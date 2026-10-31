"""Thin wrapper around ``docker run`` for launching sibling stage containers from inside the bridge.

The bridge container has the docker CLI and ``/var/run/docker.sock`` mounted, so every ``docker run`` it issues
is executed by the Docker Desktop daemon on the HOST. The daemon resolves bind-mount *sources* on the host
side, therefore sources must be host paths, not the in-bridge ``/workspace/...`` paths. See
:func:`host_path` for the spike results and :meth:`lidar_bridge.paths.RunContext.mount` for the helper that
stage code should use to build :class:`Mount` objects.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

log = logging.getLogger("lidar_bridge.docker")

# Image tag -> compose service that builds it (used for friendly "image not built" errors).
IMAGE_SERVICES = {
    "lidar-3dfin": "dfin",
    "lidar-smartqsm": "smartqsm",
    "lidar-smartqsm-gpu": "smartqsm-gpu",  # in the compose profile `gpu`; naming the service on the command line enables it
    "lidar-arbor": "arbor",
    "lidar-bridge": "bridge",
}

_WSL_PATH = re.compile(r"^/mnt/([A-Za-z])(?:/(.*))?$")


class DockerRunError(RuntimeError):
    """A ``docker`` invocation failed (non-zero exit, timeout, or docker unavailable)."""

    def __init__(self, message: str, *, returncode: int | None = None, command: Sequence[str] = (), output: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.command = list(command)
        self.output = output


class ImageNotFoundError(DockerRunError):
    """A stage image has not been built yet."""


@dataclass(frozen=True)
class Mount:
    """One bind mount for a sibling container. ``source`` is a HOST path (already mapped, see ``host_path``)."""

    source: str
    target: str
    readonly: bool = False

    def __post_init__(self) -> None:
        for what, value in (("source", self.source), ("target", self.target)):
            if "," in value:
                # --mount is a comma-separated key=value list; a comma in a path would silently corrupt it.
                raise ValueError(f"Mount {what} {value!r} contains a comma, which docker --mount cannot express")
        if not self.target.startswith("/"):
            raise ValueError(f"Mount target {self.target!r} must be an absolute path inside the container")

    def to_args(self) -> list[str]:
        spec = f"type=bind,source={self.source},target={self.target}"
        if self.readonly:
            spec += ",readonly"
        return ["--mount", spec]


def host_path(host_project_dir: str, relative: str = "") -> str:
    """Return the HOST spelling of ``<host_project_dir>/<relative>`` for use as a bind-mount source.

    SPIKE RESULT (WP4, Docker Desktop 29.1.3 for Windows / WSL2 backend, docker CLI 27 inside the bridge
    container talking to the host daemon through the mounted /var/run/docker.sock). The same project data
    directory was bind-mounted into an ``alpine`` sibling container using each spelling below, both with
    ``-v SRC:/data`` and ``--mount type=bind,source=SRC,target=/data``. A "WORKS" spelling saw the marker
    file written by the bridge into /workspace/data (so it is the real Windows folder) and a file written by the
    sibling was visible on the Windows side and from the bridge, and could be deleted by the bridge::

        D:/Podyplomowe/05_Proximity_Lidar/data                       WORKS  (-v and --mount)  <- CHOSEN
        D:\\Podyplomowe\\05_Proximity_Lidar\\data                      WORKS  (-v and --mount)
        d:/... (lower-case drive), trailing slash                    WORKS
        /run/desktop/mnt/host/d/Podyplomowe/05_Proximity_Lidar/data  WORKS  (-v and --mount)
        //d/Podyplomowe/05_Proximity_Lidar/data                      WORKS  (-v and --mount)
        /d/Podyplomowe/05_Proximity_Lidar/data                       WORKS  (-v and --mount)
        /host_mnt/d/Podyplomowe/05_Proximity_Lidar/data              WORKS  (-v and --mount)
        /mnt/d/Podyplomowe/05_Proximity_Lidar/data                   FAILS SILENTLY: rc=0 but this is a path in
                                                                     the docker-desktop VM, not the Windows
                                                                     folder. Docker creates an empty phantom
                                                                     directory there (with --mount too), the
                                                                     sibling sees an empty dir and its output is
                                                                     lost. Never use WSL-style paths.

    Other findings that shape this module:

    * ``HOST_PROJECT_DIR`` exactly as the .bat sets it (``D:/Podyplomowe/05_Proximity_Lidar``, forward slashes)
      is the simplest spelling that works, so it is used verbatim; only ``relative`` is appended.
    * ``--mount type=bind`` is used instead of ``-v``: no colon parsing around the ``D:`` drive letter, and
      the read-only flag is explicit. Its only limitation is that paths may not contain commas (``Mount``
      rejects them).
    * NEITHER ``-v`` NOR ``--mount`` fails when the source does not exist on the Windows side: Docker Desktop
      silently creates it (a ghost directory appears next to the project). Therefore existence is checked on the
      bridge side before launching (``RunContext.mount``), never left to docker.
    * ``--network none`` works (the sibling only has ``lo``); the default bridge network has internet access.
    * stdin is only forwarded with ``-i``: piping data into ``docker run`` without ``-i`` delivers 0 bytes
      (``run_container`` therefore enables ``-i`` whenever ``stdin`` data is given). Text and binary stdin up to
      the 5 MB tested pass through ``subprocess`` intact.
    * Files created by a sibling (root inside the container) can be read, chmod-600 files included, and deleted by
      the bridge, which is also root; on the Windows side they are ordinary user-owned files.
    * Streaming ``docker run`` output line by line through ``subprocess.Popen`` works and the exit code is
      propagated.

    A WSL-style ``/mnt/<drive>/...`` value (e.g. when HOST_PROJECT_DIR was copied from a WSL shell) is converted
    to ``<DRIVE>:/...`` instead of being passed through.
    """
    base = host_project_dir.strip().replace("\\", "/")
    if not base:
        raise ValueError("host project dir is empty; set HOST_PROJECT_DIR (e.g. D:/Podyplomowe/05_Proximity_Lidar)")
    wsl = _WSL_PATH.match(base)
    if wsl:
        base = f"{wsl.group(1).upper()}:/{wsl.group(2) or ''}"
    base = base.rstrip("/") if len(base) > 1 else base
    rel = relative.replace("\\", "/").strip("/")
    out = f"{base}/{rel}" if rel else base
    if "," in out:
        raise ValueError(f"host path {out!r} contains a comma, which docker --mount cannot express")
    return out


# ---------------------------------------------------------------------------------------------------------
# docker / image helpers
# ---------------------------------------------------------------------------------------------------------
def _docker(*args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise DockerRunError("the `docker` CLI is not installed in this container") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerRunError(f"`docker {' '.join(args)}` timed out after {timeout}s (is the docker daemon responsive?)") from exc


def check_docker() -> None:
    """Fail early with a helpful message when the docker socket is not usable."""
    proc = _docker("version", "--format", "{{.Server.Version}}")
    if proc.returncode != 0:
        raise DockerRunError(
            "cannot talk to the docker daemon. Run the bridge through `docker compose run --rm -T bridge ...` so "
            "that /var/run/docker.sock is mounted, and make sure Docker Desktop is running.\n"
            f"docker said: {proc.stderr.strip()}",
            returncode=proc.returncode,
        )


def list_images() -> list[str]:
    """Return ``repo:tag`` for all local images (``<none>`` entries skipped)."""
    proc = _docker("images", "--format", "{{.Repository}}:{{.Tag}}")
    if proc.returncode != 0:
        raise DockerRunError(f"`docker images` failed: {proc.stderr.strip()}", returncode=proc.returncode)
    return [line for line in proc.stdout.splitlines() if line and "<none>" not in line]


def image_exists(tag: str) -> bool:
    """True if ``tag`` (``name`` or ``name:tag``) exists locally."""
    return _docker("image", "inspect", "--format", "{{.Id}}", tag).returncode == 0


def build_hint(tag: str) -> str:
    service = IMAGE_SERVICES.get(tag.split(":")[0])
    target = service if service else "<service>"
    return f"docker compose build {target}   (PowerShell: $env:HOST_PROJECT_DIR='D:/path/to/project' first)"


def require_image(tag: str, *, dry_run: bool = False) -> None:
    """Raise :class:`ImageNotFoundError` (a clear message, not a docker stack trace) if ``tag`` is not built.

    In ``dry_run`` the problem is only logged as a warning.
    """
    try:
        if image_exists(tag):
            return
    except DockerRunError as exc:
        if not dry_run:
            raise
        log.warning("[dry-run] cannot check for image %r: %s", tag, exc)
        return
    msg = f"docker image {tag!r} not found. Build it first:\n    {build_hint(tag)}"
    if dry_run:
        log.warning("[dry-run] %s", msg)
        return
    raise ImageNotFoundError(msg)


# ---------------------------------------------------------------------------------------------------------
# docker run
# ---------------------------------------------------------------------------------------------------------
def build_run_command(
    image: str,
    args: Sequence[str] = (),
    mounts: Sequence[Mount] = (),
    *,
    network: str | None = None,
    interactive: bool = False,
    env: Mapping[str, str] | None = None,
    gpus: bool = False,
    name: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the ``docker run --rm ...`` argv (pure function; no side effects)."""
    cmd = ["docker", "run", "--rm"]
    if interactive:
        cmd.append("-i")
    if name:
        cmd += ["--name", name]
    if network:
        cmd += ["--network", network]
    if gpus:
        cmd += ["--gpus", "all"]
    for key, value in (env or {}).items():
        cmd += ["-e", f"{key}={value}"]
    for mount in mounts:
        cmd += mount.to_args()
    cmd += [str(a) for a in extra_args]
    cmd.append(image)
    cmd += [str(a) for a in args]
    return cmd


def format_command(cmd: Sequence[str]) -> str:
    """Shell-quoted one-line rendering of an argv, for logs."""
    import shlex

    return " ".join(shlex.quote(c) for c in cmd)


_CONTAINER_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

#: Seconds each of the two cleanup commands (``docker kill``, ``docker rm -f``) may take before it is abandoned.
STOP_TIMEOUT_S = 10.0


def new_container_name(image: str) -> str:
    """Unique container name ``lidar-<image basename>-<8 hex>`` (``lidar-smartqsm-1a2b3c4d``).

    The basename is the image without registry path, tag and digest; a ``lidar-`` prefix that is already part of it is
    not doubled. Every container started by :func:`run_container` gets such a name so that it can be stopped by name
    when the pipeline is interrupted. Tests may replace this function to get deterministic names.
    """
    base = image.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
    base = _CONTAINER_NAME_CHARS.sub("-", base).strip("-.") or "container"
    if not base.startswith("lidar-"):
        base = f"lidar-{base}"
    return f"{base}-{uuid.uuid4().hex[:8]}"


def stop_container(name: str, *, timeout: float = STOP_TIMEOUT_S) -> bool:
    """Best effort ``docker kill NAME`` followed by ``docker rm -f NAME``. Never raises; problems are only logged.

    Returns True when the daemon acknowledged at least one of the two commands (a container that is already gone
    makes both fail, which is fine and only logged at debug level).
    """
    acknowledged = False
    for args in (("kill", name), ("rm", "-f", name)):
        try:
            proc = _docker(*args, timeout=timeout)
        except DockerRunError as exc:
            log.warning("could not run `docker %s`: %s", " ".join(args), exc)
            continue
        if proc.returncode == 0:
            acknowledged = True
        else:
            log.debug("`docker %s` returned %s: %s", " ".join(args), proc.returncode, (proc.stderr or "").strip())
    return acknowledged


def run_container(
    image: str,
    args: Sequence[str],
    mounts: Sequence[Mount] = (),
    *,
    network: str | None = None,
    stdin: str | bytes | None = None,
    interactive: bool = False,
    env: Mapping[str, str] | None = None,
    gpus: bool = False,
    name: str | None = None,
    dry_run: bool = False,
    timeout_s: float | None = None,
    extra_args: Sequence[str] = (),
    log_prefix: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``docker run --rm [...] image args...`` and stream its combined stdout/stderr to the log.

    Parameters
    ----------
    image, args
        Image tag and the arguments after it (the image's ENTRYPOINT receives them).
    mounts
        :class:`Mount` objects whose ``source`` is already a HOST path (use ``RunContext.mount``).
    network
        e.g. ``"none"`` to cut the container off from the network.
    stdin
        Data piped to the container's stdin. Enables ``-i`` automatically (docker drops stdin otherwise).
    interactive
        Force ``-i`` even without ``stdin`` data.
    env, gpus, name, extra_args
        ``-e K=V``; ``--gpus all``; ``--name``; any further ``docker run`` flags placed before the image
        (``["--shm-size=2g", "--user", "0"]``, ``["--entrypoint", "python"]``, ...). Without ``name`` the container is
        named ``lidar-<image basename>-<8 hex>`` (:func:`new_container_name`).
    dry_run
        Only log the full command; nothing is executed. Returns a ``CompletedProcess`` with returncode 0. The logged
        command has no generated ``--name`` (so dry-run output is reproducible); a real run adds one.
    timeout_s
        Kill the container (``docker kill``) and raise :class:`DockerRunError` after this many seconds.
    check
        Raise :class:`DockerRunError` on a non-zero exit code (default). With ``check=False`` the caller
        inspects ``returncode``.

    Returns
    -------
    subprocess.CompletedProcess
        ``stdout`` holds the combined output text (stderr is merged in), ``stderr`` is ``None``.

    Notes
    -----
    Killing the ``docker run`` client does NOT stop the container, so on ANY exception while the container runs
    (``KeyboardInterrupt``, timeout, an error while streaming its output) the container is stopped by name
    (:func:`stop_container`: ``docker kill`` then ``docker rm -f``, errors only logged) before the exception is
    re-raised. The pipeline maps SIGTERM to ``KeyboardInterrupt`` (:func:`lidar_bridge.pipeline.install_termination_handler`),
    so ``docker stop`` of the bridge triggers the same cleanup. ``kill -9`` of the bridge cannot be handled.
    """
    interactive = interactive or stdin is not None
    if not name and not dry_run:
        # Always named, so that the container can be killed by name on timeout / interrupt (the `docker run` client dies
        # with the bridge but the container does not). Dry-run keeps the command free of random names (stable logs).
        name = new_container_name(image)
    cmd = build_run_command(
        image, args, mounts, network=network, interactive=interactive, env=env, gpus=gpus, name=name, extra_args=extra_args
    )
    prefix = log_prefix or image.split(":")[0]
    if dry_run:
        log.info("[dry-run] would run: %s", format_command(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=None)

    log.info("running: %s", format_command(cmd))
    started = time.monotonic()
    result = _execute(cmd, stdin=stdin, timeout_s=timeout_s, name=name, prefix=prefix)
    elapsed = time.monotonic() - started
    log.info("%s exited with code %s after %.1f s", prefix, result.returncode, elapsed)
    if check and result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-30:])
        raise DockerRunError(
            f"container {image!r} exited with code {result.returncode}\n--- last output ---\n{tail}\n--- command ---\n{format_command(cmd)}",
            returncode=result.returncode,
            command=cmd,
            output=result.stdout or "",
        )
    return result


def _execute(
    cmd: list[str], *, stdin: str | bytes | None, timeout_s: float | None, name: str | None, prefix: str
) -> subprocess.CompletedProcess:
    """Run ``cmd``, streaming merged output to the log. Separate function so tests can replace it.

    ``name`` is the ``--name`` of the container started by ``cmd``; when given, the container is stopped on timeout and
    on any exception raised while it runs (see :func:`run_container`).
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise DockerRunError("the `docker` CLI is not installed in this container") from exc

    timed_out = threading.Event()
    timer = None
    writer = None
    lines: deque[str] = deque(maxlen=20000)
    try:
        if timeout_s:

            def _kill() -> None:
                timed_out.set()
                log.error("%s: timeout after %s s, killing container", prefix, timeout_s)
                if name:
                    stop_container(name)
                proc.kill()

            timer = threading.Timer(timeout_s, _kill)
            timer.daemon = True
            timer.start()

        if stdin is not None:
            data = stdin.encode("utf-8") if isinstance(stdin, str) else stdin

            def _feed() -> None:
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    pass  # container exited before reading all input; its exit code tells the story
                finally:
                    try:
                        proc.stdin.close()  # type: ignore[union-attr]
                    except OSError:
                        pass

            writer = threading.Thread(target=_feed, daemon=True)
            writer.start()

        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            lines.append(line)
            log.info("[%s] %s", prefix, line)
        returncode = proc.wait()
    except BaseException as exc:
        # Ctrl+C, SIGTERM mapped to KeyboardInterrupt, a failing log handler, ...: the container would keep running (and
        # writing into the output folder) after the `docker run` client is gone, so stop it first.
        _abort(proc, name, prefix, exc)
        raise
    finally:
        if timer:
            timer.cancel()
        if writer:
            writer.join(timeout=5)

    if timed_out.is_set():
        raise DockerRunError(
            f"container {prefix!r} timed out after {timeout_s} s and was killed",
            returncode=returncode,
            command=cmd,
            output="\n".join(lines),
        )
    return subprocess.CompletedProcess(cmd, returncode, stdout="\n".join(lines), stderr=None)


def _abort(proc: subprocess.Popen, name: str | None, prefix: str, exc: BaseException) -> None:
    """Stop container ``name`` and the ``docker run`` client ``proc`` after ``exc``; best effort, never raises."""
    stopped = False
    if name:
        stopped = stop_container(name)
    try:
        proc.kill()
        proc.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass
    if name and not stopped:
        # Nothing to stop yet: the daemon may only just have created the container when the client was killed.
        stopped = stop_container(name)
    cause = "interrupted" if isinstance(exc, KeyboardInterrupt) else f"aborted ({type(exc).__name__})"
    if not name:
        log.warning("%s: %s", prefix, cause)
    elif stopped:
        log.warning("%s: stopped container %s", cause, name)
    else:
        log.warning("%s: container %s was not running (or could not be stopped; check `docker ps -a`)", cause, name)

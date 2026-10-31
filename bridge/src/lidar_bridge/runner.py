"""Run one tool in its mamba environment, streaming its output to the log.

Every tool gets its own session (process group), because SmartQSM starts about 8 worker processes: a timeout or an interrupt
kills the whole group, and so does the end of the tool, so that no worker can hold on to memory or the GPU for the next tree.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence

log = logging.getLogger("lidar_bridge.runner")

ENV_3DFIN, ENV_SMARTQSM, ENV_ARBOR = "3dfin", "smartqsm", "arbor"


class ToolError(RuntimeError):
    """A tool could not be run, exited with an error or timed out. ``output`` holds what it printed."""

    def __init__(self, message: str, *, returncode: int | None = None, command: Sequence[str] = (), output: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.command = list(command)
        self.output = output


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def gpu_visible() -> bool:
    """True when the container can see an NVIDIA GPU (``docker run --gpus all`` provides ``nvidia-smi``)."""
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def run_tool(
    env: str,
    args: Sequence[str],
    *,
    stdin: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    dry_run: bool = False,
    log_prefix: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``in-env <env> <args...>``; ``stdout`` of the result holds the merged output.

    ``timeout_s`` kills the tool and raises :class:`ToolError`. ``check`` also raises on a non-zero exit code. ``dry_run`` only
    logs the command. An exception while the tool runs (Ctrl+C, ``docker stop``) kills it before the exception goes on.
    """
    cmd = ["in-env", env, *args]
    prefix = log_prefix or env
    if dry_run:
        log.info("[dry-run] would run: %s", format_command(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=None)

    log.info("running: %s", format_command(cmd))
    started = time.monotonic()
    result = _execute(cmd, stdin=stdin, extra_env=extra_env, timeout_s=timeout_s, prefix=prefix)
    log.info("%s exited with code %s after %.1f s", prefix, result.returncode, time.monotonic() - started)
    if check and result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-30:])
        raise ToolError(
            f"{prefix} exited with code {result.returncode}\n--- last output ---\n{tail}\n--- command ---\n{format_command(cmd)}",
            returncode=result.returncode,
            command=cmd,
            output=result.stdout or "",
        )
    return result


def _execute(
    cmd: list[str], *, stdin: str | None, extra_env: Mapping[str, str] | None, timeout_s: float | None, prefix: str
) -> subprocess.CompletedProcess:
    """Run ``cmd``, streaming merged output to the log. Separate function so tests can replace it."""
    # PYTHONPATH points at the pipeline code; the tools have nothing to do with it.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"} | dict(extra_env or {})
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolError(f"cannot start {prefix}: {exc}", command=cmd) from exc

    timed_out = threading.Event()
    timer = None
    lines: deque[str] = deque(maxlen=20000)
    try:
        if timeout_s:

            def _on_timeout() -> None:
                timed_out.set()
                log.error("%s: timeout after %s s, killing it", prefix, timeout_s)
                _kill_group(proc)

            timer = threading.Timer(timeout_s, _on_timeout)
            timer.daemon = True
            timer.start()

        if stdin is not None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(stdin.encode("utf-8"))
                proc.stdin.close()
            except OSError:
                pass  # the tool exited before reading its input; its exit code tells the story

        # Read in a thread and wait for the tool itself: a worker that outlives the tool would keep the pipe open and a read loop
        # here would never end.
        reader = threading.Thread(target=_pump_output, args=(proc, lines, prefix), daemon=True)
        reader.start()
        returncode = proc.wait()
    except BaseException as exc:
        cause = "interrupted" if isinstance(exc, KeyboardInterrupt) else f"aborted ({type(exc).__name__})"
        log.warning("%s: %s, killing it", prefix, cause)
        _kill_group(proc)
        raise
    finally:
        if timer:
            timer.cancel()
    _kill_group(proc)  # a worker that outlived the tool
    reader.join(timeout=5)

    if timed_out.is_set():
        raise ToolError(
            f"{prefix} timed out after {timeout_s} s and was killed",
            returncode=returncode,
            command=cmd,
            output="\n".join(lines),
        )
    return subprocess.CompletedProcess(cmd, returncode, stdout="\n".join(lines), stderr=None)


def _pump_output(proc: subprocess.Popen, lines: deque[str], prefix: str) -> None:
    """Send the tool's output to the log and to ``lines``, until the pipe closes."""
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        lines.append(line)
        log.info("[%s] %s", prefix, line)


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the tool's process group; best effort, never raises."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):  # already gone
        pass
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass

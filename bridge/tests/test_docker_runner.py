from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

from lidar_bridge import docker_runner as dr
from lidar_bridge.docker_runner import DockerRunError, ImageNotFoundError, Mount, build_run_command, run_container

DATA = Mount("D:/Podyplomowe/05_Proximity_Lidar/data", "/data")


def test_basic_command():
    cmd = build_run_command("lidar-3dfin", ["cli", "/data/in/a.laz"], [DATA])
    assert cmd == [
        "docker", "run", "--rm",
        "--mount", "type=bind,source=D:/Podyplomowe/05_Proximity_Lidar/data,target=/data",
        "lidar-3dfin", "cli", "/data/in/a.laz",
    ]  # fmt: skip


def test_readonly_network_gpus_env_name_extra_args_and_order():
    cmd = build_run_command(
        "img:1",
        ["a", "b"],
        [DATA, Mount("D:/x/in", "/in", readonly=True)],
        network="none",
        interactive=True,
        env={"A": "1", "B": "two words"},
        gpus=True,
        name="job1",
        extra_args=["--shm-size=2g"],
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "-i" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert cmd[cmd.index("--gpus") + 1] == "all"
    assert cmd[cmd.index("--name") + 1] == "job1"
    assert ["-e", "A=1"] == cmd[cmd.index("-e") : cmd.index("-e") + 2]
    assert "B=two words" in cmd
    assert "type=bind,source=D:/x/in,target=/in,readonly" in cmd
    assert cmd[-3:] == ["img:1", "a", "b"] and cmd.index("--shm-size=2g") < cmd.index("img:1")
    assert "--network" not in build_run_command("img", [], [])


def test_mount_rejects_commas_and_relative_targets():
    with pytest.raises(ValueError, match="comma"):
        Mount("D:/a,b/data", "/data")
    with pytest.raises(ValueError, match="absolute"):
        Mount("D:/a", "data")


def test_dry_run_only_logs_and_never_executes(monkeypatch, caplog):
    def boom(*a, **k):
        raise AssertionError("must not execute anything in dry_run")

    monkeypatch.setattr(dr, "_execute", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.docker"):
        result = run_container("lidar-3dfin", ["cli", "x"], [DATA], network="none", dry_run=True)
    assert result.returncode == 0
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[dry-run]" in text and "--network none" in text and "lidar-3dfin cli x" in text and "D:/Podyplomowe" in text


def test_stdin_enables_interactive_flag(monkeypatch):
    seen = {}

    def fake(cmd, *, stdin, timeout_s, name, prefix):
        seen.update(cmd=cmd, stdin=stdin)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr=None)

    monkeypatch.setattr(dr, "_execute", fake)
    run_container("alpine", ["cat"], stdin="hello")
    assert "-i" in seen["cmd"] and seen["stdin"] == "hello"
    run_container("alpine", ["true"])
    assert "-i" not in seen["cmd"]
    run_container("alpine", ["true"], interactive=True)
    assert "-i" in seen["cmd"]


def test_nonzero_exit_raises_clear_error(monkeypatch):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 7, stdout="line a\nboom happened", stderr=None)

    monkeypatch.setattr(dr, "_execute", fake)
    with pytest.raises(DockerRunError) as err:
        run_container("lidar-3dfin", ["cli"], [DATA])
    assert err.value.returncode == 7
    assert "exited with code 7" in str(err.value) and "boom happened" in str(err.value)
    assert "docker run" in str(err.value)  # the command is included for copy/paste debugging
    assert run_container("lidar-3dfin", [], check=False).returncode == 7


@pytest.mark.parametrize(
    "image, pattern",
    [
        ("lidar-smartqsm", r"lidar-smartqsm-[0-9a-f]{8}"),  # no doubled prefix
        ("lidar-3dfin:latest", r"lidar-3dfin-[0-9a-f]{8}"),
        ("alpine", r"lidar-alpine-[0-9a-f]{8}"),
        ("ghcr.io/some/Tool Box:1.2", r"lidar-Tool-Box-[0-9a-f]{8}"),  # registry/tag dropped, invalid chars replaced
        ("repo/img@sha256:abcdef", r"lidar-img-[0-9a-f]{8}"),
    ],
)
def test_new_container_name_format(image, pattern):
    name = dr.new_container_name(image)
    assert re.fullmatch(pattern, name), name
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)  # what docker accepts for --name


def test_new_container_name_is_unique():
    assert len({dr.new_container_name("lidar-smartqsm") for _ in range(200)}) == 200


def test_every_container_is_named_even_without_timeout(monkeypatch):
    seen = []

    def fake(cmd, *, name, **kw):
        seen.append((name, cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=None)

    monkeypatch.setattr(dr, "_execute", fake)
    run_container("lidar-3dfin", ["cli"])  # dfin passes no timeout
    run_container("lidar-3dfin", ["cli"], timeout_s=5)
    for name, cmd in seen:
        assert re.fullmatch(r"lidar-3dfin-[0-9a-f]{8}", name) and cmd[3:5] == ["--name", name]
        assert "--rm" in cmd  # still removed by the daemon when it exits normally
    assert seen[0][0] != seen[1][0]


def test_caller_supplied_name_wins_and_factory_is_replaceable(monkeypatch):
    seen = []
    monkeypatch.setattr(dr, "_execute", lambda cmd, *, name, **kw: seen.append(name) or subprocess.CompletedProcess(cmd, 0, stdout=""))
    run_container("alpine", [], name="mine")
    monkeypatch.setattr(dr, "new_container_name", lambda image: f"fixed-{image}")
    run_container("alpine", [])
    assert seen == ["mine", "fixed-alpine"]


def test_dry_run_command_has_no_generated_name(monkeypatch, caplog):
    monkeypatch.setattr(dr, "new_container_name", lambda image: pytest.fail("no name is generated in a dry run"))
    with caplog.at_level(logging.INFO, logger="lidar_bridge.docker"):
        result = run_container("lidar-3dfin", ["cli"], [DATA], network="none", dry_run=True)
    assert "--name" not in result.args
    assert "docker run --rm --network none --mount" in caplog.text  # reproducible dry-run output


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_execute_streams_output_pipes_stdin_and_returns_code(caplog):
    code = "import sys; data = sys.stdin.read(); print('got', data.upper()); print('err', file=sys.stderr); sys.exit(3)"
    with caplog.at_level(logging.INFO, logger="lidar_bridge.docker"):
        proc = dr._execute(_py(code), stdin="abc", timeout_s=None, name=None, prefix="t")
    assert proc.returncode == 3
    assert proc.stdout.splitlines() == ["got ABC", "err"]  # stderr merged into stdout
    assert [r.getMessage() for r in caplog.records] == ["[t] got ABC", "[t] err"]


def test_execute_handles_large_binary_safe_stdin():
    code = "import sys; print(len(sys.stdin.buffer.read()))"
    proc = dr._execute(_py(code), stdin=b"x" * 5_000_000, timeout_s=None, name=None, prefix="t")
    assert proc.stdout.strip() == "5000000"


def test_execute_without_stdin_gives_empty_stdin():
    proc = dr._execute(_py("import sys; print(repr(sys.stdin.read()))"), stdin=None, timeout_s=None, name=None, prefix="t")
    assert proc.stdout.strip() == "''"


def test_execute_timeout_kills_and_raises():
    with pytest.raises(DockerRunError, match="timed out"):
        dr._execute(_py("import time; time.sleep(30)"), stdin=None, timeout_s=0.5, name=None, prefix="t")


def test_missing_docker_cli_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(DockerRunError, match="docker"):
        dr._execute(["docker", "version"], stdin=None, timeout_s=None, name=None, prefix="t")


# -- cleanup of the container on interrupt / error / timeout (subprocess mocked) -----------------------------------------
class FakeStdout:
    """``readline`` plays a script: bytes are returned, exceptions raised, callables called (to block, for instance)."""

    def __init__(self, items):
        self.items = list(items)

    def readline(self):
        if not self.items:
            return b""
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item() if callable(item) else item


class FakeProc:
    def __init__(self, items=(), order=None):
        self.stdout = FakeStdout(items)
        self.stdin = None
        self.killed = threading.Event()
        self.kill_calls = 0
        self.order = order if order is not None else []

    def kill(self):
        self.kill_calls += 1
        self.order.append(("client", "kill"))
        self.killed.set()

    def wait(self, timeout=None):
        return 137 if self.killed.is_set() else 0


@pytest.fixture
def fake_docker(monkeypatch):
    """Replaces ``Popen`` (returns the proc stored in ``state['proc']``) and the ``docker kill/rm`` helper calls."""
    state = {"calls": [], "rc": {}, "raises": {}, "order": []}
    state["proc"] = FakeProc(order=state["order"])

    def popen(cmd, **kw):
        state["popen_cmd"] = cmd
        return state["proc"]

    def docker(*args, timeout=30):
        state["calls"].append((args, timeout))
        state["order"].append(("docker", args[0]))
        if args[0] in state["raises"]:
            raise DockerRunError(f"`docker {args[0]}` timed out")
        return subprocess.CompletedProcess(["docker", *args], state["rc"].get(args[0], 0), stdout="", stderr="Error: no such container")

    monkeypatch.setattr(dr.subprocess, "Popen", popen)
    monkeypatch.setattr(dr, "_docker", docker)
    return state


def script(state, *items):
    """Make the fake ``docker run`` client print/raise ``items``; ``state['order']`` records docker + client kills."""
    state["proc"] = FakeProc(items, order=state["order"])
    return state["proc"]


def run_fake(**kw):
    return dr._execute(["docker", "run"], stdin=None, timeout_s=kw.pop("timeout_s", None), name=kw.pop("name", "lidar-x-1a2b3c4d"), prefix="x", **kw)


def test_keyboard_interrupt_kills_and_removes_container_then_reraises(fake_docker, caplog):
    script(fake_docker, b"line 1\n", KeyboardInterrupt("SIGTERM"))
    with caplog.at_level(logging.INFO, logger="lidar_bridge.docker"), pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        run_fake()
    assert [args for args, _ in fake_docker["calls"]] == [("kill", "lidar-x-1a2b3c4d"), ("rm", "-f", "lidar-x-1a2b3c4d")]
    assert all(timeout <= 30 for _, timeout in fake_docker["calls"])  # short timeouts, never hangs the abort
    # the container is stopped BEFORE the docker run client is killed, and the client is killed at all
    assert fake_docker["order"] == [("docker", "kill"), ("docker", "rm"), ("client", "kill")]
    assert "interrupted: stopped container lidar-x-1a2b3c4d" in caplog.text


@pytest.mark.parametrize("exc", [RuntimeError("log handler exploded"), SystemExit(3), MemoryError()])
def test_any_exception_while_streaming_stops_the_container(fake_docker, exc):
    script(fake_docker, b"hello\n", exc)
    with pytest.raises(type(exc)):
        run_fake()
    assert [args[0] for args, _ in fake_docker["calls"]] == ["kill", "rm"]
    assert fake_docker["proc"].kill_calls == 1


def test_cleanup_failures_are_only_logged_and_original_exception_wins(fake_docker, caplog):
    script(fake_docker, KeyboardInterrupt())
    fake_docker["raises"] = {"kill": True, "rm": True}  # docker daemon unresponsive: both commands time out
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.docker"), pytest.raises(KeyboardInterrupt):
        run_fake()
    assert "could not run `docker kill" in caplog.text and "could not run `docker rm -f" in caplog.text
    assert "could not be stopped" in caplog.text
    assert fake_docker["proc"].kill_calls == 1  # the client is still killed


def test_container_that_is_already_gone_is_not_an_error(fake_docker, caplog):
    script(fake_docker, KeyboardInterrupt())
    fake_docker["rc"] = {"kill": 1, "rm": 1}  # "No such container"
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.docker"), pytest.raises(KeyboardInterrupt):
        run_fake()
    # tried again after the client was killed (the daemon may only just have created the container)
    assert fake_docker["order"] == [("docker", "kill"), ("docker", "rm"), ("client", "kill"), ("docker", "kill"), ("docker", "rm")]
    assert "was not running" in caplog.text


def test_no_name_means_only_the_client_is_killed(fake_docker):
    script(fake_docker, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run_fake(name=None)
    assert fake_docker["calls"] == [] and fake_docker["proc"].kill_calls == 1


def test_normal_exit_runs_no_cleanup_commands(fake_docker):
    script(fake_docker, b"a\n", b"b\n")
    result = run_fake()
    assert result.returncode == 0 and result.stdout == "a\nb" and fake_docker["calls"] == []
    assert fake_docker["proc"].kill_calls == 0


def test_timeout_kills_container_and_keeps_the_error_message(fake_docker):
    proc = script(fake_docker)
    proc.stdout = FakeStdout([b"working\n", lambda: (proc.killed.wait(10), b"")[1]])  # blocks until the client is killed
    with pytest.raises(DockerRunError, match=r"container 'x' timed out after 0\.2 s and was killed") as err:
        run_fake(timeout_s=0.2)
    assert [args for args, _ in fake_docker["calls"]] == [("kill", "lidar-x-1a2b3c4d"), ("rm", "-f", "lidar-x-1a2b3c4d")]
    assert proc.kill_calls == 1 and err.value.output == "working"


def test_run_container_passes_the_generated_name_to_execute(monkeypatch):
    monkeypatch.setattr(dr, "new_container_name", lambda image: "lidar-fixed-00000000")
    seen = {}

    def boom(cmd, *, name, **kw):
        seen.update(name=name, cmd=cmd)
        raise KeyboardInterrupt

    monkeypatch.setattr(dr, "_execute", boom)
    with pytest.raises(KeyboardInterrupt):
        run_container("lidar-smartqsm", ["x"])
    assert seen["name"] == "lidar-fixed-00000000" and "--name" in seen["cmd"]


# -- the same with a fake `docker` executable (real Popen, real signal); POSIX only ---------------------------------------
FAKE_DOCKER = """\
#!/bin/sh
echo "$@" >> "$FAKE_DOCKER_LOG"
case "$1" in
  run) echo started; exec sleep 60 ;;
  kill|rm) exit 0 ;;
esac
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell and SIGINT delivery to the main thread")
def test_sigint_during_a_real_subprocess_stops_the_named_container(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "docker"
    script.write_text(FAKE_DOCKER)
    script.chmod(0o755)
    log_file = tmp_path / "docker.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log_file))
    timer = threading.Timer(0.7, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_container("lidar-smartqsm", ["x"])
    finally:
        timer.cancel()
    assert time.monotonic() - started < 15  # did not wait for `sleep 60`
    lines = log_file.read_text().splitlines()
    run_line = next(line for line in lines if line.startswith("run "))
    name = re.search(r"--name (lidar-smartqsm-[0-9a-f]{8}) ", run_line).group(1)
    assert lines[1:] == [f"kill {name}", f"rm -f {name}"]


@pytest.mark.skipif(not os.environ.get("LIDAR_DOCKER_TESTS"), reason="set LIDAR_DOCKER_TESTS=1 to talk to the real docker daemon")
def test_real_interrupt_leaves_no_container_behind():
    """``sleep`` in a real alpine container, SIGINT to this process: the container must be gone afterwards."""
    name = f"lidar-selftest-{uuid.uuid4().hex[:8]}"
    timer = threading.Timer(3.0, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_container("alpine", ["sleep", "300"], name=name, network="none")
    finally:
        timer.cancel()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"], capture_output=True, text=True)
    assert listed.stdout.strip() == ""


def test_require_image(monkeypatch, caplog):
    monkeypatch.setattr(dr, "image_exists", lambda tag: tag == "lidar-3dfin")
    dr.require_image("lidar-3dfin")
    with pytest.raises(ImageNotFoundError, match=r"docker compose build smartqsm"):
        dr.require_image("lidar-smartqsm")
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.docker"):
        dr.require_image("lidar-arbor", dry_run=True)  # warns only
    assert "docker compose build arbor" in caplog.text


def test_build_hint_for_unknown_image():
    assert "<service>" in dr.build_hint("someone/else:1")


@pytest.mark.skipif(not os.environ.get("LIDAR_DOCKER_TESTS"), reason="set LIDAR_DOCKER_TESTS=1 to talk to the real docker daemon")
def test_real_roundtrip_through_docker_socket():
    """Spike regression: sibling container sees data/ via the host path and its output is visible to the bridge."""
    from lidar_bridge.paths import RunContext

    ctx = RunContext.from_env("_selftest")
    scratch = ctx.data_dir / "out" / "_selftest_docker"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        run_container(
            "alpine",
            ["sh", "-c", "cat > /d/from_stdin.txt && ls /d && echo hi > /d/hello.txt"],
            [ctx.mount(scratch, "/d")],
            network="none",
            stdin="piped\n",
        )
        assert (scratch / "hello.txt").read_text().strip() == "hi"
        assert (scratch / "from_stdin.txt").read_text() == "piped\n"
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)

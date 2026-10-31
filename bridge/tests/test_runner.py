"""Tests of ``lidar_bridge.runner`` with real subprocesses: a tiny fake ``in-env`` on PATH runs the rest of the command."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from lidar_bridge import runner
from lidar_bridge.runner import ToolError, gpu_visible, run_tool

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="process groups and POSIX shell scripts")

FAKE_IN_ENV = """\
#!/bin/sh
# fake in-env: drops the environment name and runs the rest of the command
shift
exec "$@"
"""


def sh(script: str) -> list[str]:
    return ["sh", "-c", script]


def process_is_dead(pid: int, wait_s: float = 5.0) -> bool:
    """True when ``pid`` is gone or a zombie (nobody reaps orphans in a container without an init process)."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        except OSError:  # no /proc: fall back to a signal probe
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            state = "R"
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def fake_in_env(tmp_path, monkeypatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "in-env"
    script.write_text(FAKE_IN_ENV, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


# -- output and exit codes --------------------------------------------------------------------------------------------
def test_command_is_in_env_then_the_environment_then_the_args(fake_in_env):
    result = run_tool("myenv", ["echo", "hi"])
    assert result.args == ["in-env", "myenv", "echo", "hi"]
    assert result.returncode == 0 and result.stdout == "hi"


def test_output_is_streamed_to_the_log_and_returned_merged(fake_in_env, caplog):
    with caplog.at_level(logging.INFO, logger="lidar_bridge.runner"):
        result = run_tool("e", sh("echo one; echo two >&2; echo three"), log_prefix="tool")
    assert result.stdout.splitlines() == ["one", "two", "three"]  # stderr is merged into stdout
    assert result.returncode == 0
    for line in ("[tool] one", "[tool] two", "[tool] three"):
        assert line in caplog.text
    assert "tool exited with code 0" in caplog.text


def test_log_prefix_defaults_to_the_environment_name(fake_in_env, caplog):
    with caplog.at_level(logging.INFO, logger="lidar_bridge.runner"):
        run_tool("arbor", ["echo", "x"])
    assert "[arbor] x" in caplog.text


def test_output_streams_while_the_tool_still_runs(fake_in_env):
    seen: list[float] = []

    class Stamp(logging.Handler):
        def emit(self, record):
            if record.getMessage() == "[e] first":
                seen.append(time.monotonic())

    handler = Stamp()
    runner.log.addHandler(handler)
    previous = runner.log.level
    runner.log.setLevel(logging.INFO)
    try:
        started = time.monotonic()
        run_tool("e", sh("echo first; sleep 1; echo last"))
        finished = time.monotonic()
    finally:
        runner.log.removeHandler(handler)
        runner.log.setLevel(previous)
    assert seen and seen[0] - started < finished - started - 0.5  # "first" was logged long before the tool ended


def test_non_zero_exit_raises_tool_error_with_the_output(fake_in_env):
    with pytest.raises(ToolError) as err:
        run_tool("e", sh("echo before; echo boom >&2; exit 3"), log_prefix="tool")
    exc = err.value
    assert exc.returncode == 3
    assert exc.command[:2] == ["in-env", "e"]
    assert "before" in exc.output and "boom" in exc.output
    assert "tool exited with code 3" in str(exc) and "boom" in str(exc)  # the message carries the tail


def test_check_false_returns_the_exit_code(fake_in_env):
    result = run_tool("e", sh("echo partial; exit 7"), check=False)
    assert result.returncode == 7 and result.stdout == "partial"


def test_dry_run_runs_nothing(fake_in_env, tmp_path, caplog):
    marker = tmp_path / "marker"
    with caplog.at_level(logging.INFO, logger="lidar_bridge.runner"):
        result = run_tool("e", sh(f"touch {marker}"), dry_run=True)
    assert not marker.exists()
    assert result.returncode == 0 and result.stdout == ""
    assert "[dry-run] would run: in-env e sh -c" in caplog.text


def test_dry_run_does_not_even_need_in_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert run_tool("e", ["whatever"], dry_run=True).returncode == 0


def test_a_missing_in_env_is_a_tool_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ToolError, match="cannot start e"):
        run_tool("e", ["echo", "x"])


def test_undecodable_output_does_not_break_the_run(fake_in_env):
    result = run_tool("e", sh(r"printf 'caf\351\n'"))
    assert result.returncode == 0 and result.stdout.startswith("caf")


# -- stdin and environment --------------------------------------------------------------------------------------------
def test_stdin_reaches_the_tool(fake_in_env):
    result = run_tool("e", ["cat"], stdin="no\nsecond line\n")
    assert result.stdout.splitlines() == ["no", "second line"]


def test_without_stdin_the_tool_sees_end_of_input(fake_in_env):
    result = run_tool("e", sh("read answer || echo eof"), timeout_s=20)
    assert result.stdout == "eof"


def test_pythonpath_is_removed_and_extra_env_is_added(fake_in_env, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/lidar/bridge/src")
    monkeypatch.setenv("KEEP_ME", "kept")
    result = run_tool("e", sh('echo "pp=[$PYTHONPATH] foo=[$FOO] keep=[$KEEP_ME]"'), extra_env={"FOO": "bar"})
    assert result.stdout == "pp=[] foo=[bar] keep=[kept]"
    assert os.environ["PYTHONPATH"] == "/opt/lidar/bridge/src"  # only the child's environment changes


def test_extra_env_may_set_pythonpath_explicitly(fake_in_env, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/old")
    result = run_tool("e", sh('echo "[$PYTHONPATH]"'), extra_env={"PYTHONPATH": "/new"})
    assert result.stdout == "[/new]"


# -- process groups ---------------------------------------------------------------------------------------------------
def test_timeout_kills_the_tool_and_its_background_workers(fake_in_env, tmp_path):
    pidfile = tmp_path / "worker.pid"
    started = time.monotonic()
    with pytest.raises(ToolError) as err:
        run_tool("e", sh(f"sleep 100 >/dev/null 2>&1 & echo $! > {pidfile}; echo started; wait"), timeout_s=2, log_prefix="tool")
    assert time.monotonic() - started < 30  # not the 100 s of the worker
    assert "timed out" in str(err.value) and "tool" in str(err.value)
    assert "started" in err.value.output
    worker = int(pidfile.read_text())
    assert process_is_dead(worker)


def test_timeout_kills_a_worker_that_holds_the_output_pipe_too(fake_in_env, tmp_path):
    pidfile = tmp_path / "worker.pid"
    started = time.monotonic()
    with pytest.raises(ToolError, match="timed out"):
        run_tool("e", sh(f"sleep 100 & echo $! > {pidfile}; wait"), timeout_s=2)
    assert time.monotonic() - started < 30
    assert process_is_dead(int(pidfile.read_text()))


def test_a_worker_left_behind_after_a_normal_exit_is_killed(fake_in_env, tmp_path):
    pidfile = tmp_path / "worker.pid"
    result = run_tool("e", sh(f"sleep 100 >/dev/null 2>&1 & echo $! > {pidfile}; echo done"))
    assert result.returncode == 0 and result.stdout == "done"
    assert process_is_dead(int(pidfile.read_text()))


def test_a_worker_that_holds_the_output_pipe_cannot_hang_a_tool_without_timeout(fake_in_env, tmp_path):
    pidfile = tmp_path / "worker.pid"
    started = time.monotonic()
    result = run_tool("e", sh(f"sleep 100 & echo $! > {pidfile}; echo done"))
    assert time.monotonic() - started < 20
    assert result.returncode == 0 and result.stdout == "done"
    assert process_is_dead(int(pidfile.read_text()))


def test_a_worker_left_behind_after_a_failure_is_killed(fake_in_env, tmp_path):
    pidfile = tmp_path / "worker.pid"
    with pytest.raises(ToolError) as err:
        run_tool("e", sh(f"sleep 100 >/dev/null 2>&1 & echo $! > {pidfile}; exit 4"))
    assert err.value.returncode == 4
    assert process_is_dead(int(pidfile.read_text()))


def test_an_interrupt_kills_the_tool_group_before_it_propagates(fake_in_env, tmp_path):
    pidfile = tmp_path / "worker.pid"

    class Interrupt(logging.Handler):
        def emit(self, record):
            if record.getMessage() == "[e] go":
                os.kill(os.getpid(), signal.SIGINT)  # a real Ctrl+C, while the main thread waits for the tool

    handler = Interrupt()
    runner.log.addHandler(handler)
    previous = runner.log.level
    runner.log.setLevel(logging.INFO)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_tool("e", sh(f"sleep 100 >/dev/null 2>&1 & echo $! > {pidfile}; echo go; wait"))
    finally:
        runner.log.removeHandler(handler)
        runner.log.setLevel(previous)
    assert process_is_dead(int(pidfile.read_text()))


def test_the_tool_runs_in_its_own_process_group(fake_in_env):
    result = run_tool("e", [sys.executable, "-c", "import os; print(os.getpgrp(), os.getpid())"])
    pgid, pid = result.stdout.split()
    assert pgid == pid  # the tool is the leader of a new session / group
    assert int(pgid) != os.getpgrp()


# -- gpu_visible ------------------------------------------------------------------------------------------------------
def test_gpu_is_not_visible_without_nvidia_smi(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # an empty directory
    assert gpu_visible() is False


@pytest.mark.parametrize("exit_code, expected", [(0, True), (9, False)])
def test_gpu_visible_follows_the_exit_code_of_nvidia_smi(monkeypatch, tmp_path, exit_code, expected):
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert gpu_visible() is expected


# -- helpers ----------------------------------------------------------------------------------------------------------
def test_format_command_quotes_for_a_shell():
    assert runner.format_command(["in-env", "e", "a b", "c"]) == "in-env e 'a b' c"


def test_environment_names():
    assert (runner.ENV_3DFIN, runner.ENV_SMARTQSM, runner.ENV_ARBOR) == ("3dfin", "smartqsm", "arbor")

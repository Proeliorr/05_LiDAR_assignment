from __future__ import annotations

import logging
import re
import signal
import sys
import threading

import pytest

from lidar_bridge import pipeline
from lidar_bridge.docker_runner import DockerRunError
from lidar_bridge.paths import RunContext
from lidar_bridge.pipeline import PipelineError, execute_pipeline, select_stages
from lidar_bridge.stages import STAGE_ORDER, StageError

HOST = "D:/Podyplomowe/05_Proximity_Lidar"


def recorder(order: list[str], fail: dict[str, Exception] | None = None):
    fns = {}
    for name in STAGE_ORDER:

        def fn(ctx, _name=name):
            order.append(_name)
            if fail and _name in fail:
                raise fail[_name]
            return {"stage": _name}

        fns[name] = fn
    return fns


@pytest.fixture
def run_ctx(project):
    (project / "data/in/ref.csv").write_text("id\n1\n")
    return RunContext(
        run_name="r1",
        host_project_dir=HOST,
        project_dir=project,
        input=project / "data/in/plot_01.laz",
        reference=project / "data/in/ref.csv",
        dry_run=True,  # skips the docker daemon check
    )


def test_all_stages_by_default():
    assert select_stages(None) == ["dfin", "split", "smartqsm", "arbor", "compare"]


def test_selection_is_reordered_to_fixed_order_and_deduped():
    assert select_stages(["compare", "smartqsm"]) == ["smartqsm", "compare"]
    assert select_stages(["arbor", "dfin", "arbor"]) == ["dfin", "arbor"]
    assert select_stages("compare, split") == ["split", "compare"]
    assert select_stages(("dfin",)) == ["dfin"]


def test_unknown_or_empty_selection_fails():
    with pytest.raises(PipelineError, match=r"unknown stage.*\['qsm'\]"):
        select_stages(["dfin", "qsm"])
    with pytest.raises(PipelineError, match="empty"):
        select_stages([])


def test_executes_selected_stages_in_order(run_ctx):
    order: list[str] = []
    results = execute_pipeline(run_ctx, select_stages(["compare", "dfin", "smartqsm"]), recorder(order))
    assert order == ["dfin", "smartqsm", "compare"]
    assert results == {"dfin": {"stage": "dfin"}, "smartqsm": {"stage": "smartqsm"}, "compare": {"stage": "compare"}}


def test_failure_stops_pipeline_and_names_stage(run_ctx, caplog):
    order: list[str] = []
    fns = recorder(order, fail={"split": StageError("no trees found")})
    with caplog.at_level(logging.INFO, logger="lidar_bridge.pipeline"), pytest.raises(PipelineError) as err:
        execute_pipeline(run_ctx, list(STAGE_ORDER), fns)
    assert order == ["dfin", "split"]  # smartqsm etc. never ran
    assert "stage 'split' failed: no trees found" in str(err.value)
    assert "FAILED" in caplog.text and "traceback" not in caplog.text.lower()  # expected error: no stack trace spam


def test_unexpected_exception_is_wrapped_and_logged_with_traceback(run_ctx, caplog):
    fns = recorder([], fail={"dfin": KeyError("boom")})
    with caplog.at_level(logging.INFO, logger="lidar_bridge.pipeline"), pytest.raises(PipelineError, match="stage 'dfin' failed"):
        execute_pipeline(run_ctx, ["dfin"], fns)
    assert "Traceback" in caplog.text


def test_timing_is_logged_per_stage(run_ctx, caplog):
    with caplog.at_level(logging.INFO, logger="lidar_bridge.pipeline"):
        execute_pipeline(run_ctx, ["dfin", "split"], recorder([]))
    text = caplog.text
    assert "stage dfin: done in" in text and "stage split: done in" in text and "total" in text


def test_preflight_requires_input_and_reference_before_anything_runs(project):
    base = dict(run_name="r1", host_project_dir=HOST, project_dir=project)
    order: list[str] = []
    no_input = RunContext(**base, reference=project / "data/in/ref.csv")
    with pytest.raises(PipelineError, match="needs input"):
        execute_pipeline(no_input, ["dfin"], recorder(order))
    with pytest.raises(PipelineError, match="needs reference"):  # compare on its own was asked for explicitly
        execute_pipeline(RunContext(**base, input=project / "data/in/plot_01.laz"), ["compare"], recorder(order))
    with pytest.raises(PipelineError, match="reference file not found"):  # a given but missing file is a typo, not "later"
        execute_pipeline(RunContext(**base, reference=project / "data/in/nope.csv"), ["compare"], recorder(order))
    with pytest.raises(PipelineError, match="reference file not found"):
        execute_pipeline(RunContext(**base, input=project / "data/in/plot_01.laz", reference=project / "data/in/nope.csv"), ["dfin", "compare"], recorder(order))
    assert order == []
    # later stages alone need neither input nor reference
    execute_pipeline(RunContext(**base, dry_run=True), ["smartqsm"], recorder(order))
    assert order == ["smartqsm"]


def test_preflight_in_dry_run_only_warns(project, caplog):
    ctx = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project, dry_run=True)
    order: list[str] = []
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.pipeline"):
        execute_pipeline(ctx, ["dfin", "compare"], recorder(order))
    assert order == ["dfin"] and "needs input" in caplog.text  # compare is skipped, not run, without a reference
    order.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.pipeline"):
        execute_pipeline(ctx, ["compare"], recorder(order))
    assert order == ["compare"] and "needs reference" in caplog.text  # asked for alone: still flagged


def test_compare_is_skipped_when_no_reference_exists_yet(project, caplog):
    """The reference file is attached later: the processing stages must run and the run must end successfully."""
    ctx = RunContext(
        run_name="r1", host_project_dir=HOST, project_dir=project, input=project / "data/in/plot_01.laz", dry_run=True
    )
    order: list[str] = []
    with caplog.at_level(logging.INFO, logger="lidar_bridge.pipeline"):
        results = execute_pipeline(ctx, list(STAGE_ORDER), recorder(order))
    assert order == ["dfin", "split", "smartqsm", "arbor"] and "compare" not in results
    assert "no reference table given: skipping the compare stage" in caplog.text
    assert "stages=[compare] run_name=r1 reference=" in caplog.text  # tells the user how to attach it later
    assert "stage compare" not in caplog.text


def test_compare_runs_when_a_reference_is_given(run_ctx):
    order: list[str] = []
    execute_pipeline(run_ctx, list(STAGE_ORDER), recorder(order))
    assert order == list(STAGE_ORDER)


def test_docker_is_checked_once_for_container_stages_only(project, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "check_docker", lambda: calls.append(1))
    ctx = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project, input=project / "data/in/plot_01.laz")
    execute_pipeline(ctx, ["dfin", "split"], recorder([]))
    assert calls == [1]
    execute_pipeline(ctx, ["split", "compare"] if False else ["split"], recorder([]))
    assert calls == [1]  # split alone does not need the daemon


def test_docker_unavailable_gives_pipeline_error(project, monkeypatch):
    def broken():
        raise DockerRunError("cannot talk to the docker daemon")

    monkeypatch.setattr(pipeline, "check_docker", broken)
    ctx = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project, input=project / "data/in/plot_01.laz")
    with pytest.raises(PipelineError, match="docker daemon"):
        execute_pipeline(ctx, ["dfin"], recorder([]))


def test_run_pipeline_entry_builds_context_and_exits_nonzero_on_failure(monkeypatch, project):
    monkeypatch.setenv("HOST_PROJECT_DIR", HOST)
    monkeypatch.setenv("LIDAR_PROJECT_DIR", str(project))
    order: list[str] = []
    fns = recorder(order)
    kwargs = dict(input="data/in/plot_01.laz", reference=None, run_name="r1", stages=["dfin", "split"], dry_run=True, **fns)
    pipeline.run_pipeline(**kwargs)
    assert order == ["dfin", "split"]

    bad = dict(kwargs, stages=["dfin"], dfin=lambda ctx: (_ for _ in ()).throw(StageError("nope")))
    with pytest.raises(SystemExit) as exit_info:
        pipeline.run_pipeline(**bad)
    assert exit_info.value.code == 1

    with pytest.raises(SystemExit):  # input outside the project
        pipeline.run_pipeline(**dict(kwargs, input="/etc/passwd"))
    with pytest.raises(SystemExit):  # unknown stage
        pipeline.run_pipeline(**dict(kwargs, stages=["nope"]))


# -- interrupts: SIGTERM -> KeyboardInterrupt, one clear log line, exit code 130 ----------------------------------------
@pytest.fixture
def restore_signals():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_is_mapped_to_keyboard_interrupt(restore_signals):
    assert pipeline.install_termination_handler() is True
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        signal.raise_signal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt, match="SIGHUP"):
        signal.raise_signal(signal.SIGHUP)


def test_termination_handler_is_not_installed_outside_the_main_thread():
    result = []
    thread = threading.Thread(target=lambda: result.append(pipeline.install_termination_handler()))
    thread.start()
    thread.join()
    assert result == [False]


def test_execute_pipeline_logs_aborted_stage_and_reraises_interrupt(run_ctx, caplog):
    fns = recorder([], fail={"split": KeyboardInterrupt("SIGTERM")})
    with caplog.at_level(logging.INFO, logger="lidar_bridge.pipeline"), pytest.raises(KeyboardInterrupt):
        execute_pipeline(run_ctx, list(STAGE_ORDER), fns)
    assert re.search(r"split\s+ABORTED", caplog.text) and "smartqsm" not in caplog.text.split("summary:")[1]


@pytest.mark.parametrize("reason, shown", [("SIGTERM", "SIGTERM"), ("", "Ctrl+C")])
def test_run_pipeline_interrupt_is_one_line_and_exit_code_130(monkeypatch, project, caplog, reason, shown):
    monkeypatch.setenv("HOST_PROJECT_DIR", HOST)
    monkeypatch.setenv("LIDAR_PROJECT_DIR", str(project))
    fns = recorder([], fail={"dfin": KeyboardInterrupt(reason)})
    kwargs = dict(input="data/in/plot_01.laz", reference=None, run_name="r1", stages=["dfin", "split"], dry_run=True, **fns)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.pipeline"), pytest.raises(SystemExit) as exit_info:
        pipeline.run_pipeline(**kwargs)
    assert exit_info.value.code == 130 == pipeline.EXIT_INTERRUPTED
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("interrupted")]
    assert lines == [f"interrupted ({shown}): run 'r1' aborted, exit code 130"]
    assert "Traceback" not in caplog.text and not any(r.exc_info for r in caplog.records)

"""The pipeline runner: executes the selected stages in the fixed order, with timing and clear failures."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from lidar_bridge.docker_runner import DockerRunError, check_docker
from lidar_bridge.paths import PathOutsideProjectError, RunContext
from lidar_bridge.stages import DOCKER_STAGES, STAGE_ORDER, StageError

log = logging.getLogger("lidar_bridge.pipeline")

StageCallable = Callable[[RunContext], Any]

# Failures that are reported as a one-line message (everything else also gets a traceback in the log).
EXPECTED_ERRORS = (StageError, DockerRunError, PathOutsideProjectError, FileNotFoundError, NotImplementedError)


#: Process exit code after an interrupt (128 + SIGINT), the usual shell convention.
EXIT_INTERRUPTED = 130


class PipelineError(RuntimeError):
    """A stage (or the setup before it) failed; the message names the stage and the cause."""


def _raise_keyboard_interrupt(signum: int, frame: Any) -> None:
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    raise KeyboardInterrupt(name)


def install_termination_handler() -> bool:
    """Make SIGTERM (and SIGHUP where it exists) raise :class:`KeyboardInterrupt` in the main thread.

    Ctrl+C already does. Without this the bridge, which runs as PID 1 inside its container, would ignore ``docker stop`` /
    ``docker compose`` shutdown (PID 1 has no default signal actions) and, after the grace period, be SIGKILLed with
    its sibling stage containers still running. Mapping the signal to ``KeyboardInterrupt`` runs the same cleanup as
    Ctrl+C (:func:`lidar_bridge.docker_runner.run_container` stops its container) and ends the run with exit code
    :data:`EXIT_INTERRUPTED`. Returns False (and installs nothing) outside the main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        return False
    installed = False
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _raise_keyboard_interrupt)
            installed = True
        except (ValueError, OSError):  # not the main thread of the main interpreter / signal not supported
            log.debug("cannot install a %s handler", name)
    return installed


def select_stages(stages: Iterable[str] | str | None) -> list[str]:
    """Validate a ``stages=[...]`` selection and return it in the FIXED pipeline order (duplicates dropped).

    ``stages=[compare,dfin]`` still runs dfin first. A comma separated string is accepted too.
    """
    if stages is None:
        return list(STAGE_ORDER)
    if isinstance(stages, str):
        stages = [s for s in (part.strip() for part in stages.split(",")) if s]
    requested = [str(s) for s in stages]
    unknown = [s for s in requested if s not in STAGE_ORDER]
    if unknown:
        raise PipelineError(f"unknown stage(s) {unknown}; valid stages are {list(STAGE_ORDER)}")
    if not requested:
        raise PipelineError(f"stages is empty; choose from {list(STAGE_ORDER)}")
    return [s for s in STAGE_ORDER if s in requested]


def preflight(ctx: RunContext, selected: list[str]) -> None:
    """Cheap checks before anything slow starts (a missing reference must not surface after a 30 min dfin run)."""
    problems: list[str] = []
    if "dfin" in selected:
        if ctx.input is None:
            problems.append("stage dfin needs input=data/in/<file>.laz")
        elif not ctx.input.is_file():
            problems.append(f"input file not found: {ctx.input} (put it under data/in/)")
    if "compare" in selected:
        if ctx.reference is None:
            problems.append("stage compare needs reference=data/in/<reference table>.csv|xlsx")
        elif not ctx.reference.is_file():
            problems.append(f"reference file not found: {ctx.reference}")
    if problems:
        if ctx.dry_run:
            for problem in problems:
                log.warning("[dry-run] %s", problem)
        else:
            raise PipelineError("; ".join(problems))
    if not ctx.dry_run and DOCKER_STAGES.intersection(selected):
        try:
            check_docker()
        except DockerRunError as exc:
            raise PipelineError(str(exc)) from exc


def skip_compare_without_reference(ctx: RunContext, selected: list[str]) -> list[str]:
    """Drop ``compare`` when no reference table was given and other stages are selected too.

    The reference measurements may only exist after the (hours long) processing stages have run, so their absence must
    not stop the run: the results stay in the run folder and ``compare`` is run later, alone, once the file exists. A
    reference that IS given but missing is still an error (``preflight``), and so is asking for ``compare`` alone.
    """
    if "compare" not in selected or ctx.reference is not None or len(selected) == 1:
        return selected
    log.warning(
        "no reference table given: skipping the compare stage. The results stay in %s; when the reference file is "
        "ready, compare without re-running anything: stages=[compare] run_name=%s reference=data/in/<file> "
        "(launcher: run_pipeline.bat --stages compare --run %s --ref data\\in\\<file>)",
        ctx.rel(ctx.run_dir), ctx.run_name, ctx.run_name,
    )  # fmt: skip
    return [stage for stage in selected if stage != "compare"]


def execute_pipeline(ctx: RunContext, selected: list[str], stage_fns: Mapping[str, StageCallable]) -> dict[str, dict]:
    """Run ``selected`` (already in order) calling ``stage_fns[name](ctx)``; returns ``{stage: result}``.

    Stops at the first failure with a :class:`PipelineError` naming the stage. Without a reference table, ``compare`` is
    skipped when other stages run (:func:`skip_compare_without_reference`).
    """
    selected = skip_compare_without_reference(ctx, selected)
    preflight(ctx, selected)
    results: dict[str, dict] = {}
    timings: list[tuple[str, float, str]] = []
    pipeline_start = time.monotonic()
    log.info(
        "run %r%s: stages %s | project %s | host %s",
        ctx.run_name,
        " [DRY RUN]" if ctx.dry_run else "",
        selected,
        ctx.project_dir,
        ctx.host_project_dir,
    )
    for name in selected:
        log.info("=== stage %s: start ===", name)
        started = time.monotonic()
        try:
            results[name] = stage_fns[name](ctx) or {}
        except KeyboardInterrupt:
            timings.append((name, time.monotonic() - started, "ABORTED"))
            _log_summary(timings, time.monotonic() - pipeline_start)
            raise
        except Exception as exc:
            elapsed = time.monotonic() - started
            timings.append((name, elapsed, "FAILED"))
            if not isinstance(exc, EXPECTED_ERRORS):
                log.exception("unexpected error in stage %s", name)
            log.error("=== stage %s: FAILED after %.1f s ===", name, elapsed)
            _log_summary(timings, time.monotonic() - pipeline_start)
            raise PipelineError(f"stage {name!r} failed: {exc}") from exc
        elapsed = time.monotonic() - started
        timings.append((name, elapsed, "ok"))
        log.info("=== stage %s: done in %.1f s ===", name, elapsed)
    _log_summary(timings, time.monotonic() - pipeline_start)
    return results


def _log_summary(timings: list[tuple[str, float, str]], total: float) -> None:
    lines = [f"  {name:<9s}{status:<8s}{seconds:8.1f} s" for name, seconds, status in timings]
    log.info("summary:\n%s\n  %-17s%8.1f s", "\n".join(lines), "total", total)


def run_pipeline(
    input: str | None,  # noqa: A002  (config key name)
    reference: str | None,
    run_name: str,
    stages: list[str],
    dry_run: bool,
    dfin: Callable[..., Any],
    split: Callable[..., Any],
    smartqsm: Callable[..., Any],
    arbor: Callable[..., Any],
    compare: Callable[..., Any],
) -> None:
    """hydra-zen task function: build the :class:`RunContext`, run the selected stages, exit 1 on failure.

    ``dfin`` ... ``compare`` arrive from hydra-zen as ``functools.partial`` objects of the ``run_<stage>``
    functions with the config values bound; each is then called with the context. An interrupt (Ctrl+C, or SIGTERM
    from ``docker stop`` once :func:`install_termination_handler` ran, as :func:`lidar_bridge.configs.main` does) stops
    the running stage container, logs one ``interrupted`` line (no traceback) and exits with code 130.
    """
    try:
        selected = select_stages(list(stages) if stages is not None else None)
        ctx = RunContext.from_env(run_name, input=input, reference=reference, dry_run=bool(dry_run))
        stage_fns = {"dfin": dfin, "split": split, "smartqsm": smartqsm, "arbor": arbor, "compare": compare}
        execute_pipeline(ctx, selected, stage_fns)
    except KeyboardInterrupt as exc:
        reason = str(exc) or "Ctrl+C"
        log.error("interrupted (%s): run %r aborted, exit code %d", reason, run_name, EXIT_INTERRUPTED)
        sys.exit(EXIT_INTERRUPTED)
    except (PipelineError, PathOutsideProjectError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)

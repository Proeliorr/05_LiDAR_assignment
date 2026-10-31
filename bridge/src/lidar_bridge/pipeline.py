"""The pipeline runner: executes the selected stages in the fixed order, with timing and clear failures."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from lidar_bridge.paths import PathOutsideDataError, RunContext
from lidar_bridge.runner import ToolError
from lidar_bridge.stages import STAGE_ORDER, StageError

log = logging.getLogger("lidar_bridge.pipeline")

StageCallable = Callable[[RunContext], Any]

# Failures that are reported as a one-line message (everything else also gets a traceback in the log).
EXPECTED_ERRORS = (StageError, ToolError, PathOutsideDataError, FileNotFoundError, NotImplementedError)


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

    Ctrl+C already does. Mapping ``docker stop`` to ``KeyboardInterrupt`` runs the same cleanup as Ctrl+C
    (:func:`lidar_bridge.runner.run_tool` kills the running tool) and ends the run with exit code :data:`EXIT_INTERRUPTED`.
    Returns False (and installs nothing) outside the main thread.
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
    """Validate a ``--stages`` selection and return it in the FIXED pipeline order (duplicates dropped).

    ``compare,dfin`` still runs dfin first. A comma separated string is accepted too.
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
            problems.append("stage dfin needs an input point cloud, e.g. in/<file>.laz")
        elif not ctx.input.is_file():
            problems.append(f"input file not found: {ctx.input}")
    if "compare" in selected:
        if ctx.reference is None:
            problems.append("stage compare needs a reference table: --ref in/<file>.csv|xlsx")
        elif not ctx.reference.is_file():
            problems.append(f"reference file not found: {ctx.reference}")
    if problems:
        if ctx.dry_run:
            for problem in problems:
                log.warning("[dry-run] %s", problem)
        else:
            raise PipelineError("; ".join(problems))


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
        "ready, compare without re-running anything: --stages compare --run %s --ref in/<file>",
        ctx.rel(ctx.run_dir), ctx.run_name,
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
        "run %r%s: stages %s | data %s",
        ctx.run_name,
        " [DRY RUN]" if ctx.dry_run else "",
        selected,
        ctx.data_dir,
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


def _log_to_file(ctx: RunContext) -> logging.Handler | None:
    """Also write the log to ``<run>/pipeline.log`` (appended, a run may be resumed); nothing in a dry run."""
    if ctx.dry_run:
        return None
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(ctx.run_dir / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def run_pipeline(
    input: str | None,  # noqa: A002  (same name as the command-line argument)
    reference: str | None,
    run_name: str,
    stages: list[str],
    dry_run: bool,
    stage_fns: Mapping[str, StageCallable],
) -> None:
    """Build the :class:`RunContext`, run the selected stages, exit 1 on failure.

    ``stage_fns`` holds the ``run_<stage>`` functions with their parameters bound; each is called with the context. An interrupt
    (Ctrl+C, or SIGTERM from ``docker stop`` once :func:`install_termination_handler` ran) kills the running tool, logs one
    ``interrupted`` line (no traceback) and exits with code 130.
    """
    handler = None
    try:
        selected = select_stages(stages)
        ctx = RunContext.from_env(run_name, input=input, reference=reference, dry_run=bool(dry_run))
        handler = _log_to_file(ctx)
        execute_pipeline(ctx, selected, stage_fns)
    except KeyboardInterrupt as exc:
        reason = str(exc) or "Ctrl+C"
        log.error("interrupted (%s): run %r aborted, exit code %d", reason, run_name, EXIT_INTERRUPTED)
        sys.exit(EXIT_INTERRUPTED)
    except (ValueError, RuntimeError) as exc:  # PipelineError, a bad path or run name, a missing data folder
        log.error("%s", exc)
        sys.exit(1)
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()

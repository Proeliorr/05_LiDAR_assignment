"""Stage 3b (WP6b): DBH / height / volume of every 3DFin tree with the R package ``arbor`` in the ``lidar-arbor`` container.

Contract
--------
Reads ``data/out/<run>/split/tree_map.csv`` and, depending on ``mode``,

``per_tree``  ``split/trees/tree_XXXX.laz`` (one QSM per tree cloud)  ->  ``arbor/metrics.csv``, log ``arbor/arbor.log``
``plot``      ``dfin/<stem>_tree_ID_dist_axes.las`` (arbor's whole scene pipeline once on the plot, the trees found by arbor are
              matched one-to-one to the 3DFin ordinals by the nearest root)  ->  ``arbor/metrics_plot.csv``, ``arbor/arbor_plot.log``

The compare stage reads ``arbor/metrics.csv`` only, so ``per_tree`` (the default) feeds it and the plot-mode table never
silently replaces it; to compare the plot run too, ``compare`` has to be pointed at ``arbor/metrics_plot.csv`` (a pipeline named
e.g. ``arbor_plot``). The two modes live side by side in ``arbor/`` and do not delete each other.

ONE container runs all trees (R start-up and package loading cost ~10 s)::

    docker run --rm --network none -e OMP_NUM_THREADS=<threads> \
        --mount ...data:/data --mount ...bridge/r:/bridge_r,readonly lidar-arbor \
        /bridge_r/arbor_metrics.R --mode per_tree --trees_dir /data/out/<run>/split/trees \
        --tree_map /data/out/<run>/split/tree_map.csv --out_csv /data/out/<run>/arbor/metrics.csv --threads <threads> [...]

The R script (``bridge/r/arbor_metrics.R``, its header documents the arbor chain and every height reference) appends one row to
the CSV after EVERY tree, so an interrupted run keeps what it did and ``resume=True`` continues with the missing ordinals.

per_tree chain (chosen on all 123 trees of the dev_full plot, numbers in the R header): 1000 synthetic ground points on the 3DFin
DTM around the tree cloud -> ``segment_ground`` -> ``wood_likelihood`` -> ``segment_semantic`` -> ``qsm``. The cloud is used at full
density (no thinning). The DBH is measured 1.3 m above the TRUE ground (not 1.3 m from arbor's QSM root, which is prolonged to
its own ground and ends up at 1.0 - 1.4 m), and ``height_m`` is also measured from the true ground.
plot chain: arbor's README scene pipeline unchanged (homogeneization, own ground, seeds, instance segmentation, ``qsf``).

metrics.csv (exact columns, in this order; ``compare`` reads the first five by name, extras are ignored)
-------------------------------------------------------------------------------------------------------
``ordinal, dbh_m, height_m, status, seconds, base_z0_m, dbh_height_above_ground_m, volume_m3, qsm_height_m, root_x, root_y,
note, mode, match_dist_m``

* ``dbh_m`` metres, from arbor's QSM at a measuring point 1.3 m above the TRUE ground (3DFin's DTM, ``Z - Z0``);
  ``dbh_height_above_ground_m`` is the height actually achieved (1.300 +- 0.005; the note says ``bh_not_converged`` otherwise).
* ``height_m`` metres: top of the QSM above the ground at the stem base (true ground, 3DFin DTM). ``qsm_height_m`` is arbor's own
  ``qsm_stats()$H`` (measured from the first stem cylinder, i.e. 0.2 - 0.5 m shorter: arbor does not count its prolongation to
  the ground).
* ``dbh_m`` and ``height_m`` are EMPTY unless ``status == ok``. A DBH that arbor only ESTIMATED (``Small tree allometry``, ``No
  valid measure``) is kept with ``status = ok`` and the arbor message in ``note`` (compare accepts ``ok`` only).
* ``status``: ``ok`` | ``no_dbh`` (QSM without a usable DBH) | ``failed`` (no QSM; in plot mode also: no arbor tree within
  ``max_match_dist`` of the 3DFin tree) | ``skipped_low_points`` (flagged in tree_map.csv) | ``error`` (unexpected R error, the
  message is in ``note``) | ``estimated_dbh`` (reserved, never written).
* ``seconds``: wall time of the tree (per_tree); plot: total scene-run time / number of arbor trees.
* ``mode``: ``per_tree`` | ``plot``; ``match_dist_m``: plot mode, distance arbor root - 3DFin (x_m, y_m).

Stale results
-------------
The rows of ``metrics.csv`` are only valid for the split they were computed on and for the settings that change what arbor
computes. Before the container starts the stage writes ``arbor/source_fingerprint.txt`` (plot mode:
``arbor/source_fingerprint_plot.txt``, so that the two modes do not overwrite each other's record): line 1 = hash of
``split/tree_map.csv`` (changes whenever dfin / split change), line 2 = the settings, ``mode=per_tree homogenize=false
params=<hash|none> skip_low_points=true`` (plot: ``mode=plot max_match_dist=1 params=<hash|none> skip_low_points=true``).
``compare`` reads the file to skip stale metrics. A resume only continues a table whose stored ``(fingerprint, settings)`` equal
the current ones; otherwise (or when a non-empty table has no fingerprint at all) the old table and log are deleted with a WARNING
and every selected tree is computed again. ``force=True`` always starts from scratch.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lidar_bridge.docker_runner import DockerRunError, require_image, run_container
from lidar_bridge.paths import RunContext
from lidar_bridge.stages._common import FINGERPRINT_FILE, StageError, read_fingerprint, tree_map_fingerprint, write_fingerprint

log = logging.getLogger("lidar_bridge.stages.arbor")

ARBOR_MODES = ("per_tree", "plot")
#: The R script (relative to the project) and where it is mounted (read-only) in the container.
R_SCRIPT_DIR = Path("bridge/r")
R_SCRIPT = "arbor_metrics.R"
R_MOUNT = "/bridge_r"
DFIN_CLOUD_SUFFIX = "_tree_ID_dist_axes.las"

#: Exact column order of the metrics file (must equal CONTRACT_COLS in ``arbor_metrics.R``).
METRICS_COLUMNS = [
    "ordinal",
    "dbh_m",
    "height_m",
    "status",
    "seconds",
    # extras
    "base_z0_m",
    "dbh_height_above_ground_m",
    "volume_m3",
    "qsm_height_m",
    "root_x",
    "root_y",
    "note",
    "mode",
    "match_dist_m",
]
OK, NO_DBH, FAILED, ESTIMATED_DBH, SKIPPED, ERROR = "ok", "no_dbh", "failed", "estimated_dbh", "skipped_low_points", "error"
STATUSES = (OK, NO_DBH, FAILED, ESTIMATED_DBH, SKIPPED, ERROR)

#: mode -> (metrics file, log file) inside ``arbor/``
OUTPUT_FILES = {"per_tree": ("metrics.csv", "arbor.log"), "plot": ("metrics_plot.csv", "arbor_plot.log")}

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
#: arbor / cli progress bars ("Anisotropy: 14% | 2.9s ETA (16 threads)", "  |=====   | 45%") redrawn with a carriage return
_PROGRESS = re.compile(r"(^\s*[\w .()/-]*?:\s*\d{1,3}%\s*\|)|(\bETA\b)|(^\s*\|[=\- >#]*\|\s*\d{1,3}%)|(^\s*\d{1,3}%\s*$)")
_PARAM_KEY = re.compile(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$")


# ---------------------------------------------------------------------------------------------------------
# log cleaning
# ---------------------------------------------------------------------------------------------------------
def clean_log_lines(text: str | None) -> list[str]:
    """Container output -> readable log lines: ANSI colours stripped, ``\\r``-redrawn progress bars dropped.

    A "line" from a progress-bar run is ``frame1\\rframe2\\r...\\rframeN``; every fragment that looks like a progress frame is
    removed and the remaining text is kept, so ``"Anisotropy: 5% | ETA\\rAnisotropy: 100% | ETA\\r[10:00:00] done"`` becomes
    ``["[10:00:00] done"]``.
    """
    out: list[str] = []
    for raw in (text or "").splitlines():
        for fragment in _ANSI.sub("", raw).split("\r"):
            fragment = fragment.rstrip()
            if fragment.strip() and not _PROGRESS.search(fragment):
                out.append(fragment)
    return out


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
def flatten_params(params: Mapping[str, Any] | None) -> str:
    """arbor parameter overrides -> ``"qsm.smooth_steps=15,qsm.max_d=0.2"`` (the ``--params`` value; empty if none).

    Accepts ``{"qsm": {"smooth_steps": 15}}`` and ``{"qsm.smooth_steps": 15}``. Keys are ``<section>.<name>`` of
    ``arbor::arbor_parameters_default``; the R script rejects unknown ones.
    """
    items: dict[str, str] = {}

    def put(key: str, value: Any) -> None:
        if not _PARAM_KEY.match(key):
            raise StageError(f"arbor.params: key {key!r} must look like '<section>.<name>', e.g. qsm.skeleton_node_distance")
        if isinstance(value, Mapping) or isinstance(value, (list, tuple, set)):
            raise StageError(f"arbor.params: {key!r} must be a scalar, got {type(value).__name__}")
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        if "," in text or "=" in text or not text:
            raise StageError(f"arbor.params: value {text!r} of {key!r} must be non-empty and contain no ',' or '='")
        items[key] = text

    for name, value in (params or {}).items():
        if isinstance(value, Mapping):
            for inner, inner_value in value.items():
                put(f"{name}.{inner}", inner_value)
        else:
            put(str(name), value)
    return ",".join(f"{k}={v}" for k, v in sorted(items.items()))


@dataclass(frozen=True)
class Tree:
    """One row of ``split/tree_map.csv`` (only what this stage needs)."""

    ordinal: int
    low_points: bool = False


def read_tree_map(path: Path) -> list[Tree]:
    """The trees of ``tree_map.csv`` in file order. Raises :class:`StageError` for an unreadable / unusable file."""
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise StageError(f"cannot read {path}: {exc}") from exc
    if "ordinal" not in raw.columns:
        raise StageError(f"{path} has no 'ordinal' column; columns found: {list(raw.columns)}")
    trees = []
    for _, row in raw.iterrows():
        try:
            ordinal = int(row["ordinal"])
        except ValueError as exc:
            raise StageError(f"{path}: invalid ordinal {row['ordinal']!r}") from exc
        low = str(row.get("low_points", "")).strip().lower() in ("true", "1", "t", "yes")
        trees.append(Tree(ordinal, low))
    if not trees:
        raise StageError(f"{path} lists no trees")
    return trees


def select_ordinals(available: list[int], trees: list[int] | None, limit: int | None = None) -> list[int]:
    """Ordinals to process: ``trees`` (validated against ``available``, sorted, duplicates dropped) or all; then ``limit``."""
    if trees is not None:
        try:
            wanted = sorted({int(t) for t in trees})
        except (TypeError, ValueError) as exc:
            raise StageError(f"arbor.trees must be a list of integers, got {trees!r}") from exc
        unknown = [t for t in wanted if t not in set(available)]
        if unknown:
            raise StageError(f"arbor.trees: ordinal(s) {unknown} are not in tree_map.csv (it holds {min(available)}..{max(available)})")
        selected = wanted
    else:
        selected = sorted(available)
    if not selected:
        raise StageError("arbor.trees is empty: nothing to do")
    if limit is not None:
        selected = selected[:limit]
    return selected


def find_dfin_cloud(ctx: RunContext) -> Path:
    """``dfin/<stem>_tree_ID_dist_axes.las`` of this run (plot mode input)."""
    dfin_dir = ctx.stage_dir("dfin")
    if not dfin_dir.is_dir():
        raise StageError(f"no 3DFin results in {dfin_dir}: run the dfin stage first (stages=[dfin,arbor], mode=plot needs its cloud)")
    if ctx.input is not None:
        candidate = dfin_dir / f"{ctx.input.stem}{DFIN_CLOUD_SUFFIX}"
        if candidate.is_file():
            return candidate
    found = sorted(dfin_dir.glob(f"*{DFIN_CLOUD_SUFFIX}"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise StageError(f"{dfin_dir} contains no *{DFIN_CLOUD_SUFFIX} file: run the dfin stage first")
    raise StageError(f"several 3DFin clouds in {dfin_dir} ({[p.name for p in found]}); pass input=<the input file> to choose one")


# ---------------------------------------------------------------------------------------------------------
# metrics.csv
# ---------------------------------------------------------------------------------------------------------
def read_metrics(path: Path) -> pd.DataFrame:
    """Read a metrics file and check its STRUCTURE (exact header, integer unique ordinals, known statuses).

    Raises :class:`StageError` otherwise. Value-level contract violations are reported by :func:`check_metrics`.
    """
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise StageError(f"{path} is empty") from exc
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise StageError(f"cannot read {path}: {exc}") from exc
    if list(df.columns) != METRICS_COLUMNS:
        raise StageError(f"{path}: columns are {list(df.columns)}, expected {METRICS_COLUMNS}")
    try:
        ordinals = df["ordinal"].astype(int)
    except ValueError as exc:
        raise StageError(f"{path}: non-integer ordinal ({exc})") from exc
    duplicated = sorted(set(ordinals[ordinals.duplicated()]))
    if duplicated:
        raise StageError(f"{path}: duplicate ordinal(s) {duplicated[:10]}")
    bad = sorted(set(df["status"]) - set(STATUSES))
    if bad:
        raise StageError(f"{path}: unknown status value(s) {bad}; allowed: {list(STATUSES)}")
    df["ordinal"] = ordinals
    return df


def _number(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def check_metrics(df: pd.DataFrame) -> list[str]:
    """Value-level contract problems of a table from :func:`read_metrics` (empty list = clean)."""
    problems = []
    for _, row in df.iterrows():
        ordinal, status = row["ordinal"], row["status"]
        dbh, height = _number(row["dbh_m"]), _number(row["height_m"])
        if status == OK:
            if not (math.isfinite(dbh) and dbh > 0):
                problems.append(f"ordinal {ordinal}: status ok but dbh_m={row['dbh_m']!r}")
            if row["height_m"] != "" and not (math.isfinite(height) and height > 0):
                problems.append(f"ordinal {ordinal}: status ok but height_m={row['height_m']!r}")
        elif row["dbh_m"] != "" or row["height_m"] != "":
            problems.append(f"ordinal {ordinal}: status {status} must have empty dbh_m and height_m")
        if not math.isfinite(_number(row["seconds"])):
            problems.append(f"ordinal {ordinal}: seconds is empty")
    return problems


def _existing_ordinals(path: Path) -> list[int]:
    """Ordinals already in ``path`` (``[]`` if the file does not exist); a file of another layout is an error."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        return sorted(read_metrics(path)["ordinal"].tolist())
    except StageError as exc:
        raise StageError(f"cannot resume from {path}: {exc}. Use force=True to discard it.") from exc


# ---------------------------------------------------------------------------------------------------------
# source fingerprint (stale-result guard)
# ---------------------------------------------------------------------------------------------------------
def fingerprint_file(mode: str) -> str:
    """Name (inside ``arbor/``) of the fingerprint file of ``mode``: ``source_fingerprint.txt`` for ``per_tree`` (the table
    ``compare`` reads), ``source_fingerprint_plot.txt`` for ``plot`` (own file: the two tables can belong to different splits)."""
    if mode == "per_tree":
        return FINGERPRINT_FILE
    stem, dot, suffix = FINGERPRINT_FILE.rpartition(".")
    return f"{stem}_{mode}{dot}{suffix}"


def settings_tag(mode: str, *, homogenize: bool, params_arg: str, max_match_dist: float, skip_low_points: bool) -> str:
    """The settings that change the rows of ``mode`` (``extra`` of the fingerprint); only those the mode really uses."""
    params = hashlib.sha256(params_arg.encode("utf-8")).hexdigest()[:8] if params_arg else "none"
    flag = "true" if skip_low_points else "false"
    if mode == "per_tree":
        return f"mode=per_tree homogenize={'true' if homogenize else 'false'} params={params} skip_low_points={flag}"
    return f"mode={mode} max_match_dist={float(max_match_dist):g} params={params} skip_low_points={flag}"


def _read_source(out_dir: Path, mode: str) -> tuple[str, str] | None:
    if mode == "per_tree":
        return read_fingerprint(out_dir)
    try:  # same two-line format as read_fingerprint, in the plot mode's own file
        lines = (out_dir / fingerprint_file(mode)).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines or not lines[0].strip():
        return None
    return lines[0].strip(), (lines[1].strip() if len(lines) > 1 else "")


def _write_source(out_dir: Path, mode: str, fingerprint: str, tag: str) -> None:
    if mode == "per_tree":
        write_fingerprint(out_dir, fingerprint, extra=tag)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fingerprint_file(mode)).write_text(f"{fingerprint}\n{tag}\n", encoding="utf-8")


def stale_reason(stored: tuple[str, str] | None, current: tuple[str, str]) -> str | None:
    """Why a table with the stored ``(fingerprint, settings)`` cannot be continued under ``current`` (``None`` = it can)."""
    if stored is None:
        return "the existing arbor results have no source fingerprint (made before it was recorded), so their split is unknown"
    if stored[0] != current[0]:
        return "split/dfin changed since the existing arbor results were made"
    if stored[1] != current[1]:
        return f"arbor settings changed since the existing results were made (was: {stored[1] or '-'}; now: {current[1] or '-'})"
    return None


# ---------------------------------------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------------------------------------
def run_arbor(
    ctx: RunContext,
    *,
    mode: str = "per_tree",
    image: str = "lidar-arbor",
    params: dict = {},  # noqa: B006  (never mutated; hydra-zen turns it into a config field)
    timeout_s: int = 7200,
    trees: list[int] | None = None,
    limit: int | None = None,
    threads: int = 8,
    skip_low_points: bool = True,
    resume: bool = True,
    force: bool = False,
    max_match_dist: float = 1.0,
    homogenize: bool = False,
) -> dict:
    """Run arbor (image ``lidar-arbor``) on the 3DFin trees and write ``arbor/metrics.csv`` (see the module docstring).

    Parameters
    ----------
    mode
        ``per_tree``: one QSM per ``split/trees/tree_XXXX.laz`` (default; feeds ``compare``). ``plot``: arbor's scene
        pipeline once on the whole 3DFin cloud (needs the dfin stage), results matched to the ordinals; written to
        ``metrics_plot.csv``.
    image
        Docker image tag (ENTRYPOINT ``Rscript``).
    params
        arbor parameter overrides ``{"qsm": {"smooth_steps": 20}}`` / ``{"qsm.smooth_steps": 20}`` (keys of
        ``arbor_parameters_default``; CLI: ``+arbor.params.qsm.smooth_steps=20``).
    timeout_s
        Limit of the single container run, seconds (0 = no limit). A full 123-tree per_tree run takes ~5 min.
    trees
        Only these ordinals (CLI: ``arbor.trees=[3,8,10]``); default all trees of ``tree_map.csv``.
    limit
        Process at most this many trees (the first ones; per_tree: of those not yet done).
    threads
        OpenMP threads of arbor (``OMP_NUM_THREADS`` of the container) and lidR/data.table threads.
    skip_low_points
        Trees flagged ``low_points`` in ``tree_map.csv`` get status ``skipped_low_points`` instead of being processed.
    resume
        per_tree: keep the rows already in ``metrics.csv`` and process only the missing ordinals; nothing to do -> no container.
        plot: reuse the table if it already covers every selected ordinal. ``False`` starts from scratch. Only a table made on the
        current split with the current settings is resumed (module docstring, "Stale results"); any other is discarded with a
        warning and recomputed.
    force
        Discard an existing table of this mode and run everything.
    max_match_dist
        plot mode: an arbor tree is matched to a 3DFin tree only if its root is at most this far (m) from the 3DFin position.
    homogenize
        per_tree: arbor's voxel homogeneization (2 cm) before the QSM chain. Default off = full point density; on, the
        chain is ~1.6x faster with an equal DBH accuracy on 12 test trees (differences are within noise).

    Returns
    -------
    dict
        ``{"dir", "metrics", "log", "mode", "n_trees", "counts", "seconds", "skipped"}``; paths relative to the project.
    """
    if mode not in ARBOR_MODES:
        raise StageError(f"arbor.mode must be one of {ARBOR_MODES}, got {mode!r}")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise StageError(f"arbor.threads must be an integer >= 1, got {threads!r}")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise StageError(f"arbor.limit must be an integer >= 1 or null, got {limit!r}")
    if not (isinstance(max_match_dist, (int, float)) and max_match_dist > 0):
        raise StageError(f"arbor.max_match_dist must be > 0, got {max_match_dist!r}")
    params_arg = flatten_params(params)

    out_dir = ctx.stage_dir("arbor")
    metrics_name, log_name = OUTPUT_FILES[mode]
    metrics_path, log_path = out_dir / metrics_name, out_dir / log_name
    split_dir = ctx.stage_dir("split")
    tree_map_path = split_dir / "tree_map.csv"
    trees_dir = split_dir / "trees"
    script_dir = ctx.project_dir / R_SCRIPT_DIR

    # -- inputs ----------------------------------------------------------------------------------------------
    problems = []
    if not tree_map_path.is_file():
        problems.append(f"{ctx.rel(tree_map_path)} not found: run the split stage first (stages=[split,arbor])")
    if mode == "per_tree" and not trees_dir.is_dir():
        problems.append(f"{ctx.rel(trees_dir)} not found: run the split stage first")
    if not (script_dir / R_SCRIPT).is_file():
        problems.append(f"{R_SCRIPT_DIR / R_SCRIPT} not found in the project")
    plot_cloud: Path | None = None
    if mode == "plot":
        try:
            plot_cloud = find_dfin_cloud(ctx)
        except StageError as exc:
            problems.append(str(exc))
    if problems and not ctx.dry_run:
        raise StageError("; ".join(problems))
    for problem in problems:
        log.warning("[dry-run] %s", problem)

    # -- selection / resume ------------------------------------------------------------------------------------
    fresh = force or not resume  # discard an existing table of this mode and run everything
    selected: list[int] | None = None  # ordinals in scope (None only in a dry run without tree_map.csv)
    if tree_map_path.is_file():
        selected = select_ordinals([t.ordinal for t in read_tree_map(tree_map_path)], trees)
    existing = [] if fresh else _existing_ordinals(metrics_path)  # fails on a table of another layout
    fingerprint = tree_map_fingerprint(split_dir)  # None only in a dry run without tree_map.csv
    tag = settings_tag(mode, homogenize=homogenize, params_arg=params_arg, max_match_dist=max_match_dist, skip_low_points=skip_low_points)
    if existing and fingerprint is not None:  # rows of another split / other settings must neither be resumed nor reported
        why = stale_reason(_read_source(out_dir, mode), (fingerprint, tag))
        if why is not None:
            log.warning(
                "%sarbor %s: %s: recomputing all trees (the old %s and %s are discarded)",
                "[dry-run] " if ctx.dry_run else "", mode, why, metrics_name, log_name,
            )  # fmt: skip
            existing, fresh = [], True
    if mode == "per_tree":
        done = [o for o in (selected or []) if o in set(existing)]
        todo = [o for o in (selected or []) if o not in set(done)]
        if limit is not None:
            todo = todo[:limit]
        expected = sorted(set(done) | set(todo))
        reuse = bool(existing)  # resume the existing table (R skips the ordinals it holds)
    else:  # plot: one run that rewrites the whole table; reused only if it already covers the selection
        target = (selected or [])[:limit]
        done = [o for o in target if o in set(existing)]
        todo = [o for o in target if o not in set(done)]
        expected = sorted(target)
        reuse = False
    if selected is not None and not todo and not fresh and metrics_path.is_file():
        log.info("arbor %s: %s already holds all %d selected tree(s); nothing to run (force=True re-runs)", mode, metrics_path.name, len(done))
        return _summary(ctx, out_dir, metrics_path, log_path, mode, read_metrics(metrics_path), 0.0, skipped=True)
    if (fresh or mode == "plot") and not ctx.dry_run:
        for stale in (metrics_path, log_path):
            if stale.exists():
                stale.unlink()
    if reuse and params_arg:
        log.warning("%s will be extended although arbor.params is set; use force=True if it holds rows from other parameters", metrics_path.name)

    # -- command -----------------------------------------------------------------------------------------------
    if not ctx.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if fingerprint is not None:  # BEFORE the run: rows R appends belong to this split even if the run is interrupted
            _write_source(out_dir, mode, fingerprint, tag)
    mounts = [ctx.mount(ctx.data_dir, "/data"), ctx.mount(script_dir, R_MOUNT, readonly=True)]
    args = [f"{R_MOUNT}/{R_SCRIPT}", "--mode", mode]
    if mode == "per_tree":
        args += ["--trees_dir", ctx.container_path(trees_dir)]
    else:
        placeholder = ctx.stage_dir("dfin") / f"<stem>{DFIN_CLOUD_SUFFIX}"
        args += ["--plot_las", ctx.container_path(plot_cloud or placeholder), "--max_match_dist", str(max_match_dist)]
    args += ["--tree_map", ctx.container_path(tree_map_path), "--out_csv", ctx.container_path(metrics_path)]
    args += ["--threads", str(threads), "--skip_low_points", "true" if skip_low_points else "false"]
    if trees is not None:
        args += ["--trees", ",".join(str(o) for o in (selected if selected is not None else sorted({int(t) for t in trees})))]
    if limit is not None:
        args += ["--limit", str(limit)]
    if reuse:
        args.append("--resume")
    if homogenize and mode == "per_tree":
        args.append("--homogenize")
    if params_arg:
        args += ["--params", params_arg]

    require_image(image, dry_run=ctx.dry_run)
    n_run = len(todo) if selected is not None else None
    log.info(
        "arbor %s: %s tree(s) to process%s -> %s", mode, n_run if n_run is not None else "?",
        f" ({len(done)} already done)" if done else "", ctx.rel(metrics_path),
    )

    started = time.monotonic()
    try:
        result = run_container(
            image,
            args,
            mounts,
            network="none",  # arbor needs no network; also makes its update check fail fast
            env={"OMP_NUM_THREADS": str(threads)},
            timeout_s=timeout_s or None,
            dry_run=ctx.dry_run,
            log_prefix="arbor",
        )
    except DockerRunError as exc:
        _write_log(log_path, clean_log_lines(exc.output), append=True)
        tail = "\n".join(clean_log_lines(exc.output)[-25:])
        partial = f" Rows already written stay in {ctx.rel(metrics_path)}; rerun with resume=True to continue." if metrics_path.is_file() else ""
        raise StageError(f"arbor container failed: {str(exc).splitlines()[0]}\n--- last output ---\n{tail}\n{partial}".rstrip()) from exc
    seconds = time.monotonic() - started

    if ctx.dry_run:
        return {"dir": str(ctx.rel(out_dir)), "metrics": str(ctx.rel(metrics_path)), "mode": mode, "dry_run": True}

    lines = clean_log_lines(result.stdout)
    _write_log(log_path, lines, append=True)
    if not metrics_path.is_file():
        raise StageError(
            f"arbor produced no {metrics_name} in {out_dir}\n--- last output ---\n" + "\n".join(lines[-25:])
        )
    table = read_metrics(metrics_path)
    missing = sorted(set(expected or []) - set(table["ordinal"]))
    if missing:
        raise StageError(
            f"{metrics_name} has no row for {len(missing)} selected tree(s): {missing[:15]}{' ...' if len(missing) > 15 else ''} "
            f"(see {ctx.rel(log_path)})"
        )
    for problem in check_metrics(table)[:20]:
        log.warning("%s: %s", metrics_name, problem)
    summary = _summary(ctx, out_dir, metrics_path, log_path, mode, table, seconds, skipped=False)
    log.info("arbor %s finished in %.1f s: %s", mode, seconds, ", ".join(f"{k}={v}" for k, v in summary["counts"].items()))
    return summary


def _write_log(path: Path, lines: list[str], *, append: bool) -> None:
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write("\n".join(lines) + "\n")


def _summary(ctx: RunContext, out_dir: Path, metrics_path: Path, log_path: Path, mode: str, table: pd.DataFrame, seconds: float, *, skipped: bool) -> dict:
    return {
        "dir": str(ctx.rel(out_dir)),
        "metrics": str(ctx.rel(metrics_path)),
        "log": str(ctx.rel(log_path)),
        "mode": mode,
        "n_trees": len(table),
        "counts": dict(Counter(table["status"])),
        "seconds": round(seconds, 1),
        "skipped": skipped,
    }

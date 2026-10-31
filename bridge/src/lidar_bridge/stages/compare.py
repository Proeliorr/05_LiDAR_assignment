"""Last stage: compare DBH and total tree height of every pipeline with the plot reference measurements.

Trees are matched by the TREE NUMBER ONLY (reference tree number == pipeline ordinal tree number 1..N == 3DFin's XLSX
row label ``T<k>``); no coordinates or distances are involved. Only DBH and total height are compared. All numbers
are in **metres** (the reference is converted from its own units, see :mod:`lidar_bridge.reference`).

Inputs (``<run>`` = ``out/<run_name>`` in the data folder)
-------------------------------------------
reference table
    ``ctx.reference`` (``--ref in/<file>.csv|xlsx``); columns / units from the stage parameters, the
    optional ``columns_file`` YAML, or auto-detection (see :mod:`lidar_bridge.reference`).

pipeline ``3dfin``: ``<run>/split/tree_map.csv`` (written by the split stage)
    ``ordinal`` (int), ``dbh_m`` (empty when 3DFin found no DBH), ``th_m`` (corrected tree height, m); other
    columns are ignored.

pipeline ``smartqsm:<config>``: ``<run>/smartqsm/<config>/metrics.csv`` (one per SmartQSM config; several runs over
``smartqsm.config`` creates several, each is compared under the name ``smartqsm:<config>``, also if there is one)
pipeline ``arbor``: ``<run>/arbor/metrics.csv``
    columns ``ordinal,dbh_m,height_m,status,seconds`` (extra columns are ignored). A row counts as a valid estimate of
    a metric only if ``status`` is ``ok`` (case-insensitive; a file without a ``status`` column is taken as all ok)
    and the value is a finite number > 0; otherwise it is "missing" for that metric.

A pipeline whose metrics file does not exist is skipped with a warning; if none exists the stage fails. A metric
for which the reference has no column is skipped (warning); reference cells that are empty or <= 0 are "no reference".

Do the results belong to the current split? (``allow_stale``)
    The tree numbers are only meaningful for the split the QSM stages ran on, and a re-split renumbers the trees. The
    smartqsm / arbor stages therefore store ``source_fingerprint.txt`` (a hash of ``split/tree_map.csv``, see
    :func:`lidar_bridge.stages._common.tree_map_fingerprint`) next to their ``metrics.csv``, and compare checks it
    against the current ``<run>/split/tree_map.csv`` for ``smartqsm:<config>`` and ``arbor`` (``3dfin`` is read from
    the current tree map and needs no check):

    * equal: fine;
    * different: the pipeline is SKIPPED with a warning ("computed on a different split, re-run the stage"; it is
      listed in the returned ``skipped``); ``allow_stale=True`` compares it anyway (still a warning, listed in the
      returned ``unverified``);
    * no fingerprint file (results written before this check existed) or no current ``split/tree_map.csv``: the
      pipeline is compared, with one warning ("cannot verify"), also listed in ``unverified``.

    Because ``3dfin`` always comes from the current tree map, a stale-only situation leaves a 3dfin-only comparison; the
    stage fails ("nothing to compare", with the reasons) only when no pipeline is left at all.

Quality flags (``id_confidence``, ``estimated``; ``exclude_ambiguous`` / ``exclude_estimated``)
    ``id_confidence``
        ``exact`` / ``resolved`` / ``ambiguous`` from ``split/tree_map.csv`` for every pipeline (``unknown``: no tree
        map / no such column / tree not in it). ``ambiguous``: the 3DFin tree number could not be aligned with the
        original numbering with confidence, so the pairing with the reference may be wrong for that tree.
    ``estimated`` (DBH only)
        True when the ``note`` column of the metrics row contains arbor's ``No valid measure`` or ``Small tree
        allometry`` (the DBH is an allometric estimate, not a measurement; the row still has status ok) or
        ``bh_not_converged`` (the DBH was taken off the 1.3 m breast height). Pipelines without a ``note`` column
        (3dfin) and SmartQSM (its notes never contain these phrases) are never flagged. Heights are never flagged.
    Both appear as columns in ``errors.csv``; ``summary.csv`` counts them among the paired trees (``n_ambiguous``,
    ``n_estimated``). By default such trees stay in the statistics and one warning per pipeline and metric gives the
    counts; ``exclude_ambiguous`` / ``exclude_estimated`` drop them from errors, statistics and plots (``n_excluded``
    in the summary counts the dropped pairs; they are then neither "missing" nor "extra").

Outputs (``<run>/compare/``, cleared at the start of a successful run)
----------------------------------------------------------------------
``errors.csv`` (long format, one row per paired tree and metric; metres)
    ``pipeline, metric (dbh|height), ordinal, ref, est, error (= est - ref), rel_error_pct (= 100 * error / ref),
    id_confidence, estimated`` (the last two: quality flags, see above)
``summary.csv`` (one row per pipeline and metric; metres unless ``_pct``)
    ``n`` paired trees; ``n_missing`` trees with a valid reference value but no valid estimate (absent from the
    metrics file, failed status, NaN / <= 0); ``n_extra`` valid estimates without a valid reference value (tree
    number not in the reference table or its cell is empty); ``n_ambiguous`` / ``n_estimated`` paired trees with
    ``id_confidence == ambiguous`` / an estimated DBH (0 for ``exclude_*``-d kinds); ``n_excluded`` pairs dropped by
    ``exclude_ambiguous`` / ``exclude_estimated``; then, over the ``n`` pairs, with e = est - ref:

    ``bias`` mean(e); ``MAE`` mean(|e|); ``RMSE`` sqrt(mean(e^2));
    ``rel_bias_pct`` / ``rel_RMSE_pct`` = 100 * bias / mean(ref) and 100 * RMSE / mean(ref) (relative to the mean
    reference value of the paired trees);
    ``R2`` = 1 - sum(e^2) / sum((ref - mean(ref))^2), the coefficient of determination of the estimates as a
    predictor of the reference **on the 1:1 line** (1 = perfect, 0 = no better than the mean, negative = worse;
    NaN for n < 2 or if all reference values are equal);
    ``slope`` / ``intercept`` of the OLS fit est ~ ref (intercept in metres; NaN for n < 2 or constant ref);
    ``R2_ols`` = squared Pearson correlation of est and ref (= R^2 of that OLS fit, ignores bias / slope errors);
    ``within_tol_pct`` = percentage of pairs with |e| <= ``tol_m`` (``compare.tolerance_dbh_cm`` /
    ``compare.tolerance_height_m``, stored in the column ``tol_m``).
``reference_used.csv``
    The parsed reference (``ordinal, dbh_m, height_m``), to check the column / unit interpretation.
``scatter_dbh.png``, ``scatter_height.png`` (``plots=True``)
    Estimate vs reference with the 1:1 line, one colour (and marker) per pipeline. DBH is drawn in cm, height in m.

The return value holds the paths (relative to the project) and the key numbers per pipeline and metric.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidar_bridge import reference as refmod  # module import: lidar_bridge.reference imports stages._common (cycle)
from lidar_bridge.paths import RunContext
from lidar_bridge.stages._common import StageError, read_fingerprint, tree_map_fingerprint

log = logging.getLogger("lidar_bridge.stages.compare")

METRICS = ("dbh", "height")
#: Name of the pipeline that is not a tool stage but the 3DFin values from ``split/tree_map.csv``.
PIPELINE_3DFIN = "3dfin"
ERROR_COLUMNS = ["pipeline", "metric", "ordinal", "ref", "est", "error", "rel_error_pct", "id_confidence", "estimated"]
SUMMARY_COLUMNS = [
    "pipeline", "metric", "n", "n_missing", "n_extra", "n_ambiguous", "n_estimated", "n_excluded", "bias", "MAE",
    "RMSE", "rel_bias_pct", "rel_RMSE_pct", "R2", "slope", "intercept", "R2_ols", "tol_m", "within_tol_pct",
]  # fmt: skip
#: ``note`` phrases (lower case) with which arbor says that the DBH of a row is not a clean measurement: allometric
#: estimate ("Small tree allometry", "No valid measure") or taken off the breast height (``bh_not_converged``).
ESTIMATED_DBH_NOTES = ("no valid measure", "small tree allometry", "bh_not_converged")
ID_AMBIGUOUS = "ambiguous"
ID_UNKNOWN = "unknown"
#: Display unit of a metric in the plots / log: (axis label, factor from metres).
PLOT_UNITS = {"dbh": ("DBH", "cm", 100.0), "height": ("Tree height", "m", 1.0)}
#: Categorical slots of the dataviz palette (validated order). Only the first three are validated for scatter
#: plots (all pairs on screen), so each pipeline also gets its own marker shape as a second encoding.
PIPELINE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PIPELINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


# ---------------------------------------------------------------------------------------------------------
# estimates of one pipeline
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Estimates:
    """Valid estimates of one pipeline: ``table`` is indexed by ordinal, columns ``dbh_m`` / ``height_m`` (NaN = none)
    and ``estimated_dbh`` (bool: the ``note`` column calls the DBH estimated, see :data:`ESTIMATED_DBH_NOTES`)."""

    name: str
    path: Path
    table: pd.DataFrame


def _valid_positive(values: pd.Series) -> pd.Series:
    return values.where(np.isfinite(values) & (values > 0))


def read_estimates(name: str, path: Path, *, height_col: str = "height_m") -> Estimates:
    """Read a pipeline's CSV (see the module docstring). ``height_col`` is ``th_m`` for 3DFin, ``height_m`` otherwise."""
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise StageError(f"{name}: {path} is empty") from exc
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise StageError(f"{name}: cannot read {path}: {exc}") from exc
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "ordinal" not in raw.columns:
        raise StageError(f"{name}: {path} has no 'ordinal' column; columns found: {list(raw.columns)}")
    present = {m: col for m, col in (("dbh", "dbh_m"), ("height", height_col)) if col in raw.columns}
    if not present:
        raise StageError(
            f"{name}: {path} has neither 'dbh_m' nor '{height_col}'; columns found: {list(raw.columns)}"
        )
    for metric, col in (("dbh", "dbh_m"), ("height", height_col)):
        if metric not in present:
            log.warning("%s: %s has no '%s' column, no %s estimates", name, path.name, col, metric)

    ordinals = raw["ordinal"].map(refmod.parse_tree_id)
    unusable = ordinals.isna()
    if unusable.any():
        log.warning("%s: ignoring %d row(s) of %s without a usable ordinal", name, int(unusable.sum()), path.name)
    ordinals = ordinals[~unusable].astype(int)
    if ordinals.duplicated().any():
        dup = sorted(set(ordinals[ordinals.duplicated(keep=False)]))
        raise StageError(f"{name}: duplicate ordinals {dup[:20]} in {path}; expected one row per tree")

    if "status" in raw.columns:
        ok = raw["status"].astype(str).str.strip().str.lower().eq("ok")
    else:
        ok = pd.Series(True, index=raw.index)
        log.debug("%s: %s has no status column, all rows taken as ok", name, path.name)

    table = pd.DataFrame(index=pd.Index(ordinals.to_numpy(), name="ordinal"))
    for metric, out in (("dbh", "dbh_m"), ("height", "height_m")):
        if metric not in present:
            table[out] = np.nan
            continue
        values = raw.loc[~unusable, present[metric]].map(refmod.parse_number).astype(float).to_numpy()
        values = _valid_positive(pd.Series(values, index=table.index))
        table[out] = values.where(ok[~unusable].to_numpy())
    if "note" in raw.columns:
        notes = raw.loc[~unusable, "note"].astype(str).str.lower().to_numpy()
        table["estimated_dbh"] = [any(phrase in note for phrase in ESTIMATED_DBH_NOTES) for note in notes]
    else:
        table["estimated_dbh"] = False
    return Estimates(name=name, path=path, table=table)


def read_id_confidence(tree_map: Path) -> pd.Series | None:
    """``id_confidence`` per ordinal from ``split/tree_map.csv`` (lower case; blank -> ``unknown``).

    ``None`` (with a warning) when the file has no such column, i.e. it was written by a split before the check existed.
    """
    try:
        raw = pd.read_csv(tree_map, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        log.warning("cannot read id_confidence from %s: %s", tree_map, exc)
        return None
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "ordinal" not in raw.columns or "id_confidence" not in raw.columns:
        log.warning(
            "%s has no id_confidence column (split from before the tree number check): ambiguous tree numbers cannot be "
            "flagged", tree_map.name,
        )
        return None
    ordinals = raw["ordinal"].map(refmod.parse_tree_id)
    usable = ordinals.notna()
    values = raw.loc[usable, "id_confidence"].str.strip().str.lower().replace("", ID_UNKNOWN)
    series = pd.Series(values.to_numpy(), index=pd.Index(ordinals[usable].astype(int).to_numpy(), name="ordinal"))
    return series[~series.index.duplicated(keep="first")]


def discover_pipelines(
    ctx: RunContext, *, allow_stale: bool = False
) -> tuple[list[tuple[str, Path, str]], list[str], dict[str, str]]:
    """``([(pipeline name, metrics csv, height column)], [reasons for what is skipped], {pipeline: warning})``.

    The third item holds a warning for every included ``smartqsm:<config>`` / ``arbor`` pipeline whose results could
    not be verified to belong to the current split (no fingerprint file, no current tree map, or stale and
    ``allow_stale``); stale pipelines without ``allow_stale`` are skipped and explained in the second item.
    """
    found: list[tuple[str, Path, str]] = []
    missing: list[str] = []
    unverified: dict[str, str] = {}

    def rel(p: Path) -> str:
        try:
            return str(ctx.rel(p))
        except ValueError:
            return str(p)

    tree_map = ctx.stage_dir("split") / "tree_map.csv"
    current = tree_map_fingerprint(ctx.stage_dir("split"))

    def belongs_to_current_split(name: str, stage_dir: Path) -> bool:
        stored = read_fingerprint(stage_dir)
        if current is None:
            unverified[name] = f"{name}: {rel(tree_map)} not found, cannot verify the results belong to the current split"
        elif stored is None:
            unverified[name] = f"{name}: no source fingerprint, cannot verify the results belong to the current split"
        elif stored[0] != current:
            if not allow_stale:
                missing.append(
                    f"{name} skipped: its results were computed on a different split (re-run the stage, or set "
                    "compare.allow_stale=true to compare them anyway)"
                )
                return False
            unverified[name] = (
                f"{name}: its results were computed on a different split, compared anyway (allow_stale=true); its tree "
                "numbers may not match the current split"
            )
        return True

    if tree_map.is_file():
        found.append((PIPELINE_3DFIN, tree_map, "th_m"))
    else:
        missing.append(f"pipeline {PIPELINE_3DFIN!r} skipped: {rel(tree_map)} not found (run the split stage)")

    configs = sorted(ctx.stage_dir("smartqsm").glob("*/metrics.csv"))
    for path in configs:
        name = f"smartqsm:{path.parent.name}"
        if belongs_to_current_split(name, path.parent):
            found.append((name, path, "height_m"))
    if not configs:
        missing.append(
            f"pipeline 'smartqsm' skipped: no {rel(ctx.stage_dir('smartqsm'))}/<config>/metrics.csv found"
        )

    arbor = ctx.stage_dir("arbor") / "metrics.csv"
    if not arbor.is_file():
        missing.append(f"pipeline 'arbor' skipped: {rel(arbor)} not found")
    elif belongs_to_current_split("arbor", arbor.parent):
        found.append(("arbor", arbor, "height_m"))
    return found, missing, unverified


# ---------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------
def pair_metric(ref: pd.Series, est: pd.Series) -> tuple[pd.DataFrame, int, int]:
    """Join two ordinal-indexed series of VALID values: ``(paired[ordinal, ref, est], n_missing, n_extra)``."""
    ref, est = ref.dropna(), est.dropna()
    common = ref.index.intersection(est.index)
    paired = pd.DataFrame({"ref": ref.loc[common], "est": est.loc[common]})
    paired.index.name = "ordinal"
    return paired.reset_index().sort_values("ordinal", ignore_index=True), len(ref.index.difference(est.index)), len(
        est.index.difference(ref.index)
    )


def error_stats(ref: np.ndarray, est: np.ndarray, tol: float) -> dict[str, float]:
    """Bias / MAE / RMSE / relative / R2 / OLS fit of ``est`` against ``ref`` (definitions: module docstring)."""
    nan = math.nan
    out = dict.fromkeys(
        ("bias", "MAE", "RMSE", "rel_bias_pct", "rel_RMSE_pct", "R2", "slope", "intercept", "R2_ols", "within_tol_pct"),
        nan,
    )
    n = len(ref)
    if n == 0:
        return out
    err = est - ref
    out["bias"] = float(err.mean())
    out["MAE"] = float(np.abs(err).mean())
    out["RMSE"] = float(np.sqrt((err**2).mean()))
    mean_ref = float(ref.mean())
    if mean_ref > 0:
        out["rel_bias_pct"] = 100.0 * out["bias"] / mean_ref
        out["rel_RMSE_pct"] = 100.0 * out["RMSE"] / mean_ref
    out["within_tol_pct"] = 100.0 * float((np.abs(err) <= tol + 1e-12).mean())
    if n >= 2:
        ss_ref = float(((ref - mean_ref) ** 2).sum())
        if ss_ref > 0:
            out["R2"] = 1.0 - float((err**2).sum()) / ss_ref
            slope = float(((ref - mean_ref) * (est - est.mean())).sum()) / ss_ref
            out["slope"] = slope
            out["intercept"] = float(est.mean()) - slope * mean_ref
            ss_est = float(((est - est.mean()) ** 2).sum())
            if ss_est > 0:
                out["R2_ols"] = float(((ref - mean_ref) * (est - est.mean())).sum()) ** 2 / (ss_ref * ss_est)
    return out


def flag_pairs(paired: pd.DataFrame, metric: str, est: Estimates, id_confidence: pd.Series | None) -> pd.DataFrame:
    """``paired`` plus the quality flags ``id_confidence`` (str) and ``estimated`` (bool, DBH only), by ordinal."""
    flagged = paired.copy()
    ordinals = flagged["ordinal"].tolist()
    confidence = {} if id_confidence is None else id_confidence.to_dict()
    flagged["id_confidence"] = [confidence.get(o, ID_UNKNOWN) for o in ordinals]
    estimated = est.table["estimated_dbh"].to_dict() if metric == "dbh" and "estimated_dbh" in est.table else {}
    flagged["estimated"] = [bool(estimated.get(o, False)) for o in ordinals]
    return flagged


def _apply_exclusions(
    paired: pd.DataFrame, label: str, *, exclude_ambiguous: bool, exclude_estimated: bool
) -> tuple[pd.DataFrame, int]:
    """Log how many of the pairs are flagged and drop them if asked: ``(remaining pairs, number dropped)``."""
    ambiguous = (paired["id_confidence"] == ID_AMBIGUOUS).to_numpy()
    estimated = paired["estimated"].to_numpy(dtype=bool)
    n = len(paired)
    for count, excluded, what, option in (
        (int(ambiguous.sum()), exclude_ambiguous, "have an ambiguous tree number (the alignment with the reference "
         "may be wrong)", "exclude_ambiguous"),
        (int(estimated.sum()), exclude_estimated, "are estimated by arbor, not measured (small tree allometry / no "
         "valid measure / breast height not converged, see the note column of its metrics.csv)", "exclude_estimated"),
    ):  # fmt: skip
        if count and excluded:
            log.info("%s: excluded %d of %d paired values that %s", label, count, n, what)
        elif count:
            log.warning("%s: %d of %d paired values %s; they are included (compare.%s=true leaves them out)", label,
                        count, n, what, option)  # fmt: skip
    drop = np.zeros(n, dtype=bool)
    if exclude_ambiguous:
        drop |= ambiguous
    if exclude_estimated:
        drop |= estimated
    return paired.loc[~drop].reset_index(drop=True), int(drop.sum())


def compare_all(
    reference: refmod.Reference,
    estimates: list[Estimates],
    tolerances: dict[str, float],
    *,
    id_confidence: pd.Series | None = None,
    exclude_ambiguous: bool = False,
    exclude_estimated: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(errors, summary)`` DataFrames (columns: ``ERROR_COLUMNS`` / ``SUMMARY_COLUMNS``) for all pipelines.

    ``id_confidence`` (ordinal -> exact / resolved / ambiguous, from ``split/tree_map.csv``; ``None``: unknown) and the
    ``estimated`` flag of the estimates are attached to every pair; ``exclude_*`` drop flagged pairs from everything.
    """
    ref_table = reference.table.set_index("ordinal")
    error_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for est in estimates:
        for metric in METRICS:
            if metric not in reference.metrics:
                continue
            col = f"{metric}_m"
            paired, n_missing, n_extra = pair_metric(_valid_positive(ref_table[col]), est.table[col])
            if len(paired) == 0:
                log.warning(
                    "%s: no %s value could be paired with the reference (%d reference trees without estimate, %d "
                    "estimates without reference). Do the reference tree numbers equal the 3DFin tree numbers T<k>?",
                    est.name, metric, n_missing, n_extra,
                )
            paired = flag_pairs(paired, metric, est, id_confidence)
            paired, n_excluded = _apply_exclusions(
                paired, f"{est.name} {metric}", exclude_ambiguous=exclude_ambiguous, exclude_estimated=exclude_estimated
            )
            err = paired["est"] - paired["ref"]
            error_rows.append(
                pd.DataFrame(
                    {
                        "pipeline": est.name,
                        "metric": metric,
                        "ordinal": paired["ordinal"],
                        "ref": paired["ref"],
                        "est": paired["est"],
                        "error": err,
                        "rel_error_pct": 100.0 * err / paired["ref"],
                        "id_confidence": paired["id_confidence"],
                        "estimated": paired["estimated"],
                    }
                )
            )
            stats = error_stats(paired["ref"].to_numpy(float), paired["est"].to_numpy(float), tolerances[metric])
            summary_rows.append(
                {"pipeline": est.name, "metric": metric, "n": len(paired), "n_missing": n_missing, "n_extra": n_extra,
                 "n_ambiguous": int((paired["id_confidence"] == ID_AMBIGUOUS).sum()),
                 "n_estimated": int(paired["estimated"].sum()), "n_excluded": n_excluded,
                 **stats, "tol_m": tolerances[metric]}  # fmt: skip
            )
    non_empty = [frame for frame in error_rows if len(frame)]
    errors = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame(columns=ERROR_COLUMNS)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    return errors[ERROR_COLUMNS], summary


_LOG_COLUMNS = (  # (summary column, header, format, left aligned)
    ("pipeline", "pipeline", "{}", True),
    ("metric", "metric", "{}", True),
    ("n", "n", "{:d}", False),
    ("n_missing", "miss", "{:d}", False),
    ("n_extra", "extra", "{:d}", False),
    ("n_ambiguous", "amb", "{:d}", False),
    ("n_estimated", "est", "{:d}", False),
    ("n_excluded", "excl", "{:d}", False),
    ("bias", "bias", "{:+.3f}", False),
    ("MAE", "MAE", "{:.3f}", False),
    ("RMSE", "RMSE", "{:.3f}", False),
    ("rel_bias_pct", "bias%", "{:+.1f}", False),
    ("rel_RMSE_pct", "RMSE%", "{:.1f}", False),
    ("R2", "R2", "{:.3f}", False),
    ("slope", "slope", "{:.3f}", False),
    ("intercept", "icpt", "{:+.3f}", False),
)


def format_summary(summary: pd.DataFrame) -> str:
    """Compact aligned text table of the summary (lengths in m, ``%`` columns relative to the mean reference)."""
    columns = [c for c in _LOG_COLUMNS if c[0] in summary.columns]
    cells = [[header for _, header, _, _ in columns]]
    for row in summary.to_dict("records"):
        line = []
        for col, _, fmt, _ in columns:
            value = row[col]
            line.append("-" if isinstance(value, float) and math.isnan(value) else fmt.format(value))
        cells.append(line)
    widths = [max(len(r[i]) for r in cells) for i in range(len(columns))]
    lines = [
        "  ".join(c.ljust(w) if left else c.rjust(w) for c, w, (_, _, _, left) in zip(r, widths, columns))
        for r in cells
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------------------
def scatter_plot(
    metric: str, errors: pd.DataFrame, summary: pd.DataFrame, pipelines: list[str], path: Path, title: str
) -> Path | None:
    """Write ``scatter_<metric>.png``: estimate vs reference with the 1:1 line. ``None`` if there is nothing to draw."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    data = errors[errors["metric"] == metric]
    if data.empty:
        log.warning("no paired %s values, scatter plot skipped", metric)
        return None
    label, unit, factor = PLOT_UNITS[metric]
    lo = float(min(data["ref"].min(), data["est"].min())) * factor
    hi = float(max(data["ref"].max(), data["est"].max())) * factor
    pad = 0.05 * (hi - lo) if hi > lo else max(0.05 * hi, 0.5)
    lo, hi = max(0.0, lo - pad), hi + pad

    fig = Figure(figsize=(6.4, 6.4), dpi=150)
    FigureCanvasAgg(fig)  # explicit Agg canvas, no pyplot / GUI backend involved
    ax = fig.add_subplot()
    ax.plot([lo, hi], [lo, hi], color="#52514e", linewidth=1.0, linestyle="--", label="1:1", zorder=1)
    for index, name in enumerate(pipelines):
        sub = data[data["pipeline"] == name]
        if sub.empty:
            continue
        stats = summary[(summary["pipeline"] == name) & (summary["metric"] == metric)].iloc[0]
        ax.scatter(
            sub["ref"] * factor,
            sub["est"] * factor,
            s=34,
            color=PIPELINE_COLORS[index % len(PIPELINE_COLORS)],
            marker=PIPELINE_MARKERS[index % len(PIPELINE_MARKERS)],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.92,
            zorder=2,
            label=f"{name} (n={int(stats['n'])}, bias {stats['bias'] * factor:+.2f} {unit}, "
            f"RMSE {stats['RMSE'] * factor:.2f} {unit})",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Reference {label.lower() if metric == 'height' else label} [{unit}]")
    ax.set_ylabel(f"Estimated {label.lower() if metric == 'height' else label} [{unit}]")
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.grid(True, color="#e3e2de", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # legend below the axes: long "smartqsm:<config>" names must not cover the data
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, frameon=False)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------------------------------------
def _number(value: Any) -> Any:
    """JSON-friendly scalar: NaN -> None, numpy numbers -> python numbers."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else round(float(value), 6)
    return value


def run_compare(
    ctx: RunContext,
    *,
    columns_file: str | None = None,
    tree_id_col: str | None = None,
    dbh_col: str | None = None,
    dbh_unit: str = "auto",
    height_col: str | None = None,
    height_unit: str = "auto",
    sheet: str | None = None,
    tolerance_dbh_cm: float = 2.0,
    tolerance_height_m: float = 1.0,
    plots: bool = True,
    allow_stale: bool = False,
    exclude_ambiguous: bool = False,
    exclude_estimated: bool = False,
) -> dict:
    """Compare DBH and tree height of 3DFin / SmartQSM / arbor with the reference table (``ctx.reference``).

    Files read and written, and all statistics: see the module docstring. Rows are joined on the tree number only.

    Parameters
    ----------
    columns_file
        Optional YAML (relative to the project) with the column mapping, sheet and csv options, see
        ``configs/reference_columns.example.yaml``. Precedence per setting: explicit parameter below > YAML >
        auto-detection by column title.
    tree_id_col, dbh_col, height_col
        Column titles in the reference table (``None``: auto-detect by aliases such as ``nr``, ``dbh``, ``d1.3``,
        ``h``, ``wysokość``; ambiguous or unknown columns are an error that lists the columns found).
    dbh_unit, height_unit
        ``m``, ``cm`` or ``mm``. ``auto``: the unit in the column title (``DBH [cm]``), else cm for DBH and m for
        height.
    sheet
        xlsx only: sheet name or 0-based index (default: the first sheet).
    tolerance_dbh_cm, tolerance_height_m
        |estimate - reference| within these counts as agreeing (``within_tol_pct`` in the summary).
    plots
        Write ``scatter_dbh.png`` / ``scatter_height.png``.
    allow_stale
        Compare ``smartqsm:<config>`` / ``arbor`` results that were computed on a different split than the current
        ``split/tree_map.csv`` (their tree numbers may not match) instead of skipping them. Still a warning.
    exclude_ambiguous
        Leave trees whose ``id_confidence`` in ``split/tree_map.csv`` is ``ambiguous`` out of the statistics.
    exclude_estimated
        Leave trees whose DBH arbor only estimated (note: No valid measure / Small tree allometry /
        bh_not_converged) out of the DBH statistics.

    Returns
    -------
    dict
        ``{"dir", "errors", "summary", "reference_used", "plots", "pipelines", "skipped", "unverified",
        "n_reference", "metrics": {pipeline: {metric: {n, n_missing, n_extra, n_ambiguous, n_estimated, n_excluded,
        bias, MAE, RMSE, R2, ...}}}}``. ``skipped``: reasons for pipelines left out; ``unverified``: pipelines
        compared without a verified link to the current split.
    """
    for what, tol in (("tolerance_dbh_cm", tolerance_dbh_cm), ("tolerance_height_m", tolerance_height_m)):
        if not tol > 0:
            raise StageError(f"compare.{what} must be > 0, got {tol!r}")
    base = refmod.ReferenceSpec()
    if columns_file:
        yaml_path = ctx.resolve(columns_file)
        if not yaml_path.is_file():
            raise StageError(f"compare.columns_file not found: {yaml_path}")
        base = refmod.spec_from_yaml(yaml_path)
    spec = refmod.merge_specs(
        base,
        tree_id_col=tree_id_col,
        dbh_col=dbh_col,
        height_col=height_col,
        dbh_unit=dbh_unit,
        height_unit=height_unit,
        sheet=sheet,
    )

    out_dir = ctx.stage_dir("compare")
    if ctx.dry_run:
        log.info("[dry-run] would compare the pipelines found under %s with %s and write %s", ctx.rel(ctx.run_dir), ctx.reference, out_dir)
        if ctx.reference is not None and ctx.reference.is_file():
            try:
                refmod.load_reference(ctx.reference, spec)  # read-only check of the column mapping
            except StageError as exc:
                log.warning("[dry-run] the reference table would be rejected: %s", exc)
        return {"dir": str(ctx.rel(out_dir)), "dry_run": True}

    if ctx.reference is None:
        raise StageError("the compare stage needs a reference table: pass --ref in/<file>.csv|xlsx")
    reference = refmod.load_reference(ctx.reference, spec)

    found, missing, unverified = discover_pipelines(ctx, allow_stale=allow_stale)
    for message in (*missing, *unverified.values()):
        log.warning("%s", message)
    if not found:
        raise StageError(
            "nothing to compare: none of split/tree_map.csv, smartqsm/<config>/metrics.csv, arbor/metrics.csv can be "
            f"used under {ctx.run_dir}. Run the earlier stages first. Details: {'; '.join(missing)}"
        )
    estimates = [read_estimates(name, path, height_col=hcol) for name, path, hcol in found]
    names = [e.name for e in estimates]
    log.info("comparing %s with the reference (%d trees, metrics %s)", names, len(reference.table), list(reference.metrics))

    tree_map = ctx.stage_dir("split") / "tree_map.csv"
    id_confidence = read_id_confidence(tree_map) if tree_map.is_file() else None
    if exclude_ambiguous and id_confidence is None:
        log.warning("exclude_ambiguous=true has no effect: no id_confidence available (see above)")
    tolerances = {"dbh": tolerance_dbh_cm / 100.0, "height": tolerance_height_m}
    errors, summary = compare_all(
        reference, estimates, tolerances,
        id_confidence=id_confidence, exclude_ambiguous=exclude_ambiguous, exclude_estimated=exclude_estimated,
    )  # fmt: skip

    try:
        ctx.reset_stage_dir("compare")  # stale plots / tables of an earlier run would be mistaken for fresh ones
    except RuntimeError as exc:
        raise StageError(str(exc)) from exc
    errors_path, summary_path, reference_path = out_dir / "errors.csv", out_dir / "summary.csv", out_dir / "reference_used.csv"
    errors.round({"ref": 6, "est": 6, "error": 6, "rel_error_pct": 4}).to_csv(errors_path, index=False)
    summary.round(6).to_csv(summary_path, index=False)
    reference.table.round(6).to_csv(reference_path, index=False)

    plot_paths: list[Path] = []
    if plots:
        title = f"Estimate vs reference ({ctx.run_name})"
        for metric in reference.metrics:
            written = scatter_plot(metric, errors, summary, names, out_dir / f"scatter_{metric}.png", title)
            if written is not None:
                plot_paths.append(written)

    log.info(
        "summary (lengths in m; %%-columns relative to the mean reference value; R2 w.r.t. the 1:1 line):\n%s",
        format_summary(summary),
    )
    log.info("wrote %s", ", ".join(p.name for p in (errors_path, summary_path, reference_path, *plot_paths)))

    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary.to_dict("records"):
        metrics.setdefault(row["pipeline"], {})[row["metric"]] = {
            k: _number(v) for k, v in row.items() if k not in ("pipeline", "metric")
        }
    return {
        "dir": str(ctx.rel(out_dir)),
        "errors": str(ctx.rel(errors_path)),
        "summary": str(ctx.rel(summary_path)),
        "reference_used": str(ctx.rel(reference_path)),
        "plots": [str(ctx.rel(p)) for p in plot_paths],
        "pipelines": names,
        "skipped": missing,
        "unverified": sorted(unverified),
        "n_reference": len(reference.table),
        "metrics": metrics,
    }

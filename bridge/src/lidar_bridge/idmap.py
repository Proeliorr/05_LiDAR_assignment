"""Alignment of 3DFin's table rows (``T1 ... Tn``) with the raw ``tree_ID`` values of its point clouds.

Why this is needed
------------------
3DFin 0.6.0 detects ``|U|`` trees in the stripe (``U`` = sorted unique ``tree_ID`` of ``<stem>_stripe.las``; the heights
file has one point per element of ``U``, in that order). The tables (XLSX / txt) however are built from a *second*
vertical-clustering pass (``compute_sections``), in which a detected tree can vanish. The tables then have only
``n = |U| - D`` rows, still in ascending raw-id order, and nothing in them says which ids were dropped. On top of that
3DFin pairs some columns by *index* instead of by tree: ``TH`` is ``tree_heights[k]`` (the k-th of ALL ids) and the X,Y
of rows without a DBH come from ``tree_vector[k]``. After the first dropped id those cells belong to the previous tree.

Only these table data belong to the row's own tree and are used as evidence here: the per-section circle fits of the
sheets ``X`` / ``Y`` / ``Diameters`` (centre and radius per tree row and section). A candidate id supports a row when its
points (in the section window, near its axis) lie on the row's circle.

Method
------
``D = |U| - n`` ids have to be dropped. Rows keep their order, so row ``k`` (0-based) is mapped to ``U[k + o_k]`` with
a non-decreasing offset ``o_k`` in ``0..D`` (``o`` increases by one right after a dropped id). The score of the pair
``(k, o)`` is the sum over the row's circles of the fraction of the candidate's points that are within ``circle_tol`` of
the circle (one pass over the big point cloud, ``O(points * (D + 1))``, no per-id re-reading). The optimal monotone path
is found by dynamic programming. Ambiguity is the *margin*: optimal total score minus the best total score of any
alignment that maps this row to a different id (resp. keeps a dropped id). Rows with a margin below ``margin_min``
are reported as ``ambiguous`` (typically a gap of rows without circles around the dropped id).
"""

from __future__ import annotations

import configparser
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

import laspy
import numpy as np
import pandas as pd

from lidar_bridge.stages._common import StageError

log = logging.getLogger("lidar_bridge.idmap")

#: ``tree_ID`` values that are never turned into a tree, even if one of them shows up in the stripe: -1 is 3DFin's
#: clustering-noise label. (3DFin 0.6.0 writes no other sentinel into ``tree_ID``; in particular 100000 is an ordinary
#: id - the fixture run has 82 of 123 ids >= 100000 - so a real tree with that id must keep its cloud.)
NON_TREE_IDS = (-1,)

REASON_DROPPED = "no_stem_section_rows"

# defaults of 3DFin 0.6.0 used when ``<stem>_config.ini`` is missing
DEFAULT_Z0_NAME = "Z0"
DEFAULT_SECTION_WID = 0.05

_LABEL = re.compile(r"^\s*T(\d+)\s*$")
_SECTION = re.compile(r"^\s*S(\d+)\s*$")


# ---------------------------------------------------------------------------------------------------------
# reading the 3DFin result files
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DfinTables:
    """Per-row data of the 3DFin tables (row ``k`` = label ``T<k+1>``). Circle arrays are ``(n, J)``, 0 = no circle."""

    source: str  # "xlsx" or "txt"
    th: np.ndarray  # 3DFin's TH column (index-paired, see module doc; NOT reliable)
    dbh: np.ndarray  # 0 where 3DFin found no DBH
    x: np.ndarray  # X, Y: own circle centre where dbh > 0, otherwise from the axis of the k-th of ALL ids
    y: np.ndarray
    sections: np.ndarray  # (J,) lower Z0 bound of every section
    x_c: np.ndarray
    y_c: np.ndarray
    radius: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.th.shape[0])

    @property
    def n_sections(self) -> int:
        return int(self.sections.shape[0])


@dataclass(frozen=True)
class DfinConfig:
    """The few 3DFin parameters (``<stem>_config.ini``) needed to interpret the clouds / tables."""

    z0_name: str = DEFAULT_Z0_NAME
    section_wid: float = DEFAULT_SECTION_WID
    found: bool = False


def read_dfin_config(dfin_dir: Path, stem: str) -> DfinConfig:
    path = Path(dfin_dir) / f"{stem}_config.ini"
    if not path.is_file():
        log.warning("%s not found: assuming the 3DFin defaults (z0_name=%s, section_wid=%s)", path.name, DEFAULT_Z0_NAME, DEFAULT_SECTION_WID)
        return DfinConfig()
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(path, encoding="utf-8")
    try:
        return DfinConfig(
            z0_name=parser.get("basic", "z0_name", fallback=DEFAULT_Z0_NAME).strip() or DEFAULT_Z0_NAME,
            section_wid=parser.getfloat("advanced", "section_wid", fallback=DEFAULT_SECTION_WID),
            found=True,
        )
    except ValueError as exc:
        raise StageError(f"cannot parse {path}: {exc}") from exc


def _cell_text(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()


def _find_sheet(sheets: dict[str, pd.DataFrame], name: str, description: str) -> pd.DataFrame:
    for key, frame in sheets.items():
        if key.strip().lower() == name.lower():
            return frame
    # fall back to the description 3DFin writes into the first cell of every sheet
    for frame in sheets.values():
        if frame.size and description.lower() in _cell_text(frame.iat[0, 0]).lower():
            return frame
    raise StageError(f"the 3DFin workbook has no sheet {name!r} (sheets: {list(sheets)})")


def _labelled_rows(raw: pd.DataFrame, header_row: int, label_col: int) -> list[int]:
    """Row indices below ``header_row`` whose label cell is ``T<k>``, checked to run T1, T2, ... without gaps."""
    rows: list[int] = []
    for r in range(header_row + 1, raw.shape[0]):
        match = _LABEL.match(_cell_text(raw.iat[r, label_col]))
        if not match:
            break
        if int(match.group(1)) != len(rows) + 1:
            raise StageError(f"row labels of the 3DFin table are not T1..Tn in order (found {match.group(0)!r} at position {len(rows) + 1})")
        rows.append(r)
    if not rows:
        raise StageError("the 3DFin table has no tree rows (labels T1..Tn not found)")
    return rows


def _read_plot_metrics(raw: pd.DataFrame) -> np.ndarray:
    """``(n, 4)`` array TH, DBH, X, Y. The header row is located by its content (it is not the first row)."""
    wanted = ["TH", "DBH", "X", "Y"]
    for r in range(raw.shape[0]):
        cells = {_cell_text(raw.iat[r, c]).upper(): c for c in range(raw.shape[1]) if _cell_text(raw.iat[r, c])}
        if all(w in cells for w in wanted):
            cols = [cells[w] for w in wanted]
            label_col = min(cols) - 1
            if label_col < 0:
                continue
            rows = _labelled_rows(raw, r, label_col)
            values = raw.iloc[rows, cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            return values
    raise StageError("sheet 'Plot Metrics': no header row with TH / DBH / X / Y found")


def _read_section_sheet(raw: pd.DataFrame, n_rows: int, sheet: str) -> np.ndarray:
    """``(n_rows, J)`` values of a per-tree-per-section sheet (header row ``S1 S2 ...`` found by content)."""
    for r in range(raw.shape[0]):
        cols = [c for c in range(raw.shape[1]) if _SECTION.match(_cell_text(raw.iat[r, c]))]
        if cols:
            label_col = cols[0] - 1
            rows = _labelled_rows(raw, r, label_col)
            if len(rows) != n_rows:
                raise StageError(f"sheet {sheet!r} has {len(rows)} tree rows but 'Plot Metrics' has {n_rows}")
            return raw.iloc[rows, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    raise StageError(f"sheet {sheet!r}: no section header row (S1, S2, ...) found")


def _read_sections_sheet(raw: pd.DataFrame) -> np.ndarray:
    """Lower Z0 bound of every section: the row labelled ``Z0`` under the ``S1 S2 ...`` header."""
    for r in range(raw.shape[0]):
        cols = [c for c in range(raw.shape[1]) if _SECTION.match(_cell_text(raw.iat[r, c]))]
        if cols and r + 1 < raw.shape[0]:
            return pd.to_numeric(raw.iloc[r + 1, cols], errors="coerce").to_numpy(float)
    raise StageError("sheet 'Sections': no section header row (S1, S2, ...) found")


def read_xlsx_tables(path: Path) -> DfinTables:
    """Parse ``<stem>.xlsx`` (sheets 'Plot Metrics', 'Diameters', 'X', 'Y', 'Sections')."""
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    except Exception as exc:  # corrupt / locked / not a workbook
        raise StageError(f"cannot read the 3DFin workbook {path}: {exc}") from exc
    metrics = _read_plot_metrics(_find_sheet(sheets, "Plot Metrics", "Total height (TH)"))
    n = metrics.shape[0]
    diameters = _read_section_sheet(_find_sheet(sheets, "Diameters", "Diameter of every section"), n, "Diameters")
    x_c = _read_section_sheet(_find_sheet(sheets, "X", "(x) coordinate of the centre"), n, "X")
    y_c = _read_section_sheet(_find_sheet(sheets, "Y", "(y) coordinate of the centre"), n, "Y")
    sections = _read_sections_sheet(_find_sheet(sheets, "Sections", "Normalized height (Z0)"))
    return _assemble(metrics, diameters, x_c, y_c, sections, "xlsx", path)


def read_txt_tables(dfin_dir: Path, stem: str) -> DfinTables:
    """Parse the ``export_txt`` variant (``<stem>_dbh_and_heights.txt``, ``_diameters.txt``, ``_X_c.txt``, ...)."""
    base = Path(dfin_dir) / stem

    def load(suffix: str, ndmin: int = 2) -> np.ndarray:
        path = Path(f"{base}{suffix}")
        if not path.is_file():
            raise StageError(f"missing 3DFin table {path.name}")
        return np.loadtxt(path, ndmin=ndmin)

    metrics = load("_dbh_and_heights.txt")
    sections = load("_sections.txt").reshape(-1)
    return _assemble(metrics, load("_diameters.txt"), load("_X_c.txt"), load("_Y_c.txt"), sections, "txt", Path(f"{base}_dbh_and_heights.txt"))


def _assemble(metrics, diameters, x_c, y_c, sections, source: str, where: Path) -> DfinTables:
    n = metrics.shape[0]
    sections = sections[~np.isnan(sections)]
    j = sections.shape[0]
    for name, arr in (("diameters", diameters), ("X", x_c), ("Y", y_c)):
        if arr.shape != (n, j):
            raise StageError(f"{where.name}: table {name!r} has shape {arr.shape}, expected ({n}, {j}) = (trees, sections)")
    if not np.all(np.diff(sections) > 0):
        raise StageError(f"{where.name}: section heights are not strictly increasing")
    return DfinTables(
        source=source,
        th=metrics[:, 0].copy(),
        dbh=np.nan_to_num(metrics[:, 1]),
        x=metrics[:, 2].copy(),
        y=metrics[:, 3].copy(),
        sections=sections,
        x_c=x_c,
        y_c=y_c,
        radius=np.nan_to_num(diameters) / 2.0,
    )


def read_tables(dfin_dir: Path, stem: str) -> DfinTables:
    """The XLSX if present, else the ``export_txt`` files."""
    xlsx = Path(dfin_dir) / f"{stem}.xlsx"
    if xlsx.is_file():
        return read_xlsx_tables(xlsx)
    if (Path(dfin_dir) / f"{stem}_dbh_and_heights.txt").is_file():
        return read_txt_tables(dfin_dir, stem)
    raise StageError(f"no 3DFin result table in {dfin_dir}: expected {stem}.xlsx or {stem}_dbh_and_heights.txt")


def read_stripe_ids(path: Path, chunk_size: int = 2_000_000) -> np.ndarray:
    """Sorted unique ``tree_ID`` of ``<stem>_stripe.las``: the authoritative list of the trees 3DFin detected."""
    found: list[np.ndarray] = []
    try:
        with laspy.open(path) as reader:
            if "tree_ID" not in reader.header.point_format.dimension_names:
                raise StageError(f"{path.name} has no 'tree_ID' dimension (is it a 3DFin stripe file?)")
            for chunk in reader.chunk_iterator(chunk_size):
                found.append(np.unique(np.asarray(chunk["tree_ID"])))
    except laspy.errors.LaspyException as exc:
        raise StageError(f"cannot read {path}: {exc}") from exc
    if not found:
        raise StageError(f"{path.name} contains no points")
    return np.unique(np.concatenate(found)).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------
# dynamic programming on the (rows x offsets) score band
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PathResult:
    offsets: np.ndarray  # (n,) chosen offset of every row
    total: float
    row_margin: np.ndarray  # (n,) total - best total with this row mapped to another id
    through: np.ndarray  # (n, D+1) best total of any path through (row, offset)


def _forward(score: np.ndarray) -> np.ndarray:
    fwd = np.empty_like(score)
    fwd[0] = score[0]
    for k in range(1, score.shape[0]):
        fwd[k] = score[k] + np.maximum.accumulate(fwd[k - 1])
    return fwd


def _backward(score: np.ndarray) -> np.ndarray:
    bwd = np.empty_like(score)
    bwd[-1] = score[-1]
    for k in range(score.shape[0] - 2, -1, -1):
        bwd[k] = score[k] + np.maximum.accumulate(bwd[k + 1][::-1])[::-1]
    return bwd


def best_monotone_path(score: np.ndarray) -> PathResult:
    """Best non-decreasing offset sequence for a ``(n, D+1)`` score band (ties prefer the lower offset).

    ``score[k, o]`` is the evidence for "row ``k`` is ``U[k + o]``". A non-decreasing offset sequence is exactly an
    order-preserving selection of ``n`` of the ``n + D`` ids.
    """
    score = np.asarray(score, dtype=float)
    if score.ndim != 2 or score.shape[0] == 0:
        raise ValueError("score must be a non-empty (rows, D+1) array")
    n, d1 = score.shape
    fwd = _forward(score)
    bwd = _backward(score)
    offsets = np.empty(n, dtype=np.int64)
    offsets[-1] = int(np.argmax(fwd[-1]))
    for k in range(n - 2, -1, -1):
        offsets[k] = int(np.argmax(fwd[k, : offsets[k + 1] + 1]))
    total = float(score[np.arange(n), offsets].sum())
    through = fwd + bwd - score
    row_margin = np.full(n, np.inf)
    if d1 > 1:
        for k in range(n):
            others = np.delete(through[k], offsets[k])
            row_margin[k] = max(0.0, total - float(others.max()))
    return PathResult(offsets=offsets, total=total, row_margin=row_margin, through=through)


def dropped_margins(path: PathResult, dropped_u_index: Sequence[int]) -> list[float]:
    """For every dropped id: total score minus the best total of an alignment that *keeps* that id."""
    n, d1 = path.through.shape
    margins = []
    for u in dropped_u_index:
        best = -np.inf
        for o in range(d1):
            k = u - o
            if 0 <= k < n:
                best = max(best, float(path.through[k, o]))
        margins.append(max(0.0, path.total - best) if np.isfinite(best) else float("inf"))
    return margins


# ---------------------------------------------------------------------------------------------------------
# evidence from the point cloud
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Evidence:
    score: np.ndarray  # (n, D+1): sum over the row's circles of the inlier fraction of the candidate's points
    supported: np.ndarray  # (n, D+1): number of circles with inlier fraction >= 0.3
    n_points: int  # points that took part (near an axis, inside a section window, id in U)


def require_dims(header: laspy.LasHeader, names: Sequence[str], path: Path) -> None:
    have = set(header.point_format.dimension_names)
    missing = [n for n in names if n not in have]
    if missing:
        raise StageError(
            f"{path.name} has no dimension(s) {missing}; found extra dims "
            f"{[e.name for e in header.point_format.extra_dimensions]}. Was it written by 3DFin?"
        )


def collect_evidence(
    cloud_path: Path,
    unique_ids: np.ndarray,
    tables: DfinTables,
    n_dropped: int,
    *,
    z0_name: str = DEFAULT_Z0_NAME,
    section_wid: float = DEFAULT_SECTION_WID,
    circle_tol: float = 0.04,
    axis_margin: float = 0.15,
    min_points: int = 5,
    chunk_size: int = 2_000_000,
) -> Evidence:
    """ONE chunked pass over ``<stem>_tree_ID_dist_axes.las``: how well do the points of the candidate ids fit the circles?

    A point of id ``U[i]`` inside the window ``[S_j, S_j + section_wid)`` and within ``max(R) + axis_margin`` of its axis
    is tested against the circle of every row ``k = i - o`` (``o = 0..D``) that has a circle in section ``j``; it is an
    inlier when ``| |p - centre| - R | <= circle_tol``. Sections with fewer than ``min_points`` candidate points count 0.
    (Windows are assumed not to overlap, i.e. ``section_len >= section_wid`` - the 3DFin default 0.2 / 0.05.)
    """
    n, n_sec = tables.radius.shape
    d1 = n_dropped + 1
    if n * n_sec * d1 * 16 > 2 * 1024**3:
        raise StageError(f"too many dropped ids ({n_dropped}) for {n} rows x {n_sec} sections: refusing to allocate the evidence table")
    radius, x_c, y_c, sections = tables.radius, tables.x_c, tables.y_c, tables.sections
    axis_thr = max(0.3, float(radius.max()) + axis_margin)
    n_u = unique_ids.shape[0]
    n_in = np.zeros(n * n_sec * d1, dtype=np.int64)
    n_tot = np.zeros(n * n_sec * d1, dtype=np.int64)
    used = 0
    with laspy.open(cloud_path) as reader:
        require_dims(reader.header, ("tree_ID", "dist_axes", z0_name), cloud_path)
        for chunk in reader.chunk_iterator(chunk_size):
            idx = np.flatnonzero(np.asarray(chunk["dist_axes"]) <= axis_thr)
            if idx.size == 0:
                continue
            tid = np.asarray(chunk["tree_ID"])[idx].astype(np.int64)
            z0 = np.asarray(chunk[z0_name])[idx]
            pos = np.searchsorted(unique_ids, tid)
            pos[pos >= n_u] = 0
            sec = np.searchsorted(sections, z0, side="right") - 1
            sec_c = np.clip(sec, 0, n_sec - 1)
            ok = (unique_ids[pos] == tid) & (sec >= 0) & (z0 < sections[sec_c] + section_wid)
            if not ok.any():
                continue
            idx, pos, sec = idx[ok], pos[ok], sec_c[ok]
            px = np.asarray(chunk.x[idx])
            py = np.asarray(chunk.y[idx])
            used += idx.size
            for o in range(d1):
                k = pos - o
                valid = (k >= 0) & (k < n)
                k0 = np.where(valid, k, 0)
                rad = radius[k0, sec]
                valid &= rad > 0
                if not valid.any():
                    continue
                flat = (k0[valid] * n_sec + sec[valid]) * d1 + o
                resid = np.abs(np.hypot(px[valid] - x_c[k0[valid], sec[valid]], py[valid] - y_c[k0[valid], sec[valid]]) - rad[valid])
                np.add.at(n_tot, flat, 1)
                np.add.at(n_in, flat[resid <= circle_tol], 1)
    n_in = n_in.reshape(n, n_sec, d1)
    n_tot = n_tot.reshape(n, n_sec, d1)
    frac = np.where(n_tot >= min_points, n_in / np.maximum(n_tot, 1), 0.0)
    return Evidence(score=frac.sum(axis=1), supported=(frac >= 0.3).sum(axis=1), n_points=used)


# ---------------------------------------------------------------------------------------------------------
# the alignment
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Alignment:
    rows: pd.DataFrame  # one row per table row, see ROW_COLUMNS
    dropped: pd.DataFrame  # ids without a table row: raw_id, u_index, reason, margin
    n_unique: int
    n_dropped: int
    total_score: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def n_ambiguous(self) -> int:
        return int((self.rows["id_confidence"] == "ambiguous").sum())

    @property
    def dropped_ids(self) -> list[int]:
        return [int(v) for v in self.dropped["raw_id"]]


ROW_COLUMNS = ["ordinal", "raw_id", "u_index", "offset", "id_confidence", "n_circles", "n_supported", "evidence", "margin"]
DROPPED_COLUMNS = ["raw_id", "u_index", "reason", "margin"]


def align_rows_to_ids(
    unique_ids: np.ndarray,
    tables: DfinTables,
    cloud_path: Path | None = None,
    *,
    z0_name: str = DEFAULT_Z0_NAME,
    section_wid: float = DEFAULT_SECTION_WID,
    margin_min: float = 2.0,
    circle_tol: float = 0.04,
    chunk_size: int = 2_000_000,
) -> Alignment:
    """Map table rows ``T1..Tn`` to raw ``tree_ID`` values (see the module docstring).

    Parameters
    ----------
    unique_ids
        Sorted unique tree ids of the stripe file (``read_stripe_ids``).
    tables
        The parsed 3DFin tables (only the circle sheets are used as evidence).
    cloud_path
        ``<stem>_tree_ID_dist_axes.las``; only read when ids were dropped (``len(unique_ids) > n_rows``).
    margin_min
        A row is ``resolved`` when the best alignment mapping it to another id scores at least this much lower
        (unit: one circle whose points all lie on it = 1.0), else ``ambiguous``. With ``D == 0`` all rows are ``exact``.

    Returns
    -------
    Alignment
        ``rows`` has the columns ``ordinal`` (1..n = row label number), ``raw_id``, ``u_index``, ``offset``,
        ``id_confidence`` (``exact`` / ``resolved`` / ``ambiguous``), ``n_circles``, ``n_supported``, ``evidence``
        (score of the assigned id) and ``margin``; ``dropped`` lists the ids that got no row.

    Raises
    ------
    StageError
        More table rows than detected trees (``D < 0``), or ids dropped but no cloud to compare with.
    """
    unique_ids = np.asarray(unique_ids, dtype=np.int64)
    if unique_ids.ndim != 1 or unique_ids.size == 0 or np.any(np.diff(unique_ids) <= 0):
        raise ValueError("unique_ids must be a sorted array of distinct ids")
    n = tables.n_rows
    n_unique = int(unique_ids.size)
    d = n_unique - n
    if d < 0:
        raise StageError(
            f"the 3DFin table has {n} tree rows but only {n_unique} distinct tree_IDs were detected in the stripe: "
            "files of different runs? (expected rows <= detected trees)"
        )
    n_circles = (tables.radius > 0).sum(axis=1)
    if d == 0:
        rows = pd.DataFrame(
            {
                "ordinal": np.arange(1, n + 1),
                "raw_id": unique_ids,
                "u_index": np.arange(n),
                "offset": 0,
                "id_confidence": "exact",
                "n_circles": n_circles,
                "n_supported": np.nan,
                "evidence": np.nan,
                "margin": np.nan,
            }
        )[ROW_COLUMNS]
        return Alignment(rows, pd.DataFrame(columns=DROPPED_COLUMNS), n_unique, 0)

    if cloud_path is None:
        raise StageError(f"{d} detected tree id(s) have no table row; the point cloud is needed to find which ones")
    log.info("%d of %d detected ids have no table row: matching the %d rows' circles against the cloud ...", d, n_unique, n)
    evidence = collect_evidence(
        cloud_path, unique_ids, tables, d, z0_name=z0_name, section_wid=section_wid, circle_tol=circle_tol, chunk_size=chunk_size
    )
    path = best_monotone_path(evidence.score)
    k = np.arange(n)
    u_index = k + path.offsets
    kept = np.zeros(n_unique, dtype=bool)
    kept[u_index] = True
    dropped_u = np.flatnonzero(~kept)
    drop_margin = dropped_margins(path, dropped_u)
    row_margin = path.row_margin
    confidence = np.where(row_margin >= margin_min, "resolved", "ambiguous")
    rows = pd.DataFrame(
        {
            "ordinal": k + 1,
            "raw_id": unique_ids[u_index],
            "u_index": u_index,
            "offset": path.offsets,
            "id_confidence": confidence,
            "n_circles": n_circles,
            "n_supported": evidence.supported[k, path.offsets],
            "evidence": evidence.score[k, path.offsets],
            "margin": row_margin,
        }
    )[ROW_COLUMNS]
    dropped = pd.DataFrame(
        {
            "raw_id": unique_ids[dropped_u],
            "u_index": dropped_u,
            "reason": REASON_DROPPED,
            "margin": drop_margin,
        }
    )[DROPPED_COLUMNS]
    alignment = Alignment(rows, dropped, n_unique, d, total_score=path.total)
    if path.total <= 0:
        alignment.notes.append("no circle of any row found support in the cloud: the alignment is a guess")
    with_circles = rows[rows["n_circles"] >= 5]
    unsupported = with_circles[with_circles["evidence"] < 0.5]
    if len(unsupported):
        alignment.notes.append(
            f"{len(unsupported)} of {len(with_circles)} rows with >=5 circles find no support in their assigned id "
            f"(ordinals {unsupported['ordinal'].head(10).tolist()}): check the alignment"
        )
    return alignment

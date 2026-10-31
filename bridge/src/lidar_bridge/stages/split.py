"""Stage 2 (WP5): split the 3DFin result into one point cloud per tree, numbered like 3DFin's table rows.

Contract
--------
Reads ``data/out/<run>/dfin/`` (``<stem>_tree_ID_dist_axes.las`` + ``<stem>_stripe.las`` + ``<stem>.xlsx`` +
``<stem>_tree_heights.las`` [+ ``_tree_locator.las``, ``_config.ini``]) and writes into ``data/out/<run>/split/``::

    trees/tree_0001.laz ...         one cloud per tree, named by the ORDINAL (= table row T1, T2, ...; 4 digits, wider if
                                    there are more than 9999 trees); same point format, scale/offset and ALL extra
                                    dimensions as the 3DFin cloud (so ``Z0`` and ``dist_axes`` survive)
    trees_ground/tree_0001.laz ...  only with ``emit_ground_variant``: the same points plus the ground skirt
                                    (``-0.5 <= Z0 < min_z0`` within ``max_axis_dist``)
    tree_map.csv                    one row per ordinal, see :data:`TREE_MAP_COLUMNS`
    excluded_ids.csv                3DFin tree ids that have no table row (and so no ordinal), see :mod:`lidar_bridge.idmap`

Trees with fewer than ``min_points`` kept points are listed in ``tree_map.csv`` (``low_points=True``, empty ``file``)
but their cloud is not written.

The cut and the stem base
-------------------------
3DFin's ``Z0`` is the height above its terrain model (DTM) at the point's own X,Y, so on a slope the points with
``Z0 >= min_z0`` of a tree include low vegetation on the DOWNHILL side whose absolute Z is far below the ground at the stem.
The QSM tools measure breast height (and the tree height) from the LOWEST POINT OF THE CLOUD, so the cloud's lowest point
has to lie at the stem base. Per tree the split therefore first estimates ``stem_ground_z`` = the terrain elevation at the
stem base = mean of ``Z - Z0`` over the tree's points closer than :data:`STEM_GROUND_AXIS_DIST` (0.5 m) to its axis (if there
are fewer than :data:`STEM_GROUND_MIN_POINTS` of them: the mean over all the tree's points, flagged ``tree`` in
``stem_ground_source``; one extra streaming pass over the dist_axes cloud). A point is kept when

    raw tree id has a table row  AND  dist_axes <= max_axis_dist  AND  Z0 >= min_z0  AND  Z >= stem_ground_z + min_z0

i.e. additionally above the HORIZONTAL plane ``min_z0`` over the stem base. ``base_z0_m`` = ``min(Z of the kept points) -
stem_ground_z`` is the height of the cloud's lowest point above the ground at the stem (so ~``min_z0``, never less), which is
exactly the offset a collector has to add: the QSM's "1.3 m" is ``1.3 + base_z0_m`` above the ground at the stem and its
height (max z - min z) lacks ``base_z0_m``. With ``emit_ground_variant`` ``trees_ground/`` holds the same points plus a
skirt: ``Z0 >= -0.5`` AND ``Z >= stem_ground_z - 0.5``.

The tree files are deterministic: the LAS header copies the source's creation date (laspy would stamp today's date, which
makes identical points differ from day to day and defeats the byte-identity check of the SmartQSM resume).

Why the ordinals need an alignment step (3DFin quirk) is described in :mod:`lidar_bridge.idmap`. In short: the table has
``D`` rows fewer than 3DFin detected trees, and its ``TH`` column (and X,Y where there is no DBH) belongs to the wrong tree
after the first dropped id. ``tree_map.csv`` therefore carries the CORRECT height (``th_m``, from ``_tree_heights.las``)
next to the table's (``th_xlsx_m``) and ``th_mismatch``.
"""

from __future__ import annotations

import datetime
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from lidar_bridge import idmap
from lidar_bridge.paths import RunContext
from lidar_bridge.stages._common import StageError

log = logging.getLogger("lidar_bridge.stages.split")

#: Lower bound (relative to Z0 = 0 and to the horizontal plane through the stem base) of the ground skirt written by
#: ``emit_ground_variant``.
GROUND_MIN_Z0 = -0.5
#: Terrain elevation at the stem base = mean of ``Z - Z0`` over the tree's points closer than this to the axis (m) ...
STEM_GROUND_AXIS_DIST = 0.5
#: ... if there are at least this many; else the mean over ALL points of the tree (``stem_ground_source`` = ``tree``).
STEM_GROUND_MIN_POINTS = 20
SOURCE_AXIS, SOURCE_TREE, SOURCE_NONE = "axis", "tree", "none"
#: ``creation_date`` of the tree files when the source file has none (laspy would write today's date: not reproducible).
FALLBACK_CREATION_DATE = datetime.date(2000, 1, 1)
#: Stem centre estimate from the cloud: points within this height window around breast height ...
BREAST_HEIGHT = 1.3
CENTRE_HALF_WINDOW = 0.15
#: ... and this close to the axis; at least this many points are required.
CENTRE_MAX_AXIS_DIST = 0.35
CENTRE_MIN_POINTS = 3
#: A DBH==0 row takes the X,Y of its own circle nearest to breast height if that is at most this far from it (m).
CIRCLE_XY_MAX_DZ = 0.8
TH_TOLERANCE_XLSX = 1e-6
TH_TOLERANCE_TXT = 6e-4  # the export_txt tables are rounded to 3 decimals
CHUNK_SIZE = 1_000_000

TREE_MAP_COLUMNS = [
    "ordinal",  # 1..n = the row label T<k> of the 3DFin table
    "raw_id",  # 3DFin's tree_ID (sparse integers; 0 is a real tree)
    "id_confidence",  # exact (no id dropped) | resolved | ambiguous (see idmap)
    "n_points",  # points in trees/tree_XXXX.laz
    "base_z0_m",  # min(Z of the kept points) - stem_ground_z_m: height of the cloud's lowest point above the ground at the stem
    "stem_ground_z_m",  # terrain elevation at the stem base (mean of Z - Z0 near the axis), same CRS as the cloud's Z
    "x_m",
    "y_m",  # tree position, see xy_source
    "dbh_m",  # 3DFin DBH, empty where 3DFin found none
    "th_m",  # CORRECT total height of this tree (tree_heights.las z0 looked up by raw_id)
    "th_xlsx_m",  # TH as written in the table (3DFin pairs it by index, wrong after a dropped id)
    "th_mismatch",
    "low_points",  # n_points < min_points: cloud not written
    "dbh_missing",
    # extras
    "file",  # path of the cloud relative to the split dir ('' if not written)
    "xy_source",  # xlsx (own DBH circle) | circle (own circle nearest to 1.3 m) | cloud (median of the stem points) | none
    "xy_cloud_offset_m",  # |(x_m, y_m) - cloud estimate|: sanity check of the id assignment
    "th_xy_offset_m",  # |top of the tree in tree_heights.las - (x_m, y_m)|: sanity check of the height ordering
    "id_margin",  # idmap score margin (empty when exact)
    "stem_ground_source",  # axis (points within 0.5 m of the axis) | tree (fallback: all points of the tree) | none
]


def tree_file_name(ordinal: int, n_trees: int) -> str:
    """``tree_0001.laz``; the number of digits grows with ``n_trees`` (never fewer than 4)."""
    width = max(4, len(str(n_trees)))
    return f"tree_{ordinal:0{width}d}.laz"


# ---------------------------------------------------------------------------------------------------------
# locating the 3DFin files
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DfinFiles:
    dir: Path
    stem: str
    cloud: Path
    stripe: Path
    heights: Path
    locator: Path

    @classmethod
    def find(cls, dfin_dir: Path, input_path: Path | None) -> DfinFiles:
        if not dfin_dir.is_dir():
            raise StageError(f"no 3DFin results in {dfin_dir}: run the dfin stage first (stages=[dfin,split])")
        suffix = "_tree_ID_dist_axes.las"
        stem = None
        if input_path is not None and (dfin_dir / f"{input_path.stem}{suffix}").is_file():
            stem = input_path.stem
        else:
            found = sorted(p.name[: -len(suffix)] for p in dfin_dir.glob(f"*{suffix}"))
            if len(found) == 1:
                stem = found[0]
            elif not found:
                raise StageError(f"{dfin_dir} contains no *{suffix} file: run the dfin stage first")
            else:
                raise StageError(f"several 3DFin results in {dfin_dir} ({found}); pass input=<the input file> to choose one")
        files = cls(
            dir=dfin_dir,
            stem=stem,
            cloud=dfin_dir / f"{stem}{suffix}",
            stripe=dfin_dir / f"{stem}_stripe.las",
            heights=dfin_dir / f"{stem}_tree_heights.las",
            locator=dfin_dir / f"{stem}_tree_locator.las",
        )
        missing = [p.name for p in (files.stripe, files.heights) if not p.is_file()]
        if missing:
            raise StageError(f"missing 3DFin result file(s) in {dfin_dir}: {', '.join(missing)}")
        return files


def _read_heights(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, z0)`` of ``<stem>_tree_heights.las``: one point per detected tree, in ascending tree_ID order."""
    try:
        las = laspy.read(path)
    except laspy.errors.LaspyException as exc:
        raise StageError(f"cannot read {path}: {exc}") from exc
    idmap.require_dims(las.header, ("z0",), path)
    return np.asarray(las.x), np.asarray(las.y), np.asarray(las["z0"], dtype=float)


def _read_locator(path: Path) -> tuple[int, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        las = laspy.read(path)
    except laspy.errors.LaspyException as exc:
        raise StageError(f"cannot read {path}: {exc}") from exc
    idmap.require_dims(las.header, ("diameters",), path)
    return len(las.points), np.asarray(las["diameters"], dtype=float)


# ---------------------------------------------------------------------------------------------------------
# the splitting pass
# ---------------------------------------------------------------------------------------------------------
@dataclass
class StemGround:
    """Terrain elevation at the stem base of every detected tree (index = position in the sorted unique ids)."""

    z: np.ndarray  # (|U|,) mean of Z - Z0, NaN where the tree has no point in the cloud
    source: np.ndarray  # (|U|,) SOURCE_AXIS | SOURCE_TREE | SOURCE_NONE
    n_axis: np.ndarray  # (|U|,) points within STEM_GROUND_AXIS_DIST of the axis
    n_all: np.ndarray  # (|U|,) all points of the tree


@dataclass
class SplitStats:
    n_points: np.ndarray  # per ordinal (index 0 unused): points in trees/
    n_ground: np.ndarray  # per ordinal: points in trees_ground/
    min_z: np.ndarray  # per ordinal: lowest Z of the kept points (inf if none)
    centre_xy: np.ndarray  # (|U|, 2) median stem position near breast height, NaN where too few points
    stem_ground: StemGround
    points_read: int = 0
    points_kept: int = 0


def collect_stem_ground(cloud: Path, unique_ids: np.ndarray, *, z0_name: str, chunk_size: int = CHUNK_SIZE) -> StemGround:
    """ONE streaming pass over the 3DFin cloud: per detected tree the terrain elevation ``Z - Z0`` at the stem base.

    Mean of ``Z - Z0`` over the tree's points with ``dist_axes < STEM_GROUND_AXIS_DIST``; with fewer than
    ``STEM_GROUND_MIN_POINTS`` such points the mean over all of the tree's points (flagged ``tree``). Only per-tree sums
    and counts are kept, so the memory is a few chunks whatever the size of the file.
    """
    n_u = unique_ids.shape[0]
    sum_axis, sum_all = np.zeros(n_u), np.zeros(n_u)
    n_axis, n_all = np.zeros(n_u, dtype=np.int64), np.zeros(n_u, dtype=np.int64)
    try:
        with laspy.open(cloud) as reader:
            idmap.require_dims(reader.header, ("tree_ID", "dist_axes", z0_name), cloud)
            for chunk in reader.chunk_iterator(chunk_size):
                tid = np.asarray(chunk["tree_ID"]).astype(np.int64)
                pos = np.searchsorted(unique_ids, tid)
                pos[pos >= n_u] = 0
                idx = np.flatnonzero(unique_ids[pos] == tid)
                if idx.size == 0:
                    continue
                height = np.asarray(chunk.z[idx]) - np.asarray(chunk[z0_name])[idx]
                finite = np.isfinite(height)
                idx, height = idx[finite], height[finite]
                p = pos[idx]
                near = np.asarray(chunk["dist_axes"])[idx] < STEM_GROUND_AXIS_DIST
                sum_all += np.bincount(p, weights=height, minlength=n_u)
                n_all += np.bincount(p, minlength=n_u)
                sum_axis += np.bincount(p[near], weights=height[near], minlength=n_u)
                n_axis += np.bincount(p[near], minlength=n_u)
    except laspy.errors.LaspyException as exc:
        raise StageError(f"cannot read {cloud}: {exc}") from exc
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(n_axis >= STEM_GROUND_MIN_POINTS, sum_axis / n_axis, np.where(n_all > 0, sum_all / n_all, np.nan))
    source = np.where(n_axis >= STEM_GROUND_MIN_POINTS, SOURCE_AXIS, np.where(n_all > 0, SOURCE_TREE, SOURCE_NONE))
    return StemGround(z=z, source=source, n_axis=n_axis, n_all=n_all)


def _output_header(src: laspy.LasHeader) -> laspy.LasHeader:
    """Header for the tree files: same version / point format (with its extra dims) / scale / offset / CRS records.

    Deterministic: laspy would stamp today's date as ``creation_date``, so the same points written on another day would
    differ in the header (the SmartQSM resume compares the files byte for byte). The date (and the other identification
    fields) are therefore taken from the source header.
    """
    header = laspy.LasHeader(version=src.version, point_format=src.point_format)
    header.scales = src.scales
    header.offsets = src.offsets
    header.global_encoding = src.global_encoding
    header.system_identifier = src.system_identifier
    header.generating_software = "lidar_bridge split"
    header.file_source_id = src.file_source_id
    header.uuid = src.uuid
    header.creation_date = src.creation_date if src.creation_date is not None else FALLBACK_CREATION_DATE
    for vlr in src.vlrs:
        if not isinstance(vlr, laspy.vlrs.known.ExtraBytesVlr):  # rebuilt by laspy from the point format
            header.vlrs.append(vlr)
    if src.evlrs:
        header.evlrs = list(src.evlrs)
    return header


class _TreeWriters:
    """One lazily opened LAZ writer per ordinal (a tree without points never gets a file)."""

    def __init__(self, directory: Path, header: laspy.LasHeader, n_trees: int):
        self.directory, self.header, self.n_trees = directory, header, n_trees
        self.writers: dict[int, laspy.LasWriter] = {}
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, ordinal: int) -> Path:
        return self.directory / tree_file_name(ordinal, self.n_trees)

    def write(self, ordinal: int, points) -> None:
        writer = self.writers.get(ordinal)
        if writer is None:
            writer = self.writers[ordinal] = laspy.open(self.path(ordinal), mode="w", header=self.header)
        writer.write_points(points)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


def _write_groups(writers: _TreeWriters, chunk, sel: np.ndarray, ordinals: np.ndarray, z: np.ndarray | None, stats_n, stats_min_z) -> None:
    """Write the points ``sel`` (indices into ``chunk``) of every ordinal to its file, keeping the original order."""
    if sel.size == 0:
        return
    order = np.argsort(ordinals, kind="stable")
    sel, ordinals = sel[order], ordinals[order]
    bounds = np.flatnonzero(np.diff(ordinals)) + 1
    for group, ords in zip(np.split(sel, bounds), np.split(ordinals, bounds)):
        ordinal = int(ords[0])
        writers.write(ordinal, chunk[group])
        stats_n[ordinal] += group.size
        if stats_min_z is not None and z is not None:
            stats_min_z[ordinal] = min(stats_min_z[ordinal], float(z[group].min()))


def split_cloud(
    cloud: Path,
    unique_ids: np.ndarray,
    ordinal_of_u: np.ndarray,
    n_trees: int,
    out_dir: Path,
    *,
    z0_name: str,
    max_axis_dist: float,
    min_z0: float,
    emit_ground_variant: bool,
    chunk_size: int = CHUNK_SIZE,
) -> SplitStats:
    """Two chunked passes over the 3DFin cloud: the stem-base ground, then write ``trees/`` (and ``trees_ground/``).

    ``ordinal_of_u[i]`` is the ordinal of ``unique_ids[i]`` (0 = this id is not split). Peak memory is a few chunks
    (``chunk_size`` points x point size), independent of the file size. The cut is described in the module docstring.
    """
    n_u = unique_ids.shape[0]
    started = time.monotonic()
    stem_ground = collect_stem_ground(cloud, unique_ids, z0_name=z0_name, chunk_size=chunk_size)
    log.info("split: stem-base ground of %d trees found (%.0f s)", n_u, time.monotonic() - started)
    stats = SplitStats(
        n_points=np.zeros(n_trees + 1, dtype=np.int64),
        n_ground=np.zeros(n_trees + 1, dtype=np.int64),
        min_z=np.full(n_trees + 1, np.inf),
        centre_xy=np.full((n_u, 2), np.nan),
        stem_ground=stem_ground,
    )
    ground_lower = min(GROUND_MIN_Z0, min_z0)  # the skirt always contains the main cloud
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    writers: list[_TreeWriters] = []
    try:
        with laspy.open(cloud) as reader:
            idmap.require_dims(reader.header, ("tree_ID", "dist_axes", z0_name), cloud)
            header = _output_header(reader.header)
            trees = _TreeWriters(out_dir / "trees", header, n_trees)
            writers.append(trees)
            ground = None
            if emit_ground_variant:
                ground = _TreeWriters(out_dir / "trees_ground", header, n_trees)
                writers.append(ground)
            total = reader.header.point_count
            for number, chunk in enumerate(reader.chunk_iterator(chunk_size), start=1):
                stats.points_read += len(chunk)
                tid = np.asarray(chunk["tree_ID"]).astype(np.int64)
                dist = np.asarray(chunk["dist_axes"])
                z0 = np.asarray(chunk[z0_name])
                z = np.asarray(chunk.z)
                pos = np.searchsorted(unique_ids, tid)
                pos[pos >= n_u] = 0
                in_u = unique_ids[pos] == tid
                ordinal = np.where(in_u, ordinal_of_u[pos], 0)
                near = (ordinal > 0) & (dist <= max_axis_dist)
                above = z - stem_ground.z[pos]  # height over the horizontal plane through the stem base (NaN: no ground)

                with np.errstate(invalid="ignore"):
                    keep = np.flatnonzero(near & (z0 >= min_z0) & (above >= min_z0))
                    _write_groups(trees, chunk, keep, ordinal[keep], z, stats.n_points, stats.min_z)
                    stats.points_kept += keep.size
                    if ground is not None:
                        skirt = np.flatnonzero(near & (z0 >= ground_lower) & (above >= ground_lower))
                        _write_groups(ground, chunk, skirt, ordinal[skirt], None, stats.n_ground, None)

                centre = np.flatnonzero(in_u & (np.abs(z0 - BREAST_HEIGHT) < CENTRE_HALF_WINDOW) & (dist < CENTRE_MAX_AXIS_DIST))
                if centre.size:
                    samples.append((pos[centre], np.asarray(chunk.x[centre]), np.asarray(chunk.y[centre])))
                if number % 10 == 0:
                    log.info("split: %.1f of %.1f M points read, %.1f M kept (%.0f s)", stats.points_read / 1e6, total / 1e6, stats.points_kept / 1e6, time.monotonic() - started)
    except laspy.errors.LaspyException as exc:
        _discard(writers, out_dir)
        raise StageError(f"cannot read {cloud}: {exc}") from exc
    except BaseException:
        _discard(writers, out_dir)
        raise
    for w in writers:
        w.close()

    if samples:
        u = np.concatenate([s[0] for s in samples])
        xs = np.concatenate([s[1] for s in samples])
        ys = np.concatenate([s[2] for s in samples])
        order = np.argsort(u, kind="stable")
        u, xs, ys = u[order], xs[order], ys[order]
        starts = np.flatnonzero(np.r_[True, np.diff(u) != 0])
        ends = np.r_[starts[1:], u.size]
        for a, b in zip(starts, ends):
            if b - a >= CENTRE_MIN_POINTS:
                stats.centre_xy[u[a]] = (np.median(xs[a:b]), np.median(ys[a:b]))
    return stats


def _discard(writers: list[_TreeWriters], out_dir: Path) -> None:
    """Failure clean-up: no half-written tree files may be mistaken for a result."""
    for w in writers:
        try:
            w.close()
        except Exception:  # noqa: BLE001  (already failing)
            pass
    for name in ("trees", "trees_ground"):
        shutil.rmtree(out_dir / name, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------
# tree_map.csv
# ---------------------------------------------------------------------------------------------------------
def _own_circle_xy(tables: idmap.DfinTables, section_wid: float) -> tuple[np.ndarray, np.ndarray]:
    """Per row: centre of the row's own circle nearest to breast height (NaN if none within ``CIRCLE_XY_MAX_DZ``)."""
    n = tables.n_rows
    mid = tables.sections + section_wid / 2.0
    dz = np.abs(mid - BREAST_HEIGHT)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    for k in range(n):
        ok = (tables.radius[k] > 0) & (dz <= CIRCLE_XY_MAX_DZ)
        if ok.any():
            j = np.flatnonzero(ok)[np.argmin(dz[ok])]
            x[k], y[k] = tables.x_c[k, j], tables.y_c[k, j]
    return x, y


def build_tree_map(
    alignment: idmap.Alignment,
    tables: idmap.DfinTables,
    stats: SplitStats,
    heights: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    section_wid: float,
    min_points: int,
    emit_ground_variant: bool,
    n_trees: int,
    written: set[int],
) -> pd.DataFrame:
    rows = alignment.rows
    n = len(rows)
    k = rows["ordinal"].to_numpy() - 1
    u = rows["u_index"].to_numpy()
    ordinal = rows["ordinal"].to_numpy()
    h_x, h_y, h_z0 = heights

    dbh = tables.dbh[k]
    has_dbh = dbh > 0
    cloud_x, cloud_y = stats.centre_xy[u, 0], stats.centre_xy[u, 1]
    circ_x, circ_y = _own_circle_xy(tables, section_wid)
    circ_x, circ_y = circ_x[k], circ_y[k]
    x = np.where(has_dbh, tables.x[k], np.where(np.isfinite(circ_x), circ_x, cloud_x))
    y = np.where(has_dbh, tables.y[k], np.where(np.isfinite(circ_y), circ_y, cloud_y))
    source = np.where(has_dbh, "xlsx", np.where(np.isfinite(circ_x), "circle", np.where(np.isfinite(cloud_x), "cloud", "none")))
    if tables.source == "txt":
        source = np.where(source == "xlsx", "table", source)

    th = h_z0[u]
    th_table = tables.th[k]
    tol = TH_TOLERANCE_XLSX if tables.source == "xlsx" else TH_TOLERANCE_TXT
    n_points = stats.n_points[ordinal]
    stem_ground_z = stats.stem_ground.z[u]
    frame = pd.DataFrame(
        {
            "ordinal": ordinal,
            "raw_id": rows["raw_id"].to_numpy(),
            "id_confidence": rows["id_confidence"].to_numpy(),
            "n_points": n_points,
            "base_z0_m": np.where(n_points > 0, stats.min_z[ordinal] - stem_ground_z, np.nan),
            "stem_ground_z_m": stem_ground_z,
            "x_m": x,
            "y_m": y,
            "dbh_m": np.where(has_dbh, dbh, np.nan),
            "th_m": th,
            "th_xlsx_m": th_table,
            "th_mismatch": np.abs(th - th_table) > tol,
            "low_points": n_points < min_points,
            "dbh_missing": ~has_dbh,
            "file": [f"trees/{tree_file_name(int(o), n_trees)}" if int(o) in written else "" for o in ordinal],
            "xy_source": source,
            "xy_cloud_offset_m": np.hypot(x - cloud_x, y - cloud_y),
            "th_xy_offset_m": np.hypot(h_x[u] - x, h_y[u] - y),
            "id_margin": rows["margin"].to_numpy(),
            "stem_ground_source": stats.stem_ground.source[u],
        }
    )
    if emit_ground_variant:
        frame["n_points_ground"] = stats.n_ground[ordinal]
    return frame[TREE_MAP_COLUMNS + (["n_points_ground"] if emit_ground_variant else [])]


def _check_height_order(heights, centre_xy) -> str | None:
    """The heights file must be ordered like the sorted ids: each top is near its own stem, not the neighbour's."""
    h_x, h_y, _ = heights
    ok = np.isfinite(centre_xy[:, 0])
    if ok.sum() < 5:
        return None
    own = np.hypot(h_x - centre_xy[:, 0], h_y - centre_xy[:, 1])[ok]
    shifted = np.hypot(np.roll(h_x, 1) - centre_xy[:, 0], np.roll(h_y, 1) - centre_xy[:, 1])[ok]
    if np.median(own) > 0.5 * np.median(shifted):
        return (
            f"tree_heights.las does not look ordered like the stripe ids (median top-to-stem distance {np.median(own):.1f} m "
            f"vs {np.median(shifted):.1f} m for the neighbouring entry): th_m may be wrong"
        )
    return None


# ---------------------------------------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------------------------------------
def run_split(
    ctx: RunContext,
    *,
    max_axis_dist: float = 3.0,
    min_z0: float = 0.10,
    min_points: int = 500,
    emit_ground_variant: bool = False,
) -> dict:
    """Split ``dfin/<stem>_tree_ID_dist_axes.las`` into per-tree clouds ``split/trees/tree_XXXX.laz`` + ``tree_map.csv``.

    The tree number (ordinal) is the row label ``T<k>`` of 3DFin's table, found by aligning the table rows with the
    detected tree ids (:func:`lidar_bridge.idmap.align_rows_to_ids`). The dfin stage must have run in the same run.

    Parameters
    ----------
    max_axis_dist
        Keep points closer than this (m) to the tree's axis ("dist_axes" extra dimension).
    min_z0
        Drop points below this height (m) above the ground (ground / low vegetation): both ``Z0 >= min_z0`` (above 3DFin's
        terrain model at the point) and ``Z >= stem_ground_z + min_z0`` (above the horizontal plane through the stem base,
        so that the cloud's lowest point lies at the stem and not on the downhill side). Keep it small: both QSM tools
        measure breast height from the lowest point of a tree's cloud, so ``base_z0_m`` in ``tree_map.csv`` (that point's
        height above the ground at the stem, ~``min_z0``) is the offset a collector has to correct for.
    min_points
        Trees with fewer kept points are flagged ``low_points`` in ``tree_map.csv`` and their cloud is not written.
    emit_ground_variant
        Additionally write ``trees_ground/``: each tree's points plus a ground skirt (``Z0 >= -0.5`` and
        ``Z >= stem_ground_z - 0.5``).
    """
    if not max_axis_dist > 0:
        raise StageError(f"split.max_axis_dist must be > 0, got {max_axis_dist}")
    if min_points < 0:
        raise StageError(f"split.min_points must be >= 0, got {min_points}")
    dfin_dir = ctx.stage_dir("dfin")
    out_dir = ctx.stage_dir("split")
    if ctx.dry_run:
        log.warning("[dry-run] would split %s into %s (max_axis_dist=%s, min_z0=%s, min_points=%s, ground variant=%s)", dfin_dir, out_dir, max_axis_dist, min_z0, min_points, emit_ground_variant)
        return {"dir": str(ctx.rel(out_dir)), "dry_run": True}

    started = time.monotonic()
    files = DfinFiles.find(dfin_dir, ctx.input)
    config = idmap.read_dfin_config(files.dir, files.stem)
    tables = idmap.read_tables(files.dir, files.stem)
    unique_ids = idmap.read_stripe_ids(files.stripe)
    heights = _read_heights(files.heights)
    n = tables.n_rows
    log.info("3DFin run %r: %d table rows (T1..T%d), %d detected tree ids, %d tree-height points", files.stem, n, n, unique_ids.size, heights[0].size)

    # -- consistency of the 3DFin result files ----------------------------------------------------------
    if heights[0].size != unique_ids.size:
        raise StageError(
            f"{files.heights.name} has {heights[0].size} points but the stripe has {unique_ids.size} distinct tree ids "
            "(3DFin writes one height per detected tree): the files are not from the same run"
        )
    locator = _read_locator(files.locator)
    if locator is None:
        log.warning("%s not found: cannot cross-check the number of trees", files.locator.name)
    else:
        n_locator, loc_dbh = locator
        if n_locator != n:
            raise StageError(f"the table has {n} tree rows but {files.locator.name} has {n_locator} points: the files are not from the same run")
        atol = 1e-6 if tables.source == "xlsx" else 2e-3
        if not np.allclose(loc_dbh, tables.dbh, atol=atol, rtol=0):
            raise StageError(f"the DBH column of the table and {files.locator.name} disagree: the files are not from the same run")

    # -- tree ids <-> table rows -------------------------------------------------------------------------
    alignment = idmap.align_rows_to_ids(
        unique_ids, tables, files.cloud, z0_name=config.z0_name, section_wid=config.section_wid
    )
    rows = alignment.rows
    if len(rows) != n or not np.array_equal(rows["ordinal"].to_numpy(), np.arange(1, n + 1)):
        raise StageError("internal error: the id alignment did not return one ordinal per table row")
    sentinel = rows["raw_id"].isin(idmap.NON_TREE_IDS).to_numpy()
    if sentinel.any():
        log.warning(
            "tree id(s) %s are 3DFin's 'no tree' labels and cannot be split; those row(s) %s get no cloud",
            rows["raw_id"][sentinel].tolist(),
            rows["ordinal"][sentinel].tolist(),
        )
    if alignment.n_dropped:
        for _, d in alignment.dropped.iterrows():
            log.warning("3DFin dropped tree id %d (index %d of %d) from its tables: reason %s, margin %.1f", d.raw_id, d.u_index, unique_ids.size, d.reason, d.margin)
    for note in alignment.notes:
        log.warning("id alignment: %s", note)

    ordinal_of_u = np.zeros(unique_ids.size, dtype=np.int64)
    split_rows = ~sentinel
    ordinal_of_u[rows["u_index"].to_numpy()[split_rows]] = rows["ordinal"].to_numpy()[split_rows]

    # -- the cut -------------------------------------------------------------------------------------------
    ctx.reset_stage_dir("split")
    stats = split_cloud(
        files.cloud,
        unique_ids,
        ordinal_of_u,
        n,
        out_dir,
        z0_name=config.z0_name,
        max_axis_dist=max_axis_dist,
        min_z0=min_z0,
        emit_ground_variant=emit_ground_variant,
    )

    empty = [(int(o), int(r)) for o, r, s in zip(rows["ordinal"], rows["raw_id"], sentinel) if not s and stats.n_points[o] == 0]
    if empty:
        shutil.rmtree(out_dir / "trees", ignore_errors=True)
        shutil.rmtree(out_dir / "trees_ground", ignore_errors=True)
        raise StageError(
            f"{len(empty)} tree(s) have no points after the cut (max_axis_dist={max_axis_dist}, min_z0={min_z0}): "
            f"(ordinal, tree_ID) {empty[:10]}. Are the 3DFin files from the same run, or is the cut too tight?"
        )
    ground_source = stats.stem_ground.source[rows["u_index"].to_numpy()]
    fallback = rows["ordinal"][(ground_source != SOURCE_AXIS) & ~sentinel].tolist()  # (no table row of a no-tree label is split)
    if fallback:
        log.warning(
            "stem-base ground of %d tree(s) (ordinals %s) estimated from all points of the tree: fewer than %d points within %.1f m of the axis "
            "(stem_ground_source=tree in tree_map.csv; base_z0_m and the plane cut are less reliable there)",
            len(fallback), fallback[:20], STEM_GROUND_MIN_POINTS, STEM_GROUND_AXIS_DIST,
        )
    low = [int(o) for o in rows["ordinal"] if stats.n_points[o] < min_points]
    for ordinal in low:  # too small to be worth a QSM: listed in tree_map.csv only
        for sub in ("trees", "trees_ground"):
            (out_dir / sub / tree_file_name(ordinal, n)).unlink(missing_ok=True)
    written = {int(o) for o in rows["ordinal"] if stats.n_points[o] >= min_points and stats.n_points[o] > 0}

    problem = _check_height_order(heights, stats.centre_xy)
    if problem:
        log.warning(problem)

    tree_map = build_tree_map(
        alignment,
        tables,
        stats,
        heights,
        section_wid=config.section_wid,
        min_points=min_points,
        emit_ground_variant=emit_ground_variant,
        n_trees=n,
        written=written,
    )
    tree_map_path = out_dir / "tree_map.csv"
    excluded_path = out_dir / "excluded_ids.csv"
    tree_map.to_csv(tree_map_path, index=False, float_format="%.6f")
    excluded = _excluded_table(alignment, rows, sentinel, heights, stats)
    excluded.to_csv(excluded_path, index=False, float_format="%.6f")

    return _summary(ctx, alignment, tree_map, stats, unique_ids, tables, heights, min_points, out_dir, tree_map_path, excluded_path, written, time.monotonic() - started)


def _excluded_table(alignment, rows, sentinel, heights, stats) -> pd.DataFrame:
    h_x, h_y, h_z0 = heights
    parts = []
    if alignment.n_dropped:
        u = alignment.dropped["u_index"].to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "raw_id": alignment.dropped["raw_id"].to_numpy(),
                    "u_index": u,
                    "reason": alignment.dropped["reason"].to_numpy(),
                    "th_m": h_z0[u],
                    "x_m": stats.centre_xy[u, 0],
                    "y_m": stats.centre_xy[u, 1],
                    "margin": alignment.dropped["margin"].to_numpy(),
                }
            )
        )
    if sentinel.any():
        u = rows["u_index"].to_numpy()[sentinel]
        parts.append(
            pd.DataFrame(
                {
                    "raw_id": rows["raw_id"].to_numpy()[sentinel],
                    "u_index": u,
                    "reason": "non_tree_id",
                    "th_m": h_z0[u],
                    "x_m": stats.centre_xy[u, 0],
                    "y_m": stats.centre_xy[u, 1],
                    "margin": np.nan,
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["raw_id", "u_index", "reason", "th_m", "x_m", "y_m", "margin"])
    return pd.concat(parts, ignore_index=True)


def _summary(ctx, alignment, tree_map, stats, unique_ids, tables, heights, min_points, out_dir, tree_map_path, excluded_path, written, seconds) -> dict:
    n = len(tree_map)
    n_mismatch = int(tree_map["th_mismatch"].sum())
    n_low = int(tree_map["low_points"].sum())
    n_ambiguous = alignment.n_ambiguous
    counts = tree_map["n_points"]
    # how 3DFin's TH column relates to the tree it should describe: tree_heights[k] for the k-th row
    shifted = 0
    if n_mismatch:
        tol = TH_TOLERANCE_XLSX if tables.source == "xlsx" else TH_TOLERANCE_TXT
        shifted = int((np.abs(tables.th - heights[2][:n]) <= tol).sum())
    ground_note = f", {int(stats.n_ground.sum())} ground-variant points" if "n_points_ground" in tree_map else ""
    base = tree_map["base_z0_m"].dropna()
    base_note = f"base_z0_m min/median/max {base.min():.2f}/{base.median():.2f}/{base.max():.2f} m" if len(base) else "base_z0_m n/a"
    n_fallback = int(((tree_map["stem_ground_source"] != SOURCE_AXIS) & (tree_map["n_points"] > 0)).sum())
    log.info(
        "split: %d trees (%d detected ids, %d dropped %s); id confidence: %s; TH differs from the table in %d rows%s; "
        "%d tree(s) below %d points; %.1f M of %.1f M points kept (n_points min/median/max %d/%d/%d); %s, %d tree(s) with a fallback "
        "stem-base ground%s; %.0f s",
        n,
        unique_ids.size,
        alignment.n_dropped,
        alignment.dropped_ids,
        tree_map["id_confidence"].value_counts().to_dict(),
        n_mismatch,
        f" (the table's TH column is index-shifted by 3DFin: {shifted} of {n} table values equal tree_heights[row])" if n_mismatch else "",
        n_low,
        min_points,
        stats.points_kept / 1e6,
        stats.points_read / 1e6,
        int(counts.min()),
        int(counts.median()),
        int(counts.max()),
        base_note,
        n_fallback,
        ground_note,
        seconds,
    )
    if n_ambiguous:
        log.warning("%d row(s) have an ambiguous tree id (ordinals %s): see id_confidence in tree_map.csv", n_ambiguous, tree_map["ordinal"][tree_map["id_confidence"] == "ambiguous"].tolist()[:20])
    return {
        "dir": str(ctx.rel(out_dir)),
        "trees_dir": str(ctx.rel(out_dir / "trees")),
        "tree_map": str(ctx.rel(tree_map_path)),
        "excluded_ids": str(ctx.rel(excluded_path)),
        "n_trees": n,
        "n_written": len(written),
        "n_detected_ids": int(unique_ids.size),
        "n_dropped": alignment.n_dropped,
        "dropped_ids": alignment.dropped_ids,
        "n_ambiguous": n_ambiguous,
        "n_th_mismatch": n_mismatch,
        "n_low_points": n_low,
        "n_stem_ground_fallback": n_fallback,
        "n_dbh_missing": int(tree_map["dbh_missing"].sum()),
        "points_read": stats.points_read,
        "points_kept": stats.points_kept,
        "seconds": round(seconds, 1),
    }

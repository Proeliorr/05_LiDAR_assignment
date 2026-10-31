"""Builder of a miniature fake 3DFin result directory (``data/out/<run>/dfin/``) for the split / idmap tests.

It reproduces what matters for the tree-id alignment, not 3DFin's algorithms:

* ``<stem>_stripe.las``, ``_tree_ID_dist_axes.las`` (point format 7, extra dims Class, Split, dist_axes, tree_ID, Z0 like
  the real file, non-default scale/offset), ``_tree_heights.las`` (one point per detected tree, ascending tree_ID order),
  ``_tree_locator.las`` (one point per TABLE row), ``_config.ini`` and ``<stem>.xlsx`` laid out like 3DFin's workbook
  (info text in the first rows, header row ``TH DBH X Y`` NOT first, labels ``T1..Tn`` one column left of it).
* Trees are vertical cylinders on a grid. Sparse raw ids such as ``0, 7, 42, 9001, 630740`` (0 is a REAL tree).
* ``dropped``: raw ids that 3DFin detected but that have no table row. The table is then built the way 3DFin 0.6.0 does it:
  ``TH`` of row k = height of the k-th of ALL detected trees, and X,Y of rows without DBH = axis position of the k-th of ALL
  trees (both therefore belong to the previous tree after the dropped id).
* The cloud carries crown / far / ground / deep-ground / noise (-1) / unassigned (100000, NOT in the stripe) points to exercise
  the filters. (3DFin 0.6.0 writes no 100000 sentinel; the points only prove that ids outside the stripe are never split.)
* The terrain is the plane ``GROUND_Z + slope_x * (x - 100) + slope_y * (y - 200)`` and ``Z0 = Z - terrain(x, y)`` exactly like
  3DFin's normalisation, so on a slope (``slope_x=0.3``) the points of a tree differ in absolute Z although they have the same
  Z0. ``brush=True`` adds low vegetation (Z0 0.1 .. 0.4 m, 0.8 .. 2.5 m from the axis): on a slope the downhill part of it lies
  far below the ground at the stem although its Z0 passes a ``min_z0`` cut.
* The point clouds are written with a fixed header ``creation_date`` (``CREATION_DATE``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

DEFAULT_IDS = (0, 7, 42, 9001, 630740)
SECTION_WID = 0.05
SECTION_LEN = 0.2
SECTION_MIN = 0.3
SCALES = (0.001, 0.001, 0.001)
OFFSETS = (100.0, 200.0, 50.0)
GROUND_Z = 100.0
CREATION_DATE = date(2024, 3, 5)

#: filter bands used for the "other" points of every tree
CROWN_DIST = (1.0, 2.5)  # kept by the default max_axis_dist=3
FAR_DIST = (3.3, 4.5)  # always dropped by the default max_axis_dist
SKIRT_Z0 = (-0.45, 0.09)  # dropped by min_z0=0.10, kept by the ground variant
DEEP_Z0 = (-1.2, -0.55)  # dropped by both
BRUSH_Z0 = (0.10, 0.40)  # low vegetation (brush=True): passes a min_z0=0.10 cut on Z0 alone
BRUSH_DIST = (0.8, 2.5)
BRUSH_N = 60


def terrain(x, y, slope_x: float, slope_y: float):
    """Elevation of the planar terrain model at ``(x, y)``, i.e. ``Z - Z0`` of a point there."""
    return GROUND_Z + slope_x * (np.asarray(x) - 100.0) + slope_y * (np.asarray(y) - 200.0)


@dataclass
class SynthTree:
    raw_id: int
    cx: float
    cy: float
    radius: float
    height: float  # the tree's true total height (z0 of its tree_heights.las point)


@dataclass
class SynthRun:
    dfin_dir: Path
    stem: str
    trees: list[SynthTree]  # all detected trees, ascending raw id (= U)
    rows: list[SynthTree]  # trees with a table row, in row order
    dropped: list[int]
    sections: np.ndarray
    th_table: np.ndarray  # TH column as 3DFin writes it
    dbh: np.ndarray
    xy_table: np.ndarray  # (n, 2) X, Y columns as 3DFin writes them
    radius: np.ndarray  # (n, J)
    x_c: np.ndarray
    y_c: np.ndarray
    cloud: dict[str, np.ndarray] = field(default_factory=dict)  # source columns of the dist_axes cloud
    n_points: dict[int, int] = field(default_factory=dict)  # raw id -> number of points in the cloud
    slope_x: float = 0.02
    slope_y: float = 0.0

    def terrain(self, x, y):
        return terrain(x, y, self.slope_x, self.slope_y)

    def stem_ground(self, raw_id: int, max_dist: float = 0.5) -> float:
        """Mean of ``Z - Z0`` over the points of ``raw_id`` closer than ``max_dist`` to its axis (what split must find)."""
        c = self.cloud
        m = (c["tid"] == raw_id) & (c["dist"] < max_dist)
        return float(np.mean(c["z"][m] - c["z0"][m]))

    @property
    def ids(self) -> list[int]:
        return [t.raw_id for t in self.trees]

    @property
    def row_ids(self) -> list[int]:
        return [t.raw_id for t in self.rows]

    def expected_ordinal(self) -> dict[int, int]:
        return {t.raw_id: i + 1 for i, t in enumerate(self.rows)}


# ---------------------------------------------------------------------------------------------------------
# LAS writers
# ---------------------------------------------------------------------------------------------------------
def write_las(
    path: Path, point_format: int, xyz: np.ndarray, extras: dict[str, tuple[type, np.ndarray]], *, scales=SCALES, offsets=OFFSETS, creation_date=CREATION_DATE
) -> None:
    las = laspy.create(point_format=point_format, file_version="1.4")
    las.header.creation_date = creation_date
    las.change_scaling(scales=np.array(scales), offsets=np.array(offsets))  # before the coordinates are quantised
    las.xyz = xyz
    las.add_extra_dims([laspy.ExtraBytesParams(name=name, type=dtype) for name, (dtype, _) in extras.items()])
    for name, (_, values) in extras.items():
        las[name] = values
    las.write(str(path))


# ---------------------------------------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------------------------------------
def make_trees(ids, rng: np.random.Generator) -> list[SynthTree]:
    ids = sorted(int(i) for i in ids)
    n = len(ids)
    cols = int(np.ceil(np.sqrt(n)))
    cells = rng.permutation(n)  # ids are NOT in spatial order, like real cluster labels
    trees = []
    for i, raw in enumerate(ids):
        cell = int(cells[i])
        trees.append(
            SynthTree(
                raw_id=raw,
                cx=105.0 + 5.0 * (cell % cols) + rng.uniform(-0.5, 0.5),
                cy=205.0 + 5.0 * (cell // cols) + rng.uniform(-0.5, 0.5),
                radius=round(0.08 + 0.015 * i + rng.uniform(0, 0.01), 4),
                height=round(5.2 + 0.37 * i, 3),
            )
        )
    return trees


def _tree_points(t: SynthTree, sections: np.ndarray, rng: np.random.Generator, n_scale: float) -> dict[str, np.ndarray]:
    """Points of one tree: (x, y, z0, dist_axes) arrays for the stem windows, the stem between windows and the clutter."""
    k = lambda n: max(3, int(n * n_scale))  # noqa: E731
    z0, dist = [], []
    for b in sections:  # points inside the section windows, on the circle
        if b + SECTION_WID < t.height - 0.1:
            m = k(30)
            z0.append(rng.uniform(b, b + SECTION_WID, m))
            dist.append(t.radius + rng.normal(0, 0.003, m))
            z0.append(rng.uniform(b, b + SECTION_WID, 4))  # a little understory clutter near the axis
            dist.append(rng.uniform(0.05, 0.6, 4))
    z0.append(rng.uniform(0.1, t.height, k(120)))  # stem between the windows
    dist.append(t.radius + rng.normal(0, 0.003, k(120)))
    z0.append(rng.uniform(0.6 * t.height, t.height, k(150)))  # crown
    dist.append(rng.uniform(*CROWN_DIST, k(150)))
    z0.append(rng.uniform(0.5, t.height, k(60)))  # far branches
    dist.append(rng.uniform(*FAR_DIST, k(60)))
    z0.append(rng.uniform(*SKIRT_Z0, k(50)))  # ground skirt
    dist.append(rng.uniform(0.1, 2.0, k(50)))
    z0.append(rng.uniform(*DEEP_Z0, k(30)))  # below the skirt
    dist.append(rng.uniform(0.1, 2.0, k(30)))
    z0, dist = np.concatenate(z0), np.concatenate(dist)
    ang = rng.uniform(0, 2 * np.pi, z0.size)
    return {"x": t.cx + dist * np.cos(ang), "y": t.cy + dist * np.sin(ang), "z0": z0, "dist": dist}


def build_dfin(
    dfin_dir: Path,
    *,
    stem: str = "plot_01",
    ids=DEFAULT_IDS,
    dropped=(),
    no_circle_ids=(),
    sparse_ids=(),
    seed: int = 7,
    export_txt: bool = False,
    nontree_ids: bool = True,
    p_missing: float = 0.2,
    slope_x: float = 0.02,
    slope_y: float = 0.0,
    brush: bool = False,
) -> SynthRun:
    """Write the fake 3DFin outputs into ``dfin_dir`` and return the ground truth.

    ``dropped``: ids detected by 3DFin but missing from its tables. ``no_circle_ids``: rows of these trees have no circle
    (no evidence for the alignment). ``sparse_ids``: trees with only a few points. ``ids`` may contain -1 (3DFin's noise
    label) or 100000 (an ordinary id for 3DFin). ``slope_x`` / ``slope_y``: gradient of the terrain model. ``brush``: add
    the low vegetation band (``BRUSH_*``); it uses its own random generator so the other points do not change.
    """
    rng = np.random.default_rng(seed)
    dfin_dir.mkdir(parents=True, exist_ok=True)
    trees = make_trees(ids, rng)
    dropped = sorted(int(d) for d in dropped)
    if not set(dropped) <= {t.raw_id for t in trees}:
        raise ValueError("dropped ids must be among the detected ids")
    rows = [t for t in trees if t.raw_id not in dropped]
    n = len(rows)
    max_h = max(t.height for t in trees)
    sections = np.arange(SECTION_MIN, max_h - 0.2, SECTION_LEN)
    n_sec = sections.size
    sparse = set(sparse_ids)

    # -- point cloud -------------------------------------------------------------------------------------
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("x", "y", "z0", "dist", "tid")}
    n_points: dict[int, int] = {}
    for t in trees:
        p = _tree_points(t, sections, rng, 0.05 if t.raw_id in sparse else 1.0)
        n_points[t.raw_id] = p["x"].size
        for key in ("x", "y", "z0", "dist"):
            cols[key].append(p[key])
        cols["tid"].append(np.full(p["x"].size, t.raw_id, dtype=np.int32))
    if brush:
        brng = np.random.default_rng(seed + 1000)
        for t in trees:
            dist = brng.uniform(*BRUSH_DIST, BRUSH_N)
            ang = brng.uniform(0, 2 * np.pi, BRUSH_N)
            cols["x"].append(t.cx + dist * np.cos(ang))
            cols["y"].append(t.cy + dist * np.sin(ang))
            cols["z0"].append(brng.uniform(*BRUSH_Z0, BRUSH_N))
            cols["dist"].append(dist)
            cols["tid"].append(np.full(BRUSH_N, t.raw_id, dtype=np.int32))
            n_points[t.raw_id] += BRUSH_N
    if nontree_ids:
        for label, lo, hi in ((-1, 0.2, 0.5), (100000, 5.0, 10.0)):  # noise near a stem / unassigned far away
            m = 120
            base = trees[0]
            dist = rng.uniform(lo, hi, m)
            ang = rng.uniform(0, 2 * np.pi, m)
            cols["x"].append(base.cx + dist * np.cos(ang))
            cols["y"].append(base.cy + dist * np.sin(ang))
            cols["z0"].append(rng.uniform(0.5, 3.0, m))
            cols["dist"].append(dist)
            cols["tid"].append(np.full(m, label, dtype=np.int32))
    cloud = {k: np.concatenate(v) for k, v in cols.items()}
    order = rng.permutation(cloud["x"].size)
    cloud = {k: v[order] for k, v in cloud.items()}
    total = cloud["x"].size
    cloud["klass"] = (np.arange(total) % 5).astype(np.int8)
    cloud["split"] = (np.arange(total) % 3).astype(np.int8)
    z = cloud["z0"] + terrain(cloud["x"], cloud["y"], slope_x, slope_y)
    cloud_path = dfin_dir / f"{stem}_tree_ID_dist_axes.las"
    write_las(
        cloud_path,
        7,
        np.column_stack([cloud["x"], cloud["y"], z]),
        {
            "Class": (np.int8, cloud["klass"]),
            "Split": (np.int8, cloud["split"]),
            "dist_axes": (np.float64, cloud["dist"]),
            "tree_ID": (np.int32, cloud["tid"]),
            "Z0": (np.float64, cloud["z0"]),
        },
    )
    written = laspy.read(cloud_path)  # the coordinates as stored (quantised): what the stages see
    cloud["x"], cloud["y"], cloud["z"] = np.asarray(written.x), np.asarray(written.y), np.asarray(written.z)

    # -- stripe: a few points of every detected tree (the authoritative id list) --------------------------
    sx, sy, sz, sid = [], [], [], []
    for t in trees:
        m = 40
        ang = rng.uniform(0, 2 * np.pi, m)
        sx.append(t.cx + t.radius * np.cos(ang))
        sy.append(t.cy + t.radius * np.sin(ang))
        sz.append(rng.uniform(0.7, 3.5, m) + GROUND_Z)
        sid.append(np.full(m, t.raw_id, dtype=np.int32))
    write_las(
        dfin_dir / f"{stem}_stripe.las",
        2,
        np.column_stack([np.concatenate(sx), np.concatenate(sy), np.concatenate(sz)]),
        {"tree_ID": (np.int32, np.concatenate(sid))},
        scales=(0.01, 0.01, 0.01),
        offsets=(0.0, 0.0, 0.0),
    )

    # -- heights: one point per DETECTED tree, ascending id; the top is up to ~1 m off the stem ----------
    write_las(
        dfin_dir / f"{stem}_tree_heights.las",
        2,
        np.array([[t.cx + 0.4, t.cy - 0.3, GROUND_Z + t.height] for t in trees]),
        {"z0": (np.float64, np.array([t.height for t in trees])), "deviated": (np.int32, np.ones(len(trees), dtype=np.int32))},
        scales=(0.01, 0.01, 0.01),
        offsets=(0.0, 0.0, 0.0),
    )

    # -- the tables, built the way 3DFin 0.6.0 builds them -----------------------------------------------
    radius = np.zeros((n, n_sec))
    x_c = np.zeros((n, n_sec))
    y_c = np.zeros((n, n_sec))
    for r, t in enumerate(rows):
        if t.raw_id in no_circle_ids:
            continue
        for j, b in enumerate(sections):
            if b + SECTION_WID < t.height - 0.1 and rng.random() > p_missing:
                radius[r, j] = t.radius + rng.normal(0, 0.001)
                x_c[r, j] = t.cx + rng.normal(0, 0.004)
                y_c[r, j] = t.cy + rng.normal(0, 0.004)
    j0 = int(np.argmin(np.abs(sections - 1.3)))
    dbh = np.zeros(n)
    xy = np.zeros((n, 2))
    for r in range(n):
        near = slice(max(0, j0 - 1), j0 + 2)
        if np.all(radius[r, near] > 0):
            dbh[r] = 2.0 * radius[r, j0]
            xy[r] = x_c[r, j0], y_c[r, j0]
        else:  # axis location of the r-th of ALL detected trees (3DFin indexes by row, not by tree)
            xy[r] = trees[r].cx, trees[r].cy
    th = np.array([trees[r].height for r in range(n)])  # tree_heights[r] truncated to n rows

    write_las(
        dfin_dir / f"{stem}_tree_locator.las",
        2,
        np.column_stack([xy, np.full(n, GROUND_Z + 1.3)]),
        {"diameters": (np.float64, dbh)},
        scales=(0.01, 0.01, 0.01),
        offsets=(0.0, 0.0, 0.0),
    )
    if export_txt:
        base = str(dfin_dir / stem)
        np.savetxt(base + "_dbh_and_heights.txt", np.column_stack([th, dbh, xy]), fmt="%.3f")
        np.savetxt(base + "_diameters.txt", radius * 2, fmt="%.3f")
        np.savetxt(base + "_X_c.txt", x_c, fmt="%.3f")
        np.savetxt(base + "_Y_c.txt", y_c, fmt="%.3f")
        np.savetxt(base + "_sections.txt", np.column_stack(sections), fmt="%.3f")
    else:
        write_xlsx(dfin_dir / f"{stem}.xlsx", th, dbh, xy, radius, x_c, y_c, sections)
    (dfin_dir / f"{stem}_config.ini").write_text(
        f"[basic]\nz0_name = Z0\n\n[advanced]\nsection_len = {SECTION_LEN}\nsection_wid = {SECTION_WID}\n", encoding="utf-8"
    )
    return SynthRun(dfin_dir, stem, trees, rows, dropped, sections, th, dbh, xy, radius, x_c, y_c, cloud, n_points, slope_x, slope_y)


def write_xlsx(path: Path, th, dbh, xy, radius, x_c, y_c, sections) -> None:
    """The workbook layout of three_d_fin.processing.io: info text rows, then the data frame at row 3 / column B."""
    n, n_sec = radius.shape
    labels = [f"T{i + 1}" for i in range(n)]
    s_cols = [f"S{i + 1}" for i in range(n_sec)]
    frames = {
        "Plot Metrics": pd.DataFrame(np.column_stack([th, dbh, xy]), index=labels, columns=["TH", "DBH", "X", "Y"]),
        "Diameters": pd.DataFrame(radius * 2, index=labels, columns=s_cols),
        "X": pd.DataFrame(x_c, index=labels, columns=s_cols),
        "Y": pd.DataFrame(y_c, index=labels, columns=s_cols),
        "Sections": pd.DataFrame([sections], index=["Z0"], columns=s_cols),
    }
    info = {
        "Plot Metrics": "Total height (TH) of each tree (T).\n        Diameter at breast height (DBH) of each tree (T).\n        (x, y) coordinates (X and Y) of each tree (T).\n        ",
        "Diameters": "Diameter of every section (S) of every tree (T).\n            Units are meters.\n            ",
        "X": "(x) coordinate of the centre of every section (S) of every tree (T).",
        "Y": "(y) coordinate of the centre of every section (S) of every tree (T).",
        "Sections": "Normalized height (Z0) of every section (S).\n        Units are meters.",
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            pd.Series(info[name]).to_excel(writer, sheet_name=name, header=False, index=False, merge_cells=False)
            if name == "Plot Metrics":
                pd.Series("This cloud has 0.001 million points and its area is 100 m2").to_excel(
                    writer, sheet_name=name, startrow=1, header=False, index=False, merge_cells=False
                )
            frame.to_excel(writer, sheet_name=name, startrow=2, startcol=1)
        # the quality sheets the real workbook has, not needed here but they must not confuse the reader
        quality = pd.DataFrame(np.ones((n, n_sec)), index=labels, columns=s_cols)
        pd.Series("Overall quality of every section (S) of every tree (T).").to_excel(
            writer, sheet_name="Q(Overall Quality 0-1)", header=False, index=False, merge_cells=False
        )
        quality.to_excel(writer, sheet_name="Q(Overall Quality 0-1)", startrow=2, startcol=1)


def remove_detected_tree(run: SynthRun, raw_id: int) -> None:
    """Make 3DFin "detect" one tree fewer than the table has rows: drop ``raw_id`` from the stripe and the heights file."""
    stripe = run.dfin_dir / f"{run.stem}_stripe.las"
    las = laspy.read(stripe)
    las.points = las.points[np.asarray(las["tree_ID"]) != raw_id]
    las.write(str(stripe))
    heights = run.dfin_dir / f"{run.stem}_tree_heights.las"
    las = laspy.read(heights)
    index = run.ids.index(raw_id)
    las.points = las.points[np.arange(len(las.points)) != index]
    las.write(str(heights))

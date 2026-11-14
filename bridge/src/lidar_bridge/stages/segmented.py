"""Split step for a cloud that arbor has already segmented, driven by a key table.

3DFin and arbor are not run again. The key table lists the trees to take (tree number and the id arbor gave the tree in
``id_dim``) and, optionally, the 3DFin and arbor results for each. The step cuts those trees out of the cloud and writes
what the later stages read: ``split/trees/tree_XXXX.laz`` and ``split/tree_map.csv`` (the "3dfin" results for
``compare``), and ``arbor/metrics.csv`` with the arbor results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from lidar_bridge.paths import RunContext
from lidar_bridge.stages._common import StageError, tree_map_fingerprint, write_fingerprint
from lidar_bridge.stages.split import CHUNK_SIZE, _output_header, _TreeWriters, tree_file_name

log = logging.getLogger("lidar_bridge.stages.segmented")

DEFAULT_COLUMNS = {
    "tree": "Lp",
    "id": "ID_Arbor",
    "dfin_dbh_cm": "DBH_3DFin_cm",
    "dfin_height_m": "H_3DFin_m",
    "arbor_dbh_cm": "DBH_Arbor_cm",
    "arbor_height_m": "H_Arbor_m",
    "arbor_note": "Arbor_log",
}


def read_key(path: Path, sheet: str | int, key_columns: dict | None) -> pd.DataFrame:
    """The key table with columns renamed to the roles of :data:`DEFAULT_COLUMNS` (absent optional roles are left out)."""
    if not path.is_file():
        raise StageError(f"key table {path} not found")
    raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=sheet)
    titles = {**DEFAULT_COLUMNS, **(key_columns or {})}
    for role in ("tree", "id"):
        if titles[role] not in raw.columns:
            raise StageError(f"key table {path.name}: no column {titles[role]!r} for the {role}; columns found: {list(raw.columns)}")
    key = pd.DataFrame({role: raw[title] for role, title in titles.items() if title in raw.columns})
    key["tree"], key["id"] = key["tree"].astype(int), key["id"].astype(int)
    for role in ("tree", "id"):
        if key[role].duplicated().any():
            raise StageError(f"key table {path.name}: duplicate values in the {role} column: {sorted(key[role][key[role].duplicated()])}")
    return key


def cut_trees(cloud: Path, ids: dict[int, int], id_dim: str, trees_dir: Path, chunk_size: int = CHUNK_SIZE):
    """Write the points of every listed id to its tree file. Returns ``(n_points, min_hag)`` per tree number.

    ``ids`` maps an id of ``id_dim`` to the tree number. ``min_hag`` is ``None`` when the cloud has no ``hag`` dimension.
    """
    n_points = dict.fromkeys(ids.values(), 0)
    min_hag = dict.fromkeys(ids.values(), np.inf)
    wanted = np.array(sorted(ids))
    with laspy.open(cloud) as reader:
        names = list(reader.header.point_format.dimension_names)
        if id_dim not in names:
            raise StageError(f"{cloud.name} has no dimension {id_dim!r}; dimensions: {names}")
        has_hag = "hag" in names
        if not has_hag:
            log.warning("%s has no 'hag' dimension: base_z0_m is set to 0", cloud.name)
        writers = _TreeWriters(trees_dir, _output_header(reader.header), max(ids.values()))
        for chunk in reader.chunk_iterator(chunk_size):
            tid = np.asarray(chunk[id_dim]).astype(np.int64)
            for tree_id in np.intersect1d(wanted, np.unique(tid)):
                tree, sel = ids[int(tree_id)], np.flatnonzero(tid == tree_id)
                writers.write(tree, chunk[sel])
                n_points[tree] += sel.size
                if has_hag:
                    min_hag[tree] = min(min_hag[tree], float(np.min(np.asarray(chunk["hag"])[sel])))
        writers.close()
    empty = sorted(tree_id for tree_id, tree in ids.items() if n_points[tree] == 0)
    if empty:
        raise StageError(f"{id_dim} value(s) {empty} from the key table have no points in {cloud.name}")
    return n_points, (min_hag if has_hag else dict.fromkeys(ids.values(), 0.0))


def run_segmented_split(
    ctx: RunContext, key: str, key_sheet: str | int, key_columns: dict | None, id_dim: str, min_points: int
) -> dict:
    """Cut the key table's trees out of ``ctx.input`` and write ``split/`` and (with arbor columns) ``arbor/metrics.csv``."""
    if ctx.input is None:
        raise StageError("split.key needs the segmented cloud: give it as the input file")
    out_dir, arbor_dir = ctx.stage_dir("split"), ctx.stage_dir("arbor")
    if ctx.dry_run:
        log.warning("[dry-run] would cut the trees of %s out of %s into %s", key, ctx.input, out_dir)
        return {"dir": str(ctx.rel(out_dir)), "dry_run": True}
    key_table = read_key(ctx.resolve(key), key_sheet, key_columns)
    trees_dir = out_dir / "trees"
    for old in trees_dir.glob("*.laz"):
        old.unlink()
    n_points, min_hag = cut_trees(ctx.input, dict(zip(key_table["id"], key_table["tree"])), id_dim, trees_dir)

    n_trees = int(key_table["tree"].max())
    tree_map = pd.DataFrame({"ordinal": key_table["tree"], "source_id": key_table["id"]})
    tree_map["n_points"] = tree_map["ordinal"].map(n_points)
    tree_map["base_z0_m"] = tree_map["ordinal"].map(min_hag)
    tree_map["low_points"] = tree_map["n_points"] < min_points
    tree_map["file"] = [f"trees/{tree_file_name(o, n_trees)}" for o in tree_map["ordinal"]]
    for ordinal in tree_map["ordinal"][tree_map["low_points"]]:
        (trees_dir / tree_file_name(ordinal, n_trees)).unlink()
    tree_map.loc[tree_map["low_points"], "file"] = ""
    tree_map["id_confidence"] = "exact"
    if "dfin_dbh_cm" in key_table:
        tree_map["dbh_m"] = key_table["dfin_dbh_cm"] / 100
    if "dfin_height_m" in key_table:
        tree_map["th_m"] = key_table["dfin_height_m"]
    tree_map = tree_map.sort_values("ordinal")
    tree_map.to_csv(out_dir / "tree_map.csv", index=False)

    if {"arbor_dbh_cm", "arbor_height_m"} <= set(key_table.columns):
        write_arbor_metrics(key_table, arbor_dir)
        write_fingerprint(arbor_dir, tree_map_fingerprint(out_dir), extra="from key table")

    base = tree_map["base_z0_m"]
    log.info("split from key table: %d trees, base_z0_m min/median/max %.2f/%.2f/%.2f m", len(tree_map), base.min(), base.median(), base.max())
    return {
        "dir": str(ctx.rel(out_dir)),
        "trees_dir": str(ctx.rel(trees_dir)),
        "tree_map": str(ctx.rel(out_dir / "tree_map.csv")),
        "n_trees": len(tree_map),
        "base_z0_m_min": float(base.min()),
        "base_z0_m_median": float(base.median()),
        "base_z0_m_max": float(base.max()),
    }


def write_arbor_metrics(key_table: pd.DataFrame, arbor_dir: Path) -> None:
    """``arbor/metrics.csv`` from the key table's arbor columns (the note is arbor's log text unless it is ``OK``)."""
    note = key_table["arbor_note"].fillna("").astype(str) if "arbor_note" in key_table else pd.Series("", index=key_table.index)
    metrics = pd.DataFrame(
        {
            "ordinal": key_table["tree"],
            "dbh_m": key_table["arbor_dbh_cm"] / 100,
            "height_m": key_table["arbor_height_m"],
            "status": "ok",
            "seconds": "",
            "note": note.where(note != "OK", ""),
        }
    ).sort_values("ordinal")
    arbor_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(arbor_dir / "metrics.csv", index=False)

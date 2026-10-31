"""The split stage on fake 3DFin output (tests/synth3dfin.py): ordinals, 3DFin's TH shift, the cut, the outputs."""

from __future__ import annotations

import datetime
import inspect
import logging
import shutil
from dataclasses import replace
import laspy
import numpy as np
import pandas as pd
import pytest
import synth3dfin as synth

from lidar_bridge.paths import RunContext
from lidar_bridge.stages import split
from lidar_bridge.stages._common import StageError

IDS = synth.DEFAULT_IDS + (11, 12, 13)  # sorted: 0, 7, 11, 12, 13, 42, 9001, 630740


def build(ctx: RunContext, **kw) -> synth.SynthRun:
    kw.setdefault("ids", IDS)
    return synth.build_dfin(ctx.stage_dir("dfin"), **kw)


def read_map(ctx: RunContext) -> pd.DataFrame:
    return pd.read_csv(ctx.stage_dir("split") / "tree_map.csv")


def tree_files(ctx: RunContext) -> list[str]:
    return sorted(p.name for p in (ctx.stage_dir("split") / "trees").glob("*.laz"))


def expected_mask(run: synth.SynthRun, raw_id: int, max_axis_dist: float = 3.0, min_z0: float = 0.10) -> np.ndarray:
    """The cut as documented: own id, near the axis, Z0 >= min_z0 AND above the horizontal plane min_z0 over the stem base."""
    c = run.cloud
    plane = run.stem_ground(raw_id) + min_z0
    return (c["tid"] == raw_id) & (c["dist"] <= max_axis_dist) & (c["z0"] >= min_z0) & (c["z"] >= plane)


def old_mask(run: synth.SynthRun, raw_id: int, max_axis_dist: float = 3.0, min_z0: float = 0.10) -> np.ndarray:
    """The cut before the plane condition existed (Z0 only): its lowest point lies on the downhill side on a slope."""
    c = run.cloud
    return (c["tid"] == raw_id) & (c["dist"] <= max_axis_dist) & (c["z0"] >= min_z0)


# -- signature / naming -----------------------------------------------------------------------------------------
def test_signature_defaults_are_the_documented_ones():
    params = inspect.signature(split.run_split).parameters
    assert {k: v.default for k, v in params.items() if k != "ctx"} == dict(
        max_axis_dist=3.0, min_z0=0.10, min_points=500, emit_ground_variant=False
    )


def test_tree_file_names_are_zero_padded_and_widen_for_big_plots():
    assert split.tree_file_name(1, 123) == "tree_0001.laz"
    assert split.tree_file_name(9999, 9999) == "tree_9999.laz"
    assert split.tree_file_name(7, 10000) == "tree_00007.laz"
    assert split.tree_file_name(123456, 123456) == "tree_123456.laz"


# -- (a) nothing dropped ----------------------------------------------------------------------------------------
def test_no_dropped_id_ordinal_is_rank_and_id_zero_is_a_real_tree(ctx):
    run = build(ctx)
    result = split.run_split(ctx)
    tm = read_map(ctx)
    assert tm["ordinal"].tolist() == list(range(1, 9))
    assert tm["raw_id"].tolist() == sorted(IDS) and tm["raw_id"].iloc[0] == 0
    assert set(tm["id_confidence"]) == {"exact"}
    assert not tm["th_mismatch"].any()
    assert tm["th_m"].tolist() == pytest.approx([t.height for t in run.trees])
    assert tree_files(ctx) == [f"tree_{i:04d}.laz" for i in range(1, 9)]
    assert (pd.read_csv(ctx.stage_dir("split") / "excluded_ids.csv")).empty
    assert result["n_trees"] == 8 and result["n_dropped"] == 0 and result["n_ambiguous"] == 0 and result["n_th_mismatch"] == 0
    assert result["dropped_ids"] == []


# -- (b) one id dropped, 3DFin's TH / XY shift ------------------------------------------------------------------------
def test_dropped_id_in_the_middle_with_the_3dfin_shift(ctx, caplog):
    caplog.set_level(logging.INFO, logger="lidar_bridge")
    run = build(ctx, dropped=(42,))  # sorted ids ...13, 42, 9001...: 42 has index 5
    result = split.run_split(ctx)
    tm = read_map(ctx)
    height = {t.raw_id: t.height for t in run.trees}
    truth = {t.raw_id: t for t in run.trees}

    assert tm["ordinal"].tolist() == list(range(1, 8)) and tm["raw_id"].tolist() == run.row_ids
    assert 42 not in tm["raw_id"].tolist() and set(tm["id_confidence"]) == {"resolved"}
    assert tree_files(ctx) == [f"tree_{i:04d}.laz" for i in range(1, 8)]
    # the correct height is the tree's own, the table's TH is the one of the previous entry from the drop on
    assert tm["th_m"].tolist() == pytest.approx([height[i] for i in run.row_ids])
    assert tm["th_xlsx_m"].tolist() == pytest.approx(run.th_table.tolist())
    assert tm["th_mismatch"].tolist() == [False] * 5 + [True] * 2  # rows 6, 7 (ids 9001, 630740) are after the drop
    assert result["n_th_mismatch"] == 2 and result["dropped_ids"] == [42] and result["n_dropped"] == 1
    # positions: the DBH>0 rows are the table's, the other rows are NOT the shifted axis location of the table
    for _, row in tm.iterrows():
        tree = truth[int(row["raw_id"])]
        assert np.hypot(row["x_m"] - tree.cx, row["y_m"] - tree.cy) < 0.3, row.to_dict()
        assert (row["xy_source"] == "xlsx") == (not row["dbh_missing"])
        assert row["dbh_m"] == pytest.approx(2 * tree.radius, abs=0.01) if not row["dbh_missing"] else np.isnan(row["dbh_m"])
    assert (tm["xy_cloud_offset_m"] < 0.5).all() and (tm["th_xy_offset_m"] < 1.5).all()
    # excluded
    ex = pd.read_csv(ctx.stage_dir("split") / "excluded_ids.csv")
    assert ex["raw_id"].tolist() == [42] and ex["reason"].tolist() == ["no_stem_section_rows"] and ex["u_index"].tolist() == [5]
    assert ex["th_m"].iloc[0] == pytest.approx(height[42])
    # every file holds exactly its own id
    for ordinal, raw in enumerate(run.row_ids, start=1):
        ids = np.unique(np.asarray(laspy.read(ctx.stage_dir("split") / "trees" / f"tree_{ordinal:04d}.laz")["tree_ID"]))
        assert ids.tolist() == [raw]
    assert "dropped tree id 42" in caplog.text and "TH differs from the table in 2 rows" in caplog.text


def test_dropped_first_id_and_zero_is_not_confused_with_missing(ctx):
    run = build(ctx, dropped=(0,))
    split.run_split(ctx)
    tm = read_map(ctx)
    assert tm["raw_id"].tolist() == sorted(IDS)[1:] and tm["th_mismatch"].all()
    assert pd.read_csv(ctx.stage_dir("split") / "excluded_ids.csv")["raw_id"].tolist() == [0]
    assert run.dropped == [0]


def test_ambiguous_alignment_is_reported_not_hidden(ctx, caplog):
    build(ctx, ids=synth.DEFAULT_IDS, dropped=(42,), no_circle_ids=(7, 9001))
    result = split.run_split(ctx)
    tm = read_map(ctx)
    assert tm["id_confidence"].tolist() == ["resolved", "ambiguous", "ambiguous", "resolved"]
    assert result["n_ambiguous"] == 2
    assert "ambiguous tree id" in caplog.text


# -- (c) inconsistent inputs ------------------------------------------------------------------------------------
def test_more_table_rows_than_detected_trees_fails(ctx):
    run = build(ctx)
    synth.remove_detected_tree(run, 9001)
    with pytest.raises(StageError, match="8 tree rows but only 7"):
        split.run_split(ctx)


def test_locator_point_count_must_equal_table_rows(ctx):
    run = build(ctx, dropped=(42,))
    path = run.dfin_dir / "plot_01_tree_locator.las"
    las = laspy.read(path)
    las.points = las.points[:-1]
    las.write(str(path))
    with pytest.raises(StageError, match="7 tree rows but plot_01_tree_locator.las has 6"):
        split.run_split(ctx)


def test_locator_diameters_must_match_the_table(ctx):
    run = build(ctx)
    path = run.dfin_dir / "plot_01_tree_locator.las"
    las = laspy.read(path)
    las["diameters"] = np.asarray(las["diameters"]) + 0.05
    las.write(str(path))
    with pytest.raises(StageError, match="DBH"):
        split.run_split(ctx)


def test_heights_count_must_equal_detected_trees(ctx):
    run = build(ctx, dropped=(42,))
    path = run.dfin_dir / "plot_01_tree_heights.las"
    las = laspy.read(path)
    las.points = las.points[:-1]
    las.write(str(path))
    with pytest.raises(StageError, match="not from the same run"):
        split.run_split(ctx)


def test_missing_dfin_results(ctx, project):
    with pytest.raises(StageError, match="run the dfin stage first"):
        split.run_split(ctx)
    ctx.stage_dir("dfin").mkdir(parents=True)
    with pytest.raises(StageError, match="contains no"):
        split.run_split(ctx)


def test_a_tree_without_points_after_the_cut_is_an_error(ctx):
    build(ctx)
    with pytest.raises(StageError, match="no points after the cut"):
        split.run_split(ctx, max_axis_dist=1e-6)
    assert not (ctx.stage_dir("split") / "trees").exists()  # no half-written result is left behind
    assert not (ctx.stage_dir("split") / "tree_map.csv").exists()


def test_invalid_parameters(ctx):
    with pytest.raises(StageError, match="max_axis_dist"):
        split.run_split(ctx, max_axis_dist=0)
    with pytest.raises(StageError, match="min_points"):
        split.run_split(ctx, min_points=-1)


def test_stem_is_taken_from_the_input_or_discovered(ctx, project):
    run = build(ctx)
    split.run_split(replace(ctx, input=None))  # a single result is found by itself
    for suffix in ("_tree_ID_dist_axes.las", "_stripe.las", "_tree_heights.las", "_tree_locator.las", ".xlsx", "_config.ini"):
        shutil.copy(run.dfin_dir / f"plot_01{suffix}", run.dfin_dir / f"other{suffix}")
    with pytest.raises(StageError, match="several 3DFin results"):
        split.run_split(replace(ctx, input=None))
    split.run_split(ctx)  # ctx.input is data/in/plot_01.laz -> stem plot_01


def test_dry_run_touches_nothing(ctx):
    result = split.run_split(replace(ctx, dry_run=True))
    assert result["dry_run"] and not ctx.stage_dir("split").exists()


# -- (d) the cut ------------------------------------------------------------------------------------------------
def test_cut_filters_by_axis_distance_and_height_and_keeps_every_dimension(ctx):
    run = build(ctx, dropped=(42,))
    split.run_split(ctx, max_axis_dist=3.0, min_z0=0.10)
    src = laspy.open(run.dfin_dir / "plot_01_tree_ID_dist_axes.las")
    for ordinal, raw in enumerate(run.row_ids, start=1):
        out = laspy.read(ctx.stage_dir("split") / "trees" / f"tree_{ordinal:04d}.laz")
        want = expected_mask(run, raw)
        c = run.cloud
        assert len(out.points) == want.sum() > 500
        # same points in the same order; extra dims bit-identical; coordinates identical within the quantisation
        np.testing.assert_array_equal(np.asarray(out["tree_ID"]), c["tid"][want])
        np.testing.assert_array_equal(np.asarray(out["Class"]), c["klass"][want])
        np.testing.assert_array_equal(np.asarray(out["Split"]), c["split"][want])
        np.testing.assert_array_equal(np.asarray(out["dist_axes"]), c["dist"][want])
        np.testing.assert_array_equal(np.asarray(out["Z0"]), c["z0"][want])
        np.testing.assert_allclose(np.asarray(out.x), c["x"][want], atol=1e-3)
        np.testing.assert_allclose(np.asarray(out.z), c["z"][want], atol=1e-3)
        # the filters really bite
        assert np.asarray(out["dist_axes"]).max() <= 3.0 and np.asarray(out["Z0"]).min() >= 0.10
        # format, scale and offset are the source's
        assert out.header.point_format.id == src.header.point_format.id == 7
        assert out.header.version == src.header.version
        np.testing.assert_array_equal(out.header.scales, src.header.scales)
        np.testing.assert_array_equal(out.header.offsets, src.header.offsets)
        assert [e.name for e in out.point_format.extra_dimensions] == ["Class", "Split", "dist_axes", "tree_ID", "Z0"]
        assert [str(e.dtype) for e in out.point_format.extra_dimensions] == ["int8", "int8", "float64", "int32", "float64"]
    src.close()


def test_cut_parameters_change_the_result(ctx):
    run = build(ctx)
    split.run_split(ctx, max_axis_dist=2.0, min_z0=0.5)
    tm = read_map(ctx)
    for _, row in tm.iterrows():
        want = expected_mask(run, int(row["raw_id"]), max_axis_dist=2.0, min_z0=0.5)
        assert row["n_points"] == want.sum()
        assert row["base_z0_m"] == pytest.approx(run.cloud["z"][want].min() - run.stem_ground(int(row["raw_id"])), abs=1e-5)
        assert 0.5 <= row["base_z0_m"] < 0.6  # the lowest kept point is just above the cut


def test_base_z0_is_the_height_of_the_lowest_kept_point_above_the_stem_ground(ctx):
    run = build(ctx)
    split.run_split(ctx)
    tm = read_map(ctx)
    for _, row in tm.iterrows():
        raw = int(row["raw_id"])
        want = expected_mask(run, raw)
        assert row["n_points"] == want.sum()
        assert row["stem_ground_z_m"] == pytest.approx(run.stem_ground(raw), abs=1e-5)
        assert row["base_z0_m"] == pytest.approx(run.cloud["z"][want].min() - run.stem_ground(raw), abs=1e-5)
        assert 0.10 - 1e-5 <= row["base_z0_m"] < 0.36  # never below min_z0; the first section window starts at 0.3
        out = laspy.read(ctx.stage_dir("split") / "trees" / f"tree_{int(row['ordinal']):04d}.laz")
        assert np.asarray(out.z).min() - row["stem_ground_z_m"] == pytest.approx(row["base_z0_m"], abs=1e-5)


def test_on_a_slope_the_lowest_point_is_at_the_stem_base_not_downhill(ctx):
    """The reviewed defect: with a Z0-only cut the cloud's lowest point (what the QSM tools call the ground) is downhill."""
    run = build(ctx, ids=synth.DEFAULT_IDS, slope_x=0.3, slope_y=0.1, brush=True)
    split.run_split(ctx)
    tm = read_map(ctx)
    for ordinal, raw in enumerate(run.row_ids, start=1):
        row = tm.iloc[ordinal - 1]
        sg = run.stem_ground(raw)
        tree = next(t for t in run.trees if t.raw_id == raw)
        assert sg == pytest.approx(float(run.terrain(tree.cx, tree.cy)), abs=0.05)  # = the terrain at the stem
        # the old cut: lowest point far below the ground at the stem (downhill brush), on average > 0.5 m
        old = old_mask(run, raw)
        assert run.cloud["z"][old].min() - sg < 0.10 - 0.3, "the fixture must reproduce the defect"
        # the new cut: the lowest point is ~min_z0 above the stem ground, and it is the real minimum of the file
        out = laspy.read(ctx.stage_dir("split") / "trees" / f"tree_{ordinal:04d}.laz")
        lowest = np.asarray(out.z).min()
        assert 0.10 - 1e-5 <= lowest - sg < 0.10 + 0.12, (raw, lowest - sg)
        assert row["base_z0_m"] == pytest.approx(lowest - sg, abs=1e-5) and 0.10 - 1e-5 <= row["base_z0_m"] < 0.22
        # brush points downhill (Z0 >= 0.1 but below the stem plane) are gone, those uphill remain
        assert row["n_points"] == expected_mask(run, raw).sum() < old.sum()
        assert (np.asarray(out["Z0"]) >= 0.10).all() and (np.asarray(out.z) >= sg + 0.10 - 1e-5).all()


def test_stem_ground_comes_from_the_axis_points_and_is_reported(ctx):
    run = build(ctx, slope_x=0.3, slope_y=0.1)
    split.run_split(ctx)
    tm = read_map(ctx)
    assert set(tm["stem_ground_source"]) == {"axis"}
    for _, row in tm.iterrows():
        raw = int(row["raw_id"])
        c = run.cloud
        near = (c["tid"] == raw) & (c["dist"] < split.STEM_GROUND_AXIS_DIST)
        assert near.sum() >= split.STEM_GROUND_MIN_POINTS
        assert row["stem_ground_z_m"] == pytest.approx(np.mean(c["z"][near] - c["z0"][near]), abs=1e-5)
        # NOT the mean over all of the tree's points (crown points are on every side of the stem, brush/far ones as well, but
        # the rule is "near the axis")
        assert row["stem_ground_z_m"] == pytest.approx(run.stem_ground(raw), abs=1e-5)


def test_few_axis_points_fall_back_to_all_points_of_the_tree_and_are_flagged(ctx, caplog):
    run = build(ctx, slope_x=0.3, slope_y=0.1)
    path = run.dfin_dir / "plot_01_tree_ID_dist_axes.las"
    las = laspy.read(path)
    tid, dist = np.asarray(las["tree_ID"]), np.asarray(las["dist_axes"]).copy()
    few, enough, none = 7, 11, 42  # tree ids: 19 axis points -> fallback, 20 -> axis, 0 -> fallback
    for raw, keep in ((few, split.STEM_GROUND_MIN_POINTS - 1), (enough, split.STEM_GROUND_MIN_POINTS), (none, 0)):
        near = np.flatnonzero((tid == raw) & (dist < split.STEM_GROUND_AXIS_DIST))
        assert near.size > keep
        dist[near[keep:]] += 0.5  # still within max_axis_dist, no longer "at the axis"
    las["dist_axes"] = dist
    las.write(str(path))
    c = laspy.read(path)
    z, z0, dist, tid = np.asarray(c.z), np.asarray(c["Z0"]), np.asarray(c["dist_axes"]), np.asarray(c["tree_ID"])

    result = split.run_split(ctx)
    tm = read_map(ctx).set_index("raw_id")
    assert tm["stem_ground_source"].to_dict() == {i: ("axis" if i not in (few, none) else "tree") for i in tm.index}
    assert result["n_stem_ground_fallback"] == 2
    for raw in (few, none):
        assert tm.loc[raw, "stem_ground_z_m"] == pytest.approx(np.mean((z - z0)[tid == raw]), abs=1e-5)
    near = (tid == enough) & (dist < split.STEM_GROUND_AXIS_DIST)
    assert near.sum() == split.STEM_GROUND_MIN_POINTS
    assert tm.loc[enough, "stem_ground_z_m"] == pytest.approx(np.mean((z - z0)[near]), abs=1e-5)
    # the cut uses the fallback ground too
    for raw in (few, none):
        want = (tid == raw) & (dist <= 3.0) & (z0 >= 0.10) & (z >= tm.loc[raw, "stem_ground_z_m"] + 0.10)
        assert tm.loc[raw, "n_points"] == want.sum()
    assert "ordinals" in caplog.text and "estimated from all points of the tree" in caplog.text


@pytest.mark.parametrize("slope", [0.02, 0.3], ids=["gentle", "steep"])
def test_ground_variant_adds_exactly_the_skirt(ctx, slope):
    run = build(ctx, dropped=(42,), slope_x=slope, slope_y=slope / 3, brush=True)
    split.run_split(ctx, emit_ground_variant=True)
    tm = read_map(ctx)
    c = run.cloud
    assert "n_points_ground" in tm.columns
    for ordinal, raw in enumerate(run.row_ids, start=1):
        sg = run.stem_ground(raw)
        near = (c["tid"] == raw) & (c["dist"] <= 3.0)
        main = expected_mask(run, raw)
        skirt = near & (c["z0"] >= -0.5) & (c["z"] >= sg - 0.5)  # the same plane idea, 0.5 m lower
        out = laspy.read(ctx.stage_dir("split") / "trees_ground" / f"tree_{ordinal:04d}.laz")
        assert len(out.points) == skirt.sum() > main.sum()
        assert (main & ~skirt).sum() == 0  # the ground variant contains the main cloud
        z0 = np.asarray(out["Z0"])
        assert z0.min() >= -0.5 and np.asarray(out.z).min() >= sg - 0.5 - 1e-5
        assert (np.asarray(out.z) < sg + 0.10).sum() > 0  # points below the stem plane (the skirt) are there ...
        np.testing.assert_array_equal(z0, c["z0"][skirt])
        assert tm["n_points_ground"].iloc[ordinal - 1] == skirt.sum() and tm["n_points"].iloc[ordinal - 1] == main.sum()
        main_file = laspy.read(ctx.stage_dir("split") / "trees" / f"tree_{ordinal:04d}.laz")
        assert (np.asarray(main_file.z) >= sg + 0.10 - 1e-5).all()  # ... and not in the main cloud
    assert not (ctx.stage_dir("split") / "trees_ground" / "tree_0008.laz").exists()  # no 8th tree


def test_without_ground_variant_there_is_no_ground_dir(ctx):
    build(ctx)
    split.run_split(ctx)
    assert not (ctx.stage_dir("split") / "trees_ground").exists()
    assert "n_points_ground" not in read_map(ctx).columns


def test_low_point_trees_are_listed_but_not_written(ctx):
    run = build(ctx, sparse_ids=(13,))
    result = split.run_split(ctx, min_points=500)
    tm = read_map(ctx)
    sparse = tm[tm["raw_id"] == 13].iloc[0]
    assert sparse["low_points"] and sparse["n_points"] < 500 and pd.isna(sparse["file"])  # no file
    assert tm["low_points"].sum() == 1 and result["n_low_points"] == 1 and result["n_written"] == 7
    assert len(tree_files(ctx)) == 7 and f"tree_{int(sparse['ordinal']):04d}.laz" not in tree_files(ctx)
    assert sparse["n_points"] == expected_mask(run, 13).sum()  # still counted, ordinal and row kept


def test_stale_outputs_of_a_previous_run_are_cleared(ctx):
    build(ctx)
    stale = ctx.stage_dir("split") / "trees" / "tree_0099.laz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    split.run_split(ctx)
    assert not stale.exists() and len(tree_files(ctx)) == 8


def test_tree_map_columns(ctx):
    build(ctx, dropped=(42,))
    split.run_split(ctx)
    tm = read_map(ctx)
    assert list(tm.columns) == split.TREE_MAP_COLUMNS and {"base_z0_m", "stem_ground_z_m", "stem_ground_source"} <= set(tm.columns)
    assert tm["dbh_m"].isna().tolist() == tm["dbh_missing"].tolist()
    assert (tm["file"] == [f"trees/tree_{i:04d}.laz" for i in range(1, 8)]).all()
    assert tm.dtypes["th_mismatch"] == bool and tm.dtypes["low_points"] == bool


# -- (e) 3DFin's noise label -1 (100000 is an ordinary id) ---------------------------------------------------------------
def test_points_labelled_minus_one_or_100000_outside_the_stripe_never_become_trees(ctx):
    run = build(ctx, nontree_ids=True)
    assert {-1, 100000} <= set(np.unique(run.cloud["tid"]).tolist())  # in the cloud, but the stripe does not list them
    split.run_split(ctx)
    assert len(tree_files(ctx)) == 8
    for f in tree_files(ctx):
        ids = np.unique(np.asarray(laspy.read(ctx.stage_dir("split") / "trees" / f)["tree_ID"]))
        assert ids.size == 1 and ids[0] not in (-1, 100000)
    tm = read_map(ctx)
    assert -1 not in tm["raw_id"].tolist() and 100000 not in tm["raw_id"].tolist()


def test_a_tree_with_id_minus_one_gets_no_cloud_but_100000_is_an_ordinary_id(ctx, caplog):
    build(ctx, ids=(-1, 5, 7000, 100000), nontree_ids=False)
    result = split.run_split(ctx)
    tm = read_map(ctx)
    assert tm["raw_id"].tolist() == [-1, 5, 7000, 100000]  # the rows keep their ordinals ...
    assert tm["n_points"].iloc[0] == 0 and (tm["n_points"].iloc[1:] > 500).all() and tm["low_points"].tolist() == [True, False, False, False]
    assert tree_files(ctx) == ["tree_0002.laz", "tree_0003.laz", "tree_0004.laz"]  # ... -1 has no cloud, 100000 has
    ex = pd.read_csv(ctx.stage_dir("split") / "excluded_ids.csv")
    assert ex["raw_id"].tolist() == [-1] and set(ex["reason"]) == {"non_tree_id"}
    assert "no tree' labels" in caplog.text and result["n_trees"] == 4 and result["n_written"] == 3


def test_a_real_tree_with_id_100000_keeps_its_cloud(ctx, caplog):
    run = build(ctx, ids=(5, 7000, 100000), nontree_ids=False, slope_x=0.2)
    result = split.run_split(ctx)
    tm = read_map(ctx)
    assert tm["raw_id"].tolist() == [5, 7000, 100000] and set(tm["id_confidence"]) == {"exact"}
    assert tree_files(ctx) == ["tree_0001.laz", "tree_0002.laz", "tree_0003.laz"]
    want = expected_mask(run, 100000)
    assert tm["n_points"].iloc[2] == want.sum() > 500 and not tm["low_points"].any()
    out = laspy.read(ctx.stage_dir("split") / "trees" / "tree_0003.laz")
    assert np.unique(np.asarray(out["tree_ID"])).tolist() == [100000]
    assert pd.read_csv(ctx.stage_dir("split") / "excluded_ids.csv").empty
    assert "no tree' labels" not in caplog.text and result["n_written"] == 3


# -- deterministic output (the SmartQSM resume compares the tree files byte for byte) -------------------------------
def fake_today(day: datetime.date):
    """A ``date`` class for ``laspy.header`` whose ``today()`` is ``day`` (laspy stamps it into new headers)."""

    class FakeDate(datetime.date):
        @classmethod
        def today(cls):
            return day

    return FakeDate


def tree_bytes(ctx: RunContext) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((ctx.stage_dir("split") / "trees").glob("*.laz"))}


def test_tree_files_do_not_depend_on_the_day_they_are_written(ctx, monkeypatch):
    run = build(ctx, dropped=(42,), slope_x=0.2)
    written = []
    for today in (datetime.date(2030, 1, 2), datetime.date(2031, 7, 9)):
        monkeypatch.setattr(laspy.header, "date", fake_today(today))
        assert laspy.LasHeader().creation_date == today  # the patch works: a fresh header would carry that day
        split.run_split(ctx)
        written.append(tree_bytes(ctx))
    assert len(written[0]) == 7 and written[0] == written[1]  # byte-identical, e.g. at header byte 90 (creation day)
    monkeypatch.undo()
    first = next(iter(written[0]))
    header = (ctx.stage_dir("split") / "trees" / first).read_bytes()[:94]
    day_of_year, year = int.from_bytes(header[90:92], "little"), int.from_bytes(header[92:94], "little")
    assert datetime.date(year, 1, 1) + datetime.timedelta(days=day_of_year - 1) == synth.CREATION_DATE  # the source's date


def test_the_header_is_copied_from_the_source(ctx):
    run = build(ctx)
    src_path = run.dfin_dir / "plot_01_tree_ID_dist_axes.las"
    las = laspy.read(src_path)  # give the source distinctive identification fields
    las.header.file_source_id = 1234
    las.header.system_identifier = "SYSX"
    las.write(str(src_path))
    src = laspy.read(src_path).header
    split.run_split(ctx)
    for name in tree_files(ctx):
        h = laspy.read(ctx.stage_dir("split") / "trees" / name).header
        assert h.creation_date == synth.CREATION_DATE == src.creation_date
        assert h.file_source_id == 1234 and h.system_identifier == "SYSX" and h.generating_software == "lidar_bridge split"
        assert h.version == src.version and h.point_format.id == src.point_format.id


def test_a_source_without_creation_date_gets_a_fixed_one(ctx, monkeypatch):
    run = build(ctx)
    path = run.dfin_dir / "plot_01_tree_ID_dist_axes.las"
    raw = bytearray(path.read_bytes())
    raw[90:94] = bytes(4)  # day of year 0, year 0 = "no date"
    path.write_bytes(bytes(raw))
    assert laspy.read(path).header.creation_date is None
    written = []
    for today in (datetime.date(2030, 1, 2), datetime.date(2031, 7, 9)):
        monkeypatch.setattr(laspy.header, "date", fake_today(today))
        split.run_split(ctx)
        written.append(tree_bytes(ctx))
    assert written[0] == written[1]
    monkeypatch.undo()
    assert laspy.read(ctx.stage_dir("split") / "trees" / "tree_0001.laz").header.creation_date == split.FALLBACK_CREATION_DATE


# -- tables exported as text ---------------------------------------------------------------------------------------
def test_export_txt_tables_work_too(ctx):
    run = build(ctx, dropped=(42,), export_txt=True)
    split.run_split(ctx)
    tm = read_map(ctx)
    assert tm["raw_id"].tolist() == run.row_ids
    assert tm["th_mismatch"].tolist() == [False] * 5 + [True] * 2  # the 3-decimal rounding of the txt is tolerated


def test_a_shuffled_heights_file_is_noticed(ctx, caplog):
    run = build(ctx, ids=synth.DEFAULT_IDS + (11, 12, 13, 14, 15, 16))
    path = run.dfin_dir / "plot_01_tree_heights.las"
    las = laspy.read(path)
    las.points = las.points[::-1].copy()  # tops no longer follow the ascending id order
    las.write(str(path))
    split.run_split(ctx)
    assert "does not look ordered like the stripe ids" in caplog.text

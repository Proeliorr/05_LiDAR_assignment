"""Tree-id alignment (idmap): table parsing, the dynamic programme, and the end-to-end alignment on fake 3DFin output."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import synth3dfin as synth

from lidar_bridge import idmap
from lidar_bridge.stages._common import StageError

IDS = synth.DEFAULT_IDS + (11, 12, 13)  # 0, 7, 11, 12, 13, 42, 9001, 630740 (8 trees)


def build(tmp_path: Path, **kw) -> synth.SynthRun:
    kw.setdefault("ids", IDS)
    return synth.build_dfin(tmp_path / "dfin", **kw)


def align(run: synth.SynthRun, **kw) -> idmap.Alignment:
    tables = idmap.read_tables(run.dfin_dir, run.stem)
    unique_ids = idmap.read_stripe_ids(run.dfin_dir / f"{run.stem}_stripe.las")
    return idmap.align_rows_to_ids(unique_ids, tables, run.dfin_dir / f"{run.stem}_tree_ID_dist_axes.las", **kw)


# -- reading the tables ---------------------------------------------------------------------------------------
def test_xlsx_tables_are_found_by_content_not_position(tmp_path):
    run = build(tmp_path, dropped=(42,))
    sheets = pd.read_excel(run.dfin_dir / "plot_01.xlsx", sheet_name=None, header=None)
    assert sheets["Plot Metrics"].iloc[0, 0].startswith("Total height")  # the header row is NOT the first row
    t = idmap.read_tables(run.dfin_dir, "plot_01")
    assert t.source == "xlsx" and t.n_rows == 7 and t.n_sections == run.sections.size
    np.testing.assert_allclose(t.th, run.th_table)
    np.testing.assert_allclose(t.dbh, run.dbh)
    np.testing.assert_allclose(t.x, run.xy_table[:, 0])
    np.testing.assert_allclose(t.y, run.xy_table[:, 1])
    np.testing.assert_allclose(t.sections, run.sections)
    np.testing.assert_allclose(t.radius, run.radius)
    np.testing.assert_allclose(t.x_c, run.x_c)
    np.testing.assert_allclose(t.y_c, run.y_c)


def test_txt_tables_export_txt(tmp_path):
    run = build(tmp_path, dropped=(42,), export_txt=True)
    assert not (run.dfin_dir / "plot_01.xlsx").exists()
    t = idmap.read_tables(run.dfin_dir, "plot_01")
    assert t.source == "txt" and t.n_rows == 7
    np.testing.assert_allclose(t.th, run.th_table, atol=1e-3)
    np.testing.assert_allclose(t.radius, run.radius, atol=1e-3)
    np.testing.assert_allclose(t.sections, run.sections, atol=1e-3)


def test_missing_table_is_a_stage_error(tmp_path):
    with pytest.raises(StageError, match="no 3DFin result table"):
        idmap.read_tables(tmp_path, "plot_01")


def test_workbook_without_header_row_is_rejected(tmp_path):
    path = tmp_path / "plot_01.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["nothing", "useful"]]).to_excel(writer, sheet_name="Plot Metrics", header=False, index=False)
    with pytest.raises(StageError, match="TH / DBH / X / Y"):
        idmap.read_tables(tmp_path, "plot_01")


def test_row_labels_must_run_t1_to_tn(tmp_path):
    path = tmp_path / "plot_01.xlsx"
    frame = pd.DataFrame([[1.0, 0, 1.0, 1.0], [2.0, 0, 2.0, 2.0]], index=["T1", "T3"], columns=["TH", "DBH", "X", "Y"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.Series("info").to_excel(writer, sheet_name="Plot Metrics", header=False, index=False)
        frame.to_excel(writer, sheet_name="Plot Metrics", startrow=2, startcol=1)
    with pytest.raises(StageError, match="not T1..Tn"):
        idmap.read_tables(tmp_path, "plot_01")


def test_stripe_ids_are_sorted_unique_and_keep_zero(tmp_path):
    run = build(tmp_path)
    ids = idmap.read_stripe_ids(run.dfin_dir / "plot_01_stripe.las")
    assert ids.tolist() == sorted(IDS) and 0 in ids.tolist()


def test_config_ini_is_parsed_and_defaults_apply(tmp_path):
    run = build(tmp_path)
    cfg = idmap.read_dfin_config(run.dfin_dir, "plot_01")
    assert cfg.found and cfg.z0_name == "Z0" and cfg.section_wid == synth.SECTION_WID
    assert not idmap.read_dfin_config(tmp_path / "nowhere", "x").found


# -- the dynamic programme ------------------------------------------------------------------------------------
def test_monotone_path_hand_computed():
    score = np.array([[5.0, 0.0], [4.0, 0.0], [0.0, 6.0], [0.0, 3.0]])
    path = idmap.best_monotone_path(score)
    assert path.offsets.tolist() == [0, 0, 1, 1] and path.total == 18.0
    # best alignment that maps the row to the other id: row 0 -> 0,1,1,1 (9); row 1 -> 5+0+6+3 (14); row 2 -> 5+4+0+3 (12);
    # row 3 -> all offset 0 (9)
    assert path.row_margin.tolist() == [9.0, 4.0, 6.0, 9.0]
    # a monotone path never uses offset 0 after offset 1
    assert np.all(np.diff(path.offsets) >= 0)


def test_monotone_path_without_evidence_is_a_tie_with_zero_margins():
    path = idmap.best_monotone_path(np.zeros((5, 3)))
    assert path.offsets.tolist() == [0] * 5 and path.total == 0.0
    assert np.all(path.row_margin == 0.0)


def test_monotone_path_single_offset_has_infinite_margin():
    path = idmap.best_monotone_path(np.array([[2.0], [3.0]]))
    assert path.offsets.tolist() == [0, 0] and np.all(np.isinf(path.row_margin))


def test_dropped_margin_is_the_cost_of_keeping_the_id():
    score = np.array([[5.0, 0.0], [4.0, 0.0], [0.0, 6.0], [0.0, 3.0]])
    path = idmap.best_monotone_path(score)
    # ids 0..4, rows -> 0,1,3,4: id 2 is dropped. Keeping it means row 1->2 (loses 4) or row 2->2 (loses 6, offset 0)
    assert idmap.dropped_margins(path, [2]) == [4.0]


def test_monotone_path_brute_force_agrees():
    rng = np.random.default_rng(3)
    for _ in range(20):
        score = rng.random((5, 3))
        path = idmap.best_monotone_path(score)
        best = max(sum(score[k, o] for k, o in enumerate(offs)) for offs in itertools.combinations_with_replacement(range(3), 5))
        assert path.total == pytest.approx(best)


# -- end to end on fake 3DFin output --------------------------------------------------------------------------
def test_no_dropped_id_is_exact_and_ordinal_is_rank(tmp_path):
    run = build(tmp_path)
    tables = idmap.read_tables(run.dfin_dir, "plot_01")
    unique = idmap.read_stripe_ids(run.dfin_dir / "plot_01_stripe.las")
    result = idmap.align_rows_to_ids(unique, tables)  # no cloud needed
    assert result.rows["ordinal"].tolist() == list(range(1, 9))
    assert result.rows["raw_id"].tolist() == sorted(IDS)
    assert result.rows["raw_id"].iloc[0] == 0  # id 0 is a real tree
    assert set(result.rows["id_confidence"]) == {"exact"}
    assert result.n_dropped == 0 and result.dropped_ids == [] and result.n_ambiguous == 0


@pytest.mark.parametrize("dropped", [(0,), (7,), (42,), (9001,), (630740,), (7, 42), (0, 13, 630740)], ids=str)
def test_dropped_ids_are_found_wherever_they_are(tmp_path, dropped):
    run = build(tmp_path, dropped=dropped)
    result = align(run)
    assert result.dropped_ids == sorted(dropped)
    assert result.rows["raw_id"].tolist() == run.row_ids
    assert result.rows["ordinal"].tolist() == list(range(1, len(run.rows) + 1))
    assert set(result.rows["id_confidence"]) == {"resolved"}
    assert (result.dropped["reason"] == idmap.REASON_DROPPED).all()
    assert (result.rows["margin"] >= 2.0).all() and (result.rows["n_supported"] > 0).all()


def test_only_minus_one_is_a_non_tree_id_and_100000_is_an_ordinary_id(tmp_path):
    assert idmap.NON_TREE_IDS == (-1,)
    ids = synth.DEFAULT_IDS + (100000,)  # 0, 7, 42, 9001, 100000, 630740
    for case, dropped in enumerate(((), (100000,), (42,))):
        run = build(tmp_path / f"case{case}", ids=ids, dropped=dropped, nontree_ids=False)
        result = align(run)
        assert result.rows["raw_id"].tolist() == run.row_ids
        assert (100000 in result.rows["raw_id"].tolist()) == (100000 not in dropped)
        assert result.dropped_ids == sorted(dropped)
        assert set(result.rows["id_confidence"]) <= {"exact", "resolved"}


def test_rows_without_circles_next_to_the_drop_are_ambiguous(tmp_path):
    # rows 7 and 9001 have no circle: the drop may be 7, 42 or 9001; the rows that do have circles stay resolved
    run = build(tmp_path, ids=synth.DEFAULT_IDS, dropped=(42,), no_circle_ids=(7, 9001))
    result = align(run)
    conf = dict(zip(result.rows["ordinal"], result.rows["id_confidence"]))
    assert conf == {1: "resolved", 2: "ambiguous", 3: "ambiguous", 4: "resolved"}
    assert result.rows["raw_id"].iloc[[0, 3]].tolist() == [0, 630740]
    assert result.n_ambiguous == 2 and len(result.dropped) == 1
    assert (result.dropped["margin"] < 2.0).all()


def test_no_circle_at_all_means_everything_is_ambiguous(tmp_path):
    run = build(tmp_path, ids=synth.DEFAULT_IDS, dropped=(42,), no_circle_ids=synth.DEFAULT_IDS)
    result = align(run)
    assert set(result.rows["id_confidence"]) == {"ambiguous"}
    assert any("guess" in note for note in result.notes)


def test_more_table_rows_than_detected_trees_is_an_error(tmp_path):
    run = build(tmp_path)  # 8 rows
    tables = idmap.read_tables(run.dfin_dir, "plot_01")
    unique = idmap.read_stripe_ids(run.dfin_dir / "plot_01_stripe.las")[:-1]
    with pytest.raises(StageError, match="8 tree rows but only 7"):
        idmap.align_rows_to_ids(unique, tables)


def test_dropped_ids_need_the_cloud(tmp_path):
    run = build(tmp_path, dropped=(42,))
    tables = idmap.read_tables(run.dfin_dir, "plot_01")
    unique = idmap.read_stripe_ids(run.dfin_dir / "plot_01_stripe.las")
    with pytest.raises(StageError, match="point cloud is needed"):
        idmap.align_rows_to_ids(unique, tables)


def test_cloud_without_the_expected_dimensions_is_rejected(tmp_path):
    run = build(tmp_path, dropped=(42,))
    tables = idmap.read_tables(run.dfin_dir, "plot_01")
    unique = idmap.read_stripe_ids(run.dfin_dir / "plot_01_stripe.las")
    with pytest.raises(StageError, match="dist_axes"):
        idmap.align_rows_to_ids(unique, tables, run.dfin_dir / "plot_01_stripe.las")


def test_unsorted_unique_ids_are_a_programming_error(tmp_path):
    run = build(tmp_path)
    tables = idmap.read_tables(run.dfin_dir, "plot_01")
    with pytest.raises(ValueError):
        idmap.align_rows_to_ids(np.array([5, 3, 9, 1, 2, 6, 7, 8]), tables)

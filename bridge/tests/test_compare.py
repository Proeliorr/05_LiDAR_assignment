from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidar_bridge.paths import RunContext
from lidar_bridge.stages import compare
from lidar_bridge.stages._common import StageError, tree_map_fingerprint, write_fingerprint

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# -- builders -------------------------------------------------------------------------------------------------
def write_reference_csv(
    path: Path, ids, dbh_cm, height_m, *, sep=";", decimal=",", encoding="utf-8-sig", labels=None
) -> Path:
    """Reference table like a Polish Excel export: 'Nr drzewa', 'DBH [cm]', 'Wysokość [m]'."""

    def fmt(x):
        return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x}".replace(".", decimal)

    lines = [sep.join(["Nr drzewa", "DBH [cm]", "Wysokość [m]"])]
    for i, d, h in zip(labels or ids, dbh_cm, height_m):
        lines.append(sep.join([str(i), fmt(d), fmt(h)]))
    path.write_bytes(("\n".join(lines) + "\n").encode(encoding))
    return path


def write_tree_map(ctx: RunContext, ordinals, dbh_m, th_m, id_confidence="exact") -> Path:
    """split/tree_map.csv; ``id_confidence``: one value for all trees, a list, or ``None`` for no such column."""
    path = ctx.stage_dir("split") / "tree_map.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"ordinal": ordinals, "dbh_m": dbh_m, "th_m": th_m, "n_points": 1000, "x": 0.0})
    if id_confidence is not None:
        frame.insert(1, "id_confidence", id_confidence)
    frame.to_csv(path, index=False)
    return path


def write_metrics(ctx: RunContext, rel_dir: str, ordinals, dbh_m, height_m, status="ok", note=None) -> Path:
    path = ctx.stage_dir(rel_dir.split("/")[0]) / "/".join(rel_dir.split("/")[1:]) / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"ordinal": ordinals, "dbh_m": dbh_m, "height_m": height_m, "status": status, "seconds": 1.5, "volume_m3": 0.2}
    )
    if note is not None:
        frame["note"] = note
    frame.to_csv(path, index=False)
    return path


def stamp(ctx: RunContext, rel_dir: str, fingerprint: str | None = None) -> None:
    """source_fingerprint.txt of a stage directory: by default that of the CURRENT split/tree_map.csv."""
    stage_dir = ctx.stage_dir(rel_dir.split("/")[0]) / "/".join(rel_dir.split("/")[1:])
    write_fingerprint(stage_dir, fingerprint or tree_map_fingerprint(ctx.stage_dir("split")), extra="homogenize=false")


@pytest.fixture
def ref_ctx(ctx: RunContext, project: Path) -> RunContext:
    """The shared ctx with a 10-tree reference: DBH 20..47 cm (step 3), height 15..24 m (step 1)."""
    reference = write_reference_csv(
        project / "data" / "in" / "ref.csv", range(1, 11), [20 + 3 * i for i in range(10)], [15 + i for i in range(10)]
    )
    return replace(ctx, reference=reference)


REF_DBH_M = np.array([20 + 3 * i for i in range(10)]) / 100.0
REF_H_M = np.array([15.0 + i for i in range(10)])
ORD = list(range(1, 11))


def read_csv(ctx: RunContext, name: str) -> pd.DataFrame:
    return pd.read_csv(ctx.stage_dir("compare") / name)


def row(df: pd.DataFrame, **match):
    mask = np.ones(len(df), dtype=bool)
    for k, v in match.items():
        mask &= (df[k] == v).to_numpy()
    assert mask.sum() == 1, f"expected exactly one row for {match}, got {mask.sum()}"
    return df[mask].iloc[0]


# -- error_stats ----------------------------------------------------------------------------------------------
def test_error_stats_hand_computed():
    ref = np.array([10.0, 20.0, 30.0, 40.0])
    est = ref + np.array([1.0, -1.0, 3.0, 1.0])
    s = compare.error_stats(ref, est, tol=1.0)
    assert s["bias"] == pytest.approx(1.0)
    assert s["MAE"] == pytest.approx(1.5)
    assert s["RMSE"] == pytest.approx(math.sqrt(3.0))
    assert s["rel_bias_pct"] == pytest.approx(100 * 1.0 / 25.0)
    assert s["rel_RMSE_pct"] == pytest.approx(100 * math.sqrt(3.0) / 25.0)
    assert s["R2"] == pytest.approx(1 - 12 / 500)  # SS_res 12, SS_tot 500
    assert s["slope"] == pytest.approx(520 / 500)
    assert s["intercept"] == pytest.approx(26 - 1.04 * 25)
    assert s["R2_ols"] == pytest.approx(520**2 / (500 * 548))
    assert s["within_tol_pct"] == pytest.approx(75.0)  # |e| <= 1 for three of four


def test_error_stats_matches_numpy_on_noisy_data():
    rng = np.random.default_rng(42)
    ref = rng.uniform(0.15, 0.6, 200)
    est = 0.9 * ref + 0.03 + rng.normal(0, 0.02, 200)
    s = compare.error_stats(ref, est, tol=0.02)
    slope, intercept = np.polyfit(ref, est, 1)
    assert s["slope"] == pytest.approx(slope) and s["intercept"] == pytest.approx(intercept)
    assert s["R2_ols"] == pytest.approx(np.corrcoef(ref, est)[0, 1] ** 2)
    assert s["bias"] == pytest.approx(np.mean(est - ref))
    assert s["RMSE"] == pytest.approx(np.sqrt(np.mean((est - ref) ** 2)))
    assert s["R2"] == pytest.approx(1 - np.sum((est - ref) ** 2) / np.sum((ref - ref.mean()) ** 2))
    assert s["R2"] <= s["R2_ols"] + 1e-12  # the 1:1 R2 can never beat the best-fit line's


def test_error_stats_guards():
    one = compare.error_stats(np.array([30.0]), np.array([31.0]), tol=2.0)
    assert one["bias"] == 1.0 and one["MAE"] == 1.0 and one["RMSE"] == 1.0 and one["within_tol_pct"] == 100.0
    assert all(math.isnan(one[k]) for k in ("R2", "slope", "intercept", "R2_ols"))
    const = compare.error_stats(np.array([5.0, 5.0, 5.0]), np.array([4.0, 5.0, 6.0]), tol=1.0)
    assert const["bias"] == 0.0 and const["MAE"] == pytest.approx(2 / 3)
    assert all(math.isnan(const[k]) for k in ("R2", "slope", "intercept", "R2_ols"))
    flat_est = compare.error_stats(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]), tol=1.0)
    assert flat_est["slope"] == 0.0 and math.isnan(flat_est["R2_ols"]) and flat_est["R2"] == pytest.approx(0.0)
    empty = compare.error_stats(np.array([]), np.array([]), tol=1.0)
    assert all(math.isnan(v) for v in empty.values())


def test_perfect_estimates():
    ref = np.array([0.2, 0.3, 0.45])
    s = compare.error_stats(ref, ref.copy(), tol=0.02)
    assert s["bias"] == 0 and s["RMSE"] == 0 and s["R2"] == 1.0 and s["slope"] == pytest.approx(1.0)
    assert s["intercept"] == pytest.approx(0.0, abs=1e-12)


# -- pairing --------------------------------------------------------------------------------------------------
def test_pair_metric_counts_missing_and_extra():
    ref = pd.Series([0.2, 0.3, 0.4, np.nan], index=pd.Index([1, 2, 3, 4], name="ordinal"))
    est = pd.Series([0.21, np.nan, 0.41, 0.5], index=pd.Index([1, 2, 3, 9], name="ordinal"))
    paired, n_missing, n_extra = compare.pair_metric(ref, est)
    assert paired["ordinal"].tolist() == [1, 3]
    assert n_missing == 1  # tree 2: reference but the estimate is NaN
    assert n_extra == 1  # tree 9: estimate without reference (tree 4's reference is NaN: not counted as missing)


# -- end to end -----------------------------------------------------------------------------------------------
def make_pipelines(ctx: RunContext) -> None:
    """3dfin: dbh +1 cm, height -0.5 m, covers all trees but 10 (no DBH for tree 4).
    smartqsm:layerwise-clustering: dbh +2 cm, height +1 m, trees 1..8 (+ extra tree 99).
    smartqsm:space-colonization: dbh -1 cm, height 0, tree 3 failed.
    arbor: dbh*1.1, height +0.5 with failed rows and invalid values."""
    d = REF_DBH_M
    h = REF_H_M
    o3 = ORD[:9]
    dbh3 = [d[i - 1] + 0.01 if i != 4 else None for i in o3]
    write_tree_map(ctx, o3, dbh3, [h[i - 1] - 0.5 for i in o3])
    o1 = ORD[:8] + [99]
    write_metrics(ctx, "smartqsm/layerwise-clustering", o1, [*(d[:8] + 0.02), 0.3], [*(h[:8] + 1.0), 20.0])
    status = ["ok"] * 10
    status[2] = "failed"
    write_metrics(ctx, "smartqsm/space-colonization", ORD, d - 0.01, h, status=status)
    dbh_a = list(d * 1.1)
    h_a = list(h + 0.5)
    status_a = ["ok"] * 10
    status_a[0] = "error"
    dbh_a[4], h_a[5], dbh_a[6], h_a[7] = float("nan"), 0.0, -0.1, float("nan")
    write_metrics(ctx, "arbor", ORD, dbh_a, h_a, status=status_a)


def test_end_to_end_numbers_naming_and_files(ref_ctx, caplog):
    make_pipelines(ref_ctx)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)

    out = ref_ctx.stage_dir("compare")
    assert sorted(p.name for p in out.iterdir()) == [
        "errors.csv", "reference_used.csv", "scatter_dbh.png", "scatter_height.png", "summary.csv",
    ]  # fmt: skip
    assert (out / "scatter_dbh.png").read_bytes().startswith(PNG_MAGIC)
    assert result["pipelines"] == ["3dfin", "smartqsm:layerwise-clustering", "smartqsm:space-colonization", "arbor"]
    assert result["plots"] == ["data/out/r1/compare/scatter_dbh.png", "data/out/r1/compare/scatter_height.png"]

    errors, summary = read_csv(ref_ctx, "errors.csv"), read_csv(ref_ctx, "summary.csv")
    assert list(errors.columns) == [
        "pipeline", "metric", "ordinal", "ref", "est", "error", "rel_error_pct", "id_confidence", "estimated",
    ]  # fmt: skip
    assert list(summary.columns[:16]) == [
        "pipeline", "metric", "n", "n_missing", "n_extra", "n_ambiguous", "n_estimated", "n_excluded", "bias", "MAE",
        "RMSE", "rel_bias_pct", "rel_RMSE_pct", "R2", "slope", "intercept",
    ]  # fmt: skip
    assert not errors.estimated.any()  # nothing flagged here
    assert (errors.id_confidence == np.where(errors.ordinal <= 9, "exact", "unknown")).all()  # tree 10 is not in the tree map
    assert (summary[["n_ambiguous", "n_estimated", "n_excluded"]] == 0).all().all()

    # 3DFin: DBH +1 cm for 8 trees (tree 4 has none, tree 10 is absent), height -0.5 for 9 trees
    s = row(summary, pipeline="3dfin", metric="dbh")
    assert (s.n, s.n_missing, s.n_extra) == (8, 2, 0)
    assert s.bias == pytest.approx(0.01) and s.MAE == pytest.approx(0.01) and s.RMSE == pytest.approx(0.01)
    paired_ref = np.delete(REF_DBH_M[:9], 3)
    assert s.rel_bias_pct == pytest.approx(100 * 0.01 / paired_ref.mean(), abs=1e-4)
    assert s.slope == pytest.approx(1.0) and s.intercept == pytest.approx(0.01) and s.R2_ols == pytest.approx(1.0)
    assert s.R2 == pytest.approx(1 - 8 * 0.01**2 / np.sum((paired_ref - paired_ref.mean()) ** 2), abs=1e-5)
    s = row(summary, pipeline="3dfin", metric="height")
    assert (s.n, s.n_missing, s.n_extra) == (9, 1, 0)
    assert s.bias == pytest.approx(-0.5) and s.RMSE == pytest.approx(0.5) and s.R2_ols == pytest.approx(1.0)

    # SmartQSM sweep: tree 99 is an extra, trees 9 and 10 are missing
    s = row(summary, pipeline="smartqsm:layerwise-clustering", metric="dbh")
    assert (s.n, s.n_missing, s.n_extra) == (8, 2, 1)
    assert s.bias == pytest.approx(0.02)
    s = row(summary, pipeline="smartqsm:layerwise-clustering", metric="height")
    assert s.bias == pytest.approx(1.0) and (s.n, s.n_missing, s.n_extra) == (8, 2, 1)
    # status != ok -> tree 3 missing for both metrics
    for metric, bias in (("dbh", -0.01), ("height", 0.0)):
        s = row(summary, pipeline="smartqsm:space-colonization", metric=metric)
        assert (s.n, s.n_missing, s.n_extra) == (9, 1, 0) and s.bias == pytest.approx(bias, abs=1e-9)
    assert 3 not in errors[errors.pipeline == "smartqsm:space-colonization"].ordinal.tolist()

    # arbor: tree 1 failed, dbh of 5 and 7 invalid (NaN, negative), height of 6 and 8 invalid (0, NaN)
    s = row(summary, pipeline="arbor", metric="dbh")
    assert (s.n, s.n_missing) == (7, 3)
    assert s.rel_bias_pct == pytest.approx(10.0, abs=1e-4)  # +10 % everywhere
    s = row(summary, pipeline="arbor", metric="height")
    assert (s.n, s.n_missing) == (7, 3) and s.bias == pytest.approx(0.5)
    arbor_dbh = errors[(errors.pipeline == "arbor") & (errors.metric == "dbh")]
    assert sorted(set(ORD) - set(arbor_dbh.ordinal)) == [1, 5, 7]

    # errors.csv: error = est - ref, rel_error_pct relative to ref, metres
    e = row(errors, pipeline="3dfin", metric="height", ordinal=2)
    assert (e.ref, e.est) == pytest.approx((16.0, 15.5)) and e.error == pytest.approx(-0.5)
    assert e.rel_error_pct == pytest.approx(100 * -0.5 / 16.0, abs=1e-4)
    e = row(errors, pipeline="smartqsm:layerwise-clustering", metric="dbh", ordinal=1)
    assert (e.ref, e.est) == pytest.approx((0.20, 0.22)) and e.error == pytest.approx(0.02)

    # reference_used.csv is the parsed reference in metres
    used = read_csv(ref_ctx, "reference_used.csv")
    assert used["dbh_m"].tolist() == pytest.approx(REF_DBH_M.tolist())

    # returned key numbers
    assert result["metrics"]["3dfin"]["dbh"]["bias"] == pytest.approx(0.01)
    assert result["metrics"]["arbor"]["height"]["n"] == 7
    assert result["n_reference"] == 10 and result["skipped"] == []

    # summary table logged at INFO
    text = caplog.text
    assert "summary (lengths in m" in text and "pipeline" in text and "RMSE%" in text
    assert "smartqsm:layerwise-clustering" in text


def test_summary_contains_tolerance_columns(ref_ctx):
    write_tree_map(ref_ctx, ORD, REF_DBH_M + np.where(np.arange(10) < 5, 0.01, 0.03), REF_H_M)
    compare.run_compare(ref_ctx, tolerance_dbh_cm=2.0, tolerance_height_m=0.5)
    summary = read_csv(ref_ctx, "summary.csv")
    s = row(summary, pipeline="3dfin", metric="dbh")
    assert s.tol_m == pytest.approx(0.02) and s.within_tol_pct == pytest.approx(50.0)
    s = row(summary, pipeline="3dfin", metric="height")
    assert s.tol_m == pytest.approx(0.5) and s.within_tol_pct == pytest.approx(100.0)


def test_only_one_pipeline_present_others_skipped_with_warning(ref_ctx, caplog):
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["arbor"] and len(result["skipped"]) == 2
    assert "pipeline '3dfin' skipped" in caplog.text and "pipeline 'smartqsm' skipped" in caplog.text
    summary = read_csv(ref_ctx, "summary.csv")
    assert set(summary.pipeline) == {"arbor"} and (summary.R2 == 1.0).all()


def test_single_smartqsm_config_keeps_prefixed_name(ref_ctx):
    write_metrics(ref_ctx, "smartqsm/space-colonization", ORD, REF_DBH_M, REF_H_M)
    result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["smartqsm:space-colonization"]


def test_no_pipeline_outputs_is_an_error(ref_ctx):
    with pytest.raises(StageError, match="nothing to compare.*pipeline '3dfin' skipped.*pipeline 'arbor' skipped"):
        compare.run_compare(ref_ctx)


def test_failed_run_keeps_previous_outputs(ref_ctx):
    out = ref_ctx.stage_dir("compare")
    out.mkdir(parents=True)
    (out / "summary.csv").write_text("previous")
    with pytest.raises(StageError):
        compare.run_compare(ref_ctx)
    assert (out / "summary.csv").read_text() == "previous"


def test_stale_outputs_are_cleared(ref_ctx):
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    out = ref_ctx.stage_dir("compare")
    out.mkdir(parents=True)
    (out / "scatter_old.png").write_bytes(b"x")
    compare.run_compare(ref_ctx)
    assert not (out / "scatter_old.png").exists()


def test_empty_overlap_gives_n0_rows_nan_stats_and_no_plot(ref_ctx, caplog):
    write_metrics(ref_ctx, "arbor", [101, 102, 103], [0.3, 0.3, 0.3], [20.0, 20.0, 20.0])
    with caplog.at_level(logging.WARNING):
        result = compare.run_compare(ref_ctx)
    summary = read_csv(ref_ctx, "summary.csv")
    for metric in ("dbh", "height"):
        s = row(summary, pipeline="arbor", metric=metric)
        assert (s.n, s.n_missing, s.n_extra) == (0, 10, 3)
        assert s[["bias", "MAE", "RMSE", "R2", "slope", "intercept"]].isna().all()
    assert len(read_csv(ref_ctx, "errors.csv")) == 0
    assert result["plots"] == [] and not list(ref_ctx.stage_dir("compare").glob("*.png"))
    assert "no dbh value could be paired" in caplog.text and "T<k>" in caplog.text
    assert result["metrics"]["arbor"]["dbh"]["bias"] is None


def test_single_pair_does_not_crash(ref_ctx):
    write_metrics(ref_ctx, "arbor", [2], [0.25], [17.5])
    compare.run_compare(ref_ctx)
    s = row(read_csv(ref_ctx, "summary.csv"), pipeline="arbor", metric="dbh")
    assert s.n == 1 and s.bias == pytest.approx(0.02) and math.isnan(s.R2) and math.isnan(s.slope)


def test_metrics_file_without_status_column_is_all_ok(ref_ctx):
    path = ref_ctx.stage_dir("arbor") / "metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"ordinal": ORD, "dbh_m": REF_DBH_M + 0.01, "height_m": REF_H_M}).to_csv(path, index=False)
    compare.run_compare(ref_ctx)
    assert row(read_csv(ref_ctx, "summary.csv"), pipeline="arbor", metric="dbh").n == 10


def test_status_is_case_and_space_insensitive_and_nan_status_is_not_ok(ref_ctx):
    path = ref_ctx.stage_dir("arbor") / "metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"ordinal": [1, 2, 3, 4], "dbh_m": [0.2] * 4, "height_m": [15.0] * 4, "status": ["OK", " ok ", None, "ok?"]}
    ).to_csv(path, index=False)
    compare.run_compare(ref_ctx)
    assert row(read_csv(ref_ctx, "summary.csv"), pipeline="arbor", metric="dbh").n == 2


def test_malformed_metrics_files(ref_ctx):
    path = ref_ctx.stage_dir("arbor") / "metrics.csv"
    path.parent.mkdir(parents=True)
    path.write_text("tree,dbh_m,height_m\n1,0.2,15\n")
    with pytest.raises(StageError, match="no 'ordinal' column.*tree"):
        compare.run_compare(ref_ctx)
    path.write_text("ordinal,volume\n1,0.2\n")
    with pytest.raises(StageError, match="neither 'dbh_m' nor 'height_m'"):
        compare.run_compare(ref_ctx)
    path.write_text("ordinal,dbh_m,height_m,status\n1,0.2,15,ok\n1,0.2,15,ok\n")
    with pytest.raises(StageError, match="duplicate ordinals \\[1\\]"):
        compare.run_compare(ref_ctx)
    path.write_text("")
    with pytest.raises(StageError, match="is empty"):
        compare.run_compare(ref_ctx)


def test_metrics_with_only_one_metric_column_and_na_markers(ref_ctx, caplog):
    path = ref_ctx.stage_dir("arbor") / "metrics.csv"
    path.parent.mkdir(parents=True)
    path.write_text("ordinal,dbh_m,status\n1,0.20,ok\n2,NA,ok\nT3,0.26,ok\n", encoding="utf-8-sig")
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        compare.run_compare(ref_ctx)
    summary = read_csv(ref_ctx, "summary.csv")
    assert row(summary, pipeline="arbor", metric="dbh").n == 2  # 'T3' ordinal accepted, NA is missing
    assert row(summary, pipeline="arbor", metric="height").n == 0
    assert "no 'height_m' column" in caplog.text


# -- reference formats through the stage ----------------------------------------------------------------------
def test_semicolon_decimal_comma_bom_reference(ctx, project):
    reference = write_reference_csv(
        project / "data" / "in" / "pl.csv", [1, 2, 3], [31.5, 28.0, 40.25], [24.5, 22.0, 27.75]
    )
    assert reference.read_bytes().startswith(b"\xef\xbb\xbf") and b"31,5" in reference.read_bytes()
    ctx = replace(ctx, reference=reference)
    write_tree_map(ctx, [1, 2, 3], [0.325, 0.28, 0.4125], [24.0, 22.5, 27.75])
    compare.run_compare(ctx)
    s = row(read_csv(ctx, "summary.csv"), pipeline="3dfin", metric="dbh")
    assert s.n == 3 and s.bias == pytest.approx((0.01 + 0.0 + 0.01) / 3, abs=1e-6)
    s = row(read_csv(ctx, "summary.csv"), pipeline="3dfin", metric="height")
    assert s.bias == pytest.approx((-0.5 + 0.5 + 0) / 3, abs=1e-9)


def test_tree_labels_T12_and_float_ids(ctx, project):
    reference = write_reference_csv(
        project / "data" / "in" / "t.csv", [12, 13, 14], [30.0, 31.0, 32.0], [20.0, 21.0, 22.0],
        labels=["T12", "13", "14.0"],
    )  # fmt: skip
    ctx = replace(ctx, reference=reference)
    write_tree_map(ctx, [12, 13, 14], [0.30, 0.31, 0.32], [20.0, 21.0, 22.0])
    compare.run_compare(ctx)
    summary = read_csv(ctx, "summary.csv")
    assert row(summary, pipeline="3dfin", metric="dbh").n == 3 and row(summary, pipeline="3dfin", metric="dbh").RMSE < 1e-9


@pytest.mark.parametrize(
    "kwargs, dbh_in, h_in",
    [
        (dict(dbh_unit="mm", height_unit="cm"), [300.0, 400.0], [2000.0, 2500.0]),
        (dict(dbh_unit="m", height_unit="m"), [0.3, 0.4], [20.0, 25.0]),
        (dict(dbh_unit="cm"), [30.0, 40.0], [20.0, 25.0]),
        (dict(), [30.0, 40.0], [20.0, 25.0]),
    ],
)
def test_unit_parameters(ctx, project, kwargs, dbh_in, h_in):
    path = project / "data" / "in" / "u.csv"
    path.write_text(f"id;dbh;h\n1;{dbh_in[0]};{h_in[0]}\n2;{dbh_in[1]};{h_in[1]}\n".replace(".", ","), encoding="utf-8")
    ctx = replace(ctx, reference=path)
    write_tree_map(ctx, [1, 2], [0.31, 0.41], [20.5, 25.5])
    compare.run_compare(ctx, **kwargs)
    summary = read_csv(ctx, "summary.csv")
    assert row(summary, pipeline="3dfin", metric="dbh").bias == pytest.approx(0.01)
    assert row(summary, pipeline="3dfin", metric="height").bias == pytest.approx(0.5)


def test_unit_taken_from_column_title(ctx, project):
    path = project / "data" / "in" / "u.csv"
    path.write_text("Nr;DBH [mm];Wysokość [cm]\n1;300;2000\n2;400;2500\n", encoding="utf-8")
    ctx = replace(ctx, reference=path)
    write_tree_map(ctx, [1, 2], [0.31, 0.41], [20.5, 25.5])
    compare.run_compare(ctx)
    assert row(read_csv(ctx, "summary.csv"), pipeline="3dfin", metric="dbh").bias == pytest.approx(0.01)


def test_xlsx_reference_with_sheet_param(ctx, project):
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.title = "Info"
    wb.active.append(["not the data"])
    wb.active.append(["still not"])
    ws = wb.create_sheet("Pomiary")
    ws.append(["Nr drzewa", "Pierśnica [cm]", "Wysokość [m]"])
    for i, (d, h) in enumerate([(30.5, 20.0), ("41,5", "25,5"), (35.0, None)], start=1):
        ws.append([f"T{i}", d, h])
    path = project / "data" / "in" / "ref.xlsx"
    wb.save(path)
    ctx = replace(ctx, reference=path)
    write_tree_map(ctx, [1, 2, 3], [0.315, 0.425, 0.36], [20.5, 26.0, 21.0])
    with pytest.raises(StageError, match="no tree number column found"):
        compare.run_compare(ctx)  # the first sheet is the info sheet
    compare.run_compare(ctx, sheet="Pomiary")
    summary = read_csv(ctx, "summary.csv")
    s = row(summary, pipeline="3dfin", metric="dbh")
    assert s.n == 3 and s.bias == pytest.approx((0.01 + 0.01 + 0.01) / 3, abs=1e-9)
    s = row(summary, pipeline="3dfin", metric="height")
    assert s.n == 2 and s.n_extra == 1 and s.bias == pytest.approx(0.5)  # tree 3 has no reference height
    compare.run_compare(ctx, sheet="1")  # index spelling, as hydra passes it


def test_duplicate_reference_ids_fail_the_stage(ctx, project):
    reference = write_reference_csv(project / "data" / "in" / "d.csv", [1, 2, 2], [30, 31, 32], [20, 21, 22])
    ctx = replace(ctx, reference=reference)
    write_tree_map(ctx, [1, 2], [0.3, 0.31], [20.0, 21.0])
    with pytest.raises(StageError, match="duplicate tree numbers"):
        compare.run_compare(ctx)


def test_reference_without_height_column_compares_dbh_only(ctx, project, caplog):
    path = project / "data" / "in" / "dbh_only.csv"
    path.write_text("Nr;DBH\n1;30\n2;40\n", encoding="utf-8")
    ctx = replace(ctx, reference=path)
    write_tree_map(ctx, [1, 2], [0.31, 0.41], [20.0, 25.0])
    with caplog.at_level(logging.WARNING):
        result = compare.run_compare(ctx)
    summary = read_csv(ctx, "summary.csv")
    assert set(summary.metric) == {"dbh"}
    assert result["plots"] == ["data/out/r1/compare/scatter_dbh.png"]
    assert not (ctx.stage_dir("compare") / "scatter_height.png").exists()


def test_explicit_columns_and_columns_file_precedence(ctx, project):
    path = project / "data" / "in" / "x.csv"
    path.write_text("A;B;C;D\n1;30;20;99\n2;40;25;99\n", encoding="utf-8")
    yaml_path = project / "configs" / "cols.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text(
        "tree_id: {column: A}\ndbh: {column: D, unit: cm}\nheight: {column: C, unit: m}\ncsv: {sep: ';'}\n",
        encoding="utf-8",
    )
    ctx = replace(ctx, reference=path)
    write_tree_map(ctx, [1, 2], [0.31, 0.41], [20.5, 25.5])
    # YAML alone maps DBH to column D (99 cm -> far off) ...
    compare.run_compare(ctx, columns_file="configs/cols.yaml")
    assert row(read_csv(ctx, "summary.csv"), pipeline="3dfin", metric="dbh").bias == pytest.approx(-0.63, abs=1e-6)  # mean of 0.31-0.99 and 0.41-0.99
    # ... the explicit stage parameter wins over the YAML, the YAML still supplies the rest
    compare.run_compare(ctx, columns_file="configs/cols.yaml", dbh_col="B")
    summary = read_csv(ctx, "summary.csv")
    assert row(summary, pipeline="3dfin", metric="dbh").bias == pytest.approx(0.01)
    assert row(summary, pipeline="3dfin", metric="height").bias == pytest.approx(0.5)


def test_columns_file_errors(ref_ctx):
    with pytest.raises(StageError, match="columns_file not found"):
        compare.run_compare(ref_ctx, columns_file="configs/missing.yaml")


def test_ambiguous_columns_fail_with_listing(ctx, project):
    path = project / "data" / "in" / "amb.csv"
    path.write_text("tree;dbh;d13;h\n1;30;30;20\n", encoding="utf-8")
    ctx = replace(ctx, reference=path)
    write_tree_map(ctx, [1], [0.3], [20.0])
    with pytest.raises(StageError, match=r"candidates 'dbh', 'd13'.*Columns found: 'tree', 'dbh', 'd13', 'h'"):
        compare.run_compare(ctx)


# -- plots / parameters / misc --------------------------------------------------------------------------------
def test_plots_false_writes_no_png(ref_ctx):
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M + 0.01, REF_H_M)
    result = compare.run_compare(ref_ctx, plots=False)
    out = ref_ctx.stage_dir("compare")
    assert result["plots"] == [] and not list(out.glob("*.png"))
    assert (out / "errors.csv").is_file() and (out / "summary.csv").is_file()


def test_scatter_plots_are_valid_images(ref_ctx):
    write_metrics(ref_ctx, "smartqsm/a", ORD, REF_DBH_M + 0.01, REF_H_M + 0.3)
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M * 1.05, REF_H_M - 0.3)
    compare.run_compare(ref_ctx)
    import matplotlib.image as mpimg

    for metric in ("dbh", "height"):
        img = mpimg.imread(ref_ctx.stage_dir("compare") / f"scatter_{metric}.png")
        assert img.shape[0] > 400 and img.shape[1] > 400


def test_invalid_parameters(ref_ctx):
    with pytest.raises(StageError, match="dbh_unit"):
        compare.run_compare(ref_ctx, dbh_unit="inch")
    with pytest.raises(StageError, match="tolerance_dbh_cm"):
        compare.run_compare(ref_ctx, tolerance_dbh_cm=0)


def test_missing_or_absent_reference(ctx):
    write_metrics(ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    with pytest.raises(StageError, match="needs a reference table"):
        compare.run_compare(ctx)
    with pytest.raises(StageError, match="reference file not found"):
        compare.run_compare(replace(ctx, reference=ctx.project_dir / "data" / "in" / "nope.csv"))


def test_dry_run_writes_nothing_and_never_raises_on_bad_reference(ctx, project, caplog):
    bad = project / "data" / "in" / "bad.csv"
    bad.write_text("foo;bar\n1;2\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(replace(ctx, reference=bad, dry_run=True))
    assert result["dry_run"] is True and not ctx.stage_dir("compare").exists()
    assert "would be rejected" in caplog.text
    assert compare.run_compare(replace(ctx, dry_run=True))["dry_run"] is True  # no reference at all


def test_signature_is_exposed_to_hydra_zen():
    from hydra_zen import instantiate

    from lidar_bridge.configs import stage_config

    partial = instantiate(stage_config(compare.run_compare))
    assert partial.keywords["plots"] is True and partial.keywords["dbh_unit"] == "auto"
    assert partial.keywords["columns_file"] is None and partial.keywords["sheet"] is None
    assert partial.keywords["allow_stale"] is False
    assert partial.keywords["exclude_ambiguous"] is False and partial.keywords["exclude_estimated"] is False
    assert set(partial.keywords) >= {
        "columns_file", "tree_id_col", "dbh_col", "dbh_unit", "height_col", "height_unit", "sheet", "plots",
        "allow_stale", "exclude_ambiguous", "exclude_estimated",
    }  # fmt: skip


def test_format_summary_is_aligned_and_marks_nan():
    df = pd.DataFrame(
        [
            dict(pipeline="3dfin", metric="dbh", n=8, n_missing=2, n_extra=0, bias=0.01, MAE=0.01, RMSE=0.0123,
                 rel_bias_pct=3.2, rel_RMSE_pct=4.1, R2=0.987, slope=1.01, intercept=-0.002),
            dict(pipeline="smartqsm:layerwise-clustering", metric="height", n=0, n_missing=10, n_extra=1,
                 bias=math.nan, MAE=math.nan, RMSE=math.nan, rel_bias_pct=math.nan, rel_RMSE_pct=math.nan,
                 R2=math.nan, slope=math.nan, intercept=math.nan),
        ]
    )  # fmt: skip
    lines = compare.format_summary(df).splitlines()
    assert len(lines) == 3 and len({len(line) for line in lines}) == 1  # header + 2 rows, all the same width
    assert lines[0].split()[:3] == ["pipeline", "metric", "n"]
    assert lines[2].split()[-1] == "-" and "+0.010" in lines[1]


# -- stale results: source fingerprint -------------------------------------------------------------------------
OTHER_SPLIT = "0123456789abcdef"


def stale_setup(ctx: RunContext) -> None:
    """3dfin (current tree map), smartqsm 'fresh' (stamped with the current split), smartqsm 'old' and arbor (stamped
    with another split)."""
    write_tree_map(ctx, ORD, REF_DBH_M + 0.01, REF_H_M)
    write_metrics(ctx, "smartqsm/fresh", ORD, REF_DBH_M + 0.02, REF_H_M)
    write_metrics(ctx, "smartqsm/old", ORD, REF_DBH_M - 0.02, REF_H_M)
    write_metrics(ctx, "arbor", ORD, REF_DBH_M * 1.1, REF_H_M)
    stamp(ctx, "smartqsm/fresh")
    stamp(ctx, "smartqsm/old", OTHER_SPLIT)
    stamp(ctx, "arbor", OTHER_SPLIT)


def test_stale_results_are_skipped_with_a_warning(ref_ctx, caplog):
    stale_setup(ref_ctx)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["3dfin", "smartqsm:fresh"]
    assert result["unverified"] == [] and len(result["skipped"]) == 2
    assert "smartqsm:old skipped: its results were computed on a different split (re-run the stage" in caplog.text
    assert "arbor skipped: its results were computed on a different split" in caplog.text
    assert "cannot verify" not in caplog.text
    assert set(read_csv(ref_ctx, "summary.csv").pipeline) == {"3dfin", "smartqsm:fresh"}
    assert set(read_csv(ref_ctx, "errors.csv").pipeline) == {"3dfin", "smartqsm:fresh"}


def test_a_new_split_makes_earlier_results_stale(ref_ctx, caplog):
    write_tree_map(ref_ctx, ORD, REF_DBH_M, REF_H_M)
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    stamp(ref_ctx, "arbor")
    assert compare.run_compare(ref_ctx)["pipelines"] == ["3dfin", "arbor"]
    write_tree_map(ref_ctx, ORD[::-1], REF_DBH_M[::-1], REF_H_M[::-1] + 1.0)  # a re-split: other numbering
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["3dfin"] and "arbor skipped" in caplog.text


def test_all_qsm_results_stale_leaves_the_3dfin_comparison(ref_ctx):
    write_tree_map(ref_ctx, ORD, REF_DBH_M + 0.01, REF_H_M)
    write_metrics(ref_ctx, "smartqsm/a", ORD, REF_DBH_M, REF_H_M)
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    stamp(ref_ctx, "smartqsm/a", OTHER_SPLIT)
    stamp(ref_ctx, "arbor", OTHER_SPLIT)
    result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["3dfin"]
    assert [s.split(" skipped")[0] for s in result["skipped"]] == ["smartqsm:a", "arbor"]


def test_missing_fingerprint_is_compared_with_one_warning_per_pipeline(ref_ctx, caplog):
    write_tree_map(ref_ctx, ORD, REF_DBH_M, REF_H_M)
    write_metrics(ref_ctx, "smartqsm/a", ORD, REF_DBH_M, REF_H_M)
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    (ref_ctx.stage_dir("arbor") / "source_fingerprint.txt").write_text("")  # unreadable counts as absent
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["3dfin", "smartqsm:a", "arbor"] and result["skipped"] == []
    assert result["unverified"] == ["arbor", "smartqsm:a"]
    for name in ("smartqsm:a", "arbor"):
        message = f"{name}: no source fingerprint, cannot verify the results belong to the current split"
        assert caplog.text.count(message) == 1
    assert "3dfin: no source" not in caplog.text


def test_matching_fingerprints_are_silent(ref_ctx, caplog):
    write_tree_map(ref_ctx, ORD, REF_DBH_M, REF_H_M)
    write_metrics(ref_ctx, "smartqsm/a", ORD, REF_DBH_M, REF_H_M)
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    stamp(ref_ctx, "smartqsm/a")
    stamp(ref_ctx, "arbor")
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["3dfin", "smartqsm:a", "arbor"]
    assert result["skipped"] == [] and result["unverified"] == []
    assert "fingerprint" not in caplog.text and "different split" not in caplog.text


def test_allow_stale_compares_stale_results_anyway_with_a_warning(ref_ctx, caplog):
    stale_setup(ref_ctx)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx, allow_stale=True)
    assert result["pipelines"] == ["3dfin", "smartqsm:fresh", "smartqsm:old", "arbor"]
    assert result["skipped"] == [] and result["unverified"] == ["arbor", "smartqsm:old"]  # 'fresh' is verified
    assert "smartqsm:old: its results were computed on a different split, compared anyway" in caplog.text
    assert "arbor: its results were computed on a different split, compared anyway" in caplog.text
    assert "smartqsm:fresh:" not in caplog.text
    summary = read_csv(ref_ctx, "summary.csv")
    assert row(summary, pipeline="arbor", metric="dbh").n == 10
    assert row(summary, pipeline="smartqsm:old", metric="dbh").n == 10


def test_stored_fingerprint_without_a_current_tree_map_cannot_be_verified(ref_ctx, caplog):
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M)
    stamp(ref_ctx, "arbor", OTHER_SPLIT)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)
    assert result["pipelines"] == ["arbor"] and result["unverified"] == ["arbor"]
    assert "tree_map.csv not found, cannot verify the results belong to the current split" in caplog.text


# -- quality flags: id_confidence / estimated -------------------------------------------------------------------
ARBOR_NOTES = {
    2: "Small tree allometry", 4: "No valid measure; no_ground_reference", 5: "bh_not_converged",
    6: "NO VALID MEASURE", 8: "arbor_treeID=3", 9: "Small tree allometry",
}  # fmt: skip
ESTIMATED_ORDINALS = {2, 4, 5, 6}  # tree 9 has the note too, but its status is failed: never paired


def flagged_setup(ctx: RunContext) -> None:
    """3dfin: all 10 trees, 1 'resolved', 3 and 7 'ambiguous'. smartqsm:q: exact, benign notes. arbor: exact except
    that the DBH of the 4 estimated trees is 10 cm too large; tree 9 failed."""
    confidence = ["exact"] * 10
    confidence[0], confidence[2], confidence[6] = "resolved", "ambiguous", "ambiguous"
    write_tree_map(ctx, ORD, REF_DBH_M + 0.01, REF_H_M - 0.5, id_confidence=confidence)
    write_metrics(ctx, "smartqsm/q", ORD, REF_DBH_M, REF_H_M, note="reused existing _qsm.mat")
    dbh = REF_DBH_M + np.where(np.isin(ORD, sorted(ESTIMATED_ORDINALS)), 0.10, 0.0)
    status = ["failed" if i == 9 else "ok" for i in ORD]
    write_metrics(ctx, "arbor", ORD, dbh, REF_H_M, status=status, note=[ARBOR_NOTES.get(i, "") for i in ORD])


def test_quality_flags_reach_errors_and_summary(ref_ctx, caplog):
    flagged_setup(ref_ctx)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.stages.compare"):
        result = compare.run_compare(ref_ctx)
    errors, summary = read_csv(ref_ctx, "errors.csv"), read_csv(ref_ctx, "summary.csv")

    # errors.csv: id_confidence comes from the tree map for every pipeline and metric
    assert (errors[errors.ordinal.isin([3, 7])].id_confidence == "ambiguous").all()
    assert (errors[errors.ordinal == 1].id_confidence == "resolved").all()
    assert set(errors[~errors.ordinal.isin([1, 3, 7])].id_confidence) == {"exact"}
    # ... and `estimated` is a DBH flag of the pipelines whose note says so
    flagged = errors[errors.estimated]
    assert set(flagged.ordinal) == ESTIMATED_ORDINALS
    assert set(flagged.pipeline) == {"arbor"} and set(flagged.metric) == {"dbh"}
    assert row(errors, pipeline="arbor", metric="dbh", ordinal=4).estimated
    assert not row(errors, pipeline="arbor", metric="height", ordinal=4).estimated
    assert not row(errors, pipeline="arbor", metric="dbh", ordinal=8).estimated  # an unrelated note
    assert not errors[errors.pipeline.isin(["3dfin", "smartqsm:q"])].estimated.any()

    # summary.csv counts among the paired trees; by default nothing is dropped
    columns = ["n", "n_ambiguous", "n_estimated", "n_excluded"]
    expected = {
        ("3dfin", "dbh"): (10, 2, 0, 0), ("3dfin", "height"): (10, 2, 0, 0),
        ("smartqsm:q", "dbh"): (10, 2, 0, 0),
        ("arbor", "dbh"): (9, 2, 4, 0), ("arbor", "height"): (9, 2, 0, 0),
    }  # fmt: skip
    for (pipeline, metric), counts in expected.items():
        assert tuple(row(summary, pipeline=pipeline, metric=metric)[columns]) == counts, (pipeline, metric)
    assert row(summary, pipeline="arbor", metric="dbh").bias == pytest.approx(0.4 / 9, abs=1e-5)  # the estimates are in the stats
    assert result["metrics"]["arbor"]["dbh"]["n_estimated"] == 4

    # one log line per pipeline / metric with the counts; included trees are a warning
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("arbor dbh: 4 of 9 paired values are estimated by arbor") for m in warnings)
    assert any(m.startswith("3dfin dbh: 2 of 10 paired values have an ambiguous tree number") for m in warnings)
    assert any(m.startswith("arbor height: 2 of 9 paired values have an ambiguous") for m in warnings)
    assert not any(m.startswith("arbor height") and "estimated" in m for m in warnings)
    assert not any(m.startswith("smartqsm:q dbh") and "estimated" in m for m in warnings)
    assert " amb " in caplog.text and " est " in caplog.text  # columns of the logged summary table


def test_exclude_estimated_drops_flagged_dbh_values_only(ref_ctx, caplog):
    flagged_setup(ref_ctx)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.stages.compare"):
        compare.run_compare(ref_ctx, exclude_estimated=True)
    errors, summary = read_csv(ref_ctx, "errors.csv"), read_csv(ref_ctx, "summary.csv")
    s = row(summary, pipeline="arbor", metric="dbh")
    assert (s.n, s.n_estimated, s.n_excluded, s.n_missing, s.n_extra, s.n_ambiguous) == (5, 0, 4, 1, 0, 2)
    assert s.bias == pytest.approx(0.0, abs=1e-9) and s.RMSE == pytest.approx(0.0, abs=1e-9)
    assert sorted(errors[(errors.pipeline == "arbor") & (errors.metric == "dbh")].ordinal) == [1, 3, 7, 8, 10]
    assert row(summary, pipeline="arbor", metric="height").n == 9  # heights are not estimated values
    assert row(summary, pipeline="3dfin", metric="dbh").n == 10
    assert row(summary, pipeline="smartqsm:q", metric="dbh").n == 10
    assert "arbor dbh: excluded 4 of 9 paired values" in caplog.text
    assert not any(r.levelno == logging.WARNING and "estimated" in r.getMessage() for r in caplog.records)


def test_exclude_ambiguous_drops_the_tree_from_every_pipeline_and_metric(ref_ctx, caplog):
    flagged_setup(ref_ctx)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.stages.compare"):
        compare.run_compare(ref_ctx, exclude_ambiguous=True)
    errors, summary = read_csv(ref_ctx, "errors.csv"), read_csv(ref_ctx, "summary.csv")
    assert not errors.ordinal.isin([3, 7]).any() and not (errors.id_confidence == "ambiguous").any()
    for pipeline, metric, n in (
        ("3dfin", "dbh", 8), ("3dfin", "height", 8), ("smartqsm:q", "dbh", 8), ("arbor", "dbh", 7), ("arbor", "height", 7),
    ):  # fmt: skip
        s = row(summary, pipeline=pipeline, metric=metric)
        assert (s.n, s.n_ambiguous, s.n_excluded) == (n, 0, 2), (pipeline, metric)
    s = row(summary, pipeline="arbor", metric="dbh")
    assert s.n_estimated == 4 and s.bias == pytest.approx(0.4 / 7, abs=1e-5)  # estimated values stay when only ambiguous is set
    assert "3dfin dbh: excluded 2 of 10 paired values" in caplog.text


def test_exclude_both_and_missing_vs_excluded_bookkeeping(ref_ctx):
    flagged_setup(ref_ctx)
    compare.run_compare(ref_ctx, exclude_ambiguous=True, exclude_estimated=True)
    s = row(read_csv(ref_ctx, "summary.csv"), pipeline="arbor", metric="dbh")
    assert (s.n, s.n_excluded, s.n_missing) == (3, 6, 1)  # 1, 8, 10 left; tree 9 failed (missing, not excluded)
    assert s.bias == pytest.approx(0.0, abs=1e-9)


def test_excluding_everything_gives_an_empty_row_not_a_crash(ref_ctx):
    write_tree_map(ref_ctx, ORD, REF_DBH_M, REF_H_M, id_confidence="ambiguous")
    result = compare.run_compare(ref_ctx, exclude_ambiguous=True)
    summary = read_csv(ref_ctx, "summary.csv")
    for metric in ("dbh", "height"):
        s = row(summary, pipeline="3dfin", metric=metric)
        assert (s.n, s.n_excluded) == (0, 10) and math.isnan(s.bias)
    assert len(read_csv(ref_ctx, "errors.csv")) == 0 and result["plots"] == []


def test_tree_map_without_id_confidence_column(ref_ctx, caplog):
    write_tree_map(ref_ctx, ORD, REF_DBH_M, REF_H_M, id_confidence=None)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):
        compare.run_compare(ref_ctx, exclude_ambiguous=True)
    assert "has no id_confidence column" in caplog.text and "exclude_ambiguous=true has no effect" in caplog.text
    errors, summary = read_csv(ref_ctx, "errors.csv"), read_csv(ref_ctx, "summary.csv")
    assert (errors.id_confidence == "unknown").all()
    s = row(summary, pipeline="3dfin", metric="dbh")
    assert (s.n, s.n_ambiguous, s.n_excluded) == (10, 0, 0)


def test_no_tree_map_means_unknown_confidence(ref_ctx):
    write_metrics(ref_ctx, "arbor", ORD, REF_DBH_M, REF_H_M, note="Small tree allometry")
    compare.run_compare(ref_ctx)
    errors = read_csv(ref_ctx, "errors.csv")
    assert (errors.id_confidence == "unknown").all()
    assert errors[errors.metric == "dbh"].estimated.all() and not errors[errors.metric == "height"].estimated.any()


@pytest.mark.parametrize(
    "note, expected",
    [
        ("", False), ("Small tree allometry", True), ("no valid measure", True), ("NO VALID MEASURE", True),
        ("arbor_treeID=3; bh_not_converged", True), ("No valid measure; no_ground_reference", True),
        ("reused existing _qsm.mat", False), ("no_ground_reference", False), ("breast height outside the stem", False),
        ("no_Z0_flat_ground_assumed", False),
    ],
)  # fmt: skip
def test_estimated_dbh_notes(tmp_path, note, expected):
    path = tmp_path / "metrics.csv"
    frame = pd.DataFrame(
        {"ordinal": [1, 2], "dbh_m": [0.2, 0.3], "height_m": [10.0, 11.0], "status": "ok", "note": [note, ""]}
    )
    frame.to_csv(path, index=False)
    table = compare.read_estimates("arbor", path).table
    assert bool(table.loc[1, "estimated_dbh"]) is expected and not bool(table.loc[2, "estimated_dbh"])


def test_read_id_confidence_normalises_and_tolerates_blanks(tmp_path):
    path = tmp_path / "tree_map.csv"
    path.write_text("ordinal,id_confidence\n1,Exact\n2, ambiguous \n3,\nbad,exact\n2,exact\n")
    series = compare.read_id_confidence(path)
    assert series.to_dict() == {1: "exact", 2: "ambiguous", 3: "unknown"}  # first duplicate wins, bad ordinal dropped


# -- reference units through the stage --------------------------------------------------------------------------
def test_dbh_in_metres_under_a_plain_title_fails_the_stage(ctx, project, caplog):
    path = project / "data" / "in" / "metres.csv"
    path.write_text("Nr;DBH;H\n1;0,30;20\n2;0,40;25\n", encoding="utf-8")
    ctx = replace(ctx, reference=path)
    write_tree_map(ctx, [1, 2], [0.31, 0.41], [20.5, 25.5])
    with pytest.raises(StageError, match=r"DBH column 'DBH' has median 0\.35.*Probably metres.*compare\.dbh_unit=m"):
        compare.run_compare(ctx)
    assert not ctx.stage_dir("compare").exists()  # nothing written
    compare.run_compare(ctx, dbh_unit="m")  # the escape hatch
    assert row(read_csv(ctx, "summary.csv"), pipeline="3dfin", metric="dbh").bias == pytest.approx(0.01)
    with caplog.at_level(logging.WARNING):  # an explicit (even wrong) unit is the user's call: warning only
        compare.run_compare(ctx, dbh_unit="cm")
    assert "implausible" in caplog.text
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.compare"):  # dry-run reports, never raises
        assert compare.run_compare(replace(ctx, dry_run=True))["dry_run"] is True
    assert "would be rejected" in caplog.text and "Probably metres" in caplog.text

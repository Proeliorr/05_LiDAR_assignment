from __future__ import annotations

import csv
import logging
import math
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from scipy.io import savemat

from lidar_bridge import runner
from lidar_bridge.paths import RunContext
from lidar_bridge.runner import ToolError
from lidar_bridge.stages import smartqsm
from lidar_bridge.stages._common import FINGERPRINT_FILE, StageError, read_fingerprint, tree_map_fingerprint

LC = "layerwise-clustering"
SC = "space-colonization"
GPU_PRESET = "spconv-contraction-LEAFON-GPU"
CPU_PRESET = "spconv-contraction-LEAFON-cpu"

PRESET_LC = """\
voxel_down_size: 0.01
filter: statistical_outlier_removal
kwargs_for_filter:
  nb_neighbors: 10
  std_ratio: 4.0
skeletonization:
  category: segmentation
  core_algorithm_kwargs:
    max_patch_size: 0.04
    neighborhood_size: 0.055
refinement:
  min_radius: !!float 1.0e-3
  big: !!float 1.0e10
  x0: ~
  fn: "lambda wpl, r: wpl / r"
modeling:
  resolution: 0.01
"""
PRESET_SC = """\
voxel_down_size: 0.02
filter: ~
using_density_adaptation_in_skeletonization: True
skeletonization:
  category: thinning
"""

PRESET_SPCONV = """\
voxel_down_size: 0.01
skeletonization:
  category: thinning
  core_algorithm: spconv_based_contraction
  core_algorithm_kwargs:
    ckpt_path: ckpts/for_leafon_seasons.pth
    batch_size: 16
    device: {device}
"""

EXPECTED_COLUMNS = [
    "ordinal", "dbh_m", "height_m", "status", "seconds",
    "height_raw_m", "base_z0_m", "dbh_height_above_ground_m", "num_branches", "max_order",
    "stem_volume_m3", "trunk_volume_m3", "n_points", "note",
]  # fmt: skip


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture
def sq(ctx: RunContext, project: Path) -> RunContext:
    """The shared ctx, in a project that has two SmartQSM presets."""
    cfg = project / "third_party" / "SmartQSM" / "configs"
    cfg.mkdir(parents=True)
    (cfg / f"{LC}.yaml").write_text(PRESET_LC, encoding="utf-8")
    (cfg / f"{SC}.yaml").write_text(PRESET_SC, encoding="utf-8")
    return ctx


def make_split(ctx: RunContext, trees: list[dict] | int = 3, **defaults) -> Path:
    """Write split/tree_map.csv + fake tree clouds. ``trees``: a count or a list of per-tree field dicts."""
    split = ctx.stage_dir("split")
    (split / "trees").mkdir(parents=True, exist_ok=True)
    if isinstance(trees, int):
        trees = [{} for _ in range(trees)]
    cols = ["ordinal", "raw_id", "n_points", "base_z0_m", "dbh_m", "th_m", "low_points", "file"]
    rows = []
    for i, spec in enumerate(trees, 1):
        row = dict(ordinal=i, raw_id=100 + i, n_points=5000 + i, base_z0_m=0.1, dbh_m=0.2, th_m=12.0, low_points=False)
        row.update(defaults)
        row.update(spec)
        row["file"] = spec.get("file", "" if row["low_points"] else f"trees/tree_{row['ordinal']:04d}.laz")
        if row["file"] and not spec.get("no_file"):
            (split / row["file"]).write_bytes(f"cloud of tree {row['ordinal']}".encode())
        rows.append(row)
    with (split / "tree_map.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, cols, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return split


def write_mat(path: Path, dbh_cm: float = 20.0, height_m: float = 11.0, **extra) -> None:
    treedata = dict(
        X_m=1.0, Y_m=2.0, num_branches=40, max_order=3, height_m=height_m, DBH_cm=dbh_cm,
        stem_volume_L=500.0, trunk_volume_L=150.0,
    )  # fmt: skip
    treedata.update(extra)
    savemat(path, {"QSM": {"cylinder": {"n": 1.0}, "treedata": treedata, "rundata": {"start": 0}}})


def run_smartqsm(ctx: RunContext, **kw) -> dict:
    """``smartqsm.run_smartqsm`` with the layerwise-clustering preset as default: the default of the stage is a GPU preset."""
    kw.setdefault("config", LC)
    return smartqsm.run_smartqsm(ctx, **kw)


class FakeSmartQsm:
    """Replaces ``run_tool`` in the stage module. ``behaviour[ordinal]`` = dict(mode=..., dbh_cm=..., height_m=..., rc=...)."""

    def __init__(self, ctx: RunContext, behaviour: dict[int, dict] | None = None):
        self.ctx = ctx
        self.behaviour = behaviour or {}
        self.calls: list[dict] = []
        self.hook = None  # called as hook(ordinal, laz_path) before the "tool" produces anything

    @property
    def ordinals(self) -> list[int]:
        return [c["ordinal"] for c in self.calls]

    def __call__(self, env, args, **kw):
        laz = Path(args[-1])
        ordinal = int(laz.stem.split("_")[1])
        self.calls.append(dict(env=env, args=list(args), ordinal=ordinal, laz=laz, **kw))
        if self.hook:
            self.hook(ordinal, laz)
        spec = {"mode": "ok", **self.behaviour.get(ordinal, {})}
        mat = laz.with_name(f"{laz.stem}_qsm.mat")
        mode = spec["mode"]
        if mode == "timeout":
            raise ToolError(
                f"smartqsm timed out after {kw.get('timeout_s')} s and was killed", returncode=-9, output="partial output"
            )
        if mode == "cannot_start":
            raise ToolError("cannot start smartqsm: [Errno 2] No such file or directory: 'in-env'")
        if mode == "ok":
            write_mat(mat, spec.get("dbh_cm", 20.0), spec.get("height_m", 10.0 + ordinal))
        elif mode == "corrupt":
            mat.write_bytes(b"\x00garbage" * 10)
        return subprocess.CompletedProcess(["in-env"], spec.get("rc", 0), stdout=spec.get("out", "Initialized point cloud.\nDone.\nFinished."), stderr=None)


@pytest.fixture
def fake(monkeypatch, sq) -> FakeSmartQsm:
    f = FakeSmartQsm(sq)
    monkeypatch.setattr(smartqsm, "run_tool", f)
    return f


@pytest.fixture
def sqg(sq: RunContext, project: Path) -> RunContext:
    """``sq`` plus the two spconv presets: ``...-GPU`` (device: cuda) and ``...-cpu`` (device: cpu)."""
    cfg = project / "third_party" / "SmartQSM" / "configs"
    (cfg / f"{GPU_PRESET}.yaml").write_text(PRESET_SPCONV.format(device="cuda"), encoding="utf-8")
    (cfg / f"{CPU_PRESET}.yaml").write_text(PRESET_SPCONV.format(device="cpu"), encoding="utf-8")
    return sq


def metrics(ctx: RunContext, config: str = LC) -> list[dict[str, str]]:
    path = ctx.stage_dir("smartqsm") / config / "metrics.csv"
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def by_ordinal(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(r["ordinal"]): r for r in rows}


# ---------------------------------------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------------------------------------
def test_preset_without_overrides_is_copied_verbatim(sq, fake):
    make_split(sq, 1)
    run_smartqsm(sq)
    assert (sq.stage_dir("smartqsm") / LC / "config.yaml").read_text() == PRESET_LC


def test_shallow_override_replaces_top_level_values(sq, fake):
    make_split(sq, 1)
    run_smartqsm(sq, overrides={"voxel_down_size": 0.03, "kwargs_for_filter": {"nb_neighbors": 20, "std_ratio": 2.0}})
    written = yaml.safe_load((sq.stage_dir("smartqsm") / LC / "config.yaml").read_text())
    base = yaml.safe_load(PRESET_LC)
    assert written["voxel_down_size"] == 0.03
    assert written["kwargs_for_filter"] == {"nb_neighbors": 20, "std_ratio": 2.0}
    for key in ("filter", "skeletonization", "refinement", "modeling"):  # untouched blocks survive the yaml round trip
        assert written[key] == base[key]
    assert written["refinement"]["big"] == 1.0e10 and written["refinement"]["x0"] is None
    assert list(written) == list(base)  # preset order kept


def test_overriding_a_block_replaces_it_and_warns_about_lost_keys(sq, fake, caplog):
    make_split(sq, 1)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq, overrides={"skeletonization": {"category": "thinning"}})
    written = yaml.safe_load((sq.stage_dir("smartqsm") / LC / "config.yaml").read_text())
    assert written["skeletonization"] == {"category": "thinning"}  # SHALLOW: the preset's other keys are gone
    assert "REPLACES" in caplog.text and "skeletonization.core_algorithm_kwargs" in caplog.text


def test_dotted_override_changes_one_nested_value(sq, fake):
    make_split(sq, 1)
    run_smartqsm(sq, overrides={"skeletonization.core_algorithm_kwargs.max_patch_size": 0.06})
    written = yaml.safe_load((sq.stage_dir("smartqsm") / LC / "config.yaml").read_text())
    assert written["skeletonization"]["core_algorithm_kwargs"] == {"max_patch_size": 0.06, "neighborhood_size": 0.055}
    assert written["skeletonization"]["category"] == "segmentation"


def test_overrides_are_logged(sq, fake, caplog):
    make_split(sq, 1)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq, overrides={"voxel_down_size": 0.03})
    assert "voxel_down_size: 0.01 -> 0.03" in caplog.text


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"voxel_downsize": 1}, "unknown SmartQSM key 'voxel_downsize'"),
        ({"skeletonization.nope": 1}, "no key 'nope'"),
        ({"voxel_down_size.x": 1}, "not a block"),
        ({"modeling.resolution.deeper": 1}, "not a block"),
    ],
)
def test_bad_override_keys_are_rejected(sq, fake, overrides, message):
    make_split(sq, 1)
    with pytest.raises(StageError, match=message):
        run_smartqsm(sq, overrides=overrides)
    assert not fake.calls


def test_key_known_from_another_preset_is_allowed(sq, fake):
    make_split(sq, 1)
    run_smartqsm(sq, overrides={"using_density_adaptation_in_skeletonization": False})  # only SC has it
    written = yaml.safe_load((sq.stage_dir("smartqsm") / LC / "config.yaml").read_text())
    assert written["using_density_adaptation_in_skeletonization"] is False


def test_unknown_preset_lists_the_available_ones(sq, fake):
    make_split(sq, 1)
    with pytest.raises(StageError, match=r"unknown SmartQSM config 'nope'.*layerwise-clustering.*space-colonization"):
        run_smartqsm(sq, config="nope")


def test_missing_submodule_is_a_clear_error(ctx, monkeypatch):
    monkeypatch.setattr(smartqsm, "run_tool", FakeSmartQsm(ctx))
    make_split(ctx, 1)
    with pytest.raises(StageError, match="submodule"):
        run_smartqsm(ctx)


@pytest.mark.parametrize("bad", ["../x", "a/b", "with space", "", "..", ".hidden"])
def test_config_name_is_validated_because_it_becomes_a_directory(sq, fake, bad):
    make_split(sq, 1)
    with pytest.raises(StageError, match="invalid smartqsm.config"):
        run_smartqsm(sq, config=bad)


def test_yaml_suffix_in_config_name_is_tolerated(sq, fake):
    make_split(sq, 1)
    assert run_smartqsm(sq, config=f"{SC}.yaml")["config"] == SC


def test_sweep_over_presets_uses_separate_directories(sq, fake):
    make_split(sq, 2)
    run_smartqsm(sq, config=LC)
    run_smartqsm(sq, config=SC, overrides={"voxel_down_size": 0.05})
    base = sq.stage_dir("smartqsm")
    assert sorted(p.name for p in base.iterdir()) == [LC, SC]
    for name in (LC, SC):
        assert {"config.yaml", "metrics.csv", "work", "logs"} <= {p.name for p in (base / name).iterdir()}
        assert len(metrics(sq, name)) == 2
    assert yaml.safe_load((base / SC / "config.yaml").read_text())["voxel_down_size"] == 0.05
    assert yaml.safe_load((base / LC / "config.yaml").read_text())["voxel_down_size"] == 0.01
    # each run pointed the tool at its own config and work dir
    lc_call = next(c for c in fake.calls if f"/{LC}/" in c["args"][-1])
    sc_call = next(c for c in fake.calls if f"/{SC}/" in c["args"][-1])
    assert lc_call["args"][4] == str(base / LC / "config.yaml")
    assert sc_call["args"][4] == str(base / SC / "config.yaml")


# ---------------------------------------------------------------------------------------------------------
# tool command
# ---------------------------------------------------------------------------------------------------------
def test_tool_command_is_the_documented_contract(sq, fake):
    make_split(sq, 1)
    run_smartqsm(sq, timeout_s=77)
    (call,) = fake.calls
    out = sq.stage_dir("smartqsm") / LC
    assert call["env"] == "smartqsm"
    assert call["args"] == ["smartqsm-run", "-t", "-y", "-c", str(out / "config.yaml"), str(out / "work" / "tree_0001.laz")]
    assert call["stdin"] == "no\n"
    assert call["timeout_s"] == 77 and call["check"] is False


def test_real_run_tool_argv_and_stdin(sq, monkeypatch):
    """Through the real run_tool: only the process launcher is replaced."""
    make_split(sq, 1)
    seen = {}

    def fake_execute(cmd, *, stdin, extra_env, timeout_s, prefix):
        seen.update(cmd=cmd, stdin=stdin, timeout_s=timeout_s, extra_env=extra_env)
        write_mat(sq.stage_dir("smartqsm") / LC / "work" / "tree_0001_qsm.mat")
        return subprocess.CompletedProcess(cmd, 0, stdout="Done.", stderr=None)

    monkeypatch.setattr(runner, "_execute", fake_execute)
    run_smartqsm(sq, timeout_s=90)
    out = sq.stage_dir("smartqsm") / LC
    assert seen["stdin"] == "no\n"
    assert seen["cmd"] == [
        "in-env", "smartqsm", "smartqsm-run", "-t", "-y", "-c", str(out / "config.yaml"), str(out / "work" / "tree_0001.laz"),
    ]  # fmt: skip
    assert seen["timeout_s"] == 90


# ---------------------------------------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------------------------------------
def test_default_config_is_the_leafoff_gpu_preset(sqg, fake):
    assert smartqsm.DEFAULT_CONFIG == "spconv-contraction-LEAFOFF-GPU"
    (sqg.install_dir / smartqsm.CONFIGS_DIR / f"{smartqsm.DEFAULT_CONFIG}.yaml").write_text(
        PRESET_SPCONV.format(device="cuda"), encoding="utf-8"
    )
    make_split(sqg, 1)
    result = smartqsm.run_smartqsm(sqg)  # no config given
    assert result["config"] == "spconv-contraction-LEAFOFF-GPU" and result["gpu"] is True
    assert (sqg.stage_dir("smartqsm") / smartqsm.DEFAULT_CONFIG / "metrics.csv").is_file()


def test_a_gpu_preset_is_reported_as_gpu_run(sqg, fake):
    make_split(sqg, 2)
    result = run_smartqsm(sqg, config=GPU_PRESET)
    assert [c["env"] for c in fake.calls] == ["smartqsm"] * 2
    assert fake.calls[0]["args"][:4] == ["smartqsm-run", "-t", "-y", "-c"]
    assert result["gpu"] is True and result["counts"] == {"ok": 2}


def test_a_cpu_preset_is_reported_as_cpu_run(sqg, fake):
    make_split(sqg, 1)
    result = run_smartqsm(sqg, config=CPU_PRESET)
    assert fake.calls[0]["env"] == "smartqsm"
    assert result["gpu"] is False


def test_a_cuda_device_override_counts_as_a_gpu_preset(sqg, fake):
    make_split(sqg, 1)
    over = {"skeletonization.core_algorithm_kwargs.device": "cuda"}
    assert run_smartqsm(sqg, config=CPU_PRESET, overrides=over)["gpu"] is True  # a -cpu preset turned into a GPU one
    fake.calls.clear()
    result = run_smartqsm(  # and the other way round: a -GPU preset forced onto the CPU
        sqg, config=GPU_PRESET, overrides={"skeletonization.core_algorithm_kwargs.device": "cpu"}, force=True
    )
    assert result["gpu"] is False


def test_a_gpu_preset_does_not_warn(sqg, fake, caplog):
    make_split(sqg, 1)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sqg, config=GPU_PRESET)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_gpu_is_derived_from_the_preset_name_alone_in_a_dry_run(ctx):
    """Dry run of the whole pipeline without presets on disk: the preset text is unknown, the name decides."""
    dry = replace(ctx, dry_run=True)
    assert smartqsm.run_smartqsm(dry, config=GPU_PRESET)["gpu"] is True
    assert smartqsm.run_smartqsm(dry, config=CPU_PRESET)["gpu"] is False


@pytest.mark.parametrize(
    "config, device, expected",
    [
        ({"skeletonization": {"core_algorithm_kwargs": {"device": "cuda"}}}, "cuda", True),
        ({"skeletonization": {"core_algorithm_kwargs": {"device": "CUDA:0"}}}, "cuda:0", True),
        ({"skeletonization": {"core_algorithm_kwargs": {"device": "cpu"}}}, "cpu", False),
        ({"skeletonization": {"category": "segmentation"}}, None, False),
        ({"skeletonization": None}, None, False),
        ({}, None, False),
    ],
)
def test_preset_device_and_uses_cuda(config, device, expected):
    assert smartqsm.preset_device(config) == device
    assert smartqsm.uses_cuda("any-preset", config) is expected


def test_uses_cuda_falls_back_to_the_name_suffix_only_without_a_device():
    assert smartqsm.uses_cuda("x-GPU", None) is True and smartqsm.uses_cuda("x-cpu", None) is False
    assert smartqsm.uses_cuda("x-GPU", {"skeletonization": {"category": "thinning"}}) is True
    assert smartqsm.uses_cuda("x-GPU", {"skeletonization": {"core_algorithm_kwargs": {"device": "cpu"}}}) is False


CUDA_OOM = (
    "Initialized point cloud.\nTraceback (most recent call last):\n  File \"/opt/SmartQSM/core/core_algorithms/"
    "spconv_based_contraction.py\", line 102, in _contract\n"
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.56 GiB. GPU 0 has a total capacity of 6.00 GiB"
)


def test_cuda_out_of_memory_fails_that_tree_with_the_reason_and_the_others_go_on(sqg, monkeypatch):
    f = FakeSmartQsm(sqg, {2: {"mode": "nomat", "out": CUDA_OOM}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sqg, 3)
    result = run_smartqsm(sqg, config=GPU_PRESET)
    rows = by_ordinal(metrics(sqg, GPU_PRESET))
    assert [rows[i]["status"] for i in (1, 2, 3)] == ["ok", "failed", "ok"] and f.ordinals == [1, 2, 3]
    assert "CUDA out of memory" in rows[2]["note"] and "batch_size" in rows[2]["note"]
    assert "no tree_0002_qsm.mat produced" in rows[2]["note"] and rows[2]["dbh_m"] == ""
    assert "OutOfMemoryError" in (sqg.stage_dir("smartqsm") / GPU_PRESET / "logs" / "tree_0002.log").read_text()  # full output kept
    assert result["counts"] == {"ok": 2, "failed": 1}


def test_gpu_diagnosis_is_only_added_for_gpu_runs(sqg, monkeypatch):
    f = FakeSmartQsm(sqg, {1: {"mode": "nomat", "out": CUDA_OOM}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sqg, 2)
    run_smartqsm(sqg, config=CPU_PRESET)
    row = by_ordinal(metrics(sqg, CPU_PRESET))[1]
    assert row["status"] == "failed" and "batch_size" not in row["note"] and "OutOfMemoryError" in row["note"]


@pytest.mark.parametrize(
    "output, fragment",
    [
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB", "CUDA out of memory"),
        ("RuntimeError: CUDA error: out of memory", "CUDA out of memory"),
        ("RuntimeError: CUDA error: no kernel image is available for execution on the device", "CUDA is not usable"),
        ("ValueError: boom", ""),
        ("", ""),
    ],
)
def test_diagnose_gpu_failure(output, fragment):
    text = smartqsm.diagnose_gpu_failure(output)
    assert (fragment in text) if fragment else text == ""


def test_timeout_zero_means_no_limit(sq, fake):
    make_split(sq, 1)
    run_smartqsm(sq, timeout_s=0)
    assert fake.calls[0]["timeout_s"] is None


def test_tree_cloud_is_copied_to_work_and_original_untouched(sq, fake):
    split = make_split(sq, 1)
    run_smartqsm(sq)
    work = sq.stage_dir("smartqsm") / LC / "work"
    assert (work / "tree_0001.laz").read_bytes() == b"cloud of tree 1"
    assert (split / "trees" / "tree_0001.laz").read_bytes() == b"cloud of tree 1"
    assert sorted(p.name for p in (split / "trees").iterdir()) == ["tree_0001.laz"]  # SmartQSM output went to work/ only


def test_tool_log_is_saved(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"out": "line a\nline b"}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 1)
    run_smartqsm(sq)
    assert (sq.stage_dir("smartqsm") / LC / "logs" / "tree_0001.log").read_text() == "line a\nline b"


# ---------------------------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------------------------
def test_trees_selects_ordinals_in_ascending_order(sq, fake):
    make_split(sq, 5)
    result = run_smartqsm(sq, trees=[4, 2, 4])
    assert fake.ordinals == [2, 4]
    assert [int(r["ordinal"]) for r in metrics(sq)] == [2, 4]
    assert result["n_selected"] == 2


def test_unknown_ordinal_is_an_error(sq, fake):
    make_split(sq, 3)
    with pytest.raises(StageError, match=r"ordinal\(s\) \[7\] are not in tree_map.csv"):
        run_smartqsm(sq, trees=[1, 7])
    with pytest.raises(StageError, match="empty"):
        run_smartqsm(sq, trees=[])
    assert not fake.calls


def test_limit_takes_the_first_n(sq, fake):
    make_split(sq, 5)
    run_smartqsm(sq, limit=2)
    assert fake.ordinals == [1, 2]
    with pytest.raises(StageError, match="positive"):
        run_smartqsm(sq, limit=0)


def test_low_points_trees_are_recorded_and_not_run(sq, fake):
    make_split(sq, [{}, {"low_points": True, "n_points": 120}, {}])
    result = run_smartqsm(sq)
    assert fake.ordinals == [1, 3]
    rows = by_ordinal(metrics(sq))
    assert rows[2]["status"] == "skipped_low_points" and rows[2]["n_points"] == "120"
    assert rows[2]["dbh_m"] == "" and rows[2]["height_m"] == "" and rows[2]["seconds"] == ""
    assert result["counts"] == {"ok": 2, "skipped_low_points": 1}


def test_limit_does_not_count_skipped_low_points_trees(sq, fake):
    make_split(sq, [{"low_points": True}, {}, {}, {}])
    run_smartqsm(sq, limit=2)
    assert fake.ordinals == [2, 3]
    assert [r["status"] for r in metrics(sq)] == ["skipped_low_points", "ok", "ok"]


def test_skip_low_points_false_reports_missing_cloud(sq, fake):
    make_split(sq, [{}, {"low_points": True}])
    run_smartqsm(sq, skip_low_points=False)
    assert fake.ordinals == [1]
    row = by_ordinal(metrics(sq))[2]
    assert row["status"] == "error" and "cloud not found" in row["note"]


def test_missing_tree_map_is_a_clear_error(sq, fake):
    with pytest.raises(StageError, match="run the split stage first"):
        run_smartqsm(sq)


def test_tree_map_without_file_column_uses_conventional_names(sq, fake):
    split = sq.stage_dir("split")
    (split / "trees").mkdir(parents=True)
    (split / "trees" / "tree_0001.laz").write_bytes(b"x")
    (split / "tree_map.csv").write_text("ordinal,n_points,base_z0_m\n1,900,0.1\n")
    run_smartqsm(sq)
    assert fake.ordinals == [1]


def test_read_tree_map_parses_types(sq):
    make_split(sq, [{"base_z0_m": ""}, {"low_points": True, "n_points": 33}])
    first, second = smartqsm.read_tree_map(sq.stage_dir("split") / "tree_map.csv")
    assert math.isnan(first.base_z0_m) and first.low_points is False and first.file == "trees/tree_0001.laz"
    assert second.low_points is True and second.n_points == 33 and second.file == ""


# ---------------------------------------------------------------------------------------------------------
# metrics.csv
# ---------------------------------------------------------------------------------------------------------
def test_metrics_csv_column_contract(sq, fake):
    make_split(sq, 2)
    run_smartqsm(sq)
    with (sq.stage_dir("smartqsm") / LC / "metrics.csv").open(newline="") as fh:
        assert next(csv.reader(fh)) == EXPECTED_COLUMNS
    assert smartqsm.METRICS_COLUMNS == EXPECTED_COLUMNS
    assert EXPECTED_COLUMNS[:5] == ["ordinal", "dbh_m", "height_m", "status", "seconds"]  # what WP7 reads


def test_ok_row_values_and_height_correction(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"dbh_cm": 16.5, "height_m": 12.4}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, [{"base_z0_m": 0.1234, "n_points": 4321}])
    run_smartqsm(sq)
    (row,) = metrics(sq)
    assert row["status"] == "ok" and row["note"] == ""
    assert float(row["dbh_m"]) == pytest.approx(0.165)  # cm -> m
    assert float(row["height_raw_m"]) == pytest.approx(12.4)
    assert float(row["height_m"]) == pytest.approx(12.4 + 0.1234)  # + base_z0_m: height above the TRUE ground
    assert float(row["base_z0_m"]) == pytest.approx(0.1234)
    assert float(row["dbh_height_above_ground_m"]) == pytest.approx(1.3 + 0.1234)
    assert row["num_branches"] == "40" and row["max_order"] == "3" and row["n_points"] == "4321"
    assert float(row["stem_volume_m3"]) == pytest.approx(0.5) and float(row["trunk_volume_m3"]) == pytest.approx(0.15)
    assert float(row["seconds"]) >= 0


def test_no_dbh_when_dbh_is_zero(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"dbh_cm": 0.0, "height_m": 3.0}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 1)
    run_smartqsm(sq)
    (row,) = metrics(sq)
    assert row["status"] == "no_dbh" and row["dbh_m"] == "" and row["height_m"] == ""  # empty unless ok
    assert float(row["height_raw_m"]) == 3.0 and "1.3 m" in row["note"]


def _treedata(dbh_cm=18.6, height_m=13.5, volume_m3=0.4, branches=300):
    return dict(DBH_cm=dbh_cm, height_m=height_m, num_branches=branches, max_order=3, stem_volume_L=volume_m3 * 1000, trunk_volume_L=100.0)


def test_degenerate_model_is_suspect_and_withholds_dbh_and_height():
    """Seen with spconv LEAFON-GPU: 1 branch and 397 m3 for a 13 m tree of 18 cm DBH, which must not pass as status ok."""
    tree = smartqsm.Tree(ordinal=35, base_z0_m=0.1, n_points=225912)
    result = smartqsm.result_from_treedata(tree, _treedata(volume_m3=397.0, branches=1), seconds=60.0)
    assert result.status == "suspect" and "suspect" in smartqsm.STATUSES
    assert math.isnan(result.dbh_m) and math.isnan(result.height_m)  # compare only uses status ok with finite values
    assert "implausible model" in result.note and "397" in result.note and "1 branches" in result.note
    assert "another smartqsm.config" in result.note
    assert result.stem_volume_m3 == pytest.approx(397.0) and result.num_branches == 1  # the evidence stays in the row


def test_volume_check_threshold_and_missing_volume():
    tree = smartqsm.Tree(ordinal=1, base_z0_m=0.1, n_points=5000)
    cylinder = math.pi / 4 * 0.186**2 * 13.6  # DBH 18.6 cm, height 13.5 + 0.1 above ground
    just_below = smartqsm.result_from_treedata(tree, _treedata(volume_m3=0.99 * smartqsm.IMPLAUSIBLE_VOLUME_RATIO * cylinder))
    just_above = smartqsm.result_from_treedata(tree, _treedata(volume_m3=1.01 * smartqsm.IMPLAUSIBLE_VOLUME_RATIO * cylinder))
    assert just_below.status == "ok" and just_above.status == "suspect"
    typical = smartqsm.result_from_treedata(tree, _treedata(volume_m3=0.4))  # ~0.5x a DBH cylinder
    assert typical.status == "ok" and typical.dbh_m == pytest.approx(0.186) and typical.height_m == pytest.approx(13.6)
    no_volume = _treedata()
    del no_volume["stem_volume_L"]
    assert smartqsm.result_from_treedata(tree, no_volume).status == "ok"  # nothing to judge by -> not flagged


def test_missing_base_z0_leaves_height_uncorrected_with_a_note(sq, fake):
    make_split(sq, [{"base_z0_m": ""}])
    run_smartqsm(sq)
    (row,) = metrics(sq)
    assert row["status"] == "ok" and float(row["height_m"]) == float(row["height_raw_m"])
    assert row["base_z0_m"] == "" and row["dbh_height_above_ground_m"] == "" and "base_z0_m unknown" in row["note"]


def test_treedata_without_dbh_field_is_an_error_row(sq, monkeypatch):
    class NoDbh(FakeSmartQsm):
        def __call__(self, env, args, **kw):
            r = super().__call__(env, args, **kw)
            savemat(self.calls[-1]["laz"].with_name("tree_0001_qsm.mat"), {"QSM": {"treedata": {"height_m": 5.0}}})
            return r

    monkeypatch.setattr(smartqsm, "run_tool", NoDbh(sq))
    make_split(sq, [{}, {}])
    with pytest.raises(StageError):  # both trees broken -> nothing usable
        run_smartqsm(sq, trees=[1])
    (row,) = metrics(sq)
    assert row["status"] == "error" and "DBH_cm" in row["note"] and row["dbh_m"] == ""


def test_failed_when_no_mat_is_produced_even_with_exit_code_zero(sq, monkeypatch):
    f = FakeSmartQsm(sq, {2: {"mode": "nomat", "out": "Traceback...\nValueError: boom"}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 3)
    result = run_smartqsm(sq)
    rows = by_ordinal(metrics(sq))
    assert [rows[i]["status"] for i in (1, 2, 3)] == ["ok", "failed", "ok"]  # the loop carries on
    assert "no tree_0002_qsm.mat produced" in rows[2]["note"] and "ValueError: boom" in rows[2]["note"]
    assert rows[2]["dbh_m"] == "" and rows[2]["height_m"] == "" and rows[2]["seconds"] != ""
    assert result["counts"] == {"ok": 2, "failed": 1}


def test_nonzero_exit_code_is_recorded(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"mode": "nomat", "rc": 2}, 2: {}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 2)
    run_smartqsm(sq)
    row = by_ordinal(metrics(sq))[1]
    assert row["status"] == "failed" and "exit code 2" in row["note"]


def test_timeout_status_and_next_tree_still_runs(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"mode": "timeout"}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 2)
    result = run_smartqsm(sq, timeout_s=42)
    rows = by_ordinal(metrics(sq))
    assert rows[1]["status"] == "timeout" and "42 s" in rows[1]["note"] and rows[1]["dbh_m"] == ""
    assert rows[2]["status"] == "ok"
    assert result["counts"] == {"ok": 1, "timeout": 1}
    assert (sq.stage_dir("smartqsm") / LC / "logs" / "tree_0001.log").read_text() == "partial output"


def test_tool_error_other_than_timeout_is_an_error_row(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"mode": "cannot_start"}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 2)
    run_smartqsm(sq)
    row = by_ordinal(metrics(sq))[1]
    assert row["status"] == "error" and "in-env" in row["note"]


def test_corrupt_mat_is_an_error_row(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"mode": "corrupt"}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 2)
    run_smartqsm(sq)
    row = by_ordinal(metrics(sq))[1]
    assert row["status"] == "error" and "cannot read" in row["note"]


def test_stage_fails_only_if_every_attempted_tree_failed(sq, monkeypatch):
    f = FakeSmartQsm(sq, {1: {"mode": "nomat"}, 2: {"mode": "timeout"}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, [{}, {}, {"low_points": True}])
    with pytest.raises(StageError, match="no result for any of the 2 tree"):
        run_smartqsm(sq)
    assert len(metrics(sq)) == 3  # metrics.csv is there for the post-mortem


def test_metrics_csv_is_written_after_every_tree(sq, fake):
    make_split(sq, 4)
    snapshots = {}

    def hook(ordinal, laz):
        path = sq.stage_dir("smartqsm") / LC / "metrics.csv"
        snapshots[ordinal] = [r["ordinal"] for r in metrics(sq)] if path.exists() else None

    fake.hook = hook
    run_smartqsm(sq)
    # when tree k starts, the file already holds trees 1..k-1
    assert snapshots == {1: None, 2: ["1"], 3: ["1", "2"], 4: ["1", "2", "3"]}
    assert len(metrics(sq)) == 4
    assert not list((sq.stage_dir("smartqsm") / LC).glob("*.tmp"))  # atomic write leaves no temp file


def test_interrupt_keeps_finished_trees_and_resume_continues(sq, fake):
    make_split(sq, 4)

    def interrupt(ordinal, laz):
        if ordinal == 3:
            raise KeyboardInterrupt

    fake.hook = interrupt
    with pytest.raises(KeyboardInterrupt):
        run_smartqsm(sq)
    assert [r["ordinal"] for r in metrics(sq)] == ["1", "2"]

    fake.hook = None
    fake.calls.clear()
    result = run_smartqsm(sq)
    assert fake.ordinals == [3, 4]  # 1 and 2 are collected from their existing .mat
    rows = by_ordinal(metrics(sq))
    assert sorted(rows) == [1, 2, 3, 4] and all(r["status"] == "ok" for r in rows.values())
    assert rows[1]["seconds"] != "" and "reused" in rows[1]["note"]  # timing of the first run kept
    assert result["counts"] == {"ok": 4}


def test_progress_log_has_counter_and_eta(sq, fake, caplog):
    make_split(sq, 3)
    with caplog.at_level(logging.INFO, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq)
    assert re.search(r"\[1/3\] tree 1 ok \d+ s, eta \d+:\d\d min", caplog.text)
    assert re.search(r"\[3/3\] tree 3 ok \d+ s\n", caplog.text + "\n")  # no eta after the last one


@pytest.mark.parametrize("seconds, text", [(9060, "2:31 h"), (3600, "1:00 h"), (245, "4:05 min"), (0, "0:00 min"), (-5, "0:00 min")])
def test_format_eta(seconds, text):
    assert smartqsm.format_eta(seconds) == text


def test_return_value(sq, fake):
    make_split(sq, [{}, {"low_points": True}])
    result = run_smartqsm(sq)
    assert result["dir"] == f"out/r1/smartqsm/{LC}" and result["metrics"] == f"out/r1/smartqsm/{LC}/metrics.csv"
    assert result["config"] == LC and result["n_selected"] == 2
    assert result["gpu"] is False
    assert result["counts"] == {"ok": 1, "skipped_low_points": 1} and result["seconds"] >= 0


# ---------------------------------------------------------------------------------------------------------
# resume / force
# ---------------------------------------------------------------------------------------------------------
def test_resume_reuses_existing_results(sq, fake):
    make_split(sq, 3)
    run_smartqsm(sq)
    first = metrics(sq)
    fake.calls.clear()
    result = run_smartqsm(sq)
    assert fake.calls == []  # nothing re-run
    second = metrics(sq)
    assert [(r["ordinal"], r["status"], r["dbh_m"], r["height_m"], r["seconds"]) for r in second] == [
        (r["ordinal"], r["status"], r["dbh_m"], r["height_m"], r["seconds"]) for r in first
    ]
    assert result["counts"] == {"ok": 3}


def test_resume_runs_only_missing_trees(sq, fake):
    make_split(sq, 3)
    run_smartqsm(sq, trees=[1, 3])
    fake.calls.clear()
    run_smartqsm(sq)
    assert fake.ordinals == [2]
    assert [r["ordinal"] for r in metrics(sq)] == ["1", "2", "3"]


def test_failed_trees_are_retried_on_resume(sq, monkeypatch):
    f = FakeSmartQsm(sq, {2: {"mode": "nomat"}})
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 2)
    run_smartqsm(sq)
    f.calls.clear()
    f.behaviour = {}
    run_smartqsm(sq)
    assert f.ordinals == [2]
    assert [r["status"] for r in metrics(sq)] == ["ok", "ok"]


@pytest.mark.parametrize("kwargs", [{"force": True}, {"resume": False}])
def test_force_and_no_resume_rerun_everything(sq, fake, kwargs):
    make_split(sq, 3)
    run_smartqsm(sq)
    fake.calls.clear()
    run_smartqsm(sq, **kwargs)
    assert fake.ordinals == [1, 2, 3]


def test_stale_mat_is_deleted_before_a_rerun_so_a_failure_is_not_masked(sq, monkeypatch):
    make_split(sq, 2)
    f = FakeSmartQsm(sq)
    monkeypatch.setattr(smartqsm, "run_tool", f)
    run_smartqsm(sq)
    f.behaviour = {1: {"mode": "nomat"}}
    run_smartqsm(sq, force=True)
    assert [r["status"] for r in metrics(sq)] == ["failed", "ok"]
    assert not (sq.stage_dir("smartqsm") / LC / "work" / "tree_0001_qsm.mat").exists()


def test_changed_input_cloud_invalidates_the_old_result(sq, fake):
    split = make_split(sq, 2)
    run_smartqsm(sq)
    fake.calls.clear()
    (split / "trees" / "tree_0002.laz").write_bytes(b"split was re-run with other parameters")
    run_smartqsm(sq)
    assert fake.ordinals == [2]


def test_unreadable_old_result_is_rerun(sq, fake):
    make_split(sq, 2)
    run_smartqsm(sq)
    fake.calls.clear()
    (sq.stage_dir("smartqsm") / LC / "work" / "tree_0001_qsm.mat").write_bytes(b"half a file")
    run_smartqsm(sq)
    assert fake.ordinals == [1]


def test_changed_config_refuses_to_reuse_results_unless_forced(sq, fake):
    make_split(sq, 2)
    run_smartqsm(sq)
    fake.calls.clear()
    with pytest.raises(StageError, match="differs from the config"):
        run_smartqsm(sq, overrides={"voxel_down_size": 0.05})
    assert not fake.calls
    assert yaml.safe_load((sq.stage_dir("smartqsm") / LC / "config.yaml").read_text())["voxel_down_size"] == 0.01  # untouched

    run_smartqsm(sq, overrides={"voxel_down_size": 0.05}, force=True)
    assert fake.ordinals == [1, 2]
    assert yaml.safe_load((sq.stage_dir("smartqsm") / LC / "config.yaml").read_text())["voxel_down_size"] == 0.05


def test_same_overrides_again_resume_fine(sq, fake):
    make_split(sq, 2)
    run_smartqsm(sq, overrides={"voxel_down_size": 0.05})
    fake.calls.clear()
    run_smartqsm(sq, overrides={"voxel_down_size": 0.05})
    assert fake.calls == []


def test_subset_run_keeps_the_rows_of_the_other_trees(sq, monkeypatch):
    f = FakeSmartQsm(sq)
    monkeypatch.setattr(smartqsm, "run_tool", f)
    make_split(sq, 4)
    run_smartqsm(sq)
    first = by_ordinal(metrics(sq))
    f.calls.clear()
    f.behaviour = {2: {"dbh_cm": 33.0}}
    run_smartqsm(sq, trees=[2], force=True)  # re-run ONE tree
    assert f.ordinals == [2]
    rows = by_ordinal(metrics(sq))
    assert sorted(rows) == [1, 2, 3, 4]  # the other rows survive
    assert float(rows[2]["dbh_m"]) == pytest.approx(0.33) and rows[2]["dbh_m"] != first[2]["dbh_m"]
    for ordinal in (1, 3, 4):
        assert rows[ordinal] == first[ordinal]  # byte for byte what the earlier run wrote
    assert [r["ordinal"] for r in metrics(sq)] == ["1", "2", "3", "4"]  # sorted, same columns
    with (sq.stage_dir("smartqsm") / LC / "metrics.csv").open(newline="") as fh:
        assert next(csv.reader(fh)) == EXPECTED_COLUMNS


def test_subset_run_without_force_reuses_the_tree_and_keeps_the_table(sq, fake):
    make_split(sq, 3)
    run_smartqsm(sq)
    first = metrics(sq)
    fake.calls.clear()
    result = run_smartqsm(sq, trees=[2])
    assert fake.calls == [] and result["n_selected"] == 1 and result["counts"] == {"ok": 1}
    after = by_ordinal(metrics(sq))
    assert sorted(after) == [1, 2, 3] and "reused" in after[2]["note"]
    assert after[1] == by_ordinal(first)[1] and after[3] == by_ordinal(first)[3]


def test_subset_runs_accumulate_into_one_sorted_table(sq, fake):
    make_split(sq, 4)
    run_smartqsm(sq, trees=[3])
    assert [r["ordinal"] for r in metrics(sq)] == ["3"]
    run_smartqsm(sq, trees=[1])
    run_smartqsm(sq, trees=[4, 2])
    assert [r["ordinal"] for r in metrics(sq)] == ["1", "2", "3", "4"]
    assert not list((sq.stage_dir("smartqsm") / LC).glob("*.tmp"))


def test_metrics_are_merged_after_every_tree_of_a_subset_run(sq, fake):
    make_split(sq, 4)
    run_smartqsm(sq)
    snapshots = {}

    def hook(ordinal, laz):
        snapshots[ordinal] = [r["ordinal"] for r in metrics(sq)]

    fake.hook = hook
    run_smartqsm(sq, trees=[2, 3], force=True)
    assert snapshots == {2: ["1", "2", "3", "4"], 3: ["1", "2", "3", "4"]}  # never a shortened table in between


def test_fingerprint_is_written_next_to_the_metrics(sq, fake):
    split = make_split(sq, 2)
    run_smartqsm(sq)
    out = sq.stage_dir("smartqsm") / LC
    fingerprint, extra = read_fingerprint(out)
    assert (out / FINGERPRINT_FILE).is_file() and (out / "metrics.csv").is_file()
    assert fingerprint == tree_map_fingerprint(split)
    assert re.fullmatch(r"config=[0-9a-f]{8} skip_low_points=true", extra)
    run_smartqsm(sq, skip_low_points=False, force=True, trees=[1])
    assert read_fingerprint(out)[1].endswith("skip_low_points=false") and read_fingerprint(out)[0] == fingerprint


def test_fingerprint_is_in_place_after_the_first_tree(sq, fake):
    make_split(sq, 3)
    seen = {}

    def hook(ordinal, laz):
        seen[ordinal] = read_fingerprint(sq.stage_dir("smartqsm") / LC)

    fake.hook = hook
    run_smartqsm(sq)
    assert seen[1] is None and seen[2] is not None and seen[3] == seen[2]  # written together with the first metrics.csv


def test_subset_run_after_a_changed_tree_map_drops_the_old_rows_with_a_warning(sq, fake, caplog):
    make_split(sq, 3)
    run_smartqsm(sq)
    old = read_fingerprint(sq.stage_dir("smartqsm") / LC)[0]
    make_split(sq, [{"n_points": 9999}, {}, {}])  # the split stage ran again: tree_map.csv differs
    fake.calls.clear()
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq, trees=[2])
    assert "older rows from a different split were dropped" in caplog.text
    assert [r["ordinal"] for r in metrics(sq)] == ["2"]  # the old rows of 1 and 3 belong to the previous split
    new = read_fingerprint(sq.stage_dir("smartqsm") / LC)[0]
    assert new != old and new == tree_map_fingerprint(sq.stage_dir("split"))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq, trees=[3])  # now consistent again: merge, no warning
    assert "dropped" not in caplog.text and [r["ordinal"] for r in metrics(sq)] == ["2", "3"]


def test_full_run_after_a_changed_split_rewrites_the_table(sq, fake, caplog):
    make_split(sq, 4)
    run_smartqsm(sq)
    make_split(sq, 3, n_points=777)  # 3DFin now finds 3 trees
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq)
    assert "older rows from a different split were dropped" in caplog.text
    assert [r["ordinal"] for r in metrics(sq)] == ["1", "2", "3"]  # no leftover row of ordinal 4


def test_table_without_a_fingerprint_is_not_merged(sq, fake, caplog):
    make_split(sq, 3)
    run_smartqsm(sq)
    (sq.stage_dir("smartqsm") / LC / FINGERPRINT_FILE).unlink()  # a table from before fingerprints existed
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq, trees=[2])
    assert "older rows from a different split were dropped" in caplog.text and "no source fingerprint" in caplog.text
    assert [r["ordinal"] for r in metrics(sq)] == ["2"]
    assert read_fingerprint(sq.stage_dir("smartqsm") / LC) is not None


def test_first_run_has_nothing_to_drop_and_no_warning(sq, fake, caplog):
    make_split(sq, 2)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq, trees=[2])
    assert "dropped" not in caplog.text and [r["ordinal"] for r in metrics(sq)] == ["2"]


def test_forced_config_change_does_not_keep_rows_of_the_old_config(sq, fake, caplog):
    make_split(sq, 3)
    run_smartqsm(sq)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.stages.smartqsm"):
        run_smartqsm(sq, overrides={"voxel_down_size": 0.05}, force=True, trees=[2])  # work/ is discarded
    assert "older rows from a different configuration were dropped" in caplog.text
    assert [r["ordinal"] for r in metrics(sq)] == ["2"]
    caplog.clear()
    run_smartqsm(sq, overrides={"voxel_down_size": 0.05}, trees=[1])  # same (new) config: merge
    assert [r["ordinal"] for r in metrics(sq)] == ["1", "2"]


def test_full_forced_run_after_a_changed_split_still_works(sq, fake):
    make_split(sq, 2)
    run_smartqsm(sq)
    make_split(sq, [{"n_points": 1234}, {}])
    fake.calls.clear()
    run_smartqsm(sq, force=True)
    assert fake.ordinals == [1, 2] and [r["ordinal"] for r in metrics(sq)] == ["1", "2"]


def test_interrupted_subset_run_keeps_the_merged_table(sq, fake):
    make_split(sq, 4)
    run_smartqsm(sq)

    def interrupt(ordinal, laz):
        if ordinal == 3:
            raise KeyboardInterrupt

    fake.hook = interrupt
    with pytest.raises(KeyboardInterrupt):
        run_smartqsm(sq, trees=[2, 3], force=True)
    assert [r["ordinal"] for r in metrics(sq)] == ["1", "2", "3", "4"]  # tree 3 keeps its old row, 2 is the new one


def test_dry_run_with_existing_results_touches_nothing(sq, fake, monkeypatch):
    make_split(sq, 3)
    run_smartqsm(sq)
    monkeypatch.setattr(smartqsm, "run_tool", runner.run_tool)  # the real one only logs in a dry run
    out = sq.stage_dir("smartqsm") / LC
    before = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    make_split(sq, [{"n_points": 4242}, {}, {}])  # even a stale table is only reported on, never changed
    dry = replace(sq, dry_run=True)
    run_smartqsm(dry, trees=[2])
    assert {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


# ---------------------------------------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------------------------------------
def test_dry_run_logs_commands_and_touches_nothing(sq, caplog):
    make_split(sq, [{}, {"low_points": True}, {}])
    dry = replace(sq, dry_run=True)
    with caplog.at_level(logging.INFO):
        result = run_smartqsm(dry, overrides={"voxel_down_size": 0.03})
    assert not (sq.run_dir / "smartqsm").exists()  # no config.yaml, no work dir, no metrics.csv
    text = caplog.text
    out = sq.stage_dir("smartqsm") / LC
    prefix = ["in-env", "smartqsm", "smartqsm-run", "-t", "-y", "-c", str(out / "config.yaml")]
    assert f"[dry-run] would run: {runner.format_command([*prefix, str(out / 'work' / 'tree_0001.laz')])}" in text
    assert str(out / "work" / "tree_0003.laz") in text and "tree_0002.laz" not in text  # low-points tree is not run
    assert "tree 2: skipped" in text
    assert result["dry_run"] is True and result["n_selected"] == 3


def test_dry_run_with_a_gpu_preset_shows_the_commands(sqg, caplog):
    make_split(sqg, [{}, {}])
    dry = replace(sqg, dry_run=True)
    with caplog.at_level(logging.INFO):
        result = run_smartqsm(dry, config=GPU_PRESET)
    out = sqg.stage_dir("smartqsm") / GPU_PRESET
    assert "[dry-run] would run: in-env smartqsm smartqsm-run -t -y -c " in caplog.text
    assert str(out / "work" / "tree_0001.laz") in caplog.text and str(out / "work" / "tree_0002.laz") in caplog.text
    assert result["dry_run"] is True and result["gpu"] is True
    assert not (sqg.run_dir / "smartqsm").exists()


def test_dry_run_without_split_or_presets_only_warns(ctx, caplog):
    """The CLI dry run of the whole pipeline has no split dir and (in tests) no presets."""
    dry = replace(ctx, dry_run=True)
    with caplog.at_level(logging.INFO):
        result = smartqsm.run_smartqsm(dry)
    assert "[dry-run]" in caplog.text and "placeholder tree" in caplog.text and "tree_0001.laz" in caplog.text
    assert result["dry_run"] is True
    assert not ctx.run_dir.exists()


def test_dry_run_without_split_accepts_an_explicit_tree_selection(ctx, caplog):
    """Rehearsing a long run with smartqsm.trees=[...] must not fail just because split has not produced tree_map.csv."""
    dry = replace(ctx, dry_run=True)
    with caplog.at_level(logging.INFO):
        result = smartqsm.run_smartqsm(dry, trees=[3, 14, 29])
    assert result["n_selected"] == 3
    assert all(f"tree_{o:04d}.laz" in caplog.text for o in (3, 14, 29))
    assert not ctx.run_dir.exists()

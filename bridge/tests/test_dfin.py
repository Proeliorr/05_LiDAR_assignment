from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

import pytest

from lidar_bridge.paths import RunContext
from lidar_bridge.runner import ToolError
from lidar_bridge.stages import dfin
from lidar_bridge.stages._common import StageError


def read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(path, encoding="utf-8")
    return parser


class FakeTool:
    """Replaces run_tool in the dfin module; optionally 'produces' the 3DFin outputs."""

    def __init__(self, ctx, produce=("table", "tree_las", "stripe_las"), export_txt=False, returncode=0, stem="plot_01"):
        self.ctx, self.produce, self.export_txt, self.returncode, self.stem = ctx, produce, export_txt, returncode, stem
        self.calls: list[dict] = []

    def __call__(self, env, args, **kw):
        self.calls.append(dict(env=env, args=list(args), **kw))
        out = self.ctx.stage_dir("dfin")
        self.ini_at_call = read_ini(out / "params.ini") if (out / "params.ini").exists() else None
        names = dfin.expected_outputs(self.stem, self.export_txt)
        for key in self.produce:
            (out / names[key]).write_bytes(b"data")
        if self.returncode:
            raise ToolError("3dfin exited with code 1", returncode=self.returncode)
        return subprocess.CompletedProcess(["in-env"], 0, stdout="3DFin output line", stderr=None)


@pytest.fixture
def fake(monkeypatch, ctx):
    f = FakeTool(ctx)
    monkeypatch.setattr(dfin, "run_tool", f)
    return f


# -- params.ini ---------------------------------------------------------------------------------------------
def test_params_ini_is_template_copy_with_case_preserved(project):
    parser = dfin.build_params(project)
    out = dfin.write_params_ini(parser, project / "p.ini")
    written = read_ini(out)
    assert written.sections() == ["basic", "advanced", "expert"]
    assert written["basic"]["z0_name"] == "Z0" and written["expert"]["number_of_points"] == "1000"
    assert "maximum_height" in written["advanced"]
    assert "\r" not in out.read_text()


def test_ini_overrides_nested_dotted_and_bare(project):
    parser = dfin.build_params(
        project,
        {"basic": {"upper_limit": 4.0}, "advanced.maximum_height": 30, "minimum_points": 25, "basic.z0_name": "z"},
    )
    assert parser["basic"]["upper_limit"] == "4.0"
    assert parser["advanced"]["maximum_height"] == "30"
    assert parser["expert"]["minimum_points"] == "25"
    assert parser["basic"]["z0_name"] == "z"
    assert parser["basic"]["lower_limit"] == "0.7"  # untouched


def test_ini_override_bool_is_lowercase(project):
    parser = dfin.build_params(project, {"basic.upper_limit": True})
    assert parser["basic"]["upper_limit"] == "true"


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"basic": {"upper_limt": 1}}, "unknown key 'upper_limt'"),
        ({"nosuch": {"a": 1}}, "unknown 3DFin section"),
        ({"nosuch.a": 1}, "unknown 3DFin section"),
        ({"typo": 1}, "unknown 3DFin parameter"),
        ({"dup": 1}, "ambiguous"),
        ({"basic": {"upper_limit": {"x": 1}}}, "scalar"),
    ],
)
def test_ini_override_errors(project, bad, message):
    with pytest.raises(StageError, match=message):
        dfin.build_params(project, bad)


def test_missing_template_is_a_clear_error(tmp_path):
    with pytest.raises(StageError, match="submodule"):
        dfin.load_template(tmp_path)


def test_real_3dfin_template_parses_and_has_no_ambiguous_keys():
    root = Path(__file__).resolve().parents[2]
    template = root / dfin.TEMPLATE_INI
    if not template.is_file():
        pytest.skip("3DFin submodule not mounted")
    parser = dfin.load_template(root)
    keys = [k for s in dfin.INI_SECTIONS for k in parser[s]]
    assert len(keys) == len(set(keys)), "bare-key overrides would be ambiguous"
    assert {"upper_limit", "lower_limit", "maximum_height", "number_of_points"} <= set(keys)
    flat = dfin.flatten_ini_overrides(parser, {"upper_limit": 4.5})
    assert flat == {("basic", "upper_limit"): "4.5"}


# -- the stage ------------------------------------------------------------------------------------------------
def test_run_dfin_builds_documented_command(ctx, fake):
    result = dfin.run_dfin(ctx, normalize=True, denoise=True, export_txt=False)
    (call,) = fake.calls
    data = ctx.data_dir
    assert call["env"] == "3dfin"
    assert call["args"] == [
        "3DFin",
        "cli",
        str(data / "in/plot_01.laz"),
        str(data / "out/r1/dfin"),
        str(data / "out/r1/dfin/params.ini"),
        "--normalize",
        "--denoise",
    ]
    assert call["dry_run"] is False
    assert result["table"] == "out/r1/dfin/plot_01.xlsx"
    assert result["tree_las"] == "out/r1/dfin/plot_01_tree_ID_dist_axes.las"
    assert result["stripe_las"] == "out/r1/dfin/plot_01_stripe.las"
    assert result["stem"] == "plot_01"


def test_flags_follow_parameters(ctx, monkeypatch):
    f = FakeTool(ctx, export_txt=True)
    monkeypatch.setattr(dfin, "run_tool", f)
    result = dfin.run_dfin(ctx, normalize=False, denoise=False, export_txt=True)
    assert f.calls[0]["args"][5:] == ["--export_txt"]
    assert result["table"].endswith("plot_01_dbh_and_heights.txt")


def test_params_ini_written_with_overrides_before_run(ctx, fake):
    dfin.run_dfin(ctx, ini_overrides={"basic": {"upper_limit": 4.2}, "expert.minimum_points": 30})
    ini = fake.ini_at_call  # params.ini as it was when the tool was started
    assert ini["basic"]["upper_limit"] == "4.2" and ini["expert"]["minimum_points"] == "30"
    assert ini["basic"]["lower_limit"] == "0.7"


def test_stale_outputs_are_wiped_before_run(ctx, project, monkeypatch):
    out = ctx.stage_dir("dfin")
    out.mkdir(parents=True)
    (out / "plot_01.xlsx").write_text("stale")
    (out / "plot_01_tree_ID_dist_axes.las").write_text("stale")
    (out / "other_run_leftover.las").write_text("stale")
    f = FakeTool(ctx, produce=())  # the "tool" produces nothing
    monkeypatch.setattr(dfin, "run_tool", f)
    with pytest.raises(StageError, match="missing plot_01.xlsx"):  # stale xlsx must not satisfy the check
        dfin.run_dfin(ctx)
    assert sorted(p.name for p in out.iterdir()) == ["params.ini"]


def test_missing_outputs_fail_even_with_exit_code_zero(ctx, monkeypatch):
    f = FakeTool(ctx, produce=("table", "stripe_las"))  # no tree_ID_dist_axes.las
    monkeypatch.setattr(dfin, "run_tool", f)
    with pytest.raises(StageError) as err:
        dfin.run_dfin(ctx)
    assert "missing plot_01_tree_ID_dist_axes.las" in str(err.value)
    assert "3DFin output line" in str(err.value)  # tool output tail is included


def test_empty_output_file_fails(ctx, monkeypatch):
    class Empty(FakeTool):
        def __call__(self, *a, **k):
            r = super().__call__(*a, **k)
            (self.ctx.stage_dir("dfin") / "plot_01.xlsx").write_bytes(b"")
            return r

    monkeypatch.setattr(dfin, "run_tool", Empty(ctx))
    with pytest.raises(StageError, match="empty plot_01.xlsx"):
        dfin.run_dfin(ctx)


def test_tool_failure_propagates(ctx, monkeypatch):
    monkeypatch.setattr(dfin, "run_tool", FakeTool(ctx, produce=(), returncode=1))
    with pytest.raises(ToolError):
        dfin.run_dfin(ctx)


def test_input_requirements(project, monkeypatch, fake):
    base = dict(run_name="r1", data_dir=project / "data", install_dir=project)
    with pytest.raises(StageError, match="needs an input"):
        dfin.run_dfin(RunContext(**base))
    with pytest.raises(StageError, match="not found"):
        dfin.run_dfin(RunContext(**base, input=project / "data/in/missing.laz"))
    (project / "data/in/notes.txt").write_text("x")
    with pytest.raises(StageError, match=r"\.las/\.laz"):
        dfin.run_dfin(RunContext(**base, input=project / "data/in/notes.txt"))
    assert not fake.calls


def test_input_outside_data_is_rejected_by_context(project):
    from lidar_bridge.paths import PathOutsideDataError

    ctx = RunContext(run_name="r1", data_dir=project / "data", install_dir=project)
    with pytest.raises(PathOutsideDataError, match="outside the mounted data folder"):
        ctx.resolve("/etc/passwd.laz")
    with pytest.raises(PathOutsideDataError):
        ctx.resolve("../elsewhere/a.laz")


def test_dry_run_has_no_side_effects_and_logs_command(project, monkeypatch, caplog):
    import logging

    ctx = RunContext(
        run_name="r1",
        data_dir=project / "data",
        install_dir=project,
        input=project / "data/in/plot_01.laz",
        dry_run=True,
    )
    with caplog.at_level(logging.INFO):
        result = dfin.run_dfin(ctx, denoise=True)
    assert not ctx.run_dir.exists()  # nothing created: no dir, no params.ini
    text = caplog.text
    assert "[dry-run] would run: in-env 3dfin 3DFin cli" in text
    assert "--normalize --denoise" in text
    assert result["dry_run"] is True


def test_expected_outputs():
    assert dfin.expected_outputs("p", False) == {
        "table": "p.xlsx",
        "tree_las": "p_tree_ID_dist_axes.las",
        "stripe_las": "p_stripe.las",
    }
    assert dfin.expected_outputs("p", True)["table"] == "p_dbh_and_heights.txt"

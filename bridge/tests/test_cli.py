"""Tests of the command line (``lidar_bridge.cli``): argument handling, ``--params`` file, GPU check, stage binding."""

from __future__ import annotations

import functools
import re
import signal
from pathlib import Path

import pytest

from lidar_bridge import cli
from lidar_bridge.pipeline import PipelineError
from lidar_bridge.stages import STAGE_FUNCTIONS, STAGE_ORDER
from lidar_bridge.stages.smartqsm import DEFAULT_CONFIG

GPU_PRESET = "spconv-contraction-LEAFON-GPU"
CPU_PRESET = "space-colonization"


class PipelineSpy:
    """Replaces ``cli.run_pipeline``; keeps the call so that a test can look at the bound stage functions."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, input, reference, run_name, stages, dry_run, stage_fns):  # noqa: A002
        self.calls.append(
            dict(input=input, reference=reference, run_name=run_name, stages=list(stages), dry_run=dry_run, stage_fns=dict(stage_fns))
        )

    @property
    def call(self) -> dict:
        (call,) = self.calls
        return call

    def keywords(self, stage: str) -> dict:
        return dict(self.call["stage_fns"][stage].keywords)


@pytest.fixture(autouse=True)
def restore_signals():
    """``main`` installs SIGTERM / SIGHUP handlers; put the test process back as it was."""
    names = [name for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)]
    saved = {name: signal.getsignal(getattr(signal, name)) for name in names}
    yield
    for name, handler in saved.items():
        signal.signal(getattr(signal, name), handler)


@pytest.fixture
def spy(monkeypatch) -> PipelineSpy:
    spy = PipelineSpy()
    monkeypatch.setattr(cli, "run_pipeline", spy)
    monkeypatch.setattr(cli.runner, "gpu_visible", lambda: True)
    return spy


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("LIDAR_DATA_DIR", str(data))
    return data


def exits_with(code: int, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == code


# -- input and run name -----------------------------------------------------------------------------------------------
def test_no_input_and_no_run_is_a_usage_error(spy, capsys):
    exits_with(2, [])
    assert spy.calls == []
    assert "give an input point cloud, or --run" in capsys.readouterr().err
    exits_with(2, ["--stages", "compare", "--ref", "in/ref.csv"])
    assert spy.calls == []


def test_the_run_name_defaults_to_the_input_stem(spy):
    cli.main(["in/plot_01.laz"])
    assert spy.call["run_name"] == "plot_01" and spy.call["input"] == "in/plot_01.laz"


@pytest.mark.parametrize("given", ["in/sub/Plot.A.las", "in\\sub\\Plot.A.las"])
def test_the_run_name_is_the_stem_for_posix_and_windows_spellings(spy, given):
    cli.main([given])
    assert spy.call["run_name"] == "Plot.A"


def test_an_explicit_run_name_wins_and_needs_no_input(spy):
    cli.main(["in/plot_01.laz", "--run", "mine"])
    assert spy.call["run_name"] == "mine"
    spy.calls.clear()
    cli.main(["--run", "earlier", "--stages", "compare", "--ref", "in/ref.csv"])
    assert spy.call["input"] is None and spy.call["run_name"] == "earlier" and spy.call["reference"] == "in/ref.csv"


def test_defaults_are_all_stages_no_reference_and_no_dry_run(spy):
    cli.main(["in/plot_01.laz"])
    assert spy.call["stages"] == list(STAGE_ORDER)
    assert spy.call["reference"] is None and spy.call["dry_run"] is False


def test_dry_run_and_reference_are_passed_on(spy):
    cli.main(["in/plot_01.laz", "--ref", "in/ref.xlsx", "--dry-run"])
    assert spy.call["dry_run"] is True and spy.call["reference"] == "in/ref.xlsx"


# -- stages -----------------------------------------------------------------------------------------------------------
def test_stages_are_selected_in_the_fixed_order(spy):
    cli.main(["in/plot_01.laz", "--stages", "compare,smartqsm"])
    assert spy.call["stages"] == ["smartqsm", "compare"]


def test_an_unknown_stages_value_is_a_usage_error(spy, capsys):
    exits_with(2, ["in/plot_01.laz", "--stages", "dfin,qsm"])
    assert spy.calls == []
    assert "unknown stage" in capsys.readouterr().err


def test_main_binds_the_stage_functions_and_passes_them_to_run_pipeline(spy):
    cli.main(["in/plot_01.laz"])
    fns = spy.call["stage_fns"]
    assert list(fns) == list(STAGE_ORDER)
    for name in STAGE_ORDER:
        assert isinstance(fns[name], functools.partial)
        assert fns[name].func is STAGE_FUNCTIONS[name]
    assert all(spy.keywords(name) == {} for name in STAGE_ORDER)  # nothing set: the stages use their own defaults


# -- flags --------------------------------------------------------------------------------------------------------------
def test_trees_limit_and_force_reach_smartqsm_and_arbor(spy):
    cli.main(["in/plot_01.laz", "--trees", "3,8", "--limit", "2", "--force"])
    for stage in ("smartqsm", "arbor"):
        assert spy.keywords(stage) == {"trees": [3, 8], "limit": 2, "force": True}
    for stage in ("dfin", "split", "compare"):
        assert spy.keywords(stage) == {}


def test_flags_that_are_not_given_are_not_passed(spy):
    cli.main(["in/plot_01.laz", "--limit", "4"])
    assert spy.keywords("smartqsm") == {"limit": 4} and spy.keywords("arbor") == {"limit": 4}


def test_smartqsm_config_reaches_smartqsm_only(spy):
    cli.main(["in/plot_01.laz", "--smartqsm-config", CPU_PRESET])
    assert spy.keywords("smartqsm") == {"config": CPU_PRESET}
    for stage in ("dfin", "split", "arbor", "compare"):
        assert "config" not in spy.keywords(stage)


@pytest.mark.parametrize("text, expected", [("3,8", [3, 8]), ("5", [5]), (" 1 , 2 ,", [1, 2]), ("", [])])
def test_parse_trees(text, expected):
    assert cli.parse_trees(text) == expected


def test_parse_trees_rejects_non_numbers():
    with pytest.raises(ValueError, match="--trees"):
        cli.parse_trees("3,x")


def test_bad_trees_is_a_usage_error(spy):
    exits_with(2, ["in/plot_01.laz", "--trees", "a,b"])
    assert spy.calls == []


def test_limit_must_be_a_number(spy):
    exits_with(2, ["in/plot_01.laz", "--limit", "many"])
    assert spy.calls == []


# -- --params file ------------------------------------------------------------------------------------------------------
def test_params_file_values_reach_their_stage(spy, data_dir):
    (data_dir / "params.yaml").write_text(
        "dfin: {denoise: true}\nsplit: {}\nsmartqsm: {timeout_s: 99, overrides: {voxel_down_size: 0.03}}\n"
        "arbor: {threads: 4}\ncompare: {plots: false}\n",
        encoding="utf-8",
    )
    cli.main(["in/plot_01.laz", "--params", "params.yaml"])
    assert spy.keywords("dfin") == {"denoise": True}
    assert spy.keywords("smartqsm") == {"timeout_s": 99, "overrides": {"voxel_down_size": 0.03}}
    assert spy.keywords("arbor") == {"threads": 4}
    assert spy.keywords("compare") == {"plots": False}


def test_a_flag_beats_the_params_file(spy, data_dir):
    (data_dir / "params.yaml").write_text(
        f"smartqsm: {{config: {GPU_PRESET}, limit: 9, trees: [1, 2], timeout_s: 10}}\narbor: {{limit: 9, threads: 2, force: false}}\n",
        encoding="utf-8",
    )
    cli.main(["in/plot_01.laz", "--params", "params.yaml", "--limit", "2", "--trees", "7", "--smartqsm-config", CPU_PRESET, "--force"])
    assert spy.keywords("smartqsm") == {"config": CPU_PRESET, "limit": 2, "trees": [7], "timeout_s": 10, "force": True}
    assert spy.keywords("arbor") == {"limit": 2, "trees": [7], "threads": 2, "force": True}  # the rest of the file still applies


def test_params_file_values_survive_when_no_flag_is_given(spy, data_dir):
    (data_dir / "params.yaml").write_text("smartqsm: {limit: 5, config: space-colonization}\n", encoding="utf-8")
    cli.main(["in/plot_01.laz", "--params", "params.yaml"])
    assert spy.keywords("smartqsm") == {"limit": 5, "config": "space-colonization"}


def test_an_empty_params_file_is_fine(spy, data_dir):
    (data_dir / "params.yaml").write_text("", encoding="utf-8")
    cli.main(["in/plot_01.laz", "--params", "params.yaml"])
    assert all(spy.keywords(name) == {} for name in STAGE_ORDER)


@pytest.mark.parametrize(
    "text, message",
    [
        ("qsm: {a: 1}\n", "unknown stage 'qsm'"),
        ("smartqsm: {nope: 1}\n", r"unknown parameter\(s\) \['nope'\] of stage smartqsm"),
        ("arbor: {trees: [1], bogus: 2}\n", "bogus"),
        ("dfin: [1, 2]\n", "must be a mapping"),
        ("- dfin\n- split\n", "mapping of stage names"),
    ],
)
def test_an_invalid_params_file_is_a_usage_error(spy, data_dir, capsys, text, message):
    (data_dir / "params.yaml").write_text(text, encoding="utf-8")
    exits_with(2, ["in/plot_01.laz", "--params", "params.yaml"])
    assert spy.calls == []
    assert re.search(message, capsys.readouterr().err)


@pytest.mark.parametrize(
    "text, message",
    [
        ("qsm: {a: 1}\n", "unknown stage 'qsm'"),
        ("smartqsm: {nope: 1}\n", r"unknown parameter\(s\) \['nope'\] of stage smartqsm"),
        ("dfin: [1, 2]\n", "must be a mapping"),
        ("- dfin\n", "mapping of stage names"),
    ],
)
def test_load_params_messages(tmp_path, text, message):
    path = tmp_path / "p.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        cli.load_params(path)


def test_load_params_accepts_every_parameter_of_every_stage(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("smartqsm: {resume: false, skip_low_points: false}\ncompare: {allow_stale: true}\narbor: {mode: plot}\n", encoding="utf-8")
    assert cli.load_params(path) == {
        "smartqsm": {"resume": False, "skip_low_points": False},
        "compare": {"allow_stale": True},
        "arbor": {"mode": "plot"},
    }


def test_a_missing_params_file_is_a_usage_error(spy, data_dir):
    exits_with(2, ["in/plot_01.laz", "--params", "nope.yaml"])
    assert spy.calls == []


# -- GPU check ----------------------------------------------------------------------------------------------------------
def no_gpu(monkeypatch):
    monkeypatch.setattr(cli.runner, "gpu_visible", lambda: False)


def test_check_gpu_raises_for_a_gpu_preset_without_a_gpu(monkeypatch):
    no_gpu(monkeypatch)
    with pytest.raises(PipelineError, match=rf"{GPU_PRESET}.*needs a GPU.*--gpus all.*--smartqsm-config"):
        cli.check_gpu({"smartqsm": {"config": GPU_PRESET}}, ["smartqsm"])


def test_check_gpu_covers_the_default_preset(monkeypatch):
    assert DEFAULT_CONFIG.endswith("-GPU")
    no_gpu(monkeypatch)
    with pytest.raises(PipelineError, match=DEFAULT_CONFIG):
        cli.check_gpu({"smartqsm": {}}, list(STAGE_ORDER))


def test_check_gpu_is_quiet_with_a_gpu(monkeypatch):
    monkeypatch.setattr(cli.runner, "gpu_visible", lambda: True)
    cli.check_gpu({"smartqsm": {"config": GPU_PRESET}}, ["smartqsm"])
    cli.check_gpu({"smartqsm": {}}, ["smartqsm"])


def test_check_gpu_is_quiet_for_a_cpu_preset(monkeypatch):
    no_gpu(monkeypatch)
    cli.check_gpu({"smartqsm": {"config": CPU_PRESET}}, ["smartqsm"])
    cli.check_gpu({"smartqsm": {"config": "spconv-contraction-LEAFON-cpu"}}, ["smartqsm"])  # only the -GPU suffix counts


def test_check_gpu_is_quiet_when_smartqsm_is_not_selected(monkeypatch):
    no_gpu(monkeypatch)
    cli.check_gpu({"smartqsm": {"config": GPU_PRESET}}, ["dfin", "split", "arbor", "compare"])
    cli.check_gpu({"smartqsm": {}}, [])


def test_check_gpu_does_not_ask_for_the_gpu_when_it_is_not_needed(monkeypatch):
    def fail():
        raise AssertionError("nvidia-smi must not be run")

    monkeypatch.setattr(cli.runner, "gpu_visible", fail)
    cli.check_gpu({"smartqsm": {"config": CPU_PRESET}}, ["smartqsm"])
    cli.check_gpu({"smartqsm": {}}, ["arbor"])


def test_main_stops_before_the_first_stage_when_a_gpu_preset_has_no_gpu(spy, monkeypatch, caplog):
    no_gpu(monkeypatch)
    exits_with(1, ["in/plot_01.laz", "--smartqsm-config", GPU_PRESET])
    assert spy.calls == []
    assert "needs a GPU" in caplog.text


def test_main_default_config_needs_a_gpu(spy, monkeypatch):
    no_gpu(monkeypatch)
    exits_with(1, ["in/plot_01.laz"])
    assert spy.calls == []


def test_main_runs_a_cpu_preset_without_a_gpu(spy, monkeypatch):
    no_gpu(monkeypatch)
    cli.main(["in/plot_01.laz", "--smartqsm-config", CPU_PRESET])
    assert spy.keywords("smartqsm") == {"config": CPU_PRESET}


def test_main_without_smartqsm_needs_no_gpu(spy, monkeypatch):
    no_gpu(monkeypatch)
    cli.main(["in/plot_01.laz", "--stages", "dfin,split,arbor"])
    assert spy.call["stages"] == ["dfin", "split", "arbor"]


def test_a_dry_run_skips_the_gpu_check(spy, monkeypatch):
    no_gpu(monkeypatch)
    cli.main(["in/plot_01.laz", "--dry-run"])
    assert spy.call["dry_run"] is True


def test_the_gpu_check_uses_the_config_from_the_params_file(spy, data_dir, monkeypatch):
    no_gpu(monkeypatch)
    (data_dir / "params.yaml").write_text(f"smartqsm: {{config: {CPU_PRESET}}}\n", encoding="utf-8")
    cli.main(["in/plot_01.laz", "--params", "params.yaml"])  # a CPU preset from the file: no GPU needed
    assert spy.keywords("smartqsm") == {"config": CPU_PRESET}
    spy.calls.clear()
    exits_with(1, ["in/plot_01.laz", "--params", "params.yaml", "--smartqsm-config", GPU_PRESET])  # the flag wins, and needs a GPU
    assert spy.calls == []


# -- helpers --------------------------------------------------------------------------------------------------------------
def test_stage_params_starts_with_an_empty_dict_per_stage():
    args = cli.build_parser().parse_args(["in/a.laz"])
    params = cli.stage_params(args, {})
    assert params == {name: {} for name in STAGE_ORDER}


def test_stage_params_does_not_change_the_loaded_file_content():
    from_file = {"smartqsm": {"limit": 5}}
    args = cli.build_parser().parse_args(["in/a.laz", "--limit", "2"])
    params = cli.stage_params(args, from_file)
    assert params["smartqsm"]["limit"] == 2 and from_file == {"smartqsm": {"limit": 5}}


def test_data_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LIDAR_DATA_DIR", str(tmp_path))
    assert cli.data_dir_from_env() == tmp_path
    monkeypatch.delenv("LIDAR_DATA_DIR")
    assert cli.data_dir_from_env() == Path("/data")

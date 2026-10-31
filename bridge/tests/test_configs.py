"""Config composition with hydra's compose API, and a few end-to-end CLI dry-runs in a throw-away project."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate
from omegaconf import OmegaConf

from lidar_bridge.configs import CONFIG_NAME, register_configs
from lidar_bridge.stages import STAGE_FUNCTIONS, STAGE_ORDER

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def cfg_factory(tmp_path):
    register_configs()
    GlobalHydra.instance().clear()

    def make(*overrides: str, hydra_node: bool = False):
        with initialize_config_dir(config_dir=str(tmp_path), version_base="1.3"):
            return compose(config_name=CONFIG_NAME, overrides=list(overrides), return_hydra_config=hydra_node)

    yield make
    GlobalHydra.instance().clear()


def test_defaults(cfg_factory):
    cfg = cfg_factory()
    assert list(cfg.stages) == list(STAGE_ORDER)
    assert cfg.dry_run is False and cfg.input is None and cfg.reference is None
    assert cfg.run_name == "run"  # no input -> generic default
    for stage in STAGE_ORDER:  # one config group per stage, targeting the stage function
        assert cfg[stage]._target_ == f"{STAGE_FUNCTIONS[stage].__module__}.{STAGE_FUNCTIONS[stage].__name__}"
        assert cfg[stage]._partial_ is True
        assert "ctx" not in cfg[stage]  # supplied at call time
    assert cfg.split.max_axis_dist == 3.0 and cfg.split.min_z0 == 0.10 and cfg.split.min_points == 500
    assert cfg.split.emit_ground_variant is False
    assert cfg.smartqsm.config == "layerwise-clustering" and cfg.smartqsm.image == "lidar-smartqsm"
    assert cfg.smartqsm.gpu is False and "timeout_s" in cfg.smartqsm
    assert cfg.arbor.mode == "per_tree" and cfg.arbor.image == "lidar-arbor"
    assert cfg.dfin.normalize is True and cfg.dfin.denoise is False and cfg.dfin.export_txt is False
    assert cfg.dfin.image == "lidar-3dfin" and cfg.dfin.ini_overrides == {}


def test_hydra_output_settings_live_inside_run_dir(cfg_factory):
    cfg = cfg_factory("run_name=r1", hydra_node=True)
    node = OmegaConf.to_container(cfg.hydra, resolve=False)
    assert node["job"]["chdir"] is False and node["job"]["name"] == "lidar_bridge"
    assert node["run"]["dir"] == "data/out/${run_name}/.hydra"
    assert node["sweep"]["dir"].startswith("data/out/${run_name}/.hydra")
    assert node["output_subdir"] == "."  # config.yaml / overrides.yaml directly in .hydra/


def test_input_reference_and_derived_run_name(cfg_factory):
    cfg = cfg_factory("input=data/in/x.laz", "reference=data/in/ref.csv")
    assert (cfg.input, cfg.reference) == ("data/in/x.laz", "data/in/ref.csv")
    assert cfg.run_name == "x"  # defaults to the input stem
    assert cfg_factory("input=data/in/x.laz", "run_name=mine").run_name == "mine"


def test_stage_selection_and_run_name(cfg_factory):
    cfg = cfg_factory("stages=[smartqsm,compare]", "run_name=r1")
    assert list(cfg.stages) == ["smartqsm", "compare"] and cfg.run_name == "r1"


def test_parameter_overrides_reach_the_partials(cfg_factory):
    cfg = cfg_factory("dfin.normalize=false", "dfin.denoise=true", "split.max_axis_dist=2.5", "split.min_points=300")
    dfin_partial, split_partial = instantiate(cfg.dfin), instantiate(cfg.split)
    assert dfin_partial.func is STAGE_FUNCTIONS["dfin"]
    assert dfin_partial.keywords["normalize"] is False and dfin_partial.keywords["denoise"] is True
    assert split_partial.keywords == dict(max_axis_dist=2.5, min_z0=0.10, min_points=300, emit_ground_variant=False)


def test_ini_overrides_from_cli_become_plain_nested_dict(cfg_factory):
    cfg = cfg_factory("+dfin.ini_overrides.basic.upper_limit=4.0", "+dfin.ini_overrides.expert.minimum_points=30")
    overrides = instantiate(cfg.dfin).keywords["ini_overrides"]
    assert overrides == {"basic": {"upper_limit": 4.0}, "expert": {"minimum_points": 30}}
    assert type(overrides) is dict and type(overrides["basic"]) is dict  # not DictConfig (hydra_convert=all)


@pytest.mark.parametrize("preset", ["layerwise-clustering", "space-colonization"])
def test_smartqsm_config_sweep_values(cfg_factory, preset):
    cfg = cfg_factory(f"smartqsm.config={preset}", "smartqsm.gpu=true", "smartqsm.timeout_s=60")
    kw = instantiate(cfg.smartqsm).keywords
    assert kw["config"] == preset and kw["gpu"] is True and kw["timeout_s"] == 60


def test_wrong_types_are_rejected(cfg_factory):
    with pytest.raises(Exception, match="Error merging override split.max_axis_dist=abc"):
        cfg_factory("split.max_axis_dist=abc")


def test_unknown_key_is_rejected_without_plus(cfg_factory):
    with pytest.raises(Exception, match="(?i)(not in|could not override)"):
        cfg_factory("dfin.no_such_param=1")


# -- end to end through the real CLI entry point (dry run, in a temp project) -----------------------------------------
def run_cli(project: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "HOST_PROJECT_DIR": "D:/Podyplomowe/05_Proximity_Lidar",
        "LIDAR_PROJECT_DIR": str(project),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "lidar_bridge", *args], cwd=project, env=env, capture_output=True, text=True, timeout=120
    )


def test_cli_dry_run_full_pipeline(project):
    proc = run_cli(project, "input=data/in/plot_01.laz", "dry_run=true", "dfin.denoise=true", "+dfin.ini_overrides.basic.upper_limit=4.0")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "docker run --rm --network none --mount type=bind,source=D:/Podyplomowe/05_Proximity_Lidar/data,target=/data" in out
    assert "lidar-3dfin cli /data/in/plot_01.laz /data/out/plot_01/dfin /data/out/plot_01/dfin/params.ini --normalize --denoise" in out
    for stage in STAGE_ORDER:
        if stage == "compare":  # no reference table yet (it is attached later): compare is skipped, the run still succeeds
            assert "stage compare" not in out and "no reference table given: skipping the compare stage" in out
        else:
            assert f"stage {stage}: done" in out
    # hydra snapshot + log are inside the run directory, nothing else was created
    hydra_dir = project / "data/out/plot_01/.hydra"
    assert {"config.yaml", "overrides.yaml", "lidar_bridge.log"} <= {p.name for p in hydra_dir.iterdir()}
    assert not (project / "data/out/plot_01/dfin").exists()
    assert "upper_limit: 4.0" in (hydra_dir / "config.yaml").read_text()


def test_cli_stage_selection(project):
    proc = run_cli(
        project, "stages=[compare,smartqsm]", "run_name=r1", "reference=data/in/ref.csv", "dry_run=true", "split.max_axis_dist=2.5"
    )  # fmt: skip
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert out.index("stage smartqsm: start") < out.index("stage compare: start")
    assert "stage dfin" not in out and "stage split" not in out and "stage arbor" not in out


def test_cli_sweep_over_smartqsm_config_shares_one_run(project):
    # the documented sweep: dfin + split ran earlier as r1; the jobs only repeat smartqsm (+ compare) under the SAME run_name
    proc = run_cli(
        project, "-m", "smartqsm.config=layerwise-clustering,space-colonization", "run_name=r1",
        "stages=[smartqsm,compare]", "dry_run=true",
    )  # fmt: skip
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "Launching 2 jobs" in out
    assert "'config': 'layerwise-clustering'" in out and "'config': 'space-colonization'" in out
    assert out.count("stage smartqsm: done") == 2 and "stage dfin" not in out and "stage split" not in out
    # one run directory; each job keeps its own hydra snapshot below .hydra/multirun/<job number>
    assert not list((project / "data/out").glob("r1_*"))
    for job in ("0", "1"):
        assert (project / "data/out/r1/.hydra/multirun" / job / "config.yaml").is_file()


def test_sweep_advice_does_not_recommend_a_run_name_per_job():
    from lidar_bridge import configs

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for text in (configs.__doc__, configs.HELP_HEADER, readme):
        assert "r1_${smartqsm.config}" not in text and "its own run_name" not in text
        assert "smartqsm.config=layerwise-clustering,space-colonization" in text
    for text in (configs.HELP_HEADER, readme):
        assert "SAME run_name" in text or "same run_name" in text or "same name" in text


def test_cli_failing_stage_is_a_one_line_error_when_not_dry_run(project):
    # split needs the dfin outputs of the same run; none exist in the temp project, so it must fail cleanly
    proc = run_cli(project, "stages=[split]", "input=data/in/plot_01.laz")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "run the dfin stage first" in out and "stage 'split' failed" in out and "Traceback" not in out


def test_cli_bad_input_path_is_a_one_line_error(project):
    proc = run_cli(project, "input=/etc/passwd", "stages=[split]", "dry_run=true")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "outside the project" in out and "Traceback" not in out


def test_cli_help(project):
    proc = run_cli(project, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "== Configuration groups ==" in proc.stdout and "dfin: default" in proc.stdout and "max_axis_dist" in proc.stdout
    assert "SAME run_name" in proc.stdout  # sweep advice shown in the help


# -- interrupts through the real entry point ------------------------------------------------------------------------------
PROBE = """import signal, sys, time
signal.signal(signal.SIGINT, signal.default_int_handler)  # in case the parent started us with SIGINT ignored
from lidar_bridge.configs import main
from lidar_bridge.stages import STAGE_FUNCTIONS


def run_arbor(ctx, *, mode: str = "per_tree"):
    print("PROBE-READY", flush=True)
    time.sleep(120)


STAGE_FUNCTIONS["arbor"] = run_arbor
main()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_cli_interrupt_ends_with_130_one_line_and_no_traceback(project, tmp_path, sig):
    """SIGTERM (docker stop) and Ctrl+C reach hydra's task function as KeyboardInterrupt and are not swallowed by hydra."""
    script = tmp_path / "probe_main.py"
    script.write_text(PROBE)
    env = {**os.environ, "PYTHONPATH": str(SRC), "HOST_PROJECT_DIR": "D:/Podyplomowe/05_Proximity_Lidar",
           "LIDAR_PROJECT_DIR": str(project), "PYTHONDONTWRITEBYTECODE": "1"}  # fmt: skip
    proc = subprocess.Popen(
        [sys.executable, str(script), "stages=[arbor]", "run_name=r1", "dry_run=true"],
        cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    ready = threading.Event()
    lines: list[str] = []

    def pump() -> None:
        for line in proc.stdout:
            lines.append(line)
            if "PROBE-READY" in line:
                ready.set()

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        assert ready.wait(60), "".join(lines)
        proc.send_signal(sig)
        code = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        reader.join(timeout=5)
    out = "".join(lines)
    assert code == 130, out
    assert f"interrupted ({sig.name}" in out or (sig == signal.SIGINT and "interrupted (Ctrl+C)" in out), out
    assert "Traceback" not in out and "KeyboardInterrupt" not in out, out
    assert "arbor    ABORTED" in out  # summary still written

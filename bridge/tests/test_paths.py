from __future__ import annotations

from pathlib import Path

import pytest

from lidar_bridge.docker_runner import Mount, host_path
from lidar_bridge.paths import PathOutsideProjectError, RunContext

HOST = "D:/Podyplomowe/05_Proximity_Lidar"


def make(project_dir="/workspace", host=HOST, **kw) -> RunContext:
    return RunContext(run_name="r1", host_project_dir=host, project_dir=Path(project_dir), **kw)


def test_host_maps_workspace_path_to_windows_spelling():
    ctx = make()
    assert ctx.host("/workspace/data") == "D:/Podyplomowe/05_Proximity_Lidar/data"
    assert ctx.host("/workspace/data/in/plot 01.laz") == "D:/Podyplomowe/05_Proximity_Lidar/data/in/plot 01.laz"
    assert ctx.host(Path("/workspace/data/out/r1/dfin")) == "D:/Podyplomowe/05_Proximity_Lidar/data/out/r1/dfin"
    assert ctx.host("/workspace") == HOST  # the project root itself
    assert ctx.host("/workspace/") == HOST


def test_host_accepts_relative_paths_and_collapses_dots():
    ctx = make()
    assert ctx.host("data/in/a.laz") == f"{HOST}/data/in/a.laz"
    assert ctx.host("./data/../data/in/a.laz") == f"{HOST}/data/in/a.laz"
    assert ctx.host("/workspace/data/out/../in") == f"{HOST}/data/in"


@pytest.mark.parametrize(
    "configured",
    [HOST, HOST + "/", "D:\\Podyplomowe\\05_Proximity_Lidar", "D:\\Podyplomowe\\05_Proximity_Lidar\\", "  " + HOST + " "],
)
def test_host_project_dir_spellings_are_normalised(configured):
    assert make(host=configured).host("/workspace/data") == f"{HOST}/data"


def test_wsl_style_host_dir_is_converted_not_passed_through():
    # /mnt/d/... silently creates an empty phantom dir inside the docker-desktop VM (spike result).
    assert host_path("/mnt/d/Podyplomowe/05_Proximity_Lidar", "data") == HOST + "/data"
    assert host_path("/mnt/d", "data") == "D:/data"


def test_other_working_host_spellings_are_left_alone():
    assert host_path("/run/desktop/mnt/host/d/Podyplomowe/x", "data") == "/run/desktop/mnt/host/d/Podyplomowe/x/data"
    assert host_path("/home/me/proj", "data/in") == "/home/me/proj/data/in"


def test_empty_host_dir_and_comma_are_rejected():
    with pytest.raises(ValueError, match="HOST_PROJECT_DIR"):
        host_path("  ", "data")
    with pytest.raises(ValueError, match="comma"):
        host_path("D:/a,b", "data")


def test_paths_outside_project_are_rejected():
    ctx = make()
    for bad in ("/etc/passwd", "/workspace/../etc", "../x", "data/../../x", "/workspacefoo/data"):
        with pytest.raises(PathOutsideProjectError):
            ctx.host(bad)


def test_windows_drive_paths_get_a_helpful_error():
    with pytest.raises(PathOutsideProjectError, match=r"Windows path.*data/in/plot\.laz"):
        make().resolve(r"D:\Podyplomowe\05_Proximity_Lidar\data\in\plot.laz")


def test_run_dir_and_stage_dirs():
    ctx = make()
    assert ctx.run_dir == Path("/workspace/data/out/r1")
    assert ctx.stage_dir("dfin") == Path("/workspace/data/out/r1/dfin")
    assert ctx.data_dir == Path("/workspace/data")


def test_container_path_relative_to_data_mount():
    ctx = make()
    assert ctx.container_path("data/in/a.laz") == "/data/in/a.laz"
    assert ctx.container_path(ctx.stage_dir("dfin")) == "/data/out/r1/dfin"
    assert ctx.container_path("data", root="/d") == "/d"
    with pytest.raises(PathOutsideProjectError):
        ctx.container_path("Example_data/a.laz")


def test_mount_uses_host_spelling_and_checks_existence(project):
    ctx = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project)
    mount = ctx.mount("data", "/data")
    assert mount == Mount(f"{HOST}/data", "/data", readonly=False)
    assert ctx.mount("data/in", "/in", readonly=True).readonly is True
    with pytest.raises(FileNotFoundError):  # docker would silently create a ghost directory otherwise
        ctx.mount("data/nope", "/x")
    assert ctx.mount("data/nope", "/x", must_exist=False).source == f"{HOST}/data/nope"
    dry = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project, dry_run=True)
    assert dry.mount("data/nope", "/x").source == f"{HOST}/data/nope"


def test_reset_stage_dir_wipes_only_that_stage(project):
    ctx = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project)
    stale = ctx.stage_dir("dfin") / "old.xlsx"
    stale.parent.mkdir(parents=True)
    stale.write_text("x")
    other = ctx.stage_dir("split") / "keep.txt"
    other.parent.mkdir(parents=True)
    other.write_text("y")
    out = ctx.reset_stage_dir("dfin")
    assert out.is_dir() and not stale.exists() and other.exists()
    with pytest.raises(ValueError):
        ctx.reset_stage_dir("../..")


def test_from_env(monkeypatch, tmp_path):
    env = {"HOST_PROJECT_DIR": HOST, "LIDAR_PROJECT_DIR": str(tmp_path)}
    ctx = RunContext.from_env("r9", input="data/in/a.laz", reference="data/in/ref.csv", dry_run=True, env=env)
    assert ctx.project_dir == tmp_path
    assert ctx.host_project_dir == HOST
    assert ctx.input == tmp_path / "data/in/a.laz"
    assert ctx.reference == tmp_path / "data/in/ref.csv"
    assert ctx.dry_run is True and ctx.run_dir == tmp_path / "data/out/r9"
    with pytest.raises(RuntimeError, match="HOST_PROJECT_DIR"):
        RunContext.from_env("r9", env={})
    default = RunContext.from_env("r9", env={"HOST_PROJECT_DIR": HOST})
    assert default.project_dir == Path("/workspace") and default.input is None


def test_context_is_frozen():
    with pytest.raises(AttributeError):
        make().run_name = "other"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", "..", ".", ".hidden", "a/b", r"a\b", "a b", "../x", "x/..", "a,b"])
def test_invalid_run_names_are_rejected(bad):
    with pytest.raises(ValueError, match="run_name"):
        RunContext(run_name=bad, host_project_dir=HOST)


@pytest.mark.parametrize("good", ["r1", "plot_01", "r1_layerwise-clustering", "dev.v2", "drzewo_ą"])
def test_valid_run_names(good):
    assert RunContext(run_name=good, host_project_dir=HOST).run_dir.name == good


def test_empty_host_project_dir_fails_at_construction():
    with pytest.raises(ValueError, match="HOST_PROJECT_DIR"):
        RunContext(run_name="r1", host_project_dir="")

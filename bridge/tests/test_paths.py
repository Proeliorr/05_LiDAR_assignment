from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from lidar_bridge.paths import PathOutsideDataError, RunContext


def make(data_dir="/data", install_dir="/opt/lidar", **kw) -> RunContext:
    return RunContext(run_name="r1", data_dir=Path(data_dir), install_dir=Path(install_dir), **kw)


def test_defaults_are_the_container_directories():
    ctx = RunContext(run_name="r1")
    assert ctx.data_dir == Path("/data") and ctx.install_dir == Path("/opt/lidar")


def test_resolve_and_rel_are_relative_to_the_data_dir():
    ctx = make()
    assert ctx.resolve("in/plot 01.laz") == Path("/data/in/plot 01.laz")
    assert ctx.resolve("./in/../in/a.laz") == Path("/data/in/a.laz")
    assert ctx.resolve("/data/out/r1") == Path("/data/out/r1")
    assert ctx.rel("in/a.laz") == PurePosixPath("in/a.laz")
    assert ctx.rel(ctx.stage_dir("dfin")) == PurePosixPath("out/r1/dfin")
    assert ctx.rel(Path("/data/out/r1/dfin/x.las")) == PurePosixPath("out/r1/dfin/x.las")
    assert ctx.rel("/data") == PurePosixPath(".")


def test_paths_outside_data_are_rejected():
    ctx = make()
    for bad in ("/etc/passwd", "/data/../etc", "../x", "in/../../x", "/datafoo/in"):
        with pytest.raises(PathOutsideDataError):
            ctx.resolve(bad)
        with pytest.raises(PathOutsideDataError):
            ctx.rel(bad)


def test_windows_drive_paths_get_a_helpful_error():
    with pytest.raises(PathOutsideDataError, match=r"Windows path.*in/plot\.laz"):
        make().resolve(r"D:\Podyplomowe\05_Proximity_Lidar\data\in\plot.laz")


def test_run_dir_and_stage_dirs():
    ctx = make()
    assert ctx.run_dir == Path("/data/out/r1")
    assert ctx.stage_dir("dfin") == Path("/data/out/r1/dfin")
    assert ctx.data_dir == Path("/data")


def test_reset_stage_dir_wipes_only_that_stage(project):
    ctx = RunContext(run_name="r1", data_dir=project / "data", install_dir=project)
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


def test_reset_stage_dir_in_a_dry_run_touches_nothing(project):
    ctx = RunContext(run_name="r1", data_dir=project / "data", install_dir=project, dry_run=True)
    assert ctx.reset_stage_dir("dfin") == ctx.stage_dir("dfin")
    assert not ctx.run_dir.exists()


def test_from_env(tmp_path):
    data = tmp_path / "mydata"
    data.mkdir()
    env = {"LIDAR_DATA_DIR": str(data), "LIDAR_INSTALL_DIR": str(tmp_path / "inst")}
    ctx = RunContext.from_env("r9", input="in/a.laz", reference="in/ref.csv", dry_run=True, env=env)
    assert ctx.data_dir == data
    assert ctx.install_dir == tmp_path / "inst"
    assert ctx.input == data / "in/a.laz"
    assert ctx.reference == data / "in/ref.csv"
    assert ctx.dry_run is True and ctx.run_dir == data / "out/r9"
    default = RunContext.from_env("r9", env={"LIDAR_DATA_DIR": str(data)})
    assert default.install_dir == Path("/opt/lidar") and default.input is None


def test_from_env_requires_an_existing_data_dir(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        RunContext.from_env("r9", env={"LIDAR_DATA_DIR": str(tmp_path / "missing")})


def test_from_env_rejects_input_outside_the_data_dir(tmp_path):
    with pytest.raises(PathOutsideDataError):
        RunContext.from_env("r9", input="../elsewhere/a.laz", env={"LIDAR_DATA_DIR": str(tmp_path)})


def test_context_is_frozen():
    with pytest.raises(AttributeError):
        make().run_name = "other"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", "..", ".", ".hidden", "a/b", r"a\b", "a b", "../x", "x/..", "a,b"])
def test_invalid_run_names_are_rejected(bad):
    with pytest.raises(ValueError, match="run name"):
        RunContext(run_name=bad)


@pytest.mark.parametrize("good", ["r1", "plot_01", "r1_layerwise-clustering", "dev.v2", "drzewo_ą"])
def test_valid_run_names(good):
    assert RunContext(run_name=good).run_dir.name == good

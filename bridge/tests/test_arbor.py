"""Tests of the arbor stage with a mocked ``run_container`` (the real R/arbor run is checked manually, see the README/report)."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import pytest

from lidar_bridge.docker_runner import DockerRunError, Mount
from lidar_bridge.paths import RunContext
from lidar_bridge.stages import arbor
from lidar_bridge.stages._common import FINGERPRINT_FILE, StageError, read_fingerprint, tree_map_fingerprint

HOST = "D:/Podyplomowe/05_Proximity_Lidar"
ORDINALS = [1, 2, 3, 4, 5]
LOW_POINTS = {4}
R_SOURCE = Path(__file__).resolve().parents[1] / "r" / "arbor_metrics.R"

PROGRESS_OUTPUT = (
    "[10:00:00] arbor 1.1.2 | mode per_tree\n"
    "Anisotropy: 14% | 2.9s ETA (16 threads)\rAnisotropy: 15% | 2.8s ETA (16 threads)\rAnisotropy: 100% | 0.0s ETA\r"
    "[10:00:03] tree    1 [1/3] ok  dbh 0.154 m\n"
    "\x1b[33m[W2] colour\x1b[0m\n"
)


def metrics_row(ordinal: int, status: str = "ok", mode: str = "per_tree", **over) -> dict:
    row = dict.fromkeys(arbor.METRICS_COLUMNS, "")
    row.update(ordinal=ordinal, status=status, seconds="1.5", mode=mode, base_z0_m="0.1")
    if status == "ok":
        row.update(dbh_m=f"{0.1 + ordinal / 100:.4f}", height_m="12.5", dbh_height_above_ground_m="1.3", volume_m3="0.2")
    row.update(over)
    return row


def write_metrics(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns or arbor.METRICS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def run_dir(project: Path) -> Path:
    """Fake run r1 with the split results (5 trees, #4 flagged low_points), the R script and a 3DFin cloud."""
    split = project / "data" / "out" / "r1" / "split"
    (split / "trees").mkdir(parents=True)
    with (split / "tree_map.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ordinal", "raw_id", "n_points", "base_z0_m", "x_m", "y_m", "dbh_m", "th_m", "low_points", "file"])
        for o in ORDINALS:
            low = o in LOW_POINTS
            writer.writerow([o, o * 10, 100, 0.1, o, -o, "" if low else 0.2, 12.0, str(low), "" if low else f"trees/tree_{o:04d}.laz"])
            if not low:
                (split / "trees" / f"tree_{o:04d}.laz").write_bytes(b"laz")
    dfin = project / "data" / "out" / "r1" / "dfin"
    dfin.mkdir(parents=True)
    (dfin / "plot_01_tree_ID_dist_axes.las").write_bytes(b"las")
    r_dir = project / "bridge" / "r"
    r_dir.mkdir(parents=True)
    (r_dir / "arbor_metrics.R").write_text("# stub", encoding="utf-8")
    return project / "data" / "out" / "r1"


class FakeDocker:
    """Replaces ``run_container`` in the arbor module and plays the R script: appends one row per tree to ``--out_csv``."""

    def __init__(self, ctx: RunContext, *, returncode: int = 0, write: bool = True, stdout: str = PROGRESS_OUTPUT, skip_ordinals=()):
        self.ctx, self.returncode, self.write, self.stdout, self.skip_ordinals = ctx, returncode, write, stdout, set(skip_ordinals)
        self.calls: list[dict] = []

    def host_path(self, container_path: str) -> Path:
        assert container_path.startswith("/data/")
        return self.ctx.data_dir / container_path[len("/data/") :]

    def __call__(self, image, args, mounts=(), **kw):
        args = list(args)
        self.calls.append(dict(image=image, args=args, mounts=list(mounts), **kw))
        if kw.get("dry_run"):
            return subprocess.CompletedProcess(["docker"], 0, stdout="", stderr=None)
        opts = {a: args[i + 1] for i, a in enumerate(args) if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--")}
        out_csv = self.host_path(opts["--out_csv"])
        if self.write:
            mode = opts["--mode"]
            tm = read_rows(self.host_path(opts["--tree_map"]))
            ordinals = [int(r["ordinal"]) for r in tm]
            if "--trees" in opts:
                ordinals = [o for o in ordinals if o in {int(t) for t in opts["--trees"].split(",")}]
            existing = [int(r["ordinal"]) for r in read_rows(out_csv)] if out_csv.is_file() and "--resume" in args else []
            todo = [o for o in ordinals if o not in existing]
            if "--limit" in opts:
                todo = todo[: int(opts["--limit"])]
            rows = read_rows(out_csv) if existing else []
            for o in todo:
                if o in self.skip_ordinals:
                    continue
                low = o in LOW_POINTS and opts.get("--skip_low_points") == "true"
                rows.append(metrics_row(o, "skipped_low_points" if low else "ok", mode=mode))
            write_metrics(out_csv, rows)
        if self.returncode:
            raise DockerRunError(f"container exited with code {self.returncode}", returncode=self.returncode, output=self.stdout)
        return subprocess.CompletedProcess(["docker"], 0, stdout=self.stdout, stderr=None)


@pytest.fixture
def fake(monkeypatch, ctx, run_dir):
    f = FakeDocker(ctx)
    monkeypatch.setattr(arbor, "run_container", f)
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    return f


def option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


DEFAULT_TAG = "mode=per_tree homogenize=false params=none skip_low_points=true"
PLOT_TAG = "mode=plot max_match_dist=1 params=none skip_low_points=true"


def seed_results(run_dir: Path, rows: list[dict], *, mode: str = "per_tree", tag: str | None = None, fingerprint: str | None = None) -> None:
    """Earlier results as the stage leaves them: the table plus the fingerprint of the current split (and default settings)."""
    name = arbor.OUTPUT_FILES[mode][0]
    write_metrics(run_dir / "arbor" / name, rows)
    fp = fingerprint or tree_map_fingerprint(run_dir / "split")
    (run_dir / "arbor" / arbor.fingerprint_file(mode)).write_text(f"{fp}\n{tag or (DEFAULT_TAG if mode == 'per_tree' else PLOT_TAG)}\n", encoding="utf-8")


def change_split(run_dir: Path, *, drop: int | None = None) -> None:
    """The split stage ran again with other parameters: tree_map.csv changes (positions; optionally fewer trees)."""
    path = run_dir / "split" / "tree_map.csv"
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    body = [r for r in body if drop is None or int(r[0]) != drop]
    for r in body:
        r[header.index("x_m")] = str(float(r[header.index("x_m")]) + 0.5)
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows([header, *body])


def ordinals_of(path: Path) -> list[int]:
    return [int(r["ordinal"]) for r in read_rows(path)]



# -- command construction -------------------------------------------------------------------------------------
def test_per_tree_command_mounts_and_env(ctx, fake, run_dir):
    result = arbor.run_arbor(ctx)
    (call,) = fake.calls
    assert call["image"] == "lidar-arbor" and call["network"] == "none"
    assert call["env"] == {"OMP_NUM_THREADS": "8"}
    assert call["mounts"] == [
        Mount(f"{HOST}/data", "/data"),
        Mount(f"{HOST}/bridge/r", "/bridge_r", readonly=True),
    ]
    args = call["args"]
    assert args[0] == "/bridge_r/arbor_metrics.R"  # the image's ENTRYPOINT is Rscript
    assert option(args, "--mode") == "per_tree"
    assert option(args, "--trees_dir") == "/data/out/r1/split/trees"
    assert option(args, "--tree_map") == "/data/out/r1/split/tree_map.csv"
    assert option(args, "--out_csv") == "/data/out/r1/arbor/metrics.csv"
    assert option(args, "--threads") == "8" and option(args, "--skip_low_points") == "true"
    for absent in ("--trees", "--limit", "--resume", "--homogenize", "--params", "--plot_las", "--max_match_dist"):
        assert absent not in args
    assert call["timeout_s"] == 7200
    assert (run_dir / "arbor" / "metrics.csv").is_file()
    assert result["metrics"] == "data/out/r1/arbor/metrics.csv" and result["mode"] == "per_tree"
    assert result["n_trees"] == 5 and result["counts"] == {"ok": 4, "skipped_low_points": 1} and result["skipped"] is False


def test_one_container_run_for_all_trees(ctx, fake):
    arbor.run_arbor(ctx)
    assert len(fake.calls) == 1


def test_threads_limit_selection_and_flags(ctx, fake):
    arbor.run_arbor(ctx, threads=4, trees=[3, 1, 3], limit=1, skip_low_points=False, homogenize=True, timeout_s=0)
    call = fake.calls[0]
    args = call["args"]
    assert call["env"] == {"OMP_NUM_THREADS": "4"} and option(args, "--threads") == "4"
    assert option(args, "--trees") == "1,3" and option(args, "--limit") == "1"
    assert option(args, "--skip_low_points") == "false"
    assert "--homogenize" in args
    assert call["timeout_s"] is None  # 0 = no limit
    rows = read_rows(ctx.stage_dir("arbor") / "metrics.csv")
    assert [r["ordinal"] for r in rows] == ["1"]


def test_params_are_forwarded_flattened(ctx, fake):
    arbor.run_arbor(ctx, params={"qsm": {"smooth_steps": 20, "broken_detection_enabled": False}, "semantic.x": 1.5})
    assert option(fake.calls[0]["args"], "--params") == "qsm.broken_detection_enabled=false,qsm.smooth_steps=20,semantic.x=1.5"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(mode="bogus"), "arbor.mode"),
        (dict(threads=0), "threads"),
        (dict(threads=2.5), "threads"),
        (dict(limit=0), "limit"),
        (dict(max_match_dist=0), "max_match_dist"),
        (dict(trees=[99]), r"\[99\].*not in tree_map"),
        (dict(trees=[]), "empty"),
        (dict(trees=["x"]), "list of integers"),
        (dict(params={"smooth_steps": 3}), "section"),
        (dict(params={"qsm.a": "1,2"}), "no ','"),
    ],
)
def test_invalid_arguments_fail_before_docker(ctx, fake, kwargs, message):
    with pytest.raises(StageError, match=message):
        arbor.run_arbor(ctx, **kwargs)
    assert fake.calls == []


# -- inputs -----------------------------------------------------------------------------------------------------
def test_missing_split_results(ctx, fake, run_dir):
    (run_dir / "split" / "tree_map.csv").unlink()
    with pytest.raises(StageError, match="split stage"):
        arbor.run_arbor(ctx)
    assert fake.calls == []


def test_missing_trees_dir_only_matters_for_per_tree(ctx, fake, run_dir):
    for f in (run_dir / "split" / "trees").iterdir():
        f.unlink()
    (run_dir / "split" / "trees").rmdir()
    with pytest.raises(StageError, match="trees not found"):
        arbor.run_arbor(ctx)
    arbor.run_arbor(ctx, mode="plot")  # does not read the tree files


def test_missing_r_script(ctx, fake, project):
    (project / "bridge" / "r" / "arbor_metrics.R").unlink()
    with pytest.raises(StageError, match="arbor_metrics.R not found"):
        arbor.run_arbor(ctx)


def test_unreadable_tree_map(ctx, fake, run_dir):
    (run_dir / "split" / "tree_map.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(StageError, match="no 'ordinal' column"):
        arbor.run_arbor(ctx)


# -- resume / force ---------------------------------------------------------------------------------------------
def test_second_run_finds_everything_done_and_starts_no_container(ctx, fake):
    arbor.run_arbor(ctx)
    result = arbor.run_arbor(ctx)
    assert len(fake.calls) == 1 and result["skipped"] is True and result["n_trees"] == 5


def test_resume_processes_only_missing_ordinals(ctx, fake, run_dir):
    seed_results(run_dir, [metrics_row(1), metrics_row(2, dbh_m="0.5000")])  # made on the current split with default settings
    result = arbor.run_arbor(ctx)
    args = fake.calls[0]["args"]
    assert "--resume" in args
    rows = read_rows(run_dir / "arbor" / "metrics.csv")
    assert [int(r["ordinal"]) for r in rows] == [1, 2, 3, 4, 5]
    assert rows[1]["dbh_m"] == "0.5000"  # kept from the earlier run
    assert result["skipped"] is False


def test_resume_with_limit_continues_where_it_stopped(ctx, fake, run_dir):
    arbor.run_arbor(ctx, limit=2)
    assert [r["ordinal"] for r in read_rows(run_dir / "arbor" / "metrics.csv")] == ["1", "2"]
    arbor.run_arbor(ctx, limit=2)
    assert [r["ordinal"] for r in read_rows(run_dir / "arbor" / "metrics.csv")] == ["1", "2", "3", "4"]
    assert "--resume" in fake.calls[1]["args"]


def test_resume_false_and_force_start_from_scratch(ctx, fake, run_dir):
    arbor.run_arbor(ctx)
    for kwargs in (dict(resume=False), dict(force=True)):
        write_metrics(run_dir / "arbor" / "metrics.csv", [metrics_row(1, dbh_m="0.9999")])
        arbor.run_arbor(ctx, **kwargs)
        assert "--resume" not in fake.calls[-1]["args"]
        rows = read_rows(run_dir / "arbor" / "metrics.csv")
        assert len(rows) == 5 and rows[0]["dbh_m"] != "0.9999"


def test_force_discards_stale_files_before_the_run(ctx, monkeypatch, run_dir):
    seen = {}

    def fake_run(image, args, mounts=(), **kw):
        seen["csv"] = (run_dir / "arbor" / "metrics.csv").exists()
        seen["log"] = (run_dir / "arbor" / "arbor.log").exists()
        raise DockerRunError("boom", returncode=1, output="")

    monkeypatch.setattr(arbor, "run_container", fake_run)
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    write_metrics(run_dir / "arbor" / "metrics.csv", [metrics_row(1)])
    (run_dir / "arbor" / "arbor.log").write_text("old", encoding="utf-8")
    with pytest.raises(StageError):
        arbor.run_arbor(ctx, force=True)
    assert seen == {"csv": False, "log": False}


def test_table_of_another_layout_is_an_error_unless_forced(ctx, fake, run_dir):
    write_metrics(run_dir / "arbor" / "metrics.csv", [{"ordinal": 1, "dbh_m": "0.1"}], columns=["ordinal", "dbh_m"])
    with pytest.raises(StageError, match="force=True"):
        arbor.run_arbor(ctx)
    assert fake.calls == []
    arbor.run_arbor(ctx, force=True)
    assert len(fake.calls) == 1


# -- stale results: source fingerprint ---------------------------------------------------------------------------
def test_fingerprint_is_written_next_to_the_metrics(ctx, fake, run_dir):
    arbor.run_arbor(ctx)
    stored = read_fingerprint(run_dir / "arbor")
    assert (run_dir / "arbor" / FINGERPRINT_FILE).is_file() and (run_dir / "arbor" / "metrics.csv").is_file()
    assert stored == (tree_map_fingerprint(run_dir / "split"), DEFAULT_TAG)
    assert arbor.fingerprint_file("per_tree") == FINGERPRINT_FILE


def test_settings_tag_strings():
    kw = dict(homogenize=False, params_arg="", max_match_dist=1.0, skip_low_points=True)
    assert arbor.settings_tag("per_tree", **kw) == DEFAULT_TAG
    assert arbor.settings_tag("plot", **kw) == PLOT_TAG
    other = arbor.settings_tag("per_tree", **{**kw, "homogenize": True, "skip_low_points": False, "params_arg": "qsm.a=1"})
    assert other.startswith("mode=per_tree homogenize=true params=") and other.endswith(" skip_low_points=false") and "params=none" not in other
    assert arbor.settings_tag("plot", **{**kw, "max_match_dist": 0.75}) == "mode=plot max_match_dist=0.75 params=none skip_low_points=true"
    # the mode ignores what it does not use: no spurious recomputation
    assert arbor.settings_tag("plot", **{**kw, "homogenize": True}) == PLOT_TAG
    assert arbor.settings_tag("per_tree", **{**kw, "max_match_dist": 3.0}) == DEFAULT_TAG


def test_plot_mode_has_its_own_fingerprint_file(ctx, fake, run_dir):
    arbor.run_arbor(ctx)
    arbor.run_arbor(ctx, mode="plot")
    fp = tree_map_fingerprint(run_dir / "split")
    assert read_fingerprint(run_dir / "arbor") == (fp, DEFAULT_TAG)  # the per_tree record (what compare reads) is not overwritten
    plot = (run_dir / "arbor" / "source_fingerprint_plot.txt").read_text(encoding="utf-8")
    assert plot == f"{fp}\n{PLOT_TAG}\n"


def test_matching_fingerprint_means_nothing_to_run_and_no_warning(ctx, fake, caplog):
    arbor.run_arbor(ctx)
    with caplog.at_level("INFO", logger="lidar_bridge.stages.arbor"):
        result = arbor.run_arbor(ctx)
    assert result["skipped"] is True and len(fake.calls) == 1
    assert "nothing to run" in caplog.text and "recomputing" not in caplog.text


def test_changed_split_recomputes_all_trees_with_a_warning(ctx, fake, run_dir, caplog):
    arbor.run_arbor(ctx)
    old_fp = read_fingerprint(run_dir / "arbor")[0]
    write_metrics(run_dir / "arbor" / "metrics.csv", [metrics_row(o, dbh_m="0.9999") for o in ORDINALS])  # the OLD values
    change_split(run_dir)  # e.g. split.max_axis_dist=2.0 (dfin/split were rebuilt)
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        result = arbor.run_arbor(ctx)
    assert "split/dfin changed since the existing arbor results were made: recomputing all trees" in caplog.text
    assert len(fake.calls) == 2 and "--resume" not in fake.calls[1]["args"] and result["skipped"] is False
    rows = read_rows(run_dir / "arbor" / "metrics.csv")
    assert [int(r["ordinal"]) for r in rows] == ORDINALS and all(r["dbh_m"] != "0.9999" for r in rows if r["status"] == "ok")
    stored = read_fingerprint(run_dir / "arbor")
    assert stored == (tree_map_fingerprint(run_dir / "split"), DEFAULT_TAG) and stored[0] != old_fp
    # and it is a normal finished state again
    assert arbor.run_arbor(ctx)["skipped"] is True and len(fake.calls) == 2


def test_changed_split_with_fewer_trees_drops_the_extra_old_ordinals(ctx, fake, run_dir):
    arbor.run_arbor(ctx)
    assert ordinals_of(run_dir / "arbor" / "metrics.csv") == ORDINALS
    change_split(run_dir, drop=5)  # 3DFin now finds 4 trees
    arbor.run_arbor(ctx)
    assert ordinals_of(run_dir / "arbor" / "metrics.csv") == [1, 2, 3, 4]  # the old row of ordinal 5 must not survive


def test_changed_split_with_a_subset_keeps_only_the_new_subset(ctx, fake, run_dir, caplog):
    arbor.run_arbor(ctx)
    change_split(run_dir)
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        arbor.run_arbor(ctx, trees=[2])
    assert "split/dfin changed" in caplog.text
    assert ordinals_of(run_dir / "arbor" / "metrics.csv") == [2] and "--resume" not in fake.calls[-1]["args"]


def test_stale_table_and_log_are_deleted_before_the_run(ctx, monkeypatch, run_dir):
    seed_results(run_dir, [metrics_row(1)], fingerprint="0123456789abcdef")  # other split
    (run_dir / "arbor" / "arbor.log").write_text("old", encoding="utf-8")
    seen = {}

    def fake_run(image, args, mounts=(), **kw):
        seen["csv"] = (run_dir / "arbor" / "metrics.csv").exists()
        seen["log"] = (run_dir / "arbor" / "arbor.log").exists()
        seen["fingerprint"] = read_fingerprint(run_dir / "arbor")
        raise DockerRunError("boom", returncode=1, output="")

    monkeypatch.setattr(arbor, "run_container", fake_run)
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    with pytest.raises(StageError):
        arbor.run_arbor(ctx)
    assert seen == {"csv": False, "log": False, "fingerprint": (tree_map_fingerprint(run_dir / "split"), DEFAULT_TAG)}


def test_table_without_a_fingerprint_is_treated_as_stale(ctx, fake, run_dir, caplog):
    write_metrics(run_dir / "arbor" / "metrics.csv", [metrics_row(o, dbh_m="0.9999") for o in ORDINALS])  # made before fingerprints existed
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        arbor.run_arbor(ctx)
    assert "no source fingerprint" in caplog.text and "recomputing all trees" in caplog.text
    assert len(fake.calls) == 1 and "--resume" not in fake.calls[0]["args"]
    assert all(r["dbh_m"] != "0.9999" for r in read_rows(run_dir / "arbor" / "metrics.csv") if r["status"] == "ok")
    assert read_fingerprint(run_dir / "arbor") is not None


@pytest.mark.parametrize(
    "kwargs, was, now",
    [
        (dict(homogenize=True), "homogenize=false", "homogenize=true"),
        (dict(params={"qsm.smooth_steps": 3}), "params=none", "params="),
        (dict(skip_low_points=False), "skip_low_points=true", "skip_low_points=false"),
    ],
)
def test_changed_settings_recompute_all_trees(ctx, fake, run_dir, caplog, kwargs, was, now):
    arbor.run_arbor(ctx)
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        arbor.run_arbor(ctx, **kwargs)
    assert "arbor settings changed" in caplog.text and was in caplog.text and now in caplog.text
    assert len(fake.calls) == 2 and "--resume" not in fake.calls[1]["args"]
    assert read_fingerprint(run_dir / "arbor")[1] != DEFAULT_TAG
    arbor.run_arbor(ctx, **kwargs)  # same settings again: nothing to do
    assert len(fake.calls) == 2


def test_same_params_do_not_look_like_a_change(ctx, fake):
    arbor.run_arbor(ctx, params={"qsm": {"smooth_steps": 20}})
    assert arbor.run_arbor(ctx, params={"qsm.smooth_steps": 20})["skipped"] is True  # same flattened params
    assert len(fake.calls) == 1


def test_subset_runs_on_the_same_split_still_resume(ctx, fake, run_dir, caplog):
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        arbor.run_arbor(ctx, trees=[1, 2])
        arbor.run_arbor(ctx, trees=[4, 5])
        arbor.run_arbor(ctx, limit=3)
    assert "recomputing" not in caplog.text
    assert all("--resume" in c["args"] for c in fake.calls[1:])
    assert sorted(ordinals_of(run_dir / "arbor" / "metrics.csv")) == [1, 2, 3, 4, 5]  # R appends in processing order


def test_force_ignores_the_fingerprint_without_a_stale_warning(ctx, fake, run_dir, caplog):
    seed_results(run_dir, [metrics_row(1, dbh_m="0.9999")])
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        arbor.run_arbor(ctx, force=True)
    assert "recomputing" not in caplog.text and "--resume" not in fake.calls[0]["args"]
    assert ordinals_of(run_dir / "arbor" / "metrics.csv") == ORDINALS
    assert read_fingerprint(run_dir / "arbor") == (tree_map_fingerprint(run_dir / "split"), DEFAULT_TAG)


def test_interrupted_run_keeps_its_fingerprint_so_the_partial_table_is_resumed(ctx, run_dir, monkeypatch, caplog):
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    monkeypatch.setattr(arbor, "run_container", FakeDocker(ctx, returncode=1, stdout="FATAL: killed\n"))  # rows written, then the container dies
    with pytest.raises(StageError):
        arbor.run_arbor(ctx, limit=2)
    assert read_fingerprint(run_dir / "arbor") == (tree_map_fingerprint(run_dir / "split"), DEFAULT_TAG)
    good = FakeDocker(ctx)
    monkeypatch.setattr(arbor, "run_container", good)
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        arbor.run_arbor(ctx)
    assert "recomputing" not in caplog.text and "--resume" in good.calls[0]["args"]
    assert ordinals_of(run_dir / "arbor" / "metrics.csv") == ORDINALS


def test_plot_mode_stale_results_are_recomputed(ctx, fake, run_dir, caplog):
    arbor.run_arbor(ctx)
    arbor.run_arbor(ctx, mode="plot")
    fp0 = tree_map_fingerprint(run_dir / "split")
    per_tree_before = (run_dir / "arbor" / "metrics.csv").read_text(encoding="utf-8")
    assert arbor.run_arbor(ctx, mode="plot")["skipped"] is True and len(fake.calls) == 2
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        arbor.run_arbor(ctx, mode="plot", max_match_dist=0.5)  # setting changed
        change_split(run_dir, drop=5)
        arbor.run_arbor(ctx, mode="plot", max_match_dist=0.5)  # split changed
    assert len(fake.calls) == 4
    assert "arbor settings changed" in caplog.text and "split/dfin changed" in caplog.text
    assert ordinals_of(run_dir / "arbor" / "metrics_plot.csv") == [1, 2, 3, 4]
    plot_record = (run_dir / "arbor" / "source_fingerprint_plot.txt").read_text(encoding="utf-8").splitlines()
    assert plot_record[0] == tree_map_fingerprint(run_dir / "split") and plot_record[0] != fp0
    assert read_fingerprint(run_dir / "arbor") == (fp0, DEFAULT_TAG)  # plot runs never touch the per_tree record
    assert (run_dir / "arbor" / "metrics.csv").read_text(encoding="utf-8") == per_tree_before  # the other mode is not touched


def test_dry_run_with_stale_results_changes_nothing(project, run_dir, monkeypatch, caplog):
    arbor_dir = run_dir / "arbor"
    seed_results(run_dir, [metrics_row(1), metrics_row(2)], fingerprint="0123456789abcdef")  # another split
    (arbor_dir / "arbor.log").write_text("old", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in arbor_dir.iterdir()}
    dry = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project, dry_run=True)
    calls = []
    monkeypatch.setattr(arbor, "run_container", lambda *a, **k: calls.append((a, k)) or subprocess.CompletedProcess(["docker"], 0, stdout="", stderr=None))
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    with caplog.at_level("WARNING", logger="lidar_bridge.stages.arbor"):
        result = arbor.run_arbor(dry)
    assert result["dry_run"] is True and "[dry-run]" in caplog.text and "split/dfin changed" in caplog.text
    ((args, kwargs),) = calls
    assert kwargs["dry_run"] is True and "--resume" not in args[1]  # the rehearsed command is a full recomputation
    assert {p.name: p.read_bytes() for p in arbor_dir.iterdir()} == before  # table, log and fingerprint untouched


# -- failures ---------------------------------------------------------------------------------------------------
def test_container_failure_is_a_stage_error_and_keeps_partial_rows(ctx, run_dir, monkeypatch):
    f = FakeDocker(ctx, returncode=1, stdout="Anisotropy: 5% | ETA\rAnisotropy: 6% | ETA\rFATAL: out of memory\n")
    monkeypatch.setattr(arbor, "run_container", f)
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    with pytest.raises(StageError) as err:
        arbor.run_arbor(ctx)
    text = str(err.value)
    assert "arbor container failed" in text and "FATAL: out of memory" in text and "Anisotropy" not in text
    assert "resume=True" in text  # the partial table is there
    assert (run_dir / "arbor" / "metrics.csv").is_file()
    assert "FATAL: out of memory" in (run_dir / "arbor" / "arbor.log").read_text(encoding="utf-8")


def test_container_ok_but_no_csv(ctx, run_dir, monkeypatch):
    monkeypatch.setattr(arbor, "run_container", FakeDocker(ctx, write=False))
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    with pytest.raises(StageError, match="produced no metrics.csv"):
        arbor.run_arbor(ctx)


def test_csv_missing_a_selected_tree(ctx, run_dir, monkeypatch):
    monkeypatch.setattr(arbor, "run_container", FakeDocker(ctx, skip_ordinals=[2, 5]))
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    with pytest.raises(StageError, match=r"no row for 2 selected tree.*\[2, 5\]"):
        arbor.run_arbor(ctx)


def test_missing_image_is_reported(ctx, run_dir, monkeypatch):
    from lidar_bridge.docker_runner import ImageNotFoundError

    def no_image(tag, **kw):
        raise ImageNotFoundError("docker image 'lidar-arbor' not found")

    monkeypatch.setattr(arbor, "require_image", no_image)
    monkeypatch.setattr(arbor, "run_container", FakeDocker(ctx))
    with pytest.raises(ImageNotFoundError):
        arbor.run_arbor(ctx)


def test_per_tree_row_problems_are_not_stage_errors(ctx, run_dir, monkeypatch):
    """A tree that failed inside R is a row, not a failure of the stage."""
    f = FakeDocker(ctx)

    def run(image, args, mounts=(), **kw):
        out = f(image, args, mounts, **kw)
        rows = read_rows(run_dir / "arbor" / "metrics.csv")
        rows[0].update(status="failed", dbh_m="", height_m="", note="qsm failed")
        rows[1].update(status="error", dbh_m="", height_m="", note="tree file not found")
        write_metrics(run_dir / "arbor" / "metrics.csv", rows)
        return out

    monkeypatch.setattr(arbor, "run_container", run)
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    result = arbor.run_arbor(ctx)
    assert result["counts"] == {"failed": 1, "error": 1, "ok": 2, "skipped_low_points": 1}


# -- log --------------------------------------------------------------------------------------------------------
def test_log_file_is_clean(ctx, fake, run_dir):
    arbor.run_arbor(ctx)
    text = (run_dir / "arbor" / "arbor.log").read_text(encoding="utf-8")
    assert "Anisotropy" not in text and "ETA" not in text and "\r" not in text and "\x1b" not in text
    assert "[10:00:03] tree    1 [1/3] ok  dbh 0.154 m" in text and "[W2] colour" in text


def test_clean_log_lines():
    out = arbor.clean_log_lines(PROGRESS_OUTPUT)
    assert out == ["[10:00:00] arbor 1.1.2 | mode per_tree", "[10:00:03] tree    1 [1/3] ok  dbh 0.154 m", "[W2] colour"]
    assert arbor.clean_log_lines("  |=====     | 45%\r  |==========| 100%\rdone\n") == ["done"]
    assert arbor.clean_log_lines("12%\r13%\r") == []
    assert arbor.clean_log_lines(None) == [] and arbor.clean_log_lines("") == []
    # a normal line that merely mentions a percentage or a colon stays
    assert arbor.clean_log_lines("tree 5: kept 75% of points") == ["tree 5: kept 75% of points"]
    assert arbor.clean_log_lines("[10:00:01] memory: R process VmHWM: 1292096 kB") == ["[10:00:01] memory: R process VmHWM: 1292096 kB"]


# -- plot mode --------------------------------------------------------------------------------------------------
def test_plot_mode_command_and_separate_output(ctx, fake, run_dir):
    write_metrics(run_dir / "arbor" / "metrics.csv", [metrics_row(o) for o in ORDINALS])  # the per_tree table
    before = (run_dir / "arbor" / "metrics.csv").read_text(encoding="utf-8")
    result = arbor.run_arbor(ctx, mode="plot", max_match_dist=0.75, threads=6)
    (call,) = fake.calls
    args = call["args"]
    assert option(args, "--mode") == "plot"
    assert option(args, "--plot_las") == "/data/out/r1/dfin/plot_01_tree_ID_dist_axes.las"
    assert option(args, "--max_match_dist") == "0.75"
    assert option(args, "--out_csv") == "/data/out/r1/arbor/metrics_plot.csv"
    assert "--trees_dir" not in args and "--resume" not in args and "--homogenize" not in args
    assert call["env"] == {"OMP_NUM_THREADS": "6"}
    assert result["metrics"] == "data/out/r1/arbor/metrics_plot.csv" and result["log"].endswith("arbor_plot.log")
    assert (run_dir / "arbor" / "metrics.csv").read_text(encoding="utf-8") == before  # compare's input is untouched
    assert {r["mode"] for r in read_rows(run_dir / "arbor" / "metrics_plot.csv")} == {"plot"}
    assert (run_dir / "arbor" / "arbor_plot.log").is_file() and not (run_dir / "arbor" / "arbor.log").exists()


def test_plot_mode_reuses_a_complete_table_and_reruns_an_incomplete_one(ctx, fake, run_dir):
    arbor.run_arbor(ctx, mode="plot")
    assert arbor.run_arbor(ctx, mode="plot")["skipped"] is True and len(fake.calls) == 1
    write_metrics(run_dir / "arbor" / "metrics_plot.csv", [metrics_row(1, mode="plot")])
    arbor.run_arbor(ctx, mode="plot")
    assert len(fake.calls) == 2 and "--resume" not in fake.calls[1]["args"]
    assert len(read_rows(run_dir / "arbor" / "metrics_plot.csv")) == 5
    arbor.run_arbor(ctx, mode="plot", force=True)
    assert len(fake.calls) == 3


def test_plot_mode_needs_the_dfin_cloud(ctx, fake, run_dir):
    (run_dir / "dfin" / "plot_01_tree_ID_dist_axes.las").unlink()
    with pytest.raises(StageError, match="no .*_tree_ID_dist_axes.las"):
        arbor.run_arbor(ctx, mode="plot")
    (run_dir / "dfin").rename(run_dir / "dfin_gone")
    with pytest.raises(StageError, match="run the dfin stage"):
        arbor.run_arbor(ctx, mode="plot")


def test_find_dfin_cloud_prefers_the_input_stem(ctx, run_dir):
    (run_dir / "dfin" / "other_tree_ID_dist_axes.las").write_bytes(b"x")
    assert arbor.find_dfin_cloud(ctx).name == "plot_01_tree_ID_dist_axes.las"  # ctx.input is data/in/plot_01.laz
    no_input = RunContext(run_name="r1", host_project_dir=HOST, project_dir=ctx.project_dir)
    with pytest.raises(StageError, match="several 3DFin clouds"):
        arbor.find_dfin_cloud(no_input)


# -- dry run ----------------------------------------------------------------------------------------------------
def test_dry_run_only_logs_the_command(project, run_dir, monkeypatch):
    dry = RunContext(run_name="r1", host_project_dir=HOST, project_dir=project, dry_run=True)
    calls = []
    monkeypatch.setattr(arbor, "run_container", lambda *a, **k: calls.append((a, k)) or subprocess.CompletedProcess(["docker"], 0, stdout="", stderr=None))
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    result = arbor.run_arbor(dry, trees=[2])
    assert result["dry_run"] is True
    ((args, kwargs),) = calls
    assert kwargs["dry_run"] is True and option(list(args[1]), "--trees") == "2"
    assert not (run_dir / "arbor").exists()  # nothing created


def test_dry_run_tolerates_missing_inputs(project, monkeypatch, caplog):
    dry = RunContext(run_name="nothing", host_project_dir=HOST, project_dir=project, dry_run=True)
    (project / "bridge" / "r").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(arbor, "run_container", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(["docker"], 0, stdout="", stderr=None))
    monkeypatch.setattr(arbor, "require_image", lambda *a, **k: None)
    with caplog.at_level("WARNING"):
        arbor.run_arbor(dry, mode="plot", trees=[5, 1])
    assert len(calls) == 1 and option(list(calls[0][1]), "--trees") == "1,5"
    assert "tree_map.csv not found" in caplog.text


# -- metrics.csv contract -----------------------------------------------------------------------------------------
def test_contract_columns_match_the_r_script():
    """METRICS_COLUMNS (python) and CONTRACT_COLS (arbor_metrics.R) must be the same list in the same order."""
    text = R_SOURCE.read_text(encoding="utf-8")
    block = re.search(r"CONTRACT_COLS <- c\((.*?)\n\)", text, re.S).group(1)
    assert re.findall(r'"([^"]+)"', block) == arbor.METRICS_COLUMNS
    assert arbor.METRICS_COLUMNS[:5] == ["ordinal", "dbh_m", "height_m", "status", "seconds"]  # what compare reads


def test_synthetic_valid_table_passes_the_contract(tmp_path):
    rows = [
        metrics_row(1),
        metrics_row(2, note="Small tree allometry"),  # estimated DBH stays ok
        metrics_row(3, "no_dbh", volume_m3="0.01", note="breast height outside the stem"),
        metrics_row(4, "failed", note="no arbor tree within 1.00 m"),
        metrics_row(5, "skipped_low_points"),
        metrics_row(6, "error", note="tree file not found"),
    ]
    write_metrics(tmp_path / "m.csv", rows)
    table = arbor.read_metrics(tmp_path / "m.csv")
    assert list(table.columns) == arbor.METRICS_COLUMNS and table["ordinal"].tolist() == [1, 2, 3, 4, 5, 6]
    assert arbor.check_metrics(table) == []


def test_contract_violations_are_reported(tmp_path):
    rows = [
        metrics_row(1, dbh_m=""),  # ok without dbh
        metrics_row(2, "failed", dbh_m="0.2"),  # dbh on a non-ok row
        metrics_row(3, height_m="-1"),
        metrics_row(4, seconds=""),
    ]
    write_metrics(tmp_path / "m.csv", rows)
    problems = arbor.check_metrics(arbor.read_metrics(tmp_path / "m.csv"))
    assert len(problems) == 4
    assert any("ordinal 1" in p and "dbh_m" in p for p in problems)
    assert any("ordinal 2" in p and "empty dbh_m" in p for p in problems)
    assert any("ordinal 3" in p and "height_m" in p for p in problems)
    assert any("ordinal 4" in p and "seconds" in p for p in problems)


@pytest.mark.parametrize(
    "rows, columns, message",
    [
        ([metrics_row(1)], arbor.METRICS_COLUMNS[1:], "columns are"),
        ([metrics_row(1)], list(reversed(arbor.METRICS_COLUMNS)), "columns are"),
        ([metrics_row(1), metrics_row(1)], None, "duplicate ordinal"),
        ([metrics_row(1, "timeout")], None, "unknown status"),
        ([metrics_row("x", "failed")], None, "non-integer ordinal"),
    ],
)
def test_structural_problems_raise(tmp_path, rows, columns, message):
    write_metrics(tmp_path / "m.csv", [{k: v for k, v in r.items() if k in (columns or arbor.METRICS_COLUMNS)} for r in rows], columns)
    with pytest.raises(StageError, match=message):
        arbor.read_metrics(tmp_path / "m.csv")


def test_empty_metrics_file(tmp_path):
    (tmp_path / "m.csv").write_text("", encoding="utf-8")
    with pytest.raises(StageError, match="empty"):
        arbor.read_metrics(tmp_path / "m.csv")


# -- helpers ------------------------------------------------------------------------------------------------------
def test_flatten_params():
    assert arbor.flatten_params(None) == "" and arbor.flatten_params({}) == ""
    assert arbor.flatten_params({"qsm": {"a": 1, "b": True}, "x.y": "name"}) == "qsm.a=1,qsm.b=true,x.y=name"
    with pytest.raises(StageError, match="scalar"):
        arbor.flatten_params({"qsm.a": [1, 2]})
    with pytest.raises(StageError, match="scalar"):
        arbor.flatten_params({"qsm": {"a": {"b": 1}}})


def test_select_ordinals():
    assert arbor.select_ordinals([3, 1, 2], None) == [1, 2, 3]
    assert arbor.select_ordinals([1, 2, 3], [3, 3, 1]) == [1, 3]
    assert arbor.select_ordinals([1, 2, 3], None, limit=2) == [1, 2]


# -- hydra ------------------------------------------------------------------------------------------------------------
def test_hydra_config_exposes_the_new_parameters(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra_zen import instantiate

    from lidar_bridge.configs import CONFIG_NAME, register_configs

    register_configs()
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(tmp_path), version_base="1.3"):
            cfg = compose(config_name=CONFIG_NAME, overrides=["arbor.trees=[3,8]", "arbor.limit=2", "arbor.mode=plot", "arbor.threads=4", "+arbor.params.qsm.smooth_steps=20"])
            default = compose(config_name=CONFIG_NAME)
    finally:
        GlobalHydra.instance().clear()
    kw = instantiate(cfg.arbor).keywords
    assert kw["trees"] == [3, 8] and kw["limit"] == 2 and kw["mode"] == "plot" and kw["threads"] == 4
    assert kw["params"] == {"qsm": {"smooth_steps": 20}}
    d = default.arbor
    assert d.trees is None and d.limit is None and d.threads == 8 and d.skip_low_points is True
    assert d.resume is True and d.force is False and d.max_match_dist == 1.0 and d.homogenize is False

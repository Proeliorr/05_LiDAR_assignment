"""Stage 3a (WP6a): run SmartQSM once per tree in the ``lidar-smartqsm`` container and collect DBH / height.

Contract
--------
Reads ``data/out/<run>/split/tree_map.csv`` + ``split/trees/tree_XXXX.laz`` (see :mod:`lidar_bridge.stages.split`) and writes
into ``data/out/<run>/smartqsm/<config>/`` (``<config>`` = the preset name, so ``-m smartqsm.config=a,b`` never collides)::

    config.yaml                   the per-run SmartQSM config: preset from third_party/SmartQSM/configs + overrides
    work/tree_0001.laz            copy of the tree cloud (SmartQSM writes its results next to its input)
    work/tree_0001_qsm.mat        SmartQSM result (QSM.treedata = tree parameters); also _branches.ply, _crown_convex.ply,
                                  _active_crown_convex.ply, _skeleton.dxf
    logs/tree_0001.log            container output of that tree
    metrics.csv                   one row per tree, REWRITTEN AFTER EVERY TREE (a 3 h run survives an interrupt); the rows of
                                  the selected trees are merged into the rows of earlier runs on the same split (see below)
    source_fingerprint.txt        hash of split/tree_map.csv + the settings the rows were made with (read by ``compare``)

The container is started once per tree, strictly sequentially (SmartQSM uses ~8 worker processes per tree)::

    docker run --rm -i --network none -v <host>/data:/data lidar-smartqsm -t -y \
        -c /data/out/<run>/smartqsm/<config>/config.yaml /data/out/<run>/smartqsm/<config>/work/tree_0001.laz
    (stdin "no\\n": answers the update-check prompt; the check itself is offline-safe with --network none)

SmartQSM's exit code is ALWAYS 0, so success = ``<stem>_qsm.mat`` exists and holds a readable ``QSM.treedata``.

GPU
---
``gpu=True`` runs the same command with ``docker run --gpus all`` in the image ``lidar-smartqsm-gpu`` (CUDA torch + spconv-cu126;
built by ``run_pipeline.bat --gpu --build-only``; ``image`` still at its default). Only the ``spconv-contraction-*-GPU`` presets
(``device: cuda``) use the card; every other preset runs on the CPU even in the GPU image (WARNING). A ``-GPU`` preset without
``gpu=True`` is refused (the container would have no GPU). A tree that does not fit into the card's memory ends like any other
SmartQSM failure: no ``.mat``, status ``failed``, the reason (CUDA out of memory, ...) in ``note`` and in ``logs/``; every tree has
its own container (the memory is freed) and the next trees still run.

metrics.csv (columns in this order; WP7 ``compare`` reads the first five by name)
----------------------------------------------------------------------------------
``ordinal, dbh_m, height_m, status, seconds`` and then the extras ``height_raw_m, base_z0_m, dbh_height_above_ground_m,
num_branches, max_order, stem_volume_m3, trunk_volume_m3, n_points, note``.

* ``dbh_m``  = ``treedata.DBH_cm / 100``.
* ``height_m`` = ``treedata.height_m + base_z0_m`` = height above the TRUE ground. SmartQSM's ``height_m`` (``height_raw_m``) is
  ``max z - min z`` of its (voxel-downsampled, outlier-filtered) cloud; the split stage cut the cloud at ``Z0 >= min_z0``, so
  the lowest point is ``base_z0_m`` (~0.10 m) above the ground and that offset is added back.
* ``dbh_height_above_ground_m`` = ``1.3 + base_z0_m``: SmartQSM measures DBH 1.3 m above the cloud's lowest point.
* ``dbh_m`` and ``height_m`` are EMPTY unless ``status == ok`` (``height_raw_m`` is still reported for ``no_dbh``).
* ``stem_volume_m3`` = ``treedata.stem_volume_L / 1000``: SmartQSM's "stem" is ALL woody parts (trunk + every branch);
  ``trunk_volume_m3`` = ``trunk_volume_L / 1000`` is the main trunk alone.
* ``status``: ``ok`` | ``no_dbh`` (``DBH_cm == 0``: the trunk does not reach 1.3 m above the lowest point) | ``failed``
  (no usable ``_qsm.mat`` although the container ran) | ``timeout`` | ``skipped_low_points`` (flagged in tree_map.csv;
  no cloud was written) | ``error`` (unreadable .mat, missing cloud, docker problem, ...) | ``suspect`` (the model is not
  plausible for this tree: woody volume more than ``IMPLAUSIBLE_VOLUME_RATIO`` times a DBH x height cylinder; its DBH and height
  are withheld). ``note`` explains non-ok rows.
* ``seconds``: wall time of the container run (a reused result keeps the value of the previous metrics.csv, else empty).

Resume
------
With ``resume=True`` (default) a tree whose ``work/<stem>_qsm.mat`` exists is not run again but still collected, provided that
(a) the .mat is readable, (b) ``work/<stem>.laz`` is byte-identical to the current split cloud, (c) ``config.yaml`` is unchanged.
Trees that previously failed/timed out have no .mat and are retried. A changed ``config.yaml`` next to existing results is a
``StageError`` (the results would belong to a different parametrisation) unless ``force=True``, which discards ``work/``.
``force=True`` (or ``resume=False``) re-runs every selected tree; stale outputs of a tree are always deleted before its run
so that an old .mat can never be mistaken for the result of a failed run.

metrics.csv merge
-----------------
``metrics.csv`` always describes the whole split, not only the last selection: ``smartqsm.trees=[5]`` replaces the row of tree 5 and
keeps every other row of the earlier runs (output sorted by ordinal, still written atomically after every tree). This is only
done when ``source_fingerprint.txt`` next to it equals the current one, i.e. the rows were made on the current ``tree_map.csv``
with the same ``config.yaml`` (and ``skip_low_points``): ``<hash of tree_map.csv>`` / ``config=<hash of config.yaml>
skip_low_points=true``. Rows of another split or configuration (or of a table without a fingerprint) are dropped with a WARNING and
the table starts empty, so that ``compare`` can never pair them with the new ordinals.
"""

from __future__ import annotations

import copy
import csv
import filecmp
import glob
import hashlib
import logging
import math
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lidar_bridge.docker_runner import DockerRunError, image_exists, require_image, run_container
from lidar_bridge.paths import RunContext
from lidar_bridge.qsm_mat import QsmMatError, read_treedata
from lidar_bridge.stages._common import StageError, read_fingerprint, tree_map_fingerprint, write_fingerprint

log = logging.getLogger("lidar_bridge.stages.smartqsm")

#: SmartQSM presets (and the source of the baked-in copies in the image), relative to the project directory.
CONFIGS_DIR = Path("third_party/SmartQSM/configs")
#: The CPU image (the default of ``image``) and the CUDA image that ``gpu=True`` selects while ``image`` is still the default.
DEFAULT_IMAGE = "lidar-smartqsm"
GPU_IMAGE = "lidar-smartqsm-gpu"
#: The GPU image is not part of a plain ``docker compose build`` (several GB), so a missing one needs this hint.
GPU_BUILD_HINT = "run_pipeline.bat --gpu --build-only   (or: docker compose --profile gpu build smartqsm-gpu)"
#: Suffix of the presets whose ``skeletonization.core_algorithm_kwargs.device`` is ``cuda``.
GPU_PRESET_SUFFIX = "-GPU"
#: SmartQSM measures DBH this far above the LOWEST point of the cloud.
BREAST_HEIGHT_M = 1.3
MAT_SUFFIX = "_qsm.mat"
#: Answer to the "check for updates?" prompt (belt and braces: --network none already makes the check fail fast).
STDIN_ANSWER = "no\n"

OK, NO_DBH, FAILED, TIMEOUT, SKIPPED, ERROR = "ok", "no_dbh", "failed", "timeout", "skipped_low_points", "error"
#: The model came out, but it cannot be a model of this tree (see IMPLAUSIBLE_VOLUME_RATIO).
SUSPECT = "suspect"
STATUSES = (OK, NO_DBH, FAILED, TIMEOUT, SKIPPED, ERROR, SUSPECT)
#: A QSM whose woody volume exceeds this many times the volume of a cylinder with the tree's own DBH and height is not a
#: model of that tree. Real trees give roughly 0.5-3 (tapered trunk plus branches); the one case seen so far was the
#: spconv LEAFON-GPU preset on a 13 m tree with 18 cm DBH: 1 branch and 397 m3, i.e. a ratio of about 4000.
IMPLAUSIBLE_VOLUME_RATIO = 30.0

METRICS_COLUMNS = [
    "ordinal",
    "dbh_m",
    "height_m",
    "status",
    "seconds",
    # extras
    "height_raw_m",
    "base_z0_m",
    "dbh_height_above_ground_m",
    "num_branches",
    "max_order",
    "stem_volume_m3",
    "trunk_volume_m3",
    "n_points",
    "note",
]
_NAN = float("nan")
_SAFE_NAME = r"^\w[\w.+=\-]*$"  # same rule as RunContext.run_name; no leading dot, no slashes


# ---------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tree:
    """One row of ``split/tree_map.csv`` (only what this stage needs)."""

    ordinal: int
    n_points: int | None = None
    base_z0_m: float = _NAN
    low_points: bool = False
    file: str = ""  # cloud path relative to the split dir; '' if none was written


@dataclass
class TreeResult:
    """One row of metrics.csv."""

    ordinal: int
    status: str
    seconds: float = _NAN
    dbh_m: float = _NAN
    height_m: float = _NAN
    height_raw_m: float = _NAN
    base_z0_m: float = _NAN
    dbh_height_above_ground_m: float = _NAN
    num_branches: int | None = None
    max_order: int | None = None
    stem_volume_m3: float = _NAN
    trunk_volume_m3: float = _NAN
    n_points: int | None = None
    note: str = ""

    def csv_row(self) -> list[str]:
        return [
            str(self.ordinal),
            _fmt(self.dbh_m),
            _fmt(self.height_m),
            self.status,
            _fmt(self.seconds, ".1f"),
            _fmt(self.height_raw_m),
            _fmt(self.base_z0_m),
            _fmt(self.dbh_height_above_ground_m),
            _fmt(self.num_branches),
            _fmt(self.max_order),
            _fmt(self.stem_volume_m3),
            _fmt(self.trunk_volume_m3),
            _fmt(self.n_points),
            self.note,
        ]


def _fmt(value: Any, spec: str = ".6g") -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, spec) if math.isfinite(value) else ""
    return str(value)


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(value)


# ---------------------------------------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------------------------------------
def check_config_name(config: str) -> str:
    """The preset name is also a directory name: accept ``layerwise-clustering`` (a trailing ``.yaml`` is dropped)."""
    name = str(config).strip()
    if name.lower().endswith((".yaml", ".yml")):
        name = name.rsplit(".", 1)[0]
    if not re.match(_SAFE_NAME, name) or name.endswith(".."):
        raise StageError(f"invalid smartqsm.config {config!r}: use the name of a preset in {CONFIGS_DIR} (no slashes or spaces)")
    return name


def available_presets(configs_dir: Path) -> list[str]:
    return sorted(p.stem for p in configs_dir.glob("*.y*ml") if p.suffix in (".yaml", ".yml"))


def load_preset(configs_dir: Path, name: str) -> tuple[dict, str, set[str]]:
    """``(config dict, raw text, top-level keys used by any preset)`` of preset ``name``."""
    path = next((configs_dir / f"{name}{ext}" for ext in (".yaml", ".yml") if (configs_dir / f"{name}{ext}").is_file()), None)
    if path is None:
        if not configs_dir.is_dir():
            raise StageError(
                f"SmartQSM presets not found: {configs_dir}. Is the submodule checked out? "
                "(git submodule update --init third_party/SmartQSM)"
            )
        raise StageError(f"unknown SmartQSM config {name!r}; available presets: {available_presets(configs_dir)}")
    text = path.read_text(encoding="utf-8")
    try:
        base = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise StageError(f"cannot parse SmartQSM preset {path}: {exc}") from exc
    if not isinstance(base, dict):
        raise StageError(f"SmartQSM preset {path} is not a mapping")
    known: set[str] = set(base)
    for other in configs_dir.glob("*.y*ml"):
        try:
            data = yaml.safe_load(other.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict):
            known.update(data)
    return base, text, known


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any] | None, known_keys: set[str] = frozenset()) -> tuple[dict, list[str]]:
    """Apply ``overrides`` to a deep copy of ``base``; returns ``(merged, human readable change list)``.

    * A plain key is a SHALLOW ``dict.update`` of the top-level preset (this is also how SmartQSM applies a sidecar yaml): the
      value REPLACES the preset's value, so overriding the block ``skeletonization`` needs the complete block. A warning is
      logged when a replaced block loses keys.
    * A dotted key (``skeletonization.core_algorithm_kwargs.max_patch_size``) sets one value inside nested blocks; the path
      must exist in the preset.
    * Top-level keys that no shipped preset uses are rejected (a typo would be silently ignored by SmartQSM).
    """
    merged = copy.deepcopy(dict(base))
    changes: list[str] = []
    for key, value in (overrides or {}).items():
        key = str(key)
        if "." in key:
            *parents, leaf = key.split(".")
            node: Any = merged
            for part in parents:
                node = node.get(part) if isinstance(node, dict) else None
                if not isinstance(node, dict):
                    raise StageError(f"overrides: {key!r}: {'.'.join(parents)!r} is not a block of the preset")
            if leaf not in node:
                raise StageError(f"overrides: {key!r}: no key {leaf!r} in that block; keys: {sorted(node)}")
            changes.append(f"{key}: {node[leaf]!r} -> {value!r}")
            node[leaf] = copy.deepcopy(value)
            continue
        if key not in merged and key not in known_keys:
            raise StageError(
                f"overrides: unknown SmartQSM key {key!r} (typo?); top-level keys of the preset: {sorted(merged)}"
            )
        old = merged.get(key)
        if isinstance(old, dict) and isinstance(value, Mapping):
            lost = sorted(set(old) - set(value))
            if lost:
                log.warning(
                    "override of block %r REPLACES it (shallow update): preset keys %s are gone. To change single values "
                    "use dotted keys, e.g. '%s.%s'",
                    key,
                    lost,
                    key,
                    lost[0],
                )
        changes.append(f"{key}: {old!r} -> {value!r}" if key in merged else f"{key}: (new) -> {value!r}")
        merged[key] = copy.deepcopy(dict(value) if isinstance(value, Mapping) else value)
    return merged, changes


def render_config(base_text: str, merged: Mapping[str, Any], changed: bool) -> str:
    """Text of the per-run config.yaml: the preset verbatim if nothing changed, else the merged mapping dumped as yaml."""
    if not changed:
        return base_text
    try:
        return yaml.safe_dump(dict(merged), sort_keys=False, default_flow_style=False, allow_unicode=True)
    except yaml.YAMLError as exc:
        raise StageError(f"overrides cannot be written as yaml: {exc}") from exc


# ---------------------------------------------------------------------------------------------------------
# tree_map.csv
# ---------------------------------------------------------------------------------------------------------
def _to_float(text: str | None) -> float:
    try:
        return float(text) if text not in (None, "") else _NAN  # type: ignore[arg-type]
    except ValueError:
        return _NAN


def _to_int(text: str | None) -> int | None:
    value = _to_float(text)
    return int(value) if math.isfinite(value) else None


def _to_bool(text: str | None) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes")


def read_tree_map(path: Path) -> list[Tree]:
    """Parse ``split/tree_map.csv`` (only ``ordinal`` is required; the other columns have defaults)."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if "ordinal" not in (reader.fieldnames or []):
            raise StageError(f"{path} has no 'ordinal' column (columns: {reader.fieldnames}); was it written by the split stage?")
        rows = list(reader)
    width = max(4, len(str(max((_to_int(r["ordinal"]) or 0 for r in rows), default=0))))
    trees = []
    for row in rows:
        ordinal = _to_int(row["ordinal"])
        if ordinal is None:
            raise StageError(f"{path}: bad ordinal {row['ordinal']!r}")
        trees.append(
            Tree(
                ordinal=ordinal,
                n_points=_to_int(row.get("n_points")),
                base_z0_m=_to_float(row.get("base_z0_m")),
                low_points=_to_bool(row.get("low_points")),
                file=(row["file"].strip() if "file" in row and row["file"] is not None else f"trees/tree_{ordinal:0{width}d}.laz"),
            )
        )
    return trees


def select_trees(all_trees: list[Tree], *, trees: list[int] | None, limit: int | None, skip_low_points: bool) -> list[Tree]:
    """Trees of this invocation in ordinal order.

    ``trees``: only these ordinals (unknown ones are an error). ``limit``: stop after that many trees that will really
    be processed; low-point trees met on the way are kept (they are recorded as ``skipped_low_points``) but do not count.
    """
    by_ordinal = {t.ordinal: t for t in all_trees}
    if trees is not None:
        wanted_ordinals = sorted({int(o) for o in trees})
        if not wanted_ordinals:
            raise StageError("smartqsm.trees is empty; omit it to process every tree")
        unknown = [o for o in wanted_ordinals if o not in by_ordinal]
        if unknown:
            raise StageError(f"smartqsm.trees: ordinal(s) {unknown} are not in tree_map.csv (it has {len(by_ordinal)} trees)")
        wanted = [by_ordinal[o] for o in wanted_ordinals]
    else:
        wanted = sorted(all_trees, key=lambda t: t.ordinal)
    if limit is not None and limit < 1:
        raise StageError(f"smartqsm.limit must be a positive number of trees, got {limit}")
    selected: list[Tree] = []
    processed = 0
    for tree in wanted:
        if skip_low_points and tree.low_points:
            selected.append(tree)
            continue
        if limit is not None and processed >= limit:
            break
        selected.append(tree)
        processed += 1
    return selected


# ---------------------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------------------
def result_from_treedata(tree: Tree, treedata: Mapping[str, float], seconds: float = _NAN, note: str = "") -> TreeResult:
    """Metrics row of a tree whose .mat was read successfully (status ok / no_dbh / error)."""
    result = TreeResult(ordinal=tree.ordinal, status=OK, seconds=seconds, base_z0_m=tree.base_z0_m, n_points=tree.n_points, note=note)
    dbh_cm, raw_height = treedata.get("DBH_cm"), treedata.get("height_m")
    if not _finite(dbh_cm) or not _finite(raw_height):
        result.status = ERROR
        result.note = _join(note, "treedata has no usable DBH_cm / height_m")
        return result
    base = tree.base_z0_m
    if not _finite(base):
        base = 0.0
        result.note = _join(result.note, "base_z0_m unknown: height and DBH height not corrected for the ground offset")
    else:
        result.dbh_height_above_ground_m = BREAST_HEIGHT_M + base
    result.height_raw_m = float(raw_height)
    result.num_branches = _as_int(treedata.get("num_branches"))
    result.max_order = _as_int(treedata.get("max_order"))
    for column, field in (("stem_volume_m3", "stem_volume_L"), ("trunk_volume_m3", "trunk_volume_L")):
        litres = treedata.get(field)
        if _finite(litres):
            setattr(result, column, litres / 1000.0)
    if dbh_cm <= 0:
        result.status = NO_DBH
        result.note = _join(result.note, f"DBH_cm == 0: the trunk does not reach {BREAST_HEIGHT_M} m above the lowest point")
        return result
    result.dbh_m = dbh_cm / 100.0
    result.height_m = float(raw_height) + base
    if _finite(result.stem_volume_m3) and result.height_m > 0:
        cylinder_m3 = math.pi / 4.0 * result.dbh_m**2 * result.height_m
        if result.stem_volume_m3 > IMPLAUSIBLE_VOLUME_RATIO * cylinder_m3:
            result.note = _join(
                result.note,
                f"implausible model: woody volume {result.stem_volume_m3:.3g} m3 is {result.stem_volume_m3 / cylinder_m3:.0f}x "
                f"a cylinder of DBH {dbh_cm:.1f} cm and height {result.height_m:.1f} m"
                f"{f' ({result.num_branches} branches)' if result.num_branches is not None else ''}; "
                "the QSM probably failed, try another smartqsm.config",
            )
            result.status = SUSPECT
            result.dbh_m = _NAN
            result.height_m = _NAN
    return result


def _as_int(value: float | None) -> int | None:
    return int(value) if _finite(value) else None


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def write_metrics(path: Path, results: Mapping[int, TreeResult], kept_rows: Mapping[int, list[str]] | None = None) -> None:
    """Atomically (temp file + rename) write metrics.csv with the rows sorted by ordinal.

    ``kept_rows`` are rows (already formatted, in :data:`METRICS_COLUMNS` order) of earlier runs; a result for the same
    ordinal replaces the kept row.
    """
    rows = dict(kept_rows or {})
    rows.update({ordinal: result.csv_row() for ordinal, result in results.items()})
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(METRICS_COLUMNS)
        for ordinal in sorted(rows):
            writer.writerow(rows[ordinal])
    os.replace(tmp, path)


def read_previous_rows(path: Path) -> dict[int, list[str]]:
    """``{ordinal: row}`` of an existing metrics.csv, the rows in :data:`METRICS_COLUMNS` order (``{}`` if unreadable)."""
    out: dict[int, list[str]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ordinal = _to_int(row.get("ordinal"))
                if ordinal is not None:
                    out[ordinal] = [row.get(column) or "" for column in METRICS_COLUMNS]
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    return out


def settings_tag(config_text: str, skip_low_points: bool) -> str:
    """The settings besides the split that the rows of metrics.csv depend on (``extra`` of the fingerprint)."""
    digest = hashlib.sha256(config_text.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:8]
    return f"config={digest} skip_low_points={'true' if skip_low_points else 'false'}"


def rows_to_keep(metrics_path: Path, out_dir: Path, fingerprint: str, tag: str) -> dict[int, list[str]]:
    """Rows of an existing metrics.csv that a new run may keep: all of them if the stored fingerprint equals the current
    ``(fingerprint, tag)``, else none (a WARNING says why)."""
    old = read_previous_rows(metrics_path)
    if not old:
        return {}
    stored = read_fingerprint(out_dir)
    if stored == (fingerprint, tag):
        return old
    if stored is None:
        why = "the table has no source fingerprint, so the split it belongs to is unknown"
        kind = "a different split"
    elif stored[0] != fingerprint:
        why = "split/tree_map.csv changed"
        kind = "a different split"
    else:
        why = f"settings changed (was: {stored[1] or '-'}; now: {tag})"
        kind = "a different configuration"
    log.warning("smartqsm: older rows from %s were dropped (%d row(s) in %s: %s); the table starts with this run's trees only",
                kind, len(old), metrics_path.name, why)  # fmt: skip
    return {}


def read_previous_seconds(path: Path) -> dict[int, float]:
    """``{ordinal: seconds}`` of an existing metrics.csv (to keep the timing of reused results)."""
    out: dict[int, float] = {}
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seconds, ordinal = _to_float(row.get("seconds")), _to_int(row.get("ordinal"))
                if ordinal is not None and math.isfinite(seconds):
                    out[ordinal] = seconds
    except (OSError, UnicodeDecodeError, csv.Error):
        pass
    return out


def format_eta(seconds: float) -> str:
    """``2:31 h`` for >= 1 h, ``4:05 min`` below."""
    seconds = max(0, int(round(seconds)))
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d} h"
    return f"{seconds // 60}:{seconds % 60:02d} min"


def _tail(text: str, lines: int = 3, width: int = 300) -> str:
    kept = [ln.strip() for ln in (text or "").splitlines() if ln.strip()][-lines:]
    joined = " | ".join(kept)
    return joined if len(joined) <= width else "..." + joined[-width:]


# ---------------------------------------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------------------------------------
def preset_device(config: Mapping[str, Any]) -> str | None:
    """``skeletonization.core_algorithm_kwargs.device`` of a (merged) preset, ``None`` if it has none."""
    skeletonization = config.get("skeletonization")
    kwargs = skeletonization.get("core_algorithm_kwargs") if isinstance(skeletonization, Mapping) else None
    device = kwargs.get("device") if isinstance(kwargs, Mapping) else None
    return str(device).strip().lower() if device is not None else None


def uses_cuda(name: str, config: Mapping[str, Any] | None) -> bool:
    """True if the preset runs on the GPU: ``device: cuda`` in the (merged) preset, else the ``-GPU`` name suffix."""
    device = preset_device(config) if config is not None else None
    if device is not None:
        return device.startswith("cuda")
    return name.endswith(GPU_PRESET_SUFFIX)


def check_gpu_setup(name: str, config: Mapping[str, Any] | None, gpu: bool) -> None:
    """Refuse a GPU preset without ``gpu=True``; warn when ``gpu=True`` is combined with a preset that cannot use CUDA."""
    cuda = uses_cuda(name, config)
    if cuda and not gpu:
        raise StageError(
            f"smartqsm.config {name!r} runs on the GPU (device: cuda) but smartqsm.gpu is false, so the container would get no "
            "GPU and SmartQSM would fail. Set smartqsm.gpu=true (run_pipeline.bat --gpu), or use a CPU preset such as "
            "spconv-contraction-LEAFON-cpu."
        )
    if gpu and not cuda:
        log.warning(
            "smartqsm.gpu=true, but the preset %r does not use CUDA, so SmartQSM will run on the CPU (the GPU is only used by "
            "the spconv-contraction-*-GPU presets, e.g. smartqsm.config=spconv-contraction-LEAFOFF-GPU)",
            name,
        )


def check_image(image: str, *, dry_run: bool) -> None:
    """The image must exist; the GPU image gets a pointer to the command that builds it (it is not in a plain build)."""
    if image != GPU_IMAGE:
        require_image(image, dry_run=dry_run)
        return
    try:
        found = image_exists(image)
    except DockerRunError as exc:
        if not dry_run:
            raise
        log.warning("[dry-run] cannot check for image %r: %s", image, exc)
        return
    if found:
        return
    message = f"docker image {image!r} not found. The GPU image is built on request only (several GB): {GPU_BUILD_HINT}"
    if dry_run:
        log.warning("[dry-run] %s", message)
        return
    raise StageError(message)


#: (pattern in the container output, short explanation for the ``note`` column), most specific first.
_GPU_DIAGNOSES = (
    (re.compile(r"out of memory", re.I), "CUDA out of memory: the tree does not fit into the GPU's free VRAM (try a smaller "
     "skeletonization.core_algorithm_kwargs.batch_size or a -cpu preset)"),
    (re.compile(r"could not select device driver|nvidia-container-cli|unknown or invalid runtime name: nvidia", re.I),
     "Docker cannot give the container a GPU (--gpus all failed): needs an NVIDIA driver and Docker Desktop WSL2 GPU support, "
     "check `docker run --rm --gpus all lidar-bridge nvidia-smi`"),
    (re.compile(r"no kernel image is available|CUDA error|CUDA driver|libcuda|cudaErrorNoDevice|No CUDA GPUs", re.I),
     "CUDA is not usable in the container (see the log for the CUDA error)"),
)  # fmt: skip


def diagnose_gpu_failure(output: str) -> str:
    """One-line reason for a failed GPU run recognised in the container output (``""`` if nothing is recognised)."""
    for pattern, text in _GPU_DIAGNOSES:
        if pattern.search(output or ""):
            return text
    return ""


# ---------------------------------------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------------------------------------
@dataclass
class _Env:
    ctx: RunContext
    image: str
    gpu: bool
    timeout_s: int
    split_dir: Path
    out_dir: Path
    work_dir: Path
    config_path: Path
    rerun_all: bool
    skip_low_points: bool
    previous_seconds: dict[int, float]

    def command(self, laz: Path) -> tuple[list[str], list]:
        args = ["-t", "-y", "-c", self.ctx.container_path(self.config_path), self.ctx.container_path(laz)]
        return args, [self.ctx.mount(self.ctx.data_dir, "/data")]


def _clean_outputs(work_dir: Path, stem: str) -> None:
    """Delete everything SmartQSM wrote for ``stem`` (``<stem>_*``), never the cloud itself."""
    for leftover in work_dir.glob(f"{glob.escape(stem)}_*"):
        if leftover.is_file():
            leftover.unlink()


def _write_log(env: _Env, stem: str, output: str) -> None:
    try:
        logs = env.out_dir / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{stem}.log").write_text(output or "", encoding="utf-8")
    except OSError as exc:  # a missing log must never fail the tree
        log.warning("cannot write the container log of %s: %s", stem, exc)


def _reuse_existing(env: _Env, tree: Tree, src: Path, dst: Path, mat: Path) -> TreeResult | None:
    """Result for a tree with an existing, consistent ``_qsm.mat`` (resume), else ``None``."""
    if env.rerun_all or not mat.is_file():
        return None
    if not dst.is_file() or not filecmp.cmp(src, dst, shallow=False):
        log.info("tree %d: %s exists but %s differs from the current split cloud: re-running", tree.ordinal, mat.name, dst.name)
        return None
    try:
        treedata = read_treedata(mat)
    except QsmMatError as exc:
        log.warning("tree %d: existing %s is unusable (%s): re-running", tree.ordinal, mat.name, exc)
        return None
    return result_from_treedata(tree, treedata, env.previous_seconds.get(tree.ordinal, _NAN), note="reused existing _qsm.mat")


def _process_tree(env: _Env, tree: Tree) -> tuple[TreeResult, bool]:
    """One tree: reuse / run the container / collect; ``ran`` is True if the container was started.

    Never raises for tree-level problems (they become status ``error``).
    """
    if env.skip_low_points and tree.low_points:
        return TreeResult(
            ordinal=tree.ordinal, status=SKIPPED, base_z0_m=tree.base_z0_m, n_points=tree.n_points,
            note="flagged low_points in tree_map.csv (too few points, no cloud written)",
        ), False  # fmt: skip
    src = env.split_dir / tree.file if tree.file else None
    if src is None or not src.is_file():
        return TreeResult(
            ordinal=tree.ordinal, status=ERROR, base_z0_m=tree.base_z0_m, n_points=tree.n_points,
            note=f"tree cloud not found: {tree.file or '(no file in tree_map.csv)'}",
        ), False  # fmt: skip
    dst = env.work_dir / src.name
    mat = env.work_dir / f"{dst.stem}{MAT_SUFFIX}"

    reused = _reuse_existing(env, tree, src, dst, mat)
    if reused is not None:
        return reused, False

    _clean_outputs(env.work_dir, dst.stem)  # a stale .mat must not be mistaken for the result of this run
    if not (dst.is_file() and filecmp.cmp(src, dst, shallow=False)):
        shutil.copyfile(src, dst)
    args, mounts = env.command(dst)

    base = dict(ordinal=tree.ordinal, base_z0_m=tree.base_z0_m, n_points=tree.n_points)
    started = time.monotonic()
    try:
        proc = run_container(
            env.image, args, mounts,
            network="none", stdin=STDIN_ANSWER, gpus=env.gpu, timeout_s=env.timeout_s or None,
            log_prefix="smartqsm", check=False,
        )  # fmt: skip
    except DockerRunError as exc:
        seconds = time.monotonic() - started
        _write_log(env, dst.stem, exc.output)
        timed_out = "timed out" in str(exc)
        return TreeResult(
            status=TIMEOUT if timed_out else ERROR, seconds=seconds,
            note=f"killed after the {env.timeout_s} s timeout" if timed_out else _tail(str(exc)),
            **base,
        ), True  # fmt: skip
    seconds = time.monotonic() - started
    output = proc.stdout or ""
    _write_log(env, dst.stem, output)

    exit_note = f"container exit code {proc.returncode}" if proc.returncode else ""
    if not mat.is_file() or mat.stat().st_size == 0:
        reason = diagnose_gpu_failure(output) if env.gpu else ""
        return TreeResult(
            status=FAILED, seconds=seconds,
            note=_join(exit_note, f"no {mat.name} produced", reason or _tail(output)),  # logs/ keeps the whole output
            **base,
        ), True  # fmt: skip
    try:
        treedata = read_treedata(mat)
    except QsmMatError as exc:
        return TreeResult(status=ERROR, seconds=seconds, note=_join(exit_note, str(exc)), **base), True
    return result_from_treedata(tree, treedata, seconds, note=exit_note), True


def _prepare_out_dir(out_dir: Path, work_dir: Path, config_text: str, force: bool) -> Path:
    """Create the output dirs and write ``config.yaml``; guard against reusing results of a different config."""
    config_path = out_dir / "config.yaml"
    if config_path.is_file() and work_dir.is_dir() and any(work_dir.glob(f"*{MAT_SUFFIX}")):
        old = config_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if old != config_text.replace("\r\n", "\n"):
            if not force:
                raise StageError(
                    f"{config_path} differs from the config the results in {work_dir} were made with, so they must not be "
                    "reused. Use force=true to discard them (smartqsm.force=true), or another smartqsm.config / run_name."
                )
            log.warning("force: the config changed, discarding the old results in %s", work_dir)
            shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8", newline="\n")
    return config_path


def run_smartqsm(
    ctx: RunContext,
    *,
    config: str = "layerwise-clustering",
    overrides: dict = {},  # noqa: B006  (never mutated; hydra-zen turns it into a config field)
    image: str = DEFAULT_IMAGE,
    gpu: bool = False,
    timeout_s: int = 1800,
    trees: list[int] | None = None,
    limit: int | None = None,
    skip_low_points: bool = True,
    resume: bool = True,
    force: bool = False,
) -> dict:
    """Run SmartQSM (image ``lidar-smartqsm``, or ``lidar-smartqsm-gpu`` with ``gpu``) on every tree of ``split/`` and write
    ``smartqsm/<config>/metrics.csv``.

    Parameters
    ----------
    config
        Name of a preset in ``third_party/SmartQSM/configs`` (``layerwise-clustering`` ~85 s per tree, ``space-colonization``,
        ``spconv-contraction-LEAFOFF-cpu`` ...). Sweepable (``-m smartqsm.config=a,b``); output goes to ``smartqsm/<config>/``.
        The ``spconv-contraction-*-GPU`` presets need ``gpu=True`` (refused otherwise).
    overrides
        Changes to the preset: top-level YAML keys (``voxel_down_size``, ``filter``, ``kwargs_for_filter``, ``skeletonization``,
        ``refinement``, ``modeling``) REPLACE the preset's value (shallow ``dict.update``); a dotted key such as
        ``skeletonization.core_algorithm_kwargs.max_patch_size`` changes a single nested value. CLI:
        ``+smartqsm.overrides.voxel_down_size=0.02``. Different overrides need a different ``run_name`` (or ``force``).
    image
        Docker image tag. While it is left at the default ``lidar-smartqsm``, ``gpu=True`` uses ``lidar-smartqsm-gpu`` instead.
    gpu
        Run with ``--gpus all`` in the CUDA image (build it with ``run_pipeline.bat --gpu --build-only``; a missing image is a
        :class:`StageError` that says so). Only the ``spconv-contraction-*-GPU`` presets use the card; with another preset a
        warning says that SmartQSM runs on the CPU.
    timeout_s
        Per tree, seconds (0 = no limit); a tree over the limit gets status ``timeout``.
    trees
        Only these tree ordinals (default: all); their rows are merged into the existing metrics.csv of the same split (the rows
        of the other trees are kept). ``limit``: only the first N trees that would be processed.
    skip_low_points
        Do not run trees flagged ``low_points`` in tree_map.csv (recorded as ``skipped_low_points``).
    resume
        Reuse an existing ``work/<stem>_qsm.mat`` instead of running the tree again (see the module docstring).
    force
        Re-run every selected tree (and discard results of a different config).

    Returns
    -------
    dict
        ``{"dir", "config", "metrics", "n_selected", "counts": {status: n}, "seconds"}``; with ``dry_run`` only the
        selection is reported. Raises :class:`StageError` if every tree that was attempted failed.
    """
    started = time.monotonic()
    log.info(
        "smartqsm stage parameters: %s",
        dict(config=config, overrides=overrides, image=image, gpu=gpu, timeout_s=timeout_s, trees=trees, limit=limit,
             skip_low_points=skip_low_points, resume=resume, force=force),
    )  # fmt: skip
    name = check_config_name(config)
    if gpu and image == DEFAULT_IMAGE:
        image = GPU_IMAGE
        log.info("smartqsm.gpu=true: using the image %r with --gpus all", image)
    split_dir = ctx.stage_dir("split")
    out_dir = ctx.stage_dir("smartqsm") / name
    work_dir = out_dir / "work"
    metrics_path = out_dir / "metrics.csv"
    tree_map = split_dir / "tree_map.csv"

    # -- per-run config.yaml -----------------------------------------------------------------------------------
    config_text: str | None
    try:
        base, base_text, known = load_preset(ctx.project_dir / CONFIGS_DIR, name)
        merged, changes = merge_overrides(base, overrides, known)
        config_text = render_config(base_text, merged, changed=bool(overrides))
        log.info(
            "SmartQSM config %r%s", name,
            " with overrides:\n  " + "\n  ".join(changes) if changes else " (preset unchanged)",
        )  # fmt: skip
    except StageError as exc:
        if not ctx.dry_run:
            raise
        log.warning("[dry-run] %s", exc)
        config_text = None
        merged = None
    check_gpu_setup(name, merged, gpu)  # also in a dry run: a wrong gpu/preset pair would fail the real run immediately

    # -- trees ---------------------------------------------------------------------------------------------------
    if tree_map.is_file():
        all_trees = read_tree_map(tree_map)
    elif ctx.dry_run:
        log.warning("[dry-run] %s does not exist yet; showing the commands for placeholder trees", tree_map)
        placeholders = sorted({int(o) for o in trees}) if trees else [1]  # explicit selections stay valid in a rehearsal
        all_trees = [Tree(ordinal=o, file=f"trees/tree_{o:04d}.laz") for o in placeholders]
    else:
        raise StageError(f"{tree_map} not found: run the split stage first (stages=[dfin,split,smartqsm])")
    selected = select_trees(all_trees, trees=trees, limit=limit, skip_low_points=skip_low_points)
    if not selected:
        raise StageError(f"no trees to process ({len(all_trees)} in {tree_map.name})")

    check_image(image, dry_run=ctx.dry_run)

    rerun_all = force or not resume
    config_path = out_dir / "config.yaml"
    env = _Env(ctx, image, gpu, timeout_s, split_dir, out_dir, work_dir, config_path, rerun_all, skip_low_points, {})

    # -- dry run: log the commands, touch nothing -----------------------------------------------------------------
    if ctx.dry_run:
        log.info("[dry-run] would write %s and %s, and process %d tree(s) one after the other", config_path, metrics_path, len(selected))
        for tree in selected:
            if skip_low_points and tree.low_points:
                log.info("[dry-run] tree %d: skipped (low_points)", tree.ordinal)
                continue
            args, mounts = env.command(work_dir / Path(tree.file or f"tree_{tree.ordinal:04d}.laz").name)
            run_container(
                image, args, mounts,
                network="none", stdin=STDIN_ANSWER, gpus=gpu, timeout_s=timeout_s or None, dry_run=True,
            )  # fmt: skip
        return {"dir": str(ctx.rel(out_dir)), "config": name, "image": image, "gpu": gpu, "n_selected": len(selected), "dry_run": True}

    assert config_text is not None
    env.config_path = _prepare_out_dir(out_dir, work_dir, config_text, force)
    env.previous_seconds = read_previous_seconds(metrics_path)
    # metrics.csv describes the whole split: this run's rows are merged into those of earlier runs on the SAME split and config
    fingerprint = tree_map_fingerprint(split_dir)
    assert fingerprint is not None  # tree_map.csv exists (checked above, no dry run here)
    tag = settings_tag(config_text, skip_low_points)
    kept_rows = rows_to_keep(metrics_path, out_dir, fingerprint, tag)
    results: dict[int, TreeResult] = {}

    def save_metrics() -> None:
        write_metrics(metrics_path, results, kept_rows)
        write_fingerprint(out_dir, fingerprint, extra=tag)  # after the table: a crash in between only costs the kept rows

    def needs_run(tree: Tree) -> bool:
        if skip_low_points and tree.low_points:
            return False
        return rerun_all or not (work_dir / f"{Path(tree.file).stem}{MAT_SUFFIX}").is_file()

    # -- the loop -------------------------------------------------------------------------------------------------------
    run_seconds: list[float] = []
    total = len(selected)
    try:
        for index, tree in enumerate(selected, 1):
            result, ran = _process_tree(env, tree)
            results[tree.ordinal] = result
            if ran:
                run_seconds.append(result.seconds)
            try:
                save_metrics()  # after EVERY tree: an interrupted run keeps what it has
            except OSError as exc:
                log.warning("cannot update %s (%s); will retry after the next tree", metrics_path, exc)
            remaining = sum(1 for t in selected[index:] if needs_run(t))
            eta = f", eta {format_eta(sum(run_seconds) / len(run_seconds) * remaining)}" if run_seconds and remaining else ""
            took = f" {result.seconds:.0f} s" if math.isfinite(result.seconds) and result.status != SKIPPED else ""
            detail = f" ({result.note})" if result.note else ""
            log.info("[%d/%d] tree %d %s%s%s%s", index, total, tree.ordinal, result.status, took, eta, detail)
    finally:
        if results:
            try:
                save_metrics()
            except OSError as exc:
                log.error("cannot write %s: %s", metrics_path, exc)

    counts = Counter(r.status for r in results.values())
    elapsed = time.monotonic() - started
    log.info(
        "SmartQSM %r finished in %s: %s -> %s", name, format_eta(elapsed),
        ", ".join(f"{status} {counts[status]}" for status in STATUSES if counts[status]),
        f"{ctx.rel(metrics_path)} ({len(set(kept_rows) | set(results))} rows)",
    )  # fmt: skip
    attempted = sum(counts[s] for s in (OK, NO_DBH, FAILED, TIMEOUT, ERROR))
    broken = counts[FAILED] + counts[TIMEOUT] + counts[ERROR]
    if attempted and broken == attempted:
        raise StageError(
            f"SmartQSM produced no result for any of the {attempted} tree(s) tried ({dict(counts)}); see "
            f"{ctx.rel(out_dir / 'logs')}/ and the 'note' column of {ctx.rel(metrics_path)}"
        )
    return {
        "dir": str(ctx.rel(out_dir)),
        "config": name,
        "image": image,
        "gpu": gpu,
        "config_file": str(ctx.rel(env.config_path)),
        "metrics": str(ctx.rel(metrics_path)),
        "n_selected": total,
        "counts": {status: counts[status] for status in STATUSES if counts[status]},
        "seconds": round(elapsed, 1),
    }

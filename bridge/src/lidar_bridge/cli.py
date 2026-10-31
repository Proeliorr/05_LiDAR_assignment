"""Command line of the pipeline: the arguments become keyword arguments of the ``run_<stage>`` functions."""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from lidar_bridge import runner
from lidar_bridge.paths import DEFAULT_DATA_DIR
from lidar_bridge.pipeline import EXIT_INTERRUPTED, PipelineError, install_termination_handler, run_pipeline, select_stages
from lidar_bridge.stages import STAGE_FUNCTIONS, STAGE_ORDER
from lidar_bridge.stages.smartqsm import DEFAULT_CONFIG, GPU_PRESET_SUFFIX

EPILOG = """\
paths are relative to the data folder mounted at /data; results go to /data/out/<run>/.

examples:
  docker run --rm --init --gpus all --network none -v D:\\path\\data:/data lidar-pipeline in/plot_01.laz
  ... lidar-pipeline in/plot_01.laz --ref in/reference.csv
  ... lidar-pipeline --stages compare --run plot_01 --ref in/reference.csv
  ... lidar-pipeline --stages smartqsm,arbor --run plot_01 --trees 3,8
  ... lidar-pipeline in/plot_01.laz --params in/params.yaml
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lidar-pipeline",
        description="3DFin -> split into trees -> SmartQSM and arbor -> compare with a reference table.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="point cloud (.las or .laz); needed when the dfin stage runs")
    parser.add_argument("--ref", help="reference table (.csv or .xlsx); without it the compare stage is skipped")
    parser.add_argument("--run", help="run name, the folder out/<run> (default: the file name of the input)")
    parser.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGE_ORDER)} (the order is fixed)")
    parser.add_argument("--trees", help="only these tree numbers in smartqsm and arbor, e.g. 3,8")
    parser.add_argument("--limit", type=int, help="only the first N trees in smartqsm and arbor")
    parser.add_argument("--smartqsm-config", help=f"SmartQSM preset (default: {DEFAULT_CONFIG})")
    parser.add_argument("--force", action="store_true", help="re-run the selected trees in smartqsm and arbor, discarding old results")
    parser.add_argument("--dry-run", action="store_true", help="log every command and run none of them")
    parser.add_argument("--params", help="YAML file with any other stage parameter: {dfin: {...}, split: {...}, smartqsm: {...}, ...}")
    return parser


def data_dir_from_env() -> Path:
    return Path(os.environ.get("LIDAR_DATA_DIR", str(DEFAULT_DATA_DIR)))


def load_params(path: Path) -> dict[str, dict[str, Any]]:
    """Read and check the ``--params`` file: every key must be a stage, and every key under it a parameter of that stage."""
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a mapping of stage names to parameters")
    params: dict[str, dict[str, Any]] = {}
    for stage, values in loaded.items():
        if stage not in STAGE_FUNCTIONS:
            raise ValueError(f"{path}: unknown stage {stage!r}; stages are {list(STAGE_ORDER)}")
        if not isinstance(values, dict):
            raise ValueError(f"{path}: the parameters of {stage} must be a mapping")
        known = [name for name in inspect.signature(STAGE_FUNCTIONS[stage]).parameters if name != "ctx"]
        unknown = [name for name in values if name not in known]
        if unknown:
            raise ValueError(f"{path}: unknown parameter(s) {unknown} of stage {stage}; its parameters are {known}")
        params[stage] = values
    return params


def parse_trees(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"--trees must be comma-separated tree numbers, got {text!r}") from None


def stage_params(args: argparse.Namespace, from_file: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parameters per stage: the ``--params`` file, then the command-line flags on top of it."""
    params = {name: dict(from_file.get(name, {})) for name in STAGE_ORDER}
    for stage in ("smartqsm", "arbor"):
        if args.trees:
            params[stage]["trees"] = parse_trees(args.trees)
        if args.limit is not None:
            params[stage]["limit"] = args.limit
        if args.force:
            params[stage]["force"] = True
    if args.smartqsm_config:
        params["smartqsm"]["config"] = args.smartqsm_config
    return params


def check_gpu(params: dict[str, dict[str, Any]], selected: list[str]) -> None:
    """Stop before the first stage when a GPU preset would only fail later, after hours of work."""
    if "smartqsm" not in selected:
        return
    config = str(params["smartqsm"].get("config", DEFAULT_CONFIG))
    if config.endswith(GPU_PRESET_SUFFIX) and not runner.gpu_visible():
        raise PipelineError(
            f"the SmartQSM preset {config!r} needs a GPU, but the container sees none. Start it with --gpus all (this needs an "
            "NVIDIA driver and Docker Desktop's WSL2 GPU support), or choose a CPU preset: --smartqsm-config space-colonization"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and run the pipeline. Exit codes: 0 success, 1 error, 2 bad command line, 130 interrupted."""
    install_termination_handler()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input and not args.run:
        parser.error("give an input point cloud, or --run with the name of an earlier run")
    run_name = args.run or PurePosixPath(args.input.replace("\\", "/")).stem
    try:
        selected = select_stages(args.stages)
        from_file = load_params(data_dir_from_env() / args.params) if args.params else {}
        params = stage_params(args, from_file)
    except (PipelineError, ValueError, OSError) as exc:
        parser.error(str(exc))

    try:
        if not args.dry_run:
            check_gpu(params, selected)
    except PipelineError as exc:
        logging.getLogger("lidar_bridge").error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERRUPTED)

    stage_fns = {name: functools.partial(STAGE_FUNCTIONS[name], **params[name]) for name in STAGE_ORDER}
    run_pipeline(args.input, args.ref, run_name, selected, args.dry_run, stage_fns)

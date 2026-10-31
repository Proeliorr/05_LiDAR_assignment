"""Pipeline stages. Each ``run_<stage>(ctx, *, <typed params>) -> dict`` is exposed to hydra-zen via its signature."""

from lidar_bridge.stages._common import StageError
from lidar_bridge.stages.arbor import run_arbor
from lidar_bridge.stages.compare import run_compare
from lidar_bridge.stages.dfin import run_dfin
from lidar_bridge.stages.smartqsm import run_smartqsm
from lidar_bridge.stages.split import run_split

# Fixed execution order; the `stages=[...]` override only selects a subset, never reorders.
STAGE_ORDER = ("dfin", "split", "smartqsm", "arbor", "compare")

STAGE_FUNCTIONS = {
    "dfin": run_dfin,
    "split": run_split,
    "smartqsm": run_smartqsm,
    "arbor": run_arbor,
    "compare": run_compare,
}

# Stages that launch containers (need the docker socket).
DOCKER_STAGES = frozenset({"dfin", "smartqsm", "arbor"})

__all__ = [
    "DOCKER_STAGES",
    "STAGE_FUNCTIONS",
    "STAGE_ORDER",
    "StageError",
    "run_arbor",
    "run_compare",
    "run_dfin",
    "run_smartqsm",
    "run_split",
]

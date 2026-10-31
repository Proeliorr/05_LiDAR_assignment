"""hydra-zen configuration: one config group per stage + the root ``pipeline`` config, and the hydra entry point.

Everything is derived from the typed signatures of the ``run_<stage>`` functions, so adding a keyword parameter
to a stage function exposes it on the command line without touching this file::

    python -m lidar_bridge input=data/in/plot_01.laz reference=data/in/ref.csv
    python -m lidar_bridge 'stages=[smartqsm,compare]' run_name=r1
    python -m lidar_bridge dfin.normalize=true split.max_axis_dist=2.5
    python -m lidar_bridge 'stages=[smartqsm,compare]' run_name=r1 -m smartqsm.config=layerwise-clustering,space-colonization
    python -m lidar_bridge --help        # hydra help with the full default config
    python -m lidar_bridge --cfg job     # print the composed config and exit
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath
from typing import Any, Optional

from hydra_zen import ZenField, ZenStore, builds, make_config, zen
from omegaconf import MISSING, OmegaConf

from lidar_bridge.pipeline import EXIT_INTERRUPTED, install_termination_handler, run_pipeline
from lidar_bridge.stages import STAGE_FUNCTIONS, STAGE_ORDER

HELP_HEADER = """\
lidar_bridge: 3DFin -> split -> SmartQSM / arbor -> compare, each stage in its own container.

Run it with   docker compose run --rm -T bridge python -m lidar_bridge [overrides]
(HOST_PROJECT_DIR must be set; run_pipeline.bat does that). Input/reference paths are relative to the project.

Examples:
  input=data/in/plot_01.laz reference=data/in/ref.csv
  stages=[smartqsm,compare] run_name=r1               (stages always run in the fixed order)
  dfin.normalize=true split.max_axis_dist=2.5
  +dfin.ini_overrides.basic.upper_limit=4.0           (3DFin parameter overrides: section.key)
  dry_run=true                                        (log every docker command, run nothing)
  -m smartqsm.config=layerwise-clustering,space-colonization    (sweep, see below)

Sweep: run dfin + split once (run_name=r1), then sweep only the later stages under the SAME run_name, e.g.
  stages=[smartqsm,compare] run_name=r1 -m smartqsm.config=layerwise-clustering,space-colonization
  (run_pipeline.bat --stages smartqsm,compare --run r1 --ref ... -- -m smartqsm.config=a,b)
The jobs run one after the other; each writes smartqsm/<config>/, so the results sit side by side under r1, and
compare (which compares every smartqsm/<config> of ONE run) sees them all - compare runs after every job, so the
comparison of the last job is the complete one. A separate run_name per job would re-run dfin and split for every
job and spread the results over several runs.
"""


def _lidar_stem(path: Optional[str]) -> str:
    """Resolver ``${lidar_stem:${input}}``: default run name = file stem of the input (``run`` if no input)."""
    if not path:
        return "run"
    return PurePosixPath(str(path).replace("\\", "/")).stem or "run"


OmegaConf.register_new_resolver("lidar_stem", _lidar_stem, replace=True)

#: Name of the root config (``config_name`` for hydra / ``compose``).
CONFIG_NAME = "pipeline"

store = ZenStore(name="lidar_bridge", overwrite_ok=True)


def stage_config(fn: Any) -> Any:
    """Config node for a stage function: all keyword parameters exposed, ``ctx`` supplied at call time.

    ``zen_partial=True`` -> instantiating yields ``functools.partial(fn, **params)``; the pipeline then calls it
    with the RunContext. ``hydra_convert="all"`` hands plain ``dict`` / ``list`` objects to the stage.
    """
    return builds(fn, populate_full_signature=True, zen_partial=True, zen_exclude=["ctx"], hydra_convert="all")


def build_root_config() -> Any:
    """The root ``pipeline`` config (a dataclass) with the stage groups in its defaults list."""
    return make_config(
        input=ZenField(hint=Optional[str], default=None),
        reference=ZenField(hint=Optional[str], default=None),
        run_name="${lidar_stem:${input}}",
        stages=ZenField(hint=list[str], default=list(STAGE_ORDER)),
        dry_run=False,
        # slots filled by the per-stage config groups of the defaults list below
        **{name: ZenField(hint=Any, default=MISSING) for name in STAGE_ORDER},
        hydra_defaults=["_self_", *({name: "default"} for name in STAGE_ORDER)],
        hydra=dict(
            job=dict(chdir=False, name="lidar_bridge"),
            # config snapshot (config.yaml, hydra.yaml, overrides.yaml) and the log live inside the run dir
            run=dict(dir="data/out/${run_name}/.hydra"),
            sweep=dict(dir="data/out/${run_name}/.hydra/multirun", subdir="${hydra.job.num}"),
            output_subdir=".",
            help=dict(app_name="lidar_bridge", header=HELP_HEADER, footer=""),
        ),
        zen_dataclass={"cls_name": "PipelineConf"},
    )


def register_configs() -> ZenStore:
    """Fill the store (idempotent) and push it into hydra's ConfigStore."""
    for name, fn in STAGE_FUNCTIONS.items():
        store(stage_config(fn), group=name, name="default")
    store(build_root_config(), name=CONFIG_NAME)
    store.add_to_hydra_store(overwrite_ok=True)
    return store


def main() -> None:
    """Console entry point: ``python -m lidar_bridge ...``.

    SIGTERM (``docker stop``) is mapped to ``KeyboardInterrupt`` first thing, like Ctrl+C. :func:`run_pipeline` turns an
    interrupt during a run into a one-line log message and exit code 130; the handler here only covers an interrupt
    during start-up (config composition), where hydra would otherwise print a traceback.
    """
    install_termination_handler()
    register_configs()
    try:
        zen(run_pipeline).hydra_main(config_name=CONFIG_NAME, config_path=None, version_base="1.3")
    except KeyboardInterrupt:
        print(f"interrupted: exit code {EXIT_INTERRUPTED}", file=sys.stderr)
        sys.exit(EXIT_INTERRUPTED)

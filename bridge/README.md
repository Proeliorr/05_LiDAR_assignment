# lidar_bridge

Orchestrator container of the pipeline `3DFin -> split -> SmartQSM / arbor -> compare`. It runs the hydra-zen
app (`python -m lidar_bridge`) and starts each stage image as a **sibling container** through the mounted
`/var/run/docker.sock`. Code is bind-mounted (`PYTHONPATH=/workspace/bridge/src`), so editing `bridge/src/**`
needs no rebuild; rebuild (`docker compose build bridge`) only when `bridge/pyproject.toml` dependencies change.

## Running (PowerShell)

```powershell
$env:HOST_PROJECT_DIR = 'D:/Podyplomowe/05_Proximity_Lidar'      # forward slashes, required
docker compose run --rm -T bridge python -m lidar_bridge --help
docker compose run --rm -T bridge python -m lidar_bridge input=data/in/plot_01.laz reference=data/in/ref.csv dry_run=true
docker compose run --rm -T bridge python -m lidar_bridge input=data/in/plot_01.laz reference=data/in/ref.csv
docker compose run --rm -T bridge python -m lidar_bridge 'stages=[smartqsm,compare]' run_name=r1
docker compose run --rm -T bridge python -m lidar_bridge dfin.normalize=true split.max_axis_dist=2.5 '+dfin.ini_overrides.basic.upper_limit=4.0'
docker compose run --rm -T bridge python -m lidar_bridge 'stages=[smartqsm,compare]' run_name=r1 -m smartqsm.config=layerwise-clustering,space-colonization
docker compose run --rm -T bridge python -m pytest /workspace/bridge/tests -q
```

Quoting: in PowerShell use single quotes around overrides that contain `$` (PowerShell would expand it), quoting the
others is harmless (`'stages=[smartqsm,compare]'`); in a `.bat` file no quoting is needed except for `%`. Paths are always
relative to the project (`data/in/...`).
`run_name` defaults to the input file stem. Outputs: `data/out/<run_name>/<stage>/`; hydra's config snapshot and log:
`data/out/<run_name>/.hydra/` (a sweep job writes to `.hydra/multirun/<n>/`).

**Sweeps (`-m`).** A sweep over `smartqsm.config` must share ONE `run_name`: `compare` compares every `smartqsm/<config>/` of a
single run, and a separate `run_name` per job would re-run dfin and split for every job and scatter the results over several
runs. So run dfin + split once, then sweep only the later stages under the same name:

```powershell
.\run_pipeline.bat data\in\plot_01.laz --run r1 --stages dfin,split
.\run_pipeline.bat --stages smartqsm,compare --run r1 --ref data\in\ref.csv -- -m smartqsm.config=layerwise-clustering,space-colonization
```

The jobs run one after the other (never two SmartQSM jobs at once); each writes its own `smartqsm/<config>/`. `compare` runs
after every job, so the comparison of the last job covers all configs.

**GPU.** `smartqsm.gpu=true` runs SmartQSM with `docker run --gpus all` in the separate image `lidar-smartqsm-gpu` (CUDA torch +
spconv-cu126; `image` is switched from `lidar-smartqsm` automatically while it is left at its default). The image is built on
request only: `run_pipeline.bat --gpu --build-only`, or `docker compose --profile gpu build smartqsm-gpu` (compose service
`smartqsm-gpu`; a plain `docker compose build` skips it). Only the `spconv-contraction-*-GPU` presets (`device: cuda`) use the card.
The stage refuses a `-GPU` preset without `smartqsm.gpu=true`, warns when `smartqsm.gpu=true` meets a preset that runs on the CPU,
and raises a `StageError` that names the build command when the GPU image is missing. A tree that fails on the card (CUDA out of
memory, no GPU in the container) is recorded as status `failed` with the reason in `note`; the other trees still run.
`run_pipeline.bat --gpu` passes `smartqsm.gpu=true` and, unless the overrides after `--` already contain `smartqsm.config=`,
`smartqsm.config=spconv-contraction-LEAFOFF-GPU` (the better of the two GPU presets on the example plot). Details and measurements: the GPU section of the top-level README.

**Interrupting.** Ctrl+C and SIGTERM (`docker stop`, compose shutdown) stop the running stage container (`docker kill`, then
`docker rm -f`, by its generated name `lidar-<image>-<8 hex>`), log `interrupted: stopped container lidar-smartqsm-1a2b3c4d` and end
the run with exit code 130 and no traceback. A hard kill of the bridge (`kill -9`, `docker kill`) cannot be handled; check
`docker ps` for stray `lidar-*` containers in that case. Resumable stages (`smartqsm`, `arbor`) continue from what already exists.

## Spike: launching sibling containers from the bridge (Docker Desktop 29.1.3, Windows, WSL2)

The daemon resolves bind-mount sources on the **host**, so the bridge must pass host paths, not `/workspace/...`.
With `HOST_PROJECT_DIR=D:/Podyplomowe/05_Proximity_Lidar`:

| source spelling (`-v` and `--mount type=bind` behave the same) | result |
|---|---|
| `D:/Podyplomowe/05_Proximity_Lidar/data` | works - **used** (`HOST_PROJECT_DIR` verbatim + relative path) |
| `D:\Podyplomowe\...`, `d:/...`, trailing `/` | work |
| `/run/desktop/mnt/host/d/...`, `//d/...`, `/d/...`, `/host_mnt/d/...` | work |
| `/mnt/d/Podyplomowe/...` (WSL spelling) | **silently wrong**: rc 0, but it is a phantom folder inside the docker-desktop VM; the sibling sees an empty dir and its output is lost |

Also verified from inside the bridge: the sibling lists the Windows folder and writes a file that is visible on Windows
and removable by the bridge (root-created files too); `--network none` leaves only `lo`; `-i` is required for stdin
(without it 0 bytes arrive; text and 5 MB / binary stdin pass intact through `subprocess`); output streams line by line and
the exit code propagates. Gotcha: `-v` **and** `--mount` both silently create a missing source folder on the Windows side
(ghost directories), so `RunContext.mount()` checks existence on the bridge side first. `docker_runner.host_path()` has the full notes.

## Contract for stage authors

```python
def run_<stage>(ctx: RunContext, *, <typed keyword params with defaults>) -> dict   # may return {} ; raise StageError on failure
```
Register nothing by hand: `configs.py` derives the hydra-zen config group from the signature, so a new keyword
parameter shows up as `<stage>.<param>=...`. A new stage needs its `run_<stage>` in `stages/<stage>.py` and an entry in
`STAGE_ORDER` / `STAGE_FUNCTIONS` (and `DOCKER_STAGES` if it starts containers) in `stages/__init__.py`, plus a parameter in
`run_pipeline` (`pipeline.py`).

`RunContext` (frozen): `run_name`, `project_dir` (`/workspace`), `host_project_dir`, `input`, `reference` (in-bridge `Path`
or `None`), `dry_run`; `run_dir` (`data/out/<run_name>`), `data_dir`, `stage_dir(name)`, `reset_stage_dir(name)`,
`resolve(p)`, `rel(p)`, `host(p)` (host spelling), `mount(p, "/target", readonly=False)` -> `Mount`,
`container_path(p, root="/data")` (path inside a container that mounts `data/` at `root`).

```python
from lidar_bridge.docker_runner import require_image, run_container
require_image(image, dry_run=ctx.dry_run)
run_container(image, ["args", ...], [ctx.mount(ctx.data_dir, "/data")], network="none", gpus=False,
              timeout_s=3600, dry_run=ctx.dry_run)       # streams to the log, raises DockerRunError on non-zero exit
```
`gpus=True` adds `--gpus all` (the SmartQSM stage uses it with the `lidar-smartqsm-gpu` image).
`run_container` names every container itself (`lidar-<image>-<8 hex>`, or your `name=`) and stops it (`docker kill`, `docker rm -f`)
when an exception, a timeout or an interrupt ends the call, so stage code needs no cleanup of its own; do not catch
`KeyboardInterrupt` / `BaseException` in a stage without re-raising. In `dry_run` the logged command carries no generated name.
Honour `ctx.dry_run` (no side effects, only log commands). Stage outputs go to `ctx.stage_dir("<stage>")`; if a parameter
is sweepable (`smartqsm.config`) namespace the output below it (`smartqsm/<config>/`). The 3DFin result files the next stages
read are listed in `stages/dfin.py` (`<stem>_tree_ID_dist_axes.las`, `<stem>.xlsx`, `<stem>_stripe.las`).

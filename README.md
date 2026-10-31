# LiDAR tree metrics pipeline (3DFin, SmartQSM, arbor)

A reproducible Docker environment for forest plot point clouds. It runs 3DFin on a plot cloud, splits the result into individual trees numbered from 1, runs SmartQSM and arbor on every tree, and, once you have a reference table, compares each tree's DBH and height with the plot's reference measurements (the reference can be attached after the processing has run). The three upstream tools are unmodified git submodules, each built into its own image. A small Python orchestrator (`bridge/`, built with hydra-zen) starts the stage containers through the Docker socket. On Windows a single command, `run_pipeline.bat`, builds the images and runs the pipeline. No Python or R is needed on the host.

## Requirements

- Windows 11 with Docker Desktop (WSL2 backend), started. Tested with Docker Desktop 29.1.3.
- Git for Windows, to clone the repository with its submodules.
- The timings and memory figures below were measured on a machine with 16 CPUs and 32 GB RAM.
- Optional, for `--gpu` (see the GPU section): an NVIDIA GPU with a current Windows driver and Docker Desktop's WSL2 GPU support. No CUDA toolkit is needed on the host. The extra image takes about 16 GB of disk.
- An input point cloud (`.las` or `.laz`), inside the project folder.
- A reference table (`.csv` or `.xlsx`) is optional at first: without one the `compare` stage is skipped (with a note) and everything else runs. When the file exists, put it inside the project folder too and run `compare` on its own (see below).

## Quick start

```powershell
git clone --recurse-submodules <repository-url> <project-folder>
cd <project-folder>
.\run_pipeline.bat Example_data\plot_01.laz
```

Every run builds the four images first (BuildKit caching keeps repeat builds fast) and then runs the stages. `Example_data/` is not tracked by git (it is listed in `.gitignore`), so keep `plot_01.laz` there. The run name defaults to the file name (`plot_01`); results go to `data\out\plot_01\`.

**Attaching the reference later.** Without `--ref` the run ends successfully after `arbor` and logs that `compare` was skipped. When your reference table is ready, copy it to `data\in\` and compare without re-running anything:

```powershell
.\run_pipeline.bat --stages compare --run plot_01 --ref data\in\reference.csv --no-build
```

(With the file at hand from the start, `--ref data\in\reference.csv` on the first command compares at the end of the same run. A `--ref` path that does not exist is an error, so a typo cannot silently skip the comparison.)

A rehearsal logs every Docker command and runs none of them. `--no-build` skips the image build:

```powershell
.\run_pipeline.bat Example_data\plot_01.laz --no-build -- dry_run=true
```

Run only some stages, and only some trees, on the earlier run `plot_01`:

```powershell
.\run_pipeline.bat --stages smartqsm,arbor --run plot_01 -- 'smartqsm.trees=[3,8]'
```

- `--stages` runs a subset in the fixed order (`dfin`, `split`, `smartqsm`, `arbor`, `compare`). Stages after `dfin` need no input file when `--run` names an existing run. `compare` on its own needs `--ref`.
- A SmartQSM run on a subset of trees keeps the rows of the other trees in `smartqsm\<config>\metrics.csv` as long as the split is unchanged (rows of a different split or settings are dropped with a warning). Trees that already have a result are reused and not run again.
- `.\run_pipeline.bat --help` lists all options. Exit code: 0 for success, 1 for an error, 2 for a bad command line, otherwise the bridge's exit code.

## The pipeline

| Stage | What it does | Runs in | Main outputs (in `data\out\<run>\`) |
|---|---|---|---|
| `dfin` | 3DFin CLI on the input cloud, height-normalised by default, headless (offscreen Qt). Checks that the expected result files exist. | `lidar-3dfin` | `dfin\<stem>.xlsx`, `dfin\<stem>_tree_ID_dist_axes.las`, `dfin\<stem>_stripe.las`, `dfin\params.ini` |
| `split` | Aligns the table rows with the detected tree IDs and cuts one cloud per tree (see Tree numbering). | `lidar-bridge` (Python) | `split\trees\tree_0001.laz` and so on, `split\tree_map.csv`, `split\excluded_ids.csv` |
| `smartqsm` | SmartQSM once per tree, one after another, with no network. Collects DBH and height. | `lidar-smartqsm` (`lidar-smartqsm-gpu` with `--gpu`), one container per tree | `smartqsm\<config>\metrics.csv`, `work\`, `logs\`, `config.yaml` |
| `arbor` | R package arbor, one container for all trees. Collects DBH and height. | `lidar-arbor` | `arbor\metrics.csv` (`per_tree`) or `arbor\metrics_plot.csv` (`plot`) |
| `compare` | Joins the results with the reference on the tree number; errors, statistics and scatter plots. | `lidar-bridge` (Python) | `compare\errors.csv`, `summary.csv`, `reference_used.csv`, `scatter_dbh.png`, `scatter_height.png` |

- The metrics files of `smartqsm` and `arbor` start with `ordinal,dbh_m,height_m,status,seconds`. `dbh_m` and `height_m` are empty unless `status` is `ok`. Other statuses are `no_dbh`, `failed`, `skipped_low_points` and `error`; SmartQSM also writes `timeout` and `suspect` (an implausible model, see the GPU section for the rule), and the `note` column says why a row is not `ok`.
- Trees with fewer than `split.min_points` (500) kept points are flagged `low_points` in `tree_map.csv`, get no cloud, and are skipped by the QSM stages.
- Per-tree clouds are cut at a horizontal plane `split.min_z0` (0.10 m) above the terrain at the stem base (`stem_ground_z_m` in `tree_map.csv`, the mean of Z minus Z0 within 0.5 m of the stem axis), in addition to the `Z0 >= min_z0` cut. Without the plane, the lowest point of a cloud on a slope lies downhill of the stem and the QSM tools measure breast height too low (on the example plot, up to 0.8 m below the stem plane for most trees).
- Heights: `height_m` means height above the true ground throughout. SmartQSM measures DBH 1.3 m above the lowest point of the tree cloud, which lies `base_z0_m` (about 0.10 m, the lowest kept point above the ground at the stem base) above the ground. Its breast height is therefore 1.3 + `base_z0_m` above the ground, and its height is corrected by adding `base_z0_m`. arbor places synthetic ground points on 3DFin's terrain model and measures breast height exactly 1.3 m above the true ground.
- Results are tied to the split they were computed on: SmartQSM and arbor store a fingerprint of `split\tree_map.csv` next to their metrics (`source_fingerprint.txt`). If the split changes (other 3DFin or split parameters), arbor recomputes everything, SmartQSM drops the old rows, and `compare` skips results made on a different split with a warning (`compare.allow_stale=true` overrides).
- arbor modes: `per_tree` (default) feeds `compare`. `plot` (`arbor.mode=plot`) lets arbor segment the whole plot once and writes `arbor\metrics_plot.csv`, which `compare` ignores. Its DBH is usable; its heights are not, because the stems come out fragmented.

## Tree numbering

A tree's number is its row label `T<k>` in the "Plot Metrics" sheet of 3DFin's XLSX table, where `k` runs from 1 to N. All outputs use this number (`tree_0001.laz`, `metrics.csv`, `tree_map.csv`), and reference tree numbers must be the same T-numbers. The raw `tree_ID` in the LAS files cannot serve as a number: its values are sparse and not contiguous, and 0 is a real tree.

3DFin 0.6.0 has a quirk. A tree without a stem section can be detected but missing from its table. On the example plot, 124 trees were detected and the table has 123 rows. After the missing tree, the TH column and the X, Y of rows without a DBH belong to the previous tree (on that plot, ordinals 48-123 were shifted). The `split` stage corrects this. It matches each table row to a detected ID with dynamic programming, using the per-section circle fits of the table's X, Y and Diameters sheets. It writes:

- `split\tree_map.csv`, one row per tree. `th_m` is the correct height (from `tree_heights.las`), `th_xlsx_m` is the value as written in the table, and `th_mismatch` flags a difference. `id_confidence` is `exact` (no ID dropped), `resolved` or `ambiguous`; check the ambiguous rows.
- `split\excluded_ids.csv`, the detected IDs that have no table row and therefore no number.

## Reference data and the comparison

- Format: CSV (the delimiter and decimal comma are detected; a UTF-8 BOM is accepted) or XLSX (first sheet, unless `compare.sheet` is set).
- Columns are found by title, ignoring case and accents: tree number (`Nr drzewa`, `tree`, `id`, `nr`), DBH (`DBH [cm]`, `dbh`, `d1.3`) and height (`H (m)`, `h`, `height`, `wysokość`). The unit is read from the title (`DBH [cm]`, `H (m)`); without one, DBH is in cm and height in m. Confirm the columns and units in the compare log and in `compare\reference_used.csv`.
- To set them explicitly, use `compare.tree_id_col`, `compare.dbh_col`, `compare.dbh_unit` (`mm`, `cm` or `m`), `compare.height_col`, `compare.height_unit` and `compare.sheet`. Or copy `configs\reference_columns.example.yaml` to `configs\reference_columns.yaml`, edit it, and add `compare.columns_file=configs/reference_columns.yaml`. Explicit parameters beat the file, which beats auto-detection.
- Rows are joined on the tree number only: `T12`, `12` and `12.0` are all tree 12. Duplicate numbers are an error. An empty or non-positive cell means "no reference". A metric whose column is missing is skipped with a warning.
- Compared pipelines: `3dfin` (from `split\tree_map.csv`, using the corrected `th_m`), one `smartqsm:<config>` per SmartQSM config, and `arbor`. A value counts only when its status is `ok`.

| Column in `summary.csv` | Meaning |
|---|---|
| `n`, `n_missing`, `n_extra` | paired trees; reference trees with no valid estimate; estimates with no reference |
| `n_ambiguous`, `n_estimated`, `n_excluded` | paired trees whose tree number alignment is `ambiguous`; paired DBH values that arbor estimated rather than measured (small-tree allometry, no valid measure); pairs dropped by `compare.exclude_ambiguous` / `compare.exclude_estimated` (both off by default, the counts are logged as warnings) |
| `bias`, `MAE`, `RMSE` | mean error, mean absolute error and root mean square error, in metres |
| `rel_bias_pct`, `rel_RMSE_pct` | bias and RMSE relative to the mean reference value |
| `R2` | 1 - SSE/SST against the 1:1 line (1 is perfect, 0 is no better than the mean) |
| `slope`, `intercept`, `R2_ols` | OLS fit of estimate on reference, and the R2 of that fit |
| `within_tol_pct` | share of trees whose absolute error is within the tolerance (2 cm DBH and 1 m height by default) |

## Configuration (hydra-zen)

The root config holds `input`, `reference`, `run_name` (default: the input file stem), `stages` (default: all) and `dry_run`. Each stage has its own group: `dfin`, `split`, `smartqsm`, `arbor` and `compare`. Their parameters come from the stage function signatures. On the `run_pipeline.bat` command line, overrides follow `--` and take the form `key=value`. `+key=value` adds a key that is not in the defaults, and `-m` starts a sweep.

| Group | Parameters (defaults) |
|---|---|
| root | `input` (none), `reference` (none), `run_name` (input file stem), `stages` (all), `dry_run` (false) |
| `dfin` | `normalize` (true), `denoise` (false), `export_txt` (false; true writes text tables instead of the `.xlsx`), `ini_overrides` ({}) |
| `split` | `max_axis_dist` (3.0 m), `min_z0` (0.10 m), `min_points` (500), `emit_ground_variant` (false) |
| `smartqsm` | `config` (layerwise-clustering), `overrides`, `gpu` (false), `trees` (all), `limit` (none), `timeout_s` (1800), `skip_low_points` (true), `resume` (true), `force` (false) |
| `arbor` | `mode` (per_tree), `params`, `trees` (all), `limit` (none), `threads` (8), `timeout_s` (7200), `skip_low_points` (true), `resume` (true), `force` (false), `max_match_dist` (1.0 m), `homogenize` (false) |
| `compare` | `columns_file`, `tree_id_col`, `dbh_col`, `dbh_unit` (auto), `height_col`, `height_unit` (auto), `sheet`, `tolerance_dbh_cm` (2.0), `tolerance_height_m` (1.0), `plots` (true), `allow_stale` (false), `exclude_ambiguous` (false), `exclude_estimated` (false) |

`dfin`, `smartqsm` and `arbor` also take an `image` parameter (the Docker image to run). For `smartqsm`, `gpu=true` switches the default image to `lidar-smartqsm-gpu` and adds `--gpus all`; a `-GPU` preset without `gpu=true` is refused (see the GPU section). `errors.csv` has one row per paired tree, pipeline and metric, including `id_confidence` and `estimated`.

```powershell
.\run_pipeline.bat Example_data\plot_01.laz --ref data\in\ref.csv -- dfin.normalize=true split.max_axis_dist=2.5
.\run_pipeline.bat Example_data\plot_01.laz --ref data\in\ref.csv -- smartqsm.config=space-colonization
.\run_pipeline.bat Example_data\plot_01.laz --ref data\in\ref.csv -- +dfin.ini_overrides.basic.upper_limit=4.0
.\run_pipeline.bat --stages smartqsm,compare --run plot_01 --ref data\in\ref.csv -- -m smartqsm.config=layerwise-clustering,space-colonization
```

- `dfin.ini_overrides` changes 3DFin's own parameter file. An unknown section or key is an error.
- A sweep runs one job per value, one after another. Each job's SmartQSM output goes to its own `smartqsm\<config>\` folder, all under the same run name, so run `dfin` and `split` once first and sweep only `--stages smartqsm,compare --run <name>`. `compare` runs after every job, so the comparison of the last job is the complete one (it covers every config folder).
- To print the composed config, run `docker compose run --rm -T bridge python -m lidar_bridge --cfg job` with `HOST_PROJECT_DIR` set as in the Tests section.

## Output folders

```text
data\out\<run>\
  .hydra\              hydra snapshot of the run (config and log)
  dfin\                3DFin results and params.ini; cleared when dfin runs
  split\               tree_map.csv, excluded_ids.csv, trees\; cleared when split runs
  smartqsm\<config>\   metrics.csv, config.yaml, work\ (tree clouds and SmartQSM results), logs\ (container output per tree)
  arbor\               metrics.csv (per_tree) or metrics_plot.csv (plot), arbor.log
  compare\             errors.csv, summary.csv, reference_used.csv, scatter_dbh.png, scatter_height.png
```

SmartQSM writes `work\tree_0001_qsm.mat` and its other result files next to each tree cloud. SmartQSM and arbor resume from what already exists. `force=true` (`smartqsm.force` or `arbor.force`) re-runs the selected trees.

## Run times and resources

Measured on the development machine (16 CPUs, 32 GB RAM) with the 20.8 M point example plot:

| Step | Time | Memory |
|---|---|---|
| `dfin` | 75-98 s | peak 5.5 GiB |
| `split` | about 25 s | chunked, independent of file size |
| `smartqsm` | 60-85 s per tree | about 3.4 GiB per tree |
| `arbor` (`per_tree`) | about 200 s for 123 trees at 8 threads | not measured |

- A full plot of about 120 trees takes roughly 2 hours or more with SmartQSM. For a first look, use `smartqsm.trees=[...]` or `smartqsm.limit=N`.
- Never run two SmartQSM jobs at the same time. Runs can be interrupted and resumed, and `metrics.csv` is rewritten after every tree.
- Ctrl+C (or `docker stop` of the bridge) stops the stage container that is running and exits with code 130. Stage containers are named `lidar-<image>-<id>`; `docker ps` shows whether one is still alive.

## GPU

Optional and SmartQSM only: 3DFin and arbor are CPU-only. Of the SmartQSM presets, only `spconv-contraction-LEAFON-GPU` and `spconv-contraction-LEAFOFF-GPU` (`device: cuda`) use the card, and only for the skeleton contraction step; modelling, refinement and export stay on the CPU. All other presets run on the CPU even in the GPU image (the launcher warns).

Requirements: an NVIDIA GPU, its Windows driver and Docker Desktop with the WSL2 backend (GPU support is built in). Check that containers can see the card:

```powershell
docker run --rm --gpus all lidar-bridge nvidia-smi
```

Build once and run (the GPU image `lidar-smartqsm-gpu` is a separate tag in the compose profile `gpu`, so a plain `docker compose build` does not build it and the CPU image `lidar-smartqsm` is never replaced):

```powershell
.\run_pipeline.bat --build-only --gpu
.\run_pipeline.bat --stages smartqsm,compare --run plot_01 --ref data\in\ref.csv --no-build --gpu -- 'smartqsm.trees=[29,35,78]'
```

`--gpu` builds the GPU image in addition to the four others, passes `smartqsm.gpu=true` and, unless the overrides after `--` already contain `smartqsm.config=`, selects `spconv-contraction-LEAFOFF-GPU` (the better of the two GPU presets on the example plot, see below; for clouds with foliage compare it with `LEAFON-GPU` on a few trees). With `--no-build` the GPU image must already exist, otherwise the stage stops with a message that names the build command. `--gpu` with a CPU preset only warns that it runs on the CPU.

What was measured (GeForce GTX 1660 Ti, 6 GB, Turing sm_75, driver 596.21, Docker Desktop 29.1.3 on WSL2; image: torch 2.14.1+cu126, spconv-cu126 2.3.8, cumm 0.7.11, CUDA 12.6 runtime from pip, no nvcc; spconv's prebuilt kernels run on sm_75 with no compilation):

- Image: 15.7 GB on disk (5.0 GB compressed). First build about 11 min (downloads about 5 GB), 6-7 min with a warm pip cache.
- A SubMConv3d / SparseConv3d / SparseInverseConv3d test on the GPU agrees with a dense PyTorch reference to 5e-7.

| Preset (trees 29, 35, 78; 218k, 219k, 364k points) | Seconds per tree | DBH in cm |
|---|---|---|
| `spconv-contraction-LEAFON-cpu` | 89, 63, 90 | 19.5, 23.2, 20.2 |
| `spconv-contraction-LEAFON-GPU` | 63, 60, 81 | 21.1, 26.8 (degenerate, see below), 31.8 |
| `spconv-contraction-LEAFOFF-cpu` | 99, 95, 145 | 11.5, 11.3, 8.8 |
| `spconv-contraction-LEAFOFF-GPU` | 78, 85, 99 | 19.9, 20.6, 26.4 |
| 3DFin / arbor | | 20.4, 18.4, 25.8 / 20.2, 18.0, 25.4 |

Heights are 14.18, 13.46 and 14.98 m for every preset. The GPU makes a tree 1.1 to 1.5 times faster, not more: the contraction step itself takes 12.6 s instead of 25.2 s (tree 29) and 17.7 s instead of 33.5 s (tree 78), the remaining 50-80 s of a tree are CPU work, and each container start costs about 10 s. The largest tree of the plot (44, 583k points) took 124 s with `LEAFOFF-GPU` (DBH 17.2 cm).

- GPU memory: the card shows 1.3 GB in use for the desktop. A tree adds about 0.7 GB (trees 29, 35, 78) and about 1.05 GB (tree 44, 583k points) at peak, so no tree of this plot comes near the roughly 4.6 GB that are free. Utilisation peaked at 91-95 % in short bursts during the contraction and was idle during the CPU steps.
- Which preset: `LEAFOFF-GPU` agreed with 3DFin and arbor within 0.5, 2.2 and 0.6 cm on these three trees. `LEAFON-GPU` did not: tree 78 is 31.8 cm and tree 35 is a degenerate model (1 branch, 397 m3 stem volume). That is why `--gpu` now defaults to `LEAFOFF-GPU`. A model like tree 35's can no longer pass as `ok`: when the woody volume is more than 30 times that of a cylinder with the tree's own DBH and height, the row gets status `suspect`, DBH and height are withheld (so `compare` ignores it) and the note explains why. Still sweep both presets on trees with field measurements (`smartqsm.config=spconv-contraction-LEAFON-GPU`).
- The CPU presets are not a reliable yardstick. The CPU `spconv` 2.3.8 wheel gives results that change from run to run (tree 35 with `LEAFON-cpu`: 23.2 cm and 360 branches once, 21.7 cm and 740 branches the next time) and, in a layer test against a dense PyTorch reference, errors of order 1 on random input. The same contraction step on the GPU is bit-for-bit repeatable. This is an observation about the CPU wheel, not a fix: the CPU presets are unchanged.
- Failures: SmartQSM exits 0 and prints only `Failed.` when a tree runs out of GPU memory; the reason would be lost with the container. The image's entrypoint therefore prints the error part of SmartQSM's own log after the run, and the stage records the tree as `failed` with `CUDA out of memory ...` in `note` and the whole output in `logs	ree_NNNN.log`. The other trees still run (each has its own container). No real tree of this plot exhausted the card, so this was provoked by capping the CUDA allocator of one tree to 3 % of the memory (a genuine `torch.OutOfMemoryError`). Filling the card with a second process did not work: SmartQSM still ran with 4.75 GB held elsewhere, so Windows evidently shares the GPU memory between processes.
- If a bigger plot does run out of memory: the knobs are `skeletonization.core_algorithm_kwargs.batch_size` (16, cubes per batch) and `cube_size` (4.0 m) in the preset, or use a `-cpu` preset. Hydra cannot express a dotted override key, so on the command line such an override has to be the whole `skeletonization` block (see Configuration). The effect of these knobs on memory was not measured.

## Repository layout

```text
.
├── run_pipeline.bat          Windows launcher: builds the images, starts the bridge
├── compose.yaml              builds the four images (and lidar-smartqsm-gpu with --profile gpu); the bridge gets the project folder and the Docker socket
├── docker\
│   ├── 3dfin\Dockerfile
│   ├── arbor\Dockerfile
│   ├── bridge\Dockerfile
│   └── smartqsm\Dockerfile, requirements.lock.txt
├── bridge\                   lidar_bridge: hydra-zen orchestrator (pyproject.toml, README.md)
│   ├── src\lidar_bridge\     configs.py, pipeline.py, docker_runner.py, paths.py, idmap.py,
│   │   │                     reference.py, qsm_mat.py, __main__.py
│   │   └── stages\           dfin.py, split.py, smartqsm.py, arbor.py, compare.py, _common.py
│   ├── r\arbor_metrics.R     arbor driver script, mounted read-only into the arbor image
│   └── tests\                pytest suite
├── configs\reference_columns.example.yaml   column mapping for the compare stage
├── third_party\              git submodules: 3DFin, SmartQSM, arbor (never modified)
├── Example_data\             example plot (not tracked by git)
├── data\in\, data\out\       your inputs and the run outputs (empty in git)
└── .gitmodules, .gitattributes, .gitignore, .dockerignore
```

The bridge source is bind-mounted, so edits under `bridge\src` need no rebuild. A change to the dependencies in `bridge\pyproject.toml` needs `docker compose build bridge`.

## Tests

The pytest tests (585 at the time of writing, 2 skipped) live in `bridge\tests`. They run inside the bridge image:

```powershell
$env:HOST_PROJECT_DIR = 'D:/path/to/project'
docker compose run --rm -T bridge python -m pytest /workspace/bridge/tests -q
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `docker was not found on PATH` | Install Docker Desktop for Windows, start it, and open a new terminal. |
| `the Docker daemon does not answer` | Start Docker Desktop and wait until it reports that the engine is running. |
| `... is outside the project folder` | Inputs must be inside the project. Copy the file into `data\in` and use the copy. |
| `has a single quote in its path` | Rename the file. The bridge also rejects paths that contain commas. |
| 3DFin or SmartQSM fails with memory errors, or is killed | WSL2 has less memory than the stage needs. Create `%UserProfile%\.wslconfig` with a `[wsl2]` section and `memory=24GB` (choose a value that fits your RAM), then run `wsl --shutdown` and restart Docker Desktop. |
| `several 3DFin results in ...` | More than one `*_tree_ID_dist_axes.las` is in `dfin\`. Give the input file on the command line (`.\run_pipeline.bat Example_data\plot_01.laz ...`) to choose one. |
| A SmartQSM tree has status `failed` | SmartQSM exits with 0 even when a tree fails, so success means `work\tree_NNNN_qsm.mat` exists. Read `smartqsm\<config>\logs\tree_NNNN.log`. |
| `... differs from the config the results ... were made with` | The folder holds results of another config. Use `smartqsm.force=true` (this discards them), or another `smartqsm.config` or `--run` name. |
| `arbor ... cannot resume from ...` | The existing `metrics.csv` has another layout. Use `arbor.force=true` to discard it. |
| Shell or R scripts fail inside the containers, or `run_pipeline.bat` misbehaves | Line endings. `.gitattributes` forces LF everywhere except `.bat` and `.cmd` files, which must stay CRLF. Convert an edited script back to LF. |
| A submodule folder is empty | The launcher runs `git submodule update --init --recursive` itself; this needs Git on the PATH. Otherwise clone with `--recurse-submodules`. |
| compare warns that no value could be paired | Reference tree numbers must be the 3DFin T-numbers. Check `compare\reference_used.csv` and the tree numbers in your reference table. |
| compare stops with "DBH column ... assumed cm ... probably metres" (or mm / height in cm) | The reference column has no unit in its title, so the default was used and the values look 100x off. Set `compare.dbh_unit` / `compare.height_unit`, or put the unit in the title (`DBH [m]`). |
| compare warns "computed on a different split" and skips smartqsm or arbor | Those results were made before the split changed. Re-run the stage, or pass `compare.allow_stale=true` to compare them anyway. |
| `HOST_PROJECT_DIR` is not set | Start the bridge through `run_pipeline.bat`, or set `$env:HOST_PROJECT_DIR` as in the Tests section. |
| `docker image 'lidar-smartqsm-gpu' not found` | The GPU image is not part of a plain build. Run `.\run_pipeline.bat --build-only --gpu`. |
| `could not select device driver "" with capabilities: [[gpu]]`, or a note `Docker cannot give the container a GPU` | Docker has no GPU access: install the NVIDIA driver, update Docker Desktop (WSL2 backend), restart it, and check `docker run --rm --gpus all lidar-bridge nvidia-smi`. Without a GPU run without `--gpu` and use a `-cpu` preset. |
| `smartqsm.config '...-GPU' runs on the GPU ... but smartqsm.gpu is false` | Add `--gpu` (or `smartqsm.gpu=true`), or choose a `-cpu` preset. |
| Warning `smartqsm.gpu=true, but the preset ... does not use CUDA` | Only the `spconv-contraction-*-GPU` presets use the card; this preset runs on the CPU. |
| A GPU tree has status `failed` and the note says `CUDA out of memory` | The tree needs more GPU memory than is free (the desktop uses some). Close GPU programs, reduce `batch_size` / `cube_size` in the preset, or use a `-cpu` preset. The other trees are unaffected. |

## Pinned versions and licences

| Component | Pin | Base image or source | Licence |
|---|---|---|---|
| 3DFin | v0.6.0, `third_party\3DFin` (unmodified) | `python:3.12-slim`, PySide6 `~=6.8.1` | GPL-3.0 |
| SmartQSM | commit `bf1ee59` = v2.3.1, `third_party\SmartQSM` (unmodified) | `python:3.12-bookworm`; CPU: torch 2.15.0+cpu, spconv 2.3.8, cumm 0.7.11. GPU (`lidar-smartqsm-gpu`): torch 2.14.1+cu126, spconv-cu126 2.3.8, cumm-cu126 0.7.11 | AGPL-3.0 |
| arbor | v1.1.2, `third_party\arbor` (unmodified) | `rocker/r-ver:4.5.3` | GPL-3.0 |
| bridge and launcher | this repository, `lidar-bridge` 0.1.0 | `python:3.12-slim` | not chosen yet |

The upstream tools are built into separate images and are coupled to the bridge only through files and command lines. SmartQSM's update check is neutralised by running it without network and answering `no` on stdin, not by editing its code.

## Known limitations

- Per-tree accuracy depends on the cloud policy (`split.max_axis_dist`, `split.min_z0`). Neighbouring crowns can leak into a tree's cloud.
- The QSM methods and 3DFin's DBH differ noticeably. On the example plot, arbor is about 2 cm MAE (about 20 % relative) away from 3DFin's DBH. Judge the methods against the field reference, not against each other.
- The SmartQSM preset matters for DBH. On the example plot the default `layerwise-clustering` over-estimated DBH on some trees (trees 29, 35, 78: 40.8, 25.3 and 34.1 cm, where 3DFin and arbor both gave about 20, 18 and 25 cm), while `space-colonization` (21.3, 19.2, 21.8 cm; about 2-3x slower) and `spconv-contraction-LEAFON-cpu` (19.5, 23.2, 20.2 cm; similar speed) agreed with them. The `spconv-contraction-*-cpu` results are not repeatable, though (see the GPU section; `LEAFOFF-cpu` gave 11.5, 11.3, 8.8 cm). Heights did not depend on the preset. Before a long run, sweep the presets on a few trees that have field measurements (`--stages smartqsm,compare --run <name> -- -m smartqsm.config=a,b smartqsm.trees=[...]`) and pick the one closest to your reference.
- The input and reference files must be inside the project folder. Paths with commas are rejected by the bridge.
- The bridge container has access to the Docker socket, which is root-equivalent on the host.
- The reference columns and units are auto-detected. Confirm them in the compare log.
- The GPU path (`--gpu`) was verified on one machine only: a GTX 1660 Ti (6 GB) with driver 596.21 and Docker Desktop 29.1.3 on WSL2, on three to four trees of the example plot. It is 1.1 to 1.5 times faster per tree than the CPU presets, not an order of magnitude, and the `LEAFON-GPU` preset produced a degenerate model for one of the three trees while `LEAFOFF-GPU` (now the `--gpu` default) did not. A real out-of-memory case on the card was not met; it was only provoked artificially.
- 3DFin 0.6.0 can leave trees without a stem section out of its table. The split stage corrects the numbering, but rows marked `ambiguous` in `tree_map.csv` should be checked by hand.
- SmartQSM is slow: a full plot of about 120 trees takes roughly 2 hours or more.
- The licence of the bridge and launcher code is not chosen yet.

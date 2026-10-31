# LiDAR tree metrics pipeline (3DFin, SmartQSM, arbor)

One Docker image for forest plot point clouds. It runs 3DFin on a plot cloud, splits the result into individual trees numbered from 1, runs SmartQSM and arbor on every tree, and, once you have a reference table, compares each tree's DBH and height with the plot's reference measurements (the reference can be attached after the processing has run).

The image holds one mamba environment per tool: `3dfin`, `arbor` and `smartqsm`. The three upstream tools are unmodified git submodules. A small Python package (`bridge/`) runs them one after another and does the splitting and comparing itself, in the `smartqsm` environment. You mount a data folder and start the container. No Python or R is needed on the host.

## Requirements

- Windows 11 with Docker Desktop (WSL2 backend), started. Tested with Docker Desktop 29.1.3.
- Git for Windows, to clone the repository with its submodules. They are only needed to build the image.
- An NVIDIA GPU with a current Windows driver, for the default SmartQSM preset. No CUDA toolkit is needed on the host. Without a GPU, choose a CPU preset (see the GPU section).
- About 20 GB of disk for the image (19.3 GB as built).
- A data folder holding your input point cloud (`.las` or `.laz`).
- Optional: a reference table (`.csv` or `.xlsx`). Without one the `compare` stage is skipped (with a note) and everything else runs. See [Reference data and the comparison](#reference-data-and-the-comparison).

The timings and memory figures below were measured on a machine with 16 CPUs and 32 GB RAM.

## Quick start

```powershell
git clone --recurse-submodules <repository-url> <project-folder>
cd <project-folder>
docker build -t lidar-pipeline .
docker run --rm --init --gpus all --network none -v D:\path\data:/data lidar-pipeline in/plot_01.laz
```

`-v` mounts your data folder at `/data`, and every path on the command line is relative to it. Results go to `out/<run>/` in the same folder. The run name defaults to the file name (`plot_01`).

- `--gpus all` gives the container the GPU. `compare` and the CPU presets do not need it.
- `--network none` is optional but recommended: nothing in the pipeline needs the network, and it makes SmartQSM's update check fail at once.
- `--init` passes Ctrl+C and `docker stop` on correctly.

In the examples below, `...` stands for `docker run --rm --init --gpus all --network none -v D:\path\data:/data`.

A reference table is optional. Without one the run ends successfully after `arbor`; see [Reference data and the comparison](#reference-data-and-the-comparison) for attaching it, then or later.

A rehearsal logs every tool command and runs none of them:

```powershell
... lidar-pipeline in/plot_01.laz --dry-run
```

Run only some stages, and only some trees, on the earlier run `plot_01`:

```powershell
... lidar-pipeline --stages smartqsm,arbor --run plot_01 --trees 3,8
```

- `--stages` runs a subset in the fixed order (`dfin`, `split`, `smartqsm`, `arbor`, `compare`). Stages after `dfin` need no input file when `--run` names an existing run. `compare` on its own needs `--ref`.
- A SmartQSM run on a subset of trees keeps the rows of the other trees in `smartqsm\<config>\metrics.csv` as long as the split is unchanged (rows of a different split or settings are dropped with a warning). Trees that already have a result are reused and not run again.
- `... lidar-pipeline --help` lists all options. Exit code: 0 for success, 1 for an error, 2 for a bad command line, 130 when interrupted.

## The pipeline

| Stage | What it does | Runs in | Main outputs (in `out\<run>\`) |
|---|---|---|---|
| `dfin` | 3DFin CLI on the input cloud, height-normalised by default, headless (offscreen Qt). Checks that the expected result files exist. | `3dfin` | `dfin\<stem>.xlsx`, `dfin\<stem>_tree_ID_dist_axes.las`, `dfin\<stem>_stripe.las`, `dfin\params.ini` |
| `split` | Aligns the table rows with the detected tree IDs and cuts one cloud per tree (see Tree numbering). | `smartqsm` | `split\trees\tree_0001.laz` and so on, `split\tree_map.csv`, `split\excluded_ids.csv` |
| `smartqsm` | SmartQSM once per tree, one after another. Collects DBH and height. | `smartqsm` | `smartqsm\<config>\metrics.csv`, `work\`, `logs\`, `config.yaml` |
| `arbor` | R package arbor, one R process for all trees. Collects DBH and height. | `arbor` | `arbor\metrics.csv` (`per_tree`) or `arbor\metrics_plot.csv` (`plot`) |
| `compare` | Joins the results with the reference on the tree number; errors, statistics and scatter plots. | `smartqsm` | `compare\errors.csv`, `summary.csv`, `reference_used.csv`, `scatter_dbh.png`, `scatter_height.png` |

- The metrics files of `smartqsm` and `arbor` start with `ordinal,dbh_m,height_m,status,seconds`. `dbh_m` and `height_m` are empty unless `status` is `ok`. Other statuses are `no_dbh`, `failed`, `skipped_low_points` and `error`; SmartQSM also writes `timeout` and `suspect` (an implausible model, see the GPU section for the rule), and the `note` column says why a row is not `ok`.
- Trees with fewer than `split.min_points` (500) kept points are flagged `low_points` in `tree_map.csv`, get no cloud, and are skipped by the QSM stages.
- Per-tree clouds are cut at a horizontal plane `split.min_z0` (0.10 m) above the terrain at the stem base (`stem_ground_z_m` in `tree_map.csv`, the mean of Z minus Z0 within 0.5 m of the stem axis), in addition to the `Z0 >= min_z0` cut. Without the plane, the lowest point of a cloud on a slope lies downhill of the stem and the QSM tools measure breast height too low (on the example plot, up to 0.8 m below the stem plane for most trees).
- Heights: `height_m` means height above the true ground throughout. SmartQSM measures DBH 1.3 m above the lowest point of the tree cloud, which lies `base_z0_m` (about 0.10 m, the lowest kept point above the ground at the stem base) above the ground. Its breast height is therefore 1.3 + `base_z0_m` above the ground, and its height is corrected by adding `base_z0_m`. arbor places synthetic ground points on 3DFin's terrain model and measures breast height exactly 1.3 m above the true ground.
- Results are tied to the split they were computed on: SmartQSM and arbor store a fingerprint of `split\tree_map.csv` next to their metrics (`source_fingerprint.txt`). If the split changes (other 3DFin or split parameters), arbor recomputes everything, SmartQSM drops the old rows, and `compare` skips results made on a different split with a warning (`allow_stale: true` under `compare` in the `--params` file overrides).
- arbor modes: `per_tree` (default) feeds `compare`. `plot` lets arbor segment the whole plot once and writes `arbor\metrics_plot.csv`, which `compare` ignores. Its DBH is usable; its heights are not, because the stems come out fragmented.

## Tree numbering

A tree's number is its row label `T<k>` in the "Plot Metrics" sheet of 3DFin's XLSX table, where `k` runs from 1 to N. All outputs use this number (`tree_0001.laz`, `metrics.csv`, `tree_map.csv`), and reference tree numbers must be the same T-numbers. The raw `tree_ID` in the LAS files cannot serve as a number: its values are sparse and not contiguous, and 0 is a real tree.

3DFin 0.6.0 has a quirk. A tree without a stem section can be detected but missing from its table. On the example plot, 124 trees were detected and the table has 123 rows. After the missing tree, the TH column and the X, Y of rows without a DBH belong to the previous tree (on that plot, ordinals 48-123 were shifted). The `split` stage corrects this. It matches each table row to a detected ID with dynamic programming, using the per-section circle fits of the table's X, Y and Diameters sheets. It writes:

- `split\tree_map.csv`, one row per tree. `th_m` is the correct height (from `tree_heights.las`), `th_xlsx_m` is the value as written in the table, and `th_mismatch` flags a difference. `id_confidence` is `exact` (no ID dropped), `resolved` or `ambiguous`; check the ambiguous rows.
- `split\excluded_ids.csv`, the detected IDs that have no table row and therefore no number.

## Reference data and the comparison

Reference data is optional. Every other stage runs without it, and only `compare` uses it. Without `--ref`, the run ends successfully after `arbor` and logs that `compare` was skipped.

**The schema is not final.** The real reference table has not been described yet. So far the only fixed requirement is that each row holds the reference data of one tree. The columns below are placeholders, and this section will be updated when the real schema is known. The rules after them describe what the `compare` stage accepts today.

**Attaching it.** Copy the table into your data folder, for example to `in/<reference-file>.csv`. Then either compare an existing run, without re-running anything:

```powershell
... lidar-pipeline --stages compare --run <run> --ref in/<reference-file>.csv
```

or give it on the first command, which compares at the end of the same run:

```powershell
... lidar-pipeline in/<point-cloud>.laz --ref in/<reference-file>.csv
```

A `--ref` path that does not exist is an error, so a typo cannot silently skip the comparison. `compare` does not need the GPU, so you can leave out `--gpus all` when it runs alone.

**Placeholder layout**, one row per tree:

| `<tree-number>` | `<dbh> [cm]` | `<height> (m)` |
|---|---|---|
| `<T-number of the tree>` | `<measured DBH>` | `<measured height>` |

**What `compare` accepts today:**

- Tree numbers must be the pipeline's own T-numbers (see Tree numbering), because rows are joined on the tree number and nothing else. Trees numbered in the field will not match them unless the numbers are first mapped to T-numbers, using `split\tree_map.csv` after a run. The pipeline does not do that mapping.
- Format: CSV (the delimiter and decimal comma are detected; a UTF-8 BOM is accepted) or XLSX (first sheet, unless `sheet` is set).
- Columns are found by title, ignoring case and accents: tree number (`Nr drzewa`, `tree`, `id`, `nr`), DBH (`DBH [cm]`, `dbh`, `d1.3`) and height (`H (m)`, `h`, `height`, `wysokość`). The unit is read from the title (`DBH [cm]`, `H (m)`); without one, DBH is in cm and height in m. Confirm the columns and units in the compare log and in `compare\reference_used.csv`.
- To set them explicitly, use `tree_id_col`, `dbh_col`, `dbh_unit` (`mm`, `cm` or `m`), `height_col`, `height_unit` and `sheet` under `compare` in the `--params` file. Or copy `configs\reference_columns.example.yaml` into your data folder, edit it, and set `columns_file: reference_columns.yaml` under `compare`. Explicit parameters beat the file, which beats auto-detection.
- Rows are joined on the tree number only: `T12`, `12` and `12.0` are all tree 12. Duplicate numbers are an error. An empty or non-positive cell means "no reference". A metric whose column is missing is skipped with a warning.
- Compared pipelines: `3dfin` (from `split\tree_map.csv`, using the corrected `th_m`), one `smartqsm:<config>` per SmartQSM config, and `arbor`. A value counts only when its status is `ok`.

| Column in `summary.csv` | Meaning |
|---|---|
| `n`, `n_missing`, `n_extra` | paired trees; reference trees with no valid estimate; estimates with no reference |
| `n_ambiguous`, `n_estimated`, `n_excluded` | paired trees whose tree number alignment is `ambiguous`; paired DBH values that arbor estimated rather than measured (small-tree allometry, no valid measure); pairs dropped by `exclude_ambiguous` / `exclude_estimated` (both off by default, the counts are logged as warnings) |
| `bias`, `MAE`, `RMSE` | mean error, mean absolute error and root mean square error, in metres |
| `rel_bias_pct`, `rel_RMSE_pct` | bias and RMSE relative to the mean reference value |
| `R2` | 1 - SSE/SST against the 1:1 line (1 is perfect, 0 is no better than the mean) |
| `slope`, `intercept`, `R2_ols` | OLS fit of estimate on reference, and the R2 of that fit |
| `within_tol_pct` | share of trees whose absolute error is within the tolerance (2 cm DBH and 1 m height by default) |

## Parameters

The common choices are command-line flags: `--trees`, `--limit` and `--force` apply to both `smartqsm` and `arbor`, and `--smartqsm-config` picks the preset. Everything else goes into a YAML file in your data folder, named with `--params`. Flags beat the file, and an unknown stage or parameter is an error.

```yaml
# params.yaml: only what you want to change
dfin:
  ini_overrides:
    basic:
      upper_limit: 4.0
split:
  max_axis_dist: 2.5
arbor:
  threads: 12
compare:
  tolerance_dbh_cm: 3.0
```

```powershell
... lidar-pipeline in/plot_01.laz --params params.yaml
```

| Stage | Parameters (defaults) |
|---|---|
| `dfin` | `normalize` (true), `denoise` (false), `export_txt` (false; true writes text tables instead of the `.xlsx`), `ini_overrides` (none) |
| `split` | `max_axis_dist` (3.0 m), `min_z0` (0.10 m), `min_points` (500), `emit_ground_variant` (false) |
| `smartqsm` | `config` (spconv-contraction-LEAFOFF-GPU), `overrides` (none), `trees` (all), `limit` (none), `timeout_s` (1800), `skip_low_points` (true), `resume` (true), `force` (false) |
| `arbor` | `mode` (per_tree), `params` (none), `trees` (all), `limit` (none), `threads` (8), `timeout_s` (7200), `skip_low_points` (true), `resume` (true), `force` (false), `max_match_dist` (1.0 m), `homogenize` (false) |
| `compare` | `columns_file`, `tree_id_col`, `dbh_col`, `dbh_unit` (auto), `height_col`, `height_unit` (auto), `sheet`, `tolerance_dbh_cm` (2.0), `tolerance_height_m` (1.0), `plots` (true), `allow_stale` (false), `exclude_ambiguous` (false), `exclude_estimated` (false) |

- `dfin.ini_overrides` changes 3DFin's own parameter file. An unknown section or key is an error.
- `errors.csv` has one row per paired tree, pipeline and metric, including `id_confidence` and `estimated`.

**Trying several presets.** Each preset's results go to their own `smartqsm\<config>\` folder, so run `dfin` and `split` once, loop over the presets under the same run name, then compare once. `compare` covers every config folder of the run:

```powershell
... lidar-pipeline in/plot_01.laz --stages dfin,split
foreach ($c in 'space-colonization','spconv-contraction-LEAFOFF-GPU') { ... lidar-pipeline --stages smartqsm --run plot_01 --trees 29,35,78 --smartqsm-config $c }
... lidar-pipeline --stages compare --run plot_01 --ref in/reference.csv
```

## Output folders

```text
out\<run>\
  pipeline.log         log of every invocation on this run, appended
  dfin\                3DFin results and params.ini; cleared when dfin runs
  split\               tree_map.csv, excluded_ids.csv, trees\; cleared when split runs
  smartqsm\<config>\   metrics.csv, config.yaml, work\ (tree clouds and SmartQSM results), logs\ (SmartQSM output per tree)
  arbor\               metrics.csv (per_tree) or metrics_plot.csv (plot), arbor.log
  compare\             errors.csv, summary.csv, reference_used.csv, scatter_dbh.png, scatter_height.png
```

SmartQSM writes `work\tree_0001_qsm.mat` and its other result files next to each tree cloud. SmartQSM and arbor resume from what already exists. `--force` re-runs the selected trees.

## Run times and resources

Measured on the development machine (16 CPUs, 32 GB RAM) with the 20.8 M point example plot. The memory figures date from before the move to one image:

| Step | Time | Memory |
|---|---|---|
| `dfin` | about 65 s | peak 5.5 GiB |
| `split` | about 30 s | chunked, independent of file size |
| `smartqsm` (default preset, GPU) | 50-90 s per tree | about 3.4 GiB per tree |
| `arbor` (`per_tree`) | about 190 s for 123 trees at 8 threads | not measured |

- A full plot of about 120 trees takes roughly 2 hours or more with SmartQSM. For a first look, use `--trees 3,8` or `--limit N`.
- Never run two containers on the same run at the same time. Runs can be interrupted and resumed, and `metrics.csv` is rewritten after every tree.
- Ctrl+C or `docker stop` kills the running tool and exits with code 130.

## GPU

3DFin and arbor are CPU-only. Of the SmartQSM presets, only `spconv-contraction-LEAFON-GPU` and `spconv-contraction-LEAFOFF-GPU` (`device: cuda`) use the card, and only for the skeleton contraction step; modelling, refinement and export stay on the CPU. All other presets run on the CPU in the same image.

The default preset is `spconv-contraction-LEAFOFF-GPU`. If the container sees no GPU, the pipeline stops before the first stage and says so. Check that containers can see the card:

```powershell
docker run --rm --gpus all --entrypoint nvidia-smi lidar-pipeline
```

**Leaf-off or leaf-on clouds.** Choose the preset by the type of your point cloud. The default suits clouds without foliage; for clouds with foliage, name the leaf-on preset:

```powershell
... lidar-pipeline in/plot_01.laz --smartqsm-config spconv-contraction-LEAFON-GPU
```

To set it once for a project, put it in the `--params` file (`smartqsm: {config: spconv-contraction-LEAFON-GPU}`). Each preset's results go to their own `smartqsm\<config>\` folder and `compare` reports each one separately, so you can run both on the same split and compare them on a few trees with field measurements.

Without a GPU, leave out `--gpus all` and choose a CPU preset:

```powershell
docker run --rm --init --network none -v D:\path\data:/data lidar-pipeline in/plot_01.laz --smartqsm-config space-colonization
```

What was measured (GeForce GTX 1660 Ti, 6 GB, Turing sm_75, driver 596.21, Docker Desktop 29.1.3 on WSL2; torch 2.14.1+cu126, spconv-cu126 2.3.8, cumm 0.7.11, CUDA 12.6 runtime from pip, no nvcc; spconv's prebuilt kernels run on sm_75 with no compilation):

- A SubMConv3d / SparseConv3d / SparseInverseConv3d test on the GPU agrees with a dense PyTorch reference to 5e-7.

| Preset (trees 29, 35, 78; 218k, 219k, 364k points) | Seconds per tree | DBH in cm |
|---|---|---|
| `spconv-contraction-LEAFON-cpu` | 89, 63, 90 | 19.5, 23.2, 20.2 |
| `spconv-contraction-LEAFON-GPU` | 63, 60, 81 | 21.1, 26.8 (degenerate, see below), 31.8 |
| `spconv-contraction-LEAFOFF-cpu` | 99, 95, 145 | 11.5, 11.3, 8.8 |
| `spconv-contraction-LEAFOFF-GPU` | 78, 85, 99 | 19.9, 20.6, 26.4 |
| 3DFin / arbor | | 20.4, 18.4, 25.8 / 20.2, 18.0, 25.4 |

Heights are 14.18, 13.46 and 14.98 m for every preset. These seconds were measured with one container per tree, of which each start cost about 10 s; in the current image `LEAFOFF-GPU` takes 66, 75 and 90 s and gives the same DBH and height to every stored digit. The GPU makes a tree 1.1 to 1.5 times faster, not more: the contraction step itself takes 12.6 s instead of 25.2 s (tree 29) and 17.7 s instead of 33.5 s (tree 78), and the remaining 50-80 s of a tree are CPU work. The largest tree of the plot (44, 583k points) took 124 s with `LEAFOFF-GPU` (DBH 17.2 cm).

- GPU memory: the card shows 1.3 GB in use for the desktop. A tree adds about 0.7 GB (trees 29, 35, 78) and about 1.05 GB (tree 44, 583k points) at peak, so no tree of this plot comes near the roughly 4.6 GB that are free. Utilisation peaked at 91-95 % in short bursts during the contraction and was idle during the CPU steps.
- Which preset: `LEAFOFF-GPU` agreed with 3DFin and arbor within 0.5, 2.2 and 0.6 cm on these three trees. `LEAFON-GPU` did not: tree 78 is 31.8 cm and tree 35 is a degenerate model (1 branch, 397 m3 stem volume). A model like tree 35's can no longer pass as `ok`: when the woody volume is more than 30 times that of a cylinder with the tree's own DBH and height, the row gets status `suspect`, DBH and height are withheld (so `compare` ignores it) and the note explains why. Still try both presets on trees with field measurements.
- The CPU `spconv` presets are not a reliable yardstick. The CPU `spconv` 2.3.8 wheel gave results that change from run to run (tree 35 with `LEAFON-cpu`: 23.2 cm and 360 branches once, 21.7 cm and 740 branches the next time) and, in a layer test against a dense PyTorch reference, errors of order 1 on random input. The same contraction step on the GPU is bit-for-bit repeatable. The image now holds the CUDA wheel only, and the `-cpu` presets in it have not been re-measured.
- Failures: SmartQSM exits 0 and prints only `Failed.` when a tree runs out of GPU memory. The image's `smartqsm-run` script therefore prints the error part of SmartQSM's own log after the run, and the stage records the tree as `failed` with `CUDA out of memory ...` in `note` and the whole output in `logs\tree_NNNN.log`. The other trees still run (each is a separate process). No real tree of this plot exhausted the card; this was provoked by capping the CUDA allocator of one tree to 3 % of the memory.
- If a bigger plot does run out of memory: the knobs are `skeletonization.core_algorithm_kwargs.batch_size` (16, cubes per batch) and `cube_size` (4.0 m), set under `smartqsm: overrides:` in the `--params` file, or use a CPU preset. The effect of these knobs on memory was not measured.

## Repository layout

```text
.
├── Dockerfile                builds the one image
├── envs\                     the three mamba environments: 3dfin.yml, arbor.yml, smartqsm.yml, requirements.lock.txt
├── docker\                   in-env (runs a command in an environment), smartqsm-run (SmartQSM plus its error log)
├── bridge\                   lidar_bridge: the pipeline code
│   ├── src\lidar_bridge\     cli.py, pipeline.py, runner.py, paths.py, idmap.py, reference.py, qsm_mat.py, __main__.py
│   │   └── stages\           dfin.py, split.py, smartqsm.py, arbor.py, compare.py, _common.py
│   ├── r\arbor_metrics.R     arbor driver script
│   └── tests\                pytest suite
├── configs\reference_columns.example.yaml   column mapping for the compare stage
├── third_party\              git submodules: 3DFin, SmartQSM, arbor (never modified)
└── .gitmodules, .gitattributes, .gitignore, .dockerignore
```

The code is copied into the image, so a change needs `docker build -t lidar-pipeline .` again. That only rebuilds the last layer. The earlier setup with one image per tool is kept in git under the tag `four-images`. On the example plot, this image reproduced that setup's `split\tree_map.csv` byte for byte and arbor's DBH and height for all 123 trees.

## Tests

The pytest tests live in `bridge\tests` and run inside the image:

```powershell
docker run --rm --entrypoint in-env lidar-pipeline smartqsm python -m pytest /opt/lidar/bridge/tests -q
```

To test edited code without rebuilding, mount it over the copy in the image with `-v ${PWD}\bridge:/opt/lidar/bridge`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `... /data does not exist: mount your data folder` | Add `-v D:\path\data:/data` to `docker run`. |
| `... needs a GPU, but the container sees none` | Add `--gpus all`. If that fails, install the NVIDIA driver, update Docker Desktop (WSL2 backend), restart it, and check `docker run --rm --gpus all --entrypoint nvidia-smi lidar-pipeline`. Without a GPU use `--smartqsm-config space-colonization`. |
| `could not select device driver "" with capabilities: [[gpu]]` | Docker has no GPU access. See the row above. |
| `... is outside the mounted data folder` | Paths are relative to the data folder. Copy the file into it. |
| `... is a Windows path` | Give the path relative to the data folder: `in/plot_01.laz`, not `D:\path\data\in\plot_01.laz`. |
| 3DFin or SmartQSM fails with memory errors, or is killed | WSL2 has less memory than the stage needs. Create `%UserProfile%\.wslconfig` with a `[wsl2]` section and `memory=24GB` (choose a value that fits your RAM), then run `wsl --shutdown` and restart Docker Desktop. |
| `several 3DFin results in ...` | More than one `*_tree_ID_dist_axes.las` is in `dfin\`. Give the input file on the command line to choose one. |
| A SmartQSM tree has status `failed` | SmartQSM exits with 0 even when a tree fails, so success means `work\tree_NNNN_qsm.mat` exists. Read `smartqsm\<config>\logs\tree_NNNN.log`. |
| `... differs from the config the results ... were made with` | The folder holds results of another config. Use `--force` (this discards them), or another `--smartqsm-config` or `--run` name. |
| `arbor ... cannot resume from ...` | The existing `metrics.csv` has another layout. Use `--force` to discard it. |
| Scripts fail inside the container with a "bad interpreter" error | Line endings. `.gitattributes` forces LF everywhere except `.bat` and `.cmd` files. Convert an edited script back to LF and rebuild. |
| `docker build` stops because a third_party folder is empty | Run `git submodule update --init --recursive`. |
| compare warns that no value could be paired | Reference tree numbers must be the 3DFin T-numbers. Check `compare\reference_used.csv` and the tree numbers in your reference table. |
| compare stops with "DBH column ... assumed cm ... probably metres" (or mm / height in cm) | The reference column has no unit in its title, so the default was used and the values look 100x off. Set `dbh_unit` / `height_unit` under `compare` in the `--params` file, or put the unit in the title (`DBH [m]`). |
| compare warns "computed on a different split" and skips smartqsm or arbor | Those results were made before the split changed. Re-run the stage, or set `allow_stale: true` under `compare` to compare them anyway. |
| A GPU tree has status `failed` and the note says `CUDA out of memory` | The tree needs more GPU memory than is free (the desktop uses some). Close GPU programs, reduce `batch_size` / `cube_size` in the preset, or use a CPU preset. The other trees are unaffected. |

## Pinned versions and licences

| Component | Pin | Environment | Licence |
|---|---|---|---|
| 3DFin | v0.6.0, `third_party\3DFin` (unmodified) | `3dfin`: Python 3.12, PySide6-Essentials `~=6.8.1` | GPL-3.0 |
| SmartQSM | commit `bf1ee59` = v2.3.1, `third_party\SmartQSM` (unmodified) | `smartqsm`: Python 3.12, torch 2.14.1+cu126, spconv-cu126 2.3.8, cumm 0.7.11, other packages from `envs\requirements.lock.txt` | AGPL-3.0 |
| arbor | v1.1.2, `third_party\arbor` (unmodified) | `arbor`: R 4.5 and lidR 4.3.2 from conda-forge | GPL-3.0 |
| pipeline code | this repository, `lidar-bridge` 0.1.0 | `smartqsm` | not chosen yet |

Base image: `condaforge/miniforge3:26.9.4-0`. Only some conda packages are pinned (`envs\*.yml`), so a rebuild can pick up newer versions of the others, such as GDAL or `r-sf`.

The upstream tools are separate programs in separate environments and are coupled to the pipeline code only through files and command lines. Check their licence terms before publishing the image, since it bundles all of them. SmartQSM's update check is neutralised by answering `no` on stdin and, with `--network none`, by having no network, not by editing its code.

## Known limitations

- Per-tree accuracy depends on the cloud policy (`split.max_axis_dist`, `split.min_z0`). Neighbouring crowns can leak into a tree's cloud.
- The QSM methods and 3DFin's DBH differ noticeably. On the example plot, arbor is about 2 cm MAE (about 20 % relative) away from 3DFin's DBH. Judge the methods against the field reference, not against each other.
- The SmartQSM preset matters for DBH. On the example plot `layerwise-clustering` over-estimated DBH on some trees (trees 29, 35, 78: 40.8, 25.3 and 34.1 cm, where 3DFin and arbor both gave about 20, 18 and 25 cm), while `space-colonization` (21.3, 19.2, 21.8 cm; about 2-3x slower) and `spconv-contraction-LEAFOFF-GPU` agreed with them much better. That evidence is three trees. Heights did not depend on the preset. `space-colonization` is not repeatable: the same cloud and config gave tree 29 a DBH between 20.9 and 21.8 cm over six runs, in this image and in the earlier one alike. `LEAFOFF-GPU` gave identical values in two runs. Before a long run, try the presets on a few trees that have field measurements (see Parameters) and pick the one closest to your reference.
- The input and reference files must be inside the mounted data folder.
- The schema of the real reference table is not known yet, so reading it has only been tested on a made-up table (`in/ref_selftest.csv`). The columns and units are auto-detected; confirm them in the compare log.
- The GPU path was verified on one machine only: a GTX 1660 Ti (6 GB) with driver 596.21 and Docker Desktop 29.1.3 on WSL2, on three to four trees of the example plot. A real out-of-memory case on the card was not met; it was only provoked artificially.
- 3DFin 0.6.0 can leave trees without a stem section out of its table. The split stage corrects the numbering, but rows marked `ambiguous` in `tree_map.csv` should be checked by hand.
- SmartQSM is slow: a full plot of about 120 trees takes roughly 2 hours or more.
- The licence of the pipeline code is not chosen yet.

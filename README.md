# LiDAR tree metrics: 3DFin, SmartQSM and arbor

This repository builds one Docker image that holds three tools for forest point clouds: 3DFin, SmartQSM (which uses the GPU) and arbor. A small Python package in `bridge/` runs them in order on a plot cloud, cuts the plot into single trees, and compares DBH and tree height with your field measurements.

## Requirements

You need Docker Desktop with the WSL2 backend, and an NVIDIA GPU with a current Windows driver for the GPU presets of SmartQSM. No CUDA toolkit, Python or R is needed on the host. The image takes about 20 GB of disk.

Get the code together with the three tools, which are git submodules in `third_party/` (3DFin, SmartQSM, arbor):

```powershell
git clone --recurse-submodules https://github.com/Proeliorr/05_LiDAR_assignment
```

## Build

```powershell
docker build -t lidar-pipeline .
```

The first build takes 20 to 40 minutes, mostly because of the torch and CUDA wheels. Then check that each environment works:

```powershell
docker run --rm --entrypoint in-env lidar-pipeline 3dfin 3DFin --version
docker run --rm --entrypoint in-env lidar-pipeline arbor Rscript -e "library(arbor)"
docker run --rm --gpus all --entrypoint in-env lidar-pipeline smartqsm python -c "import torch; print(torch.cuda.is_available())"
```

The last one should print `True`.

## Running a new plot

Put the cloud (`.las` or `.laz`) in `data/in/` and run:

```powershell
docker run --rm --init --gpus all -v D:\path\to\repo\data:/data lidar-pipeline in/plot_01.laz
```

Every path on the command line is relative to the folder you mount at `/data`. Results go to `data/out/<run>/`; the run name is the file name unless you give `--run`.

The stages run in a fixed order. `dfin` runs 3DFin on the cloud. `split` cuts one cloud per tree from 3DFin's result. `smartqsm` fits a model to each tree and reads DBH and height from it, one tree after another. `arbor` measures the same trees with arbor. `compare` joins all results with a reference table and computes the errors.

The flags you will use:

- `--run NAME` names the run folder; give it again to continue an earlier run.
- `--stages dfin,split,...` runs only some of the stages (the order stays fixed).
- `--ref FILE` is the reference table (`.csv` or `.xlsx`). Without it `compare` is skipped.
- `--params FILE.yaml` holds every other stage parameter, written as `stage: {name: value}`.
- `--smartqsm-config PRESET` chooses the SmartQSM preset. The default is `spconv-contraction-LEAFOFF-GPU`.
- `--trees 3,8` and `--limit N` restrict `smartqsm` and `arbor` to some trees.
- `--force` throws away old results of the selected trees and runs them again.

`--help` lists the rest, including `--dry-run`. SmartQSM and arbor skip trees that already have a result, so an interrupted run can be started again with the same command.

A tree's number is the row label `T<k>` of 3DFin's table (`tree_0001.laz` is `T1`). The reference table must use the same numbers. Columns for tree number, DBH and height are found by their titles; if that fails, or the units are unclear, name them under `compare` in the params file (`tree_id_col`, `dbh_col`, `dbh_unit`, `height_col`, `height_unit`, `sheet`). The compare log and `compare/reference_used.csv` show what was picked.

Files you will open in `data/out/<run>/`:

- `split/tree_map.csv`, one row per tree
- `smartqsm/<preset>/metrics.csv`
- `arbor/metrics.csv`
- `compare/summary.csv` (bias, MAE, RMSE and more for each method) and `compare/errors.csv` (one row per tree)
- `compare/scatter_dbh.png` and `compare/scatter_height.png`

Progress and errors are in `pipeline.log` in the run folder, and SmartQSM's output per tree is in `smartqsm/<preset>/logs/`.

## Starting from a cloud that is already segmented

If 3DFin and arbor were already run by hand (for example 3DFin in the CloudCompare plugin and arbor in RStudio), the pipeline can start from arbor's segmented cloud. Give that cloud as the input and name a key table under `split` in the params file. The split stage then cuts the listed trees out of the cloud and takes the 3DFin and arbor values from the table, so only SmartQSM is computed here.

```yaml
split:
  key: in/key.xlsx        # .xlsx or .csv, relative to the data folder
  key_sheet: Sheet1       # xlsx only
  id_dim: treeID          # the dimension of the cloud that holds arbor's tree id
```

The key table needs a column with the tree number (`Lp`) and one with arbor's tree id (`ID_Arbor`). The columns `DBH_3DFin_cm`, `H_3DFin_m`, `DBH_Arbor_cm`, `H_Arbor_m` and `Arbor_log` are optional. Other column titles can be given under `split.key_columns`.

```powershell
docker run --rm --init --gpus all -v D:\path\to\repo\data:/data lidar-pipeline in/segmented.laz --run plot --stages split,smartqsm --params in/params.yaml
docker run --rm --init -v D:\path\to\repo\data:/data lidar-pipeline --run plot --stages compare --ref in/reference.xlsx --params in/params.yaml
```

The first command cuts out the trees and runs SmartQSM, the second compares everything with the field values. For a cloud scanned with leaves on the trees, select the leaf-on preset `spconv-contraction-LEAFON-GPU`.

## Repository layout

```text
Dockerfile        builds the image
docker/           in-env (runs a command in one environment), smartqsm-run
envs/             conda environment files, one per tool
bridge/           the pipeline code (lidar_bridge)
third_party/      3DFin, SmartQSM, arbor (submodules, not modified)
data/in/          your input clouds
data/out/         results, one folder per run
```

## Known limits

3DFin gives understorey trees the height of the canopy. SmartQSM takes height from the lowest point of a tree's cloud; the pipeline adds the offset between that point and the ground back. A SmartQSM model whose stem volume is implausible (more than 30 times that of a cylinder with the tree's own DBH and height) is marked `suspect` and left out of the comparison. On a cloud segmented by arbor, SmartQSM's tree height is not independent of arbor, because it is measured on arbor's segment.

SmartQSM spends 1 to 2 minutes per tree on the development machine (GTX 1660 Ti), so a plot of about 120 trees takes roughly two hours; use `--trees` or `--limit` for a first look. A tree adds about 1 GB of GPU memory at its peak, and none of the trees of the example plot came near the free memory of that 6 GB card. If a tree runs out of memory it is recorded as `failed` and the others still run. Only that one GPU was tested.

## Versions and licences

The pinned versions are 3DFin v0.6.0 (GPL-3.0), arbor v1.1.2 (GPL-3.0) and SmartQSM at commit `bf1ee59` (AGPL-3.0). Each keeps its own licence in its submodule. The code of this repository is under the MIT licence in the root `LICENSE`.

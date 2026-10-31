# lidar_bridge

The pipeline code: `3DFin -> split -> SmartQSM / arbor -> compare`. It runs in the `smartqsm` environment of the image and starts each tool in its own mamba environment through `in-env` (`runner.py`). Usage is in the [root README](../README.md).

- `cli.py`: the command line. Flags and the `--params` file become keyword arguments of the `run_<stage>` functions, so a new keyword parameter of a stage can be set in the `--params` file straight away.
- `pipeline.py`: runs the selected stages in the fixed order, and handles failures and interrupts.
- `runner.py`: runs one tool, streams its output to the log, and kills its whole process group on a timeout, an interrupt or its end.
- `paths.py`: `RunContext`, the data and install folders and the layout of a run.
- `stages/`: one module per stage. `idmap.py`, `reference.py` and `qsm_mat.py` are their helpers. `r/arbor_metrics.R` is the arbor driver script.

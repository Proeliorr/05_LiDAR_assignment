# syntax=docker/dockerfile:1
#
# lidar-pipeline: 3DFin, SmartQSM (CUDA) and arbor in one image, each in its own mamba environment.
# The pipeline code (bridge/) runs in the `smartqsm` environment, which already has everything split and compare need.
#
#   docker build -t lidar-pipeline .
#   docker run --rm --init --gpus all --network none -v D:\path\data:/data lidar-pipeline in/plot_01.laz
#
# The layers go from what changes least to what changes most, so editing bridge/ rebuilds only the last layer.
# PIP_ROOT_USER_ACTION and the pip cache mounts only keep the build quiet and fast; nothing depends on them at run time.

FROM condaforge/miniforge3:26.9.4-0 AS base

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    RGL_USE_NULL=TRUE

# Libraries the imports demand although nothing is ever drawn: Qt (3DFin), open3d (SmartQSM). No display is needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libgomp1 libglib2.0-0 libxkbcommon0 libfontconfig1 libdbus-1-3 libusb-1.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --chmod=755 docker/in-env docker/smartqsm-run /usr/local/bin/

# ---------------------------------------------------------------------------------------------
# 3DFin wheel. Its CLI imports GUI modules that only the sdist build hook generates (pyside6-uic/rcc), so a plain
# `pip install ./third_party/3DFin` dies with ImportError: build the sdist, then the wheel from it.
# ---------------------------------------------------------------------------------------------
FROM base AS wheel-builder

RUN mamba create -y -n build python=3.12 pip
RUN --mount=type=cache,target=/root/.cache/pip \
    in-env build pip install build "hatchling~=1.24" "pyside6-essentials~=6.8.1"
COPY third_party/3DFin /src/3DFin
WORKDIR /src/3DFin
# The hook ignores pyside6-uic/rcc failures, so the build fails here if its outputs are missing from the sdist.
RUN in-env build python -m build --sdist --no-isolation -x --outdir /dist . \
 && tar -tzf /dist/*.tar.gz > /tmp/sdist.lst \
 && for f in gui/main_window.py gui/expert_dlg.py gui/gui_ressources_rc.py; do \
        grep -q "src/three_d_fin/$f\$" /tmp/sdist.lst || { echo "FATAL: $f missing from sdist (Qt hook did not run)"; exit 1; }; \
    done \
 && in-env build pip wheel --no-deps --no-build-isolation --wheel-dir /wheels /dist/*.tar.gz

# ---------------------------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------------------------
FROM base AS runtime

# --- arbor: R from conda-forge, the package itself from the pinned submodule ---------------------
COPY envs/arbor.yml /opt/envs/arbor.yml
RUN mamba env create -y -f /opt/envs/arbor.yml && mamba clean -afy
RUN --mount=type=bind,source=third_party/arbor,target=/mnt/arbor,readonly \
    cp -a /mnt/arbor /tmp/arbor \
 && MAKEFLAGS="-j$(nproc)" in-env arbor R CMD INSTALL --strip --no-docs --no-multiarch /tmp/arbor \
 && rm -rf /tmp/arbor \
 && in-env arbor Rscript -e 'suppressMessages({library(arbor); library(lidR); library(data.table); library(optparse)}); cat("arbor", as.character(packageVersion("arbor")), "lidR", as.character(packageVersion("lidR")), "\n")'

# --- 3DFin: requirements read from the submodule's pyproject.toml so they cannot drift from the pin ---
# Only the Qt requirement is swapped: QtCore/QtGui/QtWidgets are all in PySide6-Essentials, the Addons are not needed.
COPY envs/3dfin.yml /opt/envs/3dfin.yml
COPY third_party/3DFin/pyproject.toml /tmp/3dfin/pyproject.toml
RUN mamba env create -y -f /opt/envs/3dfin.yml && mamba clean -afy
RUN in-env 3dfin python - <<'PY'
import tomllib
with open("/tmp/3dfin/pyproject.toml", "rb") as f:
    deps = tomllib.load(f)["project"]["dependencies"]
keep = [d for d in deps if not d.lower().replace("_", "-").startswith("pyside6")]
with open("/tmp/3dfin/requirements.txt", "w") as f:
    f.write("\n".join(keep) + "\n")
PY
RUN --mount=type=cache,target=/root/.cache/pip \
    in-env 3dfin pip install -r /tmp/3dfin/requirements.txt "pyside6-essentials~=6.8.1"
RUN --mount=type=bind,from=wheel-builder,source=/wheels,target=/wheels \
    in-env 3dfin pip install --no-deps /wheels/*.whl \
 && in-env 3dfin 3DFin --version \
 && in-env 3dfin 3DFin cli --help > /dev/null

# --- SmartQSM with CUDA ---------------------------------------------------------------------------
# requirements.lock.txt is a constraints file (pip freeze of a known-good build). torch comes from the PyTorch index, so it is
# dropped from the constraints. The cu126 wheels hold sm_75 (Turing) to sm_90 kernels and pull the CUDA runtime in as pip
# packages: no CUDA toolkit is needed here, the host only needs an NVIDIA driver. CPU presets run in the same environment.
COPY envs/smartqsm.yml envs/requirements.lock.txt /opt/envs/
COPY third_party/SmartQSM/requirements.txt /opt/envs/smartqsm-requirements.txt
RUN mamba env create -y -f /opt/envs/smartqsm.yml && mamba clean -afy
RUN grep -viE '^torch(==|$)' /opt/envs/requirements.lock.txt > /opt/envs/constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    in-env smartqsm pip install -c /opt/envs/constraints.txt -r /opt/envs/smartqsm-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    in-env smartqsm pip install -c /opt/envs/constraints.txt "torch==2.14.1+cu126" \
        --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
 && in-env smartqsm pip install -c /opt/envs/constraints.txt "spconv-cu126==2.3.8" pytest

# Copied, not mounted: SmartQSM writes smartqsm.log and smartqsm.run into entrypoints/.
COPY third_party/SmartQSM /opt/lidar/third_party/SmartQSM
# Fail the build if a dependency drift breaks the imports, or if the wrong torch / spconv / cumm got installed. Needs no GPU.
RUN cd /opt/lidar/third_party/SmartQSM \
 && in-env smartqsm python -c "import sys; sys.path.insert(0, 'entrypoints'); import _common_import, _updater; import torch; print('import smoke test OK, torch', torch.__version__)" \
 && in-env smartqsm python -c "import torch, spconv, cumm; import spconv.pytorch; from external.SmartTreeXX.models.smart_tree_xx import SmartTreeXX; \
assert torch.version.cuda is not None, torch.__version__; \
assert (spconv.__version__, cumm.__version__) == ('2.3.8', '0.7.11'), (spconv.__version__, cumm.__version__); \
print('spconv model import OK: spconv', spconv.__version__, 'cumm', cumm.__version__, 'torch', torch.__version__)"

# --- the pipeline ---------------------------------------------------------------------------------
COPY third_party/3DFin/src/three_d_fin/3DFinconfig.ini /opt/lidar/third_party/3DFin/src/three_d_fin/3DFinconfig.ini
COPY bridge /opt/lidar/bridge
COPY configs /opt/lidar/configs
ENV PYTHONPATH=/opt/lidar/bridge/src
WORKDIR /opt/lidar

ENTRYPOINT ["in-env", "smartqsm", "python", "-m", "lidar_bridge"]
CMD ["--help"]

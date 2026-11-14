# syntax=docker/dockerfile:1
# lidar-pipeline: 3DFin, SmartQSM (CUDA torch) and arbor, each in its own mamba environment; bridge/ runs in `smartqsm`.
#   docker build -t lidar-pipeline .
#   docker run --rm --init --gpus all -v D:\path\data:/data lidar-pipeline in/plot_01.laz

FROM condaforge/miniforge3:26.9.4-0

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    RGL_USE_NULL=TRUE

# Libraries that Qt (3DFin) and open3d (SmartQSM) load even though nothing is drawn
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libgomp1 libglib2.0-0 libxkbcommon0 libfontconfig1 libdbus-1-3 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Helper scripts: in-env runs a command in a mamba environment, smartqsm-run runs SmartQSM
COPY --chmod=755 docker/in-env docker/smartqsm-run /usr/local/bin/
COPY envs /opt/envs
COPY third_party /opt/lidar/third_party

# arbor: R and its packages from conda-forge, arbor itself from the submodule
RUN mamba env create -y -f /opt/envs/arbor.yml && mamba clean -afy \
    && in-env arbor R CMD INSTALL --no-docs --no-multiarch /opt/lidar/third_party/arbor

# 3DFin: the Qt GUI files are generated when the sdist is built, so build the sdist and install that
RUN mamba env create -y -f /opt/envs/3dfin.yml && mamba clean -afy \
    && cd /opt/lidar/third_party/3DFin \
    && in-env 3dfin python -m build --sdist --outdir /tmp/3dfin . \
    && in-env 3dfin pip install --no-cache-dir /tmp/3dfin/*.tar.gz && rm -rf /tmp/3dfin

# SmartQSM: pinned packages, torch built for CUDA 12.6 (the host only needs an NVIDIA driver)
RUN mamba env create -y -f /opt/envs/smartqsm.yml && mamba clean -afy \
    && in-env smartqsm pip install --no-cache-dir -r /opt/envs/smartqsm-requirements.txt

COPY bridge /opt/lidar/bridge
ENV PYTHONPATH=/opt/lidar/bridge/src
WORKDIR /opt/lidar

ENTRYPOINT ["in-env", "smartqsm", "python", "-m", "lidar_bridge"]
CMD ["--help"]

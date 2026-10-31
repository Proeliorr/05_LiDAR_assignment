# SmartQSM v2.3.1 (third_party/SmartQSM, AGPL-3.0) as a headless batch tool: CPU (default) or CUDA.
#
#   docker build -f docker/smartqsm/Dockerfile -t lidar-smartqsm .                              # CPU (default)
#   docker build -f docker/smartqsm/Dockerfile -t lidar-smartqsm-gpu --build-arg WITH_GPU=1 .   # CUDA (several GB)
#
# compose.yaml builds the two under separate tags (the GPU one only with `--profile gpu`), so switching never
# rebuilds anything. The GPU image is run with `docker run --gpus all ...` and a spconv-contraction-*-GPU preset.
#
# Build context is the project root. Contract used by the bridge:
#   docker run --rm -i --network none -v <host>/data:/data lidar-smartqsm \
#       -t -y -c /data/.../config.yaml /data/.../tree_0001.laz        (stdin: "no")
# Outputs land next to the input cloud (<stem>_qsm.mat, _branches.ply, _crown_convex.ply, _skeleton.dxf);
# success == <stem>_qsm.mat exists (SmartQSM's exit code is 0 even when a tree failed).

# SmartQSM needs Python 3.12 exactly: it uses same-quote nested f-strings (3.12+) and the open3d
# wheels stop at cp312. The full (non-slim) image is used because the entrypoints import tkinter
# at module level. Pin a digest here for byte-for-byte reproducible builds.
ARG PYTHON_IMAGE=python:3.12-bookworm
FROM ${PYTHON_IMAGE}

ARG WITH_GPU=0
# CPU torch (download.pytorch.org/whl/cpu). Version recorded in requirements.lock.txt.
ARG TORCH_VERSION=2.15.0
# GPU variant only (WITH_GPU=1). The cu126 torch wheels contain sm_75 (Turing) .. sm_90 kernels and pull the CUDA
# runtime libraries in as pip packages, so the image needs no CUDA toolkit: the host only needs an NVIDIA driver.
# 2.14.1 is the newest torch on the cu126 index; spconv-cu126 2.3.8 is the newest spconv wheel (compose.yaml
# pins it as spconv-cu126==2.3.8; the smoke test at the end checks the result either way).
# NOTE: every ARG above the first RUN is part of the build-cache key of all the RUNs, so this block (names and
# defaults) is deliberately left as it was: the CPU build keeps hitting the cache of the images built before the
# GPU variant existed. New ARGs go next to the step that uses them.
ARG CUDA_TAG=cu126
ARG TORCH_CUDA_VERSION=2.14.1
ARG SPCONV_PACKAGE=spconv-cu126

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# System libraries that the imports demand even though nothing is ever drawn:
# open3d.visualization.gui -> libGL/libEGL/libusb (open3d 0.20 links libusb), tkinter -> libtk (already in base).
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
 && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libgomp1 libglib2.0-0 libxkbcommon0 libusb-1.0-0

# --- dependency layers (before the submodule COPY so source edits do not reinstall anything) ----
# requirements.lock.txt is a constraints file (pip freeze of a known-good CPU build). torch itself is
# installed separately from the PyTorch index, so it is dropped from the constraints used for the rest.
COPY docker/smartqsm/requirements.lock.txt /opt/requirements.lock.txt
COPY third_party/SmartQSM/requirements.txt /opt/smartqsm-requirements.txt
RUN grep -viE '^torch(==|$)' /opt/requirements.lock.txt > /opt/constraints.txt

# SmartQSM's own requirements (unpinned upstream; the constraints make the resolution repeatable).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt -r /opt/smartqsm-requirements.txt

# Deep-learning stack for the spconv-based contraction (configs/spconv-contraction-*). SmartQSM imports
# it in try/except, so the other algorithms (layerwise-clustering, space-colonization, ...) work without it.
#   CPU (default): CPU-only torch + the plain `spconv` wheel (a CPU build; its import works without CUDA).
#                  Use configs/spconv-contraction-*-cpu.yaml (device: cpu).
#   GPU (WITH_GPU=1): CUDA torch + the matching spconv-cuXXX wheel (its prebuilt kernels include sm_75, no nvcc or
#                  JIT compilation needed); run with `docker run --gpus all ...` and
#                  configs/spconv-contraction-*-GPU.yaml (device: cuda).
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$WITH_GPU" = "1" ]; then \
        pip install -c /opt/constraints.txt "torch==${TORCH_CUDA_VERSION}+${CUDA_TAG}" \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --extra-index-url https://pypi.org/simple \
 && pip install -c /opt/constraints.txt "${SPCONV_PACKAGE}"; \
    else \
        pip install -c /opt/constraints.txt "torch==${TORCH_VERSION}+cpu" \
            --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
 && pip install -c /opt/constraints.txt spconv; \
    fi

# --- SmartQSM itself ----------------------------------------------------------------------------
# Copied (not bind-mounted): it writes smartqsm.log / smartqsm.run into entrypoints/ and must stay writable.
COPY third_party/SmartQSM /opt/SmartQSM
WORKDIR /opt/SmartQSM

# spconv / cumm exactly as requirements.lock.txt pins them for the CPU build; the GPU build gets the same pair
# (the cumm pin also keeps pip from taking the newer cumm-cu126 0.8.2 for spconv-cu126).
ARG SPCONV_VERSION=2.3.8
ARG CUMM_VERSION=0.7.11

# Fail the build early if an unpinned dependency drift breaks the module-level imports, or if the wrong torch
# flavour / spconv / cumm got installed. Needs no GPU: torch, spconv.pytorch and the spconv model import without
# a CUDA device or libcuda (checked); the kernels themselves are only exercised at run time.
RUN python -c "import sys; sys.path.insert(0, 'entrypoints'); import _common_import, _updater; import torch; print('import smoke test OK, torch', torch.__version__)" \
 && python -c "import torch, spconv, cumm; import spconv.pytorch; from external.SmartTreeXX.models.smart_tree_xx import SmartTreeXX; \
assert (torch.version.cuda is not None) == ('${WITH_GPU}' == '1'), f'wrong torch flavour for WITH_GPU=${WITH_GPU}: {torch.__version__}'; \
assert (spconv.__version__, cumm.__version__) == ('${SPCONV_VERSION}', '${CUMM_VERSION}'), (spconv.__version__, cumm.__version__); \
print('spconv model import OK: spconv', spconv.__version__, 'cumm', cumm.__version__, 'torch', torch.__version__)"

# SmartQSM exits 0 and prints only "Failed." when a tree fails: the reason (the traceback, e.g. "CUDA out of memory") goes to
# its own log, entrypoints/smartqsm.log, which dies with the container (--rm). So the entrypoint runs it as before (same
# arguments, stdin and exit code) and then prints the tail of the error part of that log, where the bridge records it.
ENTRYPOINT ["sh", "-c", "python entrypoints/smartqsm.py \"$@\"; rc=$?; if grep -qa '[[]ERROR[]]' entrypoints/smartqsm.log 2>/dev/null; then echo '--- smartqsm.log, error part ---'; sed -n '/[[]ERROR[]]/,$p' entrypoints/smartqsm.log | tail -n 12; fi; exit $rc", "smartqsm"]

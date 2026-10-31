# syntax=docker/dockerfile:1
#
# lidar-bridge: the orchestrator. Runs the hydra-zen pipeline (python -m lidar_bridge ...) and launches the
# stage images (lidar-3dfin, lidar-smartqsm, lidar-arbor) as SIBLING containers through the mounted
# /var/run/docker.sock, so it only needs the docker CLI (no compose plugin, no daemon).
#
# Only the third-party DEPENDENCIES are baked in here. The bridge code itself is NOT installed: compose
# bind-mounts the repo at /workspace and PYTHONPATH points at bridge/src, so editing bridge/src/** needs no
# rebuild. Rebuild only when the dependency list in bridge/pyproject.toml changes.
#
# Build (repo root as context):   docker compose build bridge      (HOST_PROJECT_DIR must be set)
FROM python:3.12-slim

COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/workspace/bridge/src

# Dependencies only: read [project].dependencies (+ the `dev` extra, for pytest) out of pyproject.toml.
COPY bridge/pyproject.toml /tmp/bridge-pyproject.toml
RUN python -c "import tomllib; p = tomllib.load(open('/tmp/bridge-pyproject.toml', 'rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['dev']))" > /tmp/bridge-requirements.txt \
 && pip install -r /tmp/bridge-requirements.txt \
 && rm /tmp/bridge-pyproject.toml /tmp/bridge-requirements.txt

WORKDIR /workspace
CMD ["python", "-m", "lidar_bridge", "--help"]

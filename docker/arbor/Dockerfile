# lidar-arbor: headless runner for the R package `arbor` (r-lidar/arbor), installed from the
# pinned git submodule third_party/arbor (v1.1.2), not from r-universe.
#
# Build (context = project root, see compose.yaml):
#   docker build -f docker/arbor/Dockerfile -t lidar-arbor .
# Run:
#   docker run --rm --network none -v <host>/data:/data -v <host>/bridge/r:/bridge_r:ro \
#       lidar-arbor /bridge_r/arbor_metrics.R --tree_dir /data/...
#
# Base: rocker/r-ver:4.5.3 (Ubuntu 24.04 "noble", R 4.5.3). Its default CRAN repo is a dated
# Posit Package Manager snapshot (p3m.dev/cran/__linux__/noble/2026-04-23) serving prebuilt
# Linux binaries, so every dependency installs as a binary and the snapshot fixes versions
# (lidR 4.3.2 there; arbor needs lidR >= 4.2.2, which rocker/r-ver:4.4.x snapshots do not have).
FROM rocker/r-ver:4.5.3

ENV DEBIAN_FRONTEND=noninteractive \
    # rgl would otherwise try to open an X display on load; NULL device = no OpenGL needed.
    RGL_USE_NULL=TRUE \
    LANG=en_US.UTF-8

# Runtime (not -dev) libraries for the prebuilt sf / terra / rgl / s2 / fs binaries. arbor itself
# is compiled below but only needs R headers, Rcpp, g++ and OpenMP, all present in the base image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgdal34t64 \
      libgeos-c1t64 \
      libproj25 \
      libudunits2-0 \
      libsqlite3-0 \
      libabsl20220623t64 \
      libgl1 \
      libglu1-mesa \
      libx11-6 \
      libfreetype6 \
      libpng16-16t64 \
      libxml2 \
      libfontconfig1 \
      libharfbuzz0b \
      libfribidi0 \
      libtiff6 \
      libjpeg-turbo8 \
      libuv1t64 \
 && rm -rf /var/lib/apt/lists/*

# arbor dependencies (all binaries from the PPM snapshot). optparse is for the bridge's R script.
RUN Rscript -e 'install.packages(c("Rcpp", "data.table", "sf", "rgl", "lidR", "optparse"), Ncpus = 8L)' \
 && Rscript -e 'for (p in c("Rcpp", "data.table", "sf", "rgl", "lidR", "optparse")) library(p, character.only = TRUE)' \
 && rm -rf /tmp/downloaded_packages /tmp/Rtmp*

# Install arbor from the submodule. The source is bind-mounted read-only and copied to /tmp first
# because R CMD INSTALL writes object files into the package directory; the submodule on the
# host is never touched and the ~34 MB source does not end up in an image layer.
RUN --mount=type=bind,source=third_party/arbor,target=/mnt/arbor,readonly \
    cp -a /mnt/arbor /tmp/arbor \
 && MAKEFLAGS="-j$(nproc)" R CMD INSTALL --strip --no-docs --no-multiarch /tmp/arbor \
 && rm -rf /tmp/arbor

# Fail the build if a shared library of an installed R package cannot be resolved (tcl/tk is the
# GUI toolkit binding of base R and is irrelevant headless), then smoke-load the stack.
RUN missing="$(find /usr/local/lib/R/site-library /usr/local/lib/R/library -name '*.so' \
        -exec env LD_LIBRARY_PATH=/usr/local/lib/R/lib ldd {} + 2>/dev/null \
        | grep 'not found' | grep -v -e libtcl -e libtk | sort -u)"; \
    if [ -n "$missing" ]; then echo "Unresolved shared libraries:"; echo "$missing"; exit 1; fi \
 && Rscript -e 'suppressMessages({library(arbor); library(lidR); library(data.table); library(optparse)}); cat("arbor", as.character(packageVersion("arbor")), "lidR", as.character(packageVersion("lidR")), "\n")'

ENTRYPOINT ["Rscript"]
CMD ["--version"]

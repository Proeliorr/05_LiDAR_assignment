#!/usr/bin/env Rscript
# arbor_metrics.R - DBH / height / volume of the 3DFin trees with the R package `arbor`, for the lidar_bridge `arbor` stage.
# Runs inside the `lidar-arbor` image:
#   docker run --rm --network none -e OMP_NUM_THREADS=8 -v <host>/data:/data -v <host>/bridge/r:/bridge_r:ro \
#       lidar-arbor /bridge_r/arbor_metrics.R --mode per_tree --trees_dir /data/out/<run>/split/trees \
#       --tree_map /data/out/<run>/split/tree_map.csv --out_csv /data/out/<run>/arbor/metrics.csv
#
# OUTPUT: one CSV (--out_csv), one row per 3DFin ordinal, columns
#   ordinal,dbh_m,height_m,status,seconds, base_z0_m,dbh_height_above_ground_m,volume_m3,qsm_height_m,root_x,root_y,note,mode,match_dist_m
# See CONTRACT_COLS / the notes below. The CSV is appended to AFTER EVERY TREE (per_tree mode), so an interrupted run
# loses at most the tree in progress, and --resume continues where it stopped.
#
# ======================================================================================================================
# MODES
#   per_tree  one QSM per split/trees/tree_XXXX.laz ("lone tree" chain below).
#   plot      arbor's own scene pipeline once on the whole 3DFin cloud (README chain: hybrid_homogeneization ->
#             segment_ground -> wood_likelihood -> segment_semantic -> find_seeds -> segment_instance -> flag_small_trees
#             -> qsf), then every arbor tree is matched one-to-one to the nearest 3DFin ordinal (X_root,Y_root vs tree_map
#             x_m,y_m, <= --max_match_dist, smallest distance wins). hag comes from arbor's own segment_ground (arbor's
#             ground classification and seed finding rely on it, replacing hag by 3DFin's Z0 would make hag and the ground
#             class inconsistent). 3DFin's Z0 is only used as the common ground reference for the breast-height and
#             height numbers (below), so both modes measure against the same surface.
#
# LONE-TREE CHAIN (per_tree). Tested on all 123 trees of data/out/dev_full against 3DFin's DBH (43 trees with a 3DFin DBH):
#   readTLS(select = "xyz0")  ->  1000 synthetic ground points ON THE 3DFin DTM around the tree (arbor's own
#   add_single_tree_ground, whose points are moved from z = min(Z) to the ground surface Z - Z0)  ->  segment_ground
#   ->  wood_likelihood  ->  segment_semantic  ->  drop the synthetic points  ->  treeID := 1  ->  qsm().
#   Compared (123 trees, DBH bias / MAE vs 3DFin on the 43 trees with a DBH, tree height MAE vs 3DFin th_m, failures):
#     chosen    synthetic ground on the DTM          -0.4 / 2.2 cm   0.23 m   0 failed   (192 s for 123 trees at 8 threads)
#     V1        WP3: flat ground at min(Z)           -0.3 / 1.9 cm   0.33 m   0 failed   (same DBH within noise, heights worse:
#                                                    on slopes min(Z) is 0.1 - 0.5 m below the stem's ground)
#     hag := Z0 (no ground points at all)            +2.3 / 4.9 cm   0.29 m   10 failed/no_dbh, 58 "No valid measure": the path
#                                                    finder needs ground points; corr. of its DBH with the others only 0.1
#     V3        scene chain (seeds/instance/qsf) on the lone tree: splits one tree into several instances (a 13 m tree came
#               out as 8 m), 2-4x slower (8 trees tested)
#   The differences between the first two lines are within the method's noise (paired DBH difference MAE 1.4 cm); the DTM
#   variant was kept because its hag, prolongation to the ground and heights refer to the true ground. Full point density is
#   used (no arbor-CLI 25 % sample); --homogenize adds arbor's voxel homogeneization (default off; equal accuracy on 12 trees).
#   Without a Z0 dimension (not a 3DFin cloud) the ground is assumed flat at min(Z) - base_z0_m (note: no_Z0_flat_ground_assumed).
#   DBH of the lone trees vs 3DFin: -0.4 cm bias, 2.2 cm MAE (22 %); 20 of 123 trees get arbor's "No valid measure" (DBH
#   estimated by allometry, flagged in `note`).
#
# HEIGHT REFERENCES (the important part)
#   * arbor measures the DBH at path distance `bh` from the root of the QSM, and the root of a QSM is prolonged down to hag = 0
#     ("Prolongation to the ground"), so qsm_dbh(q, bh = 1.3) is NOT 1.3 m above the ground (on a 3DFin tree 1.0 .. 1.38 m).
#     The distance is therefore solved (Newton on the stem polyline, <= 0.005 m) until the measuring point lies 1.3 m above the
#     ground surface (3DFin DTM = Z - Z0, plane fitted to the cloud around the stem; the xyz that qsm_dbh() returns is quantised
#     to the cylinder joints, so the polyline is used). `dbh_height_above_ground_m` is the height ACHIEVED (1.300 +- 0.005
#     unless the note says bh_not_converged); the diameter is then qsm_dbh(q, bh = that distance).
#   * `qsm_height_m` is qsm_stats()$H as arbor reports it: max(endZ) - Z of the first stem cylinder, i.e. WITHOUT the
#     prolongation, so it is shorter than the tree by ~0.2 - 0.5 m. `height_m` = max(endZ of the QSM) - ground elevation at the
#     stem base, i.e. the height of the top of the QSM above the true ground. The QSM top is the highest WOOD cylinder, so
#     it can be slightly lower than the highest foliage point.
#   * Tree files are cut at Z0 >= min_z0 (base_z0_m ~ 0.1 m); that cut only matters for the fallback without Z0.
#
# STATUS
#   ok                 DBH > 0 from the QSM. If arbor says the diameter was not measured but estimated ("Small tree allometry",
#                      "No valid measure") dbh_m is kept, status stays ok and the arbor message is in `note` (compare only
#                      accepts status ok, and dropping these trees would be worse than a flagged estimate).
#   estimated_dbh      reserved (not emitted); accepted by the Python validation.
#   no_dbh             a QSM exists but no usable DBH (non-finite / <= 0 / breast height outside the stem).
#   failed             arbor produced no QSM (arbor error such as "no passage points found", "QSM Generation Failed", no wood
#                      points); plot mode: no arbor tree within --max_match_dist of the 3DFin tree.
#   skipped_low_points tree_map says low_points (cloud not written / too few points); --skip_low_points=false tries anyway.
#   error              unexpected R error outside arbor (missing tree file, ...); the message is in `note`. Never aborts the run.
#
# `seconds`: per_tree = wall time of the tree (read + chain + QSM); plot = total scene-run time / number of arbor trees.
# ======================================================================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(lidR)
  library(arbor)
})

CONTRACT_COLS <- c(
  "ordinal", "dbh_m", "height_m", "status", "seconds",
  "base_z0_m", "dbh_height_above_ground_m", "volume_m3", "qsm_height_m", "root_x", "root_y", "note", "mode", "match_dist_m"
)
ROUND_DIGITS <- c(dbh_m = 4, height_m = 3, seconds = 2, base_z0_m = 4, dbh_height_above_ground_m = 3, volume_m3 = 3,
                  qsm_height_m = 3, root_x = 3, root_y = 3, match_dist_m = 3)   # precision written to the CSV
TARGET_BH <- 1.30       # breast height above the true ground (m)
BH_TOLERANCE <- 0.005   # accepted deviation of the measuring height (m)
BH_MAX_ITER <- 8L

# ---------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------
logmsg <- function(fmt, ...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(fmt, ...)))
  flush(stdout())
}

# Run an arbor/lidR call silently: arbor prints progress bars (\r) and "[hh:mm:ss] step" lines to stdout.
quiet <- function(expr) {
  value <- NULL
  utils::capture.output(value <- suppressWarnings(suppressMessages(expr)))
  value
}

as_flag <- function(x) tolower(as.character(x)) %in% c("true", "t", "1", "yes")

num1 <- function(x) if (length(x) == 0L || is.null(x)) NA_real_ else suppressWarnings(as.numeric(x[[1L]]))

clean_note <- function(x) {
  x <- as.character(unlist(x))
  x <- x[!is.na(x) & nzchar(x)]
  x <- gsub("\033\\[[0-9;]*m", "", paste(x, collapse = "; "))   # ANSI colours in arbor messages
  x <- gsub("[\r\n]+", " ", x)
  trimws(substr(x, 1L, 300L))
}

mem_report <- function() {
  hwm <- tryCatch(grep("VmHWM", readLines("/proc/self/status"), value = TRUE), error = function(e) character())
  peak <- NA_real_
  for (p in c("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes")) {
    if (file.exists(p)) peak <- tryCatch(as.numeric(readLines(p, warn = FALSE)) / 2^20, error = function(e) NA_real_)
    if (is.finite(peak)) break
  }
  logmsg("memory: R process %s | container cgroup peak %s MiB", if (length(hwm)) gsub("\\s+", " ", hwm) else "n/a",
         if (is.finite(peak)) sprintf("%.0f", peak) else "n/a")
}

new_row <- function(ordinal, status, mode, ...) {
  row <- data.table(
    ordinal = as.integer(ordinal), dbh_m = NA_real_, height_m = NA_real_, status = status, seconds = NA_real_,
    base_z0_m = NA_real_, dbh_height_above_ground_m = NA_real_, volume_m3 = NA_real_, qsm_height_m = NA_real_,
    root_x = NA_real_, root_y = NA_real_, note = "", mode = mode, match_dist_m = NA_real_
  )
  extra <- list(...)
  for (nm in names(extra)) set(row, 1L, nm, extra[[nm]])
  row
}

append_row <- function(row, path) {
  row <- copy(row[, CONTRACT_COLS, with = FALSE])
  for (col in names(ROUND_DIGITS)) set(row, 1L, col, round(row[[col]], ROUND_DIGITS[[col]]))
  fresh <- !file.exists(path) || file.size(path) == 0
  fwrite(row, path, append = !fresh, col.names = fresh, na = "", quote = TRUE)
}

# ---------------------------------------------------------------------------------------------------------------------
# ground surface from Z - Z0 (3DFin DTM) or Z - hag (arbor)
# ---------------------------------------------------------------------------------------------------------------------
# Returns function(x, y) -> ground elevation at (x, y): plane fitted to the cell medians within a growing radius
# (handles slope), NA if no data within 6 m.
make_ground <- function(x, y, g, cell = 0.5) {
  ok <- is.finite(x) & is.finite(y) & is.finite(g)
  gt <- data.table(cx = floor(x[ok] / cell), cy = floor(y[ok] / cell), g = g[ok])
  gt <- gt[, list(g = stats::median(g)), by = list(cx, cy)]
  gx <- (gt$cx + 0.5) * cell
  gy <- (gt$cy + 0.5) * cell
  gg <- gt$g
  function(px, py) {
    vapply(seq_along(px), function(i) {
      for (r in c(1, 2, 4, 6)) {
        sel <- which((gx - px[i])^2 + (gy - py[i])^2 <= r^2)
        if (length(sel) >= 6L) {
          dx <- gx[sel] - px[i]
          dy <- gy[sel] - py[i]
          fit <- tryCatch(stats::lm.fit(cbind(1, dx, dy), gg[sel])$coefficients[[1L]], error = function(e) NA_real_)
          if (is.finite(fit)) return(fit)
          return(stats::median(gg[sel]))
        }
        if (length(sel) >= 1L) return(stats::median(gg[sel]))
      }
      NA_real_
    }, numeric(1))
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# measurements on one QSM
# ---------------------------------------------------------------------------------------------------------------------
# DBH at 1.3 m ABOVE THE GROUND.
# arbor's qsm_dbh(q, bh) measures at PATH DISTANCE `bh` from the QSM root (which is prolonged down to hag = 0), so bh = 1.3 is
# not 1.3 m above the ground, and the xyz it returns is the cylinder joint nearest to bh (quantised to ~0.1 m), not the
# interpolated point. Therefore the measuring height is derived from the stem polyline (axis_ID == 1: start/end of every cylinder,
# ordered by dist_to_root) and the path distance is solved (Newton, piecewise linear => 2-3 steps) until that point is
# TARGET_BH above the ground; only then qsm_dbh(q, bh = distance) is asked for the (smoothly interpolated) diameter.
measure_qsm <- function(q, ground_at) {
  out <- list(dbh = NA_real_, h_bh = NA_real_, bh_param = TARGET_BH, converged = NA, flags = character())
  trunk <- q[q$axis_ID == 1, ]
  trunk <- trunk[order(trunk$dist_to_root), ]
  n <- nrow(trunk)
  if (n == 0L) {
    out$flags <- "no stem axis in the QSM"
    return(out)
  }
  len <- sqrt((trunk$endX - trunk$startX)^2 + (trunk$endY - trunk$startY)^2 + (trunk$endZ - trunk$startZ)^2)
  d0 <- trunk$dist_to_root
  d_lo <- d0[[1L]]
  d_hi <- d0[[n]]
  point_at <- function(d) {
    i <- max(1L, min(n, findInterval(d, d0)))
    f <- if (len[[i]] > 0) min(1, max(0, (d - d0[[i]]) / len[[i]])) else 0
    list(x = trunk$startX[[i]] + f * (trunk$endX[[i]] - trunk$startX[[i]]),
         y = trunk$startY[[i]] + f * (trunk$endY[[i]] - trunk$startY[[i]]),
         z = trunk$startZ[[i]] + f * (trunk$endZ[[i]] - trunk$startZ[[i]]),
         dz_dd = if (len[[i]] > 0) (trunk$endZ[[i]] - trunk$startZ[[i]]) / len[[i]] else 1)
  }
  d <- TARGET_BH
  for (i in seq_len(BH_MAX_ITER)) {
    p <- point_at(d)
    h <- p$z - ground_at(p$x, p$y)
    out$bh_param <- d
    if (!is.finite(h)) {
      out$h_bh <- NA_real_
      out$converged <- FALSE
      out$flags <- c(out$flags, "no_ground_reference")
      break
    }
    out$h_bh <- h
    out$converged <- abs(h - TARGET_BH) <= BH_TOLERANCE
    if (out$converged) break
    d <- max(0.05, d + (TARGET_BH - h) / max(0.3, p$dz_dd))
  }
  if (isFALSE(out$converged) && is.finite(out$h_bh)) out$flags <- c(out$flags, "bh_not_converged")
  # qsm_dbh() returns the RADIUS instead of the diameter when the distance lies outside the stem edges
  if (out$bh_param <= d_lo || out$bh_param >= d_hi) {
    out$flags <- c(out$flags, "breast height outside the stem")
  } else {
    out$dbh <- qsm_dbh(q, bh = out$bh_param)$dbh
  }
  out
}

# Everything for one finished QSM -> list of row fields (status, dbh_m, ...). `ground_at(x, y)`: true ground elevation.
describe_qsm <- function(q, ground_at) {
  if (is.null(q) || nrow(q) == 0L) return(list(status = "failed", note = "empty QSM"))
  stats <- tryCatch(qsm_stats(q, bh = TARGET_BH), error = function(e) NULL)
  if (is.null(stats) || is.null(stats$stats_global)) return(list(status = "failed", note = "qsm_stats failed"))
  g <- stats$stats_global
  m <- measure_qsm(q, ground_at)
  gz_root <- ground_at(g$X_root, g$Y_root)
  fields <- list(
    dbh_m = m$dbh,
    height_m = if (is.finite(gz_root)) max(q$endZ) - gz_root else NA_real_,
    dbh_height_above_ground_m = m$h_bh,
    volume_m3 = g$V,
    qsm_height_m = g$H,
    root_x = g$X_root,
    root_y = g$Y_root
  )
  notes <- c(stats$note, m$flags)
  if (!is.finite(gz_root)) notes <- c(notes, "no_ground_reference")
  fields$note <- clean_note(unique(notes[nzchar(notes)]))
  fields$status <- if (is.finite(m$dbh) && m$dbh > 0) "ok" else "no_dbh"
  # arbor returns a placeholder QSM with zero volume when it could not build one at all
  if (any(grepl("QSM Generation Failed|No wood point", stats$note))) fields$status <- "failed"
  # like the SmartQSM stage: dbh_m / height_m are empty unless status == ok (qsm_height_m, volume_m3 stay as diagnostics)
  if (fields$status != "ok") {
    fields$dbh_m <- NA_real_
    fields$height_m <- NA_real_
  }
  fields
}

# "qsm.skeleton_node_distance=0.05,semantic.xxx=2" -> arbor_parameters_default with those values replaced (keys must exist)
apply_param_overrides <- function(params, spec) {
  if (is.null(spec) || !nzchar(spec)) return(params)
  for (item in strsplit(spec, ",", fixed = TRUE)[[1L]]) {
    kv <- strsplit(item, "=", fixed = TRUE)[[1L]]
    if (length(kv) != 2L) stop("bad --params item '", item, "' (expected section.key=value)")
    path <- strsplit(trimws(kv[[1L]]), ".", fixed = TRUE)[[1L]]
    if (length(path) != 2L || is.null(params[[path[[1L]]]]) || is.null(params[[path[[1L]]]][[path[[2L]]]])) {
      stop("unknown arbor parameter '", kv[[1L]], "'")
    }
    old <- params[[path[[1L]]]][[path[[2L]]]]
    val <- trimws(kv[[2L]])
    new <- if (is.logical(old)) as_flag(val) else if (is.integer(old)) suppressWarnings(as.integer(val)) else if (is.numeric(old)) suppressWarnings(as.numeric(val)) else val
    if (length(new) != 1L || (!is.character(new) && is.na(new))) stop("bad value '", val, "' for arbor parameter ", kv[[1L]])
    params[[path[[1L]]]][[path[[2L]]]] <- new
    logmsg("arbor parameter %s: %s -> %s", kv[[1L]], format(old), format(new))
  }
  params
}

# ---------------------------------------------------------------------------------------------------------------------
# per_tree mode
# ---------------------------------------------------------------------------------------------------------------------
DROP_EXTRA <- c("Class", "Split", "dist_axes", "tree_ID")   # 3DFin extra dims the QSM chain does not need

# Synthetic ground for a lone tree (arbor's add_single_tree_ground puts it flat at min(Z)): here the 1000 points lie on the
# local ground surface (3DFin DTM, `ground_at`) so that hag measured from them is the true height above ground.
add_dtm_ground <- function(las, ground_at, n = 1000L) {
  n0 <- npoints(las)
  s <- add_single_tree_ground(las, n = n)
  idx <- (n0 + 1L):npoints(s)
  gz <- ground_at(s@data$X[idx], s@data$Y[idx])
  gz[!is.finite(gz)] <- min(las$Z)
  data.table::set(s@data, i = idx, j = "Z", value = gz)
  s
}

# The arbor chain of ONE tree cloud (see the header). Returns list(q = QSM or NULL, n_wood, err); arbor errors are caught here
# ("assign_wood_from_passage: no passage points found", ...) and become status `failed`.
lone_tree_qsm <- function(las, ground_at, params) {
  tryCatch(
    {
      n0 <- npoints(las)
      set.seed(1L)                                   # add_single_tree_ground draws random points: keep every tree reproducible
      las <- quiet(add_dtm_ground(las, ground_at))   # the path finder of segment_semantic needs ground points to run to
      las <- quiet(segment_ground(las, params))      # -> Classification + hag relative to those ground points
      las <- quiet(wood_likelihood(las, params))
      las <- quiet(segment_semantic(las, params))
      las <- las[seq_len(n0)]                        # the synthetic points are the LAST rows: drop them again
      las@data$treeID <- 1L
      n_wood <- sum(las$foliage == 0L)
      q <- NULL
      utils::capture.output(q <- suppressWarnings(qsm(las, params)))
      list(q = q, n_wood = n_wood)
    },
    error = function(e) list(q = NULL, n_wood = NA_integer_, err = conditionMessage(e))
  )
}

process_tree <- function(ordinal, trow, file, opt, params) {
  t0 <- proc.time()[["elapsed"]]
  base <- num1(trow$base_z0_m)
  las <- quiet(readTLS(file, select = "xyz0"))
  drop <- intersect(DROP_EXTRA, names(las@data))
  if (length(drop)) las@data[, (drop) := NULL]

  notes <- character()
  if ("Z0" %in% names(las@data)) {
    ground_at <- make_ground(las$X, las$Y, las$Z - las$Z0, cell = 0.25)
  } else {
    z_ref <- min(las$Z) - (if (is.finite(base)) base else 0.1)
    ground_at <- function(px, py) rep(z_ref, length(px))
    las@data$Z0 <- las$Z - z_ref
    notes <- c(notes, "no_Z0_flat_ground_assumed")
  }
  if (opt$homogenize) las <- quiet(hybrid_homogeneization(las, 0.02))
  built <- lone_tree_qsm(las, ground_at, params)
  rm(las)

  f <- if (is.null(built$q)) {
    list(status = "failed", note = clean_note(c(built$err, sprintf("no QSM (%s wood points)", format(built$n_wood)))))
  } else {
    describe_qsm(built$q, ground_at)
  }
  f$note <- clean_note(c(f$note, notes))
  row <- do.call(new_row, c(list(ordinal = ordinal, status = f$status, mode = "per_tree", base_z0_m = base), f[setdiff(names(f), "status")]))
  row$seconds <- proc.time()[["elapsed"]] - t0
  row
}

tree_file_for <- function(ordinal, trow, trees_dir, n_trees) {
  rel <- if ("file" %in% names(trow)) as.character(trow$file) else ""
  if (length(rel) == 1L && !is.na(rel) && nzchar(rel)) return(file.path(trees_dir, basename(rel)))
  file.path(trees_dir, sprintf("tree_%0*d.laz", max(4L, nchar(as.character(n_trees))), ordinal))
}

# NB: a variable called `ordinal` inside `tm[...]` would be resolved as the COLUMN, hence the helper with another argument name
tm_row <- function(tm, ord) tm[which(tm$ordinal == ord), ]

select_ordinals <- function(tm, opt) {
  ords <- tm$ordinal
  if (!is.null(opt[["trees"]]) && nzchar(opt[["trees"]])) {
    want <- suppressWarnings(as.integer(strsplit(opt[["trees"]], ",")[[1L]]))
    if (anyNA(want)) stop("--trees must be a comma separated list of integers, got ", opt[["trees"]])
    unknown <- setdiff(want, ords)
    if (length(unknown)) stop("ordinals not in tree_map.csv: ", paste(unknown, collapse = ","))
    ords <- sort(unique(want))
  }
  ords
}

done_ordinals <- function(path) {
  if (!file.exists(path) || file.size(path) == 0) return(integer())
  d <- tryCatch(fread(path, select = "ordinal"), error = function(e) stop("cannot resume: ", path, " is unreadable (", conditionMessage(e), "); rerun without --resume"))
  header <- names(fread(path, nrows = 0L))
  if (!identical(header, CONTRACT_COLS)) stop("cannot resume: ", path, " has other columns than this script writes; rerun without --resume")
  as.integer(d$ordinal)
}

run_per_tree <- function(opt, tm, params) {
  if (is.null(opt$trees_dir) || !dir.exists(opt$trees_dir)) stop("--trees_dir missing or not a directory: ", opt$trees_dir)
  ords <- select_ordinals(tm, opt)
  done <- if (opt$resume) done_ordinals(opt$out_csv) else integer()
  todo <- setdiff(ords, done)
  if (!is.na(opt$limit)) todo <- head(todo, opt$limit)
  logmsg("per_tree: %d tree(s) selected, %d already in the CSV, %d to process%s", length(ords), length(intersect(ords, done)),
         length(todo), if (is.na(opt$limit)) "" else sprintf(" (limit %d)", opt$limit))
  skip_low <- as_flag(opt$skip_low_points)
  t_all <- proc.time()[["elapsed"]]
  counts <- list()
  for (k in seq_along(todo)) {
    ordinal <- todo[[k]]
    trow <- tm_row(tm, ordinal)
    row <- tryCatch(
      {
        if (skip_low && as_flag(trow$low_points)) {
          new_row(ordinal, "skipped_low_points", "per_tree", base_z0_m = num1(trow$base_z0_m), note = "low_points in tree_map.csv")
        } else {
          file <- tree_file_for(ordinal, trow, opt$trees_dir, nrow(tm))
          if (!file.exists(file)) {
            new_row(ordinal, "error", "per_tree", base_z0_m = num1(trow$base_z0_m), note = clean_note(paste("tree file not found:", file)))
          } else {
            process_tree(ordinal, trow, file, opt, params)
          }
        }
      },
      error = function(e) new_row(ordinal, "error", "per_tree", base_z0_m = num1(trow$base_z0_m), note = clean_note(conditionMessage(e)))
    )
    if (is.na(row$seconds)) row$seconds <- 0
    append_row(row, opt$out_csv)
    counts[[row$status]] <- (if (is.null(counts[[row$status]])) 0L else counts[[row$status]]) + 1L
    logmsg("tree %4d [%d/%d] %-18s dbh %s m  height %s m  (%.1f s)%s", ordinal, k, length(todo), row$status,
           if (is.na(row$dbh_m)) "-" else sprintf("%.3f", row$dbh_m),
           if (is.na(row$height_m)) "-" else sprintf("%.2f", row$height_m), row$seconds,
           if (nzchar(row$note)) paste0("  | ", row$note) else "")
    if (k %% 10L == 0L) gc(verbose = FALSE)
  }
  logmsg("per_tree finished: %d tree(s) in %.1f s; %s", length(todo), proc.time()[["elapsed"]] - t_all,
         paste(sprintf("%s=%d", names(counts), unlist(counts)), collapse = " "))
}

# ---------------------------------------------------------------------------------------------------------------------
# plot mode
# ---------------------------------------------------------------------------------------------------------------------
# one-to-one nearest-neighbour matching of arbor trees (ax, ay) to 3DFin trees (tx, ty), greedy by smallest distance
match_trees <- function(ax, ay, tx, ty, max_dist) {
  n_a <- length(ax)
  n_t <- length(tx)
  if (n_a == 0L || n_t == 0L) return(data.table(a = integer(), t = integer(), dist = numeric()))
  d <- sqrt(outer(ax, tx, function(a, b) (a - b)^2) + outer(ay, ty, function(a, b) (a - b)^2))
  d[!is.finite(d)] <- Inf
  cand <- which(d <= max_dist, arr.ind = TRUE)
  if (nrow(cand) == 0L) return(data.table(a = integer(), t = integer(), dist = numeric()))
  pairs <- data.table(a = cand[, 1L], t = cand[, 2L], dist = d[cand])
  setorder(pairs, dist)
  used_a <- logical(n_a)
  used_t <- logical(n_t)
  keep <- logical(nrow(pairs))
  for (i in seq_len(nrow(pairs))) {
    if (!used_a[pairs$a[i]] && !used_t[pairs$t[i]]) {
      keep[i] <- TRUE
      used_a[pairs$a[i]] <- TRUE
      used_t[pairs$t[i]] <- TRUE
    }
  }
  pairs[keep]
}

run_plot <- function(opt, tm, params) {
  if (is.null(opt$plot_las) || !file.exists(opt$plot_las)) stop("--plot_las missing or not a file: ", opt$plot_las)
  ords <- select_ordinals(tm, opt)
  t_all <- proc.time()[["elapsed"]]
  step <- function(label, expr) {
    t0 <- proc.time()[["elapsed"]]
    value <- quiet(expr)
    logmsg("%-34s %7.1f s", label, proc.time()[["elapsed"]] - t0)
    value
  }

  logmsg("plot: reading %s", opt$plot_las)
  las <- step("read", suppressWarnings(readTLS(opt$plot_las, select = "xyz0")))
  logmsg("plot: %.2f M points", npoints(las) / 1e6)
  drop <- intersect(DROP_EXTRA, names(las@data))
  if (length(drop)) las@data[, (drop) := NULL]
  has_z0 <- "Z0" %in% names(las@data)

  las <- step("hybrid_homogeneization", hybrid_homogeneization(las, 0.02))
  logmsg("plot: %.2f M points after homogeneization", npoints(las) / 1e6)
  las <- step("segment_ground", segment_ground(las, params))
  gc(verbose = FALSE)
  # common ground reference for the breast-height / height numbers: 3DFin DTM (Z - Z0) like per_tree, else arbor's (Z - hag)
  ground_at <- make_ground(las$X, las$Y, if (has_z0) las$Z - las$Z0 else las$Z - las$hag, cell = 0.5)
  logmsg("plot: ground reference = %s", if (has_z0) "3DFin DTM (Z - Z0)" else "arbor ground (Z - hag), no Z0 in the cloud")
  las <- step("wood_likelihood", wood_likelihood(las, params))
  las <- step("segment_semantic", segment_semantic(las, params))
  seeds <- step("find_seeds", find_seeds(las, params))
  las <- step("segment_instance", segment_instance(las, seeds, params))
  las <- step("flag_small_trees", flag_small_trees(las, max_height = 2))
  logmsg("plot: %d seed(s)", nrow(seeds@data))
  rm(seeds)
  gc(verbose = FALSE)
  qs <- step("qsf", qsf(las, min_height = 2, params = params))
  rm(las)
  gc(verbose = FALSE)
  t_chain <- proc.time()[["elapsed"]] - t_all
  logmsg("plot: %d QSM(s) built in %.1f s", length(qs), t_chain)
  mem_report()

  # measure every arbor tree, then match
  rows <- lapply(seq_along(qs), function(i) {
    q <- qs[[i]]
    tid <- attr(q, "id")
    f <- tryCatch(describe_qsm(q, ground_at), error = function(e) list(status = "error", note = clean_note(conditionMessage(e))))
    list(arbor_id = if (is.null(tid)) NA_integer_ else as.integer(tid), f = f)
  })
  ax <- vapply(rows, function(r) num1(r$f$root_x), numeric(1))
  ay <- vapply(rows, function(r) num1(r$f$root_y), numeric(1))
  usable <- which(is.finite(ax) & is.finite(ay))
  cand_t <- which(is.finite(tm$x_m) & is.finite(tm$y_m) & tm$ordinal %in% ords)
  pairs <- match_trees(ax[usable], ay[usable], tm$x_m[cand_t], tm$y_m[cand_t], opt$max_match_dist)
  pairs[, `:=`(a = usable[a], t = cand_t[t])]
  unmatched_arbor <- setdiff(seq_along(rows), pairs$a)
  logmsg("plot: %d of %d arbor tree(s) matched to a 3DFin ordinal (max %.2f m); %d dropped", nrow(pairs), length(rows), opt$max_match_dist, length(unmatched_arbor))
  if (length(unmatched_arbor)) {
    show <- head(unmatched_arbor, 20L)
    logmsg("plot: unmatched arbor tree(s): %s%s", paste(sprintf("%d@(%.1f,%.1f)", vapply(rows[show], function(r) r$arbor_id, integer(1)), ax[show], ay[show]), collapse = " "),
           if (length(unmatched_arbor) > 20L) " ..." else "")
  }

  share <- t_chain / max(1L, length(qs))
  skip_low <- as_flag(opt$skip_low_points)
  by_ordinal <- setNames(vector("list", length(ords)), as.character(ords))
  for (j in seq_len(nrow(pairs))) {
    r <- rows[[pairs$a[j]]]
    trow <- tm[pairs$t[j], ]
    f <- r$f
    f$note <- clean_note(c(f$note, sprintf("arbor_treeID=%s", r$arbor_id)))
    by_ordinal[[as.character(trow$ordinal)]] <- do.call(new_row, c(
      list(ordinal = trow$ordinal, status = f$status, mode = "plot", base_z0_m = num1(trow$base_z0_m), seconds = share, match_dist_m = pairs$dist[j]),
      f[setdiff(names(f), "status")]
    ))
  }
  out <- rbindlist(lapply(ords, function(ordinal) {
    trow <- tm_row(tm, ordinal)
    if (skip_low && as_flag(trow$low_points)) {
      return(new_row(ordinal, "skipped_low_points", "plot", base_z0_m = num1(trow$base_z0_m), note = "low_points in tree_map.csv"))
    }
    row <- by_ordinal[[as.character(ordinal)]]
    if (is.null(row)) {
      reason <- if (!is.finite(trow$x_m) || !is.finite(trow$y_m)) "3DFin tree has no x_m/y_m" else sprintf("no arbor tree within %.2f m", opt$max_match_dist)
      row <- new_row(ordinal, "failed", "plot", base_z0_m = num1(trow$base_z0_m), seconds = share, note = reason)
    }
    row
  }))
  if (!is.na(opt$limit)) out <- head(out, opt$limit)
  if (file.exists(opt$out_csv)) file.remove(opt$out_csv)
  for (i in seq_len(nrow(out))) append_row(out[i, ], opt$out_csv)
  counts <- table(out$status)
  logmsg("plot finished in %.1f s; %s", proc.time()[["elapsed"]] - t_all, paste(sprintf("%s=%d", names(counts), as.integer(counts)), collapse = " "))
}

# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------
main <- function() {
  parser <- OptionParser(
    usage = "arbor_metrics.R --mode per_tree|plot --tree_map CSV --out_csv CSV [--trees_dir DIR | --plot_las LAS] [options]",
    option_list = list(
      make_option("--mode", type = "character", default = "per_tree", help = "per_tree or plot [default %default]"),
      make_option("--trees_dir", type = "character", default = NULL, help = "per_tree: folder with tree_XXXX.laz"),
      make_option("--tree_map", type = "character", default = NULL, help = "split/tree_map.csv"),
      make_option("--plot_las", type = "character", default = NULL, help = "plot: the 3DFin cloud *_tree_ID_dist_axes.las"),
      make_option("--out_csv", type = "character", default = NULL, help = "metrics.csv to write"),
      make_option("--trees", type = "character", default = NULL, help = "comma separated ordinals (default: all in tree_map)"),
      make_option("--limit", type = "integer", default = NA_integer_, help = "process at most N trees (after --resume filtering)"),
      make_option("--resume", action = "store_true", default = FALSE, help = "per_tree: skip ordinals already in out_csv"),
      make_option("--skip_low_points", type = "character", default = "true", help = "true|false: mark tree_map low_points trees as skipped [default %default]"),
      make_option("--homogenize", action = "store_true", default = FALSE, help = "per_tree: arbor voxel homogeneization (res 2 cm) before the chain"),
      make_option("--params", type = "character", default = NULL, help = "arbor parameter overrides 'section.key=value,section.key=value'"),
      make_option("--max_match_dist", type = "double", default = 1.0, help = "plot: max distance arbor root - 3DFin tree [default %default m]"),
      make_option("--threads", type = "integer", default = 4L, help = "lidR/data.table threads (OpenMP threads of arbor: env OMP_NUM_THREADS) [default %default]")
    )
  )
  opt <- parse_args(parser)
  for (req in c("tree_map", "out_csv")) if (is.null(opt[[req]])) stop("missing --", req)
  if (!opt$mode %in% c("per_tree", "plot")) stop("--mode must be per_tree or plot, got ", opt$mode)
  options(warn = 1)
  set_lidr_threads(opt$threads)
  data.table::setDTthreads(opt$threads)
  logmsg("arbor %s | lidR %s | mode %s | threads: lidR %d, arbor/OpenMP %s (OMP_NUM_THREADS=%s)", as.character(packageVersion("arbor")),
         as.character(packageVersion("lidR")), opt$mode, get_lidr_threads(), tryCatch(arbor:::getThreads(), error = function(e) "?"), Sys.getenv("OMP_NUM_THREADS"))
  if (!file.exists(opt$tree_map)) stop("tree_map not found: ", opt$tree_map)
  tm <- fread(opt$tree_map)
  if (!"ordinal" %in% names(tm)) stop(opt$tree_map, " has no 'ordinal' column")
  for (col in c("x_m", "y_m", "base_z0_m")) if (!col %in% names(tm)) tm[[col]] <- NA_real_
  if (!"low_points" %in% names(tm)) tm$low_points <- FALSE
  dir.create(dirname(opt$out_csv), recursive = TRUE, showWarnings = FALSE)
  params <- apply_param_overrides(arbor_parameters_default, opt$params)

  if (opt$mode == "per_tree") run_per_tree(opt, tm, params) else run_plot(opt, tm, params)
  mem_report()
  invisible(NULL)
}

tryCatch(
  main(),
  error = function(e) {
    cat("FATAL: ", conditionMessage(e), "\n", sep = "", file = stderr())
    quit(save = "no", status = 1L)
  }
)

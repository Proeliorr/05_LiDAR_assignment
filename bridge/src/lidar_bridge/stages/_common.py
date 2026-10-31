"""Helpers shared by the stage modules."""

from __future__ import annotations

import hashlib
from pathlib import Path


class StageError(RuntimeError):
    """A pipeline stage failed (bad input, missing outputs, ...). The message is shown to the user as is."""


# -- split fingerprint ------------------------------------------------------------------------------------------
# The QSM stages (smartqsm, arbor) resume from earlier results, and compare reads whatever metrics files exist. All of
# them are only valid for the per-tree clouds they were computed on. ``split/tree_map.csv`` changes whenever the split
# result changes (other 3DFin parameters, max_axis_dist, min_z0, ... all alter n_points / base height / positions), so a
# hash of it tells a stage whether its stored results belong to the current split. Each stage writes the fingerprint
# next to its metrics when it finishes and compares it with the current one before it reuses or reads anything.

FINGERPRINT_FILE = "source_fingerprint.txt"


def tree_map_fingerprint(split_dir: Path) -> str | None:
    """SHA-256 (first 16 hex digits) of ``<split_dir>/tree_map.csv``; ``None`` when that file does not exist."""
    path = Path(split_dir) / "tree_map.csv"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_fingerprint(stage_dir: Path, fingerprint: str | None, *, extra: str = "") -> None:
    """Record which split (and, via ``extra``, which stage settings that matter, e.g. ``homogenize=false``) the results
    in ``stage_dir`` were computed on. A ``None`` fingerprint writes nothing."""
    if fingerprint is None:
        return
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    (Path(stage_dir) / FINGERPRINT_FILE).write_text(f"{fingerprint}\n{extra}\n", encoding="utf-8")


def read_fingerprint(stage_dir: Path) -> tuple[str, str] | None:
    """``(fingerprint, extra)`` stored by :func:`write_fingerprint`, or ``None`` when absent / unreadable."""
    try:
        lines = (Path(stage_dir) / FINGERPRINT_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines or not lines[0].strip():
        return None
    return lines[0].strip(), (lines[1].strip() if len(lines) > 1 else "")

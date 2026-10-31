"""Reading the tree-level parameters out of a SmartQSM ``<stem>_qsm.mat`` file.

SmartQSM writes a MATLAB v5 file holding one variable ``QSM`` (a struct with ``cylinder``, ``branch``, ``treedata``,
``rundata``). ``QSM.treedata`` is a 1x1 struct of scalar tree parameters (63 fields in SmartQSM 2.3.1), e.g. ``DBH_cm``
(centimetres; 0 when the trunk does not reach 1.3 m above the lowest point of the cloud), ``height_m`` (max z - min z of
the cloud), ``num_branches``, ``max_order``, ``stem_volume_L``, ``trunk_volume_L``, ``X_m``, ``Y_m``.

::

    >>> from lidar_bridge.qsm_mat import read_treedata, treedata_fields
    >>> read_treedata("tree_0003_qsm.mat")["DBH_cm"]
    16.895641326904297
    $ python -m lidar_bridge.qsm_mat tree_0003_qsm.mat        # list every field with its value (debugging)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


class QsmMatError(RuntimeError):
    """The .mat file is absent, corrupt, or does not have the ``QSM.treedata`` layout."""


def _first(obj: Any) -> Any:
    """squeeze_me normally unwraps 1x1 containers; do the same for arrays that survived (several structs -> the first)."""
    while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size >= 1:
        obj = obj.reshape(-1)[0]
    return obj


def _treedata_struct(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        raise QsmMatError(f"QSM file not found: {p}")
    if p.stat().st_size == 0:
        raise QsmMatError(f"QSM file is empty (interrupted write?): {p}")
    try:
        mat = loadmat(str(p), squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:  # scipy: MATLAB v7.3 (HDF5) is not supported
        raise QsmMatError(f"{p} is a MATLAB v7.3 (HDF5) file, which scipy cannot read: {exc}") from exc
    except Exception as exc:  # truncated / garbage files raise a zoo of ValueError, TypeError, struct.error, zlib.error ...
        raise QsmMatError(f"cannot read {p} as a MATLAB file ({type(exc).__name__}: {exc}); corrupt or truncated?") from exc
    if "QSM" not in mat:
        names = sorted(k for k in mat if not k.startswith("__"))
        raise QsmMatError(f"{p} has no 'QSM' variable (found {names})")
    qsm = _first(mat["QSM"])
    treedata = getattr(qsm, "treedata", None)
    if treedata is None and isinstance(qsm, dict):
        treedata = qsm.get("treedata")
    treedata = _first(treedata)
    if treedata is None or not (hasattr(treedata, "_fieldnames") or isinstance(treedata, dict)):
        raise QsmMatError(f"{p}: QSM has no 'treedata' struct")
    return treedata


def _names(treedata: Any) -> list[str]:
    return list(treedata) if isinstance(treedata, dict) else list(treedata._fieldnames)


def _get(treedata: Any, name: str) -> Any:
    return treedata[name] if isinstance(treedata, dict) else getattr(treedata, name)


def _as_float(value: Any) -> float | None:
    """A numeric scalar as float; ``nan`` for an empty array; ``None`` for anything that is not a single number."""
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype.kind not in "biuf":  # strings, nested structs, cells, complex ...
        return None
    if arr.size == 0:
        return float("nan")
    if arr.size != 1:
        return None
    return float(arr.reshape(()))


def treedata_fields(path: str | Path) -> list[str]:
    """Names of all fields of ``QSM.treedata`` in file order (for debugging / finding the right field)."""
    return _names(_treedata_struct(path))


def read_treedata(path: str | Path) -> dict[str, float]:
    """``QSM.treedata`` of a SmartQSM ``*_qsm.mat`` as ``{field: float}``.

    Robust to missing fields (they are simply absent from the result: use ``.get``), to 1x1 / 0-d array shapes,
    and to non-numeric or array-valued fields (skipped). An empty numeric field becomes ``nan``.

    Raises
    ------
    QsmMatError
        The file does not exist, is empty / truncated / not a MATLAB file, or has no ``QSM.treedata`` struct.
    """
    treedata = _treedata_struct(path)
    out: dict[str, float] = {}
    for name in _names(treedata):
        try:
            value = _as_float(_get(treedata, name))
        except AttributeError:
            continue
        if value is not None:
            out[name] = value
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m lidar_bridge.qsm_mat <tree_XXXX_qsm.mat> [...]", file=sys.stderr)
        return 2
    status = 0
    for arg in args:
        try:
            data = read_treedata(arg)
        except QsmMatError as exc:
            print(f"{arg}: {exc}", file=sys.stderr)
            status = 1
            continue
        print(f"{arg}: {len(data)} numeric treedata fields")
        for name, value in data.items():
            print(f"  {name:<40s}{value:.6g}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

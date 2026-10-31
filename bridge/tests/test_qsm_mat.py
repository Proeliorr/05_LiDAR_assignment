from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from lidar_bridge import qsm_mat
from lidar_bridge.qsm_mat import QsmMatError, read_treedata, treedata_fields

REAL_MAT = Path(__file__).resolve().parents[2] / "data/out/dev_smartqsm/spconv_cpu/tree_0001_qsm.mat"


def write_qsm(path: Path, treedata: dict, **qsm_extra) -> Path:
    """A .mat laid out like SmartQSM's: variable QSM = struct(cylinder, branch, treedata, rundata)."""
    savemat(path, {"QSM": {"cylinder": {"n": 3.0}, "branch": {"n": 2.0}, "treedata": treedata, "rundata": {"start": 0}, **qsm_extra}})
    return path


def test_scalars_come_back_as_floats_in_file_order(tmp_path):
    p = write_qsm(tmp_path / "t_qsm.mat", {"X_m": 2.5, "DBH_cm": 16.9, "num_branches": np.int32(12), "height_m": np.float32(12.8)})
    data = read_treedata(p)
    assert list(data) == ["X_m", "DBH_cm", "num_branches", "height_m"]
    assert data["DBH_cm"] == pytest.approx(16.9)
    assert data["num_branches"] == 12.0 and isinstance(data["num_branches"], float)
    assert data["height_m"] == pytest.approx(12.8, rel=1e-6)  # float32 survives


def test_field_list_helper(tmp_path):
    p = write_qsm(tmp_path / "t_qsm.mat", {"a": 1.0, "b": "text", "c": [1.0, 2.0]})
    assert treedata_fields(p) == ["a", "b", "c"]  # all fields, also the non-numeric ones


def test_missing_fields_are_absent_not_an_error(tmp_path):
    data = read_treedata(write_qsm(tmp_path / "t_qsm.mat", {"DBH_cm": 10.0}))
    assert data == {"DBH_cm": 10.0}
    assert data.get("height_m") is None


def test_odd_shapes_and_types(tmp_path):
    p = write_qsm(
        tmp_path / "t_qsm.mat",
        {"col": np.array([[7.5]]), "vec": np.array([1.0, 2.0, 3.0]), "empty": np.zeros((0, 0)), "name": "oak", "flag": True},
    )
    data = read_treedata(p)
    assert data["col"] == 7.5  # 1x1 matrix
    assert "vec" not in data and "name" not in data  # not a scalar / not a number
    assert np.isnan(data["empty"])
    assert data["flag"] == 1.0


def test_absent_file(tmp_path):
    with pytest.raises(QsmMatError, match="not found"):
        read_treedata(tmp_path / "nope_qsm.mat")


def test_empty_file(tmp_path):
    p = tmp_path / "t_qsm.mat"
    p.write_bytes(b"")
    with pytest.raises(QsmMatError, match="empty"):
        read_treedata(p)


def test_garbage_file(tmp_path):
    p = tmp_path / "t_qsm.mat"
    p.write_bytes(b"this is definitely not a matlab file" * 20)
    with pytest.raises(QsmMatError, match="cannot read"):
        read_treedata(p)


def test_truncated_file(tmp_path):
    good = write_qsm(tmp_path / "good_qsm.mat", {f"field_{i}": float(i) for i in range(60)})
    cut = tmp_path / "cut_qsm.mat"
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    with pytest.raises(QsmMatError):
        read_treedata(cut)


def test_file_without_qsm_variable(tmp_path):
    p = tmp_path / "other.mat"
    savemat(p, {"something_else": 1.0})
    with pytest.raises(QsmMatError, match="no 'QSM' variable"):
        read_treedata(p)


def test_qsm_without_treedata(tmp_path):
    p = tmp_path / "t_qsm.mat"
    savemat(p, {"QSM": {"cylinder": {"n": 1.0}}})
    with pytest.raises(QsmMatError, match="no 'treedata'"):
        read_treedata(p)


def test_treedata_that_is_not_a_struct(tmp_path):
    p = tmp_path / "t_qsm.mat"
    savemat(p, {"QSM": {"treedata": np.array([1.0, 2.0])}})
    with pytest.raises(QsmMatError, match="no 'treedata'"):
        read_treedata(p)


def test_command_line_lists_fields(tmp_path, capsys):
    p = write_qsm(tmp_path / "t_qsm.mat", {"DBH_cm": 12.5})
    assert qsm_mat.main([str(p)]) == 0
    assert "DBH_cm" in capsys.readouterr().out
    assert qsm_mat.main([str(tmp_path / "missing.mat")]) == 1
    assert qsm_mat.main([]) == 2


@pytest.mark.skipif(not REAL_MAT.is_file(), reason="WP2's real SmartQSM result is not in data/out/dev_smartqsm")
def test_real_smartqsm_result():
    fields = treedata_fields(REAL_MAT)
    assert len(fields) == 63 and {"DBH_cm", "height_m", "X_m", "Y_m", "num_branches", "max_order"} <= set(fields)
    assert {"stem_volume_L", "trunk_volume_L"} <= set(fields)
    data = read_treedata(REAL_MAT)
    assert 10 < data["DBH_cm"] < 30 and 10 < data["height_m"] < 16
    assert float(data["max_order"]).is_integer()

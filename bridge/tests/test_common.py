"""Tests of the split fingerprint helpers shared by the smartqsm, arbor and compare stages."""

from __future__ import annotations

from lidar_bridge.stages._common import (
    FINGERPRINT_FILE,
    read_fingerprint,
    tree_map_fingerprint,
    write_fingerprint,
)


def test_fingerprint_is_none_without_a_tree_map(tmp_path):
    assert tree_map_fingerprint(tmp_path) is None


def test_fingerprint_follows_the_content_of_tree_map_csv(tmp_path):
    (tmp_path / "tree_map.csv").write_text("ordinal,n_points\n1,500\n", encoding="utf-8")
    first = tree_map_fingerprint(tmp_path)
    assert first and len(first) == 16
    assert tree_map_fingerprint(tmp_path) == first  # stable

    (tmp_path / "tree_map.csv").write_text("ordinal,n_points\n1,501\n", encoding="utf-8")
    assert tree_map_fingerprint(tmp_path) != first  # any change of the split result changes it


def test_write_and_read_roundtrip(tmp_path):
    out = tmp_path / "smartqsm" / "layerwise-clustering"
    write_fingerprint(out, "abcdef0123456789", extra="config=1234abcd skip_low_points=true")
    assert (out / FINGERPRINT_FILE).is_file()
    assert read_fingerprint(out) == ("abcdef0123456789", "config=1234abcd skip_low_points=true")


def test_read_without_extra_and_missing_or_empty_file(tmp_path):
    write_fingerprint(tmp_path, "abcdef0123456789")
    assert read_fingerprint(tmp_path) == ("abcdef0123456789", "")
    assert read_fingerprint(tmp_path / "nowhere") is None
    (tmp_path / FINGERPRINT_FILE).write_text("\n", encoding="utf-8")
    assert read_fingerprint(tmp_path) is None


def test_writing_a_none_fingerprint_creates_nothing(tmp_path):
    write_fingerprint(tmp_path / "arbor", None)
    assert not (tmp_path / "arbor").exists()

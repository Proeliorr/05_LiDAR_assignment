"""Shared fixtures: a throw-away fake project tree and a RunContext on it."""

from __future__ import annotations

from pathlib import Path

import pytest

from lidar_bridge.paths import RunContext

# Same layout / case as third_party/3DFin/src/three_d_fin/3DFinconfig.ini, trimmed, plus one key ("dup") that exists
# in two sections to exercise the ambiguity check.
TEMPLATE_INI = """\
# comment
[basic]
z0_name=Z0
upper_limit=3.5
lower_limit=0.7
dup=1
[advanced]
maximum_diameter=1.0
maximum_height=25
dup=2
[expert]
### stem identification ###
number_of_points=1000
minimum_points=20
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Fake project root with data/in, data/out and the 3DFin template submodule path."""
    (tmp_path / "data" / "in").mkdir(parents=True)
    (tmp_path / "data" / "out").mkdir(parents=True)
    template = tmp_path / "third_party" / "3DFin" / "src" / "three_d_fin" / "3DFinconfig.ini"
    template.parent.mkdir(parents=True)
    template.write_text(TEMPLATE_INI, encoding="utf-8")
    (tmp_path / "data" / "in" / "plot_01.laz").write_bytes(b"not a real laz")
    return tmp_path


@pytest.fixture
def ctx(project: Path) -> RunContext:
    return RunContext(
        run_name="r1",
        data_dir=project / "data",
        install_dir=project,
        input=project / "data" / "in" / "plot_01.laz",
    )

"""Stage 1: 3DFin (CLI) in the ``lidar-3dfin`` container.

Contract
--------
Reads ``ctx.input`` (a .las/.laz inside the project) and writes into ``data/out/<run>/dfin/``::

    params.ini                          3DFin parameters (template + ini_overrides), written by this stage
    <stem>.xlsx                         DBH / height tables (or the ``<stem>_*.txt`` files with export_txt=True)
    <stem>_tree_ID_dist_axes.las        whole cloud + tree_ID and dist-to-axis extra dims  -> input of `split`
    <stem>_stripe.las                   stems found in the stripe
    <stem>_[axes|circ|tree_heights|tree_locator|dtm_points].las, <stem>_config.ini  (other 3DFin outputs)

The equivalent manual command is::

    docker run --rm -v <host>/data:/data lidar-3dfin cli /data/<input> /data/out/<run>/dfin \
        /data/out/<run>/dfin/params.ini [--normalize] [--denoise] [--export_txt]
"""

from __future__ import annotations

import configparser
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lidar_bridge.docker_runner import require_image, run_container
from lidar_bridge.paths import PathOutsideProjectError, RunContext
from lidar_bridge.stages._common import StageError

log = logging.getLogger("lidar_bridge.stages.dfin")

#: Parameter template shipped with the (unmodified) 3DFin submodule, relative to the project directory.
TEMPLATE_INI = Path("third_party/3DFin/src/three_d_fin/3DFinconfig.ini")
INI_SECTIONS = ("basic", "advanced", "expert")
LAS_SUFFIXES = (".las", ".laz")
#: Where an input that lies outside ``data/`` (e.g. Example_data/) is mounted, read-only.
EXTRA_INPUT_MOUNT = "/input"


# ---------------------------------------------------------------------------------------------------------
# params.ini
# ---------------------------------------------------------------------------------------------------------
def load_template(project_dir: Path, template: Path | None = None) -> configparser.ConfigParser:
    """Parse the 3DFin template ini, keeping the case of keys."""
    path = project_dir / (template or TEMPLATE_INI)
    if not path.is_file():
        raise StageError(
            f"3DFin parameter template not found: {path}. Is the submodule checked out? "
            "(git submodule update --init third_party/3DFin)"
        )
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]  # keep key case, 3DFin reads exact names
    parser.read(path, encoding="utf-8")
    return parser


def _as_ini_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def flatten_ini_overrides(parser: configparser.ConfigParser, overrides: Mapping[str, Any] | None) -> dict[tuple[str, str], str]:
    """Normalise ``ini_overrides`` to ``{(section, key): "value"}`` and validate against the template.

    Accepted spellings (they can be mixed)::

        {"basic": {"upper_limit": 4.0}}                nested dict (what hydra produces from
                                                       +dfin.ini_overrides.basic.upper_limit=4.0)
        {"basic.upper_limit": 4.0}                     "section.key"
        {"upper_limit": 4.0}                           bare key, if it is unique across the sections

    Unknown sections / keys raise :class:`StageError` (a typo would otherwise be silently ignored by 3DFin).
    """
    flat: dict[tuple[str, str], str] = {}

    def lookup(section: str, key: str) -> tuple[str, str]:
        if section not in parser:
            raise StageError(f"ini_overrides: unknown 3DFin section {section!r}; sections are {list(INI_SECTIONS)}")
        if key not in parser[section]:
            raise StageError(
                f"ini_overrides: unknown key {key!r} in section [{section}]; valid keys: {list(parser[section])}"
            )
        return section, key

    for name, value in (overrides or {}).items():
        if isinstance(value, Mapping):
            if name not in parser:
                raise StageError(f"ini_overrides: unknown 3DFin section {name!r}; sections are {list(INI_SECTIONS)}")
            for key, inner in value.items():
                if isinstance(inner, Mapping):
                    raise StageError(f"ini_overrides: [{name}] {key!r} must be a scalar, got a mapping")
                flat[lookup(name, str(key))] = _as_ini_value(inner)
        elif "." in name:
            section, key = name.split(".", 1)
            flat[lookup(section, key)] = _as_ini_value(value)
        else:
            owners = [s for s in INI_SECTIONS if s in parser and name in parser[s]]
            if not owners:
                raise StageError(f"ini_overrides: unknown 3DFin parameter {name!r}")
            if len(owners) > 1:
                raise StageError(f"ini_overrides: parameter {name!r} is ambiguous, use section.key ({owners})")
            flat[(owners[0], name)] = _as_ini_value(value)
    return flat


def build_params(project_dir: Path, ini_overrides: Mapping[str, Any] | None = None) -> configparser.ConfigParser:
    """Template + overrides, ready to be written."""
    parser = load_template(project_dir)
    for (section, key), value in flatten_ini_overrides(parser, ini_overrides).items():
        parser[section][key] = value
    return parser


def write_params_ini(parser: configparser.ConfigParser, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        parser.write(fh, space_around_delimiters=False)
    return dest


# ---------------------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------------------
def expected_outputs(stem: str, export_txt: bool) -> dict[str, str]:
    """File names (relative to the dfin dir) that must exist for the stage to count as successful."""
    table = f"{stem}_dbh_and_heights.txt" if export_txt else f"{stem}.xlsx"
    return {
        "table": table,
        "tree_las": f"{stem}_tree_ID_dist_axes.las",
        "stripe_las": f"{stem}_stripe.las",
    }


def verify_outputs(out_dir: Path, stem: str, export_txt: bool, output_tail: str = "") -> dict[str, Path]:
    """Check the 3DFin result files (the CLI can exit 0 or carry on after a config validation error)."""
    found: dict[str, Path] = {}
    problems = []
    for key, name in expected_outputs(stem, export_txt).items():
        path = out_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif path.stat().st_size == 0:
            problems.append(f"empty {name}")
        else:
            found[key] = path
    if problems:
        hint = f"\n--- last 3DFin output ---\n{output_tail}" if output_tail else ""
        raise StageError(
            f"3DFin did not produce the expected results in {out_dir}: {', '.join(problems)}. "
            f"Look for 'Parameters:' / 'ValidationError' / 'Input file:' lines in the 3DFin output.{hint}"
        )
    return found


# ---------------------------------------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------------------------------------
def run_dfin(
    ctx: RunContext,
    *,
    normalize: bool = True,
    denoise: bool = False,
    export_txt: bool = False,
    ini_overrides: dict = {},  # noqa: B006  (never mutated; hydra-zen turns it into a config field)
    image: str = "lidar-3dfin",
) -> dict:
    """Run 3DFin on ``ctx.input``.

    Parameters
    ----------
    normalize
        ``--normalize``: height-normalise the cloud with CSF (needed unless the input already has a Z0 field).
    denoise
        ``--denoise``: remove outliers below ground level.
    export_txt
        ``--export_txt``: write ASCII tables instead of the .xlsx workbook.
    ini_overrides
        Changes to the 3DFin parameter template, e.g. ``{"basic": {"upper_limit": 4.0}}`` or
        ``{"basic.upper_limit": 4.0}``. From the command line: ``+dfin.ini_overrides.basic.upper_limit=4.0``.
    image
        Docker image tag (ENTRYPOINT ``3DFin``).

    Returns
    -------
    dict
        ``{"dir", "stem", "table", "tree_las", "stripe_las", "params"}`` as paths relative to the project.
    """
    if ctx.input is None:
        raise StageError("the dfin stage needs an input point cloud: pass input=data/in/<file>.laz")
    in_path = ctx.input
    if in_path.suffix.lower() not in LAS_SUFFIXES:
        raise StageError(f"3DFin reads .las/.laz files, got {in_path.name!r}")
    if not in_path.is_file():
        msg = f"input file not found: {in_path} (host: {ctx.host(in_path)}). Put it under data/in/."
        if not ctx.dry_run:
            raise StageError(msg)
        log.warning("[dry-run] %s", msg)

    stem = in_path.stem  # 3DFin names its outputs after Path(input).stem
    out_dir = ctx.stage_dir("dfin")
    params_path = out_dir / "params.ini"

    parser = build_params(ctx.project_dir, ini_overrides)
    changed = flatten_ini_overrides(parser, ini_overrides)
    log.info("params.ini: 3DFin template%s", f" with overrides {dict(('.'.join(k), v) for k, v in changed.items())}" if changed else "")

    # Mounts: the whole data/ dir at /data (rw), plus the input's folder read-only if it lies outside data/.
    mounts = [ctx.mount(ctx.data_dir, "/data")]
    try:
        in_container = ctx.container_path(in_path)
    except PathOutsideProjectError:
        mounts.append(ctx.mount(in_path.parent, EXTRA_INPUT_MOUNT, readonly=True))
        in_container = f"{EXTRA_INPUT_MOUNT}/{in_path.name}"
    out_container = ctx.container_path(out_dir)
    args = ["cli", in_container, out_container, f"{out_container}/params.ini"]
    args += [flag for flag, on in (("--normalize", normalize), ("--denoise", denoise), ("--export_txt", export_txt)) if on]

    require_image(image, dry_run=ctx.dry_run)

    if ctx.dry_run:
        log.info("[dry-run] would clear %s and write %s", out_dir, params_path)
        log.debug("params.ini would contain:\n%s", _render(parser))
    else:
        ctx.reset_stage_dir("dfin")  # stale 3DFin outputs would otherwise be mistaken for fresh ones
        write_params_ini(parser, params_path)

    result = run_container(
        image,
        args,
        mounts,
        network="none",  # 3DFin needs no network
        dry_run=ctx.dry_run,
        log_prefix="3dfin",
    )

    if ctx.dry_run:
        return {"dir": str(ctx.rel(out_dir)), "stem": stem, "dry_run": True}

    tail = "\n".join((result.stdout or "").splitlines()[-25:])
    found = verify_outputs(out_dir, stem, export_txt, tail)
    log.info("3DFin finished: %s", ", ".join(p.name for p in found.values()))
    return {
        "dir": str(ctx.rel(out_dir)),
        "stem": stem,
        "params": str(ctx.rel(params_path)),
        **{key: str(ctx.rel(path)) for key, path in found.items()},
    }


def _render(parser: configparser.ConfigParser) -> str:
    return "\n".join(f"  [{s}] " + ", ".join(f"{k}={v}" for k, v in parser[s].items()) for s in parser.sections())

"""Loader of the plot reference table (tree number, DBH, total height) for the ``compare`` stage.

The reference file is whatever the field crew / Excel produced, so the loader is deliberately tolerant:

* **CSV / TSV / TXT**: encoding auto-detected (UTF-16 with BOM, UTF-8 with or without BOM, else Windows-1250),
  delimiter sniffed (``;`` TAB ``,``; also Excel's ``sep=;`` first line), decimal comma or point, thousands
  separators and blank / ``NA`` cells handled.
* **XLSX / XLSM**: one sheet (name or 0-based index), numbers stored as numbers or as text.
* Columns come from the :class:`ReferenceSpec` (explicit names) or, when a column is not given, are auto-detected
  by case-, accent- and punctuation-insensitive aliases (``tree``, ``nr``, ``numer drzewa``, ``DBH [cm]``,
  ``d1.3``, ``pierśnica``, ``H``, ``wysokość``, ...). An ambiguous or unknown column is a :class:`StageError`
  that lists the columns found.
* Tree numbers may be ``12``, ``12.0``, ``"12"``, ``"T12"``; they are normalised to ``int``.
  Duplicate tree numbers are an error (the join is on the tree number only).

Result: :class:`Reference` whose ``table`` has one row per tree with the columns ``ordinal`` (int), ``dbh_m`` and
``height_m`` (float, metres, NaN where the cell is empty / not a number). A metric whose column does not exist in
the file and was not requested explicitly is skipped (``dbh_m`` / ``height_m`` is then NaN everywhere and the
metric is absent from :attr:`Reference.metrics`).

Units: ``m`` / ``cm`` / ``mm`` per column, from the explicit setting (parameter, YAML), else the column title
(``DBH [cm]``), else the default (cm for DBH, m for height). Only a unit that came from the DEFAULT is checked
(:func:`_check_default_unit`), because there nobody said what the numbers are: a DBH median below 1 (metres under a
plain ``DBH`` title?) or above 150 (millimetres?), or a height median above 120 (centimetres / millimetres?) is a
:class:`StageError` that names the setting to change. With an explicit unit (parameter, YAML, column title) the
same kind of mismatch only logs a warning, since the user has stated what the numbers are.

YAML column mapping (``compare.columns_file``), all keys optional, see ``configs/reference_columns.example.yaml``::

    sheet: 0              # xlsx: sheet name or 0-based index
    header_row: 0         # 0-based row with the column titles (counting non-empty rows only)
    csv: {sep: ";", decimal: ",", encoding: utf-8-sig}      # omit a key for auto-detection
    tree_id: {column: "Nr drzewa"}
    dbh:     {column: "DBH [cm]", unit: cm}                 # unit: m | cm | mm
    height:  {column: "H [m]", unit: m}
"""

from __future__ import annotations

import csv
import io
import logging
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lidar_bridge.stages._common import StageError

log = logging.getLogger("lidar_bridge.reference")

#: Multiply a value in the given unit by this factor to get metres.
UNIT_FACTORS: dict[str, float] = {"m": 1.0, "cm": 0.01, "mm": 0.001}
#: Unit used when neither the parameters, the YAML nor the column title say anything.
DEFAULT_UNITS = {"dbh": "cm", "height": "m"}
AUTO = "auto"

CSV_SUFFIXES = (".csv", ".tsv", ".txt")
XLSX_SUFFIXES = (".xlsx", ".xlsm")
DELIMITERS = (";", "\t", ",")  # tie-break priority when several delimiters fit equally well

#: Aliases, in the compact form of :func:`compact_key` (lower case, no accents, only [a-z0-9]).
ALIASES: dict[str, frozenset[str]] = {
    "tree_id": frozenset(
        "tree treeid treeno treenr treenum treenumber id nr no numer nrdrzewa numerdrzewa drzewo lp nrdrz tid".split()
    ),
    "dbh": frozenset(
        "dbh d13 d130 dbh13 dbhcm dbhmm dbhm d13cm d13mm piersnica piersnicacm srednicapiersnicowa "
        "srednicapiersnicy dpiersnicy dpiersnica d".split()
    ),
    "height": frozenset(
        "h height th totalheight treeheight totheight hgt ht htot htotal hcalk hcalkowita wysokosc "
        "wysokoscdrzewa wysokosccalkowita wys wysdrzewa hdrzewa".split()
    ),
}
ROLE_LABEL = {"tree_id": "tree number", "dbh": "DBH", "height": "tree height"}

_NA_TEXT = frozenset({"", "na", "n/a", "nan", "nd", "n.d.", "null", "none", "brak", "-", "--"})
_UNIT_TOKENS = frozenset(UNIT_FACTORS)
_ID_RE = re.compile(r"^[^\d\-+.,]*?(\d+)(?:[.,]0*)?$")
_UNIT_HINT_RES = (
    re.compile(r"[\[(]\s*(mm|cm|m)\s*[\])]", re.I),
    re.compile(r"(?:^|[\s_,;/-])(mm|cm|m)$", re.I),
)


# ---------------------------------------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceSpec:
    """Where the numbers are in the reference file. ``None`` / ``"auto"`` means detect it."""

    tree_id_col: str | None = None
    dbh_col: str | None = None
    height_col: str | None = None
    dbh_unit: str | None = None  # m | cm | mm ; None or "auto": from the column title, else cm (checked, see module doc)
    height_unit: str | None = None  # m | cm | mm ; None or "auto": from the column title, else m (checked)
    sheet: str | int | None = None  # xlsx only: name or 0-based index (None: first sheet)
    header_row: int = 0  # 0-based row that holds the column titles
    sep: str | None = None  # csv only: None = sniff
    decimal: str | None = None  # csv/xlsx text cells: "," | "." | None = per value
    encoding: str | None = None  # csv only: None = detect

    def __post_init__(self) -> None:
        for what in ("dbh_unit", "height_unit"):
            unit = getattr(self, what)
            if unit not in (None, AUTO, *UNIT_FACTORS):
                raise StageError(f"{what} must be one of {[*UNIT_FACTORS, AUTO]}, got {unit!r}")
        if self.decimal not in (None, ",", "."):
            raise StageError(f"decimal must be ',' or '.' (or omitted for auto-detection), got {self.decimal!r}")
        if self.header_row < 0:
            raise StageError(f"header_row must be >= 0, got {self.header_row}")


_YAML_KEYS = {"sheet", "header_row", "csv", "tree_id", "dbh", "height"}
_YAML_SECTION_KEYS = {"tree_id": {"column"}, "dbh": {"column", "unit"}, "height": {"column", "unit"}}
_YAML_CSV_KEYS = {"sep", "decimal", "encoding"}


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", AUTO))


def _opt_str(value: Any) -> str | None:
    return None if _blank(value) else str(value)


def spec_from_yaml(path: Path) -> ReferenceSpec:
    """Read a column-mapping YAML (see the module docstring). Unknown keys are errors (typos must not be silent)."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise StageError(f"cannot read the column mapping file {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise StageError(f"{path}: expected a mapping of keys at the top level, got {type(data).__name__}")
    unknown = set(data) - _YAML_KEYS
    if unknown:
        raise StageError(f"{path}: unknown key(s) {sorted(unknown)}; valid keys are {sorted(_YAML_KEYS)}")
    sections: dict[str, dict] = {}
    for name, allowed in (*_YAML_SECTION_KEYS.items(), ("csv", _YAML_CSV_KEYS)):
        section = data.get(name) or {}
        if not isinstance(section, dict):
            raise StageError(f"{path}: '{name}' must be a mapping with keys {sorted(allowed)}")
        bad = set(section) - allowed
        if bad:
            raise StageError(f"{path}: unknown key(s) {sorted(bad)} in '{name}'; valid keys are {sorted(allowed)}")
        sections[name] = section
    sheet = data.get("sheet")
    return ReferenceSpec(
        tree_id_col=_opt_str(sections["tree_id"].get("column")),
        dbh_col=_opt_str(sections["dbh"].get("column")),
        height_col=_opt_str(sections["height"].get("column")),
        dbh_unit=_opt_str(sections["dbh"].get("unit")),
        height_unit=_opt_str(sections["height"].get("unit")),
        sheet=None if _blank(sheet) else sheet,
        header_row=int(data.get("header_row") or 0),
        sep=None if _blank(sections["csv"].get("sep")) else str(sections["csv"]["sep"]).replace("\\t", "\t"),
        decimal=_opt_str(sections["csv"].get("decimal")),
        encoding=_opt_str(sections["csv"].get("encoding")),
    )


def merge_specs(base: ReferenceSpec, **overrides: Any) -> ReferenceSpec:
    """``base`` with every override that is not ``None`` / ``"auto"`` applied (explicit beats YAML beats auto)."""
    return replace(base, **{k: v for k, v in overrides.items() if not _blank(v)})


# ---------------------------------------------------------------------------------------------------------
# cell parsing
# ---------------------------------------------------------------------------------------------------------
def parse_number(value: Any, decimal: str | None = None) -> float:
    """Cell -> float; NaN for empty / ``NA`` / unparsable cells.

    Strings accept spaces and non-breaking spaces as thousands separators. ``decimal=None`` decides per value: if
    ``,`` and ``.`` both occur the last one is the decimal separator, a lone ``,`` is a decimal comma (``"12,5"`` ->
    12.5, the Polish convention) and repeated separators are thousands separators. ``decimal=","`` / ``"."`` take
    the other character as thousands separator.
    """
    if value is None or isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else math.nan
    text = str(value).strip().replace(" ", "").replace(" ", "").replace(" ", "").replace("−", "-")
    if text.lower() in _NA_TEXT:
        return math.nan
    if decimal == ",":
        text = text.replace(".", "").replace(",", ".")
    elif decimal == ".":
        text = text.replace(",", "")
    elif "," in text and "." in text:
        decimal_char = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal_char == "," else ","
        text = text.replace(thousands, "").replace(decimal_char, ".")
    elif text.count(",") > 1:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    try:
        number = float(text)
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def parse_tree_id(value: Any) -> int | None:
    """``12``, ``12.0``, ``"12"``, ``"T12"``, ``" tree 12 "`` -> 12; ``None`` for empty / non-integer cells."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if math.isfinite(value) and value == round(value) else None
    match = _ID_RE.match(str(value).strip())
    return int(match.group(1)) if match else None


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _is_na_cell(value: Any) -> bool:
    """Empty cell or an explicit "no value" marker (``NA``, ``-``, ...): missing on purpose, not worth a warning."""
    return _is_empty(value) or (isinstance(value, str) and value.strip().lower() in _NA_TEXT)


# ---------------------------------------------------------------------------------------------------------
# reading the raw table
# ---------------------------------------------------------------------------------------------------------
def decode_bytes(raw: bytes, encoding: str | None = None) -> tuple[str, str]:
    """Decode a text file; returns ``(text, encoding used)``. Auto: UTF-16 (BOM), UTF-8 (+BOM), else Windows-1250."""
    if encoding:
        try:
            return raw.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError) as exc:
            raise StageError(f"cannot decode the reference file as {encoding!r}: {exc}") from exc
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    try:
        return raw.decode("utf-8-sig"), "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1250", errors="replace"), "cp1250"


def sniff_delimiter(lines: Sequence[str]) -> str:
    """Pick the delimiter (``;`` TAB ``,``) under which the most lines split into the same number (>= 2) of cells.

    Robust against decimal commas: in ``"1;32,5;24,1"`` the comma gives 2-3 cells and an inconsistent count over
    the lines, the semicolon gives 3 on every line.
    """
    sample = [ln for ln in lines if ln.strip()][:30]
    if not sample:
        raise StageError("the reference file is empty")
    best: tuple[float, int, int] | None = None
    best_delim = ";"
    for priority, delim in enumerate(DELIMITERS):
        counts = [len(next(csv.reader([ln], delimiter=delim))) for ln in sample]
        modal = max(set(counts), key=lambda c: (counts.count(c), c))
        if modal < 2:
            continue
        score = (counts.count(modal) / len(counts), modal, -priority)
        if best is None or score > best:
            best, best_delim = score, delim
    if best is None:
        raise StageError(
            "cannot tell the delimiter of the reference file (no ';', TAB or ',' splits it into columns). "
            "Is it really a CSV? Set csv.sep in the columns_file."
        )
    return best_delim


def _read_csv_rows(path: Path, spec: ReferenceSpec) -> list[list[Any]]:
    text, used = decode_bytes(path.read_bytes(), spec.encoding)
    sep = spec.sep
    if m := re.match(r"[ \t]*sep=(.)[ \t]*\r?\n", text):  # Excel's explicit-delimiter first line
        sep = sep or m.group(1)
        text = text[m.end() :]
    if sep is None:
        sep = sniff_delimiter([ln for ln in text.splitlines() if ln.strip()][spec.header_row :])
    log.debug("reference %s: encoding %s, delimiter %r", path.name, used, sep)
    return list(csv.reader(io.StringIO(text, newline=""), delimiter=sep))


def _read_xlsx_rows(path: Path, spec: ReferenceSpec) -> list[list[Any]]:
    try:
        book = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:  # corrupt / password protected / not an xlsx
        raise StageError(f"cannot open the reference workbook {path.name}: {exc}") from exc
    with book:
        names = list(book.sheet_names)
        sheet = spec.sheet
        if sheet is None:
            sheet = 0
        if isinstance(sheet, str) and sheet.strip().lstrip("-").isdigit() and sheet not in names:
            sheet = int(sheet)
        if isinstance(sheet, int):
            if not -len(names) <= sheet < len(names):
                raise StageError(f"sheet index {sheet} is out of range; {path.name} has sheets {names}")
            sheet = names[sheet]
        elif sheet not in names:
            raise StageError(f"sheet {sheet!r} not found in {path.name}; available sheets: {names}")
        frame = pd.read_excel(book, sheet_name=sheet, header=None, dtype=object)
    return frame.astype(object).where(frame.notna(), None).values.tolist()


def read_raw_table(path: Path, spec: ReferenceSpec) -> pd.DataFrame:
    """The sheet / csv as a DataFrame of raw cells; column names are the stripped header cells."""
    suffix = path.suffix.lower()
    if suffix in XLSX_SUFFIXES:
        rows = _read_xlsx_rows(path, spec)
    elif suffix in CSV_SUFFIXES:
        rows = _read_csv_rows(path, spec)
    else:
        raise StageError(
            f"unsupported reference file type {path.suffix!r} ({path.name}); use {', '.join(CSV_SUFFIXES + XLSX_SUFFIXES)}"
            " (save .xls as .xlsx or .csv)"
        )
    rows = [r for r in rows if any(not _is_empty(c) for c in r)]  # drop blank lines, also above the header
    if spec.header_row >= len(rows):
        raise StageError(f"{path.name}: header_row {spec.header_row} but the file has only {len(rows)} non-empty rows")
    header_cells = rows[spec.header_row]
    width = max([len(header_cells), *(len(r) for r in rows[spec.header_row + 1 :])] or [0])
    header = [("" if _is_empty(c) else str(c).strip()) for c in header_cells] + [""] * (width - len(header_cells))
    body = [list(r) + [None] * (width - len(r)) for r in rows[spec.header_row + 1 :]]
    # unique internal labels (the visible titles are kept in .attrs) so duplicate / blank titles cannot collide
    frame = pd.DataFrame(body, columns=pd.RangeIndex(width), dtype=object)
    frame.attrs["titles"] = header
    return frame


# ---------------------------------------------------------------------------------------------------------
# column detection
# ---------------------------------------------------------------------------------------------------------
def _fold(text: str) -> str:
    """Lower case without accents (``ś`` -> ``s``, ``ł`` -> ``l``)."""
    text = text.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


def compact_key(title: str) -> str:
    """Title -> ``[a-z0-9]`` key: accents folded, bracketed unit parts and a trailing unit token dropped.

    ``"DBH [cm]"`` -> ``dbh``, ``"dbh_cm"`` -> ``dbh``, ``"d1.3"`` -> ``d13``, ``"Wysokość"`` -> ``wysokosc``,
    ``"Nr drzewa"`` -> ``nrdrzewa``.
    """
    text = re.sub(r"[\[(][^\])]*[\])]", " ", _fold(title))
    tokens = [t for t in re.split(r"[^a-z0-9]+", text) if t]
    if len(tokens) > 1 and tokens[-1] in _UNIT_TOKENS:
        tokens = tokens[:-1]
    return "".join(tokens)


def unit_hint(title: str) -> str | None:
    """Unit named in a column title (``"DBH [cm]"``, ``"h (m)"``, ``"dbh_mm"``), else ``None``."""
    for pattern in _UNIT_HINT_RES:
        if m := pattern.search(title.strip()):
            return m.group(1).lower()
    return None


def _describe(titles: Iterable[str]) -> str:
    return ", ".join(repr(t) if t else "<blank>" for t in titles)


def find_column(role: str, titles: Sequence[str], requested: str | None, *, taken: dict[int, str]) -> int | None:
    """Index of the column for ``role``; ``None`` if absent and not requested. Raises on unknown / ambiguous."""
    named = [(i, t) for i, t in enumerate(titles) if t]
    if requested is not None:
        wanted = requested.strip()
        hits = [i for i, t in named if t == wanted] or [i for i, t in named if t.lower() == wanted.lower()]
        hits = hits or [i for i, t in named if compact_key(t) == compact_key(wanted) != ""]
        if not hits:
            raise StageError(
                f"{ROLE_LABEL[role]} column {requested!r} not found in the reference table. "
                f"Columns found: {_describe(t for _, t in named)}"
            )
        if len(hits) > 1:
            raise StageError(
                f"{ROLE_LABEL[role]} column {requested!r} matches several columns: {_describe(titles[i] for i in hits)}"
            )
        found = hits[0]
    else:
        hits = [i for i, t in named if compact_key(t) in ALIASES[role]]
        if not hits:
            return None
        if len(hits) > 1:
            raise StageError(
                f"cannot tell which column holds the {ROLE_LABEL[role]}: candidates {_describe(titles[i] for i in hits)}. "
                f"Name it explicitly (compare.{role}_col=...). "
                f"Columns found: {_describe(t for _, t in named)}"
            )
        found = hits[0]
    if found in taken:
        raise StageError(
            f"column {titles[found]!r} would be used for both {ROLE_LABEL[taken[found]]} and {ROLE_LABEL[role]}. "
            f"Columns found: {_describe(t for _, t in named)}"
        )
    taken[found] = role
    return found


def resolve_unit(role: str, requested: str | None, title: str | None) -> tuple[str, str]:
    """``(unit, how it was decided)`` from the explicit unit, else the column title, else the default."""
    hint = unit_hint(title) if title else None
    if requested not in (None, AUTO):
        if hint and hint != requested:
            log.warning(
                "reference %s: column %r looks like it is in %s but %s_unit=%s was requested; using %s",
                role, title, hint, role, requested, requested,
            )
        return str(requested), "given"
    if hint:
        return hint, f"from column title {title!r}"
    return DEFAULT_UNITS[role], "default"


# ---------------------------------------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """The normalised reference table.

    ``table``: ``ordinal`` (int, unique), ``dbh_m``, ``height_m`` (float, m; NaN = empty cell); sorted by ordinal.
    ``metrics``: the metrics that have a column (subset of ``("dbh", "height")``).
    ``columns`` / ``units``: the source column title and the unit used per role, for logging.
    """

    table: pd.DataFrame
    metrics: tuple[str, ...]
    columns: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    path: Path | None = None


def load_reference(path: str | Path, spec: ReferenceSpec | None = None) -> Reference:
    """Read and normalise the reference table. See the module docstring for the rules; raises :class:`StageError`."""
    path = Path(path)
    spec = spec or ReferenceSpec()
    if not path.is_file():
        raise StageError(f"reference file not found: {path}")
    frame = read_raw_table(path, spec)
    titles: list[str] = frame.attrs["titles"]
    if frame.empty:
        raise StageError(f"{path.name} has a header row but no data rows")

    taken: dict[int, str] = {}
    idx_id = find_column("tree_id", titles, spec.tree_id_col, taken=taken)
    if idx_id is None:
        raise StageError(
            f"no tree number column found in {path.name}. Name it explicitly (compare.tree_id_col=... or tree_id.column "
            f"in the columns_file); auto-detected titles are {sorted(ALIASES['tree_id'])}. "
            f"Columns found: {_describe(t for t in titles if t)}"
        )
    idx = {"dbh": find_column("dbh", titles, spec.dbh_col, taken=taken),
           "height": find_column("height", titles, spec.height_col, taken=taken)}
    metrics = tuple(m for m in ("dbh", "height") if idx[m] is not None)
    for missing in (m for m in ("dbh", "height") if idx[m] is None):
        log.warning(
            "reference %s: no %s column found (columns: %s); the %s comparison is skipped. "
            "Set compare.%s_col=... if it exists.",
            path.name, ROLE_LABEL[missing], _describe(t for t in titles if t), missing, missing,
        )
    if not metrics:
        raise StageError(
            f"{path.name} has neither a DBH nor a height column, nothing to compare. "
            f"Columns found: {_describe(t for t in titles if t)}"
        )

    # tree numbers
    ids = frame[idx_id].map(parse_tree_id)
    blank = frame[idx_id].map(_is_empty)
    bad = frame.loc[ids.isna() & ~blank, idx_id]
    if len(bad) and len(bad) == int((~blank).sum()):
        raise StageError(
            f"column {titles[idx_id]!r} holds no usable tree numbers (first values: {[str(v) for v in bad.head(5)]}); "
            "expected 12, 12.0 or T12"
        )
    if len(bad):
        log.warning(
            "reference %s: ignoring %d row(s) whose tree number is not an integer: %s",
            path.name, len(bad), [str(v) for v in bad.head(10)],
        )
    keep = ids.notna()
    ordinals = ids[keep].astype(int)
    duplicated = sorted(set(ordinals[ordinals.duplicated(keep=False)]))
    if duplicated:
        raise StageError(
            f"duplicate tree numbers in the reference ({path.name}): {duplicated[:20]}"
            f"{' ...' if len(duplicated) > 20 else ''}. The comparison joins on the tree number only, "
            "so every number may appear once."
        )

    table = pd.DataFrame({"ordinal": ordinals.to_numpy(dtype=int)})
    columns = {"tree_id": titles[idx_id]}
    units: dict[str, str] = {}
    for role, out in (("dbh", "dbh_m"), ("height", "height_m")):
        if idx[role] is None:
            table[out] = np.nan
            continue
        title = titles[idx[role]]
        unit, how = resolve_unit(role, getattr(spec, f"{role}_unit"), title)
        cells = frame.loc[keep, idx[role]]
        values = cells.map(lambda v: parse_number(v, spec.decimal)).to_numpy(dtype=float)
        unparsable = int((np.isnan(values) & ~cells.map(_is_na_cell).to_numpy(dtype=bool)).sum())
        if unparsable:
            log.warning("reference %s: %d non-numeric cell(s) in column %r treated as missing", path.name, unparsable, title)
        table[out] = values * UNIT_FACTORS[unit]
        columns[role], units[role] = title, unit
        log.info("reference %s: %s <- column %r (%s: %s)", path.name, ROLE_LABEL[role], title, unit, how)
        if how == "default":  # nobody said which unit: a wrong guess must stop the stage, not produce garbage statistics
            _check_default_unit(role, unit, values, title)
        _warn_if_implausible(role, unit, table[out], title)
    log.info("reference %s: tree number <- column %r; %d trees", path.name, titles[idx_id], len(table))
    table = table.sort_values("ordinal", ignore_index=True)
    return Reference(table=table, metrics=metrics, columns=columns, units=units, path=path)


#: Medians (of the positive values, in the DEFAULT unit: cm for DBH, m for height) beyond which the unit was almost
#: certainly not the default: DBH below ``lo`` cm (metres?) or above ``hi`` cm (millimetres?), height above ``hi`` m
#: (centimetres / millimetres?).
_DEFAULT_UNIT_MEDIAN_LIMITS = {"dbh": (1.0, 150.0), "height": (None, 120.0)}


def _check_default_unit(role: str, unit: str, values: np.ndarray, title: str) -> None:
    """Raise when the unit came from the default and the median of the values makes it implausible.

    ``values`` are the parsed cells in ``unit`` (not yet converted). An explicit unit (parameter, YAML, column title)
    never gets here: it only produces the :func:`_warn_if_implausible` warning.
    """
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return
    median = float(np.median(positive))
    lo, hi = _DEFAULT_UNIT_MEDIAN_LIMITS[role]
    label = ROLE_LABEL[role]
    head = f"{label} column {title!r} has median {median:.4g}; assumed {unit} by default, which would mean "

    def fix(suggested: str) -> str:
        return (
            f"set compare.{role}_unit={suggested} (or {role}: {{unit: {suggested}}} in the columns_file, or put the unit "
            f"in the column title, e.g. '{title} [{suggested}]'). If the values really are {unit}, say so with "
            f"compare.{role}_unit={unit}."
        )

    if lo is not None and median < lo:  # DBH: cm -> metres
        raise StageError(f"{head}{median * 10:.2g} mm trees. Probably metres: {fix('m')}")
    if median > hi:
        if role == "dbh":
            raise StageError(f"{head}{median / 100:.3g} m trees. Probably millimetres: {fix('mm')}")
        raise StageError(f"{head}{median:.4g} m tall trees. Probably centimetres (or millimetres): {fix('cm')}")


_PLAUSIBLE_MEDIAN_M = {"dbh": (0.02, 1.5), "height": (1.0, 70.0)}


def _warn_if_implausible(role: str, unit: str, metres: pd.Series, title: str) -> None:
    """Warn when the median of a converted column is outside a plausible forest range (a wrong unit is typical)."""
    valid = metres[metres > 0]
    if valid.empty:
        return
    median = float(valid.median())
    lo, hi = _PLAUSIBLE_MEDIAN_M[role]
    if not lo <= median <= hi:
        log.warning(
            "reference column %r: median %s = %.3f m after reading it as %s looks implausible (expected %.2f-%.0f m). "
            "Check %s_unit.",
            title, ROLE_LABEL[role], median, unit, lo, hi, role,
        )

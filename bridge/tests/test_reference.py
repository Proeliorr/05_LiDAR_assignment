from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidar_bridge import reference as ref
from lidar_bridge.reference import ReferenceSpec, load_reference, merge_specs, spec_from_yaml
from lidar_bridge.stages._common import StageError

POLISH_CSV = (
    "Nr drzewa;DBH [cm];Wysokość [m];Uwagi\n"
    "1;32,5;24,1;ok\n"
    "2;41,0;27,35;\n"
    "T3;28,25;;brak wysokości\n"
)


def write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def write_xlsx(path: Path, rows: list[list], sheet: str = "Sheet1", extra_sheets: dict[str, list[list]] | None = None) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    for name, sheet_rows in (extra_sheets or {}).items():
        other = wb.create_sheet(name)
        for row in sheet_rows:
            other.append(row)
    wb.save(path)
    return path


# -- cell parsing ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, decimal, expected",
    [
        ("12.5", None, 12.5),
        ("12,5", None, 12.5),
        (" 12,5 ", None, 12.5),
        ("1 234,5", None, 1234.5),
        ("1 234,5", None, 1234.5),
        ("1.234,5", None, 1234.5),
        ("1,234.5", None, 1234.5),
        ("1,234,567", None, 1234567.0),
        ("1.234.567", None, 1234567.0),
        ("1.234", ",", 1234.0),
        ("1,234", ".", 1234.0),
        ("-3,5", None, -3.5),
        (12, None, 12.0),
        (12.5, None, 12.5),
        (np.float64(7.25), None, 7.25),
    ],
)
def test_parse_number(value, decimal, expected):
    assert ref.parse_number(value, decimal) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "  ", "NA", "n/a", "-", "brak", "abc", "12 cm", float("nan"), float("inf"), True])
def test_parse_number_missing(value):
    assert math.isnan(ref.parse_number(value))


@pytest.mark.parametrize(
    "value, expected",
    [
        (12, 12), (12.0, 12), ("12", 12), ("12.0", 12), ("12,0", 12), ("T12", 12), ("t12", 12), (" T 12 ", 12),
        ("Tree_7", 7), ("T01", 1), (np.int64(5), 5), (np.float64(5.0), 5), (0, 0),
    ],
)  # fmt: skip
def test_parse_tree_id(value, expected):
    assert ref.parse_tree_id(value) == expected


@pytest.mark.parametrize("value", [None, "", "  ", 12.5, "12.5", "12a", "abc", "-3", float("nan"), True, "T"])
def test_parse_tree_id_rejects(value):
    assert ref.parse_tree_id(value) is None


@pytest.mark.parametrize(
    "title, key, unit",
    [
        ("DBH [cm]", "dbh", "cm"),
        ("dbh_cm", "dbh", "cm"),
        ("DBH (mm)", "dbh", "mm"),
        ("d1.3", "d13", None),
        ("D 1,3 m", "d13", "m"),
        ("Wysokość [m]", "wysokosc", "m"),
        ("Pierśnica", "piersnica", None),
        ("Nr drzewa", "nrdrzewa", None),
        ("H", "h", None),
        ("h, m", "h", "m"),
        ("Łączna wysokość", "lacznawysokosc", None),
    ],
)
def test_compact_key_and_unit_hint(title, key, unit):
    assert ref.compact_key(title) == key
    assert ref.unit_hint(title) == unit


# -- csv ------------------------------------------------------------------------------------------------------
def test_semicolon_decimal_comma_bom_autodetected(tmp_path):
    path = write_text(tmp_path / "ref.csv", POLISH_CSV, "utf-8-sig")
    r = load_reference(path)
    assert r.metrics == ("dbh", "height")
    assert r.columns == {"tree_id": "Nr drzewa", "dbh": "DBH [cm]", "height": "Wysokość [m]"}
    assert r.units == {"dbh": "cm", "height": "m"}
    t = r.table
    assert list(t["ordinal"]) == [1, 2, 3] and t["ordinal"].dtype.kind == "i"
    assert t["dbh_m"].tolist() == pytest.approx([0.325, 0.41, 0.2825])
    assert t["height_m"].iloc[:2].tolist() == pytest.approx([24.1, 27.35])
    assert math.isnan(t["height_m"].iloc[2])  # empty cell


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1250", "utf-16"])
def test_encodings(tmp_path, encoding):
    r = load_reference(write_text(tmp_path / f"{encoding}.csv", POLISH_CSV, encoding))
    assert r.columns["height"] == "Wysokość [m]"  # the Polish letter survived
    assert r.table["dbh_m"].iloc[0] == pytest.approx(0.325)


def test_tab_separated_utf16_excel_unicode_text(tmp_path):
    text = "tree\tdbh\theight\r\n1\t30,5\t20,5\r\n2\t40,5\t25\r\n"
    r = load_reference(write_text(tmp_path / "ref.txt", text, "utf-16"))
    assert r.table["dbh_m"].tolist() == pytest.approx([0.305, 0.405])
    assert r.table["height_m"].tolist() == pytest.approx([20.5, 25.0])


def test_comma_delimited_point_decimal(tmp_path):
    text = "tree_id,dbh_cm,height\n1,30.5,20.5\n2,40.25,25.0\n"
    r = load_reference(write_text(tmp_path / "ref.csv", text))
    assert r.table["dbh_m"].tolist() == pytest.approx([0.305, 0.4025])
    assert r.units == {"dbh": "cm", "height": "m"}


def test_sniffer_does_not_mistake_decimal_comma_for_delimiter(tmp_path):
    text = "id;d13;h\n1;3,5;2,1\n2;4,5;3,1\n"  # commas split every data line in three, never the header
    assert ref.sniff_delimiter(text.splitlines()) == ";"
    r = load_reference(write_text(tmp_path / "ref.csv", text), ReferenceSpec(dbh_unit="m"))
    assert r.table["dbh_m"].tolist() == pytest.approx([3.5, 4.5])


def test_excel_sep_line_and_blank_lines_and_quotes(tmp_path):
    text = 'sep=|\n\n"Nr"|"DBH"|"H"\n\n1|"30,5"|20\n2|41|"25,5"\n\n'
    r = load_reference(write_text(tmp_path / "ref.csv", text))
    assert r.table["dbh_m"].tolist() == pytest.approx([0.305, 0.41])
    assert r.table["height_m"].tolist() == pytest.approx([20.0, 25.5])


def test_header_row_skips_title_lines(tmp_path):
    text = "Plot 3 measured 2025-05-01;;\nsomething;;\nNr;DBH;H\n1;30;20\n"
    r = load_reference(write_text(tmp_path / "ref.csv", text), ReferenceSpec(header_row=2))
    assert list(r.table["ordinal"]) == [1]
    assert r.table["dbh_m"].iloc[0] == pytest.approx(0.30)


def test_short_rows_and_na_cells(tmp_path):
    text = "Nr;DBH;H\n1;30;NA\n2;-;\n3\n4;n/a;18\n"
    r = load_reference(write_text(tmp_path / "ref.csv", text))
    assert list(r.table["ordinal"]) == [1, 2, 3, 4]
    assert r.table["dbh_m"].isna().tolist() == [False, True, True, True]
    assert r.table["height_m"].isna().tolist() == [True, True, True, False]


def test_non_numeric_cell_is_a_warning(tmp_path, caplog):
    text = "Nr;DBH;H\n1;30;20\n2;abc;21\n"
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.reference"):
        r = load_reference(write_text(tmp_path / "ref.csv", text))
    assert math.isnan(r.table["dbh_m"].iloc[1])
    assert "1 non-numeric cell(s)" in caplog.text


# -- units ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dbh_unit, height_unit, dbh_in, height_in",
    [("m", "m", 0.31, 20.0), ("cm", "m", 31.0, 20.0), ("mm", "cm", 310.0, 2000.0), ("cm", "mm", 31.0, 20000.0)],
)
def test_unit_conversion(tmp_path, dbh_unit, height_unit, dbh_in, height_in):
    path = write_text(tmp_path / "ref.csv", f"Nr;DBH;H\n1;{dbh_in};{height_in}\n".replace(".", ","))
    r = load_reference(path, ReferenceSpec(dbh_unit=dbh_unit, height_unit=height_unit))
    assert r.table["dbh_m"].iloc[0] == pytest.approx(0.31)
    assert r.table["height_m"].iloc[0] == pytest.approx(20.0)


def test_unit_from_title_then_default_and_explicit_wins(tmp_path, caplog):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH [mm];H [cm]\n1;310;2000\n")
    r = load_reference(path)
    assert r.units == {"dbh": "mm", "height": "cm"}
    assert r.table["dbh_m"].iloc[0] == pytest.approx(0.31)
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.reference"):
        r = load_reference(path, ReferenceSpec(dbh_unit="cm", height_unit="auto"))
    assert r.units == {"dbh": "cm", "height": "cm"}
    assert "looks like it is in mm" in caplog.text


def test_implausible_explicit_unit_only_warns(tmp_path, caplog):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;310;20\n2;280;21\n")  # mm read as cm -> 3 m
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.reference"):
        r = load_reference(path, ReferenceSpec(dbh_unit="cm"))  # the user said cm: their call
    assert r.table["dbh_m"].tolist() == pytest.approx([3.10, 2.80])
    assert "implausible" in caplog.text and "dbh_unit" in caplog.text


# -- unit plausibility of the DEFAULT unit (nobody said what the numbers are) -----------------------------------
def test_dbh_in_metres_under_a_plain_title_is_an_error(tmp_path):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;0,23;20\n2;0,31;21\n3;0,41;22\n")
    with pytest.raises(StageError) as exc:
        load_reference(path)
    message = str(exc.value)
    assert "DBH column 'DBH' has median 0.31; assumed cm by default, which would mean 3.1 mm trees" in message
    assert "Probably metres" in message and "compare.dbh_unit=m" in message
    assert "dbh: {unit: m}" in message and "'DBH [m]'" in message  # the three ways out
    assert "say so with compare.dbh_unit=cm" in message


def test_dbh_in_millimetres_under_a_plain_title_is_an_error(tmp_path):
    path = write_text(tmp_path / "ref.csv", "Nr;Pierśnica;H\n1;310;20\n2;280;21\n3;405;22\n")
    with pytest.raises(StageError) as exc:
        load_reference(path)
    message = str(exc.value)
    assert "DBH column 'Pierśnica' has median 310; assumed cm by default, which would mean 3.1 m trees" in message
    assert "Probably millimetres" in message and "compare.dbh_unit=mm" in message and "'Pierśnica [mm]'" in message
    assert load_reference(path, ReferenceSpec(dbh_unit="mm")).table["dbh_m"].tolist() == pytest.approx([0.31, 0.28, 0.405])


def test_height_in_centimetres_under_a_plain_title_is_an_error(tmp_path):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;Wysokość\n1;30;2000\n2;31;2150\n3;29;1900\n")
    with pytest.raises(StageError) as exc:
        load_reference(path)
    message = str(exc.value)
    assert "tree height column 'Wysokość' has median 2000; assumed m by default, which would mean 2000 m tall trees" in message
    assert "Probably centimetres (or millimetres)" in message and "compare.height_unit=cm" in message


@pytest.mark.parametrize(
    "spec, text",
    [
        (ReferenceSpec(dbh_unit="m"), "Nr;DBH;H\n1;0,23;20\n2;0,31;21\n"),  # parameter / YAML (they build the same spec)
        (ReferenceSpec(), "Nr;DBH [m];H\n1;0,23;20\n2;0,31;21\n"),  # unit in the title
        (ReferenceSpec(), "Nr;DBH (mm);H\n1;230;20\n2;310;21\n"),
        (ReferenceSpec(height_unit="cm"), "Nr;DBH;H\n1;23;2000\n2;31;2100\n"),
        (ReferenceSpec(), "Nr;DBH;H [cm]\n1;23;2000\n2;31;2100\n"),
    ],
)
def test_explicit_unit_is_never_an_error(tmp_path, spec, text):
    r = load_reference(write_text(tmp_path / "ref.csv", text), spec)
    assert r.table["dbh_m"].tolist() == pytest.approx([0.23, 0.31])
    assert r.table["height_m"].tolist() == pytest.approx([20.0, 21.0])


def test_unit_from_yaml_counts_as_explicit(tmp_path):
    spec = spec_from_yaml(write_text(tmp_path / "cols.yaml", "dbh:\n  unit: m\nheight:\n  unit: cm\n"))
    r = load_reference(write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;0,23;2000\n2;0,31;2100\n"), spec)
    assert r.table["dbh_m"].tolist() == pytest.approx([0.23, 0.31])
    assert r.table["height_m"].tolist() == pytest.approx([20.0, 21.0])


def test_explicit_unit_that_looks_wrong_only_warns(tmp_path, caplog):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH [cm];H\n1;0,23;20\n2;0,31;21\n")  # title says cm: believed
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.reference"):
        r = load_reference(path)
    assert r.table["dbh_m"].tolist() == pytest.approx([0.0023, 0.0031])
    assert "implausible" in caplog.text


@pytest.mark.parametrize(
    "role_text, fails",
    [
        ("Nr;DBH;H\n1;1,0;20\n", False),  # median exactly 1.0 cm: not below the limit
        ("Nr;DBH;H\n1;0,99;20\n", True),
        ("Nr;DBH;H\n1;150;20\n", False),  # exactly 150: not above
        ("Nr;DBH;H\n1;150,5;20\n", True),
        ("Nr;DBH;H\n1;30;120\n", False),
        ("Nr;DBH;H\n1;30;120,5\n", True),
    ],
)
def test_default_unit_limits(tmp_path, role_text, fails):
    path = write_text(tmp_path / "ref.csv", role_text)
    if fails:
        with pytest.raises(StageError, match="assumed (cm|m) by default"):
            load_reference(path)
    else:
        load_reference(path)


def test_median_of_the_positive_values_decides(tmp_path):
    # zeros, negatives and blanks are ignored; the median of 0.3, 0.4, 0.5 is 0.4 (metres)
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;0;20\n2;-1;20\n3;NA;20\n4;0,3;20\n5;0,4;20\n6;0,5;20\n")
    with pytest.raises(StageError, match="median 0.4;"):
        load_reference(path)
    # a few legitimate big trees do not trip the check (median 40 cm)
    load_reference(write_text(tmp_path / "ok.csv", "Nr;DBH;H\n1;0,5;20\n2;40;20\n3;45;20\n4;38;20\n"))


def test_column_without_any_value_does_not_trip_the_check(tmp_path):
    load_reference(write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;;20\n2;;21\n"))


def test_invalid_unit_is_a_stage_error():
    with pytest.raises(StageError, match="dbh_unit"):
        ReferenceSpec(dbh_unit="inch")


# -- xlsx -----------------------------------------------------------------------------------------------------
def test_xlsx_numbers_text_decimal_and_labels(tmp_path):
    rows = [["Nr drzewa", "DBH [cm]", "Wysokość [m]"], ["T1", 32.5, 24.1], ["T2", "41,0", "27,35"], [3.0, 28.25, None]]
    r = load_reference(write_xlsx(tmp_path / "ref.xlsx", rows))
    assert list(r.table["ordinal"]) == [1, 2, 3]
    assert r.table["dbh_m"].tolist() == pytest.approx([0.325, 0.41, 0.2825])
    assert r.table["height_m"].iloc[:2].tolist() == pytest.approx([24.1, 27.35])
    assert math.isnan(r.table["height_m"].iloc[2])


def test_xlsx_sheet_by_name_index_and_digit_string(tmp_path):
    first = [["Nr", "DBH"], [1, 10], [2, 20]]
    second = [["Nr", "DBH"], [1, 30], [2, 40]]
    path = write_xlsx(tmp_path / "ref.xlsx", first, sheet="A", extra_sheets={"Pomiar": second})
    assert load_reference(path).table["dbh_m"].tolist() == pytest.approx([0.10, 0.20])
    assert load_reference(path, ReferenceSpec(sheet=1)).table["dbh_m"].tolist() == pytest.approx([0.30, 0.40])
    assert load_reference(path, ReferenceSpec(sheet="1")).table["dbh_m"].tolist() == pytest.approx([0.30, 0.40])
    assert load_reference(path, ReferenceSpec(sheet="Pomiar")).table["dbh_m"].tolist() == pytest.approx([0.30, 0.40])
    with pytest.raises(StageError, match=r"sheet 'Nope' not found.*\['A', 'Pomiar'\]"):
        load_reference(path, ReferenceSpec(sheet="Nope"))
    with pytest.raises(StageError, match="out of range"):
        load_reference(path, ReferenceSpec(sheet=5))


def test_xlsx_header_row_and_blank_rows(tmp_path):
    rows = [["Plot 7"], [None], ["Nr", "DBH [cm]", "H"], [1, 30, 20], [None, None, None], [2, 31, 21]]
    r = load_reference(write_xlsx(tmp_path / "ref.xlsx", rows), ReferenceSpec(header_row=1))
    assert list(r.table["ordinal"]) == [1, 2]


def test_corrupt_xlsx(tmp_path):
    path = tmp_path / "bad.xlsx"
    path.write_bytes(b"this is not a zip")
    with pytest.raises(StageError, match="cannot open the reference workbook"):
        load_reference(path)


# -- column detection -----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "header",
    [
        ["tree", "dbh", "height"],
        ["Tree_ID", "DBH_cm", "Height"],
        ["ID", "D1.3", "TH"],
        ["NR", "d13", "H"],
        ["Numer drzewa", "Pierśnica [cm]", "Wysokość [m]"],
        ["No", "DBH (cm)", "wysokosc"],
        ["Lp.", "Pierśnica", "Wys."],
    ],
)
def test_alias_detection(tmp_path, header):
    text = ";".join(header) + "\n1;30;20\n2;31;21\n"
    r = load_reference(write_text(tmp_path / "ref.csv", text, "utf-8-sig"))
    assert list(r.table["ordinal"]) == [1, 2]
    assert r.table["dbh_m"].tolist() == pytest.approx([0.30, 0.31])
    assert r.table["height_m"].tolist() == pytest.approx([20.0, 21.0])


def test_explicit_columns_win_over_aliases_and_match_loosely(tmp_path):
    text = "tree;dbh;Moje nr;Średnica;h\n9;1;1;30;20\n9;2;2;31;21\n"  # 'tree' and 'dbh' would be wrong/duplicate
    path = write_text(tmp_path / "ref.csv", text)
    r = load_reference(path, ReferenceSpec(tree_id_col="moje NR", dbh_col="Średnica", height_col="h"))
    assert list(r.table["ordinal"]) == [1, 2]
    assert r.table["dbh_m"].tolist() == pytest.approx([0.30, 0.31])


def test_ambiguous_column_lists_candidates_and_columns(tmp_path):
    text = "tree;dbh;d13;height;note\n1;30;30;20;x\n"
    with pytest.raises(StageError) as exc:
        load_reference(write_text(tmp_path / "ref.csv", text))
    message = str(exc.value)
    assert "which column holds the DBH" in message
    assert "'dbh'" in message and "'d13'" in message and "'note'" in message  # candidates and all columns
    assert "compare.dbh_col" in message


def test_explicit_column_not_found_lists_columns(tmp_path):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;30;20\n")
    with pytest.raises(StageError, match=r"DBH column 'Pierśnica' not found.*Columns found: 'Nr', 'DBH', 'H'"):
        load_reference(path, ReferenceSpec(dbh_col="Pierśnica"))


def test_no_tree_id_column_lists_columns(tmp_path):
    path = write_text(tmp_path / "ref.csv", "foo;DBH;H\n1;30;20\n")
    with pytest.raises(StageError, match=r"no tree number column found.*Columns found: 'foo', 'DBH', 'H'"):
        load_reference(path)


def test_missing_metric_column_is_skipped_with_warning(tmp_path, caplog):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;Notes\n1;30;a\n2;31;b\n")
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.reference"):
        r = load_reference(path)
    assert r.metrics == ("dbh",)
    assert r.table["height_m"].isna().all()
    assert "no tree height column found" in caplog.text


def test_no_metric_column_at_all(tmp_path):
    with pytest.raises(StageError, match="neither a DBH nor a height column"):
        load_reference(write_text(tmp_path / "ref.csv", "Nr;Notes\n1;a\n"))


def test_same_column_for_two_roles(tmp_path):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;30;20\n")
    with pytest.raises(StageError, match="both DBH and tree height"):
        load_reference(path, ReferenceSpec(dbh_col="DBH", height_col="DBH"))


# -- tree numbers ---------------------------------------------------------------------------------------------
def test_duplicate_tree_numbers(tmp_path):
    path = write_text(tmp_path / "ref.csv", "Nr;DBH;H\n1;30;20\nT2;31;21\n2;32;22\n1;33;23\n")
    with pytest.raises(StageError, match=r"duplicate tree numbers.*\[1, 2\]"):
        load_reference(path)


def test_tree_number_normalisation_mixed(tmp_path):
    rows = [["Nr", "DBH", "H"], ["T12", 30, 20], ["13", 31, 21], [14.0, 32, 22], [" T 15 ", 33, 23]]
    r = load_reference(write_xlsx(tmp_path / "ref.xlsx", rows))
    assert list(r.table["ordinal"]) == [12, 13, 14, 15]


def test_unusable_tree_numbers_dropped_with_warning_but_all_bad_is_an_error(tmp_path, caplog):
    text = "Nr;DBH;H\n1;30;20\nSuma;;\n2a;31;21\n;;\n2;32;22\n"
    with caplog.at_level(logging.WARNING, logger="lidar_bridge.reference"):
        r = load_reference(write_text(tmp_path / "ref.csv", text))
    assert list(r.table["ordinal"]) == [1, 2]
    assert "ignoring 2 row(s)" in caplog.text and "Suma" in caplog.text
    with pytest.raises(StageError, match="no usable tree numbers"):
        load_reference(write_text(tmp_path / "bad.csv", "Nr;DBH;H\nabc;30;20\ndef;31;21\n"))


def test_header_only_and_empty_files(tmp_path):
    with pytest.raises(StageError, match="no data rows"):
        load_reference(write_text(tmp_path / "h.csv", "Nr;DBH;H\n"))
    with pytest.raises(StageError, match="empty|non-empty"):
        load_reference(write_text(tmp_path / "e.csv", ""))


def test_unsupported_and_missing_file(tmp_path):
    with pytest.raises(StageError, match="unsupported reference file type"):
        load_reference(write_text(tmp_path / "ref.xls", "x"))
    with pytest.raises(StageError, match="not found"):
        load_reference(tmp_path / "nope.csv")


def test_table_is_sorted_by_tree_number(tmp_path):
    r = load_reference(write_text(tmp_path / "ref.csv", "Nr;DBH;H\n3;30;20\n1;31;21\n2;32;22\n"))
    assert list(r.table["ordinal"]) == [1, 2, 3]
    assert r.table["dbh_m"].tolist() == pytest.approx([0.31, 0.32, 0.30])


# -- YAML -----------------------------------------------------------------------------------------------------
YAML_TEXT = """\
sheet: Pomiar
header_row: 1
csv:
  sep: ";"
  decimal: ","
  encoding: cp1250
tree_id:
  column: "Nr drzewa"
dbh:
  column: "Pierśnica"
  unit: mm
height:
  column: "H [m]"
  unit: auto
"""


def test_spec_from_yaml(tmp_path):
    path = write_text(tmp_path / "cols.yaml", YAML_TEXT)
    spec = spec_from_yaml(path)
    assert spec == ReferenceSpec(
        tree_id_col="Nr drzewa", dbh_col="Pierśnica", height_col="H [m]", dbh_unit="mm", height_unit=None,
        sheet="Pomiar", header_row=1, sep=";", decimal=",", encoding="cp1250",
    )  # fmt: skip


def test_spec_from_yaml_empty_and_errors(tmp_path):
    assert spec_from_yaml(write_text(tmp_path / "e.yaml", "# nothing\n")) == ReferenceSpec()
    with pytest.raises(StageError, match=r"unknown key\(s\) \['colums'\]"):
        spec_from_yaml(write_text(tmp_path / "a.yaml", "colums: {}\n"))
    with pytest.raises(StageError, match=r"unknown key\(s\) \['colum'\] in 'dbh'"):
        spec_from_yaml(write_text(tmp_path / "b.yaml", "dbh:\n  colum: x\n"))
    with pytest.raises(StageError, match="must be a mapping"):
        spec_from_yaml(write_text(tmp_path / "c.yaml", "dbh: 5\n"))
    with pytest.raises(StageError, match="dbh_unit"):
        spec_from_yaml(write_text(tmp_path / "d.yaml", "dbh:\n  unit: furlong\n"))
    with pytest.raises(StageError, match="cannot read"):
        spec_from_yaml(write_text(tmp_path / "f.yaml", "a: [unclosed\n"))


def test_merge_specs_explicit_beats_yaml_and_auto_keeps_yaml():
    base = ReferenceSpec(dbh_col="from yaml", dbh_unit="mm", sheet="Pomiar")
    merged = merge_specs(base, dbh_col="explicit", dbh_unit="auto", height_col=None, sheet=None)
    assert merged.dbh_col == "explicit" and merged.dbh_unit == "mm" and merged.sheet == "Pomiar"
    assert merged.height_col is None


def test_yaml_drives_a_cp1250_file(tmp_path):
    ref_path = write_text(
        tmp_path / "ref.csv", "Pomiar z 2025\nNr drzewa;Pierśnica;H [m]\n1;310;20,5\n2;405;25,0\n", "cp1250"
    )
    spec = spec_from_yaml(write_text(tmp_path / "cols.yaml", YAML_TEXT))
    r = load_reference(ref_path, spec)
    assert r.table["dbh_m"].tolist() == pytest.approx([0.310, 0.405])
    assert r.table["height_m"].tolist() == pytest.approx([20.5, 25.0])


def test_real_example_yaml_parses():
    example = Path(__file__).resolve().parents[2] / "configs" / "reference_columns.example.yaml"
    if not example.is_file():
        pytest.skip("configs/ not mounted")
    spec = spec_from_yaml(example)
    assert spec.dbh_unit == "cm" and spec.tree_id_col and spec.dbh_col and spec.height_col


def test_reference_table_dtypes(tmp_path):
    r = load_reference(write_text(tmp_path / "ref.csv", POLISH_CSV, "utf-8-sig"))
    assert isinstance(r.table, pd.DataFrame) and list(r.table.columns) == ["ordinal", "dbh_m", "height_m"]
    assert r.table["dbh_m"].dtype == float and r.table["height_m"].dtype == float

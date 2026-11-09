from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from profilometry_analyzer import notices as N
from profilometry_analyzer.readers import ReadError, read_profile, read_profiles
from profilometry_analyzer.readers.columns import choose_columns
from profilometry_analyzer.units import guess_lateral_unit, normalize_unit, unit_in_header


def _table(n: int = 50, dx: float = 2.0, z_scale: float = 1.0) -> list[str]:
    x = np.arange(n) * dx
    z = np.where(x > x.mean(), 1000.0, 0.0) * z_scale
    return [f"{a:.4f},{b:.4f}" for a, b in zip(x, z)]


def test_units_normalised() -> None:
    assert normalize_unit("um") == "µm"
    assert normalize_unit("μm") == "µm"  # Greek mu
    assert normalize_unit("Angstrom") == "Å"
    assert normalize_unit("A") == "Å"
    assert normalize_unit("parsec") is None
    assert unit_in_header("Height (Å)") == "Å"
    assert unit_in_header("x [um]") == "µm"
    assert unit_in_header("z(raw)(nm)") == "nm"
    assert unit_in_header("height in nm") == "nm"
    assert unit_in_header("A") is None  # a bare letter is a name, not a unit


def test_lateral_unit_guess_uses_instrument_range() -> None:
    assert guess_lateral_unit(1000.0) == "µm"
    assert guess_lateral_unit(2.0) == "mm"  # 2 µm is below the 50 µm minimum
    assert guess_lateral_unit(5e8) == "Å"  # 50 mm
    assert guess_lateral_unit(1e12) is None


def test_csv_with_metadata_and_units(tmp_path: Path) -> None:
    lines = [
        "Sample,film A",
        "Stylus Force: 3 mg",
        "Scan Length,2000 um",
        "Scan Speed,50 um/s",
        "Stylus Radius,2 um",
        "Date,2026-11-09",
        "",
        "Lateral (µm),Height (Å)",
        *_table(n=100, dx=20.0),
    ]
    f = tmp_path / "a.csv"
    f.write_text("\n".join(lines), encoding="utf-8")
    p = read_profile(f)
    assert p.x_unit == "µm" and p.x_unit_source == "header"
    assert p.z_unit == "Å"
    assert p.z_nm.max() == pytest.approx(100.0)  # 1000 Å
    assert p.scan.stylus_force_mg == pytest.approx(3.0)
    assert p.scan.scan_length_um == pytest.approx(2000.0)
    assert p.scan.scan_speed_um_s == pytest.approx(50.0)
    assert p.scan.tip_radius_um == pytest.approx(2.0)
    assert p.scan.date == "2026-11-09"
    assert p.metadata["Sample"] == "film A"
    assert not [n for n in p.notices if n.code == N.UNIT_ASSUMED]
    # the values as read are kept
    assert p.raw_z is not None and p.raw_z.max() == pytest.approx(1000.0)


def test_unit_row_below_header(tmp_path: Path) -> None:
    f = tmp_path / "b.txt"
    f.write_text("\n".join(["x\tz", "mm\tnm", *[ln.replace(",", "\t") for ln in _table(60, 0.01)]]))
    p = read_profile(f)
    assert p.x_unit == "mm" and p.z_unit == "nm"
    assert p.length_um == pytest.approx(590.0)


def test_missing_units_warn(tmp_path: Path) -> None:
    f = tmp_path / "c.csv"
    f.write_text("\n".join(_table(80, 10.0)))
    p = read_profile(f)
    codes = [n.code for n in p.notices]
    assert codes.count(N.UNIT_ASSUMED) == 2
    assert p.z_unit == "Å" and p.x_unit == "µm"
    # explicit units remove the warning
    q = read_profile(f, x_unit="um", z_unit="nm")
    assert N.UNIT_ASSUMED not in [n.code for n in q.notices]
    assert q.z_nm.max() == pytest.approx(1000.0)


def test_scan_length_decides_lateral_unit(tmp_path: Path) -> None:
    f = tmp_path / "d.csv"
    f.write_text("\n".join(["Scan Length,5 mm", "", *_table(100, 0.05)]))
    p = read_profile(f, z_unit="nm")
    assert p.x_unit == "mm"
    assert p.length_um == pytest.approx(4950.0)


def test_semicolon_decimal_comma(tmp_path: Path) -> None:
    rows = [f"{i * 0.5:.2f};{(i % 7) * 1.25:.3f}".replace(".", ",") for i in range(200)]
    f = tmp_path / "e.csv"
    f.write_text("x (um);z (nm)\n" + "\n".join(rows), encoding="cp949")
    p = read_profile(f)
    assert p.n == 200
    assert p.dx_um == pytest.approx(0.5)


def test_raw_column_preferred_over_processed() -> None:
    x, z, notes = choose_columns(["Lateral (um)", "Levelled (A)", "Raw (A)"], 3)
    assert (x, z) == (0, 2)
    x, z, notes = choose_columns(["x", "Normalized height"], 2)
    assert z == 1 and any("processed" in n for n in notes)


def test_explicit_columns_by_name_and_index(tmp_path: Path) -> None:
    lines = ["pos (um),a (nm),b (nm)"] + [f"{i},{i * 2},{i * 3}" for i in range(50)]
    f = tmp_path / "f.csv"
    f.write_text("\n".join(lines))
    p = read_profile(f, z_col="b (nm)")
    assert p.z_nm[10] == pytest.approx(30.0)
    p = read_profile(f, z_col=1)
    assert p.z_nm[10] == pytest.approx(20.0)


def test_unsorted_x_is_sorted(tmp_path: Path) -> None:
    lines = ["x (um),z (nm)"] + [f"{i},{i}" for i in range(49, -1, -1)]
    f = tmp_path / "g.csv"
    f.write_text("\n".join(lines))
    p = read_profile(f)
    assert np.all(np.diff(p.x_um) > 0)
    assert p.history[0]["sorted"] is True


def test_xlsx(tmp_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Scan"
    ws.append(["Stylus Force", "2 mg"])
    ws.append([])
    ws.append(["x (um)", "z (nm)"])
    for i in range(40):
        ws.append([i * 5.0, 0.0 if i < 20 else 50.0])
    f = tmp_path / "h.xlsx"
    wb.save(f)
    p = read_profile(f)
    assert p.n == 40
    assert p.scan.stylus_force_mg == pytest.approx(2.0)
    assert p.reader["sheet"] == "Scan"


def test_opdx_rejected(tmp_path: Path) -> None:
    f = tmp_path / "scan.OPDx"
    f.write_bytes(b"\x00\x01binary")
    with pytest.raises(ReadError, match="OPDx"):
        read_profiles(f)


def test_no_table(tmp_path: Path) -> None:
    f = tmp_path / "empty.csv"
    f.write_text("hello\nworld\n")
    with pytest.raises(ReadError):
        read_profiles(f)


def test_several_blocks(tmp_path: Path) -> None:
    block1 = ["x (um),z (nm)", *_table(30, 1.0)]
    block2 = ["x (um),z (nm)", *_table(60, 1.0)]
    f = tmp_path / "two.csv"
    f.write_text("\n".join([*block1, "", "second scan", *block2]))
    profs = read_profiles(f)
    assert [p.n for p in profs] == [30, 60]
    assert read_profile(f).n == 60
    assert read_profile(f, block=0).n == 30


def test_out_of_range_scan_length_warns(tmp_path: Path) -> None:
    f = tmp_path / "big.csv"
    f.write_text("x (mm),z (nm)\n" + "\n".join(f"{i * 10},{i}" for i in range(20)))
    p = read_profile(f)
    assert N.OUTSIDE_INSTRUMENT_RANGE in [n.code for n in p.notices]


def test_reader_options_recorded(tmp_path: Path) -> None:
    f = tmp_path / "opt.csv"
    f.write_text("\n".join(_table(40, 10.0)))
    p = read_profile(f, z_unit="nm")
    assert p.reader["options"] == {"z_unit": "nm"}

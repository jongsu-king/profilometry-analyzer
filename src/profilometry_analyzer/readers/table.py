"""From rows of cells (text lines or worksheet rows) to profiles.

Both the text reader and the Excel reader turn their input into ``list[list[str]]`` and
hand it to :func:`profiles_from_rows`, so header, unit and metadata handling is the
same for every format:

* numeric blocks are runs of rows whose cells are all numbers (at least two columns,
  at least ``MIN_ROWS`` rows); the largest block is the scan, other large blocks become
  further profiles;
* the text row directly above a block is its header, unless it holds only units, in
  which case it is the unit row and the header is the row above;
* every other text row is kept as metadata (``key: value``, ``key, value`` or
  ``key<TAB>value``); stylus force, scan length, speed, stylus radius, number of points
  and date are recognised by generic words and converted to mg, µm, µm/s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import notices as N
from ..instrument import DEFAULT_INSTRUMENT, InstrumentSpec
from ..notices import Notice
from ..profile import Profile, ScanInfo
from ..units import (
    DEFAULT_Z_UNIT,
    TO_NM,
    TO_UM,
    guess_lateral_unit,
    is_unit_token,
    normalize_unit,
    unit_in_header,
)
from .columns import choose_columns, resolve_column

MIN_ROWS = 10
_NUMBER = re.compile(r"[-+]?(?:\d+[.,]?\d*|[.,]\d+)(?:[eE][-+]?\d+)?")


class ReadError(ValueError):
    """No profile could be read from the input."""


def is_number(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def numeric_row(cells: list[str]) -> list[float] | None:
    vals = [c for c in cells if c.strip() != ""]
    if len(vals) < 2 or not all(is_number(c) for c in vals):
        return None
    return [float(c) for c in vals]


@dataclass
class Block:
    start: int
    stop: int
    data: np.ndarray
    header: list[str] | None
    units_row: list[str] | None
    header_rows: tuple[int, ...]


def find_blocks(rows: list[list[str]], min_rows: int = MIN_ROWS) -> list[Block]:
    blocks: list[Block] = []
    i = 0
    n = len(rows)
    while i < n:
        first = numeric_row(rows[i])
        if first is None:
            i += 1
            continue
        width = len(first)
        j = i
        values: list[list[float]] = []
        while j < n:
            r = numeric_row(rows[j])
            if r is None or len(r) != width:
                break
            values.append(r)
            j += 1
        if len(values) >= min_rows:
            header, units_row, used = _header_above(rows, i, width)
            blocks.append(Block(i, j, np.array(values, dtype=float), header, units_row, used))
        i = j if j > i else i + 1
    return blocks


def _cells(row: list[str]) -> list[str]:
    return [c.strip() for c in row if c.strip() != ""]


def _header_above(
    rows: list[list[str]], start: int, width: int
) -> tuple[list[str] | None, list[str] | None, tuple[int, ...]]:
    k = start - 1
    while k >= 0 and not _cells(rows[k]):
        k -= 1
    if k < 0:
        return None, None, ()
    above = _cells(rows[k])
    if len(above) == width and all(is_unit_token(c) for c in above):
        m = k - 1
        while m >= 0 and not _cells(rows[m]):
            m -= 1
        if m >= 0 and len(_cells(rows[m])) == width:
            return _cells(rows[m]), above, (m, k)
        return None, above, (k,)
    if len(above) == width and not numeric_row(rows[k]):
        return above, None, (k,)
    return None, None, ()


# ---------------------------------------------------------------- metadata
def metadata_from_rows(rows: list[list[str]], skip: set[int]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for idx, row in enumerate(rows):
        if idx in skip:
            continue
        cells = _cells(row)
        if not cells or numeric_row(row) is not None:
            continue
        if len(cells) == 1:
            m = re.match(r"^\s*([^:=]+?)\s*[:=]\s*(.*)$", cells[0])
            if not m:
                key, value = cells[0], ""
            else:
                key, value = m.group(1), m.group(2)
        else:
            key, value = cells[0], " ".join(cells[1:])
            if key.endswith((":", "=")):
                key = key[:-1]
        key = key.strip()
        if not key:
            continue
        name, k = key, 2
        while name in meta:
            name = f"{key} ({k})"
            k += 1
        meta[name] = value.strip()
    return meta


def _number_and_unit(key: str, value: str) -> tuple[float | None, str | None]:
    m = _NUMBER.search(value)
    if not m:
        return None, None
    num = float(m.group(0).replace(",", "."))
    rest = value[m.end() :].strip()
    unit: str | None = rest.split()[0].lower() if rest else None
    if unit is None:
        bracket = re.search(r"[\(\[]\s*([^\)\]]+)\s*[\)\]]", key)
        unit = bracket.group(1).strip().lower() if bracket else None
    return num, unit


_LENGTH_TO_UM = {"um": 1.0, "µm": 1.0, "μm": 1.0, "micron": 1.0, "microns": 1.0, "mm": 1e3}
_LENGTH_TO_UM |= {"nm": 1e-3, "å": 1e-4, "a": 1e-4, "m": 1e6}


def scan_info_from_metadata(meta: dict[str, str]) -> tuple[ScanInfo, list[str]]:
    """Scan parameters recognised in metadata, and notes on what was skipped."""
    info = ScanInfo()
    notes: list[str] = []
    for key, value in meta.items():
        k = key.lower()
        num, unit = _number_and_unit(key, value)
        if num is None:
            if ("date" in k or k == "time") and info.date is None and value:
                info.set("date", value, f"file: {key}")
            continue
        u = (unit or "").strip(".,;").replace("μ", "µ")
        if "force" in k and info.stylus_force_mg is None:
            if u in ("", "mg"):
                info.set("stylus_force_mg", num, f"file: {key}")
            elif u in ("un", "µn"):
                info.set("stylus_force_mg", num / 9.80665, f"file: {key} (µN -> mg)")
            elif u == "mn":
                info.set("stylus_force_mg", num * 1e3 / 9.80665, f"file: {key} (mN -> mg)")
            else:
                notes.append(f"force unit {unit!r} in {key!r} not understood")
        elif "radius" in k and info.tip_radius_um is None:
            factor = _LENGTH_TO_UM.get(u, 1.0 if u == "" else None)
            if factor is None:
                notes.append(f"radius unit {unit!r} in {key!r} not understood")
            else:
                info.set("tip_radius_um", num * factor, f"file: {key}")
        elif ("speed" in k or "velocity" in k) and info.scan_speed_um_s is None:
            base = u.split("/")[0] if "/" in u else u
            factor = _LENGTH_TO_UM.get(base, 1.0 if base == "" else None)
            if factor is None:
                notes.append(f"speed unit {unit!r} in {key!r} not understood")
            else:
                info.set("scan_speed_um_s", num * factor, f"file: {key}")
        elif ("length" in k or k in ("scan range", "range")) and info.scan_length_um is None:
            factor = _LENGTH_TO_UM.get(u, 1.0 if u == "" else None)
            if factor is None:
                notes.append(f"length unit {unit!r} in {key!r} not understood")
            else:
                info.set("scan_length_um", num * factor, f"file: {key}")
        elif ("points" in k or k in ("npts", "n points")) and info.n_points is None:
            info.set("n_points", int(round(num)), f"file: {key}")
        elif ("duration" in k or "scan time" in k) and info.duration_s is None:
            info.set("duration_s", num, f"file: {key}")
        elif ("date" in k or k == "time") and info.date is None:
            info.set("date", value, f"file: {key}")
    return info, notes


# ---------------------------------------------------------------- profiles
def profiles_from_rows(
    rows: list[list[str]],
    *,
    path: str | None,
    name: str,
    reader: dict[str, Any],
    x_col: int | str | None = None,
    z_col: int | str | None = None,
    x_unit: str | None = None,
    z_unit: str | None = None,
    block: int | None = None,
    instrument: InstrumentSpec = DEFAULT_INSTRUMENT,
) -> list[Profile]:
    blocks = find_blocks(rows)
    if not blocks:
        raise ReadError(f"no numeric table with at least {MIN_ROWS} rows and two columns was found")
    largest = max(b.data.shape[0] for b in blocks)
    if block is not None:
        if not 0 <= block < len(blocks):
            raise ReadError(f"block {block} does not exist (file has {len(blocks)})")
        chosen = [(block, blocks[block])]
    else:
        chosen = [(i, b) for i, b in enumerate(blocks) if b.data.shape[0] >= 0.1 * largest]
    skip: set[int] = set()
    for b in blocks:
        skip.update(range(b.start, b.stop))
        skip.update(b.header_rows)
    meta = metadata_from_rows(rows, skip)
    scan, scan_notes = scan_info_from_metadata(meta)

    out: list[Profile] = []
    for k, b in chosen:
        prof = _profile_from_block(
            b,
            index=k,
            n_blocks=len(blocks),
            path=path,
            name=name if len(chosen) == 1 else f"{name} [{k + 1}]",
            reader=reader,
            meta=meta,
            scan=scan,
            scan_notes=scan_notes,
            x_col=x_col,
            z_col=z_col,
            x_unit=x_unit,
            z_unit=z_unit,
            instrument=instrument,
        )
        out.append(prof)
    return out


def _unit_option(unit: str | None, what: str) -> str | None:
    if unit is None:
        return None
    u = normalize_unit(unit)
    if u is None or u not in TO_UM:
        raise ReadError(f"unknown {what} unit {unit!r} (use Å, nm, µm/um or mm)")
    return u


def _profile_from_block(
    b: Block,
    *,
    index: int,
    n_blocks: int,
    path: str | None,
    name: str,
    reader: dict[str, Any],
    meta: dict[str, str],
    scan: ScanInfo,
    scan_notes: list[str],
    x_col: int | str | None,
    z_col: int | str | None,
    x_unit: str | None,
    z_unit: str | None,
    instrument: InstrumentSpec,
) -> Profile:
    data = b.data
    ncol = data.shape[1]
    header = b.header
    col_notes: list[str] = []
    xi = resolve_column(x_col, header, ncol)
    zi = resolve_column(z_col, header, ncol)
    if xi is None or zi is None:
        auto_x, auto_z, col_notes = choose_columns(header, ncol)
        if xi is None:
            xi = auto_x if zi != auto_x else auto_z
        if zi is None:
            zi = auto_z if auto_z != xi else auto_x
    if xi == zi:
        raise ReadError("the lateral and height columns are the same")
    if header is None:
        c0, c1 = data[:, xi], data[:, zi]
        if not _monotonic(c0) and _monotonic(c1) and x_col is None and z_col is None:
            xi, zi = zi, xi
            col_notes.append("second column is monotonic: taken as x")
    x_name = header[xi] if header else f"column {xi + 1}"
    z_name = header[zi] if header else f"column {zi + 1}"
    raw_x = data[:, xi].astype(float)
    raw_z = data[:, zi].astype(float)

    notices: list[Notice] = []
    # ---- lateral unit
    xu_user = _unit_option(x_unit, "lateral")
    xu_head = unit_in_header(x_name) or (
        normalize_unit(b.units_row[xi]) if b.units_row and xi < len(b.units_row) else None
    )
    span = float(np.nanmax(raw_x) - np.nanmin(raw_x))
    if xu_user:
        xu, xsrc = xu_user, "user"
        if xu_head and xu_head != xu_user:
            notices.append(
                Notice(
                    N.UNIT_FROM_USER,
                    N.INFO,
                    f"lateral unit {xu_user} given by the user although the header says {xu_head}",
                )
            )
    elif xu_head:
        xu, xsrc = xu_head, "header"
    else:
        xu = None
        if scan.scan_length_um:
            ratios = {u: span * TO_UM[u] / scan.scan_length_um for u in ("µm", "mm", "nm", "Å")}
            best = min(ratios, key=lambda u: abs(np.log(max(ratios[u], 1e-30))))
            if 0.8 <= ratios[best] <= 1.25:
                xu = best
                xsrc = "assumed (matches the scan length in the file)"
        if xu is None:
            xu = guess_lateral_unit(
                span, instrument.scan_length_min_um, instrument.scan_length_max_um
            )
            xsrc = "assumed (scan length inside the instrument range)"
        if xu is None:
            xu = "µm"
            xsrc = "assumed (default)"
        notices.append(
            Notice(
                N.UNIT_ASSUMED,
                N.WARNING,
                f"the lateral unit is not in the file; {xu} assumed"
                f" ({xsrc.split('(')[-1].rstrip(')')}). Set the unit explicitly to remove this"
                " warning.",
                {"axis": "x", "unit": xu},
            )
        )
    # ---- height unit
    zu_user = _unit_option(z_unit, "height")
    zu_head = unit_in_header(z_name) or (
        normalize_unit(b.units_row[zi]) if b.units_row and zi < len(b.units_row) else None
    )
    if zu_user:
        zu, zsrc = zu_user, "user"
        if zu_head and zu_head != zu_user:
            notices.append(
                Notice(
                    N.UNIT_FROM_USER,
                    N.INFO,
                    f"height unit {zu_user} given by the user although the header says {zu_head}",
                )
            )
    elif zu_head:
        zu, zsrc = zu_head, "header"
    else:
        zu, zsrc = DEFAULT_Z_UNIT, "assumed (default)"
        notices.append(
            Notice(
                N.UNIT_ASSUMED,
                N.WARNING,
                f"the height unit is not in the file; {zu} assumed. A wrong height unit scales"
                " every thickness by 10x or more - set it explicitly.",
                {"axis": "z", "unit": zu},
            )
        )

    assert xu is not None
    finite = np.isfinite(raw_x) & np.isfinite(raw_z)
    order = np.argsort(raw_x[finite], kind="stable")
    x_um = raw_x[finite][order] * TO_UM[xu]
    z_nm = raw_z[finite][order] * TO_NM[zu]
    hist: list[dict[str, Any]] = [
        {
            "step": "read",
            "x_column": x_name,
            "z_column": z_name,
            "x_unit": xu,
            "z_unit": zu,
            "dropped_nonfinite": int((~finite).sum()),
            "sorted": bool(np.any(np.diff(raw_x[finite]) < 0)),
        }
    ]
    prof = Profile(
        x_um=x_um,
        z_nm=z_nm,
        name=name,
        path=path,
        x_unit=xu,
        z_unit=zu,
        x_unit_source=xsrc,
        z_unit_source=zsrc,
        raw_x=raw_x.copy(),
        raw_z=raw_z.copy(),
        metadata=dict(meta),
        scan=ScanInfo.from_dict(scan.to_dict()),
        history=hist,
        notices=notices,
        reader={
            **reader,
            "block_index": index,
            "n_blocks": n_blocks,
            "first_data_row": b.start,
            "header": header,
            "units_row": b.units_row,
            "x_column": x_name,
            "z_column": z_name,
            "x_column_index": xi,
            "z_column_index": zi,
            "column_notes": col_notes,
            "metadata_notes": scan_notes,
        },
    )
    prof.notices.extend(plausibility(prof, instrument))
    return prof


def _monotonic(v: np.ndarray) -> bool:
    d = np.diff(v[np.isfinite(v)])
    return bool(d.size) and (bool(np.all(d > 0)) or bool(np.all(d < 0)))


def plausibility(prof: Profile, inst: InstrumentSpec) -> list[Notice]:
    """Checks of a freshly read profile against the instrument's nominal limits."""
    out: list[Notice] = []
    d = np.diff(prof.x_um)
    if d.size and np.median(d) > 0:
        spread = float(np.max(np.abs(d - np.median(d))) / np.median(d))
        if spread > 0.01:
            out.append(
                Notice(
                    N.NONUNIFORM_SPACING,
                    N.INFO,
                    f"point spacing varies by up to {spread:.1%}; filters resample to a uniform"
                    " grid",
                    {"max_relative_deviation": spread},
                )
            )
    length = prof.length_um
    if not inst.scan_length_min_um * 0.999 <= length <= inst.scan_length_max_um * 1.001:
        out.append(
            Notice(
                N.OUTSIDE_INSTRUMENT_RANGE,
                N.WARNING,
                f"scan length {length:.4g} µm lies outside the instrument range "
                f"{inst.scan_length_min_um:g}-{inst.scan_length_max_um:g} µm: check the lateral"
                " unit",
                {"what": "scan_length", "value_um": length},
            )
        )
    if prof.n > inst.max_points:
        out.append(
            Notice(
                N.OUTSIDE_INSTRUMENT_RANGE,
                N.INFO,
                f"{prof.n} points is more than the instrument's {inst.max_points} per scan",
                {"what": "points", "value": prof.n},
            )
        )
    span = float(np.nanmax(prof.z_nm) - np.nanmin(prof.z_nm))
    if span > inst.vertical_max_nm:
        out.append(
            Notice(
                N.OUTSIDE_INSTRUMENT_RANGE,
                N.WARNING,
                f"height range {span / 1e3:.4g} µm exceeds the instrument's "
                f"{inst.vertical_max_nm / 1e3:g} µm: check the height unit",
                {"what": "vertical_range", "value_nm": span},
            )
        )
    sc = prof.scan
    if sc.stylus_force_mg is not None and not (
        inst.force_min_mg <= sc.stylus_force_mg <= inst.force_max_mg
    ):
        out.append(
            Notice(
                N.OUTSIDE_INSTRUMENT_RANGE,
                N.INFO,
                f"stylus force {sc.stylus_force_mg:g} mg outside {inst.force_min_mg:g}-"
                f"{inst.force_max_mg:g} mg",
                {"what": "force", "value_mg": sc.stylus_force_mg},
            )
        )
    if sc.scan_speed_um_s is not None and not (
        inst.speed_min_um_s <= sc.scan_speed_um_s <= inst.speed_max_um_s
    ):
        out.append(
            Notice(
                N.OUTSIDE_INSTRUMENT_RANGE,
                N.INFO,
                f"scan speed {sc.scan_speed_um_s:g} µm/s outside {inst.speed_min_um_s:g}-"
                f"{inst.speed_max_um_s:g} µm/s",
                {"what": "speed", "value_um_s": sc.scan_speed_um_s},
            )
        )
    if sc.scan_length_um and abs(length - sc.scan_length_um) > 0.05 * sc.scan_length_um:
        out.append(
            Notice(
                N.OUTSIDE_INSTRUMENT_RANGE,
                N.WARNING,
                f"the data span {length:.4g} µm but the file says the scan length is "
                f"{sc.scan_length_um:.4g} µm: check the lateral unit",
                {"what": "scan_length_mismatch"},
            )
        )
    return out

"""Excel workbook reader (.xlsx, .xlsm) via openpyxl.

The worksheet is turned into rows of cell texts and handed to the same table logic as
text files, so headers, unit rows and metadata rows work the same way.  By default the
first worksheet that contains a numeric table is read; ``sheet`` picks one by name or
0-based index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..instrument import DEFAULT_INSTRUMENT, InstrumentSpec
from ..profile import Profile
from .table import ReadError, find_blocks, profiles_from_rows


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return repr(float(value))
    return str(value).strip()


def sheet_rows(ws: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in ws.iter_rows(values_only=True):
        cells = [_cell_text(v) for v in row]
        while cells and cells[-1] == "":
            cells.pop()
        rows.append(cells)
    return rows


def read_excel(
    path: str | Path,
    *,
    sheet: int | str | None = None,
    x_col: int | str | None = None,
    z_col: int | str | None = None,
    x_unit: str | None = None,
    z_unit: str | None = None,
    block: int | None = None,
    instrument: InstrumentSpec = DEFAULT_INSTRUMENT,
    **_: Any,
) -> list[Profile]:
    from openpyxl import load_workbook

    p = Path(path)
    wb = load_workbook(p, read_only=True, data_only=True)
    try:
        names = list(wb.sheetnames)
        if sheet is None:
            chosen = None
            for nm in names:
                rows = sheet_rows(wb[nm])
                if find_blocks(rows):
                    chosen = nm
                    break
            if chosen is None:
                raise ReadError("no worksheet contains a numeric table")
        elif isinstance(sheet, int) or str(sheet).isdigit():
            idx = int(sheet)
            if not 0 <= idx < len(names):
                raise ReadError(f"sheet index {idx} out of range (0..{len(names) - 1})")
            chosen = names[idx]
            rows = sheet_rows(wb[chosen])
        else:
            if sheet not in names:
                raise ReadError(f"sheet {sheet!r} not found; sheets: {names}")
            chosen = str(sheet)
            rows = sheet_rows(wb[chosen])
    finally:
        wb.close()
    name = p.stem if len(names) == 1 else f"{p.stem} [{chosen}]"
    return profiles_from_rows(
        rows,
        path=str(p),
        name=name,
        reader={"format": "excel", "sheet": chosen, "sheets": names},
        x_col=x_col,
        z_col=z_col,
        x_unit=x_unit,
        z_unit=z_unit,
        block=block,
        instrument=instrument,
    )

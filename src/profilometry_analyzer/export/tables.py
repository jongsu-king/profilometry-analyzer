"""CSV and Excel tables (no pandas needed)."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..notices import Notice


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float | np.floating):
        return float(v) if np.isfinite(v) else ""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, list | tuple | dict):
        return json.dumps(v, ensure_ascii=False)
    return v


def columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = columns(rows)
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in cols])
    return p


def notice_rows(notices: Sequence[Notice], **extra: Any) -> list[dict[str, Any]]:
    return [
        {**extra, "severity": n.severity, "code": n.code, "message": n.message} for n in notices
    ]


def write_xlsx(path: str | Path, sheets: Mapping[str, Sequence[dict[str, Any]]]) -> Path:
    """One worksheet per table; header row bold and frozen."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    first = True
    for name, rows in sheets.items():
        ws = wb.active if first else wb.create_sheet()
        assert ws is not None
        ws.title = name[:31]
        first = False
        cols = columns(rows)
        ws.append(cols)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([_cell(r.get(c)) for c in cols])
        ws.freeze_panes = "A2"
        for i, c in enumerate(cols, start=1):
            width = max([len(str(c))] + [len(str(_cell(r.get(c)))) for r in rows[:200]])
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(width + 2, 8), 60)
    wb.save(p)
    return p


def profile_rows(profile: Any, extra: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
    """Point table of a profile: position, height as read (nm), processed height, valid."""
    raw = profile.raw_z
    rows = []
    cols = extra or {}
    read: dict[str, Any] = next((h for h in profile.history if h.get("step") == "read"), {})
    aligned = (
        raw is not None
        and raw.size == profile.n
        and not read.get("sorted", False)
        and not any(h.get("step") == "crop" for h in profile.history)
    )
    for i in range(profile.n):
        row: dict[str, Any] = {
            "x_um": float(profile.x_um[i]),
            "z_nm": float(profile.z_nm[i]),
            "valid": bool(profile.valid[i]),
        }
        if aligned:
            row["z_file_units"] = float(raw[i])
        for k, v in cols.items():
            row[k] = float(v[i])
        rows.append(row)
    return rows

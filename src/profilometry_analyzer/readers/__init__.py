"""File readers.  :func:`read_profiles` picks the reader from the file extension.

Supported: delimited text (``.csv``, ``.txt``, ``.tsv``, ``.dat``, ``.asc``) and Excel
(``.xlsx``, ``.xlsm``).  Binary instrument files (``.OPDx``) are not supported in this
version; export the scan as CSV from the instrument software.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..profile import Profile
from .excel_reader import read_excel
from .table import ReadError
from .text_reader import read_text

TEXT_SUFFIXES = (".csv", ".txt", ".tsv", ".dat", ".asc", ".prn")
EXCEL_SUFFIXES = (".xlsx", ".xlsm")
SUPPORTED_SUFFIXES = TEXT_SUFFIXES + EXCEL_SUFFIXES
UNSUPPORTED = {
    ".opdx": "OPDx binary files are not supported in v0.1; export the scan as CSV",
    ".opd": "OPD binary files are not supported; export the scan as CSV",
    ".xls": "legacy .xls workbooks are not supported; save as .xlsx or CSV",
}

READER_OPTIONS = (
    "delimiter",
    "encoding",
    "sheet",
    "x_col",
    "z_col",
    "x_unit",
    "z_unit",
    "block",
)


def read_profiles(path: str | Path, **options: Any) -> list[Profile]:
    """All profiles in a file (usually one)."""
    p = Path(path)
    if not p.is_file():
        raise ReadError(f"file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in UNSUPPORTED:
        raise ReadError(UNSUPPORTED[suffix])
    reader = read_excel if suffix in EXCEL_SUFFIXES else read_text
    profiles = reader(p, **options)
    used = {k: v for k, v in options.items() if k in READER_OPTIONS and v is not None}
    for prof in profiles:
        prof.reader["options"] = dict(used)
    return profiles


def read_profile(path: str | Path, **options: Any) -> Profile:
    """The profile of a file; with several tables, the one with the most points."""
    profiles = read_profiles(path, **options)
    return max(profiles, key=lambda p: p.n)


__all__ = [
    "EXCEL_SUFFIXES",
    "READER_OPTIONS",
    "SUPPORTED_SUFFIXES",
    "TEXT_SUFFIXES",
    "ReadError",
    "read_excel",
    "read_profile",
    "read_profiles",
    "read_text",
]

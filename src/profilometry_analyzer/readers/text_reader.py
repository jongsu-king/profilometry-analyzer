"""Delimited text reader (CSV, TSV, semicolon, whitespace).

Instrument CSV exports put metadata lines above the table; those lines are kept (see
:mod:`.table`).  Encoding is tried as UTF-8 (with BOM), CP949 and Latin-1; decimal
commas are accepted when the delimiter is not a comma.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..instrument import DEFAULT_INSTRUMENT, InstrumentSpec
from ..profile import Profile
from .table import ReadError, is_number, profiles_from_rows

DELIMITERS = ("\t", ",", ";", "whitespace")


def decode(path: Path, encoding: str | None = None) -> tuple[str, str]:
    raw = path.read_bytes()
    candidates = [encoding] if encoding else ["utf-8-sig", "cp949", "latin-1"]
    for enc in candidates:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def tokens(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        toks = line.split()
    else:
        toks = [t.strip().strip('"').strip() for t in line.split(delimiter)]
    if delimiter != "," and any("," in t and "." not in t for t in toks):
        fixed = [t.replace(",", ".") if is_number(t.replace(",", ".")) else t for t in toks]
        toks = fixed
    while toks and toks[-1] == "":
        toks.pop()
    return toks


def _numeric_width(line: str, delimiter: str) -> int:
    toks = [t for t in tokens(line, delimiter) if t != ""]
    if len(toks) < 2 or not all(is_number(t) for t in toks):
        return 0
    return len(toks)


def sniff_delimiter(lines: list[str]) -> str:
    sample = [ln for ln in lines if ln.strip()]
    if len(sample) > 4000:
        step = len(sample) // 2000
        sample = sample[::step]
    best, best_score = "whitespace", -1
    for delim in DELIMITERS:
        score = sum(1 for ln in sample if _numeric_width(ln, delim) >= 2)
        if score > best_score:
            best, best_score = delim, score
    if best_score <= 0:
        raise ReadError("no line with at least two numeric columns was found")
    return best


def read_text(
    path: str | Path,
    *,
    delimiter: str | None = None,
    encoding: str | None = None,
    x_col: int | str | None = None,
    z_col: int | str | None = None,
    x_unit: str | None = None,
    z_unit: str | None = None,
    block: int | None = None,
    instrument: InstrumentSpec = DEFAULT_INSTRUMENT,
    **_: Any,
) -> list[Profile]:
    p = Path(path)
    text, enc = decode(p, encoding)
    lines = text.splitlines()
    delim = delimiter or sniff_delimiter(lines)
    rows = [tokens(ln, delim) for ln in lines]
    reader = {
        "format": "text",
        "encoding": enc,
        "delimiter": "whitespace" if delim == "whitespace" else delim,
    }
    return profiles_from_rows(
        rows,
        path=str(p),
        name=p.stem,
        reader=reader,
        x_col=x_col,
        z_col=z_col,
        x_unit=x_unit,
        z_unit=z_unit,
        block=block,
        instrument=instrument,
    )

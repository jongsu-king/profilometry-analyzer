"""Header heuristics shared by the text and Excel readers.

Profilometer exports name their columns in many ways (``x (um)``, ``Scan Length``,
``Lateral [µm]``, ``Height (Å)``, ``z(raw)``, ``Profile``...).  The helpers here decide
which column is the lateral axis and which one holds the measured heights, and prefer
a raw height column over one the instrument software already levelled or filtered.
No vendor column name is assumed: the rules match generic words only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

X_PATTERNS = ("lateral", "distance", "position", "scan length", "horizontal", "x-axis", "x axis")
X_EXACT = frozenset({"x", "pos", "position", "distance", "length", "lateral", "scan", "xpos"})
Z_PATTERNS = ("height", "depth", "profile", "vertical", "z-axis", "z axis", "topograph", "raw")
Z_EXACT = frozenset({"z", "y", "h", "height", "data", "value", "raw", "zpos", "level"})
# Height columns that are already processed by the instrument software; used only when
# no raw column exists or when asked for by name.
PROCESSED_PATTERNS = (
    "level",
    "normali",
    "filter",
    "rough",
    "wavi",
    "smooth",
    "fit",
    "corrected",
    "flatten",
)


def base_name(header: str) -> str:
    """Lower-case header without bracketed parts: ``"Height (Å)" -> "height"``."""
    return re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", header.lower()).strip(" _-:")


def header_matches(header: str, patterns: Sequence[str], exact: frozenset[str]) -> bool:
    base = base_name(header)
    return base in exact or any(p in base for p in patterns)


def is_x_header(header: str) -> bool:
    base = base_name(header)
    if header_matches(header, X_PATTERNS, X_EXACT):
        return True
    return bool(re.match(r"^x\b|^x\d*$", base))


def is_z_header(header: str) -> bool:
    if is_x_header(header):
        return False
    base = base_name(header)
    if header_matches(header, Z_PATTERNS, Z_EXACT):
        return True
    return bool(re.match(r"^z\b|^z\d*$", base))


def is_processed_header(header: str) -> bool:
    base = base_name(header)
    if "raw" in base:
        return False
    return any(p in base for p in PROCESSED_PATTERNS)


def choose_columns(header: Sequence[str] | None, ncol: int) -> tuple[int, int, list[str]]:
    """``(x_index, z_index, notes)`` for a table with this header.

    The first lateral-looking column is x.  Among height-looking columns a raw one wins
    over processed ones; without any match the first column after x is used.
    """
    notes: list[str] = []
    if not header or len(header) != ncol:
        return 0, 1, ["no usable header: first column taken as x, second as z"]
    x_idx = next((i for i, h in enumerate(header) if is_x_header(h)), None)
    if x_idx is None:
        x_idx = 0
        notes.append(f"no lateral header recognised: column {header[0]!r} taken as x")
    candidates = [i for i, h in enumerate(header) if i != x_idx and is_z_header(h)]
    raw = [i for i in candidates if not is_processed_header(header[i])]
    if raw:
        z_idx = raw[0]
    elif candidates:
        z_idx = candidates[0]
        notes.append(f"only processed height columns found; using {header[z_idx]!r}")
    else:
        z_idx = 1 if x_idx != 1 else 0
        notes.append(f"no height header recognised: column {header[z_idx]!r} taken as z")
    others = [header[i] for i in range(ncol) if i not in (x_idx, z_idx)]
    if others:
        notes.append("other columns not used: " + ", ".join(repr(o) for o in others))
    return x_idx, z_idx, notes


def resolve_column(spec: int | str | None, header: Sequence[str] | None, ncol: int) -> int | None:
    """Column given by index (0-based) or by header text (case-insensitive)."""
    if spec is None:
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        idx = int(spec)
        if not 0 <= idx < ncol:
            raise ValueError(f"column index {idx} out of range (0..{ncol - 1})")
        return idx
    if header:
        wanted = spec.strip().lower()
        for i, h in enumerate(header):
            if h.strip().lower() == wanted:
                return i
        for i, h in enumerate(header):
            if base_name(h) == base_name(spec):
                return i
    raise ValueError(f"column {spec!r} not found in header {list(header or [])}")

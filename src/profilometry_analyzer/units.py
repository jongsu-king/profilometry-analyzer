"""Length units of the lateral axis (x) and the height axis (z).

Internally every profile stores x in micrometres and z in nanometres.  Units are read
from the column header when it names one (``Height (Å)``, ``x [um]``, ``z(nm)``); when it
does not, the lateral unit is chosen so that the scan length falls inside the instrument
range and the height unit is *assumed* - and the profile carries an ``UNIT_ASSUMED``
warning that is repeated in every output.
"""

from __future__ import annotations

import re

# Factor to convert one unit into micrometres.
TO_UM: dict[str, float] = {
    "Å": 1e-4,
    "nm": 1e-3,
    "µm": 1.0,
    "mm": 1e3,
    "m": 1e6,
}
TO_NM: dict[str, float] = {k: v * 1e3 for k, v in TO_UM.items()}

LENGTH_UNITS = ("Å", "nm", "µm", "mm")

# Assumed height unit when the file does not say.  Chosen as a documented default,
# not inferred: a magnitude cannot tell Å from nm.  Override with ``z_unit``.
DEFAULT_Z_UNIT = "Å"

_ALIASES: dict[str, str] = {
    "å": "Å",
    "a": "Å",
    "ang": "Å",
    "angs": "Å",
    "angstrom": "Å",
    "angstroms": "Å",
    "ångström": "Å",
    "ångstrom": "Å",
    "a°": "Å",
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "nanometre": "nm",
    "nanometres": "nm",
    "um": "µm",
    "µm": "µm",
    "μm": "µm",
    "micron": "µm",
    "microns": "µm",
    "micrometer": "µm",
    "micrometers": "µm",
    "micrometre": "µm",
    "micrometres": "µm",
    "mic": "µm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "m": "m",
}


def normalize_unit(text: str | None) -> str | None:
    """``"um"`` / ``"μm"`` / ``"microns"`` -> ``"µm"``; ``"A"`` / ``"Angstrom"`` -> ``"Å"``.

    Returns ``None`` for anything that is not a length unit.
    """
    if text is None:
        return None
    t = text.strip().strip(".").lower().replace("Å", "å")  # angstrom sign
    return _ALIASES.get(t)


def unit_in_header(header: str) -> str | None:
    """The unit a column header names, or ``None``.

    Looks inside brackets first (``Height (Å)``, ``x [um]``, ``z(raw)(A)`` - the last
    bracket that holds a unit wins), then at a trailing token after ``/`` or ``in``
    (``x / um``, ``height in nm``).  A bare letter outside brackets is never read as a
    unit, so a column called ``A`` stays a name.
    """
    found: str | None = None
    for m in re.finditer(r"[\(\[\{]\s*([^\)\]\}]*?)\s*[\)\]\}]", header):
        u = normalize_unit(m.group(1))
        if u is not None:
            found = u
    if found is not None:
        return found
    tail = re.search(r"(?:/|\bin\b)\s*([A-Za-zµμÅå°]+)\s*$", header.strip())
    if tail:
        return normalize_unit(tail.group(1))
    return None


def is_unit_token(token: str) -> bool:
    """True for a cell that holds only a unit (a separate unit row under the header)."""
    t = token.strip().strip("()[]{}")
    return normalize_unit(t) is not None and t.lower() not in {"m"}


def guess_lateral_unit(span: float, min_um: float = 50.0, max_um: float = 55_000.0) -> str | None:
    """Lateral unit that puts a scan of ``span`` file units inside ``[min_um, max_um]``.

    Preference order is µm, mm, nm, Å (the common export units); ``None`` when no unit
    gives an instrument-sized scan.
    """
    if not span > 0:
        return None
    for unit in ("µm", "mm", "nm", "Å"):
        length = span * TO_UM[unit]
        if min_um <= length <= max_um:
            return unit
    return None

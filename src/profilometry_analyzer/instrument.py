"""Nominal instrument limits used for plausibility warnings.

The defaults describe the stylus profilometer this project was written for (a Bruker
DektakXT-E at the Incheon National University core facility; values from the facility's
equipment page, checked 2026-11-06).  They are only used to *warn*; no result is
clipped or changed by them.  Change them for another instrument with
:func:`InstrumentSpec.replace` or the command-line options.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class InstrumentSpec:
    name: str = "stylus profilometer (DektakXT-E class defaults)"
    vertical_min_nm: float = 20.0
    """Lower end of the quoted vertical measuring range."""
    vertical_max_nm: float = 1_000_000.0
    """Upper end of the quoted vertical measuring range (1,000 µm)."""
    vertical_resolution_nm: float = 0.1
    """Best vertical resolution (1 Å, in the 6.55 µm range)."""
    step_repeatability_nm: float = 0.5
    """Step height repeatability, 1σ (5 Å on a 1 µm step)."""
    scan_length_min_um: float = 50.0
    scan_length_max_um: float = 55_000.0
    max_points: int = 120_000
    force_min_mg: float = 1.0
    force_max_mg: float = 15.0
    speed_min_um_s: float = 2.0
    speed_max_um_s: float = 200.0
    tip_radius_um: float = 2.0
    """Stylus radius used when the file does not give one.  The facility page lists
    1.25 µm and its 2022 leaflet 2 µm; the larger value is the default because it widens
    the excluded edge zones and raises more narrow-feature warnings, i.e. errs on the
    safe side.  Pass the radius of the stylus actually mounted."""
    cone_angle_deg: float = 60.0
    """Full angle of the stylus cone used when the file does not give one.  60° is the
    usual nominal angle of roughness styli (90° the other common one), and a DektakXT
    facility lists 60° for its 2 µm stylus; it only matters for features deeper than
    ``r (1 - sin 30°)`` = r / 2.  Pass the angle of the stylus actually mounted."""
    soft_film_max_force_mg: float = 3.0
    """Highest stylus force that does not trigger a warning on a film marked soft
    (polymers: start at 1-3 mg)."""

    def replace(self, **changes: Any) -> InstrumentSpec:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> InstrumentSpec:
        if not d:
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


DEFAULT_INSTRUMENT = InstrumentSpec()

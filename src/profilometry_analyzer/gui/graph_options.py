"""Display options of the live graph (Origin's "Plot Details" in small).

Kept free of Qt so they can be stored as JSON in the settings and tested on their own.
Positions are always analysed in µm and heights in nm; these options only change how
the graph shows them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

X_UNITS = ("µm", "mm")
Z_UNITS = ("auto", "Å", "nm", "µm")
X_FACTOR = {"µm": 1.0, "mm": 1e-3}
Z_FACTOR = {"Å": 10.0, "nm": 1.0, "µm": 1e-3}


@dataclass(frozen=True)
class GraphOptions:
    x_unit: str = "µm"
    z_unit: str = "auto"
    """``auto``: nm below 5 µm of height, else µm."""
    x_min: float | None = None
    """Axis limits in display units; ``None`` = automatic."""
    x_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    font_pt: int = 10
    line_width: float = 1.2
    show_legend: bool = True
    show_annotations: bool = True
    """Dimension lines with the measured heights and widths."""
    show_residuals: bool = True
    """Second layer with the residuals of the plane fits."""
    overlay_offset_nm: float = 0.0
    """Vertical offset between overlaid scans (Origin's "Offset Y")."""

    def __post_init__(self) -> None:
        if self.x_unit not in X_UNITS:
            raise ValueError(f"x unit must be one of {X_UNITS}")
        if self.z_unit not in Z_UNITS:
            raise ValueError(f"z unit must be one of {Z_UNITS}")
        for lo, hi in ((self.x_min, self.x_max), (self.z_min, self.z_max)):
            if lo is not None and hi is not None and not lo < hi:
                raise ValueError("axis minimum must be below the maximum")
        if not 6 <= self.font_pt <= 24:
            raise ValueError("font size must lie between 6 and 24 pt")
        if not 0.3 <= self.line_width <= 6:
            raise ValueError("line width must lie between 0.3 and 6")

    # ------------------------------------------------------------------ scaling
    @property
    def sx(self) -> float:
        return X_FACTOR[self.x_unit]

    def z_factor(self, span_nm: float) -> tuple[float, str]:
        """Factor from nm to the display unit, and the unit, for data of this span."""
        if self.z_unit == "auto":
            return (1e-3, "µm") if span_nm >= 5e3 else (1.0, "nm")
        return Z_FACTOR[self.z_unit], self.z_unit

    # ------------------------------------------------------------------ storage
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: Any) -> GraphOptions:
        """Options from stored JSON; anything unreadable gives the defaults."""
        try:
            data = json.loads(str(text)) if text else {}
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        except (ValueError, TypeError):
            return cls()

    def with_changes(self, **changes: Any) -> GraphOptions:
        return replace(self, **changes)


__all__ = ["X_FACTOR", "X_UNITS", "Z_FACTOR", "Z_UNITS", "GraphOptions"]

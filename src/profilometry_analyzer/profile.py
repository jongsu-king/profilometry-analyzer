"""The line profile and its processing history.

A :class:`Profile` holds the lateral positions in micrometres and the heights in
nanometres, the values exactly as read from the file (``raw_x``/``raw_z`` in file
units), the scan parameters found in the file or given by the user, a validity mask
(points removed as spikes are ``False``) and the list of processing steps that turned
the raw values into the current heights.  Processing functions never modify a profile
in place; they return a new one with one more ``history`` entry.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .notices import Notice

FloatArray = np.ndarray


@dataclass
class ScanInfo:
    """Scan parameters; ``sources`` records where each value came from."""

    stylus_force_mg: float | None = None
    scan_length_um: float | None = None
    scan_speed_um_s: float | None = None
    tip_radius_um: float | None = None
    n_points: int | None = None
    duration_s: float | None = None
    date: str | None = None
    soft_film: bool = False
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stylus_force_mg": self.stylus_force_mg,
            "scan_length_um": self.scan_length_um,
            "scan_speed_um_s": self.scan_speed_um_s,
            "tip_radius_um": self.tip_radius_um,
            "n_points": self.n_points,
            "duration_s": self.duration_s,
            "date": self.date,
            "soft_film": self.soft_film,
            "sources": dict(self.sources),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ScanInfo:
        d = d or {}
        return cls(
            stylus_force_mg=d.get("stylus_force_mg"),
            scan_length_um=d.get("scan_length_um"),
            scan_speed_um_s=d.get("scan_speed_um_s"),
            tip_radius_um=d.get("tip_radius_um"),
            n_points=d.get("n_points"),
            duration_s=d.get("duration_s"),
            date=d.get("date"),
            soft_film=bool(d.get("soft_film", False)),
            sources=dict(d.get("sources") or {}),
        )

    def set(self, name: str, value: Any, source: str) -> None:
        setattr(self, name, value)
        self.sources[name] = source


@dataclass
class Profile:
    x_um: FloatArray
    z_nm: FloatArray
    name: str = "profile"
    path: str | None = None
    x_unit: str = "µm"
    z_unit: str = "nm"
    x_unit_source: str = "default"
    z_unit_source: str = "default"
    raw_x: FloatArray | None = None
    raw_z: FloatArray | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    scan: ScanInfo = field(default_factory=ScanInfo)
    mask: np.ndarray | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    notices: list[Notice] = field(default_factory=list)
    reader: dict[str, Any] = field(default_factory=dict)
    """How the file was read (columns, delimiter, encoding, block) - for verify."""

    def __post_init__(self) -> None:
        self.x_um = np.asarray(self.x_um, dtype=float)
        self.z_nm = np.asarray(self.z_nm, dtype=float)
        if self.x_um.shape != self.z_nm.shape or self.x_um.ndim != 1:
            raise ValueError("x and z must be one-dimensional arrays of the same length")
        if self.x_um.size < 3:
            raise ValueError("a profile needs at least three points")
        if self.mask is None:
            self.mask = np.isfinite(self.x_um) & np.isfinite(self.z_nm)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
        if self.raw_x is None:
            self.raw_x = self.x_um.copy()
        if self.raw_z is None:
            self.raw_z = self.z_nm.copy()

    # ------------------------------------------------------------------ properties
    @property
    def n(self) -> int:
        return int(self.x_um.size)

    @property
    def valid(self) -> np.ndarray:
        assert self.mask is not None
        return self.mask

    @property
    def dx_um(self) -> float:
        """Median point spacing."""
        d = np.diff(self.x_um)
        return float(np.median(np.abs(d))) if d.size else float("nan")

    @property
    def length_um(self) -> float:
        return float(np.nanmax(self.x_um) - np.nanmin(self.x_um))

    @property
    def tip_radius_um(self) -> float | None:
        return self.scan.tip_radius_um

    # ------------------------------------------------------------------ processing
    def derive(
        self,
        z_nm: FloatArray | None = None,
        *,
        step: dict[str, Any],
        mask: np.ndarray | None = None,
        notices: list[Notice] | None = None,
    ) -> Profile:
        """A new profile with changed heights/mask and one more history entry."""
        new = copy.copy(self)
        new.z_nm = np.asarray(self.z_nm if z_nm is None else z_nm, dtype=float).copy()
        new.mask = (self.valid if mask is None else np.asarray(mask, dtype=bool)).copy()
        new.history = [*self.history, step]
        new.notices = [*self.notices, *(notices or [])]
        new.metadata = dict(self.metadata)
        new.scan = copy.deepcopy(self.scan)
        new.reader = dict(self.reader)
        return new

    def crop(self, x_start_um: float, x_stop_um: float) -> Profile:
        """The part of the profile between two positions (raw values cropped alike)."""
        a, b = sorted((float(x_start_um), float(x_stop_um)))
        keep = (self.x_um >= a) & (self.x_um <= b)
        if keep.sum() < 3:
            raise ValueError(f"the range {a:g}-{b:g} µm holds fewer than three points")
        new = copy.copy(self)
        new.x_um = self.x_um[keep].copy()
        new.z_nm = self.z_nm[keep].copy()
        new.mask = self.valid[keep].copy()
        if self.raw_x is not None and self.raw_x.size == keep.size:
            new.raw_x = self.raw_x[keep].copy()
        if self.raw_z is not None and self.raw_z.size == keep.size:
            new.raw_z = self.raw_z[keep].copy()
        new.history = [*self.history, {"step": "crop", "x_start_um": a, "x_stop_um": b}]
        new.notices = list(self.notices)
        new.metadata = dict(self.metadata)
        new.scan = copy.deepcopy(self.scan)
        new.reader = dict(self.reader)
        return new

    def with_scan(self, **values: Any) -> Profile:
        """A copy whose scan parameters are overridden by the user (recorded)."""
        new = copy.copy(self)
        new.scan = copy.deepcopy(self.scan)
        for k, v in values.items():
            if v is not None:
                new.scan.set(k, v, "user")
        new.history = [*self.history, {"step": "scan_parameters", "values": values}]
        return new

    def summary(self) -> dict[str, Any]:
        """Plain description for ``info`` and the GUI."""
        z = self.z_nm[self.valid]
        return {
            "name": self.name,
            "path": self.path,
            "points": self.n,
            "valid_points": int(self.valid.sum()),
            "length_um": self.length_um,
            "dx_um": self.dx_um,
            "x_unit": self.x_unit,
            "x_unit_source": self.x_unit_source,
            "z_unit": self.z_unit,
            "z_unit_source": self.z_unit_source,
            "z_min_nm": float(np.min(z)) if z.size else float("nan"),
            "z_max_nm": float(np.max(z)) if z.size else float("nan"),
            "scan": self.scan.to_dict(),
            "metadata": dict(self.metadata),
            "history": list(self.history),
            "notices": [n.to_dict() for n in self.notices],
            "reader": dict(self.reader),
        }

"""Warnings and notes attached to profiles and results.

A :class:`Notice` is data, not a Python warning: it travels with the profile or result it
belongs to and ends up in every table, figure caption and project file, so a number is
never reported without the conditions that limit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INFO = "info"
WARNING = "warning"
ERROR = "error"
SEVERITIES = (INFO, WARNING, ERROR)

# Codes (stable identifiers; the message text may change between versions).
UNIT_ASSUMED = "UNIT_ASSUMED"
UNIT_FROM_USER = "UNIT_FROM_USER"
NONUNIFORM_SPACING = "NONUNIFORM_SPACING"
SPIKES_REMOVED = "SPIKES_REMOVED"
OUTSIDE_INSTRUMENT_RANGE = "OUTSIDE_INSTRUMENT_RANGE"
NO_STEP_FOUND = "NO_STEP_FOUND"
SMALL_STEP = "SMALL_STEP"
NARROW_FEATURE = "NARROW_FEATURE"
BURR_EXCLUDED = "BURR_EXCLUDED"
NONPARALLEL_PLANES = "NONPARALLEL_PLANES"
SHORT_PLANE = "SHORT_PLANE"
A1_REGION_TRUNCATED = "A1_REGION_TRUNCATED"
A1_BOTTOM_NOT_FLAT = "A1_BOTTOM_NOT_FLAT"
STEP_BELOW_NOISE = "STEP_BELOW_NOISE"
SOFT_FILM_HIGH_FORCE = "SOFT_FILM_HIGH_FORCE"
SOFT_FILM_FORCE_UNKNOWN = "SOFT_FILM_FORCE_UNKNOWN"
BOW_REMOVED = "BOW_REMOVED"
SAMPLING_TOO_COARSE = "SAMPLING_TOO_COARSE"
EVALUATION_TOO_SHORT = "EVALUATION_TOO_SHORT"
STEP_IN_ROUGHNESS = "STEP_IN_ROUGHNESS"
SETTING_CLASS_FROM_MEASUREMENT = "SETTING_CLASS_FROM_MEASUREMENT"
STONEY_THICK_FILM = "STONEY_THICK_FILM"
STONEY_LARGE_DEFLECTION = "STONEY_LARGE_DEFLECTION"
BOW_NOT_SPHERICAL = "BOW_NOT_SPHERICAL"
CURVATURE_CHANGE_NOT_SIGNIFICANT = "CURVATURE_CHANGE_NOT_SIGNIFICANT"
SCAN_MISMATCH = "SCAN_MISMATCH"
SUBSTRATE_VALUE_UNVERIFIED = "SUBSTRATE_VALUE_UNVERIFIED"
LEVELING_REGION_EMPTY = "LEVELING_REGION_EMPTY"
LEVELING_REGIONS_DIFFER = "LEVELING_REGIONS_DIFFER"
MULTIPLE_STEPS = "MULTIPLE_STEPS"
FILTER_LONGER_THAN_DATA = "FILTER_LONGER_THAN_DATA"
LAMBDA_S_SKIPPED = "LAMBDA_S_SKIPPED"
RESAMPLED = "RESAMPLED"
STRESS_NO_REFERENCE = "STRESS_NO_REFERENCE"
SUBSTRATE_THICKNESS_TYPICAL = "SUBSTRATE_THICKNESS_TYPICAL"
TIP_RADIUS_DEFAULT = "TIP_RADIUS_DEFAULT"
READ_FAILED = "READ_FAILED"
CORRELATED_RESIDUALS = "CORRELATED_RESIDUALS"


@dataclass(frozen=True)
class Notice:
    """One condition that limits or qualifies a result."""

    code: str
    severity: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "data": _plain(self.data),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Notice:
        return cls(str(d["code"]), str(d["severity"]), str(d["message"]), dict(d.get("data") or {}))

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


def _plain(value: Any) -> Any:
    """Make numpy scalars JSON-friendly."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


def unique(notices: list[Notice]) -> list[Notice]:
    """Drop exact duplicates while keeping the order."""
    seen: set[tuple[str, str]] = set()
    out: list[Notice] = []
    for n in notices:
        key = (n.code, n.message)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def worst(notices: list[Notice]) -> str:
    """Highest severity in a list (``"ok"`` when empty)."""
    if any(n.severity == ERROR for n in notices):
        return ERROR
    if any(n.severity == WARNING for n in notices):
        return WARNING
    if notices:
        return INFO
    return "ok"

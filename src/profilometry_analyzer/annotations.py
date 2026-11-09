"""Dimension lines and fit residuals for figures and the live view.

Both the Matplotlib figures and the desktop plot draw the same things from a step
result: a vertical dimension line with arrow heads for each measured height (from the
reference plane to the measured plane, at the position where the height was
evaluated), a horizontal one for the width W of an ISO 5436-1 A1 groove or ridge, and
the residuals of the plane fits.  This module computes that geometry once, in µm and
nm, without any plotting library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.steps import PlaneFit, StepMeasurement
from .profile import Profile

MAX_EDGE_DIMENSIONS = 12


@dataclass(frozen=True)
class Dimension:
    kind: str
    """``"height"`` (vertical) or ``"width"`` (horizontal)."""
    x0_um: float
    z0_nm: float
    x1_um: float
    z1_nm: float
    value: float
    """Height in nm or width in µm (always positive)."""
    u: float | None
    label: str
    offset: bool = False
    """Draw the line beside ``x0`` (with extension lines) instead of on it - used for an
    edge, where a line on the edge itself would hide the data."""


@dataclass
class Residuals:
    name: str
    x_um: np.ndarray
    r_nm: np.ndarray
    """Residuals of the points that entered the plane fit."""
    x_excluded_um: np.ndarray
    r_excluded_nm: np.ndarray
    """Residuals of points inside the plane range that were excluded (burrs, outliers)."""
    sigma_nm: float


def fmt_uncertainty(u: float, sig: int = 2) -> str:
    """``u`` with ``sig`` significant digits in plain notation (0.000025, not 2.5e-05)."""
    if u == 0 or not np.isfinite(u):
        return f"{u:g}"
    digits = max(0, sig - 1 - int(np.floor(np.log10(abs(u)))))
    return f"{u:.{digits}f}"


def _fmt_height(h: float, u: float | None, unit_scale: float = 1.0, unit: str = "nm") -> str:
    v = abs(h) * unit_scale
    if u is None or not np.isfinite(u):
        return f"{v:.4g} {unit}"
    return f"{v:.4g} ± {fmt_uncertainty(u * unit_scale)} {unit}"


def _planes_around(planes: list[PlaneFit], x_um: float) -> tuple[PlaneFit, PlaneFit] | None:
    ordered = sorted(planes, key=lambda p: p.x_start_um)
    left = [p for p in ordered if p.x_stop_um <= x_um]
    right = [p for p in ordered if p.x_start_um >= x_um]
    if not left or not right:
        return None
    return left[-1], right[0]


def step_dimensions(m: StepMeasurement, label_unit: str = "nm") -> list[Dimension]:
    """Dimension lines of a step measurement (empty when nothing was measured)."""
    scale = {"Å": 10.0, "nm": 1.0, "µm": 1e-3}.get(label_unit, 1.0)
    out: list[Dimension] = []
    if not m.planes:
        return out
    if m.kind in ("groove", "ridge") and len(m.planes) >= 2 and m.position_um is not None:
        outer, inner = m.planes[0], m.planes[1]
        c = float(m.position_um)
        z_out, z_in = float(outer.line.at(c)), float(inner.line.at(c))
        word = "d" if m.kind == "groove" else "h"
        out.append(
            Dimension(
                "height",
                c,
                z_out,
                c,
                z_in,
                abs(z_in - z_out),
                m.u_nm,
                f"{word} = {_fmt_height(z_in - z_out, m.u_nm, scale, label_unit)}",
            )
        )
        if m.width_um:
            w = float(m.width_um)
            # inside the feature, away from the height label at mid-height
            zm = (
                z_in + 0.75 * (z_out - z_in)
                if m.kind == "groove"
                else z_out + 0.25 * (z_in - z_out)
            )
            out.append(
                Dimension(
                    "width",
                    c - w / 2,
                    zm,
                    c + w / 2,
                    zm,
                    w,
                    None,
                    f"W = {w:.4g} µm",
                )
            )
        return out
    if m.kind == "regions" and len(m.planes) == 2:
        a, b = m.planes
        xb = 0.5 * (b.x_start_um + b.x_stop_um)
        za, zb = float(a.line.at(xb)), float(b.line.at(xb))
        out.append(
            Dimension(
                "height",
                xb,
                za,
                xb,
                zb,
                abs(zb - za),
                m.u_nm,
                f"Δz = {_fmt_height(zb - za, m.u_nm, scale, label_unit)}",
            )
        )
        return out
    rows = [r for r in m.per_edge if r.get("height_nm") is not None and r.get("x_um") is not None]
    single = m.kind == "edge"
    for r in rows[:MAX_EDGE_DIMENSIONS]:
        x = float(r["x_um"])
        pair = _planes_around(m.planes, x)
        if pair is None:
            continue
        a, b = pair
        za, zb = float(a.line.at(x)), float(b.line.at(x))
        u = r.get("u_nm")
        text = (
            f"h = {_fmt_height(zb - za, u, scale, label_unit)}"
            if single
            else f"{abs(zb - za) * scale:.4g}"
        )
        out.append(Dimension("height", x, za, x, zb, abs(zb - za), u, text, offset=True))
    return out


def plane_residuals(m: StepMeasurement, profile: Profile) -> list[Residuals]:
    """Residual of every plane fit, separated into used and excluded points."""
    x, z, valid = profile.x_um, profile.z_nm, profile.valid
    out: list[Residuals] = []
    for p in m.planes:
        rng = (x >= p.x_start_um) & (x <= p.x_stop_um) & valid
        offered = rng if p.mask is None else (np.asarray(p.mask, dtype=bool) & valid)
        used = offered if p.used is None else (np.asarray(p.used, dtype=bool) & offered)
        excl = offered & ~used
        r = z - p.line.at(x)
        out.append(
            Residuals(
                name=p.name,
                x_um=x[used],
                r_nm=r[used],
                x_excluded_um=x[excl],
                r_excluded_nm=r[excl],
                sigma_nm=float(p.line.sigma),
            )
        )
    return out


def parabola_residuals(x_um: np.ndarray, z_nm: np.ndarray, fit: object) -> np.ndarray:
    """Residual of a stress curvature fit (``coefs`` = a, b, c about ``x0_um``)."""
    a, b, c = fit.coefs  # type: ignore[attr-defined]
    t = np.asarray(x_um, dtype=float) - fit.x0_um  # type: ignore[attr-defined]
    return np.asarray(z_nm, dtype=float) - (a * t**2 + b * t + c)


__all__ = [
    "Dimension",
    "fmt_uncertainty",
    "Residuals",
    "parabola_residuals",
    "plane_residuals",
    "step_dimensions",
]

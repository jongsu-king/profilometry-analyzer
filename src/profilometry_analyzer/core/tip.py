"""Stylus geometry: a cone with a rounded tip, and what such a stylus can and cannot see.

A profilometer stylus is a cone (full angle ``cone_angle_deg``, typically 60° or 90°)
that ends in a sphere of radius ``radius_um`` touching the cone tangentially.  At the
lateral distance ``u`` from the apex its surface lies ``d(u)`` above the apex::

    d(u) = r - sqrt(r^2 - u^2)                          |u| <= r cos(a)
    d(u) = r (1 - sin(a)) + (|u| - r cos(a)) / tan(a)   beyond   (a = half angle)

so the cone only matters for features deeper than ``r (1 - sin a)`` (1 µm for a 2 µm,
60° stylus).  Without a cone (angle 0) the sphere alone is used, as in the first
version of this program.

The recorded profile is the apex height ``t(x) = max_u [s(x + u) - d(u)]`` (a grey-scale
dilation of the surface ``s``).  The erosion ``r(x) = min_u [t(x + u) + d(u)]`` is the
highest surface that gives the same record: ``s <= r <= t``, and ``r = s`` wherever the
stylus touched the surface.  A point is *reached* when, for some stylus position, it is
the only place where the stylus touches ``r`` (within the noise) - the certainty map of
J. S. Villarrubia, J. Res. NIST 102, 425 (1997), written here from the description in
that paper.  Points that are never reached lie in valleys narrower than the stylus, at
the foot of walls steeper than its flank, or on the floor of a narrow groove: there the
surface is somewhere below ``r`` and nothing in the scan says where.

The morphology works on point indices and assumes the near-uniform spacing of a stylus
scan (the median spacing is used).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .stats import runs


@dataclass(frozen=True)
class Tip:
    """Stylus shape: sphere of ``radius_um`` on a cone of full angle ``cone_angle_deg``
    (0: sphere only; both 0: an ideal point)."""

    radius_um: float
    cone_angle_deg: float = 60.0

    def __post_init__(self) -> None:
        if self.radius_um < 0:
            raise ValueError("the stylus radius cannot be negative")
        if not 0 <= self.cone_angle_deg < 180:
            raise ValueError("the cone angle must be between 0 (no cone) and 180 degrees")

    @property
    def has_cone(self) -> bool:
        return self.cone_angle_deg > 0

    @property
    def _half(self) -> float:
        return math.radians(self.cone_angle_deg / 2.0)

    @property
    def tangent_um(self) -> float:
        """Lateral distance from the apex where the sphere meets the cone."""
        return self.radius_um * math.cos(self._half) if self.has_cone else self.radius_um

    @property
    def tangent_depth_nm(self) -> float:
        """Depth below the apex where the sphere meets the cone; deeper features see
        the cone."""
        r = self.radius_um
        return 1e3 * r * (1.0 - math.sin(self._half)) if self.has_cone else 1e3 * r

    @property
    def flank_slope(self) -> float:
        """Steepest wall (nm per µm) the flank can follow; infinite without a cone."""
        return 1e3 / math.tan(self._half) if self.has_cone else math.inf

    def drop_nm(self, u_um: np.ndarray | float) -> np.ndarray:
        """Height of the stylus surface above its apex at lateral offset ``u`` (inf where
        a sphere without a cone has no surface)."""
        a = np.abs(np.asarray(u_um, dtype=float))
        r = self.radius_um
        with np.errstate(invalid="ignore"):
            sphere = 1e3 * (r - np.sqrt(np.maximum(r * r - a * a, 0.0)))
        if not self.has_cone:
            return np.where(a <= r, sphere, np.inf)
        ut = self.tangent_um
        cone = self.tangent_depth_nm + (a - ut) * self.flank_slope
        return np.where(a <= ut, sphere, cone)

    def half_width_um(self, depth_nm: float) -> float:
        """Half the width of the stylus at ``depth_nm`` above its apex: a groove narrower
        than twice this cannot be entered that deep."""
        d = max(float(depth_nm), 0.0) / 1e3
        r = self.radius_um
        if d * 1e3 <= self.tangent_depth_nm:
            return math.sqrt(max(2.0 * r * d - d * d, 0.0))
        if not self.has_cone:
            return r
        return self.tangent_um + (d * 1e3 - self.tangent_depth_nm) / self.flank_slope

    def reaches_floor(self, width_um: float, depth_nm: float) -> bool:
        """Whether the apex touches the floor of a rectangular groove ``width_um`` wide
        and ``depth_nm`` deep (the stylus centred in the groove)."""
        return width_um >= 2.0 * self.half_width_um(depth_nm)

    def describe(self) -> str:
        if not self.has_cone:
            return f"{self.radius_um:g} µm sphere"
        return f"{self.radius_um:g} µm radius, {self.cone_angle_deg:g}° cone"

    def to_dict(self) -> dict[str, Any]:
        return {"radius_um": self.radius_um, "cone_angle_deg": self.cone_angle_deg}


def _spacing(x: np.ndarray) -> float:
    return float(np.median(np.diff(x))) if x.size > 1 else 1.0


def _kernel(dx: float, tip: Tip, span_nm: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Index offsets and stylus heights that can matter for a height range ``span_nm``."""
    reach = tip.half_width_um(span_nm)
    kmax = min(int(math.ceil(reach / dx)) + 1, max(n - 1, 0))
    k = np.arange(-kmax, kmax + 1)
    d = tip.drop_nm(k * dx)
    keep = np.isfinite(d) & (d <= span_nm + 1e-9)
    keep[kmax] = True
    return k[keep], d[keep]


def _shifted(z: np.ndarray, k: int, fill: float) -> np.ndarray:
    """``z[i + k]`` for every ``i``, ``fill`` outside the array."""
    out = np.full(z.size, fill)
    if k >= 0:
        out[: z.size - k] = z[k:]
    else:
        out[-k:] = z[: z.size + k]
    return out


def dilate(
    x: np.ndarray, z_nm: np.ndarray, tip: Tip, valid: np.ndarray | None = None
) -> np.ndarray:
    """What the stylus records over the surface ``z_nm`` (apex heights)."""
    z = np.where(valid, z_nm, -np.inf) if valid is not None else np.asarray(z_nm, float)
    fin = z[np.isfinite(z)]
    if fin.size == 0:
        return z.copy()
    k, d = _kernel(_spacing(x), tip, float(np.ptp(fin)), z.size)
    out = np.full(z.size, -np.inf)
    for kk, dd in zip(k.tolist(), d.tolist()):
        np.maximum(out, _shifted(z, kk, -np.inf) - dd, out=out)
    return out


def erode(x: np.ndarray, t_nm: np.ndarray, tip: Tip, valid: np.ndarray | None = None) -> np.ndarray:
    """The highest surface that the stylus would record as ``t_nm``."""
    t = np.where(valid, t_nm, np.inf) if valid is not None else np.asarray(t_nm, float)
    fin = t[np.isfinite(t)]
    if fin.size == 0:
        return t.copy()
    k, d = _kernel(_spacing(x), tip, float(np.ptp(fin)), t.size)
    out = np.full(t.size, np.inf)
    for kk, dd in zip(k.tolist(), d.tolist()):
        np.minimum(out, _shifted(t, kk, np.inf) + dd, out=out)
    return out


def reached_points(
    x: np.ndarray, r_nm: np.ndarray, tip: Tip, tol_nm: float, valid: np.ndarray | None = None
) -> np.ndarray:
    """Points of the reconstruction ``r_nm`` that the stylus touched (certainty map).

    For every stylus position the contact set is where the stylus surface comes within
    ``tol_nm`` of ``r``.  When that set is compact - no wider than a sphere resting on a
    flat surface within the tolerance, or a single point where the straight cone flank
    is involved - its points are reached; when it is spread out (the stylus hangs on two
    corners, or lies along a surface it shaped itself) the position certifies nothing.
    """
    n = r_nm.size
    ok = np.isfinite(r_nm) if valid is None else (valid & np.isfinite(r_nm))
    if n == 0 or not ok.any():
        return np.zeros(n, dtype=bool)
    dx = _spacing(x)
    r = np.where(ok, r_nm, -np.inf)
    k, d = _kernel(dx, tip, float(np.ptp(r[ok])) + tol_nm, n)
    top = np.full(n, -np.inf)
    for kk, dd in zip(k.tolist(), d.tolist()):
        np.maximum(top, _shifted(r, kk, -np.inf) - dd, out=top)
    lo = np.full(n, np.iinfo(np.int64).max)
    hi = np.full(n, np.iinfo(np.int64).min)
    for kk, dd in zip(k.tolist(), d.tolist()):
        near = (_shifted(r, kk, -np.inf) - dd) >= top - tol_nm
        lo = np.where(near & (kk < lo), kk, lo)
        hi = np.where(near & (kk > hi), kk, hi)
    flat = 2.0 * math.sqrt(2.0 * tip.radius_um * max(tol_nm, 0.0) / 1e3) if tip.radius_um else 0.0
    max_span = int(math.ceil(flat / dx)) + 2
    span = hi - lo
    # the flank is straight: lying along a flank that it shaped itself, the stylus touches
    # a whole stretch, and only a single point (a corner) is a contact there
    on_flank = np.maximum(np.abs(lo), np.abs(hi)) * dx > tip.tangent_um if tip.has_cone else False
    allowed = np.where(on_flank, 1, max_span)
    good = np.isfinite(top) & (span >= 0) & (span <= allowed)
    idx = np.arange(n)
    out = np.zeros(n, dtype=bool)
    for j in range(max_span + 1):
        sel = good & (j <= span)
        tgt = idx[sel] + lo[sel] + j
        tgt = tgt[(tgt >= 0) & (tgt < n)]
        out[tgt] = True
    return out & ok


def noise_sigma(z_nm: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Point-to-point noise from the second differences (robust)."""
    z = z_nm[valid] if valid is not None else z_nm
    if z.size < 5:
        return 0.0
    d2 = np.diff(z, 2)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / math.sqrt(6.0))


@dataclass
class TipCheck:
    """Where the stylus touched the surface, and the highest surface consistent with the
    record."""

    tip: Tip
    reconstruction_nm: np.ndarray
    reached: np.ndarray
    unreached_um: list[tuple[float, float]] = field(default_factory=list)
    tol_nm: float = 0.0
    min_run_points: int = 3
    floors: list[bool] = field(default_factory=list)
    """Per region: the record rises on both sides (the floor of a dip or valley the
    stylus did not reach), not on one side only (the foot of a wall)."""
    depth_nm: list[float] = field(default_factory=list)
    """Per region: how far the lowest recorded point lies below the lower of the two
    sides (a lower bound of the true depth there)."""

    @property
    def unreached(self) -> np.ndarray:
        return ~self.reached

    def unreached_fraction(self, mask: np.ndarray | None = None) -> float:
        """Share of the points (of ``mask``) inside an unreached region."""
        m = self.region_mask() if mask is None else self.region_mask() & mask
        base = int(mask.sum()) if mask is not None else self.reached.size
        return float(m.sum()) / base if base else 0.0

    def region_mask(self) -> np.ndarray:
        """Points inside the reported unreached regions (short noise gaps removed)."""
        return self._mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "stylus": self.tip.to_dict(),
            "tolerance_nm": self.tol_nm,
            "unreached_um": [list(r) for r in self.unreached_um],
            "floor": list(self.floors),
            "recorded_depth_nm": list(self.depth_nm),
            "unreached_percent": 100.0 * self.unreached_fraction(),
            "unreached_length_um": float(sum(b - a for a, b in self.unreached_um)),
        }

    _mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), repr=False)


def check_tip(
    x: np.ndarray,
    z_nm: np.ndarray,
    tip: Tip,
    valid: np.ndarray | None = None,
    *,
    tol_nm: float | None = None,
    min_run_points: int = 3,
) -> TipCheck:
    """Reconstruction and reached points of a recorded profile.

    ``tol_nm`` defaults to three times the point-to-point noise; unreached runs shorter
    than ``min_run_points`` are noise (a low reading between two higher ones) and are not
    reported.  Two hidden runs are joined when the points between them lie no higher than
    the hidden points on either side and are fewer than a sphere's flat contact within
    the tolerance, or than a fifth of the shorter run: with noise, a few points at the
    bottom of a V traced in a narrow trench pass the test by chance (and the despiker may
    have removed the tip of the V).
    """
    ok = np.isfinite(z_nm) if valid is None else (valid & np.isfinite(z_nm))
    tol = 3.0 * noise_sigma(z_nm, ok) if tol_nm is None else float(tol_nm)
    rec = erode(x, z_nm, tip, ok)
    rec = np.where(ok, rec, np.nan)
    reached = reached_points(x, np.where(ok, rec, np.nan), tip, tol, ok)
    hidden = ok & ~reached
    # a few reached points inside a hidden region are ties, not contacts: no gap narrower
    # than the contact of a sphere on a flat surface within the tolerance is kept
    dx = _spacing(x)
    flat = 2.0 * math.sqrt(2.0 * tip.radius_um * tol / 1e3) if tip.radius_um else 0.0
    gap = max(min_run_points, int(math.ceil(flat / dx)) + 2)
    hid = runs(hidden)
    for (a0, b0), (a1, b1) in zip(hid[:-1], hid[1:]):
        if a1 - b0 > max(gap, 0.2 * min(b0 - a0, b1 - a1)):
            continue
        # only a dip between the two hidden runs is a tie (the bottom of a V traced in a
        # narrow trench); a peak there is where the stylus rested between two valleys
        inside = rec[b0:a1][np.isfinite(rec[b0:a1])]
        if inside.size == 0 or float(inside.max()) <= max(rec[b0 - 1], rec[a1]):
            hidden[b0:a1] = True
    mask = np.zeros(x.size, dtype=bool)
    regions: list[tuple[float, float]] = []
    floors: list[bool] = []
    depths: list[float] = []
    zr = np.where(ok, z_nm, np.nan)
    for a, b in runs(hidden):
        if b - a < min_run_points:
            continue
        mask[a:b] = True
        regions.append((float(x[a]), float(x[b - 1])))
        inside = zr[a:b][np.isfinite(zr[a:b])]
        left = _edge_value(zr, a - 1, -1)
        right = _edge_value(zr, b, 1)
        low = float(inside.min()) if inside.size else np.nan
        rise = min(left, right) - low
        floors.append(bool(np.isfinite(rise) and rise > max(5.0 * tol, 1.0)))
        depths.append(float(rise) if np.isfinite(rise) else 0.0)
    return TipCheck(
        tip=tip,
        reconstruction_nm=rec,
        reached=reached | ~ok,
        unreached_um=regions,
        tol_nm=tol,
        min_run_points=min_run_points,
        floors=floors,
        depth_nm=depths,
        _mask=mask,
    )


def _edge_value(z: np.ndarray, i: int, step: int) -> float:
    """First finite value from index ``i`` outwards (nan when there is none)."""
    while 0 <= i < z.size:
        if np.isfinite(z[i]):
            return float(z[i])
        i += step
    return float("nan")

"""Tilt (levelling) and bow correction.

* :func:`level_regions` - the manual method, like the two reference cursors of the
  instrument software: a straight line is fitted to the points of two (or more)
  reference regions and subtracted.  With ``same_level=True`` (default) the regions are
  assumed to lie on one plane; with ``same_level=False`` they share the slope but each
  has its own offset (one region on the film, one on the substrate).
* :func:`level_auto` - edges are found from the smoothed derivative and masked; a trend
  shared by all plateaus, with one offset per plateau, is fitted robustly (Huber or
  RANSAC) so that steps never tilt the line.
* Bow (``order >= 2``) is the same shared fit with a polynomial trend.  It is a separate
  step with its own history entry and notice, so a step height is never silently
  computed on a bow-corrected profile.

After levelling the lowest plateau (or the reference regions) is at zero height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import notices as N
from ..notices import Notice
from ..profile import Profile
from .edges import EdgeSet, detect_edges
from .stats import fit_line, fit_shared


@dataclass
class LevelResult:
    method: str
    order: int
    coefs: list[float]
    """Trend ``sum_k c_k ((x - x0)/scale)^k`` (k = 1..order), nm."""
    x0_um: float
    scale_um: float
    zero_nm: float
    """Height subtracted after the trend so that the reference level is zero."""
    sigma_nm: float
    regions_um: list[tuple[float, float]] = field(default_factory=list)
    offsets_nm: dict[int, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    notices: list[Notice] = field(default_factory=list)

    @property
    def slope_nm_per_um(self) -> float:
        return float(self.coefs[0] / self.scale_um) if self.coefs else 0.0

    def trend(self, x: np.ndarray) -> np.ndarray:
        t = (np.asarray(x, dtype=float) - self.x0_um) / self.scale_um
        out = np.zeros_like(t)
        for k, c in enumerate(self.coefs, start=1):
            out += c * t**k
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "order": self.order,
            "coefs_nm": list(self.coefs),
            "x0_um": self.x0_um,
            "scale_um": self.scale_um,
            "slope_nm_per_um": self.slope_nm_per_um,
            "slope_mrad": self.slope_nm_per_um,  # 1 nm/µm = 1 mrad
            "zero_nm": self.zero_nm,
            "sigma_nm": self.sigma_nm,
            "regions_um": [list(r) for r in self.regions_um],
            "offsets_nm": {str(k): v for k, v in self.offsets_nm.items()},
            "info": dict(self.info),
        }


def _region_mask(x: np.ndarray, region: tuple[float, float]) -> np.ndarray:
    a, b = sorted((float(region[0]), float(region[1])))
    return (x >= a) & (x <= b)


def level_regions(
    profile: Profile,
    regions: list[tuple[float, float]],
    *,
    same_level: bool = True,
    order: int = 1,
) -> tuple[Profile, LevelResult]:
    """Subtract a line (or polynomial) fitted to the reference regions."""
    if len(regions) < 1:
        raise ValueError("at least one reference region is needed")
    x, z, valid = profile.x_um, profile.z_nm, profile.valid
    notes: list[Notice] = []
    masks = [_region_mask(x, r) & valid for r in regions]
    for r, m in zip(regions, masks):
        if m.sum() < 3:
            raise ValueError(
                f"reference region {r[0]:g}-{r[1]:g} µm holds {int(m.sum())} valid points"
            )
    if len(regions) == 1 and not same_level:
        same_level = True
    sel = np.zeros(x.size, dtype=bool)
    labels = np.full(x.size, -1, dtype=int)
    for i, m in enumerate(masks):
        sel |= m
        labels[m & (labels < 0)] = i
    lab = np.zeros(int(sel.sum()), dtype=int) if same_level else labels[sel]
    fit = fit_shared(x[sel], z[sel], lab, order=order, method="lsq")
    trend = fit.trend(x)
    zero = float(min(fit.offsets.values())) if not same_level else float(fit.offsets[0])
    z_new = z - trend - zero
    if same_level and len(regions) >= 2:
        # the same regions with one offset each: do they really share a level?
        sep = fit_shared(x[sel], z[sel], labels[sel], order=order, method="lsq")
        offs = list(sep.offsets.values())
        spread = float(max(offs) - min(offs))
        resid = float(sep.sigma) if np.isfinite(sep.sigma) else 0.0
        if spread > 3 * max(resid, 1e-12):
            notes.append(
                Notice(
                    N.LEVELING_REGIONS_DIFFER,
                    N.WARNING,
                    f"the reference regions are not on one level (offsets differ by {spread:.4g}"
                    f" nm, residual {resid:.3g} nm): a line through both tilts the profile. If"
                    " one lies on the film and one on the substrate, level with separate levels",
                    {"spread_nm": spread, "residual_nm": resid},
                )
            )
    res = LevelResult(
        method="regions" + ("" if same_level else "-separate"),
        order=order,
        coefs=[float(c) for c in fit.coefs],
        x0_um=fit.x0,
        scale_um=fit.scale,
        zero_nm=zero,
        sigma_nm=float(fit.sigma),
        regions_um=[(float(min(r)), float(max(r))) for r in regions],
        offsets_nm={k: float(v) for k, v in fit.offsets.items()},
        notices=notes,
    )
    step = {"step": "level", **res.to_dict()}
    return profile.derive(z_new, step=step, notices=notes), res


def level_auto(
    profile: Profile,
    *,
    method: str = "huber",
    order: int = 1,
    edges: EdgeSet | None = None,
    seed: int = 0,
    **edge_options: Any,
) -> tuple[Profile, LevelResult, EdgeSet]:
    """Robust trend shared by all plateaus (edges masked); see the module docstring."""
    x, z, valid = profile.x_um, profile.z_nm, profile.valid
    edge_options.setdefault("tip_radius_um", profile.scan.tip_radius_um or 2.0)
    eset = edges or detect_edges(x, z, valid, **edge_options)
    labels = eset.labels()
    use = labels >= 0
    notes: list[Notice] = []
    if use.sum() < order + 3:
        notes.append(
            Notice(
                N.LEVELING_REGION_EMPTY,
                N.WARNING,
                "too few plateau points for an automatic levelling; the whole profile was used",
            )
        )
        use = valid.copy()
        labels = np.where(use, 0, -1)
    fit = fit_shared(x[use], z[use], labels[use], order=order, method=method, seed=seed)
    trend = fit.trend(x)
    zero = float(min(fit.offsets.values()))
    z_new = z - trend - zero
    info = {"method_used": fit.method, "points_used": int(use.sum()), **fit.info}
    if order >= 2:
        lin = fit_shared(x[use], z[use], labels[use], order=1, method=method, seed=seed)
        bow = fit.trend(x) - lin.trend(x)
        bow -= float(np.mean(bow))
        pv = float(np.ptp(bow))
        info["bow_peak_to_valley_nm"] = pv
        notes.append(
            Notice(
                N.BOW_REMOVED,
                N.INFO,
                f"a polynomial bow of order {order} was removed (peak-to-valley"
                f" {pv:.4g} nm beyond the straight line); step heights below are computed on the"
                " bow-corrected profile",
                {"order": order, "peak_to_valley_nm": pv},
            )
        )
    res = LevelResult(
        method=f"auto-{fit.method}",
        order=order,
        coefs=[float(c) for c in fit.coefs],
        x0_um=fit.x0,
        scale_um=fit.scale,
        zero_nm=zero,
        sigma_nm=float(fit.sigma),
        offsets_nm={k: float(v) for k, v in fit.offsets.items()},
        info=info,
        notices=notes,
    )
    step = {"step": "level", **res.to_dict()}
    return profile.derive(z_new, step=step, notices=notes), res, eset


def level_none(profile: Profile) -> tuple[Profile, LevelResult]:
    res = LevelResult("none", 0, [], float(np.mean(profile.x_um)), 1.0, 0.0, float("nan"))
    return profile.derive(step={"step": "level", "method": "none"}), res


def line_through(profile: Profile, region: tuple[float, float]) -> dict[str, Any]:
    """Least-squares line of one region (used by the GUI read-out)."""
    m = _region_mask(profile.x_um, region) & profile.valid
    if m.sum() < 2:
        raise ValueError("region holds fewer than two points")
    return fit_line(profile.x_um[m], profile.z_nm[m]).to_dict()

"""Step height and film thickness.

Methods
-------
``planes``
    A single film edge (and every edge of a staircase or multi-step profile): a line is
    fitted to the usable part of the plateau on each side (transition and margins
    removed, raised points beyond ``k`` sigma excluded) and the height difference is
    taken at the edge centre.
``iso5436-a1``
    A groove (scratch) or a ridge (stripe of film), evaluated the way ISO 5436-1 defines
    a type A1 depth standard: with the groove width ``W`` (distance between the edge
    centres) and centre ``c``, the upper level is one line over ``c ± 1.5 W`` with the
    groove and ``W/3`` next to each groove edge ignored, the lower level is the central
    third of the groove (``c ± W/6``), and the depth is the distance from the upper line
    to the centre of the lower line.  ISO 5436-1 notes that wide ridges with flat tops
    are equivalent; a ridge is evaluated with the roles of the levels exchanged.
``periodic``
    Repeated lines: every edge is measured with ``planes``; the mean, standard deviation
    and number of edges are reported, together with the period.
``regions``
    Two reference regions given by the user (GUI cursors or ``--regions``).

Uncertainty
-----------
``u_fit`` combines the covariances of the two lines at the evaluation point; the
covariances are inflated for correlated residuals (roughness, smoothing) through the
effective number of points.  ``u_slope = |m_2 - m_1| * gap / 2`` is the change of the
height difference across the excluded gap between the two fitted regions when the
planes are not parallel.  ``u = sqrt(u_fit^2 + u_slope^2)`` is a standard uncertainty
(k = 1) of this one scan; it does not contain the instrument calibration or the
scatter between positions on the sample (use several scans and the batch table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import notices as N
from ..instrument import DEFAULT_INSTRUMENT, InstrumentSpec
from ..notices import Notice
from .edges import Edge, EdgeSet
from .stats import LineFit, fit_line, mad_sigma, runs


@dataclass
class PlaneFit:
    name: str
    x_start_um: float
    x_stop_um: float
    line: LineFit
    n_used: int
    n_excluded: int
    burrs: list[dict[str, Any]] = field(default_factory=list)
    used: np.ndarray | None = None

    @property
    def slope(self) -> float:
        return self.line.slope

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x_start_um": self.x_start_um,
            "x_stop_um": self.x_stop_um,
            "n_used": self.n_used,
            "n_excluded": self.n_excluded,
            "burrs": list(self.burrs),
            **self.line.to_dict(),
        }


def fit_plane(
    x: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
    k: float = 4.0,
    max_iter: int = 20,
    min_burr_points: int = 3,
) -> PlaneFit | None:
    """Line through the points of ``mask`` with points beyond ``k`` sigma excluded.

    Raised runs (residual above ``+k`` sigma) are grown outwards until the residual
    falls below one sigma, so the whole burr - not only its top - is excluded, and they
    are returned in ``burrs``.  ``None`` when fewer than three points are available.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 3:
        return None
    inl = mask.copy()
    line = fit_line(x[inl], z[inl])
    s = 0.0
    r = z - line.at(x)
    for _ in range(max_iter):
        r = z - line.at(x)
        s = mad_sigma(r[inl])
        if not s > 0:
            break
        out = mask & (np.abs(r) > k * s)
        grow = mask & (r > 1.0 * s)
        raised = np.zeros_like(out)
        for a, b in runs(grow):
            if np.any(out[a:b] & (r[a:b] > 0)):
                raised[a:b] = True
        out |= raised
        new = mask & ~out
        if new.sum() < 3 or np.array_equal(new, inl):
            break
        inl = new
        line = fit_line(x[inl], z[inl])
    r = z - line.at(x)
    burrs: list[dict[str, Any]] = []
    if s > 0:
        raised = mask & ~inl & (r > 0)
        for a, b in runs(raised):
            if b - a < min_burr_points:
                continue  # a single noisy point is not a burr (it is still excluded)
            j = a + int(np.argmax(r[a:b]))
            burrs.append(
                {
                    "x_start_um": float(x[a]),
                    "x_stop_um": float(x[b - 1]),
                    "x_peak_um": float(x[j]),
                    "height_nm": float(r[j]),
                    "points": int(b - a),
                }
            )
    xs = x[mask]
    return PlaneFit(
        name=name,
        x_start_um=float(xs.min()),
        x_stop_um=float(xs.max()),
        line=line,
        n_used=int(inl.sum()),
        n_excluded=int(mask.sum() - inl.sum()),
        burrs=burrs,
        used=inl,
    )


def _difference(a: PlaneFit, b: PlaneFit, x_at: float, gap_um: float) -> dict[str, float]:
    """``b - a`` at ``x_at`` with its uncertainty components."""
    h = float(b.line.at(x_at) - a.line.at(x_at))
    u_fit = float(np.sqrt(a.line.var_at(x_at) + b.line.var_at(x_at)))
    dm = float(b.slope - a.slope)
    u_dm = float(np.sqrt(max(a.line.cov[0, 0], 0.0) + max(b.line.cov[0, 0], 0.0)))
    u_slope = abs(dm) * max(gap_um, 0.0) / 2.0
    return {
        "height_nm": h,
        "u_nm": float(np.hypot(u_fit, u_slope)),
        "u_fit_nm": u_fit,
        "u_slope_nm": float(u_slope),
        "slope_difference_nm_per_um": dm,
        "slope_difference_std": u_dm,
        "x_um": float(x_at),
        "gap_um": float(gap_um),
        "means_difference_nm": float(
            np.mean(b.line.at(np.linspace(b.x_start_um, b.x_stop_um, 3)))
            - np.mean(a.line.at(np.linspace(a.x_start_um, a.x_stop_um, 3)))
        ),
    }


@dataclass
class StepMeasurement:
    kind: str
    method: str
    value_nm: float | None
    """Signed: edge = right minus left; groove = depth (> 0); ridge = height (> 0)."""
    u_nm: float | None
    u_fit_nm: float | None = None
    u_slope_nm: float | None = None
    position_um: float | None = None
    width_um: float | None = None
    planes: list[PlaneFit] = field(default_factory=list)
    per_edge: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    regions_um: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    notices: list[Notice] = field(default_factory=list)

    @property
    def thickness_nm(self) -> float | None:
        return None if self.value_nm is None else abs(float(self.value_nm))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "method": self.method,
            "value_nm": self.value_nm,
            "thickness_nm": self.thickness_nm,
            "u_nm": self.u_nm,
            "u_fit_nm": self.u_fit_nm,
            "u_slope_nm": self.u_slope_nm,
            "position_um": self.position_um,
            "width_um": self.width_um,
            "planes": [p.to_dict() for p in self.planes],
            "per_edge": list(self.per_edge),
            "stats": dict(self.stats),
            "regions_um": {k: [list(r) for r in v] for k, v in self.regions_um.items()},
            "notices": [n.to_dict() for n in self.notices],
        }


# ------------------------------------------------------------------ per-edge planes
def refine_edges(x: np.ndarray, z: np.ndarray, eset: EdgeSet, *, k: float = 4.0) -> None:
    """Move every edge centre to the half-height crossing between its two plateaus.

    The derivative centroid of a transition is pulled sideways by a burr or an
    asymmetric wall; the point where the profile crosses the middle between the two
    fitted plateau lines is what a groove width or an edge position means.  Edges whose
    plateaus cannot be fitted keep their centroid.
    """
    planes = [fit_plane(x, z, eset.usable(s), name="", k=k) for s in eset.segments]
    for i, e in enumerate(eset.edges):
        a, b = planes[i], planes[i + 1]
        if a is None or b is None or e.manual:
            continue
        lo = int(np.searchsorted(x, a.x_stop_um))
        hi = int(np.searchsorted(x, b.x_start_um, side="right"))
        if hi - lo < 2:
            continue
        xs, zs = x[lo:hi], z[lo:hi]
        mid = 0.5 * (a.line.at(xs) + b.line.at(xs))
        r = (zs - mid) * (1 if b.line.at(e.x_um) > a.line.at(e.x_um) else -1)
        cross = np.flatnonzero((r[:-1] < 0) & (r[1:] >= 0))
        if cross.size == 0:
            continue
        j = cross[int(np.argmin(np.abs(xs[cross] - e.x_um)))]
        t = r[j] / (r[j] - r[j + 1]) if r[j] != r[j + 1] else 0.5
        e.x_centroid_um = e.x_um
        e.x_um = float(xs[j] + t * (xs[j + 1] - xs[j]))


def measure_edges(
    x: np.ndarray, z: np.ndarray, eset: EdgeSet, *, k: float = 4.0
) -> tuple[list[PlaneFit | None], list[dict[str, Any]]]:
    """A plane per plateau and the height of every edge."""
    planes: list[PlaneFit | None] = []
    for seg in eset.segments:
        planes.append(fit_plane(x, z, eset.usable(seg), name=f"plateau {seg.label + 1}", k=k))
    per_edge: list[dict[str, Any]] = []
    for i, e in enumerate(eset.edges):
        a, b = planes[i], planes[i + 1]
        row: dict[str, Any] = {"edge": i + 1, **e.to_dict()}
        if a is None or b is None:
            row.update({"height_nm": None, "u_nm": None, "note": "plateau too short"})
        else:
            gap = b.x_start_um - a.x_stop_um
            row.update(_difference(a, b, e.x_um, gap))
            row["direction"] = "up" if row["height_nm"] > 0 else "down"
        per_edge.append(row)
    return planes, per_edge


def _burr_notices(planes: list[PlaneFit], k: float) -> list[Notice]:
    out: list[Notice] = []
    for p in planes:
        if not p.burrs:
            continue
        top = max(p.burrs, key=lambda b: b["height_nm"])
        out.append(
            Notice(
                N.BURR_EXCLUDED,
                N.WARNING,
                f"{len(p.burrs)} raised run(s) in the {p.name} plane exceed {k:g} sigma and were"
                f" excluded (highest {top['height_nm']:.4g} nm at {top['x_peak_um']:.4g} µm)",
                {"plane": p.name, "burrs": p.burrs[:10]},
            )
        )
    return out


def _correlation_notices(planes: list[PlaneFit], limit: float = 30.0) -> list[Notice]:
    """Warn when a plane has few independent points (long-range correlated roughness).

    In synthetic tests the single-scan uncertainty is well calibrated for white noise but
    too small by a factor 1.2-2.3 when the roughness is correlated over a few percent of
    the plane length, because the autocorrelation estimated from the residuals of a short
    plane misses its long tail.
    """
    weak = [p for p in planes if p.line.n_eff < limit]
    if not weak:
        return []
    worst = min(weak, key=lambda p: p.line.n_eff)
    corr = (worst.x_stop_um - worst.x_start_um) / max(worst.line.n_eff, 1.0)
    return [
        Notice(
            N.CORRELATED_RESIDUALS,
            N.WARNING,
            f"the {worst.name} plane has only about {worst.line.n_eff:.0f} independent points"
            f" (residuals correlated over ~{corr:.3g} µm): the single-scan uncertainty may be"
            " too small - use the scatter of several scans",
            {"plane": worst.name, "n_eff": worst.line.n_eff},
        )
    ]


def _nonparallel(
    diff: dict[str, float], what: str, a: PlaneFit | None = None, b: PlaneFit | None = None
) -> list[Notice]:
    """Planes that diverge: significantly different slopes whose effect on the height
    exceeds the fit uncertainty, or lines that drift apart by more than three residual
    sigmas over the length of the planes (bow left in the profile)."""
    dm = diff["slope_difference_nm_per_um"]
    sd = diff["slope_difference_std"]
    significant = abs(dm) > 3 * max(sd, 1e-300) and diff["u_slope_nm"] > diff["u_fit_nm"]
    drift = False
    if a is not None and b is not None:
        length = 0.5 * ((a.x_stop_um - a.x_start_um) + (b.x_stop_um - b.x_start_um))
        drift = abs(dm) * length > 3 * max(a.line.sigma, b.line.sigma, 1e-12)
    if significant or drift:
        return [
            Notice(
                N.NONPARALLEL_PLANES,
                N.WARNING,
                f"the planes of the {what} are not parallel (slope difference {dm:.3g} nm/µm,"
                f" {abs(dm) / max(sd, 1e-300):.1f} sigma); the height depends on where it is"
                f" taken (±{diff['u_slope_nm']:.3g} nm over the gap). Check the levelling or"
                " remove the bow",
                {"slope_difference_nm_per_um": dm},
            )
        ]
    return []


def measure_single_edge(
    x: np.ndarray, z: np.ndarray, eset: EdgeSet, edge_index: int = 0, *, k: float = 4.0
) -> StepMeasurement:
    planes, per_edge = measure_edges(x, z, eset, k=k)
    e = eset.edges[edge_index]
    a, b = planes[edge_index], planes[edge_index + 1]
    row = per_edge[edge_index]
    if a is None or b is None:
        return StepMeasurement(
            "edge",
            "planes",
            None,
            None,
            position_um=e.x_um,
            per_edge=per_edge,
            notices=[Notice(N.SHORT_PLANE, N.ERROR, "one side of the edge has no usable plane")],
        )
    m = StepMeasurement(
        kind="edge",
        method="planes",
        value_nm=row["height_nm"],
        u_nm=row["u_nm"],
        u_fit_nm=row["u_fit_nm"],
        u_slope_nm=row["u_slope_nm"],
        position_um=e.x_um,
        width_um=e.width_um,
        planes=[a, b],
        per_edge=per_edge,
        regions_um={
            "plane": [(a.x_start_um, a.x_stop_um), (b.x_start_um, b.x_stop_um)],
            "transition": [(e.x_start_um, e.x_stop_um)],
        },
    )
    m.notices += _burr_notices([a, b], k) + _nonparallel(row, "edge", a, b)
    m.notices += _correlation_notices([a, b])
    # a burr on the high side next to the edge lies in the transition or the margin
    high = b if row["height_nm"] > 0 else a
    if high is b:
        zone = (x >= e.x_um) & (x < high.x_start_um)
    else:
        zone = (x > high.x_stop_um) & (x <= e.x_um)
    zone &= eset.valid
    if zone.any():
        r = z[zone] - high.line.at(x[zone])
        s = max(high.line.sigma, 1e-12)
        if float(r.max()) > k * s:
            j = int(np.argmax(r))
            m.notices.append(
                Notice(
                    N.BURR_EXCLUDED,
                    N.WARNING,
                    f"a raised rim (burr) {float(r.max()):.4g} nm above the {high.name} plane at"
                    f" {float(x[zone][j]):.4g} µm lies in the excluded zone next to the edge;"
                    " it does not enter the height",
                    {"height_nm": float(r.max()), "x_um": float(x[zone][j]), "zone": "rim"},
                )
            )
    return m


def _a1_masks(
    x: np.ndarray, e1: Edge, e2: Edge, usable: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    w = e2.x_um - e1.x_um
    c = 0.5 * (e1.x_um + e2.x_um)
    lo, hi = c - 1.5 * w, c + 1.5 * w
    left = (x >= lo) & (x <= e1.x_um - w / 3.0) & (x < e1.x_start_um - margin)
    right = (x >= e2.x_um + w / 3.0) & (x <= hi) & (x > e2.x_stop_um + margin)
    inner = (x >= c - w / 6.0) & (x <= c + w / 6.0)
    inner &= (x > e1.x_stop_um + margin) & (x < e2.x_start_um - margin)
    info = {
        "W_um": w,
        "centre_um": c,
        "outer_um": [(lo, e1.x_um - w / 3.0), (e2.x_um + w / 3.0, hi)],
        "inner_um": [(c - w / 6.0, c + w / 6.0)],
        "truncated": bool(x.min() > lo or x.max() < hi),
        "shortened_by_margins": bool(
            e1.x_start_um - margin < e1.x_um - w / 3.0
            or e2.x_stop_um + margin > e2.x_um + w / 3.0
            or e1.x_stop_um + margin > c - w / 6.0
            or e2.x_start_um - margin < c + w / 6.0
        ),
    }
    return (left | right) & usable, inner & usable, info


def stylus_reaches_bottom(width_um: float, depth_um: float, tip_radius_um: float) -> bool:
    """Whether a spherical tip of radius ``r`` touches the floor of a rectangular groove.

    Below the floor the sphere centre sits ``r`` above it; it clears the two top corners
    when ``(W/2)^2 >= 2 r d - d^2`` for ``d < r`` and when ``W >= 2 r`` otherwise.  (A
    real stylus is a cone with a rounded tip, so steep walls are traced worse still.)
    """
    r, d, half = tip_radius_um, abs(depth_um), width_um / 2.0
    if d >= r:
        return width_um >= 2 * r
    return half * half >= 2 * r * d - d * d


def measure_a1(
    x: np.ndarray,
    z: np.ndarray,
    eset: EdgeSet,
    pair: tuple[int, int],
    *,
    kind: str,
    k: float = 4.0,
    tip_radius_um: float = 2.0,
) -> StepMeasurement:
    """Groove depth or ridge height in the ISO 5436-1 type A1 manner."""
    i1, i2 = pair
    e1, e2 = eset.edges[i1], eset.edges[i2]
    usable = eset.valid & ~eset.exclude
    outer_m, inner_m, info = _a1_masks(x, e1, e2, usable, eset.margin_um)
    notes: list[Notice] = []
    _, per_edge = measure_edges(x, z, eset, k=k)
    w, c = info["W_um"], info["centre_um"]
    if info["truncated"]:
        notes.append(
            Notice(
                N.A1_REGION_TRUNCATED,
                N.WARNING,
                f"the scan does not cover the ISO 5436-1 A1 upper regions (c ± 1.5 W ="
                f" {c - 1.5 * w:.4g} to {c + 1.5 * w:.4g} µm); the available part was used",
                {"needed_um": [c - 1.5 * w, c + 1.5 * w]},
            )
        )
    if info["shortened_by_margins"]:
        notes.append(
            Notice(
                N.A1_REGION_TRUNCATED,
                N.INFO,
                "the A1 regions were shortened so that they stay clear of the detected edge"
                f" transitions plus {eset.margin_um:.3g} µm",
            )
        )
    outer = fit_plane(x, z, outer_m, name="upper" if kind == "groove" else "base", k=k)
    inner = fit_plane(x, z, inner_m, name="bottom" if kind == "groove" else "top", k=k)
    regions = {
        "outer": [tuple(r) for r in info["outer_um"]],
        "inner": [tuple(r) for r in info["inner_um"]],
        "transition": [(e1.x_start_um, e1.x_stop_um), (e2.x_start_um, e2.x_stop_um)],
    }
    if outer is None or inner is None:
        notes.append(
            Notice(
                N.SHORT_PLANE,
                N.ERROR,
                f"the {kind} is too narrow or too close to the scan end for the A1 regions"
                f" ({'outer' if outer is None else 'inner'} region has fewer than 3 points)",
            )
        )
        return StepMeasurement(
            kind,
            "iso5436-a1",
            None,
            None,
            position_um=c,
            width_um=w,
            per_edge=per_edge,
            regions_um=regions,
            notices=notes,
        )
    gap = w / 3.0  # the inner line is evaluated at its centre; it spans W/3
    diff = _difference(outer, inner, c, gap)
    value = -diff["height_nm"] if kind == "groove" else diff["height_nm"]
    # flatness of the floor (groove) or top (ridge) compared with the outer level
    if inner.line.sigma > 3 * max(outer.line.sigma, 1e-12) and inner.n_used >= 5:
        notes.append(
            Notice(
                N.A1_BOTTOM_NOT_FLAT,
                N.WARNING,
                f"the {'floor' if kind == 'groove' else 'top'} is rougher than the outer level"
                f" (residual {inner.line.sigma:.3g} vs {outer.line.sigma:.3g} nm): a"
                " scratch that ploughs into the substrate, or a film that tears, biases the"
                " depth",
                {"inner_sigma_nm": inner.line.sigma, "outer_sigma_nm": outer.line.sigma},
            )
        )
    notes += _burr_notices([outer, inner], k) + _nonparallel(diff, kind, outer, inner)
    notes += _correlation_notices([outer, inner])
    # burrs on the rims, which lie inside the ignored W/3 zones and margins
    rim = (
        ((x >= e1.x_um - w / 3.0) & (x <= e1.x_um)) | ((x >= e2.x_um) & (x <= e2.x_um + w / 3.0))
    ) & eset.valid
    if kind == "groove" and rim.any():
        r = z[rim] - outer.line.at(x[rim])
        s = max(outer.line.sigma, 1e-12)
        if float(r.max()) > k * s:
            j = int(np.argmax(r))
            notes.append(
                Notice(
                    N.BURR_EXCLUDED,
                    N.WARNING,
                    f"a raised rim (burr) {float(r.max()):.4g} nm above the upper level at"
                    f" {float(x[rim][j]):.4g} µm lies in the ignored zone next to the groove;"
                    " it does not enter the depth",
                    {"height_nm": float(r.max()), "x_um": float(x[rim][j]), "zone": "rim"},
                )
            )
    notes += _narrow_notices(kind, w, abs(value), tip_radius_um, inner)
    return StepMeasurement(
        kind=kind,
        method="iso5436-a1",
        value_nm=value,
        u_nm=diff["u_nm"],
        u_fit_nm=diff["u_fit_nm"],
        u_slope_nm=diff["u_slope_nm"],
        position_um=c,
        width_um=w,
        planes=[outer, inner],
        per_edge=per_edge,
        stats={"a1": {k2: v for k2, v in info.items() if k2 not in ("outer_um", "inner_um")}},
        regions_um=regions,
        notices=notes,
    )


def _narrow_notices(
    kind: str, width_um: float, height_nm: float, tip_radius_um: float, inner: PlaneFit
) -> list[Notice]:
    out: list[Notice] = []
    r = tip_radius_um
    if kind == "groove" and not stylus_reaches_bottom(width_um, height_nm / 1e3, r):
        out.append(
            Notice(
                N.NARROW_FEATURE,
                N.WARNING,
                f"a stylus of radius {r:g} µm cannot reach the floor of a groove {width_um:.3g} µm"
                f" wide and {height_nm:.4g} nm deep: depth and width are underestimated",
                {"width_um": width_um, "tip_radius_um": r},
            )
        )
    elif width_um < 3 * r:
        out.append(
            Notice(
                N.NARROW_FEATURE,
                N.WARNING,
                f"the {kind} is {width_um:.3g} µm wide, less than three stylus radii"
                f" ({3 * r:.3g} µm): its width and {'depth' if kind == 'groove' else 'height'}"
                " are shaped by the stylus",
                {"width_um": width_um, "tip_radius_um": r},
            )
        )
    return out


def measure_periodic(
    x: np.ndarray, z: np.ndarray, eset: EdgeSet, *, k: float = 4.0
) -> StepMeasurement:
    planes, per_edge = measure_edges(x, z, eset, k=k)
    h = np.array([abs(r["height_nm"]) for r in per_edge if r.get("height_nm") is not None])
    xs = np.array([e.x_um for e in eset.edges])
    up = xs[[e.sign > 0 for e in eset.edges]]
    down = xs[[e.sign < 0 for e in eset.edges]]
    periods = np.concatenate([np.diff(up), np.diff(down)])
    stats: dict[str, Any] = {
        "n": int(h.size),
        "mean_nm": float(np.mean(h)) if h.size else None,
        "std_nm": float(np.std(h, ddof=1)) if h.size > 1 else None,
        "sem_nm": float(np.std(h, ddof=1) / np.sqrt(h.size)) if h.size > 1 else None,
        "period_um": float(np.mean(periods)) if periods.size else None,
        "period_std_um": float(np.std(periods, ddof=1)) if periods.size > 1 else None,
    }
    good = [p for p in planes if p is not None]
    m = StepMeasurement(
        kind="periodic",
        method="periodic",
        value_nm=stats["mean_nm"],
        u_nm=stats["sem_nm"],
        position_um=float(np.mean(xs)) if xs.size else None,
        planes=good,
        per_edge=per_edge,
        stats=stats,
        regions_um={
            "plane": [(p.x_start_um, p.x_stop_um) for p in good],
            "transition": [(e.x_start_um, e.x_stop_um) for e in eset.edges],
        },
    )
    m.notices += _burr_notices(good, k)
    return m


def measure_multi(
    x: np.ndarray, z: np.ndarray, eset: EdgeSet, kind: str, *, k: float = 4.0
) -> StepMeasurement:
    planes, per_edge = measure_edges(x, z, eset, k=k)
    good = [p for p in planes if p is not None]
    total = sum(r["height_nm"] for r in per_edge if r.get("height_nm") is not None)
    m = StepMeasurement(
        kind=kind,
        method="planes",
        value_nm=None,
        u_nm=None,
        planes=good,
        per_edge=per_edge,
        stats={"n_edges": len(per_edge), "net_height_nm": float(total)},
        regions_um={
            "plane": [(p.x_start_um, p.x_stop_um) for p in good],
            "transition": [(e.x_start_um, e.x_stop_um) for e in eset.edges],
        },
    )
    m.notices += _burr_notices(good, k)
    m.notices.append(
        Notice(
            N.MULTIPLE_STEPS,
            N.INFO,
            f"{len(per_edge)} edges ({kind}); each step is reported separately. Restrict the"
            " scan range or give the edge positions to measure one film",
        )
    )
    return m


def measure_regions(
    x: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    region_a: tuple[float, float],
    region_b: tuple[float, float],
    *,
    k: float = 4.0,
) -> StepMeasurement:
    """Height of region B above region A (manual cursors)."""
    ra, rb = sorted(region_a), sorted(region_b)
    ma = (x >= ra[0]) & (x <= ra[1]) & valid
    mb = (x >= rb[0]) & (x <= rb[1]) & valid
    a = fit_plane(x, z, ma, name="reference A", k=k)
    b = fit_plane(x, z, mb, name="measurement B", k=k)
    if a is None or b is None:
        raise ValueError("each region needs at least three valid points")
    if a.x_stop_um <= b.x_start_um:
        x_at, gap = 0.5 * (a.x_stop_um + b.x_start_um), b.x_start_um - a.x_stop_um
    elif b.x_stop_um <= a.x_start_um:
        x_at, gap = 0.5 * (b.x_stop_um + a.x_start_um), a.x_start_um - b.x_stop_um
    else:
        x_at, gap = 0.5 * (a.x_start_um + a.x_stop_um), 0.0
    diff = _difference(a, b, x_at, gap)
    m = StepMeasurement(
        kind="regions",
        method="regions",
        value_nm=diff["height_nm"],
        u_nm=diff["u_nm"],
        u_fit_nm=diff["u_fit_nm"],
        u_slope_nm=diff["u_slope_nm"],
        position_um=x_at,
        width_um=gap,
        planes=[a, b],
        per_edge=[{"edge": 1, **diff}],
        stats={"means_difference_nm": diff["means_difference_nm"]},
        regions_um={"plane": [(a.x_start_um, a.x_stop_um), (b.x_start_um, b.x_stop_um)]},
    )
    m.notices += _burr_notices([a, b], k) + _nonparallel(diff, "two regions", a, b)
    m.notices += _correlation_notices([a, b])
    return m


def pick_pair(eset: EdgeSet, kind: str) -> tuple[int, int] | None:
    """Consecutive edges forming the largest groove (down, up) or ridge (up, down)."""
    want = (-1, 1) if kind == "groove" else (1, -1)
    best, score = None, -1.0
    for i in range(len(eset.edges) - 1):
        a, b = eset.edges[i], eset.edges[i + 1]
        if (a.sign, b.sign) == want:
            s = min(abs(a.height_nm), abs(b.height_nm))
            if s > score:
                best, score = (i, i + 1), s
    return best


def measure(
    x: np.ndarray,
    z: np.ndarray,
    eset: EdgeSet,
    kind: str,
    *,
    k: float = 4.0,
    tip_radius_um: float = 2.0,
) -> StepMeasurement:
    """Dispatch on the profile kind (from :func:`.edges.classify` or forced)."""
    if not eset.edges or kind == "none":
        return StepMeasurement(
            "none",
            "none",
            None,
            None,
            notices=[
                Notice(
                    N.NO_STEP_FOUND,
                    N.WARNING,
                    "no step above the noise threshold"
                    f" ({eset.threshold.get('height_threshold_nm', float('nan')):.3g} nm) was"
                    " found; give the edge positions or reference regions",
                    dict(eset.threshold),
                )
            ],
        )
    if kind == "edge":
        i = int(np.argmax([abs(e.height_nm) for e in eset.edges]))
        m = measure_single_edge(x, z, eset, i, k=k)
        if len(eset.edges) > 1:
            m.notices.append(
                Notice(
                    N.MULTIPLE_STEPS,
                    N.INFO,
                    f"{len(eset.edges)} edges found; the largest (at {eset.edges[i].x_um:.4g} µm)"
                    " was measured",
                )
            )
        return m
    if kind in ("groove", "ridge"):
        pair = pick_pair(eset, kind)
        if pair is None:
            return StepMeasurement(
                kind,
                "iso5436-a1",
                None,
                None,
                notices=[
                    Notice(
                        N.NO_STEP_FOUND,
                        N.WARNING,
                        f"no {'down-up' if kind == 'groove' else 'up-down'} edge pair found for"
                        f" a {kind}",
                    )
                ],
            )
        m = measure_a1(x, z, eset, pair, kind=kind, k=k, tip_radius_um=tip_radius_um)
        if len(eset.edges) > 2:
            m.notices.append(
                Notice(
                    N.MULTIPLE_STEPS,
                    N.INFO,
                    f"{len(eset.edges)} edges found; the {kind} between edges {pair[0] + 1} and"
                    f" {pair[1] + 1} was measured",
                )
            )
        return m
    if kind == "periodic":
        return measure_periodic(x, z, eset, k=k)
    return measure_multi(x, z, eset, kind, k=k)


def range_notices(
    m: StepMeasurement,
    noise_nm: float,
    inst: InstrumentSpec = DEFAULT_INSTRUMENT,
) -> list[Notice]:
    """Warnings that depend on the size of the result."""
    out: list[Notice] = []
    t = m.thickness_nm
    if t is None:
        return out
    if t < inst.vertical_min_nm:
        out.append(
            Notice(
                N.SMALL_STEP,
                N.WARNING,
                f"the step ({t:.3g} nm) is below the lower end of the instrument's vertical"
                f" range ({inst.vertical_min_nm:g} nm)",
                {"thickness_nm": t},
            )
        )
    elif t < 2 * inst.vertical_min_nm:
        out.append(
            Notice(
                N.SMALL_STEP,
                N.WARNING,
                f"the step ({t:.3g} nm) is close to the lower end of the instrument's vertical"
                f" range ({inst.vertical_min_nm:g} nm); check against a reference or another"
                " method",
                {"thickness_nm": t},
            )
        )
    if m.u_nm is not None and (t < 3 * m.u_nm or t < 5 * noise_nm):
        out.append(
            Notice(
                N.STEP_BELOW_NOISE,
                N.WARNING,
                f"the step ({t:.3g} nm) is not clearly above the noise ({noise_nm:.3g} nm) or"
                f" its uncertainty ({m.u_nm:.3g} nm)",
            )
        )
    return out

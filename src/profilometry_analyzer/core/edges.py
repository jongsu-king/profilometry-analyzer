"""Step-edge detection.

An edge is a run of points where the smoothed derivative ``dz/dx`` stands out from its
median by more than ``k`` robust standard deviations *and* the height changes across
the run by more than ``k_height`` times the height noise (or ``min_step_nm``).  The run
is grown outwards while the derivative stays above ``k_extend`` sigmas, so it covers
the whole transition, and same-sign runs separated by a short gap are merged.

The flat parts between edges are *segments* (plateaus).  Each segment loses a margin
next to every edge (``margin_um``, default two stylus radii) so that the rounded
corners drawn by the stylus never enter a plane fit.  A segment that is too short to
fit after the margins (a burr on the rim of a scratch, a dust particle, a narrow line)
is dissolved: its two edges become one transition, or - when the net height change is
below the threshold - a narrow feature that is excluded from every fit and reported.

Wide, gentle edges (a spin-coated or shadow-masked film edge running out over tens to
hundreds of micrometres) do not stand out in a derivative smoothed over one stylus
radius.  With ``multiscale`` (the default) the derivative is computed again with
smoothings ``3, 9, 27 ...`` times wider, up to a fortieth of the scan, and edges found
only there are added.  At these scales a tilt or a parabolic bow of the whole scan
would look like a large derivative, so a robust straight line through the derivative
is removed first, and runs near the scan ends are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .. import notices as N
from ..notices import Notice
from .stats import diff_noise, mad_sigma, runs


@dataclass
class Edge:
    """One transition between two levels; indices refer to the profile arrays."""

    start: int
    stop: int
    """Exclusive."""
    x_um: float
    """Centre: derivative-weighted centroid of the transition (or the given position)."""
    x_start_um: float
    x_stop_um: float
    sign: int
    """+1 when the surface goes up from left to right, -1 when it goes down."""
    height_nm: float
    """Approximate height change across the run (smoothed profile)."""
    manual: bool = False
    x_centroid_um: float | None = None
    """Derivative centroid, kept when the centre is refined to the half-height point."""
    smooth_um: float | None = None
    """Derivative smoothing at which the edge was found (``None``: the base smoothing)."""
    widened_um: tuple[float, float] = (0.0, 0.0)
    """How far the transition was widened on the left and right side because the plateau
    next to it was not yet flat (:func:`~profilometry_analyzer.core.steps.widen_transitions`)."""

    @property
    def width_um(self) -> float:
        return self.x_stop_um - self.x_start_um

    @property
    def direction(self) -> str:
        return "up" if self.sign > 0 else "down"

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_um": self.x_um,
            "x_start_um": self.x_start_um,
            "x_stop_um": self.x_stop_um,
            "width_um": self.width_um,
            "direction": self.direction,
            "height_nm": self.height_nm,
            "manual": self.manual,
            "x_centroid_um": self.x_centroid_um,
            "smooth_um": self.smooth_um,
            "widened_um": list(self.widened_um),
        }


@dataclass
class Segment:
    """A plateau between two edges (or between an edge and the end of the scan)."""

    label: int
    start: int
    stop: int
    use_start: int
    use_stop: int
    """Usable part after the margins (stop exclusive); may be empty."""

    def to_dict(self, x: np.ndarray) -> dict[str, Any]:
        def at(i: int) -> float:
            return float(x[min(max(i, 0), x.size - 1)])

        return {
            "label": self.label,
            "x_start_um": at(self.start),
            "x_stop_um": at(self.stop - 1),
            "use_x_start_um": at(self.use_start) if self.use_stop > self.use_start else None,
            "use_x_stop_um": at(self.use_stop - 1) if self.use_stop > self.use_start else None,
        }


@dataclass
class EdgeSet:
    """Edges, plateaus and the detection parameters of one profile."""

    edges: list[Edge]
    segments: list[Segment]
    exclude: np.ndarray
    """Points excluded from all plane fits (dissolved narrow features and their margins)."""
    valid: np.ndarray
    smooth_um: float
    margin_um: float
    min_plane_um: float
    threshold: dict[str, float]
    features: list[dict[str, Any]] = field(default_factory=list)
    notices: list[Notice] = field(default_factory=list)

    def usable(self, seg: Segment) -> np.ndarray:
        """Boolean mask of the points of ``seg`` that may enter a plane fit."""
        m = np.zeros(self.valid.shape, dtype=bool)
        if seg.use_stop > seg.use_start:
            m[seg.use_start : seg.use_stop] = True
        return m & self.valid & ~self.exclude

    def labels(self) -> np.ndarray:
        """Segment label of every usable point, -1 elsewhere."""
        lab = np.full(self.valid.shape, -1, dtype=int)
        for seg in self.segments:
            lab[self.usable(seg)] = seg.label
        return lab

    def to_dict(self, x: np.ndarray) -> dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "segments": [s.to_dict(x) for s in self.segments],
            "smooth_um": self.smooth_um,
            "margin_um": self.margin_um,
            "min_plane_um": self.min_plane_um,
            "threshold": dict(self.threshold),
            "features": list(self.features),
        }


def _fill_invalid(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    zz = np.asarray(z, dtype=float).copy()
    bad = ~valid | ~np.isfinite(zz)
    if bad.any() and (~bad).any():
        idx = np.arange(zz.size)
        zz[bad] = np.interp(idx[bad], idx[~bad], zz[~bad])
    return zz


def default_smooth_um(dx: float, tip_radius_um: float) -> float:
    """Derivative smoothing (Gaussian sigma): one stylus radius, at least 3 points."""
    return float(max(tip_radius_um, 3.0 * dx))


def default_margin_um(tip_radius_um: float) -> float:
    """Exclusion beyond each detected transition: two stylus radii."""
    return 2.0 * tip_radius_um


def default_min_plane_um(dx: float, tip_radius_um: float) -> float:
    """Shortest usable plateau: three stylus radii and at least ten points."""
    return float(max(3.0 * tip_radius_um, 10.0 * dx))


def smoothed_derivative(
    x: np.ndarray, z: np.ndarray, valid: np.ndarray, smooth_um: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(z_smooth, dz/dx)`` with a Gaussian of sigma ``smooth_um``."""
    dx = float(np.median(np.diff(x)))
    sig = max(smooth_um / dx, 1.0)
    zs = gaussian_filter1d(_fill_invalid(z, valid), sig, mode="nearest")
    return zs, np.gradient(zs, x)


def _robust_poly(x: np.ndarray, y: np.ndarray, valid: np.ndarray, order: int = 1) -> np.ndarray:
    """Polynomial through ``y`` with points beyond 2.5 robust sigmas left out."""
    use = valid & np.isfinite(y)
    if use.sum() <= order + 1:
        return np.zeros_like(y)
    c = float(np.mean(x))
    h = float(max(np.ptp(x) / 2.0, 1e-12))
    t = (x - c) / h
    coef = np.polyfit(t[use], y[use], order)
    for _ in range(20):
        r = y - np.polyval(coef, t)
        s = mad_sigma(r[use])
        if not s > 0:
            break
        new = valid & np.isfinite(y) & (np.abs(r) <= 2.5 * s)
        if new.sum() <= order + 1 or np.array_equal(new, use):
            break
        use = new
        coef = np.polyfit(t[use], y[use], order)
    return np.asarray(np.polyval(coef, t))


def _signed_runs(dd: np.ndarray, thresh: float) -> list[tuple[int, int, int]]:
    out = [(a, b, 1) for a, b in runs(dd > thresh)]
    out += [(a, b, -1) for a, b in runs(dd < -thresh)]
    return sorted(out)


def _coarse_derivative(x: np.ndarray, zf: np.ndarray, s: float) -> np.ndarray:
    """``dz/dx`` smoothed with a Gaussian of sigma ``s``, computed on block means of
    about ``s / 4`` (the block averaging is negligible next to the Gaussian) and
    interpolated back - a long kernel on every point would be slow for 100k-point scans."""
    dx = float(np.median(np.diff(x)))
    q = max(int(s / (4.0 * dx)), 1)
    if q == 1:
        return np.gradient(gaussian_filter1d(zf, max(s / dx, 1.0), mode="nearest"), x)
    m = (x.size // q) * q
    xb = x[:m].reshape(-1, q).mean(axis=1)
    zb = zf[:m].reshape(-1, q).mean(axis=1)
    if m < x.size:
        xb = np.append(xb, x[m:].mean())
        zb = np.append(zb, zf[m:].mean())
    db = np.gradient(gaussian_filter1d(zb, max(s / (q * dx), 1.0), mode="nearest"), xb)
    return np.asarray(np.interp(x, xb, db))


def _coarse_runs(
    x: np.ndarray,
    zf: np.ndarray,
    valid: np.ndarray,
    smooth: float,
    k: float,
    k_extend: float,
    taken: np.ndarray,
    *,
    factor: float = 3.0,
) -> list[tuple[int, int, float, np.ndarray]]:
    """Edges that only a wider smoothing reveals: ``(start, stop, smooth_um, dd)``."""
    n = x.size
    dx = float(np.median(np.diff(x)))
    length = float(x[-1] - x[0])
    taken = taken.copy()
    found: list[tuple[int, int, float, np.ndarray]] = []
    s = smooth * factor
    while s <= length / 40.0:
        d = _coarse_derivative(x, zf, s)
        dd = d - _robust_poly(x, d, valid, order=1)
        sd = mad_sigma(dd[valid])
        s_pts = int(np.ceil(s / dx))
        if sd > 0:
            ext = np.abs(dd) > k_extend * sd
            guard = 3 * s_pts
            for a, b, sign in _signed_runs(dd, k * sd):
                if a <= guard or b >= n - guard or taken[a:b].any():
                    continue
                while a > 0 and ext[a - 1] and np.sign(dd[a - 1]) == sign and not taken[a - 1]:
                    a -= 1
                while b < n and ext[b] and np.sign(dd[b]) == sign and not taken[b]:
                    b += 1
                # the smoothing itself broadens the run by about one sigma on each side
                a2, b2 = a + s_pts, b - s_pts
                if b2 <= a2:
                    mid = a + int(np.argmax(np.abs(dd[a:b])))
                    a2, b2 = mid, mid + 1
                found.append((a2, b2, s, dd))
                taken[a:b] = True
        s *= factor
    return found


def detect_edges(
    x: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    tip_radius_um: float = 2.0,
    smooth_um: float | None = None,
    k: float = 6.0,
    k_extend: float = 2.0,
    k_height: float = 5.0,
    min_step_nm: float = 0.0,
    margin_um: float | None = None,
    min_plane_um: float | None = None,
    multiscale: bool = True,
) -> EdgeSet:
    """Find the step edges of a profile (see the module docstring)."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n = x.size
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    dx = float(np.median(np.diff(x)))
    smooth = float(smooth_um) if smooth_um else default_smooth_um(dx, tip_radius_um)
    margin = float(margin_um) if margin_um is not None else default_margin_um(tip_radius_um)
    min_plane = (
        float(min_plane_um) if min_plane_um is not None else default_min_plane_um(dx, tip_radius_um)
    )
    zf = _fill_invalid(z, valid)
    zs, d = smoothed_derivative(x, zf, valid, smooth)
    dd = d - float(np.median(d[valid])) if valid.any() else d - float(np.median(d))
    sd = mad_sigma(dd[valid])
    peak = float(np.max(np.abs(dd))) if dd.size else 0.0
    if not sd > 0:
        sd = max(1e-9 * peak, 1e-300)
    z_noise = diff_noise(zf[valid])
    resid_noise = mad_sigma((zf - zs)[valid])
    noise = float(max(z_noise, resid_noise)) if np.isfinite(resid_noise) else z_noise
    zrange = float(np.ptp(zf)) if zf.size else 0.0
    h_thresh = float(max(k_height * noise, min_step_nm, 1e-9 * zrange))
    gap_pts = max(int(round(2.0 * smooth / dx)), 1)

    ext = np.abs(dd) > k_extend * sd
    raw_runs: list[list[int]] = []  # [start, stop, sign]
    # runs of each sign separately: two walls closer than the smoothing never merge
    for a, b, sign in _signed_runs(dd, k * sd):
        while a > 0 and ext[a - 1] and np.sign(dd[a - 1]) == sign:
            a -= 1
        while b < n and ext[b] and np.sign(dd[b]) == sign:
            b += 1
        if raw_runs and raw_runs[-1][2] == sign and a - raw_runs[-1][1] <= gap_pts:
            raw_runs[-1][1] = max(raw_runs[-1][1], b)
        elif raw_runs and a < raw_runs[-1][1]:
            # overlapping opposite-sign runs: cut at the boundary
            a = raw_runs[-1][1]
            if b > a:
                raw_runs.append([a, b, sign])
        else:
            raw_runs.append([a, b, sign])

    def make_edge(a: int, b: int, manual: bool = False, deriv: np.ndarray | None = None) -> Edge:
        lo = max(a - 1, 0)
        hi = min(b, n - 1)
        h = float(zs[hi] - zs[lo])
        w = np.abs((dd if deriv is None else deriv)[a:b])
        xc = float(np.sum(w * x[a:b]) / np.sum(w)) if w.sum() > 0 else float(x[(a + b) // 2])
        return Edge(
            start=a,
            stop=b,
            x_um=xc,
            x_start_um=float(x[a]),
            x_stop_um=float(x[min(b, n) - 1]),
            sign=1 if h > 0 else -1,
            height_nm=h,
            manual=manual,
        )

    edges = [make_edge(a, b) for a, b, _ in raw_runs]
    edges = [e for e in edges if abs(e.height_nm) >= h_thresh]
    if multiscale and n >= 200:
        taken = np.zeros(n, dtype=bool)
        for e in edges:
            taken[e.start : e.stop] = True
        for a, b, s_c, dd_c in _coarse_runs(x, zf, valid, smooth, k, k_extend, taken):
            e = make_edge(a, b, deriv=dd_c)
            e.smooth_um = s_c
            if abs(e.height_nm) >= h_thresh:
                edges.append(e)
        edges.sort(key=lambda e: e.start)
    exclude = np.zeros(n, dtype=bool)
    features: list[dict[str, Any]] = []
    edges = _dissolve_narrow(
        x, zs, edges, exclude, features, margin, min_plane, h_thresh, make_edge
    )
    segments = build_segments(x, edges, margin)
    eset = EdgeSet(
        edges=edges,
        segments=segments,
        exclude=exclude,
        valid=valid.copy(),
        smooth_um=smooth,
        margin_um=margin,
        min_plane_um=min_plane,
        threshold={
            "k_derivative": k,
            "k_extend": k_extend,
            "derivative_sigma_nm_per_um": sd,
            "height_noise_nm": noise,
            "height_threshold_nm": h_thresh,
            "multiscale": float(multiscale),
        },
        features=features,
    )
    _prune_small_steps(x, zf, eset, h_thresh, make_edge)
    eset.notices.extend(_segment_notices(x, eset))
    for e in eset.edges:
        if e.smooth_um is not None:
            eset.notices.append(
                Notice(
                    N.WIDE_EDGE,
                    N.INFO,
                    f"the edge at {e.x_um:.4g} µm runs out over about {e.width_um:.3g} µm and was"
                    f" found only with a {e.smooth_um:.3g} µm derivative smoothing (the base"
                    f" smoothing is {smooth:.3g} µm)",
                    {"x_um": e.x_um, "width_um": e.width_um, "smooth_um": e.smooth_um},
                )
            )
    for f in features:
        eset.notices.append(
            Notice(
                N.NARROW_FEATURE,
                N.WARNING,
                f"narrow {f['kind']} at {f['x_um']:.4g} µm ({f['width_um']:.3g} µm wide,"
                f" {f['extreme_nm']:+.4g} nm) is shorter than the shortest plane"
                f" ({min_plane:.3g} µm) and was excluded from the plane fits. To measure it,"
                " lower the smoothing (--smooth), the margins or the shortest plane, or give"
                " the edge positions",
                f,
            )
        )
    return eset


def _dissolve_narrow(
    x: np.ndarray,
    zs: np.ndarray,
    edges: list[Edge],
    exclude: np.ndarray,
    features: list[dict[str, Any]],
    margin: float,
    min_plane: float,
    h_thresh: float,
    make_edge: Any,
) -> list[Edge]:
    """Merge the edges around interior plateaus that are too short to fit."""
    edges = list(edges)
    while len(edges) >= 2:
        gaps = []
        for i in range(len(edges) - 1):
            usable = (edges[i + 1].x_start_um - margin) - (edges[i].x_stop_um + margin)
            gaps.append(usable)
        i = int(np.argmin(gaps))
        if gaps[i] >= min_plane:
            break
        a, b = edges[i].start, edges[i + 1].stop
        merged = make_edge(a, b)
        if abs(merged.height_nm) >= h_thresh:
            edges[i : i + 2] = [merged]
        else:
            seg = zs[a:b]
            base = 0.5 * (zs[max(a - 1, 0)] + zs[min(b, zs.size - 1)])
            dev = seg - base
            j = int(np.argmax(np.abs(dev)))
            lo = int(np.searchsorted(x, x[a] - margin))
            hi = int(np.searchsorted(x, x[min(b, x.size) - 1] + margin, side="right"))
            exclude[lo:hi] = True
            features.append(
                {
                    "kind": "bump" if dev[j] > 0 else "dip",
                    "x_um": float(x[a + j]),
                    "x_start_um": float(x[a]),
                    "x_stop_um": float(x[min(b, x.size) - 1]),
                    "width_um": float(x[min(b, x.size) - 1] - x[a]),
                    "extreme_nm": float(dev[j]),
                }
            )
            del edges[i : i + 2]
    return edges


def build_segments(x: np.ndarray, edges: list[Edge], margin: float) -> list[Segment]:
    """Plateaus between consecutive edges, each shrunk by ``margin`` next to an edge."""
    n = x.size
    bounds = [0, *[v for e in edges for v in (e.start, e.stop)], n]
    segs: list[Segment] = []
    for k in range(len(edges) + 1):
        a, b = bounds[2 * k], bounds[2 * k + 1]
        if b <= a:
            b = a
        ua, ub = a, b
        if k > 0:
            ua = int(np.searchsorted(x, x[min(a, n - 1)] + margin))
        if k < len(edges):
            ub = int(np.searchsorted(x, x[max(b - 1, 0)] - margin, side="right"))
        ua, ub = max(ua, a), min(ub, b)
        segs.append(Segment(label=k, start=a, stop=b, use_start=ua, use_stop=max(ub, ua)))
    return segs


def _prune_small_steps(
    x: np.ndarray, z: np.ndarray, eset: EdgeSet, h_thresh: float, make_edge: Any
) -> None:
    """Drop edges whose neighbouring plateaus (medians) differ by less than the threshold."""
    changed = True
    while changed and eset.edges:
        changed = False
        med = []
        for seg in eset.segments:
            m = eset.usable(seg)
            med.append(float(np.median(z[m])) if m.any() else np.nan)
        for i, _e in enumerate(eset.edges):
            a, b = med[i], med[i + 1]
            if np.isfinite(a) and np.isfinite(b) and abs(b - a) < h_thresh:
                del eset.edges[i]
                eset.segments = build_segments(x, eset.edges, eset.margin_um)
                changed = True
                break
        if not changed:
            # refresh signs and heights from the plateau medians
            for i, e in enumerate(eset.edges):
                a, b = med[i], med[i + 1]
                if np.isfinite(a) and np.isfinite(b):
                    e.height_nm = float(b - a)
                    e.sign = 1 if b > a else -1


def _segment_notices(x: np.ndarray, eset: EdgeSet) -> list[Notice]:
    out: list[Notice] = []
    for seg in eset.segments:
        m = eset.usable(seg)
        if not m.any():
            length = 0.0
        else:
            xs = x[m]
            length = float(xs[-1] - xs[0])
        if length < eset.min_plane_um:
            out.append(
                Notice(
                    N.SHORT_PLANE,
                    N.WARNING,
                    f"plateau {seg.label + 1} has only {length:.3g} µm ({int(m.sum())} points)"
                    f" left after the edge margins; the shortest reliable plane is"
                    f" {eset.min_plane_um:.3g} µm",
                    {"segment": seg.label, "usable_um": length, "points": int(m.sum())},
                )
            )
    return out


def edges_at(
    x: np.ndarray,
    z: np.ndarray,
    positions_um: list[float],
    valid: np.ndarray | None = None,
    *,
    tip_radius_um: float = 2.0,
    halfwidth_um: float | None = None,
    margin_um: float | None = None,
    min_plane_um: float | None = None,
) -> EdgeSet:
    """An :class:`EdgeSet` from edge positions given by the user (GUI cursors, CLI).

    Each edge is a transition of ``±halfwidth_um`` around the position (default: two
    stylus radii); its direction comes from the plateau medians on both sides.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n = x.size
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    dx = float(np.median(np.diff(x)))
    hw = float(halfwidth_um) if halfwidth_um is not None else 2.0 * tip_radius_um
    margin = float(margin_um) if margin_um is not None else default_margin_um(tip_radius_um)
    min_plane = (
        float(min_plane_um) if min_plane_um is not None else default_min_plane_um(dx, tip_radius_um)
    )
    edges: list[Edge] = []
    for p in sorted(float(v) for v in positions_um):
        a = int(np.searchsorted(x, p - hw))
        b = int(np.searchsorted(x, p + hw, side="right"))
        a, b = max(a, 0), min(max(b, a + 1), n)
        edges.append(Edge(a, b, p, float(x[a]), float(x[b - 1]), 1, 0.0, manual=True))
    for i in range(1, len(edges)):
        if edges[i].start < edges[i - 1].stop:
            edges[i].start = edges[i - 1].stop
    zf = _fill_invalid(z, valid)
    eset = EdgeSet(
        edges=edges,
        segments=build_segments(x, edges, margin),
        exclude=np.zeros(n, dtype=bool),
        valid=valid.copy(),
        smooth_um=0.0,
        margin_um=margin,
        min_plane_um=min_plane,
        threshold={"manual_halfwidth_um": hw},
    )
    for i, e in enumerate(eset.edges):
        left = eset.usable(eset.segments[i])
        right = eset.usable(eset.segments[i + 1])
        if left.any() and right.any():
            e.height_nm = float(np.median(zf[right]) - np.median(zf[left]))
            e.sign = 1 if e.height_nm > 0 else -1
    eset.notices.extend(_segment_notices(x, eset))
    return eset


def classify(edges: list[Edge], periodic_cv: float = 0.25) -> str:
    """``none``, ``edge``, ``groove``, ``ridge``, ``staircase``, ``periodic`` or ``multi``.

    * one edge: a film edge (``edge``);
    * down then up: a groove or scratch (``groove``); up then down: a line or stripe of
      film (``ridge``, evaluated like a groove with the sign reversed);
    * all edges in the same direction: ``staircase``;
    * at least four alternating edges with regular spacing (coefficient of variation of
      the same-direction spacings below ``periodic_cv``): ``periodic``;
    * anything else: ``multi`` (every edge is reported separately).
    """
    n = len(edges)
    if n == 0:
        return "none"
    if n == 1:
        return "edge"
    signs = [e.sign for e in edges]
    if n == 2:
        if signs == [-1, 1]:
            return "groove"
        if signs == [1, -1]:
            return "ridge"
        return "staircase"
    if all(s == signs[0] for s in signs):
        return "staircase"
    alternating = all(signs[i] != signs[i + 1] for i in range(n - 1))
    if alternating and n >= 4:
        xs = np.array([e.x_um for e in edges])
        sp = np.concatenate([np.diff(xs[0::2]), np.diff(xs[1::2])])
        if sp.size >= 2 and np.mean(sp) > 0 and np.std(sp) / np.mean(sp) < periodic_cv:
            return "periodic"
    return "multi"

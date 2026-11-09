"""Small numerical helpers: robust scale, line fits with covariance, robust fits with a
shared slope and per-segment offsets, uniform resampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAD_TO_SIGMA = 1.482602218505602


def mad_sigma(v: np.ndarray) -> float:
    """Robust standard deviation (scaled median absolute deviation)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def diff_noise(z: np.ndarray) -> float:
    """White-noise level from first differences (insensitive to slow trends)."""
    d = np.diff(np.asarray(z, dtype=float))
    return mad_sigma(d) / np.sqrt(2.0) if d.size else float("nan")


def lag1_autocorrelation(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if r.size < 3:
        return 0.0
    r = r - r.mean()
    den = float(np.dot(r, r))
    if den <= 0:
        return 0.0
    return float(np.dot(r[:-1], r[1:]) / den)


def autocorrelation(r: np.ndarray) -> np.ndarray:
    """Sample autocorrelation of a sequence (lags 0..n-1), computed with an FFT."""
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    n = r.size
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(r, m)
    acf = np.fft.irfft(f * np.conjugate(f), m)[:n]
    return np.asarray(acf / acf[0]) if acf[0] > 0 else np.zeros(n)


def integrated_time(r: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time ``tau = 1 + 2 sum_k rho(k)`` with Sokal's window.

    The sum stops at the smallest lag ``M`` with ``M >= c * tau(M)``, which keeps the
    noisy far tail of the sample autocorrelation out of the estimate.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    if n < 4:
        return 1.0
    rho = autocorrelation(r)
    tau = 1.0
    for m in range(1, n // 2):
        tau += 2.0 * rho[m]
        if m >= c * tau:
            break
    return float(max(tau, 1.0))


def n_effective(r: np.ndarray) -> float:
    """Effective number of independent residuals: ``n / tau``.

    Roughness, waviness and smoothing make neighbouring residuals correlated, so the
    textbook regression covariance would be too optimistic.  ``tau`` is the integrated
    autocorrelation time (:func:`integrated_time`), which handles a mixture of white
    noise and correlated roughness (a lag-1 AR(1) model does not).
    """
    n = len(r)
    return float(max(2.0, n / integrated_time(np.asarray(r, dtype=float))))


@dataclass
class LineFit:
    slope: float
    intercept: float
    """Value at ``x0``."""
    x0: float
    cov: np.ndarray
    """Covariance of ``(slope, intercept)``, inflated for correlated residuals."""
    sigma: float
    n: int
    n_eff: float
    x_min: float
    x_max: float

    def at(self, x: float | np.ndarray) -> Any:
        return self.slope * (np.asarray(x, dtype=float) - self.x0) + self.intercept

    def var_at(self, x: float) -> float:
        v = np.array([x - self.x0, 1.0])
        return float(v @ self.cov @ v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slope_nm_per_um": self.slope,
            "intercept_nm": self.intercept,
            "x0_um": self.x0,
            "sigma_nm": self.sigma,
            "n": self.n,
            "n_eff": self.n_eff,
            "x_min_um": self.x_min,
            "x_max_um": self.x_max,
            "slope_std": float(np.sqrt(max(self.cov[0, 0], 0.0))),
            "intercept_std": float(np.sqrt(max(self.cov[1, 1], 0.0))),
        }


def fit_line(x: np.ndarray, z: np.ndarray, x0: float | None = None) -> LineFit:
    """Least-squares line with a covariance corrected for residual autocorrelation."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.size < 2:
        raise ValueError("a line fit needs at least two points")
    c = float(np.mean(x)) if x0 is None else float(x0)
    A = np.column_stack([x - c, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    r = z - A @ coef
    dof = max(x.size - 2, 1)
    s2 = float(np.dot(r, r) / dof)
    neff = n_effective(r)
    inflate = max(x.size / neff, 1.0)
    try:
        cov = s2 * np.linalg.inv(A.T @ A) * inflate
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)
    return LineFit(
        slope=float(coef[0]),
        intercept=float(coef[1]),
        x0=c,
        cov=cov,
        sigma=float(np.sqrt(s2)),
        n=int(x.size),
        n_eff=neff,
        x_min=float(x.min()),
        x_max=float(x.max()),
    )


@dataclass
class SharedFit:
    """``z = sum_k c_k ((x - x0)/scale)^k + offset[label]`` (k = 1..order)."""

    coefs: np.ndarray
    offsets: dict[int, float]
    x0: float
    scale: float
    order: int
    sigma: float
    weights: np.ndarray
    inliers: np.ndarray
    method: str
    info: dict[str, Any] = field(default_factory=dict)

    def trend(self, x: np.ndarray) -> np.ndarray:
        t = (np.asarray(x, dtype=float) - self.x0) / self.scale
        out = np.zeros_like(t)
        for k, c in enumerate(self.coefs, start=1):
            out += c * t**k
        return out

    @property
    def slope(self) -> float:
        """Slope of the trend at x0 (nm per µm)."""
        return float(self.coefs[0] / self.scale) if self.coefs.size else 0.0


def _design(x: np.ndarray, labels: np.ndarray, order: int, x0: float, scale: float) -> tuple:
    t = (x - x0) / scale
    uniq = np.unique(labels)
    cols = [t**k for k in range(1, order + 1)]
    cols += [(labels == u).astype(float) for u in uniq]
    return np.column_stack(cols), uniq


def fit_shared(
    x: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    *,
    order: int = 1,
    method: str = "huber",
    huber_k: float = 1.345,
    max_iter: int = 50,
    ransac_iter: int = 400,
    seed: int = 0,
) -> SharedFit:
    """Polynomial trend shared by all segments, one offset per segment label.

    ``method``: ``"lsq"`` (ordinary least squares), ``"huber"`` (iteratively
    reweighted, Huber weights, scale from the MAD), ``"ransac"`` (consensus of
    two-point slopes within one segment, then least squares on the inliers; trend order
    1 only - higher orders fall back to Huber after the RANSAC inlier selection).
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels)
    if x.size < order + 2:
        raise ValueError("not enough points for the trend fit")
    x0 = float(np.mean(x))
    scale = float(max(np.ptp(x) / 2.0, 1e-12))
    A, uniq = _design(x, labels, order, x0, scale)
    w = np.ones_like(z)
    inliers = np.ones(z.shape, dtype=bool)
    info: dict[str, Any] = {}

    def solve(weights: np.ndarray) -> np.ndarray:
        sw = np.sqrt(weights)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
        return np.asarray(coef)

    if method == "ransac":
        inliers = _ransac_inliers(x, z, labels, seed=seed, n_iter=ransac_iter)
        info["ransac_inliers"] = int(inliers.sum())
        w = inliers.astype(float)
        coef = solve(w)
        if order > 1:
            method_used = "ransac+huber"
            coef, w = _huber_loop(A, z, coef, w, huber_k, max_iter, inliers)
        else:
            method_used = "ransac"
    elif method == "huber":
        coef = solve(w)
        coef, w = _huber_loop(A, z, coef, w, huber_k, max_iter, inliers)
        method_used = "huber"
    elif method == "lsq":
        coef = solve(w)
        method_used = "lsq"
    else:
        raise ValueError(f"unknown fit method {method!r}")
    r = z - A @ coef
    sigma = mad_sigma(r[inliers]) if inliers.any() else float("nan")
    offsets = {int(u): float(coef[order + i]) for i, u in enumerate(uniq)}
    return SharedFit(
        coefs=np.asarray(coef[:order], dtype=float),
        offsets=offsets,
        x0=x0,
        scale=scale,
        order=order,
        sigma=sigma,
        weights=w,
        inliers=inliers,
        method=method_used,
        info=info,
    )


def _huber_loop(
    A: np.ndarray,
    z: np.ndarray,
    coef: np.ndarray,
    w0: np.ndarray,
    k: float,
    max_iter: int,
    inliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    w = w0.copy()
    for _ in range(max_iter):
        r = z - A @ coef
        s = mad_sigma(r[inliers]) if inliers.any() else 0.0
        if not s > 0:
            break
        u = np.abs(r) / (k * s)
        w_new = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12)) * inliers
        sw = np.sqrt(w_new)
        new, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
        if np.allclose(new, coef, rtol=1e-10, atol=1e-12 * (1 + np.abs(coef).max())):
            coef, w = new, w_new
            break
        coef, w = new, w_new
    return coef, w


def _ransac_inliers(
    x: np.ndarray, z: np.ndarray, labels: np.ndarray, *, seed: int, n_iter: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    groups = {int(u): np.flatnonzero(labels == u) for u in uniq}
    noise = max(
        float(np.nanmedian([diff_noise(z[idx]) for idx in groups.values() if idx.size > 3])), 1e-12
    )
    thresh = 3.0 * noise * np.sqrt(2.0)
    best = np.ones(z.shape, dtype=bool)
    best_score = -1
    for _ in range(n_iter):
        g = groups[int(rng.choice(uniq, p=probs))]
        if g.size < 4:
            continue
        i, j = rng.choice(g, size=2, replace=False)
        if abs(x[j] - x[i]) < 0.1 * np.ptp(x[g]):
            continue
        a = (z[j] - z[i]) / (x[j] - x[i])
        r = np.empty_like(z)
        for idx in groups.values():
            rr = z[idx] - a * x[idx]
            r[idx] = rr - np.median(rr)
        inl = np.abs(r) < thresh
        score = int(inl.sum())
        if score > best_score:
            best, best_score = inl, score
    return best


def uniform_grid(
    x: np.ndarray, z: np.ndarray, tol: float = 0.01
) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(x, z, resampled)``: the data on a uniform grid (linear interpolation if the
    spacing varies by more than ``tol``)."""
    d = np.diff(x)
    med = float(np.median(d))
    if med <= 0:
        raise ValueError("lateral positions must increase")
    if float(np.max(np.abs(d - med))) <= tol * med:
        return x, z, False
    n = int(round((x[-1] - x[0]) / med)) + 1
    xu = np.linspace(x[0], x[-1], n)
    return xu, np.interp(xu, x, z), True


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` of consecutive ``True`` values (stop exclusive)."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))

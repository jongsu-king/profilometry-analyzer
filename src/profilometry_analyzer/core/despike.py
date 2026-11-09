"""Spike removal.

A spike is a run of at most ``max_width`` points that stands out from a running median
by more than ``k`` robust standard deviations (dust, electrical glitches).  Monotonic
ramps - step edges - follow the running median exactly and are never flagged.  Spike
points are marked invalid (excluded from every fit) and, in the stored heights, replaced
by linear interpolation so derivative filters stay finite; the raw values are kept in
``raw_z``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .. import notices as N
from ..notices import Notice
from ..profile import Profile
from .stats import mad_sigma, runs


def find_spikes(z: np.ndarray, valid: np.ndarray, k: float = 8.0, max_width: int = 3) -> np.ndarray:
    window = 2 * max_width + 1
    zz = np.asarray(z, dtype=float).copy()
    if (~valid).any() and valid.any():
        idx = np.arange(zz.size)
        zz[~valid] = np.interp(idx[~valid], idx[valid], zz[valid])
    med = median_filter(zz, size=window, mode="nearest")
    r = zz - med
    s = mad_sigma(r[valid])
    if not s > 0:
        return np.zeros(zz.shape, dtype=bool)
    cand = (np.abs(r) > k * s) & valid
    spikes = np.zeros(zz.shape, dtype=bool)
    for a, b in runs(cand):
        if b - a <= max_width:
            spikes[a:b] = True
    return spikes


def despike(profile: Profile, k: float = 8.0, max_width: int = 3) -> Profile:
    valid = profile.valid
    spikes = find_spikes(profile.z_nm, valid, k=k, max_width=max_width)
    n = int(spikes.sum())
    new_valid = valid & ~spikes
    z = profile.z_nm.copy()
    if n and new_valid.any():
        idx = np.arange(z.size)
        z[spikes] = np.interp(idx[spikes], idx[new_valid], z[new_valid])
    step = {"step": "despike", "k": k, "max_width_points": max_width, "removed": n}
    notes: list[Notice] = []
    if n:
        xs = profile.x_um[spikes]
        notes.append(
            Notice(
                N.SPIKES_REMOVED,
                N.INFO,
                f"{n} spike point(s) excluded (|z - running median| > {k:g} sigma, at most"
                f" {max_width} points wide)",
                {"count": n, "x_um": [float(v) for v in xs[:20]]},
            )
        )
    return profile.derive(z, step=step, mask=new_valid, notices=notes)

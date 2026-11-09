"""Gaussian profile filter (ISO 16610-21).

The weighting function of the linear Gaussian profile filter is

    s(x) = 1 / (alpha * lc) * exp(-pi * (x / (alpha * lc))^2),   alpha = sqrt(ln 2 / pi),

so that the mean line transmits ``exp(-pi (alpha lc / lambda)^2)`` of a sine of
wavelength ``lambda``: exactly 50 % at the cut-off ``lambda = lc``.  The roughness
profile is the complement (``1 - transmission``).  The sampled weights are truncated at
``± lc`` (the weight there is below 1e-6 of the centre value) and normalised to sum
to one.

End effects: near the ends of the profile the window is incomplete.  This module
renormalises the weights inside the data so the mean line stays finite, but the result
is biased there; :mod:`.roughness` therefore discards half a cut-off at each end (the
run-up and run-over of a traverse, ISO 4288 / ISO 21920-3 practice).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

ALPHA = math.sqrt(math.log(2.0) / math.pi)
"""0.4697...: the Gaussian constant of ISO 16610-21."""


def gaussian_weights(dx: float, cutoff: float, truncation: float = 1.0) -> np.ndarray:
    """Sampled, normalised weights for spacing ``dx`` and cut-off ``cutoff`` (same unit)."""
    if not (dx > 0 and cutoff > 0):
        raise ValueError("spacing and cut-off must be positive")
    m = max(int(math.ceil(truncation * cutoff / dx)), 1)
    t = np.arange(-m, m + 1) * dx
    w = np.exp(-math.pi * (t / (ALPHA * cutoff)) ** 2)
    return w / w.sum()


def mean_line(z: np.ndarray, dx: float, cutoff: float, truncation: float = 1.0) -> np.ndarray:
    """Gaussian low-pass of ``z`` (uniform spacing ``dx``); ends renormalised."""
    z = np.asarray(z, dtype=float)
    w = gaussian_weights(dx, cutoff, truncation)
    num = fftconvolve(z, w, mode="same")
    den = fftconvolve(np.ones_like(z), w, mode="same")
    return np.asarray(num / den)


def transmission(wavelength: float | np.ndarray, cutoff: float) -> np.ndarray:
    """Amplitude transmission of the mean line (low-pass) for a sine of ``wavelength``."""
    lam = np.asarray(wavelength, dtype=float)
    return np.asarray(np.exp(-math.pi * (ALPHA * cutoff / lam) ** 2))


def kernel_points(dx: float, cutoff: float, truncation: float = 1.0) -> int:
    return 2 * max(int(math.ceil(truncation * cutoff / dx)), 1) + 1

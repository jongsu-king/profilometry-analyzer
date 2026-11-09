"""Analytical checks: Gaussian filter, sine roughness, Stoney (README: validation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from profilometry_analyzer import notices as N
from profilometry_analyzer.analysis import analyze_roughness, analyze_stress
from profilometry_analyzer.core.filters import ALPHA, gaussian_weights, mean_line, transmission
from profilometry_analyzer.core.roughness import (
    RoughnessSettings,
    class_for_ra,
    mean_element_width,
    roughness,
)
from profilometry_analyzer.core.stress import (
    M_SI100,
    M_SI111,
    SUBSTRATES,
    StressSettings,
    fit_curvature,
    stoney_mpa,
)
from profilometry_analyzer.profile import Profile

from .conftest import make


def sine_profile(A: float, lam: float, length: float = 2000.0, n: int = 40_000) -> Profile:
    x = np.linspace(0, length, n)
    return Profile(x, A * np.sin(2 * np.pi * x / lam), name="sine")


# ------------------------------------------------------------------ filter
def test_gaussian_constant() -> None:
    assert pytest.approx(math.sqrt(math.log(2) / math.pi)) == ALPHA
    assert transmission(80.0, 80.0) == pytest.approx(0.5)


@pytest.mark.parametrize("lc", [80.0, 250.0, 800.0])
def test_mean_line_transmits_half_at_cutoff(lc: float) -> None:
    """A sine of wavelength λc keeps 50 % of its amplitude in the mean line."""
    dx = lc / 400
    x = np.arange(0, 20 * lc, dx)
    z = np.sin(2 * np.pi * x / lc)
    m = mean_line(z, dx, lc)
    core = (x > 2 * lc) & (x < 18 * lc)
    amp = np.sqrt(2 * np.mean(m[core] ** 2))
    assert amp == pytest.approx(0.5, abs=0.003)


def test_transmission_curve_matches_theory() -> None:
    lc, dx = 250.0, 0.5
    for lam in (60.0, 125.0, 500.0, 1000.0):
        x = np.arange(0, 40 * max(lam, lc), dx)
        m = mean_line(np.sin(2 * np.pi * x / lam), dx, lc)
        core = (x > 2 * max(lam, lc)) & (x < x[-1] - 2 * max(lam, lc))
        amp = np.sqrt(2 * np.mean(m[core] ** 2))
        assert amp == pytest.approx(float(transmission(lam, lc)), abs=0.003), lam


def test_weights_normalised_and_truncated() -> None:
    w = gaussian_weights(0.5, 80.0)
    assert w.sum() == pytest.approx(1.0)
    assert w.size == 2 * 160 + 1
    assert w[0] / w.max() < 1e-6


# ------------------------------------------------------------------ roughness
def test_sine_ra_rq_analytical() -> None:
    """Ra = 2A/π and Rq = A/√2 for a sine (no λs, λc far above the wavelength)."""
    A, lam = 10.0, 20.0
    r = roughness(
        sine_profile(A, lam, length=6000, n=60_000),
        RoughnessSettings(lambda_c_um=800.0, lambda_s_um=0),
    )
    p = r.parameters
    assert p["Ra"] == pytest.approx(2 * A / math.pi, rel=1e-3)
    assert p["Rq"] == pytest.approx(A / math.sqrt(2), rel=1e-3)
    assert p["Rt"] == pytest.approx(2 * A, rel=1e-3)
    assert p["Rz"] == pytest.approx(2 * A, rel=1e-3)
    assert p["Rp"] == pytest.approx(A, rel=1e-3) and p["Rv"] == pytest.approx(A, rel=1e-3)
    assert p["Rsk"] == pytest.approx(0.0, abs=1e-3)
    assert p["Rku"] == pytest.approx(1.5, rel=1e-3)
    assert p["Rsm"] == pytest.approx(lam, rel=1e-3)
    assert r.n_sections == 6  # (6000 - 800) // 800


def test_filtered_sine_amplitude_follows_both_filters() -> None:
    """With λs and λc on, Ra = 2A/π · T_s(λ) · (1 − T_c(λ))."""
    A, lam, lc, ls = 12.0, 40.0, 80.0, 2.5
    r = roughness(sine_profile(A, lam), RoughnessSettings(lambda_c_um=lc, lambda_s_um=ls))
    expected = 2 * A / math.pi * float(transmission(lam, ls)) * (1 - float(transmission(lam, lc)))
    assert r.parameters["Ra"] == pytest.approx(expected, rel=2e-3)


def test_auto_cutoff_from_ra() -> None:
    assert class_for_ra(0.015).name == "Sc1"
    assert class_for_ra(0.05).name == "Sc2"
    assert class_for_ra(1.0).name == "Sc3"
    assert class_for_ra(5.0).name == "Sc4"
    assert class_for_ra(50.0).name == "Sc5"
    res = analyze_roughness(sine_profile(10.0, 20.0, length=1500, n=15_000))
    assert res.result.setting_class == "Sc1" and res.result.lambda_c_um == 80.0
    assert N.SETTING_CLASS_FROM_MEASUREMENT in [n.code for n in res.notices]


def test_short_evaluation_and_coarse_sampling_warn() -> None:
    res = analyze_roughness(
        sine_profile(10.0, 20.0, length=300, n=3_000), RoughnessSettings(lambda_c_um=80.0)
    )
    assert N.EVALUATION_TOO_SHORT in [n.code for n in res.notices]
    assert res.result.n_sections == 2
    res = analyze_roughness(
        sine_profile(10.0, 20.0, length=2000, n=1_000), RoughnessSettings(lambda_c_um=80.0)
    )
    assert N.SAMPLING_TOO_COARSE in [n.code for n in res.notices]
    res = analyze_roughness(
        sine_profile(10.0, 20.0, length=100, n=1_000), RoughnessSettings(lambda_c_um=80.0)
    )
    err = [n for n in res.notices if n.code == N.EVALUATION_TOO_SHORT]
    assert err and err[0].severity == N.ERROR


def test_step_inside_roughness_range_warns() -> None:
    prof, _ = make("edge", height_nm=200.0, noise_nm=0.5, seed=3)
    res = analyze_roughness(prof, RoughnessSettings(lambda_c_um=80.0))
    assert N.STEP_IN_ROUGHNESS in [n.code for n in res.notices]
    res = analyze_roughness(prof, RoughnessSettings(lambda_c_um=80.0, region_um=(0, 480)))
    assert N.STEP_IN_ROUGHNESS not in [n.code for n in res.notices]


def test_rsm_discrimination_merges_small_crossings() -> None:
    x = np.linspace(0, 100, 10_001)
    big = np.sin(2 * np.pi * x / 20)
    small = 0.02 * np.sin(2 * np.pi * x / 0.5)
    rz = 2.0
    assert mean_element_width(x, big + small, 0.1 * rz, 0.01 * 80) == pytest.approx(20, rel=0.02)


# ------------------------------------------------------------------ stress
def test_silicon_biaxial_moduli() -> None:
    assert pytest.approx(180.3, abs=0.1) == M_SI100
    assert pytest.approx(229.1, abs=0.1) == M_SI111
    assert all(s.source for s in SUBSTRATES.values())
    assert not SUBSTRATES["soda-lime"].verified


def test_stoney_hand_calculation() -> None:
    """σ = M h_s² / (6 h_f) Δκ: Si(100), 525 µm, 500 nm, R = ∞ -> 50 m."""
    m, hs, hf = 180.32, 525e-6, 500e-9
    dk = 1 / 50.0
    hand = m * 1e9 * hs**2 / (6 * hf) * dk / 1e6
    assert hand == pytest.approx(331.36, abs=0.05)
    assert stoney_mpa(m, 525.0, 500.0, dk * 1e-6) == pytest.approx(hand, rel=1e-12)


def _wafer(
    radius_m: float, length_um: float = 30_000.0, n: int = 6000, noise: float = 0.0, seed: int = 0
) -> Profile:
    x = np.linspace(0, length_um, n)
    k = 1.0 / (radius_m * 1e6)  # 1/µm
    z_um = 0.5 * k * (x - length_um / 2) ** 2
    rng = np.random.default_rng(seed)
    return Profile(x, z_um * 1e3 + rng.normal(scale=noise, size=n) if noise else z_um * 1e3)


def test_curvature_of_a_parabola() -> None:
    f = fit_curvature(_wafer(80.0))
    assert f.radius_m == pytest.approx(80.0, rel=1e-6)
    assert fit_curvature(_wafer(-40.0)).radius_m == pytest.approx(-40.0, rel=1e-6)


def test_stress_sign_and_value() -> None:
    pre, post = _wafer(-400.0, noise=1.0, seed=1), _wafer(60.0, noise=1.0, seed=2)
    st = StressSettings(substrate="si100", substrate_thickness_um=525.0, film_thickness_nm=400.0)
    res = analyze_stress(pre, post, st).result
    dk = 1 / 60.0 - 1 / -400.0
    expected = M_SI100 * 1e9 * (525e-6) ** 2 / (6 * 400e-9) * dk / 1e6
    assert res.stress_mpa == pytest.approx(expected, rel=2e-3)
    assert res.kind == "tensile" and res.stress_mpa > 0
    back = analyze_stress(
        pre,
        post,
        StressSettings(
            substrate="si100", substrate_thickness_um=525, film_thickness_nm=400, backside=True
        ),
    )
    assert back.result.stress_mpa == pytest.approx(-res.stress_mpa)
    comp = analyze_stress(post, pre, st).result
    assert comp.kind == "compressive"


def test_stress_assumption_warnings() -> None:
    flat = _wafer(1e9, noise=2.0, seed=3)
    st = StressSettings(substrate="si100", substrate_thickness_um=100.0, film_thickness_nm=3000.0)
    res = analyze_stress(None, flat, st)
    c = [n.code for n in res.notices]
    assert N.STRESS_NO_REFERENCE in c
    assert N.CURVATURE_CHANGE_NOT_SIGNIFICANT in c
    assert N.STONEY_THICK_FILM in c
    res = analyze_stress(
        None,
        _wafer(5.0),
        StressSettings(substrate="soda-lime", substrate_thickness_um=200, film_thickness_nm=100),
    )
    c = [n.code for n in res.notices]
    assert N.SUBSTRATE_VALUE_UNVERIFIED in c and N.STONEY_LARGE_DEFLECTION in c
    res = analyze_stress(
        None, _wafer(50.0), StressSettings(substrate="si100", film_thickness_nm=100)
    )
    assert N.SUBSTRATE_THICKNESS_TYPICAL in [n.code for n in res.notices]
    assert res.result.substrate_thickness_um == 525.0


def test_non_spherical_bow_and_mismatch() -> None:
    x = np.linspace(0, 30_000, 6000)
    t = (x - 15_000) / 15_000
    quartic = Profile(x, 500 * t**2 + 800 * t**4 + np.random.default_rng(0).normal(0, 1, x.size))
    short = _wafer(100.0, length_um=20_000, n=4000)
    st = StressSettings(substrate="si100", substrate_thickness_um=525, film_thickness_nm=200)
    c = [n.code for n in analyze_stress(short, quartic, st).notices]
    assert N.BOW_NOT_SPHERICAL in c and N.SCAN_MISMATCH in c


def test_user_modulus() -> None:
    st = StressSettings(
        substrate=None,
        youngs_modulus_gpa=100.0,
        poisson_ratio=0.25,
        substrate_thickness_um=500.0,
        film_thickness_nm=100.0,
    )
    res = analyze_stress(None, _wafer(20.0), st).result
    assert res.biaxial_modulus_gpa == pytest.approx(100 / 0.75)
    with pytest.raises(ValueError):
        analyze_stress(
            None, _wafer(20.0), StressSettings(substrate="unobtainium", film_thickness_nm=100)
        )

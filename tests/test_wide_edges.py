"""Wide, gently running-out edges and the residual-correlation model of the uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from profilometry_analyzer import notices as N
from profilometry_analyzer.analysis import StepSettings, analyze_step
from profilometry_analyzer.cli import main
from profilometry_analyzer.core.edges import detect_edges
from profilometry_analyzer.core.stats import correlation_time, fit_line
from profilometry_analyzer.core.steps import fit_plane
from profilometry_analyzer.synthetic import write_csv

from .conftest import make

OFF = StepSettings(multiscale=False, widen=False)
WIDE = {"kind": "edge", "length_um": 2000.0, "n_points": 20_000, "height_nm": 100.0}


def codes(res: object) -> set[str]:
    return {n.code for n in res.notices}  # type: ignore[attr-defined]


# ----------------------------------------------------------- correlation model
def test_correlation_time_white_noise_is_one(rng: np.random.Generator) -> None:
    pieces = [rng.normal(size=5000) for _ in range(2)]
    info = correlation_time(pieces, n_params=4)
    assert 0.9 <= info["tau"] <= 1.3
    assert info["correlated_fraction"] < 0.05


def test_correlation_time_exponential_process(rng: np.random.Generator) -> None:
    phi = 0.95  # AR(1): tau = (1 + phi) / (1 - phi) = 39
    e = rng.normal(size=60_000)
    r = np.empty_like(e)
    r[0] = e[0]
    for i in range(1, e.size):
        r[i] = phi * r[i - 1] + e[i]
    info = correlation_time([r[:30_000], r[30_000:]], n_params=4)
    assert 0.8 * 39 <= info["tau"] <= 1.4 * 39


def test_correction_for_short_planes_raises_tau(rng: np.random.Generator) -> None:
    """Gaussian-smoothed noise, sigma 200 points (tau = 2 sqrt(pi) 200 = 709), on two
    planes of 4500 points: the summed sample autocorrelation falls far short."""
    true_tau = 2 * np.sqrt(np.pi) * 200
    ratios_sum, ratios_model = [], []
    for seed in range(12):
        g = np.random.default_rng(seed)
        w = gaussian_filter1d(g.normal(size=12_000), 200)
        pieces = []
        for a in (500, 6500):
            seg = w[a : a + 4500]
            t = np.arange(seg.size)
            pieces.append(seg - np.polyval(np.polyfit(t, seg, 1), t))
        info = correlation_time(pieces, n_params=4)
        ratios_sum.append(info["tau_sum"] / true_tau)
        ratios_model.append(info["tau"] / true_tau)
    assert np.median(ratios_sum) < 0.6
    assert 0.7 <= np.median(ratios_model) <= 1.8


def test_line_with_tau_scales_covariance(rng: np.random.Generator) -> None:
    x = np.linspace(0, 100, 2000)
    line = fit_line(x, rng.normal(size=x.size))
    wider = line.with_tau(4 * line.tau)
    assert wider.cov == pytest.approx(line.cov * 4, rel=1e-9)
    assert wider.n_eff == pytest.approx(line.n_eff / 4, rel=1e-9)
    assert line.with_tau(1.0).tau == 1.0


def test_uncertainty_covers_long_correlated_roughness() -> None:
    z = []
    for seed in range(24):
        p, t = make(
            "edge",
            height_nm=100.0,
            noise_nm=0.5,
            roughness_nm=2.0,
            roughness_corr_um=20.0,
            tip_radius_um=2.0,
            seed=2000 + seed,
        )
        m = analyze_step(p).measurement
        assert m.value_nm is not None and m.u_nm
        z.append((m.value_nm - t["value_nm"]) / m.u_nm)
    za = np.abs(np.array(z))
    assert np.mean(za < 2) >= 0.85
    assert 0.5 <= float(np.std(z)) <= 1.4
    # the old per-plane sum gave std(z) of about 1.8 here


# ------------------------------------------------------------------ wide edges
def test_wide_logistic_edge_is_widened_and_unbiased() -> None:
    errs_off, errs_on = [], []
    for seed in range(4):
        p, _ = make(**WIDE, wall_um=200.0, noise_nm=1.0, tilt_nm_per_um=0.03, seed=seed)
        off = analyze_step(p, OFF).measurement
        res = analyze_step(p)
        on = res.measurement
        assert off.value_nm is not None and on.value_nm is not None and on.u_nm
        errs_off.append(off.value_nm - 100.0)
        errs_on.append(on.value_nm - 100.0)
        assert N.TRANSITION_WIDENED in codes(res)
        e = res.edges.edges[0]
        assert e.widened_um[0] > 0 and e.widened_um[1] > 0
        assert on.u_tail_nm and on.u_tail_nm > 0
        assert abs(on.value_nm - 100.0) < 3 * on.u_nm
    assert np.mean(errs_off) < -1.5  # fixed margins: the tails pull the planes together
    assert abs(np.mean(errs_on)) < 0.3


def test_wide_ramp_found_only_with_multiscale() -> None:
    p, _ = make(**WIDE, wall_um=600.0, wall_shape="ramp", noise_nm=1.0, seed=3)
    off = analyze_step(p, OFF)
    assert off.measurement.kind == "none"
    res = analyze_step(p)
    assert res.measurement.kind == "edge"
    assert N.WIDE_EDGE in codes(res)
    assert res.edges.edges[0].smooth_um is not None
    assert res.measurement.value_nm == pytest.approx(100.0, abs=0.3)


def test_very_wide_edge_warns_that_plateaus_are_not_flat() -> None:
    p, _ = make(**WIDE, wall_um=600.0, noise_nm=1.0, seed=1)
    res = analyze_step(p)
    assert res.measurement.kind == "edge"
    assert N.TRANSITION_TAIL in codes(res)
    tail = next(n for n in res.notices if n.code == N.TRANSITION_TAIL)
    assert tail.severity == N.WARNING
    assert res.measurement.value_nm is not None and res.measurement.value_nm < 100.0


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "flat", "bow_nm": 40.0, "tilt_nm_per_um": 0.05, "noise_nm": 0.5},
        {"kind": "flat", "noise_nm": 0.3, "roughness_nm": 2.0, "roughness_corr_um": 20.0},
        {"kind": "flat", "noise_nm": 1.0, "length_um": 12_000.0, "n_points": 120_000},
    ],
)
def test_multiscale_adds_no_edges_to_flat_profiles(spec: dict) -> None:
    for seed in range(3):
        p, _ = make(**spec, seed=seed, tip_radius_um=2.0)
        assert detect_edges(p.x_um, p.z_nm, p.valid, multiscale=True).edges == []


def test_sharp_edges_and_burrs_are_not_widened() -> None:
    for seed in range(5):
        p, _ = make("edge", height_nm=120.0, burr_nm=40.0, noise_nm=0.5, seed=seed)
        res = analyze_step(p)
        assert N.TRANSITION_WIDENED not in codes(res)
        assert res.measurement.u_tail_nm == 0.0
        assert res.edges.edges[0].widened_um == (0.0, 0.0)


def test_widening_can_be_switched_off() -> None:
    p, _ = make(**WIDE, wall_um=200.0, noise_nm=1.0, seed=0)
    res = analyze_step(p, StepSettings(widen=False))
    assert N.TRANSITION_WIDENED not in codes(res)
    assert res.edges.edges[0].widened_um == (0.0, 0.0)


def test_noise_point_is_not_reported_as_burr(rng: np.random.Generator) -> None:
    x = np.linspace(0, 500, 5000)
    z = rng.normal(0, 1.0, x.size)
    z[2500], z[2499], z[2501] = 6.0, 1.5, 1.5  # one outlier with two mildly raised neighbours
    pf = fit_plane(x, z, np.ones(x.size, dtype=bool), name="p")
    assert pf is not None and pf.burrs == []
    assert not pf.used[2500]  # still excluded from the fit
    z2 = z.copy()
    z2 += 12.0 * np.exp(-0.5 * ((x - 100.0) / 1.0) ** 2)  # a real burr, 20 points wide
    pf2 = fit_plane(x, z2, np.ones(x.size, dtype=bool), name="p")
    assert pf2 is not None and len(pf2.burrs) == 1


# ----------------------------------------------------------------- interfaces
def test_cli_flags_and_tail_column(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p, _ = make(**WIDE, wall_um=200.0, noise_nm=1.0, seed=0, tip_radius_um=2.0)
    f = write_csv(p, tmp_path / "wide_1.csv", metadata={"Stylus Radius": "2 um"})
    assert main(["step", str(f), "--json"]) == 0
    on = json.loads(capsys.readouterr().out)
    assert on["measurement"]["u_tail_nm"] > 0
    assert any(n["code"] == N.TRANSITION_WIDENED for n in on["notices"])
    assert main(["step", str(f), "--json", "--single-scale", "--no-widen"]) == 0
    off = json.loads(capsys.readouterr().out)
    assert off["edges"]["threshold"]["multiscale"] == 0.0
    assert off["edges"]["edges"][0]["widened_um"] == [0.0, 0.0]
    assert off["measurement"]["u_tail_nm"] == 0.0
    assert on["measurement"]["value_nm"] > off["measurement"]["value_nm"] + 1.0
    out = tmp_path / "o"
    assert main(["step", str(f), "-o", str(out), "--formats", "json", "--no-widen"]) == 0
    project = json.loads((out / "wide_1_step.json").read_text())
    assert project["settings"]["widen"] is False and project["settings"]["multiscale"] is True
    assert main(["verify", str(out / "wide_1_step.json")]) == 0

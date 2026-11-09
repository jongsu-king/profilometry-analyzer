"""Synthetic recovery of step heights (README: validation, synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from profilometry_analyzer import notices as N
from profilometry_analyzer.analysis import StepSettings, analyze_step
from profilometry_analyzer.core.despike import despike, find_spikes
from profilometry_analyzer.core.edges import classify, detect_edges
from profilometry_analyzer.core.leveling import level_auto, level_regions
from profilometry_analyzer.core.steps import stylus_reaches_bottom
from profilometry_analyzer.synthetic import SynthSpec, synthesize, trace

from .conftest import make


def codes(res) -> list[str]:  # type: ignore[no-untyped-def]
    return [n.code for n in res.notices]


# ------------------------------------------------------------------ building blocks
def test_spikes_found_and_steps_kept() -> None:
    prof, truth = make("edge", height_nm=200, noise_nm=0.5, spikes=5, seed=2)
    sp = find_spikes(prof.z_nm, prof.valid)
    assert sp.sum() == 5
    found = np.sort(prof.x_um[sp])
    assert np.allclose(found, np.sort(truth["spike_x_um"]))
    d = despike(prof)
    assert d.valid.sum() == prof.n - 5
    assert d.raw_z is not None and np.array_equal(d.raw_z, prof.raw_z)  # raw kept
    assert d.history[-1]["step"] == "despike"


def test_stylus_trace_rounds_a_groove() -> None:
    x = np.linspace(0, 100, 2001)
    z = np.where(np.abs(x - 50) < 1.0, -1000.0, 0.0)  # 2 µm wide, 1 µm deep
    t = trace(x, z, radius_um=2.0)
    assert t.min() > -1000.0  # cannot reach the floor
    assert not stylus_reaches_bottom(2.0, 1.0, 2.0)
    assert stylus_reaches_bottom(50.0, 1.0, 2.0)


@pytest.mark.parametrize(
    ("kind", "expected", "n_edges"),
    [
        ("edge", "edge", 1),
        ("groove", "groove", 2),
        ("ridge", "ridge", 2),
        ("periodic", "periodic", 20),
        ("staircase", "staircase", 2),
        ("flat", "none", 0),
    ],
)
def test_classification(kind: str, expected: str, n_edges: int) -> None:
    prof, _ = make(kind, height_nm=80, noise_nm=0.4, tip_radius_um=2.0, seed=3)
    lp, _, _ = level_auto(prof)
    es = detect_edges(lp.x_um, lp.z_nm, lp.valid, tip_radius_um=2.0)
    assert classify(es.edges) == expected
    assert len(es.edges) == n_edges


def test_auto_levelling_is_not_tilted_by_the_step() -> None:
    prof, truth = make("edge", height_nm=500, tilt_nm_per_um=0.2, noise_nm=1.0, seed=5)
    for method in ("huber", "ransac", "lsq"):
        _, lev, _ = level_auto(prof, method=method)
        assert lev.slope_nm_per_um == pytest.approx(0.2, abs=2e-4), method


def test_manual_regions_same_and_separate_levels() -> None:
    prof, _ = make("edge", height_nm=300, tilt_nm_per_um=0.1, noise_nm=0.5, seed=6)
    # both regions on the lower level: one line through them would be fine only if the
    # regions share a level; here they do not -> warning, and the separate fit is right
    lp, lev = level_regions(prof, [(50, 400), (600, 950)])
    assert N.LEVELING_REGIONS_DIFFER in [n.code for n in lev.notices]
    lp2, lev2 = level_regions(prof, [(50, 400), (600, 950)], same_level=False)
    assert lev2.slope_nm_per_um == pytest.approx(0.1, abs=5e-4)
    assert not lev2.notices
    # regions on one plane: no warning
    flat, _ = make("flat", tilt_nm_per_um=0.1, noise_nm=0.5, seed=6)
    _, lev3 = level_regions(flat, [(50, 300), (700, 950)])
    assert lev3.slope_nm_per_um == pytest.approx(0.1, abs=5e-4) and not lev3.notices


# ------------------------------------------------------------------ recovery
@pytest.mark.parametrize("seed", range(6))
def test_edge_recovered_with_tilt_noise_spikes(seed: int) -> None:
    prof, truth = make(
        "edge",
        height_nm=150.0,
        tilt_nm_per_um=0.08,
        noise_nm=1.0,
        spikes=4,
        tip_radius_um=2.0,
        seed=seed,
    )
    res = analyze_step(prof)
    m = res.measurement
    assert m.kind == "edge"
    assert m.value_nm == pytest.approx(150.0, abs=max(4 * m.u_nm, 0.05))
    assert m.u_nm < 0.2
    assert abs(m.position_um - 500.0) < 1.0
    assert N.SPIKES_REMOVED in codes(res)


def test_down_edge_is_negative() -> None:
    prof, _ = make("edge", height_nm=-90.0, noise_nm=0.5, seed=1)
    m = analyze_step(prof).measurement
    assert m.value_nm == pytest.approx(-90.0, abs=0.2)
    assert m.thickness_nm == pytest.approx(90.0, abs=0.2)


def test_bow_is_a_separate_correction() -> None:
    prof, _ = make("edge", height_nm=100.0, bow_nm=40.0, noise_nm=0.5, tip_radius_um=2.0, seed=8)
    straight = analyze_step(prof, StepSettings(bow_order=1))
    bowed = analyze_step(prof, StepSettings(bow_order=2))
    assert N.BOW_REMOVED in codes(bowed) and N.BOW_REMOVED not in codes(straight)
    assert bowed.measurement.value_nm == pytest.approx(100.0, abs=0.3)
    # the straight levelling leaves non-parallel planes and says so
    assert N.NONPARALLEL_PLANES in codes(straight)
    assert [h["step"] for h in bowed.profile.history].count("level") == 1


def test_groove_iso5436_a1_with_burrs() -> None:
    prof, truth = make(
        "groove",
        height_nm=250.0,
        width_um=180.0,
        burr_nm=80.0,
        tilt_nm_per_um=-0.05,
        noise_nm=0.8,
        tip_radius_um=2.0,
        seed=9,
    )
    res = analyze_step(prof)
    m = res.measurement
    assert m.kind == "groove" and m.method == "iso5436-a1"
    assert m.value_nm == pytest.approx(250.0, abs=0.5)
    assert m.width_um == pytest.approx(180.0, abs=2.0)
    assert N.BURR_EXCLUDED in codes(res)
    # A1 geometry: outer regions end W/3 before the edges, inner = central third
    w, c = m.width_um, m.position_um
    (l0, l1), (r0, r1) = m.regions_um["outer"]
    assert l0 == pytest.approx(c - 1.5 * w) and r1 == pytest.approx(c + 1.5 * w)
    ((i0, i1),) = m.regions_um["inner"]
    assert i1 - i0 == pytest.approx(w / 3)


def test_burr_in_plane_is_excluded_and_reported() -> None:
    # a wide burr reaching far into the upper plane
    prof, _ = make(
        "groove",
        height_nm=200.0,
        width_um=150.0,
        burr_nm=40.0,
        burr_width_um=60.0,
        noise_nm=0.5,
        tip_radius_um=2.0,
        seed=10,
    )
    res = analyze_step(prof)
    assert res.measurement.value_nm == pytest.approx(200.0, abs=1.0)
    assert N.BURR_EXCLUDED in codes(res)


def test_no_false_burr_on_clean_edge() -> None:
    for seed in range(5):
        prof, _ = make("edge", height_nm=100.0, noise_nm=0.6, tip_radius_um=2.0, seed=seed)
        assert N.BURR_EXCLUDED not in codes(analyze_step(prof))


def test_ridge_height() -> None:
    prof, _ = make(
        "ridge", height_nm=60.0, width_um=120.0, noise_nm=0.3, tip_radius_um=2.0, seed=11
    )
    m = analyze_step(prof).measurement
    assert m.kind == "ridge"
    assert m.value_nm == pytest.approx(60.0, abs=0.3)


def test_periodic_lines() -> None:
    prof, truth = make(
        "periodic", height_nm=45.0, period_um=100.0, noise_nm=0.4, tip_radius_um=2.0, seed=12
    )
    m = analyze_step(prof).measurement
    assert m.kind == "periodic"
    assert m.stats["n"] == truth["n_edges"]
    assert m.stats["mean_nm"] == pytest.approx(45.0, abs=0.2)
    assert m.stats["period_um"] == pytest.approx(100.0, abs=0.5)
    assert m.stats["std_nm"] < 0.5


def test_staircase_reports_each_step() -> None:
    prof, _ = make("staircase", height_nm=70.0, n_steps=3, noise_nm=0.4, seed=13)
    res = analyze_step(prof)
    m = res.measurement
    assert m.kind == "staircase" and m.value_nm is None
    assert [round(r["height_nm"]) for r in m.per_edge] == [70, 70, 70]
    assert N.MULTIPLE_STEPS in codes(res)


def test_forced_mode_and_manual_edges() -> None:
    prof, _ = make("staircase", height_nm=70.0, n_steps=2, noise_nm=0.4, seed=14)
    res = analyze_step(prof, StepSettings(mode="edge"))
    assert res.measurement.kind == "edge"
    manual = analyze_step(prof, StepSettings(edges_um=[333.3, 666.7]))
    assert all(e.manual for e in manual.edges.edges)
    assert [round(r["height_nm"], 1) for r in manual.measurement.per_edge] == [70.0, 70.0]
    one, _ = make("edge", height_nm=40.0, noise_nm=0.3, seed=14)
    m = analyze_step(one, StepSettings(edges_um=[503.0])).measurement
    assert m.value_nm == pytest.approx(40.0, abs=0.2)


def test_manual_regions_step() -> None:
    prof, _ = make("edge", height_nm=110.0, tilt_nm_per_um=0.03, noise_nm=0.5, seed=15)
    res = analyze_step(prof, StepSettings(regions_um=[(100, 400), (600, 900)]))
    assert res.measurement.method == "regions"
    assert res.measurement.value_nm == pytest.approx(110.0, abs=0.3)


def test_crop() -> None:
    prof, _ = make("staircase", height_nm=50.0, n_steps=2, noise_nm=0.3, seed=16)
    res = analyze_step(prof, StepSettings(crop_um=(0, 500)))
    assert res.measurement.kind == "edge"
    assert res.profile.x_um.max() <= 500


def test_narrow_groove_warning() -> None:
    prof, _ = make(
        "groove",
        height_nm=800.0,
        width_um=6.0,
        wall_um=0.3,
        noise_nm=0.5,
        tip_radius_um=2.0,
        n_points=20_000,
        length_um=400,
        seed=17,
    )
    res = analyze_step(prof, StepSettings(mode="groove", min_plane_um=0.5, margin_um=0.5))
    assert N.NARROW_FEATURE in codes(res)


def test_small_step_and_noise_warnings() -> None:
    prof, _ = make("edge", height_nm=25.0, noise_nm=0.3, seed=18)
    res = analyze_step(prof)
    assert N.SMALL_STEP in codes(res)
    prof, _ = make("edge", height_nm=8.0, noise_nm=0.3, seed=18)
    res = analyze_step(prof)
    assert N.SMALL_STEP in codes(res)


def test_no_step() -> None:
    prof, _ = make("flat", noise_nm=0.5, seed=19)
    res = analyze_step(prof)
    assert res.measurement.kind == "none" and res.thickness_nm is None
    assert N.NO_STEP_FOUND in codes(res)


def test_soft_film_force_checks() -> None:
    prof, _ = make("edge", height_nm=100.0, noise_nm=0.5, seed=20)
    res = analyze_step(prof, StepSettings(soft_film=True, stylus_force_mg=10.0))
    assert N.SOFT_FILM_HIGH_FORCE in codes(res)
    res = analyze_step(prof, StepSettings(soft_film=True, stylus_force_mg=2.0))
    assert N.SOFT_FILM_HIGH_FORCE not in codes(res)
    res = analyze_step(prof, StepSettings(soft_film=True))
    assert N.SOFT_FILM_FORCE_UNKNOWN in codes(res)


def test_tip_radius_default_noted() -> None:
    prof, _ = make("edge", height_nm=100.0, noise_nm=0.5, seed=21)
    res = analyze_step(prof)
    assert N.TIP_RADIUS_DEFAULT in codes(res) and res.tip_radius_um == 2.0
    res = analyze_step(prof, StepSettings(tip_radius_um=1.25))
    assert N.TIP_RADIUS_DEFAULT not in codes(res) and res.tip_radius_um == 1.25


def test_input_profile_is_not_modified() -> None:
    prof, _ = make("edge", height_nm=100.0, tilt_nm_per_um=0.1, noise_nm=0.5, spikes=2, seed=22)
    z0 = prof.z_nm.copy()
    analyze_step(prof)
    assert np.array_equal(prof.z_nm, z0) and prof.valid.all()


def test_uncertainty_coverage_white_noise() -> None:
    """|error| < 2u for about 95 % of scans with white noise (40 seeds)."""
    z = []
    for seed in range(40):
        syn = synthesize(
            SynthSpec(
                kind="edge",
                height_nm=100,
                tilt_nm_per_um=0.05,
                noise_nm=1.0,
                tip_radius_um=2.0,
                seed=100 + seed,
            )
        )
        m = analyze_step(syn.profile).measurement
        z.append((m.value_nm - 100.0) / m.u_nm)
    z = np.abs(np.array(z))
    assert 0.85 <= float(np.mean(z < 2)) <= 1.0
    assert 0.6 < float(np.std(z)) * 0 + float(np.sqrt(np.mean(z**2))) < 1.5


def test_correlated_roughness_warns() -> None:
    prof, _ = make(
        "edge", height_nm=100.0, noise_nm=0.2, roughness_nm=2.0, roughness_corr_um=20.0, seed=23
    )
    assert N.CORRELATED_RESIDUALS in codes(analyze_step(prof))

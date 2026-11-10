"""Stylus shape (sphere on a cone), the points it cannot touch, and u_tail of A1 depths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from profilometry_analyzer import notices as N
from profilometry_analyzer.analysis import StepSettings, analyze_roughness, analyze_step
from profilometry_analyzer.cli import main
from profilometry_analyzer.core.stats import runs
from profilometry_analyzer.core.steps import stylus_reaches_bottom
from profilometry_analyzer.core.tip import Tip, check_tip, dilate, erode
from profilometry_analyzer.readers import read_profile
from profilometry_analyzer.synthetic import trace, write_csv

from .conftest import make

TIP = Tip(2.0, 60.0)


def codes(res: object) -> set[str]:
    return {n.code for n in res.notices}  # type: ignore[attr-defined]


def _sphere_trace(x: np.ndarray, z: np.ndarray, r: float) -> np.ndarray:
    """The sphere-only tracing of the first version, kept here as the reference."""
    dx = float(np.median(np.diff(x)))
    m = int(np.ceil(r / dx))
    out = np.full_like(z, -np.inf)
    for k in range(-m, m + 1):
        u = k * dx
        if abs(u) > r:
            continue
        lo, hi = max(0, -k), min(z.size, z.size - k)
        cand = np.full(z.size, -np.inf)
        cand[lo:hi] = z[lo + k : hi + k] - (r - np.sqrt(r * r - u * u)) * 1e3
        np.maximum(out, cand, out=out)
    return out


# ------------------------------------------------------------------- geometry
def test_tip_geometry() -> None:
    assert TIP.tangent_um == pytest.approx(np.sqrt(3.0))
    assert TIP.tangent_depth_nm == pytest.approx(1000.0)  # r (1 - sin 30°)
    assert TIP.flank_slope == pytest.approx(1e3 * np.sqrt(3.0))
    # the surface is continuous where the sphere meets the cone, and the inverse holds
    u = TIP.tangent_um
    assert TIP.drop_nm(u - 1e-9) == pytest.approx(TIP.drop_nm(u + 1e-9), abs=1e-4)
    for depth in (50.0, 999.0, 1001.0, 5000.0):
        assert TIP.drop_nm(TIP.half_width_um(depth)) == pytest.approx(depth, rel=1e-9)
    assert TIP.half_width_um(5000.0) == pytest.approx(np.sqrt(3) + 4.0 / np.sqrt(3))
    sphere = Tip(2.0, 0.0)
    assert sphere.half_width_um(5000.0) == 2.0 and not sphere.has_cone
    with pytest.raises(ValueError):
        Tip(2.0, 180.0)


def test_floor_reach_with_cone() -> None:
    # 5 µm deep trench, 2 µm stylus: the sphere alone would need 4 µm, the cone 8.08 µm
    assert stylus_reaches_bottom(6.0, 5.0, 2.0)  # sphere only (first version)
    assert not stylus_reaches_bottom(6.0, 5.0, 2.0, 60.0)
    assert stylus_reaches_bottom(8.2, 5.0, 2.0, 60.0)
    # shallow grooves do not see the cone
    for w in (0.5, 1.0, 2.0):
        assert stylus_reaches_bottom(w, 0.1, 2.0) == stylus_reaches_bottom(w, 0.1, 2.0, 60.0)


def test_sphere_tracing_unchanged_and_cone_matters_only_deep(rng: np.random.Generator) -> None:
    x = np.linspace(0, 200, 4001)
    z = gaussian_filter1d(rng.normal(size=x.size), 5) * 300.0
    z[1800:2200] -= 3000.0
    assert np.allclose(trace(x, z, 2.0), _sphere_trace(x, z, 2.0))
    shallow = np.where(x > 100, 600.0, 0.0)
    assert np.allclose(trace(x, shallow, 2.0, 60.0), trace(x, shallow, 2.0))
    deep = np.where(x > 100, 5000.0, 0.0)
    assert not np.allclose(trace(x, deep, 2.0, 60.0), trace(x, deep, 2.0))


def test_erosion_bounds_the_surface(rng: np.random.Generator) -> None:
    x = np.linspace(0, 300, 6001)
    s = gaussian_filter1d(rng.normal(size=x.size), 4) * 400.0
    s[3000:3100] -= 2500.0
    t = dilate(x, s, TIP)
    r = erode(x, t, TIP)
    assert np.all(s <= r + 1e-9) and np.all(r <= t + 1e-9)
    assert np.allclose(dilate(x, r, TIP), t)  # the record is reproduced exactly


# ---------------------------------------------------------------- reach check
def _true_hidden(x: np.ndarray, s: np.ndarray, tip: Tip) -> np.ndarray:
    """Points of the true surface that are never the contact point."""
    dx = float(x[1] - x[0])
    hit = np.zeros(s.size, dtype=bool)
    best = np.full(s.size, -np.inf)
    arg = np.zeros(s.size, dtype=int)
    m = int(np.ceil(tip.half_width_um(float(np.ptp(s))) / dx)) + 1
    for k in range(-m, m + 1):
        d = float(tip.drop_nm(k * dx))
        g = np.full(s.size, -np.inf)
        lo, hi = max(0, -k), min(s.size, s.size - k)
        g[lo:hi] = s[lo + k : hi + k] - d
        better = g > best
        best, arg = np.where(better, g, best), np.where(better, k, arg)
    idx = np.arange(s.size) + arg
    hit[idx[(idx >= 0) & (idx < s.size)]] = True
    out = np.zeros(s.size, dtype=bool)
    for a, b in runs(~hit):
        if b - a >= 3:
            out[a:b] = True
    return out


def test_hidden_foot_of_a_wall_matches_the_geometry(rng: np.random.Generator) -> None:
    x = np.linspace(0, 200, 4001)
    for height in (100.0, 5000.0):
        t = dilate(x, np.where(x > 100, -height, 0.0), TIP) + rng.normal(0, 1.0, x.size)
        c = check_tip(x, t, TIP)
        assert len(c.unreached_um) == 1
        a, b = c.unreached_um[0]
        assert 99.9 <= a <= 100.5
        assert b == pytest.approx(100 + TIP.half_width_um(height), abs=0.3)


def test_no_hidden_points_on_noisy_flat_or_gentle_surfaces(rng: np.random.Generator) -> None:
    x = np.linspace(0, 1000, 20_001)
    for sigma in (0.3, 1.0, 3.0):
        assert check_tip(x, rng.normal(0, sigma, x.size), TIP).unreached_um == []
    s = gaussian_filter1d(rng.normal(size=x.size), 20) * 50.0  # 1 µm correlation
    s *= 50.0 / np.std(s)
    assert check_tip(x, dilate(x, s, TIP) + rng.normal(0, 0.5, x.size), TIP).unreached_um == []


def test_rough_surface_hidden_points_are_conservative(rng: np.random.Generator) -> None:
    x = np.linspace(0, 600, 12_001)
    w = gaussian_filter1d(rng.normal(size=x.size), 6)  # 0.3 µm correlation
    s = 100.0 * w / np.std(w)
    truth = _true_hidden(x, s, TIP)
    est = check_tip(x, dilate(x, s, TIP) + rng.normal(0, 0.5, x.size), TIP).region_mask()
    assert truth.mean() > 0.3
    assert np.mean(est & ~truth) < 0.02  # hardly any point called hidden that was touched
    assert 0.6 * truth.mean() <= est.mean() <= truth.mean() + 0.02


def test_narrow_trench_is_one_hidden_region(rng: np.random.Generator) -> None:
    x = np.linspace(0, 300, 6001)
    s = np.where(np.abs(x - 150) < 4.0, -5000.0, 0.0)
    c = check_tip(x, dilate(x, s, TIP) + rng.normal(0, 1.0, x.size), TIP)
    assert len(c.unreached_um) == 1
    a, b = c.unreached_um[0]
    assert a == pytest.approx(146.0, abs=0.5) and b == pytest.approx(154.0, abs=0.5)


# ------------------------------------------------------------------- analysis
def test_film_edge_has_no_stylus_notices() -> None:
    p, _ = make("edge", height_nm=120.0, noise_nm=0.5, tip_radius_um=2.0, seed=1)
    res = analyze_step(p)
    assert res.tip_check is not None and res.tip_check.unreached_um == []
    assert not codes(res) & {N.TIP_UNREACHED, N.TIP_LIMITED_WALL, N.CONE_ANGLE_DEFAULT}
    assert res.cone_angle_deg == 60.0 and res.cone_angle_source == "instrument default"


def test_tall_step_reports_hidden_foot_and_flank() -> None:
    p, _ = make("edge", height_nm=5000.0, wall_um=0.05, noise_nm=1.0, tip_radius_um=2.0, seed=2)
    res = analyze_step(p)
    m = res.measurement
    assert m.value_nm is not None and m.u_nm
    assert abs(m.value_nm - 5000.0) < 3 * m.u_nm + 0.5  # the height is not affected
    assert {N.TIP_UNREACHED, N.TIP_LIMITED_WALL, N.CONE_ANGLE_DEFAULT} <= codes(res)
    hidden = next(n for n in res.notices if n.code == N.TIP_UNREACHED)
    assert hidden.severity == N.INFO and "floor" not in hidden.message  # foot of the wall
    (a, b), *_ = res.tip_check.unreached_um  # type: ignore[union-attr]
    assert b - a == pytest.approx(TIP.half_width_um(5000.0), abs=0.6)
    res2 = analyze_step(p, StepSettings(cone_angle_deg=90.0))
    assert res2.cone_angle_deg == 90.0 and res2.cone_angle_source == "user"
    assert N.CONE_ANGLE_DEFAULT not in codes(res2)
    # recorded with a 60° cone, the wall is steeper than a 90° cone could follow
    wall = next(n for n in res2.notices if n.code == N.TIP_LIMITED_WALL)
    assert wall.severity == N.WARNING and "sharper than assumed" in wall.message
    off = analyze_step(p, StepSettings(tip_check=False))
    assert off.tip_check is None and N.TIP_UNREACHED not in codes(off)
    assert off.measurement.value_nm == m.value_nm


def test_floor_region_the_stylus_cannot_reach_warns() -> None:
    p, _ = make(
        "groove",
        height_nm=5000.0,
        width_um=10.0,
        wall_um=0.05,
        noise_nm=1.0,
        tip_radius_um=2.0,
        length_um=300.0,
        n_points=6000,
        seed=3,
    )
    res = analyze_step(p, StepSettings(regions_um=[(100.0, 130.0), (148.0, 152.0)]))
    w = [n for n in res.notices if n.code == N.TIP_UNREACHED]
    assert w and w[0].severity == N.WARNING and w[0].data["plane_percent"] > 20
    assert abs(res.measurement.value_nm or 0.0) < 4900.0  # the floor was not reached


def test_floor_of_a_narrow_trench_is_reported_as_not_reached() -> None:
    # 5 µm wide: the 2 µm, 60° stylus hangs on the rims with its apex drop(2.5 µm) down
    p, _ = make(
        "groove",
        height_nm=3000.0,
        width_um=5.0,
        wall_um=0.05,
        noise_nm=0.5,
        tip_radius_um=2.0,
        length_um=300.0,
        n_points=6000,
        seed=4,
    )
    res = analyze_step(p)
    floor = [n for n in res.notices if n.code == N.TIP_UNREACHED and "floor" in n.message]
    assert floor and "60° cone" in floor[0].message
    hanging = float(TIP.drop_nm(2.5))  # 2330 nm instead of 3000 nm
    assert floor[0].data["recorded_depth_nm"] == pytest.approx(hanging, abs=40.0)
    assert not stylus_reaches_bottom(5.0, 3.0, 2.0, 60.0)


def test_roughness_reports_hidden_valleys() -> None:
    p, _ = make(
        "flat",
        length_um=1500.0,
        n_points=30_000,
        roughness_nm=100.0,
        roughness_corr_um=0.3,
        noise_nm=0.5,
        tip_radius_um=2.0,
        seed=5,
    )
    res = analyze_roughness(p)
    assert res.result.stylus["unreached_percent"] > 10
    assert N.TIP_UNREACHED in codes(res)
    gentle, _ = make("sine", sine_amplitude_nm=12.0, sine_wavelength_um=40.0, tip_radius_um=2.0)
    r2 = analyze_roughness(gentle)
    assert r2.result.stylus["unreached_percent"] == 0.0 and N.TIP_UNREACHED not in codes(r2)


# ------------------------------------------------------------------ A1 u_tail
def _wide_groove(wall: float, seed: int, width: float = 400.0) -> object:
    p, _ = make(
        "groove",
        height_nm=100.0,
        width_um=width,
        wall_um=wall,
        length_um=2000.0,
        n_points=20_000,
        noise_nm=1.0,
        tip_radius_um=2.0,
        seed=seed,
    )
    return analyze_step(p)


def test_a1_depth_gets_tail_uncertainty() -> None:
    z = []
    for seed in range(6):
        res = _wide_groove(100.0, seed)
        m = res.measurement  # type: ignore[attr-defined]
        assert m.method == "iso5436-a1" and m.value_nm is not None and m.u_nm
        assert N.TRANSITION_WIDENED in codes(res)
        assert m.u_tail_nm is not None and m.u_tail_nm > 0.0
        z.append((m.value_nm - 100.0) / m.u_nm)
    assert np.mean(np.abs(z) < 2) >= 5 / 6
    sharp, _ = make("groove", height_nm=240.0, width_um=150.0, noise_nm=0.8, seed=4)
    assert analyze_step(sharp).measurement.u_tail_nm == 0.0


def test_a1_floor_that_still_bends_warns_and_widens_u() -> None:
    warned = 0
    for seed in range(6):
        res = _wide_groove(150.0, seed)
        m = res.measurement  # type: ignore[attr-defined]
        assert m.value_nm < 100.0 and abs(m.value_nm - 100.0) < 2 * m.u_nm  # ~1 nm too shallow
        bend = [n for n in res.notices if n.code == N.A1_BOTTOM_NOT_FLAT]  # type: ignore[attr-defined]
        if bend:
            warned += 1
            assert "bends" in bend[0].message and m.u_tail_nm >= bend[0].data["bend_nm"]
    assert warned >= 5


def test_a1_bow_left_in_the_profile_warns() -> None:
    p, _ = make("groove", height_nm=100.0, width_um=200.0, noise_nm=0.5, bow_nm=40.0, seed=1)
    res = analyze_step(p)
    assert N.A1_OUTER_CURVED in codes(res)
    assert res.measurement.value_nm == pytest.approx(108.9, abs=0.5)  # the bow's doing
    res2 = analyze_step(p, StepSettings(bow_order=2))
    assert N.A1_OUTER_CURVED not in codes(res2)
    assert res2.measurement.value_nm == pytest.approx(100.0, abs=0.1)


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "groove", "height_nm": 250.0, "width_um": 180.0, "burr_nm": 80.0},
        {"kind": "ridge", "height_nm": 60.0, "width_um": 120.0},
        {"kind": "groove", "height_nm": 100.0, "width_um": 300.0, "roughness_nm": 2.0},
    ],
)
def test_no_curvature_warnings_on_flat_levels(spec: dict) -> None:
    for seed in range(5):
        p, _ = make(**spec, noise_nm=0.5, tip_radius_um=2.0, seed=seed)
        res = analyze_step(p)
        assert not codes(res) & {N.A1_OUTER_CURVED}
        assert not any(n.code == N.A1_BOTTOM_NOT_FLAT and "bends" in n.message for n in res.notices)


# ----------------------------------------------------------------- interfaces
def test_cone_angle_from_file_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p, _ = make("edge", height_nm=5000.0, wall_um=0.05, noise_nm=1.0, tip_radius_um=2.0, seed=2)
    meta = {"Stylus Radius": "2 um", "Stylus Cone Angle": "60 deg"}
    f = write_csv(p, tmp_path / "tall_1.csv", metadata=meta)
    prof = read_profile(f)
    assert prof.scan.cone_angle_deg == pytest.approx(60.0)
    assert main(["step", str(f), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cone_angle_source"].startswith("file")
    assert out["tip_check"]["unreached_um"]
    assert any(n["code"] == N.TIP_LIMITED_WALL for n in out["notices"])
    assert main(["step", str(f), "--json", "--cone-angle", "90", "--no-tip-check"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cone_angle_deg"] == 90.0 and out["tip_check"] is None
    o = tmp_path / "o"
    assert main(["step", str(f), "-o", str(o), "--formats", "json,png", "--cone-angle", "90"]) == 0
    capsys.readouterr()
    assert (o / "tall_1_step.png").exists()
    assert main(["verify", str(o / "tall_1_step.json")]) == 0
    assert main(["roughness", str(f), "--cone-angle", "90", "--range", "0:450"]) == 0
    assert "90° cone" in capsys.readouterr().out

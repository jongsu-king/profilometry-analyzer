"""Synthetic and analytical validation behind the README table.

    python examples/validate_synthetic.py            # prints a Markdown table

Every case is generated with a known truth (step height, tilt, bow, noise, burrs,
spikes, stylus radius) by ``profilometry_analyzer.synthetic`` and analysed with the
default settings unless stated.  These checks say nothing about real instrument data.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from profilometry_analyzer.analysis import StepSettings, analyze_step
from profilometry_analyzer.core.filters import mean_line, transmission
from profilometry_analyzer.core.roughness import RoughnessSettings, roughness
from profilometry_analyzer.core.stats import runs
from profilometry_analyzer.core.stress import M_SI100, stoney_mpa
from profilometry_analyzer.core.tip import Tip, check_tip, dilate
from profilometry_analyzer.profile import Profile
from profilometry_analyzer.synthetic import SynthSpec, synthesize

N_SEEDS = 30


FIXED = StepSettings(multiscale=False, widen=False)
"""The detection of the first version: one derivative smoothing, fixed edge margins."""


TAIL_WARNINGS: list[int] = []


def _errors(truth: float, settings: StepSettings | None, kw: dict) -> tuple[np.ndarray, np.ndarray]:
    err, z = [], []
    TAIL_WARNINGS.clear()
    for seed in range(N_SEEDS):
        syn = synthesize(SynthSpec(seed=1000 + seed, tip_radius_um=2.0, **kw))
        res = analyze_step(syn.profile, settings)
        TAIL_WARNINGS.append(int(any(n.code == "TRANSITION_TAIL" for n in res.notices)))
        m = res.measurement
        if m.value_nm is None:
            err.append(math.nan)
            z.append(math.nan)
            continue
        e = m.value_nm - truth
        err.append(e)
        z.append(abs(e) / m.u_nm if m.u_nm else math.nan)
    return np.array(err), np.array(z)


def _cells(err: np.ndarray, z: np.ndarray) -> str:
    found = np.isfinite(err)
    if not found.any():
        return "not found | - | -"
    miss = "" if found.all() else f" ({int((~found).sum())} not found)"
    return (
        f"{np.nanmean(err):+.3f}{miss} | {np.nanmax(np.abs(err)):.3f} | {np.mean(z[found] < 2):.0%}"
    )


def step_case(name: str, truth: float, settings: StepSettings | None = None, **kw: object) -> str:
    err, z = _errors(truth, settings, kw)
    return f"| {name} | {truth:g} | {_cells(err, z)} |"


def wide_case(name: str, truth: float, **kw: object) -> str:
    spec = {"kind": "edge", "length_um": 2000.0, "n_points": 20_000, "height_nm": truth, **kw}
    before = _cells(*_errors(truth, FIXED, spec))
    after = _cells(*_errors(truth, None, spec))
    return f"| {name} | {truth:g} | {before} | {after} | {sum(TAIL_WARNINGS)}/{N_SEEDS} |"


TIP = Tip(2.0, 60.0)


def tall_step_case(height: float) -> str:
    """Vertical step traced by a 2 µm, 60° stylus: hidden foot and height."""
    foot, err, z, flank = [], [], [], 0
    for seed in range(N_SEEDS):
        syn = synthesize(
            SynthSpec(
                kind="edge",
                height_nm=height,
                wall_um=0.05,
                noise_nm=1.0,
                tip_radius_um=2.0,
                seed=3000 + seed,
            )
        )
        res = analyze_step(syn.profile)
        m = res.measurement
        regions = res.tip_check.unreached_um if res.tip_check else []
        foot.append(max((b - a for a, b in regions), default=0.0))
        flank += any(n.code == "TIP_LIMITED_WALL" for n in res.notices)
        if m.value_nm is not None and m.u_nm:
            err.append(m.value_nm - height)
            z.append(abs(m.value_nm - height) / m.u_nm)
    f = np.array(foot)
    return (
        f"| {height:g} | {TIP.half_width_um(height):.2f} | {f.mean():.2f} ± {f.std(ddof=1):.2f} |"
        f" {np.mean(err):+.3f} | {np.mean(np.array(z) < 2):.0%} | {flank}/{N_SEEDS} |"
    )


def trench_case(width: float, depth: float = 3000.0, seeds: int = 10) -> str:
    """Narrow trench: does the program say the floor was not reached, and how deep did
    the stylus record it?"""
    expect = min(depth, float(TIP.drop_nm(width / 2)))
    flagged, rec = 0, []
    for seed in range(seeds):
        syn = synthesize(
            SynthSpec(
                kind="groove",
                height_nm=depth,
                width_um=width,
                wall_um=0.05,
                noise_nm=0.5,
                tip_radius_um=2.0,
                length_um=300.0,
                n_points=6000,
                seed=4000 + seed,
            )
        )
        res = analyze_step(syn.profile)
        floor = [n for n in res.notices if n.code == "TIP_UNREACHED" and "floor" in n.message]
        flagged += bool(floor)
        if floor:
            rec.append(floor[0].data["recorded_depth_nm"])
    reach = "yes" if TIP.reaches_floor(width, depth) else "no"
    got = f"{np.mean(rec):.0f}" if rec else "-"
    return f"| {width:g} | {reach} | {expect:.0f} | {got} | {flagged}/{seeds} |"


def _true_hidden(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Points of the true surface never the contact point of the stylus."""
    dx = float(x[1] - x[0])
    best = np.full(s.size, -np.inf)
    arg = np.zeros(s.size, dtype=int)
    m = int(np.ceil(TIP.half_width_um(float(np.ptp(s))) / dx)) + 1
    for k in range(-m, m + 1):
        g = np.full(s.size, -np.inf)
        lo, hi = max(0, -k), min(s.size, s.size - k)
        g[lo:hi] = s[lo + k : hi + k] - float(TIP.drop_nm(k * dx))
        better = g > best
        best, arg = np.where(better, g, best), np.where(better, k, arg)
    hit = np.zeros(s.size, dtype=bool)
    idx = np.arange(s.size) + arg
    hit[idx[(idx >= 0) & (idx < s.size)]] = True
    out = np.zeros(s.size, dtype=bool)
    for a, b in runs(~hit):
        if b - a >= 3:
            out[a:b] = True
    return out


def rough_reach_case(corr: float, rms: float, seeds: int = 3) -> str:
    x = np.linspace(0, 1500, 30_001)
    dx = float(x[1] - x[0])
    true_h, est_h, false_h = [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(5000 + seed)
        w = gaussian_filter1d(rng.normal(size=x.size), corr / dx)
        s = rms * w / np.std(w)
        truth = _true_hidden(x, s)
        est = check_tip(x, dilate(x, s, TIP) + rng.normal(0, 0.5, x.size), TIP).region_mask()
        true_h.append(truth.mean())
        est_h.append(est.mean())
        false_h.append(np.mean(est & ~truth))
    return (
        f"| {rms:g} nm, {corr:g} µm | {100 * np.mean(true_h):.1f} % |"
        f" {100 * np.mean(est_h):.1f} % | {100 * np.mean(false_h):.2f} % |"
    )


def a1_tail_case(width: float, wall: float) -> str:
    """A1 groove with gently sloping walls: u with and without u_tail, and how often the
    floor was found still bending."""
    err, z_with, z_without, tails, bent = [], [], [], [], 0
    for seed in range(N_SEEDS):
        syn = synthesize(
            SynthSpec(
                kind="groove",
                height_nm=100.0,
                width_um=width,
                wall_um=wall,
                length_um=2000.0,
                n_points=20_000,
                noise_nm=1.0,
                tip_radius_um=2.0,
                seed=6000 + seed,
            )
        )
        res = analyze_step(syn.profile)
        m = res.measurement
        bent += any(n.code == "A1_BOTTOM_NOT_FLAT" for n in res.notices)
        if m.value_nm is None or not m.u_nm:
            continue
        e = m.value_nm - 100.0
        tail = m.u_tail_nm or 0.0
        err.append(e)
        tails.append(tail)
        z_with.append(abs(e) / m.u_nm)
        z_without.append(abs(e) / math.sqrt(max(m.u_nm**2 - tail**2, 1e-30)))
    return (
        f"| W {width:g} µm, walls {wall:g} µm | {np.mean(err):+.3f} |"
        f" {np.max(np.abs(err)):.3f} | {np.mean(tails):.3f} |"
        f" {np.mean(np.array(z_without) < 2):.0%} | {np.mean(np.array(z_with) < 2):.0%} |"
        f" {bent}/{N_SEEDS} |"
    )


def bow_groove_case(order: int) -> str:
    err, warned = [], 0
    for seed in range(N_SEEDS):
        syn = synthesize(
            SynthSpec(
                kind="groove",
                height_nm=100.0,
                width_um=200.0,
                noise_nm=0.5,
                bow_nm=40.0,
                tip_radius_um=2.0,
                seed=1000 + seed,
            )
        )
        res = analyze_step(syn.profile, StepSettings(bow_order=order))
        warned += any(n.code == "A1_OUTER_CURVED" for n in res.notices)
        if res.measurement.value_nm is not None:
            err.append(res.measurement.value_nm - 100.0)
    return f"| {order} | {np.mean(err):+.3f} | {warned}/{N_SEEDS} |"


def main() -> None:
    print(f"Step recovery, {N_SEEDS} noise realisations each (nm):\n")
    print("| case | true (nm) | mean error | max abs error | share with error < 2u |")
    print("|---|---|---|---|---|")
    print(
        step_case(
            "film edge, tilt 0.08 nm/µm, noise 1 nm, 4 spikes",
            150.0,
            kind="edge",
            height_nm=150.0,
            tilt_nm_per_um=0.08,
            noise_nm=1.0,
            spikes=4,
        )
    )
    print(
        step_case(
            "film edge, bow 40 nm (bow order 2)",
            100.0,
            StepSettings(bow_order=2),
            kind="edge",
            height_nm=100.0,
            bow_nm=40.0,
            noise_nm=0.5,
        )
    )
    print(
        step_case(
            "scratch (A1), W 180 µm, burrs 80 nm, tilt",
            250.0,
            kind="groove",
            height_nm=250.0,
            width_um=180.0,
            burr_nm=80.0,
            tilt_nm_per_um=-0.05,
            noise_nm=0.8,
        )
    )
    print(
        step_case(
            "ridge (A1), W 120 µm", 60.0, kind="ridge", height_nm=60.0, width_um=120.0, noise_nm=0.3
        )
    )
    print(
        step_case(
            "periodic lines, period 100 µm",
            45.0,
            kind="periodic",
            height_nm=45.0,
            period_um=100.0,
            noise_nm=0.4,
        )
    )
    for corr in (5.0, 20.0):
        print(
            step_case(
                f"film edge, correlated roughness Rq 2 nm ({corr:g} µm)",
                100.0,
                kind="edge",
                height_nm=100.0,
                noise_nm=0.5,
                roughness_nm=2.0,
                roughness_corr_um=corr,
            )
        )
    print(
        "\nWide, gently running-out film edges in a 2 mm scan (noise 1 nm, tilt"
        " 0.03 nm/µm); first version (one smoothing, fixed margins) against the default"
        " (multi-scale detection, widened transitions):\n"
    )
    print(
        "| wall | true (nm) | fixed: mean error | max abs | < 2u |"
        " default: mean error | max abs | < 2u | TRANSITION_TAIL warning |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    common = {"noise_nm": 1.0, "tilt_nm_per_um": 0.03}
    for wall in (50.0, 100.0, 200.0, 400.0):
        print(wide_case(f"logistic, 10-90 % over {wall:g} µm", 100.0, wall_um=wall, **common))
    print(wide_case("straight ramp, 600 µm", 100.0, wall_um=600.0, wall_shape="ramp", **common))
    print(
        "\nA1 grooves 100 nm deep with gently sloping walls (logistic, 2 mm scan, noise 1 nm):"
        " u with and without u_tail:\n"
    )
    print(
        "| groove | mean error | max abs | mean u_tail | < 2u without u_tail | < 2u with |"
        " A1_BOTTOM_NOT_FLAT |"
    )
    print("|---|---|---|---|---|---|---|")
    for width, wall in ((400.0, 100.0), (400.0, 150.0), (600.0, 250.0)):
        print(a1_tail_case(width, wall))
    print("\nGroove (W 200 µm, 100 nm) in a scan with a 40 nm bow:\n")
    print("| bow order | mean error | A1_OUTER_CURVED |")
    print("|---|---|---|")
    for order in (1, 2):
        print(bow_groove_case(order))
    print(
        "\nStylus 2 µm radius, 60° cone. Vertical steps (noise 1 nm): hidden foot next to"
        " the wall and the height:\n"
    )
    print(
        "| height (nm) | hidden foot, theory (µm) | found (µm) | height mean error (nm) |"
        " < 2u | TIP_LIMITED_WALL |"
    )
    print("|---|---|---|---|---|---|")
    for h in (100.0, 1000.0, 5000.0):
        print(tall_step_case(h))
    print("\nTrenches 3000 nm deep (noise 0.5 nm, 10 seeds): floor reached and recorded depth:\n")
    print("| width (µm) | floor reachable | recorded depth, theory (nm) | reported | flagged |")
    print("|---|---|---|---|---|")
    for w in (4.0, 5.0, 5.5, 7.0):
        print(trench_case(w))
    print(
        "\nPoints the stylus did not touch on random rough surfaces (1.5 mm, noise 0.5 nm,"
        " 3 seeds), against the true contact points:\n"
    )
    print("| roughness Rq, correlation | truly hidden | found hidden | found but touched |")
    print("|---|---|---|---|")
    for corr, rms in ((0.3, 100.0), (0.5, 60.0), (1.0, 50.0)):
        print(rough_reach_case(corr, rms))
    print("\nAnalytical checks:\n")
    A, lam = 10.0, 20.0
    x = np.linspace(0, 6000, 60_000)
    r = roughness(
        Profile(x, A * np.sin(2 * np.pi * x / lam)),
        RoughnessSettings(lambda_c_um=800.0, lambda_s_um=0),
    ).parameters
    print(
        f"- sine A = {A} nm: Ra = {r['Ra']:.5f} (2A/π = {2 * A / math.pi:.5f}),"
        f" Rq = {r['Rq']:.5f} (A/√2 = {A / math.sqrt(2):.5f}),"
        f" Rsm = {r['Rsm']:.4f} µm (λ = {lam})"
    )
    for lc in (80.0, 800.0):
        dx = lc / 400
        xx = np.arange(0, 20 * lc, dx)
        m = mean_line(np.sin(2 * np.pi * xx / lc), dx, lc)
        core = (xx > 2 * lc) & (xx < 18 * lc)
        amp = float(np.sqrt(2 * np.mean(m[core] ** 2)))
        print(
            f"- Gaussian mean line at λ = λc = {lc:g} µm: transmission {amp:.4f}"
            f" (theory {float(transmission(lc, lc)):.4f})"
        )
    hand = M_SI100 * 1e9 * (525e-6) ** 2 / (6 * 500e-9) * (1 / 50.0) / 1e6
    prog = stoney_mpa(M_SI100, 525.0, 500.0, 1 / 50.0 * 1e-6)
    print(
        f"- Stoney, Si(100) M = {M_SI100:.2f} GPa, h_s 525 µm, h_f 500 nm, R ∞ -> 50 m:"
        f" program {prog:.3f} MPa, hand {hand:.3f} MPa"
    )


if __name__ == "__main__":
    main()

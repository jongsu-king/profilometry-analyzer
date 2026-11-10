"""Synthetic and analytical validation behind the README table.

    python examples/validate_synthetic.py            # prints a Markdown table

Every case is generated with a known truth (step height, tilt, bow, noise, burrs,
spikes, stylus radius) by ``profilometry_analyzer.synthetic`` and analysed with the
default settings unless stated.  These checks say nothing about real instrument data.
"""

from __future__ import annotations

import math

import numpy as np

from profilometry_analyzer.analysis import StepSettings, analyze_step
from profilometry_analyzer.core.filters import mean_line, transmission
from profilometry_analyzer.core.roughness import RoughnessSettings, roughness
from profilometry_analyzer.core.stress import M_SI100, stoney_mpa
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

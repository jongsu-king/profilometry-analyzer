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


def step_case(name: str, truth: float, settings: StepSettings | None = None, **kw: object) -> str:
    err, z = [], []
    for seed in range(N_SEEDS):
        syn = synthesize(SynthSpec(seed=1000 + seed, tip_radius_um=2.0, **kw))  # type: ignore[arg-type]
        m = analyze_step(syn.profile, settings).measurement
        assert m.value_nm is not None
        e = m.value_nm - truth
        err.append(e)
        z.append(abs(e) / m.u_nm if m.u_nm else math.nan)
    err_a, z_a = np.array(err), np.array(z)
    return (
        f"| {name} | {truth:g} | {np.mean(err_a):+.3f} | {np.max(np.abs(err_a)):.3f} |"
        f" {np.mean(z_a < 2):.0%} |"
    )


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
    print(
        step_case(
            "film edge, correlated roughness Rq 2 nm (5 µm)",
            100.0,
            kind="edge",
            height_nm=100.0,
            noise_nm=0.5,
            roughness_nm=2.0,
            roughness_corr_um=5.0,
        )
    )
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

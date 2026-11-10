"""Synthetic stylus profiles with known step height, tilt, bow, noise, burrs and spikes.

Used by the tests, the ``synth`` command and the validation in the README.  The surface
is built first (edges with a finite wall width), then traced with a stylus - a sphere
on a cone (morphological dilation: the apex follows the highest contact,
:mod:`.core.tip`) - then tilt, bow, noise and spikes are added, the order in which they
arise on the instrument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .core.tip import Tip, dilate
from .profile import Profile, ScanInfo


@dataclass
class SynthSpec:
    kind: str = "edge"
    """``edge``, ``groove``, ``ridge``, ``periodic``, ``staircase``, ``flat``, ``sine``."""
    length_um: float = 1000.0
    n_points: int = 10_000
    height_nm: float = 100.0
    """Step height (film thickness); groove depth; ridge height; staircase: each step."""
    position_um: float | None = None
    """Edge position or groove/ridge centre (default: middle)."""
    width_um: float = 200.0
    """Groove or ridge width (between half-height points)."""
    wall_um: float = 2.0
    """Width of a wall: 10-90 % rise of a logistic step, or the full length of a ramp."""
    wall_shape: str = "logistic"
    """``logistic`` (smooth, exponential tails) or ``ramp`` (straight slope, sharp corners)."""
    period_um: float = 100.0
    n_steps: int = 2
    tilt_nm_per_um: float = 0.0
    bow_nm: float = 0.0
    """Sag of a parabolic bow over the scan (positive: concave, ends higher)."""
    noise_nm: float = 0.0
    roughness_nm: float = 0.0
    """RMS of a correlated (Gaussian-smoothed) roughness."""
    roughness_corr_um: float = 5.0
    burr_nm: float = 0.0
    """Height of a burr on each rim of a groove (or on the high side of an edge)."""
    burr_width_um: float = 8.0
    spikes: int = 0
    spike_nm: float = 500.0
    tip_radius_um: float = 0.0
    """Stylus radius for the tracing (0: ideal point)."""
    cone_angle_deg: float = 60.0
    """Full angle of the stylus cone (0: a sphere without a cone).  Only features deeper
    than ``r (1 - sin(angle / 2))`` - 1 µm for 2 µm and 60° - see the cone."""
    sine_amplitude_nm: float = 10.0
    sine_wavelength_um: float = 50.0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Synthetic:
    profile: Profile
    truth: dict[str, Any] = field(default_factory=dict)


def _smooth_step(x: np.ndarray, x0: float, wall: float, shape: str = "logistic") -> np.ndarray:
    """0 -> 1 step at ``x0``: a logistic whose 10-90 % rise is ``wall``, or a straight
    ramp ``wall`` long."""
    if wall <= 0:
        return (x >= x0).astype(float)
    if shape == "ramp":
        return np.asarray(np.clip((x - x0) / wall + 0.5, 0.0, 1.0))
    if shape != "logistic":
        raise ValueError("wall_shape must be logistic or ramp")
    s = wall / 4.394  # 10-90 % of a logistic is 4.394 scale units
    return 0.5 * (1.0 + np.tanh((x - x0) / (2.0 * s)))


def trace(
    x: np.ndarray, z_nm: np.ndarray, radius_um: float, cone_angle_deg: float = 0.0
) -> np.ndarray:
    """Profile recorded by a stylus of ``radius_um`` on a cone of ``cone_angle_deg``
    (apex height, nm; ``radius_um = 0`` and no cone: an ideal point)."""
    if radius_um <= 0 and cone_angle_deg <= 0:
        return z_nm.copy()
    return dilate(x, z_nm, Tip(radius_um, cone_angle_deg))


def surface(spec: SynthSpec) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.linspace(0.0, spec.length_um, spec.n_points)
    c = spec.position_um if spec.position_um is not None else spec.length_um / 2
    h = spec.height_nm
    truth: dict[str, Any] = {"kind": spec.kind}
    z = np.zeros_like(x)
    if spec.kind == "edge":
        z = h * _smooth_step(x, c, spec.wall_um, spec.wall_shape)
        truth.update(value_nm=h, thickness_nm=abs(h), position_um=c)
        if spec.burr_nm:
            hi_side = c + spec.wall_um if h > 0 else c - spec.wall_um
            z += spec.burr_nm * np.exp(-0.5 * ((x - hi_side) / (spec.burr_width_um / 4)) ** 2)
    elif spec.kind in ("groove", "ridge"):
        a, b = c - spec.width_um / 2, c + spec.width_um / 2
        box = _smooth_step(x, a, spec.wall_um, spec.wall_shape) - _smooth_step(
            x, b, spec.wall_um, spec.wall_shape
        )
        sign = -1.0 if spec.kind == "groove" else 1.0
        z = h + sign * h * box if spec.kind == "groove" else h * box
        truth.update(value_nm=abs(h), thickness_nm=abs(h), position_um=c, width_um=spec.width_um)
        if spec.burr_nm:
            for e in (a, b):
                outside = e - spec.burr_width_um / 2 if e == a else e + spec.burr_width_um / 2
                z += spec.burr_nm * np.exp(-0.5 * ((x - outside) / (spec.burr_width_um / 4)) ** 2)
            truth["burr_nm"] = spec.burr_nm
    elif spec.kind == "periodic":
        z = np.zeros_like(x)
        starts = np.arange(spec.period_um / 4, spec.length_um, spec.period_um)
        n = 0
        for s0 in starts:
            s1 = s0 + spec.period_um / 2
            if s1 + spec.period_um / 4 > spec.length_um:
                break
            z += h * (
                _smooth_step(x, s0, spec.wall_um, spec.wall_shape)
                - _smooth_step(x, s1, spec.wall_um, spec.wall_shape)
            )
            n += 2
        truth.update(value_nm=abs(h), thickness_nm=abs(h), period_um=spec.period_um, n_edges=n)
    elif spec.kind == "staircase":
        pos = np.linspace(0, spec.length_um, spec.n_steps + 2)[1:-1]
        for p in pos:
            z += h * _smooth_step(x, p, spec.wall_um, spec.wall_shape)
        truth.update(step_nm=h, n_edges=spec.n_steps, positions_um=pos.tolist())
    elif spec.kind == "sine":
        A, lam = spec.sine_amplitude_nm, spec.sine_wavelength_um
        z = A * np.sin(2 * np.pi * x / lam)
        truth.update(
            Ra=2 * A / np.pi,
            Rq=A / np.sqrt(2),
            Rt=2 * A,
            Rz=2 * A,
            Rp=A,
            Rv=A,
            Rsk=0.0,
            Rku=1.5,
            Rsm=lam,
        )
    elif spec.kind == "flat":
        truth.update(value_nm=0.0)
    else:
        raise ValueError(f"unknown synthetic kind {spec.kind!r}")
    return x, z, truth


def synthesize(spec: SynthSpec | None = None, **kw: Any) -> Synthetic:
    """A synthetic profile (heights nm, positions µm) and the truth it was built from."""
    spec = spec or SynthSpec(**kw)
    rng = np.random.default_rng(spec.seed)
    x, z, truth = surface(spec)
    if spec.roughness_nm:
        from scipy.ndimage import gaussian_filter1d

        dx = spec.length_um / (spec.n_points - 1)
        w = rng.normal(size=x.size)
        w = gaussian_filter1d(w, max(spec.roughness_corr_um / dx, 0.5))
        z = z + spec.roughness_nm * w / np.std(w)
    z = trace(x, z, spec.tip_radius_um, spec.cone_angle_deg if spec.tip_radius_um > 0 else 0.0)
    t = (x - spec.length_um / 2) / (spec.length_um / 2)
    z = z + spec.tilt_nm_per_um * (x - spec.length_um / 2) + spec.bow_nm * t**2
    if spec.noise_nm:
        z = z + rng.normal(scale=spec.noise_nm, size=x.size)
    spike_idx: list[int] = []
    if spec.spikes:
        spike_idx = sorted(
            int(i) for i in rng.choice(np.arange(20, x.size - 20), spec.spikes, replace=False)
        )
        for i in spike_idx:
            z[i] += spec.spike_nm * (1 if rng.random() > 0.3 else -1)
    truth.update(
        tilt_nm_per_um=spec.tilt_nm_per_um,
        bow_nm=spec.bow_nm,
        noise_nm=spec.noise_nm,
        spike_x_um=[float(x[i]) for i in spike_idx],
        spec=spec.to_dict(),
    )
    scan = ScanInfo(
        scan_length_um=spec.length_um,
        n_points=spec.n_points,
        tip_radius_um=spec.tip_radius_um or None,
        sources={"scan_length_um": "synthetic", "n_points": "synthetic"},
    )
    prof = Profile(
        x_um=x,
        z_nm=z,
        name=f"synthetic-{spec.kind}",
        x_unit="µm",
        z_unit="nm",
        x_unit_source="synthetic",
        z_unit_source="synthetic",
        scan=scan,
        history=[{"step": "synthesize", **spec.to_dict()}],
    )
    return Synthetic(prof, truth)


def write_csv(
    profile: Profile,
    path: str | Path,
    *,
    z_unit: str = "Å",
    metadata: dict[str, str] | None = None,
    header: bool = True,
) -> Path:
    """Write a profile as a generic CSV with metadata lines above the table.

    The layout is generic (``key,value`` lines, then ``x (µm),z (unit)``); it is not a copy
    of any instrument's export format.
    """
    from .units import TO_NM

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    lines = [f"{k},{v}" for k, v in meta.items()]
    if lines:
        lines.append("")
    if header:
        lines.append(f"x (µm),z ({z_unit})")
    z = profile.z_nm / TO_NM[z_unit]
    lines += [f"{a:.6f},{b:.6f}" for a, b in zip(profile.x_um, z)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def demo_set(out: str | Path, seed: int = 1) -> list[Path]:
    """A folder of synthetic scans: three samples x three scans, a groove, roughness and
    a stress pair.  Truth values go to ``truth.json``."""
    import json

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    truth: dict[str, Any] = {}
    meta = {"Stylus Force": "3 mg", "Scan Length": "1000 um", "Stylus Radius": "2 um"}
    for s, h in (("A", 85.0), ("B", 152.0), ("C", 310.0)):
        for k in range(3):
            syn = synthesize(
                SynthSpec(
                    kind="edge",
                    height_nm=h + 0.6 * (k - 1),
                    tilt_nm_per_um=0.04 * (k + 1),
                    noise_nm=0.6,
                    spikes=2,
                    tip_radius_um=2.0,
                    seed=seed + 10 * k + ord(s),
                )
            )
            f = write_csv(syn.profile, out / f"film{s}_{k + 1}.csv", metadata=meta)
            files.append(f)
            truth[f.name] = syn.truth["thickness_nm"]
    g = synthesize(
        SynthSpec(
            kind="groove",
            height_nm=240.0,
            width_um=150.0,
            burr_nm=60.0,
            tilt_nm_per_um=-0.03,
            noise_nm=0.8,
            tip_radius_um=2.0,
            seed=seed + 99,
        )
    )
    files.append(write_csv(g.profile, out / "scratch_1.csv", metadata=meta))
    truth["scratch_1.csv"] = g.truth["thickness_nm"]
    r = synthesize(
        SynthSpec(
            kind="sine",
            length_um=1000.0,
            n_points=20_000,
            sine_amplitude_nm=12.0,
            sine_wavelength_um=40.0,
            noise_nm=0.0,
            seed=seed,
        )
    )
    rough_dir = out / "roughness"
    files.append(write_csv(r.profile, rough_dir / "sine_A12nm_40um.csv", metadata=meta))
    truth["roughness/sine_A12nm_40um.csv"] = {
        **{k: r.truth[k] for k in ("Ra", "Rq", "Rsm")},
        "note": "unfiltered values: reproduced with --lc 800 --ls 0 (λc far above 40 µm)",
    }
    stress_dir = out / "stress"
    meta_s = {"Stylus Force": "3 mg", "Scan Length": "30000 um"}
    pre = synthesize(
        SynthSpec(
            kind="flat", length_um=30_000, n_points=6000, bow_nm=-300.0, noise_nm=1.0, seed=seed + 5
        )
    )
    post = synthesize(
        SynthSpec(
            kind="flat", length_um=30_000, n_points=6000, bow_nm=900.0, noise_nm=1.0, seed=seed + 6
        )
    )
    files.append(write_csv(pre.profile, stress_dir / "wafer_pre.csv", metadata=meta_s))
    files.append(write_csv(post.profile, stress_dir / "wafer_post.csv", metadata=meta_s))
    # curvature of the parabola bow*t^2, t = (x - L/2)/(L/2): z'' = 2 bow / (L/2)^2
    half = 15_000.0
    k_pre = 2 * (-300.0) / half**2 * 1e-3
    k_post = 2 * 900.0 / half**2 * 1e-3
    truth["stress"] = {"curvature_pre_per_m": k_pre * 1e6, "curvature_post_per_m": k_post * 1e6}
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return files

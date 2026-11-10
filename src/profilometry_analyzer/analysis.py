"""The analysis pipelines shared by the command line, the batch runner and the GUI.

Every pipeline takes a profile as read from a file and a settings object, never
changes its input, and returns a result whose ``to_dict()`` goes into the project file
and whose ``key_numbers()`` are compared by ``verify``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import notices as N
from .core.despike import despike
from .core.edges import EdgeSet, classify, detect_edges, edges_at
from .core.leveling import LevelResult, level_auto, level_none, level_regions
from .core.roughness import RoughnessResult, RoughnessSettings, roughness
from .core.steps import (
    StepMeasurement,
    measure,
    measure_regions,
    range_notices,
    refine_edges,
    widen_transitions,
)
from .core.stress import StressResult, StressSettings, stress
from .core.tip import Tip, TipCheck, check_tip
from .instrument import DEFAULT_INSTRUMENT, InstrumentSpec
from .notices import Notice
from .profile import Profile

STEP_MODES = ("auto", "edge", "groove", "ridge", "periodic", "staircase", "multi")
LEVEL_MODES = ("auto", "regions", "none")


def _tuples(v: Any) -> list[tuple[float, float]] | None:
    if not v:
        return None
    return [(float(a), float(b)) for a, b in v]


@dataclass
class StepSettings:
    """Everything that decides a step-height result (stored in the project file)."""

    despike: bool = True
    spike_k: float = 8.0
    spike_width: int = 3
    crop_um: tuple[float, float] | None = None
    level: str = "auto"
    level_method: str = "huber"
    level_regions_um: list[tuple[float, float]] | None = None
    level_separate: bool = False
    bow_order: int = 1
    """1: straight-line levelling only; 2-6: a polynomial bow is removed as well."""
    mode: str = "auto"
    edges_um: list[float] | None = None
    regions_um: list[tuple[float, float]] | None = None
    """Two regions (reference A, measurement B) for a manual step height."""
    smooth_um: float | None = None
    edge_k: float = 6.0
    min_step_nm: float = 0.0
    margin_um: float | None = None
    min_plane_um: float | None = None
    burr_k: float = 4.0
    multiscale: bool = True
    """Look for wide, gentle edges with wider derivative smoothings as well."""
    widen: bool = True
    """Widen a transition while the plateau next to it still runs towards the other level."""
    tip_radius_um: float | None = None
    cone_angle_deg: float | None = None
    """Full angle of the stylus cone (default: file, else the instrument's 60°)."""
    stylus_force_mg: float | None = None
    soft_film: bool = False
    tip_check: bool = True
    """Find the points the stylus could not touch (:mod:`.core.tip`)."""
    seed: int = 0

    def __post_init__(self) -> None:
        if self.level not in LEVEL_MODES:
            raise ValueError(f"level must be one of {LEVEL_MODES}")
        if self.mode not in STEP_MODES:
            raise ValueError(f"mode must be one of {STEP_MODES}")
        if self.level_method not in ("huber", "ransac", "lsq"):
            raise ValueError("level_method must be huber, ransac or lsq")
        if not 1 <= int(self.bow_order) <= 6:
            raise ValueError("bow_order must be between 1 and 6")
        self.level_regions_um = _tuples(self.level_regions_um)
        self.regions_um = _tuples(self.regions_um)
        if self.crop_um:
            self.crop_um = (float(self.crop_um[0]), float(self.crop_um[1]))
        if self.edges_um:
            self.edges_um = [float(v) for v in self.edges_um]
        if self.regions_um is not None and len(self.regions_um) != 2:
            raise ValueError("regions_um needs exactly two regions (reference, measurement)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("level_regions_um", "regions_um"):
            if d[k] is not None:
                d[k] = [list(r) for r in d[k]]
        if d["crop_um"] is not None:
            d["crop_um"] = list(d["crop_um"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> StepSettings:
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if known.get("crop_um"):
            known["crop_um"] = tuple(known["crop_um"])
        return cls(**known)


def prepare(profile: Profile, st: StepSettings | None = None, **scan: Any) -> Profile:
    """Crop, scan-parameter overrides and spike removal."""
    st = st or StepSettings()
    p = profile
    if st.crop_um:
        p = p.crop(*st.crop_um)
    overrides = {
        "tip_radius_um": st.tip_radius_um,
        "cone_angle_deg": st.cone_angle_deg,
        "stylus_force_mg": st.stylus_force_mg,
        "soft_film": True if st.soft_film else None,
        **scan,
    }
    if any(v is not None for v in overrides.values()):
        p = p.with_scan(**overrides)
    if st.despike:
        p = despike(p, k=st.spike_k, max_width=st.spike_width)
    return p


def tip_radius(profile: Profile, inst: InstrumentSpec = DEFAULT_INSTRUMENT) -> tuple[float, str]:
    if profile.scan.tip_radius_um:
        return float(profile.scan.tip_radius_um), profile.scan.sources.get("tip_radius_um", "file")
    return inst.tip_radius_um, "instrument default"


def cone_angle(profile: Profile, inst: InstrumentSpec = DEFAULT_INSTRUMENT) -> tuple[float, str]:
    if profile.scan.cone_angle_deg is not None:
        src = profile.scan.sources.get("cone_angle_deg", "file")
        return float(profile.scan.cone_angle_deg), src
    return inst.cone_angle_deg, "instrument default"


@dataclass
class StepResult:
    profile: Profile
    """The processed (levelled) profile; ``profile.raw_z`` holds the file values."""
    source: Profile
    """The profile as read (before crop, despike and levelling)."""
    settings: StepSettings
    level: LevelResult
    edges: EdgeSet
    measurement: StepMeasurement
    kind_detected: str
    tip_radius_um: float
    tip_radius_source: str
    notices: list[Notice] = field(default_factory=list)
    cone_angle_deg: float = DEFAULT_INSTRUMENT.cone_angle_deg
    cone_angle_source: str = "instrument default"
    tip_check: TipCheck | None = None
    """Points the stylus could not touch and the reconstruction (``None``: not run)."""

    @property
    def thickness_nm(self) -> float | None:
        return self.measurement.thickness_nm

    @property
    def status(self) -> str:
        return N.worst(self.notices)

    def key_numbers(self) -> dict[str, Any]:
        m = self.measurement
        out: dict[str, Any] = {
            "kind": m.kind,
            "method": m.method,
            "value_nm": m.value_nm,
            "u_nm": m.u_nm,
            "slope_nm_per_um": self.level.slope_nm_per_um,
            "n_edges": len(self.edges.edges),
        }
        for i, row in enumerate(m.per_edge):
            out[f"edge{i + 1}_x_um"] = row.get("x_um")
            out[f"edge{i + 1}_height_nm"] = row.get("height_nm")
        return out

    def row(self) -> dict[str, Any]:
        m = self.measurement
        warn = [n.code for n in self.notices if n.severity in (N.WARNING, N.ERROR)]
        return {
            "file": self.source.path or self.source.name,
            "name": self.source.name,
            "kind": m.kind,
            "method": m.method,
            "thickness_nm": m.thickness_nm,
            "value_nm": m.value_nm,
            "u_nm": m.u_nm,
            "u_fit_nm": m.u_fit_nm,
            "u_slope_nm": m.u_slope_nm,
            "u_tail_nm": m.u_tail_nm,
            "position_um": m.position_um,
            "width_um": m.width_um,
            "n_edges": len(self.edges.edges),
            "periodic_mean_nm": m.stats.get("mean_nm"),
            "periodic_std_nm": m.stats.get("std_nm"),
            "periodic_n": m.stats.get("n"),
            "period_um": m.stats.get("period_um"),
            "slope_nm_per_um": self.level.slope_nm_per_um,
            "level_method": self.level.method,
            "bow_order": self.settings.bow_order,
            "tip_radius_um": self.tip_radius_um,
            "cone_angle_deg": self.cone_angle_deg,
            "unreached_percent": (
                100.0 * self.tip_check.unreached_fraction() if self.tip_check else None
            ),
            "stylus_force_mg": self.profile.scan.stylus_force_mg,
            "z_unit": self.source.z_unit,
            "z_unit_source": self.source.z_unit_source,
            "x_unit": self.source.x_unit,
            "status": self.status,
            "warnings": ";".join(dict.fromkeys(warn)),
        }

    def to_dict(self) -> dict[str, Any]:
        x = self.profile.x_um
        return {
            "analysis": "step",
            "measurement": self.measurement.to_dict(),
            "kind_detected": self.kind_detected,
            "level": self.level.to_dict(),
            "edges": self.edges.to_dict(x),
            "tip_radius_um": self.tip_radius_um,
            "tip_radius_source": self.tip_radius_source,
            "cone_angle_deg": self.cone_angle_deg,
            "cone_angle_source": self.cone_angle_source,
            "tip_check": self.tip_check.to_dict() if self.tip_check else None,
            "profile": self.source.summary(),
            "history": list(self.profile.history),
            "key_numbers": self.key_numbers(),
            "notices": [n.to_dict() for n in self.notices],
        }


def _edge_options(st: StepSettings, tip: float) -> dict[str, Any]:
    return {
        "tip_radius_um": tip,
        "smooth_um": st.smooth_um,
        "k": st.edge_k,
        "min_step_nm": st.min_step_nm,
        "margin_um": st.margin_um,
        "min_plane_um": st.min_plane_um,
        "multiscale": st.multiscale,
    }


def soft_film_notices(p: Profile, inst: InstrumentSpec) -> list[Notice]:
    if not p.scan.soft_film:
        return []
    f = p.scan.stylus_force_mg
    if f is None:
        return [
            Notice(
                N.SOFT_FILM_FORCE_UNKNOWN,
                N.WARNING,
                "the film is marked soft but the stylus force is not known; with polymers start"
                f" at 1-{inst.soft_film_max_force_mg:g} mg - a stylus that ploughs makes the film"
                " look thinner",
            )
        ]
    if f > inst.soft_film_max_force_mg:
        return [
            Notice(
                N.SOFT_FILM_HIGH_FORCE,
                N.WARNING,
                f"stylus force {f:g} mg on a film marked soft (polymers: start at"
                f" 1-{inst.soft_film_max_force_mg:g} mg): the stylus may plough into the film and"
                " the thickness comes out too small",
                {"force_mg": f},
            )
        ]
    return []


def analyze_step(
    profile: Profile,
    settings: StepSettings | None = None,
    instrument: InstrumentSpec = DEFAULT_INSTRUMENT,
) -> StepResult:
    """Step height / film thickness of one scan."""
    st = settings or StepSettings()
    p = prepare(profile, st)
    tip, tip_src = tip_radius(p, instrument)
    cone, cone_src = cone_angle(p, instrument)
    stylus = Tip(tip, cone)
    opts = _edge_options(st, tip)
    x = p.x_um
    if st.level == "auto":
        lp, lev, _ = level_auto(p, method=st.level_method, order=st.bow_order, seed=st.seed, **opts)
    elif st.level == "regions":
        if not st.level_regions_um:
            raise ValueError("level='regions' needs level_regions_um")
        lp, lev = level_regions(
            p, st.level_regions_um, same_level=not st.level_separate, order=st.bow_order
        )
    else:
        lp, lev = level_none(p)
    if st.edges_um:
        eset = edges_at(
            x,
            lp.z_nm,
            st.edges_um,
            lp.valid,
            tip_radius_um=tip,
            margin_um=st.margin_um,
            min_plane_um=st.min_plane_um,
        )
    else:
        eset = detect_edges(x, lp.z_nm, lp.valid, **opts)
    widen_notes: list[Notice] = []
    if st.widen and not st.regions_um and eset.edges:
        widen_notes = widen_transitions(x, lp.z_nm, eset, k=st.burr_k)
        if widen_notes and st.level == "auto":
            # level again with the wider transitions kept out of the trend fit
            lp, lev, _ = level_auto(
                p, method=st.level_method, order=st.bow_order, edges=eset, seed=st.seed
            )
    if not st.edges_um:
        refine_edges(x, lp.z_nm, eset, k=st.burr_k)
    detected = classify(eset.edges)
    if st.regions_um:
        meas = measure_regions(
            x, lp.z_nm, lp.valid, st.regions_um[0], st.regions_um[1], k=st.burr_k
        )
    else:
        kind = detected if st.mode == "auto" else st.mode
        if kind == "periodic" and len(eset.edges) < 2:
            kind = detected
        meas = measure(x, lp.z_nm, eset, kind, k=st.burr_k, tip_radius_um=tip, cone_angle_deg=cone)
    noise = float(eset.threshold.get("height_noise_nm", 0.0))
    meas.notices += range_notices(meas, noise, instrument)
    notes = list(lp.notices) + list(eset.notices) + widen_notes + list(meas.notices)
    notes += soft_film_notices(lp, instrument)
    reach = check_tip(x, lp.z_nm, stylus, lp.valid) if st.tip_check else None
    if reach is not None:
        notes += tip_notices(reach, meas)
    notes += wall_notices(x, lp.z_nm, eset, stylus)
    depth = float(np.ptp(lp.z_nm[lp.valid])) if lp.valid.any() else 0.0
    if cone_src == "instrument default" and stylus.has_cone and depth > stylus.tangent_depth_nm:
        notes.append(
            Notice(
                N.CONE_ANGLE_DEFAULT,
                N.INFO,
                f"the profile spans {depth:.4g} nm, more than the {stylus.tangent_depth_nm:.4g} nm"
                f" a {tip:g} µm sphere covers, so the stylus cone shapes the walls; its angle"
                f" is not in the file and {cone:g}° was assumed (set the angle of the mounted"
                " stylus)",
                {"cone_angle_deg": cone},
            )
        )
    if tip_src == "instrument default":
        notes.append(
            Notice(
                N.TIP_RADIUS_DEFAULT,
                N.INFO,
                f"stylus radius not in the file: {tip:g} µm assumed for the edge margins and the"
                " narrow-feature checks (set the radius of the mounted stylus)",
            )
        )
    return StepResult(
        profile=lp,
        source=profile,
        settings=st,
        level=lev,
        edges=eset,
        measurement=meas,
        kind_detected=detected,
        tip_radius_um=tip,
        tip_radius_source=tip_src,
        notices=N.unique(notes),
        cone_angle_deg=cone,
        cone_angle_source=cone_src,
        tip_check=reach,
    )


def tip_notices(reach: TipCheck, meas: StepMeasurement) -> list[Notice]:
    """Where the stylus did not touch the surface, and whether the fitted levels use
    such points."""
    if not reach.unreached_um:
        return []
    length = sum(b - a for a, b in reach.unreached_um)
    longest = max(reach.unreached_um, key=lambda r: r[1] - r[0])
    worst_plane, worst_share = None, 0.0
    for plane in meas.planes:
        used = plane.used
        if used is None or not used.any():
            continue
        share = reach.unreached_fraction(used)
        if share > worst_share:
            worst_plane, worst_share = plane, share
    where = (
        f"{len(reach.unreached_um)} region(s), {length:.3g} µm in all, the longest"
        f" {longest[1] - longest[0]:.3g} µm at {0.5 * (longest[0] + longest[1]):.4g} µm"
    )
    data = {"unreached_um": [list(r) for r in reach.unreached_um], "stylus": reach.tip.to_dict()}
    out: list[Notice] = []
    floors = [i for i, f in enumerate(reach.floors) if f]
    if floors:
        j = max(floors, key=lambda i: reach.depth_nm[i])
        a, b = reach.unreached_um[j]
        out.append(
            Notice(
                N.TIP_UNREACHED,
                N.INFO,
                f"the stylus ({reach.tip.describe()}) did not reach the floor of"
                f" {len(floors)} dip(s) narrower than itself; the deepest, {b - a:.3g} µm at"
                f" {0.5 * (a + b):.4g} µm, reads {reach.depth_nm[j]:.4g} nm deep, which is only a"
                " lower bound of its true depth (and its width is the stylus's)",
                {**data, "floors": floors, "recorded_depth_nm": reach.depth_nm[j]},
            )
        )
    if worst_plane is not None and worst_share > 0.02:
        return [
            Notice(
                N.TIP_UNREACHED,
                N.WARNING,
                f"the stylus ({reach.tip.describe()}) did not touch the surface in {where};"
                f" {100 * worst_share:.2g} % of the points of the {worst_plane.name} plane are"
                " among them: that level is only an upper bound (a floor the stylus cannot"
                " reach reads too high) - narrow the region or use a finer stylus",
                {**data, "plane": worst_plane.name, "plane_percent": 100 * worst_share},
            )
        ]
    if len(floors) < len(reach.unreached_um):
        out.append(
            Notice(
                N.TIP_UNREACHED,
                N.INFO,
                f"the stylus ({reach.tip.describe()}) did not touch the surface in {where} (the"
                " foot of a steep wall, or a gap narrower than the stylus); the fitted levels"
                " do not use these points",
                data,
            )
        )
    return out


def wall_notices(x: np.ndarray, z: np.ndarray, eset: EdgeSet, tip: Tip) -> list[Notice]:
    """Walls as steep as the stylus flank: their recorded shape is the stylus's."""
    if not tip.has_cone or not eset.edges or x.size < 5:
        return []
    out: list[Notice] = []
    slope = np.abs(np.gradient(z, x))
    for i, e in enumerate(eset.edges):
        seg = slope[e.start : e.stop]
        if seg.size < 3:
            continue
        steep = float(np.percentile(seg, 90))
        if steep > 1.15 * tip.flank_slope:
            out.append(
                Notice(
                    N.TIP_LIMITED_WALL,
                    N.WARNING,
                    f"the wall at {e.x_um:.4g} µm is steeper ({steep / 1e3:.3g} µm/µm) than the"
                    f" flank of a {tip.cone_angle_deg:g}° stylus cone can record"
                    f" ({tip.flank_slope / 1e3:.3g}): the mounted stylus is sharper than"
                    " assumed - check the cone angle; the regions it could not touch are"
                    " wrong until then (the height is not affected)",
                    {"edge": i + 1, "slope": steep / 1e3, "flank_slope": tip.flank_slope / 1e3},
                )
            )
        elif steep >= 0.9 * tip.flank_slope:
            out.append(
                Notice(
                    N.TIP_LIMITED_WALL,
                    N.INFO,
                    f"the wall at {e.x_um:.4g} µm is as steep as the flank of the"
                    f" {tip.cone_angle_deg:g}° stylus cone ({steep / 1e3:.3g} vs"
                    f" {tip.flank_slope / 1e3:.3g} µm/µm): its recorded shape and width are the"
                    " stylus's, not the film edge's; the height is not affected",
                    {"edge": i + 1, "slope": steep / 1e3, "flank_slope": tip.flank_slope / 1e3},
                )
            )
    return out


# ---------------------------------------------------------------- roughness
@dataclass
class RoughnessAnalysis:
    profile: Profile
    source: Profile
    settings: RoughnessSettings
    result: RoughnessResult
    despiked: bool
    notices: list[Notice] = field(default_factory=list)

    def key_numbers(self) -> dict[str, Any]:
        return {
            **self.result.parameters,
            "lambda_c_um": self.result.lambda_c_um,
            "lambda_s_um": self.result.lambda_s_um,
            "n_sections": self.result.n_sections,
        }

    def row(self) -> dict[str, Any]:
        warn = [n.code for n in self.notices if n.severity in (N.WARNING, N.ERROR)]
        r = self.result
        return {
            "file": self.source.path or self.source.name,
            "name": self.source.name,
            **{
                f"{k}{'_um' if k == 'Rsm' else '' if k in ('Rsk', 'Rku') else '_nm'}": v
                for k, v in r.parameters.items()
            },
            "lambda_c_um": r.lambda_c_um,
            "lambda_s_um": r.lambda_s_um,
            "setting_class": r.setting_class,
            "n_sections": r.n_sections,
            "evaluation_length_um": r.evaluation_um[1] - r.evaluation_um[0],
            "dx_um": r.dx_um,
            "status": N.worst(self.notices),
            "warnings": ";".join(dict.fromkeys(warn)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": "roughness",
            "result": self.result.to_dict(),
            "profile": self.source.summary(),
            "history": list(self.profile.history),
            "key_numbers": self.key_numbers(),
            "notices": [n.to_dict() for n in self.notices],
        }


def analyze_roughness(
    profile: Profile,
    settings: RoughnessSettings | None = None,
    *,
    despike_first: bool = True,
    crop_um: tuple[float, float] | None = None,
) -> RoughnessAnalysis:
    st = settings or RoughnessSettings()
    p = profile.crop(*crop_um) if crop_um else profile
    if despike_first:
        p = despike(p)
    res = roughness(p, st)
    notes = N.unique(list(p.notices) + list(res.notices))
    return RoughnessAnalysis(p, profile, st, res, despike_first, notes)


# ---------------------------------------------------------------- stress
@dataclass
class StressAnalysis:
    pre: Profile | None
    post: Profile
    sources: list[Profile]
    settings: StressSettings
    result: StressResult
    notices: list[Notice] = field(default_factory=list)

    def key_numbers(self) -> dict[str, Any]:
        r = self.result
        return {
            "stress_mpa": r.stress_mpa,
            "u_stress_mpa": r.u_stress_mpa,
            "curvature_post_per_m": r.post.kappa_per_um * 1e6,
            "curvature_pre_per_m": r.pre.kappa_per_um * 1e6 if r.pre else None,
        }

    def row(self) -> dict[str, Any]:
        r = self.result
        warn = [n.code for n in self.notices if n.severity in (N.WARNING, N.ERROR)]
        return {
            "pre": self.sources[0].path if self.pre is not None else None,
            "post": self.sources[-1].path,
            "stress_mpa": r.stress_mpa,
            "u_stress_mpa": r.u_stress_mpa,
            "kind": r.kind,
            "radius_pre_m": r.pre.radius_m if r.pre else None,
            "radius_post_m": r.post.radius_m,
            "biaxial_modulus_gpa": r.biaxial_modulus_gpa,
            "substrate_thickness_um": r.substrate_thickness_um,
            "film_thickness_nm": r.film_thickness_nm,
            "status": N.worst(self.notices),
            "warnings": ";".join(dict.fromkeys(warn)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": "stress",
            "result": self.result.to_dict(),
            "profiles": [s.summary() for s in self.sources],
            "key_numbers": self.key_numbers(),
            "notices": [n.to_dict() for n in self.notices],
        }


def analyze_stress(pre: Profile | None, post: Profile, settings: StressSettings) -> StressAnalysis:
    a = despike(pre) if (pre is not None and settings.despike) else pre
    b = despike(post) if settings.despike else post
    res = stress(a, b, settings)
    notes = list(b.notices)
    if a is not None:
        notes += list(a.notices)
    notes += res.notices
    sources = [s for s in (pre, post) if s is not None]
    return StressAnalysis(a, b, sources, settings, res, N.unique(notes))


def residual_noise(profile: Profile) -> float:
    """White-noise estimate of a profile (nm), for read-outs."""
    from .core.stats import diff_noise

    return float(diff_noise(profile.z_nm[profile.valid]))


__all__ = [
    "LEVEL_MODES",
    "STEP_MODES",
    "RoughnessAnalysis",
    "RoughnessSettings",
    "StepResult",
    "StepSettings",
    "StressAnalysis",
    "StressSettings",
    "analyze_roughness",
    "analyze_step",
    "analyze_stress",
    "prepare",
]

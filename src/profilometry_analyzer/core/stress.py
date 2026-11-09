"""Film stress from wafer curvature (Stoney equation).

    sigma_f = [E_s / (1 - nu_s)] * h_s^2 / (6 h_f) * (1/R_post - 1/R_pre)

``E_s / (1 - nu_s)`` is the biaxial modulus of the substrate, ``h_s`` and ``h_f`` the
substrate and film thicknesses.  The curvature ``1/R`` of each long scan comes from a
quadratic fit ``z = a (x - x0)^2 + b (x - x0) + c``:
``kappa = 2 a / (1 + b^2)^(3/2)``.

Sign convention (of this program): the scans are taken on the film side, before and
after deposition.  ``kappa > 0`` when that surface is concave (centre of curvature
above it, the profile curves upwards).  A film under tension pulls the surface it sits
on into a concave shape, so ``sigma_f > 0`` is **tensile** and ``sigma_f < 0``
**compressive**.  Scans from the back side reverse the sign (``backside=True``).  The
sign convention of the instrument software was not checked.

Assumptions checked by notices: film much thinner than the substrate (Stoney),
deflection small compared with the substrate thickness, a uniform (spherical)
curvature, a curvature change larger than its uncertainty, and comparable scans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import notices as N
from ..notices import Notice
from ..profile import Profile
from .stats import mad_sigma, n_effective


@dataclass(frozen=True)
class Substrate:
    key: str
    name: str
    biaxial_modulus_gpa: float
    """``E / (1 - nu)`` for the in-plane biaxial state."""
    youngs_modulus_gpa: float | None
    poisson_ratio: float | None
    typical_thickness_um: float | None
    source: str
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "biaxial_modulus_gpa": self.biaxial_modulus_gpa,
            "youngs_modulus_gpa": self.youngs_modulus_gpa,
            "poisson_ratio": self.poisson_ratio,
            "typical_thickness_um": self.typical_thickness_um,
            "source": self.source,
            "verified": self.verified,
        }


# Silicon: biaxial moduli computed from the elastic constants c11 = 165.7, c12 = 63.9,
# c44 = 79.6 GPa (as tabulated by Hopcroft, Nix & Kenny, J. Microelectromech. Syst. 19,
# 229 (2010)): (100): c11 + c12 - 2 c12^2 / c11; (111): 6 c44 (c11 + 2 c12) /
# (c11 + 2 c12 + 4 c44).
_C11, _C12, _C44 = 165.7, 63.9, 79.6
M_SI100 = _C11 + _C12 - 2 * _C12**2 / _C11
M_SI111 = 6 * _C44 * (_C11 + 2 * _C12) / (_C11 + 2 * _C12 + 4 * _C44)
_SI_SRC = (
    "computed from Si elastic constants c11 = 165.7, c12 = 63.9, c44 = 79.6 GPa"
    " (Hopcroft, Nix & Kenny, J. Microelectromech. Syst. 19, 229 (2010))"
)


def _glass(key: str, name: str, e: float, nu: float, src: str, verified: bool = True) -> Substrate:
    return Substrate(key, name, e / (1 - nu), e, nu, None, src, verified)


SUBSTRATES: dict[str, Substrate] = {
    s.key: s
    for s in (
        Substrate("si100", "Si(100)", M_SI100, None, None, 525.0, _SI_SRC),
        Substrate("si111", "Si(111)", M_SI111, None, None, 525.0, _SI_SRC),
        _glass(
            "eagle-xg",
            "Corning EAGLE XG glass",
            73.6,
            0.23,
            "Corning EAGLE XG product information sheet (E = 73.6 GPa, ν = 0.23)",
        ),
        _glass(
            "borofloat33",
            "SCHOTT BOROFLOAT 33",
            64.0,
            0.20,
            "SCHOTT BOROFLOAT 33 data sheet (E = 64 GPa, ν = 0.2)",
        ),
        _glass(
            "fused-silica",
            "Fused silica (Corning 7980)",
            72.7,
            0.16,
            "Corning HPFS 7980 data sheet (E = 72.7 GPa, ν = 0.16)",
        ),
        _glass(
            "soda-lime",
            "Soda-lime glass (typical)",
            70.0,
            0.20,
            "typical value (EN 572-1 lists about 70 GPa and 0.2); not checked against a"
            " data sheet - replace with the supplier's value",
            verified=False,
        ),
    )
}


@dataclass
class CurvatureFit:
    kappa_per_um: float
    u_kappa_per_um: float
    coefs: tuple[float, float, float]
    """``a, b, c`` of ``a (x - x0)^2 + b (x - x0) + c`` (nm, µm)."""
    x0_um: float
    length_um: float
    sigma_nm: float
    n: int
    n_eff: float
    quartic_improvement: float
    """Residual variance ratio quadratic/quartic (1 = no improvement)."""
    sag_nm: float
    """Peak-to-valley of the fitted parabola over the scan."""

    @property
    def radius_m(self) -> float:
        return float(1e-6 / self.kappa_per_um) if self.kappa_per_um else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "curvature_per_m": self.kappa_per_um * 1e6,
            "u_curvature_per_m": self.u_kappa_per_um * 1e6,
            "radius_m": self.radius_m,
            "coefs": list(self.coefs),
            "x0_um": self.x0_um,
            "length_um": self.length_um,
            "sigma_nm": self.sigma_nm,
            "n": self.n,
            "n_eff": self.n_eff,
            "quartic_improvement": self.quartic_improvement,
            "sag_nm": self.sag_nm,
        }


def fit_curvature(profile: Profile, region_um: tuple[float, float] | None = None) -> CurvatureFit:
    """Curvature of a long scan from a quadratic fit (heights nm, positions µm)."""
    x, z, m = profile.x_um, profile.z_nm, profile.valid.copy()
    if region_um:
        a, b = sorted(region_um)
        m &= (x >= a) & (x <= b)
    if m.sum() < 10:
        raise ValueError("a curvature fit needs at least 10 valid points")
    xs, zs = x[m], z[m]
    x0 = float(np.mean(xs))
    half = max(float(np.ptp(xs)) / 2.0, 1e-12)
    t = (xs - x0) / half
    A = np.column_stack([t**2, t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    r = zs - A @ coef
    dof = max(xs.size - 3, 1)
    s2 = float(r @ r / dof)
    neff = n_effective(r)
    cov = s2 * np.linalg.inv(A.T @ A) * max(xs.size / neff, 1.0)
    a = float(coef[0]) / half**2
    b = float(coef[1]) / half
    u_a = float(np.sqrt(max(cov[0, 0], 0.0))) / half**2
    # nm per µm^2 -> 1/µm: z in nm, x in µm => d2z/dx2 [nm/µm^2] = 1e-3 [µm/µm^2]
    slope = b * 1e-3
    kappa = 2 * a * 1e-3 / (1 + slope**2) ** 1.5
    u_kappa = 2 * u_a * 1e-3
    A4 = np.column_stack([t**4, t**3, t**2, t, np.ones_like(t)])
    c4, *_ = np.linalg.lstsq(A4, zs, rcond=None)
    r4 = zs - A4 @ c4
    ratio = float((r @ r) / max(float(r4 @ r4), 1e-300))
    par = A @ coef
    return CurvatureFit(
        kappa_per_um=float(kappa),
        u_kappa_per_um=float(u_kappa),
        coefs=(a, b, float(coef[2])),
        x0_um=x0,
        length_um=float(np.ptp(xs)),
        sigma_nm=float(np.sqrt(s2)),
        n=int(xs.size),
        n_eff=float(neff),
        quartic_improvement=ratio,
        sag_nm=float(np.ptp(par)),
    )


@dataclass
class StressSettings:
    substrate: str | None = "si100"
    biaxial_modulus_gpa: float | None = None
    """Overrides the substrate table (``E/(1-ν)``)."""
    youngs_modulus_gpa: float | None = None
    poisson_ratio: float | None = None
    substrate_thickness_um: float | None = None
    film_thickness_nm: float | None = None
    region_um: tuple[float, float] | None = None
    backside: bool = False
    despike: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "biaxial_modulus_gpa": self.biaxial_modulus_gpa,
            "youngs_modulus_gpa": self.youngs_modulus_gpa,
            "poisson_ratio": self.poisson_ratio,
            "substrate_thickness_um": self.substrate_thickness_um,
            "film_thickness_nm": self.film_thickness_nm,
            "region_um": list(self.region_um) if self.region_um else None,
            "backside": self.backside,
            "despike": self.despike,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> StressSettings:
        d = dict(d or {})
        if d.get("region_um"):
            d["region_um"] = tuple(d["region_um"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class StressResult:
    stress_mpa: float
    u_stress_mpa: float
    kind: str
    """``tensile`` or ``compressive`` (or ``not significant``)."""
    biaxial_modulus_gpa: float
    modulus_source: str
    substrate_thickness_um: float
    film_thickness_nm: float
    pre: CurvatureFit | None
    post: CurvatureFit
    delta_kappa_per_um: float
    u_delta_kappa_per_um: float
    notices: list[Notice] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stress_mpa": self.stress_mpa,
            "u_stress_mpa": self.u_stress_mpa,
            "kind": self.kind,
            "biaxial_modulus_gpa": self.biaxial_modulus_gpa,
            "modulus_source": self.modulus_source,
            "substrate_thickness_um": self.substrate_thickness_um,
            "film_thickness_nm": self.film_thickness_nm,
            "pre": self.pre.to_dict() if self.pre else None,
            "post": self.post.to_dict(),
            "delta_curvature_per_m": self.delta_kappa_per_um * 1e6,
            "u_delta_curvature_per_m": self.u_delta_kappa_per_um * 1e6,
            "radius_pre_m": self.pre.radius_m if self.pre else None,
            "radius_post_m": self.post.radius_m,
            "sign_convention": "positive = tensile (film-side scans; see documentation)",
            "notices": [n.to_dict() for n in self.notices],
        }


def stoney_mpa(
    biaxial_modulus_gpa: float,
    substrate_thickness_um: float,
    film_thickness_nm: float,
    delta_kappa_per_um: float,
) -> float:
    """Stoney stress in MPa from SI-consistent inputs (GPa, µm, nm, 1/µm)."""
    m = biaxial_modulus_gpa * 1e9
    hs = substrate_thickness_um * 1e-6
    hf = film_thickness_nm * 1e-9
    dk = delta_kappa_per_um * 1e6
    return float(m * hs**2 / (6.0 * hf) * dk / 1e6)


def resolve_modulus(st: StressSettings) -> tuple[float, str, list[Notice]]:
    notes: list[Notice] = []
    if st.biaxial_modulus_gpa:
        return float(st.biaxial_modulus_gpa), "user (biaxial modulus)", notes
    if st.youngs_modulus_gpa and st.poisson_ratio is not None:
        m = st.youngs_modulus_gpa / (1 - st.poisson_ratio)
        return (
            float(m),
            f"user (E = {st.youngs_modulus_gpa:g} GPa, ν = {st.poisson_ratio:g})",
            notes,
        )
    if not st.substrate or st.substrate not in SUBSTRATES:
        raise ValueError(
            f"unknown substrate {st.substrate!r}; choose one of {sorted(SUBSTRATES)} or give"
            " the modulus"
        )
    sub = SUBSTRATES[st.substrate]
    if not sub.verified:
        notes.append(
            Notice(
                N.SUBSTRATE_VALUE_UNVERIFIED,
                N.WARNING,
                f"the modulus of {sub.name} is a typical value, not a checked data-sheet value:"
                f" {sub.source}",
                sub.to_dict(),
            )
        )
    return sub.biaxial_modulus_gpa, f"{sub.name}: {sub.source}", notes


def stress(pre: Profile | None, post: Profile, settings: StressSettings) -> StressResult:
    """Stoney stress from the scans before (optional) and after deposition."""
    st = settings
    notes: list[Notice] = []
    if not st.film_thickness_nm or st.film_thickness_nm <= 0:
        raise ValueError("the film thickness is required (nm)")
    m_gpa, m_src, mnotes = resolve_modulus(st)
    notes += mnotes
    hs = st.substrate_thickness_um
    if not hs:
        sub = SUBSTRATES.get(st.substrate or "")
        if sub and sub.typical_thickness_um:
            hs = sub.typical_thickness_um
            notes.append(
                Notice(
                    N.SUBSTRATE_THICKNESS_TYPICAL,
                    N.WARNING,
                    f"substrate thickness not given; {hs:g} µm (typical for {sub.name}) was"
                    " used. The stress scales with its square - measure it",
                )
            )
        else:
            raise ValueError("the substrate thickness is required (µm)")
    fpost = fit_curvature(post, st.region_um)
    fpre = fit_curvature(pre, st.region_um) if pre is not None else None
    kpre = fpre.kappa_per_um if fpre else 0.0
    upre = fpre.u_kappa_per_um if fpre else 0.0
    if fpre is None:
        notes.append(
            Notice(
                N.STRESS_NO_REFERENCE,
                N.WARNING,
                "no scan before deposition: the substrate is assumed flat (1/R_pre = 0)."
                " Wafers are rarely flat to this level; scan the same line before deposition",
            )
        )
    dk = fpost.kappa_per_um - kpre
    udk = float(np.hypot(fpost.u_kappa_per_um, upre))
    sign = -1.0 if st.backside else 1.0
    s = sign * stoney_mpa(m_gpa, hs, st.film_thickness_nm, dk)
    us = abs(stoney_mpa(m_gpa, hs, st.film_thickness_nm, udk))
    if abs(dk) < 2 * udk:
        kind = "not significant"
        notes.append(
            Notice(
                N.CURVATURE_CHANGE_NOT_SIGNIFICANT,
                N.WARNING,
                f"the curvature change ({dk * 1e6:.3g} 1/m) is within twice its uncertainty"
                f" ({udk * 1e6:.3g} 1/m)",
            )
        )
    else:
        kind = "tensile" if s > 0 else "compressive"
    ratio = st.film_thickness_nm * 1e-3 / hs
    if ratio > 0.01:
        notes.append(
            Notice(
                N.STONEY_THICK_FILM,
                N.WARNING if ratio <= 0.05 else N.ERROR,
                f"the film is {ratio:.1%} of the substrate thickness; Stoney's equation assumes"
                " a film much thinner than the substrate and its error grows with this ratio"
                " (and with the film-to-substrate modulus ratio)",
                {"ratio": ratio},
            )
        )
    for label, f, prof in (("before", fpre, pre), ("after", fpost, post)):
        if f is None or prof is None:
            continue
        if f.sag_nm * 1e-3 > 0.1 * hs:
            notes.append(
                Notice(
                    N.STONEY_LARGE_DEFLECTION,
                    N.WARNING,
                    f"the deflection over the {label} scan ({f.sag_nm * 1e-3:.3g} µm) exceeds a"
                    f" tenth of the substrate thickness ({hs:g} µm); Stoney assumes small"
                    " deflections",
                )
            )
        noise = mad_sigma(np.diff(prof.z_nm[prof.valid])) / np.sqrt(2)
        if f.quartic_improvement > 2.0 and f.sigma_nm > 3 * max(noise, 1e-12):
            notes.append(
                Notice(
                    N.BOW_NOT_SPHERICAL,
                    N.WARNING,
                    f"the {label} scan is not a uniform curvature (a quartic fits"
                    f" {f.quartic_improvement:.1f}x better than the parabola): the curvature is"
                    " an average and the Stoney stress a mean value",
                )
            )
    if pre is not None:
        la, lb = post.length_um, pre.length_um
        if abs(la - lb) > 0.02 * max(la, lb) or abs(post.dx_um - pre.dx_um) > 0.05 * pre.dx_um:
            notes.append(
                Notice(
                    N.SCAN_MISMATCH,
                    N.WARNING,
                    f"the scans differ (length {lb:.5g} vs {la:.5g} µm, spacing {pre.dx_um:.3g}"
                    f" vs {post.dx_um:.3g} µm): scan the same line with the same settings",
                )
            )
    return StressResult(
        stress_mpa=s,
        u_stress_mpa=us,
        kind=kind,
        biaxial_modulus_gpa=m_gpa,
        modulus_source=m_src,
        substrate_thickness_um=float(hs),
        film_thickness_nm=float(st.film_thickness_nm),
        pre=fpre,
        post=fpost,
        delta_kappa_per_um=dk,
        u_delta_kappa_per_um=udk,
        notices=notes,
    )

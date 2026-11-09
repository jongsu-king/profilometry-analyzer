"""Profile roughness parameters (ISO 21920-2) with Gaussian filtering (ISO 16610-21).

Procedure
---------
1. The evaluated range (whole profile or a user range, e.g. one plateau) is put on a
   uniform grid; spike points are interpolated.
2. Form removal: a least-squares polynomial (default: a line, order 1).
3. ``lambda_s`` short-wave filter (Gaussian low-pass) -> primary profile.
4. ``lambda_c`` filter: the Gaussian mean line is the waviness, the remainder the
   roughness profile.
5. Half a cut-off is discarded at each end; the evaluation length is the largest whole
   number of section lengths (``lsc = lambda_c``) that fits, centred.

Parameters
----------
Ra, Rq, Rsk, Rku and Rt over the evaluation length; Rp, Rv and Rz as the mean of the
section values (ISO 21920-2 evaluates Rz, Rp and Rv per section); Rsm is the mean
width of the profile elements over the evaluation length, with the height
discrimination at 10 % of Rz and the width discrimination at 1 % of the section length.
The Rsm discrimination follows the ISO 4287 convention; the ISO 21920-2 text was not
available when this was written and the result is labelled accordingly.

Default cut-offs
----------------
When ``lambda_c`` is not given, the setting class (ISO 21920-3; the values below follow
the class table as summarised by Polytec (2023), the original text was not checked) is
chosen from the measured Ra, iterating until the class no longer changes.  The standard
chooses the class from the specification on the drawing; choosing it from the
measurement is reported with a notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import notices as N
from ..notices import Notice
from ..profile import Profile
from .edges import detect_edges
from .filters import kernel_points, mean_line
from .stats import uniform_grid


@dataclass(frozen=True)
class SettingClass:
    name: str
    lambda_s_um: float
    lambda_c_um: float
    max_spacing_um: float
    ra_max_um: float
    """Upper Ra limit of the class (``inf`` for the last)."""


SETTING_CLASSES = (
    SettingClass("Sc1", 2.5, 80.0, 0.5, 0.02),
    SettingClass("Sc2", 2.5, 250.0, 0.5, 0.1),
    SettingClass("Sc3", 2.5, 800.0, 0.5, 2.0),
    SettingClass("Sc4", 8.0, 2500.0, 1.5, 10.0),
    SettingClass("Sc5", 25.0, 8000.0, 5.0, float("inf")),
)
SETTING_CLASS_SOURCE = (
    "ISO 21920-3 setting classes as summarised by Polytec (2023); original text not checked"
)


def class_for_ra(ra_um: float) -> SettingClass:
    for sc in SETTING_CLASSES:
        if ra_um <= sc.ra_max_um:
            return sc
    return SETTING_CLASSES[-1]


def class_for_cutoff(lambda_c_um: float) -> SettingClass | None:
    for sc in SETTING_CLASSES:
        if abs(sc.lambda_c_um - lambda_c_um) <= 1e-6 * sc.lambda_c_um:
            return sc
    return None


@dataclass
class RoughnessSettings:
    lambda_c_um: float | None = None
    """Cut-off of the roughness filter; ``None``: setting class from the measured Ra."""
    lambda_s_um: float | None = None
    """Short-wave cut-off; ``None``: the value of the setting class; ``0``: no λs filter."""
    n_sections: int = 5
    form_order: int = 1
    region_um: tuple[float, float] | None = None
    height_discrimination: float = 0.10
    width_discrimination: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_c_um": self.lambda_c_um,
            "lambda_s_um": self.lambda_s_um,
            "n_sections": self.n_sections,
            "form_order": self.form_order,
            "region_um": list(self.region_um) if self.region_um else None,
            "height_discrimination": self.height_discrimination,
            "width_discrimination": self.width_discrimination,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RoughnessSettings:
        d = dict(d or {})
        if d.get("region_um"):
            d["region_um"] = tuple(d["region_um"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class RoughnessResult:
    parameters: dict[str, float]
    """Ra, Rq, Rz, Rp, Rv, Rt, Rsk, Rku (nm or -), Rsm (µm)."""
    lambda_c_um: float
    lambda_s_um: float | None
    setting_class: str | None
    n_sections: int
    evaluation_um: tuple[float, float]
    dx_um: float
    x_um: np.ndarray
    primary_nm: np.ndarray
    waviness_nm: np.ndarray
    roughness_nm: np.ndarray
    evaluated: np.ndarray
    sections: list[dict[str, float]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    notices: list[Notice] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "units": {
                k: ("µm" if k == "Rsm" else "" if k in ("Rsk", "Rku") else "nm")
                for k in self.parameters
            },
            "lambda_c_um": self.lambda_c_um,
            "lambda_s_um": self.lambda_s_um,
            "setting_class": self.setting_class,
            "n_sections": self.n_sections,
            "evaluation_um": list(self.evaluation_um),
            "evaluation_length_um": self.evaluation_um[1] - self.evaluation_um[0],
            "dx_um": self.dx_um,
            "sections": list(self.sections),
            "iterations": list(self.iterations),
            "rsm_discrimination": "ISO 4287 convention (10 % of Rz, 1 % of section length)",
            "notices": [n.to_dict() for n in self.notices],
        }


# ---------------------------------------------------------------- parameters
def amplitude_parameters(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=float)
    rq = float(np.sqrt(np.mean(r**2)))
    return {
        "Ra": float(np.mean(np.abs(r))),
        "Rq": rq,
        "Rsk": float(np.mean(r**3) / rq**3) if rq > 0 else float("nan"),
        "Rku": float(np.mean(r**4) / rq**4) if rq > 0 else float("nan"),
        "Rt": float(np.max(r) - np.min(r)),
    }


def section_parameters(r: np.ndarray, n_sections: int) -> tuple[dict[str, float], list[dict]]:
    parts = np.array_split(np.asarray(r, dtype=float), max(n_sections, 1))
    rows = []
    for i, p in enumerate(parts):
        rp, rv = float(np.max(p)), float(-np.min(p))
        rows.append({"section": i + 1, "Rp": rp, "Rv": rv, "Rz": rp + rv})
    return {
        "Rp": float(np.mean([s["Rp"] for s in rows])),
        "Rv": float(np.mean([s["Rv"] for s in rows])),
        "Rz": float(np.mean([s["Rz"] for s in rows])),
    }, rows


def mean_element_width(x: np.ndarray, r: np.ndarray, height_min: float, width_min: float) -> float:
    """Rsm: mean width of the profile elements (peak + adjacent valley).

    The profile is cut at the mean line into runs of one sign.  Runs whose extreme is
    below ``height_min`` or whose width is below ``width_min`` are merged with their
    neighbours (smallest first) until every run passes.  Complete elements are the
    pairs of runs between the first and the last crossing; incomplete runs at the ends
    are not counted.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    s = np.where(r >= 0, 1, -1)
    change = np.flatnonzero(np.diff(s) != 0) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [r.size]])

    # crossing positions by linear interpolation
    def xcross(i: int) -> float:
        if i <= 0:
            return float(x[0])
        if i >= r.size:
            return float(x[-1])
        a, b = r[i - 1], r[i]
        t = a / (a - b) if a != b else 0.5
        return float(x[i - 1] + t * (x[i] - x[i - 1]))

    seg = []
    for a, b in zip(starts, stops):
        sign = int(s[a])
        ext = float(np.max(np.abs(r[a:b])))
        seg.append([xcross(int(a)), xcross(int(b)), sign, ext])
    # merge insignificant interior runs
    while len(seg) > 3:
        cand = [
            (i, seg[i][3], seg[i][1] - seg[i][0])
            for i in range(1, len(seg) - 1)
            if seg[i][3] < height_min or (seg[i][1] - seg[i][0]) < width_min
        ]
        if not cand:
            break
        i = min(cand, key=lambda c: c[1] / max(height_min, 1e-300) + c[2] / max(width_min, 1e-300))[
            0
        ]
        left, right = seg[i - 1], seg[i + 1]
        merged = [left[0], right[1], left[2], max(left[3], right[3])]
        seg[i - 1 : i + 2] = [merged]
    interior = seg[1:-1]
    n_pairs = len(interior) // 2
    if n_pairs < 1:
        return float("nan")
    used = interior[: 2 * n_pairs]
    return float((used[-1][1] - used[0][0]) / n_pairs)


# ---------------------------------------------------------------- pipeline
def _form_removed(x: np.ndarray, z: np.ndarray, order: int) -> np.ndarray:
    if order <= 0:
        return z - float(np.mean(z))
    t = (x - x.mean()) / max(np.ptp(x) / 2, 1e-12)
    coef = np.polyfit(t, z, order)
    return z - np.polyval(coef, t)


def _evaluate(
    x: np.ndarray,
    p: np.ndarray,
    dx: float,
    lc: float,
    ls: float | None,
    settings: RoughnessSettings,
) -> dict[str, Any]:
    notes: list[Notice] = []
    prim = p
    if ls and ls > 0:
        if ls < 2 * dx:
            notes.append(
                Notice(
                    N.LAMBDA_S_SKIPPED,
                    N.INFO,
                    f"λs = {ls:g} µm is shorter than two point spacings ({2 * dx:.3g} µm):"
                    " the λs filter was not applied",
                )
            )
        else:
            prim = mean_line(p, dx, ls)
    if kernel_points(dx, lc) > x.size:
        notes.append(
            Notice(
                N.FILTER_LONGER_THAN_DATA,
                N.WARNING,
                f"the λc filter window (±{lc:g} µm) is longer than the evaluated profile"
                f" ({np.ptp(x):.4g} µm); the mean line is dominated by end effects",
            )
        )
    wav = mean_line(prim, dx, lc)
    rough = prim - wav
    lo, hi = x[0] + lc / 2.0, x[-1] - lc / 2.0
    avail = hi - lo
    nsc = int(np.floor(avail / lc + 1e-9)) if avail > 0 else 0
    if nsc >= 1:
        le = nsc * lc
        mid = 0.5 * (lo + hi)
        a, b = mid - le / 2.0, mid + le / 2.0
    else:
        a, b = (lo, hi) if avail > 0 else (float(x[0]), float(x[-1]))
    ev = (x >= a - 1e-9) & (x <= b + 1e-9)
    if ev.sum() < 4:
        ev = np.ones_like(x, dtype=bool)
        a, b = float(x[0]), float(x[-1])
    r = rough[ev]
    params = amplitude_parameters(r)
    sec, rows = section_parameters(r, max(nsc, 1))
    params.update(sec)
    params["Rsm"] = mean_element_width(
        x[ev],
        r,
        settings.height_discrimination * params["Rz"],
        settings.width_discrimination * lc,
    )
    if nsc < 1:
        notes.append(
            Notice(
                N.EVALUATION_TOO_SHORT,
                N.ERROR,
                f"after discarding λc/2 at each end only {max(avail, 0):.4g} µm remain, less than"
                f" one section (λc = {lc:g} µm); the values are not comparable with ISO"
                " results. Scan longer or choose a smaller λc",
                {"available_um": float(max(avail, 0)), "lambda_c_um": lc},
            )
        )
    elif nsc < settings.n_sections:
        notes.append(
            Notice(
                N.EVALUATION_TOO_SHORT,
                N.WARNING,
                f"the evaluation length holds {nsc} section(s) of λc = {lc:g} µm instead of"
                f" {settings.n_sections}; a scan of at least"
                f" {(settings.n_sections + 1) * lc:.4g} µm is needed",
                {"n_sections": nsc, "lambda_c_um": lc},
            )
        )
    return {
        "params": params,
        "rows": rows,
        "nsc": nsc,
        "eval": (float(a), float(b)),
        "prim": prim,
        "wav": wav,
        "rough": rough,
        "ev": ev,
        "notes": notes,
    }


def roughness(profile: Profile, settings: RoughnessSettings | None = None) -> RoughnessResult:
    """Roughness parameters of a profile (see the module docstring)."""
    st = settings or RoughnessSettings()
    x, z, valid = profile.x_um, profile.z_nm, profile.valid
    sel = valid.copy()
    if st.region_um:
        a, b = sorted(st.region_um)
        sel &= (x >= a) & (x <= b)
    if sel.sum() < 10:
        raise ValueError("the evaluated range holds fewer than 10 valid points")
    i0, i1 = np.flatnonzero(sel)[[0, -1]]
    xs, zs, vs = x[i0 : i1 + 1], z[i0 : i1 + 1].copy(), valid[i0 : i1 + 1]
    if (~vs).any():
        idx = np.arange(xs.size)
        zs[~vs] = np.interp(idx[~vs], idx[vs], zs[vs])
    notes: list[Notice] = []
    xu, zu, resampled = uniform_grid(xs, zs)
    if resampled:
        notes.append(
            Notice(
                N.RESAMPLED,
                N.INFO,
                f"the point spacing is not uniform; resampled to {xu.size} points",
            )
        )
    dx = float(np.median(np.diff(xu)))
    p = _form_removed(xu, zu, st.form_order)

    iterations: list[dict[str, Any]] = []
    ls_used: float | None
    auto = st.lambda_c_um is None
    if auto:
        length = float(np.ptp(xu))
        fitting = [sc for sc in SETTING_CLASSES if 2 * sc.lambda_c_um <= length]
        sc = fitting[-1] if fitting else SETTING_CLASSES[0]
        sc = min(sc, SETTING_CLASSES[2], key=lambda c: c.lambda_c_um)
        seen: list[str] = []
        res: dict[str, Any] = {}
        for _ in range(6):
            ls = st.lambda_s_um if st.lambda_s_um is not None else sc.lambda_s_um
            res = _evaluate(xu, p, dx, sc.lambda_c_um, ls, st)
            ra_um = res["params"]["Ra"] / 1e3
            nxt = class_for_ra(ra_um)
            iterations.append(
                {"class": sc.name, "lambda_c_um": sc.lambda_c_um, "Ra_nm": ra_um * 1e3}
            )
            seen.append(sc.name)
            if nxt.name == sc.name or nxt.name in seen:
                if nxt.name != sc.name:
                    sc = min(sc, nxt, key=lambda c: c.lambda_c_um)
                    ls = st.lambda_s_um if st.lambda_s_um is not None else sc.lambda_s_um
                    res = _evaluate(xu, p, dx, sc.lambda_c_um, ls, st)
                break
            sc = nxt
        lc = sc.lambda_c_um
        ls_used = st.lambda_s_um if st.lambda_s_um is not None else sc.lambda_s_um
        cls_name: str | None = sc.name
        notes.append(
            Notice(
                N.SETTING_CLASS_FROM_MEASUREMENT,
                N.INFO,
                f"cut-offs from the measured Ra: class {sc.name} (λc = {lc:g} µm, λs ="
                f" {ls_used:g} µm). The standard chooses the class from the specification;"
                f" table: {SETTING_CLASS_SOURCE}",
                {"class": sc.name, "source": SETTING_CLASS_SOURCE},
            )
        )
    else:
        lc = float(st.lambda_c_um)  # type: ignore[arg-type]
        known = class_for_cutoff(lc)
        cls_name = known.name if known else None
        ls_used = (
            st.lambda_s_um if st.lambda_s_um is not None else (known.lambda_s_um if known else None)
        )
        res = _evaluate(xu, p, dx, lc, ls_used, st)
        sc = known if known else SettingClass("custom", ls_used or 0.0, lc, (ls_used or 0) / 5, 0)
    max_dx = sc.max_spacing_um if sc.max_spacing_um > 0 else None
    if max_dx and dx > max_dx * 1.0001:
        notes.append(
            Notice(
                N.SAMPLING_TOO_COARSE,
                N.WARNING,
                f"the point spacing ({dx:.3g} µm) is larger than the {max_dx:g} µm allowed for"
                f" λs = {ls_used if ls_used else 0:g} µm: short wavelengths are aliased and the"
                " λs filter is not meaningful. Use more points per scan",
                {"dx_um": dx, "max_spacing_um": max_dx},
            )
        )
    if ls_used and ls_used > 0 and dx > ls_used / 2:
        notes.append(
            Notice(
                N.SAMPLING_TOO_COARSE,
                N.WARNING,
                f"the point spacing ({dx:.3g} µm) exceeds half of λs ({ls_used:g} µm)",
            )
        )
    tip = profile.scan.tip_radius_um
    if tip and ls_used and tip > ls_used:
        notes.append(
            Notice(
                N.NARROW_FEATURE,
                N.INFO,
                f"the stylus radius ({tip:g} µm) is larger than λs ({ls_used:g} µm): the stylus"
                " itself filters short wavelengths more than λs does",
            )
        )
    # steps inside the evaluated range distort roughness
    try:
        eset = detect_edges(xu, zu, tip_radius_um=tip or 2.0, k_height=8.0)
        rq = res["params"]["Rq"]
        big = [e for e in eset.edges if abs(e.height_nm) > max(10 * rq, 1e-9)]
        if big:
            notes.append(
                Notice(
                    N.STEP_IN_ROUGHNESS,
                    N.WARNING,
                    f"{len(big)} step(s) inside the evaluated range (largest"
                    f" {max(abs(e.height_nm) for e in big):.4g} nm at"
                    f" {big[0].x_um:.4g} µm): restrict the range to one plateau",
                    {"x_um": [e.x_um for e in big]},
                )
            )
    except ValueError:
        pass
    notes = res["notes"] + notes
    return RoughnessResult(
        parameters=res["params"],
        lambda_c_um=lc,
        lambda_s_um=ls_used if ls_used else None,
        setting_class=cls_name,
        n_sections=res["nsc"],
        evaluation_um=res["eval"],
        dx_um=dx,
        x_um=xu,
        primary_nm=res["prim"],
        waviness_nm=res["wav"],
        roughness_nm=res["rough"],
        evaluated=res["ev"],
        sections=res["rows"],
        iterations=iterations,
        notices=N.unique(notes),
    )

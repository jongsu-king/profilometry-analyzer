"""Publication-style figures (PNG / SVG / PDF via Matplotlib).

The default ``"origin"`` style follows the OriginLab look used by the sister tool
xps-autofit-studio: white canvas, black frame on all four sides with inward ticks,
Arial-type labels, no grid, black data, red fits and a boxed legend.  ``"classic"`` is a
softer look with a light grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from ..analysis import RoughnessAnalysis, StepResult, StressAnalysis
    from ..batch import BatchResult

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FIGURE_STYLES = ("origin", "classic")
DEFAULT_FIGURE_STYLE = "origin"
FORMATS = ("png", "svg", "pdf")
ORIGIN_FONTS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
FIT = "#d62728"
RAW = "#000000"
MUTED = "#8c8c8c"
PLANE_FILL = "#2a78d6"
INNER_FILL = "#1baf7a"
EXCLUDED_FILL = "#bdbdbd"

ORIGIN_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ORIGIN_FONTS,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
}


@lru_cache(maxsize=1)
def origin_font_family() -> str:
    """The first of :data:`ORIGIN_FONTS` installed here (``DejaVu Sans`` at worst)."""
    for name in ORIGIN_FONTS:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        return name
    return "DejaVu Sans"


def _rc(style: str) -> Any:
    if style not in FIGURE_STYLES:
        raise ValueError(f"figure style must be one of {FIGURE_STYLES}, got {style!r}")
    if style != "origin":
        return {}
    return {**ORIGIN_RC, "font.sans-serif": [origin_font_family(), *ORIGIN_FONTS]}


def _style_axes(ax: Any, style: str) -> None:
    ax.set_facecolor("#ffffff")
    if style == "origin":
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.tick_params(which="both", direction="in", top=True, right=True, colors="black")
        ax.tick_params(which="major", labelsize=9, length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.minorticks_on()
        ax.grid(False)
    else:
        ax.grid(True, color="#e1e0d9", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)


def _legend(ax: Any, style: str, **kw: Any) -> None:
    if style == "origin":
        kw.setdefault("prop", {"family": origin_font_family(), "size": 7})
    else:
        kw.setdefault("fontsize", 7)
        kw.setdefault("frameon", False)
    ax.legend(**kw)


def save(
    fig: Figure, stem: str | Path, formats: Sequence[str] = ("png",), dpi: int = 200
) -> list[Path]:
    """Save ``fig`` as ``stem.<fmt>`` for every format; returns the paths."""
    out: list[Path] = []
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"figure format must be one of {FORMATS}")
        p = stem.with_name(stem.name + f".{fmt}")
        fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor())
        out.append(p)
    return out


def _height_scale(z_nm: np.ndarray) -> tuple[float, str]:
    span = float(np.nanmax(np.abs(z_nm))) if z_nm.size else 0.0
    return (1e3, "µm") if span >= 5e3 else (1.0, "nm")


def _span(
    ax: Any, region: tuple[float, float], color: str, alpha: float, label: str | None
) -> None:
    ax.axvspan(region[0], region[1], color=color, alpha=alpha, lw=0, label=label, zorder=0)


def plot_step(
    result: StepResult,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] = (6.6, 5.0),
) -> Figure:
    """Raw and levelled profile, the fitted planes and the result."""
    with plt.rc_context(_rc(style)):
        fig, (ax0, ax) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.06},
        )
        fig.patch.set_facecolor("#ffffff")
        family = origin_font_family() if style == "origin" else None
        font: dict[str, Any] = {"fontfamily": family} if family else {}
        for a in (ax0, ax):
            _style_axes(a, style)
        p, src = result.profile, result.source
        sc, unit = _height_scale(p.z_nm)
        xr = src.x_um
        ax0.plot(xr, src.z_nm / sc, color=MUTED, lw=0.7, label="as read")
        ax0.set_ylabel(f"z ({unit})", fontsize=9, **font)
        _legend(ax0, style, loc="best")
        x = p.x_um
        z = np.where(p.valid, p.z_nm, np.nan) / sc
        m = result.measurement
        first = True
        for r in m.regions_um.get("plane", []) + m.regions_um.get("outer", []):
            _span(ax, r, PLANE_FILL, 0.10, "plane region" if first else None)
            first = False
        for r in m.regions_um.get("inner", []):
            _span(ax, r, INNER_FILL, 0.15, "inner region (A1)")
        first = True
        for r in m.regions_um.get("transition", []):
            _span(ax, r, EXCLUDED_FILL, 0.35, "transition (excluded)" if first else None)
            first = False
        ax.plot(x, z, color=RAW, lw=0.8, label="levelled", zorder=2)
        bad = ~p.valid
        if bad.any():
            ax.plot(x[bad], p.z_nm[bad] / sc, "x", color=SERIES[1], ms=4, label="spike (excluded)")
        for k, plane in enumerate(m.planes):
            xs = np.linspace(plane.x_start_um, plane.x_stop_um, 50)
            ax.plot(
                xs,
                plane.line.at(xs) / sc,
                color=FIT,
                lw=1.4,
                label="plane fit" if k == 0 else None,
                zorder=3,
            )
            for b in plane.burrs:
                ax.axvspan(b["x_start_um"], b["x_stop_um"], color=SERIES[3], alpha=0.35, lw=0)
        for e in result.edges.edges:
            ax.axvline(e.x_um, color=SERIES[0], lw=0.8, ls="--", zorder=1)
        if m.thickness_nm is not None:
            u = f" ± {m.u_nm:.3g}" if m.u_nm is not None else ""
            label = {"groove": "depth", "ridge": "height", "periodic": "mean step"}.get(
                m.kind, "step"
            )
            txt = f"{label} = {m.thickness_nm:.4g}{u} nm\n{m.kind}, {m.method}"
        else:
            txt = (
                f"{m.kind}: "
                + ", ".join(
                    f"{r['height_nm']:.4g}" for r in m.per_edge if r.get("height_nm") is not None
                )
                + " nm"
            )
        ax.text(
            0.01,
            0.97,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.6},
            **font,
        )
        ax.set_xlabel("x (µm)", fontsize=9, **font)
        ax.set_ylabel(f"height ({unit})", fontsize=9, **font)
        _legend(ax, style, loc="lower right")
        if title:
            ax0.set_title(title, fontsize=10, loc="left", **font)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.10)
    return fig


def plot_roughness(
    analysis: RoughnessAnalysis,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] = (6.6, 5.0),
) -> Figure:
    r = analysis.result
    with plt.rc_context(_rc(style)):
        fig, (ax0, ax) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.06}
        )
        fig.patch.set_facecolor("#ffffff")
        family = origin_font_family() if style == "origin" else None
        font: dict[str, Any] = {"fontfamily": family} if family else {}
        for a in (ax0, ax):
            _style_axes(a, style)
        ax0.plot(r.x_um, r.primary_nm, color=RAW, lw=0.7, label="primary (form removed)")
        ax0.plot(
            r.x_um, r.waviness_nm, color=FIT, lw=1.2, label=f"mean line (λc = {r.lambda_c_um:g} µm)"
        )
        ax0.set_ylabel("z (nm)", fontsize=9, **font)
        _legend(ax0, style, loc="best")
        ev = r.evaluated
        ax.plot(
            r.x_um,
            np.where(ev, np.nan, r.roughness_nm),
            color=MUTED,
            lw=0.6,
            label="discarded (λc/2 ends)",
        )
        ax.plot(r.x_um, np.where(ev, r.roughness_nm, np.nan), color=RAW, lw=0.7, label="roughness")
        a, b = r.evaluation_um
        if r.n_sections >= 1:
            for edge in np.linspace(a, b, r.n_sections + 1):
                ax.axvline(edge, color=SERIES[0], lw=0.6, ls=":")
        ax.axhline(0, color=FIT, lw=0.8)
        pr = r.parameters
        txt = (
            f"Ra = {pr['Ra']:.4g} nm\nRq = {pr['Rq']:.4g} nm\nRz = {pr['Rz']:.4g} nm\n"
            f"Rsk = {pr['Rsk']:.3g}, Rku = {pr['Rku']:.3g}\nRsm = {pr['Rsm']:.4g} µm"
        )
        ax.text(
            0.01,
            0.97,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.6},
            **font,
        )
        ax.set_xlabel("x (µm)", fontsize=9, **font)
        ax.set_ylabel("roughness (nm)", fontsize=9, **font)
        _legend(ax, style, loc="lower right")
        if title:
            ax0.set_title(title, fontsize=10, loc="left", **font)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.10)
    return fig


def plot_stress(
    analysis: StressAnalysis,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] = (6.6, 4.4),
) -> Figure:
    r = analysis.result
    with plt.rc_context(_rc(style)):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#ffffff")
        family = origin_font_family() if style == "origin" else None
        font: dict[str, Any] = {"fontfamily": family} if family else {}
        _style_axes(ax, style)
        pairs = [("before", analysis.pre, r.pre, SERIES[0]), ("after", analysis.post, r.post, RAW)]
        for label, prof, fit, color in pairs:
            if prof is None or fit is None:
                continue
            x = prof.x_um / 1e3
            ax.plot(x, prof.z_nm / 1e3, color=color, lw=0.7, label=f"{label} deposition")
            a, b, c = fit.coefs
            xs = np.linspace(prof.x_um.min(), prof.x_um.max(), 200)
            zz = a * (xs - fit.x0_um) ** 2 + b * (xs - fit.x0_um) + c
            ax.plot(
                xs / 1e3, zz / 1e3, color=FIT, lw=1.2, ls="--", label=f"R = {fit.radius_m:.4g} m"
            )
        txt = (
            f"σ = {r.stress_mpa:.4g} ± {r.u_stress_mpa:.2g} MPa ({r.kind})\n"
            f"M_s = {r.biaxial_modulus_gpa:.4g} GPa, h_s = {r.substrate_thickness_um:g} µm,"
            f" h_f = {r.film_thickness_nm:g} nm"
        )
        ax.text(
            0.01,
            0.97,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.6},
            **font,
        )
        ax.set_xlabel("x (mm)", fontsize=9, **font)
        ax.set_ylabel("z (µm)", fontsize=9, **font)
        _legend(ax, style, loc="lower right")
        if title:
            ax.set_title(title, fontsize=10, loc="left", **font)
        fig.tight_layout()
    return fig


def plot_batch(
    batch: BatchResult,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> Figure:
    """Thickness of every scan (open symbols) and the sample mean ± standard deviation."""
    with plt.rc_context(_rc(style)):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#ffffff")
        family = origin_font_family() if style == "origin" else None
        font: dict[str, Any] = {"fontfamily": family} if family else {}
        _style_axes(ax, style)
        names = [s.sample for s in batch.samples]
        for i, s in enumerate(batch.samples):
            vals = [
                it.result.thickness_nm
                for it in batch.items
                if it.sample == s.sample and it.result is not None and it.result.thickness_nm
            ]
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
            ax.plot(
                i + jitter,
                vals,
                "o",
                mfc="none",
                mec=RAW,
                ms=5,
                mew=0.8,
                label="scan" if i == 0 else None,
            )
            if s.mean_nm is not None:
                ax.errorbar(
                    i + 0.3,
                    s.mean_nm,
                    yerr=s.std_nm or 0.0,
                    fmt="s",
                    color=FIT,
                    ms=5,
                    capsize=3,
                    lw=1.0,
                    label="mean ± s.d." if i == 0 else None,
                )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, **font)
        ax.minorticks_off()
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylabel("thickness (nm)", fontsize=9, **font)
        _legend(ax, style, loc="best")
        if title:
            ax.set_title(title, fontsize=10, loc="left", **font)
        fig.tight_layout()
    return fig


def close(fig: Figure) -> None:
    plt.close(fig)

"""Publication figures (PNG / TIFF / SVG / PDF via Matplotlib).

The default ``"origin"`` style follows the OriginLab look used by the sister tool
xps-autofit-studio: white canvas, black frame on all four sides with inward major and
minor ticks, Arial-type labels, no grid, black data, red fits and a boxed legend.
``"classic"`` is a softer look with a light grid.

Figures are laid out like Origin multi-layer graphs: stacked layers share the x axis
(profile as read, levelled profile with the plane fits, residuals of the fits), layers
carry panel letters, measured heights and widths are drawn as dimension lines with
their values, and :class:`FigureOptions` sets the journal size (single or double column,
in millimetres), the resolution, the font size and the axis units.  Text stays text in
SVG and PDF (TrueType embedding), so labels remain editable in Illustrator, Inkscape or
Origin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from ..annotations import Dimension, parabola_residuals, plane_residuals, step_dimensions  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from ..analysis import RoughnessAnalysis, StepResult, StressAnalysis
    from ..batch import BatchResult

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FIGURE_STYLES = ("origin", "classic")
DEFAULT_FIGURE_STYLE = "origin"
FORMATS = ("png", "tiff", "svg", "pdf")
FORMAT_ALIASES = {"tif": "tiff"}
ORIGIN_FONTS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
FIT = "#d62728"
RAW = "#000000"
MUTED = "#8c8c8c"
PLANE_FILL = "#2a78d6"
INNER_FILL = "#1baf7a"
EXCLUDED_FILL = "#bdbdbd"
BURR_FILL = "#eda100"
UNREACHED_FILL = "#6250d6"
MM = 1.0 / 25.4

# Journal-style figure sizes in millimetres (width, height).
SIZE_PRESETS: dict[str, tuple[float, float]] = {
    "single": (85.0, 75.0),
    "onehalf": (120.0, 95.0),
    "double": (174.0, 120.0),
    "slide": (240.0, 150.0),
}
Z_UNITS = ("auto", "Å", "nm", "µm")
X_UNITS = ("µm", "mm")

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
# Keep text as text in vector output (editable labels, searchable PDF).
VECTOR_RC: dict[str, Any] = {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}


@dataclass(frozen=True)
class FigureOptions:
    """Size, resolution, fonts, units and layers of a figure."""

    width_mm: float | None = None
    """``None``: the figure's own default size."""
    height_mm: float | None = None
    dpi: int = 300
    font_pt: float = 9.0
    x_unit: str = "µm"
    """``µm`` or ``mm``."""
    z_unit: str = "auto"
    """``auto`` (nm below 5 µm, else µm), ``Å``, ``nm`` or ``µm``."""
    annotate: bool = True
    """Dimension lines with the measured heights and widths."""
    residuals: bool = True
    """Layer with the residuals of the plane (or curvature) fits."""
    show_raw: bool = True
    """Layer with the profile as read (step figures)."""
    panel_labels: bool = True
    line_width: float = 0.8
    error_bar: str = "sd"
    """Batch chart error bars: ``sd``, ``se`` or ``ci95``."""
    title: bool = True

    def __post_init__(self) -> None:
        if self.x_unit not in X_UNITS:
            raise ValueError(f"x unit must be one of {X_UNITS}")
        if self.z_unit not in Z_UNITS:
            raise ValueError(f"z unit must be one of {Z_UNITS}")
        if self.error_bar not in ("sd", "se", "ci95"):
            raise ValueError("error bar must be sd, se or ci95")
        if self.dpi < 50 or self.dpi > 2400:
            raise ValueError("dpi must lie between 50 and 2400")

    def size_in(self, default_mm: tuple[float, float]) -> tuple[float, float]:
        w = self.width_mm or default_mm[0]
        h = self.height_mm or (default_mm[1] * w / default_mm[0])
        return w * MM, h * MM


def figure_options(
    size: str | None = None, **changes: Any
) -> FigureOptions:  # pragma: no branch - thin helper
    """Options from a size preset (``single``, ``onehalf``, ``double``, ``slide``) or a
    ``"WxH"`` string in millimetres, plus any field of :class:`FigureOptions`."""
    base = FigureOptions()
    if size:
        if size in SIZE_PRESETS:
            w, h = SIZE_PRESETS[size]
            font = 8.0 if size == "single" else 10.0 if size == "slide" else 9.0
            base = replace(base, width_mm=w, height_mm=h, font_pt=font)
        else:
            try:
                w_s, h_s = size.lower().replace("mm", "").split("x")
                base = replace(base, width_mm=float(w_s), height_mm=float(h_s))
            except ValueError as exc:
                raise ValueError(
                    f"figure size {size!r}: use {', '.join(SIZE_PRESETS)} or WIDTHxHEIGHT in mm"
                ) from exc
    clean = {k: v for k, v in changes.items() if v is not None}
    return replace(base, **clean)


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
        return dict(VECTOR_RC)
    return {**ORIGIN_RC, **VECTOR_RC, "font.sans-serif": [origin_font_family(), *ORIGIN_FONTS]}


@dataclass
class _Ctx:
    """Fonts and units of one figure."""

    style: str
    opts: FigureOptions
    font: dict[str, Any]

    @property
    def fs(self) -> float:
        return self.opts.font_pt

    @property
    def small(self) -> float:
        return max(self.opts.font_pt - 1.5, 5.0)

    @property
    def lw(self) -> float:
        return self.opts.line_width


def _ctx(style: str, opts: FigureOptions) -> _Ctx:
    family = origin_font_family() if style == "origin" else None
    return _Ctx(style, opts, {"fontfamily": family} if family else {})


def _style_axes(ax: Any, c: _Ctx) -> None:
    ax.set_facecolor("#ffffff")
    if c.style == "origin":
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.tick_params(which="both", direction="in", top=True, right=True, colors="black")
        ax.tick_params(which="major", labelsize=c.fs - 1, length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.minorticks_on()
        ax.grid(False)
    else:
        ax.grid(True, color="#e1e0d9", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=c.fs - 1)


def _legend(ax: Any, c: _Ctx, **kw: Any) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if c.style == "origin":
        kw.setdefault("prop", {"family": origin_font_family(), "size": c.small})
    else:
        kw.setdefault("fontsize", c.small)
        kw.setdefault("frameon", False)
    ax.legend(**kw)


def _panel(ax: Any, letter: str, c: _Ctx) -> None:
    if c.opts.panel_labels:
        ax.text(
            0.012,
            0.965,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=c.fs,
            fontweight="bold",
            **c.font,
        )


def _box(ax: Any, text: str, c: _Ctx, x: float = 0.07, y: float = 0.965) -> None:
    ax.text(
        x if c.opts.panel_labels else 0.012,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=c.small,
        bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.6, "pad": 2.5},
        zorder=6,
        **c.font,
    )


def _new_figure(c: _Ctx, default_mm: tuple[float, float]) -> Figure:
    w, h = c.opts.size_in(default_mm)
    fig = plt.figure(figsize=(w, h), dpi=c.opts.dpi, layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(h_pad=2 / 72, w_pad=2 / 72, hspace=0.0, wspace=0.02)  # type: ignore[call-arg]
    return fig


def normalize_formats(formats: Sequence[str]) -> list[str]:
    out = []
    for f in formats:
        f = FORMAT_ALIASES.get(f.lower(), f.lower())
        if f not in FORMATS:
            raise ValueError(f"figure format must be one of {FORMATS}")
        out.append(f)
    return out


def save(
    fig: Figure, stem: str | Path, formats: Sequence[str] = ("png",), dpi: int | None = None
) -> list[Path]:
    """Save ``fig`` as ``stem.<fmt>`` for every format; returns the paths.

    The resolution is the figure's own (set from :attr:`FigureOptions.dpi`) unless
    ``dpi`` is given.  TIFF files are LZW-compressed.
    """
    out: list[Path] = []
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(VECTOR_RC):  # type: ignore[arg-type]
        for fmt in normalize_formats(formats):
            p = stem.with_name(stem.name + f".{fmt}")
            kw: dict[str, Any] = {"dpi": dpi or "figure", "facecolor": fig.get_facecolor()}
            if fmt == "tiff":
                kw["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(p, **kw)
            out.append(p)
    return out


# ---------------------------------------------------------------- units
def z_scale(unit: str, z_nm: np.ndarray | None = None) -> tuple[float, str]:
    """Factor from nm to the display unit, and the unit."""
    if unit == "Å":
        return 10.0, "Å"
    if unit == "nm":
        return 1.0, "nm"
    if unit == "µm":
        return 1e-3, "µm"
    span = float(np.nanmax(np.abs(z_nm))) if z_nm is not None and np.size(z_nm) else 0.0
    return (1e-3, "µm") if span >= 5e3 else (1.0, "nm")


def x_scale(unit: str) -> tuple[float, str]:
    return (1e-3, "mm") if unit == "mm" else (1.0, "µm")


def _span(
    ax: Any, region: tuple[float, float], sx: float, color: str, alpha: float, label: str | None
) -> None:
    ax.axvspan(
        region[0] * sx, region[1] * sx, color=color, alpha=alpha, lw=0, label=label, zorder=0
    )


def draw_dimension(
    ax: Any, d: Dimension, sx: float, sz: float, c: _Ctx, x_range: float, z_range: float
) -> None:
    """A dimension line with arrow heads, extension ticks and its value."""
    x0, x1, z0, z1 = d.x0_um * sx, d.x1_um * sx, d.z0_nm * sz, d.z1_nm * sz
    if d.kind == "width" or not d.offset:
        ax.annotate(
            "",
            xy=(x1, z1),
            xytext=(x0, z0),
            arrowprops={
                "arrowstyle": "<|-|>",
                "lw": 0.8,
                "color": "black",
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 7,
            },
            zorder=5,
        )
    tick = 0.012
    if d.kind == "height":
        xa = x0 + 0.035 * x_range if d.offset else x0
        if d.offset:  # move the arrow beside the edge; extension lines show the levels
            ax.annotate(
                "",
                xy=(xa, z1),
                xytext=(xa, z0),
                arrowprops={
                    "arrowstyle": "<|-|>",
                    "lw": 0.8,
                    "color": "black",
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 7,
                },
                zorder=5,
            )
        for z in (z0, z1):
            ax.plot(
                [x0 - tick * x_range, xa + tick * x_range], [z, z], color="black", lw=0.6, zorder=5
            )
        ax.text(
            xa + 0.018 * x_range,
            0.5 * (z0 + z1),
            d.label,
            ha="left",
            va="center",
            fontsize=c.small,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85},
            zorder=6,
            **c.font,
        )
    else:
        for x in (x0, x1):
            ax.plot(
                [x, x], [z0 - tick * z_range, z0 + tick * z_range], color="black", lw=0.6, zorder=5
            )
        ax.text(
            0.5 * (x0 + x1),
            z0 + 0.03 * z_range,
            d.label,
            ha="center",
            va="bottom",
            fontsize=c.small,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85},
            zorder=6,
            **c.font,
        )


# ---------------------------------------------------------------- step
def _step_text(result: StepResult, unit: str, sz: float) -> str:
    m = result.measurement
    method = {"iso5436-a1": "ISO 5436-1 A1", "planes": "two planes", "regions": "cursors R, M"}
    head = f"{m.kind}, {method.get(m.method, m.method)}"
    if m.thickness_nm is None:
        heights = [r["height_nm"] for r in m.per_edge if r.get("height_nm") is not None]
        return head + "\n" + ", ".join(f"{h * sz:.4g}" for h in heights[:8]) + f" {unit}"
    word = {"groove": "d", "ridge": "h", "periodic": "mean"}.get(m.kind, "h")
    u = f" ± {m.u_nm * sz:.2g}" if m.u_nm is not None else ""
    lines = [head, f"{word} = {m.thickness_nm * sz:.5g}{u} {unit} (k = 1)"]
    if m.kind in ("groove", "ridge") and m.width_um:
        lines.append(f"W = {m.width_um:.4g} µm")
    if m.kind == "periodic" and m.stats.get("n"):
        sd = m.stats.get("std_nm")
        lines.append(f"n = {m.stats['n']}" + (f", SD = {sd * sz:.2g} {unit}" if sd else ""))
    return "\n".join(lines)


def plot_step(
    result: StepResult,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] | None = None,
    options: FigureOptions | None = None,
) -> Figure:
    """Stacked layers: profile as read, levelled profile with plane fits and dimension
    lines, residuals of the plane fits."""
    opts = options or FigureOptions()
    if figsize and not opts.width_mm:
        opts = replace(opts, width_mm=figsize[0] / MM, height_mm=figsize[1] / MM)
    c = _ctx(style, opts)
    p, src, m = result.profile, result.source, result.measurement
    sx, xu = x_scale(opts.x_unit)
    sz, zu = z_scale(opts.z_unit, np.where(p.valid, p.z_nm, np.nan))
    resid = plane_residuals(m, p) if (opts.residuals and m.planes) else []
    layers = (["raw"] if opts.show_raw else []) + ["main"] + (["resid"] if resid else [])
    ratios = {"raw": 1.0, "main": 3.0, "resid": 1.1}
    with plt.rc_context(_rc(style)):
        fig = _new_figure(c, (170.0, 135.0))
        axes = fig.subplots(
            len(layers), 1, sharex=True, gridspec_kw={"height_ratios": [ratios[k] for k in layers]}
        )
        axes = list(np.atleast_1d(axes))
        ax_of = dict(zip(layers, axes))
        for i, a in enumerate(axes):
            _style_axes(a, c)
            _panel(a, "abc"[i], c)
        ax = ax_of["main"]
        x = p.x_um * sx
        if "raw" in ax_of:
            ar = ax_of["raw"]
            s_raw, u_raw = z_scale(opts.z_unit, src.z_nm)
            ar.plot(src.x_um * sx, src.z_nm * s_raw, color=MUTED, lw=c.lw * 0.9, label="as read")
            ar.set_ylabel(f"z ({u_raw})", fontsize=c.fs, **c.font)
            _legend(ar, c, loc="lower right")
        z = np.where(p.valid, p.z_nm, np.nan) * sz
        first = True
        for r in m.regions_um.get("plane", []) + m.regions_um.get("outer", []):
            _span(ax, r, sx, PLANE_FILL, 0.10, "plane region" if first else None)
            first = False
        for r in m.regions_um.get("inner", []):
            _span(ax, r, sx, INNER_FILL, 0.15, "inner region (A1)")
        first = True
        for r in m.regions_um.get("transition", []):
            _span(ax, r, sx, EXCLUDED_FILL, 0.35, "transition (excluded)" if first else None)
            first = False
        reach = getattr(result, "tip_check", None)
        for i, r in enumerate(reach.unreached_um if reach is not None else []):
            ax.axvspan(
                r[0] * sx,
                r[1] * sx,
                facecolor="none",
                edgecolor=UNREACHED_FILL,
                hatch="////",
                lw=0,
                alpha=0.6,
                zorder=0,
                label="not touched by the stylus" if i == 0 else None,
            )
        ax.plot(x, z, color=RAW, lw=c.lw, label="levelled", zorder=2)
        bad = ~p.valid
        if bad.any():
            ax.plot(
                x[bad], p.z_nm[bad] * sz, "x", color=SERIES[1], ms=3.5, label="spike (excluded)"
            )
        burr_label = True
        for k, plane in enumerate(m.planes):
            xs = np.linspace(plane.x_start_um, plane.x_stop_um, 50)
            ax.plot(
                xs * sx,
                plane.line.at(xs) * sz,
                color=FIT,
                lw=c.lw * 1.7,
                label="plane fit" if k == 0 else None,
                zorder=3,
            )
            for b in plane.burrs:
                _span(
                    ax,
                    (b["x_start_um"], b["x_stop_um"]),
                    sx,
                    BURR_FILL,
                    0.4,
                    "burr (excluded)" if burr_label else None,
                )
                burr_label = False
        for e in result.edges.edges:
            ax.axvline(e.x_um * sx, color=SERIES[0], lw=0.6, ls="--", zorder=1)
        if opts.annotate:
            dims = step_dimensions(m, zu)
            finite = z[np.isfinite(z)]
            z_range = float(np.ptp(finite)) if finite.size else 1.0
            x_range = float(np.ptp(x)) or 1.0
            for d in dims:
                draw_dimension(ax, d, sx, sz, c, x_range, z_range or 1.0)
        _box(ax, _step_text(result, zu, sz), c)
        ax.set_ylabel(f"height ({zu})", fontsize=c.fs, **c.font)
        _legend(ax, c, loc="lower right")
        if resid:
            arz = ax_of["resid"]
            r_unit = "Å" if opts.z_unit == "Å" else "nm"
            rs = 10.0 if r_unit == "Å" else 1.0
            sig = []
            for k, rr in enumerate(resid):
                arz.plot(
                    rr.x_um * sx,
                    rr.r_nm * rs,
                    ".",
                    color=RAW,
                    ms=1.2,
                    label="used" if k == 0 else None,
                )
                if rr.x_excluded_um.size:
                    arz.plot(
                        rr.x_excluded_um * sx,
                        rr.r_excluded_nm * rs,
                        ".",
                        color=BURR_FILL,
                        ms=1.6,
                        label="excluded" if k == 0 else None,
                    )
                sig.append(rr.sigma_nm)
            arz.axhline(0.0, color=FIT, lw=0.8)
            s2 = 2.0 * float(np.median(sig)) * rs if sig else 0.0
            if s2 > 0:
                for v in (-s2, s2):
                    arz.axhline(v, color=MUTED, lw=0.6, ls="--")
                used = np.concatenate([rr.r_nm for rr in resid]) * rs
                lim = max(
                    4.0 * s2, float(np.percentile(np.abs(used), 99.5)) * 1.2 if used.size else s2
                )
                arz.set_ylim(-lim, lim)
            arz.set_ylabel(f"residual ({r_unit})", fontsize=c.fs, **c.font)
        axes[-1].set_xlabel(f"x ({xu})", fontsize=c.fs, **c.font)
        if title and opts.title:
            axes[0].set_title(title, fontsize=c.fs + 1, loc="left", **c.font)
    return fig


# ---------------------------------------------------------------- roughness
def plot_roughness(
    analysis: RoughnessAnalysis,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] | None = None,
    options: FigureOptions | None = None,
) -> Figure:
    """(a) primary profile and mean line, (b) parameter table, (c) roughness profile,
    (d) amplitude distribution, (e) material ratio (Abbott-Firestone) curve."""
    from ..core.roughness import amplitude_distribution, material_ratio_curve

    opts = options or FigureOptions()
    if figsize and not opts.width_mm:
        opts = replace(opts, width_mm=figsize[0] / MM, height_mm=figsize[1] / MM)
    c = _ctx(style, opts)
    r = analysis.result
    sx, xu = x_scale(opts.x_unit)
    sz, zu = z_scale(opts.z_unit, r.primary_nm)
    ev = r.evaluated
    rr = r.roughness_nm[ev]
    with plt.rc_context(_rc(style)):
        fig = _new_figure(c, (174.0, 120.0))
        gs = fig.add_gridspec(2, 3, width_ratios=[3.2, 1.0, 1.0], height_ratios=[1.0, 1.25])
        ax_p = fig.add_subplot(gs[0, 0])
        ax_t = fig.add_subplot(gs[0, 1:])
        ax_r = fig.add_subplot(gs[1, 0], sharex=ax_p)
        ax_a = fig.add_subplot(gs[1, 1], sharey=ax_r)
        ax_m = fig.add_subplot(gs[1, 2], sharey=ax_r)
        for a in (ax_p, ax_r, ax_a, ax_m):
            _style_axes(a, c)
        x = r.x_um * sx
        ax_p.plot(x, r.primary_nm * sz, color=RAW, lw=c.lw * 0.9, label="primary")
        ax_p.plot(
            x,
            r.waviness_nm * sz,
            color=FIT,
            lw=c.lw * 1.6,
            label=f"mean line, λc = {r.lambda_c_um:g} µm",
        )
        ax_p.set_ylabel(f"z ({zu})", fontsize=c.fs, **c.font)
        ax_p.tick_params(labelbottom=False)
        _legend(ax_p, c, loc="lower right")
        ax_r.plot(
            x,
            np.where(ev, np.nan, r.roughness_nm * sz),
            color=MUTED,
            lw=c.lw * 0.8,
            label="discarded (λc/2)",
        )
        ax_r.plot(
            x,
            np.where(ev, r.roughness_nm * sz, np.nan),
            color=RAW,
            lw=c.lw * 0.9,
            label="roughness",
        )
        a0, b0 = r.evaluation_um
        if r.n_sections >= 1:
            for edge in np.linspace(a0, b0, r.n_sections + 1):
                ax_r.axvline(edge * sx, color=SERIES[0], lw=0.6, ls=":")
        ax_r.axhline(0, color=FIT, lw=0.8)
        ax_r.set_xlabel(f"x ({xu})", fontsize=c.fs, **c.font)
        ax_r.set_ylabel(f"roughness ({zu})", fontsize=c.fs, **c.font)
        _legend(ax_r, c, loc="lower right")
        h, dens = amplitude_distribution(rr)
        if h.size:
            ax_a.fill_betweenx(h * sz, 0, dens / sz, color=SERIES[0], alpha=0.25, lw=0)
            ax_a.plot(dens / sz, h * sz, color=SERIES[0], lw=c.lw)
        ax_a.set_xlabel(f"density (1/{zu})", fontsize=c.fs, **c.font)
        ax_a.tick_params(labelleft=False)
        ax_a.set_xlim(left=0)
        mr, cc = material_ratio_curve(rr)
        if mr.size:
            ax_m.plot(mr, cc * sz, color=RAW, lw=c.lw * 1.2)
        ax_m.set_xlim(0, 100)
        ax_m.set_xlabel("material ratio (%)", fontsize=c.fs, **c.font)
        ax_m.tick_params(labelleft=False)
        for a, letter in ((ax_p, "a"), (ax_t, "b"), (ax_r, "c"), (ax_a, "d"), (ax_m, "e")):
            _panel(a, letter, c)
        ax_t.axis("off")
        pr = r.parameters
        rows = []
        for k in ("Ra", "Rq", "Rz", "Rp", "Rv", "Rt", "Rsk", "Rku", "Rsm"):
            if k not in pr:
                continue
            unit = "µm" if k == "Rsm" else "" if k in ("Rsk", "Rku") else zu
            val = pr[k] if k in ("Rsm", "Rsk", "Rku") else pr[k] * sz
            rows.append([k, f"{val:.4g}", unit])
        rows.append(["λc", f"{r.lambda_c_um:g}", "µm"])
        rows.append(["λs", f"{r.lambda_s_um or 0:g}", "µm"])
        rows.append(["ln", f"{b0 - a0:.4g}", "µm"])
        tab = ax_t.table(
            cellText=rows,
            colLabels=["parameter", "value", "unit"],
            cellLoc="left",
            colLoc="left",
            edges="horizontal",
            bbox=Bbox.from_bounds(0.14, 0.0, 0.84, 0.86),
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(c.small)
        for cell in tab.get_celld().values():
            cell.set_linewidth(0.5)
            if c.font:
                cell.get_text().set_fontfamily(c.font["fontfamily"])
        cls = f"class {r.setting_class}, " if r.setting_class else ""
        ax_t.text(
            0.14,
            0.93,
            f"ISO 21920-2 ({cls}{r.n_sections} sections)",
            transform=ax_t.transAxes,
            fontsize=c.small,
            va="bottom",
            **c.font,
        )
        if title and opts.title:
            ax_p.set_title(title, fontsize=c.fs + 1, loc="left", **c.font)
    return fig


# ---------------------------------------------------------------- stress
def plot_stress(
    analysis: StressAnalysis,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] | None = None,
    options: FigureOptions | None = None,
) -> Figure:
    """(a) scans before and after deposition with the curvature fits, (b) fit residuals."""
    opts = options or FigureOptions(x_unit="mm", z_unit="µm")
    if figsize and not opts.width_mm:
        opts = replace(opts, width_mm=figsize[0] / MM, height_mm=figsize[1] / MM)
    c = _ctx(style, opts)
    r = analysis.result
    sx, xu = x_scale(opts.x_unit)
    zs = analysis.post.z_nm
    sz, zu = z_scale(opts.z_unit, zs)
    pairs = [("before", analysis.pre, r.pre, SERIES[0]), ("after", analysis.post, r.post, RAW)]
    with plt.rc_context(_rc(style)):
        fig = _new_figure(c, (170.0, 125.0))
        n_layers = 2 if opts.residuals else 1
        axes = list(
            np.atleast_1d(
                fig.subplots(
                    n_layers, 1, sharex=True, gridspec_kw={"height_ratios": [3.0, 1.1][:n_layers]}
                )
            )
        )
        for i, a in enumerate(axes):
            _style_axes(a, c)
            _panel(a, "ab"[i], c)
        ax = axes[0]
        for label, prof, fit, color in pairs:
            if prof is None or fit is None:
                continue
            ax.plot(
                prof.x_um * sx, prof.z_nm * sz, color=color, lw=c.lw, label=f"{label} deposition"
            )
            xs = np.linspace(prof.x_um.min(), prof.x_um.max(), 200)
            a2, b2, c2 = fit.coefs
            zz = a2 * (xs - fit.x0_um) ** 2 + b2 * (xs - fit.x0_um) + c2
            ax.plot(
                xs * sx,
                zz * sz,
                color=FIT,
                lw=c.lw * 1.4,
                ls="--",
                label=f"R = {fit.radius_m:.4g} m",
            )
            if opts.residuals:
                res = parabola_residuals(prof.x_um, prof.z_nm, fit)
                axes[1].plot(prof.x_um * sx, res, ".", color=color, ms=1.0, label=label)
        txt = (
            f"σ = {r.stress_mpa:.4g} ± {r.u_stress_mpa:.2g} MPa ({r.kind})\n"
            f"E/(1−ν) = {r.biaxial_modulus_gpa:.4g} GPa\n"
            f"h_s = {r.substrate_thickness_um:g} µm, h_f = {r.film_thickness_nm:g} nm"
        )
        _box(ax, txt, c)
        ax.set_ylabel(f"z ({zu})", fontsize=c.fs, **c.font)
        _legend(ax, c, loc="best")
        if opts.residuals:
            axes[1].axhline(0, color=FIT, lw=0.8)
            axes[1].set_ylabel("residual (nm)", fontsize=c.fs, **c.font)
            _legend(axes[1], c, loc="lower right", ncol=2, markerscale=6)
        axes[-1].set_xlabel(f"x ({xu})", fontsize=c.fs, **c.font)
        if title and opts.title:
            ax.set_title(title, fontsize=c.fs + 1, loc="left", **c.font)
    return fig


# ---------------------------------------------------------------- batch
def plot_batch(
    batch: BatchResult,
    title: str = "",
    style: str = DEFAULT_FIGURE_STYLE,
    figsize: tuple[float, float] | None = None,
    options: FigureOptions | None = None,
) -> Figure:
    """Column chart in the Origin manner: mean thickness per sample as columns, error
    bars (SD, SE or 95 % CI), the individual scans as open symbols and n above each
    column."""
    opts = options or FigureOptions()
    if figsize and not opts.width_mm:
        opts = replace(opts, width_mm=figsize[0] / MM, height_mm=figsize[1] / MM)
    c = _ctx(style, opts)
    samples = batch.samples
    means = np.array([s.mean_nm if s.mean_nm is not None else np.nan for s in samples], dtype=float)
    finite = means[np.isfinite(means)]
    sz, zu = z_scale(opts.z_unit, finite if finite.size else None)
    err_name = {"sd": "SD", "se": "SE", "ci95": "95% CI"}[opts.error_bar]

    def err(s: Any) -> float:
        v = {"sd": s.std_nm, "se": s.sem_nm, "ci95": s.ci95_nm}[opts.error_bar]
        return float(v) * sz if v is not None else 0.0

    with plt.rc_context(_rc(style)):
        fig = _new_figure(c, (120.0, 90.0))
        ax = fig.add_subplot(1, 1, 1)
        _style_axes(ax, c)
        idx = np.arange(len(samples))
        heights = np.nan_to_num(means * sz)
        from matplotlib.colors import to_rgba

        ax.bar(
            idx,
            heights,
            width=0.6,
            color=[to_rgba(SERIES[i % len(SERIES)], 0.35) for i in range(len(samples))],
            edgecolor="black",
            linewidth=0.8,
            label=f"mean ± {err_name}",
            zorder=1,
        )
        errs = [err(s) for s in samples]
        ax.errorbar(
            idx,
            heights,
            yerr=errs,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=4,
            capthick=0.9,
            zorder=3,
        )
        top = 0.0
        tops: list[float] = []
        for i, s in enumerate(samples):
            vals = [
                float(it.result.thickness_nm) * sz
                for it in batch.items
                if it.sample == s.sample and it.result is not None and it.result.thickness_nm
            ]
            if vals:
                jitter = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.zeros(1)
                ax.plot(
                    i + jitter,
                    vals,
                    "o",
                    mfc="white",
                    mec=RAW,
                    ms=4,
                    mew=0.8,
                    label="scan" if i == 0 else None,
                    zorder=4,
                )
                top = max(top, max(vals))
            hi = max([heights[i] + errs[i], *vals])
            tops.append(hi)
            top = max(top, hi)
        for i, s in enumerate(samples):
            ax.text(
                i,
                tops[i] + 0.025 * top,
                f"n = {s.n}",
                ha="center",
                va="bottom",
                fontsize=c.small,
                **c.font,
            )
        ax.set_xticks(idx)
        ax.set_xticklabels([s.sample for s in samples], fontsize=c.fs - 1, **c.font)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlim(-0.6, len(samples) - 0.4)
        ax.set_ylim(0, top * 1.18 if top > 0 else 1)
        ax.set_ylabel(f"thickness ({zu})", fontsize=c.fs, **c.font)
        _legend(ax, c, loc="upper left")
        if title and opts.title:
            ax.set_title(title, fontsize=c.fs + 1, loc="left", **c.font)
    return fig


def close(fig: Figure) -> None:
    plt.close(fig)

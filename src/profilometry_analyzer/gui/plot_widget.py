"""Live profile view and batch chart.

The profile view shows the profile as read (optional, grey), the processed profile,
plane regions and plane fits, burrs and spikes, the measured heights and widths as
dimension lines, and - in a second layer that shares the x axis - the residuals of the
plane fits, like a two-layer Origin graph.  Two draggable reference regions (``R`` and
``M``, like the reference and measurement cursors of the instrument software) and one
draggable line per detected edge let the user correct the analysis: releasing a cursor
emits ``regionsEdited`` or ``edgesEdited`` and the main window re-runs the analysis.

Display units (µm or mm, Å, nm or µm), axis limits, fonts and line widths come from
:class:`~.graph_options.GraphOptions`; every signal still carries µm and nm.  The mouse
works as a data reader (the cross snaps to the nearest data point), Shift-click sets
two points A and B and reports Δx and Δz, and a double click opens the Plot Details.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..analysis import RoughnessAnalysis, StepResult, StressAnalysis
from ..annotations import Dimension, parabola_residuals, plane_residuals, step_dimensions
from ..profile import Profile
from .graph_options import GraphOptions
from .theme import PlotStyle, plot_style, rgba

_AXES = ("left", "bottom", "top", "right")
_BASE_WIDTH = 1.2


def _finite_span(*arrays: np.ndarray) -> float:
    vals = [np.abs(a[np.isfinite(a)]) for a in arrays if a is not None and np.size(a)]
    vals = [v for v in vals if v.size]
    return float(max(v.max() for v in vals)) if vals else 0.0


class ProfilePlot(QWidget):
    regionsEdited = Signal(list)
    """``[(a, b), (c, d)]`` in µm after a reference cursor (R, M) was moved."""
    edgesEdited = Signal(list)
    """Edge positions (µm) after an edge line was moved."""
    cursorMoved = Signal(float, float)
    """Data reader: the data point nearest to the mouse (µm, nm)."""
    pointsMeasured = Signal(object)
    """Shift-click A and B: ``{"a": (x, z), "b": (x, z), "dx_um", "dz_nm"}``."""
    detailsRequested = Signal()
    """Double click on the graph."""

    def __init__(
        self,
        parent: QWidget | None = None,
        style: PlotStyle | None = None,
        options: GraphOptions | None = None,
    ) -> None:
        super().__init__(parent)
        self.look: PlotStyle = style or plot_style()
        self.opts: GraphOptions = options or GraphOptions()
        self.view = pg.GraphicsLayoutWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.main = self.view.addPlot(row=0, col=0)
        self.resid = pg.PlotItem()
        self.resid.setXLink(self.main)
        self._resid_shown = False
        self.legend = self.main.addLegend(offset=(-10, 10))
        self.show_raw = True
        self._edge_lines: list[pg.InfiniteLine] = []
        self._cursors: list[pg.LinearRegionItem] = []
        self._cursor_labels: list[pg.TextItem] = []
        self._cursor_spec: list[tuple[float, float]] = []
        """Cursor positions in µm; kept when the plot is redrawn."""
        self._last: tuple[str, tuple[Any, ...]] | None = None
        self._title = ""
        self._sx = 1.0
        self._sz = 1.0
        self._zu = "nm"
        self.dimension_items: list[Any] = []
        self._reader: tuple[np.ndarray, np.ndarray] | None = None
        self._snapped: tuple[float, float] | None = None
        self.measure_points: list[tuple[float, float]] = []
        self._measure_items: list[Any] = []
        self._cross = pg.InfiniteLine(angle=90, movable=False)
        self._cross.setZValue(-10)
        self._cross.hide()
        self._marker = pg.ScatterPlotItem(size=9, symbol="o", brush=pg.mkBrush(None))
        self._marker.setZValue(20)
        self._proxy = pg.SignalProxy(
            self.main.scene().sigMouseMoved, rateLimit=40, slot=self._mouse_moved
        )
        self.main.scene().sigMouseClicked.connect(self._clicked)
        self._apply_style()

    # ------------------------------------------------------------------ units
    @property
    def x_unit(self) -> str:
        return self.opts.x_unit

    @property
    def z_unit(self) -> str:
        return self._zu

    def _scale_for(self, *z_nm: np.ndarray) -> None:
        self._sx = self.opts.sx
        self._sz, self._zu = self.opts.z_factor(_finite_span(*z_nm))
        self._labels()

    def format_point(self, x_um: float, z_nm: float) -> str:
        return (
            f"x = {x_um * self._sx:.5g} {self.opts.x_unit}   z = {z_nm * self._sz:.5g} {self._zu}"
        )

    def _lw(self, w: float) -> float:
        return w * self.opts.line_width / _BASE_WIDTH

    # ------------------------------------------------------------------ style
    def set_style(self, style: PlotStyle) -> None:
        self.look = style
        self._apply_style()
        self._replay()

    def set_options(self, options: GraphOptions) -> None:
        self.opts = options
        self._apply_style()
        self._replay()

    def _replay(self) -> None:
        if self._last is not None:
            method, args = self._last
            getattr(self, method)(*args)

    def _plots(self) -> tuple[pg.PlotItem, ...]:
        return (self.main, self.resid)

    def _apply_style(self) -> None:
        s = self.look
        self.view.setBackground(s.canvas)
        font = QFont(s.font_family) if s.font_family else QFont()
        font.setPointSize(int(self.opts.font_pt))
        for plot in self._plots():
            for name in _AXES:
                ax = plot.getAxis(name)
                ax.setPen(pg.mkPen(s.frame, width=1))
                ax.setTextPen(pg.mkPen(s.text))
                ax.setTickFont(font)
                ax.setStyle(tickLength=-6 if s.ticks_inward else 5, tickTextOffset=6)
                ax.enableAutoSIPrefix(False)  # units are chosen in Plot Details
            if s.box:
                plot.showAxis("top")
                plot.showAxis("right")
                plot.getAxis("top").setStyle(showValues=False)
                plot.getAxis("right").setStyle(showValues=False)
            else:
                plot.hideAxis("top")
                plot.hideAxis("right")
            plot.showGrid(x=s.grid, y=s.grid, alpha=0.15)
        self._labels()
        self.legend.setLabelTextColor(s.text)
        self.legend.setPen(pg.mkPen(s.legend_pen) if s.legend_pen else pg.mkPen(None))
        self.legend.setBrush(pg.mkBrush(*s.legend_fill))
        self.legend.setVisible(self.opts.show_legend)
        self._cross.setPen(pg.mkPen(s.crosshair, width=1))
        self._marker.setPen(pg.mkPen(s.series[1], width=1.5))
        self.set_title(self._title)

    def _label_style(self) -> dict[str, str]:
        s = self.look
        extra = {"color": s.text, "font-size": f"{self.opts.font_pt + 1}pt"}
        if s.font_family:
            extra["font-family"] = s.font_family
        return extra

    def _labels(self, z_title: str = "height") -> None:
        extra = self._label_style()
        if self._resid_shown:  # the x axis title sits under the residual layer
            self.main.setLabel("bottom", "")
        else:
            self.main.setLabel("bottom", f"x ({self.opts.x_unit})", **extra)
        self.main.setLabel("left", f"{z_title} ({self._zu})", **extra)
        r_unit = "Å" if self.opts.z_unit == "Å" else "nm"
        self.resid.setLabel("bottom", f"x ({self.opts.x_unit})", **extra)
        self.resid.setLabel("left", f"residual ({r_unit})", **extra)

    def set_title(self, title: str) -> None:
        self._title = title
        kw: dict[str, Any] = {"color": self.look.text, "size": f"{self.opts.font_pt + 2}pt"}
        self.main.setTitle(title or None, **kw)

    # ------------------------------------------------------------------ layers
    def _show_resid(self, on: bool) -> None:
        if on and not self._resid_shown:
            self.view.addItem(self.resid, row=1, col=0)
            layout = self.view.ci.layout
            layout.setRowStretchFactor(0, 3)
            layout.setRowStretchFactor(1, 1)
            self.main.getAxis("bottom").setStyle(showValues=False)
            self._resid_shown = True
            self._labels()
        elif not on and self._resid_shown:
            self.view.removeItem(self.resid)
            self.main.getAxis("bottom").setStyle(showValues=True)
            self._resid_shown = False
            self._labels()

    @property
    def residuals_visible(self) -> bool:
        return self._resid_shown

    # ------------------------------------------------------------------ drawing
    def _reset(self) -> None:
        for line in self._edge_lines:
            with contextlib.suppress(RuntimeError, TypeError):
                line.sigPositionChangeFinished.disconnect()
        self.main.clear()
        self.resid.clear()
        self.legend.clear()
        self._edge_lines = []
        self._cursors = []
        self._cursor_labels = []
        self.dimension_items = []
        self._measure_items = []
        self.measure_points = []
        self._reader = None
        self._snapped = None
        self.main.addItem(self._cross, ignoreBounds=True)
        self.main.addItem(self._marker, ignoreBounds=True)
        self._marker.setData([], [])

    def _finish(self) -> None:
        """Axis limits (or autorange), then the cursors where the user left them."""
        o = self.opts
        vb = self.main.getViewBox()
        vb.enableAutoRange()
        if o.x_min is not None and o.x_max is not None:
            self.main.setXRange(o.x_min, o.x_max, padding=0)
        if o.z_min is not None and o.z_max is not None:
            self.main.setYRange(o.z_min, o.z_max, padding=0)
        if self._resid_shown:
            self.resid.getViewBox().enableAutoRange(axis="y")
        if self._cursor_spec:
            self.set_cursors(list(self._cursor_spec))
        self.legend.setVisible(o.show_legend)

    def clear(self) -> None:
        self._last = None
        self._cursor_spec = []
        self._reset()
        self._show_resid(False)
        self.set_title("")

    def show_profile(self, profile: Profile, title: str = "") -> None:
        self._last = ("show_profile", (profile, title))
        self._reset()
        self._show_resid(False)
        s = self.look
        z = np.where(profile.valid, profile.z_nm, np.nan)
        self._scale_for(z)
        self.main.plot(
            profile.x_um * self._sx,
            z * self._sz,
            pen=pg.mkPen(s.raw, width=self._lw(1.2)),
            name="as read",
            connect="finite",
        )
        self._reader = (profile.x_um, z)
        self.set_title(title or profile.name)
        self._finish()

    def show_step(self, res: StepResult, title: str = "") -> None:
        self._last = ("show_step", (res, title))
        self._reset()
        s = self.look
        p = res.profile
        m = res.measurement
        z_nm = np.where(p.valid, p.z_nm, np.nan)
        self._scale_for(z_nm)
        sx, sz = self._sx, self._sz
        if self.show_raw and res.source.n == p.n:
            off = float(np.nanmedian(res.source.z_nm - p.z_nm))
            raw = np.where(p.valid, res.source.z_nm - off, np.nan)
            self.main.plot(
                res.source.x_um * sx,
                raw * sz,
                pen=pg.mkPen(s.background, width=self._lw(0.8)),
                name="as read (offset)",
                connect="finite",
            )
        regions = [*m.regions_um.get("plane", []), *m.regions_um.get("outer", [])]
        for i, r in enumerate(regions):
            self._band(r, rgba(s.series[0], 28))
            if i == 0:
                self.legend.addItem(
                    pg.BarGraphItem(
                        x=[0], height=[0], width=0, brush=pg.mkBrush(*rgba(s.series[0], 90))
                    ),
                    "plane region",
                )
        for r in m.regions_um.get("inner", []):
            self._band(r, rgba(s.series[2], 40))
        for r in m.regions_um.get("transition", []):
            self._band(r, rgba(s.background, 45))
        reach = getattr(res, "tip_check", None)
        for i, r in enumerate(reach.unreached_um if reach is not None else []):
            self._band(r, rgba(s.series[6 % len(s.series)], 70))
            if i == 0:
                self.legend.addItem(
                    pg.BarGraphItem(
                        x=[0],
                        height=[0],
                        width=0,
                        brush=pg.mkBrush(*rgba(s.series[6 % len(s.series)], 120)),
                    ),
                    "not touched by the stylus",
                )
        self.main.plot(
            p.x_um * sx,
            z_nm * sz,
            pen=pg.mkPen(s.raw, width=self._lw(1.2)),
            name="levelled",
            connect="finite",
        )
        self._reader = (p.x_um, z_nm)
        bad = ~p.valid
        if bad.any():
            self.main.plot(
                p.x_um[bad] * sx,
                p.z_nm[bad] * sz,
                pen=None,
                symbol="x",
                symbolSize=8,
                symbolPen=pg.mkPen(s.series[1]),
                name="spike (excluded)",
            )
        for k, plane in enumerate(m.planes):
            xs = np.linspace(plane.x_start_um, plane.x_stop_um, 20)
            self.main.plot(
                xs * sx,
                plane.line.at(xs) * sz,
                pen=pg.mkPen(s.envelope, width=self._lw(s.envelope_width + 0.4)),
                name="plane fit" if k == 0 else None,
            )
            for b in plane.burrs:
                self._band((b["x_start_um"], b["x_stop_um"]), rgba(s.series[3], 90))
        for e in res.edges.edges:
            line = pg.InfiniteLine(
                pos=e.x_um * sx,
                angle=90,
                movable=True,
                pen=pg.mkPen(s.series[0], width=1.5, style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen(s.series[0], width=3),
            )
            line.setToolTip("drag to move this edge; the analysis runs again")
            line.sigPositionChangeFinished.connect(self._edge_released)
            self.main.addItem(line)
            self._edge_lines.append(line)
        if self.opts.show_annotations:
            x_range = float(np.ptp(p.x_um)) * sx or 1.0
            finite = z_nm[np.isfinite(z_nm)]
            z_range = float(np.ptp(finite)) * sz if finite.size else 1.0
            self._draw_dimensions(step_dimensions(m, self._zu), x_range, z_range or 1.0)
        resid = plane_residuals(m, p) if m.planes else []
        self._show_resid(bool(self.opts.show_residuals and resid))
        if self._resid_shown:
            rs = 10.0 if self.opts.z_unit == "Å" else 1.0
            full = np.full(p.x_um.shape, np.nan)
            excl_x: list[np.ndarray] = []
            excl_r: list[np.ndarray] = []
            for rr in resid:
                idx = np.searchsorted(p.x_um, rr.x_um)
                full[idx] = rr.r_nm * rs
                excl_x.append(rr.x_excluded_um)
                excl_r.append(rr.r_excluded_nm * rs)
            self.resid.plot(
                p.x_um * sx, full, pen=pg.mkPen(s.raw, width=self._lw(0.8)), connect="finite"
            )
            ex = np.concatenate(excl_x) if excl_x else np.array([])
            if ex.size:
                self.resid.plot(
                    ex * sx,
                    np.concatenate(excl_r),
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolPen=None,
                    symbolBrush=pg.mkBrush(s.series[3]),
                )
            self.resid.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(s.envelope, width=1)))
            sig = float(np.median([rr.sigma_nm for rr in resid])) * rs
            for v in (-2 * sig, 2 * sig):
                self.resid.addItem(
                    pg.InfiniteLine(
                        pos=v,
                        angle=0,
                        pen=pg.mkPen(s.background, width=1, style=Qt.PenStyle.DashLine),
                    )
                )
        self.set_title(title or res.source.name)
        self._finish()

    def _band(self, r: tuple[float, float], color: tuple[int, int, int, int]) -> None:
        item = pg.LinearRegionItem(
            values=(r[0] * self._sx, r[1] * self._sx),
            movable=False,
            brush=pg.mkBrush(*color),
            pen=pg.mkPen(None),
        )
        item.setZValue(-20)
        self.main.addItem(item)

    def _draw_dimensions(self, dims: list[Dimension], x_range: float, z_range: float) -> None:
        s = self.look
        pen = pg.mkPen(s.text, width=1.2)
        thin = pg.mkPen(s.text, width=0.8, style=Qt.PenStyle.DashLine)
        font = QFont(s.font_family) if s.font_family else QFont()
        font.setPointSize(int(self.opts.font_pt))

        def arrow(x: float, z: float, angle: float) -> None:
            a = pg.ArrowItem(
                pos=(x, z), angle=angle, headLen=9, tipAngle=30, tailLen=None, pen=pen, brush=s.text
            )
            a.setZValue(15)
            self.main.addItem(a)
            self.dimension_items.append(a)

        for d in dims:
            x0, x1 = d.x0_um * self._sx, d.x1_um * self._sx
            z0, z1 = d.z0_nm * self._sz, d.z1_nm * self._sz
            if d.kind == "height":
                xa = x0 + 0.035 * x_range if d.offset else x0
                line = pg.PlotDataItem([xa, xa], [z0, z1], pen=pen)
                lo, hi = (z0, z1) if z0 < z1 else (z1, z0)
                arrow(xa, hi, 90)
                arrow(xa, lo, -90)
                tick = 0.012 * x_range
                for z in (z0, z1):
                    ext = pg.PlotDataItem([x0 - tick, xa + tick], [z, z], pen=thin)
                    self.main.addItem(ext, ignoreBounds=True)
                    self.dimension_items.append(ext)
                text = pg.TextItem(
                    d.label, color=s.text, anchor=(0, 0.5), fill=pg.mkBrush(*s.label_fill)
                )
                text.setPos(xa + 0.012 * x_range, 0.5 * (z0 + z1))
            else:
                line = pg.PlotDataItem([x0, x1], [z0, z0], pen=pen)
                arrow(x0, z0, 0)
                arrow(x1, z0, 180)
                text = pg.TextItem(
                    d.label, color=s.text, anchor=(0.5, 1.0), fill=pg.mkBrush(*s.label_fill)
                )
                text.setPos(0.5 * (x0 + x1), z0 + 0.02 * z_range)
            text.setFont(font)
            text.setZValue(16)
            self.main.addItem(line, ignoreBounds=True)
            self.main.addItem(text, ignoreBounds=True)
            self.dimension_items += [line, text]

    def show_roughness(self, res: RoughnessAnalysis, title: str = "") -> None:
        self._last = ("show_roughness", (res, title))
        self._reset()
        self._show_resid(False)
        s = self.look
        r = res.result
        self._scale_for(r.primary_nm)
        sx, sz = self._sx, self._sz
        self.main.plot(
            r.x_um * sx,
            r.primary_nm * sz,
            pen=pg.mkPen(s.background, width=self._lw(0.8)),
            name="primary",
        )
        self.main.plot(
            r.x_um * sx,
            r.waviness_nm * sz,
            pen=pg.mkPen(s.envelope, width=self._lw(1.6)),
            name=f"mean line (λc {r.lambda_c_um:g} µm)",
        )
        rr = np.where(r.evaluated, r.roughness_nm, np.nan)
        self.main.plot(
            r.x_um * sx,
            rr * sz,
            pen=pg.mkPen(s.raw, width=self._lw(1.0)),
            name="roughness",
            connect="finite",
        )
        self._reader = (r.x_um, rr)
        a, b = r.evaluation_um
        if r.n_sections >= 1:
            for edge in np.linspace(a, b, r.n_sections + 1):
                self.main.addItem(
                    pg.InfiniteLine(
                        pos=edge * sx,
                        angle=90,
                        movable=False,
                        pen=pg.mkPen(s.series[0], width=1, style=Qt.PenStyle.DotLine),
                    )
                )
        self.set_title(title or res.source.name)
        self._finish()

    def show_stress(self, res: StressAnalysis, title: str = "") -> None:
        self._last = ("show_stress", (res, title))
        self._reset()
        s = self.look
        r = res.result
        pairs = [
            ("before", res.pre, r.pre, s.series[0]),
            ("after", res.post, r.post, s.raw),
        ]
        self._scale_for(*(p.z_nm for _l, p, f, _c in pairs if p is not None and f is not None))
        sx, sz = self._sx, self._sz
        self._show_resid(self.opts.show_residuals)
        for label, prof, fit, color in pairs:
            if prof is None or fit is None:
                continue
            self.main.plot(
                prof.x_um * sx,
                prof.z_nm * sz,
                pen=pg.mkPen(color, width=self._lw(1.0)),
                name=f"{label} deposition",
            )
            a, b, c = fit.coefs
            xs = np.linspace(prof.x_um.min(), prof.x_um.max(), 200)
            zz = a * (xs - fit.x0_um) ** 2 + b * (xs - fit.x0_um) + c
            self.main.plot(
                xs * sx,
                zz * sz,
                pen=pg.mkPen(s.envelope, width=self._lw(1.6), style=Qt.PenStyle.DashLine),
                name=f"R = {fit.radius_m:.4g} m",
            )
            if self._resid_shown:
                self.resid.plot(
                    prof.x_um * sx,
                    parabola_residuals(prof.x_um, prof.z_nm, fit),
                    pen=pg.mkPen(color, width=self._lw(0.6)),
                )
            self._reader = (prof.x_um, prof.z_nm)
        if self._resid_shown:
            self.resid.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(s.envelope, width=1)))
        self.set_title(title or "stress")
        self._finish()

    def show_overlay(
        self, curves: list[tuple[str, np.ndarray, np.ndarray]], title: str = ""
    ) -> None:
        """Several profiles in one graph, each shifted by ``overlay_offset_nm`` (Origin's
        "Offset Y" stack); ``curves`` holds ``(name, x_um, z_nm)``."""
        self._last = ("show_overlay", (curves, title))
        self._reset()
        self._show_resid(False)
        s = self.look
        off = self.opts.overlay_offset_nm
        shifted = [(n, x, z + i * off) for i, (n, x, z) in enumerate(curves)]
        self._scale_for(*(z for _n, _x, z in shifted))
        for i, (name, x, z) in enumerate(shifted):
            label = name if not off else f"{name} (+{i * off:g} nm)"
            self.main.plot(
                x * self._sx,
                z * self._sz,
                pen=pg.mkPen(s.series[i % len(s.series)], width=self._lw(1.2)),
                name=label,
                connect="finite",
            )
        if shifted:
            self._reader = (shifted[0][1], shifted[0][2])
        self.set_title(title or f"overlay of {len(curves)} scans")
        self._finish()

    # ------------------------------------------------------------------ cursors
    def set_cursors(self, regions: list[tuple[float, float]]) -> None:
        """Show the two draggable reference regions R and M (positions in µm)."""
        for c in self._cursors:
            self.main.removeItem(c)
        for t in self._cursor_labels:
            self.main.removeItem(t)
        self._cursors, self._cursor_labels = [], []
        self._cursor_spec = [tuple(sorted((float(a), float(b)))) for a, b in regions]  # type: ignore[misc]
        s = self.look
        for name, r, color in zip(("R", "M"), self._cursor_spec, (s.series[4], s.series[2])):
            item = pg.LinearRegionItem(
                values=(r[0] * self._sx, r[1] * self._sx),
                movable=True,
                brush=pg.mkBrush(*rgba(color, 55)),
                pen=pg.mkPen(color, width=1.5),
            )
            item.setToolTip(f"cursor {name}: drag the band or its edges")
            item.setZValue(5)
            item.sigRegionChangeFinished.connect(self._cursor_released)
            self.main.addItem(item)
            label = pg.TextItem(name, color=color, anchor=(0.5, 1.0))
            self.main.addItem(label, ignoreBounds=True)
            self._cursors.append(item)
            self._cursor_labels.append(label)
        self._place_labels()

    def hide_cursors(self) -> None:
        self._cursor_spec = []
        for c in self._cursors:
            self.main.removeItem(c)
        for t in self._cursor_labels:
            self.main.removeItem(t)
        self._cursors, self._cursor_labels = [], []

    def cursor_regions(self) -> list[tuple[float, float]]:
        """Cursor positions in µm."""
        return [
            tuple(sorted(float(v) / self._sx for v in c.getRegion()))  # type: ignore[misc]
            for c in self._cursors
        ]

    def edge_positions(self) -> list[float]:
        """Edge line positions in µm."""
        return sorted(float(line.value()) / self._sx for line in self._edge_lines)

    def _place_labels(self) -> None:
        yr = self.main.getViewBox().viewRange()[1]
        for c, t in zip(self._cursors, self._cursor_labels):
            a, b = c.getRegion()
            t.setPos(0.5 * (a + b), yr[1])

    def _cursor_released(self, *_: Any) -> None:
        self._cursor_spec = self.cursor_regions()
        self._place_labels()
        self.regionsEdited.emit(self.cursor_regions())

    def _edge_released(self, *_: Any) -> None:
        self.edgesEdited.emit(self.edge_positions())

    # ------------------------------------------------------------------ data reader
    def reader_point(self, x_um: float) -> tuple[float, float] | None:
        """The data point nearest to ``x_um`` on the main curve (µm, nm)."""
        if self._reader is None:
            return None
        x, z = self._reader
        if x.size == 0:
            return None
        i = int(np.clip(np.searchsorted(x, x_um), 1, x.size - 1))
        j = i - 1 if abs(x[i - 1] - x_um) <= abs(x[i] - x_um) else i
        if not np.isfinite(z[j]):
            ok = np.flatnonzero(np.isfinite(z))
            if ok.size == 0:
                return None
            j = int(ok[np.argmin(np.abs(x[ok] - x_um))])
        return float(x[j]), float(z[j])

    def _mouse_moved(self, event: tuple) -> None:
        pos = event[0]
        if not self.main.sceneBoundingRect().contains(pos):
            self._cross.hide()
            self._marker.setData([], [])
            return
        pt = self.main.getViewBox().mapSceneToView(pos)
        x_um = float(pt.x()) / self._sx
        snap = self.reader_point(x_um)
        if snap is None:
            self._snapped = (x_um, float(pt.y()) / self._sz)
            self._cross.setPos(float(pt.x()))
        else:
            self._snapped = snap
            self._cross.setPos(snap[0] * self._sx)
            self._marker.setData([snap[0] * self._sx], [snap[1] * self._sz])
        self._cross.show()
        self.cursorMoved.emit(*self._snapped)

    def _clicked(self, ev: Any) -> None:
        if not self.main.sceneBoundingRect().contains(ev.scenePos()):
            return
        if ev.double():
            self.detailsRequested.emit()
            return
        if ev.button() == Qt.MouseButton.LeftButton and (
            ev.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            pt = self.main.getViewBox().mapSceneToView(ev.scenePos())
            self.add_measure_point(float(pt.x()) / self._sx)
            ev.accept()

    def add_measure_point(self, x_um: float) -> dict[str, Any] | None:
        """Shift-click: set point A, then B (a third click starts again)."""
        snap = self.reader_point(x_um)
        if snap is None:
            return None
        if len(self.measure_points) >= 2:
            self.clear_measure()
        self.measure_points.append(snap)
        self._draw_measure()
        if len(self.measure_points) == 2:
            (xa, za), (xb, zb) = self.measure_points
            info = {"a": (xa, za), "b": (xb, zb), "dx_um": xb - xa, "dz_nm": zb - za}
            self.pointsMeasured.emit(info)
            return info
        return None

    def clear_measure(self) -> None:
        for it in self._measure_items:
            self.main.removeItem(it)
        self._measure_items = []
        self.measure_points = []

    def _draw_measure(self) -> None:
        for it in self._measure_items:
            self.main.removeItem(it)
        self._measure_items = []
        s = self.look
        color = s.series[1]
        pts = [(x * self._sx, z * self._sz) for x, z in self.measure_points]
        sc = pg.ScatterPlotItem(
            [p[0] for p in pts],
            [p[1] for p in pts],
            size=10,
            symbol="+",
            pen=pg.mkPen(color, width=2),
        )
        sc.setZValue(21)
        self.main.addItem(sc, ignoreBounds=True)
        self._measure_items.append(sc)
        for name, (x, z) in zip("AB", pts):
            t = pg.TextItem(name, color=color, anchor=(-0.3, 1.2))
            t.setPos(x, z)
            self.main.addItem(t, ignoreBounds=True)
            self._measure_items.append(t)
        if len(pts) == 2:
            line = pg.PlotDataItem(
                [pts[0][0], pts[1][0]],
                [pts[0][1], pts[1][1]],
                pen=pg.mkPen(color, width=1.2, style=Qt.PenStyle.DashLine),
            )
            self.main.addItem(line, ignoreBounds=True)
            self._measure_items.append(line)

    def shutdown(self) -> None:
        """Disconnect the mouse proxy so no scene signal outlives the widget."""
        with contextlib.suppress(RuntimeError, TypeError):
            self._proxy.disconnect()
        with contextlib.suppress(RuntimeError, TypeError):
            self.main.scene().sigMouseClicked.disconnect(self._clicked)
        self._reset()


class BatchChart(QWidget):
    """Column chart of the sample means with error bars and the individual scans."""

    def __init__(self, parent: QWidget | None = None, style: PlotStyle | None = None) -> None:
        super().__init__(parent)
        self.look = style or plot_style()
        self.view = pg.PlotWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.plot = self.view.getPlotItem()
        self.error = "sd"
        self._batch: Any = None
        self.bars: pg.BarGraphItem | None = None
        self._apply_style()

    def set_style(self, style: PlotStyle) -> None:
        self.look = style
        self._apply_style()
        if self._batch is not None:
            self.show_batch(self._batch, self.error)

    def _apply_style(self) -> None:
        s = self.look
        self.view.setBackground(s.canvas)
        for name in _AXES:
            ax = self.plot.getAxis(name)
            ax.setPen(pg.mkPen(s.frame, width=1))
            ax.setTextPen(pg.mkPen(s.text))
            ax.setStyle(tickLength=-6 if s.ticks_inward else 5)
            ax.enableAutoSIPrefix(False)
        if s.box:
            self.plot.showAxis("top")
            self.plot.showAxis("right")
            self.plot.getAxis("top").setStyle(showValues=False)
            self.plot.getAxis("right").setStyle(showValues=False)
        self.plot.setLabel("left", "thickness (nm)", color=s.text)

    def show_batch(self, batch: Any, error: str = "sd") -> None:
        self._batch, self.error = batch, error
        s = self.look
        self.plot.clear()
        samples = batch.samples
        idx = np.arange(len(samples), dtype=float)
        means = np.array([v.mean_nm if v.mean_nm is not None else 0.0 for v in samples])
        errs = np.array(
            [
                float({"sd": v.std_nm, "se": v.sem_nm, "ci95": v.ci95_nm}[error] or 0.0)
                for v in samples
            ]
        )
        brushes = [pg.mkBrush(*rgba(s.series[i % len(s.series)], 90)) for i in range(len(samples))]
        self.bars = pg.BarGraphItem(
            x=idx, height=means, width=0.6, brushes=brushes, pen=pg.mkPen(s.frame, width=1)
        )
        self.plot.addItem(self.bars)
        self.plot.addItem(
            pg.ErrorBarItem(
                x=idx, y=means, top=errs, bottom=errs, beam=0.2, pen=pg.mkPen(s.text, width=1.2)
            )
        )
        top = float(np.max(means + errs)) if len(samples) else 1.0
        tops: list[float] = []
        for i, v in enumerate(samples):
            vals = [
                float(it.result.thickness_nm)
                for it in batch.items
                if it.sample == v.sample and it.result is not None and it.result.thickness_nm
            ]
            if vals:
                jit = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.zeros(1)
                self.plot.addItem(
                    pg.ScatterPlotItem(
                        i + jit,
                        vals,
                        size=7,
                        symbol="o",
                        pen=pg.mkPen(s.text, width=1),
                        brush=pg.mkBrush(s.canvas),
                    )
                )
                top = max(top, max(vals))
            tops.append(max([means[i] + errs[i], *vals]) if vals else means[i] + errs[i])
        for i, v in enumerate(samples):
            label = pg.TextItem(f"n = {v.n}", color=s.text, anchor=(0.5, 1.0))
            label.setPos(i, tops[i] + 0.04 * top)
            self.plot.addItem(label)
        self.plot.getAxis("bottom").setTicks([[(i, v.sample) for i, v in enumerate(samples)]])
        self.plot.setXRange(-0.6, len(samples) - 0.4, padding=0)
        self.plot.setYRange(0, top * 1.35 if top > 0 else 1, padding=0)
        name = {"sd": "SD", "se": "SE", "ci95": "95% CI"}[error]
        self.plot.setTitle(f"mean ± {name}", color=s.text)


__all__ = ["BatchChart", "ProfilePlot"]

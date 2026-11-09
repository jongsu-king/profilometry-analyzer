"""Live profile view.

Shows the profile as read (optional, grey), the processed profile, plane regions and
plane fits, burrs and spikes.  Two draggable reference regions (``R`` and ``M``, like the
reference and measurement cursors of the instrument software) and one draggable line per
detected edge let the user correct the analysis: releasing a cursor emits
``regionsEdited`` or ``edgesEdited`` and the main window re-runs the analysis.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..analysis import RoughnessAnalysis, StepResult, StressAnalysis
from ..profile import Profile
from .theme import PlotStyle, plot_style, rgba

_AXES = ("left", "bottom", "top", "right")


class ProfilePlot(QWidget):
    regionsEdited = Signal(list)
    """``[(a, b), (c, d)]`` after a reference cursor (R, M) was moved."""
    edgesEdited = Signal(list)
    """Edge positions (µm) after an edge line was moved."""
    cursorMoved = Signal(float, float)

    def __init__(self, parent: QWidget | None = None, style: PlotStyle | None = None) -> None:
        super().__init__(parent)
        self.look: PlotStyle = style or plot_style()
        self.view = pg.GraphicsLayoutWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.main = self.view.addPlot(row=0, col=0)
        self.legend = self.main.addLegend(offset=(-10, 10))
        self.show_raw = True
        self._edge_lines: list[pg.InfiniteLine] = []
        self._cursors: list[pg.LinearRegionItem] = []
        self._cursor_labels: list[pg.TextItem] = []
        self._cursor_spec: list[tuple[float, float]] = []
        """Cursor positions; kept when the plot is redrawn."""
        self._last: tuple[str, tuple[Any, ...]] | None = None
        self._title = ""
        self._cross = pg.InfiniteLine(angle=90, movable=False)
        self._cross.setZValue(-10)
        self._cross.hide()
        self._proxy = pg.SignalProxy(
            self.main.scene().sigMouseMoved, rateLimit=40, slot=self._mouse_moved
        )
        self._apply_style()

    # ------------------------------------------------------------------ style
    def set_style(self, style: PlotStyle) -> None:
        self.look = style
        self._apply_style()
        self._replay()

    def _replay(self) -> None:
        if self._last is not None:
            method, args = self._last
            getattr(self, method)(*args)

    def _apply_style(self) -> None:
        s = self.look
        self.view.setBackground(s.canvas)
        font: QFont | None = None
        if s.font_family or s.tick_font_pt:
            font = QFont(s.font_family) if s.font_family else QFont()
            if s.tick_font_pt:
                font.setPointSize(s.tick_font_pt)
        for name in _AXES:
            ax = self.main.getAxis(name)
            ax.setPen(pg.mkPen(s.frame, width=1))
            ax.setTextPen(pg.mkPen(s.text))
            ax.setTickFont(font)
            ax.setStyle(tickLength=-6 if s.ticks_inward else 5, tickTextOffset=6)
        if s.box:
            self.main.showAxis("top")
            self.main.showAxis("right")
            self.main.getAxis("top").setStyle(showValues=False)
            self.main.getAxis("right").setStyle(showValues=False)
        else:
            self.main.hideAxis("top")
            self.main.hideAxis("right")
        self.main.showGrid(x=s.grid, y=s.grid, alpha=0.15)
        extra: dict[str, str] = {"color": s.text}
        if s.font_family:
            extra["font-family"] = s.font_family
        if s.label_font_pt:
            extra["font-size"] = f"{s.label_font_pt}pt"
        self.main.setLabel("bottom", "x (µm)", **extra)
        self.main.setLabel("left", "height (nm)", **extra)
        self.legend.setLabelTextColor(s.text)
        self.legend.setPen(pg.mkPen(s.legend_pen) if s.legend_pen else pg.mkPen(None))
        self.legend.setBrush(pg.mkBrush(*s.legend_fill))
        self._cross.setPen(pg.mkPen(s.crosshair, width=1))
        self.set_title(self._title)

    def set_title(self, title: str) -> None:
        self._title = title
        s = self.look
        kw: dict[str, Any] = {"color": s.text}
        if s.label_font_pt:
            kw["size"] = f"{s.label_font_pt}pt"
        self.main.setTitle(title or None, **kw)

    # ------------------------------------------------------------------ drawing
    def _reset(self) -> None:
        for line in self._edge_lines:
            with contextlib.suppress(RuntimeError, TypeError):
                line.sigPositionChangeFinished.disconnect()
        self.main.clear()
        self.legend.clear()
        self._edge_lines = []
        self._cursors = []
        self._cursor_labels = []
        self.main.addItem(self._cross, ignoreBounds=True)

    def _finish(self) -> None:
        """Autorange, then put the cursors back where the user left them."""
        self.main.enableAutoRange()
        if self._cursor_spec:
            self.set_cursors(list(self._cursor_spec))

    def clear(self) -> None:
        self._last = None
        self._cursor_spec = []
        self._reset()
        self.set_title("")

    def show_profile(self, profile: Profile, title: str = "") -> None:
        self._last = ("show_profile", (profile, title))
        self._reset()
        s = self.look
        z = np.where(profile.valid, profile.z_nm, np.nan)
        self.main.plot(
            profile.x_um, z, pen=pg.mkPen(s.raw, width=1.2), name="as read", connect="finite"
        )
        self.set_title(title or profile.name)
        self._finish()

    def show_step(self, res: StepResult, title: str = "") -> None:
        self._last = ("show_step", (res, title))
        self._reset()
        s = self.look
        p = res.profile
        m = res.measurement
        if self.show_raw and res.source.n == p.n:
            off = float(np.nanmedian(res.source.z_nm - p.z_nm))
            raw = np.where(p.valid, res.source.z_nm - off, np.nan)
            self.main.plot(
                res.source.x_um,
                raw,
                pen=pg.mkPen(s.background, width=0.8),
                name="as read (offset)",
                connect="finite",
            )
        regions = [*m.regions_um.get("plane", []), *m.regions_um.get("outer", [])]
        for i, r in enumerate(regions):
            item = pg.LinearRegionItem(
                values=r,
                movable=False,
                brush=pg.mkBrush(*rgba(s.series[0], 28)),
                pen=pg.mkPen(None),
            )
            item.setZValue(-20)
            self.main.addItem(item)
            if i == 0:
                self.legend.addItem(
                    pg.BarGraphItem(
                        x=[0], height=[0], width=0, brush=pg.mkBrush(*rgba(s.series[0], 90))
                    ),
                    "plane region",
                )
        for r in m.regions_um.get("inner", []):
            item = pg.LinearRegionItem(
                values=r,
                movable=False,
                brush=pg.mkBrush(*rgba(s.series[2], 40)),
                pen=pg.mkPen(None),
            )
            item.setZValue(-20)
            self.main.addItem(item)
        z = np.where(p.valid, p.z_nm, np.nan)
        self.main.plot(p.x_um, z, pen=pg.mkPen(s.raw, width=1.2), name="levelled", connect="finite")
        bad = ~p.valid
        if bad.any():
            self.main.plot(
                p.x_um[bad],
                p.z_nm[bad],
                pen=None,
                symbol="x",
                symbolSize=8,
                symbolPen=pg.mkPen(s.series[1]),
                name="spike (excluded)",
            )
        for k, plane in enumerate(m.planes):
            xs = np.linspace(plane.x_start_um, plane.x_stop_um, 20)
            self.main.plot(
                xs,
                plane.line.at(xs),
                pen=pg.mkPen(s.envelope, width=s.envelope_width + 0.4),
                name="plane fit" if k == 0 else None,
            )
            for b in plane.burrs:
                item = pg.LinearRegionItem(
                    values=(b["x_start_um"], b["x_stop_um"]),
                    movable=False,
                    brush=pg.mkBrush(*rgba(s.series[3], 90)),
                    pen=pg.mkPen(None),
                )
                self.main.addItem(item)
        for e in res.edges.edges:
            line = pg.InfiniteLine(
                pos=e.x_um,
                angle=90,
                movable=True,
                pen=pg.mkPen(s.series[0], width=1.5, style=pg.QtCore.Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen(s.series[0], width=3),
            )
            line.setToolTip("drag to move this edge; the analysis runs again")
            line.sigPositionChangeFinished.connect(self._edge_released)
            self.main.addItem(line)
            self._edge_lines.append(line)
        self.set_title(title or res.source.name)
        self._finish()

    def show_roughness(self, res: RoughnessAnalysis, title: str = "") -> None:
        self._last = ("show_roughness", (res, title))
        self._reset()
        s = self.look
        r = res.result
        self.main.plot(r.x_um, r.primary_nm, pen=pg.mkPen(s.background, width=0.8), name="primary")
        self.main.plot(
            r.x_um,
            r.waviness_nm,
            pen=pg.mkPen(s.envelope, width=1.6),
            name=f"mean line (λc {r.lambda_c_um:g} µm)",
        )
        rr = np.where(r.evaluated, r.roughness_nm, np.nan)
        self.main.plot(
            r.x_um, rr, pen=pg.mkPen(s.raw, width=1.0), name="roughness", connect="finite"
        )
        a, b = r.evaluation_um
        if r.n_sections >= 1:
            for edge in np.linspace(a, b, r.n_sections + 1):
                self.main.addItem(
                    pg.InfiniteLine(
                        pos=edge,
                        angle=90,
                        movable=False,
                        pen=pg.mkPen(s.series[0], width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
                    )
                )
        self.set_title(title or res.source.name)
        self._finish()

    def show_stress(self, res: StressAnalysis, title: str = "") -> None:
        self._last = ("show_stress", (res, title))
        self._reset()
        s = self.look
        r = res.result
        for label, prof, fit, color in (
            ("before", res.pre, r.pre, s.series[0]),
            ("after", res.post, r.post, s.raw),
        ):
            if prof is None or fit is None:
                continue
            self.main.plot(
                prof.x_um, prof.z_nm, pen=pg.mkPen(color, width=1.0), name=f"{label} deposition"
            )
            a, b, c = fit.coefs
            xs = np.linspace(prof.x_um.min(), prof.x_um.max(), 200)
            zz = a * (xs - fit.x0_um) ** 2 + b * (xs - fit.x0_um) + c
            self.main.plot(
                xs,
                zz,
                pen=pg.mkPen(s.envelope, width=1.6, style=pg.QtCore.Qt.PenStyle.DashLine),
                name=f"R = {fit.radius_m:.4g} m",
            )
        self.set_title(title or "stress")
        self._finish()

    # ------------------------------------------------------------------ cursors
    def set_cursors(self, regions: list[tuple[float, float]]) -> None:
        """Show the two draggable reference regions R and M."""
        for c in self._cursors:
            self.main.removeItem(c)
        for t in self._cursor_labels:
            self.main.removeItem(t)
        self._cursors, self._cursor_labels = [], []
        self._cursor_spec = [tuple(sorted((float(a), float(b)))) for a, b in regions]  # type: ignore[misc]
        s = self.look
        for name, r, color in zip(("R", "M"), regions, (s.series[4], s.series[2])):
            item = pg.LinearRegionItem(
                values=r,
                movable=True,
                brush=pg.mkBrush(*rgba(color, 55)),
                pen=pg.mkPen(color, width=1.5),
            )
            item.setToolTip(f"cursor {name}: drag the band or its edges")
            item.setZValue(5)
            item.sigRegionChangeFinished.connect(self._cursor_released)
            self.main.addItem(item)
            label = pg.TextItem(name, color=color, anchor=(0.5, 1.0))
            label.setPos(0.5 * (r[0] + r[1]), 0)
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
        return [tuple(sorted(float(v) for v in c.getRegion())) for c in self._cursors]  # type: ignore[misc]

    def edge_positions(self) -> list[float]:
        return sorted(float(line.value()) for line in self._edge_lines)

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

    def _mouse_moved(self, event: tuple) -> None:
        pos = event[0]
        if not self.main.sceneBoundingRect().contains(pos):
            self._cross.hide()
            return
        pt = self.main.getViewBox().mapSceneToView(pos)
        self._cross.setPos(float(pt.x()))
        self._cross.show()
        self.cursorMoved.emit(float(pt.x()), float(pt.y()))

    def shutdown(self) -> None:
        """Disconnect the mouse proxy so no scene signal outlives the widget."""
        with contextlib.suppress(RuntimeError, TypeError):
            self._proxy.disconnect()
        self._reset()

"""Main window: scan list, live profile with cursors, settings, results, notices, batch.

Layout follows the instrument-software style of xps-autofit-studio: a toolbar of
command buttons, the scans on the left, the profile in the centre, settings on the
right and results / edges / notices / batch tables at the bottom.  Every analysis runs
on a :class:`~.workers.JobRunner` thread; results come back to the GUI thread.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QColor, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from .. import notices as N
from ..analysis import (
    RoughnessAnalysis,
    StepResult,
    StepSettings,
    StressAnalysis,
    analyze_roughness,
    analyze_step,
    analyze_stress,
)
from ..batch import BatchResult, run_batch, sample_name
from ..core.roughness import SETTING_CLASSES, RoughnessSettings
from ..core.stress import SUBSTRATES, StressSettings
from ..export import origin as O
from ..export.plots import FigureOptions
from ..instrument import DEFAULT_INSTRUMENT
from ..notices import Notice
from ..profile import Profile
from ..readers import SUPPORTED_SUFFIXES, ReadError, read_profile
from . import theme as T
from .dialogs import ExportFigureDialog, PlotDetailsDialog
from .graph_options import GraphOptions
from .plot_widget import BatchChart, ProfilePlot
from .workers import JobRunner
from .worksheet import WorksheetView

ORG, APP = "profilometry-analyzer", "desktop"
SEVERITY_COLOR = {"info": "#5a8fd8", "warning": "#e0a000", "error": "#d03b3b"}


@dataclass
class Scan:
    path: Path
    profile: Profile
    sample: str
    step: StepResult | None = None
    roughness: RoughnessAnalysis | None = None
    edges_um: list[float] | None = None
    """Edge positions edited by the user (None: detected)."""
    status: str = "idle"
    extra: dict[str, Any] = field(default_factory=dict)


def _spin(
    lo: float, hi: float, val: float, step: float = 1.0, dec: int = 2, special: str | None = None
) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if special:
        s.setSpecialValueText(special)
    return s


class MainWindow(QMainWindow):
    def __init__(self, synchronous_jobs: bool = False, check_updates: bool | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"Profilometry Analyzer {__version__}")
        self.resize(1440, 900)
        self.settings = QSettings(ORG, APP)
        self.jobs = JobRunner(self, synchronous=synchronous_jobs)
        self.jobs.started.connect(lambda label: self._set_status("running", label))
        self.scans: dict[str, Scan] = {}
        self.current: str | None = None
        self.batch: BatchResult | None = None
        self.stress_result: StressAnalysis | None = None
        self._last_view = "step"

        self.plot = ProfilePlot(
            self,
            T.plot_style(str(self.settings.value("ui/plot", "origin"))),
            GraphOptions.from_json(self.settings.value("ui/graph", "")),
        )
        self.plot.regionsEdited.connect(self._cursors_moved)
        self.plot.edgesEdited.connect(self._edges_moved)
        self.plot.cursorMoved.connect(self._cursor_readout)
        self.plot.pointsMeasured.connect(self._points_measured)
        self.plot.detailsRequested.connect(self.plot_details)
        self.fig_options, self.fig_formats, self.fig_style, self.fig_preset = self._load_figure()
        self.setCentralWidget(self.plot)
        self._build_toolbar()
        self._build_menu()
        self._build_scans_dock()
        self._build_settings_dock()
        self._build_results_dock()
        self.cursor_label = QLabel("")
        self.cursor_label.setObjectName("cursor")
        self.measure_label = QLabel("")
        self.measure_label.setObjectName("cursor")
        self.status_label = QLabel("")
        self.update_label = QLabel("")
        self.update_label.setObjectName("update")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.update_label)
        self.statusBar().addPermanentWidget(self.measure_label)
        self.statusBar().addPermanentWidget(self.cursor_label)
        self._set_status("idle", "open a scan (CSV, TXT or xlsx export)")
        want_updates = check_updates
        if want_updates is None:
            from ..update_check import disabled_by_environment

            want_updates = (
                not synchronous_jobs
                and not disabled_by_environment()
                and bool(self.settings.value("updates/check", True, type=bool))
            )
        if want_updates:
            QTimer.singleShot(1500, self._check_updates)

    # ------------------------------------------------------------------ building
    def _build_toolbar(self) -> None:
        tb = QToolBar("commands")
        tb.setMovable(False)
        self.addToolBar(tb)

        def add(text: str, slot: Any, tip: str, primary: bool = False) -> QAction:
            act = QAction(text, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            tb.addAction(act)
            if primary:
                w = tb.widgetForAction(act)
                if w is not None:
                    w.setObjectName("primary")
            return act

        add("OPEN", self.open_files, "open scan files")
        add("OPEN FOLDER", self.open_folder, "open every scan of a folder")
        tb.addSeparator()
        self.act_step = add("STEP", self.analyze_current, "step height of the selected scan", True)
        add("STEP ALL", self.analyze_all, "step height of every scan and the sample table")
        add("ROUGHNESS", self.roughness_current, "ISO 21920-2 roughness of the selected scan")
        add("STRESS", self.stress_run, "Stoney stress from the before/after scans")
        add("OVERLAY", self.overlay, "overlay the selected scans (or the scans of the sample)")
        tb.addSeparator()
        add("EXPORT", self.export, "tables, figures and project file of the shown result")
        add("FIGURE", self.export_figure, "publication figure: journal size, dpi, TIFF/SVG/PDF")

    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction("Open scans…", self.open_files)
        m.addAction("Open folder…", self.open_folder)
        m.addAction("Export results…", self.export)
        m.addAction("Export figure…", self.export_figure)
        m.addAction("Export Origin worksheet…", self.export_worksheet)
        m.addSeparator()
        m.addAction("Quit", self.close)
        e = mb.addMenu("&Edit")
        act = e.addAction("Copy graph", self.copy_graph)
        act.setShortcut(QKeySequence("Ctrl+Shift+C"))
        e.addAction("Copy worksheet (Long Name / Units / Comments + data)", self.copy_worksheet)
        e.addAction("Clear A/B markers", self.plot.clear_measure)
        v = mb.addMenu("&View")
        for name in T.THEMES:
            v.addAction(f"{name.title()} theme", partial(self.set_theme, name))
        v.addSeparator()
        for name in T.PLOT_STYLE_NAMES:
            v.addAction(f"{name.title()} plot", partial(self.set_plot_style, name))
        self.act_raw = QAction("Show profile as read", self, checkable=True, checked=True)
        self.act_raw.toggled.connect(self._toggle_raw)
        v.addAction(self.act_raw)
        o = self.plot.opts
        self.act_resid = QAction("Residual layer", self, checkable=True, checked=o.show_residuals)
        self.act_resid.toggled.connect(lambda on: self._toggle_option("show_residuals", on))
        self.act_dims = QAction("Dimension lines", self, checkable=True, checked=o.show_annotations)
        self.act_dims.toggled.connect(lambda on: self._toggle_option("show_annotations", on))
        v.addAction(self.act_resid)
        v.addAction(self.act_dims)
        v.addSeparator()
        act = v.addAction("Plot details…", self.plot_details)
        act.setShortcut(QKeySequence("Ctrl+D"))
        h = mb.addMenu("&Help")
        self.act_updates = QAction("Check for updates at startup", self, checkable=True)
        self.act_updates.setChecked(bool(self.settings.value("updates/check", True, type=bool)))
        self.act_updates.toggled.connect(lambda on: self.settings.setValue("updates/check", on))
        h.addAction(self.act_updates)
        h.addAction("Check for updates now", self._check_updates)
        h.addAction("About", self._about)

    def _build_scans_dock(self) -> None:
        dock = QDockWidget("SCANS", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["scan", "thickness (nm)", "status"])
        self.tree.setColumnWidth(0, 150)
        self.tree.setColumnWidth(1, 140)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        dock.setWidget(self.tree)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_settings_dock(self) -> None:
        dock = QDockWidget("SETTINGS", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        tabs = QTabWidget()
        tabs.addTab(self._step_panel(), "STEP")
        tabs.addTab(self._roughness_panel(), "ROUGHNESS")
        tabs.addTab(self._stress_panel(), "STRESS")
        dock.setWidget(tabs)
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _step_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("LEVELLING")
        f = QFormLayout(g)
        self.cb_level = QComboBox()
        self.cb_level.addItems(["auto", "regions", "none"])
        self.cb_level.setToolTip("auto: robust fit with the steps masked; regions: cursors R and M")
        self.cb_level.currentTextChanged.connect(self._mode_changed)
        self.cb_method = QComboBox()
        self.cb_method.addItems(["huber", "ransac", "lsq"])
        self.chk_separate = QCheckBox("R and M on different levels")
        self.sp_bow = QSpinBox()
        self.sp_bow.setRange(1, 6)
        self.sp_bow.setToolTip("1 = tilt only; 2-6 = also remove a polynomial bow")
        f.addRow("method", self.cb_level)
        f.addRow("robust fit", self.cb_method)
        f.addRow("", self.chk_separate)
        f.addRow("bow order", self.sp_bow)
        lay.addWidget(g)
        g = QGroupBox("STEP")
        f = QFormLayout(g)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["auto", "edge", "groove", "ridge", "periodic", "staircase", "multi"])
        self.chk_cursor_step = QCheckBox("measure M above R (cursors)")
        self.chk_cursor_step.toggled.connect(self._mode_changed)
        self.sp_smooth = _spin(0, 1000, 0, 0.5, 2, "auto")
        self.sp_edge_k = _spin(2, 50, 6, 0.5, 1)
        self.sp_burr_k = _spin(2, 20, 4, 0.5, 1)
        self.sp_margin = _spin(0, 1000, 0, 0.5, 2, "auto")
        self.chk_multiscale = QCheckBox("find wide, gentle edges")
        self.chk_multiscale.setChecked(True)
        self.chk_multiscale.setToolTip(
            "also search with 3, 9, 27 ... times wider derivative smoothing"
        )
        self.chk_widen = QCheckBox("widen transitions until the plateaus are flat")
        self.chk_widen.setChecked(True)
        self.chk_widen.setToolTip(
            "grow a transition while the plateau next to it still runs towards the other level"
        )
        self.btn_reset_edges = QPushButton("Reset edited edges")
        self.btn_reset_edges.clicked.connect(self._reset_edges)
        f.addRow("profile kind", self.cb_mode)
        f.addRow("", self.chk_cursor_step)
        f.addRow("smoothing σ (µm)", self.sp_smooth)
        f.addRow("edge threshold (σ)", self.sp_edge_k)
        f.addRow("burr exclusion (σ)", self.sp_burr_k)
        f.addRow("edge margin (µm)", self.sp_margin)
        f.addRow("", self.chk_multiscale)
        f.addRow("", self.chk_widen)
        f.addRow("", self.btn_reset_edges)
        lay.addWidget(g)
        g = QGroupBox("STYLUS AND FILM")
        f = QFormLayout(g)
        self.sp_tip = _spin(0, 50, 0, 0.25, 2, "from file")
        self.sp_cone = _spin(0, 170, 0, 5, 0, "from file")
        self.sp_cone.setToolTip(
            "full angle of the stylus cone; 'from file' uses the file, else 60°. It matters"
            " only for features deeper than half the stylus radius"
        )
        self.sp_force = _spin(0, 50, 0, 0.5, 1, "from file")
        self.chk_soft = QCheckBox("soft film (polymer)")
        self.chk_tip_check = QCheckBox("mark points the stylus could not touch")
        self.chk_tip_check.setChecked(True)
        self.chk_tip_check.setToolTip(
            "the foot of steep walls and gaps narrower than the stylus: there the surface"
            " lies somewhere below the record"
        )
        f.addRow("stylus radius (µm)", self.sp_tip)
        f.addRow("stylus cone angle (°)", self.sp_cone)
        f.addRow("stylus force (mg)", self.sp_force)
        f.addRow("", self.chk_soft)
        f.addRow("", self.chk_tip_check)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    def _roughness_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("FILTERS (ISO 16610-21)")
        f = QFormLayout(g)
        self.cb_lc = QComboBox()
        self.cb_lc.addItem("auto (class from Ra)", None)
        for sc in SETTING_CLASSES:
            self.cb_lc.addItem(f"{sc.lambda_c_um:g} µm ({sc.name})", sc.lambda_c_um)
        self.cb_ls = QComboBox()
        self.cb_ls.addItem("class value", None)
        self.cb_ls.addItem("none", 0.0)
        for v in (2.5, 8.0, 25.0):
            self.cb_ls.addItem(f"{v:g} µm", v)
        self.sp_form = QSpinBox()
        self.sp_form.setRange(0, 6)
        self.sp_form.setValue(1)
        self.sp_sections = QSpinBox()
        self.sp_sections.setRange(1, 20)
        self.sp_sections.setValue(5)
        self.chk_rough_cursor = QCheckBox("evaluate inside cursor R only")
        self.chk_rough_cursor.toggled.connect(self._mode_changed)
        f.addRow("λc", self.cb_lc)
        f.addRow("λs", self.cb_ls)
        f.addRow("form order", self.sp_form)
        f.addRow("sections", self.sp_sections)
        f.addRow("", self.chk_rough_cursor)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    def _stress_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("SCANS")
        f = QFormLayout(g)
        self.cb_pre = QComboBox()
        self.cb_post = QComboBox()
        f.addRow("before", self.cb_pre)
        f.addRow("after", self.cb_post)
        lay.addWidget(g)
        g = QGroupBox("SUBSTRATE AND FILM")
        f = QFormLayout(g)
        self.cb_sub = QComboBox()
        for s in SUBSTRATES.values():
            self.cb_sub.addItem(s.name + ("" if s.verified else " (unverified)"), s.key)
        self.cb_sub.currentIndexChanged.connect(self._substrate_changed)
        self.sp_mod = _spin(0, 2000, 0, 1, 2, "table")
        self.sp_hs = _spin(1, 20000, 525, 5, 1)
        self.sp_hf = _spin(0, 1e6, 0, 10, 1, "required")
        self.chk_back = QCheckBox("scans on the back side")
        self.lbl_source = QLabel("")
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setObjectName("caption")
        f.addRow("substrate", self.cb_sub)
        f.addRow("E/(1-ν) (GPa)", self.sp_mod)
        f.addRow("h_s (µm)", self.sp_hs)
        f.addRow("h_f (nm)", self.sp_hf)
        f.addRow("", self.chk_back)
        f.addRow(self.lbl_source)
        lay.addWidget(g)
        lay.addStretch(1)
        self._substrate_changed()
        return w

    def _build_results_dock(self) -> None:
        dock = QDockWidget("RESULTS", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.result_tabs = QTabWidget()
        self.tbl_result = QTableWidget(0, 2)
        self.tbl_result.setHorizontalHeaderLabels(["quantity", "value"])
        self.tbl_result.horizontalHeader().setStretchLastSection(True)
        self.tbl_edges = QTableWidget(0, 0)
        self.lst_notices = QListWidget()
        self.lst_notices.setWordWrap(True)
        self.tbl_batch = QTableWidget(0, 0)
        self.batch_chart = BatchChart(self, self.plot.look)
        self.batch_split = QSplitter(Qt.Orientation.Horizontal)
        self.batch_split.addWidget(self.tbl_batch)
        self.batch_split.addWidget(self.batch_chart)
        self.batch_split.setSizes([700, 420])
        self.worksheet = WorksheetView(self)
        self.worksheet.set_header_color(T.current().panel_raised)
        self.result_tabs.addTab(self.tbl_result, "RESULT")
        self.result_tabs.addTab(self.tbl_edges, "EDGES")
        self.result_tabs.addTab(self.lst_notices, "NOTICES")
        self.result_tabs.addTab(self.batch_split, "BATCH")
        self.result_tabs.addTab(self.worksheet, "WORKSHEET")
        dock.setWidget(self.result_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    # ------------------------------------------------------------------ status
    def _set_status(self, state: str, text: str = "") -> None:
        color = T.STATUS_COLORS.get(state, T.STATUS_COLORS["idle"])
        word = T.STATUS_TEXT.get(state, state.upper())
        self.status_label.setText(f"<span style='color:{color}'>●</span> <b>{word}</b>  {text}")

    def _cursor_readout(self, x: float, z: float) -> None:
        self.cursor_label.setText(self.plot.format_point(x, z))

    def _points_measured(self, info: dict[str, Any]) -> None:
        p = self.plot
        dx = info["dx_um"] * p.opts.sx
        dz_scale = p.opts.z_factor(abs(info["dz_nm"]) if p.opts.z_unit == "auto" else 0.0)
        self.measure_label.setText(
            f"A→B  Δx = {dx:.5g} {p.opts.x_unit}   Δz = {info['dz_nm'] * dz_scale[0]:.5g}"
            f" {dz_scale[1]}"
        )

    # ------------------------------------------------------------------ files
    def open_files(self) -> None:
        exts = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
        files, _ = QFileDialog.getOpenFileNames(self, "Open scans", "", f"Scans ({exts});;All (*)")
        self.add_files([Path(f) for f in files])

    def open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open folder of scans")
        if d:
            from ..batch import find_files

            self.add_files(find_files(d))

    def add_files(self, paths: list[Path]) -> list[str]:
        added, errors = [], []
        for p in paths:
            key = str(Path(p).resolve())
            if key in self.scans:
                continue
            try:
                prof = read_profile(p)
            except (ReadError, ValueError, OSError) as exc:
                errors.append(f"{Path(p).name}: {exc}")
                continue
            self.scans[key] = Scan(Path(p), prof, sample_name(p))
            added.append(key)
        self._rebuild_tree()
        self._refresh_stress_lists()
        if errors:
            self._set_status("error", "; ".join(errors)[:300])
        elif added:
            self._set_status("idle", f"{len(added)} scan(s) loaded")
            self.select(added[0])
        return added

    def _rebuild_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for key, sc in self.scans.items():
            parent = groups.get(sc.sample)
            if parent is None:
                parent = QTreeWidgetItem([sc.sample, "", ""])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.tree.addTopLevelItem(parent)
                groups[sc.sample] = parent
            t = sc.step.thickness_nm if sc.step else None
            item = QTreeWidgetItem([sc.path.name, f"{t:.4g}" if t is not None else "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            color = T.STATUS_COLORS.get(sc.status, T.STATUS_COLORS["idle"])
            item.setText(2, T.STATUS_TEXT.get(sc.status, sc.status))
            item.setForeground(2, QBrush(QColor(color)))
            parent.addChild(item)
            if key == self.current:
                item.setSelected(True)
        if self.batch is not None:
            for s in self.batch.samples:
                parent = groups.get(s.sample)
                if parent is not None and s.mean_nm is not None:
                    sd = f" ± {s.std_nm:.2g}" if s.std_nm is not None else ""
                    parent.setText(1, f"{s.mean_nm:.4g}{sd} (n={s.n})")
        self.tree.expandAll()
        self.tree.blockSignals(False)

    def _refresh_stress_lists(self) -> None:
        for cb in (self.cb_pre, self.cb_post):
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            if cb is self.cb_pre:
                cb.addItem("(none: assume flat)", None)
            for key, sc in self.scans.items():
                cb.addItem(sc.path.name, key)
            idx = cb.findData(cur)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            cb.blockSignals(False)

    def select(self, key: str) -> None:
        self.current = key
        sc = self.scans[key]
        self._mark_selected(key)
        if sc.step is not None:
            self._show_step(sc.step)
        else:
            self.plot.show_profile(sc.profile, sc.path.name)
            self._fill_notices(sc.profile.notices)
            self._ensure_cursors(sc.profile)

    def _mark_selected(self, key: str) -> None:
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top is None:
                continue
            for j in range(top.childCount()):
                child = top.child(j)
                if child is not None:
                    child.setSelected(child.data(0, Qt.ItemDataRole.UserRole) == key)
        self.tree.blockSignals(False)

    def _selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if items:
            key = items[0].data(0, Qt.ItemDataRole.UserRole)
            if key:
                self.select(str(key))

    # ------------------------------------------------------------------ cursors
    def _needs_cursors(self) -> bool:
        return (
            self.cb_level.currentText() == "regions"
            or self.chk_cursor_step.isChecked()
            or self.chk_rough_cursor.isChecked()
        )

    def _ensure_cursors(self, prof: Profile) -> None:
        if not self._needs_cursors():
            self.plot.hide_cursors()
            return
        regions = self.plot.cursor_regions()
        if len(regions) != 2:
            x0, x1 = float(prof.x_um.min()), float(prof.x_um.max())
            span = x1 - x0
            regions = [(x0 + 0.05 * span, x0 + 0.30 * span), (x0 + 0.70 * span, x0 + 0.95 * span)]
        self.plot.set_cursors(regions)

    def _mode_changed(self, *_: Any) -> None:
        if self.current:
            self._ensure_cursors(self.scans[self.current].profile)

    def _cursors_moved(self, regions: list[tuple[float, float]]) -> None:
        if self.current and (
            self.cb_level.currentText() == "regions" or self.chk_cursor_step.isChecked()
        ):
            self.analyze_current()

    def _edges_moved(self, positions: list[float]) -> None:
        if self.current:
            self.scans[self.current].edges_um = positions
            self.analyze_current()

    def _reset_edges(self) -> None:
        if self.current:
            self.scans[self.current].edges_um = None
            self.analyze_current()

    # ------------------------------------------------------------------ settings
    def step_settings(self, scan: Scan | None = None) -> StepSettings:
        cursors = self.plot.cursor_regions()
        level = self.cb_level.currentText()
        use_cursors = len(cursors) == 2
        return StepSettings(
            level=level if (level != "regions" or use_cursors) else "auto",
            level_method=self.cb_method.currentText(),
            level_regions_um=cursors if (level == "regions" and use_cursors) else None,
            level_separate=self.chk_separate.isChecked(),
            bow_order=self.sp_bow.value(),
            mode=self.cb_mode.currentText(),
            regions_um=cursors if (self.chk_cursor_step.isChecked() and use_cursors) else None,
            edges_um=scan.edges_um if scan else None,
            smooth_um=self.sp_smooth.value() or None,
            edge_k=self.sp_edge_k.value(),
            burr_k=self.sp_burr_k.value(),
            margin_um=self.sp_margin.value() or None,
            multiscale=self.chk_multiscale.isChecked(),
            widen=self.chk_widen.isChecked(),
            tip_radius_um=self.sp_tip.value() or None,
            cone_angle_deg=self.sp_cone.value() or None,
            stylus_force_mg=self.sp_force.value() or None,
            soft_film=self.chk_soft.isChecked(),
            tip_check=self.chk_tip_check.isChecked(),
        )

    def roughness_settings(self) -> RoughnessSettings:
        cursors = self.plot.cursor_regions()
        region = cursors[0] if (self.chk_rough_cursor.isChecked() and cursors) else None
        return RoughnessSettings(
            lambda_c_um=self.cb_lc.currentData(),
            lambda_s_um=self.cb_ls.currentData(),
            n_sections=self.sp_sections.value(),
            form_order=self.sp_form.value(),
            region_um=region,
        )

    def stress_settings(self) -> StressSettings:
        return StressSettings(
            substrate=self.cb_sub.currentData(),
            biaxial_modulus_gpa=self.sp_mod.value() or None,
            substrate_thickness_um=self.sp_hs.value(),
            film_thickness_nm=self.sp_hf.value() or None,
            backside=self.chk_back.isChecked(),
        )

    def _substrate_changed(self, *_: Any) -> None:
        s = SUBSTRATES.get(self.cb_sub.currentData())
        if s:
            self.lbl_source.setText(f"M = {s.biaxial_modulus_gpa:.2f} GPa — {s.source}")

    # ------------------------------------------------------------------ analyses
    def analyze_current(self) -> None:
        if not self.current:
            self._set_status("idle", "open a scan first")
            return
        key = self.current
        sc = self.scans[key]
        try:
            st = self.step_settings(sc)
        except ValueError as exc:
            self._set_status("error", str(exc))
            return
        prof = sc.profile

        def job() -> StepResult:
            return analyze_step(prof, st, DEFAULT_INSTRUMENT)

        def done(res: StepResult) -> None:
            sc.step = res
            sc.status = N.worst(res.notices)
            self._last_view = "step"
            self._rebuild_tree()
            if self.current == key:
                self._show_step(res)
            self._set_status(sc.status, self._step_summary(res))

        self.jobs.queue(job, done, self._job_failed, f"step: {sc.path.name}")

    def analyze_all(self) -> None:
        if not self.scans:
            return
        paths = [sc.path for sc in self.scans.values()]
        st = self.step_settings(None)

        def job() -> BatchResult:
            return run_batch(paths, st)

        def done(res: BatchResult) -> None:
            self.batch = res
            by_path = {str(Path(it.path).resolve()): it for it in res.items}
            for key, sc in self.scans.items():
                it = by_path.get(key)
                if it is None:
                    continue
                if it.result is not None and sc.edges_um is None:
                    sc.step = it.result
                    sc.status = N.worst(it.result.notices)
                elif it.error:
                    sc.status = "failed"
            self._rebuild_tree()
            self._fill_batch(res)
            self.batch_chart.show_batch(res, self.fig_options.error_bar)
            self.result_tabs.setCurrentWidget(self.batch_split)
            if self.current and self.scans[self.current].step:
                self._show_step(self.scans[self.current].step)  # type: ignore[arg-type]
            self._set_status("ok", f"{len(res.items)} scans, {len(res.samples)} samples")

        self.jobs.queue(job, done, self._job_failed, f"batch: {len(paths)} scans")

    def roughness_current(self) -> None:
        if not self.current:
            return
        sc = self.scans[self.current]
        st = self.roughness_settings()
        stylus = {
            "tip_radius_um": self.sp_tip.value() or None,
            "cone_angle_deg": self.sp_cone.value() or None,
        }
        prof = sc.profile.with_scan(**stylus) if any(stylus.values()) else sc.profile

        def job() -> RoughnessAnalysis:
            return analyze_roughness(prof, st)

        def done(res: RoughnessAnalysis) -> None:
            sc.roughness = res
            self._last_view = "roughness"
            self.plot.show_roughness(res, sc.path.name)
            rows = [
                (k, f"{v:.5g} {'µm' if k == 'Rsm' else '' if k in ('Rsk', 'Rku') else 'nm'}")
                for k, v in res.result.parameters.items()
            ]
            r = res.result
            rows += [
                ("λc", f"{r.lambda_c_um:g} µm"),
                ("λs", f"{r.lambda_s_um or 0:g} µm"),
                ("class", r.setting_class or "-"),
                ("sections", str(r.n_sections)),
                ("evaluation length", f"{r.evaluation_um[1] - r.evaluation_um[0]:.5g} µm"),
            ]
            self._fill_result(rows)
            self._fill_notices(res.notices)
            self.worksheet.set_columns(O.roughness_worksheet(res), sc.path.name)
            self._set_status(N.worst(res.notices), f"Ra = {r.parameters['Ra']:.4g} nm")

        self.jobs.queue(job, done, self._job_failed, f"roughness: {sc.path.name}")

    def stress_run(self) -> None:
        post_key = self.cb_post.currentData()
        if not post_key:
            self._set_status("idle", "load the scans and choose the 'after' scan (STRESS tab)")
            return
        pre_key = self.cb_pre.currentData()
        post = self.scans[post_key].profile
        pre = self.scans[pre_key].profile if pre_key else None
        st = self.stress_settings()

        def job() -> StressAnalysis:
            return analyze_stress(pre, post, st)

        def done(res: StressAnalysis) -> None:
            self.stress_result = res
            self._last_view = "stress"
            self.plot.show_stress(res, "stress")
            r = res.result
            rows = [
                ("stress", f"{r.stress_mpa:.5g} ± {r.u_stress_mpa:.3g} MPa ({r.kind})"),
                ("R before", f"{r.pre.radius_m:.5g} m" if r.pre else "flat (assumed)"),
                ("R after", f"{r.post.radius_m:.5g} m"),
                ("E/(1-ν)", f"{r.biaxial_modulus_gpa:.5g} GPa"),
                ("modulus source", r.modulus_source),
                ("h_s / h_f", f"{r.substrate_thickness_um:g} µm / {r.film_thickness_nm:g} nm"),
                ("sign", "positive = tensile"),
            ]
            self._fill_result(rows)
            self._fill_notices(res.notices)
            self.worksheet.set_columns(O.stress_worksheet(res), "stress")
            self._set_status(N.worst(res.notices), rows[0][1])

        self.jobs.queue(job, done, self._job_failed, "stress")

    def _job_failed(self, text: str) -> None:
        last = text.strip().splitlines()[-1] if text.strip() else text
        self._set_status("error", last)
        self.lst_notices.clear()
        self.lst_notices.addItem(text)

    # ------------------------------------------------------------------ results
    @staticmethod
    def _step_summary(res: StepResult) -> str:
        m = res.measurement
        if m.thickness_nm is None:
            return f"{m.kind}: {len(m.per_edge)} edge(s)"
        u = f" ± {m.u_nm:.3g}" if m.u_nm is not None else ""
        return f"{m.kind}: {m.thickness_nm:.5g}{u} nm"

    def _show_step(self, res: StepResult) -> None:
        self.plot.show_raw = self.act_raw.isChecked()
        self.plot.show_step(res, res.source.name)
        self._ensure_cursors(res.profile)
        m = res.measurement
        rows: list[tuple[str, str]] = [
            ("profile kind", f"{m.kind} (detected {res.kind_detected})"),
            ("method", m.method),
        ]
        if m.thickness_nm is not None:
            what = {"groove": "depth", "ridge": "height", "periodic": "mean step"}.get(
                m.kind, "step"
            )
            rows.append((what, f"{m.thickness_nm:.5g} nm"))
            if m.u_nm is not None:
                rows.append(
                    (
                        "u (k=1)",
                        f"{m.u_nm:.3g} nm (fit {m.u_fit_nm or 0:.2g},"
                        f" slope {m.u_slope_nm or 0:.2g}, tail {m.u_tail_nm or 0:.2g})",
                    )
                )
        if m.position_um is not None:
            rows.append(("position", f"{m.position_um:.5g} µm"))
        if m.width_um and m.kind in ("groove", "ridge"):
            rows.append(("width W", f"{m.width_um:.4g} µm"))
        if m.kind == "periodic":
            s = m.stats
            rows += [
                ("n edges", str(s.get("n"))),
                ("s.d.", f"{s.get('std_nm') or 0:.3g} nm"),
                ("period", f"{s.get('period_um') or 0:.4g} µm"),
            ]
        rows += [
            ("levelling", f"{res.level.method}, slope {res.level.slope_nm_per_um:.4g} nm/µm"),
            ("stylus radius", f"{res.tip_radius_um:g} µm ({res.tip_radius_source})"),
            ("stylus cone", f"{res.cone_angle_deg:g}° ({res.cone_angle_source})"),
        ]
        if res.tip_check is not None:
            hidden = res.tip_check.unreached_um
            rows.append(
                (
                    "not touched",
                    f"{100 * res.tip_check.unreached_fraction():.2g} % of the scan"
                    f" ({len(hidden)} region{'s' if len(hidden) != 1 else ''})",
                )
            )
        rows += [
            (
                "units",
                f"x {res.source.x_unit} ({res.source.x_unit_source}),"
                f" z {res.source.z_unit} ({res.source.z_unit_source})",
            ),
        ]
        self._fill_result(rows)
        self._fill_edges(m.per_edge)
        self._fill_notices(res.notices)
        self.worksheet.set_columns(O.step_worksheet(res), res.source.name)

    def _fill_result(self, rows: list[tuple[str, str]]) -> None:
        t = self.tbl_result
        t.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            t.setItem(i, 0, QTableWidgetItem(k))
            t.setItem(i, 1, QTableWidgetItem(v))
        t.resizeColumnToContents(0)

    def _fill_table(self, t: QTableWidget, rows: list[dict[str, Any]], cols: list[str]) -> None:
        t.clear()
        t.setColumnCount(len(cols))
        t.setRowCount(len(rows))
        t.setHorizontalHeaderLabels(cols)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                v = r.get(c)
                text = f"{v:.5g}" if isinstance(v, float) else ("" if v is None else str(v))
                t.setItem(i, j, QTableWidgetItem(text))
        t.resizeColumnsToContents()

    def _fill_edges(self, per_edge: list[dict[str, Any]]) -> None:
        self._fill_table(
            self.tbl_edges,
            per_edge,
            [
                "edge",
                "x_um",
                "direction",
                "height_nm",
                "u_nm",
                "u_fit_nm",
                "u_slope_nm",
                "u_tail_nm",
                "width_um",
            ],
        )

    def _fill_batch(self, res: BatchResult) -> None:
        self._fill_table(
            self.tbl_batch,
            res.sample_rows(),
            [
                "sample",
                "n",
                "thickness_mean_nm",
                "thickness_std_nm",
                "thickness_sem_nm",
                "thickness_ci95_nm",
                "thickness_cv_percent",
                "mean_single_scan_u_nm",
                "kinds",
                "warnings",
            ],
        )

    def _fill_notices(self, notices: list[Notice]) -> None:
        self.lst_notices.clear()
        for n in notices:
            item = QListWidgetItem(f"[{n.severity.upper()}] {n.code}: {n.message}")
            item.setForeground(QBrush(QColor(SEVERITY_COLOR.get(n.severity, "#888888"))))
            self.lst_notices.addItem(item)
        warn = sum(1 for n in notices if n.severity != N.INFO)
        self.result_tabs.setTabText(2, f"NOTICES ({warn})" if warn else "NOTICES")

    # ------------------------------------------------------------------ export
    def export(self, folder: str | None = None) -> list[Path]:
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not folder:
            return []
        out = Path(folder)
        from ..export import plots
        from ..export.project import save_project
        from ..export.tables import notice_rows, profile_rows, write_csv, write_xlsx

        written: list[Path] = []
        sc = self.scans.get(self.current or "")
        if self._last_view == "roughness" and sc and sc.roughness:
            res = sc.roughness
            stem = out / f"{sc.path.stem}_roughness"
            written.append(write_csv(stem.with_suffix(".csv"), [res.row()]))
            written.append(save_project(res, stem.with_suffix(".json")))
            written.append(
                O.write_origin_xlsx(
                    stem.with_name(stem.name + "_origin.xlsx"),
                    {
                        "Profile": O.roughness_worksheet(res),
                        "Curves": O.roughness_curves_worksheet(res),
                        "Parameters": O.roughness_parameters_worksheet(res),
                    },
                )
            )
            fig = plots.plot_roughness(res, title=sc.path.name, options=self._figure_options())
        elif self._last_view == "stress" and self.stress_result:
            res2 = self.stress_result
            stem = out / "stress"
            written.append(write_csv(stem.with_suffix(".csv"), [res2.row()]))
            written.append(save_project(res2, stem.with_suffix(".json")))
            written.append(
                O.write_origin_xlsx(
                    stem.with_name(stem.name + "_origin.xlsx"), {"Scans": O.stress_worksheet(res2)}
                )
            )
            fig = plots.plot_stress(res2, title="stress", options=self._figure_options(stress=True))
        elif sc and sc.step:
            res3 = sc.step
            stem = out / f"{sc.path.stem}_step"
            written.append(write_csv(stem.with_suffix(".csv"), [res3.row()]))
            written.append(
                write_xlsx(
                    stem.with_suffix(".xlsx"),
                    {
                        "Result": [res3.row()],
                        "Edges": res3.measurement.per_edge,
                        "Notices": notice_rows(res3.notices),
                        "Profile": profile_rows(res3.profile),
                    },
                )
            )
            written.append(save_project(res3, stem.with_suffix(".json"), DEFAULT_INSTRUMENT))
            written.append(
                O.write_origin_xlsx(
                    stem.with_name(stem.name + "_origin.xlsx"),
                    {
                        "Profile": O.step_worksheet(res3),
                        "Edges": O.step_edges_worksheet(res3),
                        "Residuals": O.plane_residual_worksheet(res3),
                    },
                )
            )
            fig = plots.plot_step(res3, title=sc.path.name, options=self._figure_options())
        else:
            self._set_status("idle", "nothing to export yet")
            return []
        try:
            written += plots.save(fig, stem, ["png", "svg"])
        finally:
            plots.close(fig)
        if self.batch is not None:
            written.append(write_csv(out / "batch_samples.csv", self.batch.sample_rows()))
            written.append(write_csv(out / "batch_scans.csv", self.batch.scan_rows()))
            written.append(save_project(self.batch, out / "batch.json", DEFAULT_INSTRUMENT))
            written.append(
                O.write_origin_xlsx(
                    out / "batch_origin.xlsx",
                    {
                        "Samples": O.batch_worksheet(self.batch),
                        "Scans": O.batch_scans_worksheet(self.batch),
                    },
                )
            )
            fig = plots.plot_batch(self.batch, options=self._figure_options())
            try:
                written += plots.save(fig, out / "batch_samples", ["png", "svg"])
            finally:
                plots.close(fig)
        self._set_status("ok", f"wrote {len(written)} files to {out}")
        return written

    # ------------------------------------------------------------------ view
    def set_theme(self, name: str) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if isinstance(app, QApplication):
            theme = T.apply_theme(app, name)
            self.settings.setValue("ui/theme", theme.name)
            self.worksheet.set_header_color(theme.panel_raised)
            if self.plot.look.name == "instrument":
                self.plot.set_style(T.plot_style("instrument", theme))
                self.batch_chart.set_style(self.plot.look)

    def set_plot_style(self, name: str) -> None:
        self.plot.set_style(T.plot_style(name))
        self.batch_chart.set_style(self.plot.look)
        self.settings.setValue("ui/plot", name)

    def _toggle_raw(self, on: bool) -> None:
        self.plot.show_raw = on
        if self.current and self.scans[self.current].step:
            self._show_step(self.scans[self.current].step)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ graph
    def set_graph_options(self, opts: GraphOptions) -> None:
        self.plot.set_options(opts)
        self.settings.setValue("ui/graph", opts.to_json())
        for act, on in (
            (self.act_resid, opts.show_residuals),
            (self.act_dims, opts.show_annotations),
        ):
            act.blockSignals(True)
            act.setChecked(on)
            act.blockSignals(False)

    def _toggle_option(self, name: str, on: bool) -> None:
        self.set_graph_options(self.plot.opts.with_changes(**{name: on}))

    def plot_details(self) -> None:
        dlg = PlotDetailsDialog(self.plot.opts, self)
        dlg.applied.connect(self.set_graph_options)
        dlg.exec()

    def copy_graph(self) -> QPixmap:
        """The live graph as an image on the clipboard (paste into slides or reports)."""
        pix = self.plot.grab()
        QApplication.clipboard().setPixmap(pix)
        self._set_status("ok", "graph copied to the clipboard")
        return pix

    def copy_worksheet(self) -> str:
        text = self.worksheet.sheet.as_tsv()
        QApplication.clipboard().setText(text)
        self._set_status("ok", "worksheet copied (paste into Origin or Excel)")
        return text

    def _selected_keys(self) -> list[str]:
        keys = []
        for item in self.tree.selectedItems():
            key = item.data(0, Qt.ItemDataRole.UserRole)
            if key:
                keys.append(str(key))
        return keys

    def overlay(self, keys: list[str] | None = None) -> None:
        """Selected scans (or every scan of the current sample) in one graph."""
        if keys is None:
            keys = self._selected_keys()
            if len(keys) < 2 and self.current:
                sample = self.scans[self.current].sample
                keys = [k for k, sc in self.scans.items() if sc.sample == sample]
        keys = [k for k in keys if k in self.scans]
        if not keys:
            self._set_status("idle", "open scans first")
            return
        curves = []
        for k in keys:
            sc = self.scans[k]
            if sc.step is not None:
                p = sc.step.profile
                z = np.where(p.valid, p.z_nm, np.nan)
                curves.append((sc.path.stem, p.x_um, z))
            else:
                p = sc.profile
                z = np.where(p.valid, p.z_nm, np.nan)
                curves.append((sc.path.stem, p.x_um, z - float(np.nanmedian(z))))
        self._last_view = "overlay"
        self.plot.show_overlay(curves)
        self._set_status("ok", f"overlay of {len(curves)} scans (levelled where analysed)")

    # ------------------------------------------------------------------ figures
    def _load_figure(self) -> tuple[FigureOptions, list[str], str, str]:
        try:
            data = json.loads(str(self.settings.value("ui/figure", "") or "{}"))
            opts = FigureOptions(**data.get("options", {}))
            formats = [str(f) for f in data.get("formats", ["png", "svg"])]
            return (
                opts,
                formats,
                str(data.get("style", "origin")),
                str(data.get("preset", "double")),
            )
        except (ValueError, TypeError):
            return FigureOptions(), ["png", "svg"], "origin", "double"

    def _save_figure_settings(self) -> None:
        data = {
            "options": asdict(self.fig_options),
            "formats": self.fig_formats,
            "style": self.fig_style,
            "preset": self.fig_preset,
        }
        self.settings.setValue("ui/figure", json.dumps(data, ensure_ascii=False))

    def _figure_options(self, stress: bool = False) -> FigureOptions:
        """Figure options with the axis units of the live graph."""
        from dataclasses import replace

        g = self.plot.opts
        x_unit, z_unit = g.x_unit, g.z_unit
        if stress and g.x_unit == "µm" and g.z_unit == "auto":
            x_unit, z_unit = "mm", "µm"
        return replace(self.fig_options, x_unit=x_unit, z_unit=z_unit)

    def export_figure(
        self,
        folder: str | None = None,
        options: FigureOptions | None = None,
        formats: list[str] | None = None,
        style: str | None = None,
    ) -> list[Path]:
        """Publication figure of the shown result (and of the batch, if there is one)."""
        from ..export import plots

        if options is None:
            dlg = ExportFigureDialog(
                self.fig_options, self.fig_formats, self.fig_style, self.fig_preset, self
            )
            if dlg.exec() != ExportFigureDialog.DialogCode.Accepted:
                return []
            options, formats, style = dlg.result_options()
            self.fig_options, self.fig_formats, self.fig_style = options, formats, style
            self.fig_preset = dlg.preset()
            self._save_figure_settings()
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Save figures to folder")
        if not folder:
            return []
        self.fig_options = options
        formats = formats or self.fig_formats
        style = style or self.fig_style
        out = Path(folder)
        sc = self.scans.get(self.current or "")
        jobs: list[tuple[Any, Path]] = []
        if self._last_view == "roughness" and sc and sc.roughness:
            fig = plots.plot_roughness(
                sc.roughness, title=sc.path.name, style=style, options=self._figure_options()
            )
            jobs.append((fig, out / f"{sc.path.stem}_roughness"))
        elif self._last_view == "stress" and self.stress_result:
            fig = plots.plot_stress(
                self.stress_result, title="stress", style=style, options=self._figure_options(True)
            )
            jobs.append((fig, out / "stress"))
        elif sc and sc.step:
            fig = plots.plot_step(
                sc.step, title=sc.path.name, style=style, options=self._figure_options()
            )
            jobs.append((fig, out / f"{sc.path.stem}_step"))
        if self.batch is not None:
            fig = plots.plot_batch(self.batch, style=style, options=self._figure_options())
            jobs.append((fig, out / "batch_samples"))
        if not jobs:
            self._set_status("idle", "nothing to draw yet")
            return []
        written: list[Path] = []
        for fig, stem in jobs:
            try:
                written += plots.save(fig, stem, formats)
            finally:
                plots.close(fig)
        self._set_status("ok", f"wrote {len(written)} figure file(s) to {out}")
        return written

    def export_worksheet(self, path: str | None = None) -> Path | None:
        """The worksheet shown in the WORKSHEET tab as an Origin-ready CSV."""
        cols = self.worksheet.sheet.columns
        if not cols:
            self._set_status("idle", "no worksheet yet: analyse a scan first")
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export worksheet", "worksheet_origin.csv", "CSV (*.csv)"
            )
        if not path:
            return None
        p = O.write_origin_csv(path, cols)
        self._set_status("ok", f"wrote {p}")
        return p

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "Profilometry Analyzer",
            f"<b>Profilometry Analyzer {__version__}</b><br>Step height, roughness and film"
            " stress from stylus profilometer scans.<br>MIT License.<br><br>Dektak is a"
            " trademark of Bruker; this program is not affiliated with or endorsed by"
            " Bruker.",
        )

    def _check_updates(self) -> None:
        from ..update_check import UpdateCheckError, check_for_update

        def job() -> Any:
            try:
                return check_for_update()
            except UpdateCheckError as exc:
                return exc

        def done(info: Any) -> None:
            if isinstance(info, Exception):
                self.update_label.setText("")
            elif info.is_newer:
                self.update_label.setText(
                    f"<a href='{info.url}'>version {info.latest} available</a>"
                )
                self.update_label.setOpenExternalLinks(True)

        self.jobs.queue(job, done, lambda _t: None, "update check")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt name
        self.jobs.shutdown()
        self.plot.shutdown()
        super().closeEvent(event)


def smoke_mode() -> bool:
    return bool(os.environ.get("PROFILOMETRY_ANALYZER_SMOKE"))


__all__ = ["MainWindow", "Scan"]

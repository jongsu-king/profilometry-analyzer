"""Plot Details and Export Figure dialogs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..export.plots import FORMATS, SIZE_PRESETS, FigureOptions
from .graph_options import X_UNITS, Z_UNITS, GraphOptions


def _limit_edit(value: float | None) -> QLineEdit:
    e = QLineEdit("" if value is None else f"{value:g}")
    e.setPlaceholderText("auto")
    e.setMaximumWidth(110)
    return e


def _read_limit(edit: QLineEdit) -> float | None:
    t = edit.text().strip().replace(",", ".")
    if not t or t.lower() == "auto":
        return None
    return float(t)


class PlotDetailsDialog(QDialog):
    """Axis units and limits, fonts, line width and layers of the live graph."""

    applied = Signal(object)

    def __init__(self, options: GraphOptions, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Details")
        self._start = options
        lay = QVBoxLayout(self)

        g = QGroupBox("AXES")
        f = QFormLayout(g)
        self.cb_x = QComboBox()
        self.cb_x.addItems(list(X_UNITS))
        self.cb_x.setCurrentText(options.x_unit)
        self.cb_z = QComboBox()
        self.cb_z.addItems(list(Z_UNITS))
        self.cb_z.setCurrentText(options.z_unit)
        self.ed_xmin, self.ed_xmax = _limit_edit(options.x_min), _limit_edit(options.x_max)
        self.ed_zmin, self.ed_zmax = _limit_edit(options.z_min), _limit_edit(options.z_max)
        f.addRow("x unit", self.cb_x)
        f.addRow("x from / to", self._pair(self.ed_xmin, self.ed_xmax))
        f.addRow("height unit", self.cb_z)
        f.addRow("height from / to", self._pair(self.ed_zmin, self.ed_zmax))
        lay.addWidget(g)

        g = QGroupBox("LINES AND TEXT")
        f = QFormLayout(g)
        self.sp_font = QSpinBox()
        self.sp_font.setRange(6, 24)
        self.sp_font.setValue(options.font_pt)
        self.sp_line = QDoubleSpinBox()
        self.sp_line.setRange(0.3, 6.0)
        self.sp_line.setSingleStep(0.1)
        self.sp_line.setValue(options.line_width)
        self.sp_offset = QDoubleSpinBox()
        self.sp_offset.setRange(0.0, 1e7)
        self.sp_offset.setDecimals(1)
        self.sp_offset.setSuffix(" nm")
        self.sp_offset.setValue(options.overlay_offset_nm)
        self.sp_offset.setToolTip("vertical offset between overlaid scans (0 = none)")
        f.addRow("font size (pt)", self.sp_font)
        f.addRow("line width", self.sp_line)
        f.addRow("overlay offset", self.sp_offset)
        lay.addWidget(g)

        g = QGroupBox("LAYERS")
        v = QVBoxLayout(g)
        self.chk_legend = QCheckBox("legend")
        self.chk_legend.setChecked(options.show_legend)
        self.chk_annot = QCheckBox("dimension lines (h, d, W)")
        self.chk_annot.setChecked(options.show_annotations)
        self.chk_resid = QCheckBox("residual layer below the profile")
        self.chk_resid.setChecked(options.show_residuals)
        for w in (self.chk_legend, self.chk_annot, self.chk_resid):
            v.addWidget(w)
        lay.addWidget(g)

        self.err = QLabel("")
        self.err.setObjectName("caption")
        lay.addWidget(self.err)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Reset
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        reset_btn = buttons.button(QDialogButtonBox.StandardButton.Reset)
        if apply_btn is not None:
            apply_btn.clicked.connect(self.apply)
        if reset_btn is not None:
            reset_btn.clicked.connect(self.reset)
        lay.addWidget(buttons)

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(QLabel("…"))
        h.addWidget(b)
        return w

    def options(self) -> GraphOptions:
        """The options in the dialog; raises ``ValueError`` for invalid input."""
        return GraphOptions(
            x_unit=self.cb_x.currentText(),
            z_unit=self.cb_z.currentText(),
            x_min=_read_limit(self.ed_xmin),
            x_max=_read_limit(self.ed_xmax),
            z_min=_read_limit(self.ed_zmin),
            z_max=_read_limit(self.ed_zmax),
            font_pt=self.sp_font.value(),
            line_width=self.sp_line.value(),
            show_legend=self.chk_legend.isChecked(),
            show_annotations=self.chk_annot.isChecked(),
            show_residuals=self.chk_resid.isChecked(),
            overlay_offset_nm=self.sp_offset.value(),
        )

    def apply(self) -> bool:
        try:
            opts = self.options()
        except ValueError as exc:
            self.err.setText(f"⚠ {exc}")
            return False
        self.err.setText("")
        self.applied.emit(opts)
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()

    def reset(self) -> None:
        d = GraphOptions()
        self.cb_x.setCurrentText(d.x_unit)
        self.cb_z.setCurrentText(d.z_unit)
        for e in (self.ed_xmin, self.ed_xmax, self.ed_zmin, self.ed_zmax):
            e.setText("")
        self.sp_font.setValue(d.font_pt)
        self.sp_line.setValue(d.line_width)
        self.sp_offset.setValue(d.overlay_offset_nm)
        self.chk_legend.setChecked(d.show_legend)
        self.chk_annot.setChecked(d.show_annotations)
        self.chk_resid.setChecked(d.show_residuals)


PRESET_LABELS = {
    "single": "single column (85 mm)",
    "onehalf": "1.5 column (120 mm)",
    "double": "double column (174 mm)",
    "slide": "slide (240 mm)",
    "custom": "custom",
}


class ExportFigureDialog(QDialog):
    """Publication figure: journal size, resolution, formats, layers."""

    def __init__(
        self,
        options: FigureOptions,
        formats: list[str],
        style: str = "origin",
        preset: str = "double",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Figure")
        lay = QVBoxLayout(self)
        g = QGroupBox("PAGE")
        f = QFormLayout(g)
        self.cb_preset = QComboBox()
        for key, label in PRESET_LABELS.items():
            self.cb_preset.addItem(label, key)
        self.sp_w = QDoubleSpinBox()
        self.sp_h = QDoubleSpinBox()
        for s in (self.sp_w, self.sp_h):
            s.setRange(30.0, 600.0)
            s.setDecimals(0)
            s.setSuffix(" mm")
        self.sp_w.setValue(options.width_mm or SIZE_PRESETS["double"][0])
        self.sp_h.setValue(options.height_mm or SIZE_PRESETS["double"][1])
        self.cb_dpi = QComboBox()
        for d in (150, 300, 600, 1200):
            self.cb_dpi.addItem(f"{d} dpi", d)
        self.cb_dpi.setCurrentIndex(max(self.cb_dpi.findData(options.dpi), 0))
        self.sp_font = QDoubleSpinBox()
        self.sp_font.setRange(5.0, 18.0)
        self.sp_font.setSingleStep(0.5)
        self.sp_font.setValue(options.font_pt)
        self.cb_style = QComboBox()
        self.cb_style.addItems(["origin", "classic"])
        self.cb_style.setCurrentText(style)
        f.addRow("size", self.cb_preset)
        f.addRow("width", self.sp_w)
        f.addRow("height", self.sp_h)
        f.addRow("resolution", self.cb_dpi)
        f.addRow("font (pt)", self.sp_font)
        f.addRow("style", self.cb_style)
        lay.addWidget(g)

        g = QGroupBox("CONTENT")
        v = QVBoxLayout(g)
        self.chk_annot = QCheckBox("dimension lines")
        self.chk_annot.setChecked(options.annotate)
        self.chk_resid = QCheckBox("residual layer")
        self.chk_resid.setChecked(options.residuals)
        self.chk_raw = QCheckBox("profile as read (step)")
        self.chk_raw.setChecked(options.show_raw)
        self.chk_panels = QCheckBox("panel letters (a), (b), …")
        self.chk_panels.setChecked(options.panel_labels)
        self.cb_err = QComboBox()
        self.cb_err.addItem("SD", "sd")
        self.cb_err.addItem("SE", "se")
        self.cb_err.addItem("95% CI", "ci95")
        self.cb_err.setCurrentIndex(max(self.cb_err.findData(options.error_bar), 0))
        for w in (self.chk_annot, self.chk_resid, self.chk_raw, self.chk_panels):
            v.addWidget(w)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("batch error bars"))
        h.addWidget(self.cb_err)
        v.addWidget(row)
        lay.addWidget(g)

        g = QGroupBox("FORMATS")
        h = QHBoxLayout(g)
        self.fmt_checks: dict[str, QCheckBox] = {}
        for fmt in FORMATS:
            c = QCheckBox(fmt.upper())
            c.setChecked(fmt in formats)
            self.fmt_checks[fmt] = c
            h.addWidget(c)
        lay.addWidget(g)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.cb_preset.currentIndexChanged.connect(self._preset_changed)
        self._base = options
        idx = self.cb_preset.findData(preset)
        self.cb_preset.setCurrentIndex(idx if idx >= 0 else 2)
        if preset == "custom":
            self.sp_w.setValue(options.width_mm or SIZE_PRESETS["double"][0])
            self.sp_h.setValue(options.height_mm or SIZE_PRESETS["double"][1])
        self._preset_changed()  # also when the index did not change (no signal then)
        self.sp_font.setValue(options.font_pt)  # keep the font the user chose last

    def _preset_changed(self, *_: Any) -> None:
        key = self.cb_preset.currentData()
        custom = key == "custom"
        self.sp_w.setEnabled(custom)
        self.sp_h.setEnabled(custom)
        if key in SIZE_PRESETS and hasattr(self, "_base"):
            w, h = SIZE_PRESETS[key]
            self.sp_w.setValue(w)
            self.sp_h.setValue(h)
            self.sp_font.setValue(8.0 if key == "single" else 10.0 if key == "slide" else 9.0)

    def preset(self) -> str:
        return str(self.cb_preset.currentData())

    def result_options(self) -> tuple[FigureOptions, list[str], str]:
        opts = replace(
            self._base,
            width_mm=self.sp_w.value(),
            height_mm=self.sp_h.value(),
            dpi=int(self.cb_dpi.currentData()),
            font_pt=self.sp_font.value(),
            annotate=self.chk_annot.isChecked(),
            residuals=self.chk_resid.isChecked(),
            show_raw=self.chk_raw.isChecked(),
            panel_labels=self.chk_panels.isChecked(),
            error_bar=str(self.cb_err.currentData()),
        )
        formats = [f for f, c in self.fmt_checks.items() if c.isChecked()] or ["png"]
        return opts, formats, self.cb_style.currentText()


__all__ = ["ExportFigureDialog", "PlotDetailsDialog"]

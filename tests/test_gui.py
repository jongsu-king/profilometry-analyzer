from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK", "1")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QCoreApplication, QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from profilometry_analyzer.synthetic import demo_set  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    QCoreApplication.setOrganizationName("profilometry-analyzer-tests")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    inst = QApplication.instance() or QApplication([])
    from profilometry_analyzer.gui.theme import apply_theme

    apply_theme(inst, "dark")  # type: ignore[arg-type]
    return inst  # type: ignore[return-value]


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("guidemo")
    demo_set(out)
    return out


@pytest.fixture
def window(app: QApplication) -> Iterator[object]:
    from profilometry_analyzer.gui.main_window import MainWindow

    w = MainWindow(synchronous_jobs=True, check_updates=False)
    w.show()
    yield w
    # never just drop a window: close, delete and let Qt process it (see conftest)
    w.close()
    w.deleteLater()
    app.processEvents()


def test_open_analyze_and_export(window, demo: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    w = window
    keys = w.add_files([demo / "filmA_1.csv", demo / "filmA_2.csv", demo / "scratch_1.csv"])
    assert len(keys) == 3 and w.current == keys[0]
    assert w.tree.topLevelItemCount() == 2  # samples filmA and scratch
    w.analyze_current()
    res = w.scans[keys[0]].step
    assert res is not None and res.thickness_nm == pytest.approx(84.4, abs=0.3)
    assert w.tbl_result.rowCount() > 3
    assert len(w.plot.edge_positions()) == 1
    files = w.export(str(tmp_path / "exp"))
    names = {p.name for p in files}
    assert "filmA_1_step.json" in names and "filmA_1_step.png" in names


def test_edit_edges_and_cursors(window, demo: Path) -> None:  # type: ignore[no-untyped-def]
    w = window
    (key,) = w.add_files([demo / "filmB_1.csv"])
    w.analyze_current()
    before = w.scans[key].step.thickness_nm
    # drag the edge line: the analysis runs again with the user's edge
    w._edges_moved([505.0])
    after = w.scans[key].step
    assert after.edges.edges[0].manual
    assert after.thickness_nm == pytest.approx(before, abs=0.5)
    w._reset_edges()
    assert not w.scans[key].step.edges.edges[0].manual
    # cursors R and M: manual step measurement
    w.chk_cursor_step.setChecked(True)
    w.plot.set_cursors([(100.0, 400.0), (600.0, 900.0)])
    w._cursors_moved(w.plot.cursor_regions())
    assert w.scans[key].step.measurement.method == "regions"
    assert w.scans[key].step.thickness_nm == pytest.approx(before, abs=0.5)
    w.chk_cursor_step.setChecked(False)


def test_wide_edge_options(window, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from profilometry_analyzer.synthetic import SynthSpec, synthesize, write_csv

    w = window
    syn = synthesize(
        SynthSpec(kind="edge", length_um=2000, n_points=20_000, wall_um=200, noise_nm=1.0, seed=0)
    )
    f = write_csv(syn.profile, tmp_path / "wide_1.csv")
    (key,) = w.add_files([f])
    assert w.chk_multiscale.isChecked() and w.chk_widen.isChecked()
    st = w.step_settings()
    assert st.multiscale and st.widen
    w.analyze_current()
    on = w.scans[key].step
    assert on.edges.edges[0].widened_um[0] > 0
    assert on.thickness_nm == pytest.approx(100.0, abs=0.4)
    w.chk_widen.setChecked(False)
    w.chk_multiscale.setChecked(False)
    st = w.step_settings()
    assert not st.multiscale and not st.widen
    w.analyze_current()
    assert w.scans[key].step.edges.edges[0].widened_um == (0.0, 0.0)
    w.chk_widen.setChecked(True)
    w.chk_multiscale.setChecked(True)


def test_stylus_options(window, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from profilometry_analyzer.synthetic import SynthSpec, synthesize, write_csv

    w = window
    syn = synthesize(
        SynthSpec(kind="edge", height_nm=5000, wall_um=0.05, noise_nm=1.0, tip_radius_um=2.0)
    )
    f = write_csv(syn.profile, tmp_path / "tall_1.csv", metadata={"Stylus Radius": "2 um"})
    (key,) = w.add_files([f])
    st = w.step_settings()
    assert st.cone_angle_deg is None and st.tip_check and w.chk_tip_check.isChecked()
    w.analyze_current()
    res = w.scans[key].step
    assert res.tip_check is not None and res.tip_check.unreached_um
    assert res.cone_angle_source == "instrument default"
    w.sp_cone.setValue(45)
    w.chk_tip_check.setChecked(False)
    assert w.step_settings().cone_angle_deg == 45
    w.analyze_current()
    res = w.scans[key].step
    assert res.tip_check is None and res.cone_angle_deg == 45
    w.roughness_current()
    assert w.scans[key].roughness.result.stylus["cone_angle_deg"] == 45
    w.sp_cone.setValue(0)
    w.chk_tip_check.setChecked(True)


def test_batch_roughness_stress(window, demo: Path) -> None:  # type: ignore[no-untyped-def]
    w = window
    w.add_files(sorted(demo.glob("film*.csv")))
    w.analyze_all()
    assert w.batch is not None and len(w.batch.samples) == 3
    assert w.tbl_batch.rowCount() == 3
    (rk,) = w.add_files([demo / "roughness" / "sine_A12nm_40um.csv"])
    w.select(rk)
    w.roughness_current()
    assert w.scans[rk].roughness is not None
    pre, post = w.add_files([demo / "stress" / "wafer_pre.csv", demo / "stress" / "wafer_post.csv"])
    w.cb_pre.setCurrentIndex(w.cb_pre.findData(pre))
    w.cb_post.setCurrentIndex(w.cb_post.findData(post))
    w.sp_hf.setValue(500.0)
    w.stress_run()
    assert w.stress_result is not None and w.stress_result.result.kind == "tensile"


def test_themes_and_plot_styles(window, demo: Path) -> None:  # type: ignore[no-untyped-def]
    w = window
    w.add_files([demo / "filmC_1.csv"])
    w.analyze_current()
    for theme in ("light", "dark"):
        w.set_theme(theme)
    for style in ("instrument", "origin"):
        w.set_plot_style(style)
        assert w.plot.look.name == style
    w.act_raw.setChecked(False)
    w.act_raw.setChecked(True)


def test_threaded_jobs_finish_cleanly(app: QApplication) -> None:
    """Many real QThread jobs in a row: quit() -> wait() -> drop references."""
    from profilometry_analyzer.gui.workers import JobRunner

    runner = JobRunner()
    results: list[int] = []
    for i in range(30):
        runner.queue(lambda i=i: i * 2, results.append, lambda t: results.append(-1))
    import time

    t0 = time.time()
    while (runner.busy or runner.pending) and time.time() - t0 < 20:
        app.processEvents()
        time.sleep(0.002)
    assert results == [i * 2 for i in range(30)]
    assert not runner.busy and runner._worker is None
    failures: list[str] = []
    runner.queue(lambda: 1 / 0, results.append, failures.append)
    t0 = time.time()
    while runner.busy and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.002)
    assert failures and "ZeroDivisionError" in failures[0]
    runner.shutdown()
    runner.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------- Origin-style GUI
def test_graph_options_storage_and_validation() -> None:
    from profilometry_analyzer.gui.graph_options import GraphOptions

    o = GraphOptions(x_unit="mm", z_unit="Å", x_min=0.1, x_max=0.9, font_pt=12)
    assert GraphOptions.from_json(o.to_json()) == o
    assert GraphOptions.from_json("not json") == GraphOptions()
    assert GraphOptions.from_json('{"x_unit": "cm"}') == GraphOptions()
    assert o.sx == 1e-3 and o.z_factor(10.0) == (10.0, "Å")
    assert GraphOptions().z_factor(6000.0) == (1e-3, "µm")
    for bad in ({"x_unit": "cm"}, {"x_min": 2.0, "x_max": 1.0}, {"font_pt": 3}):
        with pytest.raises(ValueError):
            GraphOptions(**bad)  # type: ignore[arg-type]


def test_display_units_keep_positions_in_um(window, demo: Path) -> None:  # type: ignore[no-untyped-def]
    from profilometry_analyzer.gui.graph_options import GraphOptions

    w = window
    (key,) = w.add_files([demo / "filmB_1.csv"])
    w.analyze_current()
    edge_um = w.plot.edge_positions()[0]
    w.chk_cursor_step.setChecked(True)
    w.plot.set_cursors([(100.0, 400.0), (600.0, 900.0)])
    w.set_graph_options(GraphOptions(x_unit="mm", z_unit="µm"))
    curves = [it for it in w.plot.main.listDataItems() if isinstance(it, pg.PlotDataItem)]
    xs = [float(np.nanmax(it.xData)) for it in curves if it.xData is not None and it.xData.size]
    assert max(xs) == pytest.approx(1.0, abs=0.01)  # 1000 µm shown as 1 mm
    assert w.plot.edge_positions()[0] == pytest.approx(edge_um)
    assert w.plot.cursor_regions()[0] == pytest.approx((100.0, 400.0))
    assert "mm" in w.plot.format_point(500.0, 150.0) and "µm" in w.plot.format_point(500.0, 150.0)
    w._cursors_moved(w.plot.cursor_regions())
    assert w.scans[key].step.thickness_nm == pytest.approx(152.0, abs=0.6)
    w.chk_cursor_step.setChecked(False)
    w.set_graph_options(GraphOptions(x_min=400.0, x_max=600.0))
    xr = w.plot.main.getViewBox().viewRange()[0]
    assert xr == pytest.approx([400.0, 600.0], abs=1e-6)
    w.set_graph_options(GraphOptions())


def test_residual_layer_and_dimension_lines(window, demo: Path) -> None:  # type: ignore[no-untyped-def]
    w = window
    (key,) = w.add_files([demo / "scratch_1.csv"])
    w.analyze_current()
    assert w.plot.residuals_visible and len(w.plot.dimension_items) > 0
    w.act_resid.setChecked(False)
    w.act_dims.setChecked(False)
    assert not w.plot.residuals_visible and w.plot.dimension_items == []
    assert not w.plot.opts.show_residuals
    w.act_resid.setChecked(True)
    w.act_dims.setChecked(True)
    assert w.plot.residuals_visible
    w.plot.show_profile(w.scans[key].profile)
    assert not w.plot.residuals_visible  # no fit, no residuals


def test_data_reader_and_two_point_measurement(window, demo: Path) -> None:  # type: ignore[no-untyped-def]
    w = window
    w.add_files([demo / "filmA_1.csv"])
    w.analyze_current()
    x, z = w.plot.reader_point(250.004)
    prof = w.scans[w.current].step.profile
    assert x in prof.x_um and z == pytest.approx(float(prof.z_nm[np.searchsorted(prof.x_um, x)]))
    assert w.plot.add_measure_point(200.0) is None
    info = w.plot.add_measure_point(800.0)
    assert info is not None and info["dx_um"] == pytest.approx(600.0, abs=0.2)
    assert info["dz_nm"] == pytest.approx(84.4, abs=4.0)
    assert "Δx" in w.measure_label.text() and "Δz" in w.measure_label.text()
    w.plot.add_measure_point(300.0)  # a third click starts again
    assert len(w.plot.measure_points) == 1
    w.plot.clear_measure()
    assert w.plot.measure_points == []


def test_overlay_copy_and_worksheet(window, demo: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from profilometry_analyzer.gui.graph_options import GraphOptions

    w = window
    keys = w.add_files([demo / "filmA_1.csv", demo / "filmA_2.csv", demo / "filmB_1.csv"])
    w.analyze_current()
    w.set_graph_options(GraphOptions(overlay_offset_nm=50.0))
    w.overlay(keys)
    curves = [it for it in w.plot.main.listDataItems() if isinstance(it, pg.PlotDataItem)]
    names = [it.name() for it in curves]
    assert len(names) == 3 and "+50 nm" in names[1]
    w.set_graph_options(GraphOptions())
    w.tree.clearSelection()
    w.select(keys[0])
    w.overlay()  # nothing selected beyond one scan: the scans of its sample
    assert sum(isinstance(it, pg.PlotDataItem) for it in w.plot.main.listDataItems()) == 2
    pix = w.copy_graph()
    assert not pix.isNull()
    w.select(keys[0])
    sheet = w.worksheet.sheet
    assert sheet.rowCount() == w.scans[keys[0]].step.profile.n + 3
    assert sheet.headerData(0, Qt.Orientation.Horizontal) == "A(X)"
    assert sheet.headerData(1, Qt.Orientation.Vertical) == "Units"
    text = w.copy_worksheet()
    assert text.startswith("Position\tHeight as read") and "\nµm\tnm" in text
    from PySide6.QtWidgets import QApplication

    assert QApplication.clipboard().text().startswith("Position")
    p = w.export_worksheet(str(tmp_path / "ws.csv"))
    assert p is not None and p.read_text(encoding="utf-8-sig").startswith("Position,")


def test_export_figure_and_batch_chart(window, demo: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from profilometry_analyzer.export.plots import figure_options

    w = window
    w.add_files(sorted(demo.glob("film*.csv")))
    w.analyze_all()
    assert w.batch_chart.bars is not None
    assert w.tbl_batch.columnCount() >= 8
    files = w.export_figure(
        str(tmp_path / "f"),
        options=figure_options("single", dpi=120),
        formats=["png", "tiff", "svg"],
    )
    names = {f.name for f in files}
    assert any(n.endswith("_step.tiff") for n in names)
    assert {"batch_samples.png", "batch_samples.svg"} <= names
    out = w.export(str(tmp_path / "e"))
    assert any(p.name.endswith("_origin.xlsx") for p in out)
    assert any(p.name == "batch_origin.xlsx" for p in out)


def test_plot_details_and_export_dialogs(app: QApplication) -> None:
    from profilometry_analyzer.export.plots import FigureOptions
    from profilometry_analyzer.gui.dialogs import ExportFigureDialog, PlotDetailsDialog
    from profilometry_analyzer.gui.graph_options import GraphOptions

    got: list[GraphOptions] = []
    dlg = PlotDetailsDialog(GraphOptions(), None)
    dlg.applied.connect(got.append)
    dlg.cb_x.setCurrentText("mm")
    dlg.cb_z.setCurrentText("Å")
    dlg.ed_zmin.setText("-10")
    dlg.ed_zmax.setText("3000")
    dlg.sp_font.setValue(12)
    dlg.chk_resid.setChecked(False)
    assert dlg.apply()
    assert got[-1] == GraphOptions(
        x_unit="mm", z_unit="Å", z_min=-10.0, z_max=3000.0, font_pt=12, show_residuals=False
    )
    dlg.ed_xmin.setText("5")
    dlg.ed_xmax.setText("1")
    assert not dlg.apply() and "minimum" in dlg.err.text()
    dlg.reset()
    assert dlg.options() == GraphOptions()
    dlg.deleteLater()

    ex = ExportFigureDialog(FigureOptions(), ["png"], "origin", "single")
    assert ex.sp_w.value() == 85.0 and not ex.sp_w.isEnabled()
    ex.cb_preset.setCurrentIndex(ex.cb_preset.findData("custom"))
    ex.sp_w.setValue(140.0)
    ex.fmt_checks["tiff"].setChecked(True)
    ex.cb_err.setCurrentIndex(ex.cb_err.findData("ci95"))
    opts, formats, style = ex.result_options()
    assert opts.width_mm == 140.0 and opts.error_bar == "ci95"
    assert formats == ["png", "tiff"] and style == "origin" and ex.preset() == "custom"
    ex.deleteLater()
    app.processEvents()

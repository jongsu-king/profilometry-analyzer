from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK", "1")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QCoreApplication, QSettings  # noqa: E402
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

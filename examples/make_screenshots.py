"""Regenerate the screenshots and example figures in docs/images from synthetic scans.

QT_QPA_PLATFORM=offscreen python examples/make_screenshots.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK", "1")

from PySide6.QtCore import QCoreApplication, QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from profilometry_analyzer.analysis import analyze_roughness  # noqa: E402
from profilometry_analyzer.core.roughness import RoughnessSettings  # noqa: E402
from profilometry_analyzer.export import plots  # noqa: E402
from profilometry_analyzer.gui.graph_options import GraphOptions  # noqa: E402
from profilometry_analyzer.gui.main_window import MainWindow  # noqa: E402
from profilometry_analyzer.gui.theme import apply_theme  # noqa: E402
from profilometry_analyzer.readers import read_profile  # noqa: E402
from profilometry_analyzer.synthetic import demo_set  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> int:
    # keep the user's own settings out of the screenshots
    QCoreApplication.setOrganizationName("profilometry-analyzer-screenshots")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, "dark")  # type: ignore[arg-type]
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        demo_set(tmp)
        w = MainWindow(synchronous_jobs=True, check_updates=False)
        w.set_graph_options(GraphOptions())
        w.resize(1500, 920)
        w.show()
        files = sorted(Path(tmp).glob("*.csv"))
        w.add_files(files)
        w.analyze_all()
        key = str((Path(tmp) / "scratch_1.csv").resolve())
        w.select(key)
        w.result_tabs.setCurrentIndex(0)
        app.processEvents()
        w.grab().save(str(OUT / "gui_groove.png"))

        w.result_tabs.setCurrentWidget(w.batch_split)
        w.select(str((Path(tmp) / "filmB_2.csv").resolve()))
        w.chk_cursor_step.setChecked(True)
        w.plot.set_cursors([(120.0, 420.0), (580.0, 880.0)])
        w.analyze_current()
        w.plot.add_measure_point(300.0)
        w.plot.add_measure_point(700.0)
        app.processEvents()
        w.grab().save(str(OUT / "gui_cursors_batch.png"))
        w.chk_cursor_step.setChecked(False)

        w.select(str((Path(tmp) / "filmC_1.csv").resolve()))
        w.set_graph_options(GraphOptions(x_unit="mm", overlay_offset_nm=0.0))
        w.result_tabs.setCurrentWidget(w.worksheet)
        w.overlay([k for k in w.scans if Path(k).stem.startswith(("filmA", "filmB", "filmC"))])
        app.processEvents()
        w.grab().save(str(OUT / "gui_overlay_worksheet.png"))
        w.set_graph_options(GraphOptions())

        step = w.scans[key].step
        assert step is not None
        opts = plots.figure_options("double", dpi=150)
        fig = plots.plot_step(step, title="scratch_1", options=opts)
        plots.save(fig, OUT / "figure_step", ["png"])
        plots.close(fig)
        rr = analyze_roughness(
            read_profile(Path(tmp) / "roughness" / "sine_A12nm_40um.csv"),
            RoughnessSettings(lambda_c_um=250.0, lambda_s_um=0.0),
        )
        fig = plots.plot_roughness(rr, title="sine_A12nm_40um", options=opts)
        plots.save(fig, OUT / "figure_roughness", ["png"])
        plots.close(fig)
        assert w.batch is not None
        fig = plots.plot_batch(w.batch, options=plots.figure_options("single", dpi=150))
        plots.save(fig, OUT / "figure_batch", ["png"])
        plots.close(fig)
        w.close()
        w.deleteLater()
        app.processEvents()
    print(f"screenshots and figures in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

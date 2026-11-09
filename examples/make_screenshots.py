"""Regenerate the screenshots in docs/images from synthetic scans.

QT_QPA_PLATFORM=offscreen python examples/make_screenshots.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK", "1")

from PySide6.QtWidgets import QApplication  # noqa: E402

from profilometry_analyzer.gui.main_window import MainWindow  # noqa: E402
from profilometry_analyzer.gui.theme import apply_theme  # noqa: E402
from profilometry_analyzer.synthetic import demo_set  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, "dark")  # type: ignore[arg-type]
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        demo_set(tmp)
        w = MainWindow(synchronous_jobs=True, check_updates=False)
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
        w.result_tabs.setCurrentWidget(w.tbl_batch)
        w.select(str((Path(tmp) / "filmB_2.csv").resolve()))
        w.chk_cursor_step.setChecked(True)
        w.plot.set_cursors([(120.0, 420.0), (580.0, 880.0)])
        w.analyze_current()
        app.processEvents()
        w.grab().save(str(OUT / "gui_cursors_batch.png"))
        w.close()
        w.deleteLater()
        app.processEvents()
    print(f"screenshots in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

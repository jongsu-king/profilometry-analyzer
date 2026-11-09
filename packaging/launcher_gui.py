"""Entry point of the frozen desktop application.

``PROFILOMETRY_ANALYZER_SMOKE=1`` opens the window, waits briefly and exits with 0 so the
packaged build can be checked on a headless runner (with ``QT_QPA_PLATFORM=offscreen``).
"""

from __future__ import annotations

import os
import sys

import frozen_env


def main() -> int:
    frozen_env.prepare()
    from profilometry_analyzer.gui import run

    if os.environ.get("PROFILOMETRY_ANALYZER_SMOKE"):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)
        QTimer.singleShot(1500, app.quit)
    return int(run(sys.argv))


if __name__ == "__main__":
    sys.exit(main())

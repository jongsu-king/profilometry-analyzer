"""Desktop application (PySide6 + PyQtGraph).

Qt is imported lazily so the analysis engine never depends on it.  Start with
``profilometry-analyzer gui`` or ``python -m profilometry_analyzer.gui``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

GUI_INSTALL_HINT = (
    "The desktop application needs PySide6 and pyqtgraph. "
    "Install them with:  pip install 'profilometry-analyzer[gui]'"
)


def run(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(f"{GUI_INSTALL_HINT}\n({exc})") from exc
    from PySide6.QtCore import QSettings

    from .main_window import APP, ORG, MainWindow
    from .theme import DEFAULT_THEME, THEMES, apply_theme

    args = list(argv if argv is not None else sys.argv)
    app = QApplication.instance()
    if app is None:
        app = QApplication(args)
    theme = str(QSettings(ORG, APP).value("ui/theme", DEFAULT_THEME))
    apply_theme(cast(QApplication, app), theme if theme in THEMES else DEFAULT_THEME)
    window = MainWindow()
    files = [Path(a) for a in args[1:] if Path(a).is_file()]
    if files:
        window.add_files(files)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())

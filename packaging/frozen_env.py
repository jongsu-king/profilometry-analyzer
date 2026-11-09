"""Environment preparation shared by the frozen launchers.

Matplotlib keeps its font cache in a configuration directory; inside a PyInstaller
bundle the default location is not persistent, so every start would rebuild the
cache (about ten seconds).  Point it to a per-user cache folder instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_FOLDER = "profilometry-analyzer"


def user_cache_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / APP_FOLDER
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_FOLDER / "cache"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / APP_FOLDER


def prepare() -> None:
    # PyInstaller's matplotlib runtime hook points MPLCONFIGDIR at a fresh temporary
    # directory on every start, which forces a ~10 s font-cache rebuild each time.
    # In the frozen build we always redirect it to a persistent per-user folder.
    if getattr(sys, "frozen", False) or "MPLCONFIGDIR" not in os.environ:
        target = user_cache_dir() / "matplotlib"
        try:
            target.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(target)
        except OSError:
            pass  # fall back to matplotlib's own choice
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

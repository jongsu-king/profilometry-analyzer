# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build: one folder with the desktop app and the command-line tool.

    pyinstaller packaging/profilometry_analyzer.spec --noconfirm

Produces ``dist/Profilometry Analyzer/`` (Windows, Linux) or
``dist/Profilometry Analyzer.app`` (macOS) containing

* ``Profilometry Analyzer``  - the windowed desktop application
* ``profilometry-analyzer``  - the console command-line tool

Both share one ``_internal`` runtime, so the folder is about half the size of two
separate builds.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
PKG = ROOT / "packaging"
sys.path.insert(0, str(SRC))
import profilometry_analyzer  # noqa: E402

VERSION = profilometry_analyzer.__version__
APP_NAME = "Profilometry Analyzer"
CLI_NAME = "profilometry-analyzer"
IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
ICON = str(PKG / ("icon.ico" if IS_WIN else "icon.icns" if IS_MAC else "icon.png"))

datas = collect_data_files("pyqtgraph", includes=["**/*.png", "**/*.ui", "**/*.svg"])
datas += collect_data_files("matplotlib", includes=["mpl-data/**"])
hiddenimports = (
    collect_submodules("scipy.special")
    + collect_submodules("scipy.signal")
    + collect_submodules("scipy.ndimage")
    + [
        "scipy._lib.messagestream",
        "scipy._cyutility",
        "openpyxl",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
        "matplotlib.backends.backend_pdf",
        "pyqtgraph",
        "pyqtgraph.exporters",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
    ]
)
excludes = [
    "tkinter",
    "_tkinter",
    "IPython",
    "jupyter",
    "pytest",
    "mypy",
    "ruff",
    "pandas",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSerialBus",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtTextToSpeech",
    "PySide6.QtHttpServer",
    "PySide6.QtSpatialAudio",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtUiTools",
    "PySide6.QtTest",
    "PySide6.QtSql",
    "PySide6.QtNetworkAuth",
]

common = dict(
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

gui_a = Analysis([str(PKG / "launcher_gui.py")], **common)
cli_a = Analysis([str(PKG / "launcher_cli.py")], **common)

gui_pyz = PYZ(gui_a.pure)
cli_pyz = PYZ(cli_a.pure)

gui_exe = EXE(
    gui_pyz,
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON,
)
cli_exe = EXE(
    cli_pyz,
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name=CLI_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=ICON,
)

coll = COLLECT(
    gui_exe,
    cli_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON,
        bundle_identifier="io.github.jongsu-king.profilometry-analyzer",
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "12.0",
            "NSHumanReadableCopyright": "MIT License",
        },
    )

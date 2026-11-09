"""Themes and plot styles for the desktop application (taken from xps-autofit-studio).

Two application themes share one stylesheet template: the instrument-style **dark**
theme (flat panels, hairline borders, status lamps) and a **light** theme with the
same layout.  Colours follow the validated palette of the figures
(``export.plots``): the dark theme uses the dark-surface steps for series, the
light theme the light-surface steps, so plot and panels read as one system.

The plot has its own style, independent of the theme: **origin** is a
publication look in the manner of OriginLab graphs (white canvas, black frame on
all four sides, inward ticks, Arial labels, no grid, open black data symbols, red
envelope), **instrument** follows the theme (dark or light canvas, grid, axes on
the left and bottom only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtWidgets import QApplication

RGBA = tuple[int, int, int, int]

# Series colours of the figures (``export.plots.SERIES``); kept identical there.
SERIES_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
SERIES_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]


@dataclass(frozen=True)
class Theme:
    """Colour tokens of an application theme."""

    name: str
    surface: str
    panel: str
    panel_raised: str
    plot_bg: str
    border: str
    ink: str
    ink_secondary: str
    muted: str
    accent: str
    accent_dim: str
    grid: str
    series: tuple[str, ...]
    label_fill: RGBA
    """Background of the labels that float on the plot (component centres, element lines)."""


DARK = Theme(
    name="dark",
    surface="#1a1a19",
    panel="#222221",
    panel_raised="#2a2a28",
    plot_bg="#131312",
    border="#383835",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    accent="#3987e5",
    accent_dim="#1c5cab",
    grid="#2c2c2a",
    series=tuple(SERIES_DARK),
    label_fill=(20, 20, 19, 190),
)

LIGHT = Theme(
    name="light",
    surface="#f3f2ee",
    panel="#e9e8e2",
    panel_raised="#dfded7",
    plot_bg="#fcfcfb",
    border="#c4c2b9",
    ink="#1a1a19",
    ink_secondary="#44433f",
    muted="#7a7872",
    accent="#1f5fbf",
    accent_dim="#c9dcf5",
    grid="#dddbd3",
    series=tuple(SERIES_LIGHT),
    label_fill=(252, 252, 251, 215),
)

THEMES: dict[str, Theme] = {DARK.name: DARK, LIGHT.name: LIGHT}
DEFAULT_THEME = DARK.name

# Backwards-compatible module constants: the dark theme's tokens.
SURFACE = DARK.surface
PANEL = DARK.panel
PANEL_RAISED = DARK.panel_raised
PLOT_BG = DARK.plot_bg
BORDER = DARK.border
INK = DARK.ink
INK_SECONDARY = DARK.ink_secondary
MUTED = DARK.muted
ACCENT = DARK.accent
ACCENT_DIM = DARK.accent_dim
GRID = DARK.grid

STATUS_COLORS = {
    "idle": "#5a5955",
    "running": ACCENT,
    "ok": "#0ca30c",
    "info": "#0ca30c",
    "warning": "#fab219",
    "error": "#d03b3b",
    "failed": "#d03b3b",
}

STATUS_TEXT = {
    "idle": "NOT ANALYSED",
    "running": "ANALYSING",
    "ok": "OK",
    "info": "OK",
    "warning": "CHECK WARNINGS",
    "error": "ERROR",
    "failed": "FAILED",
}


def build_qss(t: Theme) -> str:
    """The application stylesheet for a theme."""
    return f"""
QMainWindow, QDialog, QWidget {{
    background: {t.surface};
    color: {t.ink};
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}}
QToolBar {{
    background: {t.panel};
    border-bottom: 1px solid {t.border};
    spacing: 6px;
    padding: 4px 8px;
}}
QToolBar QToolButton {{
    background: {t.panel_raised};
    border: 1px solid {t.border};
    border-radius: 3px;
    padding: 6px 12px;
    color: {t.ink};
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QToolBar QToolButton:hover {{ border-color: {t.accent}; }}
QToolBar QToolButton:pressed {{ background: {t.accent_dim}; }}
QToolBar QToolButton:disabled {{ color: {t.muted}; }}
QToolBar QToolButton#primary {{ background: {t.accent_dim}; border-color: {t.accent}; }}
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    color: {t.ink_secondary};
    font-weight: 700;
    letter-spacing: 1px;
}}
QDockWidget::title {{
    background: {t.panel};
    border-bottom: 1px solid {t.border};
    padding: 5px 10px;
    text-align: left;
}}
QGroupBox {{
    border: 1px solid {t.border};
    border-radius: 3px;
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
    color: {t.ink_secondary};
    font-weight: 700;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QTreeWidget, QTableWidget, QTableView, QPlainTextEdit, QListWidget {{
    background: {t.plot_bg};
    border: 1px solid {t.border};
    gridline-color: {t.grid};
    alternate-background-color: {t.panel};
    selection-background-color: {t.accent_dim};
    selection-color: {t.ink};
}}
QHeaderView::section {{
    background: {t.panel};
    color: {t.ink_secondary};
    border: none;
    border-right: 1px solid {t.border};
    border-bottom: 1px solid {t.border};
    padding: 4px 6px;
    font-weight: 600;
}}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background: {t.plot_bg};
    border: 1px solid {t.border};
    border-radius: 2px;
    padding: 3px 6px;
    color: {t.ink};
    selection-background-color: {t.accent_dim};
}}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {{
    border-color: {t.accent};
}}
QComboBox QAbstractItemView {{
    background: {t.panel};
    border: 1px solid {t.border};
    selection-background-color: {t.accent_dim};
}}
QPushButton {{
    background: {t.panel_raised};
    border: 1px solid {t.border};
    border-radius: 3px;
    padding: 5px 12px;
    color: {t.ink};
}}
QPushButton:hover {{ border-color: {t.accent}; }}
QPushButton:pressed {{ background: {t.accent_dim}; }}
QPushButton:disabled {{ color: {t.muted}; }}
QPushButton#primary {{ background: {t.accent_dim}; border-color: {t.accent}; font-weight: 700; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {t.border};
    border-radius: 2px;
    background: {t.plot_bg};
}}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}
QStatusBar {{
    background: {t.panel};
    border-top: 1px solid {t.border};
    color: {t.ink_secondary};
}}
QProgressBar {{
    background: {t.plot_bg};
    border: 1px solid {t.border};
    border-radius: 2px;
    text-align: center;
    color: {t.ink};
    height: 10px;
}}
QProgressBar::chunk {{ background: {t.accent}; }}
QSplitter::handle {{ background: {t.border}; }}
QScrollBar:vertical {{ background: {t.panel}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 4px; min-height: 20px; }}
QScrollBar:horizontal {{ background: {t.panel}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {t.border}; border-radius: 4px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QTabWidget::pane {{ border: 1px solid {t.border}; }}
QTabBar::tab {{
    background: {t.panel};
    border: 1px solid {t.border};
    border-bottom: none;
    padding: 5px 12px;
    color: {t.ink_secondary};
}}
QTabBar::tab:selected {{ background: {t.panel_raised}; color: {t.ink}; }}
QLabel#readout, QPlainTextEdit#readout {{
    font-family: "SF Mono", Menlo, Consolas, monospace;
    color: {t.ink};
}}
QLabel#cursor {{
    font-family: "SF Mono", Menlo, Consolas, monospace;
    color: {t.ink_secondary};
    padding: 0 8px;
}}
QLabel#caption {{ color: {t.muted}; font-size: 11px; }}
QLabel#update {{
    padding: 0 10px;
    font-weight: 600;
}}
QMenuBar {{ background: {t.panel}; border-bottom: 1px solid {t.border}; }}
QMenuBar::item:selected {{ background: {t.accent_dim}; }}
QMenu {{ background: {t.panel}; border: 1px solid {t.border}; }}
QMenu::item:selected {{ background: {t.accent_dim}; }}
QToolTip {{ background: {t.panel_raised}; color: {t.ink}; border: 1px solid {t.border}; }}
"""


QSS = build_qss(DARK)

_current: Theme = DARK


def current() -> Theme:
    """The theme applied last (dark until :func:`apply_theme` is called)."""
    return _current


def apply_theme(app: QApplication, name: str = DEFAULT_THEME) -> Theme:
    """Apply a theme to the application; widgets restyle through the stylesheet.

    Items that paint their own colours (the plot) ask :func:`current` or
    :func:`plot_style` afterwards.
    """
    import pyqtgraph as pg

    global _current
    theme = THEMES.get(name, DARK)
    app.setStyle("Fusion")
    app.setStyleSheet(build_qss(theme))
    pg.setConfigOptions(background=theme.plot_bg, foreground=theme.ink_secondary, antialias=True)
    _current = theme
    return theme


# --------------------------------------------------------------------- plot styles
@dataclass(frozen=True)
class PlotStyle:
    """How the live spectrum view draws itself."""

    name: str
    canvas: str
    """Background of the whole graph area (canvas and margins)."""
    frame: str
    """Axis lines and ticks."""
    text: str
    """Tick labels, axis titles, legend text, plot title."""
    font_family: str
    tick_font_pt: int
    label_font_pt: int
    ticks_inward: bool
    box: bool
    """Axes on all four sides (top and right without labels)."""
    grid: bool
    raw: str
    """Pen of the open data symbols."""
    envelope: str
    background: str
    residual: str
    sigma_fill: RGBA
    series: tuple[str, ...]
    fill_alpha: int
    """Opacity (0-255) of the area between a component and the background."""
    label_fill: RGBA
    legend_pen: str | None
    legend_fill: RGBA
    crosshair: str
    envelope_width: float
    symbol_size: int


ORIGIN = PlotStyle(
    name="origin",
    canvas="#ffffff",
    frame="#000000",
    text="#000000",
    font_family="Arial",
    tick_font_pt=10,
    label_font_pt=12,
    ticks_inward=True,
    box=True,
    grid=False,
    raw="#000000",
    envelope="#d62728",
    background="#7f7f7f",
    residual="#000000",
    sigma_fill=(190, 190, 190, 110),
    series=tuple(SERIES_LIGHT),
    fill_alpha=70,
    label_fill=(255, 255, 255, 205),
    legend_pen="#000000",
    legend_fill=(255, 255, 255, 255),
    crosshair="#b0b0b0",
    envelope_width=1.6,
    symbol_size=5,
)

PLOT_STYLE_NAMES = ("origin", "instrument")
DEFAULT_PLOT_STYLE = "origin"


def instrument_style(theme: Theme) -> PlotStyle:
    """The plot style that follows a theme's canvas, grid and series colours."""
    return PlotStyle(
        name="instrument",
        canvas=theme.plot_bg,
        frame=theme.grid if theme.name == "dark" else theme.border,
        text=theme.ink_secondary,
        font_family="",
        tick_font_pt=0,
        label_font_pt=0,
        ticks_inward=False,
        box=False,
        grid=True,
        raw=theme.muted,
        envelope=theme.ink,
        background=theme.muted,
        residual=theme.ink_secondary,
        sigma_fill=(*_rgb(theme.grid), 160),
        series=theme.series,
        fill_alpha=45,
        label_fill=theme.label_fill,
        legend_pen=None,
        legend_fill=(0, 0, 0, 0),
        crosshair=theme.grid,
        envelope_width=2.0,
        symbol_size=4,
    )


def plot_style(name: str = DEFAULT_PLOT_STYLE, theme: Theme | None = None) -> PlotStyle:
    """``"origin"`` (publication look) or ``"instrument"`` (follows the theme)."""
    if name == "instrument":
        return instrument_style(theme or current())
    return ORIGIN


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba(hex_color: str, alpha: int) -> RGBA:
    return (*_rgb(hex_color), alpha)


def describe(style: PlotStyle) -> dict[str, Any]:
    """A plain dictionary of the style (for logs and tests)."""
    return {
        "name": style.name,
        "canvas": style.canvas,
        "box": style.box,
        "ticks_inward": style.ticks_inward,
        "grid": style.grid,
        "font_family": style.font_family,
    }

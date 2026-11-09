"""Origin-style worksheet view.

Columns are named ``A(X)``, ``B(Y)``, ``C(yEr)``... and the first three rows are the
Long Name, Units and Comments header rows, as in an Origin workbook.  The model reads
the column arrays lazily, so a 120,000-point scan opens instantly.  ``Copy`` puts the
selection - or the whole sheet with its three header rows - on the clipboard as
tab-separated text that pastes straight into an Origin worksheet or Excel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QAbstractItemView, QApplication, QTableView, QWidget

from ..export.origin import HEADER_ROWS, Column, column_letters, n_rows

Index = QModelIndex | QPersistentModelIndex


def _text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float | np.floating):
        return "" if not np.isfinite(v) else f"{float(v):.7g}"
    if isinstance(v, np.integer | np.bool_):
        return str(int(v))
    return str(v)


class WorksheetModel(QAbstractTableModel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.columns: list[Column] = []
        self.title = ""
        self._rows = 0
        self.header_brush = QBrush(QColor("#2a2a28"))

    def set_columns(self, cols: Sequence[Column], title: str = "") -> None:
        self.beginResetModel()
        self.columns = list(cols)
        self.title = title
        self._rows = n_rows(self.columns)
        self.endResetModel()

    def rowCount(self, parent: Index = QModelIndex()) -> int:  # noqa: N802, B008 - Qt API
        return 0 if parent.isValid() or not self.columns else self._rows + len(HEADER_ROWS)

    def columnCount(self, parent: Index = QModelIndex()) -> int:  # noqa: N802, B008 - Qt API
        return 0 if parent.isValid() else len(self.columns)

    def cell(self, row: int, col: int) -> str:
        c = self.columns[col]
        if row < len(HEADER_ROWS):
            return (c.long_name, c.units, c.comments)[row]
        return _text(c.cell(row - len(HEADER_ROWS)))

    def data(self, index: Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return self.cell(r, c)
        if role == Qt.ItemDataRole.BackgroundRole and r < len(HEADER_ROWS):
            return self.header_brush
        if role == Qt.ItemDataRole.FontRole and r < len(HEADER_ROWS):
            f = QFont()
            f.setBold(r == 0)
            f.setItalic(r == 2)
            return f
        if role == Qt.ItemDataRole.TextAlignmentRole and r >= len(HEADER_ROWS):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(  # noqa: N802 - Qt API
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            letters = column_letters(len(self.columns))
            if 0 <= section < len(self.columns):
                return f"{letters[section]}({self.columns[section].designation})"
            return None
        if section < len(HEADER_ROWS):
            return HEADER_ROWS[section]
        return str(section - len(HEADER_ROWS) + 1)

    def as_tsv(self, rows: Sequence[int] | None = None, cols: Sequence[int] | None = None) -> str:
        rr = range(self.rowCount()) if rows is None else rows
        cc = range(self.columnCount()) if cols is None else cols
        return "\n".join("\t".join(self.cell(r, c) for c in cc) for r in rr) + "\n"


class WorksheetView(QTableView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sheet = WorksheetModel(self)
        self.setModel(self.sheet)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setDefaultSectionSize(20)
        self.horizontalHeader().setDefaultSectionSize(118)
        self._copy = QShortcut(QKeySequence.StandardKey.Copy, self, self.copy_selection)

    def set_columns(self, cols: Sequence[Column], title: str = "") -> None:
        self.sheet.set_columns(cols, title)

    def set_header_color(self, color: str) -> None:
        self.sheet.header_brush = QBrush(QColor(color))

    def copy_selection(self) -> str:
        """Selected cells (or the whole sheet) as tab-separated text on the clipboard."""
        idx = self.selectedIndexes()
        if idx:
            rows = sorted({i.row() for i in idx})
            cols = sorted({i.column() for i in idx})
            text = self.sheet.as_tsv(rows, cols)
        else:
            text = self.sheet.as_tsv()
        QApplication.clipboard().setText(text)
        return text


__all__ = ["WorksheetModel", "WorksheetView"]

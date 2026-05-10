from __future__ import annotations

from PyQt6 import QtWidgets

from app.core.diagnostics import collect_diagnostics
from app.core.paths import AppPaths
from app.engines.manager import EngineManager


class DiagnosticsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, paths: AppPaths, manager: EngineManager) -> None:
        super().__init__(parent)
        self.paths = paths
        self.manager = manager
        self.setWindowTitle("Diagnostyka")
        self.resize(860, 560)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Element", "Status", "Szczegoly"])
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        buttons = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Odswiez")
        refresh_btn.clicked.connect(self.refresh)
        close_btn = QtWidgets.QPushButton("Zamknij")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(refresh_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

    def refresh(self) -> None:
        self.table.setRowCount(0)
        for name, status, detail in collect_diagnostics(self.paths, self.manager):
            self._add_row(name, status, detail)
        self.table.resizeRowsToContents()

    def _add_row(self, name: str, status: str, detail: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, value in enumerate((name, status, diagnostic_table_text(detail))):
            item = QtWidgets.QTableWidgetItem(value)
            if col == 2:
                item.setToolTip(detail)
            self.table.setItem(row, col, item)


def show_diagnostics(parent: QtWidgets.QWidget, paths: AppPaths, manager: EngineManager) -> None:
    dialog = DiagnosticsDialog(parent, paths, manager)
    dialog.exec()


def diagnostic_table_text(text: str, limit: int = 140) -> str:
    compacted = " ".join(str(text or "").split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: max(0, int(limit) - 3)].rstrip() + "..."

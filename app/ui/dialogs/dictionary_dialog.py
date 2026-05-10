from __future__ import annotations

import json
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from app.core.dictionary import sanitize_dictionary


class DictionaryTable(QtWidgets.QTableWidget):
    def keyPressEvent(self, event):  # noqa: N802
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self._paste()
            event.accept()
            return
        super().keyPressEvent(event)

    def _paste(self) -> None:
        text = (QtWidgets.QApplication.clipboard().text() or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return
        row = max(0, self.currentRow())
        col = min(max(0, self.currentColumn()), 1)
        for row_offset, line in enumerate(text.split("\n")):
            if line == "":
                continue
            target_row = row + row_offset
            while target_row >= self.rowCount():
                self.insertRow(self.rowCount())
            for col_offset, value in enumerate(line.split("\t")[: 2 - col]):
                item = self.item(target_row, col + col_offset)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.setItem(target_row, col + col_offset, item)
                item.setText(value.strip())


class DictionaryDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, engine_name: str, data: dict[str, str]) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Slownik - {engine_name}")
        self.resize(720, 520)
        self._build_ui()
        self._load(data)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        search_row = QtWidgets.QHBoxLayout()
        search_row.addWidget(QtWidgets.QLabel("Szukaj:"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("np. bat")
        self.search.textChanged.connect(self._filter)
        search_row.addWidget(self.search, 1)
        root.addLayout(search_row)

        self.table = DictionaryTable(0, 2)
        self.table.setHorizontalHeaderLabels(["Oryginal", "Zamiana"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        root.addWidget(self.table, 1)

        actions = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Dodaj")
        add_btn.clicked.connect(self._add_row)
        remove_btn = QtWidgets.QPushButton("Usun zaznaczone")
        remove_btn.clicked.connect(self._remove_selected)
        load_btn = QtWidgets.QPushButton("Wczytaj z pliku")
        load_btn.clicked.connect(self._load_from_file)
        save_as_btn = QtWidgets.QPushButton("Zapisz jako...")
        save_as_btn.clicked.connect(self._save_as_file)
        clear_btn = QtWidgets.QPushButton("Wyczysc slownik")
        clear_btn.clicked.connect(self._clear_dictionary)
        actions.addWidget(add_btn)
        actions.addWidget(remove_btn)
        actions.addWidget(load_btn)
        actions.addWidget(save_as_btn)
        actions.addWidget(clear_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Anuluj")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QtWidgets.QPushButton("Zapisz")
        save_btn.clicked.connect(self.accept)
        footer.addWidget(cancel_btn)
        footer.addWidget(save_btn)
        root.addLayout(footer)

    def _load(self, data: dict[str, str]) -> None:
        for key, value in sorted(data.items(), key=lambda item: item[0].lower()):
            self._add_row(key, value)
        self._filter()

    def _add_row(self, key: str = "", value: str = "") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))
        self.table.scrollToBottom()
        self._filter()

    def _filter(self) -> None:
        needle = self.search.text().strip()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            key = (item.text() if item is not None else "").strip()
            self.table.setRowHidden(row, not should_show_dictionary_row(key, needle))

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _load_from_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Wczytaj slownik",
            "",
            "Slownik JSON (*.json);;Wszystkie pliki (*)",
        )
        if not path:
            return
        try:
            data, count = load_dictionary_external_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Blad slownika", f"Nie udalo sie wczytac slownika:\n{exc}")
            return
        self.table.setRowCount(0)
        self._load(data)
        QtWidgets.QMessageBox.information(self, "Slownik", f"Wczytano wpisy: {count}")

    def _save_as_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Zapisz slownik jako",
            "dictionary.json",
            "Slownik JSON (*.json);;Wszystkie pliki (*)",
        )
        if not path:
            return
        save_path = Path(path)
        if save_path.suffix.lower() != ".json":
            save_path = save_path.with_suffix(".json")
        try:
            count, skipped = save_dictionary_external_file(save_path, self.read_data())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Blad slownika", f"Nie udalo sie zapisac slownika:\n{exc}")
            return
        suffix = f", pominieto: {skipped}" if skipped else ""
        QtWidgets.QMessageBox.information(self, "Slownik", f"Zapisano wpisy: {count}{suffix}")

    def _clear_dictionary(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Wyczysc slownik",
            "Wyczysc wszystkie wpisy w oknie slownika?\nZmiana trafi do aktywnego slownika dopiero po kliknieciu Zapisz.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.table.setRowCount(0)
        self._add_row()
        self._filter()

    def read_data(self) -> dict[str, str]:
        data: dict[str, str] = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            key = (key_item.text() if key_item is not None else "").strip()
            value = (value_item.text() if value_item is not None else "").strip()
            if key:
                data[key] = value
        return data


def edit_dictionary(
    parent: QtWidgets.QWidget | None,
    engine_name: str,
    data: dict[str, str],
) -> tuple[bool, dict[str, str]]:
    dialog = DictionaryDialog(parent, engine_name, data)
    ok = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    if not ok:
        return False, data
    return True, dialog.read_data()


def should_show_dictionary_row(key: str, needle: str) -> bool:
    normalized_needle = str(needle or "").strip().casefold()
    if not normalized_needle:
        return True
    normalized_key = str(key or "").strip().casefold()
    if not normalized_key:
        return True
    return normalized_key.startswith(normalized_needle)


def load_dictionary_external_file(path: Path) -> tuple[dict[str, str], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Plik slownika musi zawierac obiekt JSON.")
    sanitized, _skipped = sanitize_dictionary({str(key): str(value) for key, value in data.items()})
    return sanitized, len(sanitized)


def save_dictionary_external_file(path: Path, data: dict[str, str]) -> tuple[int, int]:
    sanitized, skipped = sanitize_dictionary(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(sanitized), skipped

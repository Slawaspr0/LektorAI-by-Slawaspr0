from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets

from app.core.log_cleanup import LOG_CLEANUP_OPTIONS, cleanup_logs, preview_log_cleanup
from app.core.paths import AppPaths


class LogCleanupDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, paths: AppPaths, active_app_log_path: Path | None = None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.active_app_log_path = active_app_log_path
        self.checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self.setWindowTitle("Czyszczenie logow aplikacji")
        self.resize(520, 240)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        group = QtWidgets.QGroupBox("Co wyczyscic")
        group_layout = QtWidgets.QVBoxLayout(group)
        for option in LOG_CLEANUP_OPTIONS:
            checkbox = QtWidgets.QCheckBox(option.label)
            self.checkboxes[option.option_id] = checkbox
            group_layout.addWidget(checkbox)
        root.addWidget(group)

        self.status_label = QtWidgets.QLabel("Gotowe")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Odswiez")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_clear_selected = QtWidgets.QPushButton("Wyczysc zaznaczone")
        self.btn_clear_selected.clicked.connect(self.clear_selected)
        self.btn_clear_all = QtWidgets.QPushButton("Wyczysc wszystko")
        self.btn_clear_all.clicked.connect(self.clear_all)
        self.btn_close = QtWidgets.QPushButton("Zamknij")
        self.btn_close.clicked.connect(self.accept)
        actions.addWidget(self.btn_refresh)
        actions.addStretch(1)
        actions.addWidget(self.btn_clear_selected)
        actions.addWidget(self.btn_clear_all)
        actions.addWidget(self.btn_close)
        root.addLayout(actions)

    def refresh(self) -> None:
        counts = preview_log_cleanup(self.paths, active_app_log_path=self.active_app_log_path)
        for option in LOG_CLEANUP_OPTIONS:
            checkbox = self.checkboxes[option.option_id]
            count = int(counts.get(option.option_id, 0))
            checkbox.setText(f"{option.label} ({count})")
            checkbox.setEnabled(count > 0)
            if count <= 0:
                checkbox.setChecked(False)
        total = sum(int(value) for value in counts.values())
        self.status_label.setText(f"Do wyczyszczenia: {total}")

    def clear_selected(self) -> None:
        selected = {option_id for option_id, checkbox in self.checkboxes.items() if checkbox.isChecked() and checkbox.isEnabled()}
        if not selected:
            self.status_label.setText("Nie wybrano nic do czyszczenia.")
            return
        self._clear(selected)

    def clear_all(self) -> None:
        selected = {option_id for option_id, checkbox in self.checkboxes.items() if checkbox.isEnabled()}
        if not selected:
            self.status_label.setText("Nie ma nic do czyszczenia.")
            return
        self._clear(selected)

    def _clear(self, selected: set[str]) -> None:
        result = cleanup_logs(self.paths, selected, active_app_log_path=self.active_app_log_path)
        if result.errors:
            message = (
                f"Usunieto plikow: {result.files_removed}, folderow: {result.dirs_removed}. "
                f"Bledy: {len(result.errors)}"
            )
            QtWidgets.QMessageBox.warning(self, "Czyszczenie logow", "\n".join(result.errors[:8]))
        else:
            message = f"Usunieto plikow: {result.files_removed}, folderow: {result.dirs_removed}."
        self.refresh()
        self.status_label.setText(message)


def show_log_cleanup(parent: QtWidgets.QWidget, paths: AppPaths, active_app_log_path: Path | None = None) -> None:
    dialog = LogCleanupDialog(parent, paths, active_app_log_path)
    dialog.exec()

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from app.engines.manager import EngineManager, EngineRemovalError
from app.engines.schemas import EngineKind, EngineState, EngineStatus
from app.engines.status import format_engine_state
from app.ui.dialogs.scrollable_text_dialog import confirm_scrollable_text, show_scrollable_text


def should_enable_keep_settings_remove(is_local: bool, has_removable_payload: bool, busy: bool) -> bool:
    return is_local and has_removable_payload and not busy


def clear_engine_status_cache(manager: EngineManager) -> None:
    manager.clear_package_check_cache()


def initial_install_button_label() -> str:
    return "Zainstaluj"


class InstallWorker(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(str)

    def __init__(self, manager: EngineManager, engine_id: str, torch_variant: str | None = None) -> None:
        super().__init__()
        self.manager = manager
        self.engine_id = engine_id
        self.torch_variant = torch_variant

    def run(self) -> None:
        try:
            self.manager.install_local_engine(self.engine_id, self.message.emit, torch_variant=self.torch_variant)
            self.finished_ok.emit(f"TTS {self.engine_id}: gotowy")
        except Exception as exc:
            self.failed.emit(f"TTS {self.engine_id}: BLAD instalacji: {exc}")


class TTSManagerDialog(QtWidgets.QDialog):
    changed = QtCore.pyqtSignal()
    message = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None, manager: EngineManager) -> None:
        super().__init__(parent)
        self.manager = manager
        self.states: list[EngineState] = []
        self.install_worker: InstallWorker | None = None
        self.setWindowTitle("Menadzer TTS")
        self.resize(820, 520)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Grupa", "Silnik", "Status", "Komponenty", "Uwagi"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        root.addWidget(self.table, 1)

        self.status_label = QtWidgets.QLabel("Gotowe")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        self.btn_install = QtWidgets.QPushButton(initial_install_button_label())
        self.btn_install.clicked.connect(self.prepare_selected)
        self.btn_preview = QtWidgets.QPushButton("Wymagania")
        self.btn_preview.clicked.connect(self.show_install_preview)
        self.btn_keep = QtWidgets.QPushButton("Usun model TTS, zostaw ustawienia")
        self.btn_keep.clicked.connect(self.remove_keep_settings)
        self.btn_remove = QtWidgets.QPushButton("Usun calkowicie")
        self.btn_remove.clicked.connect(self.remove_completely)
        self.btn_refresh = QtWidgets.QPushButton("Odswiez")
        self.btn_refresh.clicked.connect(self.refresh)
        actions.addWidget(self.btn_install)
        actions.addWidget(self.btn_preview)
        actions.addWidget(self.btn_keep)
        actions.addWidget(self.btn_remove)
        actions.addStretch(1)
        actions.addWidget(self.btn_refresh)
        root.addLayout(actions)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        close_btn = QtWidgets.QPushButton("Zamknij")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def refresh(self) -> None:
        clear_engine_status_cache(self.manager)
        self.states = self.manager.list_states()
        self.table.setRowCount(0)
        for state in self.states:
            row = self.table.rowCount()
            self.table.insertRow(row)
            group = "Lokalne" if state.definition.kind == EngineKind.LOCAL else "Internetowe"
            values = [
                group,
                state.definition.display_name,
                state.status.value,
                ", ".join(state.components),
                state.reason,
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, state.definition.engine_id)
                self.table.setItem(row, col, item)
        self._sync_buttons()

    def closeEvent(self, event):  # noqa: N802
        if self.install_worker is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Instalacja trwa",
                "Poczekaj na zakonczenie instalacji TTS. Szczegoly sa zapisywane w install.log.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def selected_state(self) -> EngineState | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        engine_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        return next((state for state in self.states if state.definition.engine_id == engine_id), None)

    def _sync_buttons(self) -> None:
        state = self.selected_state()
        is_local = bool(state and state.definition.kind == EngineKind.LOCAL)
        busy = self.install_worker is not None
        has_engine_dir = bool(state and self.manager.engine_dir_exists(state.definition.engine_id))
        has_removable_payload = bool(state and self.manager.removable_payload_exists(state.definition.engine_id))
        self.btn_install.setText(self._install_button_text(state))
        self.btn_install.setEnabled(is_local and not busy)
        self.btn_preview.setEnabled(is_local and not busy)
        self.btn_keep.setEnabled(should_enable_keep_settings_remove(is_local, has_removable_payload, busy))
        self.btn_remove.setEnabled(is_local and has_engine_dir and not busy)
        self.btn_refresh.setEnabled(not busy)

    def _install_button_text(self, state: EngineState | None) -> str:
        if state is None or state.definition.kind != EngineKind.LOCAL:
            return initial_install_button_label()
        if state.status == EngineStatus.BROKEN:
            return "Napraw instalacje"
        if state.status in {EngineStatus.READY, EngineStatus.INSTALLED_NO_MODEL}:
            return "Przeinstaluj / aktualizuj"
        return initial_install_button_label()

    def show_install_preview(self) -> None:
        state = self.selected_state()
        if not state:
            return
        variants = self.manager.local_install_variants(state.definition.engine_id)
        if variants:
            sections: list[str] = []
            for idx, variant in enumerate(variants, start=1):
                details = "\n".join(self.manager.local_install_preview(state.definition.engine_id, torch_variant=variant.variant_id))
                sections.append(
                    "\n".join(
                        [
                            f"WARIANT {idx}: {variant.label}",
                            variant.description,
                            "-" * 72,
                            details,
                        ]
                    )
                )
            text = ("\n\n" + "=" * 72 + "\n\n").join(sections)
        else:
            text = "\n".join(self.manager.local_install_preview(state.definition.engine_id))
        show_scrollable_text(self, "Wymagania TTS", "Plan instalacji lokalnego silnika TTS:", text)

    def prepare_selected(self) -> None:
        state = self.selected_state()
        if not state:
            return
        selected_variant = self.choose_torch_variant(state)
        if selected_variant is None and self.manager.local_install_variants(state.definition.engine_id):
            return
        plan = "\n".join(self.manager.local_install_preview(state.definition.engine_id, torch_variant=selected_variant))
        reinstall_runtime = bool(self.manager.local_runtime_exists(state.definition.engine_id) and self.manager.local_install_variants(state.definition.engine_id))
        prompt = "Rozpoczac instalacje lokalnego TTS? To moze potrwac dlugo i pobrac duzo danych."
        if reinstall_runtime:
            prompt = (
                "Przeinstalowac lokalny TTS? Program utworzy czyste srodowisko dla wybranego wariantu PyTorch, "
                "zostawiajac Twoje ustawienia i slownik."
            )
        reply = confirm_scrollable_text(
            self,
            "Instalacja TTS",
            prompt,
            plan,
        )
        if not reply:
            return
        if reinstall_runtime:
            try:
                self.manager.remove_engine_keep_user_settings(state.definition.engine_id)
                self.message.emit(f"TTS {state.definition.engine_id}: usunieto stare srodowisko, ustawienia zostaja")
            except EngineRemovalError as exc:
                self.status_label.setText(str(exc))
                self.message.emit(f"TTS {state.definition.engine_id}: BLAD przygotowania reinstalacji - {exc}")
                QtWidgets.QMessageBox.warning(self, "Nie mozna przeinstalowac TTS", str(exc))
                self.refresh()
                return
        self.install_worker = InstallWorker(self.manager, state.definition.engine_id, torch_variant=selected_variant)
        self.status_label.setText(f"TTS {state.definition.engine_id}: instalacja rozpoczeta")
        self.install_worker.message.connect(self._install_message)
        self.install_worker.failed.connect(self._install_failed)
        self.install_worker.finished_ok.connect(self._install_finished)
        self._sync_buttons()
        self.install_worker.start()

    def choose_torch_variant(self, state: EngineState) -> str | None:
        variants = self.manager.local_install_variants(state.definition.engine_id)
        if not variants:
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Wybierz PyTorch")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        label = QtWidgets.QLabel("Wybierz wariant PyTorch dla tego silnika TTS:")
        label.setWordWrap(True)
        layout.addWidget(label)

        group = QtWidgets.QButtonGroup(dialog)
        buttons: list[QtWidgets.QRadioButton] = []
        for idx, variant in enumerate(variants):
            button = QtWidgets.QRadioButton(f"{variant.label}\n{variant.description}")
            button.setChecked(idx == 0)
            button.setProperty("variant_id", variant.variant_id)
            button.setMinimumHeight(44)
            group.addButton(button)
            buttons.append(button)
            layout.addWidget(button)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Anuluj")
        ok_btn = QtWidgets.QPushButton("Dalej")
        cancel_btn.clicked.connect(dialog.reject)
        ok_btn.clicked.connect(dialog.accept)
        actions.addWidget(cancel_btn)
        actions.addWidget(ok_btn)
        layout.addLayout(actions)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        selected = group.checkedButton()
        if selected is None:
            return variants[0].variant_id
        return str(selected.property("variant_id") or variants[0].variant_id)

    def remove_keep_settings(self) -> None:
        state = self.selected_state()
        if not state:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Usun TTS",
            f"Usunac model/runtime {state.definition.display_name}, zostawiajac tylko config.json i dictionary.json?",
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.manager.remove_engine_keep_user_settings(state.definition.engine_id)
        except EngineRemovalError as exc:
            self.status_label.setText(str(exc))
            self.message.emit(f"TTS {state.definition.engine_id}: BLAD usuwania - {exc}")
            QtWidgets.QMessageBox.warning(self, "Nie mozna usunac TTS", str(exc))
            self.refresh()
            return
        self.message.emit(f"TTS {state.definition.engine_id}: usunieto runtime, zostawiono ustawienia")
        self.refresh()
        self.changed.emit()

    def remove_completely(self) -> None:
        state = self.selected_state()
        if not state:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Usun calkowicie",
            f"Usunac calkowicie {state.definition.display_name} wraz z ustawieniami i slownikiem?",
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.manager.remove_engine_completely(state.definition.engine_id)
        except EngineRemovalError as exc:
            self.status_label.setText(str(exc))
            self.message.emit(f"TTS {state.definition.engine_id}: BLAD usuwania - {exc}")
            QtWidgets.QMessageBox.warning(self, "Nie mozna usunac TTS", str(exc))
            self.refresh()
            return
        self.message.emit(f"TTS {state.definition.engine_id}: usunieto calkowicie")
        self.refresh()
        self.changed.emit()

    def _install_failed(self, message: str) -> None:
        self.status_label.setText(message)
        self.message.emit(message)
        self.install_worker = None
        self.refresh()
        self.changed.emit()

    def _install_finished(self, message: str) -> None:
        self.status_label.setText(message)
        self.message.emit(message)
        self.install_worker = None
        self.refresh()
        self.changed.emit()

    def _install_message(self, message: str) -> None:
        self.status_label.setText(message)
        self.message.emit(message)

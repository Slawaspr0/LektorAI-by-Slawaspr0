from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path
from time import monotonic

from PyQt6 import QtCore, QtGui, QtWidgets

from app.core.config import AppConfigStore
from app.core.dictionary import load_dictionary, save_dictionary
from app.core.logging import setup_app_logger
from app.core.media_tools import (
    BINARY_LOOKUP_HINT,
    DEFAULT_AAC_BITRATE,
    DEFAULT_BACKGROUND_LUFS,
    DEFAULT_BACKGROUND_WEIGHT,
    DEFAULT_LEKTOR_LUFS,
    DEFAULT_LEKTOR_DELAY_MS,
    DEFAULT_LEKTOR_WEIGHT,
    LEKTOR_DELAY_STEP_MS,
    MAX_LEKTOR_DELAY_MS,
    MIN_LEKTOR_DELAY_MS,
    VIDEO_EXTENSIONS,
    find_ffmpeg,
    find_mkvmerge,
    find_ffprobe,
    is_video_file,
    sanitize_aac_bitrate,
    sanitize_audio_weight,
    sanitize_lektor_delay_ms,
    sanitize_lufs,
)
from app.core.paths import AppPaths, build_paths
from app.core.version import APP_NAME, APP_VERSION
from app.engines.config_validation import validate_engine_config
from app.engines.manager import EngineManager
from app.engines.schemas import EngineKind, EngineState
from app.engines.status import format_engine_state
from app.pipeline.subtitles import SUPPORTED_SUBTITLE_EXTENSIONS
from app.pipeline.progress import (
    FILE_PROGRESS_TOTAL,
    decode_progress_marker,
    format_progress_status,
    progress_value_for_stage,
    safe_unit_eta_seconds,
)
from app.pipeline.tts_job import run_tts_job
from app.ui.dialogs.diagnostics_dialog import show_diagnostics
from app.ui.dialogs.dictionary_dialog import edit_dictionary
from app.ui.dialogs.scrollable_text_dialog import confirm_scrollable_text, show_scrollable_text
from app.ui.dialogs.settings_dialog import edit_engine_settings
from app.ui.dialogs.tts_manager_dialog import TTSManagerDialog


def progress_bar_style() -> str:
    return """
            QProgressBar {
                border: 1px solid #7a7f8a;
                border-radius: 4px;
                text-align: center;
                color: #111111;
                background: #f1f3f5;
                min-height: 22px;
            }
            QProgressBar::chunk {
                background-color: #12d979;
                border-radius: 3px;
            }
            """


def compact_app_log_message(message: str, limit: int = 260) -> str:
    text = " ".join(str(message or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * max(0, limit)
    return text[: limit - 3].rstrip() + "..."


def clear_engine_status_cache(manager: EngineManager) -> None:
    manager.clear_package_check_cache()


def should_enable_start_button(engine_id: str, controls_enabled: bool, engine_selectable: bool = True) -> bool:
    return bool(str(engine_id or "").strip()) and controls_enabled and engine_selectable


def should_enable_engine_actions(engine_id: str, engine_selectable: bool, controls_enabled: bool) -> bool:
    return bool(str(engine_id or "").strip()) and engine_selectable and controls_enabled


def stored_engine_after_selection(engine_id: str, engine_selectable: bool) -> str:
    normalized = str(engine_id or "").strip()
    return normalized if normalized and engine_selectable else ""


def bool_config_value(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "tak", "on"}:
            return True
        if normalized in {"0", "false", "no", "nie", "off"}:
            return False
    if isinstance(value, int) and not isinstance(value, bool):
        if value in {0, 1}:
            return bool(value)
    return default


def engine_combo_label(group_name: str, state: EngineState) -> str:
    if state.definition.kind == EngineKind.INTERNET:
        return f"{group_name}: {state.definition.display_name}"
    return f"{group_name}: {format_engine_state(state)}"


def aac_quality_options() -> tuple[str, ...]:
    return ("192k", "256k", "320k", "384k", "448k", "640k")


def aac_quality_label(value: str) -> str:
    bitrate = sanitize_aac_bitrate(value)
    return f"{bitrate[:-1]} kb/s"


def main_window_refresh_keeps_engine_signals_blocked() -> bool:
    return True


def main_window_minimum_size() -> tuple[int, int]:
    return (980, 620)


def slider_value_to_weight(value: int) -> float:
    return sanitize_audio_weight(float(value) / 10.0, DEFAULT_BACKGROUND_WEIGHT)


def weight_to_slider_value(value: float) -> int:
    return int(round(sanitize_audio_weight(value, DEFAULT_BACKGROUND_WEIGHT) * 10))


def audio_defaults_summary() -> dict[str, object]:
    return {
        "lektor_lufs": DEFAULT_LEKTOR_LUFS,
        "lektor_weight": DEFAULT_LEKTOR_WEIGHT,
        "background_lufs": DEFAULT_BACKGROUND_LUFS,
        "background_weight": DEFAULT_BACKGROUND_WEIGHT,
        "aac_bitrate": DEFAULT_AAC_BITRATE,
        "lektor_delay_ms": DEFAULT_LEKTOR_DELAY_MS,
        "create_stereo_for_surround": True,
    }


def format_lektor_delay_label(value: int) -> str:
    value = sanitize_lektor_delay_ms(value)
    if value > 0:
        return f"+{value} ms"
    return f"{value} ms"


class QueueListWidget(QtWidgets.QListWidget):
    files_dropped = QtCore.pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class PipelineWorker(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    diagnostic = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    tts_progress = QtCore.pyqtSignal(int, int, str)
    output_ready = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(str)

    def __init__(
        self,
        files: list[Path],
        engine_id: str,
        manager: EngineManager,
        aac_bitrate: str,
        lektor_lufs: int,
        lektor_weight: float,
        background_lufs: int,
        background_weight: float,
        lektor_delay_ms: int,
        create_stereo_for_surround: bool,
    ) -> None:
        super().__init__()
        self.files = files
        self.engine_id = engine_id
        self.manager = manager
        self.aac_bitrate = sanitize_aac_bitrate(aac_bitrate)
        self.lektor_lufs = sanitize_lufs(lektor_lufs, DEFAULT_LEKTOR_LUFS)
        self.lektor_weight = sanitize_audio_weight(lektor_weight, DEFAULT_LEKTOR_WEIGHT)
        self.background_lufs = sanitize_lufs(background_lufs, DEFAULT_BACKGROUND_LUFS)
        self.background_weight = sanitize_audio_weight(background_weight, DEFAULT_BACKGROUND_WEIGHT)
        self.lektor_delay_ms = sanitize_lektor_delay_ms(lektor_delay_ms)
        self.create_stereo_for_surround = bool(create_stereo_for_surround)
        self.file_started_at: float | None = None
        self.tts_started_at: float | None = None
        self.current_segment_total = 0

    def run(self) -> None:
        total = len(self.files)
        failed: list[tuple[str, str]] = []
        output_dirs: list[str] = []
        total_generation_seconds = 0.0
        try:
            for index, source_path in enumerate(self.files, 1):
                if self.isInterruptionRequested():
                    self.finished_ok.emit("Przerwano przez uzytkownika")
                    return
                self.progress.emit(index - 1, total)
                self.file_started_at = monotonic()
                self.tts_started_at = None
                self.current_segment_total = 0
                self.tts_progress.emit(progress_value_for_stage("prepare"), FILE_PROGRESS_TOTAL, f"Aktualny plik: przygotowanie - {source_path.name}")
                self.message.emit(f"Przetwarzanie: {source_path.name}")
                try:
                    result = run_tts_job(
                        source_path,
                        self.engine_id,
                        self.manager.paths,
                        self.manager,
                        self._progress_message,
                        aac_bitrate=self.aac_bitrate,
                        lektor_lufs=self.lektor_lufs,
                        lektor_weight=self.lektor_weight,
                        background_lufs=self.background_lufs,
                        background_weight=self.background_weight,
                        lektor_delay_ms=self.lektor_delay_ms,
                        create_stereo_for_surround=self.create_stereo_for_surround,
                        cancel_requested=self.isInterruptionRequested,
                    )
                except Exception as exc:
                    failed.append((source_path.name, str(exc)))
                    self.diagnostic.emit(
                        "BLAD pliku "
                        f"{source_path.name}\n"
                        f"Silnik: {self.engine_id}\n"
                        f"Typ: {type(exc).__name__}\n"
                        f"Komunikat: {exc}\n"
                        f"{traceback.format_exc()}"
                    )
                    self.message.emit(f"BLAD pliku {source_path.name}: {exc}")
                    self.progress.emit(index, total)
                    continue
                total_generation_seconds += float(result.generation_seconds)
                self.message.emit(f"{self.engine_id}: zapisano {result.segment_count} segmentow")
                self.message.emit(f"{self.engine_id}: czas generowania {format_duration(result.generation_seconds)}")
                if result.qc_warning_count:
                    self.message.emit(f"Audio QC: podejrzane segmenty: {result.qc_warning_count}")
                if result.subtitle_path.is_file():
                    self.message.emit(f"Napisy: {result.subtitle_path.name}")
                if result.lektor_m4a_path is not None:
                    self.message.emit(f"Audio: {result.lektor_m4a_path.name}")
                if result.output_video_path is not None:
                    self.message.emit(f"Wideo: {result.output_video_path.name}")
                workspace = str(result.workspace)
                output_dirs.append(workspace)
                self.output_ready.emit(workspace)
                self.progress.emit(index, total)
                self.tts_progress.emit(FILE_PROGRESS_TOTAL, FILE_PROGRESS_TOTAL, "Aktualny plik: gotowe")
                if self.isInterruptionRequested():
                    self.finished_ok.emit("Przerwano przez uzytkownika")
                    return
            done = total - len(failed)
            self.message.emit(f"Kolejka: gotowe {done}/{total}, bledy {len(failed)}, TTS {format_duration(total_generation_seconds)}")
            if output_dirs:
                unique_output_dirs = list(dict.fromkeys(output_dirs))
                self.message.emit(f"Wyniki: foldery {len(unique_output_dirs)}, ostatni {Path(unique_output_dirs[-1]).name}")
            if failed:
                for name, error in failed[:8]:
                    self.message.emit(f"Nieudany plik: {name} - {error}")
                if len(failed) > 8:
                    self.message.emit(f"Nieudane pliki: +{len(failed) - 8} wiecej")
                self.finished_ok.emit(f"Zakonczono z bledami: {len(failed)}")
            else:
                self.finished_ok.emit("Zakonczono")
        except Exception as exc:
            self.diagnostic.emit(
                "BLAD krytyczny pipeline\n"
                f"Silnik: {self.engine_id}\n"
                f"Typ: {type(exc).__name__}\n"
                f"Komunikat: {exc}\n"
                f"{traceback.format_exc()}"
            )
            self.failed.emit(str(exc))

    def _progress_message(self, message: str) -> None:
        marker = decode_progress_marker(message)
        if marker is not None:
            stage, ratio, label = marker
            value = progress_value_for_stage(stage, ratio)
            elapsed = max(0.0, monotonic() - self.file_started_at) if self.file_started_at is not None else None
            status = format_progress_status(label or "Aktualny plik", elapsed_seconds=elapsed)
            self.tts_progress.emit(value, FILE_PROGRESS_TOTAL, status)
            return
        self.message.emit(message)
        segment_total_match = re.search(r":\s+(\d+)\s+segmentow\b", message, flags=re.IGNORECASE)
        if segment_total_match:
            try:
                self.current_segment_total = int(segment_total_match.group(1))
            except ValueError:
                self.current_segment_total = 0
            self.tts_started_at = monotonic()
            self.tts_progress.emit(
                progress_value_for_stage("tts", 0.0),
                FILE_PROGRESS_TOTAL,
                format_progress_status("Generowanie TTS", f"0/{self.current_segment_total}"),
            )
            return
        match = re.search(r"segment\s+(\d+)\s*/\s*(\d+)", message, flags=re.IGNORECASE)
        if not match:
            return
        try:
            done = int(match.group(1))
            total = int(match.group(2))
        except ValueError:
            return
        if self.tts_started_at is None:
            self.tts_started_at = monotonic()
        elapsed = max(0.0, monotonic() - self.tts_started_at)
        ratio = done / max(1, total)
        eta = safe_unit_eta_seconds(done, total, elapsed)
        label = format_progress_status("Generowanie TTS", f"{done}/{total}", elapsed, eta)
        self.tts_progress.emit(progress_value_for_stage("tts", ratio), FILE_PROGRESS_TOTAL, label)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, paths: AppPaths, logger: logging.Logger, log_path: Path) -> None:
        super().__init__()
        self.paths = paths
        self.logger = logger
        self.log_path = log_path
        self.config_store = AppConfigStore(paths.config_path)
        self.config = self.config_store.load()
        self.engine_manager = EngineManager(paths)
        self.engine_states: list[EngineState] = []
        self.worker: PipelineWorker | None = None
        self.tts_started_at: float | None = None

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        min_width, min_height = main_window_minimum_size()
        self.setMinimumSize(min_width, min_height)
        self.statusBar().setSizeGripEnabled(True)
        self._build_ui()
        self._restore_window_state()
        self.log(f"Aplikacja uruchomiona: {APP_NAME} {APP_VERSION}")
        self.refresh_engines()

    def closeEvent(self, event):  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "Konwersja trwa",
                "Najpierw przerwij kolejke albo poczekaj na zakonczenie aktualnego pliku.",
            )
            event.ignore()
            return
        self._save_window_state()
        self.log("Aplikacja zamknieta")
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter)

        left_panel = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_panel)
        self.queue = QueueListWidget()
        self.queue.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.queue.files_dropped.connect(self.add_file_paths)
        self.queue_status = QtWidgets.QLabel("Pliki: 0 | wideo: 0 | napisy/tekst: 0")
        self.btn_add = QtWidgets.QPushButton("Dodaj pliki")
        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove_files = QtWidgets.QPushButton("Usun zaznaczone")
        self.btn_remove_files.clicked.connect(self.remove_selected_files)
        self.btn_clear_files = QtWidgets.QPushButton("Wyczysc")
        self.btn_clear_files.clicked.connect(self.clear_files)
        self.btn_move_up = QtWidgets.QPushButton("W gore")
        self.btn_move_up.clicked.connect(self.move_selected_up)
        self.btn_move_down = QtWidgets.QPushButton("W dol")
        self.btn_move_down.clicked.connect(self.move_selected_down)
        self.btn_sort_files = QtWidgets.QPushButton("Sortuj")
        self.btn_sort_files.clicked.connect(self.sort_queue)
        queue_actions_top = QtWidgets.QHBoxLayout()
        queue_actions_top.addWidget(self.btn_add)
        queue_actions_top.addWidget(self.btn_remove_files)
        queue_actions_top.addWidget(self.btn_clear_files)
        queue_actions_bottom = QtWidgets.QHBoxLayout()
        queue_actions_bottom.addWidget(self.btn_move_up)
        queue_actions_bottom.addWidget(self.btn_move_down)
        queue_actions_bottom.addWidget(self.btn_sort_files)
        left.addWidget(QtWidgets.QLabel("Kolejka"))
        left.addWidget(self.queue_status)
        left.addWidget(self.queue, 1)
        left.addLayout(queue_actions_top)
        left.addLayout(queue_actions_bottom)
        self.main_splitter.addWidget(left_panel)

        middle_panel = QtWidgets.QWidget()
        middle = QtWidgets.QVBoxLayout(middle_panel)
        self.engine_combo = QtWidgets.QComboBox()
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        self.btn_dictionary = QtWidgets.QPushButton("Slownik")
        self.btn_dictionary.setEnabled(False)
        self.btn_dictionary.clicked.connect(self.open_dictionary)
        self.btn_settings = QtWidgets.QPushButton("Ustawienia TTS")
        self.btn_settings.setEnabled(False)
        self.btn_settings.clicked.connect(self.open_engine_settings)
        self.btn_tts_manager = QtWidgets.QPushButton("Menadzer TTS")
        self.btn_tts_manager.clicked.connect(self.open_tts_manager)
        self.btn_diagnostics = QtWidgets.QPushButton("Diagnostyka")
        self.btn_diagnostics.clicked.connect(self.open_diagnostics)
        self.btn_open_output = QtWidgets.QPushButton("Otworz wynik")
        self.btn_open_output.setEnabled(False)
        self.btn_open_output.clicked.connect(self.open_last_output)
        self.last_output_dir: Path | None = None
        self.lektor_lufs_row, self.lektor_lufs_slider, self.lektor_lufs_value = self._create_slider_row(
            "Glosnosc lektora (LUFS)",
            -30,
            -8,
            self.config_store.lektor_lufs(),
            "Docelowa glosnosc finalnej sciezki lektora po normalizacji.",
            lambda value: f"{value} LUFS",
        )
        self.lektor_lufs_slider.valueChanged.connect(self._on_lektor_lufs_changed)
        self.lektor_weight_row, self.lektor_weight_slider, self.lektor_weight_value = self._create_slider_row(
            "Waga lektora",
            1,
            30,
            weight_to_slider_value(self.config_store.lektor_weight()),
            "Udzial lektora w miksie z oryginalnym audio filmu. Wyzej = lektor bardziej z przodu.",
            lambda value: f"{value / 10:.1f}",
        )
        self.lektor_weight_slider.valueChanged.connect(self._on_lektor_weight_changed)
        self.background_lufs_row, self.background_lufs_slider, self.background_lufs_value = self._create_slider_row(
            "Glosnosc tla (LUFS)",
            -30,
            -8,
            self.config_store.background_lufs(),
            "Docelowa glosnosc oryginalnego audio filmu przed zmiksowaniem z lektorem.",
            lambda value: f"{value} LUFS",
        )
        self.background_lufs_slider.valueChanged.connect(self._on_background_lufs_changed)
        self.background_weight_row, self.background_weight_slider, self.background_weight_value = self._create_slider_row(
            "Waga tla",
            1,
            30,
            weight_to_slider_value(self.config_store.background_weight()),
            "Udzial oryginalnego audio filmu w miksie z lektorem. Wyzej = glosniejsze tlo.",
            lambda value: f"{value / 10:.1f}",
        )
        self.background_weight_slider.valueChanged.connect(self._on_background_weight_changed)
        self.aac_quality_combo = QtWidgets.QComboBox()
        for option in aac_quality_options():
            self.aac_quality_combo.addItem(aac_quality_label(option), option)
        current_bitrate = self.config_store.aac_bitrate()
        current_index = self.aac_quality_combo.findData(current_bitrate)
        self.aac_quality_combo.setCurrentIndex(current_index if current_index >= 0 else self.aac_quality_combo.findData("256k"))
        self.aac_quality_combo.setToolTip("Jakosc finalnej sciezki lektora AAC dodawanej do MKV. Wyższy bitrate to wiekszy plik i mniejsza kompresja.")
        self.aac_quality_combo.currentIndexChanged.connect(self._on_aac_quality_changed)
        self.create_stereo_for_surround_checkbox = QtWidgets.QCheckBox("Tworz dodatkowa sciezke stereo przy 5.1")
        self.create_stereo_for_surround_checkbox.setChecked(self.config_store.create_stereo_for_surround())
        self.create_stereo_for_surround_checkbox.setToolTip(
            "Gdy zrodlo ma audio 5.1, program utworzy dodatkowa sciezke PL 2.0 obok PL 5.1. "
            "Wylacz, jesli potrzebujesz tylko sciezki PL 5.1."
        )
        self.create_stereo_for_surround_checkbox.toggled.connect(self._on_create_stereo_for_surround_toggled)
        lektor_delay_tooltip = (
            "Przesuwa wszystkie kwestie lektora wzgledem timestampow napisow. "
            "Im wyzsza wartosc, tym pozniej lektor zaczyna wzgledem napisow. "
            "W klasycznym voice-over lektor zwykle startuje po oryginalnej kwestii; "
            "zrodla podaja ok. 1-2 s, czasem wiecej. Przy napisach zwykle testuj subtelnie +100 do +500 ms."
        )
        self.lektor_delay_row, self.lektor_delay_slider, self.lektor_delay_value = self._create_slider_row(
            "Przesuniecie lektora",
            MIN_LEKTOR_DELAY_MS,
            MAX_LEKTOR_DELAY_MS,
            self.config_store.lektor_delay_ms(),
            lektor_delay_tooltip,
            format_lektor_delay_label,
        )
        self.lektor_delay_slider.setSingleStep(LEKTOR_DELAY_STEP_MS)
        self.lektor_delay_slider.setPageStep(LEKTOR_DELAY_STEP_MS * 5)
        self.lektor_delay_slider.valueChanged.connect(self._on_lektor_delay_changed)
        audio_buttons = QtWidgets.QHBoxLayout()
        self.btn_save_audio_settings = QtWidgets.QPushButton("Zapisz")
        self.btn_save_audio_settings.setToolTip("Zapisuje aktualne globalne ustawienia glosnosci i jakosci sciezki lektora w config.json.")
        self.btn_save_audio_settings.clicked.connect(self.save_audio_settings)
        self.btn_restore_audio_defaults = QtWidgets.QPushButton("Przywroc")
        self.btn_restore_audio_defaults.setToolTip("Przywraca domyslne ustawienia glosnosci, wag i jakosci sciezki lektora.")
        self.btn_restore_audio_defaults.clicked.connect(self.restore_audio_defaults)
        audio_buttons.addWidget(self.btn_save_audio_settings)
        audio_buttons.addWidget(self.btn_restore_audio_defaults)
        self.btn_start = QtWidgets.QPushButton("START")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_job)
        self.btn_stop = QtWidgets.QPushButton("Przerwij")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_job)
        self.progress_label = QtWidgets.QLabel("Gotowe")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m")
        self.progress_bar.setStyleSheet(progress_bar_style())
        self.tts_progress_label = QtWidgets.QLabel("TTS: -")
        self.tts_progress_bar = QtWidgets.QProgressBar()
        self.tts_progress_bar.setRange(0, 1)
        self.tts_progress_bar.setValue(0)
        self.tts_progress_bar.setTextVisible(True)
        self.tts_progress_bar.setFormat("%v/%m")
        self.tts_progress_bar.setStyleSheet(self.progress_bar.styleSheet())
        middle.addWidget(QtWidgets.QLabel("Silnik TTS"))
        middle.addWidget(self.engine_combo)
        middle.addWidget(self.btn_settings)
        middle.addWidget(self.btn_dictionary)
        middle.addWidget(self.btn_tts_manager)
        middle.addWidget(self.btn_diagnostics)
        middle.addWidget(self.btn_open_output)
        middle.addWidget(self.lektor_lufs_row)
        middle.addWidget(self.lektor_weight_row)
        middle.addWidget(self.background_lufs_row)
        middle.addWidget(self.background_weight_row)
        middle.addWidget(QtWidgets.QLabel("Jakosc sciezki lektora"))
        middle.addWidget(self.aac_quality_combo)
        middle.addWidget(self.create_stereo_for_surround_checkbox)
        middle.addWidget(self.lektor_delay_row)
        middle.addLayout(audio_buttons)
        middle.addStretch(1)
        middle.addWidget(self.progress_label)
        middle.addWidget(self.progress_bar)
        middle.addWidget(self.tts_progress_label)
        middle.addWidget(self.tts_progress_bar)
        middle.addWidget(self.btn_start)
        middle.addWidget(self.btn_stop)
        self.main_splitter.addWidget(middle_panel)

        right_panel = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_panel)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        right.addWidget(QtWidgets.QLabel("Log aplikacji"))
        right.addWidget(self.log_view, 1)
        self.main_splitter.addWidget(right_panel)
        for index in range(3):
            self.main_splitter.setStretchFactor(index, 1)
        self.main_splitter.setSizes([420, 420, 420])

    def _restore_window_state(self) -> None:
        state = self.config_store.window_state()
        width = int(state.get("width", 1280))
        height = int(state.get("height", 780))
        self.resize(width, height)
        self._center_on_current_screen()
        mode = str(state.get("mode", "normal") or "normal").lower()
        if mode == "fullscreen":
            self.showFullScreen()
        elif mode == "maximized":
            self.showMaximized()

    def _save_window_state(self) -> None:
        mode = "fullscreen" if self.isFullScreen() else "maximized" if self.isMaximized() else "normal"
        geometry = self.normalGeometry() if mode != "normal" else self.geometry()
        size = geometry.size() if geometry.isValid() else self.size()
        self.config_store.set_window_state(size.width(), size.height(), mode)

    def _center_on_current_screen(self) -> None:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        top_left = frame.topLeft()
        top_left.setX(max(available.left(), min(top_left.x(), available.right() - frame.width())))
        top_left.setY(max(available.top(), min(top_left.y(), available.bottom() - frame.height())))
        self.move(top_left)

    def refresh_engines(self, clear_cache: bool = True) -> None:
        if clear_cache:
            clear_engine_status_cache(self.engine_manager)
        self.engine_states = self.engine_manager.list_states()
        self.engine_combo.blockSignals(True)
        self.engine_combo.clear()
        self.engine_combo.addItem("Wybierz silnik TTS", "")
        for group_name, kind in (("Lokalne", EngineKind.LOCAL), ("Internetowe", EngineKind.INTERNET)):
            self.engine_combo.insertSeparator(self.engine_combo.count())
            for state in self.engine_states:
                if state.definition.kind != kind:
                    continue
                label = engine_combo_label(group_name, state)
                self.engine_combo.addItem(label, state.definition.engine_id)
                index = self.engine_combo.count() - 1
                if not state.selectable:
                    model = self.engine_combo.model()
                    item = model.item(index)
                    if item is not None:
                        item.setEnabled(False)
        self.restore_last_engine()
        self.engine_combo.blockSignals(False)
        self.on_engine_changed()

    def selected_engine_id(self) -> str:
        return str(self.engine_combo.currentData() or "")

    def restore_last_engine(self) -> None:
        last_engine = self.config_store.last_engine()
        if not last_engine:
            return
        state = next((item for item in self.engine_states if item.definition.engine_id == last_engine), None)
        if state is None or not state.selectable:
            self.config_store.set_last_engine("")
            return
        index = self.engine_combo.findData(last_engine)
        if index >= 0:
            self.engine_combo.setCurrentIndex(index)

    def on_engine_changed(self) -> None:
        engine_id = self.selected_engine_id()
        actions_enabled = self._selected_engine_actions_enabled(True)
        self.btn_dictionary.setEnabled(actions_enabled)
        self.btn_settings.setEnabled(actions_enabled)
        if engine_id and actions_enabled:
            self.engine_manager.ensure_engine_config(engine_id)
            self.log(f"Wybrano TTS: {engine_id}")
        self.config_store.set_last_engine(stored_engine_after_selection(engine_id, actions_enabled))
        self._update_start_button(True)

    def add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Wybierz pliki",
            self._file_dialog_start_dir(),
            "Napisy / tekst (*.srt *.txt);;Wideo (*.mkv *.mp4 *.avi *.mov *.webm *.wmv *.m4v);;Wszystkie pliki (*)",
        )
        self.add_file_paths(files)

    def add_file_paths(self, files: list[str]) -> None:
        added = 0
        skipped = 0
        unsupported = 0
        supported_extensions = {*SUPPORTED_SUBTITLE_EXTENSIONS, *VIDEO_EXTENSIONS}
        accepted_paths: list[Path] = [Path(self.queue.item(i).text()) for i in range(self.queue.count())]
        candidate_paths: list[Path] = []
        for file_path in files:
            path = Path(file_path)
            if not path.is_file() or path.suffix.lower() not in supported_extensions:
                unsupported += 1
                continue
            candidate_paths.append(path)
        if candidate_paths:
            self.config_store.set_last_file_dir(str(candidate_paths[0].resolve().parent))
        video_context = accepted_paths + [path for path in candidate_paths if is_video_file(path)]
        for path in candidate_paths:
            if is_sidecar_for_existing_video(path, accepted_paths):
                skipped += 1
                continue
            if is_sidecar_for_existing_video(path, video_context):
                skipped += 1
                continue
            if self._queue_contains(str(path)) or path_in_list(path, accepted_paths):
                skipped += 1
                continue
            self.queue.addItem(str(path))
            accepted_paths.append(path)
            added += 1
        if added:
            self.log(f"Kolejka: dodano {added} plik(ow)")
        if skipped:
            self.log(f"Kolejka: pominieto duplikaty/sidecary {skipped}")
        if unsupported:
            self.log(f"Kolejka: pominieto nieobslugiwane typy {unsupported}")
        self.update_queue_status()

    def remove_selected_files(self) -> None:
        selected = self.queue.selectedItems()
        if not selected:
            return
        for item in selected:
            row = self.queue.row(item)
            self.queue.takeItem(row)
        self.log(f"Kolejka: usunieto {len(selected)} plik(ow)")
        self.update_queue_status()

    def clear_files(self) -> None:
        count = self.queue.count()
        if count <= 0:
            return
        self.queue.clear()
        self.log(f"Kolejka: wyczyszczono {count} plik(ow)")
        self.update_queue_status()

    def move_selected_up(self) -> None:
        rows = sorted({index.row() for index in self.queue.selectedIndexes()})
        if not rows or rows[0] <= 0:
            return
        for row in rows:
            item = self.queue.takeItem(row)
            self.queue.insertItem(row - 1, item)
            item.setSelected(True)

    def move_selected_down(self) -> None:
        rows = sorted({index.row() for index in self.queue.selectedIndexes()}, reverse=True)
        if not rows or rows[0] >= self.queue.count() - 1:
            return
        for row in rows:
            item = self.queue.takeItem(row)
            self.queue.insertItem(row + 1, item)
            item.setSelected(True)

    def sort_queue(self) -> None:
        if self.queue.count() <= 1:
            return
        paths = [self.queue.item(i).text() for i in range(self.queue.count())]
        paths.sort(key=natural_path_key)
        self.queue.clear()
        for path in paths:
            self.queue.addItem(path)
        self.log("Kolejka: posortowano po nazwie pliku")
        self.update_queue_status()

    def _queue_contains(self, file_path: str) -> bool:
        normalized = str(Path(file_path).resolve()).lower()
        for index in range(self.queue.count()):
            item_path = str(Path(self.queue.item(index).text()).resolve()).lower()
            if item_path == normalized:
                return True
        return False

    def _file_dialog_start_dir(self) -> str:
        last_dir = self.config_store.last_file_dir()
        if last_dir and Path(last_dir).is_dir():
            return last_dir
        return str(Path.home())

    def update_queue_status(self) -> None:
        files = [Path(self.queue.item(i).text()) for i in range(self.queue.count())]
        video_count = sum(1 for path in files if is_video_file(path))
        subtitle_count = sum(1 for path in files if path.suffix.lower() in SUPPORTED_SUBTITLE_EXTENSIONS)
        self.queue_status.setText(f"Pliki: {len(files)} | wideo: {video_count} | napisy/tekst: {subtitle_count}")

    def open_dictionary(self) -> None:
        engine_id = self.selected_engine_id()
        if not engine_id:
            return
        path = self.engine_manager.ensure_engine_dictionary(engine_id)
        state = next((s for s in self.engine_states if s.definition.engine_id == engine_id), None)
        engine_name = state.definition.display_name if state else engine_id
        ok, new_data = edit_dictionary(self, engine_name, load_dictionary(path))
        if not ok:
            return
        count, skipped = save_dictionary(path, new_data)
        self.log(f"Slownik {engine_id}: zapisano {count} wpisow")
        if skipped:
            self.log(f"Slownik {engine_id}: pominieto {skipped} wpisow")

    def open_engine_settings(self) -> None:
        engine_id = self.selected_engine_id()
        if not engine_id:
            return
        config_path = self.engine_manager.ensure_engine_config(engine_id)
        state = next((s for s in self.engine_states if s.definition.engine_id == engine_id), None)
        engine_name = state.definition.display_name if state else engine_id
        default_config = state.definition.default_config if state else {}
        if edit_engine_settings(self, engine_name, engine_id, config_path, default_config):
            self.log(f"Ustawienia {engine_id}: zapisano")
            self.refresh_engines()

    def open_tts_manager(self) -> None:
        dialog = TTSManagerDialog(self, self.engine_manager)
        dialog.message.connect(self.log)
        dialog.changed.connect(self.refresh_engines)
        dialog.exec()

    def open_diagnostics(self) -> None:
        show_diagnostics(self, self.paths, self.engine_manager)

    def start_job(self) -> None:
        engine_id = self.selected_engine_id()
        if not engine_id:
            QtWidgets.QMessageBox.warning(self, "Brak TTS", "Wybierz zainstalowany silnik TTS.")
            return
        state = self.engine_manager.state_for(engine_id)
        if not state.selectable:
            QtWidgets.QMessageBox.warning(self, "TTS niedostepny", f"{state.definition.display_name}: {state.reason}")
            self.refresh_engines()
            return
        config_errors = self.validate_selected_engine_config(engine_id)
        if config_errors:
            show_scrollable_text(self, "Ustawienia TTS", "Popraw ustawienia TTS przed startem:", "\n".join(config_errors))
            return
        if self.queue.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Brak plikow", "Dodaj pliki do kolejki.")
            return
        files = [Path(self.queue.item(i).text()) for i in range(self.queue.count())]
        queue_errors = self.validate_queue_files(files)
        if queue_errors:
            show_scrollable_text(self, "Kolejka", "Popraw problemy w kolejce przed startem:", "\n".join(queue_errors))
            return
        if not self.confirm_start(engine_id, files):
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.last_output_dir = None
        self.btn_open_output.setEnabled(False)
        self._set_queue_controls_enabled(False)
        self.worker = PipelineWorker(
            files,
            engine_id,
            self.engine_manager,
            self.selected_aac_bitrate(),
            self.selected_lektor_lufs(),
            self.selected_lektor_weight(),
            self.selected_background_lufs(),
            self.selected_background_weight(),
            self.selected_lektor_delay_ms(),
            self.selected_create_stereo_for_surround(),
        )
        self.worker.message.connect(self.log)
        self.worker.diagnostic.connect(self.log_diagnostic)
        self.worker.progress.connect(self.on_pipeline_progress)
        self.worker.tts_progress.connect(self.on_tts_progress)
        self.worker.output_ready.connect(self.on_output_ready)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.finished_ok.connect(self.on_pipeline_finished)
        self.on_pipeline_progress(0, len(files))
        self.on_tts_progress(0, FILE_PROGRESS_TOTAL, "Aktualny plik: oczekiwanie")
        self.worker.start()

    def stop_job(self) -> None:
        if self.worker is None:
            return
        self.worker.requestInterruption()
        self.btn_stop.setEnabled(False)
        self.log("Przerwanie: zatrzymuje aktualny proces TTS")

    def open_last_output(self) -> None:
        if self.last_output_dir is None or not self.last_output_dir.exists():
            QtWidgets.QMessageBox.warning(self, "Wynik", "Brak folderu wyniku do otwarcia.")
            self.btn_open_output.setEnabled(False)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.last_output_dir)))

    def validate_selected_engine_config(self, engine_id: str) -> list[str]:
        config_path = self.engine_manager.ensure_engine_config(engine_id)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                config = {}
        except Exception:
            config = {}
        return validate_engine_config(engine_id, config)

    def validate_queue_files(self, files: list[Path]) -> list[str]:
        errors: list[str] = []
        supported_extensions = {*SUPPORTED_SUBTITLE_EXTENSIONS, *VIDEO_EXTENSIONS}
        ffmpeg = find_ffmpeg(self.paths)
        ffprobe = find_ffprobe(self.paths)
        mkvmerge = find_mkvmerge(self.paths)
        has_video = any(is_video_file(path) for path in files)
        if ffmpeg is None:
            errors.append(missing_ffmpeg_message())
        if has_video and ffprobe is None:
            errors.append(missing_ffprobe_message())
        if has_video and mkvmerge is None:
            errors.append(missing_mkvmerge_message())
        for path in files:
            if not path.exists():
                errors.append(f"Brak pliku: {path}")
                continue
            if not path.is_file():
                errors.append(f"To nie jest plik: {path}")
                continue
            if path.suffix.lower() not in supported_extensions:
                errors.append(f"Nieobslugiwany typ pliku: {path.name}")
        if len(errors) > 12:
            errors = errors[:12] + [f"... oraz {len(errors) - 12} kolejnych problemow"]
        return errors

    def confirm_start(self, engine_id: str, files: list[Path]) -> bool:
        state = self.engine_manager.state_for(engine_id)
        text = build_start_confirmation_text(
            engine_name=state.definition.display_name,
            file_count=len(files),
            video_count=sum(1 for path in files if is_video_file(path)),
            keep_lektor_assets=self.selected_engine_keep_lektor_assets(engine_id),
            aac_bitrate=self.selected_aac_bitrate(),
            lektor_delay_ms=self.selected_lektor_delay_ms(),
            ffmpeg_ok=find_ffmpeg(self.paths) is not None,
            ffprobe_ok=find_ffprobe(self.paths) is not None,
            mkvmerge_ok=find_mkvmerge(self.paths) is not None,
        )
        return confirm_scrollable_text(self, "Start konwersji", "Rozpoczac konwersje?", text)

    def on_pipeline_progress(self, done: int, total: int) -> None:
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat("%v/%m")
        if done >= total:
            self.progress_label.setText(f"Kolejka: gotowe {done}/{total}")
        else:
            self.progress_label.setText(f"Kolejka: plik {done + 1}/{total}")

    def on_tts_progress(self, done: int, total: int, label: str) -> None:
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self.tts_progress_bar.setRange(0, total)
        self.tts_progress_bar.setValue(done)
        self.tts_progress_bar.setFormat("%p%")
        self.tts_progress_label.setText(label)

    def on_output_ready(self, folder: str) -> None:
        self.last_output_dir = Path(folder)
        self.btn_open_output.setEnabled(self.last_output_dir.exists())

    def on_pipeline_failed(self, message: str) -> None:
        self.log(f"BLAD: {message}")
        self.btn_stop.setEnabled(False)
        self._set_queue_controls_enabled(True)
        self.progress_label.setText("Przerwano")
        self.tts_progress_label.setText("TTS: -")
        self.tts_progress_bar.setRange(0, 1)
        self.tts_progress_bar.setValue(0)
        self.tts_started_at = None
        self.worker = None

    def on_pipeline_finished(self, message: str) -> None:
        self.log(message)
        self.btn_stop.setEnabled(False)
        self._set_queue_controls_enabled(True)
        self.progress_label.setText("Gotowe")
        self.tts_progress_label.setText("TTS: -")
        self.tts_progress_bar.setRange(0, 1)
        self.tts_progress_bar.setValue(0)
        self.tts_started_at = None
        self.worker = None

    def _set_queue_controls_enabled(self, enabled: bool) -> None:
        self.btn_add.setEnabled(enabled)
        self.btn_remove_files.setEnabled(enabled)
        self.btn_move_up.setEnabled(enabled)
        self.btn_move_down.setEnabled(enabled)
        self.btn_sort_files.setEnabled(enabled)
        self.btn_clear_files.setEnabled(enabled)
        self.engine_combo.setEnabled(enabled)
        actions_enabled = self._selected_engine_actions_enabled(enabled)
        self.btn_settings.setEnabled(actions_enabled)
        self.btn_dictionary.setEnabled(actions_enabled)
        self.btn_tts_manager.setEnabled(enabled)
        self.aac_quality_combo.setEnabled(enabled)
        self.create_stereo_for_surround_checkbox.setEnabled(enabled)
        self.lektor_lufs_slider.setEnabled(enabled)
        self.lektor_weight_slider.setEnabled(enabled)
        self.background_lufs_slider.setEnabled(enabled)
        self.background_weight_slider.setEnabled(enabled)
        self.lektor_delay_slider.setEnabled(enabled)
        self.btn_save_audio_settings.setEnabled(enabled)
        self.btn_restore_audio_defaults.setEnabled(enabled)
        self._update_start_button(enabled)

    def selected_aac_bitrate(self) -> str:
        return sanitize_aac_bitrate(self.aac_quality_combo.currentData())

    def selected_engine_keep_lektor_assets(self, engine_id: str) -> bool:
        try:
            config_path = self.engine_manager.ensure_engine_config(engine_id)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                return False
            diagnostic_keys = (
                "save_processed_subtitles",
                "save_run_reports",
                "save_lektor_segments",
                "save_lektor_track_before_normalization",
                "save_lektor_track_after_normalization",
                "save_audio_mix_steps",
            )
            return any(bool_config_value(config.get(key), False) for key in diagnostic_keys)
        except Exception:
            return False

    def _on_aac_quality_changed(self) -> None:
        self.config_store.set_aac_bitrate(self.selected_aac_bitrate())

    def save_audio_settings(self) -> None:
        self._persist_audio_settings()
        self.log("Ustawienia audio: zapisano")

    def restore_audio_defaults(self) -> None:
        defaults = audio_defaults_summary()
        self.lektor_lufs_slider.setValue(int(defaults["lektor_lufs"]))
        self.lektor_weight_slider.setValue(weight_to_slider_value(float(defaults["lektor_weight"])))
        self.background_lufs_slider.setValue(int(defaults["background_lufs"]))
        self.background_weight_slider.setValue(weight_to_slider_value(float(defaults["background_weight"])))
        self.lektor_delay_slider.setValue(int(defaults["lektor_delay_ms"]))
        self.create_stereo_for_surround_checkbox.setChecked(bool(defaults["create_stereo_for_surround"]))
        index = self.aac_quality_combo.findData(str(defaults["aac_bitrate"]))
        if index >= 0:
            self.aac_quality_combo.setCurrentIndex(index)
        self._persist_audio_settings()
        self.log("Ustawienia audio: przywrocono domyslne")

    def _persist_audio_settings(self) -> None:
        self.config_store.set_lektor_lufs(self.selected_lektor_lufs())
        self.config_store.set_lektor_weight(self.selected_lektor_weight())
        self.config_store.set_background_lufs(self.selected_background_lufs())
        self.config_store.set_background_weight(self.selected_background_weight())
        self.config_store.set_aac_bitrate(self.selected_aac_bitrate())
        self.config_store.set_lektor_delay_ms(self.selected_lektor_delay_ms())
        self.config_store.set_create_stereo_for_surround(self.selected_create_stereo_for_surround())

    def selected_lektor_lufs(self) -> int:
        return sanitize_lufs(self.lektor_lufs_slider.value(), DEFAULT_LEKTOR_LUFS)

    def selected_lektor_weight(self) -> float:
        return slider_value_to_weight(self.lektor_weight_slider.value())

    def selected_background_lufs(self) -> int:
        return sanitize_lufs(self.background_lufs_slider.value(), DEFAULT_BACKGROUND_LUFS)

    def selected_background_weight(self) -> float:
        return slider_value_to_weight(self.background_weight_slider.value())

    def selected_lektor_delay_ms(self) -> int:
        return sanitize_lektor_delay_ms(self.lektor_delay_slider.value())

    def selected_create_stereo_for_surround(self) -> bool:
        return bool(self.create_stereo_for_surround_checkbox.isChecked())

    def _on_create_stereo_for_surround_toggled(self, checked: bool) -> None:
        self.config_store.set_create_stereo_for_surround(bool(checked))

    def _on_lektor_lufs_changed(self, value: int) -> None:
        self.lektor_lufs_value.setText(f"{value} LUFS")
        self.config_store.set_lektor_lufs(value)

    def _on_lektor_weight_changed(self, value: int) -> None:
        weight = slider_value_to_weight(value)
        self.lektor_weight_value.setText(f"{weight:.1f}")
        self.config_store.set_lektor_weight(weight)

    def _on_background_lufs_changed(self, value: int) -> None:
        self.background_lufs_value.setText(f"{value} LUFS")
        self.config_store.set_background_lufs(value)

    def _on_background_weight_changed(self, value: int) -> None:
        weight = slider_value_to_weight(value)
        self.background_weight_value.setText(f"{weight:.1f}")
        self.config_store.set_background_weight(weight)

    def _on_lektor_delay_changed(self, value: int) -> None:
        delay_ms = sanitize_lektor_delay_ms(value)
        if delay_ms != value:
            self.lektor_delay_slider.blockSignals(True)
            self.lektor_delay_slider.setValue(delay_ms)
            self.lektor_delay_slider.blockSignals(False)
        self.lektor_delay_value.setText(format_lektor_delay_label(delay_ms))
        self.config_store.set_lektor_delay_ms(delay_ms)

    def _create_slider_row(self, label: str, minimum: int, maximum: int, value: int, tooltip: str, formatter) -> tuple[QtWidgets.QWidget, QtWidgets.QSlider, QtWidgets.QLabel]:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(label)
        help_button = QtWidgets.QToolButton()
        help_button.setText("?")
        help_button.setAutoRaise(True)
        help_button.setToolTip(tooltip)
        help_button.setFixedWidth(22)
        value_label = QtWidgets.QLabel(formatter(value))
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        value_label.setMinimumWidth(76)
        header.addWidget(title)
        header.addWidget(help_button)
        header.addStretch(1)
        header.addWidget(value_label)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(max(minimum, min(maximum, int(value))))
        slider.setToolTip(tooltip)
        layout.addLayout(header)
        layout.addWidget(slider)
        return widget, slider, value_label

    def _update_start_button(self, controls_enabled: bool) -> None:
        engine_id = self.selected_engine_id()
        state = next((item for item in self.engine_states if item.definition.engine_id == engine_id), None)
        self.btn_start.setEnabled(should_enable_start_button(engine_id, controls_enabled, bool(state and state.selectable)))

    def _selected_engine_actions_enabled(self, controls_enabled: bool) -> bool:
        engine_id = self.selected_engine_id()
        state = next((item for item in self.engine_states if item.definition.engine_id == engine_id), None)
        return should_enable_engine_actions(engine_id, bool(state and state.selectable), controls_enabled)

    def log(self, message: str) -> None:
        self.logger.info(str(message or ""))
        compacted = compact_app_log_message(message)
        self.log_view.appendPlainText(compacted)

    def log_diagnostic(self, message: str) -> None:
        self.logger.info("DIAGNOSTYKA\n%s", str(message or "").rstrip())


def run_app(app_dir: Path) -> int:
    app = QtWidgets.QApplication([])
    paths = build_paths(app_dir)
    logger, log_path = setup_app_logger(paths.logs_dir)
    window = MainWindow(paths, logger, log_path)
    window.show()
    return app.exec()


def format_duration(seconds: float) -> str:
    seconds_i = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds_i, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}min {seconds_part:02d}s"
    if minutes:
        return f"{minutes}min {seconds_part:02d}s"
    return f"{seconds_part}s"


def build_start_confirmation_text(
    engine_name: str,
    file_count: int,
    video_count: int,
    keep_lektor_assets: bool,
    aac_bitrate: str,
    lektor_delay_ms: int,
    ffmpeg_ok: bool,
    ffprobe_ok: bool,
    mkvmerge_ok: bool = True,
) -> str:
    asset_mode = "zachowaj segmenty i raporty" if keep_lektor_assets else "usun diagnostyke po udanym muxie"
    ffprobe_status = "OK" if ffprobe_ok else ("brak, potrzebny dla wideo" if video_count else "brak, niepotrzebny dla samych napisow")
    mkvmerge_status = "OK" if mkvmerge_ok else ("brak, wymagany dla wideo" if video_count else "brak, niepotrzebny dla samych napisow")
    lines = [
        "Rozpoczac konwersje?",
        "",
        f"Silnik TTS: {engine_name}",
        f"Pliki w kolejce: {file_count}",
        f"Pliki wideo: {video_count}",
        f"Pliki lektora: {asset_mode}",
        f"Jakosc sciezki lektora AAC: {aac_quality_label(aac_bitrate)}",
        f"Przesuniecie lektora: {format_lektor_delay_label(lektor_delay_ms)}",
        f"ffmpeg: {'OK' if ffmpeg_ok else 'brak'}",
        f"ffprobe: {ffprobe_status}",
        f"mkvmerge: {mkvmerge_status}",
    ]
    if not ffmpeg_ok or (video_count and (not ffprobe_ok or not mkvmerge_ok)):
        lines.extend(["", f"Narzedzia: {BINARY_LOOKUP_HINT}"])
    lines.extend(["", "Po starcie ustawienia TTS i kolejka beda zablokowane do zakonczenia albo przerwania."])
    return "\n".join(lines)


def missing_ffmpeg_message() -> str:
    return f"Brak ffmpeg. Bez ffmpeg aplikacja nie zlozy gotowej sciezki lektora. {BINARY_LOOKUP_HINT}"


def missing_ffprobe_message() -> str:
    return f"Brak ffprobe. Bez ffprobe aplikacja nie obsluzy poprawnie plikow wideo. {BINARY_LOOKUP_HINT}"


def missing_mkvmerge_message() -> str:
    return f"Brak mkvmerge. Bez MKVToolNix aplikacja nie zapisze bezpiecznie kontenera MKV. {BINARY_LOOKUP_HINT}"


def natural_path_key(path_text: str) -> list:
    path = Path(path_text)
    name = path.name.casefold()
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    key.append(str(path.parent).casefold())
    return key


def is_sidecar_for_existing_video(path: Path, existing_paths: list[Path]) -> bool:
    if path.suffix.lower() not in SUPPORTED_SUBTITLE_EXTENSIONS:
        return False
    path_stem_lower = path.stem.lower()
    for existing in existing_paths:
        if not is_video_file(existing):
            continue
        video_stem = existing.stem.lower()
        if path.parent.resolve() != existing.parent.resolve():
            continue
        if path_stem_lower == video_stem or path_stem_lower.startswith(video_stem + "."):
            return True
    return False


def path_in_list(path: Path, paths: list[Path]) -> bool:
    normalized = str(path.resolve()).casefold()
    return any(str(existing.resolve()).casefold() == normalized for existing in paths)

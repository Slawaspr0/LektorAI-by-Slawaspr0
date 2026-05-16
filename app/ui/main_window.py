from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
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
    probe_media_duration,
    sanitize_aac_bitrate,
    sanitize_audio_weight,
    sanitize_lektor_delay_ms,
    sanitize_lufs,
)
from app.core.paths import AppPaths, build_paths
from app.core.version import APP_NAME, APP_VERSION
from app.updater.core import UpdateCheckResult, check_for_updates
from app.engines.config_validation import validate_engine_config
from app.engines.config_schema import (
    WHISPER_QC_MODELS,
    whisper_qc_compute_type_labels_for_options,
    whisper_qc_compute_type_options_for_device,
)
from app.engines.manager import EngineManager
from app.engines.schemas import EngineKind, EngineState
from app.engines.status import format_engine_state
from app.engines.voice_sample_rules import (
    validate_voice_sample_duration,
    voice_sample_rule,
)
from app.pipeline.subtitles import SUPPORTED_SUBTITLE_EXTENSIONS
from app.pipeline.progress import (
    FILE_PROGRESS_TOTAL,
    decode_progress_marker,
    format_progress_status,
    progress_value_for_stage,
    safe_unit_eta_seconds,
)
from app.pipeline.tts_job import run_tts_job
from app.pipeline.workspace import engine_short_code
from app.stt.job import SttSettings, SUPPORTED_STT_INPUT_EXTENSIONS, is_stt_input_file, run_stt_job
from app.stt.languages import STT_LANGUAGE_OPTIONS
from app.stt.whisper_cpp_runtime import WHISPER_CPP_RUNTIME_PACKAGES, whisper_cpp_runtime_ready
from app.ui.dialogs.diagnostics_dialog import show_diagnostics
from app.ui.dialogs.dictionary_dialog import edit_dictionary
from app.ui.dialogs.log_cleanup_dialog import show_log_cleanup
from app.ui.dialogs.scrollable_text_dialog import confirm_scrollable_text, show_scrollable_text
from app.ui.dialogs.settings_dialog import edit_engine_settings
from app.ui.dialogs.tts_manager_dialog import TTSManagerDialog


STT_ACCURACY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Szybciej", "fast"),
    ("Standard", "standard"),
    ("Dokladniej", "accurate"),
)
STT_VAD_SENSITIVITY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Lagodna", "gentle"),
    ("Standardowa", "standard"),
    ("Mocniejsza", "strong"),
)
STT_ENGINE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("faster-whisper", "faster_whisper"),
    ("whisper.cpp", "whisper_cpp"),
    ("WhisperX", "whisperx"),
)
STT_WHISPER_CPP_THREAD_OPTIONS: tuple[tuple[str, int], ...] = (
    ("Auto", 0),
    ("2", 2),
    ("4", 4),
    ("6", 6),
    ("8", 8),
    ("12", 12),
    ("16", 16),
    ("24", 24),
    ("32", 32),
)
STT_WHISPER_CPP_RUNTIME_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (package.label, package.variant) for package in WHISPER_CPP_RUNTIME_PACKAGES.values()
)


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


def polish_combo_box(combo: QtWidgets.QComboBox) -> QtWidgets.QComboBox:
    combo.setMinimumHeight(28)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
    return combo


class UpdateButton(QtWidgets.QPushButton):
    UPDATE_NEON_GREEN = "#39ff14"

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._update_available = False

    def set_update_available(self, value: bool) -> None:
        self._update_available = bool(value)
        self.update()

    def update_available(self) -> bool:
        return self._update_available

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        if self._update_available:
            border_rect = self.rect().adjusted(1, 1, -2, -2)
            painter.setPen(QtGui.QPen(QtGui.QColor(self.UPDATE_NEON_GREEN), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(border_rect, 4, 4)
        size = 9
        margin = 7
        y = max(0, (self.height() - size) // 2)
        rect = QtCore.QRect(self.width() - size - margin, y, size, size)
        color = QtGui.QColor(self.UPDATE_NEON_GREEN) if self._update_available else self.palette().button().color()
        painter.setPen(QtGui.QPen(QtGui.QColor("#151515"), 1))
        painter.setBrush(QtGui.QBrush(color))
        painter.drawEllipse(rect)


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
    engine_name = f"{state.definition.display_name} [{engine_short_code(state.definition.engine_id)}]"
    if state.definition.kind == EngineKind.INTERNET:
        return f"{group_name}: {engine_name}"
    text = f"{engine_name} - {state.status.value}"
    if state.reason:
        text += f" ({state.reason})"
    return f"{group_name}: {text}"


def aac_quality_options() -> tuple[str, ...]:
    return ("192k", "256k", "320k", "384k", "448k", "640k")


def aac_quality_label(value: str) -> str:
    bitrate = sanitize_aac_bitrate(value)
    return f"{bitrate[:-1]} kb/s"


def main_window_refresh_keeps_engine_signals_blocked() -> bool:
    return True


def worker_message_should_refresh_engine_status(message: str) -> bool:
    text = " ".join(str(message or "").strip().lower().split())
    if text.startswith("model tts: "):
        return "model w cache" in text or "ladowanie modelu" in text or "ładowanie modelu" in text
    return bool(re.fullmatch(r"segment\s+\d+/\d+", text))


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


class SttWorker(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    diagnostic = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    output_ready = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(str)

    def __init__(self, files: list[Path], paths: AppPaths, settings: SttSettings) -> None:
        super().__init__()
        self.files = files
        self.paths = paths
        self.settings = settings

    def run(self) -> None:
        total = len(self.files)
        failed: list[tuple[str, str]] = []
        output_dirs: list[str] = []
        try:
            for index, source_path in enumerate(self.files, 1):
                if self.isInterruptionRequested():
                    self.finished_ok.emit("STT: przerwano przez uzytkownika")
                    return
                self.progress.emit(index - 1, total)
                self.message.emit(f"STT: przetwarzanie {source_path.name}")
                try:
                    result = run_stt_job(
                        source_path,
                        self.paths,
                        self.settings,
                        progress=self.message.emit,
                        cancel_requested=self.isInterruptionRequested,
                    )
                except Exception as exc:
                    if self.isInterruptionRequested() or "Przerwano przez uzytkownika" in str(exc):
                        self.message.emit("STT: przerwano przez uzytkownika")
                        self.finished_ok.emit("STT: przerwano przez uzytkownika")
                        return
                    failed.append((source_path.name, str(exc)))
                    self.diagnostic.emit(
                        "BLAD STT pliku "
                        f"{source_path.name}\n"
                        f"Typ: {type(exc).__name__}\n"
                        f"Komunikat: {exc}\n"
                        f"{traceback.format_exc()}"
                    )
                    self.message.emit(f"BLAD STT pliku {source_path.name}: {exc}")
                    self.progress.emit(index, total)
                    continue
                output_dirs.append(str(result.workspace))
                self.output_ready.emit(str(result.workspace))
                self.message.emit(f"STT: zapisano {result.segment_count} segmentow w {format_duration(result.duration_seconds)}")
                self.message.emit(f"STT: wynik {result.output_srt.name}")
                self.progress.emit(index, total)
                if self.isInterruptionRequested():
                    self.finished_ok.emit("STT: przerwano przez uzytkownika")
                    return
            done = total - len(failed)
            self.message.emit(f"STT: gotowe {done}/{total}, bledy {len(failed)}")
            if output_dirs:
                unique_output_dirs = list(dict.fromkeys(output_dirs))
                self.message.emit(f"STT: foldery wynikow {len(unique_output_dirs)}, ostatni {Path(unique_output_dirs[-1]).name}")
            if failed:
                for name, error in failed[:8]:
                    self.message.emit(f"STT nieudany plik: {name} - {error}")
                if len(failed) > 8:
                    self.message.emit(f"STT nieudane pliki: +{len(failed) - 8} wiecej")
                self.finished_ok.emit(f"STT zakonczono z bledami: {len(failed)}")
            else:
                self.finished_ok.emit("STT zakonczono")
        except Exception as exc:
            self.diagnostic.emit(
                "BLAD krytyczny STT\n"
                f"Typ: {type(exc).__name__}\n"
                f"Komunikat: {exc}\n"
                f"{traceback.format_exc()}"
            )
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    update_check_finished = QtCore.pyqtSignal(object, bool)

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
        self.stt_worker: SttWorker | None = None
        self.tts_started_at: float | None = None
        self.engine_status_refreshed_during_job = False
        self.update_check_result: UpdateCheckResult | None = None
        self.update_check_running = False

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        min_width, min_height = main_window_minimum_size()
        self.setMinimumSize(min_width, min_height)
        self.statusBar().setSizeGripEnabled(True)
        self._build_ui()
        self._restore_window_state()
        self.update_check_finished.connect(self.on_update_check_finished)
        self.log(f"Aplikacja uruchomiona: {APP_NAME} {APP_VERSION}")
        self.refresh_engines()
        QtCore.QTimer.singleShot(1500, lambda: self.start_update_check(show_result=False))

    def closeEvent(self, event):  # noqa: N802
        if self._is_any_worker_running():
            QtWidgets.QMessageBox.warning(
                self,
                "Praca trwa",
                "Najpierw przerwij zadanie albo poczekaj na zakonczenie aktualnego pliku.",
            )
            event.ignore()
            return
        self._save_window_state()
        self.log("Aplikacja zamknieta")
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        tts_tab = QtWidgets.QWidget()
        tts_root = QtWidgets.QHBoxLayout(tts_tab)
        self.tabs.addTab(tts_tab, "TTS")
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        tts_root.addWidget(self.main_splitter)

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
        self.btn_log_cleanup = QtWidgets.QPushButton("Czyszczenie logow")
        self.btn_log_cleanup.clicked.connect(self.open_log_cleanup)
        self.btn_update = UpdateButton("Aktualizacja")
        self.btn_update.clicked.connect(self.open_update)
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
        middle.addWidget(self.btn_log_cleanup)
        middle.addWidget(self.btn_update)
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
        self._build_stt_tab()

    def _build_stt_tab(self) -> None:
        stt_tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(stt_tab)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        left_panel = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_panel)
        self.stt_queue = QueueListWidget()
        self.stt_queue.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.stt_queue.files_dropped.connect(self.add_stt_file_paths)
        self.stt_queue_status = QtWidgets.QLabel("Pliki: 0 | wideo: 0 | audio: 0")
        self.btn_stt_add = QtWidgets.QPushButton("Dodaj pliki")
        self.btn_stt_add.clicked.connect(self.add_stt_files)
        self.btn_stt_remove = QtWidgets.QPushButton("Usun zaznaczone")
        self.btn_stt_remove.clicked.connect(self.remove_selected_stt_files)
        self.btn_stt_clear = QtWidgets.QPushButton("Wyczysc")
        self.btn_stt_clear.clicked.connect(self.clear_stt_files)
        self.btn_stt_sort = QtWidgets.QPushButton("Sortuj")
        self.btn_stt_sort.clicked.connect(self.sort_stt_queue)
        stt_actions_top = QtWidgets.QHBoxLayout()
        stt_actions_top.addWidget(self.btn_stt_add)
        stt_actions_top.addWidget(self.btn_stt_remove)
        stt_actions_top.addWidget(self.btn_stt_clear)
        stt_actions_bottom = QtWidgets.QHBoxLayout()
        stt_actions_bottom.addWidget(self.btn_stt_sort)
        left.addWidget(QtWidgets.QLabel("Kolejka STT"))
        left.addWidget(self.stt_queue_status)
        left.addWidget(self.stt_queue, 1)
        left.addLayout(stt_actions_top)
        left.addLayout(stt_actions_bottom)
        splitter.addWidget(left_panel)

        settings = self.config_store.stt_settings()
        middle_panel = QtWidgets.QScrollArea()
        middle_panel.setWidgetResizable(True)
        middle_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        middle_content = QtWidgets.QWidget()
        middle = QtWidgets.QVBoxLayout(middle_content)
        middle.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        middle_panel.setWidget(middle_content)
        self.stt_engine_combo = polish_combo_box(QtWidgets.QComboBox())
        for label, value in STT_ENGINE_OPTIONS:
            self.stt_engine_combo.addItem(label, value)
        self._set_combo_data(self.stt_engine_combo, settings.engine)
        self.stt_engine_combo.setToolTip("Silnik rozpoznawania mowy.")
        self.stt_engine_combo.currentIndexChanged.connect(self._on_stt_engine_changed)

        self.stt_model_combo = polish_combo_box(QtWidgets.QComboBox())
        for model in WHISPER_QC_MODELS:
            self.stt_model_combo.addItem(model, model)
        self._set_combo_data(self.stt_model_combo, settings.model)
        self.stt_model_combo.setToolTip("Model do tworzenia napisow z mowy.")
        self.stt_model_combo.currentIndexChanged.connect(self._on_stt_model_changed)

        self.stt_language_combo = polish_combo_box(QtWidgets.QComboBox())
        for value, label in STT_LANGUAGE_OPTIONS:
            self.stt_language_combo.addItem(label, value)
        self._set_combo_data(self.stt_language_combo, settings.language)
        self.stt_language_combo.setToolTip("Jezyk rozpoznawanej mowy. Auto pozwala modelowi samemu wykryc jezyk.")
        self.stt_language_combo.currentIndexChanged.connect(self._on_stt_language_changed)

        self.stt_device_combo = polish_combo_box(QtWidgets.QComboBox())
        device_choices = self.engine_manager.torch_device_choices("", include_auto=False)
        for label, value in zip(device_choices.labels, device_choices.values):
            self.stt_device_combo.addItem(label, value)
        self._set_combo_data(self.stt_device_combo, settings.device)
        self.stt_device_combo.setToolTip("Urzadzenie dla STT. CPU jest najbezpieczniejsze, GPU moze byc szybsze.")
        self.stt_device_combo.currentIndexChanged.connect(self._on_stt_device_changed)

        self.stt_compute_combo = polish_combo_box(QtWidgets.QComboBox())
        self.stt_compute_combo.setToolTip("Tryb pracy STT. Na CPU dostepny jest tylko int8.")
        self.stt_compute_combo.currentIndexChanged.connect(self._on_stt_compute_type_changed)
        self._refresh_stt_compute_options(settings.compute_type)

        self.stt_accuracy_combo = polish_combo_box(QtWidgets.QComboBox())
        for label, value in STT_ACCURACY_OPTIONS:
            self.stt_accuracy_combo.addItem(label, value)
        self._set_combo_data(self.stt_accuracy_combo, settings.accuracy)
        self.stt_accuracy_combo.setToolTip("Okresla balans miedzy szybkoscia i dokladnoscia rozpoznawania mowy.")
        self.stt_accuracy_combo.currentIndexChanged.connect(self._on_stt_accuracy_changed)

        self.stt_vad_checkbox = QtWidgets.QCheckBox("Pomijanie ciszy")
        self.stt_vad_checkbox.setChecked(settings.vad_enabled)
        self.stt_vad_checkbox.setToolTip("Pomija fragmenty bez mowy, zeby ograniczyc powtarzane lub zmyslone napisy.")
        self.stt_vad_checkbox.toggled.connect(self._on_stt_vad_enabled_changed)

        self.stt_vad_sensitivity_combo = polish_combo_box(QtWidgets.QComboBox())
        for label, value in STT_VAD_SENSITIVITY_OPTIONS:
            self.stt_vad_sensitivity_combo.addItem(label, value)
        self._set_combo_data(self.stt_vad_sensitivity_combo, settings.vad_sensitivity)
        self.stt_vad_sensitivity_combo.setToolTip("Okresla, jak mocno program pomija cisze i tlo bez mowy.")
        self.stt_vad_sensitivity_combo.setEnabled(settings.vad_enabled)
        self.stt_vad_sensitivity_combo.currentIndexChanged.connect(self._on_stt_vad_sensitivity_changed)

        self.stt_whisper_cpp_threads_combo = polish_combo_box(QtWidgets.QComboBox())
        for label, value in STT_WHISPER_CPP_THREAD_OPTIONS:
            self.stt_whisper_cpp_threads_combo.addItem(label, value)
        self._set_combo_data(self.stt_whisper_cpp_threads_combo, settings.whisper_cpp_threads)
        self.stt_whisper_cpp_threads_combo.setToolTip("Liczba watkow CPU dla whisper.cpp. Auto dobiera wartosc samodzielnie.")
        self.stt_whisper_cpp_threads_combo.currentIndexChanged.connect(self._on_stt_whisper_cpp_threads_changed)

        self.stt_whisper_cpp_runtime_combo = polish_combo_box(QtWidgets.QComboBox())
        for label, value in STT_WHISPER_CPP_RUNTIME_OPTIONS:
            self.stt_whisper_cpp_runtime_combo.addItem(label, value)
        self._set_combo_data(self.stt_whisper_cpp_runtime_combo, settings.whisper_cpp_runtime)
        self.stt_whisper_cpp_runtime_combo.setToolTip("Wybierz CPU albo CUDA. Runtime pobierze sie przy pierwszym uzyciu.")
        self.stt_whisper_cpp_runtime_combo.currentIndexChanged.connect(self._on_stt_whisper_cpp_runtime_changed)

        self.stt_whisper_cpp_device_combo = polish_combo_box(QtWidgets.QComboBox())
        whisper_cpp_device_choices = self.engine_manager.torch_device_choices("", include_auto=True)
        for label, value in zip(whisper_cpp_device_choices.labels, whisper_cpp_device_choices.values):
            self.stt_whisper_cpp_device_combo.addItem(label, value)
        self._set_combo_data(self.stt_whisper_cpp_device_combo, settings.whisper_cpp_device)
        self.stt_whisper_cpp_device_combo.setToolTip("Urzadzenie dla whisper.cpp CUDA. Auto zostawia wybor programowi.")
        self.stt_whisper_cpp_device_combo.currentIndexChanged.connect(self._on_stt_whisper_cpp_device_changed)

        self.stt_whisper_cpp_runtime_status = QtWidgets.QLabel("")
        self.stt_whisper_cpp_runtime_status.setWordWrap(True)

        self.stt_save_audio_checkbox = QtWidgets.QCheckBox("Przygotowane audio")
        self.stt_save_audio_checkbox.setChecked(settings.save_prepared_audio)
        self.stt_save_audio_checkbox.setToolTip("Zachowuje audio przygotowane do rozpoznawania mowy.")
        self.stt_save_audio_checkbox.toggled.connect(self._on_stt_save_prepared_audio_changed)

        self.stt_save_report_checkbox = QtWidgets.QCheckBox("Raport STT")
        self.stt_save_report_checkbox.setChecked(settings.save_report)
        self.stt_save_report_checkbox.setToolTip("Zachowuje podsumowanie przebiegu STT.")
        self.stt_save_report_checkbox.toggled.connect(self._on_stt_save_report_changed)

        self.stt_save_log_checkbox = QtWidgets.QCheckBox("Log STT")
        self.stt_save_log_checkbox.setChecked(settings.save_log)
        self.stt_save_log_checkbox.setToolTip("Zachowuje prosty log przebiegu STT.")
        self.stt_save_log_checkbox.toggled.connect(self._on_stt_save_log_changed)

        self.stt_postprocess_checkbox = QtWidgets.QCheckBox("Formatowanie LektorAI")
        self.stt_postprocess_checkbox.setChecked(settings.postprocess_enabled)
        self.stt_postprocess_checkbox.setToolTip(
            "Po wlaczeniu aplikacja oczyszcza i uklada napisy. Po wylaczeniu zapisuje surowszy wynik silnika STT."
        )
        self.stt_postprocess_checkbox.toggled.connect(self._on_stt_postprocess_enabled_changed)

        self.btn_stt_open_output = QtWidgets.QPushButton("Otworz wynik")
        self.btn_stt_open_output.setEnabled(False)
        self.btn_stt_open_output.clicked.connect(self.open_last_stt_output)
        self.last_stt_output_dir: Path | None = None
        self.stt_progress_label = QtWidgets.QLabel("STT: -")
        self.stt_progress_bar = QtWidgets.QProgressBar()
        self.stt_progress_bar.setRange(0, 1)
        self.stt_progress_bar.setValue(0)
        self.stt_progress_bar.setTextVisible(True)
        self.stt_progress_bar.setFormat("%v/%m")
        self.stt_progress_bar.setStyleSheet(progress_bar_style())
        self.btn_stt_start = QtWidgets.QPushButton("START STT")
        self.btn_stt_start.clicked.connect(self.start_stt_job)
        self.btn_stt_stop = QtWidgets.QPushButton("Przerwij")
        self.btn_stt_stop.setEnabled(False)
        self.btn_stt_stop.clicked.connect(self.stop_stt_job)

        middle.addWidget(QtWidgets.QLabel("Silnik STT"))
        middle.addWidget(self.stt_engine_combo)
        middle.addWidget(QtWidgets.QLabel("Model"))
        middle.addWidget(self.stt_model_combo)
        middle.addWidget(QtWidgets.QLabel("Jezyk"))
        middle.addWidget(self.stt_language_combo)
        middle.addWidget(self.stt_postprocess_checkbox)

        self.stt_faster_whisper_box = QtWidgets.QGroupBox("Ustawienia faster-whisper")
        faster_whisper_layout = QtWidgets.QVBoxLayout(self.stt_faster_whisper_box)
        self.stt_device_label = QtWidgets.QLabel("Urzadzenie")
        faster_whisper_layout.addWidget(self.stt_device_label)
        faster_whisper_layout.addWidget(self.stt_device_combo)
        self.stt_compute_label = QtWidgets.QLabel("Tryb pracy")
        faster_whisper_layout.addWidget(self.stt_compute_label)
        faster_whisper_layout.addWidget(self.stt_compute_combo)
        faster_whisper_layout.addWidget(self.stt_vad_checkbox)
        self.stt_vad_sensitivity_label = QtWidgets.QLabel("Czulosc pomijania ciszy")
        faster_whisper_layout.addWidget(self.stt_vad_sensitivity_label)
        faster_whisper_layout.addWidget(self.stt_vad_sensitivity_combo)
        self.stt_accuracy_label = QtWidgets.QLabel("Dokladnosc")
        faster_whisper_layout.addWidget(self.stt_accuracy_label)
        faster_whisper_layout.addWidget(self.stt_accuracy_combo)
        self.stt_whisperx_note = QtWidgets.QLabel("WhisperX uzywa wlasnego wykrywania mowy i wyrownania timestampow.")
        self.stt_whisperx_note.setWordWrap(True)
        faster_whisper_layout.addWidget(self.stt_whisperx_note)

        self.stt_whisper_cpp_box = QtWidgets.QGroupBox("Ustawienia whisper.cpp")
        whisper_cpp_layout = QtWidgets.QVBoxLayout(self.stt_whisper_cpp_box)
        whisper_cpp_layout.addWidget(QtWidgets.QLabel("Runtime"))
        whisper_cpp_layout.addWidget(self.stt_whisper_cpp_runtime_combo)
        whisper_cpp_layout.addWidget(QtWidgets.QLabel("Urzadzenie CUDA"))
        whisper_cpp_layout.addWidget(self.stt_whisper_cpp_device_combo)
        whisper_cpp_layout.addWidget(QtWidgets.QLabel("Watki CPU"))
        whisper_cpp_layout.addWidget(self.stt_whisper_cpp_threads_combo)
        whisper_cpp_layout.addWidget(self.stt_whisper_cpp_runtime_status)

        middle.addSpacing(8)
        middle.addWidget(self.stt_faster_whisper_box)
        middle.addWidget(self.stt_whisper_cpp_box)
        middle.addSpacing(8)
        diagnostics_box = QtWidgets.QGroupBox("Opcje diagnostyczne")
        diagnostics_layout = QtWidgets.QVBoxLayout(diagnostics_box)
        diagnostics_layout.addWidget(self.stt_save_audio_checkbox)
        diagnostics_layout.addWidget(self.stt_save_report_checkbox)
        diagnostics_layout.addWidget(self.stt_save_log_checkbox)
        middle.addWidget(diagnostics_box)
        middle.addWidget(self.btn_stt_open_output)
        middle.addStretch(1)
        middle.addWidget(self.stt_progress_label)
        middle.addWidget(self.stt_progress_bar)
        middle.addWidget(self.btn_stt_start)
        middle.addWidget(self.btn_stt_stop)
        splitter.addWidget(middle_panel)

        right_panel = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_panel)
        self.stt_log_view = QtWidgets.QPlainTextEdit()
        self.stt_log_view.setReadOnly(True)
        self.stt_log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        right.addWidget(QtWidgets.QLabel("Log STT"))
        right.addWidget(self.stt_log_view, 1)
        splitter.addWidget(right_panel)
        for index in range(3):
            splitter.setStretchFactor(index, 1)
        splitter.setSizes([420, 420, 420])
        self.tabs.addTab(stt_tab, "STT")
        self._load_stt_controls_for_engine(str(self.stt_engine_combo.currentData() or "faster_whisper"))
        self._refresh_stt_engine_settings_visibility()

    def _set_combo_data(self, combo: QtWidgets.QComboBox, value: str | int) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

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

    def refresh_engines(
        self,
        clear_cache: bool = True,
        log_selection: bool = True,
        controls_enabled: bool | None = None,
        store_selection: bool = True,
    ) -> None:
        if controls_enabled is None:
            controls_enabled = self.engine_combo.isEnabled()
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
        self.on_engine_changed(log_selection=log_selection, controls_enabled=controls_enabled, store_selection=store_selection)

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

    def on_engine_changed(
        self,
        index: int | None = None,
        log_selection: bool = True,
        controls_enabled: bool | None = None,
        store_selection: bool = True,
    ) -> None:
        if controls_enabled is None:
            controls_enabled = self.engine_combo.isEnabled()
        engine_id = self.selected_engine_id()
        actions_enabled = self._selected_engine_actions_enabled(bool(controls_enabled))
        self.btn_dictionary.setEnabled(actions_enabled)
        self.btn_settings.setEnabled(actions_enabled)
        if engine_id and actions_enabled:
            self.engine_manager.ensure_engine_config(engine_id)
            if log_selection:
                self.log(f"Wybrano TTS: {engine_id}")
        if store_selection:
            self.config_store.set_last_engine(stored_engine_after_selection(engine_id, actions_enabled))
        self._update_start_button(bool(controls_enabled))

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

    def add_stt_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Wybierz pliki STT",
            self._file_dialog_start_dir(),
            "Audio / wideo (*.mkv *.mp4 *.avi *.mov *.webm *.wmv *.m4v *.aac *.ac3 *.flac *.m4a *.mp3 *.ogg *.opus *.wav *.wma);;Wszystkie pliki (*)",
        )
        self.add_stt_file_paths(files)

    def add_stt_file_paths(self, files: list[str]) -> None:
        added = 0
        skipped = 0
        unsupported = 0
        accepted_paths: list[Path] = [Path(self.stt_queue.item(i).text()) for i in range(self.stt_queue.count())]
        for file_path in files:
            path = Path(file_path)
            if not path.is_file() or not is_stt_input_file(path):
                unsupported += 1
                continue
            if path_in_list(path, accepted_paths):
                skipped += 1
                continue
            self.stt_queue.addItem(str(path))
            accepted_paths.append(path)
            added += 1
        if accepted_paths:
            self.config_store.set_last_file_dir(str(accepted_paths[-1].resolve().parent))
        if added:
            self.log_stt(f"STT: dodano {added} plik(ow)")
        if skipped:
            self.log_stt(f"STT: pominieto duplikaty {skipped}")
        if unsupported:
            self.log_stt(f"STT: pominieto nieobslugiwane typy {unsupported}")
        self.update_stt_queue_status()

    def remove_selected_stt_files(self) -> None:
        selected = self.stt_queue.selectedItems()
        if not selected:
            return
        for item in selected:
            row = self.stt_queue.row(item)
            self.stt_queue.takeItem(row)
        self.log_stt(f"STT: usunieto {len(selected)} plik(ow)")
        self.update_stt_queue_status()

    def clear_stt_files(self) -> None:
        count = self.stt_queue.count()
        if count <= 0:
            return
        self.stt_queue.clear()
        self.log_stt(f"STT: wyczyszczono {count} plik(ow)")
        self.update_stt_queue_status()

    def sort_stt_queue(self) -> None:
        if self.stt_queue.count() <= 1:
            return
        paths = [self.stt_queue.item(i).text() for i in range(self.stt_queue.count())]
        paths.sort(key=natural_path_key)
        self.stt_queue.clear()
        for path in paths:
            self.stt_queue.addItem(path)
        self.log_stt("STT: posortowano po nazwie pliku")
        self.update_stt_queue_status()

    def update_stt_queue_status(self) -> None:
        files = [Path(self.stt_queue.item(i).text()) for i in range(self.stt_queue.count())]
        video_count = sum(1 for path in files if is_video_file(path))
        audio_count = len(files) - video_count
        self.stt_queue_status.setText(f"Pliki: {len(files)} | wideo: {video_count} | audio: {audio_count}")

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
        if edit_engine_settings(
            self,
            engine_name,
            engine_id,
            config_path,
            default_config,
            self.engine_settings_option_overrides(engine_id),
        ):
            self.log(f"Ustawienia {engine_id}: zapisano")
            self.refresh_engines()

    def engine_settings_option_overrides(self, engine_id: str) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
        tts_devices = self.engine_manager.torch_device_choices(engine_id, include_auto=True)
        stt_devices = self.engine_manager.torch_device_choices(engine_id, include_auto=False)
        return {
            "device": (tts_devices.values, tts_devices.labels),
            "whisper_qc_device": (stt_devices.values, stt_devices.labels),
        }

    def open_tts_manager(self) -> None:
        dialog = TTSManagerDialog(self, self.engine_manager)
        dialog.message.connect(self.log)
        dialog.changed.connect(self.refresh_engines)
        dialog.exec()

    def open_diagnostics(self) -> None:
        show_diagnostics(self, self.paths, self.engine_manager)

    def open_log_cleanup(self) -> None:
        show_log_cleanup(self, self.paths, self.log_path)

    def open_update(self) -> None:
        if self._is_any_worker_running():
            QtWidgets.QMessageBox.warning(self, "Aktualizacja", "Najpierw przerwij albo zakoncz aktualne zadanie.")
            return
        if self.update_check_result is not None and self.update_check_result.update_available:
            self.start_update()
            return
        self.start_update_check(show_result=True)

    def start_update_check(self, show_result: bool) -> None:
        if self.update_check_running:
            if show_result:
                QtWidgets.QMessageBox.information(self, "Aktualizacja", "Sprawdzanie aktualizacji juz trwa.")
            return
        self.update_check_running = True
        self.btn_update.setEnabled(False)

        def run_check() -> None:
            result = check_for_updates(self.paths.app_dir)
            self.update_check_finished.emit(result, bool(show_result))

        threading.Thread(target=run_check, daemon=True).start()

    def on_update_check_finished(self, result: UpdateCheckResult, show_result: bool) -> None:
        self.update_check_running = False
        self.update_check_result = result
        self.btn_update.setEnabled(not self._is_any_worker_running())
        self.btn_update.set_update_available(bool(result.ok and result.update_available))
        if result.ok and result.update_available:
            self.btn_update.setToolTip("Dostepna jest aktualizacja programu.")
            if not show_result:
                self.log("Aktualizacja: dostepna nowa wersja")
                return
        elif result.ok:
            self.btn_update.setToolTip("Masz najnowsza wersje programu.")
        else:
            self.btn_update.setToolTip("Nie udalo sie sprawdzic aktualizacji.")
        if not show_result:
            return
        if not result.ok:
            QtWidgets.QMessageBox.warning(self, "Aktualizacja", f"Nie udalo sie sprawdzic aktualizacji.\n{result.error}")
            return
        if result.update_available:
            self.start_update()
        else:
            QtWidgets.QMessageBox.information(self, "Aktualizacja", "Masz najnowsza wersje programu.")

    def start_update(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Aktualizacja",
            "Program zamknie sie, pobierze aktualizacje i uruchomi ponownie.\nKontynuowac?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        updater_path = self.paths.app_dir / "UPDATER.py"
        if not updater_path.is_file():
            QtWidgets.QMessageBox.warning(self, "Aktualizacja", "Brakuje pliku UPDATER.py.")
            return
        try:
            subprocess.Popen(
                [sys.executable, str(updater_path), "--app-dir", str(self.paths.app_dir), "--pid", str(os.getpid())],
                cwd=str(self.paths.app_dir),
                close_fds=True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Aktualizacja", f"Nie udalo sie uruchomic aktualizatora.\n{exc}")
            return
        QtWidgets.QApplication.quit()

    def start_job(self) -> None:
        if self.stt_worker is not None and self.stt_worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "STT trwa", "Najpierw zakoncz albo przerwij aktualne zadanie STT.")
            return
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
        self.engine_status_refreshed_during_job = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.last_output_dir = None
        self.btn_open_output.setEnabled(False)
        self._set_queue_controls_enabled(False)
        self._set_stt_controls_enabled(False)
        self.btn_stt_stop.setEnabled(False)
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
        self.worker.message.connect(self.on_worker_message)
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

    def start_stt_job(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "TTS trwa", "Najpierw zakoncz albo przerwij aktualne zadanie TTS.")
            return
        if self.stt_queue.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Brak plikow", "Dodaj pliki audio lub wideo do kolejki STT.")
            return
        files = [Path(self.stt_queue.item(i).text()) for i in range(self.stt_queue.count())]
        errors = self.validate_stt_queue_files(files)
        if errors:
            show_scrollable_text(self, "Kolejka STT", "Popraw problemy w kolejce STT przed startem:", "\n".join(errors))
            return
        settings = self.selected_stt_settings()
        self.last_stt_output_dir = None
        self.btn_stt_open_output.setEnabled(False)
        self._set_stt_controls_enabled(False)
        self._set_queue_controls_enabled(False)
        self.stt_progress_bar.setRange(0, 0)
        self.stt_progress_label.setText("STT: start")
        self.stt_worker = SttWorker(files, self.paths, settings)
        self.stt_worker.message.connect(self.on_stt_worker_message)
        self.stt_worker.diagnostic.connect(self.log_diagnostic)
        self.stt_worker.progress.connect(self.on_stt_progress)
        self.stt_worker.output_ready.connect(self.on_stt_output_ready)
        self.stt_worker.failed.connect(self.on_stt_failed)
        self.stt_worker.finished_ok.connect(self.on_stt_finished)
        self.on_stt_progress(0, len(files))
        self.stt_worker.start()

    def stop_stt_job(self) -> None:
        if self.stt_worker is None:
            return
        self.stt_worker.requestInterruption()
        self.btn_stt_stop.setEnabled(False)
        self.log_stt("STT: zatrzymuje aktualny proces")

    def validate_stt_queue_files(self, files: list[Path]) -> list[str]:
        errors: list[str] = []
        if find_ffmpeg(self.paths) is None:
            errors.append(missing_ffmpeg_message())
        if find_ffprobe(self.paths) is None:
            errors.append(missing_ffprobe_message())
        for path in files:
            if not path.exists():
                errors.append(f"Brak pliku: {path}")
                continue
            if not path.is_file():
                errors.append(f"To nie jest plik: {path}")
                continue
            if path.suffix.lower() not in SUPPORTED_STT_INPUT_EXTENSIONS:
                errors.append(f"Nieobslugiwany typ pliku STT: {path.name}")
        if len(errors) > 12:
            errors = errors[:12] + [f"... oraz {len(errors) - 12} kolejnych problemow"]
        return errors

    def on_worker_message(self, message: str) -> None:
        self.log(message)
        if self.engine_status_refreshed_during_job:
            return
        if not worker_message_should_refresh_engine_status(message):
            return
        self.engine_status_refreshed_during_job = True
        self.refresh_engines(clear_cache=False, log_selection=False, controls_enabled=False, store_selection=False)

    def on_stt_worker_message(self, message: str) -> None:
        self.log_stt(message)

    def on_stt_progress(self, done: int, total: int) -> None:
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self.stt_progress_bar.setRange(0, total)
        self.stt_progress_bar.setValue(done)
        self.stt_progress_bar.setFormat("%v/%m")
        if done >= total:
            self.stt_progress_label.setText(f"STT: gotowe {done}/{total}")
        else:
            self.stt_progress_label.setText(f"STT: plik {done + 1}/{total}")

    def on_stt_output_ready(self, folder: str) -> None:
        self.last_stt_output_dir = Path(folder)
        self.btn_stt_open_output.setEnabled(self.last_stt_output_dir.exists())

    def on_stt_failed(self, message: str) -> None:
        self.log_stt(f"BLAD STT: {message}")
        self.btn_stt_stop.setEnabled(False)
        self._set_stt_controls_enabled(True)
        self._set_queue_controls_enabled(True)
        self._refresh_stt_whisper_cpp_runtime_controls()
        self.stt_progress_label.setText("STT: przerwano")
        self.stt_progress_bar.setRange(0, 1)
        self.stt_progress_bar.setValue(0)
        self.stt_worker = None

    def on_stt_finished(self, message: str) -> None:
        self.log_stt(message)
        self.btn_stt_stop.setEnabled(False)
        self._set_stt_controls_enabled(True)
        self._set_queue_controls_enabled(True)
        self._refresh_stt_whisper_cpp_runtime_controls()
        self.stt_progress_label.setText("STT: gotowe")
        self.stt_progress_bar.setRange(0, 1)
        self.stt_progress_bar.setValue(0)
        self.stt_worker = None

    def open_last_output(self) -> None:
        if self.last_output_dir is None or not self.last_output_dir.exists():
            QtWidgets.QMessageBox.warning(self, "Wynik", "Brak folderu wyniku do otwarcia.")
            self.btn_open_output.setEnabled(False)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.last_output_dir)))

    def open_last_stt_output(self) -> None:
        if self.last_stt_output_dir is None or not self.last_stt_output_dir.exists():
            QtWidgets.QMessageBox.warning(self, "Wynik STT", "Brak folderu wyniku STT do otwarcia.")
            self.btn_stt_open_output.setEnabled(False)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.last_stt_output_dir)))

    def validate_selected_engine_config(self, engine_id: str) -> list[str]:
        config_path = self.engine_manager.ensure_engine_config(engine_id)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                config = {}
        except Exception:
            config = {}
        errors = validate_engine_config(engine_id, config)
        errors.extend(self.validate_selected_voice_sample_duration(engine_id, config))
        return errors

    def validate_selected_voice_sample_duration(self, engine_id: str, config: dict) -> list[str]:
        rule = voice_sample_rule(engine_id)
        if rule is None:
            return []
        value = str(config.get(rule.config_key, "") or "").strip()
        if not value:
            return []
        path = Path(value)
        if not path.is_file():
            return []
        ffprobe = find_ffprobe(self.paths)
        if ffprobe is None:
            return [f"{rule.label}: nie moge sprawdzic dlugosci probki glosu, bo brakuje ffprobe. {BINARY_LOOKUP_HINT}"]
        try:
            duration_seconds = probe_media_duration(ffprobe, path)
        except Exception:
            return [f"{rule.label}: nie mozna odczytac dlugosci pliku audio. Uzyj poprawnego WAV/MP3/FLAC."]
        return validate_voice_sample_duration(engine_id, duration_seconds)

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
        self._set_stt_controls_enabled(True)
        self.refresh_engines(log_selection=False)
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
        self._set_stt_controls_enabled(True)
        self.refresh_engines(log_selection=False)
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
        self.btn_log_cleanup.setEnabled(enabled)
        self.btn_update.setEnabled(enabled and not self.update_check_running)
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

    def _set_stt_controls_enabled(self, enabled: bool) -> None:
        self.btn_stt_add.setEnabled(enabled)
        self.btn_stt_remove.setEnabled(enabled)
        self.btn_stt_clear.setEnabled(enabled)
        self.btn_stt_sort.setEnabled(enabled)
        self.stt_engine_combo.setEnabled(enabled)
        self.stt_model_combo.setEnabled(enabled)
        self.stt_language_combo.setEnabled(enabled)
        self.stt_device_combo.setEnabled(enabled)
        settings = self.config_store.stt_settings()
        engine = str(self.stt_engine_combo.currentData() or "faster_whisper")
        preferred_compute = settings.whisperx_compute_type if engine == "whisperx" else settings.compute_type
        self._refresh_stt_compute_options(preferred_compute)
        self.stt_compute_combo.setEnabled(enabled and self.stt_compute_combo.count() > 1)
        self.stt_accuracy_combo.setEnabled(enabled)
        self.stt_vad_checkbox.setEnabled(enabled)
        self.stt_vad_sensitivity_combo.setEnabled(enabled and self.stt_vad_checkbox.isChecked())
        self.stt_whisper_cpp_runtime_combo.setEnabled(enabled)
        self.stt_whisper_cpp_device_combo.setEnabled(enabled)
        self.stt_whisper_cpp_threads_combo.setEnabled(enabled)
        self.stt_postprocess_checkbox.setEnabled(enabled)
        self.stt_save_audio_checkbox.setEnabled(enabled)
        self.stt_save_report_checkbox.setEnabled(enabled)
        self.stt_save_log_checkbox.setEnabled(enabled)
        self.btn_stt_start.setEnabled(enabled)
        self.btn_stt_stop.setEnabled(not enabled)
        self._refresh_stt_engine_settings_visibility()

    def _is_any_worker_running(self) -> bool:
        return bool(
            (self.worker is not None and self.worker.isRunning())
            or (self.stt_worker is not None and self.stt_worker.isRunning())
        )

    def selected_aac_bitrate(self) -> str:
        return sanitize_aac_bitrate(self.aac_quality_combo.currentData())

    def selected_stt_settings(self) -> SttSettings:
        engine = str(self.stt_engine_combo.currentData() or "faster_whisper")
        stored_settings = self.config_store.stt_settings()
        device = self.selected_stt_device()
        compute_type = str(self.stt_compute_combo.currentData() or "int8")
        faster_device = device if engine == "faster_whisper" else stored_settings.device
        faster_compute_type = compute_type if engine == "faster_whisper" else stored_settings.compute_type
        whisperx_device = device if engine == "whisperx" else stored_settings.whisperx_device
        whisperx_compute_type = compute_type if engine == "whisperx" else stored_settings.whisperx_compute_type
        return SttSettings(
            engine=engine,
            model=str(self.stt_model_combo.currentData() or "small"),
            language=str(self.stt_language_combo.currentData() or "auto"),
            device=faster_device,
            compute_type=faster_compute_type,
            accuracy=str(self.stt_accuracy_combo.currentData() or "standard"),
            vad_enabled=self.stt_vad_checkbox.isChecked(),
            vad_sensitivity=str(self.stt_vad_sensitivity_combo.currentData() or "standard"),
            whisper_cpp_runtime=str(self.stt_whisper_cpp_runtime_combo.currentData() or "cpu"),
            whisper_cpp_device=str(self.stt_whisper_cpp_device_combo.currentData() or "auto"),
            whisper_cpp_threads=int(self.stt_whisper_cpp_threads_combo.currentData() or 0),
            whisperx_device=whisperx_device,
            whisperx_compute_type=whisperx_compute_type,
            postprocess_enabled=self.stt_postprocess_checkbox.isChecked(),
            save_prepared_audio=self.stt_save_audio_checkbox.isChecked(),
            save_report=self.stt_save_report_checkbox.isChecked(),
            save_log=self.stt_save_log_checkbox.isChecked(),
        )

    def selected_stt_device(self) -> str:
        return str(self.stt_device_combo.currentData() or "cpu")

    def _on_stt_engine_changed(self) -> None:
        engine = str(self.stt_engine_combo.currentData() or "faster_whisper")
        self.config_store.set_stt_engine(engine)
        self._load_stt_controls_for_engine(engine)
        self._refresh_stt_engine_settings_visibility()

    def _on_stt_model_changed(self) -> None:
        self.config_store.set_stt_model(str(self.stt_model_combo.currentData() or "small"))

    def _on_stt_language_changed(self) -> None:
        self.config_store.set_stt_language(str(self.stt_language_combo.currentData() or "auto"))

    def _on_stt_device_changed(self) -> None:
        device = self.selected_stt_device()
        engine = str(self.stt_engine_combo.currentData() or "faster_whisper")
        if engine == "whisperx":
            self.config_store.set_stt_whisperx_device(device)
            self._refresh_stt_compute_options(self.config_store.stt_settings().whisperx_compute_type)
        elif engine == "faster_whisper":
            self.config_store.set_stt_device(device)
            self._refresh_stt_compute_options(self.config_store.stt_settings().compute_type)

    def _on_stt_compute_type_changed(self) -> None:
        value = str(self.stt_compute_combo.currentData() or "int8")
        engine = str(self.stt_engine_combo.currentData() or "faster_whisper")
        if engine == "whisperx":
            self.config_store.set_stt_whisperx_compute_type(value)
        elif engine == "faster_whisper":
            self.config_store.set_stt_compute_type(value)

    def _on_stt_accuracy_changed(self) -> None:
        self.config_store.set_stt_accuracy(str(self.stt_accuracy_combo.currentData() or "standard"))

    def _on_stt_vad_enabled_changed(self, checked: bool) -> None:
        self.config_store.set_stt_vad_enabled(checked)
        self.stt_vad_sensitivity_combo.setEnabled(checked and self.btn_stt_start.isEnabled())

    def _on_stt_vad_sensitivity_changed(self) -> None:
        self.config_store.set_stt_vad_sensitivity(str(self.stt_vad_sensitivity_combo.currentData() or "standard"))

    def _on_stt_postprocess_enabled_changed(self, checked: bool) -> None:
        self.config_store.set_stt_postprocess_enabled(checked)

    def _on_stt_whisper_cpp_runtime_changed(self) -> None:
        self.config_store.set_stt_whisper_cpp_runtime(str(self.stt_whisper_cpp_runtime_combo.currentData() or "cpu"))
        self._refresh_stt_whisper_cpp_runtime_controls()

    def _on_stt_whisper_cpp_device_changed(self) -> None:
        self.config_store.set_stt_whisper_cpp_device(str(self.stt_whisper_cpp_device_combo.currentData() or "auto"))

    def _on_stt_whisper_cpp_threads_changed(self) -> None:
        self.config_store.set_stt_whisper_cpp_threads(int(self.stt_whisper_cpp_threads_combo.currentData() or 0))

    def _on_stt_save_prepared_audio_changed(self, checked: bool) -> None:
        self.config_store.set_stt_save_prepared_audio(checked)

    def _on_stt_save_report_changed(self, checked: bool) -> None:
        self.config_store.set_stt_save_report(checked)

    def _on_stt_save_log_changed(self, checked: bool) -> None:
        self.config_store.set_stt_save_log(checked)

    def _refresh_stt_compute_options(self, preferred_value: str | None = None) -> None:
        device = self.selected_stt_device() if hasattr(self, "stt_device_combo") else "cpu"
        options = whisper_qc_compute_type_options_for_device(device)
        labels = whisper_qc_compute_type_labels_for_options(options)
        current = str(preferred_value or self.stt_compute_combo.currentData() or "int8")
        if current not in options:
            current = options[0]
        self.stt_compute_combo.blockSignals(True)
        self.stt_compute_combo.clear()
        for label, value in zip(labels, options):
            self.stt_compute_combo.addItem(label, value)
        self._set_combo_data(self.stt_compute_combo, current)
        self.stt_compute_combo.setEnabled(len(options) > 1)
        self.stt_compute_combo.blockSignals(False)
        engine = str(self.stt_engine_combo.currentData() or "faster_whisper")
        if engine == "whisperx":
            self.config_store.set_stt_whisperx_compute_type(current)
        elif engine == "faster_whisper":
            self.config_store.set_stt_compute_type(current)

    def _load_stt_controls_for_engine(self, engine: str) -> None:
        if not hasattr(self, "stt_model_combo"):
            return
        settings = self.config_store.stt_settings()
        combo_values: tuple[tuple[QtWidgets.QComboBox, str | int], ...] = (
            (self.stt_model_combo, settings.model),
            (self.stt_language_combo, settings.language),
            (self.stt_accuracy_combo, settings.accuracy),
            (self.stt_vad_sensitivity_combo, settings.vad_sensitivity),
            (self.stt_whisper_cpp_runtime_combo, settings.whisper_cpp_runtime),
            (self.stt_whisper_cpp_device_combo, settings.whisper_cpp_device),
            (self.stt_whisper_cpp_threads_combo, settings.whisper_cpp_threads),
        )
        for combo, value in combo_values:
            combo.blockSignals(True)
            self._set_combo_data(combo, value)
            combo.blockSignals(False)
        checkboxes: tuple[tuple[QtWidgets.QCheckBox, bool], ...] = (
            (self.stt_vad_checkbox, settings.vad_enabled),
            (self.stt_save_audio_checkbox, settings.save_prepared_audio),
            (self.stt_save_report_checkbox, settings.save_report),
            (self.stt_save_log_checkbox, settings.save_log),
            (self.stt_postprocess_checkbox, settings.postprocess_enabled),
        )
        for checkbox, checked in checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        self._load_stt_device_controls_for_engine(engine)
        self._refresh_stt_whisper_cpp_runtime_controls()

    def _load_stt_device_controls_for_engine(self, engine: str) -> None:
        if not hasattr(self, "stt_device_combo"):
            return
        settings = self.config_store.stt_settings()
        if engine == "whisperx":
            device = settings.whisperx_device
            compute_type = settings.whisperx_compute_type
        else:
            device = settings.device
            compute_type = settings.compute_type
        self.stt_device_combo.blockSignals(True)
        self._set_combo_data(self.stt_device_combo, device)
        self.stt_device_combo.blockSignals(False)
        self._refresh_stt_compute_options(compute_type)

    def _refresh_stt_engine_settings_visibility(self) -> None:
        if not hasattr(self, "stt_engine_combo"):
            return
        engine = str(self.stt_engine_combo.currentData() or "faster_whisper")
        is_faster_whisper = engine == "faster_whisper"
        is_whisperx = engine == "whisperx"
        self.stt_faster_whisper_box.setVisible(is_faster_whisper or is_whisperx)
        self.stt_faster_whisper_box.setTitle("Ustawienia WhisperX" if is_whisperx else "Ustawienia faster-whisper")
        self.stt_vad_checkbox.setVisible(is_faster_whisper)
        self.stt_vad_sensitivity_label.setVisible(is_faster_whisper)
        self.stt_vad_sensitivity_combo.setVisible(is_faster_whisper)
        self.stt_whisperx_note.setVisible(is_whisperx)
        self.stt_whisper_cpp_box.setVisible(engine == "whisper_cpp")
        self._refresh_stt_whisper_cpp_runtime_controls()

    def _refresh_stt_whisper_cpp_runtime_controls(self) -> None:
        if not hasattr(self, "stt_whisper_cpp_runtime_combo"):
            return
        enabled = self.btn_stt_start.isEnabled() if hasattr(self, "btn_stt_start") else True
        runtime = str(self.stt_whisper_cpp_runtime_combo.currentData() or "cpu")
        is_cuda = runtime == "cuda"
        self.stt_whisper_cpp_device_combo.setEnabled(enabled and is_cuda)
        if hasattr(self, "stt_whisper_cpp_runtime_status"):
            ready = whisper_cpp_runtime_ready(self.paths, runtime)
            status = "Runtime: gotowy" if ready else "Runtime: pobierze sie przy pierwszym uzyciu"
            self.stt_whisper_cpp_runtime_status.setText(status)

    def selected_engine_keep_lektor_assets(self, engine_id: str) -> bool:
        try:
            config_path = self.engine_manager.ensure_engine_config(engine_id)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                return False
            diagnostic_keys = (
                "save_processed_subtitles",
                "save_quality_report",
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

    def log_stt(self, message: str) -> None:
        self.logger.info(str(message or ""))
        compacted = compact_app_log_message(message)
        self.stt_log_view.appendPlainText(compacted)

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

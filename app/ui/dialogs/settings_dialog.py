from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtWidgets

from app.engines.config_schema import (
    ConfigField,
    DIAGNOSTIC_FIELD_KEYS,
    DIAGNOSTIC_MIX_FIELD_KEYS,
    DIAGNOSTIC_REPORT_FIELD_KEYS,
    DIAGNOSTIC_SEGMENT_FIELD_KEYS,
    DIAGNOSTIC_TRACK_FIELD_KEYS,
    is_audio_qc_field,
    is_diagnostic_field,
    is_speech_qc_field,
    visible_fields_for,
)


class EdgeProsodySlider(QtWidgets.QWidget):
    def __init__(
        self,
        value: Any,
        suffix: str,
        minimum: int,
        maximum: int,
        step: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.suffix = suffix
        self.step = max(1, int(step))
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(minimum))
        self.slider.setMaximum(int(maximum))
        self.slider.setSingleStep(self.step)
        self.slider.setPageStep(max(self.step, 10))
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(max(self.step * 4, 20))
        self.label = QtWidgets.QLabel()
        self.label.setMinimumWidth(58)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label, 0)
        self.slider.valueChanged.connect(self._on_value_changed)
        self.set_edge_value(value)

    def edge_value(self) -> str:
        return format_edge_slider_value(self.slider.value(), self.suffix)

    def set_edge_value(self, value: Any) -> None:
        self.slider.setValue(
            edge_slider_value_for_widget(value, self.suffix, self.slider.minimum(), self.slider.maximum(), self.step)
        )
        self._update_label()

    def _on_value_changed(self, value: int) -> None:
        snapped = edge_slider_value_for_widget(value, self.suffix, self.slider.minimum(), self.slider.maximum(), self.step)
        if snapped != value:
            self.slider.blockSignals(True)
            self.slider.setValue(snapped)
            self.slider.blockSignals(False)
        self._update_label()

    def _update_label(self) -> None:
        self.label.setText(self.edge_value())


class EngineSettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        engine_name: str,
        engine_id: str,
        config: dict[str, Any],
        default_config: dict[str, Any],
    ) -> None:
        super().__init__(parent)
        self.engine_id = engine_id
        self.config = dict(config)
        self.default_config = dict(default_config)
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.setWindowTitle(f"Ustawienia TTS - {engine_name}")
        self.resize(680, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        fields = visible_fields_for(self.engine_id)
        model_fields = tuple(
            field
            for field in fields
            if not is_audio_qc_field(field) and not is_speech_qc_field(field) and not is_diagnostic_field(field)
        )
        audio_qc_fields = tuple(field for field in fields if is_audio_qc_field(field))
        speech_qc_fields = tuple(field for field in fields if is_speech_qc_field(field))
        diagnostic_fields = tuple(
            sorted(
                (field for field in fields if is_diagnostic_field(field)),
                key=lambda field: DIAGNOSTIC_FIELD_KEYS.index(field.key)
                if field.key in DIAGNOSTIC_FIELD_KEYS
                else len(DIAGNOSTIC_FIELD_KEYS),
            )
        )

        content_layout.addLayout(self._form_for(model_fields))
        if audio_qc_fields:
            self._add_section(content_layout, "Kontrola audio", "Techniczna kontrola wygenerowanego pliku audio.")
            content_layout.addLayout(self._form_for(audio_qc_fields))
        if speech_qc_fields:
            self._add_section(content_layout, "Kontrola mowy", "Kontrola zgodnosci wypowiedzianego tekstu z napisami przez faster-whisper.")
            content_layout.addLayout(self._form_for(speech_qc_fields))
        if diagnostic_fields:
            self._add_section(
                content_layout,
                "Opcje diagnostyczne",
                "Opcje pomocne przy testowaniu i debugowaniu wynikow TTS. Zwykly uzytkownik najczesciej nie musi ich zmieniac.",
            )
            for title, keys in diagnostic_field_groups():
                grouped = tuple(field for field in diagnostic_fields if field.key in keys)
                if not grouped:
                    continue
                self._add_subsection(content_layout, title)
                content_layout.addLayout(self._form_for(grouped))
        content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Zapisz")
        self.btn_restore = QtWidgets.QPushButton("Przywroc")
        self.btn_exit = QtWidgets.QPushButton("Wyjdz")
        self.btn_save.clicked.connect(self.accept)
        self.btn_restore.clicked.connect(self.restore_defaults)
        self.btn_exit.clicked.connect(self.reject)
        buttons.addWidget(self.btn_save)
        buttons.addWidget(self.btn_restore)
        buttons.addWidget(self.btn_exit)
        root.addLayout(buttons)

    def _add_section(self, layout: QtWidgets.QVBoxLayout, title: str, tooltip: str) -> None:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)
        label = QtWidgets.QLabel(title)
        label.setToolTip(tooltip)
        layout.addWidget(label)

    def _add_subsection(self, layout: QtWidgets.QVBoxLayout, title: str) -> None:
        label = QtWidgets.QLabel(title)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label)

    def _form_for(self, fields: tuple[ConfigField, ...]) -> QtWidgets.QFormLayout:
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for field in fields:
            widget = self._widget_for(field)
            form.addRow(self._label_widget(field), widget)
            self.widgets[field.key] = widget
        return form

    def restore_defaults(self) -> None:
        self.config = dict(self.default_config)
        for field in visible_fields_for(self.engine_id):
            self._set_widget_value(field, self.default_config.get(field.key, ""))

    def _label_widget(self, field: ConfigField) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QtWidgets.QLabel(field.label)
        label.setMinimumWidth(150)
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        if field.show_help and str(field.tooltip or "").strip():
            help_button = QtWidgets.QToolButton()
            help_button.setText(settings_help_button_label())
            help_button.setToolTip(field.tooltip)
            help_button.setAutoRaise(True)
            help_button.setFixedSize(22, 22)
            help_button.clicked.connect(lambda _checked=False, title=field.label, text=field.tooltip: self._show_field_help(title, text))
            layout.addWidget(help_button, 0)
        return container

    def _show_field_help(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.information(self, title, text)

    def _widget_for(self, field: ConfigField) -> QtWidgets.QWidget:
        value = self.config.get(field.key, "")
        if field.field_type == "bool":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(coerce_bool_for_widget(value))
            return widget
        if field.field_type == "int":
            widget = QtWidgets.QSpinBox()
            widget.setMinimum(int(field.minimum if field.minimum is not None else -2147483648))
            widget.setMaximum(int(field.maximum if field.maximum is not None else 2147483647))
            widget.setSingleStep(int(field.step if field.step is not None else 1))
            widget.setValue(coerce_int_for_widget(value, widget.minimum(), widget.maximum()))
            return widget
        if field.field_type == "float":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setDecimals(3)
            widget.setMinimum(float(field.minimum if field.minimum is not None else -999999.0))
            widget.setMaximum(float(field.maximum if field.maximum is not None else 999999.0))
            widget.setSingleStep(float(field.step if field.step is not None else 0.1))
            widget.setValue(coerce_float_for_widget(value, widget.minimum(), widget.maximum()))
            return widget
        if field.field_type == "path":
            return self._path_widget(str(value), field.tooltip)
        if field.field_type == "choice":
            widget = QtWidgets.QComboBox()
            if field.option_labels and len(field.option_labels) == len(field.options):
                for label, option in zip(field.option_labels, field.options):
                    widget.addItem(label, option)
            else:
                widget.addItems(list(field.options))
            selected = choice_value_for_widget(value, field.options, field.options[0] if field.options else "")
            index = widget.findData(selected)
            if index < 0:
                index = widget.findText(selected)
            if index >= 0:
                widget.setCurrentIndex(index)
            return widget
        if field.field_type in {"percent_slider", "hz_slider"}:
            suffix = "%" if field.field_type == "percent_slider" else "Hz"
            return EdgeProsodySlider(
                value,
                suffix,
                int(field.minimum if field.minimum is not None else -100),
                int(field.maximum if field.maximum is not None else 100),
                int(field.step if field.step is not None else 1),
            )
        widget = QtWidgets.QLineEdit(str(value))
        widget.setMinimumWidth(320)
        if field.secret:
            widget.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        return widget

    def _path_widget(self, value: str, tooltip: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit(value)
        edit.setMinimumWidth(260)
        button = QtWidgets.QPushButton("Wybierz")
        button.clicked.connect(lambda: self._choose_audio_file(edit))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return container

    def _choose_audio_file(self, edit: QtWidgets.QLineEdit) -> None:
        start_dir = str(Path(edit.text()).parent) if edit.text().strip() else str(Path.home())
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Wybierz plik audio",
            start_dir,
            "Audio (*.wav *.mp3 *.flac);;WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac);;Wszystkie pliki (*)",
        )
        if file_path:
            edit.setText(file_path)

    def values(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field in visible_fields_for(self.engine_id):
            widget = self.widgets[field.key]
            if hasattr(widget, "edge_value"):
                data[field.key] = widget.edge_value()
            elif isinstance(widget, QtWidgets.QCheckBox):
                data[field.key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QSpinBox):
                data[field.key] = int(widget.value())
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                data[field.key] = float(widget.value())
            elif isinstance(widget, QtWidgets.QComboBox):
                data[field.key] = choice_data_for_widget(widget)
            elif isinstance(widget, QtWidgets.QLineEdit):
                data[field.key] = widget.text().strip()
            else:
                line_edit = widget.findChild(QtWidgets.QLineEdit)
                data[field.key] = line_edit.text().strip() if line_edit is not None else ""
        return data

    def _set_widget_value(self, field: ConfigField, value: Any) -> None:
        widget = self.widgets.get(field.key)
        if widget is None:
            return
        if hasattr(widget, "set_edge_value"):
            widget.set_edge_value(value)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.setChecked(coerce_bool_for_widget(value))
        elif isinstance(widget, QtWidgets.QSpinBox):
            widget.setValue(coerce_int_for_widget(value, widget.minimum(), widget.maximum()))
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            widget.setValue(coerce_float_for_widget(value, widget.minimum(), widget.maximum()))
        elif isinstance(widget, QtWidgets.QComboBox):
            selected = choice_value_for_widget(value, field.options, field.options[0] if field.options else "")
            index = widget.findData(selected)
            if index < 0:
                index = widget.findText(selected)
            widget.setCurrentIndex(index if index >= 0 else 0)
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(str(value or ""))
        else:
            line_edit = widget.findChild(QtWidgets.QLineEdit)
            if line_edit is not None:
                line_edit.setText(str(value or ""))


def edit_engine_settings(
    parent: QtWidgets.QWidget,
    engine_name: str,
    engine_id: str,
    config_path: Path,
    default_config: dict[str, Any] | None = None,
) -> bool:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            config = {}
    except Exception:
        config = {}
    dialog = EngineSettingsDialog(parent, engine_name, engine_id, config, default_config or {})
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return False
    updated = merge_engine_settings_values(config, dialog.values())
    config_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def merge_engine_settings_values(config: dict[str, Any], visible_values: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config or {})
    updated.update(dict(visible_values or {}))
    return updated


def diagnostic_field_groups() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("Napisy i raporty", DIAGNOSTIC_REPORT_FIELD_KEYS),
        ("Segmenty TTS", DIAGNOSTIC_SEGMENT_FIELD_KEYS),
        ("Sciezki lektora", DIAGNOSTIC_TRACK_FIELD_KEYS),
        ("Etapy miksowania audio", DIAGNOSTIC_MIX_FIELD_KEYS),
    )


def coerce_bool_for_widget(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "tak", "on"}:
            return True
        if lowered in {"false", "0", "no", "nie", "off", ""}:
            return False
    return False


def coerce_int_for_widget(value: Any, minimum: int, maximum: int) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not int")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("fractional int")
            number = int(value)
        else:
            text = str(value).strip()
            if not text or any(char in text for char in ".,"):
                raise ValueError("invalid int text")
            number = int(text)
    except Exception:
        number = int(minimum)
    return max(int(minimum), min(int(maximum), number))


def coerce_float_for_widget(value: Any, minimum: float, maximum: float) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not float")
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            raise ValueError("non-finite float")
    except Exception:
        number = float(minimum)
    return max(float(minimum), min(float(maximum), number))


def choice_value_for_widget(value: Any, options: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in options else str(default or "")


def choice_data_for_widget(widget: QtWidgets.QComboBox) -> str:
    data = widget.currentData()
    if data is not None:
        return str(data).strip()
    return widget.currentText().strip()


def edge_slider_value_for_widget(value: Any, suffix: str, minimum: int, maximum: int, step: int = 1) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not edge slider value")
        if isinstance(value, int):
            number = value
        else:
            text = str(value or "").strip()
            match = re.fullmatch(r"([+-])(\d+)" + re.escape(suffix), text)
            if match is None:
                raise ValueError("invalid edge slider text")
            number = int(match.group(2))
            if match.group(1) == "-":
                number = -number
    except Exception:
        number = 0
    clamped = max(int(minimum), min(int(maximum), int(number)))
    step = max(1, int(step))
    if step > 1:
        clamped = int(round(clamped / step) * step)
        clamped = max(int(minimum), min(int(maximum), clamped))
    return clamped


def format_edge_slider_value(value: int, suffix: str) -> str:
    number = int(value)
    sign = "+" if number >= 0 else ""
    return f"{sign}{number}{suffix}"


def settings_help_button_label() -> str:
    return "?"

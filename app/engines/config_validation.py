from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import math
import os
import re

from app.engines.config_schema import EDGE_PITCH_MAX, EDGE_PITCH_MIN, EDGE_RATE_MAX, EDGE_RATE_MIN, fields_for
from app.core.media_tools import supported_voice_sample_extensions


def validate_engine_config(engine_id: str, config: dict[str, Any]) -> list[str]:
    errors = _validate_schema_values(engine_id, config)
    if engine_id == "chatterbox":
        errors.extend(_validate_optional_audio(config, "audio_prompt_path", "probka glosu Chatterbox", supported_voice_sample_extensions(), "WAV/MP3/FLAC"))
    elif engine_id == "omnivoice":
        errors.extend(_validate_optional_audio(config, "reference_audio_path", "probka glosu OmniVoice", supported_voice_sample_extensions(), "WAV/MP3/FLAC"))
    elif engine_id == "openai":
        errors.extend(_validate_openai(config))
    elif engine_id == "edge":
        errors.extend(_validate_edge(config))
    errors.extend(_validate_device(config))
    if engine_id in {"edge", "openai"}:
        errors.extend(validate_whisper_qc_dependency(config))
    return errors


def validate_whisper_qc_dependency(config: dict[str, Any], finder=None) -> list[str]:
    if not _bool_config(config.get("whisper_qc_enabled"), False):
        return []
    finder = finder or importlib.util.find_spec
    if finder("faster_whisper"):
        return []
    return ["Kontrola tekstu Whisper: brak biblioteki faster-whisper. Zainstaluj requirements albo wylacz te opcje."]


def _validate_schema_values(engine_id: str, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in fields_for(engine_id):
        if field.key not in config:
            continue
        value = config.get(field.key)
        label = field.label
        if field.field_type == "int":
            try:
                if isinstance(value, bool):
                    raise ValueError("bool is not int config")
                if isinstance(value, float):
                    if not math.isfinite(value) or not value.is_integer():
                        raise ValueError("float is not whole int config")
                    number = int(value)
                else:
                    value_text = str(value).strip()
                    if not re.fullmatch(r"[+-]?\d+", value_text):
                        raise ValueError("int config is not digits")
                    number = int(value_text)
            except Exception:
                errors.append(f"{label}: wartosc musi byc liczba calkowita.")
                continue
            if field.minimum is not None and number < int(field.minimum):
                errors.append(f"{label}: minimum to {int(field.minimum)}.")
            if field.maximum is not None and number > int(field.maximum):
                errors.append(f"{label}: maksimum to {int(field.maximum)}.")
        elif field.field_type == "float":
            try:
                if isinstance(value, bool):
                    raise ValueError("bool is not float config")
                number = float(value)
            except Exception:
                errors.append(f"{label}: wartosc musi byc liczba.")
                continue
            if not math.isfinite(number):
                errors.append(f"{label}: wartosc musi byc skonczona liczba.")
                continue
            if field.minimum is not None and number < float(field.minimum):
                errors.append(f"{label}: minimum to {field.minimum:g}.")
            if field.maximum is not None and number > float(field.maximum):
                errors.append(f"{label}: maksimum to {field.maximum:g}.")
        elif field.field_type == "bool" and not isinstance(value, bool):
            errors.append(f"{label}: wartosc musi byc true/false.")
        elif field.field_type == "choice":
            if str(value or "").strip() not in field.options:
                allowed = ", ".join(field.options)
                errors.append(f"{label}: wybierz jedna z opcji: {allowed}.")
    return errors


def _validate_edge(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    voice = str(config.get("voice", "") or "").strip()
    rate = str(config.get("rate", "") or "").strip()
    pitch = str(config.get("pitch", "") or "").strip()
    if not voice:
        errors.append("Edge: glos nie moze byc pusty.")
    if rate and not re.fullmatch(r"[+-]\d+%", rate):
        errors.append("Edge: predkosc musi miec format np. +0%, -10%, +15%.")
    elif rate and not _edge_number_in_range(rate, "%", EDGE_RATE_MIN, EDGE_RATE_MAX):
        errors.append(f"Edge: predkosc musi byc w zakresie od {EDGE_RATE_MIN}% do +{EDGE_RATE_MAX}%.")
    if pitch and not re.fullmatch(r"[+-]\d+Hz", pitch):
        errors.append("Edge: pitch musi miec format np. +0Hz, -5Hz, +10Hz.")
    elif pitch and not _edge_number_in_range(pitch, "Hz", EDGE_PITCH_MIN, EDGE_PITCH_MAX):
        errors.append(f"Edge: barwa glosu musi byc w zakresie od {EDGE_PITCH_MIN}Hz do +{EDGE_PITCH_MAX}Hz.")
    return errors


def _edge_number_in_range(value: str, suffix: str, minimum: int, maximum: int) -> bool:
    match = re.fullmatch(r"([+-])(\d+)" + re.escape(suffix), str(value or "").strip())
    if match is None:
        return False
    number = int(match.group(2))
    if match.group(1) == "-":
        number = -number
    return int(minimum) <= number <= int(maximum)


def _validate_openai(config: dict[str, Any]) -> list[str]:
    errors = []
    if not str(config.get("api_key", "") or "").strip() and not os.environ.get("OPENAI_API_KEY"):
        errors.append("Brak klucza API OpenAI albo OPENAI_API_KEY.")
    if not str(config.get("model", "") or "").strip():
        errors.append("OpenAI: model nie moze byc pusty.")
    if not str(config.get("voice", "") or "").strip():
        errors.append("OpenAI: glos nie moze byc pusty.")
    return errors


def _validate_device(config: dict[str, Any]) -> list[str]:
    device = str(config.get("device", "") or "").strip()
    if not device:
        return []
    if device == "auto" or device == "cpu" or re.fullmatch(r"cuda(:\d+)?", device):
        return []
    return ["Urzadzenie: wpisz auto, cpu, cuda albo cuda:N."]


def _validate_optional_audio(config: dict[str, Any], key: str, label: str, allowed_suffixes: tuple[str, ...], suffix_label: str) -> list[str]:
    value = str(config.get(key, "") or "").strip()
    if not value:
        return []
    path = Path(value)
    if not path.is_file():
        return [f"{label}: plik nie istnieje."]
    if path.suffix.lower() not in allowed_suffixes:
        return [f"{label}: plik powinien byc {suffix_label}."]
    return []


def _bool_config(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "tak", "yes", "on"}:
        return True
    if text in {"0", "false", "nie", "no", "off"}:
        return False
    return default

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import broken_json_backup_path
from app.core.media_tools import (
    DEFAULT_AAC_BITRATE,
    DEFAULT_BACKGROUND_LUFS,
    DEFAULT_BACKGROUND_WEIGHT,
    DEFAULT_LEKTOR_LUFS,
    DEFAULT_LEKTOR_DELAY_MS,
    DEFAULT_LEKTOR_WEIGHT,
    sanitize_aac_bitrate,
    sanitize_audio_weight,
    sanitize_lektor_delay_ms,
    sanitize_lufs,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "ui": {
        "theme": "dark",
        "last_file_dir": "",
        "window": {
            "width": 1280,
            "height": 780,
            "mode": "normal",
        },
    },
    "tts": {
        "last_engine": "",
    },
    "output": {
        "aac_bitrate": DEFAULT_AAC_BITRATE,
        "lektor_lufs": DEFAULT_LEKTOR_LUFS,
        "lektor_weight": DEFAULT_LEKTOR_WEIGHT,
        "background_lufs": DEFAULT_BACKGROUND_LUFS,
        "background_weight": DEFAULT_BACKGROUND_WEIGHT,
        "lektor_delay_ms": DEFAULT_LEKTOR_DELAY_MS,
        "create_stereo_for_surround": True,
    },
}


@dataclass
class AppConfigStore:
    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            self.data = json.loads(json.dumps(DEFAULT_CONFIG))
            self.save()
            return self.data
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("config root is not an object")
            self.data = _normalize_config(_merge_defaults(DEFAULT_CONFIG, loaded))
            if self.data != loaded:
                self.save()
        except Exception:
            broken = broken_json_backup_path(self.path)
            self.path.replace(broken)
            self.data = json.loads(json.dumps(DEFAULT_CONFIG))
            self.save()
        return self.data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def set_last_engine(self, engine_id: str) -> None:
        self.data.setdefault("tts", {})["last_engine"] = str(engine_id or "").strip()
        self.save()

    def last_engine(self) -> str:
        tts = self.data.get("tts", {})
        if not isinstance(tts, dict):
            return ""
        return str(tts.get("last_engine", "") or "").strip()

    def aac_bitrate(self) -> str:
        output = self.data.get("output", {})
        if not isinstance(output, dict):
            return DEFAULT_AAC_BITRATE
        return sanitize_aac_bitrate(output.get("aac_bitrate", DEFAULT_AAC_BITRATE))

    def set_aac_bitrate(self, value: str) -> None:
        self.data.setdefault("output", {})["aac_bitrate"] = sanitize_aac_bitrate(value)
        self.save()

    def lektor_lufs(self) -> int:
        output = self.data.get("output", {})
        return sanitize_lufs(output.get("lektor_lufs") if isinstance(output, dict) else None, DEFAULT_LEKTOR_LUFS)

    def set_lektor_lufs(self, value: int | float | str) -> None:
        self.data.setdefault("output", {})["lektor_lufs"] = sanitize_lufs(value, DEFAULT_LEKTOR_LUFS)
        self.save()

    def lektor_weight(self) -> float:
        output = self.data.get("output", {})
        return sanitize_audio_weight(output.get("lektor_weight") if isinstance(output, dict) else None, DEFAULT_LEKTOR_WEIGHT)

    def set_lektor_weight(self, value: int | float | str) -> None:
        self.data.setdefault("output", {})["lektor_weight"] = sanitize_audio_weight(value, DEFAULT_LEKTOR_WEIGHT)
        self.save()

    def background_lufs(self) -> int:
        output = self.data.get("output", {})
        return sanitize_lufs(output.get("background_lufs") if isinstance(output, dict) else None, DEFAULT_BACKGROUND_LUFS)

    def set_background_lufs(self, value: int | float | str) -> None:
        self.data.setdefault("output", {})["background_lufs"] = sanitize_lufs(value, DEFAULT_BACKGROUND_LUFS)
        self.save()

    def background_weight(self) -> float:
        output = self.data.get("output", {})
        return sanitize_audio_weight(output.get("background_weight") if isinstance(output, dict) else None, DEFAULT_BACKGROUND_WEIGHT)

    def set_background_weight(self, value: int | float | str) -> None:
        self.data.setdefault("output", {})["background_weight"] = sanitize_audio_weight(value, DEFAULT_BACKGROUND_WEIGHT)
        self.save()

    def lektor_delay_ms(self) -> int:
        output = self.data.get("output", {})
        return sanitize_lektor_delay_ms(output.get("lektor_delay_ms") if isinstance(output, dict) else None)

    def set_lektor_delay_ms(self, value: int | float | str) -> None:
        self.data.setdefault("output", {})["lektor_delay_ms"] = sanitize_lektor_delay_ms(value)
        self.save()

    def create_stereo_for_surround(self) -> bool:
        output = self.data.get("output", {})
        if not isinstance(output, dict):
            return True
        return _coerce_bool(output.get("create_stereo_for_surround", True), True)

    def set_create_stereo_for_surround(self, value: bool) -> None:
        self.data.setdefault("output", {})["create_stereo_for_surround"] = bool(value)
        self.save()

    def last_file_dir(self) -> str:
        ui = self.data.get("ui", {})
        if not isinstance(ui, dict):
            return ""
        return str(ui.get("last_file_dir", "") or "").strip()

    def set_last_file_dir(self, value: str) -> None:
        self.data.setdefault("ui", {})["last_file_dir"] = str(value or "").strip()
        self.save()

    def window_state(self) -> dict[str, Any]:
        ui = self.data.get("ui", {})
        if not isinstance(ui, dict):
            return dict(DEFAULT_CONFIG["ui"]["window"])
        window = ui.get("window", {})
        if not isinstance(window, dict):
            return dict(DEFAULT_CONFIG["ui"]["window"])
        return dict(window)

    def set_window_state(self, width: int, height: int, mode: str) -> None:
        window = self.data.setdefault("ui", {}).setdefault("window", {})
        if not isinstance(window, dict):
            self.data.setdefault("ui", {})["window"] = {}
            window = self.data["ui"]["window"]
        window["width"] = _coerce_int(width, DEFAULT_CONFIG["ui"]["window"]["width"], 900, 6000)
        window["height"] = _coerce_int(height, DEFAULT_CONFIG["ui"]["window"]["height"], 600, 4000)
        window["mode"] = _normalize_window_mode(mode)
        self.save()


def _merge_defaults(defaults: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(defaults))
    for key, value in loaded.items():
        default_value = result.get(key)
        if isinstance(default_value, dict):
            if isinstance(value, dict):
                result[key] = _merge_defaults(default_value, value)
            continue
        else:
            result[key] = value
    return result


def _normalize_config(data: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(data))
    if not isinstance(normalized.get("version"), int) or isinstance(normalized.get("version"), bool):
        normalized["version"] = DEFAULT_CONFIG["version"]

    ui = normalized.setdefault("ui", {})
    if isinstance(ui, dict):
        if not isinstance(ui.get("theme"), str) or not ui.get("theme", "").strip():
            ui["theme"] = DEFAULT_CONFIG["ui"]["theme"]
        else:
            ui["theme"] = ui["theme"].strip()
        if not isinstance(ui.get("last_file_dir"), str):
            ui["last_file_dir"] = DEFAULT_CONFIG["ui"]["last_file_dir"]
        else:
            ui["last_file_dir"] = ui["last_file_dir"].strip()
        window = ui.setdefault("window", {})
        if not isinstance(window, dict):
            window = {}
            ui["window"] = window
        window["width"] = _coerce_int(window.get("width"), DEFAULT_CONFIG["ui"]["window"]["width"], 900, 6000)
        window["height"] = _coerce_int(window.get("height"), DEFAULT_CONFIG["ui"]["window"]["height"], 600, 4000)
        window["mode"] = _normalize_window_mode(window.get("mode", DEFAULT_CONFIG["ui"]["window"]["mode"]))

    tts = normalized.setdefault("tts", {})
    if isinstance(tts, dict):
        if not isinstance(tts.get("last_engine"), str):
            tts["last_engine"] = DEFAULT_CONFIG["tts"]["last_engine"]
        else:
            tts["last_engine"] = tts["last_engine"].strip()

    output = normalized.setdefault("output", {})
    if isinstance(output, dict):
        output["aac_bitrate"] = sanitize_aac_bitrate(output.get("aac_bitrate", DEFAULT_AAC_BITRATE))
        output["lektor_lufs"] = sanitize_lufs(output.get("lektor_lufs", DEFAULT_LEKTOR_LUFS), DEFAULT_LEKTOR_LUFS)
        output["lektor_weight"] = sanitize_audio_weight(output.get("lektor_weight", DEFAULT_LEKTOR_WEIGHT), DEFAULT_LEKTOR_WEIGHT)
        output["background_lufs"] = sanitize_lufs(output.get("background_lufs", DEFAULT_BACKGROUND_LUFS), DEFAULT_BACKGROUND_LUFS)
        output["background_weight"] = sanitize_audio_weight(output.get("background_weight", DEFAULT_BACKGROUND_WEIGHT), DEFAULT_BACKGROUND_WEIGHT)
        output["lektor_delay_ms"] = sanitize_lektor_delay_ms(output.get("lektor_delay_ms", DEFAULT_LEKTOR_DELAY_MS))
        output["create_stereo_for_surround"] = _coerce_bool(output.get("create_stereo_for_surround", True), True)

    return normalized


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "tak", "on"}:
            return True
        if normalized in {"0", "false", "no", "nie", "off"}:
            return False
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        if value in {0, 1}:
            return bool(value)
    return default


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = default
    return max(minimum, min(maximum, number))


def _normalize_window_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"normal", "maximized", "fullscreen"}:
        return mode
    return DEFAULT_CONFIG["ui"]["window"]["mode"]

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
from app.engines.config_schema import WHISPER_QC_MODELS, whisper_qc_effective_compute_type
from app.stt.job import SttSettings
from app.stt.languages import STT_LANGUAGE_CODES
from app.stt.whisper_cpp_runtime import sanitize_whisper_cpp_device, sanitize_whisper_cpp_runtime


DEFAULT_STT_ENGINE = "faster_whisper"
STT_ENGINE_OPTIONS = ("faster_whisper", "whisper_cpp", "whisperx")
DEFAULT_STT_MODEL = "small"
DEFAULT_STT_LANGUAGE = "auto"
DEFAULT_STT_DEVICE = "cpu"
DEFAULT_STT_COMPUTE_TYPE = "int8"
DEFAULT_STT_ACCURACY = "standard"
DEFAULT_STT_VAD_ENABLED = True
DEFAULT_STT_VAD_SENSITIVITY = "standard"
DEFAULT_STT_WHISPER_CPP_RUNTIME = "cpu"
DEFAULT_STT_WHISPER_CPP_DEVICE = "auto"
DEFAULT_STT_WHISPER_CPP_THREADS = 0
DEFAULT_STT_WHISPERX_DEVICE = "cpu"
DEFAULT_STT_WHISPERX_COMPUTE_TYPE = "int8"
DEFAULT_STT_POSTPROCESS_ENABLED = True
DEFAULT_STT_SAVE_PREPARED_AUDIO = False
DEFAULT_STT_SAVE_REPORT = False
DEFAULT_STT_SAVE_LOG = False
STT_ACCURACY_OPTIONS = ("fast", "standard", "accurate")
STT_VAD_SENSITIVITY_OPTIONS = ("gentle", "standard", "strong")


DEFAULT_STT_ENGINE_CONFIGS: dict[str, dict[str, Any]] = {
    "faster_whisper": {
        "model": DEFAULT_STT_MODEL,
        "language": DEFAULT_STT_LANGUAGE,
        "device": DEFAULT_STT_DEVICE,
        "compute_type": DEFAULT_STT_COMPUTE_TYPE,
        "accuracy": DEFAULT_STT_ACCURACY,
        "vad_enabled": DEFAULT_STT_VAD_ENABLED,
        "vad_sensitivity": DEFAULT_STT_VAD_SENSITIVITY,
        "postprocess_enabled": DEFAULT_STT_POSTPROCESS_ENABLED,
        "save_prepared_audio": DEFAULT_STT_SAVE_PREPARED_AUDIO,
        "save_report": DEFAULT_STT_SAVE_REPORT,
        "save_log": DEFAULT_STT_SAVE_LOG,
    },
    "whisper_cpp": {
        "model": DEFAULT_STT_MODEL,
        "language": DEFAULT_STT_LANGUAGE,
        "runtime": DEFAULT_STT_WHISPER_CPP_RUNTIME,
        "device": DEFAULT_STT_WHISPER_CPP_DEVICE,
        "threads": DEFAULT_STT_WHISPER_CPP_THREADS,
        "accuracy": DEFAULT_STT_ACCURACY,
        "vad_enabled": DEFAULT_STT_VAD_ENABLED,
        "vad_sensitivity": DEFAULT_STT_VAD_SENSITIVITY,
        "postprocess_enabled": DEFAULT_STT_POSTPROCESS_ENABLED,
        "save_prepared_audio": DEFAULT_STT_SAVE_PREPARED_AUDIO,
        "save_report": DEFAULT_STT_SAVE_REPORT,
        "save_log": DEFAULT_STT_SAVE_LOG,
    },
    "whisperx": {
        "model": DEFAULT_STT_MODEL,
        "language": DEFAULT_STT_LANGUAGE,
        "device": DEFAULT_STT_WHISPERX_DEVICE,
        "compute_type": DEFAULT_STT_WHISPERX_COMPUTE_TYPE,
        "accuracy": DEFAULT_STT_ACCURACY,
        "vad_enabled": DEFAULT_STT_VAD_ENABLED,
        "vad_sensitivity": DEFAULT_STT_VAD_SENSITIVITY,
        "postprocess_enabled": DEFAULT_STT_POSTPROCESS_ENABLED,
        "save_prepared_audio": DEFAULT_STT_SAVE_PREPARED_AUDIO,
        "save_report": DEFAULT_STT_SAVE_REPORT,
        "save_log": DEFAULT_STT_SAVE_LOG,
    },
}


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
    "stt": {
        "engine": DEFAULT_STT_ENGINE,
        "model": DEFAULT_STT_MODEL,
        "language": DEFAULT_STT_LANGUAGE,
        "device": DEFAULT_STT_DEVICE,
        "compute_type": DEFAULT_STT_COMPUTE_TYPE,
        "accuracy": DEFAULT_STT_ACCURACY,
        "vad_enabled": DEFAULT_STT_VAD_ENABLED,
        "vad_sensitivity": DEFAULT_STT_VAD_SENSITIVITY,
        "whisper_cpp_runtime": DEFAULT_STT_WHISPER_CPP_RUNTIME,
        "whisper_cpp_device": DEFAULT_STT_WHISPER_CPP_DEVICE,
        "whisper_cpp_threads": DEFAULT_STT_WHISPER_CPP_THREADS,
        "whisperx_device": DEFAULT_STT_WHISPERX_DEVICE,
        "whisperx_compute_type": DEFAULT_STT_WHISPERX_COMPUTE_TYPE,
        "whisperx_device_user_set": False,
        "postprocess_enabled": DEFAULT_STT_POSTPROCESS_ENABLED,
        "save_prepared_audio": DEFAULT_STT_SAVE_PREPARED_AUDIO,
        "save_report": DEFAULT_STT_SAVE_REPORT,
        "save_log": DEFAULT_STT_SAVE_LOG,
        "engine_configs": DEFAULT_STT_ENGINE_CONFIGS,
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

    def stt_settings(self) -> SttSettings:
        stt = self.data.get("stt", {})
        if not isinstance(stt, dict):
            stt = {}
        engine = sanitize_stt_engine(stt.get("engine", DEFAULT_STT_ENGINE))
        active = self._stt_engine_config(engine)
        faster_whisper = self._stt_engine_config("faster_whisper")
        whisper_cpp = self._stt_engine_config("whisper_cpp")
        whisperx = self._stt_engine_config("whisperx")
        device = sanitize_stt_device(faster_whisper.get("device", DEFAULT_STT_DEVICE))
        whisperx_device = sanitize_stt_device(whisperx.get("device", DEFAULT_STT_WHISPERX_DEVICE))
        return SttSettings(
            engine=engine,
            model=sanitize_stt_model(active.get("model", DEFAULT_STT_MODEL)),
            language=sanitize_stt_language(active.get("language", DEFAULT_STT_LANGUAGE)),
            device=device,
            compute_type=whisper_qc_effective_compute_type(device, str(faster_whisper.get("compute_type", DEFAULT_STT_COMPUTE_TYPE) or "")),
            accuracy=sanitize_stt_accuracy(active.get("accuracy", DEFAULT_STT_ACCURACY)),
            vad_enabled=_coerce_bool(active.get("vad_enabled", DEFAULT_STT_VAD_ENABLED), DEFAULT_STT_VAD_ENABLED),
            vad_sensitivity=sanitize_stt_vad_sensitivity(active.get("vad_sensitivity", DEFAULT_STT_VAD_SENSITIVITY)),
            whisper_cpp_runtime=sanitize_whisper_cpp_runtime(whisper_cpp.get("runtime", DEFAULT_STT_WHISPER_CPP_RUNTIME)),
            whisper_cpp_device=sanitize_whisper_cpp_device(whisper_cpp.get("device", DEFAULT_STT_WHISPER_CPP_DEVICE)),
            whisper_cpp_threads=sanitize_stt_whisper_cpp_threads(whisper_cpp.get("threads", DEFAULT_STT_WHISPER_CPP_THREADS)),
            whisperx_device=whisperx_device,
            whisperx_compute_type=whisper_qc_effective_compute_type(whisperx_device, str(whisperx.get("compute_type", DEFAULT_STT_WHISPERX_COMPUTE_TYPE) or "")),
            postprocess_enabled=_coerce_bool(active.get("postprocess_enabled", DEFAULT_STT_POSTPROCESS_ENABLED), DEFAULT_STT_POSTPROCESS_ENABLED),
            save_prepared_audio=_coerce_bool(active.get("save_prepared_audio", DEFAULT_STT_SAVE_PREPARED_AUDIO), DEFAULT_STT_SAVE_PREPARED_AUDIO),
            save_report=_coerce_bool(active.get("save_report", DEFAULT_STT_SAVE_REPORT), DEFAULT_STT_SAVE_REPORT),
            save_log=_coerce_bool(active.get("save_log", DEFAULT_STT_SAVE_LOG), DEFAULT_STT_SAVE_LOG),
        )

    def set_stt_engine(self, value: str) -> None:
        self.data.setdefault("stt", {})["engine"] = sanitize_stt_engine(value)
        self.save()

    def set_stt_model(self, value: str) -> None:
        model = sanitize_stt_model(value)
        self._active_stt_engine_config()["model"] = model
        self.data.setdefault("stt", {})["model"] = model
        self.save()

    def set_stt_language(self, value: str) -> None:
        language = sanitize_stt_language(value)
        self._active_stt_engine_config()["language"] = language
        self.data.setdefault("stt", {})["language"] = language
        self.save()

    def set_stt_device(self, value: str) -> None:
        stt = self.data.setdefault("stt", {})
        config = self._stt_engine_config("faster_whisper")
        device = sanitize_stt_device(value)
        config["device"] = device
        config["compute_type"] = whisper_qc_effective_compute_type(device, str(config.get("compute_type", DEFAULT_STT_COMPUTE_TYPE) or ""))
        stt["device"] = device
        stt["compute_type"] = config["compute_type"]
        self.save()

    def set_stt_compute_type(self, value: str) -> None:
        stt = self.data.setdefault("stt", {})
        config = self._stt_engine_config("faster_whisper")
        device = sanitize_stt_device(config.get("device", DEFAULT_STT_DEVICE))
        config["compute_type"] = whisper_qc_effective_compute_type(device, value)
        stt["compute_type"] = config["compute_type"]
        self.save()

    def set_stt_accuracy(self, value: str) -> None:
        accuracy = sanitize_stt_accuracy(value)
        self._active_stt_engine_config()["accuracy"] = accuracy
        self.data.setdefault("stt", {})["accuracy"] = accuracy
        self.save()

    def set_stt_vad_enabled(self, value: bool) -> None:
        self._active_stt_engine_config()["vad_enabled"] = bool(value)
        self.data.setdefault("stt", {})["vad_enabled"] = bool(value)
        self.save()

    def set_stt_vad_sensitivity(self, value: str) -> None:
        sensitivity = sanitize_stt_vad_sensitivity(value)
        self._active_stt_engine_config()["vad_sensitivity"] = sensitivity
        self.data.setdefault("stt", {})["vad_sensitivity"] = sensitivity
        self.save()

    def set_stt_postprocess_enabled(self, value: bool) -> None:
        self._active_stt_engine_config()["postprocess_enabled"] = bool(value)
        self.data.setdefault("stt", {})["postprocess_enabled"] = bool(value)
        self.save()

    def set_stt_whisper_cpp_runtime(self, value: str) -> None:
        runtime = sanitize_whisper_cpp_runtime(value)
        self._stt_engine_config("whisper_cpp")["runtime"] = runtime
        self.data.setdefault("stt", {})["whisper_cpp_runtime"] = runtime
        self.save()

    def set_stt_whisper_cpp_device(self, value: str) -> None:
        device = sanitize_whisper_cpp_device(value)
        self._stt_engine_config("whisper_cpp")["device"] = device
        self.data.setdefault("stt", {})["whisper_cpp_device"] = device
        self.save()

    def set_stt_whisper_cpp_threads(self, value: int | str) -> None:
        threads = sanitize_stt_whisper_cpp_threads(value)
        self._stt_engine_config("whisper_cpp")["threads"] = threads
        self.data.setdefault("stt", {})["whisper_cpp_threads"] = threads
        self.save()

    def set_stt_whisperx_device(self, value: str) -> None:
        stt = self.data.setdefault("stt", {})
        config = self._stt_engine_config("whisperx")
        device = sanitize_stt_device(value)
        config["device"] = device
        config["compute_type"] = whisper_qc_effective_compute_type(device, str(config.get("compute_type", DEFAULT_STT_WHISPERX_COMPUTE_TYPE) or ""))
        stt["whisperx_device"] = device
        stt["whisperx_device_user_set"] = True
        stt["whisperx_compute_type"] = config["compute_type"]
        self.save()

    def set_stt_whisperx_compute_type(self, value: str) -> None:
        stt = self.data.setdefault("stt", {})
        config = self._stt_engine_config("whisperx")
        device = sanitize_stt_device(config.get("device", DEFAULT_STT_WHISPERX_DEVICE))
        config["compute_type"] = whisper_qc_effective_compute_type(device, value)
        stt["whisperx_compute_type"] = config["compute_type"]
        self.save()

    def set_stt_save_prepared_audio(self, value: bool) -> None:
        self._active_stt_engine_config()["save_prepared_audio"] = bool(value)
        self.data.setdefault("stt", {})["save_prepared_audio"] = bool(value)
        self.save()

    def set_stt_save_report(self, value: bool) -> None:
        self._active_stt_engine_config()["save_report"] = bool(value)
        self.data.setdefault("stt", {})["save_report"] = bool(value)
        self.save()

    def set_stt_save_log(self, value: bool) -> None:
        self._active_stt_engine_config()["save_log"] = bool(value)
        self.data.setdefault("stt", {})["save_log"] = bool(value)
        self.save()

    def _active_stt_engine_config(self) -> dict[str, Any]:
        stt = self.data.setdefault("stt", {})
        engine = sanitize_stt_engine(stt.get("engine", DEFAULT_STT_ENGINE)) if isinstance(stt, dict) else DEFAULT_STT_ENGINE
        return self._stt_engine_config(engine)

    def _stt_engine_config(self, engine: str) -> dict[str, Any]:
        stt = self.data.setdefault("stt", {})
        if not isinstance(stt, dict):
            stt = {}
            self.data["stt"] = stt
        engine = sanitize_stt_engine(engine)
        configs = stt.setdefault("engine_configs", {})
        if not isinstance(configs, dict):
            configs = {}
            stt["engine_configs"] = configs
        config = configs.get(engine)
        if not isinstance(config, dict):
            config = _default_stt_engine_config(engine)
            configs[engine] = config
        return config

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

    stt = normalized.setdefault("stt", {})
    if not isinstance(stt, dict):
        stt = json.loads(json.dumps(DEFAULT_CONFIG["stt"]))
        normalized["stt"] = stt
    stt["engine"] = sanitize_stt_engine(stt.get("engine", DEFAULT_STT_ENGINE))
    stt["model"] = sanitize_stt_model(stt.get("model", DEFAULT_STT_MODEL))
    stt["language"] = sanitize_stt_language(stt.get("language", DEFAULT_STT_LANGUAGE))
    stt["device"] = sanitize_stt_device(stt.get("device", DEFAULT_STT_DEVICE))
    stt["compute_type"] = whisper_qc_effective_compute_type(
        stt["device"],
        str(stt.get("compute_type", DEFAULT_STT_COMPUTE_TYPE) or ""),
    )
    stt["accuracy"] = sanitize_stt_accuracy(stt.get("accuracy", DEFAULT_STT_ACCURACY))
    stt["vad_enabled"] = _coerce_bool(stt.get("vad_enabled", DEFAULT_STT_VAD_ENABLED), DEFAULT_STT_VAD_ENABLED)
    stt["vad_sensitivity"] = sanitize_stt_vad_sensitivity(stt.get("vad_sensitivity", DEFAULT_STT_VAD_SENSITIVITY))
    stt["postprocess_enabled"] = _coerce_bool(
        stt.get("postprocess_enabled", DEFAULT_STT_POSTPROCESS_ENABLED),
        DEFAULT_STT_POSTPROCESS_ENABLED,
    )
    stt["whisper_cpp_runtime"] = sanitize_whisper_cpp_runtime(
        stt.get("whisper_cpp_runtime", DEFAULT_STT_WHISPER_CPP_RUNTIME)
    )
    stt["whisper_cpp_device"] = sanitize_whisper_cpp_device(
        stt.get("whisper_cpp_device", DEFAULT_STT_WHISPER_CPP_DEVICE)
    )
    stt["whisper_cpp_threads"] = sanitize_stt_whisper_cpp_threads(stt.get("whisper_cpp_threads", DEFAULT_STT_WHISPER_CPP_THREADS))
    stt["whisperx_device_user_set"] = _coerce_bool(stt.get("whisperx_device_user_set", False), False)
    if stt["whisperx_device_user_set"]:
        stt["whisperx_device"] = sanitize_stt_device(stt.get("whisperx_device", DEFAULT_STT_WHISPERX_DEVICE))
    else:
        stt["whisperx_device"] = DEFAULT_STT_WHISPERX_DEVICE
    stt["whisperx_compute_type"] = whisper_qc_effective_compute_type(
        stt["whisperx_device"],
        str(stt.get("whisperx_compute_type", DEFAULT_STT_WHISPERX_COMPUTE_TYPE) or ""),
    )
    stt["save_prepared_audio"] = _coerce_bool(stt.get("save_prepared_audio", DEFAULT_STT_SAVE_PREPARED_AUDIO), DEFAULT_STT_SAVE_PREPARED_AUDIO)
    stt["save_report"] = _coerce_bool(stt.get("save_report", DEFAULT_STT_SAVE_REPORT), DEFAULT_STT_SAVE_REPORT)
    stt["save_log"] = _coerce_bool(stt.get("save_log", DEFAULT_STT_SAVE_LOG), DEFAULT_STT_SAVE_LOG)
    stt["engine_configs"] = _normalize_stt_engine_configs(stt)

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


def _default_stt_engine_config(engine: str) -> dict[str, Any]:
    engine = sanitize_stt_engine(engine)
    return json.loads(json.dumps(DEFAULT_STT_ENGINE_CONFIGS[engine]))


def _normalize_stt_engine_configs(stt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_configs = stt.get("engine_configs")
    if not isinstance(raw_configs, dict) or (
        _stt_engine_configs_are_defaults(raw_configs) and _legacy_stt_fields_are_customized(stt)
    ):
        raw_configs = _legacy_stt_engine_configs(stt)
    normalized: dict[str, dict[str, Any]] = {}
    for engine in STT_ENGINE_OPTIONS:
        raw_config = raw_configs.get(engine)
        if not isinstance(raw_config, dict):
            raw_config = {}
        merged = _merge_defaults(_default_stt_engine_config(engine), raw_config)
        normalized[engine] = _normalize_stt_engine_config(engine, merged)
    return normalized


def _stt_engine_configs_are_defaults(configs: dict[str, Any]) -> bool:
    for engine in STT_ENGINE_OPTIONS:
        if configs.get(engine) != DEFAULT_STT_ENGINE_CONFIGS[engine]:
            return False
    return True


def _legacy_stt_fields_are_customized(stt: dict[str, Any]) -> bool:
    legacy_defaults = {
        "model": DEFAULT_STT_MODEL,
        "language": DEFAULT_STT_LANGUAGE,
        "device": DEFAULT_STT_DEVICE,
        "compute_type": DEFAULT_STT_COMPUTE_TYPE,
        "accuracy": DEFAULT_STT_ACCURACY,
        "vad_enabled": DEFAULT_STT_VAD_ENABLED,
        "vad_sensitivity": DEFAULT_STT_VAD_SENSITIVITY,
        "postprocess_enabled": DEFAULT_STT_POSTPROCESS_ENABLED,
        "whisper_cpp_runtime": DEFAULT_STT_WHISPER_CPP_RUNTIME,
        "whisper_cpp_device": DEFAULT_STT_WHISPER_CPP_DEVICE,
        "whisper_cpp_threads": DEFAULT_STT_WHISPER_CPP_THREADS,
        "whisperx_compute_type": DEFAULT_STT_WHISPERX_COMPUTE_TYPE,
        "save_prepared_audio": DEFAULT_STT_SAVE_PREPARED_AUDIO,
        "save_report": DEFAULT_STT_SAVE_REPORT,
        "save_log": DEFAULT_STT_SAVE_LOG,
    }
    for key, default in legacy_defaults.items():
        if stt.get(key, default) != default:
            return True
    return _coerce_bool(stt.get("whisperx_device_user_set", False), False)


def _legacy_stt_engine_configs(stt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_common = {
        "model": stt.get("model", DEFAULT_STT_MODEL),
        "language": stt.get("language", DEFAULT_STT_LANGUAGE),
        "accuracy": stt.get("accuracy", DEFAULT_STT_ACCURACY),
        "vad_enabled": stt.get("vad_enabled", DEFAULT_STT_VAD_ENABLED),
        "vad_sensitivity": stt.get("vad_sensitivity", DEFAULT_STT_VAD_SENSITIVITY),
        "postprocess_enabled": stt.get("postprocess_enabled", DEFAULT_STT_POSTPROCESS_ENABLED),
        "save_prepared_audio": stt.get("save_prepared_audio", DEFAULT_STT_SAVE_PREPARED_AUDIO),
        "save_report": stt.get("save_report", DEFAULT_STT_SAVE_REPORT),
        "save_log": stt.get("save_log", DEFAULT_STT_SAVE_LOG),
    }
    whisperx_user_set = _coerce_bool(stt.get("whisperx_device_user_set", False), False)
    whisperx_device = stt.get("whisperx_device", DEFAULT_STT_WHISPERX_DEVICE) if whisperx_user_set else DEFAULT_STT_WHISPERX_DEVICE
    return {
        "faster_whisper": {
            **base_common,
            "device": stt.get("device", DEFAULT_STT_DEVICE),
            "compute_type": stt.get("compute_type", DEFAULT_STT_COMPUTE_TYPE),
        },
        "whisper_cpp": {
            **base_common,
            "runtime": stt.get("whisper_cpp_runtime", DEFAULT_STT_WHISPER_CPP_RUNTIME),
            "device": stt.get("whisper_cpp_device", DEFAULT_STT_WHISPER_CPP_DEVICE),
            "threads": stt.get("whisper_cpp_threads", DEFAULT_STT_WHISPER_CPP_THREADS),
        },
        "whisperx": {
            **base_common,
            "device": whisperx_device,
            "compute_type": stt.get("whisperx_compute_type", DEFAULT_STT_WHISPERX_COMPUTE_TYPE),
        },
    }


def _normalize_stt_engine_config(engine: str, config: dict[str, Any]) -> dict[str, Any]:
    normalized = _default_stt_engine_config(engine)
    normalized.update(config)
    normalized["model"] = sanitize_stt_model(normalized.get("model", DEFAULT_STT_MODEL))
    normalized["language"] = sanitize_stt_language(normalized.get("language", DEFAULT_STT_LANGUAGE))
    normalized["accuracy"] = sanitize_stt_accuracy(normalized.get("accuracy", DEFAULT_STT_ACCURACY))
    normalized["vad_enabled"] = _coerce_bool(normalized.get("vad_enabled", DEFAULT_STT_VAD_ENABLED), DEFAULT_STT_VAD_ENABLED)
    normalized["vad_sensitivity"] = sanitize_stt_vad_sensitivity(normalized.get("vad_sensitivity", DEFAULT_STT_VAD_SENSITIVITY))
    normalized["postprocess_enabled"] = _coerce_bool(
        normalized.get("postprocess_enabled", DEFAULT_STT_POSTPROCESS_ENABLED),
        DEFAULT_STT_POSTPROCESS_ENABLED,
    )
    normalized["save_prepared_audio"] = _coerce_bool(
        normalized.get("save_prepared_audio", DEFAULT_STT_SAVE_PREPARED_AUDIO),
        DEFAULT_STT_SAVE_PREPARED_AUDIO,
    )
    normalized["save_report"] = _coerce_bool(normalized.get("save_report", DEFAULT_STT_SAVE_REPORT), DEFAULT_STT_SAVE_REPORT)
    normalized["save_log"] = _coerce_bool(normalized.get("save_log", DEFAULT_STT_SAVE_LOG), DEFAULT_STT_SAVE_LOG)
    if engine == "whisper_cpp":
        normalized["runtime"] = sanitize_whisper_cpp_runtime(normalized.get("runtime", DEFAULT_STT_WHISPER_CPP_RUNTIME))
        normalized["device"] = sanitize_whisper_cpp_device(normalized.get("device", DEFAULT_STT_WHISPER_CPP_DEVICE))
        normalized["threads"] = sanitize_stt_whisper_cpp_threads(normalized.get("threads", DEFAULT_STT_WHISPER_CPP_THREADS))
    else:
        default_device = DEFAULT_STT_WHISPERX_DEVICE if engine == "whisperx" else DEFAULT_STT_DEVICE
        default_compute = DEFAULT_STT_WHISPERX_COMPUTE_TYPE if engine == "whisperx" else DEFAULT_STT_COMPUTE_TYPE
        normalized["device"] = sanitize_stt_device(normalized.get("device", default_device))
        normalized["compute_type"] = whisper_qc_effective_compute_type(
            normalized["device"],
            str(normalized.get("compute_type", default_compute) or ""),
        )
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


def sanitize_stt_model(value: Any) -> str:
    model = str(value or "").strip()
    return model if model in WHISPER_QC_MODELS else DEFAULT_STT_MODEL


def sanitize_stt_engine(value: Any) -> str:
    engine = str(value or "").strip().lower()
    return engine if engine in STT_ENGINE_OPTIONS else DEFAULT_STT_ENGINE


def sanitize_stt_language(value: Any) -> str:
    language = str(value or "").strip().lower()
    return language if language in STT_LANGUAGE_CODES else DEFAULT_STT_LANGUAGE


def sanitize_stt_accuracy(value: Any) -> str:
    accuracy = str(value or "").strip().lower()
    return accuracy if accuracy in STT_ACCURACY_OPTIONS else DEFAULT_STT_ACCURACY


def sanitize_stt_vad_sensitivity(value: Any) -> str:
    sensitivity = str(value or "").strip().lower()
    return sensitivity if sensitivity in STT_VAD_SENSITIVITY_OPTIONS else DEFAULT_STT_VAD_SENSITIVITY


def sanitize_stt_whisper_cpp_threads(value: Any) -> int:
    return _coerce_int(value, DEFAULT_STT_WHISPER_CPP_THREADS, 0, 64)


def sanitize_stt_device(value: Any) -> str:
    device = str(value or "").strip().lower()
    if device == "cpu" or device == "cuda":
        return device
    if re_match_cuda_device(device):
        return device
    return DEFAULT_STT_DEVICE


def re_match_cuda_device(value: str) -> bool:
    if not value.startswith("cuda:"):
        return False
    suffix = value.split(":", 1)[1]
    return suffix.isdigit()


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

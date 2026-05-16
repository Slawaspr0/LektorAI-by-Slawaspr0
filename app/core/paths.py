from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    app_dir: Path

    @property
    def config_path(self) -> Path:
        return self.app_dir / "config.json"

    @property
    def logs_dir(self) -> Path:
        return self.app_dir / "logs"

    @property
    def runtime_engines_dir(self) -> Path:
        return self.app_dir / "engines"

    @property
    def runtime_stt_dir(self) -> Path:
        return self.app_dir / "stt"

    @property
    def temp_dir(self) -> Path:
        return self.app_dir / "temp"

    @property
    def cache_dir(self) -> Path:
        return self.app_dir / "cache"

    @property
    def whisper_cache_dir(self) -> Path:
        return self.faster_whisper_cache_dir

    @property
    def app_packages_dir(self) -> Path:
        return self.app_dir / "packages"

    def engine_dir(self, engine_id: str) -> Path:
        return self.runtime_engines_dir / engine_id

    def stt_dir(self, stt_id: str) -> Path:
        return self.runtime_stt_dir / stt_id

    @property
    def faster_whisper_stt_dir(self) -> Path:
        return self.stt_dir("faster_whisper")

    @property
    def faster_whisper_packages_dir(self) -> Path:
        return self.faster_whisper_stt_dir / "packages"

    @property
    def faster_whisper_cache_dir(self) -> Path:
        return self.faster_whisper_stt_dir / "cache"

    @property
    def whisper_cpp_stt_dir(self) -> Path:
        return self.stt_dir("whisper_cpp")

    @property
    def whisper_cpp_runtime_bin_dir(self) -> Path:
        return self.whisper_cpp_stt_dir / "bin"

    @property
    def whisper_cpp_runtime_metadata_path(self) -> Path:
        return self.whisper_cpp_stt_dir / "runtime.json"

    @property
    def whisper_cpp_models_dir(self) -> Path:
        return self.whisper_cpp_stt_dir / "models"

    @property
    def whisperx_stt_dir(self) -> Path:
        return self.stt_dir("whisperx")

    @property
    def whisperx_venv_dir(self) -> Path:
        return self.whisperx_stt_dir / "venv"

    @property
    def whisperx_python_path(self) -> Path:
        return self.whisperx_venv_dir / "Scripts" / "python.exe"

    @property
    def whisperx_cache_dir(self) -> Path:
        return self.whisperx_stt_dir / "cache"


def build_paths(app_dir: Path) -> AppPaths:
    return AppPaths(app_dir=app_dir.resolve())

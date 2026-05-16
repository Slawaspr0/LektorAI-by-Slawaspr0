from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Callable

from app.core.paths import AppPaths
from app.stt.cuda_runtime import CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID, cuda_runtime_env, cuda_runtime_ready, ensure_cuda_runtime


PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
WHISPERX_REQUIREMENT = "whisperx>=3.8.5,<4"
WHISPERX_TORCH_REQUIREMENTS = (
    "torch==2.8.0+cu128",
    "torchaudio==2.8.0+cu128",
    "torchvision==0.23.0+cu128",
)
WHISPERX_PROGRESS_RE_TEXT = r"Progress:\s*(\d+(?:\.\d+)?)%"


def whisperx_runtime_ready(paths: AppPaths) -> bool:
    python_path = paths.whisperx_python_path
    if not python_path.is_file():
        return False
    return _can_import_whisperx(python_path)


def ensure_whisperx_runtime(
    paths: AppPaths,
    progress: Callable[[str], None] | None = None,
) -> Path:
    paths.whisperx_stt_dir.mkdir(parents=True, exist_ok=True)
    paths.whisperx_cache_dir.mkdir(parents=True, exist_ok=True)
    python_path = paths.whisperx_python_path
    if whisperx_runtime_ready(paths):
        return python_path

    install_log = paths.whisperx_stt_dir / "install.log"
    if not python_path.is_file():
        _emit(progress, "WhisperX: tworzenie srodowiska - prosze czekac")
        _run_install_step(
            [sys.executable, "-m", "venv", str(paths.whisperx_venv_dir)],
            install_log,
            "Tworzenie srodowiska WhisperX",
        )
    if not python_path.is_file():
        raise RuntimeError("WhisperX: nie udalo sie utworzyc srodowiska.")

    _emit(progress, "WhisperX: instalacja podstawowych pakietow - prosze czekac")
    _run_install_step(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"],
        install_log,
        "Aktualizacja pip/wheel/setuptools",
    )
    _emit(progress, "WhisperX: instalacja PyTorch CU128 - prosze czekac")
    _run_install_step(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--index-url",
            PYTORCH_CU128_INDEX,
            *WHISPERX_TORCH_REQUIREMENTS,
        ],
        install_log,
        "Instalacja PyTorch CU128 dla WhisperX",
    )
    _emit(progress, "WhisperX: instalacja modulu - prosze czekac")
    _run_install_step(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--extra-index-url",
            PYTORCH_CU128_INDEX,
            WHISPERX_REQUIREMENT,
        ],
        install_log,
        "Instalacja WhisperX",
    )
    if not _can_import_whisperx(python_path):
        raise RuntimeError("WhisperX: modul zostal zainstalowany, ale nie mozna go uruchomic. Szczegoly w stt/whisperx/install.log.")
    _emit(progress, "WhisperX: modul gotowy")
    return python_path


def build_whisperx_command(
    python_path: Path,
    input_wav: Path,
    output_dir: Path,
    model: str,
    model_dir: Path,
    language: str = "auto",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 8,
    beam_size: int = 5,
    no_align: bool = False,
) -> list[str]:
    device_type, device_index = whisperx_device_args(device)
    command = [
        str(python_path),
        "-m",
        "whisperx",
        str(input_wav),
        "--model",
        normalize_whisperx_model_name(model),
        "--model_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "json",
        "--device",
        device_type,
        "--device_index",
        str(device_index),
        "--compute_type",
        normalize_whisperx_compute_type(compute_type, device_type),
        "--batch_size",
        str(max(1, int(batch_size or 8))),
        "--beam_size",
        str(max(1, int(beam_size or 5))),
        "--condition_on_previous_text",
        "False",
        "--print_progress",
        "True",
        "--verbose",
        "True",
        "--segment_resolution",
        "sentence",
    ]
    normalized_language = str(language or "auto").strip().lower()
    if normalized_language and normalized_language != "auto":
        command.extend(["--language", normalized_language])
    if no_align:
        command.append("--no_align")
    return command


def whisperx_device_args(device: str) -> tuple[str, int]:
    normalized = str(device or "cpu").strip().lower()
    if normalized == "cuda":
        return "cuda", 0
    if normalized.startswith("cuda:") and normalized.split(":", 1)[1].isdigit():
        return "cuda", int(normalized.split(":", 1)[1])
    return "cpu", 0


def whisperx_device_needs_cuda(device: str) -> bool:
    device_type, _ = whisperx_device_args(device)
    return device_type == "cuda"


def ensure_whisperx_gpu_runtime(paths: AppPaths, progress: Callable[[str], None] | None = None) -> None:
    if cuda_runtime_ready(paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID):
        _emit(progress, "WhisperX: biblioteki GPU gotowe")
        return
    ensure_cuda_runtime(paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID, progress=progress)
    if not cuda_runtime_ready(paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID):
        raise RuntimeError("WhisperX GPU: nie znaleziono wymaganych bibliotek CUDA. Ustaw WhisperX na CPU albo sprobuj ponownie.")
    _emit(progress, "WhisperX: biblioteki GPU gotowe")


def whisperx_runtime_env(paths: AppPaths, device: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if not whisperx_device_needs_cuda(device):
        return env
    return cuda_runtime_env(paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID, env)


def normalize_whisperx_compute_type(compute_type: str, device_type: str = "cpu") -> str:
    normalized = str(compute_type or "").strip().lower()
    if device_type == "cpu":
        return "int8" if normalized != "float32" else "float32"
    return normalized if normalized in {"float16", "float32", "int8"} else "float16"


def normalize_whisperx_model_name(model: str) -> str:
    normalized = str(model or "small").strip()
    if normalized == "turbo":
        return "large-v3-turbo"
    if normalized == "large":
        return "large-v3"
    return normalized or "small"


def _can_import_whisperx(python_path: Path) -> bool:
    if not python_path.is_file():
        return False
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import whisperx; import torch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except Exception:
        return False
    return result.returncode == 0


def _run_install_step(command: list[str], install_log: Path, title: str) -> None:
    install_log.parent.mkdir(parents=True, exist_ok=True)
    with install_log.open("a", encoding="utf-8") as log:
        log.write(f"\n== {title} ==\n")
        log.write("> " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log.write(result.stdout or "")
        log.write(f"\nKod wyjscia: {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"WhisperX: instalacja nie powiodla sie. Szczegoly w stt/whisperx/install.log.")


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)

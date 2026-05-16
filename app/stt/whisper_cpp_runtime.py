from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.download import download_file_with_progress
from app.core.paths import AppPaths
from app.stt.cuda_runtime import CUDA_RUNTIME_WHISPER_CPP_ID, cuda_runtime_env, ensure_cuda_runtime


WHISPER_CPP_RELEASE_TAG = "whispercpp-windows-x64-v1"
WHISPER_CPP_RELEASE_BASE_URL = (
    "https://github.com/Slawaspr0/LektorAI-by-Slawaspr0/releases/download/"
    f"{WHISPER_CPP_RELEASE_TAG}"
)
WHISPER_CPP_MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
WHISPER_CPP_EXECUTABLE_NAMES = (
    "whisper-cli.exe",
    "main.exe",
)


@dataclass(frozen=True)
class WhisperCppRuntimePackage:
    variant: str
    label: str
    archive_name: str
    url: str


WHISPER_CPP_RUNTIME_PACKAGES: dict[str, WhisperCppRuntimePackage] = {
    "cpu": WhisperCppRuntimePackage(
        variant="cpu",
        label="CPU",
        archive_name="whispercpp-cpu-avx2-windows-x64.zip",
        url=f"{WHISPER_CPP_RELEASE_BASE_URL}/whispercpp-cpu-avx2-windows-x64.zip",
    ),
    "cuda": WhisperCppRuntimePackage(
        variant="cuda",
        label="CUDA",
        archive_name="whispercpp-cuda13-modern-windows-x64.zip",
        url=f"{WHISPER_CPP_RELEASE_BASE_URL}/whispercpp-cuda13-modern-windows-x64.zip",
    ),
}


def whisper_cpp_model_file_name(model_name: str) -> str:
    model = normalize_whisper_cpp_model_name(model_name)
    return f"ggml-{model}.bin"


def normalize_whisper_cpp_model_name(model_name: str) -> str:
    model = str(model_name or "small").strip()
    if model == "turbo":
        model = "large-v3-turbo"
    elif model == "large":
        model = "large-v3"
    return model or "small"


def sanitize_whisper_cpp_runtime(value: str) -> str:
    runtime = str(value or "").strip().lower()
    return runtime if runtime in WHISPER_CPP_RUNTIME_PACKAGES else "cpu"


def sanitize_whisper_cpp_device(value: str) -> str:
    device = str(value or "").strip().lower()
    if device in {"auto", "cpu", "cuda"}:
        return device
    if device.startswith("cuda:") and device.split(":", 1)[1].isdigit():
        return device
    return "auto"


def whisper_cpp_runtime_download_label(package: WhisperCppRuntimePackage) -> str:
    return f"whisper.cpp: pobieranie plikow programu ({package.label})"


def find_whisper_cpp_executable(paths: AppPaths) -> Path | None:
    candidates: list[Path] = []
    for name in WHISPER_CPP_EXECUTABLE_NAMES:
        candidates.extend(
            [
                paths.whisper_cpp_runtime_bin_dir / name,
                paths.whisper_cpp_stt_dir / name,
                paths.app_dir / name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for name in WHISPER_CPP_EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def ensure_whisper_cpp_runtime(
    paths: AppPaths,
    variant: str = "cpu",
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    variant = sanitize_whisper_cpp_runtime(variant)
    if whisper_cpp_runtime_ready(paths, variant):
        executable = find_whisper_cpp_executable(paths)
        if executable is not None:
            if variant == "cuda":
                ensure_cuda_runtime(
                    paths,
                    CUDA_RUNTIME_WHISPER_CPP_ID,
                    progress=progress,
                    cancel_requested=cancel_requested,
                )
            return executable
    package = WHISPER_CPP_RUNTIME_PACKAGES[variant]
    _emit(progress, f"whisper.cpp: przygotowanie runtime {package.label}")
    archive_path = _obtain_runtime_archive(paths, package, progress=progress, cancel_requested=cancel_requested)
    _raise_if_cancelled(cancel_requested)
    _emit(progress, f"whisper.cpp: instalacja runtime {package.label}")
    install_whisper_cpp_runtime(paths, package, archive_path)
    executable = find_whisper_cpp_executable(paths)
    if executable is None:
        raise RuntimeError("Nie udalo sie przygotowac whisper.cpp runtime.")
    if variant == "cuda":
        ensure_cuda_runtime(
            paths,
            CUDA_RUNTIME_WHISPER_CPP_ID,
            progress=progress,
            cancel_requested=cancel_requested,
        )
    _emit(progress, f"whisper.cpp: runtime {package.label} gotowy")
    return executable


def whisper_cpp_runtime_env(paths: AppPaths, variant: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if sanitize_whisper_cpp_runtime(variant) != "cuda":
        return env
    return cuda_runtime_env(paths, CUDA_RUNTIME_WHISPER_CPP_ID, env)


def whisper_cpp_runtime_ready(paths: AppPaths, variant: str = "cpu") -> bool:
    executable = find_whisper_cpp_executable(paths)
    if executable is None:
        return False
    metadata = read_whisper_cpp_runtime_metadata(paths)
    if not metadata:
        return True
    expected_variant = sanitize_whisper_cpp_runtime(variant)
    if str(metadata.get("variant", "") or "").strip().lower() != expected_variant:
        return False
    required = [
        paths.whisper_cpp_runtime_bin_dir / "whisper.dll",
        paths.whisper_cpp_runtime_bin_dir / "ggml.dll",
    ]
    if expected_variant == "cuda":
        required.append(paths.whisper_cpp_runtime_bin_dir / "ggml-cuda.dll")
    return all(path.is_file() for path in required)


def read_whisper_cpp_runtime_metadata(paths: AppPaths) -> dict[str, object]:
    try:
        path = paths.whisper_cpp_runtime_metadata_path
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def install_whisper_cpp_runtime(paths: AppPaths, package: WhisperCppRuntimePackage, archive_path: Path) -> None:
    bin_dir = paths.whisper_cpp_runtime_bin_dir
    bin_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.exe", "*.dll", "*.lib", "*.exp"):
        for path in bin_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute() or any(part in {"", ".", ".."} for part in member_path.parts):
                raise RuntimeError(f"Nieprawidlowa sciezka w paczce whisper.cpp: {member.filename}")
            target = bin_dir / member_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    paths.whisper_cpp_runtime_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    paths.whisper_cpp_runtime_metadata_path.write_text(
        json.dumps(
            {
                "variant": package.variant,
                "label": package.label,
                "archive": package.archive_name,
                "release": WHISPER_CPP_RELEASE_TAG,
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def find_whisper_cpp_model(paths: AppPaths, model_name: str) -> Path | None:
    file_name = whisper_cpp_model_file_name(model_name)
    candidates = [
        paths.whisper_cpp_models_dir / file_name,
        paths.whisper_cpp_stt_dir / file_name,
        paths.app_dir / "models" / file_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ensure_whisper_cpp_model(
    paths: AppPaths,
    model_name: str,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    existing = find_whisper_cpp_model(paths, model_name)
    if existing is not None:
        return existing
    model = normalize_whisper_cpp_model_name(model_name)
    file_name = whisper_cpp_model_file_name(model)
    target = paths.whisper_cpp_models_dir / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{WHISPER_CPP_MODEL_BASE_URL}/{file_name}"
    label = f"whisper.cpp: pobieranie modelu {model}"
    part_path = target.with_suffix(target.suffix + ".part")
    try:
        download_file_with_progress(
            url,
            part_path,
            label=label,
            progress=progress,
            cancel_requested=cancel_requested,
            timeout_s=30.0,
        )
        if target.exists():
            target.unlink()
        part_path.replace(target)
    except Exception:
        try:
            part_path.unlink()
        except OSError:
            pass
        raise
    _emit(progress, f"whisper.cpp: model {model} gotowy")
    return target


def build_whisper_cpp_command(
    exe_path: Path,
    model_path: Path,
    input_wav: Path,
    output_base: Path,
    language: str = "auto",
    threads: int = 0,
    device: str = "auto",
) -> list[str]:
    command = [
        str(exe_path),
        "-m",
        str(model_path),
        "-f",
        str(input_wav),
        "-osrt",
        "-of",
        str(output_base),
        "-np",
        "-pp",
        "-sns",
    ]
    normalized_language = str(language or "auto").strip().lower() or "auto"
    command.extend(["-l", normalized_language])
    if int(threads or 0) > 0:
        command.extend(["-t", str(int(threads))])
    normalized_device = sanitize_whisper_cpp_device(device)
    if normalized_device == "cpu":
        command.append("-ng")
    elif normalized_device.startswith("cuda:"):
        command.extend(["-dev", normalized_device.split(":", 1)[1]])
    return command


def _obtain_runtime_archive(
    paths: AppPaths,
    package: WhisperCppRuntimePackage,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    local_archive = find_local_runtime_archive(paths, package.archive_name)
    if local_archive is not None:
        _emit(progress, f"whisper.cpp: uzywam lokalnej paczki {package.archive_name}")
        return local_archive
    download_dir = paths.whisper_cpp_stt_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / package.archive_name
    download_file_with_progress(
        package.url,
        target,
        label=whisper_cpp_runtime_download_label(package),
        progress=progress,
        cancel_requested=cancel_requested,
        timeout_s=30.0,
    )
    return target


def find_local_runtime_archive(paths: AppPaths, archive_name: str) -> Path | None:
    candidates = (
        paths.app_dir / "Releases" / archive_name,
        paths.app_dir.parent / "Releases" / archive_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Przerwano przez uzytkownika")

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


CUDA_RUNTIME_RELEASE_TAG = "cuda-runtime-win-x64-v1"
CUDA_RUNTIME_RELEASE_BASE_URL = (
    "https://github.com/Slawaspr0/LektorAI-by-Slawaspr0/releases/download/"
    f"{CUDA_RUNTIME_RELEASE_TAG}"
)
CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID = "cuda12-ctranslate2-pytorch-win-x64"
CUDA_RUNTIME_WHISPER_CPP_ID = "cuda13-whispercpp-win-x64"
ENV_CUDA_RUNTIME_DIRS = "LEKTORAI_CUDA_RUNTIME_DIRS"


@dataclass(frozen=True)
class CudaRuntimePackage:
    package_id: str
    label: str
    archive_name: str
    url: str
    required_dlls: tuple[str, ...]


CUDA_RUNTIME_PACKAGES: dict[str, CudaRuntimePackage] = {
    CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID: CudaRuntimePackage(
        package_id=CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID,
        label="CUDA 12 dla faster-whisper",
        archive_name="cuda12-ctranslate2-pytorch-win-x64.zip",
        url=f"{CUDA_RUNTIME_RELEASE_BASE_URL}/cuda12-ctranslate2-pytorch-win-x64.zip",
        required_dlls=("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudart64_12.dll"),
    ),
    CUDA_RUNTIME_WHISPER_CPP_ID: CudaRuntimePackage(
        package_id=CUDA_RUNTIME_WHISPER_CPP_ID,
        label="CUDA 13 dla whisper.cpp",
        archive_name="cuda13-whispercpp-win-x64.zip",
        url=f"{CUDA_RUNTIME_RELEASE_BASE_URL}/cuda13-whispercpp-win-x64.zip",
        required_dlls=("cublas64_13.dll", "cublasLt64_13.dll", "cudart64_13.dll"),
    ),
}


def cuda_runtime_dll_dir(paths: AppPaths, package_id: str) -> Path:
    package = _runtime_package(package_id)
    return paths.cuda_runtime_pack_dir(package.package_id)


def cuda_runtime_ready(paths: AppPaths, package_id: str) -> bool:
    package = _runtime_package(package_id)
    dll_dir = cuda_runtime_dll_dir(paths, package.package_id)
    return all((dll_dir / dll_name).is_file() for dll_name in package.required_dlls)


def cuda_runtime_env(paths: AppPaths, package_id: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    dll_dir = cuda_runtime_dll_dir(paths, package_id)
    if not dll_dir.is_dir():
        return env
    _prepend_env_path(env, "PATH", str(dll_dir))
    existing_dirs = env.get(ENV_CUDA_RUNTIME_DIRS, "")
    env[ENV_CUDA_RUNTIME_DIRS] = str(dll_dir) if not existing_dirs else str(dll_dir) + os.pathsep + existing_dirs
    return env


def ensure_cuda_runtime(
    paths: AppPaths,
    package_id: str,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    package = _runtime_package(package_id)
    if cuda_runtime_ready(paths, package.package_id):
        return cuda_runtime_dll_dir(paths, package.package_id)
    _emit(progress, f"CUDA Runtime: przygotowanie {package.label}")
    archive_path = _obtain_runtime_archive(paths, package, progress=progress, cancel_requested=cancel_requested)
    _raise_if_cancelled(cancel_requested)
    _emit(progress, f"CUDA Runtime: instalacja {package.label}")
    install_cuda_runtime(paths, package, archive_path)
    if not cuda_runtime_ready(paths, package.package_id):
        missing = ", ".join(
            dll_name for dll_name in package.required_dlls if not (cuda_runtime_dll_dir(paths, package.package_id) / dll_name).is_file()
        )
        raise RuntimeError(f"CUDA Runtime: paczka {package.label} nie zawiera wymaganych plikow: {missing}")
    _emit(progress, f"CUDA Runtime: {package.label} gotowy")
    return cuda_runtime_dll_dir(paths, package.package_id)


def install_cuda_runtime(paths: AppPaths, package: CudaRuntimePackage, archive_path: Path) -> None:
    root_dir = paths.cuda_runtime_root_dir.resolve()
    target_dir = paths.cuda_runtime_pack_dir(package.package_id).resolve()
    if root_dir not in target_dir.parents:
        raise RuntimeError("CUDA Runtime: nieprawidlowa sciezka instalacji")
    temp_dir = target_dir.with_name(target_dir.name + ".tmp")
    _safe_rmtree(temp_dir, root_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute() or any(part in {"", ".", ".."} for part in member_path.parts):
                raise RuntimeError(f"Nieprawidlowa sciezka w paczce CUDA Runtime: {member.filename}")
            target = temp_dir / member_path.name
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    metadata_path = temp_dir / "runtime.json"
    metadata_path.write_text(
        json.dumps(
            {
                "package_id": package.package_id,
                "label": package.label,
                "archive": package.archive_name,
                "release": CUDA_RUNTIME_RELEASE_TAG,
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _safe_rmtree(target_dir, root_dir)
    temp_dir.replace(target_dir)


def find_local_cuda_runtime_archive(paths: AppPaths, archive_name: str) -> Path | None:
    candidates = (
        paths.app_dir / "Releases" / archive_name,
        paths.app_dir.parent / "Releases" / archive_name,
        paths.app_dir / "cuda_runtime_packs" / archive_name,
        paths.app_dir.parent / "cuda_runtime_packs" / archive_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _obtain_runtime_archive(
    paths: AppPaths,
    package: CudaRuntimePackage,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    local_archive = find_local_cuda_runtime_archive(paths, package.archive_name)
    if local_archive is not None:
        _emit(progress, f"CUDA Runtime: uzywam lokalnej paczki {package.archive_name}")
        return local_archive
    target = paths.cuda_runtime_downloads_dir / package.archive_name
    part_path = target.with_suffix(target.suffix + ".part")
    try:
        download_file_with_progress(
            package.url,
            part_path,
            label=f"CUDA Runtime: pobieranie {package.label}",
            progress=progress,
            cancel_requested=cancel_requested,
            timeout_s=30.0,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        part_path.replace(target)
        return target
    except Exception:
        try:
            part_path.unlink()
        except OSError:
            pass
        raise


def _runtime_package(package_id: str) -> CudaRuntimePackage:
    normalized = str(package_id or "").strip()
    package = CUDA_RUNTIME_PACKAGES.get(normalized)
    if package is None:
        raise RuntimeError(f"Nieznana paczka CUDA Runtime: {package_id}")
    return package


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    current = env.get(key, "")
    entries = [part for part in current.split(os.pathsep) if part]
    if value in entries:
        return
    env[key] = value if not current else value + os.pathsep + current


def _safe_rmtree(path: Path, root_dir: Path) -> None:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if root_dir not in resolved.parents:
        raise RuntimeError("CUDA Runtime: odmowa usuniecia katalogu poza runtime")
    if resolved.exists():
        shutil.rmtree(resolved)


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Przerwano przez uzytkownika")

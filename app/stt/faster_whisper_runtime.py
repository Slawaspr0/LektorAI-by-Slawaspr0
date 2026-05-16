from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

from app.core.paths import AppPaths
from app.stt.cuda_runtime import (
    CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID,
    cuda_runtime_dll_dir,
    cuda_runtime_ready,
    ensure_cuda_runtime,
)


ENV_PACKAGES_DIRS = "LEKTORAI_STT_FASTER_WHISPER_PACKAGES_DIRS"
ENV_CACHE_DIR = "LEKTORAI_STT_FASTER_WHISPER_CACHE_DIR"
LEGACY_ENV_PACKAGES_DIR = "LEKTORAI_APP_PACKAGES_DIR"
LEGACY_ENV_CACHE_DIR = "LEKTORAI_WHISPER_CACHE_DIR"
ENV_DLL_DIRS = "LEKTORAI_STT_FASTER_WHISPER_DLL_DIRS"
STT_REQUIREMENTS = ("faster-whisper>=1.2,<2", "PyYAML>=6.0.3,<7")
_CUDA_DLL_SUBDIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("nvidia", "cublas", "bin"), ("cublas64_12.dll", "cublasLt64_12.dll")),
    (("nvidia", "cudnn", "bin"), ("cudnn64_9.dll",)),
    (("nvidia", "cuda_runtime", "bin"), ("cudart64_12.dll",)),
    (("torch", "lib"), ("cublas64_12.dll", "cudnn64_9.dll", "cudart64_12.dll")),
)
_DLL_DIRECTORY_HANDLES: list[object] = []


def faster_whisper_package_dirs(paths: AppPaths) -> tuple[Path, ...]:
    dirs = [paths.faster_whisper_packages_dir]
    venv_site_packages = paths.faster_whisper_stt_dir / "venv" / "Lib" / "site-packages"
    dirs.append(venv_site_packages)
    if paths.app_packages_dir != paths.faster_whisper_packages_dir:
        dirs.append(paths.app_packages_dir)
    return tuple(_unique_paths(dirs))


def faster_whisper_package_dirs_for_cache(cache_dir: Path) -> tuple[Path, ...]:
    stt_dir = cache_dir.parent
    dirs = [
        stt_dir / "packages",
        stt_dir / "venv" / "Lib" / "site-packages",
    ]
    try:
        app_dir = stt_dir.parent.parent
    except Exception:
        app_dir = None
    if app_dir is not None:
        dirs.append(app_dir / "packages")
    return tuple(_unique_paths(dirs))


def faster_whisper_worker_env(paths: AppPaths) -> dict[str, str]:
    package_dirs = os.pathsep.join(str(path) for path in faster_whisper_package_dirs(paths))
    cache_dir = str(paths.faster_whisper_cache_dir)
    env = {
        ENV_PACKAGES_DIRS: package_dirs,
        ENV_CACHE_DIR: cache_dir,
        LEGACY_ENV_PACKAGES_DIR: str(paths.app_packages_dir),
        LEGACY_ENV_CACHE_DIR: cache_dir,
    }
    dll_dirs = faster_whisper_cuda_dll_dirs(paths)
    if dll_dirs:
        dll_text = os.pathsep.join(str(path) for path in dll_dirs)
        env[ENV_DLL_DIRS] = dll_text
        current_path = os.environ.get("PATH", "")
        env["PATH"] = dll_text if not current_path else dll_text + os.pathsep + current_path
    return env


def faster_whisper_device_needs_cuda(device: str) -> bool:
    normalized = str(device or "").strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def faster_whisper_cuda_dll_dirs_for_package_dirs(package_dirs: Iterable[Path]) -> tuple[Path, ...]:
    dirs: list[Path] = []
    for package_dir in package_dirs:
        for relative_parts, dll_names in _CUDA_DLL_SUBDIRS:
            candidate = package_dir.joinpath(*relative_parts)
            if candidate.is_dir() and any((candidate / dll_name).is_file() for dll_name in dll_names):
                dirs.append(candidate)
    return tuple(_unique_paths(dirs))


def faster_whisper_cuda_dll_dirs(paths: AppPaths) -> tuple[Path, ...]:
    dirs: list[Path] = []
    if cuda_runtime_ready(paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID):
        dirs.append(cuda_runtime_dll_dir(paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID))
    dirs.extend(faster_whisper_cuda_dll_dirs_for_package_dirs(faster_whisper_package_dirs(paths)))
    dirs.extend(_external_torch_cuda_dll_dirs(paths))
    return tuple(_unique_paths(dirs))


def ensure_faster_whisper_dll_search_path_for_package_dirs(package_dirs: Iterable[Path]) -> None:
    ensure_faster_whisper_dll_search_path(faster_whisper_cuda_dll_dirs_for_package_dirs(package_dirs))


def ensure_faster_whisper_dll_search_path(dll_dirs: Iterable[Path]) -> None:
    for dll_dir in dll_dirs:
        dll_text = str(dll_dir)
        if dll_text not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = dll_text + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is None:
            continue
        try:
            _DLL_DIRECTORY_HANDLES.append(add_dll_directory(dll_text))
        except OSError:
            continue


def ensure_faster_whisper_import_path(paths: AppPaths) -> None:
    for package_dir in reversed(faster_whisper_package_dirs(paths)):
        if not package_dir.is_dir():
            continue
        package_text = str(package_dir)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
    ensure_faster_whisper_dll_search_path(faster_whisper_cuda_dll_dirs(paths))


def ensure_faster_whisper_import_path_for_cache(cache_dir: Path) -> None:
    for package_dir in reversed(faster_whisper_package_dirs_for_cache(cache_dir)):
        if not package_dir.is_dir():
            continue
        package_text = str(package_dir)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
    ensure_faster_whisper_dll_search_path_for_package_dirs(faster_whisper_package_dirs_for_cache(cache_dir))


def import_faster_whisper(paths: AppPaths):
    ensure_faster_whisper_import_path(paths)
    return importlib.import_module("faster_whisper")


def import_faster_whisper_for_cache(cache_dir: Path):
    ensure_faster_whisper_import_path_for_cache(cache_dir)
    return importlib.import_module("faster_whisper")


def ensure_faster_whisper_runtime(paths: AppPaths, progress: Callable[[str], None] | None = None, device: str = "cpu") -> None:
    if _can_import_from_portable_packages(paths):
        if faster_whisper_device_needs_cuda(device):
            ensure_faster_whisper_gpu_runtime(paths, progress)
        return
    packages_dir = paths.faster_whisper_packages_dir
    install_log = paths.faster_whisper_stt_dir / "install.log"
    packages_dir.mkdir(parents=True, exist_ok=True)
    paths.faster_whisper_cache_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "Whisper QC: instalacja modulu kontroli mowy - prosze czekac")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(packages_dir),
        "--prefer-binary",
        *STT_REQUIREMENTS,
    ]
    env = _pip_install_env(paths)
    with install_log.open("w", encoding="utf-8") as log:
        log.write("Instalacja modulu STT faster-whisper dla Whisper QC.\n")
        log.write("> " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        log.write(result.stdout or "")
        log.write(f"\nKod wyjscia: {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError("Whisper QC: nie udalo sie zainstalowac modulu kontroli mowy. Szczegoly w stt/faster_whisper/install.log.")
    importlib.invalidate_caches()
    if not _can_import_from_portable_packages(paths):
        raise RuntimeError("Whisper QC: modul kontroli mowy zostal zainstalowany, ale nie mozna go uruchomic. Szczegoly w stt/faster_whisper/install.log.")
    if faster_whisper_device_needs_cuda(device):
        ensure_faster_whisper_gpu_runtime(paths, progress)
    _emit(progress, "Whisper QC: modul kontroli mowy gotowy")


def ensure_faster_whisper_gpu_runtime(paths: AppPaths, progress: Callable[[str], None] | None = None) -> None:
    if _has_faster_whisper_cuda_runtime(paths):
        ensure_faster_whisper_dll_search_path(faster_whisper_cuda_dll_dirs(paths))
        return
    ensure_cuda_runtime(paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID, progress=progress)
    if not _has_faster_whisper_cuda_runtime(paths):
        raise RuntimeError("Whisper QC GPU: nie znaleziono wymaganych bibliotek CUDA 12. Ustaw Whisper QC na CPU albo sprobuj ponownie.")
    ensure_faster_whisper_dll_search_path(faster_whisper_cuda_dll_dirs(paths))
    _emit(progress, "Whisper QC: biblioteki GPU gotowe")


def faster_whisper_import_problem(paths: AppPaths) -> str:
    ensure_faster_whisper_import_path(paths)
    if importlib.util.find_spec("faster_whisper") is None:
        return "faster-whisper"
    try:
        importlib.import_module("faster_whisper")
    except ModuleNotFoundError as exc:
        return str(exc.name or "faster-whisper")
    except Exception as exc:
        return f"faster-whisper ({type(exc).__name__})"
    return ""


def faster_whisper_missing_message() -> str:
    return "Whisper QC: brak silnika STT faster-whisper. Zainstaluj wymagania STT albo wylacz Whisper QC."


def _can_import_from_portable_packages(paths: AppPaths) -> bool:
    package_dirs = [path for path in faster_whisper_package_dirs(paths) if path.is_dir()]
    if not package_dirs:
        return False
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in package_dirs)
    dll_dirs = faster_whisper_cuda_dll_dirs(paths)
    if dll_dirs:
        dll_text = os.pathsep.join(str(path) for path in dll_dirs)
        env[ENV_DLL_DIRS] = dll_text
        env["PATH"] = dll_text + os.pathsep + env.get("PATH", "")
    script = "import faster_whisper; from faster_whisper import WhisperModel"
    try:
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=30,
        )
    except Exception:
        return False
    return result.returncode == 0


def _has_faster_whisper_cuda_runtime(paths: AppPaths) -> bool:
    dll_dirs = faster_whisper_cuda_dll_dirs(paths)
    found_names = {path.name.lower() for dll_dir in dll_dirs for path in dll_dir.glob("*.dll")}
    return {"cublas64_12.dll", "cudnn64_9.dll", "cudart64_12.dll"}.issubset(found_names)


def _external_torch_cuda_dll_dirs(paths: AppPaths) -> tuple[Path, ...]:
    candidates: list[Path] = []
    candidates.append(paths.whisperx_venv_dir / "Lib" / "site-packages" / "torch" / "lib")
    engines_dir = paths.runtime_engines_dir
    if engines_dir.is_dir():
        for engine_dir in engines_dir.iterdir():
            candidates.append(engine_dir / "venv" / "Lib" / "site-packages" / "torch" / "lib")
    return tuple(path for path in _unique_paths(candidates) if path.is_dir() and (path / "cublas64_12.dll").is_file())


def _pip_install_env(paths: AppPaths) -> dict[str, str]:
    env = dict(os.environ)
    temp_dir = paths.faster_whisper_stt_dir / "temp" / "pip"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_text = str(temp_dir)
    env["TMP"] = temp_text
    env["TEMP"] = temp_text
    env["TMPDIR"] = temp_text
    compat_dir = paths.app_dir / "app" / "engines" / "pip_compat"
    if compat_dir.is_dir():
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(compat_dir) if not current_pythonpath else str(compat_dir) + os.pathsep + current_pythonpath
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        pass


def _unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result

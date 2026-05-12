from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.core.paths import AppPaths


ENV_PACKAGES_DIRS = "LEKTORAI_STT_FASTER_WHISPER_PACKAGES_DIRS"
ENV_CACHE_DIR = "LEKTORAI_STT_FASTER_WHISPER_CACHE_DIR"
LEGACY_ENV_PACKAGES_DIR = "LEKTORAI_APP_PACKAGES_DIR"
LEGACY_ENV_CACHE_DIR = "LEKTORAI_WHISPER_CACHE_DIR"
STT_REQUIREMENTS = ("faster-whisper>=1.2,<2", "PyYAML>=6.0.3,<7")


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
    return {
        ENV_PACKAGES_DIRS: package_dirs,
        ENV_CACHE_DIR: cache_dir,
        LEGACY_ENV_PACKAGES_DIR: str(paths.app_packages_dir),
        LEGACY_ENV_CACHE_DIR: cache_dir,
    }


def ensure_faster_whisper_import_path(paths: AppPaths) -> None:
    for package_dir in reversed(faster_whisper_package_dirs(paths)):
        if not package_dir.is_dir():
            continue
        package_text = str(package_dir)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)


def ensure_faster_whisper_import_path_for_cache(cache_dir: Path) -> None:
    for package_dir in reversed(faster_whisper_package_dirs_for_cache(cache_dir)):
        if not package_dir.is_dir():
            continue
        package_text = str(package_dir)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)


def import_faster_whisper(paths: AppPaths):
    ensure_faster_whisper_import_path(paths)
    return importlib.import_module("faster_whisper")


def import_faster_whisper_for_cache(cache_dir: Path):
    ensure_faster_whisper_import_path_for_cache(cache_dir)
    return importlib.import_module("faster_whisper")


def ensure_faster_whisper_runtime(paths: AppPaths, progress: Callable[[str], None] | None = None) -> None:
    if _can_import_from_portable_packages(paths):
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
        *STT_REQUIREMENTS,
    ]
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
        )
        log.write(result.stdout or "")
        log.write(f"\nKod wyjscia: {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError("Whisper QC: nie udalo sie zainstalowac modulu kontroli mowy. Szczegoly w stt/faster_whisper/install.log.")
    importlib.invalidate_caches()
    if not _can_import_from_portable_packages(paths):
        raise RuntimeError("Whisper QC: modul kontroli mowy zostal zainstalowany, ale nie mozna go uruchomic. Szczegoly w stt/faster_whisper/install.log.")
    _emit(progress, "Whisper QC: modul kontroli mowy gotowy")


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

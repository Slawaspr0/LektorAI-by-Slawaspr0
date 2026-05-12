from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.paths import AppPaths


@dataclass(frozen=True)
class LogCleanupOption:
    option_id: str
    label: str
    description: str


@dataclass(frozen=True)
class LogCleanupResult:
    files_removed: int = 0
    dirs_removed: int = 0
    errors: tuple[str, ...] = ()


LOG_CLEANUP_OPTIONS = (
    LogCleanupOption(
        "app_logs",
        "Logi aplikacji",
        "Usuwa logi dzialania aplikacji.",
    ),
    LogCleanupOption(
        "engine_logs",
        "Logi silnikow TTS",
        "Usuwa logi pracy silnikow TTS.",
    ),
    LogCleanupOption(
        "install_logs",
        "Logi instalacji TTS",
        "Usuwa logi instalacji silnikow TTS.",
    ),
    LogCleanupOption(
        "engine_temp",
        "Foldery temp silnikow TTS",
        "Usuwa tymczasowe pliki pracy silnikow TTS.",
    ),
)


def cleanup_option_labels() -> dict[str, str]:
    return {option.option_id: option.label for option in LOG_CLEANUP_OPTIONS}


def cleanup_logs(
    paths: AppPaths,
    selected_options: set[str],
    active_app_log_path: Path | None = None,
) -> LogCleanupResult:
    selected = {str(option) for option in selected_options}
    app_root = paths.app_dir.resolve()
    active_app_log = _active_app_log(paths, active_app_log_path)
    files_removed = 0
    dirs_removed = 0
    errors: list[str] = []

    def remove_file(path: Path) -> None:
        nonlocal files_removed
        if not _is_inside(path, app_root):
            errors.append(f"Pominieto sciezke poza aplikacja: {path}")
            return
        try:
            if path.is_file():
                path.unlink()
                files_removed += 1
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    def remove_dir(path: Path) -> None:
        nonlocal dirs_removed
        if not _is_inside(path, app_root):
            errors.append(f"Pominieto sciezke poza aplikacja: {path}")
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
                dirs_removed += 1
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    if "app_logs" in selected:
        for path in _files_in_dir(paths.logs_dir, {".log"}):
            if _same_path(path, active_app_log):
                continue
            remove_file(path)

    engine_dirs = _engine_dirs(paths)
    if "engine_logs" in selected:
        for engine_dir in engine_dirs:
            logs_dir = engine_dir / "logs"
            for path in _files_in_dir(logs_dir, {".log", ".json"}):
                if path.name.endswith(".log") or path.name.endswith(".analysis.json"):
                    remove_file(path)

    if "install_logs" in selected:
        for engine_dir in engine_dirs:
            remove_file(engine_dir / "install.log")

    if "engine_temp" in selected:
        for engine_dir in engine_dirs:
            remove_dir(engine_dir / "temp")

    return LogCleanupResult(files_removed=files_removed, dirs_removed=dirs_removed, errors=tuple(errors))


def preview_log_cleanup(paths: AppPaths, active_app_log_path: Path | None = None) -> dict[str, int]:
    engine_dirs = _engine_dirs(paths)
    active_app_log = _active_app_log(paths, active_app_log_path)
    return {
        "app_logs": sum(1 for path in _files_in_dir(paths.logs_dir, {".log"}) if not _same_path(path, active_app_log)),
        "engine_logs": sum(
            1
            for engine_dir in engine_dirs
            for path in _files_in_dir(engine_dir / "logs", {".log", ".json"})
            if path.name.endswith(".log") or path.name.endswith(".analysis.json")
        ),
        "install_logs": sum(1 for engine_dir in engine_dirs if (engine_dir / "install.log").is_file()),
        "engine_temp": sum(1 for engine_dir in engine_dirs if (engine_dir / "temp").is_dir()),
    }


def _engine_dirs(paths: AppPaths) -> list[Path]:
    if not paths.runtime_engines_dir.is_dir():
        return []
    return [path for path in paths.runtime_engines_dir.iterdir() if path.is_dir()]


def _files_in_dir(directory: Path, suffixes: set[str]):
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except Exception:
        return False


def _active_app_log(paths: AppPaths, active_app_log_path: Path | None) -> Path | None:
    if active_app_log_path is None:
        return None
    try:
        active = active_app_log_path.resolve()
        active.relative_to(paths.logs_dir.resolve())
        return active
    except Exception:
        return None


def _same_path(path: Path, other: Path | None) -> bool:
    if other is None:
        return False
    try:
        return path.resolve() == other.resolve()
    except Exception:
        return False

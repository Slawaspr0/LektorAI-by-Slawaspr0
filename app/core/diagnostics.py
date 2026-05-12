from __future__ import annotations

import sys

from app.core.media_tools import BINARY_LOOKUP_HINT, find_ffmpeg, find_ffprobe, find_mkvmerge
from app.core.paths import AppPaths
from app.core.version import APP_NAME, APP_VERSION
from app.engines.manager import EngineManager
from app.engines.schemas import EngineStatus
from app.stt.faster_whisper_runtime import faster_whisper_import_problem


def collect_diagnostics(paths: AppPaths, manager: EngineManager) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    rows.append(("Aplikacja", "OK", f"{APP_NAME} {APP_VERSION}"))
    rows.append(("Python", "OK", f"{sys.version.split()[0]} | {sys.executable}"))
    rows.append(_path_row("Folder aplikacji", paths.app_dir))
    rows.append(_path_row("Config", paths.config_path))
    rows.append(_path_row("Logi", paths.logs_dir))
    rows.append(_runtime_engines_row(paths))
    rows.extend(_app_package_rows(("edge_tts", "openai"), paths))
    rows.append(_stt_faster_whisper_row(paths))
    rows.append(_binary_row("ffmpeg", find_ffmpeg(paths)))
    rows.append(_binary_row("ffprobe", find_ffprobe(paths)))
    rows.append(_binary_row("mkvmerge", find_mkvmerge(paths)))
    for state in manager.list_states():
        if state.status in {EngineStatus.READY, EngineStatus.REQUIRES_INTERNET}:
            status = "OK"
        elif state.selectable:
            status = "UWAGA"
        else:
            status = "NIE"
        components = ", ".join(state.components)
        detail = f"{state.reason} | {components}" if state.reason and components else state.reason or components
        rows.append((f"TTS: {state.definition.display_name}", status, detail))
    return rows


def format_diagnostics(rows: list[tuple[str, str, str]]) -> str:
    if not rows:
        return ""
    name_width = min(32, max(len(row[0]) for row in rows))
    status_width = min(10, max(len(row[1]) for row in rows))
    lines = []
    for name, status, detail in rows:
        lines.append(f"{name:<{name_width}}  {status:<{status_width}}  {detail}")
    return "\n".join(lines)


def _path_row(name, path) -> tuple[str, str, str]:
    return (name, "OK" if path.exists() else "brak", str(path))


def _runtime_engines_row(paths: AppPaths) -> tuple[str, str, str]:
    if paths.runtime_engines_dir.exists():
        return ("Runtime TTS", "OK", str(paths.runtime_engines_dir))
    return ("Runtime TTS", "OK", f"{paths.runtime_engines_dir} | utworzy sie przy konfiguracji/instalacji TTS")


def _binary_row(name, path, optional: bool = False) -> tuple[str, str, str]:
    if path:
        return (name, "OK", str(path))
    status = "opcjonalny" if optional else "brak"
    detail = f"opcjonalny | {BINARY_LOOKUP_HINT}" if optional else BINARY_LOOKUP_HINT
    return (name, status, detail)


def _app_package_rows(package_names: tuple[str, ...], paths: AppPaths) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for name in package_names:
        import importlib.util
        found = importlib.util.find_spec(name) is not None
        detail = "import OK" if found else f"brak importu | {sys.executable} -m pip install -r {paths.app_dir / 'requirements.txt'}"
        rows.append((f"Pakiet: {name}", "OK" if found else "brak", detail))
    return rows


def _stt_faster_whisper_row(paths: AppPaths) -> tuple[str, str, str]:
    problem = faster_whisper_import_problem(paths)
    if not problem:
        return ("STT: faster-whisper", "OK", str(paths.faster_whisper_stt_dir))
    return ("STT: faster-whisper", "brak", f"{problem} | program przygotuje modul przy pierwszym uzyciu kontroli mowy")

from __future__ import annotations

from app.core.media_tools import find_ffmpeg, find_ffprobe, find_mkvmerge
from app.core.paths import AppPaths
from app.core.version import APP_NAME
from app.engines.manager import EngineManager
from app.engines.schemas import EngineKind
from app.stt.faster_whisper_runtime import faster_whisper_import_problem


def build_preflight_report(paths: AppPaths, manager: EngineManager) -> tuple[bool, list[str]]:
    states = manager.list_states()
    selectable = [state for state in states if state.selectable]
    local_missing = [state.definition.display_name for state in states if state.definition.kind == EngineKind.LOCAL and not state.selectable]
    blockers: list[str] = []
    warnings: list[str] = []

    if find_ffmpeg(paths) is None:
        blockers.append("Brak ffmpeg.")
    if find_ffprobe(paths) is None:
        blockers.append("Brak ffprobe.")
    if find_mkvmerge(paths) is None:
        blockers.append("Brak mkvmerge.")
    if not selectable:
        blockers.append("Brak gotowego silnika TTS.")
    whisper_missing = faster_whisper_import_problem(paths)
    if whisper_missing:
        warnings.append(f"Whisper QC: modul kontroli mowy nie jest jeszcze przygotowany ({whisper_missing}). Program zainstaluje go przy pierwszym uzyciu kontroli mowy")
    if local_missing:
        warnings.append("Lokalne TTS niezainstalowane: " + ", ".join(local_missing))

    lines = [f"Preflight {APP_NAME}"]
    lines.append(f"Silniki gotowe: {len(selectable)}")
    if selectable:
        lines.append("Gotowe TTS: " + ", ".join(state.definition.display_name for state in selectable))
    if blockers:
        lines.append("Blokery:")
        lines.extend(f"- {item}" for item in blockers)
    if warnings:
        lines.append("Uwagi:")
        lines.extend(f"- {item}" for item in warnings)
    if not blockers:
        lines.append("Status: OK do podstawowego testu aplikacji")
    else:
        lines.append("Status: wymaga poprawy przed testem")
    return (not blockers, lines)

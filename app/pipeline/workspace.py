from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core.logging import safe_name

SOURCE_STEM_LIMIT = 54

ENGINE_SHORT_CODES = {
    "chatterbox": "CTB",
    "omnivoice": "OMV",
    "coqui_xtts": "XTTS",
    "piper": "PIP",
    "edge": "EDG",
    "openai": "OAI",
}


def lektorai_workspace_for(source_path: Path) -> Path:
    return source_path.resolve().parent / "LektorAI"


def engine_short_code(engine_id: str) -> str:
    normalized = str(engine_id or "").strip().lower()
    if normalized in ENGINE_SHORT_CODES:
        return ENGINE_SHORT_CODES[normalized]
    fallback = safe_name(normalized).replace("_", "")
    return (fallback[:6] or "TTS").upper()


def compact_run_timestamp(created_at: datetime | None = None) -> str:
    value = created_at or datetime.now()
    return value.strftime("%y%m%d_%H%M%S")


def compact_source_stem(source_path: Path, limit: int = SOURCE_STEM_LIMIT) -> str:
    text = safe_name(source_path.stem)
    if len(text) <= limit:
        return text or "plik"
    return text[:limit].rstrip("._-") or "plik"


def next_output_stem(
    workspace: Path,
    source_path: Path,
    engine_id: str,
    created_at: datetime | None = None,
) -> str:
    base = f"{compact_run_timestamp(created_at)}_{compact_source_stem(source_path)}_{engine_short_code(engine_id)}"
    candidate = base
    counter = 2
    while (
        (workspace / f"{candidate}.mkv").exists()
        or (workspace / f"{candidate}.mp4").exists()
        or (workspace / f"{candidate}.srt").exists()
    ):
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def lektor_assets_dir(workspace: Path, output_stem: str) -> Path:
    return workspace / output_stem

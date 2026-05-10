from __future__ import annotations

from pathlib import Path

from app.core.logging import safe_name


def lektorai_workspace_for(source_path: Path) -> Path:
    return source_path.resolve().parent / "LektorAI"


def next_output_stem(workspace: Path, source_path: Path, engine_id: str) -> str:
    base = f"{safe_name(source_path.stem)}_{safe_name(engine_id)}"
    candidate = base
    counter = 2
    while (
        (workspace / f"{candidate}.mkv").exists()
        or (workspace / f"{candidate}.mp4").exists()
        or (workspace / f"{candidate}.srt").exists()
        or (workspace / f"{candidate}_lektor").exists()
    ):
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def lektor_assets_dir(workspace: Path, output_stem: str) -> Path:
    return workspace / f"{output_stem}_lektor"

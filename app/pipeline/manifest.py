from __future__ import annotations

import csv
from pathlib import Path

from app.pipeline.subtitles import SubtitleSegment


def write_segments_manifest(
    path: Path,
    segments: list[SubtitleSegment],
    generated_segments: list[tuple[int, Path]],
) -> None:
    audio_paths = [audio_path for _start_ms, audio_path in generated_segments]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            [
                "ordinal",
                "srt_index",
                "start_ms",
                "end_ms",
                "duration_ms",
                "audio_file",
                "text",
            ]
        )
        for ordinal, segment in enumerate(segments, 1):
            audio_path = audio_paths[ordinal - 1] if ordinal <= len(audio_paths) else None
            writer.writerow(
                [
                    ordinal,
                    segment.index,
                    segment.start_ms,
                    segment.end_ms,
                    max(0, int(segment.end_ms) - int(segment.start_ms)),
                    audio_path.name if audio_path is not None else "",
                    segment.text,
                ]
            )


def write_skipped_segments_manifest(path: Path, segments: list[SubtitleSegment], reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["index", "start_ms", "end_ms", "duration_ms", "reason"])
        for segment in segments:
            writer.writerow(
                [
                    segment.index,
                    segment.start_ms,
                    segment.end_ms,
                    max(0, int(segment.end_ms) - int(segment.start_ms)),
                    reason,
                ]
            )

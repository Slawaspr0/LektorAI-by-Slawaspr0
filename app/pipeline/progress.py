from __future__ import annotations

import re


FILE_PROGRESS_TOTAL = 1000
PROGRESS_MARKER_PREFIX = "__LEKTORAI_PROGRESS__|"

_STAGE_RANGES = {
    "prepare": (0, 50),
    "tts": (50, 750),
    "manifest": (750, 770),
    "audio_qc": (770, 800),
    "timeline": (800, 850),
    "normalization": (850, 930),
    "encoding": (930, 960),
    "mux": (960, 980),
    "summary": (980, 1000),
    "done": (1000, 1000),
}


def progress_value_for_stage(stage: str, ratio: float | None = None) -> int:
    start, end = _STAGE_RANGES.get(str(stage or "").strip(), _STAGE_RANGES["prepare"])
    if ratio is None:
        return int(round((start + end) / 2))
    ratio = max(0.0, min(1.0, float(ratio)))
    return int(round(start + ((end - start) * ratio)))


def safe_unit_eta_seconds(done: int, total: int, elapsed_seconds: float, minimum_done: int = 5) -> float | None:
    done = max(0, int(done))
    total = max(0, int(total))
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    if total <= 0 or done <= 0 or done < int(minimum_done) or done >= total:
        return None
    return round((elapsed_seconds / done) * (total - done), 1)


def ffmpeg_progress_ratio(line: str, duration_seconds: float | None) -> float | None:
    duration = float(duration_seconds or 0.0)
    if duration <= 0:
        return None
    seconds = _ffmpeg_progress_seconds(line)
    if seconds is None:
        return None
    return round(max(0.0, min(1.0, seconds / duration)), 4)


def format_progress_status(stage: str, detail: str = "", elapsed_seconds: float | None = None, eta_seconds: float | None = None) -> str:
    text = str(stage or "").strip() or "Aktualny plik"
    detail = str(detail or "").strip()
    if detail:
        text = f"{text}: {detail}"
    if elapsed_seconds is not None:
        text += f" | czas {_format_duration(elapsed_seconds)}"
    if eta_seconds is not None:
        text += f" | ETA {_format_duration(eta_seconds)}"
    return text


def encode_progress_marker(stage: str, ratio: float | None = None, label: str = "") -> str:
    ratio_text = "" if ratio is None else f"{max(0.0, min(1.0, float(ratio))):.4f}"
    return PROGRESS_MARKER_PREFIX + "|".join((str(stage or "").strip(), ratio_text, str(label or "").strip()))


def decode_progress_marker(message: str) -> tuple[str, float | None, str] | None:
    text = str(message or "")
    if not text.startswith(PROGRESS_MARKER_PREFIX):
        return None
    payload = text[len(PROGRESS_MARKER_PREFIX) :]
    parts = payload.split("|", 2)
    while len(parts) < 3:
        parts.append("")
    stage = parts[0].strip()
    ratio = None
    if parts[1].strip():
        try:
            ratio = max(0.0, min(1.0, float(parts[1])))
        except ValueError:
            ratio = None
    label = parts[2].strip()
    return stage, ratio, label


def _ffmpeg_progress_seconds(line: str) -> float | None:
    text = str(line or "").strip()
    match = re.match(r"out_time_(?:us|ms)=(\d+)", text)
    if match:
        return int(match.group(1)) / 1_000_000.0
    match = re.match(r"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return (hours * 3600) + (minutes * 60) + seconds
    return None


def _format_duration(seconds: float) -> str:
    seconds_i = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds_i, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}min {seconds_part:02d}s"
    if minutes:
        return f"{minutes}min {seconds_part:02d}s"
    return f"{seconds_part}s"

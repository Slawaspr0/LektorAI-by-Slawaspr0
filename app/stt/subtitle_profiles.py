from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SttSubtitleProfile:
    profile_id: str
    label: str
    max_line_chars: int
    max_lines: int
    max_chars: int
    max_cps: float
    min_duration_ms: int


FALLBACK_SUBTITLE_PROFILE = SttSubtitleProfile(
    profile_id="global",
    label="globalny",
    max_line_chars=42,
    max_lines=1,
    max_chars=64,
    max_cps=0.0,
    min_duration_ms=800,
)

ENGLISH_USA_SUBTITLE_PROFILE = SttSubtitleProfile(
    profile_id="en-us",
    label="English USA",
    max_line_chars=42,
    max_lines=1,
    max_chars=84,
    max_cps=0.0,
    min_duration_ms=800,
)


def subtitle_profile_for_language(language: str | None) -> SttSubtitleProfile:
    normalized = str(language or "").strip().lower().replace("_", "-")
    if normalized in {"en", "eng", "english", "english-us", "en-us", "en-usa"}:
        return ENGLISH_USA_SUBTITLE_PROFILE
    return FALLBACK_SUBTITLE_PROFILE

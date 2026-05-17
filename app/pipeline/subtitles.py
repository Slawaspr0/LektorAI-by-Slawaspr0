from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUBTITLE_EXTENSIONS = (".srt", ".txt")


@dataclass(frozen=True)
class SubtitleSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str


_TTS_TEXT_TRANSLATION = str.maketrans(
    {
        "\u00ab": '"',
        "\u00bb": '"',
        "\u201e": '"',
        "\u201c": '"',
        "\u201d": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u201a": "'",
        "\u2018": "'",
        "\u2019": "'",
        "\u2032": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)

_TTS_MOJIBAKE_REPLACEMENTS = (
    ("â€ž", '"'),
    ("â€œ", '"'),
    ("â€", '"'),
    ("â€™", "'"),
    ("â€˜", "'"),
    ("â€“", "-"),
    ("â€”", "-"),
    ("â€¦", "..."),
    ("Ã¢â‚¬â„¢", "'"),
)

_TTS_BAD_ENCODING_ARTIFACTS = (
    "\ufffd",
    "Ã‚",
    "Ãƒ",
    "Ã…",
    "Ä¹",
)


def apply_dictionary(text: str, dictionary: dict[str, str]) -> str:
    text = clean_subtitle_text(text)
    rules = sorted(dictionary.items(), key=lambda item: (-len(str(item[0])), str(item[0]).lower()))
    for source, replacement in rules:
        source = str(source).strip()
        if len(source) < 2 or not re.search(r"\w", source, flags=re.UNICODE):
            continue
        literal_replacement = str(replacement).strip()
        text = re.sub(
            rf"\b{re.escape(source)}\b",
            lambda _match, value=literal_replacement: value,
            text,
            flags=re.IGNORECASE,
        )
    return normalize_subtitle_spacing(text)


def clean_subtitle_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"\\+[Nn]", " ", text)
    text = re.sub(r"\{\\[^}]+\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u266a", " ").replace("\u266b", " ")
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    text = re.sub(r"(^|\s)[\-–—]\s*", r"\1", text)
    return normalize_subtitle_spacing(text)


def normalize_subtitle_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_tts_text(text: str) -> str:
    text = html.unescape(str(text))
    for source, replacement in _TTS_MOJIBAKE_REPLACEMENTS:
        text = text.replace(source, replacement)
    for artifact in _TTS_BAD_ENCODING_ARTIFACTS:
        text = text.replace(artifact, "")
    text = re.sub(r"\\+[NnHh]", " ", text)
    text = re.sub(r"\{\\[^}]+\}", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\((?:jap\.?|niem\.?|ang\.?|fr\.?|hiszp\.?|wlos\.?|ros\.?|po [^)]+)\)", " ", text, flags=re.IGNORECASE)
    text = text.translate(_TTS_TEXT_TRANSLATION)
    text = re.sub(r"^\s*[\-–—]\s*", "", text)
    text = re.sub(r"^\s*[\w .'\-ĄĆĘŁŃÓŚŹŻąćęłńóśźż]{2,40}:\s*", "", text)
    text = re.sub(r"\s*&\s*", " i ", text)
    text = re.sub(r"(?:->|←|→|↑|↓|↔|⇒|⇐|⇑|⇓|➜|➡)", " ", text)
    text = re.sub(r"[♪♫♬★•◆●■□▲▼♦♣♠♥✓✔✕✖]", " ", text)
    text = re.sub(r"\s*(?:\.{3,})\s*$", "", text)
    text = re.sub(r"\s*(?:\.{3,})\s*", ", ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)
    return normalize_subtitle_spacing(text)


def load_srt(path: Path) -> list[SubtitleSegment]:
    content = _read_text(path)
    blocks = re.split(r"\n\s*\n", content.strip())
    segments: list[SubtitleSegment] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line_index = 1 if "-->" in lines[1] else 0
        if "-->" not in lines[time_line_index]:
            continue
        if time_line_index > 0:
            try:
                segment_index = int(lines[0].strip())
            except ValueError:
                segment_index = len(segments) + 1
        else:
            segment_index = len(segments) + 1
        start_raw, end_raw = [part.strip() for part in lines[time_line_index].split("-->", 1)]
        text = " ".join(lines[time_line_index + 1 :]).strip()
        if not text:
            continue
        segments.append(
            SubtitleSegment(
                index=segment_index,
                start_ms=_timestamp_to_ms(start_raw),
                end_ms=_timestamp_to_ms(end_raw),
                text=text,
            )
        )
    return segments


def load_txt_as_segment(path: Path) -> list[SubtitleSegment]:
    text = _read_text(path).strip()
    if not text:
        return []
    return [SubtitleSegment(index=1, start_ms=0, end_ms=max(1000, len(text) * 60), text=text)]


def save_srt(path: Path, segments: list[SubtitleSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, segment in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ms_to_timestamp(segment.start_ms)} --> {_ms_to_timestamp(segment.end_ms)}")
        lines.append(segment.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "windows-1250", "iso-8859-2", "cp1250"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _timestamp_to_ms(value: str) -> int:
    value = value.strip().replace(",", ".")
    match = re.match(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", value)
    if not match:
        return 0
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    fraction = (match.group(4) or "")[:3].ljust(3, "0")
    millis = int(fraction) if fraction else 0
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def _ms_to_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    seconds = (ms % 60_000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

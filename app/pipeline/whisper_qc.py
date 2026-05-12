from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.stt.faster_whisper_runtime import (
    faster_whisper_missing_message,
    import_faster_whisper_for_cache,
)


@dataclass(frozen=True)
class WhisperQCResult:
    text: str
    similarity: float
    score: int
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.warnings


_WHISPER_MODELS: dict[tuple[str, str, str, str], object] = {}


def score_whisper_transcript(expected_text: str, transcript: str, threshold: float = 0.62) -> WhisperQCResult:
    similarity = text_similarity(expected_text, transcript)
    penalty = whisper_similarity_penalty(similarity, threshold)
    warnings: list[str] = []
    if penalty:
        warnings.append("whisper niezgodny")
    if not normalize_for_whisper_qc(transcript):
        warnings.append("whisper pusty")
        penalty = max(penalty, 85)
    return WhisperQCResult(
        text=str(transcript or "").strip(),
        similarity=round(float(similarity), 4),
        score=int(penalty),
        warnings=tuple(warnings),
    )


def transcribe_audio_with_faster_whisper(audio_path: Path, settings: dict, cache_dir: Path) -> str:
    try:
        faster_whisper = import_faster_whisper_for_cache(cache_dir)
        WhisperModel = faster_whisper.WhisperModel
    except ModuleNotFoundError as exc:
        if exc.name == "faster_whisper":
            raise RuntimeError(faster_whisper_missing_message()) from exc
        raise

    model_name = str(settings.get("whisper_qc_model", "small") or "small").strip() or "small"
    language = str(settings.get("whisper_qc_language", "pl") or "pl").strip() or "pl"
    device = str(settings.get("whisper_qc_device", "cpu") or "cpu").strip() or "cpu"
    compute_type = str(settings.get("whisper_qc_compute_type", "int8") or "int8").strip() or "int8"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (model_name, device, compute_type, str(cache_dir))
    model = _WHISPER_MODELS.get(key)
    if model is None:
        model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=str(cache_dir))
        _WHISPER_MODELS[key] = model
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        beam_size=1,
        vad_filter=False,
        word_timestamps=False,
        without_timestamps=True,
    )
    return " ".join(str(segment.text or "").strip() for segment in segments).strip()


def text_similarity(expected_text: str, transcript: str) -> float:
    expected = normalize_for_whisper_qc(expected_text)
    actual = normalize_for_whisper_qc(transcript)
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    return round(float(SequenceMatcher(None, expected, actual).ratio()), 4)


def whisper_similarity_penalty(similarity: float, threshold: float) -> int:
    similarity = _bounded_float(similarity, 0.0, 1.0)
    threshold = _bounded_float(threshold, 0.0, 1.0)
    if similarity >= threshold:
        return 0
    gap = threshold - similarity
    return max(15, min(95, int(round(gap * 140))))


def normalize_for_whisper_qc(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").casefold())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", without_marks)).strip()

def _bounded_float(value, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = minimum
    if number != number or number in {float("inf"), float("-inf")}:
        number = minimum
    return max(minimum, min(maximum, number))

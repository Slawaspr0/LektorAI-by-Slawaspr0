from __future__ import annotations

import csv
import math
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.media_tools import convert_audio_to_wav


QC_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class AudioQCResult:
    index: int
    path: Path
    duration_ms: int
    subtitle_ms: int
    peak_db: float
    rms_db: float
    leading_db: float
    trailing_db: float
    clipped_ratio: float
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.warnings


def analyze_generated_segments(
    ffmpeg: Path,
    generated_segments: list[tuple[int, Path]],
    subtitles,
    report_path: Path,
    temp_dir: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> list[AudioQCResult]:
    if len(generated_segments) != len(subtitles):
        raise RuntimeError(
            f"Audio QC: liczba segmentow audio ({len(generated_segments)}) nie zgadza sie z napisami ({len(subtitles)})."
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        _remove_dir(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        results: list[AudioQCResult] = []
        for idx, (_start_ms, audio_path) in enumerate(generated_segments):
            _raise_if_cancelled(cancel_requested)
            subtitle = subtitles[idx]
            temp_wav = temp_dir / f"qc_{idx + 1:05d}.wav"
            convert_audio_to_wav(ffmpeg, audio_path, temp_wav, cancel_requested=cancel_requested)
            results.append(
                analyze_wav_segment(
                    index=int(getattr(subtitle, "index", idx + 1)),
                    path=audio_path,
                    wav_path=temp_wav,
                    subtitle_ms=max(0, int(subtitle.end_ms) - int(subtitle.start_ms)),
                )
            )
        write_audio_qc_report(report_path, results)
        return results
    finally:
        _remove_dir(temp_dir)


def analyze_audio_candidate(
    ffmpeg: Path,
    audio_path: Path,
    subtitle,
    temp_wav: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> AudioQCResult:
    _raise_if_cancelled(cancel_requested)
    convert_audio_to_wav(ffmpeg, audio_path, temp_wav, cancel_requested=cancel_requested)
    return analyze_wav_segment(
        index=int(getattr(subtitle, "index", 0)),
        path=audio_path,
        wav_path=temp_wav,
        subtitle_ms=max(0, int(subtitle.end_ms) - int(subtitle.start_ms)),
    )


def analyze_wav_segment(index: int, path: Path, wav_path: Path, subtitle_ms: int) -> AudioQCResult:
    samples, sample_rate = _read_wav_mono_i16(wav_path)
    duration_ms = int(round((len(samples) / sample_rate) * 1000)) if sample_rate > 0 else 0
    peak = max((abs(sample) for sample in samples), default=0) / 32768.0
    rms = _rms(samples)
    leading_db = _window_rms_db(samples, sample_rate, 0, 250)
    trailing_db = _window_rms_db(samples, sample_rate, max(0, duration_ms - 250), duration_ms)
    clipped = sum(1 for sample in samples if abs(sample) >= 32760)
    clipped_ratio = clipped / len(samples) if samples else 0.0

    warnings: list[str] = []
    if not samples or duration_ms <= 0:
        warnings.append("pusty segment")
    if samples and _db(rms) < -55.0:
        warnings.append("prawie cisza")
    if subtitle_ms >= 500 and duration_ms < max(120, int(subtitle_ms * 0.35)):
        warnings.append("podejrzanie krotki")
    if subtitle_ms >= 500 and duration_ms > int(subtitle_ms * 2.4) + 1200:
        warnings.append("podejrzanie dlugi")
    if duration_ms > 80 and leading_db > -12.0:
        warnings.append("glosny poczatek")
    if duration_ms > 80 and trailing_db > -12.0:
        warnings.append("glosny koniec")
    if clipped_ratio > 0.002:
        warnings.append("clipping")

    return AudioQCResult(
        index=index,
        path=path,
        duration_ms=duration_ms,
        subtitle_ms=subtitle_ms,
        peak_db=_db(peak),
        rms_db=_db(rms),
        leading_db=leading_db,
        trailing_db=trailing_db,
        clipped_ratio=clipped_ratio,
        warnings=tuple(warnings),
    )


def write_audio_qc_report(path: Path, results: list[AudioQCResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            [
                "index",
                "file",
                "duration_ms",
                "subtitle_ms",
                "peak_db",
                "rms_db",
                "leading_db",
                "trailing_db",
                "clipped_ratio",
                "score",
                "warnings",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.index,
                    result.path.name,
                    result.duration_ms,
                    result.subtitle_ms,
                    f"{result.peak_db:.2f}",
                    f"{result.rms_db:.2f}",
                    f"{result.leading_db:.2f}",
                    f"{result.trailing_db:.2f}",
                    f"{result.clipped_ratio:.6f}",
                    score_audio_qc(result),
                    " | ".join(result.warnings),
                ]
            )


def summarize_audio_qc(results: list[AudioQCResult]) -> str:
    warning_count = sum(1 for result in results if result.warnings)
    if warning_count == 0:
        return "Audio QC: brak podejrzanych segmentow"
    first = [str(result.index) for result in results if result.warnings][:8]
    suffix = "" if warning_count <= len(first) else f" +{warning_count - len(first)}"
    return f"Audio QC: podejrzane segmenty {', '.join(first)}{suffix}"


def score_audio_qc(result: AudioQCResult) -> int:
    score = 0
    for warning in result.warnings:
        if warning == "pusty segment":
            score += 100
        elif warning == "prawie cisza":
            score += 85
        elif warning == "podejrzanie krotki":
            score += 70
        elif warning == "podejrzanie dlugi":
            score += 45
        elif warning in {"glosny poczatek", "glosny koniec"}:
            score += 25
        elif warning == "clipping":
            score += 35
        else:
            score += 10
    return score


def _read_wav_mono_i16(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if channels != 1 or sample_width != 2:
            raise RuntimeError(f"Nieprawidlowy WAV QC: {path}")
        data = wav.readframes(wav.getnframes())
    samples = array("h")
    samples.frombytes(data)
    return samples, int(sample_rate)


def _rms(samples: array) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        value = int(sample) / 32768.0
        total += value * value
    return math.sqrt(total / len(samples))


def _window_rms_db(samples: array, sample_rate: int, start_ms: int, end_ms: int) -> float:
    if not samples or sample_rate <= 0 or end_ms <= start_ms:
        return -120.0
    start = max(0, int(start_ms * sample_rate / 1000))
    end = min(len(samples), int(end_ms * sample_rate / 1000))
    if end <= start:
        return -120.0
    window = samples[start:end]
    return _db(_rms(window))


def _db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(max(value, 1e-12))


def _remove_dir(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Przerwano przez uzytkownika")

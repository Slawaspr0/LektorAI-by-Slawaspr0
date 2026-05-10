from __future__ import annotations

import shutil
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.media_tools import convert_audio_to_wav


SAMPLE_RATE = 48000
SEGMENT_EDGE_FADE_MS = 6


@dataclass(frozen=True)
class TimelineBuildStats:
    shifted_count: int = 0
    max_shift_ms: int = 0


def build_lektor_wav(
    ffmpeg: Path,
    segment_paths: list[tuple[int, Path]],
    output_wav: Path,
    temp_dir: Path,
    minimum_duration_s: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> TimelineBuildStats:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    track = array("h")
    shifted_count = 0
    max_shift_ms = 0
    queue_cursor_sample = 0

    try:
        for index, (start_ms, segment_path) in enumerate(sorted(segment_paths, key=lambda item: int(item[0])), 1):
            _raise_if_cancelled(cancel_requested)
            temp_wav = temp_dir / f"segment_{index:05d}.wav"
            samples = _read_segment_samples(ffmpeg, segment_path, temp_wav, cancel_requested=cancel_requested)
            _apply_segment_edge_fade(samples, SAMPLE_RATE, SEGMENT_EDGE_FADE_MS)
            desired_start_sample = max(0, int(start_ms * SAMPLE_RATE / 1000))
            start_sample = max(desired_start_sample, queue_cursor_sample)
            if start_sample > desired_start_sample:
                shift_ms = int(round((start_sample - desired_start_sample) * 1000 / SAMPLE_RATE))
                shifted_count += 1
                max_shift_ms = max(max_shift_ms, shift_ms)
            required = start_sample + len(samples)
            if len(track) < required:
                track.extend([0] * (required - len(track)))
            _mix_into(track, samples, start_sample)
            queue_cursor_sample = max(queue_cursor_sample, required)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not track:
        track.extend([0] * SAMPLE_RATE)
    if minimum_duration_s is not None:
        _raise_if_cancelled(cancel_requested)
        minimum_samples = int(max(0.0, float(minimum_duration_s)) * SAMPLE_RATE)
        if len(track) < minimum_samples:
            track.extend([0] * (minimum_samples - len(track)))

    _raise_if_cancelled(cancel_requested)
    _write_wav_mono_i16(output_wav, track)
    return TimelineBuildStats(shifted_count=shifted_count, max_shift_ms=max_shift_ms)


def _read_segment_samples(
    ffmpeg: Path,
    segment_path: Path,
    temp_wav: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> array:
    try:
        return _read_wav_mono_i16(segment_path)
    except RuntimeError:
        convert_audio_to_wav(ffmpeg, segment_path, temp_wav, cancel_requested=cancel_requested)
        return _read_wav_mono_i16(temp_wav)


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Przerwano przez uzytkownika")


def _read_wav_mono_i16(path: Path) -> array:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        if channels != 1 or sample_width != 2 or rate != SAMPLE_RATE:
            raise RuntimeError(f"Nieprawidlowy WAV timeline: {path}")
        data = wav.readframes(wav.getnframes())
    samples = array("h")
    samples.frombytes(data)
    return samples


def _write_wav_mono_i16(path: Path, samples: array) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


def _apply_segment_edge_fade(samples: array, sample_rate: int, fade_ms: int) -> None:
    fade_len = min(len(samples) // 2, int(sample_rate * max(0, int(fade_ms)) / 1000))
    if fade_len <= 0:
        return
    for index in range(fade_len):
        fade_in_scale = index / fade_len
        fade_out_scale = (fade_len - index - 1) / fade_len
        samples[index] = int(samples[index] * fade_in_scale)
        samples[-index - 1] = int(samples[-index - 1] * fade_out_scale)


def _mix_into(track: array, samples: array, start_sample: int) -> None:
    for offset, value in enumerate(samples):
        pos = start_sample + offset
        mixed = int(track[pos]) + int(value)
        if mixed > 32767:
            mixed = 32767
        elif mixed < -32768:
            mixed = -32768
        track[pos] = mixed

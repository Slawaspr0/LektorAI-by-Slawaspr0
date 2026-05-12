from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Callable

from app.core.paths import AppPaths
from app.pipeline.progress import ffmpeg_progress_ratio


VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".wmv", ".m4v"}
POLISH_MARKERS = {"pl", "pol", "polish", "polski", "polskie"}
TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text"}
BITMAP_SUBTITLE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}
BINARY_LOOKUP_HINT = "Dodaj ffmpeg.exe, ffprobe.exe i mkvmerge.exe do PATH albo umiesc je w folderze aplikacji obok START.py."
VOICE_SAMPLE_EXTENSIONS = (".wav", ".mp3", ".flac")
VOICE_SAMPLE_RATE_BY_ENGINE = {
    "chatterbox": 24000,
    "omnivoice": 24000,
    "coqui_xtts": 24000,
}
AAC_BITRATE_OPTIONS = ("192k", "256k", "320k", "384k", "448k", "640k")
DEFAULT_AAC_BITRATE = "384k"
DEFAULT_LEKTOR_LUFS = -14
DEFAULT_LEKTOR_WEIGHT = 2.3
DEFAULT_BACKGROUND_LUFS = -18
DEFAULT_BACKGROUND_WEIGHT = 1.6
DEFAULT_LEKTOR_DELAY_MS = 500
MIN_LEKTOR_DELAY_MS = 0
MAX_LEKTOR_DELAY_MS = 3000
LEKTOR_DELAY_STEP_MS = 50
OUTPUT_AUDIO_SAMPLE_RATE = 48000
LOUDNORM_TRUE_PEAK = -2
LOUDNORM_LRA = 11
MIX_LIMITER_LIMIT = 0.90


def find_binary(paths: AppPaths, name: str) -> Path | None:
    local_name = f"{name}.exe" if os.name == "nt" else name
    found = shutil.which(local_name) or shutil.which(name)
    if found:
        return Path(found)
    local_path = paths.app_dir / local_name
    return local_path if local_path.is_file() else None


def find_ffmpeg(paths: AppPaths) -> Path | None:
    return find_binary(paths, "ffmpeg")


def find_ffprobe(paths: AppPaths) -> Path | None:
    return find_binary(paths, "ffprobe")


def find_mkvmerge(paths: AppPaths) -> Path | None:
    return find_binary(paths, "mkvmerge")


def supported_voice_sample_extensions() -> tuple[str, ...]:
    return VOICE_SAMPLE_EXTENSIONS


def voice_sample_sample_rate(engine_id: str) -> int:
    return int(VOICE_SAMPLE_RATE_BY_ENGINE.get(str(engine_id), 24000))


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def probe_subtitle_streams(ffprobe: Path, video_path: Path) -> list[dict]:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name:stream_disposition=default,forced:stream_tags=language,title",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "ffprobe subtitle probe failed").strip())
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    return streams if isinstance(streams, list) else []


def extract_first_subtitle_to_srt(
    ffmpeg: Path,
    ffprobe: Path,
    video_path: Path,
    output_srt: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    streams = probe_subtitle_streams(ffprobe, video_path)
    if not streams:
        raise RuntimeError("Nie znaleziono napisow w kontenerze wideo.")
    subtitle_index = select_text_subtitle_stream_index(streams)
    if subtitle_index is None:
        raise RuntimeError("Nie znaleziono tekstowych napisow w kontenerze wideo. Napisy bitmapowe PGS/VobSub wymagaja OCR albo zewnetrznego pliku SRT/TXT.")
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command_capture_text(
        [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-i",
            str(video_path),
            "-map_metadata",
            "-1",
            "-map",
            f"0:s:{subtitle_index}",
            str(output_srt),
        ],
        timeout=600,
        cancel_requested=cancel_requested,
    )
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-12:])
        raise RuntimeError(f"Nie udalo sie wypakowac napisow do SRT.\n{tail}")


def select_subtitle_stream_index(streams: list[dict]) -> int:
    if not streams:
        return 0
    scored = [(score_subtitle_stream(stream), pos) for pos, stream in enumerate(streams)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return int(scored[0][1])


def select_text_subtitle_stream_index(streams: list[dict]) -> int | None:
    text_streams = [
        (stream, pos)
        for pos, stream in enumerate(streams)
        if str(stream.get("codec_name", "") or "").strip().lower() in TEXT_SUBTITLE_CODECS
    ]
    if not text_streams:
        return None
    scored = [(score_subtitle_stream(stream), pos) for stream, pos in text_streams]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return int(scored[0][1])


def score_subtitle_stream(stream: dict) -> int:
    tags = stream.get("tags", {}) if isinstance(stream.get("tags", {}), dict) else {}
    disposition = stream.get("disposition", {}) if isinstance(stream.get("disposition", {}), dict) else {}
    language = str(tags.get("language", "") or "").strip().lower()
    title = str(tags.get("title", "") or "").strip().lower()
    codec = str(stream.get("codec_name", "") or "").strip().lower()
    is_default = str(disposition.get("default", "") or "").strip().lower() in {"1", "true", "yes"}
    is_forced = str(disposition.get("forced", "") or "").strip().lower() in {"1", "true", "yes"}
    score = 0
    language_tokens = set(re.findall(r"[a-ząćęłńóśźż]+", language, flags=re.IGNORECASE))
    if language in POLISH_MARKERS or POLISH_MARKERS & language_tokens:
        score += 100
    if POLISH_MARKERS & set(re.findall(r"[a-ząćęłńóśźż]+", title, flags=re.IGNORECASE)):
        score += 40
    if is_default:
        score += 25
    if is_forced or "forced" in title or "wymus" in title:
        score -= 150
    if codec in TEXT_SUBTITLE_CODECS:
        score += 120
    if codec in BITMAP_SUBTITLE_CODECS:
        score -= 120
    return score


def probe_audio_stream_count(ffprobe: Path, video_path: Path) -> int:
    return len(probe_audio_streams(ffprobe, video_path))


def probe_media_duration(ffprobe: Path, media_path: Path) -> float:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return max(0.0, float((result.stdout or "").strip()))
    except Exception:
        return 0.0


def probe_audio_streams(ffprobe: Path, video_path: Path) -> list[dict]:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,channels,channel_layout,sample_rate:stream_tags=language,title",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    return streams if isinstance(streams, list) else []


def primary_audio_channels(audio_streams: list[dict]) -> int:
    if not audio_streams:
        return 0
    try:
        return max(0, int(audio_streams[0].get("channels") or 0))
    except Exception:
        return 0


def audio_stream_summary(stream: dict | None) -> str:
    if not stream:
        return "brak danych audio"
    codec = str(stream.get("codec_name") or "unknown").upper()
    layout = str(stream.get("channel_layout") or "").strip()
    channels = _format_audio_channels(stream.get("channels"), layout)
    sample_rate = _format_sample_rate(stream.get("sample_rate"))
    parts = [codec]
    if channels:
        parts.append(channels)
    if sample_rate:
        parts.append(sample_rate)
    return ", ".join(parts)


def wav_audio_diagnostics(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as wav:
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        sample_rate = int(wav.getframerate())
        frames = int(wav.getnframes())
        peak = 0
        while True:
            data = wav.readframes(65536)
            if not data:
                break
            if sample_width == 2:
                samples = array("h")
                samples.frombytes(data)
                if samples:
                    peak = max(peak, max(abs(int(sample)) for sample in samples))
            else:
                peak = max(peak, max(data, default=0))
    duration_s = frames / sample_rate if sample_rate > 0 else 0.0
    if peak <= 0:
        peak_dbfs = -120.0
    elif sample_width == 2:
        peak_dbfs = 20.0 * math.log10(min(1.0, peak / 32768.0))
    else:
        peak_dbfs = 20.0 * math.log10(min(1.0, peak / float(max(1, (1 << (8 * sample_width - 1))))))
    return {
        "duration_s": float(duration_s),
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "peak_dbfs": round(float(peak_dbfs), 2),
    }


def _format_audio_channels(channels: object, layout: str = "") -> str:
    layout_clean = layout.strip()
    if layout_clean and layout_clean.lower() != "unknown":
        return layout_clean
    try:
        channel_count = int(channels or 0)
    except Exception:
        channel_count = 0
    if channel_count == 1:
        return "mono"
    if channel_count == 2:
        return "stereo"
    if channel_count > 0:
        return f"{channel_count} kan."
    return ""


def _format_sample_rate(sample_rate: object) -> str:
    try:
        rate = int(sample_rate or 0)
    except Exception:
        return ""
    if rate <= 0:
        return ""
    khz = rate / 1000.0
    if rate % 1000 == 0:
        return f"{int(khz)} kHz"
    return f"{khz:.1f} kHz"


def convert_audio_to_wav(
    ffmpeg: Path,
    input_path: Path,
    output_wav: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(OUTPUT_AUDIO_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    try:
        result = _run_command_capture_text(command, timeout=120, cancel_requested=cancel_requested)
    except RuntimeError as exc:
        raise RuntimeError(f"Nie udalo sie przekonwertowac audio do WAV.\n{exc}") from exc
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-12:])
        raise RuntimeError(f"Nie udalo sie przekonwertowac audio do WAV.\n{tail}")

class CommandCaptureResult:
    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = int(returncode)
        self.output = str(output or "")


def _run_command_capture_text(
    command: list[str],
    timeout: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> CommandCaptureResult:
    process = subprocess.Popen(
        [
            *[str(part) for part in command],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str] = queue.Queue()
    output_lines: list[str] = []

    def read_output() -> None:
        try:
            for output_line in process.stdout:
                output_queue.put(output_line)
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + max(1, int(timeout))
    cancelled = False
    timed_out = False
    try:
        while process.poll() is None:
            _drain_text_output(output_queue, output_lines)
            if cancel_requested is not None and cancel_requested():
                cancelled = True
                _terminate_process_tree(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_tree(process)
                break
            time.sleep(0.1)
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            return_code = process.wait(timeout=10)
        reader.join(timeout=2)
        _drain_text_output(output_queue, output_lines)
    except Exception:
        _terminate_process_tree(process)
        raise
    if cancelled:
        raise RuntimeError("Przerwano przez uzytkownika")
    if timed_out:
        raise RuntimeError("Proces przekroczyl limit czasu")
    return CommandCaptureResult(return_code, "".join(output_lines))


def trim_fixed_and_fade_wav_edges(
    input_wav: Path,
    output_wav: Path,
    trim_start_ms: int = 200,
    trim_end_ms: int = 900,
    fade_ms: int = 12,
) -> None:
    samples, sample_rate = _read_wav_mono_i16_for_edges(input_wav)
    start = int(sample_rate * max(0, int(trim_start_ms)) / 1000)
    end_cut = int(sample_rate * max(0, int(trim_end_ms)) / 1000)
    stop = max(start, len(samples) - end_cut)
    trimmed = samples[start:stop] if samples else []
    _apply_i16_fade_out(trimmed, sample_rate, fade_ms)
    _write_wav_mono_i16_for_edges(output_wav, trimmed, sample_rate)


def _read_wav_mono_i16_for_edges(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if channels != 1 or sample_width != 2:
            raise RuntimeError(f"Nieprawidlowy WAV do obrobki krawedzi: {path}")
        data = wav.readframes(wav.getnframes())
    samples = [int.from_bytes(data[pos : pos + 2], "little", signed=True) for pos in range(0, len(data), 2)]
    return samples, int(sample_rate)


def _write_wav_mono_i16_for_edges(path: Path, samples: list[int], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(int(max(-32768, min(32767, sample))).to_bytes(2, "little", signed=True) for sample in samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(frames)


def _apply_i16_fade_out(samples: list[int], sample_rate: int, fade_ms: int) -> None:
    fade_len = min(len(samples) // 2, int(sample_rate * max(0, int(fade_ms)) / 1000))
    if fade_len <= 0:
        return
    for index in range(fade_len):
        scale = index / fade_len
        samples[-index - 1] = int(samples[-index - 1] * scale)


def prepare_voice_sample_command(ffmpeg: Path, input_path: Path, output_wav: Path, sample_rate: int, enhance: bool = True) -> list[str]:
    sr = max(8000, int(sample_rate))
    if enhance:
        # For 24 kHz prompts the gentle low-pass must stay below Nyquist.
        lowpass = 11000 if sr <= 24000 else 15000
        audio_filter = f"highpass=f=55,lowpass=f={lowpass},afftdn=nf=-20,loudnorm=I=-20:TP=-2:LRA=11"
    else:
        audio_filter = "loudnorm=I=-20:TP=-2:LRA=11"
    return [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-af",
        audio_filter,
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]


def prepare_voice_sample(
    ffmpeg: Path,
    input_path: Path,
    output_wav: Path,
    sample_rate: int,
    enhance: bool = True,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command_capture_text(
        prepare_voice_sample_command(ffmpeg, input_path, output_wav, sample_rate, enhance=enhance),
        timeout=180,
        cancel_requested=cancel_requested,
    )
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-12:])
        raise RuntimeError(f"Nie udalo sie przygotowac probki glosu.\n{tail}")


def sanitize_aac_bitrate(value: str | int | None) -> str:
    text = str(value or "").strip().lower().replace(" ", "")
    if text.isdigit():
        text = f"{text}k"
    return text if text in AAC_BITRATE_OPTIONS else DEFAULT_AAC_BITRATE


def sanitize_lufs(value: str | int | float | None, default: int) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = int(default)
    return max(-30, min(-8, number))


def sanitize_audio_weight(value: str | int | float | None, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    number = max(0.1, min(3.0, number))
    return round(number, 1)


def sanitize_lektor_delay_ms(value: str | int | float | None) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = DEFAULT_LEKTOR_DELAY_MS
    number = max(MIN_LEKTOR_DELAY_MS, min(MAX_LEKTOR_DELAY_MS, number))
    return int(round(number / LEKTOR_DELAY_STEP_MS) * LEKTOR_DELAY_STEP_MS)


def encode_wav_to_aac_command(ffmpeg: Path, input_wav: Path, output_m4a: Path, bitrate: str | int | None = DEFAULT_AAC_BITRATE) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_wav),
        "-map_metadata",
        "-1",
        "-ar",
        str(OUTPUT_AUDIO_SAMPLE_RATE),
        "-c:a",
        "aac",
        "-b:a",
        sanitize_aac_bitrate(bitrate),
        str(output_m4a),
    ]


def encode_wav_to_aac(
    ffmpeg: Path,
    input_wav: Path,
    output_m4a: Path,
    bitrate: str | int | None = DEFAULT_AAC_BITRATE,
    progress_callback=None,
    duration_seconds: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    output_m4a.parent.mkdir(parents=True, exist_ok=True)
    command = encode_wav_to_aac_command(ffmpeg, input_wav, output_m4a, bitrate)
    if progress_callback is not None:
        try:
            run_ffmpeg_with_progress(command, duration_seconds, progress_callback, timeout=300, cancel_requested=cancel_requested)
            return
        except RuntimeError as exc:
            raise RuntimeError(f"Nie udalo sie zakodowac lektora do AAC.\n{exc}") from exc
    result = _run_command_capture_text(command, timeout=300, cancel_requested=cancel_requested)
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-12:])
        raise RuntimeError(f"Nie udalo sie zakodowac lektora do AAC.\n{tail}")


def ffmpeg_command_with_progress(command: list[str]) -> list[str]:
    if not command:
        return []
    return [str(command[0]), "-nostats", "-progress", "pipe:1", *[str(part) for part in command[1:]]]


def run_ffmpeg_with_progress(
    command: list[str],
    duration_seconds: float | None,
    progress_callback=None,
    timeout: int = 1800,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    progress_command = ffmpeg_command_with_progress(command)
    process = subprocess.Popen(
        progress_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str] = queue.Queue()
    tail: list[str] = []

    def read_output() -> None:
        try:
            for output_line in process.stdout:
                output_queue.put(output_line)
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + max(1, int(timeout))
    cancelled = False
    timed_out = False
    try:
        while process.poll() is None:
            _drain_ffmpeg_progress_output(output_queue, tail, duration_seconds, progress_callback)
            if cancel_requested is not None and cancel_requested():
                cancelled = True
                _terminate_process_tree(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_tree(process)
                break
            time.sleep(0.1)
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            return_code = process.wait(timeout=10)
        reader.join(timeout=2)
        _drain_ffmpeg_progress_output(output_queue, tail, duration_seconds, progress_callback)
    except Exception:
        _terminate_process_tree(process)
        raise
    if cancelled:
        raise RuntimeError("Przerwano przez uzytkownika")
    if timed_out:
        raise RuntimeError("ffmpeg przekroczyl limit czasu")
    if return_code != 0:
        raise RuntimeError("\n".join(tail[-80:]) or "ffmpeg failed")


def _drain_text_output(output_queue: queue.Queue[str], output_lines: list[str]) -> None:
    while True:
        try:
            output_lines.append(output_queue.get_nowait())
        except queue.Empty:
            return


def _drain_ffmpeg_progress_output(
    output_queue: queue.Queue[str],
    tail: list[str],
    duration_seconds: float | None,
    progress_callback,
) -> None:
    while True:
        try:
            line = output_queue.get_nowait()
        except queue.Empty:
            return
        stripped = line.strip()
        if stripped:
            tail.append(stripped)
            del tail[:-160]
        ratio = ffmpeg_progress_ratio(stripped, duration_seconds)
        if ratio is not None and progress_callback is not None:
            progress_callback(ratio)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=20,
            )
            if completed.returncode == 0:
                return
        except Exception:
            pass
    try:
        process.terminate()
        process.wait(timeout=10)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def normalize_lektor_wav_command(ffmpeg: Path, input_wav: Path, output_wav: Path, target_lufs: int | float = DEFAULT_LEKTOR_LUFS) -> list[str]:
    lufs = sanitize_lufs(target_lufs, DEFAULT_LEKTOR_LUFS)
    return [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_wav),
        "-map_metadata",
        "-1",
        "-af",
        f"loudnorm=I={lufs}:TP={LOUDNORM_TRUE_PEAK}:LRA={LOUDNORM_LRA}:linear=true:dual_mono=true",
        "-ar",
        str(OUTPUT_AUDIO_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]


def normalize_lektor_wav(
    ffmpeg: Path,
    input_wav: Path,
    output_wav: Path,
    target_lufs: int | float = DEFAULT_LEKTOR_LUFS,
    progress_callback=None,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if Path(ffmpeg).exists():
        try:
            _normalize_lektor_wav_loudnorm_two_pass(
                ffmpeg,
                input_wav,
                output_wav,
                target_lufs,
                progress_callback=progress_callback,
                cancel_requested=cancel_requested,
            )
            normalized, sample_rate = _read_mono_i16_wav(output_wav)
            _smooth_detected_silence_onsets(normalized, sample_rate)
            _write_mono_i16_wav(output_wav, normalized, sample_rate)
            return
        except FileNotFoundError:
            pass
    samples, sample_rate = _read_mono_i16_wav(input_wav)
    normalized = _normalize_active_speech_samples(samples, sanitize_lufs(target_lufs, DEFAULT_LEKTOR_LUFS))
    _write_mono_i16_wav(output_wav, normalized, sample_rate)


def _normalize_lektor_wav_loudnorm_two_pass(
    ffmpeg: Path,
    input_wav: Path,
    output_wav: Path,
    target_lufs: int | float,
    progress_callback=None,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    lufs = sanitize_lufs(target_lufs, DEFAULT_LEKTOR_LUFS)
    measure_filter = (
        f"loudnorm=I={lufs}:TP={LOUDNORM_TRUE_PEAK}:LRA={LOUDNORM_LRA}:"
        "linear=true:dual_mono=true:print_format=json"
    )
    measure = _run_command_capture_text(
        [
            str(ffmpeg),
            "-hide_banner",
            "-i",
            str(input_wav),
            "-af",
            measure_filter,
            "-f",
            "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ],
        timeout=900,
        cancel_requested=cancel_requested,
    )
    if measure.returncode != 0:
        tail = "\n".join((measure.output or "").splitlines()[-12:])
        raise RuntimeError(f"Nie udalo sie zmierzyc loudnorm lektora.\n{tail}")
    stats = _parse_loudnorm_json(measure.output)
    apply_filter = (
        f"loudnorm=I={lufs}:TP={LOUDNORM_TRUE_PEAK}:LRA={LOUDNORM_LRA}:"
        f"measured_I={stats['input_i']}:"
        f"measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:"
        f"measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:"
        "linear=true:dual_mono=true:print_format=none,"
        f"aresample={OUTPUT_AUDIO_SAMPLE_RATE}"
    )
    apply_command = [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-i",
            str(input_wav),
            "-map_metadata",
            "-1",
            "-af",
            apply_filter,
            "-ar",
            str(OUTPUT_AUDIO_SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    if progress_callback is not None:
        try:
            run_ffmpeg_with_progress(
                apply_command,
                _wav_duration_seconds(input_wav),
                progress_callback,
                timeout=900,
                cancel_requested=cancel_requested,
            )
            return
        except RuntimeError as exc:
            raise RuntimeError(f"Nie udalo sie znormalizowac sciezki lektora loudnorm.\n{exc}") from exc
    result = _run_command_capture_text(apply_command, timeout=900, cancel_requested=cancel_requested)
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-12:])
        raise RuntimeError(f"Nie udalo sie znormalizowac sciezki lektora loudnorm.\n{tail}")


def _parse_loudnorm_json(text: str) -> dict[str, str]:
    matches = re.findall(r"\{[\s\S]*?\}", text or "")
    if not matches:
        raise RuntimeError("Brak statystyk JSON z loudnorm.")
    data = json.loads(matches[-1])
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError("Niepelne statystyki loudnorm: " + ", ".join(missing))
    return {key: str(data[key]) for key in required}


def _read_mono_i16_wav(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = int(wav.getframerate())
        if channels != 1 or sample_width != 2:
            raise RuntimeError(f"Nieprawidlowy WAV lektora do normalizacji: {path}")
        data = wav.readframes(wav.getnframes())
    samples = array("h")
    samples.frombytes(data)
    return samples, sample_rate


def _write_mono_i16_wav(path: Path, samples: array, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(samples.tobytes())


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = int(wav.getframerate())
            frames = int(wav.getnframes())
        return frames / rate if rate > 0 else 0.0
    except Exception:
        return 0.0


def _normalize_active_speech_samples(samples: array, target_lufs: int) -> array:
    if not samples:
        return array("h")
    peak = max((abs(int(sample)) for sample in samples), default=0)
    if peak <= 0:
        return array("h", samples)

    active_threshold = max(96, int(peak * 0.003))
    active = [int(sample) for sample in samples if abs(int(sample)) >= active_threshold]
    if not active:
        return array("h", samples)

    active_rms = math.sqrt(sum(sample * sample for sample in active) / len(active))
    if active_rms <= 0:
        return array("h", samples)

    target_rms = 32768.0 * (10 ** (float(target_lufs) / 20.0))
    gain_by_rms = target_rms / active_rms
    gain_by_peak = (32767.0 * 0.95) / peak
    gain = max(0.05, min(gain_by_rms, gain_by_peak, 8.0))

    out = array("h")
    for sample in samples:
        value = int(round(int(sample) * gain))
        if value > 32767:
            value = 32767
        elif value < -32768:
            value = -32768
        out.append(value)
    return out


def _smooth_detected_silence_onsets(samples: array, sample_rate: int, silence_ms: int = 20, fade_ms: int = 6) -> None:
    if not samples:
        return
    silence_frames = max(1, int(sample_rate * max(1, int(silence_ms)) / 1000))
    fade_frames = max(1, int(sample_rate * max(1, int(fade_ms)) / 1000))
    silence_threshold = 24
    active_threshold = 192
    silent_run = 0
    index = 0
    total = len(samples)
    while index < total:
        value = abs(int(samples[index]))
        if value <= silence_threshold:
            silent_run += 1
            index += 1
            continue
        if silent_run >= silence_frames and value >= active_threshold:
            limit = min(total, index + fade_frames)
            for pos in range(index, limit):
                scale = (pos - index) / fade_frames
                samples[pos] = int(samples[pos] * scale)
            index = limit
            silent_run = 0
            continue
        silent_run = 0
        index += 1


def apply_short_audio_fade(
    ffmpeg: Path,
    input_path: Path,
    output_path: Path,
    fade_seconds: float = 0.018,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fade = max(0.005, min(float(fade_seconds), 0.08))
    result = _run_command_capture_text(
        [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-i",
            str(input_path),
            "-map_metadata",
            "-1",
            "-af",
            f"afade=t=in:st=0:d={fade:.3f},areverse,afade=t=in:st=0:d={fade:.3f},areverse",
            str(output_path),
        ],
        timeout=120,
        cancel_requested=cancel_requested,
    )
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-12:])
        raise RuntimeError(f"Nie udalo sie wygladzic krawedzi audio.\n{tail}")


def mux_lektor_track(
    ffmpeg: Path,
    ffprobe: Path,
    video_path: Path,
    lektor_audio: Path,
    output_video: Path,
    title: str,
    lektor_weight: float = DEFAULT_LEKTOR_WEIGHT,
    background_lufs: int = DEFAULT_BACKGROUND_LUFS,
    background_weight: float = DEFAULT_BACKGROUND_WEIGHT,
    bitrate: str | int | None = DEFAULT_AAC_BITRATE,
    create_stereo_for_surround: bool = True,
    diagnostic_dir: Path | None = None,
    keep_mixing_steps: bool = False,
    mkvmerge: Path | None = None,
    progress_callback=None,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    if mkvmerge is None:
        raise RuntimeError(
            "Brak mkvmerge. MKVToolNix jest wymagany do bezpiecznego zapisu kontenera MKV. "
            f"{BINARY_LOOKUP_HINT}"
        )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    audio_streams = probe_audio_streams(ffprobe, video_path)
    new_audio_index = len(audio_streams)
    primary_channels = primary_audio_channels(audio_streams)
    temp_output = _temporary_output_path(output_video, "mux_temp")
    _unlink_if_exists(temp_output)
    if new_audio_index > 0:
        stage_dir = diagnostic_dir if keep_mixing_steps and diagnostic_dir is not None else output_video.parent
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_prefix = output_video.stem
        source_audio_path = _stage_audio_path(stage_dir, stage_prefix, "tlo_zrodlowe", ".wav", keep_mixing_steps)
        pl_stereo_path = _stage_audio_path(stage_dir, stage_prefix, "pl_2_0", ".m4a", keep_mixing_steps)
        pl_surround_path = _stage_audio_path(stage_dir, stage_prefix, "pl_5_1", ".m4a", keep_mixing_steps)
        temp_stage_paths = [source_audio_path, pl_stereo_path, pl_surround_path]
        for path in temp_stage_paths:
            _unlink_if_exists(path)
        extract_command = extract_primary_audio_command(ffmpeg, video_path, source_audio_path)
        create_surround_track = primary_channels >= 6
        create_stereo_track = not create_surround_track or bool(create_stereo_for_surround)
        prepared_tracks: list[Path] = []
        prepared_labels: list[str] = []
        try:
            if progress_callback is not None:
                _run_stage_with_progress(extract_command, probe_media_duration(ffprobe, video_path), progress_callback, 0.0, 0.05, timeout=1200, cancel_requested=cancel_requested)
            else:
                extract_result = _run_command_capture_text(extract_command, timeout=1200, cancel_requested=cancel_requested)
                if extract_result.returncode != 0:
                    tail = "\n".join((extract_result.output or "").splitlines()[-24:])
                    raise RuntimeError(f"Nie udalo sie wyciagnac sciezki audio tla.\n{tail}")
            mix_duration = probe_media_duration(ffprobe, video_path)
            if create_stereo_track:
                stereo_command = mix_lektor_stereo_audio_command(
                    ffmpeg,
                    source_audio_path,
                    lektor_audio,
                    pl_stereo_path,
                    lektor_weight=lektor_weight,
                    background_lufs=background_lufs,
                    background_weight=background_weight,
                    bitrate=bitrate,
                )
                if progress_callback is not None:
                    end = 0.45 if create_surround_track else 0.85
                    _run_stage_with_progress(stereo_command, mix_duration, progress_callback, 0.05, end, timeout=3600, cancel_requested=cancel_requested)
                else:
                    stereo_result = _run_command_capture_text(stereo_command, timeout=3600, cancel_requested=cancel_requested)
                    if stereo_result.returncode != 0:
                        tail = "\n".join((stereo_result.output or "").splitlines()[-80:])
                        raise RuntimeError(f"Nie udalo sie przygotowac sciezki PL 2.0.\n{tail}")
                prepared_tracks.append(pl_stereo_path)
                prepared_labels.append("2.0")
            if create_surround_track:
                surround_command = mix_lektor_surround_audio_command(
                    ffmpeg,
                    source_audio_path,
                    lektor_audio,
                    pl_surround_path,
                    lektor_weight=lektor_weight,
                    background_lufs=background_lufs,
                    background_weight=background_weight,
                    bitrate=bitrate,
                )
                if progress_callback is not None:
                    start = 0.45 if create_stereo_track else 0.05
                    _run_stage_with_progress(surround_command, mix_duration, progress_callback, start, 0.85, timeout=3600, cancel_requested=cancel_requested)
                else:
                    surround_result = _run_command_capture_text(surround_command, timeout=3600, cancel_requested=cancel_requested)
                    if surround_result.returncode != 0:
                        tail = "\n".join((surround_result.output or "").splitlines()[-80:])
                        raise RuntimeError(f"Nie udalo sie przygotowac sciezki PL 5.1.\n{tail}")
                prepared_tracks.append(pl_surround_path)
                prepared_labels.append("5.1")
            remux_command = remux_with_prepared_lektor_audio_mkvmerge_command(
                mkvmerge,
                video_path,
                tuple(prepared_tracks),
                temp_output,
                title,
                track_labels=tuple(prepared_labels),
                source_audio_streams=tuple(audio_streams),
            )
            if progress_callback is not None:
                progress_callback(0.85)
            remux_result = _run_command_capture_text(remux_command, timeout=1200, cancel_requested=cancel_requested)
            if remux_result.returncode != 0:
                tail = "\n".join((remux_result.output or "").splitlines()[-80:])
                raise RuntimeError(f"Nie udalo sie zremuksowac wideo z gotowym audio lektora przez MKVToolNix.\n{tail}")
            if progress_callback is not None:
                progress_callback(1.0)
            _replace_completed_output(temp_output, output_video)
            return
        except RuntimeError as exc:
            raise RuntimeError(f"Nie udalo sie dodac sciezki lektora do wideo.\n{exc}") from exc
        finally:
            if not keep_mixing_steps:
                for path in temp_stage_paths:
                    _unlink_if_exists(path)
            _unlink_if_exists(temp_output)

    raise RuntimeError("Nie znaleziono sciezki audio w pliku wideo, nie ma do czego domiksowac lektora.")


def _temporary_output_path(path: Path, marker: str) -> Path:
    return path.with_name(f"{path.stem}_{marker}{path.suffix}")


def _replace_completed_output(temp_path: Path, output_path: Path) -> None:
    if not temp_path.exists():
        raise RuntimeError(f"Brak pliku tymczasowego wyniku: {temp_path.name}")
    try:
        temp_path.replace(output_path)
    except OSError as exc:
        raise RuntimeError(f"Nie udalo sie zapisac pliku wynikowego: {output_path}") from exc


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _stage_audio_path(stage_dir: Path, stage_prefix: str, suffix: str, extension: str, keep: bool) -> Path:
    prefix = safe_file_stem(stage_prefix) or "lektor"
    name = f"{prefix}_{suffix}{extension}" if keep else f"{prefix}_{suffix}_tmp{extension}"
    return stage_dir / name


def safe_file_stem(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:96]


def _run_stage_with_progress(
    command: list[str],
    duration_seconds: float | None,
    progress_callback,
    start: float,
    end: float,
    timeout: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    stage_start = max(0.0, min(1.0, float(start)))
    stage_end = max(stage_start, min(1.0, float(end)))

    def scaled_progress(ratio: float | None) -> None:
        try:
            value = 0.0 if ratio is None else max(0.0, min(1.0, float(ratio)))
        except Exception:
            value = 0.0
        progress_callback(stage_start + ((stage_end - stage_start) * value))

    run_ffmpeg_with_progress(
        command,
        duration_seconds,
        scaled_progress,
        timeout=timeout,
        cancel_requested=cancel_requested,
    )


def extract_primary_audio_command(ffmpeg: Path, video_path: Path, output_audio: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-map_metadata",
        "-1",
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        f"asetpts=PTS-STARTPTS,aresample={OUTPUT_AUDIO_SAMPLE_RATE}",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(OUTPUT_AUDIO_SAMPLE_RATE),
        str(output_audio),
    ]


def mix_lektor_stereo_audio_command(
    ffmpeg: Path,
    background_audio: Path,
    lektor_m4a: Path,
    output_audio: Path,
    lektor_weight: float = DEFAULT_LEKTOR_WEIGHT,
    background_lufs: int = DEFAULT_BACKGROUND_LUFS,
    background_weight: float = DEFAULT_BACKGROUND_WEIGHT,
    bitrate: str | int | None = DEFAULT_AAC_BITRATE,
) -> list[str]:
    bg_lufs = sanitize_lufs(background_lufs, DEFAULT_BACKGROUND_LUFS)
    bg_weight = sanitize_audio_weight(background_weight, DEFAULT_BACKGROUND_WEIGHT)
    lector_weight = sanitize_audio_weight(lektor_weight, DEFAULT_LEKTOR_WEIGHT)
    bg_gain, lector_gain = _overlay_mix_gains(bg_weight, lector_weight)
    limiter = _mix_limiter_filter()
    filter_complex = ";".join(
        [
            f"[0:a:0]asetpts=PTS-STARTPTS,aresample={OUTPUT_AUDIO_SAMPLE_RATE},loudnorm=I={bg_lufs}:TP={LOUDNORM_TRUE_PEAK},aformat=channel_layouts=stereo,volume={bg_gain}[bg20]",
            f"[1:a:0]asetpts=PTS-STARTPTS,aresample={OUTPUT_AUDIO_SAMPLE_RATE},aformat=channel_layouts=stereo,volume={lector_gain}[lector20]",
            f"[bg20][lector20]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,{limiter}[lektor_pl_2_0]",
        ]
    )
    return [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(background_audio),
        "-i",
        str(lektor_m4a),
        "-map_metadata",
        "-1",
        "-filter_complex",
        filter_complex,
        "-map",
        "[lektor_pl_2_0]",
        "-c:a",
        "aac",
        "-ar",
        str(OUTPUT_AUDIO_SAMPLE_RATE),
        "-b:a",
        sanitize_aac_bitrate(bitrate),
        str(output_audio),
    ]


def mix_lektor_surround_audio_command(
    ffmpeg: Path,
    background_audio: Path,
    lektor_m4a: Path,
    output_audio: Path,
    lektor_weight: float = DEFAULT_LEKTOR_WEIGHT,
    background_lufs: int = DEFAULT_BACKGROUND_LUFS,
    background_weight: float = DEFAULT_BACKGROUND_WEIGHT,
    bitrate: str | int | None = DEFAULT_AAC_BITRATE,
) -> list[str]:
    bg_lufs = sanitize_lufs(background_lufs, DEFAULT_BACKGROUND_LUFS)
    bg_weight = sanitize_audio_weight(background_weight, DEFAULT_BACKGROUND_WEIGHT)
    lector_weight = sanitize_audio_weight(lektor_weight, DEFAULT_LEKTOR_WEIGHT)
    bg_gain, lector_gain = _overlay_mix_gains(bg_weight, lector_weight)
    limiter = _mix_limiter_filter()
    filter_complex = ";".join(
        [
            f"[0:a:0]asetpts=PTS-STARTPTS,aresample={OUTPUT_AUDIO_SAMPLE_RATE},loudnorm=I={bg_lufs}:TP={LOUDNORM_TRUE_PEAK},aformat=channel_layouts=5.1,volume={bg_gain}[bg51]",
            f"[1:a:0]asetpts=PTS-STARTPTS,aresample={OUTPUT_AUDIO_SAMPLE_RATE},aformat=channel_layouts=mono,volume={lector_gain}[lector_mono]",
            "[lector_mono]pan=5.1|FL=0*c0|FR=0*c0|FC=c0|LFE=0*c0|BL=0*c0|BR=0*c0[lector51]",
            f"[bg51][lector51]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,{limiter}[lektor_pl_5_1]",
        ]
    )
    return [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(background_audio),
        "-i",
        str(lektor_m4a),
        "-map_metadata",
        "-1",
        "-filter_complex",
        filter_complex,
        "-map",
        "[lektor_pl_5_1]",
        "-c:a",
        "aac",
        "-ar",
        str(OUTPUT_AUDIO_SAMPLE_RATE),
        "-b:a",
        sanitize_aac_bitrate(bitrate),
        str(output_audio),
    ]


def remux_with_prepared_lektor_audio_mkvmerge_command(
    mkvmerge: Path,
    video_path: Path,
    prepared_audio_paths: tuple[Path, ...],
    output_video: Path,
    title: str,
    track_labels: tuple[str, ...] = (),
    source_audio_streams: tuple[dict, ...] = (),
) -> list[str]:
    tracks = tuple(Path(path) for path in prepared_audio_paths)
    labels = tuple(str(label or "").strip() for label in track_labels)
    command = [
        str(mkvmerge),
        "-o",
        str(output_video),
        "--no-audio",
        "--no-global-tags",
        "--no-track-tags",
        "--default-track-flag",
        "0:yes",
        str(video_path),
    ]
    for stream_index, audio_path in enumerate(tracks):
        label = labels[stream_index] if stream_index < len(labels) else ""
        track_title = f"{title} {label}".strip()
        command += [
            "--language",
            "0:pol",
            "--track-name",
            f"0:{track_title}",
            "--default-track-flag",
            f"0:{'yes' if stream_index == 0 else 'no'}",
            str(audio_path),
        ]
    command += [
        "--no-video",
        "--no-subtitles",
        "--no-attachments",
        "--no-chapters",
        "--no-global-tags",
        "--no-track-tags",
    ]
    for stream in source_audio_streams:
        try:
            source_track_id = int(stream.get("index"))
        except Exception:
            continue
        command += ["--default-track-flag", f"{source_track_id}:no"]
    command += [str(video_path)]
    return command


def _overlay_mix_gains(background_weight: float, lektor_weight: float) -> tuple[str, str]:
    bg = sanitize_audio_weight(background_weight, DEFAULT_BACKGROUND_WEIGHT)
    lector = sanitize_audio_weight(lektor_weight, DEFAULT_LEKTOR_WEIGHT)
    total = max(bg + lector, 0.001)
    background_gain = min(1.0, max(0.05, bg / DEFAULT_BACKGROUND_WEIGHT))
    lektor_gain = min(1.0, max(0.05, lector / total))
    return _ffmpeg_float(background_gain), _ffmpeg_float(lektor_gain)


def _mix_limiter_filter() -> str:
    return f"alimiter=limit={_ffmpeg_float(MIX_LIMITER_LIMIT)}:level=false:latency=1"


def _ffmpeg_float(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")

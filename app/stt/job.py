from __future__ import annotations

import json
import shutil
import time
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.core.media_tools import (
    BINARY_LOOKUP_HINT,
    VIDEO_EXTENSIONS,
    _run_command_capture_text,
    find_ffmpeg,
    find_ffprobe,
    primary_audio_channels,
    probe_audio_streams,
)
from app.core.paths import AppPaths
from app.engines.config_schema import faster_whisper_device_kwargs, whisper_qc_effective_compute_type
from app.pipeline.subtitles import SubtitleSegment, load_srt, save_srt
from app.pipeline.workspace import compact_run_timestamp, compact_source_stem, lektorai_workspace_for
from app.stt.faster_whisper_runtime import ensure_faster_whisper_runtime, import_faster_whisper_for_cache
from app.stt.subtitle_profiles import FALLBACK_SUBTITLE_PROFILE, SttSubtitleProfile, subtitle_profile_for_language
from app.stt.whisper_cpp_runtime import (
    build_whisper_cpp_command,
    ensure_whisper_cpp_runtime,
    ensure_whisper_cpp_model,
    sanitize_whisper_cpp_device,
    sanitize_whisper_cpp_runtime,
    whisper_cpp_runtime_env,
)
from app.stt.whisperx_runtime import (
    WHISPERX_PROGRESS_RE_TEXT,
    build_whisperx_command,
    ensure_whisperx_gpu_runtime,
    ensure_whisperx_runtime,
    whisperx_runtime_env,
)


AUDIO_EXTENSIONS = (".aac", ".ac3", ".dts", ".dtshd", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma")
SUPPORTED_STT_INPUT_EXTENSIONS = (*VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)
_STT_WHISPER_MODELS: dict[tuple[str, str, str, str], object] = {}
STT_ACCURACY_BEAM_SIZE = {
    "fast": 1,
    "standard": 5,
    "accurate": 8,
}
STT_VAD_MIN_SILENCE_MS = {
    "gentle": 1000,
    "standard": 500,
    "strong": 250,
}
STT_ENGINE_CODES = {
    "faster_whisper": "FW",
    "whisper_cpp": "WCPP",
    "whisperx": "WX",
}
WHISPER_CPP_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d{1,3})%")
WHISPERX_PROGRESS_RE = re.compile(WHISPERX_PROGRESS_RE_TEXT)
NON_DIALOGUE_EXACT_TEXTS = {
    "applause",
    "background music",
    "cheering",
    "chuckling",
    "clapping",
    "dramatic music",
    "dramatic music playing",
    "footsteps",
    "gunshots",
    "inaudible",
    "instrumental music",
    "laughing",
    "laughter",
    "music",
    "music continues",
    "music playing",
    "ominous music",
    "sad music",
    "sigh",
    "sighs",
    "silence",
    "soft music",
    "suspenseful music",
    "tense music",
    "the end",
    "theme music",
    "thunder",
    "upbeat music",
}
NON_DIALOGUE_KEYWORDS = {
    "applause",
    "cheering",
    "chuckling",
    "clapping",
    "footsteps",
    "gunshots",
    "inaudible",
    "laughing",
    "laughter",
    "music",
    "sigh",
    "sighs",
    "silence",
    "thunder",
}
COMMON_DIALOGUE_REPEATS = {
    "ah",
    "go",
    "ha",
    "hey",
    "hm",
    "hmm",
    "no",
    "oh",
    "ok",
    "okay",
    "stop",
    "uh",
    "um",
    "wait",
    "whoa",
    "yeah",
    "yes",
}


@dataclass(frozen=True)
class SttSettings:
    engine: str = "faster_whisper"
    model: str = "small"
    language: str = "auto"
    device: str = "cpu"
    compute_type: str = "int8"
    accuracy: str = "standard"
    vad_enabled: bool = True
    vad_sensitivity: str = "standard"
    whisper_cpp_runtime: str = "cpu"
    whisper_cpp_device: str = "auto"
    whisper_cpp_threads: int = 0
    whisperx_device: str = "cpu"
    whisperx_compute_type: str = "int8"
    postprocess_enabled: bool = True
    open_workspace_on_finish: bool = False
    save_prepared_audio: bool = False
    save_report: bool = False
    save_log: bool = False


@dataclass(frozen=True)
class SttJobResult:
    source_path: Path
    workspace: Path
    output_srt: Path
    segment_count: int
    duration_seconds: float


@dataclass(frozen=True)
class SttRemovedSegment:
    source_index: int
    start_ms: int
    end_ms: int
    text: str
    stage: str
    reason: str


def is_stt_input_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_STT_INPUT_EXTENSIONS


def run_stt_job(
    source_path: Path,
    paths: AppPaths,
    settings: SttSettings,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> SttJobResult:
    engine = str(settings.engine or "faster_whisper").strip().lower()
    if engine == "whisper_cpp":
        return run_whisper_cpp_stt(
            source_path,
            paths,
            settings,
            progress=progress,
            cancel_requested=cancel_requested,
        )
    if engine == "whisperx":
        return run_whisperx_stt(
            source_path,
            paths,
            settings,
            progress=progress,
            cancel_requested=cancel_requested,
        )
    return run_faster_whisper_stt(
        source_path,
        paths,
        settings,
        progress=progress,
        cancel_requested=cancel_requested,
    )


def run_faster_whisper_stt(
    source_path: Path,
    paths: AppPaths,
    settings: SttSettings,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> SttJobResult:
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Brak pliku: {source_path}")
    if not is_stt_input_file(source_path):
        raise ValueError(f"Nieobslugiwany plik STT: {source_path.name}")

    ffmpeg = find_ffmpeg(paths)
    if ffmpeg is None:
        raise RuntimeError(f"Brak ffmpeg. {BINARY_LOOKUP_HINT}")
    ffprobe = find_ffprobe(paths)
    if ffprobe is None:
        raise RuntimeError(f"Brak ffprobe. {BINARY_LOOKUP_HINT}")

    workspace = lektorai_workspace_for(source_path)
    output_stem = next_stt_output_stem(workspace, source_path, engine="faster_whisper")
    output_dir = workspace / output_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = paths.faster_whisper_stt_dir / "temp" / output_stem
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = temp_dir / "stt_audio.wav"
    output_srt = output_dir / f"{output_stem}.srt"

    started_at = time.monotonic()
    events: list[str] = []

    def emit(message: str) -> None:
        events.append(message)
        _emit(progress, message)

    device = str(settings.device or "cpu").strip() or "cpu"
    emit("STT: przygotowanie modulu faster-whisper")
    ensure_faster_whisper_runtime(paths, progress=progress, device=device)
    _raise_if_cancelled(cancel_requested)

    emit("STT: przygotowanie audio")
    channels = primary_audio_channels(probe_audio_streams(ffprobe, source_path))
    extract_audio_for_stt(ffmpeg, source_path, temp_wav, channels, cancel_requested=cancel_requested)
    _raise_if_cancelled(cancel_requested)

    compute_type = whisper_qc_effective_compute_type(device, settings.compute_type)
    emit("STT: ladowanie modelu")
    model = load_stt_whisper_model(
        settings.model,
        device,
        compute_type,
        paths.faster_whisper_cache_dir,
    )

    language = str(settings.language or "auto").strip().lower()
    transcribe_kwargs = default_stt_transcribe_kwargs(settings)
    if language and language != "auto":
        transcribe_kwargs["language"] = language

    emit("STT: transkrypcja")
    raw_segments, info = model.transcribe(str(temp_wav), **transcribe_kwargs)
    detected_language = getattr(info, "language", "") or ""
    if detected_language:
        emit(f"STT: wykryty jezyk {detected_language}")
    subtitle_profile = subtitle_profile_for_language(language if language != "auto" else detected_language)
    emit(f"STT: format napisow {subtitle_profile.label}")
    segments: list[SubtitleSegment] = []
    raw_segment_count = 0
    for index, segment in enumerate(raw_segments, 1):
        _raise_if_cancelled(cancel_requested)
        text = str(getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        raw_segment_count += 1
        start_ms = max(0, int(round(float(getattr(segment, "start", 0.0) or 0.0) * 1000)))
        end_ms = max(start_ms + 1, int(round(float(getattr(segment, "end", 0.0) or 0.0) * 1000)))
        segments.append(SubtitleSegment(index=len(segments) + 1, start_ms=start_ms, end_ms=end_ms, text=text))
        if index == 1 or index % 10 == 0:
            emit(f"STT: segment {index}")

    if not segments:
        raise RuntimeError("STT nie wykryl zadnej mowy w pliku.")

    if not settings.postprocess_enabled:
        emit("STT: formatowanie LektorAI wylaczone")
        save_srt(output_srt, segments)
        emit(f"STT: zapisano napisy {output_srt.name}")
        duration_seconds = max(0.0, time.monotonic() - started_at)
        save_stt_diagnostics(
            output_dir=output_dir,
            temp_wav=temp_wav,
            source_path=source_path,
            settings=settings,
            transcribe_kwargs=transcribe_kwargs,
            detected_language=detected_language,
            subtitle_profile=subtitle_profile,
            raw_segment_count=raw_segment_count,
            final_segment_count=len(segments),
            duration_seconds=duration_seconds,
            events=events,
            removed_segments=[],
        )
        try:
            temp_wav.unlink(missing_ok=True)
            temp_dir.rmdir()
        except OSError:
            pass
        return SttJobResult(
            source_path=source_path,
            workspace=output_dir,
            output_srt=output_srt,
            segment_count=len(segments),
            duration_seconds=duration_seconds,
        )

    removed_segments: list[SttRemovedSegment] = []
    segments, non_dialogue_removed = filter_stt_dialogue_segments(
        segments,
        removed_segments=removed_segments,
    )
    if non_dialogue_removed:
        emit(f"STT: pominieto opisy niedialogowe {non_dialogue_removed}")
    if not segments:
        raise RuntimeError("STT nie wykryl zadnego dialogu w pliku.")
    segments, repeated_removed = filter_repeated_stt_hallucinations(
        segments,
        removed_segments=removed_segments,
    )
    if repeated_removed:
        emit(f"STT: pominieto powtorzone fragmenty {repeated_removed}")
    if not segments:
        raise RuntimeError("STT nie wykryl zadnego dialogu w pliku.")

    segments, merged_segments = merge_short_stt_segments(segments, profile=subtitle_profile)
    if merged_segments:
        emit(f"STT: scalono krotkie fragmenty {merged_segments}")
    segments = split_stt_subtitle_segments(segments, profile=subtitle_profile)
    save_srt(output_srt, segments)
    emit(f"STT: zapisano napisy {output_srt.name}")
    duration_seconds = max(0.0, time.monotonic() - started_at)
    save_stt_diagnostics(
        output_dir=output_dir,
        temp_wav=temp_wav,
        source_path=source_path,
        settings=settings,
        transcribe_kwargs=transcribe_kwargs,
        detected_language=detected_language,
        subtitle_profile=subtitle_profile,
        raw_segment_count=raw_segment_count,
        final_segment_count=len(segments),
        duration_seconds=duration_seconds,
        events=events,
        removed_segments=removed_segments,
    )
    try:
        temp_wav.unlink(missing_ok=True)
        temp_dir.rmdir()
    except OSError:
        pass

    return SttJobResult(
        source_path=source_path,
        workspace=output_dir,
        output_srt=output_srt,
        segment_count=len(segments),
        duration_seconds=duration_seconds,
    )


def run_whisper_cpp_stt(
    source_path: Path,
    paths: AppPaths,
    settings: SttSettings,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> SttJobResult:
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Brak pliku: {source_path}")
    if not is_stt_input_file(source_path):
        raise ValueError(f"Nieobslugiwany plik STT: {source_path.name}")

    ffmpeg = find_ffmpeg(paths)
    if ffmpeg is None:
        raise RuntimeError(f"Brak ffmpeg. {BINARY_LOOKUP_HINT}")
    ffprobe = find_ffprobe(paths)
    if ffprobe is None:
        raise RuntimeError(f"Brak ffprobe. {BINARY_LOOKUP_HINT}")

    workspace = lektorai_workspace_for(source_path)
    output_stem = next_stt_output_stem(workspace, source_path, engine="whisper_cpp")
    output_dir = workspace / output_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = paths.whisper_cpp_stt_dir / "temp" / output_stem
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = temp_dir / "stt_audio.wav"
    output_base = temp_dir / "whisper_cpp_result"
    raw_srt = output_base.with_suffix(".srt")
    output_srt = output_dir / f"{output_stem}.srt"

    started_at = time.monotonic()
    events: list[str] = []

    def emit(message: str) -> None:
        events.append(message)
        _emit(progress, message)

    emit("STT: przygotowanie modulu whisper.cpp")
    runtime_variant = sanitize_whisper_cpp_runtime(settings.whisper_cpp_runtime)
    runtime_device = "cpu" if runtime_variant == "cpu" else sanitize_whisper_cpp_device(settings.whisper_cpp_device)
    exe_path = ensure_whisper_cpp_runtime(
        paths,
        runtime_variant,
        progress=progress,
        cancel_requested=cancel_requested,
    )
    model_path = ensure_whisper_cpp_model(
        paths,
        settings.model,
        progress=progress,
        cancel_requested=cancel_requested,
    )
    _raise_if_cancelled(cancel_requested)

    emit("STT: przygotowanie audio")
    channels = primary_audio_channels(probe_audio_streams(ffprobe, source_path))
    extract_audio_for_stt(ffmpeg, source_path, temp_wav, channels, cancel_requested=cancel_requested)
    _raise_if_cancelled(cancel_requested)

    language = str(settings.language or "auto").strip().lower()
    command = build_whisper_cpp_command(
        exe_path=exe_path,
        model_path=model_path,
        input_wav=temp_wav,
        output_base=output_base,
        language=language,
        threads=int(settings.whisper_cpp_threads or 0),
        device=runtime_device,
    )
    emit("STT: transkrypcja")
    last_transcription_progress = -1

    def on_whisper_cpp_output(line: str) -> None:
        nonlocal last_transcription_progress
        match = WHISPER_CPP_PROGRESS_RE.search(line)
        if not match:
            return
        percent = max(0, min(100, int(match.group(1))))
        rounded = (percent // 10) * 10
        if rounded <= 0 or rounded <= last_transcription_progress:
            return
        last_transcription_progress = rounded
        emit(f"whisper.cpp: transkrypcja {rounded}%")

    try:
        result = _run_command_capture_text(
            command,
            timeout=7200,
            cancel_requested=cancel_requested,
            output_callback=on_whisper_cpp_output,
            env=whisper_cpp_runtime_env(paths, runtime_variant),
        )
    except OSError as exc:
        if runtime_variant == "cuda":
            raise RuntimeError(
                "Nie udalo sie uruchomic whisper.cpp CUDA. "
                "Sprawdz runtime CUDA w aplikacji albo wybierz runtime CPU."
            ) from exc
        raise
    except RuntimeError as exc:
        if runtime_variant == "cuda" and _looks_like_missing_cuda_runtime(str(exc)):
            raise RuntimeError(
                "Nie udalo sie uruchomic whisper.cpp CUDA. "
                "Sprawdz runtime CUDA w aplikacji albo wybierz runtime CPU."
            ) from exc
        raise
    if result.returncode == 0 and last_transcription_progress < 100:
        emit("whisper.cpp: transkrypcja 100%")
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-20:])
        if runtime_variant == "cuda" and _looks_like_missing_cuda_runtime(result.output):
            raise RuntimeError(
                "Nie udalo sie uruchomic whisper.cpp CUDA. "
                "Sprawdz runtime CUDA w aplikacji albo wybierz runtime CPU.\n"
                f"{tail}"
            )
        raise RuntimeError(f"whisper.cpp nie utworzyl napisow.\n{tail}")
    if not raw_srt.is_file():
        tail = "\n".join((result.output or "").splitlines()[-20:])
        raise RuntimeError(f"whisper.cpp nie zapisal pliku SRT.\n{tail}")

    raw_segments = load_srt(raw_srt)
    if not raw_segments:
        raise RuntimeError("STT nie wykryl zadnej mowy w pliku.")
    raw_segment_count = len(raw_segments)

    subtitle_profile = subtitle_profile_for_language(language)
    emit(f"STT: format napisow {subtitle_profile.label}")
    if not settings.postprocess_enabled:
        emit("STT: formatowanie LektorAI wylaczone")
        shutil.copy2(raw_srt, output_srt)
        emit(f"STT: zapisano napisy {output_srt.name}")
        duration_seconds = max(0.0, time.monotonic() - started_at)
        save_stt_diagnostics(
            output_dir=output_dir,
            temp_wav=temp_wav,
            source_path=source_path,
            settings=settings,
            transcribe_kwargs={
                "command": command,
                "runtime": runtime_variant,
                "device": runtime_device,
                "threads": int(settings.whisper_cpp_threads or 0),
            },
            detected_language="" if language == "auto" else language,
            subtitle_profile=subtitle_profile,
            raw_segment_count=raw_segment_count,
            final_segment_count=len(raw_segments),
            duration_seconds=duration_seconds,
            events=events,
            removed_segments=[],
        )
        try:
            temp_wav.unlink(missing_ok=True)
            raw_srt.unlink(missing_ok=True)
            temp_dir.rmdir()
        except OSError:
            pass
        return SttJobResult(
            source_path=source_path,
            workspace=output_dir,
            output_srt=output_srt,
            segment_count=len(raw_segments),
            duration_seconds=duration_seconds,
        )
    removed_segments: list[SttRemovedSegment] = []
    raw_segments, non_dialogue_removed = filter_stt_dialogue_segments(
        raw_segments,
        removed_segments=removed_segments,
    )
    if non_dialogue_removed:
        emit(f"STT: pominieto opisy niedialogowe {non_dialogue_removed}")
    if not raw_segments:
        raise RuntimeError("STT nie wykryl zadnego dialogu w pliku.")
    raw_segments, repeated_removed = filter_repeated_stt_hallucinations(
        raw_segments,
        removed_segments=removed_segments,
    )
    if repeated_removed:
        emit(f"STT: pominieto powtorzone fragmenty {repeated_removed}")
    if not raw_segments:
        raise RuntimeError("STT nie wykryl zadnego dialogu w pliku.")
    raw_segments, merged_segments = merge_short_stt_segments(raw_segments, profile=subtitle_profile)
    if merged_segments:
        emit(f"STT: scalono krotkie fragmenty {merged_segments}")
    segments = split_stt_subtitle_segments(raw_segments, profile=subtitle_profile)
    save_srt(output_srt, segments)
    emit(f"STT: zapisano napisy {output_srt.name}")
    duration_seconds = max(0.0, time.monotonic() - started_at)
    save_stt_diagnostics(
        output_dir=output_dir,
        temp_wav=temp_wav,
        source_path=source_path,
        settings=settings,
        transcribe_kwargs={
            "command": command,
            "runtime": runtime_variant,
            "device": runtime_device,
            "threads": int(settings.whisper_cpp_threads or 0),
        },
        detected_language="" if language == "auto" else language,
        subtitle_profile=subtitle_profile,
        raw_segment_count=raw_segment_count,
        final_segment_count=len(segments),
        duration_seconds=duration_seconds,
        events=events,
        removed_segments=removed_segments,
    )
    try:
        temp_wav.unlink(missing_ok=True)
        raw_srt.unlink(missing_ok=True)
        temp_dir.rmdir()
    except OSError:
        pass

    return SttJobResult(
        source_path=source_path,
        workspace=output_dir,
        output_srt=output_srt,
        segment_count=len(segments),
        duration_seconds=duration_seconds,
    )


def run_whisperx_stt(
    source_path: Path,
    paths: AppPaths,
    settings: SttSettings,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> SttJobResult:
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Brak pliku: {source_path}")
    if not is_stt_input_file(source_path):
        raise ValueError(f"Nieobslugiwany plik STT: {source_path.name}")

    ffmpeg = find_ffmpeg(paths)
    if ffmpeg is None:
        raise RuntimeError(f"Brak ffmpeg. {BINARY_LOOKUP_HINT}")
    ffprobe = find_ffprobe(paths)
    if ffprobe is None:
        raise RuntimeError(f"Brak ffprobe. {BINARY_LOOKUP_HINT}")

    workspace = lektorai_workspace_for(source_path)
    output_stem = next_stt_output_stem(workspace, source_path, engine="whisperx")
    output_dir = workspace / output_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = paths.whisperx_stt_dir / "temp" / output_stem
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = temp_dir / "stt_audio.wav"
    raw_json = temp_dir / "stt_audio.json"
    output_srt = output_dir / f"{output_stem}.srt"

    started_at = time.monotonic()
    events: list[str] = []

    def emit(message: str) -> None:
        events.append(message)
        _emit(progress, message)

    device = str(settings.whisperx_device or "cpu").strip() or "cpu"
    compute_type = whisper_qc_effective_compute_type(device, settings.whisperx_compute_type)
    language = str(settings.language or "auto").strip().lower()

    emit("STT: przygotowanie modulu WhisperX")
    python_path = ensure_whisperx_runtime(paths, progress=progress)
    if device.startswith("cuda"):
        ensure_whisperx_gpu_runtime(paths, progress=progress)
    _raise_if_cancelled(cancel_requested)

    emit("STT: przygotowanie audio")
    channels = primary_audio_channels(probe_audio_streams(ffprobe, source_path))
    extract_audio_for_stt(ffmpeg, source_path, temp_wav, channels, cancel_requested=cancel_requested)
    _raise_if_cancelled(cancel_requested)

    command = build_whisperx_command(
        python_path=python_path,
        input_wav=temp_wav,
        output_dir=temp_dir,
        model=settings.model,
        model_dir=paths.whisperx_cache_dir,
        language=language,
        device=device,
        compute_type=compute_type,
        batch_size=8,
        beam_size=STT_ACCURACY_BEAM_SIZE.get(str(settings.accuracy or "standard"), 5),
        no_align=False,
    )
    emit("STT: transkrypcja")
    last_transcription_progress = -1

    def on_whisperx_output(line: str) -> None:
        nonlocal last_transcription_progress
        match = WHISPERX_PROGRESS_RE.search(line)
        if not match:
            return
        percent = max(0, min(100, int(float(match.group(1)))))
        rounded = (percent // 10) * 10
        if rounded <= 0 or rounded <= last_transcription_progress:
            return
        last_transcription_progress = rounded
        emit(f"WhisperX: transkrypcja {rounded}%")

    try:
        result = _run_command_capture_text(
            command,
            timeout=7200,
            cancel_requested=cancel_requested,
            output_callback=on_whisperx_output,
            env=whisperx_runtime_env(paths, device),
        )
    except RuntimeError as exc:
        if device.startswith("cuda"):
            raise RuntimeError(
                "Nie udalo sie uruchomic WhisperX na GPU. "
                "Sprawdz biblioteki CUDA Runtime w aplikacji albo wybierz CPU.\n"
                f"{exc}"
            ) from exc
        raise
    if result.returncode == 0 and last_transcription_progress < 100:
        emit("WhisperX: transkrypcja 100%")
    if result.returncode != 0:
        tail = "\n".join((result.output or "").splitlines()[-30:])
        if device.startswith("cuda"):
            raise RuntimeError(
                "WhisperX nie utworzyl napisow na GPU. "
                "Sprawdz biblioteki CUDA Runtime w aplikacji albo wybierz CPU.\n"
                f"{tail}"
            )
        raise RuntimeError(f"WhisperX nie utworzyl napisow.\n{tail}")
    if not raw_json.is_file():
        tail = "\n".join((result.output or "").splitlines()[-30:])
        raise RuntimeError(f"WhisperX nie zapisal pliku JSON.\n{tail}")

    raw_segments, detected_language = load_whisperx_json_segments(raw_json)
    if not raw_segments:
        raise RuntimeError("STT nie wykryl zadnej mowy w pliku.")
    raw_segment_count = len(raw_segments)

    subtitle_profile = subtitle_profile_for_language(language if language != "auto" else detected_language)
    emit(f"STT: format napisow {subtitle_profile.label}")
    if not settings.postprocess_enabled:
        emit("STT: formatowanie LektorAI wylaczone")
        save_srt(output_srt, raw_segments)
        emit(f"STT: zapisano napisy {output_srt.name}")
        duration_seconds = max(0.0, time.monotonic() - started_at)
        save_stt_diagnostics(
            output_dir=output_dir,
            temp_wav=temp_wav,
            source_path=source_path,
            settings=settings,
            transcribe_kwargs={
                "command": command,
                "device": device,
                "compute_type": compute_type,
                "batch_size": 8,
                "align": True,
            },
            detected_language=detected_language,
            subtitle_profile=subtitle_profile,
            raw_segment_count=raw_segment_count,
            final_segment_count=len(raw_segments),
            duration_seconds=duration_seconds,
            events=events,
            removed_segments=[],
        )
        try:
            temp_wav.unlink(missing_ok=True)
            raw_json.unlink(missing_ok=True)
            temp_dir.rmdir()
        except OSError:
            pass
        return SttJobResult(
            source_path=source_path,
            workspace=output_dir,
            output_srt=output_srt,
            segment_count=len(raw_segments),
            duration_seconds=duration_seconds,
        )
    removed_segments: list[SttRemovedSegment] = []
    raw_segments, non_dialogue_removed = filter_stt_dialogue_segments(
        raw_segments,
        removed_segments=removed_segments,
    )
    if non_dialogue_removed:
        emit(f"STT: pominieto opisy niedialogowe {non_dialogue_removed}")
    if not raw_segments:
        raise RuntimeError("STT nie wykryl zadnego dialogu w pliku.")
    raw_segments, repeated_removed = filter_repeated_stt_hallucinations(
        raw_segments,
        removed_segments=removed_segments,
    )
    if repeated_removed:
        emit(f"STT: pominieto powtorzone fragmenty {repeated_removed}")
    if not raw_segments:
        raise RuntimeError("STT nie wykryl zadnego dialogu w pliku.")

    raw_segments, merged_segments = merge_short_stt_segments(raw_segments, profile=subtitle_profile)
    if merged_segments:
        emit(f"STT: scalono krotkie fragmenty {merged_segments}")
    segments = split_stt_subtitle_segments(raw_segments, profile=subtitle_profile)
    save_srt(output_srt, segments)
    emit(f"STT: zapisano napisy {output_srt.name}")
    duration_seconds = max(0.0, time.monotonic() - started_at)
    save_stt_diagnostics(
        output_dir=output_dir,
        temp_wav=temp_wav,
        source_path=source_path,
        settings=settings,
        transcribe_kwargs={
            "command": command,
            "device": device,
            "compute_type": compute_type,
            "batch_size": 8,
            "align": True,
        },
        detected_language=detected_language,
        subtitle_profile=subtitle_profile,
        raw_segment_count=raw_segment_count,
        final_segment_count=len(segments),
        duration_seconds=duration_seconds,
        events=events,
        removed_segments=removed_segments,
    )
    try:
        temp_wav.unlink(missing_ok=True)
        raw_json.unlink(missing_ok=True)
        temp_dir.rmdir()
    except OSError:
        pass

    return SttJobResult(
        source_path=source_path,
        workspace=output_dir,
        output_srt=output_srt,
        segment_count=len(segments),
        duration_seconds=duration_seconds,
    )


def load_stt_whisper_model(model_name: str, device: str, compute_type: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = stt_model_cache_key(model_name, device, compute_type, cache_dir)
    model = _STT_WHISPER_MODELS.get(key)
    if model is not None:
        return model
    faster_whisper = import_faster_whisper_for_cache(cache_dir)
    WhisperModel = faster_whisper.WhisperModel
    model = WhisperModel(
        model_name,
        **faster_whisper_device_kwargs(device),
        compute_type=compute_type,
        download_root=str(cache_dir),
    )
    _STT_WHISPER_MODELS[key] = model
    return model


def load_whisperx_json_segments(json_path: Path) -> tuple[list[SubtitleSegment], str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return [], ""
    detected_language = str(data.get("language", "") or "").strip().lower()
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        return [], detected_language
    segments: list[SubtitleSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start_ms = max(0, int(round(float(item.get("start", 0.0) or 0.0) * 1000)))
        end_ms = max(start_ms + 1, int(round(float(item.get("end", 0.0) or 0.0) * 1000)))
        segments.append(
            SubtitleSegment(
                index=len(segments) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )
    return segments, detected_language


def default_stt_transcribe_kwargs(settings: SttSettings | None = None) -> dict[str, object]:
    settings = settings or SttSettings()
    accuracy = str(settings.accuracy or "standard").strip().lower()
    vad_sensitivity = str(settings.vad_sensitivity or "standard").strip().lower()
    kwargs: dict[str, object] = {
        "task": "transcribe",
        "beam_size": STT_ACCURACY_BEAM_SIZE.get(accuracy, STT_ACCURACY_BEAM_SIZE["standard"]),
        "temperature": 0.0,
        "vad_filter": bool(settings.vad_enabled),
        "condition_on_previous_text": False,
        "word_timestamps": False,
    }
    if settings.vad_enabled:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": STT_VAD_MIN_SILENCE_MS.get(
                vad_sensitivity,
                STT_VAD_MIN_SILENCE_MS["standard"],
            )
        }
    return kwargs


def save_stt_diagnostics(
    output_dir: Path,
    temp_wav: Path,
    source_path: Path,
    settings: SttSettings,
    transcribe_kwargs: dict[str, object],
    detected_language: str,
    subtitle_profile: SttSubtitleProfile,
    raw_segment_count: int,
    final_segment_count: int,
    duration_seconds: float,
    events: list[str],
    removed_segments: list[SttRemovedSegment] | None = None,
) -> None:
    removed_segments = list(removed_segments or [])
    if settings.save_prepared_audio:
        try:
            shutil.copy2(temp_wav, output_dir / "stt_audio_prepared.wav")
        except OSError:
            events.append("STT: nie udalo sie zachowac przygotowanego audio")
    if settings.save_report:
        removed_srt_name = "stt_removed_segments.srt"
        report = {
            "source_file": source_path.name,
            "source_path": str(source_path),
            "settings": {
                "engine": settings.engine,
                "model": settings.model,
                "language": settings.language,
                "device": settings.device,
                "compute_type": settings.compute_type,
                "accuracy": settings.accuracy,
                "vad_enabled": settings.vad_enabled,
                "vad_sensitivity": settings.vad_sensitivity,
                "postprocess_enabled": settings.postprocess_enabled,
                "whisper_cpp_runtime": settings.whisper_cpp_runtime,
                "whisper_cpp_device": settings.whisper_cpp_device,
                "whisper_cpp_threads": settings.whisper_cpp_threads,
                "whisperx_device": settings.whisperx_device,
                "whisperx_compute_type": settings.whisperx_compute_type,
            },
            "transcribe": transcribe_kwargs,
            "detected_language": detected_language,
            "subtitle_profile": {
                "id": subtitle_profile.profile_id,
                "label": subtitle_profile.label,
                "max_line_chars": subtitle_profile.max_line_chars,
                "max_lines": subtitle_profile.max_lines,
                "max_cps": subtitle_profile.max_cps,
            },
            "segments": {
                "raw": raw_segment_count,
                "final": final_segment_count,
            },
            "removed_segments": {
                "count": len(removed_segments),
                "srt_file": removed_srt_name if removed_segments else "",
                "items": stt_removed_segments_payload(removed_segments),
            },
            "duration_seconds": round(float(duration_seconds), 3),
        }
        try:
            (output_dir / "stt_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            events.append("STT: nie udalo sie zachowac raportu STT")
        if removed_segments:
            try:
                save_srt(output_dir / removed_srt_name, stt_removed_segments_as_srt(removed_segments))
                events.append(f"STT: zapisano odrzucone fragmenty {removed_srt_name}")
            except OSError:
                events.append("STT: nie udalo sie zachowac odrzuconych fragmentow")
    if settings.save_log:
        try:
            (output_dir / "stt_log.txt").write_text("\n".join(events) + "\n", encoding="utf-8")
        except OSError:
            pass


def stt_removed_segments_payload(removed_segments: list[SttRemovedSegment]) -> list[dict[str, object]]:
    return [
        {
            "source_index": item.source_index,
            "start": stt_ms_to_timestamp(item.start_ms),
            "end": stt_ms_to_timestamp(item.end_ms),
            "start_ms": item.start_ms,
            "end_ms": item.end_ms,
            "stage": item.stage,
            "reason": item.reason,
            "text": item.text,
        }
        for item in removed_segments
    ]


def stt_removed_segments_as_srt(removed_segments: list[SttRemovedSegment]) -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            index=index,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            text=f"[{item.reason}] {item.text}".strip(),
        )
        for index, item in enumerate(removed_segments, 1)
    ]


def remember_removed_stt_segment(
    removed_segments: list[SttRemovedSegment] | None,
    segment: SubtitleSegment,
    stage: str,
    reason: str,
) -> None:
    if removed_segments is None:
        return
    removed_segments.append(
        SttRemovedSegment(
            source_index=int(segment.index),
            start_ms=int(segment.start_ms),
            end_ms=int(segment.end_ms),
            text=str(segment.text or "").strip(),
            stage=stage,
            reason=reason,
        )
    )


def stt_ms_to_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def split_stt_subtitle_segments(
    segments: list[SubtitleSegment],
    max_chars: int | None = None,
    max_line_chars: int | None = None,
    profile: SttSubtitleProfile | None = None,
) -> list[SubtitleSegment]:
    profile = _effective_subtitle_profile(profile, max_chars=max_chars, max_line_chars=max_line_chars)
    result: list[SubtitleSegment] = []
    for segment in segments:
        segment_max_chars = _max_chars_for_segment_duration(segment, profile)
        parts = split_stt_text(segment.text, max_chars=segment_max_chars, profile=profile)
        if not parts:
            continue
        if len(parts) == 1:
            result.append(
                SubtitleSegment(
                    index=len(result) + 1,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=parts[0],
                )
            )
            continue
        duration = max(1, int(segment.end_ms) - int(segment.start_ms))
        weights = [max(1, len(part.replace("\n", " "))) for part in parts]
        total_weight = max(1, sum(weights))
        current_start = int(segment.start_ms)
        elapsed_weight = 0
        for part_index, (part, weight) in enumerate(zip(parts, weights), 1):
            elapsed_weight += weight
            if part_index == len(parts):
                current_end = int(segment.end_ms)
            else:
                current_end = int(segment.start_ms) + round(duration * elapsed_weight / total_weight)
                current_end = max(current_start + 1, min(current_end, int(segment.end_ms)))
            result.append(
                SubtitleSegment(
                    index=len(result) + 1,
                    start_ms=current_start,
                    end_ms=current_end,
                    text=part,
                )
            )
            current_start = current_end
    return result


def filter_stt_dialogue_segments(
    segments: list[SubtitleSegment],
    removed_segments: list[SttRemovedSegment] | None = None,
) -> tuple[list[SubtitleSegment], int]:
    result: list[SubtitleSegment] = []
    removed = 0
    for segment in segments:
        if is_stt_non_dialogue_text(segment.text):
            removed += 1
            remember_removed_stt_segment(
                removed_segments,
                segment,
                stage="dialogue_filter",
                reason="opis niedialogowy",
            )
            continue
        result.append(
            SubtitleSegment(
                index=len(result) + 1,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=segment.text,
            )
        )
    return result, removed


def filter_repeated_stt_hallucinations(
    segments: list[SubtitleSegment],
    max_gap_ms: int = 500,
    max_group_duration_ms: int = 3000,
    removed_segments: list[SttRemovedSegment] | None = None,
) -> tuple[list[SubtitleSegment], int]:
    result: list[SubtitleSegment] = []
    removed = 0
    index = 0
    while index < len(segments):
        repeat = _repeated_word_signature(segments[index].text)
        if repeat is None:
            result.append(segments[index])
            index += 1
            continue
        word, repeats = repeat
        group_end = index + 1
        total_repeats = repeats
        while group_end < len(segments):
            next_repeat = _repeated_word_signature(segments[group_end].text)
            gap_ms = int(segments[group_end].start_ms) - int(segments[group_end - 1].end_ms)
            if next_repeat is None or next_repeat[0] != word or gap_ms > max_gap_ms:
                break
            total_repeats += next_repeat[1]
            group_end += 1
        group = segments[index:group_end]
        group_duration_ms = int(group[-1].end_ms) - int(group[0].start_ms)
        if total_repeats >= 3 and word not in COMMON_DIALOGUE_REPEATS and group_duration_ms <= max_group_duration_ms:
            removed += len(group)
            for item in group:
                remember_removed_stt_segment(
                    removed_segments,
                    item,
                    stage="repeat_filter",
                    reason=f"powtorzony fragment: {word}",
                )
            index = group_end
            continue
        result.extend(group)
        index = group_end
    return [
        SubtitleSegment(index=new_index, start_ms=item.start_ms, end_ms=item.end_ms, text=item.text)
        for new_index, item in enumerate(result, 1)
    ], removed


def merge_short_stt_segments(
    segments: list[SubtitleSegment],
    profile: SttSubtitleProfile | None = None,
    max_gap_ms: int = 800,
    max_merge_chars: int | None = None,
) -> tuple[list[SubtitleSegment], int]:
    profile = _effective_subtitle_profile(profile)
    merge_limit = int(max_merge_chars) if max_merge_chars is not None else max(240, profile.max_chars * 4)
    merged: list[SubtitleSegment] = []
    merge_count = 0
    for segment in segments:
        text = re.sub(r"\s+", " ", str(segment.text or "")).strip()
        if not text:
            continue
        current = SubtitleSegment(
            index=0,
            start_ms=int(segment.start_ms),
            end_ms=max(int(segment.start_ms), int(segment.end_ms)),
            text=text,
        )
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        if _should_merge_stt_segments(previous, current, max_gap_ms=max_gap_ms, max_merge_chars=merge_limit):
            merged[-1] = SubtitleSegment(
                index=0,
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, current.end_ms),
                text=f"{previous.text} {current.text}".strip(),
            )
            merge_count += 1
        else:
            merged.append(current)
    return [
        SubtitleSegment(index=index, start_ms=item.start_ms, end_ms=item.end_ms, text=item.text)
        for index, item in enumerate(merged, 1)
    ], merge_count


def _should_merge_stt_segments(
    previous: SubtitleSegment,
    current: SubtitleSegment,
    max_gap_ms: int,
    max_merge_chars: int,
) -> bool:
    gap_ms = int(current.start_ms) - int(previous.end_ms)
    if gap_ms < 0:
        gap_ms = 0
    if gap_ms > max(0, int(max_gap_ms)):
        return False
    combined = f"{previous.text} {current.text}".strip()
    if len(combined) > max(1, int(max_merge_chars)):
        return False
    if not _stt_text_ends_sentence(previous.text):
        return True
    if _is_single_word_sentence_tail(current.text):
        return True
    return False


def _stt_text_ends_sentence(text: str) -> bool:
    stripped = str(text or "").strip().rstrip("\"'”’)]}")
    return bool(stripped) and stripped[-1] in ".?!"


def _repeated_word_signature(text: str) -> tuple[str, int] | None:
    words = [word.lower() for word in _text_words(text)]
    if not words:
        return None
    if len(set(words)) != 1:
        return None
    return words[0], len(words)


def _is_single_word_sentence_tail(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return len(_text_words(stripped)) == 1 and _stt_text_ends_sentence(stripped)


def _text_words(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z]+(?:['-][0-9A-Za-z]+)?", str(text or ""))


def is_stt_non_dialogue_text(text: str) -> bool:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return True
    if re.fullmatch(r"[♪♫♬♩\s]+", raw):
        return True
    if raw.startswith(("♪", "♫", "♬", "♩")) or raw.endswith(("♪", "♫", "♬", "♩")):
        return True

    stripped = raw.strip()
    wrapped = False
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == "*":
        wrapped = True
        stripped = stripped[1:-1].strip()
    elif len(stripped) >= 2 and ((stripped[0], stripped[-1]) in {("[", "]"), ("(", ")"), ("{", "}")}):
        wrapped = True
        stripped = stripped[1:-1].strip()

    normalized = re.sub(r"[^0-9a-zA-Z]+", " ", stripped).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return True
    if normalized in NON_DIALOGUE_EXACT_TEXTS:
        return True
    if wrapped and any(keyword in normalized.split() for keyword in NON_DIALOGUE_KEYWORDS):
        return True
    return False


def split_stt_text(
    text: str,
    max_chars: int | None = None,
    max_line_chars: int | None = None,
    profile: SttSubtitleProfile | None = None,
) -> list[str]:
    profile = _effective_subtitle_profile(profile, max_chars=max_chars, max_line_chars=max_line_chars)
    max_chars = profile.max_chars
    max_line_chars = profile.max_line_chars
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars or _is_complete_stt_sentence(normalized):
        return [wrap_stt_subtitle_text(normalized, max_line_chars)]
    sentence_parts = [part.strip() for part in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", normalized) if part.strip()]
    source_parts = sentence_parts if len(sentence_parts) > 1 else [normalized]
    result: list[str] = []
    for part in source_parts:
        if len(part) <= max_chars or _is_complete_stt_sentence(part):
            result.append(part)
            continue
        result.extend(_split_long_stt_text_part(part, max_chars))
    return [wrap_stt_subtitle_text(part, max_line_chars) for part in result if part.strip()]


def wrap_stt_subtitle_text(text: str, max_line_chars: int = FALLBACK_SUBTITLE_PROFILE.max_line_chars) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    return normalized


def _split_long_stt_text_part(text: str, max_chars: int) -> list[str]:
    prepared = re.sub(r"([,;:])\s+", r"\1|", str(text or "").strip())
    chunks: list[str] = []
    current = ""
    for raw_piece in prepared.split("|"):
        piece = raw_piece.strip()
        if not piece:
            continue
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_split_words_to_limit(chunk, max_chars))
    return final


def _split_words_to_limit(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for word in str(text or "").split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _is_complete_stt_sentence(text: str) -> bool:
    stripped = str(text or "").strip()
    if not _stt_text_ends_sentence(stripped):
        return False
    return bool(_text_words(stripped))


def _effective_subtitle_profile(
    profile: SttSubtitleProfile | None,
    max_chars: int | None = None,
    max_line_chars: int | None = None,
) -> SttSubtitleProfile:
    source = profile or FALLBACK_SUBTITLE_PROFILE
    line_limit = int(max_line_chars if max_line_chars is not None else source.max_line_chars)
    max_lines = max(1, int(source.max_lines))
    char_limit = int(max_chars if max_chars is not None else source.max_chars)
    char_limit = max(1, char_limit)
    return SttSubtitleProfile(
        profile_id=source.profile_id,
        label=source.label,
        max_line_chars=max(1, line_limit),
        max_lines=max_lines,
        max_chars=char_limit,
        max_cps=max(0.0, float(source.max_cps)),
        min_duration_ms=max(1, int(source.min_duration_ms)),
    )


def _max_chars_for_segment_duration(segment: SubtitleSegment, profile: SttSubtitleProfile) -> int:
    duration_ms = max(profile.min_duration_ms, int(segment.end_ms) - int(segment.start_ms))
    if profile.max_cps <= 0:
        return profile.max_chars
    cps_limit = max(1, int((duration_ms / 1000.0) * profile.max_cps))
    return max(1, min(profile.max_chars, max(profile.max_line_chars, cps_limit)))


def stt_model_cache_key(model_name: str, device: str, compute_type: str, cache_dir: Path) -> tuple[str, str, str, str]:
    return (
        str(model_name or "small").strip() or "small",
        str(device or "cpu").strip() or "cpu",
        str(compute_type or "int8").strip() or "int8",
        str(cache_dir),
    )


def extract_audio_for_stt(
    ffmpeg: Path,
    source_path: Path,
    output_wav: Path,
    channels: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    command = _extract_audio_command(ffmpeg, source_path, output_wav, use_center_channel=channels >= 6)
    result = _run_command_capture_text(command, timeout=1800, cancel_requested=cancel_requested)
    if result.returncode == 0:
        return
    if channels >= 6:
        fallback = _extract_audio_command(ffmpeg, source_path, output_wav, use_center_channel=False)
        result = _run_command_capture_text(fallback, timeout=1800, cancel_requested=cancel_requested)
        if result.returncode == 0:
            return
    tail = "\n".join((result.output or "").splitlines()[-20:])
    raise RuntimeError(f"Nie udalo sie przygotowac audio dla STT.\n{tail}")


def next_stt_output_stem(
    workspace: Path,
    source_path: Path,
    created_at: datetime | None = None,
    engine: str = "faster_whisper",
) -> str:
    workspace.mkdir(parents=True, exist_ok=True)
    engine_code = STT_ENGINE_CODES.get(str(engine or "").strip().lower(), "STT")
    base = f"{compact_run_timestamp(created_at)}_{compact_source_stem(source_path)}_STT_{engine_code}"
    candidate = base
    counter = 2
    while (workspace / candidate).exists() or (workspace / f"{candidate}.srt").exists():
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def _extract_audio_command(ffmpeg: Path, source_path: Path, output_wav: Path, use_center_channel: bool) -> list[str]:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-map_metadata",
        "-1",
    ]
    if use_center_channel:
        command.extend(["-af", "pan=mono|c0=FC,aresample=16000"])
    else:
        command.extend(["-ac", "1", "-ar", "16000"])
    command.extend(["-c:a", "pcm_s16le", str(output_wav)])
    return command


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Przerwano przez uzytkownika")


def _looks_like_missing_cuda_runtime(output: str) -> bool:
    text = str(output or "").lower()
    return any(
        marker in text
        for marker in (
            "cublas64_13.dll",
            "cublaslt64_13.dll",
            "cudart64_13.dll",
            "cuda toolkit",
            "could not load",
            "nie mozna odnalezc",
            "nie można odnaleźć",
            "brakuje",
        )
    )

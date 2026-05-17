from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from app.core.dictionary import load_dictionary
from app.core.media_tools import (
    BINARY_LOOKUP_HINT,
    audio_stream_log_lines,
    audio_stream_summary,
    apply_short_audio_fade,
    convert_audio_to_wav,
    encode_wav_to_aac,
    extract_first_subtitle_to_srt,
    find_ffmpeg,
    find_mkvmerge,
    find_ffprobe,
    is_video_file,
    mux_lektor_track,
    normalize_lektor_wav,
    primary_audio_channels,
    probe_audio_streams,
    probe_media_duration,
    prepare_voice_sample,
    sanitize_lektor_delay_ms,
    selected_background_audio_stream,
    surround_label_for_channels,
    supported_voice_sample_extensions,
    trim_fixed_and_fade_wav_edges,
    voice_sample_sample_rate,
    wav_audio_diagnostics,
)
from app.core.paths import AppPaths
from app.core.version import APP_NAME, APP_VERSION
from app.engines.builtin.edge import synthesize_edge_mp3_sync
from app.engines.builtin.openai_tts import synthesize_openai_wav_sync
from app.engines.manager import EngineManager
from app.engines.protocol import EngineRequest, EngineResult, SegmentRequest, write_request
from app.engines.runner import DEFAULT_WORKER_TIMEOUT_S, EngineWorkerRunner
from app.engines.schemas import EngineStatus
from app.engines.voice_sample_rules import (
    validate_voice_sample_duration,
    voice_sample_config_key,
    voice_sample_rule,
)
from app.pipeline.audio_qc import analyze_audio_candidate, analyze_generated_segments, score_audio_qc, summarize_audio_qc
from app.pipeline.audio_timeline import build_lektor_wav
from app.pipeline.manifest import write_segments_manifest, write_skipped_segments_manifest
from app.pipeline.progress import encode_progress_marker
from app.pipeline.subtitles import (
    SUPPORTED_SUBTITLE_EXTENSIONS,
    SubtitleSegment,
    apply_dictionary,
    load_srt,
    load_txt_as_segment,
    normalize_tts_text,
    save_srt,
)
from app.pipeline.summary import rel_path, write_run_error, write_run_summary
from app.pipeline.whisper_qc import score_whisper_transcript, transcribe_audio_with_faster_whisper
from app.pipeline.workspace import engine_short_code, lektor_assets_dir, lektorai_workspace_for, next_output_stem
from app.stt.faster_whisper_runtime import ensure_faster_whisper_runtime


SIDECAR_SUFFIXES = (
    ".pl",
    ".pol",
    ".polish",
    ".polski",
    ".polskie",
    ".PL",
    ".POL",
    ".POLISH",
    ".POLSKI",
    ".POLSKIE",
    "",
    ".forced",
    ".forced.pl",
    ".pl.forced",
)

DETERMINISTIC_RETRY_ENGINES = {"piper", "supertonic"}
PUNCTUATION_RETRY_ENGINES = {"piper", "supertonic"}

EDGE_TRIM_FADE_MS = 12


def local_worker_timeout_seconds() -> int:
    return DEFAULT_WORKER_TIMEOUT_S


@dataclass(frozen=True)
class TTSJobResult:
    engine_id: str
    output_stem: str
    workspace: Path
    subtitle_path: Path
    lektor_dir: Path
    segment_count: int
    generation_seconds: float
    qc_warning_count: int = 0
    manifest_path: Path | None = None
    summary_path: Path | None = None
    lektor_wav_path: Path | None = None
    lektor_before_normalization_path: Path | None = None
    lektor_m4a_path: Path | None = None
    encoded_lektor_m4a_path: Path | None = None
    output_video_path: Path | None = None


@dataclass
class QCRetrySummary:
    audio_retry_segments: set[int] = field(default_factory=set)
    audio_extra_attempts: int = 0
    speech_retry_segments: set[int] = field(default_factory=set)
    speech_extra_attempts: int = 0

    def record_audio_retry(self, segment_ordinal: int) -> None:
        self.audio_retry_segments.add(int(segment_ordinal))
        self.audio_extra_attempts += 1

    def record_speech_retry(self, segment_ordinal: int) -> None:
        self.speech_retry_segments.add(int(segment_ordinal))
        self.speech_extra_attempts += 1


@dataclass(frozen=True)
class GenerationOutput:
    generated_segments: list[tuple[int, Path]]
    generation_seconds: float
    segment_analysis: list[dict[str, Any]]


def run_tts_job(
    source_path: Path,
    engine_id: str,
    paths: AppPaths,
    manager: EngineManager,
    progress,
    keep_lektor_assets: bool | None = None,
    aac_bitrate: str = "256k",
    lektor_lufs: int = -14,
    lektor_weight: float = 2.0,
    background_lufs: int = -18,
    background_weight: float = 1.0,
    lektor_delay_ms: int = 0,
    create_stereo_for_surround: bool = True,
    cancel_requested: Callable[[], bool] | None = None,
) -> TTSJobResult:
    job_started = perf_counter()
    run_started_at = datetime.now()
    source_path = source_path.resolve()
    config_path = manager.ensure_engine_config(engine_id)
    dictionary_path = manager.ensure_engine_dictionary(engine_id)
    config = _load_json(config_path)
    dictionary = load_dictionary(dictionary_path)
    keep_lektor_assets = bool(keep_lektor_assets) if keep_lektor_assets is not None else False

    workspace = lektorai_workspace_for(source_path)
    workspace.mkdir(parents=True, exist_ok=True)
    output_stem = next_output_stem(workspace, source_path, engine_id, created_at=run_started_at)
    subtitle_path = workspace / f"{output_stem}.srt"
    lektor_dir = lektor_assets_dir(workspace, output_stem)
    segments_dir = lektor_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _run_tts_job_prepared(
            source_path=source_path,
            engine_id=engine_id,
            paths=paths,
            manager=manager,
            progress=progress,
            keep_lektor_assets=keep_lektor_assets,
            aac_bitrate=aac_bitrate,
            lektor_lufs=lektor_lufs,
            lektor_weight=lektor_weight,
            background_lufs=background_lufs,
            background_weight=background_weight,
            lektor_delay_ms=lektor_delay_ms,
            create_stereo_for_surround=create_stereo_for_surround,
            cancel_requested=cancel_requested,
            job_started=job_started,
            config=config,
            dictionary=dictionary,
            workspace=workspace,
            output_stem=output_stem,
            subtitle_path=subtitle_path,
            lektor_dir=lektor_dir,
            segments_dir=segments_dir,
            run_started_at=run_started_at,
        )
    except Exception as exc:
        try:
            write_run_error(
                _run_report_path(lektor_dir, output_stem, "error"),
                {
                    "report_type": "error",
                    "run_id": output_stem,
                    "run_timestamp": run_started_at.isoformat(timespec="seconds"),
                    "source_filename": source_path.name,
                    "source_stem": source_path.stem,
                    "tts_engine_short": engine_short_code(engine_id),
                    "llm_analysis_hint": "Ten raport opisuje nieudany przebieg konwersji. Sprawdz error_type, error, engine_id, ustawienia TTS i etap, na ktorym praca zostala przerwana.",
                    "app_name": APP_NAME,
                    "app_version": APP_VERSION,
                    "source_path": str(source_path),
                    "source_name": source_path.name,
                    "engine_id": engine_id,
                    "output_stem": output_stem,
                    "workspace": str(workspace),
                    "pipeline_seconds": round(float(perf_counter() - job_started), 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception as write_exc:
            progress(f"Run error: nie zapisano run_error.json ({write_exc})")
        raise


def _run_tts_job_prepared(
    source_path: Path,
    engine_id: str,
    paths: AppPaths,
    manager: EngineManager,
    progress,
    keep_lektor_assets: bool,
    aac_bitrate: str,
    lektor_lufs: int,
    lektor_weight: float,
    background_lufs: int,
    background_weight: float,
    lektor_delay_ms: int,
    create_stereo_for_surround: bool,
    cancel_requested: Callable[[], bool] | None,
    job_started: float,
    config: dict,
    dictionary: dict,
    workspace: Path,
    output_stem: str,
    subtitle_path: Path,
    lektor_dir: Path,
    segments_dir: Path,
    run_started_at: datetime,
) -> TTSJobResult:
    _raise_if_cancelled(cancel_requested)
    diagnostics = _diagnostic_keep_flags(config, keep_lektor_assets)
    source_duration, source_audio_streams = _source_media_diagnostics(source_path, paths)
    if is_video_file(source_path):
        progress(_format_source_media_message(source_duration, source_audio_streams))
    input_subtitle_path = _prepare_input_subtitles(source_path, paths, lektor_dir, progress, cancel_requested=cancel_requested)
    _raise_if_cancelled(cancel_requested)
    segments = _load_segments(input_subtitle_path)
    if not segments:
        raise RuntimeError("Brak tekstu do syntezy.")

    normalize_text = _bool_config(config.get("normalize_tts_text"), True)
    cleaned_segments = []
    for segment in segments:
        text = apply_dictionary(segment.text, dictionary)
        if normalize_text:
            text = normalize_tts_text(text)
        cleaned_segments.append(
            SubtitleSegment(
                index=segment.index,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=text,
            )
        )
    skipped_empty_segments = [segment for segment in cleaned_segments if not segment.text.strip()]
    empty_text_count = len(skipped_empty_segments)
    processed = [segment for segment in cleaned_segments if segment.text.strip()]
    if not processed:
        raise RuntimeError("Brak tekstu do syntezy po oczyszczeniu napisow.")
    if empty_text_count:
        progress(f"Napisy: pominieto puste segmenty {empty_text_count}")
    save_srt(subtitle_path, processed)
    lektor_delay_ms = sanitize_lektor_delay_ms(lektor_delay_ms)

    progress(f"{engine_id}: {len(processed)} segmentow")
    quality_controls = _quality_controls_summary(engine_id, config, ffmpeg_present=find_ffmpeg(paths) is not None)
    progress(f"Lancuch TTS: {_quality_chain_label(quality_controls)}")
    _emit_file_progress(progress, "tts", 0.0, "Generowanie segmentow TTS")
    if engine_id == "edge":
        generation = _generate_edge(processed, segments_dir, config, paths, lektor_dir, progress, cancel_requested)
    elif engine_id == "openai":
        generation = _generate_openai(processed, segments_dir, config, paths, lektor_dir, progress, cancel_requested)
    else:
        generation = _generate_local(
            processed,
            source_path,
            engine_id,
            segments_dir,
            lektor_dir,
            config,
            dictionary,
            paths,
            manager,
            progress,
            cancel_requested,
        )
    generated_segments = generation.generated_segments
    generation_seconds = generation.generation_seconds
    segment_analysis = generation.segment_analysis
    _raise_if_cancelled(cancel_requested)
    _emit_file_progress(progress, "tts", 1.0, "Generowanie segmentow TTS")

    manifest_path = lektor_dir / "segmenty.csv"
    _emit_file_progress(progress, "manifest", 0.5, "Manifest segmentow")
    write_segments_manifest(manifest_path, processed, generated_segments)
    progress("Manifest segmentow: zapisano")
    skipped_manifest_path: Path | None = None
    if skipped_empty_segments:
        skipped_manifest_path = lektor_dir / "skipped_segments.csv"
        write_skipped_segments_manifest(skipped_manifest_path, skipped_empty_segments, "pusty tekst po czyszczeniu")

    qc_warning_count = 0
    ffmpeg_for_qc = find_ffmpeg(paths)
    if ffmpeg_for_qc is not None:
        _emit_file_progress(progress, "audio_qc", 0.0, "Audio QC")
        progress("Audio QC: analiza segmentow")
        qc_results = analyze_generated_segments(
            ffmpeg_for_qc,
            generated_segments,
            processed,
            lektor_dir / "audio_qc.csv",
            lektor_dir / "temp_qc",
            cancel_requested=cancel_requested,
        )
        qc_warning_count = sum(1 for result in qc_results if result.warnings)
        progress(summarize_audio_qc(qc_results))
        _emit_file_progress(progress, "audio_qc", 1.0, "Audio QC")
    else:
        progress("Audio QC: pominieto, brak ffmpeg")

    lektor_wav_path: Path | None = None
    lektor_before_normalization_path: Path | None = None
    lektor_m4a_path: Path | None = None
    encoded_lektor_m4a_path: Path | None = None
    output_video_path: Path | None = None
    lektor_before_diagnostics: dict[str, int | float] | None = None
    lektor_after_diagnostics: dict[str, int | float] | None = None
    lektor_encoded_duration: float | None = None
    lektor_encoded_audio_streams: list[dict] = []
    ffmpeg = find_ffmpeg(paths)
    if ffmpeg is not None:
        _emit_file_progress(progress, "timeline", 0.0, "Skladanie sciezki lektora")
        progress("Skladanie sciezki lektora")
        lektor_before_normalization_path = lektor_dir / "lektor_przed_normalizacja.wav"
        minimum_timeline_duration = source_duration if is_video_file(source_path) else None
        if minimum_timeline_duration:
            progress(f"Sciezka lektora: dopelnienie cisza do czasu wideo {_format_duration_seconds(minimum_timeline_duration)}")
        timeline_segments = apply_lektor_delay_to_segments(generated_segments, lektor_delay_ms)
        if lektor_delay_ms:
            progress(f"Synchronizacja: przesuniecie lektora {_format_signed_duration_ms(lektor_delay_ms)}")
        timeline_stats = build_lektor_wav(
            ffmpeg,
            timeline_segments,
            lektor_before_normalization_path,
            lektor_dir / "temp_wav",
            minimum_duration_s=minimum_timeline_duration,
            cancel_requested=cancel_requested,
        )
        if timeline_stats.shifted_count:
            progress(f"Synchronizacja: opozniono {timeline_stats.shifted_count} kwestii, max +{_format_duration_ms(timeline_stats.max_shift_ms)}")
        else:
            progress("Synchronizacja: brak opoznien miedzy kwestiami")
        lektor_before_diagnostics = _safe_wav_diagnostics(lektor_before_normalization_path)
        if lektor_before_diagnostics:
            progress(f"Sciezka lektora WAV: {_format_wav_diagnostics(lektor_before_diagnostics)}")
        _emit_file_progress(progress, "timeline", 1.0, "Skladanie sciezki lektora")
        lektor_wav_path = lektor_dir / "lektor_po_normalizacji.wav"
        _emit_file_progress(progress, "normalization", 0.0, "Normalizacja sciezki lektora")
        progress(f"Normalizacja sciezki lektora: cel {int(lektor_lufs)} LUFS")
        normalize_lektor_wav(
            ffmpeg,
            lektor_before_normalization_path,
            lektor_wav_path,
            lektor_lufs,
            progress_callback=lambda ratio: _emit_file_progress(progress, "normalization", ratio, "Normalizacja sciezki lektora"),
            cancel_requested=cancel_requested,
        )
        lektor_after_diagnostics = _safe_wav_diagnostics(lektor_wav_path)
        if lektor_after_diagnostics:
            progress(f"Po normalizacji: {_format_wav_diagnostics(lektor_after_diagnostics)}")
        lektor_m4a_path = lektor_dir / "lektor_sciezka_audio.m4a"
        _emit_file_progress(progress, "encoding", 0.0, "Kodowanie AAC")
        progress(f"Kodowanie sciezki lektora: AAC {aac_bitrate}, 48 kHz")
        encode_wav_to_aac(
            ffmpeg,
            lektor_wav_path,
            lektor_m4a_path,
            aac_bitrate,
            progress_callback=lambda ratio: _emit_file_progress(progress, "encoding", ratio, "Kodowanie AAC"),
            duration_seconds=minimum_timeline_duration,
            cancel_requested=cancel_requested,
        )
        _emit_file_progress(progress, "encoding", 1.0, "Kodowanie AAC")
        encoded_lektor_m4a_path = lektor_m4a_path
        ffprobe_for_encoded = find_ffprobe(paths)
        if ffprobe_for_encoded is not None:
            lektor_encoded_duration = probe_media_duration(ffprobe_for_encoded, lektor_m4a_path)
            lektor_encoded_audio_streams = probe_audio_streams(ffprobe_for_encoded, lektor_m4a_path)
            progress(
                f"Audio lektora AAC: {_format_duration_seconds(lektor_encoded_duration)}, "
                f"{audio_stream_summary(lektor_encoded_audio_streams[0] if lektor_encoded_audio_streams else None)}"
            )
        progress(f"Audio lektora: {lektor_m4a_path.name}")
    else:
        progress("Sciezka lektora: pominieto, brak ffmpeg")

    audio_mix_stage_files: dict[str, Path] = {}
    if is_video_file(source_path):
        ffprobe = find_ffprobe(paths)
        if ffmpeg is None or ffprobe is None or lektor_wav_path is None:
            raise RuntimeError(f"Brak ffmpeg/ffprobe. {BINARY_LOOKUP_HINT}")
        mkvmerge = find_mkvmerge(paths)
        if mkvmerge is None:
            raise RuntimeError(f"Brak mkvmerge. MKVToolNix jest wymagany do zapisu wynikowego MKV. {BINARY_LOOKUP_HINT}")
        output_video_path = workspace / f"{output_stem}.mkv"
        audio_streams = source_audio_streams or probe_audio_streams(ffprobe, source_path)
        channel_count = primary_audio_channels(audio_streams)
        surround_label = surround_label_for_channels(channel_count)
        create_surround_track = bool(surround_label)
        create_stereo_track = (not create_surround_track) or bool(create_stereo_for_surround)
        output_track_label = _audio_output_track_label(surround_label, create_stereo_track)
        primary_audio_stream = selected_background_audio_stream(audio_streams)
        primary_audio = audio_stream_summary(primary_audio_stream)
        for line in audio_stream_log_lines(audio_streams):
            progress(line)
        progress(
            f"Miks audio: tlo {primary_audio} + lektor mono -> "
            f"{output_track_label}"
        )
        progress(
            f"Poziomy miksu: tlo {int(background_lufs)} LUFS x{_format_float(background_weight)}, "
            f"lektor {int(lektor_lufs)} LUFS x{_format_float(lektor_weight)}"
        )
        progress("Remux MKV: MKVToolNix")
        progress(f"Dodawanie sciezki lektora do MKV: {output_track_label}")
        _emit_file_progress(progress, "mux", 0.0, "Dodawanie do MKV")
        mux_lektor_track(
            ffmpeg,
            ffprobe,
            source_path,
            lektor_wav_path,
            output_video_path,
            f"{APP_NAME} {engine_id}",
            lektor_weight=lektor_weight,
            background_lufs=background_lufs,
            background_weight=background_weight,
            bitrate=aac_bitrate,
            create_stereo_for_surround=create_stereo_for_surround,
            diagnostic_dir=lektor_dir,
            keep_mixing_steps=diagnostics["save_audio_mix_steps"],
            mkvmerge=mkvmerge,
            progress_callback=lambda ratio: _emit_file_progress(progress, "mux", ratio, "Dodawanie do MKV"),
            cancel_requested=cancel_requested,
        )
        _emit_file_progress(progress, "mux", 1.0, "Dodawanie do MKV")
        output_audio_streams = probe_audio_streams(ffprobe, output_video_path)
        progress(_format_output_audio_message(output_audio_streams, surround_label, create_stereo_track))
        progress(f"Wideo wynikowe: {output_video_path.name}")
        audio_mix_stage_files = _expected_audio_mix_stage_files(output_video_path, lektor_dir, surround_label, create_stereo_track)
        if not diagnostics["save_audio_mix_steps"]:
            _unlink_if_file(lektor_m4a_path)
            lektor_m4a_path = None

    analysis_path = _run_report_path(lektor_dir, output_stem, "analysis")
    analysis_payload = _build_run_analysis(
        source_path=source_path,
        input_subtitle_path=input_subtitle_path,
        engine_id=engine_id,
        output_stem=output_stem,
        run_started_at=run_started_at,
        segments=processed,
        segment_analysis=segment_analysis,
        config=config,
        dictionary=dictionary,
        quality_controls=quality_controls,
        generation_seconds=generation_seconds,
        pipeline_seconds=perf_counter() - job_started,
        aac_bitrate=aac_bitrate,
        lektor_lufs=lektor_lufs,
        lektor_weight=lektor_weight,
        background_lufs=background_lufs,
        background_weight=background_weight,
        lektor_delay_ms=lektor_delay_ms,
        create_stereo_for_surround=create_stereo_for_surround,
        diagnostics=diagnostics,
        source_duration=source_duration,
        source_audio_streams=source_audio_streams,
        lektor_before_diagnostics=lektor_before_diagnostics,
        lektor_after_diagnostics=lektor_after_diagnostics,
        lektor_encoded_duration=lektor_encoded_duration,
        lektor_encoded_audio_streams=lektor_encoded_audio_streams,
    )
    write_run_summary(analysis_path, analysis_payload)
    if diagnostics["save_quality_report"]:
        progress("Raport jakosci: zapisano")

    summary_path = _run_report_path(lektor_dir, output_stem, "summary")
    _emit_file_progress(progress, "summary", 0.5, "Podsumowanie przebiegu")
    pipeline_seconds = perf_counter() - job_started
    write_run_summary(
        summary_path,
        {
            "report_type": "summary",
            "run_id": output_stem,
            "run_timestamp": run_started_at.isoformat(timespec="seconds"),
            "source_filename": source_path.name,
            "source_stem": source_path.stem,
            "tts_engine_short": engine_short_code(engine_id),
            "llm_analysis_hint": "To skrot przebiegu. Do oceny jakosci ustawien TTS uzyj raportu jakosci, bo zawiera segmenty, proby, score i retry.",
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "source_path": str(source_path),
            "source_name": source_path.name,
            "input_subtitle_path": str(input_subtitle_path),
            "engine_id": engine_id,
            "output_stem": output_stem,
            "input_segment_count": len(segments),
            "segment_count": len(processed),
            "skipped_empty_text_count": int(empty_text_count),
            "dictionary_entry_count": int(len(dictionary)),
            "generation_seconds": round(float(generation_seconds), 3),
            "pipeline_seconds": round(float(pipeline_seconds), 3),
            "qc_warning_count": int(qc_warning_count),
            "quality_controls": quality_controls,
            "audio_output": {
                "codec": "AAC",
                "bitrate": str(aac_bitrate),
                "lektor_lufs": int(lektor_lufs),
                "lektor_weight": float(lektor_weight),
                "background_lufs": int(background_lufs),
                "background_weight": float(background_weight),
                "lektor_delay_ms": int(lektor_delay_ms),
                "create_stereo_for_surround": bool(create_stereo_for_surround),
            },
            "diagnostic_keep": diagnostics,
            "audio_diagnostics": {
                "source_duration_s": round(float(source_duration or 0.0), 3),
                "source_primary_audio": audio_stream_summary(selected_background_audio_stream(source_audio_streams)),
                "lektor_before_normalization": lektor_before_diagnostics or {},
                "lektor_after_normalization": lektor_after_diagnostics or {},
                "lektor_encoded_duration_s": round(float(lektor_encoded_duration or 0.0), 3),
                "lektor_encoded_audio": audio_stream_summary(lektor_encoded_audio_streams[0] if lektor_encoded_audio_streams else None),
            },
            "workspace": str(workspace),
            "files": {
                "subtitle": rel_path(subtitle_path, workspace),
                "manifest": rel_path(manifest_path, workspace),
                "skipped_segments": rel_path(skipped_manifest_path, workspace),
                "audio_qc": rel_path(lektor_dir / "audio_qc.csv", workspace) if (lektor_dir / "audio_qc.csv").exists() else "",
                "run_analysis": rel_path(analysis_path, workspace) if diagnostics["save_quality_report"] else "",
                "lektor_przed_normalizacja": rel_path_if_exists(lektor_before_normalization_path, workspace),
                "lektor_wav": rel_path_if_exists(lektor_wav_path, workspace),
                "lektor_po_normalizacji": rel_path_if_exists(lektor_wav_path, workspace),
                "lektor_m4a": rel_path_if_exists(lektor_m4a_path, workspace),
                "lektor_m4a_encoded": rel_path(encoded_lektor_m4a_path, workspace),
                "audio_tlo_zrodlowe": rel_path_if_exists(audio_mix_stage_files.get("source_audio"), workspace),
                "audio_pl_2_0": rel_path_if_exists(audio_mix_stage_files.get("pl_2_0"), workspace),
                "audio_pl_5_1": rel_path_if_exists(audio_mix_stage_files.get("pl_5_1"), workspace),
                "audio_pl_7_1": rel_path_if_exists(audio_mix_stage_files.get("pl_7_1"), workspace),
                "output_video": rel_path(output_video_path, workspace),
            },
        },
    )
    if diagnostics["save_run_reports"]:
        progress("Podsumowanie przebiegu: zapisano")
    _emit_file_progress(progress, "done", 1.0, "Aktualny plik gotowy")

    if output_video_path is not None:
        _cleanup_successful_video_run(
            diagnostics=diagnostics,
            lektor_dir=lektor_dir,
            segments_dir=segments_dir,
            subtitle_path=subtitle_path,
            input_subtitle_path=input_subtitle_path,
            manifest_path=manifest_path,
            skipped_manifest_path=skipped_manifest_path,
            audio_qc_path=lektor_dir / "audio_qc.csv",
            analysis_path=analysis_path,
            summary_path=summary_path,
            lektor_before_normalization_path=lektor_before_normalization_path,
            lektor_after_normalization_path=lektor_wav_path,
            lektor_m4a_path=lektor_m4a_path,
            audio_mix_stage_files=audio_mix_stage_files,
        )
        progress("Pliki robocze: uporzadkowane wedlug opcji diagnostycznych")
        subtitle_path = _existing_file_or_none(subtitle_path) or subtitle_path
        if diagnostics["save_run_reports"]:
            manifest_path = _existing_file_or_none(manifest_path)
            summary_path = _existing_file_or_none(summary_path)
        else:
            manifest_path = None
            summary_path = None
        lektor_before_normalization_path = _existing_file_or_none(lektor_before_normalization_path)
        lektor_wav_path = _existing_file_or_none(lektor_wav_path)
        lektor_m4a_path = _existing_file_or_none(lektor_m4a_path)

    return TTSJobResult(
        engine_id=engine_id,
        output_stem=output_stem,
        workspace=workspace,
        subtitle_path=subtitle_path,
        lektor_dir=lektor_dir,
        segment_count=len(processed),
        generation_seconds=generation_seconds,
        qc_warning_count=qc_warning_count,
        manifest_path=manifest_path,
        summary_path=summary_path,
        lektor_wav_path=lektor_wav_path,
        lektor_m4a_path=lektor_m4a_path,
        output_video_path=output_video_path,
    )


def _cleanup_lektor_debug_files(
    config: dict,
    lektor_dir: Path,
    segments_dir: Path,
    lektor_before_normalization_path: Path | None,
    lektor_after_normalization_path: Path | None,
) -> None:
    if not _bool_config(config.get("save_lektor_segments"), True):
        shutil.rmtree(segments_dir, ignore_errors=True)
    if not _bool_config(config.get("save_lektor_track_before_normalization"), False):
        _unlink_if_file(lektor_before_normalization_path)
    if not _bool_config(config.get("save_lektor_track_after_normalization"), True):
        _unlink_if_file(lektor_after_normalization_path)
    _cleanup_empty_dir(segments_dir)
    _cleanup_empty_dir(lektor_dir)


def _run_report_path(lektor_dir: Path, output_stem: str, report_type: str) -> Path:
    return lektor_dir / f"{output_stem}_{safe_report_suffix(report_type)}.json"


def safe_report_suffix(report_type: str) -> str:
    normalized = str(report_type or "").strip().lower()
    if normalized in {"analysis", "summary", "error"}:
        return normalized
    return re.sub(r"[^a-z0-9_-]+", "_", normalized).strip("_") or "report"


def _diagnostic_keep_flags(config: dict, keep_all_legacy: bool = False) -> dict[str, bool]:
    keys = (
        "save_processed_subtitles",
        "save_quality_report",
        "save_run_reports",
        "save_lektor_segments",
        "save_lektor_track_before_normalization",
        "save_lektor_track_after_normalization",
        "save_audio_mix_steps",
    )
    if keep_all_legacy:
        return {key: True for key in keys}
    return {key: _bool_config(config.get(key), False) for key in keys}


def _cleanup_successful_video_run(
    diagnostics: dict[str, bool],
    lektor_dir: Path,
    segments_dir: Path,
    subtitle_path: Path,
    input_subtitle_path: Path,
    manifest_path: Path | None,
    skipped_manifest_path: Path | None,
    audio_qc_path: Path,
    analysis_path: Path | None,
    summary_path: Path | None,
    lektor_before_normalization_path: Path | None,
    lektor_after_normalization_path: Path | None,
    lektor_m4a_path: Path | None,
    audio_mix_stage_files: dict[str, Path],
) -> None:
    if not diagnostics.get("save_processed_subtitles", False):
        _unlink_if_file(subtitle_path)
    if not diagnostics.get("save_quality_report", False):
        _unlink_if_file(analysis_path)
    if not diagnostics.get("save_run_reports", False):
        for path in (manifest_path, skipped_manifest_path, audio_qc_path, summary_path):
            _unlink_if_file(path)
        if _is_relative_to(input_subtitle_path, lektor_dir):
            _unlink_if_file(input_subtitle_path)
    if not diagnostics.get("save_lektor_segments", False):
        shutil.rmtree(segments_dir, ignore_errors=True)
    if not diagnostics.get("save_lektor_track_before_normalization", False):
        _unlink_if_file(lektor_before_normalization_path)
    if not diagnostics.get("save_lektor_track_after_normalization", False):
        _unlink_if_file(lektor_after_normalization_path)
    if not diagnostics.get("save_audio_mix_steps", False):
        _unlink_if_file(lektor_m4a_path)
        for path in audio_mix_stage_files.values():
            _unlink_if_file(path)
    _cleanup_empty_dir(segments_dir)
    _cleanup_empty_dirs(lektor_dir)


def _cleanup_empty_dirs(*paths: Path) -> None:
    for path in paths:
        _cleanup_empty_dir(path)


def _build_run_analysis(
    source_path: Path,
    input_subtitle_path: Path,
    engine_id: str,
    output_stem: str,
    run_started_at: datetime,
    segments: list[SubtitleSegment],
    segment_analysis: list[dict[str, Any]],
    config: dict,
    dictionary: dict,
    quality_controls: dict,
    generation_seconds: float,
    pipeline_seconds: float,
    aac_bitrate: str,
    lektor_lufs: int,
    lektor_weight: float,
    background_lufs: int,
    background_weight: float,
    lektor_delay_ms: int,
    create_stereo_for_surround: bool,
    diagnostics: dict[str, bool],
    source_duration: float | None,
    source_audio_streams: list[dict],
    lektor_before_diagnostics: dict | None,
    lektor_after_diagnostics: dict | None,
    lektor_encoded_duration: float | None,
    lektor_encoded_audio_streams: list[dict],
) -> dict[str, Any]:
    records = _merge_segment_analysis(segments, segment_analysis)
    return {
        "schema": "lektorai.run_analysis.v1",
        "report_type": "analysis",
        "run_id": output_stem,
        "run_timestamp": run_started_at.isoformat(timespec="seconds"),
        "source_filename": source_path.name,
        "source_stem": source_path.stem,
        "tts_engine_short": engine_short_code(engine_id),
        "llm_analysis_hint": "Porownaj raporty analysis z tych samych source_filename. Oceniaj ustawienia TTS po aggregates, worst_segments_by_score, liczbie retry, final_score, whisper_similarity i ostrzezeniach QC. Nizszy score jest lepszy.",
        "score_meaning": "0 = najlepszy wynik QC; im wyzszy score, tym gorsza zgodnosc/jakosc kandydata",
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "source": {
            "path": str(source_path),
            "name": source_path.name,
            "duration_s": round(float(source_duration or 0.0), 3),
            "primary_audio": audio_stream_summary(selected_background_audio_stream(source_audio_streams)),
        },
        "subtitles": {
            "input_path": str(input_subtitle_path),
            "processed_segment_count": len(segments),
            "dictionary_entry_count": int(len(dictionary or {})),
        },
        "engine": {
            "id": str(engine_id),
            "settings": _sanitize_settings_snapshot(config),
        },
        "quality_controls": quality_controls,
        "audio_output": {
            "codec": "AAC",
            "bitrate": str(aac_bitrate),
            "lektor_lufs": int(lektor_lufs),
            "lektor_weight": float(lektor_weight),
            "background_lufs": int(background_lufs),
            "background_weight": float(background_weight),
            "lektor_delay_ms": int(lektor_delay_ms),
            "create_stereo_for_surround": bool(create_stereo_for_surround),
        },
        "timings": {
            "generation_seconds": round(float(generation_seconds), 3),
            "pipeline_seconds": round(float(pipeline_seconds), 3),
        },
        "diagnostic_keep": diagnostics,
        "audio_diagnostics": {
            "lektor_before_normalization": lektor_before_diagnostics or {},
            "lektor_after_normalization": lektor_after_diagnostics or {},
            "lektor_encoded_duration_s": round(float(lektor_encoded_duration or 0.0), 3),
            "lektor_encoded_audio": audio_stream_summary(lektor_encoded_audio_streams[0] if lektor_encoded_audio_streams else None),
        },
        "output_stem": output_stem,
        "aggregates": _aggregate_segment_analysis(records),
        "segments": records,
    }


def _sanitize_settings_snapshot(settings: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted((settings or {}).keys(), key=str):
        value = settings.get(key)
        lowered = str(key).lower()
        if any(secret in lowered for secret in ("api_key", "token", "secret", "password")):
            result[str(key)] = "***" if str(value or "").strip() else ""
            continue
        result[str(key)] = _json_safe_value(value)
    return result


def _json_safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _merge_segment_analysis(segments: list[SubtitleSegment], segment_analysis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {int(item.get("srt_index", item.get("segment_id", -1))): item for item in segment_analysis if isinstance(item, dict)}
    records: list[dict[str, Any]] = []
    for ordinal, segment in enumerate(segments, 1):
        source = dict(by_id.get(int(segment.index), {}))
        attempts = _bounded_int(source.get("attempts", 1), 1, 999)
        selected_attempt = _bounded_int(source.get("selected_attempt", 1), 1, max(1, attempts))
        retries = max(0, _bounded_int(source.get("retries", max(0, attempts - 1)), 0, 999))
        record = {
            "ordinal": int(ordinal),
            "srt_index": int(segment.index),
            "start_ms": int(segment.start_ms),
            "end_ms": int(segment.end_ms),
            "subtitle_duration_ms": max(0, int(segment.end_ms) - int(segment.start_ms)),
            "text": str(segment.text),
            "audio_file": str(source.get("audio_file", "")),
            "attempts": attempts,
            "retries": retries,
            "selected_attempt": selected_attempt,
            "final_score": _optional_number(source.get("final_score", source.get("qc_score"))),
            "whisper_similarity": _optional_number(source.get("whisper_similarity")),
            "whisper_text": str(source.get("whisper_text", "") or ""),
            "qc_warnings": [str(item) for item in source.get("qc_warnings", ()) or ()],
            "attempt_details": _normalized_attempt_details(source.get("attempt_details", ())),
        }
        records.append(record)
    return records


def _normalized_attempt_details(value) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result.append({str(key): _json_safe_value(val) for key, val in item.items()})
    return result


def _aggregate_segment_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["final_score"]) for record in records if record.get("final_score") is not None]
    similarities = [float(record["whisper_similarity"]) for record in records if record.get("whisper_similarity") is not None]
    retry_records = [record for record in records if int(record.get("retries", 0) or 0) > 0]
    selected_counter: Counter[str] = Counter(str(int(record.get("selected_attempt", 0) or 0)) for record in records)
    return {
        "segment_count": len(records),
        "segments_with_retry": len(retry_records),
        "extra_attempts": int(sum(int(record.get("retries", 0) or 0) for record in records)),
        "max_attempts": int(max((int(record.get("attempts", 0) or 0) for record in records), default=0)),
        "selected_attempt_distribution": dict(sorted(selected_counter.items(), key=lambda item: int(item[0]))),
        "score": _numeric_summary(scores, lower_is_better=True),
        "whisper_similarity": _numeric_summary(similarities, lower_is_better=False),
        "worst_segments_by_score": _worst_segments_by_score(records, limit=12),
        "retry_segments": [
            {
                "ordinal": int(record.get("ordinal", 0) or 0),
                "srt_index": int(record.get("srt_index", 0) or 0),
                "attempts": int(record.get("attempts", 0) or 0),
                "selected_attempt": int(record.get("selected_attempt", 0) or 0),
                "final_score": record.get("final_score"),
            }
            for record in retry_records
        ],
    }


def _numeric_summary(values: list[float], lower_is_better: bool) -> dict[str, Any]:
    if not values:
        return {"count": 0, "lower_is_better": bool(lower_is_better)}
    return {
        "count": len(values),
        "lower_is_better": bool(lower_is_better),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "average": round(sum(values) / len(values), 4),
        "zero_count": sum(1 for value in values if value == 0),
        "positive_count": sum(1 for value in values if value > 0),
    }


def _worst_segments_by_score(records: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    scored = [record for record in records if record.get("final_score") is not None]
    scored.sort(key=lambda item: float(item.get("final_score") or 0), reverse=True)
    result: list[dict[str, Any]] = []
    for record in scored[: max(1, int(limit))]:
        result.append(
            {
                "ordinal": int(record.get("ordinal", 0) or 0),
                "srt_index": int(record.get("srt_index", 0) or 0),
                "final_score": record.get("final_score"),
                "attempts": int(record.get("attempts", 0) or 0),
                "selected_attempt": int(record.get("selected_attempt", 0) or 0),
                "text": str(record.get("text", "")),
            }
        )
    return result


def _optional_number(value):
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _expected_audio_mix_stage_files(
    output_video_path: Path,
    lektor_dir: Path,
    surround_label: str,
    create_stereo_track: bool,
) -> dict[str, Path]:
    prefix = output_video_path.stem
    result: dict[str, Path] = {
        "source_audio": lektor_dir / f"{prefix}_tlo_zrodlowe.wav",
    }
    if create_stereo_track:
        result["pl_2_0"] = lektor_dir / f"{prefix}_pl_2_0.m4a"
    if surround_label:
        result[f"pl_{surround_label.replace('.', '_')}"] = lektor_dir / f"{prefix}_pl_{surround_label.replace('.', '_')}.m4a"
    return result


def _unlink_if_file(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _existing_file_or_none(path: Path | None) -> Path | None:
    return path if path is not None and path.is_file() else None


def rel_path_if_exists(path: Path | None, base: Path) -> str:
    existing = _existing_file_or_none(path)
    return rel_path(existing, base) if existing is not None else ""


def _cleanup_empty_dir(path: Path) -> None:
    try:
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def apply_lektor_delay_to_segments(segment_paths: list[tuple[int, Path]], delay_ms: int) -> list[tuple[int, Path]]:
    delay_ms = sanitize_lektor_delay_ms(delay_ms)
    if not delay_ms:
        return list(segment_paths)
    return [(max(0, int(start_ms) + delay_ms), path) for start_ms, path in segment_paths]


def _emit_file_progress(progress, stage: str, ratio: float | None = None, label: str = "") -> None:
    try:
        progress(encode_progress_marker(stage, ratio, label))
    except Exception:
        pass


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Przerwano przez uzytkownika")


def _generate_edge(
    segments: list[SubtitleSegment],
    segments_dir: Path,
    config: dict,
    paths: AppPaths,
    lektor_dir: Path,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> GenerationOutput:
    generated: list[tuple[int, Path]] = []
    segment_analysis: list[dict[str, Any]] = []
    audio_attempts = _bounded_int(config.get("audio_qc_retry_attempts", 2), 1, 5)
    speech_attempts = _bounded_int(config.get("whisper_qc_retry_attempts", 1), 1, 5)
    _ensure_whisper_qc_runtime_if_enabled(config, paths, progress, cancel_requested)
    ffmpeg = find_ffmpeg(paths)
    qc_temp_dir = lektor_dir / "temp_edge_retry_qc"
    if qc_temp_dir.exists():
        shutil.rmtree(qc_temp_dir)
    qc_temp_dir.mkdir(parents=True, exist_ok=True)
    retry_summary = QCRetrySummary()
    started = perf_counter()
    try:
        for ordinal, segment in enumerate(segments, 1):
            _raise_if_cancelled(cancel_requested)
            edge_tuning_enabled = _edge_tuning_enabled(config, ffmpeg)
            output_suffix = ".wav" if edge_tuning_enabled else ".mp3"
            output_path = segments_dir / f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}{output_suffix}"
            progress(f"Segment {ordinal}/{len(segments)}")
            selected_path, analysis = _generate_edge_with_retry(
                segment,
                ordinal,
                len(segments),
                output_path,
                segments_dir,
                config,
                ffmpeg,
                paths.whisper_cache_dir,
                qc_temp_dir,
                audio_attempts,
                speech_attempts,
                retry_summary,
                progress,
                cancel_requested,
            )
            if selected_path != output_path:
                if output_path.exists():
                    output_path.unlink()
                shutil.move(str(selected_path), str(output_path))
            _cleanup_edge_candidates(segments_dir, segment, output_path)
            generated.append((segment.start_ms, output_path))
            analysis["audio_file"] = output_path.name
            segment_analysis.append(analysis)
    finally:
        shutil.rmtree(qc_temp_dir, ignore_errors=True)
    seconds = perf_counter() - started
    _emit_builtin_qc_retry_summary("Edge", config, ffmpeg, retry_summary, progress)
    progress(f"Generowanie segmentow {_format_duration_seconds(seconds)}")
    return GenerationOutput(generated, seconds, segment_analysis)


def _generate_edge_with_retry(
    segment: SubtitleSegment,
    ordinal: int,
    total_segments: int,
    output_path: Path,
    segments_dir: Path,
    config: dict,
    ffmpeg: Path | None,
    whisper_cache_dir: Path,
    qc_temp_dir: Path,
    audio_attempts: int,
    speech_attempts: int,
    retry_summary: QCRetrySummary,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[int, Path]] = []
    attempt_details: list[dict[str, Any]] = []
    best_score = 10**9
    best_warnings: tuple[str, ...] = ()
    best_path = output_path
    best_attempt = 1
    best_whisper_similarity = None
    best_whisper_text = ""
    audio_enabled = _bool_config(config.get("audio_qc_enabled"), False) and ffmpeg is not None
    speech_enabled = _bool_config(config.get("whisper_qc_enabled"), False)
    audio_limit = audio_attempts if audio_enabled else 1
    speech_limit = speech_attempts if speech_enabled else 1
    retry_texts = _edge_retry_text_variants(segment.text, max(1, audio_limit * speech_limit))
    candidate_no = 0
    for speech_attempt in range(1, speech_limit + 1):
        if candidate_no >= len(retry_texts):
            break
        _raise_if_cancelled(cancel_requested)
        audio_best_score = 10**9
        audio_best_warnings: tuple[str, ...] = ()
        audio_best_path = output_path
        audio_best_attempt = 1
        for audio_attempt in range(1, audio_limit + 1):
            if candidate_no >= len(retry_texts):
                break
            _raise_if_cancelled(cancel_requested)
            candidate_no += 1
            edge_tuning_enabled = _edge_tuning_enabled(config, ffmpeg)
            candidate_path = output_path if candidate_no == 1 else _edge_candidate_path(segments_dir, segment, candidate_no, edge_tuning_enabled)
            raw_candidate_path = _edge_raw_candidate_path(segments_dir, segment, candidate_no)
            synthesize_edge_mp3_sync(
                text=retry_texts[candidate_no - 1],
                output_path=raw_candidate_path,
                voice=str(config.get("voice") or "pl-PL-MarekNeural"),
                rate=str(config.get("rate") or "+0%"),
                pitch=str(config.get("pitch") or "+0Hz"),
            )
            if edge_tuning_enabled and ffmpeg is not None:
                _prepare_edge_candidate_edges(ffmpeg, raw_candidate_path, candidate_path, qc_temp_dir, segment, candidate_no, config, cancel_requested)
                raw_candidate_path.unlink(missing_ok=True)
            else:
                if candidate_path.exists():
                    candidate_path.unlink()
                shutil.move(str(raw_candidate_path), str(candidate_path))
            candidates.append((candidate_no, candidate_path))
            score, warnings = _score_builtin_audio_candidate(
                candidate_path,
                segment,
                ffmpeg,
                qc_temp_dir / f"edge_{segment.index:05d}_{candidate_no}.wav",
                audio_enabled,
                cancel_requested,
            )
            detail = _builtin_attempt_detail(
                attempt=candidate_no,
                audio_attempt=audio_attempt,
                speech_attempt=speech_attempt,
                text_variant=_edge_retry_variant_label(retry_texts[candidate_no - 1]),
                audio_file=candidate_path.name,
                audio_qc_score=score,
                audio_qc_warnings=warnings,
            )
            attempt_details.append(detail)
            if score < audio_best_score:
                audio_best_score = score
                audio_best_warnings = warnings
                audio_best_path = candidate_path
                audio_best_attempt = candidate_no
            if score == 0:
                break
            if audio_attempt < audio_limit:
                retry_summary.record_audio_retry(ordinal)
                progress(_format_short_qc_retry_message("Edge", "kontrola audio", ordinal, total_segments, audio_attempt + 1, audio_limit))
        score = audio_best_score
        warnings = audio_best_warnings
        whisper_similarity = None
        whisper_text = ""
        if speech_enabled:
            _raise_if_cancelled(cancel_requested)
            speech_score, speech_warnings, whisper_text, whisper_similarity = _score_builtin_speech_candidate(audio_best_path, segment, config, whisper_cache_dir)
            score += speech_score
            warnings = tuple(list(warnings) + list(speech_warnings))
            _update_builtin_attempt_detail(
                attempt_details,
                audio_best_attempt,
                speech_score=speech_score,
                speech_warnings=speech_warnings,
                whisper_text=whisper_text,
                whisper_similarity=whisper_similarity,
                final_score=score,
                final_warnings=warnings,
            )
        if score < best_score:
            best_score = score
            best_warnings = warnings
            best_path = audio_best_path
            best_attempt = audio_best_attempt
            best_whisper_similarity = whisper_similarity
            best_whisper_text = whisper_text
        if score == 0:
            break
        if speech_attempt < speech_limit and candidate_no < len(retry_texts):
            retry_summary.record_speech_retry(ordinal)
            progress(_format_short_qc_retry_message("Edge", "kontrola mowy", ordinal, total_segments, speech_attempt + 1, speech_limit))
    _mark_selected_attempt(attempt_details, best_attempt)
    return best_path, _builtin_segment_analysis(segment, best_path, attempt_details, best_attempt, best_score, best_warnings, best_whisper_text, best_whisper_similarity)


def _edge_candidate_path(segments_dir: Path, segment: SubtitleSegment, attempt: int, edge_tuning_enabled: bool = False) -> Path:
    suffix = ".wav" if edge_tuning_enabled else ".mp3"
    return segments_dir / f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}_try{attempt}{suffix}"


def _edge_raw_candidate_path(segments_dir: Path, segment: SubtitleSegment, attempt: int) -> Path:
    return segments_dir / f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}_try{attempt}_raw.mp3"


def _cleanup_edge_candidates(segments_dir: Path, segment: SubtitleSegment, keep_path: Path) -> None:
    pattern = f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}_try*.*"
    for candidate in segments_dir.glob(pattern):
        if candidate != keep_path:
            candidate.unlink(missing_ok=True)


def _edge_tuning_enabled(config: dict, ffmpeg: Path | None) -> bool:
    return ffmpeg is not None and bool(config.get("edge_apply_segment_fade", True))


def _prepare_edge_candidate_edges(
    ffmpeg: Path,
    raw_mp3_path: Path,
    output_wav_path: Path,
    temp_dir: Path,
    segment: SubtitleSegment,
    attempt: int,
    config: dict,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    temp_wav = temp_dir / f"edge_{segment.index:05d}_{attempt}_raw.wav"
    convert_audio_to_wav(ffmpeg, raw_mp3_path, temp_wav, cancel_requested=cancel_requested)
    trim_fixed_and_fade_wav_edges(
        temp_wav,
        output_wav_path,
        trim_start_ms=_bounded_int(config.get("edge_trim_start_ms", 200), 0, 1000),
        trim_end_ms=_bounded_int(config.get("edge_trim_end_ms", 900), 0, 2000),
        fade_ms=EDGE_TRIM_FADE_MS,
    )


def _edge_retry_text(text: str, attempt: int) -> str:
    variants = _edge_retry_text_variants(text, max(1, int(attempt)))
    index = min(max(1, int(attempt)) - 1, len(variants) - 1)
    return variants[index]


_RETRY_TERMINAL_PUNCTUATION = ".,!?:;"


def _edge_retry_text_variants(text: str, limit: int = 5) -> list[str]:
    limit = max(1, int(limit))
    stripped = str(text or "").strip()
    if not stripped:
        return [str(text or "")]
    base = _strip_retry_terminal_punctuation(stripped)
    if not base:
        return [stripped]
    return [stripped, base + ".", base + ",", base + "!", base + "?"][:limit]


def _strip_retry_terminal_punctuation(text: str) -> str:
    base = str(text or "").strip()
    while base and base[-1] in _RETRY_TERMINAL_PUNCTUATION:
        base = base[:-1].rstrip()
    return base


def _generate_openai(
    segments: list[SubtitleSegment],
    segments_dir: Path,
    config: dict,
    paths: AppPaths,
    lektor_dir: Path,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> GenerationOutput:
    generated: list[tuple[int, Path]] = []
    segment_analysis: list[dict[str, Any]] = []
    audio_attempts = _bounded_int(config.get("audio_qc_retry_attempts", 2), 1, 5)
    speech_attempts = _bounded_int(config.get("whisper_qc_retry_attempts", 1), 1, 5)
    _ensure_whisper_qc_runtime_if_enabled(config, paths, progress, cancel_requested)
    ffmpeg = find_ffmpeg(paths)
    qc_temp_dir = lektor_dir / "temp_openai_retry_qc"
    if qc_temp_dir.exists():
        shutil.rmtree(qc_temp_dir)
    qc_temp_dir.mkdir(parents=True, exist_ok=True)
    retry_summary = QCRetrySummary()
    started = perf_counter()
    try:
        for ordinal, segment in enumerate(segments, 1):
            _raise_if_cancelled(cancel_requested)
            output_path = segments_dir / f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}.wav"
            progress(f"Segment {ordinal}/{len(segments)}")
            selected_path, analysis = _generate_openai_with_retry(
                segment,
                ordinal,
                len(segments),
                output_path,
                segments_dir,
                config,
                ffmpeg,
                paths.whisper_cache_dir,
                qc_temp_dir,
                audio_attempts,
                speech_attempts,
                retry_summary,
                progress,
                cancel_requested,
            )
            if selected_path != output_path:
                if output_path.exists():
                    output_path.unlink()
                shutil.move(str(selected_path), str(output_path))
            if ffmpeg is not None:
                _apply_openai_fade(ffmpeg, output_path, lektor_dir, cancel_requested=cancel_requested)
            _cleanup_openai_candidates(segments_dir, segment, output_path)
            generated.append((segment.start_ms, output_path))
            analysis["audio_file"] = output_path.name
            segment_analysis.append(analysis)
    finally:
        shutil.rmtree(qc_temp_dir, ignore_errors=True)
        shutil.rmtree(lektor_dir / "temp_openai_fade", ignore_errors=True)
    seconds = perf_counter() - started
    _emit_builtin_qc_retry_summary("OpenAI", config, ffmpeg, retry_summary, progress)
    progress(f"Generowanie segmentow {_format_duration_seconds(seconds)}")
    return GenerationOutput(generated, seconds, segment_analysis)


def _generate_openai_with_retry(
    segment: SubtitleSegment,
    ordinal: int,
    total_segments: int,
    output_path: Path,
    segments_dir: Path,
    config: dict,
    ffmpeg: Path | None,
    whisper_cache_dir: Path,
    qc_temp_dir: Path,
    audio_attempts: int,
    speech_attempts: int,
    retry_summary: QCRetrySummary,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[int, Path]] = []
    attempt_details: list[dict[str, Any]] = []
    best_score = 10**9
    best_warnings: tuple[str, ...] = ()
    best_path = output_path
    best_attempt = 1
    best_whisper_similarity = None
    best_whisper_text = ""
    audio_enabled = _bool_config(config.get("audio_qc_enabled"), False) and ffmpeg is not None
    speech_enabled = _bool_config(config.get("whisper_qc_enabled"), False)
    audio_limit = audio_attempts if audio_enabled else 1
    speech_limit = speech_attempts if speech_enabled else 1
    candidate_no = 0
    for speech_attempt in range(1, speech_limit + 1):
        _raise_if_cancelled(cancel_requested)
        audio_best_score = 10**9
        audio_best_warnings: tuple[str, ...] = ()
        audio_best_path = output_path
        audio_best_attempt = max(1, candidate_no + 1)
        for audio_attempt in range(1, audio_limit + 1):
            _raise_if_cancelled(cancel_requested)
            candidate_no += 1
            candidate_path = output_path if candidate_no == 1 else _openai_candidate_path(segments_dir, segment, candidate_no)
            api_key = str(config.get("api_key") or os.environ.get("OPENAI_API_KEY") or "")
            if not api_key.strip():
                raise RuntimeError("OpenAI TTS: brak api_key albo zmiennej OPENAI_API_KEY.")
            synthesize_openai_wav_sync(
                text=_edge_retry_text(segment.text, candidate_no),
                output_path=candidate_path,
                model=str(config.get("model") or "gpt-4o-mini-tts"),
                voice=str(config.get("voice") or "marin"),
                api_key=api_key,
                instructions=str(config.get("instructions") or ""),
            )
            candidates.append((candidate_no, candidate_path))
            score, warnings = _score_builtin_audio_candidate(
                candidate_path,
                segment,
                ffmpeg,
                qc_temp_dir / f"openai_{segment.index:05d}_{candidate_no}.wav",
                audio_enabled,
                cancel_requested,
            )
            attempt_details.append(
                _builtin_attempt_detail(
                    attempt=candidate_no,
                    audio_attempt=audio_attempt,
                    speech_attempt=speech_attempt,
                    text_variant=_edge_retry_variant_label(_edge_retry_text(segment.text, candidate_no)),
                    audio_file=candidate_path.name,
                    audio_qc_score=score,
                    audio_qc_warnings=warnings,
                )
            )
            if score < audio_best_score:
                audio_best_score = score
                audio_best_warnings = warnings
                audio_best_path = candidate_path
                audio_best_attempt = candidate_no
            if score == 0:
                break
            if audio_attempt < audio_limit:
                retry_summary.record_audio_retry(ordinal)
                progress(_format_short_qc_retry_message("OpenAI", "kontrola audio", ordinal, total_segments, audio_attempt + 1, audio_limit))
        score = audio_best_score
        warnings = audio_best_warnings
        whisper_similarity = None
        whisper_text = ""
        if speech_enabled:
            _raise_if_cancelled(cancel_requested)
            speech_score, speech_warnings, whisper_text, whisper_similarity = _score_builtin_speech_candidate(audio_best_path, segment, config, whisper_cache_dir)
            score += speech_score
            warnings = tuple(list(warnings) + list(speech_warnings))
            _update_builtin_attempt_detail(
                attempt_details,
                audio_best_attempt,
                speech_score=speech_score,
                speech_warnings=speech_warnings,
                whisper_text=whisper_text,
                whisper_similarity=whisper_similarity,
                final_score=score,
                final_warnings=warnings,
            )
        if score < best_score:
            best_score = score
            best_warnings = warnings
            best_path = audio_best_path
            best_attempt = audio_best_attempt
            best_whisper_similarity = whisper_similarity
            best_whisper_text = whisper_text
        if score == 0:
            break
        if speech_attempt < speech_limit:
            retry_summary.record_speech_retry(ordinal)
            progress(_format_short_qc_retry_message("OpenAI", "kontrola mowy", ordinal, total_segments, speech_attempt + 1, speech_limit))
    _mark_selected_attempt(attempt_details, best_attempt)
    return best_path, _builtin_segment_analysis(segment, best_path, attempt_details, best_attempt, best_score, best_warnings, best_whisper_text, best_whisper_similarity)


def _openai_candidate_path(segments_dir: Path, segment: SubtitleSegment, attempt: int) -> Path:
    return segments_dir / f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}_try{attempt}.wav"


def _cleanup_openai_candidates(segments_dir: Path, segment: SubtitleSegment, keep_path: Path) -> None:
    pattern = f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}_try*.wav"
    for candidate in segments_dir.glob(pattern):
        if candidate != keep_path:
            candidate.unlink(missing_ok=True)


def _apply_openai_fade(
    ffmpeg: Path,
    output_path: Path,
    lektor_dir: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    _raise_if_cancelled(cancel_requested)
    temp_dir = lektor_dir / "temp_openai_fade"
    temp_dir.mkdir(parents=True, exist_ok=True)
    faded_path = temp_dir / output_path.name
    apply_short_audio_fade(ffmpeg, output_path, faded_path, fade_seconds=0.012, cancel_requested=cancel_requested)
    shutil.move(str(faded_path), str(output_path))


def _score_builtin_audio_candidate(
    candidate_path: Path,
    segment: SubtitleSegment,
    ffmpeg: Path | None,
    qc_wav_path: Path,
    enabled: bool,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[int, tuple[str, ...]]:
    if not enabled or ffmpeg is None:
        return 0, ()
    qc = analyze_audio_candidate(ffmpeg, candidate_path, segment, qc_wav_path, cancel_requested=cancel_requested)
    return int(score_audio_qc(qc)), tuple(str(warning) for warning in qc.warnings)


def _score_builtin_speech_candidate(
    candidate_path: Path,
    segment: SubtitleSegment,
    config: dict,
    whisper_cache_dir: Path,
) -> tuple[int, tuple[str, ...], str, float]:
    transcript = transcribe_audio_with_faster_whisper(candidate_path, config, whisper_cache_dir)
    threshold = _bounded_float(config.get("whisper_qc_min_similarity", 0.62), 0.0, 1.0)
    whisper = score_whisper_transcript(segment.text, transcript, threshold)
    warnings = [str(warning) for warning in whisper.warnings]
    return int(whisper.score), tuple(warning for warning in warnings if warning), str(whisper.text), float(whisper.similarity)


def _builtin_attempt_detail(
    attempt: int,
    audio_attempt: int,
    speech_attempt: int,
    text_variant: str,
    audio_file: str,
    audio_qc_score: int,
    audio_qc_warnings: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "attempt": int(attempt),
        "audio_attempt": int(audio_attempt),
        "speech_attempt": int(speech_attempt),
        "text_variant": str(text_variant),
        "audio_file": str(audio_file),
        "audio_qc_score": int(audio_qc_score),
        "audio_qc_warnings": [str(warning) for warning in audio_qc_warnings],
        "whisper_checked": False,
        "whisper_score": None,
        "whisper_similarity": None,
        "whisper_text": "",
        "final_score": int(audio_qc_score),
        "final_warnings": [str(warning) for warning in audio_qc_warnings],
        "selected": False,
    }


def _update_builtin_attempt_detail(
    attempt_details: list[dict[str, Any]],
    attempt: int,
    speech_score: int,
    speech_warnings: tuple[str, ...],
    whisper_text: str,
    whisper_similarity: float | None,
    final_score: int,
    final_warnings: tuple[str, ...],
) -> None:
    for detail in attempt_details:
        if int(detail.get("attempt", 0) or 0) == int(attempt):
            detail["whisper_checked"] = True
            detail["whisper_score"] = int(speech_score)
            detail["whisper_similarity"] = _optional_number(whisper_similarity)
            detail["whisper_text"] = str(whisper_text or "")
            detail["final_score"] = int(final_score)
            detail["final_warnings"] = [str(warning) for warning in final_warnings]
            return


def _mark_selected_attempt(attempt_details: list[dict[str, Any]], selected_attempt: int) -> None:
    for detail in attempt_details:
        detail["selected"] = int(detail.get("attempt", 0) or 0) == int(selected_attempt)


def _edge_retry_variant_label(candidate_text: str) -> str:
    stripped = str(candidate_text or "").rstrip()
    if not stripped:
        return "oryginal"
    if stripped[-1] in _RETRY_TERMINAL_PUNCTUATION:
        return f"koncowka_{stripped[-1]}"
    return "oryginal"


def _builtin_segment_analysis(
    segment: SubtitleSegment,
    audio_path: Path,
    attempt_details: list[dict[str, Any]],
    selected_attempt: int,
    final_score: int,
    qc_warnings: tuple[str, ...],
    whisper_text: str,
    whisper_similarity: float | None,
) -> dict[str, Any]:
    return {
        "srt_index": int(segment.index),
        "audio_file": audio_path.name,
        "attempts": len(attempt_details) or 1,
        "selected_attempt": int(selected_attempt),
        "retries": max(0, (len(attempt_details) or 1) - 1),
        "final_score": int(final_score),
        "qc_warnings": [str(warning) for warning in qc_warnings],
        "whisper_text": str(whisper_text or ""),
        "whisper_similarity": _optional_number(whisper_similarity),
        "attempt_details": attempt_details,
    }


def _local_worker_segment_analysis(
    segments: list[SubtitleSegment],
    generated: list[tuple[int, Path]],
    result: EngineResult,
) -> list[dict[str, Any]]:
    by_id = {int(segment.segment_id): segment for segment in result.segments}
    audio_by_index = {int(segment.index): generated[index][1] for index, segment in enumerate(segments) if index < len(generated)}
    analysis: list[dict[str, Any]] = []
    for segment in segments:
        segment_result = by_id.get(int(segment.index))
        audio_path = audio_by_index.get(int(segment.index))
        if segment_result is None:
            continue
        attempts = int(segment_result.attempts or 1)
        analysis.append(
            {
                "srt_index": int(segment.index),
                "audio_file": audio_path.name if audio_path is not None else Path(segment_result.output_path or "").name,
                "attempts": attempts,
                "selected_attempt": int(segment_result.selected_attempt or 1),
                "retries": int(segment_result.retries or max(0, attempts - 1)),
                "final_score": _optional_number(segment_result.qc_score),
                "qc_warnings": [str(warning) for warning in segment_result.qc_warnings],
                "whisper_text": str(segment_result.whisper_text or ""),
                "whisper_similarity": _optional_number(segment_result.whisper_similarity),
                "attempt_details": _normalized_attempt_details(segment_result.attempt_details),
            }
        )
    return analysis


def _score_builtin_candidate(
    engine_id: str,
    candidate_path: Path,
    segment: SubtitleSegment,
    config: dict,
    ffmpeg: Path | None,
    whisper_cache_dir: Path,
    qc_wav_path: Path,
) -> tuple[int, tuple[str, ...]]:
    score = 0
    warnings: list[str] = []
    audio_score, audio_warnings = _score_builtin_audio_candidate(
        candidate_path,
        segment,
        ffmpeg,
        qc_wav_path,
        _bool_config(config.get("audio_qc_enabled"), False),
    )
    score += audio_score
    warnings.extend(audio_warnings)
    if _bool_config(config.get("whisper_qc_enabled"), False):
        speech_score, speech_warnings, _whisper_text, _whisper_similarity = _score_builtin_speech_candidate(candidate_path, segment, config, whisper_cache_dir)
        score += speech_score
        warnings.extend(speech_warnings)
    return int(score), tuple(warning for warning in warnings if warning)


def _format_builtin_qc_retry_message(
    engine_name: str,
    ordinal: int,
    total_segments: int,
    next_attempt: int,
    retry_attempts: int,
    score: int,
    warnings: tuple[str, ...],
) -> str:
    details = ", ".join(str(warning) for warning in warnings[:3] if str(warning).strip())
    suffix = f", {details}" if details else ""
    failed_attempt = max(1, int(next_attempt) - 1)
    return f"{_qc_log_prefix(engine_name)}: segment {ordinal}/{total_segments}, odrzucono probe {failed_attempt}/{retry_attempts}, kara QC {score}{suffix}; ponawiam {next_attempt}/{retry_attempts}"


def _format_builtin_qc_selected_message(
    engine_name: str,
    ordinal: int,
    total_segments: int,
    selected_attempt: int,
    attempt_count: int,
    score: int,
    warnings: tuple[str, ...],
) -> str:
    details = ", ".join(str(warning) for warning in warnings[:3] if str(warning).strip())
    suffix = f", {details}" if details else ""
    return f"{_qc_log_prefix(engine_name)}: segment {ordinal}/{total_segments}, wybrano probe {selected_attempt}/{attempt_count}, kara QC {score}{suffix}"


def _qc_log_prefix(engine_name: str) -> str:
    name = str(engine_name or "").strip()
    if not name:
        return "QC"
    return name if name.endswith("QC") else f"{name} QC"


def _emit_builtin_qc_retry_summary(
    engine_id: str,
    config: dict,
    ffmpeg: Path | None,
    retry_summary: QCRetrySummary,
    progress,
) -> None:
    audio_enabled = ffmpeg is not None and _bool_config(config.get("audio_qc_enabled"), False)
    speech_enabled = _bool_config(config.get("whisper_qc_enabled"), False)
    if audio_enabled:
        _emit_qc_retry_summary_line(
            engine_id,
            "kontrola audio",
            len(retry_summary.audio_retry_segments),
            retry_summary.audio_extra_attempts,
            progress,
        )
    if speech_enabled:
        _emit_qc_retry_summary_line(
            engine_id,
            "kontrola mowy",
            len(retry_summary.speech_retry_segments),
            retry_summary.speech_extra_attempts,
            progress,
        )


def _emit_qc_retry_summary_line(engine_id: str, label: str, segment_count: int, extra_attempts: int, progress) -> None:
    if int(extra_attempts) > 0:
        progress(
            f"{label} ponawiala {_format_polish_count(segment_count, 'segment', 'segmenty', 'segmentow')}; "
            f"dodatkowe {_format_polish_count(extra_attempts, 'proba', 'proby', 'prob')}"
        )
    else:
        progress(f"{label}: sprawdzono, bez ponowien")


def _format_short_qc_retry_message(
    engine_id: str,
    label: str,
    ordinal: int,
    total_segments: int,
    next_attempt: int,
    retry_attempts: int,
) -> str:
    module_name = "Audio QC" if "audio" in str(label).lower() else "Whisper QC"
    return f"{module_name} - segment {ordinal}/{total_segments}, proba {next_attempt}/{retry_attempts}"


def _format_polish_count(count: int, singular: str, few: str, many: str) -> str:
    value = abs(int(count))
    if value == 1:
        form = singular
    elif value % 10 in (2, 3, 4) and value % 100 not in (12, 13, 14):
        form = few
    else:
        form = many
    return f"{int(count)} {form}"


def _quality_controls_summary(engine_id: str, config: dict, ffmpeg_present: bool) -> dict:
    audio_enabled = bool(ffmpeg_present) and _bool_config(config.get("audio_qc_enabled"), False)
    speech_enabled = _bool_config(config.get("whisper_qc_enabled"), False)
    edge_tuning_enabled = engine_id == "edge" and bool(ffmpeg_present) and _bool_config(config.get("edge_apply_segment_fade"), True)
    omnivoice_tuning_enabled = engine_id == "omnivoice" and _bool_config(config.get("omnivoice_trim_edges"), False)
    xtts_tuning_enabled = engine_id == "coqui_xtts" and _bool_config(config.get("xtts_trim_trailing_silence"), True)
    audio_attempts = _bounded_int(config.get("audio_qc_retry_attempts", 1), 1, 5)
    speech_attempts = _bounded_int(config.get("whisper_qc_retry_attempts", 1), 1, 5)
    if engine_id in DETERMINISTIC_RETRY_ENGINES:
        audio_attempts = 1
        if engine_id not in PUNCTUATION_RETRY_ENGINES:
            speech_attempts = 1
    chain = ["TTS"]
    if edge_tuning_enabled or omnivoice_tuning_enabled or xtts_tuning_enabled:
        if engine_id == "omnivoice":
            chain.append("Wycinanie ciszy na brzegach")
        elif engine_id == "coqui_xtts":
            chain.append("Wycinanie koncowej ciszy")
        else:
            chain.append("Przytnij i wygladz brzegi")
    if audio_enabled:
        chain.append(f"Audio QC x{audio_attempts}" if audio_attempts > 1 else "Audio QC")
    if speech_enabled:
        chain.append(f"Whisper QC x{speech_attempts}" if speech_attempts > 1 else "Whisper QC")
    chain.append("final")
    return {
        "engine_id": str(engine_id),
        "chain": chain,
        "chain_label": " -> ".join(chain),
        "audio_qc_enabled": audio_enabled,
        "audio_qc_retry_attempts": audio_attempts,
        "edge_tuning_enabled": edge_tuning_enabled,
        "omnivoice_tuning_enabled": omnivoice_tuning_enabled,
        "xtts_tuning_enabled": xtts_tuning_enabled,
        "whisper_qc_enabled": speech_enabled,
        "whisper_qc_retry_attempts": speech_attempts,
        "whisper_qc_model": str(config.get("whisper_qc_model", "small") or "small").strip() or "small",
        "whisper_qc_min_similarity": _bounded_float(config.get("whisper_qc_min_similarity", 0.62), 0.0, 1.0),
    }


def _quality_chain_label(summary: dict) -> str:
    label = str(summary.get("chain_label") or "").strip()
    if label:
        return label
    chain = summary.get("chain")
    if isinstance(chain, list) and chain:
        return " -> ".join(str(item) for item in chain)
    return "TTS -> final"


def _generate_local(
    segments: list[SubtitleSegment],
    source_path: Path,
    engine_id: str,
    segments_dir: Path,
    lektor_dir: Path,
    config: dict,
    dictionary: dict,
    paths: AppPaths,
    manager: EngineManager,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> GenerationOutput:
    engine_dir = paths.engine_dir(engine_id)
    _ensure_whisper_qc_runtime_if_enabled(config, paths, progress, cancel_requested)
    worker_script = engine_dir / "worker.py"
    if not worker_script.exists():
        raise RuntimeError(f"TTS {engine_id} nie ma worker.py. Zainstaluj silnik ponownie w menadzerze TTS.")
    manager.install_worker_script(engine_id)

    job_id = uuid4().hex
    runner = EngineWorkerRunner(paths, manager)
    run_paths = runner.build_run_paths(engine_id, source_path.name, job_id)
    request = _build_local_engine_request(
        engine_id=engine_id,
        source_name=source_path.name,
        job_id=job_id,
        segments=segments,
        segments_dir=segments_dir,
        config=_prepare_local_voice_sample(engine_id, config, paths, lektor_dir, progress, cancel_requested=cancel_requested),
        dictionary=dictionary,
    )

    write_request(run_paths.request_path, request)
    progress(f"{engine_id}: start workera")
    started = perf_counter()
    progress_counter = {"done": 0, "model_activity": "", "model_activity_seen": set()}
    result = runner.run_worker(
        engine_id,
        worker_script,
        run_paths,
        timeout_s=local_worker_timeout_seconds(),
        progress=lambda line: _local_worker_progress(engine_id, len(segments), progress_counter, progress, line),
        cancel_requested=cancel_requested,
    )
    seconds = perf_counter() - started
    if not result.ok:
        first_error = next((segment.error for segment in result.segments if not segment.ok and segment.error), result.error)
        raise RuntimeError(first_error or f"TTS {engine_id}: worker zakonczyl prace bledem")
    _emit_local_worker_qc_summary(engine_id, config, result, progress)
    generated = _validated_worker_generated_audio(engine_id, request.segments, segments, result, run_paths.request_path.parent)
    segment_analysis = _local_worker_segment_analysis(segments, generated, result)
    progress(f"Generowanie segmentow {_format_duration_seconds(seconds)}")
    return GenerationOutput(generated, seconds, segment_analysis)


def _emit_local_worker_qc_summary(engine_id: str, config: dict, result: EngineResult, progress) -> None:
    retry_count = sum(int(segment.retries) for segment in result.segments)
    retry_segments = sum(1 for segment in result.segments if int(segment.retries) > 0)
    suspicious_count = sum(1 for segment in result.segments if segment.qc_score is not None and float(segment.qc_score) > 0)
    audio_enabled = _bool_config(config.get("audio_qc_enabled"), False)
    speech_enabled = _bool_config(config.get("whisper_qc_enabled"), False)
    if speech_enabled:
        _emit_qc_retry_summary_line(engine_id, "kontrola mowy", retry_segments, retry_count, progress)
    elif audio_enabled:
        _emit_qc_retry_summary_line(engine_id, "kontrola audio", retry_segments, retry_count, progress)
    elif retry_count:
        _emit_qc_retry_summary_line(engine_id, "kontrola jakosci", retry_segments, retry_count, progress)
    if suspicious_count:
        progress(f"Kontrola jakosci oznaczyla {_format_polish_count(suspicious_count, 'segment', 'segmenty', 'segmentow')} jako podejrzane")
        warning_summary = _local_worker_qc_warning_summary(result)
        if warning_summary:
            progress(f"Ostrzezenia kontroli jakosci: {warning_summary}")


def _local_worker_qc_warning_summary(result: EngineResult, limit: int = 4) -> str:
    counter: Counter[str] = Counter()
    for segment in result.segments:
        for warning in getattr(segment, "qc_warnings", ()) or ():
            warning_text = str(warning).strip()
            if warning_text:
                counter[warning_text] += 1
    if not counter:
        return ""
    parts = [f"{warning}: {count}" for warning, count in counter.most_common(max(1, int(limit)))]
    remaining = len(counter) - len(parts)
    if remaining > 0:
        parts.append(f"+{remaining}")
    return ", ".join(parts)


def _build_local_engine_request(
    engine_id: str,
    source_name: str,
    job_id: str,
    segments: list[SubtitleSegment],
    segments_dir: Path,
    config: dict,
    dictionary: dict,
) -> EngineRequest:
    requested_segments = [
        SegmentRequest(
            segment_id=segment.index,
            text=segment.text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            output_path=str(segments_dir / f"{segment.index:03d}_{segment.start_ms:09d}_{segment.end_ms:09d}.wav"),
        )
        for segment in segments
    ]
    return EngineRequest(
        engine_id=engine_id,
        source_name=source_name,
        job_id=job_id,
        segments=requested_segments,
        settings=config,
        dictionary={str(key): str(value) for key, value in (dictionary or {}).items()},
    )


def _prepare_local_voice_sample(
    engine_id: str,
    config: dict,
    paths: AppPaths,
    lektor_dir: Path,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict:
    _raise_if_cancelled(cancel_requested)
    key = _local_voice_sample_config_key(engine_id)
    if not key:
        return dict(config)
    source_text = str(config.get(key, "") or "").strip()
    if not source_text:
        return dict(config)
    source_path = Path(source_text)
    if not source_path.is_file():
        return dict(config)
    if source_path.suffix.lower() not in supported_voice_sample_extensions():
        return dict(config)
    _validate_local_voice_sample_duration(engine_id, source_path, paths)
    ffmpeg = find_ffmpeg(paths)
    if ffmpeg is None:
        raise RuntimeError(f"Brak ffmpeg do przygotowania probki glosu. {BINARY_LOOKUP_HINT}")

    enhance = _should_enhance_voice_sample(engine_id, config)
    prepared_path = _prepared_voice_sample_path(engine_id, source_path, paths, enhance)
    if not prepared_path.exists():
        progress(f"{engine_id}: przygotowanie probki glosu")
        prepare_voice_sample(
            ffmpeg,
            source_path,
            prepared_path,
            voice_sample_sample_rate(engine_id),
            enhance=enhance,
            cancel_requested=cancel_requested,
        )
    if engine_id != "chatterbox" and _bool_config(config.get("save_prepared_voice_sample"), False):
        debug_copy = lektor_dir / f"probka_glosu_{engine_id}_przygotowana.wav"
        if debug_copy.resolve() != prepared_path.resolve():
            debug_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prepared_path, debug_copy)
            progress(f"{engine_id}: zapisano przygotowana probke glosu")
    updated = dict(config)
    updated[key] = str(prepared_path)
    return updated


def _local_voice_sample_config_key(engine_id: str) -> str:
    return voice_sample_config_key(engine_id)


def _validate_local_voice_sample_duration(engine_id: str, source_path: Path, paths: AppPaths) -> None:
    rule = voice_sample_rule(engine_id)
    if rule is None:
        return
    ffprobe = find_ffprobe(paths)
    if ffprobe is None:
        raise RuntimeError(f"{rule.label}: nie moge sprawdzic dlugosci probki glosu, bo brakuje ffprobe. {BINARY_LOOKUP_HINT}")
    try:
        duration_seconds = probe_media_duration(ffprobe, source_path)
    except Exception as exc:
        raise RuntimeError(f"{rule.label}: nie mozna odczytac dlugosci pliku audio. Uzyj poprawnego WAV/MP3/FLAC.") from exc
    errors = validate_voice_sample_duration(engine_id, duration_seconds)
    if errors:
        raise RuntimeError(errors[0])


def _should_enhance_voice_sample(engine_id: str, config: dict) -> bool:
    if engine_id in {"chatterbox", "omnivoice", "coqui_xtts"}:
        return False
    return not _bool_config(config.get("disable_voice_sample_enhancement"), False)


def _prepared_voice_sample_path(engine_id: str, source_path: Path, paths: AppPaths, enhance: bool) -> Path:
    try:
        stat = source_path.stat()
        fingerprint_source = f"{source_path.resolve()}|{stat.st_size}|{int(stat.st_mtime_ns)}|{voice_sample_sample_rate(engine_id)}|enhance={int(bool(enhance))}"
    except OSError:
        fingerprint_source = f"{source_path.resolve()}|{voice_sample_sample_rate(engine_id)}|enhance={int(bool(enhance))}"
    digest = hashlib.sha256(fingerprint_source.encode("utf-8", errors="replace")).hexdigest()[:16]
    suffix = "ulepszona" if enhance else "bez_ulepszania"
    return paths.engine_dir(engine_id) / "cache" / "voice_samples" / f"{source_path.stem}_{suffix}_{digest}.wav"


def _validated_worker_generated_audio(
    engine_id: str,
    requested_segments: list[SegmentRequest],
    input_segments: list[SubtitleSegment],
    result: EngineResult,
    temp_dir: Path,
) -> list[tuple[int, Path]]:
    generated = _generated_audio_from_worker_result(engine_id, requested_segments, input_segments, result)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return generated


def _generated_audio_from_worker_result(
    engine_id: str,
    requested_segments: list[SegmentRequest],
    input_segments: list[SubtitleSegment],
    result: EngineResult,
) -> list[tuple[int, Path]]:
    if len(requested_segments) != len(input_segments):
        raise RuntimeError(f"TTS {engine_id}: niespojna liczba segmentow w zadaniu")
    seen_result_ids: set[int] = set()
    for segment in result.segments:
        if segment.segment_id in seen_result_ids:
            raise RuntimeError(f"TTS {engine_id}: worker zwrocil zduplikowany segment {segment.segment_id}")
        seen_result_ids.add(segment.segment_id)
    requested_ids = {segment.segment_id for segment in requested_segments}
    extra_result_ids = sorted(seen_result_ids - requested_ids)
    if extra_result_ids:
        raise RuntimeError(f"TTS {engine_id}: worker zwrocil niezamowiony segment {extra_result_ids[0]}")
    result_by_id = {segment.segment_id: segment for segment in result.segments}
    generated: list[tuple[int, Path]] = []
    for request, input_segment in zip(requested_segments, input_segments):
        segment_result = result_by_id.get(request.segment_id)
        if segment_result is None:
            raise RuntimeError(f"TTS {engine_id}: worker nie zwrocil segmentu {request.segment_id}")
        if not segment_result.ok:
            raise RuntimeError(segment_result.error or f"TTS {engine_id}: segment {request.segment_id} zakonczony bledem")
        _validate_worker_retry_diagnostics(engine_id, request.segment_id, segment_result)
        output_path = Path(segment_result.output_path or request.output_path)
        if output_path.resolve() != Path(request.output_path).resolve():
            raise RuntimeError(f"TTS {engine_id}: worker zwrocil inna sciezke audio dla segmentu {request.segment_id}: {output_path}")
        if not output_path.is_file():
            raise RuntimeError(f"TTS {engine_id}: worker nie utworzyl pliku audio dla segmentu {request.segment_id}: {output_path}")
        generated.append((input_segment.start_ms, output_path))
    return generated


def _validate_worker_retry_diagnostics(engine_id: str, segment_id: int, segment_result) -> None:
    attempts = int(getattr(segment_result, "attempts", 0) or 0)
    selected_attempt = int(getattr(segment_result, "selected_attempt", 0) or 0)
    retries = int(getattr(segment_result, "retries", 0) or 0)
    if attempts < 0 or selected_attempt < 0 or retries < 0:
        raise RuntimeError(f"TTS {engine_id}: nieprawidlowa diagnostyka retry segmentu {segment_id}")
    if attempts == 0 and selected_attempt == 0:
        return
    if attempts <= 0 or selected_attempt <= 0 or selected_attempt > attempts or retries >= attempts:
        raise RuntimeError(f"TTS {engine_id}: nieprawidlowa diagnostyka retry segmentu {segment_id}")


def _local_worker_progress(engine_id: str, total_segments: int, counter: dict[str, object], progress, line: str) -> None:
    match = re.search(r": segment\s+(\d+)\s+OK", line)
    if not match:
        retry_message = _local_worker_retry_progress_message(engine_id, total_segments, line)
        if retry_message is not None:
            progress(retry_message)
            return
        model_activity = _model_activity_message_for_worker_line(engine_id, line)
        seen = counter.setdefault("model_activity_seen", set())
        if not isinstance(seen, set):
            seen = set()
            counter["model_activity_seen"] = seen
        if model_activity is not None and model_activity not in seen:
            seen.add(model_activity)
            counter["model_activity"] = model_activity
            progress(model_activity)
        return
    counter["done"] = min(total_segments, int(counter.get("done", 0)) + 1)
    done = counter["done"]
    progress(f"Segment {done}/{total_segments}")


def _local_worker_retry_progress_message(engine_id: str, total_segments: int, line: str) -> str | None:
    text = str(line or "").strip()
    match = re.search(r": segment\s+(\d+),\s+(audio|mowa)\s+proba\s+(\d+)/(\d+)", text, re.IGNORECASE)
    if not match:
        return None
    attempt = int(match.group(3))
    if attempt <= 1:
        return None
    label = "kontrola audio" if match.group(2).lower() == "audio" else "kontrola mowy"
    message = _format_short_qc_retry_message(engine_id, label, int(match.group(1)), total_segments, attempt, int(match.group(4)))
    score_match = re.search(r"\bqc=([+-]?\d+)", text, re.IGNORECASE)
    if score_match:
        message += f", score: {score_match.group(1).strip()}"
    return message


def _model_activity_message_for_worker_line(engine_id: str, line: str) -> str | None:
    text = str(line or "").strip().lower()
    if not text:
        return None
    if text.startswith("whisper qc:"):
        if "sprawdzanie modelu" in text:
            return "Whisper QC: sprawdzanie obecnosci modelu"
        if "model w cache" in text:
            return "Whisper QC: model w cache"
        if "pobieranie modelu" in text:
            return "Whisper QC: pobieranie modelu - prosze czekac"
        if "ladowanie modelu" in text or "ładowanie modelu" in text:
            return "Whisper QC: ladowanie modelu - prosze czekac"
        return None
    if not text.startswith(f"{engine_id.lower()}:"):
        return None
    if "sprawdzanie modelu" in text:
        return "Model TTS: sprawdzanie obecnosci modelu"
    if "model w cache" in text:
        return "Model TTS: model w cache"
    if "pobieranie modelu" in text:
        return "Model TTS: pobieranie modelu - prosze czekac"
    if "ladowanie modelu" in text or "ładowanie modelu" in text:
        return "Model TTS: ladowanie modelu - prosze czekac"
    if re.search(r"\b(cache|cached)\b", text):
        return "Model TTS: sprawdzanie cache modelu"
    return None


def _ensure_common_whisper_cache(paths: AppPaths) -> None:
    common_cache = paths.whisper_cache_dir
    if _has_cache_payload(common_cache):
        return
    legacy_caches: list[Path] = [paths.cache_dir / "whisper"]
    legacy_root = paths.runtime_engines_dir
    if legacy_root.is_dir():
        legacy_caches.extend(legacy_root.glob("*/cache/whisper"))
    for legacy_cache in legacy_caches:
        if legacy_cache.resolve() == common_cache.resolve():
            continue
        if not _has_cache_payload(legacy_cache):
            continue
        common_cache.mkdir(parents=True, exist_ok=True)
        for item in legacy_cache.iterdir():
            if item.name == ".locks":
                continue
            destination = common_cache / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, destination, dirs_exist_ok=True)
                elif item.is_file() and not destination.exists():
                    shutil.copy2(item, destination)
            except OSError:
                continue


def _ensure_whisper_qc_runtime_if_enabled(
    config: dict,
    paths: AppPaths,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    if not _bool_config(config.get("whisper_qc_enabled"), False):
        return
    _raise_if_cancelled(cancel_requested)
    ensure_faster_whisper_runtime(paths, progress, device=str(config.get("whisper_qc_device", "cpu") or "cpu"))
    _raise_if_cancelled(cancel_requested)
    _ensure_common_whisper_cache(paths)


def _has_cache_payload(path: Path) -> bool:
    if not path.is_dir():
        return False
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            relative = item.relative_to(path)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == ".locks":
            continue
        return True
    return False


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _bounded_int(value, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def _bounded_float(value, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = minimum
    if number != number or number in {float("inf"), float("-inf")}:
        number = minimum
    return max(minimum, min(maximum, number))


def _bool_config(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "tak", "yes", "on"}:
        return True
    if text in {"0", "false", "nie", "no", "off"}:
        return False
    return default


def _format_duration_seconds(seconds: float) -> str:
    seconds_i = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds_i, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}min {seconds_part:02d}s"
    if minutes:
        return f"{minutes}min {seconds_part:02d}s"
    return f"{seconds_part}s"


def _format_duration_ms(milliseconds: int | float) -> str:
    ms_i = max(0, int(round(float(milliseconds))))
    hours, remainder = divmod(ms_i, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, ms_part = divmod(remainder, 1000)
    if hours:
        return f"{hours}h {minutes:02d}min {seconds:02d}s {ms_part:03d}ms"
    if minutes:
        return f"{minutes}min {seconds:02d}s {ms_part:03d}ms"
    if seconds:
        return f"{seconds}s {ms_part:03d}ms"
    return f"{ms_part}ms"


def _format_signed_duration_ms(milliseconds: int | float) -> str:
    ms_i = int(round(float(milliseconds)))
    sign = "+" if ms_i >= 0 else "-"
    return f"{sign}{_format_duration_ms(abs(ms_i))}"


def _source_media_diagnostics(source_path: Path, paths: AppPaths) -> tuple[float | None, list[dict]]:
    if not is_video_file(source_path):
        return None, []
    ffprobe = find_ffprobe(paths)
    if ffprobe is None:
        return None, []
    duration = probe_media_duration(ffprobe, source_path)
    audio_streams = probe_audio_streams(ffprobe, source_path)
    return (duration if duration > 0 else None), audio_streams


def _format_source_media_message(duration_s: float | None, audio_streams: list[dict]) -> str:
    duration = _format_duration_seconds(duration_s) if duration_s else "nieznany czas"
    audio = audio_stream_summary(selected_background_audio_stream(audio_streams))
    return f"Zrodlo: {duration}, audio tlo: {audio}"


def _safe_wav_diagnostics(path: Path | None) -> dict[str, int | float] | None:
    if path is None:
        return None
    try:
        return wav_audio_diagnostics(path)
    except Exception:
        return None


def _format_wav_diagnostics(diagnostics: dict[str, int | float]) -> str:
    duration = _format_duration_seconds(float(diagnostics.get("duration_s", 0.0)))
    channels = int(diagnostics.get("channels", 0) or 0)
    sample_rate = int(diagnostics.get("sample_rate", 0) or 0)
    channel_label = "mono" if channels == 1 else ("stereo" if channels == 2 else f"{channels} kan.")
    rate_label = f"{sample_rate // 1000} kHz" if sample_rate and sample_rate % 1000 == 0 else f"{sample_rate / 1000:.1f} kHz"
    peak_label = _format_peak_dbfs(float(diagnostics.get("peak_dbfs", -120.0)))
    return f"{duration}, {channel_label}, {rate_label}, peak {peak_label}"


def _format_peak_dbfs(value: float) -> str:
    if value <= -119.0:
        return "cisza"
    return f"{value:.1f} dBFS"


def _audio_output_track_label(surround_label: str, create_stereo_track: bool) -> str:
    labels = []
    if create_stereo_track:
        labels.append("PL 2.0")
    if surround_label:
        labels.append(f"PL {surround_label}")
    return " + ".join(labels) if labels else "PL audio"


def _format_output_audio_message(audio_streams: list[dict], surround_label: str, create_stereo_track: bool = True) -> str:
    expected_labels = []
    if create_stereo_track:
        expected_labels.append("PL 2.0")
    if surround_label:
        expected_labels.append(f"PL {surround_label}")
    expected_count = max(1, len(expected_labels))
    created = audio_streams[:expected_count]
    labels = []
    for index, stream in enumerate(created):
        name = expected_labels[index] if index < len(expected_labels) else f"PL {index + 1}"
        labels.append(f"{name}: {audio_stream_summary(stream)}")
    if not labels:
        labels.append("brak danych audio")
    default_label = expected_labels[0] if expected_labels else "PL"
    return f"MKV audio: {'; '.join(labels)}, domyslna {default_label}"


def _format_float(value: int | float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _minimum_timeline_duration_for_source(source_path: Path, paths: AppPaths) -> float | None:
    if not is_video_file(source_path):
        return None
    ffprobe = find_ffprobe(paths)
    if ffprobe is None:
        return None
    duration = probe_media_duration(ffprobe, source_path)
    return duration if duration > 0 else None


def _load_segments(path: Path) -> list[SubtitleSegment]:
    if path.suffix.lower() == ".srt":
        return load_srt(path)
    if path.suffix.lower() == ".txt":
        return load_txt_as_segment(path)
    raise RuntimeError("Obslugiwane wejscia na tym etapie: .srt, .txt oraz wideo z napisami tekstowymi.")


def _prepare_input_subtitles(
    source_path: Path,
    paths: AppPaths,
    lektor_dir: Path,
    progress,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    suffix = source_path.suffix.lower()
    if suffix in SUPPORTED_SUBTITLE_EXTENSIONS:
        return source_path
    if not is_video_file(source_path):
        raise RuntimeError("Nieobslugiwany typ pliku.")

    sidecar = _find_sidecar_subtitles(source_path)
    if sidecar is not None:
        progress(f"Napisy zewnetrzne: {sidecar.name}")
        return sidecar

    ffmpeg = find_ffmpeg(paths)
    ffprobe = find_ffprobe(paths)
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError(f"Brak ffmpeg/ffprobe. {BINARY_LOOKUP_HINT}")

    extracted_srt = lektor_dir / "extracted_subtitles.srt"
    progress("Wypakowywanie napisow z wideo")
    try:
        extract_first_subtitle_to_srt(ffmpeg, ffprobe, source_path, extracted_srt, cancel_requested=cancel_requested)
        return extracted_srt
    except RuntimeError:
        raise


def _find_sidecar_subtitles(video_path: Path) -> Path | None:
    parent = video_path.resolve().parent
    stem = video_path.stem
    for suffix in SIDECAR_SUFFIXES:
        for extension in SUPPORTED_SUBTITLE_EXTENSIONS:
            candidate = parent / f"{stem}{suffix}{extension}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None

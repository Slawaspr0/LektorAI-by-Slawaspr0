from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import wave
import zipfile
from array import array
from datetime import datetime
from pathlib import Path

from app.core.dictionary import sanitize_dictionary, save_dictionary
from app.core.config import AppConfigStore
from app.core.diagnostics import collect_diagnostics
from app.core.download import progress_percent_step_for_size
from app.core.gpu_devices import build_device_choices, detect_cuda_devices
from app.core.logging import app_log_path, broken_json_backup_path, engine_log_path, safe_name, setup_app_logger
from app.core.log_cleanup import cleanup_logs, preview_log_cleanup
from app.core.media_tools import (
    BINARY_LOOKUP_HINT,
    DEFAULT_LEKTOR_DELAY_MS,
    MAX_LEKTOR_DELAY_MS,
    MIN_LEKTOR_DELAY_MS,
    OUTPUT_AUDIO_SAMPLE_RATE,
    audio_stream_log_lines,
    audio_stream_summary,
    extract_primary_audio_command,
    find_binary,
    find_ffmpeg,
    find_mkvmerge,
    find_ffprobe,
    encode_wav_to_aac_command,
    mix_lektor_stereo_and_surround_audio_command,
    mix_lektor_stereo_audio_command,
    mix_lektor_surround_audio_command,
    normalize_lektor_wav,
    normalize_lektor_wav_command,
    prepare_voice_sample_command,
    probe_media_duration,
    primary_audio_channels,
    remux_with_prepared_lektor_audio_mkvmerge_command,
    sanitize_aac_bitrate,
    sanitize_lektor_delay_ms,
    ffmpeg_command_with_progress,
    select_background_audio_stream_index,
    selected_background_audio_stream,
    select_subtitle_stream_index,
    select_text_subtitle_stream_index,
    supported_voice_sample_extensions,
    trim_fixed_and_fade_wav_edges,
    voice_sample_sample_rate,
    wav_audio_diagnostics,
)
from app.core.paths import build_paths
from app.core.preflight import build_preflight_report
from app.core.version import APP_NAME, APP_VERSION
from app.cli.engine_commands import remove_engine_command
from app.engines.config_schema import (
    faster_whisper_device_kwargs,
    fields_for,
    is_audio_qc_field,
    is_diagnostic_field,
    is_speech_qc_field,
    visible_fields_for,
    whisper_qc_effective_compute_type,
    whisper_qc_compute_type_options_for_device,
)
from app.engines.config_validation import validate_engine_config, validate_whisper_qc_dependency
from app.engines.voice_sample_rules import (
    validate_voice_sample_duration,
    voice_sample_duration_help,
    voice_sample_rule,
)
from app.engines.install_specs import INSTALL_SPECS, TORCH_CU126_INDEX, TORCH_CU128_INDEX, get_install_spec, list_install_variants
from app.engines.manager import EngineManager, should_recreate_venv
import app.engines.manager as engine_manager_module
from app.engines.protocol import (
    EngineRequest,
    EngineResult,
    SegmentRequest,
    SegmentResult,
    read_result,
    write_request,
    write_result,
)
from app.engines.registry import get_engine_definitions
from app.engines.runner import DEFAULT_WORKER_TIMEOUT_S
from app.engines.schemas import EngineStatus
from app.pipeline.manifest import write_segments_manifest
from app.pipeline.progress import (
    FILE_PROGRESS_TOTAL,
    decode_progress_marker,
    encode_progress_marker,
    ffmpeg_progress_ratio,
    format_progress_status,
    progress_value_for_stage,
    safe_unit_eta_seconds,
)
from app.pipeline.audio_qc import analyze_generated_segments
from app.pipeline.subtitles import (
    SUPPORTED_SUBTITLE_EXTENSIONS,
    SubtitleSegment,
    apply_dictionary,
    clean_subtitle_text,
    load_srt,
    normalize_tts_text,
    save_srt,
)
from app.pipeline.whisper_qc import (
    faster_whisper_missing_message,
    normalize_for_whisper_qc,
    score_whisper_transcript,
    text_similarity,
)
from app.stt.faster_whisper_runtime import (
    faster_whisper_cuda_dll_dirs_for_package_dirs,
    faster_whisper_device_needs_cuda,
    faster_whisper_package_dirs,
    faster_whisper_worker_env,
)
from app.stt.cuda_runtime import (
    CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID,
    CUDA_RUNTIME_PACKAGES,
    CUDA_RUNTIME_WHISPER_CPP_ID,
    cuda_runtime_dll_dir,
    cuda_runtime_env,
    cuda_runtime_ready,
    ensure_cuda_runtime,
)
from app.updater.core import (
    SOURCE_ZIP_URL,
    cache_busted_update_url,
    check_for_updates,
    is_protected_update_path,
    is_update_available,
    read_update_info_from_zip,
    safe_relative_path,
    update_info_from_dict,
    update_info_matches,
    write_local_update_info,
)
from app.pipeline.tts_job import (
    _build_local_engine_request,
    _find_sidecar_subtitles,
    _format_builtin_qc_retry_message,
    _format_builtin_qc_selected_message,
    _format_short_qc_retry_message,
    _edge_retry_text,
    _edge_retry_text_variants,
    _aggregate_segment_analysis,
    _build_run_analysis,
    _sanitize_settings_snapshot,
    _generated_audio_from_worker_result,
    _cleanup_lektor_debug_files,
    _cleanup_successful_video_run,
    _ensure_common_whisper_cache,
    apply_lektor_delay_to_segments,
    _local_worker_qc_warning_summary,
    _local_worker_segment_analysis,
    _local_worker_progress,
    _local_worker_retry_progress_message,
    _model_activity_message_for_worker_line,
    _quality_chain_label,
    _quality_controls_summary,
    _should_encode_standalone_lektor_audio,
    _should_run_final_audio_qc,
    _validated_worker_generated_audio,
    PipelineTimings,
    local_worker_timeout_seconds,
    run_tts_job,
)
from app.pipeline.audio_timeline import SAMPLE_RATE, SEGMENT_EDGE_FADE_MS, build_lektor_wav
from app.pipeline.workspace import engine_short_code, lektor_assets_dir, lektorai_workspace_for, next_output_stem
from app.stt.job import (
    SttRemovedSegment,
    SttSettings,
    default_stt_transcribe_kwargs,
    filter_stt_dialogue_segments,
    filter_repeated_stt_hallucinations,
    is_stt_non_dialogue_text,
    is_stt_input_file,
    load_whisperx_json_segments,
    merge_short_stt_segments,
    next_stt_output_stem,
    save_stt_diagnostics,
    split_stt_subtitle_segments,
    split_stt_text,
    stt_removed_segments_as_srt,
    stt_removed_segments_payload,
    stt_model_cache_key,
    wrap_stt_subtitle_text,
)
from app.stt.languages import STT_LANGUAGE_CODES, STT_LANGUAGE_OPTIONS, stt_language_label
from app.stt.subtitle_profiles import (
    ENGLISH_USA_SUBTITLE_PROFILE,
    FALLBACK_SUBTITLE_PROFILE,
    subtitle_profile_for_language,
)
from app.stt.whisper_cpp_runtime import (
    WHISPER_CPP_RUNTIME_PACKAGES,
    build_whisper_cpp_command,
    normalize_whisper_cpp_model_name,
    sanitize_whisper_cpp_device,
    sanitize_whisper_cpp_runtime,
    whisper_cpp_model_file_name,
    whisper_cpp_runtime_download_label,
    whisper_cpp_runtime_env,
)
from app.stt.whisperx_runtime import (
    build_whisperx_command,
    ensure_whisperx_gpu_runtime,
    normalize_whisperx_compute_type,
    normalize_whisperx_model_name,
    whisperx_runtime_env,
    whisperx_device_args,
)
from app.ui.main_window import (
    audio_defaults_summary,
    build_start_confirmation_text,
    clear_engine_status_cache,
    compact_app_log_message,
    engine_combo_label,
    aac_quality_label,
    aac_quality_options,
    format_lektor_delay_label,
    format_duration,
    is_sidecar_for_existing_video,
    missing_ffmpeg_message,
    missing_media_tools_message,
    missing_mkvmerge_message,
    missing_ffprobe_message,
    natural_path_key,
    progress_bar_style,
    should_enable_engine_actions,
    should_enable_start_button,
    stored_engine_after_selection,
    worker_message_should_refresh_engine_status,
    main_window_minimum_size,
    main_window_refresh_keeps_engine_signals_blocked,
    slider_value_to_weight,
    weight_to_slider_value,
)
from app.ui.dialogs.settings_dialog import (
    choice_value_for_widget,
    coerce_bool_for_widget,
    coerce_float_for_widget,
    coerce_int_for_widget,
    choice_data_for_widget,
    edge_slider_value_for_widget,
    format_edge_slider_value,
    merge_engine_settings_values,
    diagnostic_field_groups,
    settings_help_button_label,
    should_show_vram_info_button,
)
from app.ui.dialogs.dictionary_dialog import should_show_dictionary_row
from app.ui.dialogs.dictionary_dialog import load_dictionary_external_file, save_dictionary_external_file
from app.ui.dialogs.diagnostics_dialog import diagnostic_table_text
from app.ui.dialogs.scrollable_text_dialog import scrollable_details_line_wrap_mode
from app.ui.dialogs.tts_manager_dialog import initial_install_button_label, should_enable_keep_settings_remove


def run_self_test(app_dir: Path) -> list[str]:
    messages: list[str] = []
    paths = build_paths(app_dir)
    _assert(paths.app_dir == app_dir.resolve(), "portable app_dir mismatch")
    _assert(paths.app_packages_dir == app_dir.resolve() / "packages", "portable app packages dir mismatch")
    _assert(paths.cache_dir == app_dir.resolve() / "cache", "portable app cache dir mismatch")
    _assert(paths.runtime_stt_dir == app_dir.resolve() / "stt", "portable STT runtime dir mismatch")
    _assert(paths.cuda_runtime_root_dir == app_dir.resolve() / "runtime" / "cuda", "portable CUDA runtime root mismatch")
    _assert(paths.cuda_runtime_downloads_dir == app_dir.resolve() / "runtime" / "cuda" / "downloads", "portable CUDA runtime downloads mismatch")
    _assert(
        paths.cuda_runtime_pack_dir(CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID)
        == app_dir.resolve() / "runtime" / "cuda" / CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID,
        "portable CUDA 12 runtime dir mismatch",
    )
    _assert(paths.stt_dir("faster_whisper") == app_dir.resolve() / "stt" / "faster_whisper", "portable faster-whisper STT dir mismatch")
    _assert(paths.whisper_cpp_stt_dir == app_dir.resolve() / "stt" / "whisper_cpp", "portable whisper.cpp STT dir mismatch")
    _assert(paths.whisper_cpp_runtime_bin_dir == app_dir.resolve() / "stt" / "whisper_cpp" / "bin", "portable whisper.cpp runtime dir mismatch")
    _assert(paths.whisper_cpp_runtime_metadata_path == app_dir.resolve() / "stt" / "whisper_cpp" / "runtime.json", "portable whisper.cpp runtime metadata mismatch")
    _assert(paths.whisperx_stt_dir == app_dir.resolve() / "stt" / "whisperx", "portable WhisperX STT dir mismatch")
    _assert(paths.whisperx_venv_dir == app_dir.resolve() / "stt" / "whisperx" / "venv", "portable WhisperX venv dir mismatch")
    _assert(paths.whisperx_cache_dir == app_dir.resolve() / "stt" / "whisperx" / "cache", "portable WhisperX cache dir mismatch")
    _assert(paths.faster_whisper_packages_dir == app_dir.resolve() / "stt" / "faster_whisper" / "packages", "portable faster-whisper packages dir mismatch")
    _assert(paths.faster_whisper_cache_dir == app_dir.resolve() / "stt" / "faster_whisper" / "cache", "portable faster-whisper cache dir mismatch")
    _assert(paths.whisper_cache_dir == paths.faster_whisper_cache_dir, "legacy Whisper cache alias mismatch")
    _assert(faster_whisper_package_dirs(paths)[0] == paths.faster_whisper_packages_dir, "faster-whisper should prefer STT packages")
    messages.append("paths: OK")

    start_module = _load_start_module(app_dir / "START.py")
    _assert(getattr(start_module, "APP_NAME", None) == APP_NAME, "START.py APP_NAME mismatch")
    _assert(getattr(start_module, "APP_VERSION", None) == APP_VERSION, "START.py APP_VERSION mismatch")
    _assert(hasattr(start_module, "APP_PACKAGES_DIR"), "START.py should expose portable app packages dir")
    messages.append("version sync: OK")

    update_json = app_dir / "update.json"
    _assert(update_json.is_file(), "update.json missing")
    local_update = update_info_from_dict(json.loads(update_json.read_text(encoding="utf-8")))
    _assert(local_update.app_name == APP_NAME, "update metadata app name mismatch")
    _assert(local_update.version == APP_VERSION, "update metadata version mismatch")
    _assert(local_update.zip_url == SOURCE_ZIP_URL, "update metadata zip URL mismatch")
    newer_update = update_info_from_dict({"version": APP_VERSION, "build_id": "9999.01.01.1", "zip_url": SOURCE_ZIP_URL})
    _assert(is_update_available(local_update, newer_update), "updater should detect newer build with same app version")
    same_update = update_info_from_dict(json.loads(update_json.read_text(encoding="utf-8")))
    _assert(not is_update_available(local_update, same_update), "updater should not flag identical metadata")
    _assert(update_info_matches(local_update, same_update), "updater metadata identity should match identical update info")
    _assert(not update_info_matches(local_update, newer_update), "updater metadata identity should reject different build id")
    cache_busted = cache_busted_update_url(SOURCE_ZIP_URL, "build 1")
    _assert("lektorai_build=build+1" in cache_busted, "updater cache-busted URL should include build id")
    update_zip_probe = app_dir / "_self_test_update_package.zip"
    update_write_probe_dir = app_dir / "_self_test_update_write"
    try:
        with zipfile.ZipFile(update_zip_probe, "w") as archive:
            archive.writestr(
                "LektorAI-main/update.json",
                json.dumps(
                    {
                        "app_name": APP_NAME,
                        "version": APP_VERSION,
                        "build_id": "123",
                        "zip_url": SOURCE_ZIP_URL,
                        "remove": [],
                    },
                    ensure_ascii=False,
                ),
            )
        zip_update = read_update_info_from_zip(update_zip_probe)
        _assert(zip_update is not None and zip_update.build_id == "123", "updater should read metadata from source ZIP")
        update_write_probe_dir.mkdir(parents=True, exist_ok=True)
        write_local_update_info(update_write_probe_dir, zip_update)
        written_update = update_info_from_dict(json.loads((update_write_probe_dir / "update.json").read_text(encoding="utf-8")))
        _assert(update_info_matches(zip_update, written_update), "updater should write remote metadata after update")
    finally:
        try:
            update_zip_probe.unlink()
        except OSError:
            pass
        _cleanup_tree(update_write_probe_dir)
    _assert(safe_relative_path("app/core/version.py") == Path("app/core/version.py"), "updater safe path mismatch")
    _assert(safe_relative_path("../config.json") is None, "updater should reject parent traversal")
    _assert(is_protected_update_path(Path("logs/app.log")), "updater should protect logs")
    _assert(is_protected_update_path(Path("runtime/cuda/file.dll")), "updater should protect CUDA runtime")
    _assert(is_protected_update_path(Path("config.json")), "updater should protect config")
    _assert(not is_protected_update_path(Path("app/core/version.py")), "updater should allow app code updates")
    remote_update_probe = app_dir / "_self_test_update_remote.json"
    try:
        remote_update_probe.write_text(
            json.dumps({"version": APP_VERSION, "build_id": "9999.01.01.1", "zip_url": SOURCE_ZIP_URL}, indent=2),
            encoding="utf-8",
        )
        check_result = check_for_updates(app_dir, info_url=remote_update_probe.as_uri())
        _assert(check_result.ok and check_result.update_available, "updater local probe should detect available update")
    finally:
        try:
            remote_update_probe.unlink()
        except OSError:
            pass
    messages.append("updater metadata: OK")

    _assert(progress_percent_step_for_size(5 * 1024 * 1024) == 0, "small downloads should not report percent steps")
    _assert(progress_percent_step_for_size(50 * 1024 * 1024) == 20, "medium downloads should report 20 percent steps")
    _assert(progress_percent_step_for_size(500 * 1024 * 1024) == 10, "large downloads should report 10 percent steps")
    messages.append("download progress policy: OK")

    version_result = subprocess.run(
        [sys.executable, "-B", str(app_dir / "START.py"), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    _assert(version_result.returncode == 0, "START.py --version should exit with 0")
    _assert(version_result.stdout.strip() == f"{APP_NAME} {APP_VERSION}", "START.py --version output mismatch")
    messages.append("launcher version: OK")

    help_probe_dir = app_dir / "_self_test_help_runtime"
    saved_srt_path = app_dir / "_self_test_saved.srt"
    try:
        help_probe_dir.mkdir(parents=True, exist_ok=True)
        help_start = app_dir / "START.py"
        help_result = subprocess.run(
            [sys.executable, "-B", str(help_start), "--help"],
            cwd=str(help_probe_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        _assert(help_result.returncode == 0, "START.py --help should exit with 0")
        _assert("START.py --version" in help_result.stdout, "START.py --help output missing version command")
        _assert(
            "--remove-engine ID      usuwa caly folder lokalnego TTS" in help_result.stdout,
            "START.py --help remove wording mismatch",
        )
        _assert(
            not (help_probe_dir / "config.json").exists()
            and not (help_probe_dir / "logs").exists()
            and not (help_probe_dir / "engines").exists(),
            "START.py --help should not create runtime files in cwd",
        )
    finally:
        _cleanup_tree(help_probe_dir)
    messages.append("launcher help: OK")

    cli_probe_dir = app_dir / "_self_test_cli_readonly_cwd"
    try:
        cli_probe_dir.mkdir(parents=True, exist_ok=True)
        cli_commands = [
            (
                ["--diagnose"],
                ("Aplikacja", "Python", "TTS: Edge TTS"),
                "START.py --diagnose",
            ),
            (
                ["--list-engines"],
                ("Edge TTS", "Chatterbox", "OmniVoice", "Piper TTS", "Coqui XTTS-v2", "Supertonic"),
                "START.py --list-engines",
            ),
            (
                ["--engine-install-plan", "chatterbox"],
                ("Silnik: Chatterbox", "Venv:", "Pakiety:"),
                "START.py --engine-install-plan chatterbox",
            ),
            (
                ["--engine-install-plan", "chatterbox", "--torch-cuda", "cu128"],
                ("Silnik: Chatterbox", "PyTorch CU128", TORCH_CU128_INDEX),
                "START.py --engine-install-plan chatterbox --torch-cuda cu128",
            ),
        ]
        for args, expected_parts, label in cli_commands:
            result = subprocess.run(
                [sys.executable, "-B", str(app_dir / "START.py"), *args],
                cwd=str(cli_probe_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
            _assert(result.returncode == 0, f"{label} should exit with 0")
            for expected in expected_parts:
                _assert(expected in result.stdout, f"{label} output missing {expected!r}")
        preflight_result = subprocess.run(
            [sys.executable, "-B", str(app_dir / "START.py"), "--preflight"],
            cwd=str(cli_probe_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        _assert(preflight_result.returncode in {0, 1}, "START.py --preflight should exit with readiness status")
        _assert(f"Preflight {APP_NAME}" in preflight_result.stdout, "START.py --preflight output missing header")
        _assert("Silniki gotowe" in preflight_result.stdout, "START.py --preflight output missing engine count")
        _assert(
            not (cli_probe_dir / "config.json").exists()
            and not (cli_probe_dir / "logs").exists()
            and not (cli_probe_dir / "engines").exists(),
            "readonly CLI commands should not create runtime files in cwd",
        )
    finally:
        _cleanup_tree(cli_probe_dir)
    messages.append("launcher readonly CLI: OK")

    log_cleanup_probe_dir = app_dir / "_self_test_log_cleanup"
    try:
        cleanup_paths = build_paths(log_cleanup_probe_dir)
        (cleanup_paths.logs_dir).mkdir(parents=True, exist_ok=True)
        engine_dir = cleanup_paths.runtime_engines_dir / "piper"
        engine_logs_dir = engine_dir / "logs"
        engine_cache_dir = engine_dir / "cache"
        engine_temp_dir = engine_dir / "temp"
        engine_logs_dir.mkdir(parents=True, exist_ok=True)
        engine_cache_dir.mkdir(parents=True, exist_ok=True)
        engine_temp_dir.mkdir(parents=True, exist_ok=True)
        old_app_log_path = cleanup_paths.logs_dir / "app_2026.05.12.10.00.00.log"
        current_app_log_path = cleanup_paths.logs_dir / "app_2026.05.12.10.05.00.log"
        old_app_log_path.write_text("old app log", encoding="utf-8")
        current_app_log_path.write_text("current app log", encoding="utf-8")
        (cleanup_paths.logs_dir / "keep.txt").write_text("keep", encoding="utf-8")
        (engine_logs_dir / "piper_2026.05.12.10.00.00.Film.log").write_text("engine log", encoding="utf-8")
        (engine_logs_dir / "piper_2026.05.12.10.00.00.Film.analysis.json").write_text("{}", encoding="utf-8")
        (engine_logs_dir / "keep.json").write_text("{}", encoding="utf-8")
        (engine_dir / "install.log").write_text("install", encoding="utf-8")
        (engine_dir / "config.json").write_text("{}", encoding="utf-8")
        (engine_dir / "dictionary.json").write_text("{}", encoding="utf-8")
        (engine_cache_dir / "model.bin").write_text("model", encoding="utf-8")
        (engine_temp_dir / "request.json").write_text("{}", encoding="utf-8")
        preview = preview_log_cleanup(cleanup_paths, active_app_log_path=current_app_log_path)
        _assert(preview["app_logs"] == 1, "log cleanup app log preview mismatch")
        _assert(preview["engine_logs"] == 2, "log cleanup engine log preview mismatch")
        _assert(preview["install_logs"] == 1, "log cleanup install log preview mismatch")
        _assert(preview["engine_temp"] == 1, "log cleanup temp preview mismatch")
        result = cleanup_logs(
            cleanup_paths,
            {"app_logs", "engine_logs", "install_logs", "engine_temp"},
            active_app_log_path=current_app_log_path,
        )
        _assert(not result.errors, "log cleanup should not report errors: " + "; ".join(result.errors))
        _assert(result.files_removed == 4, "log cleanup removed file count mismatch")
        _assert(result.dirs_removed == 1, "log cleanup removed dir count mismatch")
        _assert(not old_app_log_path.exists(), "old app log should be removed")
        _assert(current_app_log_path.is_file(), "current app log should stay")
        _assert((cleanup_paths.logs_dir / "keep.txt").is_file(), "non-log app file should stay")
        _assert(not (engine_logs_dir / "piper_2026.05.12.10.00.00.Film.log").exists(), "engine log should be removed")
        _assert(not (engine_logs_dir / "piper_2026.05.12.10.00.00.Film.analysis.json").exists(), "old analysis copy should be removed")
        _assert((engine_logs_dir / "keep.json").is_file(), "generic engine json should stay")
        _assert(not (engine_dir / "install.log").exists(), "install log should be removed")
        _assert((engine_dir / "config.json").is_file(), "engine config should stay")
        _assert((engine_dir / "dictionary.json").is_file(), "engine dictionary should stay")
        _assert((engine_cache_dir / "model.bin").is_file(), "engine cache should stay")
        _assert(not engine_temp_dir.exists(), "engine temp should be removed")
    finally:
        _cleanup_tree(log_cleanup_probe_dir)
    messages.append("log cleanup: OK")

    _assert(not (app_dir / "app" / "app").exists(), "nested app/app directory should not exist")
    forbidden_path_fragments = tuple(
        drive + separator + "LektorAI" + separator + folder
        for drive, separator in (("C:", "\\"), ("C:", "/"))
        for folder in ("TEST", "FINAL")
    )
    offenders: list[str] = []
    for scan_file in _portable_scan_files(app_dir):
        text = scan_file.read_text(encoding="utf-8", errors="replace")
        for fragment in forbidden_path_fragments:
            if fragment in text:
                offenders.append(f"{scan_file.relative_to(app_dir)}: {fragment}")
    _assert(not offenders, "hardcoded portable paths found: " + "; ".join(offenders[:5]))
    messages.append("portable layout: OK")

    portable_bins_dir = app_dir / "_self_test_portable_bins"
    portable_bin_paths = build_paths(portable_bins_dir)
    try:
        bin_dir = portable_bins_dir / "bin"
        tools_dir = portable_bins_dir / "tools"
        bin_dir.mkdir(parents=True, exist_ok=True)
        tools_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        mkvmerge_name = "mkvmerge.exe" if os.name == "nt" else "mkvmerge"
        path_dir = portable_bins_dir / "_path"
        path_dir.mkdir()
        path_ffmpeg = path_dir / ffmpeg_name
        path_ffprobe = path_dir / ffprobe_name
        path_mkvmerge = path_dir / mkvmerge_name
        path_ffmpeg.write_text("", encoding="utf-8")
        path_ffprobe.write_text("", encoding="utf-8")
        path_mkvmerge.write_text("", encoding="utf-8")
        (portable_bins_dir / ffmpeg_name).write_text("", encoding="utf-8")
        (portable_bins_dir / ffprobe_name).write_text("", encoding="utf-8")
        (portable_bins_dir / mkvmerge_name).write_text("", encoding="utf-8")
        (bin_dir / ffmpeg_name).write_text("", encoding="utf-8")
        (tools_dir / ffprobe_name).write_text("", encoding="utf-8")
        original_which = find_binary.__globals__["shutil"].which

        def fake_which(binary_name):
            mapping = {
                ffmpeg_name: path_ffmpeg,
                "ffmpeg": path_ffmpeg,
                ffprobe_name: path_ffprobe,
                "ffprobe": path_ffprobe,
                mkvmerge_name: path_mkvmerge,
                "mkvmerge": path_mkvmerge,
            }
            candidate = mapping.get(str(binary_name))
            return str(candidate) if candidate is not None else None

        find_binary.__globals__["shutil"].which = fake_which
        _assert(find_ffmpeg(portable_bin_paths) == path_ffmpeg, "portable ffmpeg PATH lookup mismatch")
        _assert(find_ffprobe(portable_bin_paths) == path_ffprobe, "portable ffprobe PATH lookup mismatch")
        _assert(find_mkvmerge(portable_bin_paths) == path_mkvmerge, "portable mkvmerge PATH lookup mismatch")
        find_binary.__globals__["shutil"].which = lambda _binary_name: None
        _assert(find_ffmpeg(portable_bin_paths) == portable_bins_dir / ffmpeg_name, "portable ffmpeg root fallback mismatch")
        _assert(find_ffprobe(portable_bin_paths) == portable_bins_dir / ffprobe_name, "portable ffprobe root fallback mismatch")
        _assert(find_mkvmerge(portable_bin_paths) == portable_bins_dir / mkvmerge_name, "portable mkvmerge root fallback mismatch")
        find_binary.__globals__["shutil"].which = original_which
        _assert("GitHub" in BINARY_LOOKUP_HINT and "README" in BINARY_LOOKUP_HINT and "START.py" in BINARY_LOOKUP_HINT, "binary lookup hint mismatch")
        _assert("PATH" in BINARY_LOOKUP_HINT and "runtime" not in BINARY_LOOKUP_HINT and "bin/tools" not in BINARY_LOOKUP_HINT, "binary lookup hint should mention PATH and root app folder only")
    finally:
        try:
            find_binary.__globals__["shutil"].which = original_which
        except UnboundLocalError:
            pass
        _cleanup_tree(portable_bins_dir)
    messages.append("portable binary lookup: OK")

    subtitle_streams = [
        {"codec_name": "subrip", "tags": {"language": "eng", "title": "Plain subtitles"}},
        {"codec_name": "subrip", "tags": {"language": "", "title": "Polish subtitles"}},
    ]
    _assert(select_subtitle_stream_index(subtitle_streams) == 1, "plain title should not count as Polish marker")
    subtitle_streams = [
        {"codec_name": "hdmv_pgs_subtitle", "tags": {"language": "pol", "title": "Polskie PGS"}},
        {"codec_name": "subrip", "tags": {"language": "eng", "title": "English SRT"}},
    ]
    _assert(select_subtitle_stream_index(subtitle_streams) == 1, "bitmap subtitles should not beat text subtitles")
    _assert(select_text_subtitle_stream_index(subtitle_streams) == 1, "text subtitle selector mismatch")
    bitmap_only_streams = [
        {"codec_name": "hdmv_pgs_subtitle", "tags": {"language": "pol", "title": "Polskie PGS"}},
        {"codec_name": "dvd_subtitle", "tags": {"language": "eng", "title": "English VobSub"}},
    ]
    _assert(select_text_subtitle_stream_index(bitmap_only_streams) is None, "bitmap-only subtitles should not be selected as text")
    forced_streams = [
        {"codec_name": "subrip", "tags": {"language": "pol", "title": "Polskie forced"}},
        {"codec_name": "subrip", "tags": {"language": "eng", "title": "English full"}},
    ]
    _assert(select_text_subtitle_stream_index(forced_streams) == 1, "forced subtitles should not beat full subtitles")
    regional_language_streams = [
        {"codec_name": "subrip", "tags": {"language": "eng", "title": "English full"}},
        {"codec_name": "subrip", "tags": {"language": "pl-PL", "title": "Full subtitles"}},
    ]
    _assert(select_text_subtitle_stream_index(regional_language_streams) == 1, "regional Polish language tag mismatch")
    forced_disposition_streams = [
        {"codec_name": "subrip", "disposition": {"forced": 1}, "tags": {"language": "pol", "title": "Polskie"}},
        {"codec_name": "subrip", "tags": {"language": "eng", "title": "English full"}},
    ]
    _assert(
        select_text_subtitle_stream_index(forced_disposition_streams) == 1,
        "forced disposition should not beat full subtitles",
    )
    default_disposition_streams = [
        {"codec_name": "subrip", "tags": {"language": "eng", "title": "English A"}},
        {"codec_name": "subrip", "disposition": {"default": 1}, "tags": {"language": "eng", "title": "English B"}},
    ]
    _assert(
        select_text_subtitle_stream_index(default_disposition_streams) == 1,
        "default disposition should break equal subtitle score ties",
    )
    messages.append("subtitle stream scoring: OK")

    audio_streams = [
        {
            "index": 1,
            "codec_name": "ac3",
            "channels": 2,
            "channel_layout": "stereo",
            "sample_rate": "48000",
            "tags": {"language": "pol", "title": "Dubbing PL"},
        },
        {
            "index": 2,
            "codec_name": "dts",
            "channels": 6,
            "channel_layout": "5.1",
            "sample_rate": "48000",
            "tags": {"language": "eng", "title": "Original"},
        },
    ]
    _assert(select_background_audio_stream_index(audio_streams) == 1, "audio background should skip Polish tracks when alternatives exist")
    _assert(selected_background_audio_stream(audio_streams) is audio_streams[1], "selected audio stream helper mismatch")
    _assert(primary_audio_channels(audio_streams) == 6, "selected background stream should drive surround detection")
    audio_log = "\n".join(audio_stream_log_lines(audio_streams))
    _assert(
        "Dubbing PL" in audio_log and "Original" in audio_log and "wybrana jako tlo" in audio_log,
        "audio stream log should show available metadata and selected track",
    )
    single_polish_audio = [
        {"index": 1, "codec_name": "ac3", "channels": 2, "tags": {"language": "pol", "title": "Polski dubbing"}}
    ]
    _assert(select_background_audio_stream_index(single_polish_audio) == 0, "single audio track should be used even if tagged Polish")
    title_marked_audio = [
        {"index": 1, "codec_name": "ac3", "channels": 2, "tags": {"language": "und", "title": "Polski dubbing"}},
        {"index": 2, "codec_name": "eac3", "channels": 6, "tags": {"language": "eng", "title": "Original"}},
    ]
    _assert(select_background_audio_stream_index(title_marked_audio) == 1, "Polish title should be treated as Polish audio metadata")
    all_polish_audio = [
        {"index": 1, "codec_name": "ac3", "channels": 2, "tags": {"language": "pol", "title": "PL"}},
        {"index": 2, "codec_name": "eac3", "channels": 6, "tags": {"language": "pl-PL", "title": "Polski"}},
    ]
    _assert(select_background_audio_stream_index(all_polish_audio) == 0, "all-Polish audio should fall back to first track")
    messages.append("audio stream selection: OK")

    sibling_test_dir = app_dir.parent / "TEST"
    sibling_final_dir = app_dir.parent / "FINAL"
    if sibling_test_dir.is_dir() and sibling_final_dir.is_dir():
        shared_files = ("START.py", "UPDATER.py", "update.json", "requirements.txt")
        mismatches = [
            name
            for name in shared_files
            if (sibling_test_dir / name).read_bytes() != (sibling_final_dir / name).read_bytes()
        ]
        _assert(not mismatches, "TEST/FINAL shared files mismatch: " + ", ".join(mismatches))
        messages.append("test/final shared files: OK")

        test_app_files = _relative_app_files(sibling_test_dir / "app")
        final_app_files = _relative_app_files(sibling_final_dir / "app")
        missing_in_final = sorted(test_app_files - final_app_files)
        extra_in_final = sorted(final_app_files - test_app_files)
        _assert(not missing_in_final, "FINAL app missing files: " + ", ".join(str(rel) for rel in missing_in_final[:5]))
        _assert(not extra_in_final, "FINAL app has extra files: " + ", ".join(str(rel) for rel in extra_in_final[:5]))
        app_mismatches = [
            rel
            for rel in sorted(test_app_files)
            if (sibling_test_dir / "app" / rel).read_bytes() != (sibling_final_dir / "app" / rel).read_bytes()
        ]
        _assert(not app_mismatches, "TEST/FINAL app files mismatch: " + ", ".join(str(rel) for rel in app_mismatches[:5]))
        messages.append("test/final app files: OK")

    requirements_path = app_dir / "requirements.txt"
    _assert(requirements_path.is_file(), "requirements.txt missing")
    app_requirements = _requirement_names(requirements_path)
    expected_app_requirements = {"pyqt6", "edge-tts", "openai"}
    _assert(expected_app_requirements.issubset(app_requirements), "app requirements missing expected packages")
    forbidden_app_requirements = {
        "torch",
        "torchaudio",
        "torchvision",
        "chatterbox-tts",
        "protobuf",
        "transformers",
        "diffusers",
        "onnx",
        "onnxruntime",
        "onnxruntime-gpu",
        "soundfile",
    }
    heavy_packages = sorted(app_requirements & forbidden_app_requirements)
    _assert(not heavy_packages, "heavy local TTS packages in app requirements: " + ", ".join(heavy_packages))
    messages.append("app requirements: OK")

    if app_dir.name.upper() == "FINAL":
        allowed_final_items = {
            "START.py",
            "UPDATER.py",
            "update.json",
            "requirements.txt",
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
            "lektorAI_screen.jpg",
            "config.json",
            "app",
            "cache",
            "logs",
            "stt",
            "runtime",
            "ffmpeg.exe",
            "ffprobe.exe",
            "mkvmerge.exe",
            "ffmpeg",
            "ffprobe",
            "mkvmerge",
        }
        unexpected_final_items = sorted(
            child.name
            for child in app_dir.iterdir()
            if child.name not in allowed_final_items and child.name != "__pycache__" and not child.name.startswith("_self_test_")
        )
        _assert(
            not unexpected_final_items,
            "FINAL package contains unexpected top-level items: " + ", ".join(unexpected_final_items[:8]),
        )
        forbidden_final_items = (
            "engines",
            "bin",
            "tools",
            "bin/ffmpeg.exe",
            "bin/ffprobe.exe",
            "bin/mkvmerge.exe",
            "bin/ffmpeg",
            "bin/ffprobe",
            "bin/mkvmerge",
            "tools/ffmpeg.exe",
            "tools/ffprobe.exe",
            "tools/mkvmerge.exe",
            "tools/ffmpeg",
            "tools/ffprobe",
            "tools/mkvmerge",
        )
        present_final_items = [name for name in forbidden_final_items if (app_dir / name).exists()]
        _assert(
            not present_final_items,
            "FINAL package contains runtime/heavy items: " + ", ".join(present_final_items),
        )
        messages.append("final release package: OK")
    elif app_dir.name.upper() == "TEST":
        allowed_test_items = {
            "START.py",
            "UPDATER.py",
            "update.json",
            "requirements.txt",
            "config.json",
            "app",
            "cache",
            "logs",
            "engines",
            "stt",
            "ffmpeg.exe",
            "ffprobe.exe",
            "mkvmerge.exe",
            "ffmpeg",
            "ffprobe",
            "mkvmerge",
            "bin",
            "tools",
            "packages",
            "napisy_testowe_wymowa_lektora_40.srt",
        }
        unexpected_test_items = sorted(
            child.name
            for child in app_dir.iterdir()
            if child.name not in allowed_test_items and child.name != "__pycache__" and not child.name.startswith("_self_test_")
        )
        _assert(
            not unexpected_test_items,
            "TEST package contains unexpected top-level items: " + ", ".join(unexpected_test_items[:8]),
        )
        messages.append("test package layout: OK")

    _assert("chatterbox_onnx_pl" not in INSTALL_SPECS, "archived Chatterbox ONNX PL should not be installable")
    _assert("fish_speech" not in INSTALL_SPECS, "archived Fish Speech should not be installable")
    _assert("vibevoice" not in INSTALL_SPECS, "archived VibeVoice should not be installable")
    local_specs = {engine_id: get_install_spec(engine_id) for engine_id in ("chatterbox", "omnivoice", "piper", "coqui_xtts", "supertonic")}
    for engine_id, spec in local_specs.items():
        _assert(spec.engine_id == engine_id, f"{engine_id} install spec id mismatch")
        _assert(spec.requirements, f"{engine_id} install spec has no requirements")
        _assert(spec.import_checks, f"{engine_id} install spec has no import checks")
        _assert(spec.package_installer in {"pip", "uv"}, f"{engine_id} install spec has invalid package installer")
        _assert(
            not any("faster-whisper" in requirement for requirement in spec.requirements),
            f"{engine_id} should use shared STT faster-whisper instead of installing it into venv",
        )
        _assert(
            "faster_whisper" not in spec.import_checks,
            f"{engine_id} install import checks should not validate app-level faster_whisper",
        )
        if engine_id not in {"piper", "supertonic"}:
            _assert(spec.torch_requirements, f"{engine_id} torch install spec has no torch requirements")
            _assert(
                any(requirement.startswith("torch") for requirement in spec.torch_requirements),
                f"{engine_id} install spec missing torch package",
            )
            _assert(
                any(requirement.startswith("torch") for requirement in spec.constraints),
                f"{engine_id} install spec missing torch constraint",
            )
        else:
            _assert(not spec.torch_requirements, f"{engine_id} should not install PyTorch")
            _assert(not spec.constraints, f"{engine_id} should not write torch constraints")
    _assert(local_specs["chatterbox"].torch_requirements == ("torch==2.6.0", "torchaudio==2.6.0"), "chatterbox should use original torch pair")
    _assert(local_specs["chatterbox"].torch_index_url == TORCH_CU126_INDEX, "chatterbox torch index mismatch")
    _assert(
        any("chatterbox-tts" in requirement and "github.com/resemble-ai/chatterbox" in requirement for requirement in local_specs["chatterbox"].requirements),
        "chatterbox should install current upstream repo",
    )
    _assert("chatterbox-tts" not in local_specs["chatterbox"].no_deps_requirements, "chatterbox should install original deps")
    _assert(not local_specs["chatterbox"].allowed_pip_check_prefixes, "chatterbox should not tolerate dependency conflicts in original env")
    chatterbox_variants = {variant.variant_id: variant for variant in list_install_variants("chatterbox")}
    _assert(set(chatterbox_variants) == {"cu126", "cu128"}, "chatterbox should expose CU126/CU128 install variants")
    _assert(chatterbox_variants["cu126"].spec is local_specs["chatterbox"], "chatterbox CU126 variant should use default spec")
    _assert(chatterbox_variants["cu128"].spec.torch_index_url == TORCH_CU128_INDEX, "chatterbox CU128 variant should use CU128 torch index")
    _assert(chatterbox_variants["cu128"].spec.torch_requirements == ("torch==2.8.0+cu128", "torchaudio==2.8.0+cu128"), "chatterbox CU128 variant should use torch 2.8 CU128")
    _assert(chatterbox_variants["cu128"].spec.no_deps_requirements, "chatterbox CU128 variant should install upstream package without deps")
    _assert(chatterbox_variants["cu128"].spec.allowed_pip_check_prefixes, "chatterbox CU128 variant should tolerate known torch metadata conflicts")
    _assert(local_specs["omnivoice"].torch_requirements == ("torch==2.8.0+cu128", "torchaudio==2.8.0+cu128"), "omnivoice should use its upstream torch pair")
    _assert(local_specs["omnivoice"].torch_index_url == TORCH_CU128_INDEX, "omnivoice torch index mismatch")
    _assert("omnivoice==0.1.5" in local_specs["omnivoice"].requirements, "omnivoice should install pinned PyPI release")
    _assert("piper-tts==1.4.2" in local_specs["piper"].requirements, "piper should install pinned upstream PyPI release")
    _assert("supertonic" in local_specs["supertonic"].requirements, "supertonic should install upstream PyPI release")
    _assert(local_specs["coqui_xtts"].torch_requirements == ("torch==2.8.0+cu126", "torchaudio==2.8.0+cu126"), "coqui XTTS should use a CUDA 12.6 torch pair")
    _assert(local_specs["coqui_xtts"].torch_index_url == TORCH_CU126_INDEX, "coqui XTTS torch index mismatch")
    coqui_variants = {variant.variant_id: variant for variant in list_install_variants("coqui_xtts")}
    _assert(set(coqui_variants) == {"cu126", "cu128"}, "coqui XTTS should expose CU126/CU128 install variants")
    _assert(coqui_variants["cu126"].spec is local_specs["coqui_xtts"], "coqui XTTS CU126 variant should use default spec")
    _assert(coqui_variants["cu128"].spec.torch_index_url == TORCH_CU128_INDEX, "coqui XTTS CU128 variant should use CU128 torch index")
    _assert(coqui_variants["cu128"].spec.torch_requirements == ("torch==2.8.0+cu128", "torchaudio==2.8.0+cu128"), "coqui XTTS CU128 variant should use torch 2.8 CU128")
    _assert(not list_install_variants("omnivoice"), "omnivoice already uses CU128 and should not show torch variants")
    _assert(not list_install_variants("piper"), "piper should not show torch variants")
    _assert(not list_install_variants("supertonic"), "supertonic should not show torch variants")
    _assert(
        any("coqui-tts" in requirement and "github.com/idiap/coqui-ai-TTS" in requirement for requirement in local_specs["coqui_xtts"].requirements),
        "coqui XTTS should install current upstream repo",
    )
    messages.append("local install specs: OK")

    manager = EngineManager(paths)
    manager._package_check_cache[("chatterbox|dummy", "chatterbox")] = ("chatterbox",)
    clear_engine_status_cache(manager)
    _assert(not manager._package_check_cache, "engine package cache should be clearable before explicit refresh")
    states = manager.list_states()
    engine_ids = {state.definition.engine_id for state in states}
    _assert(any(state.definition.engine_id == "edge" for state in states), "missing edge definition")
    _assert(any(state.definition.engine_id == "openai" for state in states), "missing openai definition")
    _assert(any(state.definition.engine_id == "chatterbox" for state in states), "missing chatterbox definition")
    _assert("chatterbox_onnx_pl" not in engine_ids, "archived Chatterbox ONNX PL should not be listed as active engine")
    _assert(any(state.definition.engine_id == "omnivoice" for state in states), "missing omnivoice definition")
    _assert(any(state.definition.engine_id == "piper" for state in states), "missing piper definition")
    _assert(any(state.definition.engine_id == "coqui_xtts" for state in states), "missing coqui XTTS definition")
    _assert(any(state.definition.engine_id == "supertonic" for state in states), "missing Supertonic definition")
    _assert("fish_speech" not in engine_ids, "archived Fish Speech should not be listed as active engine")
    _assert("f5_tts" not in engine_ids, "archived F5-TTS should not be listed as active engine")
    _assert("vibevoice" not in engine_ids, "archived VibeVoice should not be listed as active engine")
    for state in states:
        if state.definition.engine_id in {"chatterbox", "omnivoice", "piper", "coqui_xtts", "supertonic"}:
            install_checks = set(get_install_spec(state.definition.engine_id).import_checks)
            if state.definition.engine_id == "chatterbox":
                _assert(
                    "faster_whisper" not in state.definition.import_checks,
                    f"{state.definition.engine_id} runtime status should not validate shared STT faster_whisper",
                )
            _assert(install_checks.issubset(set(state.definition.import_checks)), f"{state.definition.engine_id} registry import checks should cover install import checks")
    missing_runtime_manager = EngineManager(build_paths(app_dir / "_self_test_missing_runtime"))
    _assert(not missing_runtime_manager.local_runtime_exists("chatterbox"), "missing venv should not count as local runtime")
    missing_script = manager._import_missing_script(("json", "missing_package_for_self_test"))
    _assert("missing_package_for_self_test" in missing_script, "missing import script should include checked names")
    diagnostics_script = manager._import_diagnostics_script(("json", "missing_package_for_self_test"))
    _assert("IMPORT OK" in diagnostics_script, "import diagnostics should log successful imports")
    _assert("IMPORT MISSING" in diagnostics_script, "import diagnostics should log missing imports")
    _assert("__file__" in diagnostics_script, "import diagnostics should log package origin")
    messages.append("engine registry: OK")

    readonly_probe_dir = app_dir / "_self_test_readonly_probe"
    readonly_paths = build_paths(readonly_probe_dir)
    readonly_manager = EngineManager(readonly_paths)
    _assert(not readonly_probe_dir.exists(), "readonly probe dir should start missing")
    readonly_manager.list_states()
    readonly_rows = collect_diagnostics(readonly_paths, readonly_manager)
    preflight_ok, preflight_lines = build_preflight_report(readonly_paths, readonly_manager)
    _assert(isinstance(preflight_ok, bool), "preflight should return boolean status")
    _assert(any(f"Preflight {APP_NAME}" in line for line in preflight_lines), "preflight report missing header")
    _assert(
        importlib.util.find_spec("faster_whisper") is not None
        or any("faster-whisper" in line for line in preflight_lines),
        "preflight should warn when faster-whisper is missing",
    )
    _assert(not readonly_probe_dir.exists(), "preflight should not create app runtime dirs")
    readonly_ffmpeg_row = next(row for row in readonly_rows if row[0] == "ffmpeg")
    readonly_mkvmerge_row = next(row for row in readonly_rows if row[0] == "mkvmerge")
    readonly_fw_row = next(row for row in readonly_rows if row[0] == "STT: faster-whisper")
    _assert(readonly_fw_row[1] in {"OK", "brak"}, "faster-whisper STT diagnostic row status mismatch")
    _assert(
        readonly_fw_row[1] == "OK" or "pierwszym uzyciu" in readonly_fw_row[2],
        "missing STT diagnostic should explain lazy first-use setup",
    )
    _assert(
        readonly_ffmpeg_row[2] == BINARY_LOOKUP_HINT or readonly_ffmpeg_row[1] == "OK",
        "missing ffmpeg diagnostic should show lookup hint",
    )
    _assert(
        readonly_mkvmerge_row[2] == BINARY_LOOKUP_HINT or readonly_mkvmerge_row[1] == "OK",
        "missing mkvmerge diagnostic should show lookup hint",
    )
    _assert(not readonly_probe_dir.exists(), "diagnostic/list commands should not create app runtime dirs")
    _assert(faster_whisper_device_needs_cuda("cuda"), "faster-whisper GPU runtime should detect cuda device")
    _assert(faster_whisper_device_needs_cuda("cuda:1"), "faster-whisper GPU runtime should detect indexed cuda device")
    _assert(not faster_whisper_device_needs_cuda("cpu"), "faster-whisper GPU runtime should not treat CPU as CUDA")
    _assert(
        CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID in CUDA_RUNTIME_PACKAGES,
        "CUDA runtime should expose CUDA 12 CTranslate2/PyTorch pack",
    )
    _assert(
        CUDA_RUNTIME_WHISPER_CPP_ID in CUDA_RUNTIME_PACKAGES,
        "CUDA runtime should expose CUDA 13 whisper.cpp pack",
    )
    _assert(
        CUDA_RUNTIME_PACKAGES[CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID].archive_name == "cuda12-ctranslate2-pytorch-win-x64.zip",
        "CUDA 12 runtime archive name mismatch",
    )
    _assert(
        CUDA_RUNTIME_PACKAGES[CUDA_RUNTIME_WHISPER_CPP_ID].archive_name == "cuda13-whispercpp-win-x64.zip",
        "CUDA 13 runtime archive name mismatch",
    )
    cuda_install_probe_dir = app_dir / "_self_test_cuda_runtime_install"
    cuda_install_paths = build_paths(cuda_install_probe_dir)
    try:
        local_releases_dir = cuda_install_probe_dir / "Releases"
        local_releases_dir.mkdir(parents=True)
        package = CUDA_RUNTIME_PACKAGES[CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID]
        archive_path = local_releases_dir / package.archive_name
        with zipfile.ZipFile(archive_path, "w") as archive:
            for dll_name in package.required_dlls:
                archive.writestr(dll_name, "")
        progress_lines: list[str] = []
        installed_dir = ensure_cuda_runtime(
            cuda_install_paths,
            CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID,
            progress=progress_lines.append,
        )
        _assert(installed_dir == cuda_runtime_dll_dir(cuda_install_paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID), "CUDA runtime install dir mismatch")
        _assert(cuda_runtime_ready(cuda_install_paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID), "CUDA runtime local archive install should prepare DLLs")
        _assert(any("lokalnej paczki" in line for line in progress_lines), "CUDA runtime should prefer local release archive")
    finally:
        _cleanup_tree(cuda_install_probe_dir)
    fw_cuda_probe_dir = app_dir / "_self_test_fw_cuda_runtime"
    fw_cuda_paths = build_paths(fw_cuda_probe_dir)
    try:
        cuda12_dir = cuda_runtime_dll_dir(fw_cuda_paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID)
        cuda12_dir.mkdir(parents=True)
        for dll_name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudart64_12.dll"):
            (cuda12_dir / dll_name).write_text("", encoding="utf-8")
        _assert(cuda_runtime_ready(fw_cuda_paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID), "CUDA 12 runtime pack should be detected")
        cuda_env = cuda_runtime_env(fw_cuda_paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID)
        _assert(str(cuda12_dir) in cuda_env.get("PATH", "").split(os.pathsep), "CUDA runtime env should prepend CUDA 12 DLL dir")

        cublas_dir = fw_cuda_paths.faster_whisper_packages_dir / "nvidia" / "cublas" / "bin"
        cudnn_dir = fw_cuda_paths.faster_whisper_packages_dir / "nvidia" / "cudnn" / "bin"
        runtime_dir = fw_cuda_paths.faster_whisper_packages_dir / "nvidia" / "cuda_runtime" / "bin"
        cublas_dir.mkdir(parents=True)
        cudnn_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)
        (cublas_dir / "cublas64_12.dll").write_text("", encoding="utf-8")
        (cudnn_dir / "cudnn64_9.dll").write_text("", encoding="utf-8")
        (runtime_dir / "cudart64_12.dll").write_text("", encoding="utf-8")
        dll_dirs = faster_whisper_cuda_dll_dirs_for_package_dirs((fw_cuda_paths.faster_whisper_packages_dir,))
        _assert(dll_dirs == (cublas_dir, cudnn_dir, runtime_dir), "faster-whisper CUDA DLL dirs should be stable and ordered")
        worker_env = faster_whisper_worker_env(fw_cuda_paths)
        path_entries = worker_env.get("PATH", "").split(os.pathsep)
        _assert(str(cuda12_dir) in path_entries, "faster-whisper worker PATH should include shared CUDA 12 runtime dir")
        _assert(str(cublas_dir) in path_entries, "faster-whisper worker PATH should include cuBLAS DLL dir")
        _assert(str(cudnn_dir) in path_entries, "faster-whisper worker PATH should include cuDNN DLL dir")
        _assert(str(runtime_dir) in path_entries, "faster-whisper worker PATH should include CUDA runtime DLL dir")
    finally:
        _cleanup_tree(fw_cuda_probe_dir)
    messages.append("readonly diagnostics: OK")

    install_preview_dir = app_dir / "_self_test_install_preview"
    install_preview_paths = build_paths(install_preview_dir)
    install_preview_manager = EngineManager(install_preview_paths)
    try:
        _assert(not install_preview_dir.exists(), "install preview dir should start missing")
        for engine_id in ("chatterbox", "omnivoice", "coqui_xtts"):
            lines = install_preview_manager.local_install_preview(engine_id)
            joined = "\n".join(lines)
            _assert("PyTorch CUDA:" in joined, f"install preview missing torch section for {engine_id}")
            _assert(f"  --index-url {get_install_spec(engine_id).torch_index_url}" in joined, f"install preview missing torch index for {engine_id}")
            _assert("Constraints:" in joined, f"install preview missing constraints for {engine_id}")
            _assert(
                any(f"  {constraint}" in joined for constraint in get_install_spec(engine_id).constraints),
                f"install preview missing constraint entries for {engine_id}",
            )
            _assert("Pakiety:" in joined, f"install preview missing requirements for {engine_id}")
            _assert("Instalator pakietow:" in joined, f"install preview missing package installer for {engine_id}")
            _assert("faster-whisper" not in joined, f"install preview should not install faster-whisper inside {engine_id}")
            _assert("Pakiety bez zaleznosci:" not in joined, f"{engine_id} should not use no-deps install section")
            _assert("  faster_whisper" not in joined, f"install preview should not show app-level faster_whisper for {engine_id}")
            _assert("Venv:" in joined, f"install preview missing venv path for {engine_id}")
            engine_dir = install_preview_paths.engine_dir(engine_id)
            torch_command = install_preview_manager._torch_install_command(
                engine_dir / "venv" / "Scripts" / "python.exe",
                get_install_spec(engine_id),
                engine_dir / "constraints.txt",
            )
            _assert("-c" in torch_command and str(engine_dir / "constraints.txt") in torch_command, f"torch install command missing constraints for {engine_id}")
        _assert(local_specs["supertonic"].package_installer == "uv", "supertonic should prefer uv installer")
        supertonic_preview = "\n".join(install_preview_manager.local_install_preview("supertonic"))
        _assert("Instalator pakietow: uv" in supertonic_preview, "supertonic preview should show uv installer")
        supertonic_dir = install_preview_paths.engine_dir("supertonic")
        uv_command = install_preview_manager._uv_requirements_install_command(
            supertonic_dir / "venv" / "Scripts" / "python.exe",
            supertonic_dir / "requirements.txt",
            supertonic_dir / "constraints.txt",
        )
        _assert(uv_command[1:4] == ["-m", "uv", "pip"], "uv package install command should use engine python module")
        _assert("--python" in uv_command, "uv package install command should target engine venv")
        uv_bootstrap_command = install_preview_manager._uv_bootstrap_command(supertonic_dir / "venv" / "Scripts" / "python.exe")
        _assert(uv_bootstrap_command[-1] == "uv", "uv bootstrap command should install uv into engine venv")
        for engine_id in ("chatterbox", "coqui_xtts"):
            cu128_preview = "\n".join(install_preview_manager.local_install_preview(engine_id, torch_variant="cu128"))
            _assert("PyTorch CU128" in cu128_preview, f"{engine_id} CU128 preview missing variant label")
            _assert(TORCH_CU128_INDEX in cu128_preview, f"{engine_id} CU128 preview missing torch index")
            _assert("+cu128" in cu128_preview, f"{engine_id} CU128 preview missing CUDA 12.8 torch package")
            cu128_command = install_preview_manager._torch_install_command(
                install_preview_paths.engine_dir(engine_id) / "venv" / "Scripts" / "python.exe",
                get_install_spec(engine_id, "cu128"),
                install_preview_paths.engine_dir(engine_id) / "constraints.txt",
            )
            _assert(TORCH_CU128_INDEX in cu128_command, f"{engine_id} CU128 install command missing index")
        try:
            install_preview_manager.local_install_preview("omnivoice", torch_variant="cu128")
            raise AssertionError("omnivoice CU128 variant preview should fail because it is already default")
        except ValueError:
            pass
        piper_preview = "\n".join(install_preview_manager.local_install_preview("piper"))
        _assert("PyTorch CUDA:" not in piper_preview, "piper install preview should not show torch section")
        _assert("PyTorch: nie wymagany" in piper_preview, "piper install preview should explain missing torch section")
        _assert("piper-tts==1.4.2" in piper_preview, "piper install preview missing piper package")
        supertonic_preview = "\n".join(install_preview_manager.local_install_preview("supertonic"))
        _assert("PyTorch CUDA:" not in supertonic_preview, "supertonic install preview should not show torch section")
        _assert("PyTorch: nie wymagany" in supertonic_preview, "supertonic install preview should explain missing torch section")
        _assert("supertonic" in supertonic_preview, "supertonic install preview missing supertonic package")
        _cleanup_tree(install_preview_dir)
        build_tools_command = install_preview_manager._build_tools_install_command(Path("python.exe"))
        _assert("--upgrade" not in build_tools_command, "build tools install should not upgrade pip on every retry")
        _assert("wheel" in build_tools_command and "setuptools" in build_tools_command, "build tools command missing expected packages")
        tolerated_pip_check = "chatterbox-tts 0.1.7 has requirement torch==2.6.0; python_version < \"3.14\", but you have torch 2.11.0+cu126."
        _assert(
            not install_preview_manager._is_allowed_pip_check_line(local_specs["chatterbox"], tolerated_pip_check),
            "chatterbox original env should not tolerate torch metadata conflicts",
        )
        _assert(
            not install_preview_manager._is_allowed_pip_check_line(local_specs["chatterbox"], "other-package 1.0 requires missing-package"),
            "unexpected pip check conflicts should not be tolerated",
        )
        try:
            install_preview_manager.local_install_preview("edge")
            raise AssertionError("edge install preview should fail")
        except ValueError:
            pass
        _assert(not install_preview_dir.exists(), "install preview should not create app runtime dirs")
    finally:
        _cleanup_tree(install_preview_dir)
    messages.append("install preview readonly: OK")

    old_openai_key = os.environ.pop("OPENAI_API_KEY", None)
    openai_probe_dir = app_dir / "_self_test_openai_key"
    openai_paths = build_paths(openai_probe_dir)
    openai_manager = EngineManager(openai_paths)
    try:
        no_key_state = openai_manager.state_for("openai")
        _assert(no_key_state.status == EngineStatus.REQUIRES_CONFIG, "openai without key should require config")
        _assert("api_key: brak" in no_key_state.components, "openai missing key label mismatch")
        config_path = openai_manager.ensure_engine_config("openai")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["api_key"] = "test-key"
        config_path.write_text(json.dumps(config_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config_key_state = openai_manager.state_for("openai")
        _assert("api_key: config" in config_key_state.components, "openai config key label mismatch")
        os.environ["OPENAI_API_KEY"] = "env-test-key"
        env_key_state = openai_manager.state_for("openai")
        _assert("api_key: ENV" in env_key_state.components, "openai env key label mismatch")
    finally:
        if old_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_openai_key
        _cleanup_tree(openai_probe_dir)
    messages.append("openai api key source: OK")

    log_probe_dir = app_dir / "_self_test_engine_logs"
    original_timestamp = engine_log_path.__globals__["timestamp"]
    try:
        engine_log_path.__globals__["timestamp"] = lambda: "2026.01.02.03.04.05"
        app_log_path.__globals__["timestamp"] = lambda: "2026.01.02.03.04.05"
        first_app_log = app_log_path(log_probe_dir)
        first_app_log.write_text("test\n", encoding="utf-8")
        second_app_log = app_log_path(log_probe_dir)
        _assert(first_app_log != second_app_log, "app log path should avoid collisions")
        _assert(second_app_log.name == "app_2026.01.02.03.04.05_2.log", f"unexpected app log suffix: {second_app_log.name}")
        first_log = engine_log_path(log_probe_dir, "chatterbox", "Film.mkv")
        first_log.write_text("test\n", encoding="utf-8")
        second_log = engine_log_path(log_probe_dir, "chatterbox", "Film.mkv")
        _assert(first_log != second_log, "engine log path should avoid collisions")
        _assert(second_log.name.endswith("_2.log"), f"unexpected engine log suffix: {second_log.name}")
        long_log = engine_log_path(log_probe_dir, "chatterbox", ("Zażółć" * 40) + ".mkv")
        _assert(len(long_log.name) < 140, f"engine log name should be bounded: {len(long_log.name)}")
        polish_filename = (
            "Za"
            + chr(0x017C)
            + chr(0x00F3)
            + chr(0x0142)
            + chr(0x0107)
            + " g"
            + chr(0x0119)
            + chr(0x015B)
            + "l"
            + chr(0x0105)
            + " ja"
            + chr(0x017A)
            + chr(0x0144)
            + " "
            + chr(0x0141)
            + chr(0x00F3)
            + "d"
            + chr(0x017A)
            + ".mkv"
        )
        _assert(safe_name(polish_filename) == "Zazolc_gesla_jazn_Lodz.mkv", "safe_name Polish transliteration mismatch")
    finally:
        engine_log_path.__globals__["timestamp"] = original_timestamp
        app_log_path.__globals__["timestamp"] = original_timestamp
        _cleanup_tree(log_probe_dir)
    messages.append("log naming: OK")

    app_logger_probe_dir = app_dir / "_self_test_app_logger"
    original_timestamp = app_log_path.__globals__["timestamp"]
    try:
        app_log_path.__globals__["timestamp"] = lambda: "2026.01.02.03.04.05"
        logger, first_log = setup_app_logger(app_logger_probe_dir)
        logger.info("first")
        logger, second_log = setup_app_logger(app_logger_probe_dir)
        logger.info("second")
        for handler in list(logger.handlers):
            handler.flush()
            logger.removeHandler(handler)
            handler.close()
        _assert(first_log.name == "app_2026.01.02.03.04.05.log", "unexpected first app logger name")
        _assert(second_log.name == "app_2026.01.02.03.04.05_2.log", "unexpected second app logger name")
        _assert("first" in first_log.read_text(encoding="utf-8"), "first app log missing message")
        _assert("second" in second_log.read_text(encoding="utf-8"), "second app log missing message")
    finally:
        app_log_path.__globals__["timestamp"] = original_timestamp
        _cleanup_tree(app_logger_probe_dir)
    messages.append("app logger reopen: OK")

    broken_probe_dir = app_dir / "_self_test_broken_json"
    broken_probe_dir.mkdir(parents=True, exist_ok=True)
    broken_json = broken_probe_dir / "config.json"
    original_timestamp = broken_json_backup_path.__globals__["timestamp"]
    try:
        broken_json_backup_path.__globals__["timestamp"] = lambda: "2026.01.02.03.04.05"
        first_backup = broken_json_backup_path(broken_json)
        first_backup.write_text("broken-1\n", encoding="utf-8")
        second_backup = broken_json_backup_path(broken_json)
        _assert(first_backup != second_backup, "broken json backup path should avoid collisions")
        _assert(
            second_backup.name == "config.broken.2026.01.02.03.04.05_2.json",
            f"unexpected broken backup suffix: {second_backup.name}",
        )
    finally:
        broken_json_backup_path.__globals__["timestamp"] = original_timestamp
        _cleanup_tree(broken_probe_dir)
    messages.append("broken json naming: OK")

    broken_runtime_dir = app_dir / "_self_test_broken_runtime_json"
    broken_runtime_paths = build_paths(broken_runtime_dir)
    original_timestamp = broken_json_backup_path.__globals__["timestamp"]
    try:
        broken_json_backup_path.__globals__["timestamp"] = lambda: "2026.01.02.03.04.05"
        broken_config = broken_runtime_dir / "config.json"
        broken_runtime_dir.mkdir(parents=True, exist_ok=True)
        broken_config.write_text("{broken config", encoding="utf-8")
        AppConfigStore(broken_config).load()
        first_config_backup = broken_runtime_dir / "config.broken.2026.01.02.03.04.05.json"
        _assert(first_config_backup.is_file(), "broken app config backup missing")
        _assert(json.loads(broken_config.read_text(encoding="utf-8")).get("version") == 1, "app config not recreated")
        broken_config.write_text("{broken config again", encoding="utf-8")
        AppConfigStore(broken_config).load()
        second_config_backup = broken_runtime_dir / "config.broken.2026.01.02.03.04.05_2.json"
        _assert(second_config_backup.is_file(), "second broken app config backup missing")

        broken_manager = EngineManager(broken_runtime_paths)
        engine_config_path = broken_runtime_paths.engine_dir("chatterbox") / "config.json"
        engine_config_path.parent.mkdir(parents=True, exist_ok=True)
        engine_config_path.write_text("{broken engine config", encoding="utf-8")
        broken_manager.ensure_engine_config("chatterbox")
        first_engine_config_backup = engine_config_path.parent / "config.broken.2026.01.02.03.04.05.json"
        _assert(first_engine_config_backup.is_file(), "broken engine config backup missing")
        recreated_engine_config = json.loads(engine_config_path.read_text(encoding="utf-8"))
        _assert(recreated_engine_config.get("t3_model") == "v2", "engine config not recreated")
        engine_config_path.write_text("{broken engine config again", encoding="utf-8")
        broken_manager.ensure_engine_config("chatterbox")
        second_engine_config_backup = engine_config_path.parent / "config.broken.2026.01.02.03.04.05_2.json"
        _assert(second_engine_config_backup.is_file(), "second broken engine config backup missing")
        engine_config_path.write_text('["not", "an", "object"]\n', encoding="utf-8")
        broken_manager.ensure_engine_config("chatterbox")
        third_engine_config_backup = engine_config_path.parent / "config.broken.2026.01.02.03.04.05_3.json"
        _assert(third_engine_config_backup.is_file(), "non-object engine config backup missing")
        recreated_non_object_config = json.loads(engine_config_path.read_text(encoding="utf-8"))
        _assert(
            recreated_non_object_config.get("t3_model") == "v2",
            "non-object engine config not recreated",
        )

        dictionary_path = broken_runtime_paths.engine_dir("chatterbox") / "dictionary.json"
        dictionary_path.write_text("{broken dictionary", encoding="utf-8")
        broken_manager.ensure_engine_dictionary("chatterbox")
        first_dictionary_backup = dictionary_path.parent / "dictionary.broken.2026.01.02.03.04.05.json"
        _assert(first_dictionary_backup.is_file(), "broken dictionary backup missing")
        _assert(json.loads(dictionary_path.read_text(encoding="utf-8")) == {}, "dictionary not recreated")
        dictionary_path.write_text("{broken dictionary again", encoding="utf-8")
        broken_manager.ensure_engine_dictionary("chatterbox")
        second_dictionary_backup = dictionary_path.parent / "dictionary.broken.2026.01.02.03.04.05_2.json"
        _assert(second_dictionary_backup.is_file(), "second broken dictionary backup missing")
        dictionary_path.write_text('["not", "a", "dictionary"]\n', encoding="utf-8")
        broken_manager.ensure_engine_dictionary("chatterbox")
        third_dictionary_backup = dictionary_path.parent / "dictionary.broken.2026.01.02.03.04.05_3.json"
        _assert(third_dictionary_backup.is_file(), "non-object dictionary backup missing")
        _assert(json.loads(dictionary_path.read_text(encoding="utf-8")) == {}, "non-object dictionary not recreated")
    finally:
        broken_json_backup_path.__globals__["timestamp"] = original_timestamp
        _cleanup_tree(broken_runtime_dir)
    messages.append("broken JSON quarantine: OK")

    lazy_dictionary_dir = app_dir / "_self_test_lazy_dictionary"
    lazy_dictionary_paths = build_paths(lazy_dictionary_dir)
    lazy_dictionary_manager = EngineManager(lazy_dictionary_paths)
    try:
        config_path = lazy_dictionary_manager.ensure_engine_config("edge")
        dictionary_path = lazy_dictionary_paths.engine_dir("edge") / "dictionary.json"
        _assert(config_path.is_file(), "engine config should be created on engine selection")
        _assert(config_path.read_text(encoding="utf-8").endswith("\n"), "engine config should end with newline")
        _assert(not dictionary_path.exists(), "dictionary should not be created with engine config")
        created_dictionary_path = lazy_dictionary_manager.ensure_engine_dictionary("edge")
        _assert(created_dictionary_path == dictionary_path, "dictionary path mismatch")
        _assert(json.loads(dictionary_path.read_text(encoding="utf-8")) == {}, "lazy dictionary content mismatch")
        dictionary_path.write_text(
            json.dumps({"alfa": "al fa", "x": "skip", "batman": "bat man", "joker": ""}, ensure_ascii=False),
            encoding="utf-8",
        )
        lazy_dictionary_manager.ensure_engine_dictionary("edge")
        normalized_dictionary_text = dictionary_path.read_text(encoding="utf-8")
        normalized_dictionary = json.loads(normalized_dictionary_text)
        _assert(normalized_dictionary == {"alfa": "al fa", "batman": "bat man"}, "existing dictionary normalize mismatch")
        _assert(
            normalized_dictionary_text.index('"alfa"') < normalized_dictionary_text.index('"batman"'),
            "existing dictionary normalize sort mismatch",
        )
        _assert(normalized_dictionary_text.endswith("\n"), "existing dictionary normalize newline mismatch")
    finally:
        _cleanup_tree(lazy_dictionary_dir)
    messages.append("lazy dictionary creation: OK")

    merge_config_dir = app_dir / "_self_test_engine_config_merge"
    merge_paths = build_paths(merge_config_dir)
    merge_manager = EngineManager(merge_paths)
    merge_config_path = merge_paths.engine_dir("chatterbox") / "config.json"
    try:
        merge_config_path.parent.mkdir(parents=True, exist_ok=True)
        merge_config_path.write_text(
            json.dumps(
                {
                    "t3_model": "v3",
                    "cfg_weight": True,
                    "whisper_qc_retry_attempts": "4",
                    "custom_private_flag": "keep-me",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        merge_manager.ensure_engine_config("chatterbox")
        merged_config = json.loads(merge_config_path.read_text(encoding="utf-8"))
        _assert(merged_config.get("t3_model") == "v3", "engine config merge overwrote user T3 model")
        _assert(float(merged_config.get("cfg_weight")) == 1.9, "engine config merge did not repair bad float")
        _assert(merged_config.get("whisper_qc_retry_attempts") == 4, "engine config merge did not coerce numeric int")
        _assert(merged_config.get("custom_private_flag") == "keep-me", "engine config merge removed unknown key")
        _assert(merged_config.get("seed") == 12345, "engine config merge did not add missing seed")
    finally:
        _cleanup_tree(merge_config_dir)
    messages.append("engine config merge: OK")

    temp_app_dir = app_dir / "_self_test_engine_manager"
    temp_paths = build_paths(temp_app_dir)
    temp_manager = EngineManager(temp_paths)
    engine_dir = temp_paths.engine_dir("chatterbox")
    try:
        temp_manager.prepare_local_engine("chatterbox")
        prepared_items = sorted(path.name for path in engine_dir.iterdir())
        _assert(prepared_items == ["install.log"], f"prepare local engine should not create user config: {prepared_items}")
        temp_manager.remove_engine_completely("chatterbox")
        temp_manager._ensure_venv = lambda _venv_dir, _log: None
        temp_manager._run_logged = lambda _command, _log, env=None: None
        temp_manager._run_pip_check = lambda _python_path, _spec, _log: None
        temp_manager._run_import_checks = lambda _definition, _python_path, _log: None
        temp_manager.install_worker_script = lambda engine_id: (temp_paths.engine_dir(engine_id) / "worker.py")
        temp_manager.install_local_engine("chatterbox")
        installed_items = sorted(path.name for path in engine_dir.iterdir())
        _assert("config.json" not in installed_items, f"install should not create user config: {installed_items}")
        _assert("dictionary.json" not in installed_items, f"install should not create user dictionary: {installed_items}")
        temp_manager.remove_engine_completely("chatterbox")
        temp_manager.ensure_engine_config("chatterbox")
        temp_manager.ensure_engine_dictionary("chatterbox")
        (engine_dir / "venv").mkdir(parents=True, exist_ok=True)
        (engine_dir / "cache").mkdir(parents=True, exist_ok=True)
        (engine_dir / "logs").mkdir(parents=True, exist_ok=True)
        (engine_dir / "install.log").write_text("test\n", encoding="utf-8")
        _assert(
            not temp_manager._has_model_cache(engine_dir),
            "empty local TTS cache should not count as downloaded model",
        )
        (engine_dir / "cache" / "whisper" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
        (engine_dir / "cache" / "whisper" / "model.bin").write_text("whisper\n", encoding="utf-8")
        _assert(
            not temp_manager._has_model_cache(engine_dir),
            "Whisper QC cache should not count as downloaded TTS model",
        )
        (engine_dir / "cache" / "hf" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
        (engine_dir / "cache" / "hf" / "model.bin").write_text("model\n", encoding="utf-8")
        _assert(temp_manager._has_model_cache(engine_dir), "non-empty local TTS cache should count as model payload")
        temp_manager.remove_engine_keep_user_settings("chatterbox")
        remaining = sorted(path.name for path in engine_dir.iterdir())
        _assert(remaining == ["config.json", "dictionary.json"], f"keep settings remove mismatch: {remaining}")
        _assert(not temp_manager.local_runtime_exists("chatterbox"), "removed runtime should not be selectable")
        temp_manager.remove_engine_completely("chatterbox")
        _assert(not engine_dir.exists(), "complete remove should delete engine dir")
        venv_env = temp_manager._venv_creation_env(engine_dir / "venv")
        _assert(venv_env["TEMP"].startswith(str(engine_dir)), "venv TEMP should stay inside engine dir")
        _assert(venv_env["TMP"].startswith(str(engine_dir)), "venv TMP should stay inside engine dir")
        _assert("pip_compat" in venv_env.get("PYTHONPATH", ""), "venv pip bootstrap should load pip compatibility patch")
        install_env = temp_manager._install_env(engine_dir)
        _assert(install_env["TEMP"].startswith(str(engine_dir)), "pip install TEMP should stay inside engine dir")
        _assert(install_env["TMP"].startswith(str(engine_dir)), "pip install TMP should stay inside engine dir")
        _assert(install_env["PYTHONIOENCODING"] == "utf-8", "pip install env should force UTF-8 stdio")
        _assert("pip_compat" in install_env.get("PYTHONPATH", ""), "pip install env should load pip compatibility patch")
        worker_env = temp_manager._worker_env()
        stt_env = faster_whisper_worker_env(temp_paths)
        _assert(worker_env.get("PYTHONIOENCODING") == "utf-8", "worker env should force UTF-8 stdio")
        _assert(worker_env.get("PYTHONUTF8") == "1", "worker env should enable Python UTF-8 mode")
        _assert(
            str(temp_paths.faster_whisper_packages_dir) in stt_env.get("LEKTORAI_STT_FASTER_WHISPER_PACKAGES_DIRS", ""),
            "STT env should expose faster-whisper packages",
        )
        _assert(
            worker_env.get("LEKTORAI_STT_FASTER_WHISPER_PACKAGES_DIRS") == stt_env.get("LEKTORAI_STT_FASTER_WHISPER_PACKAGES_DIRS"),
            "worker env should expose STT faster-whisper packages",
        )
        _assert(str(temp_paths.app_packages_dir) not in worker_env.get("PYTHONPATH", ""), "worker PYTHONPATH should not shadow venv packages")
        _assert(
            worker_env.get("LEKTORAI_STT_FASTER_WHISPER_CACHE_DIR") == str(temp_paths.faster_whisper_cache_dir),
            "worker env should point to STT faster-whisper cache",
        )
        legacy_app_whisper_payload = temp_paths.cache_dir / "whisper" / "models--app" / "file.bin"
        legacy_app_whisper_payload.parent.mkdir(parents=True, exist_ok=True)
        legacy_app_whisper_payload.write_text("model\n", encoding="utf-8")
        legacy_whisper_payload = temp_paths.engine_dir("edge") / "cache" / "whisper" / "models--test" / "file.bin"
        legacy_whisper_payload.parent.mkdir(parents=True, exist_ok=True)
        legacy_whisper_payload.write_text("model\n", encoding="utf-8")
        _ensure_common_whisper_cache(temp_paths)
        _assert(
            (temp_paths.whisper_cache_dir / "models--test" / "file.bin").is_file(),
            "legacy per-engine Whisper cache should migrate to shared cache",
        )
        _assert(
            (temp_paths.faster_whisper_cache_dir / "models--app" / "file.bin").is_file(),
            "legacy app Whisper cache should migrate to STT faster-whisper cache",
        )
    finally:
        _cleanup_tree(temp_app_dir)
    messages.append("engine removal: OK")

    temp_cli_dir = app_dir / "_self_test_cli_remove"
    temp_cli_paths = build_paths(temp_cli_dir)
    temp_cli_manager = EngineManager(temp_cli_paths)
    temp_cli_engine_dir = temp_cli_paths.engine_dir("chatterbox")
    try:
        code, message = remove_engine_command(temp_cli_manager, "edge", False)
        _assert(code == 2 and "wbudowany" in message, "built-in CLI remove should be blocked")
        code, message = remove_engine_command(temp_cli_manager, "nieistnieje", False)
        _assert(code == 2 and "Nieznany" in message, "unknown CLI remove should fail")
        code, message = remove_engine_command(temp_cli_manager, "chatterbox", True)
        _assert(code == 0 and "nie ma lokalnego runtime" in message, "missing runtime CLI remove mismatch")
        temp_cli_manager.ensure_engine_config("chatterbox")
        temp_cli_manager.ensure_engine_dictionary("chatterbox")
        (temp_cli_engine_dir / "cache").mkdir(parents=True, exist_ok=True)
        code, message = remove_engine_command(temp_cli_manager, "chatterbox", True)
        _assert(code == 0 and "ustawienia zachowane" in message, "cache-only keep settings CLI remove mismatch")
        remaining = sorted(path.name for path in temp_cli_engine_dir.iterdir())
        _assert(remaining == ["config.json", "dictionary.json"], f"CLI cache-only keep remove mismatch: {remaining}")
        venv_python = temp_cli_manager._venv_python(temp_cli_engine_dir)
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("", encoding="utf-8")
        (temp_cli_engine_dir / "cache").mkdir(parents=True, exist_ok=True)
        code, message = remove_engine_command(temp_cli_manager, "chatterbox", True)
        _assert(code == 0 and "ustawienia zachowane" in message, "keep settings CLI remove mismatch")
        remaining = sorted(path.name for path in temp_cli_engine_dir.iterdir())
        _assert(remaining == ["config.json", "dictionary.json"], f"CLI keep remove mismatch: {remaining}")
        code, message = remove_engine_command(temp_cli_manager, "chatterbox", True)
        _assert(
            code == 0 and "nie ma lokalnego runtime" in message and "ustawienia" in message,
            "config-only keep remove CLI message mismatch",
        )
        remaining = sorted(path.name for path in temp_cli_engine_dir.iterdir())
        _assert(remaining == ["config.json", "dictionary.json"], f"CLI config-only keep remove mismatch: {remaining}")
        code, message = remove_engine_command(temp_cli_manager, "chatterbox", False)
        _assert(code == 0 and "calkowicie" in message, "complete CLI remove mismatch")
        _assert(not temp_cli_engine_dir.exists(), "CLI complete remove should delete engine dir")
    finally:
        _cleanup_tree(temp_cli_dir)
    messages.append("CLI engine remove: OK")

    sample = "{\\an8}<i>Batman</i>\\N" + chr(0x266A)
    cleaned = clean_subtitle_text(sample)
    _assert(cleaned == "Batman", f"subtitle clean mismatch: {cleaned!r}")
    dialog_cleaned = clean_subtitle_text("<i>- Czas?</i>\n- Nie teraz.")
    _assert(dialog_cleaned == "Czas? Nie teraz.", f"subtitle dialog dash clean mismatch: {dialog_cleaned!r}")
    compact_dialog_cleaned = clean_subtitle_text("-Czas?\n–Nie teraz.")
    _assert(
        compact_dialog_cleaned == "Czas? Nie teraz.",
        f"subtitle compact dialog dash clean mismatch: {compact_dialog_cleaned!r}",
    )
    hyphen_cleaned = clean_subtitle_text("biało-czerwony")
    _assert(hyphen_cleaned == "biało-czerwony", f"subtitle internal hyphen should stay: {hyphen_cleaned!r}")
    replaced = apply_dictionary(sample, {"batman": "bat man"})
    _assert(replaced == "bat man", f"dictionary apply mismatch: {replaced!r}")
    literal_replaced = apply_dictionary("Batman", {"batman": r"bat\man"})
    _assert(literal_replaced == r"bat\man", f"dictionary replacement should be literal: {literal_replaced!r}")
    tts_normalized = normalize_tts_text("„Cześć” — powiedział…\u00a0OK")
    _assert(tts_normalized == '"Cześć" - powiedział, OK', f"TTS text normalization mismatch: {tts_normalized!r}")
    noisy_tts_text = normalize_tts_text(r"{\an8}<i>[muzyka]</i>\h JAN: „Cześć”… & patrz -> ★")
    _assert(noisy_tts_text == '"Cześć", i patrz', f"TTS noisy text cleanup mismatch: {noisy_tts_text!r}")
    messages.append("subtitle cleanup/dictionary: OK")

    sanitized, skipped = sanitize_dictionary(
        {
            " batman ": "bat man",
            "Batman": "duplicate",
            "joker": "",
            "x": "skip",
        }
    )
    _assert(sanitized == {"batman": "bat man"} and skipped == 3, "dictionary sanitize mismatch")
    dictionary_save_path = app_dir / "_self_test_dictionary_save" / "dictionary.json"
    try:
        count, skipped_save = save_dictionary(dictionary_save_path, {"alfa": "al fa", "batman": "bat man"})
        saved_dictionary_text = dictionary_save_path.read_text(encoding="utf-8")
        _assert(count == 2 and skipped_save == 0, "dictionary save count mismatch")
        _assert(saved_dictionary_text.endswith("\n"), "dictionary save should end with newline")
        _assert(saved_dictionary_text.index('"alfa"') < saved_dictionary_text.index('"batman"'), "dictionary save sort mismatch")
    finally:
        _cleanup_tree(dictionary_save_path.parent)
    messages.append("dictionary sanitize: OK")

    edge_errors = validate_engine_config("edge", {"voice": "", "rate": "fast", "pitch": "low"})
    _assert(any("glos" in error for error in edge_errors), "edge voice validation missing")
    _assert(any("predkosc" in error for error in edge_errors), "edge rate validation missing")
    edge_range_errors = validate_engine_config("edge", {"voice": "pl-PL-MarekNeural", "rate": "+150%", "pitch": "-150Hz"})
    _assert(any("predkosc" in error for error in edge_range_errors), "edge rate range validation missing")
    _assert(any("barwa" in error for error in edge_range_errors), "edge pitch range validation missing")
    bool_number_errors = validate_engine_config("chatterbox", {"cfg_weight": True})
    _assert(any("Stabilnosc tekstu" in error for error in bool_number_errors), "float bool validation missing")
    openai_bool_number_errors = validate_engine_config("openai", {"audio_qc_retry_attempts": True})
    _assert(any("Liczba prob kontroli audio" in error for error in openai_bool_number_errors), "int bool validation missing")
    fractional_int_errors = validate_engine_config("chatterbox", {"cfg_weight": 1.25, "whisper_qc_retry_attempts": 1.5})
    _assert(
        any("Liczba prob" in error for error in fractional_int_errors),
        "fractional int validation missing",
    )
    nonfinite_errors = validate_engine_config("chatterbox", {"cfg_weight": "nan"})
    _assert(any("skonczona" in error for error in nonfinite_errors), "non-finite float validation missing")
    device_errors = validate_engine_config("chatterbox", {"device": "gpu0"})
    _assert(any("Urzadzenie" in error for error in device_errors), "device validation label should be user friendly")
    whisper_device_errors = validate_engine_config("chatterbox", {"whisper_qc_device": "auto"})
    _assert(
        any("Urzadzenie kontroli mowy" in error for error in whisper_device_errors),
        "Whisper QC device validation label should be user friendly",
    )
    whisper_compute_errors = validate_engine_config("chatterbox", {"whisper_qc_compute_type": "fast"})
    _assert(
        any("Tryb kontroli mowy" in error for error in whisper_compute_errors),
        "Whisper QC compute type validation label should be user friendly",
    )
    _assert(
        whisper_qc_compute_type_options_for_device("cpu") == ("int8",),
        "Whisper QC on CPU should expose only int8",
    )
    _assert(
        "float16" in whisper_qc_compute_type_options_for_device("cuda:1"),
        "Whisper QC on CUDA should expose float16",
    )
    _assert(
        whisper_qc_effective_compute_type("cpu", "float16") == "int8",
        "Whisper QC CPU should force int8 compute type",
    )
    _assert(
        faster_whisper_device_kwargs("cuda:1") == {"device": "cuda", "device_index": 1},
        "faster-whisper cuda:N kwargs mismatch",
    )
    _assert(
        faster_whisper_device_kwargs("cpu") == {"device": "cpu"},
        "faster-whisper CPU kwargs mismatch",
    )
    omnivoice_range_errors = validate_engine_config("omnivoice", {"num_step": 2, "guidance_scale": 5.0, "speed": 2.0})
    _assert(any("Kroki inferencji" in error for error in omnivoice_range_errors), "omnivoice num_step range validation missing")
    _assert(any("CFG" in error for error in omnivoice_range_errors), "omnivoice CFG range validation missing")
    _assert(any("Predkosc" in error for error in omnivoice_range_errors), "omnivoice speed range validation missing")
    whisper_missing_errors = validate_whisper_qc_dependency({"whisper_qc_enabled": True}, lambda _name: False)
    _assert(any("faster-whisper" in error for error in whisper_missing_errors), "Whisper QC dependency validation missing")
    _assert(
        any("wylacz" in error.lower() or "requirements" in error.lower() for error in whisper_missing_errors),
        "Whisper QC dependency validation should suggest a fix",
    )
    whisper_disabled_errors = validate_whisper_qc_dependency({"whisper_qc_enabled": False}, lambda _name: False)
    _assert(whisper_disabled_errors == [], "disabled Whisper QC should not require dependency")
    local_whisper_errors = validate_engine_config("chatterbox", {"whisper_qc_enabled": True})
    _assert(
        not any("faster-whisper" in error for error in local_whisper_errors),
        "local TTS Whisper QC should be validated through its venv status, not the main app env",
    )
    _assert(voice_sample_rule("edge") is None, "edge should not have voice sample duration rules")
    _assert(voice_sample_rule("chatterbox").max_seconds == 30.0, "chatterbox voice sample max should be 30s")
    _assert(voice_sample_rule("omnivoice").max_seconds == 10.0, "omnivoice voice sample max should be 10s")
    _assert(voice_sample_rule("coqui_xtts").max_seconds == 10.0, "coqui XTTS voice sample max should be 10s")
    _assert(validate_voice_sample_duration("piper", 999.0) == [], "piper should not validate reference voice duration")
    _assert(validate_voice_sample_duration("omnivoice", 0.4), "very short OmniVoice sample should be rejected")
    _assert(any("za dluga" in error for error in validate_voice_sample_duration("omnivoice", 11.0)), "long OmniVoice sample should be rejected")
    _assert(any("30s" in error for error in validate_voice_sample_duration("chatterbox", 31.0)), "long Chatterbox sample should mention 30s max")
    _assert("3-10s" in voice_sample_duration_help("coqui_xtts"), "voice sample tooltip should include recommended duration")
    optional_audio_probe = app_dir / "_self_test_optional_audio.txt"
    optional_mp3_probe = app_dir / "_self_test_optional_audio.mp3"
    optional_flac_probe = app_dir / "_self_test_optional_audio.flac"
    try:
        optional_audio_probe.write_text("not wav\n", encoding="utf-8")
        optional_audio_errors = validate_engine_config("chatterbox", {"audio_prompt_path": str(optional_audio_probe)})
        _assert(any("WAV/MP3/FLAC" in error for error in optional_audio_errors), "optional voice sample extension validation missing")
        optional_mp3_probe.write_text("fake mp3 placeholder\n", encoding="utf-8")
        optional_flac_probe.write_text("fake flac placeholder\n", encoding="utf-8")
        chatterbox_mp3_errors = validate_engine_config("chatterbox", {"audio_prompt_path": str(optional_mp3_probe)})
        _assert(not chatterbox_mp3_errors, "chatterbox should accept MP3 voice samples")
        chatterbox_flac_errors = validate_engine_config("chatterbox", {"audio_prompt_path": str(optional_flac_probe)})
        _assert(not chatterbox_flac_errors, "chatterbox should accept FLAC voice samples for automatic preparation")
        omnivoice_mp3_errors = validate_engine_config("omnivoice", {"reference_audio_path": str(optional_mp3_probe)})
        _assert(not omnivoice_mp3_errors, "omnivoice should accept MP3 voice samples")
        coqui_mp3_errors = validate_engine_config("coqui_xtts", {"speaker_wav_path": str(optional_mp3_probe)})
        _assert(not coqui_mp3_errors, "coqui XTTS should accept MP3 voice samples")
    finally:
        for probe in (optional_audio_probe, optional_mp3_probe, optional_flac_probe):
            try:
                probe.unlink()
            except OSError:
                pass
    _assert(fields_for("tada") == (), "removed TADA should not have active config schema")
    defaults = {definition.engine_id: definition.default_config for definition in get_engine_definitions()}
    edge_fields = {field.key: field for field in fields_for("edge")}
    edge_field_keys = set(edge_fields)
    _assert("edge_apply_segment_fade" in edge_field_keys, "edge fade option should be visible in settings schema")
    _assert(edge_fields["edge_apply_segment_fade"].label == "Przytnij i wygladz brzegi", "edge tuning label mismatch")
    _assert("edge_trim_mode" not in edge_field_keys, "edge automatic trim mode should not be exposed")
    _assert(edge_fields["edge_trim_start_ms"].minimum == 0 and edge_fields["edge_trim_start_ms"].maximum == 1000, "edge trim start range mismatch")
    _assert(edge_fields["edge_trim_end_ms"].minimum == 0 and edge_fields["edge_trim_end_ms"].maximum == 2000, "edge trim end range mismatch")
    _assert(defaults["edge"].get("voice") == "pl-PL-MarekNeural", "edge default voice should be Marek")
    _assert(defaults["edge"].get("rate") == "+0%" and defaults["edge"].get("pitch") == "+0Hz", "edge default sliders should be neutral")
    _assert(defaults["edge"].get("edge_apply_segment_fade") is True, "edge trim should be enabled by default")
    _assert(defaults["edge"].get("edge_trim_start_ms") == 180 and defaults["edge"].get("edge_trim_end_ms") == 800, "edge default trim values mismatch")
    _assert(defaults["edge"].get("whisper_qc_enabled") is True, "edge Whisper QC should be enabled by default")
    _assert(defaults["edge"].get("whisper_qc_retry_attempts") == 3, "edge default Whisper retry attempts mismatch")
    _assert(defaults["edge"].get("whisper_qc_model") == "small", "edge default Whisper model mismatch")
    _assert(defaults["edge"].get("whisper_qc_min_similarity") == 0.92, "edge default Whisper similarity mismatch")
    _assert("edge_trim_fade_ms" not in edge_field_keys, "edge trim fade should stay hidden/internal")
    _assert(edge_fields["voice"].field_type == "choice", "edge voice should be a dropdown")
    _assert(edge_fields["voice"].show_help is False, "edge voice should not show a help button")
    _assert(edge_fields["voice"].option_labels == ("Marek", "Zofia"), "edge voice dropdown should use simple labels")
    _assert(edge_fields["rate"].field_type == "percent_slider", "edge rate should be a percent slider")
    _assert(edge_fields["pitch"].field_type == "hz_slider", "edge pitch should be a Hz slider")
    _assert(edge_fields["rate"].minimum == -100 and edge_fields["rate"].maximum == 100, "edge rate slider range mismatch")
    _assert(edge_fields["pitch"].minimum == -50 and edge_fields["pitch"].maximum == 50, "edge pitch slider range mismatch")
    _assert(edge_fields["rate"].step == 1 and edge_fields["pitch"].step == 1, "edge sliders should use precise step 1")
    edge_builtin_text = (app_dir / "app" / "engines" / "builtin" / "edge.py").read_text(encoding="utf-8")
    _assert("max_attempts = 3" in edge_builtin_text and "await asyncio.sleep" in edge_builtin_text, "edge builtin should retry transient online TTS failures")
    _assert(DEFAULT_WORKER_TIMEOUT_S >= 8 * 60 * 60, "local worker timeout should allow long quality runs")
    _assert(
        local_worker_timeout_seconds() == DEFAULT_WORKER_TIMEOUT_S,
        "pipeline should use the central local worker timeout",
    )
    _assert(supported_voice_sample_extensions() == (".wav", ".mp3", ".flac"), "voice sample supported extensions mismatch")
    _assert(voice_sample_sample_rate("chatterbox") == 24000, "chatterbox prepared sample rate mismatch")
    _assert(voice_sample_sample_rate("omnivoice") == 24000, "omnivoice prepared sample rate mismatch")
    _assert(voice_sample_sample_rate("coqui_xtts") == 24000, "coqui XTTS prepared sample rate mismatch")
    voice_prepare_command = prepare_voice_sample_command(
        Path("ffmpeg.exe"),
        Path("input.flac"),
        Path("prepared.wav"),
        sample_rate=24000,
        enhance=True,
    )
    voice_prepare_text = " ".join(str(part) for part in voice_prepare_command)
    _assert("-ac 1" in voice_prepare_text, "voice sample preparation should force mono")
    _assert("-ar 24000" in voice_prepare_text, "voice sample preparation should force engine sample rate")
    _assert("afftdn" in voice_prepare_text and "loudnorm" in voice_prepare_text, "voice sample preparation should clean and normalize")
    _assert("highpass" in voice_prepare_text and "lowpass" in voice_prepare_text, "voice sample preparation should use gentle filters")
    voice_convert_only_command = prepare_voice_sample_command(
        Path("ffmpeg.exe"),
        Path("input.mp3"),
        Path("prepared.wav"),
        sample_rate=24000,
        enhance=False,
    )
    voice_convert_only_text = " ".join(str(part) for part in voice_convert_only_command)
    _assert("loudnorm" in voice_convert_only_text, "voice sample preparation without enhancers should still normalize")
    _assert("afftdn" not in voice_convert_only_text, "voice sample preparation without enhancers should skip denoise")
    _assert("highpass" not in voice_convert_only_text and "lowpass" not in voice_convert_only_text, "voice sample preparation without enhancers should skip band filters")
    lektor_normalize_command = normalize_lektor_wav_command(Path("ffmpeg.exe"), Path("lektor_przed_normalizacja.wav"), Path("lektor_po_normalizacji.wav"))
    lektor_normalize_text = " ".join(str(part) for part in lektor_normalize_command)
    _assert("loudnorm=I=-14" in lektor_normalize_text, "lektor track normalization command mismatch")
    _assert("linear=true" in lektor_normalize_text and "dual_mono=true" in lektor_normalize_text, "lektor loudnorm should use linear dual-mono mode")
    _assert(f"-ar {OUTPUT_AUDIO_SAMPLE_RATE}" in lektor_normalize_text, "lektor track normalization should force output sample rate")
    _assert("-map_metadata -1" in lektor_normalize_text, "lektor track normalization should clear metadata")
    custom_lektor_normalize_text = " ".join(str(part) for part in normalize_lektor_wav_command(Path("ffmpeg.exe"), Path("lektor.wav"), Path("out.wav"), -20))
    _assert("loudnorm=I=-20" in custom_lektor_normalize_text, "lektor track normalization should accept selected LUFS")
    sparse_norm_dir = app_dir / "_self_test_sparse_normalization"
    try:
        sparse_norm_dir.mkdir(parents=True, exist_ok=True)
        sparse_input = sparse_norm_dir / "sparse.wav"
        sparse_output = sparse_norm_dir / "sparse_norm.wav"
        fade_len = int(48000 * 0.006)
        sparse_samples = [0] * 48000 + [int(1000 * index / max(1, fade_len)) for index in range(fade_len)] + [1000] * 4800
        _write_test_wav(sparse_input, sparse_samples, sample_rate=48000)
        normalize_lektor_wav(Path("ffmpeg.exe"), sparse_input, sparse_output, -14)
        _assert(abs(_wav_sample_at(sparse_output, 48000)) <= 24, "sparse lektor normalization should keep silent-to-speech boundary near silence")
        _assert(_wav_max_delta_range(sparse_output, 48000, 48000 + fade_len) < 1000, "sparse lektor normalization should not create onset impulse")
    finally:
        _cleanup_tree(sparse_norm_dir)
    _assert(sanitize_aac_bitrate("384k") == "384k", "AAC bitrate sanitizer should accept 384k")
    _assert(sanitize_aac_bitrate("640") == "640k", "AAC bitrate sanitizer should add k suffix")
    _assert(sanitize_aac_bitrate("bad") == "384k", "AAC bitrate sanitizer should fallback to default")
    _assert(sanitize_lektor_delay_ms(123) == 100, "lektor delay should snap to configured step")
    _assert(sanitize_lektor_delay_ms(-5000) == 0, "lektor delay should clamp negative values to zero")
    _assert(sanitize_lektor_delay_ms(5000) == MAX_LEKTOR_DELAY_MS, "lektor delay should clamp to maximum")
    extract_audio_text = " ".join(str(part) for part in extract_primary_audio_command(Path("ffmpeg.exe"), Path("film.mkv"), Path("source_audio.wav")))
    _assert("-map 0:a:0" in extract_audio_text and "-c:a pcm_s16le" in extract_audio_text, "primary audio extraction should decode the selected background stream to PCM")
    extract_second_audio_text = " ".join(
        str(part)
        for part in extract_primary_audio_command(
            Path("ffmpeg.exe"),
            Path("film.mkv"),
            Path("source_audio.wav"),
            audio_stream_index=1,
        )
    )
    _assert("-map 0:a:1" in extract_second_audio_text, "primary audio extraction should support selected background stream index")
    _assert("asetpts=PTS-STARTPTS" in extract_audio_text and "first_pts=0" not in extract_audio_text, "primary audio extraction should normalize timestamps without async padding")
    _assert(
        audio_stream_summary({"codec_name": "eac3", "channels": 6, "channel_layout": "5.1", "sample_rate": "48000"})
        == "EAC3, 5.1, 48 kHz",
        "audio stream summary should be compact and diagnostic-friendly",
    )
    _assert(audio_stream_summary({}) == "brak danych audio", "empty audio stream summary should be explicit")
    wav_diag_dir = app_dir / "_self_test_wav_diagnostics"
    try:
        wav_diag_dir.mkdir(parents=True, exist_ok=True)
        wav_diag_path = wav_diag_dir / "diag.wav"
        _write_test_wav(wav_diag_path, [0, 16384, -32768, 0], sample_rate=48000)
        wav_diag = wav_audio_diagnostics(wav_diag_path)
        _assert(wav_diag["channels"] == 1, "wav diagnostics should report channel count")
        _assert(wav_diag["sample_rate"] == 48000, "wav diagnostics should report sample rate")
        _assert(0.0 < float(wav_diag["duration_s"]) < 0.001, "wav diagnostics should report short duration")
        _assert(float(wav_diag["peak_dbfs"]) == 0.0, "wav diagnostics should report full-scale peak")
    finally:
        _cleanup_tree(wav_diag_dir)
    aac_command_text = " ".join(str(part) for part in encode_wav_to_aac_command(Path("ffmpeg.exe"), Path("lektor.wav"), Path("lektor.m4a"), "384k"))
    _assert("-b:a 384k" in aac_command_text, "AAC encode command should use selected bitrate")
    _assert(f"-ar {OUTPUT_AUDIO_SAMPLE_RATE}" in aac_command_text, "AAC encode command should force output sample rate")
    progress_command_text = " ".join(str(part) for part in ffmpeg_command_with_progress(["ffmpeg.exe", "-hide_banner", "-y", "-i", "in.wav", "out.m4a"]))
    _assert("-progress pipe:1" in progress_command_text and "-nostats" in progress_command_text, "ffmpeg progress command should enable machine progress")
    stereo_stage_text = " ".join(
        str(part)
        for part in mix_lektor_stereo_audio_command(
            Path("ffmpeg.exe"),
            Path("source_audio.wav"),
            Path("lektor.m4a"),
            Path("pl_2_0.m4a"),
            lektor_weight=2.3,
            background_lufs=-18,
            background_weight=1.6,
            bitrate="384k",
        )
    )
    stereo_with_layout_stage_text = " ".join(
        str(part)
        for part in mix_lektor_stereo_audio_command(
            Path("ffmpeg.exe"),
            Path("source_audio.wav"),
            Path("lektor.m4a"),
            Path("pl_2_0.m4a"),
            lektor_weight=2.3,
            background_lufs=-18,
            background_weight=1.6,
            bitrate="384k",
            channel_layout="7.1",
        )
    )
    surround_stage_text = " ".join(
        str(part)
        for part in mix_lektor_surround_audio_command(
            Path("ffmpeg.exe"),
            Path("source_audio.wav"),
            Path("lektor.m4a"),
            Path("pl_5_1.m4a"),
            lektor_weight=2.3,
            background_lufs=-18,
            background_weight=1.6,
            bitrate="384k",
        )
    )
    surround_71_stage_text = " ".join(
        str(part)
        for part in mix_lektor_surround_audio_command(
            Path("ffmpeg.exe"),
            Path("source_audio.wav"),
            Path("lektor.m4a"),
            Path("pl_7_1.m4a"),
            lektor_weight=2.3,
            background_lufs=-18,
            background_weight=1.6,
            bitrate="384k",
            channel_layout="7.1",
        )
    )
    _assert("aformat=channel_layouts=stereo" in stereo_stage_text, "stereo mix stage should downmix background before adding lektor")
    _assert("volume=1" in stereo_stage_text and "volume=0.5897" in stereo_stage_text, "stereo mix stage should scale background and lektor predictably")
    _assert(stereo_stage_text.count("asetpts=PTS-STARTPTS") >= 2, "stereo mix stage should reset audio PTS")
    _assert("async=1:first_pts=0" not in stereo_stage_text, "stereo mix stage should not use async first_pts padding")
    _assert("normalize=0" in stereo_stage_text and "alimiter=limit=0.9:level=false:latency=1" in stereo_stage_text, "stereo mix stage should keep limiter protection")
    _assert("duration=longest" in stereo_stage_text, "stereo mix stage should keep PL audio alive for the longest input")
    _assert("-c:a aac" in stereo_stage_text and "-b:a 384k" in stereo_stage_text, "stereo mix stage should encode prepared PL audio")
    _assert("aformat=channel_layouts=stereo" in stereo_with_layout_stage_text, "stereo mix should accept surround layout hint and still output stereo")
    _assert("pan=5.1|FL=0*c0|FR=0*c0|FC=c0" in surround_stage_text, "surround mix should place lektor only in center channel")
    _assert("FL=0.70*c0" not in surround_stage_text and "FR=0.70*c0" not in surround_stage_text, "surround mix should not spread lektor to front left/right")
    _assert("aformat=channel_layouts=7.1" in surround_71_stage_text, "7.1 surround mix should preserve 7.1 layout")
    _assert("pan=7.1|FL=0*c0|FR=0*c0|FC=c0" in surround_71_stage_text, "7.1 surround mix should place lektor only in center channel")
    _assert("BL=0*c0|BR=0*c0" in surround_71_stage_text and "SL=0*c0|SR=0*c0" in surround_71_stage_text, "7.1 surround mix should keep lektor out of side and back channels")
    combined_mix_text = " ".join(
        str(part)
        for part in mix_lektor_stereo_and_surround_audio_command(
            Path("ffmpeg.exe"),
            Path("source_audio.wav"),
            Path("lektor.wav"),
            Path("pl_2_0.m4a"),
            Path("pl_7_1.m4a"),
            lektor_weight=2.3,
            background_lufs=-18,
            background_weight=1.6,
            bitrate="384k",
            channel_layout="7.1",
        )
    )
    _assert("pl_2_0.m4a" in combined_mix_text and "pl_7_1.m4a" in combined_mix_text, "combined mix should create stereo and surround outputs in one ffmpeg command")
    _assert(combined_mix_text.count("-filter_complex") == 1, "combined mix should use one filter graph")
    _assert("[lektor_pl_2_0]" in combined_mix_text and "[lektor_pl_7_1]" in combined_mix_text, "combined mix should expose both prepared track labels")
    _assert("asplit=2" in combined_mix_text, "combined mix should split decoded streams instead of running two separate commands")
    mkvmerge_remux_text = " ".join(
        str(part)
        for part in remux_with_prepared_lektor_audio_mkvmerge_command(
            Path("mkvmerge.exe"),
            Path("film.mkv"),
            (Path("pl_2_0.m4a"), Path("pl_5_1.m4a")),
            Path("out.mkv"),
            f"{APP_NAME} edge",
            track_labels=("2.0", "5.1"),
            source_audio_streams=({"index": 1},),
        )
    )
    _assert("--no-audio" in mkvmerge_remux_text and "film.mkv" in mkvmerge_remux_text, "mkvmerge remux should copy source video/subtitles without original audio first")
    _assert("--language 0:pol" in mkvmerge_remux_text and f"0:{APP_NAME} edge 2.0" in mkvmerge_remux_text, "mkvmerge remux should tag prepared PL tracks")
    _assert("--default-track-flag 0:yes" in mkvmerge_remux_text and "--default-track-flag 1:no" in mkvmerge_remux_text, "mkvmerge remux should set default flags for PL and original audio")
    _assert(validate_engine_config("edge", defaults["edge"]) == [], "edge default config should be valid")
    _assert(validate_engine_config("chatterbox", defaults["chatterbox"]) == [], "chatterbox default config should be valid")
    _assert(validate_engine_config("omnivoice", defaults["omnivoice"]) == [], "omnivoice default config should be valid")
    _assert(validate_engine_config("piper", defaults["piper"]) == [], "piper default config should be valid")
    _assert(validate_engine_config("coqui_xtts", defaults["coqui_xtts"]) == [], "coqui XTTS default config should be valid")
    _assert(validate_engine_config("supertonic", defaults["supertonic"]) == [], "Supertonic default config should be valid")
    openai_config = dict(defaults["openai"])
    openai_config["api_key"] = "test-key"
    _assert(validate_engine_config("openai", openai_config) == [], "openai configured default should be valid")
    _assert("chatterbox_onnx_pl" not in defaults, "archived Chatterbox ONNX PL should not have active defaults")
    _assert(fields_for("chatterbox_onnx_pl") == (), "archived Chatterbox ONNX PL should not expose config fields")
    _assert(visible_fields_for("chatterbox_onnx_pl") == (), "archived Chatterbox ONNX PL should not expose visible config fields")
    _assert("fish_speech" not in defaults, "archived Fish Speech should not have active defaults")
    _assert(fields_for("fish_speech") == (), "archived Fish Speech should not expose config fields")
    _assert(visible_fields_for("fish_speech") == (), "archived Fish Speech should not expose visible config fields")
    _assert("vibevoice" not in defaults, "archived VibeVoice should not have active defaults")
    _assert(fields_for("vibevoice") == (), "archived VibeVoice should not expose config fields")
    _assert(visible_fields_for("vibevoice") == (), "archived VibeVoice should not expose visible config fields")
    for engine_id in ("edge", "openai", "chatterbox", "omnivoice", "piper", "coqui_xtts", "supertonic"):
        field_map = {field.key: field for field in fields_for(engine_id)}
        field_labels = {field.key: field.label for field in fields_for(engine_id)}
        _assert(defaults[engine_id].get("open_workspace_on_finish") is False, f"{engine_id} open workspace default should be disabled")
        _assert(
            field_labels.get("open_workspace_on_finish") == "Otworz folder po pracy",
            f"{engine_id} open workspace label should be user friendly",
        )
        _assert(defaults[engine_id].get("normalize_tts_text") is True, f"{engine_id} text cleanup default should be enabled")
        _assert(
            field_labels.get("normalize_tts_text") == "Czyszczenie tekstu",
            f"{engine_id} text cleanup label should be user friendly",
        )
        if engine_id in {"edge", "chatterbox", "omnivoice", "piper", "coqui_xtts", "supertonic"}:
            _assert("audio_qc_enabled" not in field_labels, f"{engine_id} Audio QC should not be exposed in TTS settings")
            _assert("audio_qc_retry_attempts" not in field_labels, f"{engine_id} Audio QC retry should not be exposed in TTS settings")
        else:
            _assert(
                field_labels.get("audio_qc_enabled") == "Wlacz kontrole audio",
                f"{engine_id} Audio QC toggle label should be user friendly",
            )
            _assert(
                field_labels.get("audio_qc_retry_attempts") == "Liczba prob kontroli audio",
                f"{engine_id} Audio QC retry label should be user friendly",
            )
        _assert(
            field_labels.get("whisper_qc_enabled") == "Wlacz kontrole mowy",
            f"{engine_id} Whisper QC label should be user friendly",
        )
        _assert(
            field_labels.get("whisper_qc_retry_attempts") == "Liczba prob",
            f"{engine_id} Whisper QC retry label should be user friendly",
        )
        whisper_retry_field = field_map.get("whisper_qc_retry_attempts")
        _assert(whisper_retry_field is not None, f"{engine_id} Whisper QC retry field missing")
        _assert(whisper_retry_field.maximum == 5, f"{engine_id} Whisper QC retry should be locked to original plus 4 retries")
        _assert("4 ponowienia" in whisper_retry_field.tooltip, f"{engine_id} Whisper QC retry tooltip should explain retry cap")
        _assert(
            field_labels.get("whisper_qc_model") == "Model",
            f"{engine_id} Whisper model label should be user friendly",
        )
        _assert(
            field_labels.get("whisper_qc_min_similarity") == "Zgodnosc tekstu",
            f"{engine_id} Whisper similarity label should be user friendly",
        )
        _assert(field_labels.get("save_processed_subtitles") == "Napisy po obrobce", f"{engine_id} processed subtitle save label mismatch")
        _assert(field_labels.get("save_quality_report") == "Raport jakosci", f"{engine_id} quality report save label mismatch")
        _assert(field_labels.get("save_run_reports") == "Raporty techniczne", f"{engine_id} run report save label mismatch")
        _assert(field_labels.get("save_lektor_segments") == "Segmenty lektora", f"{engine_id} segment save label mismatch")
        _assert(field_labels.get("save_lektor_track_before_normalization") == "Sciezka przed normalizacja", f"{engine_id} pre-normalization save label mismatch")
        _assert(field_labels.get("save_lektor_track_after_normalization") == "Sciezka po normalizacji", f"{engine_id} post-normalization save label mismatch")
        _assert(field_labels.get("save_audio_mix_steps") == "Etapy miksowania audio", f"{engine_id} mix stage save label mismatch")
        _assert("save_lektor_assets" not in field_labels, f"{engine_id} vague lektor assets option should not be visible")
        _assert(defaults[engine_id].get("save_processed_subtitles") is False, f"{engine_id} should not save processed subtitles by default")
        _assert(defaults[engine_id].get("save_quality_report") is False, f"{engine_id} should not save quality report by default")
        _assert(defaults[engine_id].get("save_run_reports") is False, f"{engine_id} should not save run reports by default")
        _assert(defaults[engine_id].get("save_lektor_segments") is False, f"{engine_id} should not save segments by default")
        _assert(defaults[engine_id].get("save_lektor_track_before_normalization") is False, f"{engine_id} should not save pre-normalization track by default")
        _assert(defaults[engine_id].get("save_lektor_track_after_normalization") is False, f"{engine_id} should not save post-normalization track by default")
        _assert(defaults[engine_id].get("save_audio_mix_steps") is False, f"{engine_id} should not save audio mix steps by default")
        if "audio_qc_enabled" in defaults[engine_id]:
            _assert(defaults[engine_id].get("audio_qc_enabled") is False, f"{engine_id} Audio QC should be disabled by default")
        expected_whisper_enabled = engine_id in {"edge", "chatterbox", "omnivoice", "piper", "coqui_xtts", "supertonic"}
        _assert(
            defaults[engine_id].get("whisper_qc_enabled") is expected_whisper_enabled,
            f"{engine_id} Whisper QC default mismatch",
        )
        _assert(int(defaults[engine_id].get("whisper_qc_retry_attempts", 0)) >= 1, f"{engine_id} Whisper QC retry attempts should have a default")
        _assert(defaults[engine_id].get("whisper_qc_device") == "cpu", f"{engine_id} Whisper QC should default to CPU")
        _assert(defaults[engine_id].get("whisper_qc_compute_type") == "int8", f"{engine_id} Whisper QC should default to safe int8")
        _assert(field_labels.get("whisper_qc_device") == "Urzadzenie", f"{engine_id} Whisper QC device label should be simple")
        _assert(field_labels.get("whisper_qc_compute_type") == "Tryb pracy", f"{engine_id} Whisper QC compute label should be simple")
    gpu_choices = build_device_choices(
        (
            {"index": 0, "name": "NVIDIA RTX 3090", "total_memory": 25769803776},
            {"index": 1, "name": "NVIDIA RTX 3090", "total_memory": 25769803776},
        ),
        include_auto=True,
    )
    _assert(gpu_choices.values == ("auto", "cpu", "cuda:0", "cuda:1"), "GPU choices should expose detected CUDA devices")
    _assert("RTX 3090" in gpu_choices.labels[2] and "24.0 GB" in gpu_choices.labels[2], "GPU choice labels should show card name and VRAM")
    stt_gpu_choices = build_device_choices(({"index": 1, "name": "NVIDIA RTX 3090", "total_memory": 25769803776},), include_auto=False)
    _assert(stt_gpu_choices.values == ("cpu", "cuda:1"), "STT choices should default to CPU without auto")
    detector_calls: list[str] = []
    fast_devices = detect_cuda_devices(
        prefer_nvidia_smi=True,
        torch_detector=lambda *_args, **_kwargs: detector_calls.append("torch") or (),
        smi_detector=lambda **_kwargs: detector_calls.append("smi") or ({"index": 0, "name": "Fast GPU", "total_memory": 8589934592},),
    )
    _assert(fast_devices == ({"index": 0, "name": "Fast GPU", "total_memory": 8589934592},), "Fast GUI GPU detection should use nvidia-smi devices")
    _assert(detector_calls == ["smi"], "Fast GUI GPU detection should not import torch when nvidia-smi works")
    original_detect_cuda_devices = engine_manager_module.detect_cuda_devices
    try:
        manager_detector_calls: list[bool] = []

        def fake_detect_cuda_devices(_python_path, **kwargs):
            manager_detector_calls.append(bool(kwargs.get("prefer_nvidia_smi")))
            return ({"index": 0, "name": "Fast GPU", "total_memory": 8589934592},)

        engine_manager_module.detect_cuda_devices = fake_detect_cuda_devices
        manager_choices = EngineManager(paths).torch_device_choices("chatterbox", include_auto=True)
    finally:
        engine_manager_module.detect_cuda_devices = original_detect_cuda_devices
    _assert(manager_detector_calls == [True], "TTS settings GPU choices should use fast nvidia-smi-first detection")
    _assert(manager_choices.values == ("auto", "cpu", "cuda:0"), "TTS settings GPU choices should include fast-detected GPU")
    for engine_id in ("chatterbox",):
        _assert("seed" in {field.key for field in fields_for(engine_id)}, f"{engine_id} seed should stay in config schema")
        _assert("seed" not in {field.key for field in visible_fields_for(engine_id)}, f"{engine_id} seed should be hidden from UI")
    chatterbox_visible_keys = {field.key for field in visible_fields_for("chatterbox")}
    _assert("language_id" not in chatterbox_visible_keys, "chatterbox language should be fixed to Polish and hidden from UI")
    _assert("save_prepared_voice_sample" not in chatterbox_visible_keys, "chatterbox prepared voice sample debug option should be removed")
    _assert("disable_voice_sample_enhancement" not in chatterbox_visible_keys, "chatterbox voice sample enhancement option should be removed")
    _assert("language_id" not in defaults["chatterbox"], "chatterbox language should not be user config")
    _assert("save_prepared_voice_sample" not in defaults["chatterbox"], "chatterbox prepared voice sample debug default should be removed")
    _assert("disable_voice_sample_enhancement" not in defaults["chatterbox"], "chatterbox voice sample enhancement default should be removed")
    _assert("audio_qc_enabled" not in defaults["chatterbox"], "chatterbox Audio QC default should be removed")
    _assert("audio_qc_retry_attempts" not in defaults["chatterbox"], "chatterbox Audio QC retry default should be removed")
    _assert(chatterbox_visible_keys == {field.key for field in visible_fields_for("chatterbox")}, "chatterbox visible key cache mismatch")
    _assert("trim_leading_silence" in chatterbox_visible_keys, "chatterbox leading silence trim should be visible")
    _assert(defaults["chatterbox"].get("trim_leading_silence") is True, "chatterbox leading silence trim should be enabled by default")
    _assert(defaults["chatterbox"].get("t3_model") == "v2", "chatterbox should default to v2")
    _assert(defaults["chatterbox"].get("cfg_weight") == 1.9, "chatterbox CFG default mismatch")
    _assert(defaults["chatterbox"].get("exaggeration") == 0.1, "chatterbox exaggeration default mismatch")
    chatterbox_field_labels = {field.key: field.label for field in fields_for("chatterbox")}
    _assert(chatterbox_field_labels.get("t3_model") == "Wersja modelu Chatterbox", "chatterbox T3 model label should be user friendly")
    _assert(chatterbox_field_labels.get("trim_leading_silence") == "Wycinanie poczatkowej ciszy", "chatterbox leading trim label should be user friendly")
    _assert(chatterbox_field_labels.get("cfg_weight") == "Stabilnosc tekstu (CFG)", "chatterbox CFG label should be user friendly")
    _assert(chatterbox_field_labels.get("exaggeration") == "Ekspresja glosu", "chatterbox exaggeration label should be user friendly")
    omnivoice_visible_keys = {field.key for field in visible_fields_for("omnivoice")}
    _assert("language" not in omnivoice_visible_keys and "language_id" not in omnivoice_visible_keys, "omnivoice language should be fixed to Polish and hidden from UI")
    _assert("audio_qc_enabled" not in defaults["omnivoice"], "omnivoice Audio QC default should be removed")
    _assert("audio_qc_retry_attempts" not in defaults["omnivoice"], "omnivoice Audio QC retry default should be removed")
    _assert(defaults["omnivoice"].get("num_step") == 48, "omnivoice num_step default mismatch")
    _assert(defaults["omnivoice"].get("guidance_scale") == 3.8, "omnivoice CFG default mismatch")
    _assert(defaults["omnivoice"].get("speed") == 1.0, "omnivoice speed default mismatch")
    _assert(defaults["omnivoice"].get("denoise") is True, "omnivoice denoise default mismatch")
    _assert(defaults["omnivoice"].get("preprocess_prompt") is False, "omnivoice preprocess prompt should be disabled by default")
    _assert(defaults["omnivoice"].get("postprocess_output") is False, "omnivoice factory postprocess should be disabled by default")
    _assert(defaults["omnivoice"].get("omnivoice_trim_edges") is True, "omnivoice silence edge trim should be enabled by default")
    omnivoice_fields = {field.key: field for field in fields_for("omnivoice")}
    _assert(omnivoice_fields["num_step"].minimum == 4 and omnivoice_fields["num_step"].maximum == 64, "omnivoice step range mismatch")
    _assert(omnivoice_fields["guidance_scale"].minimum == 0.0 and omnivoice_fields["guidance_scale"].maximum == 4.0, "omnivoice CFG range mismatch")
    _assert(omnivoice_fields["speed"].minimum == 0.5 and omnivoice_fields["speed"].maximum == 1.5, "omnivoice speed range mismatch")
    _assert("Zakres 4-64" in omnivoice_fields["num_step"].tooltip, "omnivoice step tooltip should describe min/max")
    _assert("Zakres 0.0-4.0" in omnivoice_fields["guidance_scale"].tooltip, "omnivoice CFG tooltip should describe min/max")
    _assert("Zakres 0.5-1.5" in omnivoice_fields["speed"].tooltip, "omnivoice speed tooltip should describe min/max")
    _assert("preprocess_prompt" not in omnivoice_visible_keys, "omnivoice factory preprocess should be hidden from UI")
    _assert("postprocess_output" not in omnivoice_visible_keys, "omnivoice factory postprocess should be hidden from UI")
    _assert("omnivoice_trim_edges" in omnivoice_visible_keys, "omnivoice silence edge trim should be visible")
    _assert("omnivoice_trim_start_ms" not in omnivoice_visible_keys, "omnivoice manual trim start should not be visible")
    _assert("omnivoice_trim_end_ms" not in omnivoice_visible_keys, "omnivoice manual trim end should not be visible")
    _assert("tada" not in defaults, "removed TADA should not have active defaults")
    piper_fields = {field.key: field for field in fields_for("piper")}
    _assert(defaults["piper"].get("voice") == "pl_PL-mc_speech-medium", "piper default voice mismatch")
    _assert(defaults["piper"].get("length_scale") == 1.1, "piper default length scale mismatch")
    _assert(defaults["piper"].get("noise_scale") == 0.05, "piper default noise scale mismatch")
    _assert(defaults["piper"].get("noise_w_scale") == 0.05, "piper default noise width scale mismatch")
    _assert(piper_fields["voice"].field_type == "choice", "piper voice should be a dropdown")
    _assert("pl_PL-darkman-medium" in piper_fields["voice"].options and "pl_PL-gosia-medium" in piper_fields["voice"].options, "piper Polish voices missing")
    _assert(piper_fields["length_scale"].minimum == 0.5 and piper_fields["length_scale"].maximum == 2.0, "piper length scale range mismatch")
    _assert(piper_fields["noise_scale"].minimum == 0.0 and piper_fields["noise_scale"].maximum == 1.5, "piper noise range mismatch")
    _assert(defaults["piper"].get("whisper_qc_enabled") is True, "piper Whisper QC should be enabled by default")
    _assert(defaults["piper"].get("whisper_qc_retry_attempts") == 5, "piper Whisper QC retry default mismatch")
    _assert(defaults["piper"].get("whisper_qc_model") == "small", "piper Whisper QC model default mismatch")
    _assert(defaults["piper"].get("whisper_qc_min_similarity") == 0.93, "piper Whisper QC similarity default mismatch")
    coqui_visible_keys = {field.key for field in visible_fields_for("coqui_xtts")}
    coqui_fields = {field.key: field for field in fields_for("coqui_xtts")}
    _assert("device" in coqui_visible_keys, "coqui XTTS device should be visible")
    _assert("language" not in coqui_visible_keys and "model_name" not in coqui_visible_keys, "coqui XTTS model/language should be fixed and hidden")
    _assert(defaults["coqui_xtts"].get("device") == "auto", "coqui XTTS default device mismatch")
    _assert(defaults["coqui_xtts"].get("speaker_wav_path") == "", "coqui XTTS default sample should be empty")
    _assert(defaults["coqui_xtts"].get("speaker") == "Anna", "coqui XTTS default builtin speaker mismatch")
    _assert("Anna" in coqui_fields["speaker"].options, "coqui XTTS Anna speaker should be selectable")
    _assert(defaults["coqui_xtts"].get("temperature") == 0.1, "coqui XTTS default temperature mismatch")
    _assert(defaults["coqui_xtts"].get("length_penalty") == 1.0, "coqui XTTS default length penalty mismatch")
    _assert(defaults["coqui_xtts"].get("repetition_penalty") == 9.0, "coqui XTTS default repetition penalty mismatch")
    _assert(defaults["coqui_xtts"].get("top_k") == 100, "coqui XTTS default top_k mismatch")
    _assert(defaults["coqui_xtts"].get("top_p") == 1.0, "coqui XTTS default top_p mismatch")
    _assert(defaults["coqui_xtts"].get("builtin_voice_speed") == 1.6, "coqui XTTS builtin speed default mismatch")
    _assert(defaults["coqui_xtts"].get("voice_sample_speed") == 1.3, "coqui XTTS sample speed default mismatch")
    _assert("speed" not in coqui_visible_keys, "coqui XTTS legacy speed should not be visible")
    _assert(defaults["coqui_xtts"].get("xtts_trim_trailing_silence") is True, "coqui XTTS trailing silence trim should be enabled by default")
    _assert("xtts_trim_trailing_silence" in coqui_visible_keys, "coqui XTTS trailing silence trim should be visible")
    _assert(coqui_fields["builtin_voice_speed"].minimum == 0.5 and coqui_fields["builtin_voice_speed"].maximum == 2.0, "coqui XTTS builtin speed range mismatch")
    _assert(coqui_fields["voice_sample_speed"].minimum == 0.5 and coqui_fields["voice_sample_speed"].maximum == 2.0, "coqui XTTS sample speed range mismatch")
    _assert(coqui_fields["temperature"].minimum == 0.05 and coqui_fields["temperature"].maximum == 1.5, "coqui XTTS temperature range mismatch")
    _assert(coqui_fields["top_p"].minimum == 0.05 and coqui_fields["top_p"].maximum == 1.0, "coqui XTTS top_p range mismatch")
    _assert(defaults["coqui_xtts"].get("whisper_qc_enabled") is True, "coqui XTTS Whisper QC should be enabled by default")
    _assert(defaults["coqui_xtts"].get("whisper_qc_retry_attempts") == 4, "coqui XTTS Whisper QC retry default mismatch")
    _assert(defaults["coqui_xtts"].get("whisper_qc_model") == "small", "coqui XTTS Whisper QC model default mismatch")
    _assert(defaults["coqui_xtts"].get("whisper_qc_min_similarity") == 0.93, "coqui XTTS Whisper QC similarity default mismatch")
    supertonic_fields = {field.key: field for field in fields_for("supertonic")}
    supertonic_visible_keys = {field.key for field in visible_fields_for("supertonic")}
    _assert(defaults["supertonic"].get("voice") == "M5", "Supertonic default voice mismatch")
    _assert(defaults["supertonic"].get("speed") == 1.05, "Supertonic default speed mismatch")
    _assert(defaults["supertonic"].get("total_steps") == 12, "Supertonic default total steps mismatch")
    _assert(defaults["supertonic"].get("max_chunk_length") == 360, "Supertonic default chunk length mismatch")
    _assert(defaults["supertonic"].get("supertonic_trim_edges") is True, "Supertonic silence edge trim should be enabled by default")
    _assert(defaults["supertonic"].get("open_workspace_on_finish") is False, "Supertonic should not open workspace by default")
    _assert(defaults["supertonic"].get("save_quality_report") is False, "Supertonic quality report should be disabled by default")
    _assert(defaults["supertonic"].get("save_run_reports") is False, "Supertonic run reports should be disabled by default")
    _assert(supertonic_fields["voice"].field_type == "choice", "Supertonic voice should be a dropdown")
    _assert("M1" in supertonic_fields["voice"].options and "F5" in supertonic_fields["voice"].options, "Supertonic voice options mismatch")
    _assert(supertonic_fields["speed"].minimum == 0.5 and supertonic_fields["speed"].maximum == 2.0, "Supertonic speed range mismatch")
    _assert(supertonic_fields["total_steps"].minimum == 2 and supertonic_fields["total_steps"].maximum == 12, "Supertonic steps range mismatch")
    _assert(supertonic_fields["max_chunk_length"].label == "Dlugosc fragmentu tekstu", "Supertonic chunk length label should mention text")
    _assert("supertonic_trim_edges" in supertonic_visible_keys, "Supertonic silence edge trim should be visible")
    _assert(supertonic_fields["supertonic_trim_edges"].label == "Wycinanie ciszy", "Supertonic trim label should be user friendly")
    _assert("silence_duration" not in supertonic_visible_keys, "Supertonic silence duration should stay hidden")
    _assert(defaults["supertonic"].get("whisper_qc_enabled") is True, "Supertonic Whisper QC should be enabled by default")
    _assert(defaults["supertonic"].get("whisper_qc_retry_attempts") == 5, "Supertonic Whisper QC retry default mismatch")
    _assert(defaults["supertonic"].get("whisper_qc_model") == "small", "Supertonic Whisper QC model default mismatch")
    _assert(defaults["supertonic"].get("whisper_qc_device") == "cpu", "Supertonic Whisper QC device default mismatch")
    _assert(defaults["supertonic"].get("whisper_qc_min_similarity") == 0.95, "Supertonic Whisper QC similarity default mismatch")
    messages.append("config validation: OK")

    for engine_id in ("edge", "openai"):
        field_keys = {field.key for field in fields_for(engine_id)}
        _assert("whisper_qc_enabled" in field_keys, f"{engine_id} missing Whisper QC toggle")
        _assert("whisper_qc_model" in field_keys, f"{engine_id} missing Whisper QC model")
        whisper_field = next(field for field in fields_for(engine_id) if field.key == "whisper_qc_model")
        _assert(whisper_field.field_type == "choice", f"{engine_id} Whisper QC model should be a dropdown")
        _assert("small" in whisper_field.options and "large-v3" in whisper_field.options and "turbo" in whisper_field.options, f"{engine_id} Whisper QC model options missing")
        _assert(not any(option.endswith(".en") or option.startswith("distil-") for option in whisper_field.options), f"{engine_id} Whisper QC should only list multilingual Polish-capable models")
        _assert(whisper_field.label == "Model", f"{engine_id} Whisper QC model label should be concise")
        _assert(whisper_field.tooltip == "Do wyboru rozne modele dla kontroli mowy.", f"{engine_id} Whisper QC tooltip should be concise")
    for engine_id in ("chatterbox", "omnivoice", "piper", "coqui_xtts", "supertonic"):
        field_keys = {field.key for field in fields_for(engine_id)}
        _assert("whisper_qc_enabled" in field_keys, f"{engine_id} missing Whisper QC toggle")
        _assert("whisper_qc_model" in field_keys, f"{engine_id} missing Whisper QC model")
        whisper_field = next(field for field in fields_for(engine_id) if field.key == "whisper_qc_model")
        _assert(whisper_field.field_type == "choice", f"{engine_id} Whisper QC model should be a dropdown")
        _assert("small" in whisper_field.options and "large-v3" in whisper_field.options and "turbo" in whisper_field.options, f"{engine_id} Whisper QC model options missing")
        _assert(not any(option.endswith(".en") or option.startswith("distil-") for option in whisper_field.options), f"{engine_id} Whisper QC should only list multilingual Polish-capable models")
        _assert(whisper_field.label == "Model", f"{engine_id} Whisper QC model label should be concise")
        _assert(whisper_field.tooltip == "Do wyboru rozne modele dla kontroli mowy.", f"{engine_id} Whisper QC tooltip should be concise")
    for engine_id in ("chatterbox", "omnivoice", "coqui_xtts"):
        device_field = next(field for field in fields_for(engine_id) if field.key == "device")
        _assert(device_field.field_type == "choice", f"{engine_id} device should be a dropdown")
        _assert(device_field.options == ("auto", "cpu"), f"{engine_id} base device options should stay safe before GPU detection")
    messages.append("Whisper QC config fields: OK")

    _assert(normalize_for_whisper_qc("No już, silos!") == "no juz silos", "Whisper QC normalize mismatch")
    _assert(text_similarity("mechanicy naprzod", "mechanicy naprzod") == 1.0, "Whisper QC exact similarity mismatch")
    good_whisper = score_whisper_transcript("mechanicy naprzod", "mechanicy naprzod", 0.70)
    bad_whisper = score_whisper_transcript("mechanicy naprzod", "mekanaci nabrzut", 0.70)
    _assert(good_whisper.score == 0 and not good_whisper.warnings, "Whisper QC exact match should not warn")
    _assert(bad_whisper.score > good_whisper.score and "whisper niezgodny" in bad_whisper.warnings, "Whisper QC mismatch should warn")
    _assert("faster-whisper" in faster_whisper_missing_message(), "Whisper QC missing dependency message mismatch")
    quality_summary = _quality_controls_summary(
        "edge",
        {"audio_qc_enabled": True, "audio_qc_retry_attempts": "3", "whisper_qc_enabled": True, "whisper_qc_retry_attempts": "2", "whisper_qc_model": "small"},
        ffmpeg_present=True,
    )
    _assert(quality_summary["audio_qc_retry_attempts"] == 3, "quality summary retry attempts mismatch")
    _assert(quality_summary["whisper_qc_enabled"] is True, "quality summary Whisper toggle mismatch")
    _assert(quality_summary["whisper_qc_retry_attempts"] == 2, "quality summary Whisper retry attempts mismatch")
    _assert(quality_summary["whisper_qc_model"] == "small", "quality summary Whisper model mismatch")
    _assert(
        _quality_chain_label(quality_summary) == "TTS -> Przytnij i wygladz brzegi -> Audio QC x3 -> Whisper QC x2 -> final",
        "quality chain label mismatch",
    )
    no_qc_summary = _quality_controls_summary("edge", {"audio_qc_enabled": False, "whisper_qc_enabled": False}, ffmpeg_present=True)
    _assert(
        _quality_chain_label(no_qc_summary) == "TTS -> Przytnij i wygladz brzegi -> final",
        "edge tuning should stay in chain when QC is disabled",
    )
    raw_edge_summary = _quality_controls_summary("edge", {"audio_qc_enabled": False, "whisper_qc_enabled": False, "edge_apply_segment_fade": False}, ffmpeg_present=True)
    _assert(_quality_chain_label(raw_edge_summary) == "TTS -> final", "disabled edge tuning chain label mismatch")
    omnivoice_summary = _quality_controls_summary("omnivoice", {"omnivoice_trim_edges": True, "whisper_qc_enabled": True, "whisper_qc_retry_attempts": "2"}, ffmpeg_present=True)
    _assert(_quality_chain_label(omnivoice_summary) == "TTS -> Wycinanie ciszy na brzegach -> Whisper QC x2 -> final", "omnivoice quality chain label mismatch")
    piper_summary = _quality_controls_summary("piper", {"whisper_qc_enabled": False}, ffmpeg_present=True)
    _assert(_quality_chain_label(piper_summary) == "TTS -> final", "piper quality chain label mismatch")
    piper_whisper_summary = _quality_controls_summary("piper", {"whisper_qc_enabled": True, "whisper_qc_retry_attempts": "4"}, ffmpeg_present=True)
    _assert(_quality_chain_label(piper_whisper_summary) == "TTS -> Whisper QC x4 -> final", "piper punctuation Whisper QC chain label mismatch")
    _assert(piper_whisper_summary["whisper_qc_retry_attempts"] == 4, "piper punctuation Whisper QC should advertise retry attempts")
    supertonic_whisper_summary = _quality_controls_summary("supertonic", {"whisper_qc_enabled": True, "whisper_qc_retry_attempts": "5"}, ffmpeg_present=True)
    _assert(_quality_chain_label(supertonic_whisper_summary) == "TTS -> Whisper QC x5 -> final", "Supertonic Whisper QC chain label mismatch")
    coqui_summary = _quality_controls_summary("coqui_xtts", {"whisper_qc_enabled": True, "whisper_qc_retry_attempts": "3"}, ffmpeg_present=True)
    _assert(_quality_chain_label(coqui_summary) == "TTS -> Wycinanie koncowej ciszy -> Whisper QC x3 -> final", "coqui XTTS quality chain label mismatch")
    raw_coqui_summary = _quality_controls_summary("coqui_xtts", {"xtts_trim_trailing_silence": False, "whisper_qc_enabled": False}, ffmpeg_present=True)
    _assert(_quality_chain_label(raw_coqui_summary) == "TTS -> final", "disabled coqui XTTS tuning chain label mismatch")
    builtin_retry_message = _format_builtin_qc_retry_message(
        "Edge",
        3,
        10,
        2,
        4,
        35,
        ("glosny koniec", "whisper niezgodny"),
    )
    _assert(
        builtin_retry_message == "Edge QC: segment 3/10, odrzucono probe 1/4, kara QC 35, glosny koniec, whisper niezgodny; ponawiam 2/4",
        f"built-in QC retry message mismatch: {builtin_retry_message}",
    )
    builtin_whisper_retry_message = _format_builtin_qc_retry_message(
        "Edge Whisper QC",
        3,
        10,
        3,
        4,
        49,
        ("whisper niezgodny",),
    )
    _assert(
        builtin_whisper_retry_message == "Edge Whisper QC: segment 3/10, odrzucono probe 2/4, kara QC 49, whisper niezgodny; ponawiam 3/4",
        f"built-in Whisper QC retry message mismatch: {builtin_whisper_retry_message}",
    )
    local_piper_retry_message = _local_worker_retry_progress_message(
        "piper",
        216,
        "piper: segment 12, mowa proba 2/5, wariant=kropka, qc=45, ostrzezenia=whisper niezgodny",
    )
    _assert(
        local_piper_retry_message == "Whisper QC - segment 12/216, proba 2/5, score: 45",
        f"local Piper retry UI message mismatch: {local_piper_retry_message}",
    )
    builtin_short_retry_message = _format_short_qc_retry_message(
        "Edge",
        "kontrola mowy",
        28,
        40,
        2,
        5,
        score=15,
    )
    _assert(
        builtin_short_retry_message == "Whisper QC - segment 28/40, proba 2/5, score: 15",
        f"built-in short retry UI message mismatch: {builtin_short_retry_message}",
    )
    builtin_selected_message = _format_builtin_qc_selected_message(
        "Edge",
        3,
        10,
        2,
        4,
        12,
        ("lekko za dlugi",),
    )
    _assert(
        builtin_selected_message == "Edge QC: segment 3/10, wybrano probe 2/4, kara QC 12, lekko za dlugi",
        f"built-in QC selected message mismatch: {builtin_selected_message}",
    )
    dot_retry_variants = _edge_retry_text_variants("Test.", 5)
    plain_retry_variants = _edge_retry_text_variants("Test", 5)
    question_retry_variants = _edge_retry_text_variants("Test?", 5)
    ellipsis_retry_variants = _edge_retry_text_variants("Test...", 5)
    _assert(dot_retry_variants == ["Test.", "Test.", "Test,", "Test!", "Test?"], f"Edge dot retry variants should keep original and then controlled punctuation: {dot_retry_variants}")
    _assert(plain_retry_variants == ["Test", "Test.", "Test,", "Test!", "Test?"], f"Edge plain retry variants should add controlled punctuation: {plain_retry_variants}")
    _assert(question_retry_variants == ["Test?", "Test.", "Test,", "Test!", "Test?"], f"Edge question retry variants should strip punctuation before adding controlled suffixes: {question_retry_variants}")
    _assert(ellipsis_retry_variants == ["Test...", "Test.", "Test,", "Test!", "Test?"], f"Edge ellipsis retry variants should normalize trailing punctuation for retries: {ellipsis_retry_variants}")
    _assert(_edge_retry_text("Test?", 2) == "Test.", "Edge retry 1 should normalize existing punctuation to dot")
    _assert(_edge_retry_text("Test?", 5) == "Test?", "Edge retry 4 should use question mark after normalized base")
    worker_template = _load_module(app_dir / "app" / "engines" / "worker_templates" / "local_tts_worker.py", "_lektorai_worker_template_self_test")
    piper_retry_variants = worker_template.retry_text_variants("piper", "Test?", 5)
    _assert(piper_retry_variants == ["Test?", "Test.", "Test,", "Test!", "Test?"], f"Piper retry variants should match Edge punctuation strategy: {piper_retry_variants}")
    supertonic_retry_variants = worker_template.retry_text_variants("supertonic", "Test?", 5)
    _assert(supertonic_retry_variants == ["Test?", "Test.", "Test,", "Test!", "Test?"], f"Supertonic retry variants should match Edge punctuation strategy: {supertonic_retry_variants}")
    messages.append("Whisper QC scoring: OK")

    srt_path = app_dir / "_self_test_sample.srt"
    srt_path.write_text(
        "34\n00:00:01,000 --> 00:00:02,000\n<i>Test</i>\n\n"
        "40\n00:00:03,999 --> 00:00:04,250\nDrugi\n\n",
        encoding="utf-8",
    )
    try:
        segments = load_srt(srt_path)
        _assert(len(segments) == 2, "SRT parser count mismatch")
        _assert([segment.index for segment in segments] == [34, 40], "SRT parser should preserve original indexes")
        _assert(segments[0].start_ms == 1000 and segments[0].end_ms == 2000, "SRT parser time mismatch")
        _assert(segments[1].start_ms == 3999 and segments[1].end_ms == 4250, "SRT parser millisecond mismatch")
        save_srt(saved_srt_path, segments)
        _assert(saved_srt_path.read_text(encoding="utf-8").endswith("\n"), "saved SRT should end with newline")
    finally:
        try:
            srt_path.unlink()
        except OSError:
            pass
        try:
            saved_srt_path.unlink()
        except OSError:
            pass
    messages.append("SRT parser: OK")

    manifest_probe_dir = app_dir / "_self_test_manifest"
    manifest_path = manifest_probe_dir / "segmenty.csv"
    try:
        duplicate_start_segments = [
            SubtitleSegment(1, 1000, 1500, "Pierwszy"),
            SubtitleSegment(2, 1000, 1800, "Drugi"),
        ]
        generated_audio = [
            (1000, manifest_probe_dir / "audio_1.wav"),
            (1000, manifest_probe_dir / "audio_2.wav"),
        ]
        write_segments_manifest(manifest_path, duplicate_start_segments, generated_audio)
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        _assert("audio_1.wav" in manifest_lines[1], "manifest first duplicate start audio mismatch")
        _assert("audio_2.wav" in manifest_lines[2], "manifest second duplicate start audio mismatch")
    finally:
        _cleanup_tree(manifest_probe_dir)
    messages.append("segments manifest: OK")

    worker_result_dir = app_dir / "_self_test_worker_result"
    try:
        worker_result_dir.mkdir(parents=True, exist_ok=True)
        missing_audio_path = worker_result_dir / "001.wav"
        requested_segments = [
            SegmentRequest(
                segment_id=1,
                text="Test",
                start_ms=1000,
                end_ms=2000,
                output_path=str(missing_audio_path),
            )
        ]
        input_segments = [SubtitleSegment(1, 1000, 2000, "Test")]
        worker_result = EngineResult(
            engine_id="chatterbox",
            job_id="self-test",
            ok=True,
            segments=[SegmentResult(segment_id=1, ok=True, output_path=str(missing_audio_path))],
        )
        try:
            _generated_audio_from_worker_result("chatterbox", requested_segments, input_segments, worker_result)
            raise AssertionError("worker result without audio file should fail")
        except RuntimeError as exc:
            _assert("nie utworzyl pliku audio" in str(exc), "worker missing audio error mismatch")
        duplicate_result = EngineResult(
            engine_id="chatterbox",
            job_id="self-test",
            ok=True,
            segments=[
                SegmentResult(segment_id=1, ok=True, output_path=str(missing_audio_path)),
                SegmentResult(segment_id=1, ok=True, output_path=str(missing_audio_path)),
            ],
        )
        try:
            _generated_audio_from_worker_result("chatterbox", requested_segments, input_segments, duplicate_result)
            raise AssertionError("worker duplicate segment result should fail")
        except RuntimeError as exc:
            _assert("zduplikowany segment" in str(exc), "worker duplicate segment error mismatch")
        extra_result = EngineResult(
            engine_id="chatterbox",
            job_id="self-test",
            ok=True,
            segments=[
                SegmentResult(segment_id=1, ok=True, output_path=str(missing_audio_path)),
                SegmentResult(segment_id=99, ok=True, output_path=str(missing_audio_path)),
            ],
        )
        try:
            _generated_audio_from_worker_result("chatterbox", requested_segments, input_segments, extra_result)
            raise AssertionError("worker extra segment result should fail")
        except RuntimeError as exc:
            _assert("niezamowiony segment" in str(exc), "worker extra segment error mismatch")
        wrong_audio_path = worker_result_dir / "999.wav"
        wrong_audio_path.write_text("fake wav", encoding="utf-8")
        wrong_path_result = EngineResult(
            engine_id="chatterbox",
            job_id="self-test",
            ok=True,
            segments=[SegmentResult(segment_id=1, ok=True, output_path=str(wrong_audio_path))],
        )
        try:
            _generated_audio_from_worker_result("chatterbox", requested_segments, input_segments, wrong_path_result)
            raise AssertionError("worker wrong output path should fail")
        except RuntimeError as exc:
            _assert("inna sciezke audio" in str(exc), "worker wrong output path error mismatch")
        invalid_diagnostics_result = EngineResult(
            engine_id="chatterbox",
            job_id="self-test",
            ok=True,
            segments=[
                SegmentResult(
                    segment_id=1,
                    ok=True,
                    output_path=str(missing_audio_path),
                    attempts=2,
                    selected_attempt=5,
                    retries=1,
                )
            ],
        )
        try:
            _generated_audio_from_worker_result("chatterbox", requested_segments, input_segments, invalid_diagnostics_result)
            raise AssertionError("worker invalid retry diagnostics should fail")
        except RuntimeError as exc:
            _assert("diagnostyka retry" in str(exc), "worker invalid retry diagnostics error mismatch")
        warning_summary = _local_worker_qc_warning_summary(
            EngineResult(
                engine_id="chatterbox",
                job_id="self-test",
                ok=True,
                segments=[
                    SegmentResult(segment_id=1, ok=True, qc_warnings=("glosny koniec", "clipping")),
                    SegmentResult(segment_id=2, ok=True, qc_warnings=("glosny koniec",)),
                ],
            )
        )
        _assert(warning_summary == "glosny koniec: 2, clipping: 1", f"worker QC warning summary mismatch: {warning_summary}")
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "Fetching 6 files") is None,
            "worker progress should not classify generic Hugging Face fetch lines as downloads",
        )
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "chatterbox: pobieranie modelu t3=v3 na cuda") == "Model TTS: pobieranie modelu - prosze czekac",
            "worker progress should classify explicit TTS model download",
        )
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "chatterbox: sprawdzanie modelu t3=v3 na cuda") == "Model TTS: sprawdzanie obecnosci modelu",
            "worker progress should classify explicit TTS model check",
        )
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "chatterbox: model w cache t3=v3") == "Model TTS: model w cache",
            "worker progress should classify explicit TTS model cache hit",
        )
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "chatterbox: ladowanie modelu t3=v3 na cuda") == "Model TTS: ladowanie modelu - prosze czekac",
            "worker progress should classify model loading",
        )
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "whisper qc: pobieranie modelu large-v3-turbo na cuda") == "Whisper QC: pobieranie modelu - prosze czekac",
            "worker progress should classify Whisper QC model download separately",
        )
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "whisper qc: model w cache large-v3-turbo") == "Whisper QC: model w cache",
            "worker progress should classify Whisper QC model cache hit",
        )
        _assert(
            _model_activity_message_for_worker_line("chatterbox", "segment 1 OK") is None,
            "worker progress should ignore normal segment logs for model activity",
        )
        captured_progress: list[str] = []
        duplicate_counter = {"done": 0, "model_activity": "", "model_activity_seen": set()}
        for worker_line in (
            "piper: sprawdzanie modelu pl_PL-gosia-medium",
            "piper: model w cache pl_PL-gosia-medium",
            "piper: segment 1 OK",
            "piper: sprawdzanie modelu pl_PL-gosia-medium",
            "piper: model w cache pl_PL-gosia-medium",
            "piper: segment 2 OK",
        ):
            _local_worker_progress("piper", 2, duplicate_counter, captured_progress.append, worker_line)
        _assert(
            captured_progress
            == [
                "Model TTS: sprawdzanie obecnosci modelu",
                "Model TTS: model w cache",
                "Segment 1/2",
                "Segment 2/2",
            ],
            f"worker progress should suppress repeated model activity logs: {captured_progress}",
        )
        temp_dir = worker_result_dir / "temp"
        temp_dir.mkdir()
        (temp_dir / "request.json").write_text("{}\n", encoding="utf-8")
        try:
            _validated_worker_generated_audio("chatterbox", requested_segments, input_segments, worker_result, temp_dir)
            raise AssertionError("validated worker result without audio file should fail")
        except RuntimeError:
            _assert(temp_dir.exists(), "failed worker validation should keep temp diagnostics")
        missing_audio_path.write_text("fake wav", encoding="utf-8")
        generated = _validated_worker_generated_audio("chatterbox", requested_segments, input_segments, worker_result, temp_dir)
        _assert(generated == [(1000, missing_audio_path)], "validated worker generated audio mismatch")
        _assert(not temp_dir.exists(), "successful worker validation should cleanup temp diagnostics")
    finally:
        _cleanup_tree(worker_result_dir)
    messages.append("worker result validation: OK")

    protocol_json_dir = app_dir / "_self_test_protocol_json"
    try:
        request_path = protocol_json_dir / "request.json"
        result_path = protocol_json_dir / "result.json"
        write_request(
            request_path,
            EngineRequest(
                engine_id="chatterbox",
                source_name="Film.mkv",
                job_id="self-test",
                segments=[SegmentRequest(1, "Test", 1000, 2000, str(protocol_json_dir / "001.wav"))],
            ),
        )
        write_result(
            result_path,
            EngineResult(
                "chatterbox",
                "self-test",
                True,
                [
                    SegmentResult(
                        segment_id=1,
                        ok=True,
                        attempts=2,
                        selected_attempt=2,
                        qc_score=15,
                        qc_warnings=("glosny koniec",),
                        attempt_details=(
                            {"attempt": 1, "final_score": 30, "selected": False},
                            {"attempt": 2, "final_score": 15, "selected": True},
                        ),
                    )
                ],
            ),
        )
        _assert(request_path.read_text(encoding="utf-8").endswith("\n"), "request JSON should end with newline")
        _assert(result_path.read_text(encoding="utf-8").endswith("\n"), "result JSON should end with newline")
        diagnostic_result = read_result(result_path)
        _assert(diagnostic_result.segments[0].attempts == 2, "worker result attempts JSON mismatch")
        _assert(diagnostic_result.segments[0].selected_attempt == 2, "worker result selected attempt JSON mismatch")
        _assert(diagnostic_result.segments[0].qc_warnings == ("glosny koniec",), "worker result QC warnings JSON mismatch")
        _assert(diagnostic_result.segments[0].attempt_details[1]["selected"] is True, "worker result attempt details JSON mismatch")
        generated_audio_path = protocol_json_dir / "001.wav"
        generated_audio_path.write_text("fake wav", encoding="utf-8")
        local_analysis = _local_worker_segment_analysis(
            [SubtitleSegment(1, 1000, 2000, "Test")],
            [(1000, generated_audio_path)],
            diagnostic_result,
        )
        _assert(local_analysis[0]["final_score"] == 15, "local worker analysis score mismatch")
        _assert(local_analysis[0]["attempt_details"][1]["final_score"] == 15, "local worker analysis attempt score mismatch")
        settings_snapshot = _sanitize_settings_snapshot({"api_key": "secret", "voice": "marek", "token": ""})
        _assert(settings_snapshot["api_key"] == "***" and settings_snapshot["token"] == "", "settings snapshot should mask secrets")
        analysis_payload = _build_run_analysis(
            source_path=Path("Film.mkv"),
            input_subtitle_path=Path("Film.srt"),
            engine_id="chatterbox",
            output_stem="260512_144531_Film_CTB",
            run_started_at=datetime(2026, 5, 12, 14, 45, 31),
            segments=[SubtitleSegment(1, 1000, 2000, "Test")],
            segment_analysis=local_analysis,
            config={"api_key": "secret", "cfg_weight": 1.9},
            dictionary={"vox": "woks"},
            quality_controls={"chain_label": "TTS -> Whisper QC x2 -> final"},
            generation_seconds=12.3,
            pipeline_seconds=15.4,
            aac_bitrate="384k",
            lektor_lufs=-14,
            lektor_weight=2.3,
            background_lufs=-18,
            background_weight=1.6,
            lektor_delay_ms=500,
            create_stereo_for_surround=True,
            diagnostics={"save_run_reports": True},
            source_duration=3600.0,
            source_audio_streams=[],
            lektor_before_diagnostics=None,
            lektor_after_diagnostics=None,
            lektor_encoded_duration=None,
            lektor_encoded_audio_streams=[],
        )
        _assert(analysis_payload["schema"] == "lektorai.run_analysis.v1", "run analysis schema mismatch")
        _assert(analysis_payload["report_type"] == "analysis", "run analysis report type mismatch")
        _assert(analysis_payload["run_id"] == "260512_144531_Film_CTB", "run analysis id mismatch")
        _assert(analysis_payload["run_timestamp"] == "2026-05-12T14:45:31", "run analysis timestamp mismatch")
        _assert(analysis_payload["source_filename"] == "Film.mkv", "run analysis source filename mismatch")
        _assert(analysis_payload["tts_engine_short"] == "CTB", "run analysis engine short mismatch")
        _assert("Nizszy score" in analysis_payload["llm_analysis_hint"], "run analysis LLM hint mismatch")
        _assert(analysis_payload["engine"]["settings"]["api_key"] == "***", "run analysis should mask api key")
        _assert(analysis_payload["aggregates"]["segments_with_retry"] == 1, "run analysis retry aggregate mismatch")
        _assert(analysis_payload["segments"][0]["attempt_details"][1]["selected"] is True, "run analysis attempts mismatch")
        report_cleanup_dir = app_dir / "_self_test_report_cleanup"
        try:
            for keep_quality_report, keep_technical_reports in ((False, True), (True, False)):
                _cleanup_tree(report_cleanup_dir)
                lektor_dir = report_cleanup_dir / "run"
                segments_dir = lektor_dir / "segments"
                segments_dir.mkdir(parents=True, exist_ok=True)
                subtitle = lektor_dir / "run.srt"
                input_subtitle = lektor_dir / "input.srt"
                manifest = lektor_dir / "segmenty.csv"
                skipped = lektor_dir / "skipped_segments.csv"
                audio_qc = lektor_dir / "audio_qc.csv"
                analysis = lektor_dir / "run_analysis.json"
                summary = lektor_dir / "run_summary.json"
                for path in (subtitle, input_subtitle, manifest, skipped, audio_qc, analysis, summary):
                    path.write_text("x", encoding="utf-8")
                _cleanup_successful_video_run(
                    diagnostics={
                        "save_processed_subtitles": False,
                        "save_run_reports": keep_technical_reports,
                        "save_quality_report": keep_quality_report,
                        "save_lektor_segments": False,
                        "save_lektor_track_before_normalization": False,
                        "save_lektor_track_after_normalization": False,
                        "save_audio_mix_steps": False,
                    },
                    lektor_dir=lektor_dir,
                    segments_dir=segments_dir,
                    subtitle_path=subtitle,
                    input_subtitle_path=input_subtitle,
                    manifest_path=manifest,
                    skipped_manifest_path=skipped,
                    audio_qc_path=audio_qc,
                    analysis_path=analysis,
                    summary_path=summary,
                    lektor_before_normalization_path=None,
                    lektor_after_normalization_path=None,
                    lektor_m4a_path=None,
                    audio_mix_stage_files={},
                )
                _assert(analysis.exists() is keep_quality_report, "quality report should only follow its own diagnostic option")
                _assert(summary.exists() is keep_technical_reports, "summary report should follow technical reports option")
                _assert(manifest.exists() is keep_technical_reports, "manifest should follow technical reports option")
                _assert(audio_qc.exists() is keep_technical_reports, "Audio QC report should follow technical reports option")
        finally:
            _cleanup_tree(report_cleanup_dir)
        _assert(not _should_run_final_audio_qc({}, ffmpeg_present=True), "final Audio QC should be skipped when diagnostics are disabled")
        _assert(_should_run_final_audio_qc({"save_run_reports": True}, ffmpeg_present=True), "final Audio QC should run for technical reports")
        _assert(_should_run_final_audio_qc({"save_quality_report": True}, ffmpeg_present=True), "final Audio QC should run for quality report")
        _assert(not _should_run_final_audio_qc({"save_run_reports": True}, ffmpeg_present=False), "final Audio QC should still require ffmpeg")
        _assert(_should_encode_standalone_lektor_audio(Path("dialog.srt"), {}), "audio-only jobs should encode standalone lektor track")
        _assert(not _should_encode_standalone_lektor_audio(Path("film.mkv"), {}), "video jobs should skip standalone M4A when mix steps are not saved")
        _assert(_should_encode_standalone_lektor_audio(Path("film.mkv"), {"save_audio_mix_steps": True}), "video diagnostics should keep standalone M4A")
        timings = PipelineTimings()
        timings.add("TTS", 1.234)
        timings.add("TTS", 0.766)
        timings.add("Miks MKV", 2.0)
        timing_lines = timings.summary_lines()
        _assert(timings.as_dict()["TTS"] == 2.0, "pipeline timings should accumulate repeated stages")
        _assert(timing_lines[0] == "Czasy etapow:", "pipeline timings should start with a compact header")
        _assert(any("TTS: 2s" in line for line in timing_lines), "pipeline timings should format short seconds")
        result_path.write_text(
            json.dumps(
                {
                    "engine_id": "chatterbox",
                    "job_id": "whisper",
                    "ok": True,
                    "segments": [
                        {
                            "segment_id": 1,
                            "ok": True,
                            "whisper_text": "mechanicy naprzod",
                            "whisper_similarity": 0.91,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        whisper_result = read_result(result_path)
        _assert(whisper_result.segments[0].whisper_text == "mechanicy naprzod", "worker Whisper text JSON mismatch")
        _assert(whisper_result.segments[0].whisper_similarity == 0.91, "worker Whisper similarity JSON mismatch")
        result_path.write_text(
            json.dumps({"engine_id": "chatterbox", "job_id": "self-test", "ok": True, "segments": {"bad": "shape"}}),
            encoding="utf-8",
        )
        malformed_result = read_result(result_path)
        _assert(malformed_result.segments == [], "malformed result segments should be ignored")
        result_path.write_text(
            json.dumps(
                {
                    "engine_id": "chatterbox",
                    "job_id": "future",
                    "ok": True,
                    "segments": [
                        {
                            "segment_id": 1,
                            "ok": True,
                            "output_path": str(protocol_json_dir / "001.wav"),
                            "future_qc_field": "ignored",
                        }
                    ],
                    "future_engine_field": "ignored",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        future_result = read_result(result_path)
        _assert(future_result.segments[0].segment_id == 1, "worker result should ignore unknown segment fields")
        request_with_dictionary = EngineRequest(
            engine_id="chatterbox",
            source_name="Film.mkv",
            job_id="self-test",
            segments=[],
            dictionary={"batman": "bat man"},
        )
        write_request(request_path, request_with_dictionary)
        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        _assert(request_data.get("dictionary") == {"batman": "bat man"}, "worker request dictionary should persist")
        built_request = _build_local_engine_request(
            engine_id="chatterbox",
            source_name="Film.mkv",
            job_id="self-test",
            segments=[SubtitleSegment(1, 1000, 2000, "woks")],
            segments_dir=protocol_json_dir,
            config={"seed": 12345},
            dictionary={"batman": "bat man"},
        )
        _assert(built_request.dictionary == {"batman": "bat man"}, "local worker request should include dictionary")
    finally:
        _cleanup_tree(protocol_json_dir)
    messages.append("worker protocol JSON: OK")

    worker_template_dir = app_dir / "_self_test_worker_template"
    try:
        worker_template_dir.mkdir(parents=True, exist_ok=True)
        worker_template = app_dir / "app" / "engines" / "worker_templates" / "local_tts_worker.py"
        worker_module = _load_module(worker_template, "_lektorai_worker_template_self_test")
        _assert(
            worker_module.normalize_for_whisper_qc("No już, silos!") == "no juz silos",
            "worker Whisper QC normalize mismatch",
        )
        _assert(
            worker_module.text_similarity("mechanicy naprzod", "mechanicy naprzod") == 1.0,
            "worker Whisper QC exact similarity mismatch",
        )
        _assert(
            worker_module.whisper_similarity_penalty(0.40, 0.70) > worker_module.whisper_similarity_penalty(0.68, 0.70),
            "worker Whisper QC penalty should punish larger mismatch",
        )
        _assert(
            worker_module.effective_retry_limits(
                "piper",
                {"audio_qc_enabled": True, "audio_qc_retry_attempts": 5, "whisper_qc_enabled": True, "whisper_qc_retry_attempts": 4},
            )
            == (1, 4),
            "deterministic Piper should keep Whisper QC attempts for punctuation variants",
        )
        _assert(
            worker_module.effective_retry_settings(
                "piper",
                {"audio_qc_enabled": True, "audio_qc_retry_attempts": 5, "whisper_qc_enabled": True, "whisper_qc_retry_attempts": 4},
            )
            == (1, 4, True, True),
            "deterministic Piper should keep Whisper QC enabled with punctuation variants",
        )
        _assert(
            worker_module.retry_text_variants("piper", "Ty", 5) == ["Ty", "Ty.", "Ty,", "Ty!", "Ty?"],
            "piper retry variants should add controlled punctuation",
        )
        _assert(
            worker_module.retry_text_for_attempt("piper", "Ty?", 2) == "Ty.",
            "piper retry variants should strip existing trailing punctuation before adding retry punctuation",
        )
        _assert(
            worker_module.retry_variant_label("piper", "Ty.") == "kropka",
            "piper retry variant label for dot mismatch",
        )
        _assert(
            worker_module.effective_retry_limits(
                "chatterbox",
                {"audio_qc_enabled": True, "audio_qc_retry_attempts": 5, "whisper_qc_enabled": True, "whisper_qc_retry_attempts": 4},
            )
            == (5, 4),
            "non-deterministic engines should keep configured QC retry limits",
        )
        _assert(
            worker_module.effective_retry_settings(
                "chatterbox",
                {"audio_qc_enabled": True, "audio_qc_retry_attempts": 5, "whisper_qc_enabled": True, "whisper_qc_retry_attempts": 4},
            )
            == (5, 4, True, True),
            "non-deterministic engines should keep QC flags enabled",
        )
        old_env = {name: os.environ.get(name) for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")}
        try:
            os.environ["HF_HOME"] = str(app_dir / "_outside_hf_cache")
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(app_dir / "_outside_hub_cache")
            os.environ["TRANSFORMERS_CACHE"] = str(app_dir / "_outside_transformers_cache")
            original_cwd = Path.cwd()
            os.chdir(worker_template_dir)
            worker_module.setup_cache("chatterbox")
            _assert(
                os.environ.get("HF_HOME") == str(app_dir / "_outside_hf_cache"),
                "worker should preserve global HF_HOME so auth tokens remain visible",
            )
            _assert(
                os.environ.get("HUGGINGFACE_HUB_CACHE") == str(worker_template_dir / "cache" / "hf"),
                "worker should force per-engine Hugging Face cache",
            )
            _assert(
                os.environ.get("TRANSFORMERS_CACHE") == str(worker_template_dir / "cache" / "transformers"),
                "worker should force per-engine Transformers cache",
            )
        finally:
            os.chdir(original_cwd)
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        worker_request = worker_template_dir / "request.json"
        worker_result = worker_template_dir / "result.json"
        worker_request.write_text(
            json.dumps({"engine_id": "chatterbox", "job_id": "empty", "settings": {}, "segments": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-B", str(worker_template), str(worker_request), str(worker_result)],
            cwd=str(worker_template_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        _assert(completed.returncode == 0, "worker template empty request failed")
        result_text = worker_result.read_text(encoding="utf-8")
        _assert(result_text.endswith("\n"), "worker template result should end with newline")
        result_data = json.loads(result_text)
        _assert(result_data.get("ok") is True and result_data.get("segments") == [], "worker template empty result mismatch")
        worker_error_request = worker_template_dir / "request_error.json"
        worker_error_result = worker_template_dir / "result_error.json"
        worker_error_request.write_text(
            json.dumps(
                {
                    "engine_id": "unsupported",
                    "job_id": "error",
                    "settings": {},
                    "segments": [
                        {
                            "segment_id": 7,
                            "text": "Test",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "output_path": str(worker_template_dir / "missing.wav"),
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        error_completed = subprocess.run(
            [sys.executable, "-B", str(worker_template), str(worker_error_request), str(worker_error_result)],
            cwd=str(worker_template_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        _assert(error_completed.returncode == 1, "worker template error run should exit with 1")
        error_data = json.loads(worker_error_result.read_text(encoding="utf-8"))
        error_segment = error_data["segments"][0]
        _assert(error_segment.get("segment_id") == 7 and error_segment.get("ok") is False, "worker template error segment mismatch")
        _assert("attempts" in error_segment and "selected_attempt" in error_segment, "worker template error diagnostics missing")
        _assert(isinstance(error_segment.get("qc_warnings"), list), "worker template error QC warnings should be a list")
        worker_bad_request = worker_template_dir / "request_bad.json"
        worker_bad_result = worker_template_dir / "result_bad.json"
        worker_bad_request.write_text(
            json.dumps({"engine_id": "chatterbox", "job_id": "bad", "settings": 1, "segments": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        bad_completed = subprocess.run(
            [sys.executable, "-B", str(worker_template), str(worker_bad_request), str(worker_bad_result)],
            cwd=str(worker_template_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        _assert(bad_completed.returncode == 1, "worker template bad request should exit with 1")
        bad_data = json.loads(worker_bad_result.read_text(encoding="utf-8"))
        _assert(bad_data.get("engine_id") == "chatterbox" and bad_data.get("job_id") == "bad", "worker top-level error should preserve ids")
        worker_template_text = worker_template.read_text(encoding="utf-8")
        _assert("_patch_chatterbox_watermarker" in worker_template_text, "worker template should patch broken Perth watermarker before Chatterbox init")
        _assert("PerthImplicitWatermarker" in worker_template_text, "worker template should mention Chatterbox watermarker patch")
        _assert("t3_model=t3_model" in worker_template_text, "worker template should pass Chatterbox multilingual T3 model")
        _assert("_CHATTERBOX_MODEL_KEY" in worker_template_text, "worker template should reload Chatterbox when T3 model changes")
        _assert('language_id="pl"' in worker_template_text, "worker template should hardcode Polish Chatterbox language")
        _assert("trim_leading_silence" in worker_template_text, "worker template should support Chatterbox leading silence trim")
        _assert("pre_roll_ms=50" in worker_template_text, "worker template should keep 50 ms before Chatterbox speech")
        _assert("consecutive_frames=3" in worker_template_text, "worker template should avoid trimming on a single noisy frame")
        _assert("synthesize_omnivoice" in worker_template_text, "worker template should support OmniVoice")
        _assert('language="pl"' in worker_template_text, "worker template should hardcode Polish OmniVoice language")
        _assert("reference_audio_path" in worker_template_text, "worker template should use OmniVoice reference audio")
        _assert("guidance_scale" in worker_template_text and "num_step" in worker_template_text, "worker template should pass OmniVoice generation controls")
        _assert("trim_omnivoice_silence_edges_np" in worker_template_text, "worker template should support automatic OmniVoice edge trimming")
        _assert("synthesize_piper" in worker_template_text, "worker template should support Piper")
        _assert("PiperVoice.load" in worker_template_text, "worker template should load Piper voices")
        _assert("download_voice" in worker_template_text, "worker template should download Piper voices lazily")
        _assert("synthesize_coqui_xtts" in worker_template_text, "worker template should support Coqui XTTS")
        _assert("tts_models/multilingual/multi-dataset/xtts_v2" in worker_template_text, "worker template should use XTTS-v2 model")
        _assert("COQUI_TOS_AGREED" in worker_template_text, "worker template should accept Coqui XTTS TOS non-interactively")
        _assert("trim_xtts_trailing_silence_np" in worker_template_text, "worker template should trim Coqui XTTS trailing silence")
        _assert("synthesize_supertonic" in worker_template_text, "worker template should support Supertonic")
        _assert("supertonic-3" in worker_template_text, "worker template should use Supertonic 3 model")
        _assert('"lang": "pl"' in worker_template_text, "worker template should hardcode Polish Supertonic language")
        _assert("supertonic_trim_edges" in worker_template_text, "worker template should support Supertonic silence edge trimming")
        _assert("trim_supertonic_silence_edges_np" in worker_template_text, "worker template should trim Supertonic silence on both edges")
        _assert("synthesize_vibevoice" not in worker_template_text, "archived VibeVoice should not be active in worker template")
        _assert("microsoft/VibeVoice-Realtime-0.5B" not in worker_template_text, "archived VibeVoice model should not be active in worker template")
        _assert("Sticzu/vibevoice-polish-voices" not in worker_template_text, "archived VibeVoice voices should not be active in worker template")
        _assert("vibevoice_trim_edges" not in worker_template_text, "archived VibeVoice trim option should not be active in worker template")
        _assert("synthesize_fish_speech" not in worker_template_text, "archived Fish Speech should not be active in worker template")
        _assert("fishaudio/s2-pro" not in worker_template_text, "archived Fish Speech model should not be active in worker template")
        _assert("ServeTTSRequest" not in worker_template_text, "archived Fish Speech inference engine should not be active in worker template")
        _assert("synthesize_f5_tts" not in worker_template_text, "archived F5-TTS should not be active in worker template")
        _assert('language="pl"' in worker_template_text, "worker template should hardcode Polish where model API supports it")
        _assert("pobieranie/ladowanie" not in worker_template_text, "worker template should not use ambiguous model activity logs")
        _assert("configure_worker_stdio()" in worker_template_text, "worker template should configure UTF-8 stdio")
        _assert('reconfigure(encoding="utf-8", errors="replace")' in worker_template_text, "worker template should replace unencodable stdout/stderr characters")
        _assert("PYTHONIOENCODING" in worker_template_text and "PYTHONUTF8" in worker_template_text, "worker template should set UTF-8 stdio environment")
        _assert("has_local_model_cache" in worker_template_text, "worker template should distinguish model download from loading")
        _assert("LEKTORAI_STT_FASTER_WHISPER_CACHE_DIR" in worker_template_text, "worker template should use STT faster-whisper cache")
        _assert("LEKTORAI_STT_FASTER_WHISPER_PACKAGES_DIRS" in worker_template_text, "worker template should load STT packages lazily for Whisper QC")
        _assert("has_whisper_model_cache" in worker_template_text, "worker template should check cache for the selected Whisper model")
        _assert("_AUTO_DEVICE" in worker_template_text and "return _AUTO_DEVICE" in worker_template_text, "worker template should keep auto device stable for whole worker run")
    finally:
        _cleanup_tree(worker_template_dir)
    messages.append("worker template protocol: OK")

    audio_qc_mismatch_dir = app_dir / "_self_test_audio_qc_mismatch"
    try:
        audio_qc_mismatch_dir.mkdir(parents=True, exist_ok=True)
        try:
            analyze_generated_segments(
                app_dir / "ffmpeg.exe",
                [],
                [SubtitleSegment(1, 1000, 2000, "Test")],
                audio_qc_mismatch_dir / "audio_qc.csv",
                audio_qc_mismatch_dir / "temp",
            )
            raise AssertionError("Audio QC segment count mismatch should fail")
        except RuntimeError as exc:
            _assert("liczba segmentow" in str(exc), "Audio QC segment count mismatch error mismatch")
    finally:
        _cleanup_tree(audio_qc_mismatch_dir)
    messages.append("audio QC validation: OK")

    edge_trim_dir = app_dir / "_self_test_edge_trim"
    try:
        edge_trim_dir.mkdir(parents=True, exist_ok=True)
        edge_input = edge_trim_dir / "input.wav"
        edge_manual_output = edge_trim_dir / "manual.wav"
        _write_test_wav(edge_input, [0] * 44100 + [8000] * 4410 + [0] * 88200)
        trim_fixed_and_fade_wav_edges(edge_input, edge_manual_output, trim_start_ms=200, trim_end_ms=900, fade_ms=12)
        input_frames = _wav_frame_count(edge_input)
        manual_frames = _wav_frame_count(edge_manual_output)
        expected_manual_frames = input_frames - int(44100 * 200 / 1000) - int(44100 * 900 / 1000)
        _assert(manual_frames == expected_manual_frames, "manual edge trim should remove fixed start/end")
        _assert(_wav_first_sample(edge_manual_output) == 0, "manual edge trim should not fade in or alter the first kept sample")
    finally:
        _cleanup_tree(edge_trim_dir)
    messages.append("edge tuning: OK")

    timeline_dir = app_dir / "_self_test_audio_timeline"
    try:
        timeline_dir.mkdir(parents=True, exist_ok=True)
        first_segment = timeline_dir / "first.wav"
        second_segment = timeline_dir / "second.wav"
        queued_track = timeline_dir / "queued.wav"
        _write_test_wav(first_segment, [1000] * SAMPLE_RATE, sample_rate=SAMPLE_RATE)
        _write_test_wav(second_segment, [2000] * (SAMPLE_RATE // 2), sample_rate=SAMPLE_RATE)
        stats = build_lektor_wav(
            app_dir / "ffmpeg.exe",
            [(0, first_segment), (500, second_segment)],
            queued_track,
            timeline_dir / "temp",
            minimum_duration_s=2.0,
        )
        _assert(stats.shifted_count == 1, "overlapping timeline segment should be queued")
        _assert(490 <= stats.max_shift_ms <= 510, f"timeline queue shift mismatch: {stats.max_shift_ms}")
        _assert(_wav_sample_rate(queued_track) == SAMPLE_RATE, "queued timeline should be built at 48 kHz")
        _assert(_wav_frame_count(queued_track) == SAMPLE_RATE * 2, "queued timeline should not trim segments and should pad to requested duration")
        _assert(_wav_first_sample(queued_track) == 0, "queued timeline should fade in segment edge to avoid clicks")
        _assert(SEGMENT_EDGE_FADE_MS <= 10, "timeline edge fade should stay short enough to preserve speech")
        shifted_segments = apply_lektor_delay_to_segments([(1000, first_segment), (25, second_segment)], 125)
        _assert(shifted_segments[0][0] == 1100 and shifted_segments[1][0] == 125, "lektor delay should shift timestamps by sanitized step")
        early_segments = apply_lektor_delay_to_segments([(100, first_segment)], -250)
        _assert(early_segments[0][0] == 100, "negative lektor delay should be sanitized to zero")
        try:
            build_lektor_wav(
                app_dir / "ffmpeg.exe",
                [(0, first_segment)],
                timeline_dir / "cancelled.wav",
                timeline_dir / "temp_cancelled",
                cancel_requested=lambda: True,
            )
            raise AssertionError("timeline build should stop when cancellation is requested")
        except RuntimeError as exc:
            _assert("Przerwano" in str(exc), "timeline cancellation error mismatch")
        _assert(not (timeline_dir / "temp_cancelled").exists(), "cancelled timeline should clean temp dir")
    finally:
        _cleanup_tree(timeline_dir)
    messages.append("audio timeline queue: OK")

    debug_cleanup_dir = app_dir / "_self_test_debug_cleanup"
    try:
        segments_cleanup_dir = debug_cleanup_dir / "segments"
        segments_cleanup_dir.mkdir(parents=True, exist_ok=True)
        (segments_cleanup_dir / "001.wav").write_text("segment", encoding="utf-8")
        before_cleanup = debug_cleanup_dir / "lektor_przed_normalizacja.wav"
        after_cleanup = debug_cleanup_dir / "lektor_po_normalizacji.wav"
        before_cleanup.write_text("before", encoding="utf-8")
        after_cleanup.write_text("after", encoding="utf-8")
        _cleanup_lektor_debug_files(
            {
                "save_lektor_segments": False,
                "save_lektor_track_before_normalization": False,
                "save_lektor_track_after_normalization": True,
            },
            debug_cleanup_dir,
            segments_cleanup_dir,
            before_cleanup,
            after_cleanup,
        )
        _assert(not segments_cleanup_dir.exists(), "disabled segment saving should remove segment dir")
        _assert(not before_cleanup.exists(), "disabled pre-normalization saving should remove track")
        _assert(after_cleanup.exists(), "enabled post-normalization saving should keep track")
    finally:
        _cleanup_tree(debug_cleanup_dir)

    source_path = app_dir / "Film.mkv"
    run_time = datetime(2026, 5, 12, 14, 45, 31)
    workspace = lektorai_workspace_for(source_path)
    _assert(safe_name("Zażółć gęślą jaźń") == "Zazolc_gesla_jazn", "safe name Polish transliteration mismatch")
    _assert(len(safe_name("a" * 200)) == 80, "safe name length limit mismatch")
    _assert(engine_short_code("chatterbox") == "CTB", "chatterbox short code mismatch")
    _assert(engine_short_code("omnivoice") == "OMV", "omnivoice short code mismatch")
    _assert(engine_short_code("coqui_xtts") == "XTTS", "coqui short code mismatch")
    _assert(engine_short_code("piper") == "PIP", "piper short code mismatch")
    _assert(engine_short_code("supertonic") == "SPT", "Supertonic short code mismatch")
    _assert(engine_short_code("edge") == "EDG", "edge short code mismatch")
    _assert(engine_short_code("openai") == "OAI", "openai short code mismatch")
    stem = next_output_stem(workspace, source_path, "edge", created_at=run_time)
    _assert(stem == "260512_144531_Film_EDG", f"output stem mismatch: {stem}")
    _assert(lektor_assets_dir(workspace, stem).name == "260512_144531_Film_EDG", "lektor dir mismatch")
    long_source_path = app_dir / ("Bardzo_dlugaaa_nazwa_filmu_" + ("x" * 120) + ".mkv")
    long_stem = next_output_stem(workspace, long_source_path, "chatterbox", created_at=run_time)
    _assert(long_stem.startswith("260512_144531_Bardzo_dlugaaa_nazwa_filmu_"), "long output stem prefix mismatch")
    _assert(long_stem.endswith("_CTB"), "long output stem engine code mismatch")
    _assert(len(long_stem) <= 80, "output stem should stay short")
    collision_workspace = app_dir / "_self_test_workspace_collision"
    try:
        collision_workspace.mkdir(parents=True, exist_ok=True)
        base_collision = "260512_144531_Film_EDG"
        (collision_workspace / f"{base_collision}.mp4").write_text("", encoding="utf-8")
        collision_stem = next_output_stem(collision_workspace, source_path, "edge", created_at=run_time)
        _assert(collision_stem == f"{base_collision}_2", f"mp4 collision stem mismatch: {collision_stem}")
        (collision_workspace / f"{base_collision}.mp4").unlink()
        (collision_workspace / f"{base_collision}.mkv").write_text("", encoding="utf-8")
        collision_stem = next_output_stem(collision_workspace, source_path, "edge", created_at=run_time)
        _assert(collision_stem == f"{base_collision}_2", f"mkv collision stem mismatch: {collision_stem}")
        (collision_workspace / f"{base_collision}_2.srt").write_text("", encoding="utf-8")
        collision_stem = next_output_stem(collision_workspace, source_path, "edge", created_at=run_time)
        _assert(collision_stem == f"{base_collision}_3", f"srt collision stem mismatch: {collision_stem}")
        (collision_workspace / f"{base_collision}_3").mkdir()
        collision_stem = next_output_stem(collision_workspace, source_path, "edge", created_at=run_time)
        _assert(collision_stem == f"{base_collision}_3", f"lektor dir should not rename stem: {collision_stem}")
    finally:
        _cleanup_tree(collision_workspace)
    messages.append("workspace naming: OK")

    failure_app_dir = app_dir / "_self_test_run_error_app"
    failure_source_dir = app_dir / "_self_test_run_error_media"
    failure_source = failure_source_dir / "Pusty.srt"
    try:
        failure_source_dir.mkdir(parents=True, exist_ok=True)
        failure_source.write_text("", encoding="utf-8")
        failure_paths = build_paths(failure_app_dir)
        try:
            run_tts_job(failure_source, "edge", failure_paths, EngineManager(failure_paths), lambda _msg: None)
            raise AssertionError("empty subtitle run should fail")
        except RuntimeError as exc:
            _assert("Brak tekstu do syntezy" in str(exc), "unexpected run error message")
        failure_engine_dir = failure_paths.engine_dir("edge")
        _assert((failure_engine_dir / "config.json").is_file(), "run should create engine config on first use")
        _assert((failure_engine_dir / "dictionary.json").is_file(), "run should create engine dictionary on first use")
        lektor_dirs = [path for path in (failure_source_dir / "LektorAI").glob("*_Pusty_EDG") if path.is_dir()]
        _assert(len(lektor_dirs) == 1, "failed run lektor dir naming mismatch")
        error_paths = list(lektor_dirs[0].glob("*_Pusty_EDG_error.json"))
        _assert(len(error_paths) == 1, "run error report naming mismatch")
        error_path = error_paths[0]
        error_data = json.loads(error_path.read_text(encoding="utf-8"))
        _assert(error_data.get("status") == "failed", "run_error.json status mismatch")
        _assert(error_data.get("run_id") == lektor_dirs[0].name, "run_error.json run id mismatch")
        _assert(error_data.get("report_type") == "error", "run_error.json report type mismatch")
        _assert(error_data.get("source_filename") == "Pusty.srt", "run_error.json source filename mismatch")
        _assert(error_data.get("tts_engine_short") == "EDG", "run_error.json engine short mismatch")
        _assert(error_data.get("engine_id") == "edge", "run_error.json engine mismatch")
        _assert(error_data.get("error_type") == "RuntimeError", "run_error.json error type mismatch")
        _assert(error_data.get("source_name") == "Pusty.srt", "run_error.json source mismatch")
        _assert("traceback" in error_data and "RuntimeError" in str(error_data.get("traceback")), "run_error.json should include traceback diagnostics")
    finally:
        _cleanup_tree(failure_app_dir)
        _cleanup_tree(failure_source_dir)
    messages.append("run error summary: OK")

    video_sidecar_source = app_dir / "_self_test_video.mkv"
    sidecar_path = app_dir / "_self_test_video.polish.srt"
    uppercase_sidecar_path = app_dir / "_self_test_video2.PL.SRT"
    generic_sidecar_video = app_dir / "_self_test_video3.mkv"
    generic_sidecar_path = app_dir / "_self_test_video3.srt"
    polish_sidecar_path = app_dir / "_self_test_video3.pl.srt"
    forced_sidecar_video = app_dir / "_self_test_video4.mkv"
    forced_sidecar_path = app_dir / "_self_test_video4.forced.pl.srt"
    full_polish_sidecar_path = app_dir / "_self_test_video4.pl.srt"
    video_sidecar_source.write_text("", encoding="utf-8")
    sidecar_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n", encoding="utf-8")
    uppercase_video_sidecar_source = app_dir / "_self_test_video2.mkv"
    uppercase_video_sidecar_source.write_text("", encoding="utf-8")
    uppercase_sidecar_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n", encoding="utf-8")
    generic_sidecar_video.write_text("", encoding="utf-8")
    generic_sidecar_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nGeneric\n", encoding="utf-8")
    polish_sidecar_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nPolski\n", encoding="utf-8")
    forced_sidecar_video.write_text("", encoding="utf-8")
    forced_sidecar_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nForced\n", encoding="utf-8")
    full_polish_sidecar_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nFull\n", encoding="utf-8")
    try:
        _assert(_find_sidecar_subtitles(video_sidecar_source) == sidecar_path, "sidecar lookup mismatch")
        _assert(_find_sidecar_subtitles(uppercase_video_sidecar_source) == uppercase_sidecar_path, "uppercase sidecar lookup mismatch")
        _assert(_find_sidecar_subtitles(generic_sidecar_video) == polish_sidecar_path, "polish sidecar should beat generic sidecar")
        _assert(
            _find_sidecar_subtitles(forced_sidecar_video) == full_polish_sidecar_path,
            "full polish sidecar should beat forced sidecar",
        )
    finally:
        for path in (
            video_sidecar_source,
            sidecar_path,
            uppercase_video_sidecar_source,
            uppercase_sidecar_path,
            generic_sidecar_video,
            generic_sidecar_path,
            polish_sidecar_path,
            forced_sidecar_video,
            forced_sidecar_path,
            full_polish_sidecar_path,
        ):
            try:
                path.unlink()
            except OSError:
                pass
    messages.append("sidecar lookup: OK")

    _assert(format_duration(8) == "8s", "short duration format mismatch")
    _assert(format_duration(65) == "1min 05s", "minute duration format mismatch")
    _assert(format_duration(3661) == "1h 01min 01s", "hour duration format mismatch")
    _assert(FILE_PROGRESS_TOTAL == 1000, "file progress total mismatch")
    _assert(progress_value_for_stage("prepare") == 25, "prepare progress value mismatch")
    _assert(progress_value_for_stage("tts", 0.5) == 400, "TTS weighted progress midpoint mismatch")
    _assert(progress_value_for_stage("mux", 0.5) == 970, "mux weighted progress midpoint mismatch")
    _assert(progress_value_for_stage("done") == 1000, "done progress value mismatch")
    _assert(safe_unit_eta_seconds(3, 100, 30.0) is None, "ETA should wait for enough units")
    _assert(safe_unit_eta_seconds(10, 100, 50.0) == 450.0, "ETA by units mismatch")
    _assert(ffmpeg_progress_ratio("out_time_us=5000000", 10.0) == 0.5, "ffmpeg out_time_us ratio mismatch")
    _assert(ffmpeg_progress_ratio("out_time_ms=5000000", 10.0) == 0.5, "ffmpeg out_time_ms ratio mismatch")
    _assert(ffmpeg_progress_ratio("out_time=00:00:05.000000", 10.0) == 0.5, "ffmpeg out_time ratio mismatch")
    _assert(format_progress_status("TTS", "10/100", 50.0, 450.0) == "TTS: 10/100 | czas 50s | ETA 7min 30s", "progress status format mismatch")
    marker = encode_progress_marker("mux", 0.25, "Dodawanie do MKV")
    _assert(decode_progress_marker(marker) == ("mux", 0.25, "Dodawanie do MKV"), "progress marker roundtrip mismatch")
    _assert(decode_progress_marker("normalny log") is None, "normal log should not decode as progress marker")
    progress_style = progress_bar_style()
    _assert("color: #111111" in progress_style, "progress text color should stay dark")
    _assert("background-color: #12d979" in progress_style, "progress chunk color mismatch")
    compacted_log = compact_app_log_message("ImportError: " + ("x" * 500), limit=80)
    _assert(len(compacted_log) <= 80 and compacted_log.endswith("..."), "long app log messages should be compacted")
    _assert(aac_quality_options() == ("192k", "256k", "320k", "384k", "448k", "640k"), "AAC quality options mismatch")
    _assert(aac_quality_label("384k") == "384 kb/s", "AAC quality label mismatch")
    _assert(aac_quality_label("bad") == "384 kb/s", "AAC quality invalid label fallback mismatch")
    _assert(audio_defaults_summary()["lektor_lufs"] == -14, "audio defaults lektor LUFS mismatch")
    _assert(audio_defaults_summary()["aac_bitrate"] == "384k", "audio defaults AAC bitrate mismatch")
    _assert(audio_defaults_summary()["lektor_weight"] == 2.3, "audio defaults lektor weight mismatch")
    _assert(audio_defaults_summary()["background_lufs"] == -18, "audio defaults background LUFS mismatch")
    _assert(audio_defaults_summary()["background_weight"] == 1.6, "audio defaults background weight mismatch")
    _assert(audio_defaults_summary()["lektor_delay_ms"] == DEFAULT_LEKTOR_DELAY_MS, "audio defaults lektor delay mismatch")
    _assert(audio_defaults_summary()["create_stereo_for_surround"] is True, "audio defaults should create additional stereo for 5.1 sources")
    _assert(format_lektor_delay_label(100) == "+100 ms", "positive lektor delay label mismatch")
    _assert(format_lektor_delay_label(-100) == "0 ms", "negative lektor delay label mismatch")
    _assert(format_lektor_delay_label(0) == "0 ms", "zero lektor delay label mismatch")
    internet_state = next(state for state in manager.list_states() if state.definition.engine_id == "edge")
    _assert(engine_combo_label("Internetowe", internet_state) == "Internetowe: Edge TTS [EDG]", "internet combo label should show short code")
    _assert(
        main_window_refresh_keeps_engine_signals_blocked(),
        "engine refresh should keep combo signals blocked while restoring last engine",
    )
    _assert(
        worker_message_should_refresh_engine_status("Model TTS: model w cache"),
        "engine status should refresh after model cache confirmation",
    )
    _assert(
        worker_message_should_refresh_engine_status("Model TTS: ladowanie modelu - prosze czekac"),
        "engine status should refresh when model loading starts",
    )
    _assert(
        worker_message_should_refresh_engine_status("Segment 1/216"),
        "engine status should refresh no later than first generated segment",
    )
    _assert(
        not worker_message_should_refresh_engine_status("Whisper QC: model w cache"),
        "Whisper QC cache messages should not refresh TTS engine status",
    )
    _assert(main_window_minimum_size() == (980, 620), "main window minimum size mismatch")
    _assert(slider_value_to_weight(20) == 2.0, "audio weight slider conversion mismatch")
    _assert(weight_to_slider_value(1.5) == 15, "audio weight reverse slider conversion mismatch")
    confirm = build_start_confirmation_text("Edge TTS", 2, 1, True, "384k", 100, True, True, True)
    _assert("Silnik TTS: Edge TTS" in confirm and "Pliki wideo: 1" in confirm, "preflight text mismatch")
    _assert("Jakosc sciezki lektora AAC: 384 kb/s" in confirm, "preflight should show AAC quality")
    _assert("Przesuniecie lektora: +100 ms" in confirm, "preflight should show lektor delay")
    _assert(BINARY_LOOKUP_HINT not in confirm, "preflight should not show binary hint when tools are present")
    confirm_missing_tools = build_start_confirmation_text("Edge TTS", 2, 1, True, "256k", 0, False, False, False)
    _assert(BINARY_LOOKUP_HINT in confirm_missing_tools, "preflight should show binary hint when tools are missing")
    missing_tools_message = missing_media_tools_message(("mkvmerge.exe", "ffmpeg.exe", "ffprobe.exe"))
    _assert("- ffmpeg.exe" in missing_tools_message, "missing tools message should list ffmpeg")
    _assert("- ffprobe.exe" in missing_tools_message, "missing tools message should list ffprobe")
    _assert("- mkvmerge.exe" in missing_tools_message, "missing tools message should list mkvmerge")
    _assert("GitHub" in missing_tools_message and "README" in missing_tools_message, "missing tools message should point to GitHub README")
    _assert(BINARY_LOOKUP_HINT in missing_ffmpeg_message(), "ffmpeg validation message should show lookup hint")
    _assert(BINARY_LOOKUP_HINT in missing_ffprobe_message(), "ffprobe validation message should show lookup hint")
    _assert(BINARY_LOOKUP_HINT in missing_mkvmerge_message(), "mkvmerge validation message should show lookup hint")
    _assert(not should_enable_start_button("", True), "start button should be disabled without selected engine")
    _assert(not should_enable_start_button("edge", False), "start button should be disabled while controls are locked")
    _assert(not should_enable_start_button("chatterbox", True, False), "start button should be disabled for unavailable engine")
    _assert(should_enable_start_button("edge", True), "start button should be enabled for selected engine")
    _assert(not should_enable_engine_actions("", True, True), "engine actions should be disabled without selected engine")
    _assert(not should_enable_engine_actions("chatterbox", False, True), "engine actions should be disabled for unavailable engine")
    _assert(not should_enable_engine_actions("edge", True, False), "engine actions should be disabled while controls are locked")
    _assert(should_enable_engine_actions("edge", True, True), "engine actions should be enabled for selectable engine")
    _assert(stored_engine_after_selection("edge", True) == "edge", "selected engine should be stored")
    _assert(stored_engine_after_selection("", False) == "", "empty engine selection should clear stored engine")
    _assert(stored_engine_after_selection("chatterbox", False) == "", "unavailable engine selection should clear stored engine")
    sorted_names = sorted(["Odcinek 10.mkv", "Odcinek 2.mkv", "Odcinek 1.mkv"], key=natural_path_key)
    _assert(sorted_names == ["Odcinek 1.mkv", "Odcinek 2.mkv", "Odcinek 10.mkv"], "natural sort mismatch")
    mixed_sorted_names = sorted(["10.mkv", "Odcinek 2.mkv", "2.mkv"], key=natural_path_key)
    _assert(mixed_sorted_names == ["2.mkv", "10.mkv", "Odcinek 2.mkv"], "natural sort mixed prefix mismatch")
    same_name_paths = sorted(
        [str(app_dir / "B" / "Odcinek 1.mkv"), str(app_dir / "A" / "Odcinek 1.mkv")],
        key=natural_path_key,
    )
    _assert(Path(same_name_paths[0]).parent.name == "A", "natural sort tie-breaker mismatch")
    sidecar_video = app_dir / "_self_test_queue_sidecar.mkv"
    sidecar_same_dir = app_dir / "_self_test_queue_sidecar.pl.srt"
    sidecar_numbered_neighbor = app_dir / "_self_test_queue_sidecar2.srt"
    sidecar_other_dir = app_dir / "_self_test_queue_other" / "_self_test_queue_sidecar.pl.srt"
    sidecar_video.write_text("", encoding="utf-8")
    sidecar_same_dir.write_text("", encoding="utf-8")
    sidecar_numbered_neighbor.write_text("", encoding="utf-8")
    sidecar_other_dir.parent.mkdir(exist_ok=True)
    sidecar_other_dir.write_text("", encoding="utf-8")
    try:
        _assert(
            is_sidecar_for_existing_video(sidecar_same_dir, [sidecar_video]),
            "queue sidecar detection mismatch",
        )
        _assert(
            not is_sidecar_for_existing_video(sidecar_other_dir, [sidecar_video]),
            "queue sidecar should not match a different folder",
        )
        _assert(
            not is_sidecar_for_existing_video(sidecar_numbered_neighbor, [sidecar_video]),
            "queue sidecar should not match numbered neighbor",
        )
        _assert(
            not is_sidecar_for_existing_video(sidecar_same_dir, []),
            "standalone subtitle should stay selectable",
        )
    finally:
        for path in (sidecar_video, sidecar_same_dir, sidecar_numbered_neighbor, sidecar_other_dir):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            sidecar_other_dir.parent.rmdir()
        except OSError:
            pass
    messages.append("UI helpers: OK")

    _assert(coerce_bool_for_widget("false") is False, "settings dialog bool string false mismatch")
    _assert(coerce_bool_for_widget("true") is True, "settings dialog bool string true mismatch")
    _assert(coerce_bool_for_widget("tekst") is False, "settings dialog unknown bool string mismatch")
    _assert(coerce_int_for_widget("7", 1, 10) == 7, "settings dialog int string mismatch")
    _assert(coerce_int_for_widget("7.5", 1, 10) == 1, "settings dialog fractional int should fallback")
    _assert(coerce_int_for_widget(99, 1, 10) == 10, "settings dialog int clamp mismatch")
    _assert(coerce_float_for_widget("nan", 0.1, 2.0) == 0.1, "settings dialog nan float should fallback")
    _assert(settings_help_button_label() == "?", "settings dialog help button label mismatch")
    diagnostic_group_keys = [key for _, keys in diagnostic_field_groups() for key in keys]
    _assert(len(diagnostic_group_keys) == len(set(diagnostic_group_keys)), "diagnostic fields should appear in one group only")
    _assert(is_diagnostic_field("save_processed_subtitles"), "processed subtitles should be diagnostic")
    _assert(is_diagnostic_field("save_quality_report"), "quality report should be diagnostic")
    _assert(is_diagnostic_field("save_run_reports"), "run reports should be diagnostic")
    _assert(is_diagnostic_field("save_lektor_segments"), "save lektor segments should be diagnostic")
    _assert(is_diagnostic_field("save_lektor_track_before_normalization"), "pre-normalization track should be diagnostic")
    _assert(is_diagnostic_field("save_lektor_track_after_normalization"), "post-normalization track should be diagnostic")
    _assert(is_diagnostic_field("save_audio_mix_steps"), "audio mix steps should be diagnostic")
    _assert(not is_diagnostic_field("save_lektor_assets"), "vague lektor assets option should be removed")
    _assert(not is_diagnostic_field("save_prepared_voice_sample"), "prepared voice sample option should be removed")
    _assert(not is_diagnostic_field("voice"), "normal voice setting should not be diagnostic")
    _assert(is_audio_qc_field("audio_qc_enabled"), "Audio QC toggle should be in audio section")
    _assert(is_audio_qc_field("audio_qc_retry_attempts"), "Audio QC retry should be in audio section")
    _assert(is_speech_qc_field("whisper_qc_enabled"), "Whisper QC toggle should be in speech section")
    _assert(is_speech_qc_field("whisper_qc_retry_attempts"), "Whisper QC retry should be in speech section")
    _assert(is_speech_qc_field("whisper_qc_model"), "Whisper QC model should be in speech section")
    _assert(not is_audio_qc_field("voice") and not is_speech_qc_field("voice"), "normal voice should stay in model settings section")
    _assert(choice_value_for_widget("medium", ("small", "medium"), "small") == "medium", "settings dialog choice value mismatch")
    _assert(choice_value_for_widget("bogus", ("small", "medium"), "small") == "small", "settings dialog invalid choice fallback mismatch")
    class _ChoiceDataProbe:
        def currentData(self) -> str:
            return "pl-PL-MarekNeural"

        def currentText(self) -> str:
            return "Marek"

    test_combo = _ChoiceDataProbe()
    _assert(choice_data_for_widget(test_combo) == "pl-PL-MarekNeural", "settings dialog should save choice data")
    _assert(edge_slider_value_for_widget("+15%", "%", -100, 100, 5) == 15, "edge slider percent parse mismatch")
    _assert(edge_slider_value_for_widget("-17Hz", "Hz", -100, 100, 5) == -15, "edge slider should snap to step")
    _assert(edge_slider_value_for_widget("+150%", "%", -100, 100, 5) == 100, "edge slider should clamp high values")
    _assert(format_edge_slider_value(0, "%") == "+0%", "edge slider percent format mismatch")
    _assert(format_edge_slider_value(-10, "Hz") == "-10Hz", "edge slider Hz format mismatch")
    merged_visible_settings = merge_engine_settings_values({"seed": 12345, "cfg_value": 2.0}, {"cfg_value": 1.5})
    _assert(merged_visible_settings == {"seed": 12345, "cfg_value": 1.5}, "settings merge should preserve hidden seed")
    _assert(initial_install_button_label() == "Zainstaluj", "TTS manager initial install label mismatch")
    manager_dialog_source = (app_dir / "app" / "ui" / "dialogs" / "tts_manager_dialog.py").read_text(encoding="utf-8")
    _assert("btn_install_cu128" not in manager_dialog_source, "TTS manager should use one install button and variant dialog")
    _assert("def choose_torch_variant" in manager_dialog_source, "TTS manager should show PyTorch variant dialog after install click")
    _assert(should_show_dictionary_row("Batman", "bat"), "dictionary search should match prefix")
    _assert(not should_show_dictionary_row("Alfa", "bat"), "dictionary search should hide non-prefix")
    _assert(should_show_dictionary_row("", "bat"), "dictionary search should keep empty editable rows visible")
    dictionary_export_path = app_dir / "_self_test_dictionary_export.json"
    dictionary_import_path = app_dir / "_self_test_dictionary_import.json"
    try:
        count, skipped = save_dictionary_external_file(dictionary_export_path, {"Vox": "woks", " Batman ": "batman"})
        _assert(count == 2 and skipped == 0, "dictionary external save count mismatch")
        exported_dictionary = json.loads(dictionary_export_path.read_text(encoding="utf-8"))
        _assert(list(exported_dictionary.keys()) == ["Batman", "Vox"], "dictionary external save should sanitize and sort")
        dictionary_import_path.write_text(
            json.dumps({"": "puste", "Silos": "sajlos", "vox": "woks"}, ensure_ascii=False),
            encoding="utf-8",
        )
        imported_dictionary, import_count = load_dictionary_external_file(dictionary_import_path)
        _assert(import_count == 2, "dictionary external load count mismatch")
        _assert(imported_dictionary == {"Silos": "sajlos", "vox": "woks"}, "dictionary external load sanitize mismatch")
    finally:
        for path in (dictionary_export_path, dictionary_import_path):
            try:
                path.unlink()
            except OSError:
                pass
    long_diagnostic_text = "brak importu | " + ("C:\\bardzo\\dluga\\sciezka\\" * 10)
    _assert(
        diagnostic_table_text(long_diagnostic_text, limit=80).endswith("..."),
        "diagnostics table should compact long details",
    )
    _assert(
        scrollable_details_line_wrap_mode() == "WidgetWidth",
        "scrollable details should wrap long lines",
    )
    _assert(SUPPORTED_SUBTITLE_EXTENSIONS == (".srt", ".txt"), "supported subtitle extension constant mismatch")
    _assert(
        apply_dictionary("Mam 2 kg.", {"2 kg": "dwa kilo"}) == "Mam dwa kilo.",
        "dictionary should apply user-defined subtitle replacements without automatic text normalization",
    )
    _assert(coerce_float_for_widget(9.0, 0.1, 2.0) == 2.0, "settings dialog float clamp mismatch")
    _assert(not should_enable_keep_settings_remove(False, True, False), "keep remove should be disabled for internet engines")
    _assert(not should_enable_keep_settings_remove(True, False, False), "keep remove should be disabled without runtime")
    _assert(not should_enable_keep_settings_remove(True, True, True), "keep remove should be disabled while busy")
    _assert(should_enable_keep_settings_remove(True, True, False), "keep remove should be enabled for local runtime")
    _assert(not should_recreate_venv(False, False), "missing venv should be created without rebuild path")
    _assert(should_recreate_venv(True, False), "existing venv with broken pip should be recreated")
    _assert(not should_recreate_venv(True, True), "existing venv with working pip should be reused")
    _assert(should_show_vram_info_button("chatterbox"), "Chatterbox settings should show VRAM info")
    _assert(should_show_vram_info_button("omnivoice"), "OmniVoice settings should show VRAM info")
    _assert(should_show_vram_info_button("coqui_xtts"), "Coqui XTTS settings should show VRAM info")
    _assert(not should_show_vram_info_button("vibevoice"), "archived VibeVoice settings should not show VRAM info")
    _assert(not should_show_vram_info_button("fish_speech"), "archived Fish Speech settings should not show VRAM info")
    _assert(not should_show_vram_info_button("piper"), "Piper settings should not show VRAM info")
    _assert(not should_show_vram_info_button("edge"), "Edge settings should not show VRAM info")
    messages.append("settings dialog helpers: OK")

    config_store = AppConfigStore(app_dir / "_self_test_config.json")
    try:
        config_store.load()
        _assert((app_dir / "_self_test_config.json").read_text(encoding="utf-8").endswith("\n"), "app config save should end with newline")
        config_store.set_last_file_dir(f"  {app_dir}  ")
        _assert(config_store.last_file_dir() == str(app_dir), "last file dir config mismatch")
        config_store.set_last_engine("  edge  ")
        _assert(config_store.last_engine() == "edge", "last engine should be stripped before saving")
        config_store.data["ui"]["last_file_dir"] = f"  {app_dir}  "
        config_store.data["tts"]["last_engine"] = "  edge  "
        _assert(config_store.last_file_dir() == str(app_dir), "last file dir getter should strip manual spaces")
        _assert(config_store.last_engine() == "edge", "last engine getter should strip manual spaces")
        config_store.set_aac_bitrate("384k")
        _assert(config_store.aac_bitrate() == "384k", "AAC bitrate setter mismatch")
        config_store.set_aac_bitrate("not-valid")
        _assert(config_store.aac_bitrate() == "384k", "AAC bitrate setter should sanitize invalid values")
        config_store.set_lektor_delay_ms(123)
        _assert(config_store.lektor_delay_ms() == 100, "lektor delay setter should sanitize values")
        config_store.set_create_stereo_for_surround(False)
        _assert(config_store.create_stereo_for_surround() is False, "surround stereo option setter mismatch")
        config_store.set_create_stereo_for_surround(True)
        _assert(config_store.create_stereo_for_surround() is True, "surround stereo option should accept true")
        _assert(config_store.stt_settings() == SttSettings(), "default STT settings mismatch")
        default_stt = config_store.stt_settings()
        _assert(default_stt.accuracy == "standard", "default STT accuracy mismatch")
        _assert(default_stt.vad_enabled is True, "default STT VAD should be enabled")
        _assert(default_stt.vad_sensitivity == "standard", "default STT VAD sensitivity mismatch")
        _assert(default_stt.whisperx_device == "cpu", "default WhisperX device should be CPU")
        _assert(default_stt.whisperx_compute_type == "int8", "default WhisperX compute type should be int8")
        _assert(default_stt.postprocess_enabled is True, "default STT postprocessing should be enabled")
        _assert(default_stt.open_workspace_on_finish is False, "default STT open workspace should be disabled")
        _assert(default_stt.save_prepared_audio is False, "default STT prepared audio diagnostic should be disabled")
        _assert(default_stt.save_report is False, "default STT report diagnostic should be disabled")
        _assert(default_stt.save_log is False, "default STT log diagnostic should be disabled")
        config_store.set_stt_model("large-v3")
        config_store.set_stt_language("pl")
        config_store.set_stt_device("cpu")
        config_store.set_stt_compute_type("float16")
        config_store.set_stt_accuracy("accurate")
        config_store.set_stt_vad_enabled(False)
        config_store.set_stt_vad_sensitivity("strong")
        config_store.set_stt_postprocess_enabled(False)
        config_store.set_stt_open_workspace_on_finish(True)
        config_store.set_stt_save_prepared_audio(True)
        config_store.set_stt_save_report(True)
        config_store.set_stt_save_log(True)
        stt_settings = config_store.stt_settings()
        _assert(stt_settings.model == "large-v3", "STT model setter mismatch")
        _assert(stt_settings.language == "pl", "STT language setter mismatch")
        _assert(stt_settings.device == "cpu", "STT device setter mismatch")
        _assert(stt_settings.compute_type == "int8", "STT CPU compute type should be forced to int8")
        _assert(stt_settings.accuracy == "accurate", "STT accuracy setter mismatch")
        _assert(stt_settings.vad_enabled is False, "STT VAD enabled setter mismatch")
        _assert(stt_settings.vad_sensitivity == "strong", "STT VAD sensitivity setter mismatch")
        _assert(stt_settings.postprocess_enabled is False, "STT postprocessing setter mismatch")
        _assert(stt_settings.open_workspace_on_finish is True, "STT open workspace setter mismatch")
        _assert(stt_settings.save_prepared_audio is True, "STT prepared audio diagnostic setter mismatch")
        _assert(stt_settings.save_report is True, "STT report diagnostic setter mismatch")
        _assert(stt_settings.save_log is True, "STT log diagnostic setter mismatch")
        config_store.set_stt_whisperx_device("cuda:1")
        config_store.set_stt_whisperx_compute_type("float16")
        whisperx_stt_settings = config_store.stt_settings()
        _assert(whisperx_stt_settings.whisperx_device == "cuda:1", "WhisperX device setter mismatch")
        _assert(whisperx_stt_settings.whisperx_compute_type == "float16", "WhisperX compute type setter mismatch")
        config_store.set_stt_whisperx_device("cpu")
        _assert(config_store.stt_settings().whisperx_compute_type == "int8", "WhisperX CPU compute type should be forced to int8")
        config_store.set_stt_engine("whisperx")
        whisperx_defaults = config_store.stt_settings()
        _assert(whisperx_defaults.model == "small", "WhisperX should keep an independent model setting")
        _assert(whisperx_defaults.whisperx_device == "cpu", "WhisperX should default to CPU independently")
        _assert(whisperx_defaults.postprocess_enabled is True, "WhisperX postprocessing should default to enabled")
        _assert(whisperx_defaults.open_workspace_on_finish is False, "WhisperX open workspace should default to disabled")
        _assert(whisperx_defaults.save_report is False, "WhisperX diagnostics should default to disabled")
        config_store.set_stt_model("medium")
        config_store.set_stt_postprocess_enabled(False)
        config_store.set_stt_open_workspace_on_finish(True)
        config_store.set_stt_save_report(True)
        config_store.set_stt_whisperx_device("cuda:1")
        config_store.set_stt_engine("faster_whisper")
        faster_again = config_store.stt_settings()
        _assert(faster_again.model == "large-v3", "faster-whisper model should not be changed by WhisperX")
        _assert(faster_again.device == "cpu", "faster-whisper device should stay independent from WhisperX")
        _assert(faster_again.postprocess_enabled is False, "faster-whisper postprocessing should stay independent from WhisperX")
        _assert(faster_again.open_workspace_on_finish is True, "faster-whisper open workspace should stay independent from WhisperX")
        _assert(faster_again.save_report is True, "faster-whisper diagnostics should stay independent from WhisperX")
        config_store.reset_stt_engine_defaults("faster_whisper")
        faster_reset = config_store.stt_settings()
        _assert(faster_reset.model == "small", "faster-whisper reset should restore model")
        _assert(faster_reset.postprocess_enabled is True, "faster-whisper reset should restore postprocessing")
        _assert(faster_reset.open_workspace_on_finish is False, "faster-whisper reset should restore open workspace")
        _assert(faster_reset.save_report is False, "faster-whisper reset should restore diagnostics")
        config_store.set_stt_engine("whisperx")
        config_store.set_stt_model("medium")
        config_store.set_stt_whisperx_device("cuda:1")
        config_store.reset_stt_engine_defaults("whisperx")
        whisperx_reset = config_store.stt_settings()
        _assert(whisperx_reset.model == "small", "WhisperX reset should restore model")
        _assert(whisperx_reset.whisperx_device == "cpu", "WhisperX reset should restore device")
        _assert(whisperx_reset.whisperx_compute_type == "int8", "WhisperX reset should restore compute type")
        _assert(whisperx_reset.open_workspace_on_finish is False, "WhisperX reset should restore open workspace")
        config_store.set_stt_engine("whisper_cpp")
        whisper_cpp_defaults = config_store.stt_settings()
        _assert(whisper_cpp_defaults.model == "small", "whisper.cpp should keep an independent model setting")
        _assert(whisper_cpp_defaults.whisper_cpp_runtime == "cpu", "whisper.cpp should default to CPU runtime")
        _assert(whisper_cpp_defaults.whisper_cpp_device == "auto", "whisper.cpp CUDA device should default to auto")
        _assert(whisper_cpp_defaults.postprocess_enabled is True, "whisper.cpp postprocessing should default to enabled")
        _assert(whisper_cpp_defaults.open_workspace_on_finish is False, "whisper.cpp open workspace should default to disabled")
        _assert(whisper_cpp_defaults.save_report is False, "whisper.cpp diagnostics should default to disabled")
        config_store.set_stt_model("large-v3")
        config_store.set_stt_whisper_cpp_runtime("cuda")
        config_store.set_stt_whisper_cpp_threads(12)
        config_store.reset_stt_engine_defaults("whisper_cpp")
        whisper_cpp_reset = config_store.stt_settings()
        _assert(whisper_cpp_reset.model == "small", "whisper.cpp reset should restore model")
        _assert(whisper_cpp_reset.whisper_cpp_runtime == "cpu", "whisper.cpp reset should restore runtime")
        _assert(whisper_cpp_reset.whisper_cpp_threads == 0, "whisper.cpp reset should restore threads")
        _assert(whisper_cpp_reset.open_workspace_on_finish is False, "whisper.cpp reset should restore open workspace")
        config_store.set_stt_accuracy("bad")
        config_store.set_stt_vad_sensitivity("bad")
        _assert(config_store.stt_settings().accuracy == "standard", "STT should sanitize unknown accuracy")
        _assert(config_store.stt_settings().vad_sensitivity == "standard", "STT should sanitize unknown VAD sensitivity")
        config_store.set_stt_language("ja")
        _assert(config_store.stt_settings().language == "ja", "STT should accept Japanese language code")
        config_store.set_stt_language("de")
        _assert(config_store.stt_settings().language == "de", "STT should accept German language code")
        config_store.set_stt_language("bad-language")
        _assert(config_store.stt_settings().language == "auto", "STT should sanitize unknown language code")
    finally:
        try:
            (app_dir / "_self_test_config.json").unlink()
        except OSError:
            pass
    messages.append("config helpers: OK")

    config_merge_path = app_dir / "_self_test_config_merge.json"
    try:
        config_merge_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "ui": {
                        "theme": "dark",
                        "last_file_dir": "D:/Filmy",
                        "custom_ui_flag": "keep-ui",
                    },
                    "tts": {
                        "last_engine": "edge",
                    },
                    "custom_root_flag": "keep-root",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        merged_store = AppConfigStore(config_merge_path)
        merged_store.load()
        merged_data = json.loads(config_merge_path.read_text(encoding="utf-8"))
        _assert(merged_data["ui"]["last_file_dir"] == "D:/Filmy", "app config merge overwrote last dir")
        _assert(merged_data["tts"]["last_engine"] == "edge", "app config merge overwrote last engine")
        _assert(merged_data["ui"]["custom_ui_flag"] == "keep-ui", "app config merge removed nested custom key")
        _assert(merged_data["custom_root_flag"] == "keep-root", "app config merge removed root custom key")
        _assert(merged_data["ui"]["window"]["width"] == 1280, "app config merge did not persist window width default")
        _assert(merged_data["ui"]["window"]["height"] == 780, "app config merge did not persist window height default")
        _assert(merged_data["ui"]["window"]["mode"] == "normal", "app config merge did not persist window mode default")
        _assert(merged_data["output"]["aac_bitrate"] == "384k", "app config merge did not persist AAC bitrate default")
        _assert(merged_data["output"]["lektor_lufs"] == -14, "app config merge did not persist lektor LUFS default")
        _assert(merged_data["output"]["lektor_weight"] == 2.3, "app config merge did not persist lektor weight default")
        _assert(merged_data["output"]["background_lufs"] == -18, "app config merge did not persist background LUFS default")
        _assert(merged_data["output"]["background_weight"] == 1.6, "app config merge did not persist background weight default")
        _assert(merged_data["output"]["lektor_delay_ms"] == DEFAULT_LEKTOR_DELAY_MS, "app config merge did not persist lektor delay default")
        merged_store.set_window_state(1440, 900, "maximized")
        window_state = merged_store.window_state()
        _assert(window_state["width"] == 1440 and window_state["height"] == 900, "window state dimensions did not persist")
        _assert(window_state["mode"] == "maximized", "window state mode did not persist")
    finally:
        try:
            config_merge_path.unlink()
        except OSError:
            pass
    messages.append("app config merge: OK")

    config_normalize_path = app_dir / "_self_test_config_normalize.json"
    try:
        config_normalize_path.write_text(
            json.dumps(
                {
                    "version": "bad-version",
                    "ui": {
                        "theme": 123,
                        "last_file_dir": 456,
                        "window": {
                            "width": "tiny",
                            "height": 99999,
                            "mode": "bad",
                        },
                        "custom_ui_flag": "keep-ui",
                    },
                    "tts": {
                        "last_engine": 789,
                        "custom_tts_flag": "keep-tts",
                    },
                    "output": {
                        "aac_bitrate": "640",
                        "lektor_lufs": "-99",
                        "lektor_weight": "9.9",
                        "background_lufs": "bad",
                        "background_weight": "0",
                        "lektor_delay_ms": "9999",
                        "custom_output_flag": "keep-output",
                    },
                    "custom_root_flag": "keep-root",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        normalize_store = AppConfigStore(config_normalize_path)
        normalize_store.load()
        normalized_data = json.loads(config_normalize_path.read_text(encoding="utf-8"))
        _assert(normalized_data["version"] == 1, "app config normalize should restore bad version")
        _assert(normalized_data["ui"]["theme"] == "dark", "app config normalize should restore bad theme")
        _assert(normalized_data["ui"]["last_file_dir"] == "", "app config normalize should restore bad last dir")
        _assert(normalized_data["ui"]["window"]["width"] == 1280, "app config normalize should restore bad window width")
        _assert(normalized_data["ui"]["window"]["height"] == 4000, "app config normalize should clamp bad window height")
        _assert(normalized_data["ui"]["window"]["mode"] == "normal", "app config normalize should restore bad window mode")
        _assert(normalized_data["tts"]["last_engine"] == "", "app config normalize should restore bad last engine")
        _assert(normalized_data["output"]["aac_bitrate"] == "640k", "app config normalize should sanitize AAC bitrate")
        _assert(normalized_data["output"]["lektor_lufs"] == -30, "app config normalize should clamp lektor LUFS")
        _assert(normalized_data["output"]["lektor_weight"] == 3.0, "app config normalize should clamp lektor weight")
        _assert(normalized_data["output"]["background_lufs"] == -18, "app config normalize should restore background LUFS")
        _assert(normalized_data["output"]["background_weight"] == 0.1, "app config normalize should clamp background weight")
        _assert(normalized_data["output"]["lektor_delay_ms"] == MAX_LEKTOR_DELAY_MS, "app config normalize should clamp lektor delay")
        _assert(normalized_data["ui"]["custom_ui_flag"] == "keep-ui", "app config normalize removed nested ui custom key")
        _assert(normalized_data["tts"]["custom_tts_flag"] == "keep-tts", "app config normalize removed nested tts custom key")
        _assert(normalized_data["output"]["custom_output_flag"] == "keep-output", "app config normalize removed nested output custom key")
        _assert(normalized_data["custom_root_flag"] == "keep-root", "app config normalize removed root custom key")
    finally:
        try:
            config_normalize_path.unlink()
        except OSError:
            pass
    messages.append("app config normalize: OK")

    config_bad_sections_path = app_dir / "_self_test_config_bad_sections.json"
    try:
        config_bad_sections_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "ui": "bad-ui-section",
                    "tts": 123,
                    "output": False,
                    "custom_root_flag": "keep-root",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        bad_sections_store = AppConfigStore(config_bad_sections_path)
        bad_sections_store.load()
        bad_sections_data = json.loads(config_bad_sections_path.read_text(encoding="utf-8"))
        _assert(isinstance(bad_sections_data.get("ui"), dict), "bad ui section should be restored")
        _assert(isinstance(bad_sections_data.get("tts"), dict), "bad tts section should be restored")
        _assert(isinstance(bad_sections_data.get("output"), dict), "bad output section should be restored")
        _assert(bad_sections_data["ui"]["last_file_dir"] == "", "bad ui section default mismatch")
        _assert(bad_sections_data["tts"]["last_engine"] == "", "bad tts section default mismatch")
        _assert(bad_sections_data["stt"]["engine"] == "faster_whisper", "bad stt section default mismatch")
        bad_sections_store.set_stt_engine("whisper_cpp")
        _assert(bad_sections_store.stt_settings().engine == "whisper_cpp", "STT engine selection should persist")
        bad_sections_store.set_stt_engine("whisperx")
        _assert(bad_sections_store.stt_settings().engine == "whisperx", "WhisperX STT engine selection should persist")
        bad_sections_store.set_stt_whisper_cpp_threads("12")
        _assert(bad_sections_store.stt_settings().whisper_cpp_threads == 12, "whisper.cpp thread setting should persist")
        _assert(bad_sections_data["output"]["aac_bitrate"] == "384k", "bad output section AAC bitrate default mismatch")
        _assert(bad_sections_data["custom_root_flag"] == "keep-root", "custom root key should survive bad section repair")
        bad_sections_store.set_last_file_dir(str(app_dir))
        bad_sections_store.set_last_engine("edge")
        _assert(bad_sections_store.last_file_dir() == str(app_dir), "repaired config last dir write mismatch")
        _assert(bad_sections_store.last_engine() == "edge", "repaired config last engine write mismatch")
    finally:
        try:
            config_bad_sections_path.unlink()
        except OSError:
            pass
    messages.append("app config bad sections: OK")

    stt_output_probe_dir = app_dir / "_self_test_stt_workspace"
    try:
        video_path = stt_output_probe_dir / "Film testowy.mkv"
        audio_path = stt_output_probe_dir / "Glos.wav"
        dts_path = stt_output_probe_dir / "Glos.dts"
        subtitle_path = stt_output_probe_dir / "Napisy.srt"
        stt_output_probe_dir.mkdir(parents=True, exist_ok=True)
        video_path.write_text("", encoding="utf-8")
        audio_path.write_text("", encoding="utf-8")
        dts_path.write_text("", encoding="utf-8")
        subtitle_path.write_text("", encoding="utf-8")
        _assert(is_stt_input_file(video_path), "STT should accept video input")
        _assert(is_stt_input_file(audio_path), "STT should accept audio input")
        _assert(is_stt_input_file(dts_path), "STT should accept DTS audio input")
        _assert(not is_stt_input_file(subtitle_path), "STT should not accept subtitle input")
        first_stem = next_stt_output_stem(stt_output_probe_dir, video_path)
        _assert(first_stem.endswith("_Film_testowy_STT_FW"), "STT output stem should include source and engine")
        (stt_output_probe_dir / first_stem).mkdir()
        second_stem = next_stt_output_stem(stt_output_probe_dir, video_path)
        _assert(second_stem == f"{first_stem}_2", "STT output stem should avoid collisions")
        cpp_stem = next_stt_output_stem(stt_output_probe_dir, video_path, engine="whisper_cpp")
        _assert(cpp_stem.endswith("_Film_testowy_STT_WCPP"), "whisper.cpp STT output stem should include engine code")
    finally:
        _cleanup_tree(stt_output_probe_dir)
    messages.append("STT helpers: OK")
    _assert(STT_LANGUAGE_CODES[0] == "auto", "STT language list should start with auto")
    _assert({"pl", "en", "de", "ja", "zh"}.issubset(set(STT_LANGUAGE_CODES)), "STT language list should include common Whisper languages")
    _assert(len(STT_LANGUAGE_OPTIONS) > 80, "STT should expose full Whisper language list")
    _assert(stt_language_label("ja").startswith("ja - "), "STT language label should include language code")
    _assert(
        stt_model_cache_key("small", "cuda:1", "float16", paths.faster_whisper_cache_dir)
        == ("small", "cuda:1", "float16", str(paths.faster_whisper_cache_dir)),
        "STT model cache key mismatch",
    )
    stt_kwargs = default_stt_transcribe_kwargs()
    _assert(stt_kwargs["vad_filter"] is True, "STT should enable VAD to reduce silence hallucinations")
    _assert(stt_kwargs["condition_on_previous_text"] is False, "STT should not carry previous text between windows")
    _assert(stt_kwargs["temperature"] == 0.0, "STT should use deterministic transcription by default")
    accurate_stt_kwargs = default_stt_transcribe_kwargs(SttSettings(accuracy="accurate", vad_sensitivity="strong"))
    _assert(accurate_stt_kwargs["beam_size"] == 8, "STT accurate mode should increase beam size")
    _assert(accurate_stt_kwargs["vad_parameters"]["min_silence_duration_ms"] == 250, "STT strong VAD should use shorter silence threshold")
    fast_no_vad_kwargs = default_stt_transcribe_kwargs(SttSettings(accuracy="fast", vad_enabled=False))
    _assert(fast_no_vad_kwargs["beam_size"] == 1, "STT fast mode should reduce beam size")
    _assert(fast_no_vad_kwargs["vad_filter"] is False, "STT should allow disabling VAD")
    _assert("vad_parameters" not in fast_no_vad_kwargs, "STT disabled VAD should not pass VAD parameters")
    _assert(whisper_cpp_model_file_name("large-v3") == "ggml-large-v3.bin", "whisper.cpp model file mismatch")
    _assert(whisper_cpp_model_file_name("turbo") == "ggml-large-v3-turbo.bin", "whisper.cpp turbo model file mismatch")
    _assert(whisper_cpp_model_file_name("large") == "ggml-large-v3.bin", "whisper.cpp large alias mismatch")
    _assert(normalize_whisper_cpp_model_name("large") == "large-v3", "whisper.cpp large model normalize mismatch")
    _assert(set(WHISPER_CPP_RUNTIME_PACKAGES) == {"cpu", "cuda"}, "whisper.cpp runtime variants mismatch")
    _assert(sanitize_whisper_cpp_runtime("CUDA") == "cuda", "whisper.cpp runtime sanitize mismatch")
    _assert(sanitize_whisper_cpp_device("cuda:1") == "cuda:1", "whisper.cpp device sanitize mismatch")
    _assert(
        whisper_cpp_runtime_download_label(WHISPER_CPP_RUNTIME_PACKAGES["cuda"]) == "whisper.cpp: pobieranie plikow programu (CUDA)",
        "whisper.cpp CUDA runtime download label should not be confused with CUDA DLL runtime",
    )
    cpp_cuda_probe_dir = app_dir / "_self_test_cpp_cuda_runtime"
    cpp_cuda_paths = build_paths(cpp_cuda_probe_dir)
    try:
        cuda13_dir = cuda_runtime_dll_dir(cpp_cuda_paths, CUDA_RUNTIME_WHISPER_CPP_ID)
        cuda13_dir.mkdir(parents=True)
        for dll_name in ("cublas64_13.dll", "cublasLt64_13.dll", "cudart64_13.dll"):
            (cuda13_dir / dll_name).write_text("", encoding="utf-8")
        cpp_cuda_env = whisper_cpp_runtime_env(cpp_cuda_paths, "cuda")
        _assert(str(cuda13_dir) in cpp_cuda_env.get("PATH", "").split(os.pathsep), "whisper.cpp CUDA env should include CUDA 13 runtime dir")
        cpp_cpu_env = whisper_cpp_runtime_env(cpp_cuda_paths, "cpu", base_env={"PATH": "base"})
        _assert(cpp_cpu_env.get("PATH") == "base", "whisper.cpp CPU env should not add CUDA runtime dir")
    finally:
        _cleanup_tree(cpp_cuda_probe_dir)
    cpp_command = build_whisper_cpp_command(
        exe_path=Path("whisper-cli.exe"),
        model_path=Path("models") / "ggml-small.bin",
        input_wav=Path("audio.wav"),
        output_base=Path("wynik"),
        language="pl",
        threads=6,
    )
    _assert(cpp_command[:6] == ["whisper-cli.exe", "-m", str(Path("models") / "ggml-small.bin"), "-f", "audio.wav", "-osrt"], "whisper.cpp command core args mismatch")
    _assert("-l" in cpp_command and "pl" in cpp_command, "whisper.cpp command should pass language")
    _assert("-t" in cpp_command and "6" in cpp_command, "whisper.cpp command should pass thread count")
    _assert("-pp" in cpp_command, "whisper.cpp command should enable progress output")
    _assert("-sns" in cpp_command, "whisper.cpp command should suppress non-speech tokens")
    cpp_cpu_command = build_whisper_cpp_command(
        exe_path=Path("whisper-cli.exe"),
        model_path=Path("models") / "ggml-small.bin",
        input_wav=Path("audio.wav"),
        output_base=Path("wynik"),
        device="cpu",
    )
    _assert("-ng" in cpp_cpu_command, "whisper.cpp CPU command should disable GPU")
    cpp_cuda_command = build_whisper_cpp_command(
        exe_path=Path("whisper-cli.exe"),
        model_path=Path("models") / "ggml-small.bin",
        input_wav=Path("audio.wav"),
        output_base=Path("wynik"),
        device="cuda:1",
    )
    _assert("-dev" in cpp_cuda_command and "1" in cpp_cuda_command, "whisper.cpp CUDA command should select GPU")
    _assert(whisperx_device_args("cuda:1") == ("cuda", 1), "WhisperX should split cuda device index")
    _assert(whisperx_device_args("cpu") == ("cpu", 0), "WhisperX should accept CPU device")
    _assert(normalize_whisperx_compute_type("float16", "cpu") == "int8", "WhisperX CPU should avoid float16")
    _assert(normalize_whisperx_compute_type("float16", "cuda") == "float16", "WhisperX CUDA should keep float16")
    _assert(normalize_whisperx_model_name("turbo") == "large-v3-turbo", "WhisperX turbo alias mismatch")
    whisperx_cuda_probe_dir = app_dir / "_self_test_whisperx_cuda_runtime"
    whisperx_cuda_paths = build_paths(whisperx_cuda_probe_dir)
    try:
        whisperx_cuda12_dir = cuda_runtime_dll_dir(whisperx_cuda_paths, CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID)
        whisperx_cuda12_dir.mkdir(parents=True)
        for dll_name in CUDA_RUNTIME_PACKAGES[CUDA_RUNTIME_CTRANSLATE2_PYTORCH_ID].required_dlls:
            (whisperx_cuda12_dir / dll_name).write_text("", encoding="utf-8")
        whisperx_gpu_progress: list[str] = []
        ensure_whisperx_gpu_runtime(whisperx_cuda_paths, progress=whisperx_gpu_progress.append)
        _assert("WhisperX: biblioteki GPU gotowe" in whisperx_gpu_progress, "WhisperX should report ready GPU libraries even when runtime is already installed")
        whisperx_gpu_env = whisperx_runtime_env(whisperx_cuda_paths, "cuda:1", {"PATH": "BASE"})
        _assert(str(whisperx_cuda12_dir) in whisperx_gpu_env.get("PATH", "").split(os.pathsep), "WhisperX GPU env should include shared CUDA 12 runtime dir")
        whisperx_cpu_env = whisperx_runtime_env(whisperx_cuda_paths, "cpu", {"PATH": "BASE"})
        _assert(str(whisperx_cuda12_dir) not in whisperx_cpu_env.get("PATH", "").split(os.pathsep), "WhisperX CPU env should not include CUDA runtime dir")
    finally:
        _cleanup_tree(whisperx_cuda_probe_dir)
    whisperx_command = build_whisperx_command(
        python_path=Path("python.exe"),
        input_wav=Path("audio.wav"),
        output_dir=Path("out"),
        model="small",
        model_dir=Path("cache"),
        language="pl",
        device="cuda:1",
        compute_type="float16",
        batch_size=8,
        beam_size=5,
    )
    _assert(whisperx_command[:3] == ["python.exe", "-m", "whisperx"], "WhisperX command core args mismatch")
    _assert("--output_format" in whisperx_command and "json" in whisperx_command, "WhisperX command should write JSON")
    _assert("--device" in whisperx_command and "cuda" in whisperx_command, "WhisperX command should use CUDA")
    _assert("--device_index" in whisperx_command and "1" in whisperx_command, "WhisperX command should select GPU index")
    _assert("--language" in whisperx_command and "pl" in whisperx_command, "WhisperX command should pass selected language")
    whisperx_json_probe = app_dir / "_self_test_whisperx.json"
    try:
        whisperx_json_probe.write_text(
            json.dumps(
                {
                    "language": "en",
                    "segments": [
                        {"start": 1.25, "end": 2.5, "text": "Hello there."},
                        {"start": 2.5, "end": 3.0, "text": ""},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        whisperx_segments, whisperx_language = load_whisperx_json_segments(whisperx_json_probe)
        _assert(whisperx_language == "en", "WhisperX JSON parser should preserve detected language")
        _assert(len(whisperx_segments) == 1 and whisperx_segments[0].start_ms == 1250, "WhisperX JSON parser should convert timestamps")
    finally:
        whisperx_json_probe.unlink(missing_ok=True)
    _assert(subtitle_profile_for_language("en") == ENGLISH_USA_SUBTITLE_PROFILE, "STT should use English USA profile for English")
    _assert(subtitle_profile_for_language("english") == ENGLISH_USA_SUBTITLE_PROFILE, "STT should accept normalized English language names")
    _assert(subtitle_profile_for_language("pl") == FALLBACK_SUBTITLE_PROFILE, "STT should use global fallback until language profile exists")
    _assert(is_stt_non_dialogue_text("*Dramatic music*"), "STT should treat music descriptions as non-dialogue")
    _assert(is_stt_non_dialogue_text("[applause]"), "STT should treat bracketed sound descriptions as non-dialogue")
    _assert(is_stt_non_dialogue_text("The End"), "STT should treat common silence hallucinations as non-dialogue")
    _assert(not is_stt_non_dialogue_text("I love music."), "STT should not drop normal dialogue containing the word music")
    removed_stt_details: list[SttRemovedSegment] = []
    dialogue_segments, removed_non_dialogue = filter_stt_dialogue_segments(
        [
            SubtitleSegment(index=1, start_ms=0, end_ms=1000, text="*Dramatic music*"),
            SubtitleSegment(index=2, start_ms=1000, end_ms=2000, text="Are you okay?"),
            SubtitleSegment(index=3, start_ms=2000, end_ms=3000, text="[applause]"),
        ],
        removed_segments=removed_stt_details,
    )
    _assert(removed_non_dialogue == 2 and len(dialogue_segments) == 1, "STT should filter obvious non-dialogue captions")
    _assert(dialogue_segments[0].index == 1 and dialogue_segments[0].text == "Are you okay?", "STT dialogue filter should reindex segments")
    _assert(len(removed_stt_details) == 2 and removed_stt_details[0].start_ms == 0, "STT should remember removed non-dialogue timestamps")
    repeated_removed_details: list[SttRemovedSegment] = []
    repeated_segments, repeated_removed = filter_repeated_stt_hallucinations(
        [
            SubtitleSegment(index=1, start_ms=152700, end_ms=153240, text="Trigger, trigger."),
            SubtitleSegment(index=2, start_ms=153240, end_ms=153540, text="Trigger, trigger."),
            SubtitleSegment(index=3, start_ms=155000, end_ms=156000, text="Real dialogue."),
        ],
        removed_segments=repeated_removed_details,
    )
    _assert(repeated_removed == 2 and len(repeated_segments) == 1, "STT should filter repeated short hallucination groups")
    _assert(repeated_segments[0].index == 1 and repeated_segments[0].text == "Real dialogue.", "STT repeated filter should reindex remaining segments")
    _assert(len(repeated_removed_details) == 2 and "trigger" in repeated_removed_details[0].reason, "STT should remember repeated hallucination timestamps")
    removed_payload = stt_removed_segments_payload(repeated_removed_details)
    _assert(removed_payload[0]["start"] == "00:02:32,700", "STT removed payload should include readable timestamps")
    removed_srt_segments = stt_removed_segments_as_srt(repeated_removed_details)
    _assert(removed_srt_segments[0].text.startswith("[powtorzony fragment: trigger]"), "STT removed SRT should include removal reason")
    stt_diag_dir = app_dir / "_self_test_stt_diag"
    _cleanup_tree(stt_diag_dir)
    stt_diag_dir.mkdir(parents=True, exist_ok=True)
    stt_diag_events = ["STT: test"]
    try:
        save_stt_diagnostics(
            output_dir=stt_diag_dir,
            temp_wav=stt_diag_dir / "missing.wav",
            source_path=app_dir / "Film testowy.mkv",
            settings=SttSettings(save_report=True, save_log=True),
            transcribe_kwargs={"beam_size": 5},
            detected_language="en",
            subtitle_profile=ENGLISH_USA_SUBTITLE_PROFILE,
            raw_segment_count=3,
            final_segment_count=1,
            duration_seconds=1.25,
            events=stt_diag_events,
            removed_segments=repeated_removed_details,
        )
        stt_report = json.loads((stt_diag_dir / "stt_report.json").read_text(encoding="utf-8"))
        _assert(stt_report["removed_segments"]["count"] == 2, "STT report should include removed postprocessing fragments")
        _assert((stt_diag_dir / "stt_removed_segments.srt").is_file(), "STT diagnostics should save removed fragments as SRT")
        _assert("00:02:32,700 --> 00:02:33,240" in (stt_diag_dir / "stt_removed_segments.srt").read_text(encoding="utf-8"), "STT removed SRT should preserve timestamps")
    finally:
        _cleanup_tree(stt_diag_dir)
    repeated_single_word_segments, repeated_single_word_removed = filter_repeated_stt_hallucinations(
        [
            SubtitleSegment(index=1, start_ms=152700, end_ms=152970, text="Trigger,"),
            SubtitleSegment(index=2, start_ms=152970, end_ms=153240, text="trigger."),
            SubtitleSegment(index=3, start_ms=153240, end_ms=153390, text="Trigger,"),
            SubtitleSegment(index=4, start_ms=153390, end_ms=153540, text="trigger."),
            SubtitleSegment(index=5, start_ms=155000, end_ms=156000, text="Real dialogue."),
        ]
    )
    _assert(repeated_single_word_removed == 4 and len(repeated_single_word_segments) == 1, "STT should filter repeated single-word hallucination groups")
    single_uncommon_word, single_uncommon_word_removed = filter_repeated_stt_hallucinations(
        [SubtitleSegment(index=1, start_ms=1000, end_ms=1600, text="Trigger.")]
    )
    _assert(single_uncommon_word_removed == 0 and len(single_uncommon_word) == 1, "STT should keep a single suspicious word")
    common_repeat, common_repeat_removed = filter_repeated_stt_hallucinations(
        [
            SubtitleSegment(index=1, start_ms=1000, end_ms=1300, text="No, no."),
            SubtitleSegment(index=2, start_ms=1300, end_ms=1600, text="No, no."),
        ]
    )
    _assert(common_repeat_removed == 0 and len(common_repeat) == 2, "STT should keep common dialogue repeats")
    short_tail_segments, short_tail_merge_count = merge_short_stt_segments(
        [
            SubtitleSegment(index=1, start_ms=162920, end_ms=164232, text="Skiff's had a vital signs on"),
            SubtitleSegment(index=2, start_ms=164232, end_ms=164420, text="her."),
            SubtitleSegment(index=3, start_ms=169280, end_ms=169986, text="I'm going to check"),
            SubtitleSegment(index=4, start_ms=169986, end_ms=170260, text="pupils."),
        ],
        profile=ENGLISH_USA_SUBTITLE_PROFILE,
    )
    formatted_short_tail = split_stt_subtitle_segments(short_tail_segments, profile=ENGLISH_USA_SUBTITLE_PROFILE)
    _assert(short_tail_merge_count == 2, "STT should merge short tail fragments")
    _assert(
        [item.text for item in formatted_short_tail]
        == ["Skiff's had a vital signs on her.", "I'm going to check pupils."],
        "STT should keep short one-word tails in the same subtitle cue",
    )
    inbound_tail_segments, inbound_tail_merge_count = merge_short_stt_segments(
        [
            SubtitleSegment(index=1, start_ms=126000, end_ms=128119, text="Get her loaded up and tell County we've got one"),
            SubtitleSegment(index=2, start_ms=128119, end_ms=128480, text="inbound."),
        ],
        profile=ENGLISH_USA_SUBTITLE_PROFILE,
    )
    _assert(inbound_tail_merge_count == 1, "STT should merge a single-word sentence tail from the next timestamp")
    _assert(
        split_stt_subtitle_segments(inbound_tail_segments, profile=ENGLISH_USA_SUBTITLE_PROFILE)[0].text
        == "Get her loaded up and tell County we've got one inbound.",
        "STT should keep a single-word sentence tail with the previous cue",
    )
    _assert(
        "\n" not in wrap_stt_subtitle_text("alpha beta gamma delta watermelon", max_line_chars=25),
        "STT output should keep each cue as a single line",
    )
    old_repeated_segments, old_repeated_removed = filter_repeated_stt_hallucinations(
        [
            SubtitleSegment(index=1, start_ms=152700, end_ms=152970, text="Trigger,"),
            SubtitleSegment(index=2, start_ms=152970, end_ms=153240, text="trigger."),
            SubtitleSegment(index=3, start_ms=153240, end_ms=153390, text="Trigger,"),
            SubtitleSegment(index=4, start_ms=155000, end_ms=156000, text="Real dialogue."),
        ]
    )
    _assert(old_repeated_removed == 3 and len(old_repeated_segments) == 1, "STT should also filter three-item repeated hallucination groups")
    merged_short_segments, short_merge_count = merge_short_stt_segments(
        [
            SubtitleSegment(index=1, start_ms=99140, end_ms=99764, text="What happened to"),
            SubtitleSegment(index=2, start_ms=99764, end_ms=99920, text="you?"),
            SubtitleSegment(index=3, start_ms=100500, end_ms=101100, text="Next sentence."),
        ],
        profile=ENGLISH_USA_SUBTITLE_PROFILE,
    )
    _assert(short_merge_count == 1, "STT should merge sentence fragments split by whisper.cpp")
    _assert(merged_short_segments[0].text == "What happened to you?", "STT should preserve merged sentence text")
    _assert(len(merged_short_segments) == 2 and merged_short_segments[0].index == 1 and merged_short_segments[1].index == 2, "STT merged segments should be reindexed")
    medical_fragments, medical_merge_count = merge_short_stt_segments(
        [
            SubtitleSegment(index=1, start_ms=194140, end_ms=195551, text="We're coming in with a Caucasian"),
            SubtitleSegment(index=2, start_ms=195551, end_ms=195860, text="female,"),
            SubtitleSegment(index=3, start_ms=196220, end_ms=197100, text="mid to late 20s,"),
            SubtitleSegment(index=4, start_ms=197400, end_ms=198160, text="in severe shock,"),
            SubtitleSegment(index=5, start_ms=198380, end_ms=198796, text="multiple"),
            SubtitleSegment(index=6, start_ms=198796, end_ms=199420, text="lacerations,"),
            SubtitleSegment(index=7, start_ms=199860, end_ms=200622, text="possible gunshot"),
            SubtitleSegment(index=8, start_ms=200622, end_ms=200860, text="wound"),
            SubtitleSegment(index=9, start_ms=200860, end_ms=201485, text="through the left"),
            SubtitleSegment(index=10, start_ms=201485, end_ms=201680, text="hand,"),
            SubtitleSegment(index=11, start_ms=201980, end_ms=202489, text="through and"),
            SubtitleSegment(index=12, start_ms=202489, end_ms=202860, text="through."),
        ],
        profile=ENGLISH_USA_SUBTITLE_PROFILE,
    )
    _assert(medical_merge_count == 11 and len(medical_fragments) == 1, "STT should rebuild heavily split whisper.cpp sentences before formatting")
    _assert("possible gunshot wound through the left hand" in medical_fragments[0].text, "STT should preserve text while rebuilding split sentence")
    medical_formatted = split_stt_subtitle_segments(medical_fragments, profile=ENGLISH_USA_SUBTITLE_PROFILE)
    _assert(len(medical_formatted) == 1, "STT should keep rebuilt complete sentences together for translation and TTS")
    complete_sentence_fragments, complete_sentence_merge_count = merge_short_stt_segments(
        [
            SubtitleSegment(index=1, start_ms=388100, end_ms=392426, text="You made it very clear a long time ago that you're not interested in being my"),
            SubtitleSegment(index=2, start_ms=392426, end_ms=392819, text="sister,"),
            SubtitleSegment(index=3, start_ms=392819, end_ms=393100, text="okay?"),
        ],
        profile=ENGLISH_USA_SUBTITLE_PROFILE,
    )
    _assert(complete_sentence_merge_count == 2 and len(complete_sentence_fragments) == 1, "STT should merge complete sentence fragments split by timestamps")
    complete_sentence_formatted = split_stt_subtitle_segments(complete_sentence_fragments, profile=ENGLISH_USA_SUBTITLE_PROFILE)
    _assert(
        [item.text for item in complete_sentence_formatted]
        == ["You made it very clear a long time ago that you're not interested in being my sister, okay?"],
        "STT should keep a complete sentence in one cue even when it is longer than subtitle display limits",
    )
    long_stt_text = "are you okay? Can you hear me? I've got someone on the south lawn. We need paramedics over here"
    split_text = split_stt_text(long_stt_text)
    _assert(
        split_text
        == [
            "are you okay?",
            "Can you hear me?",
            "I've got someone on the south lawn.",
            "We need paramedics over here",
        ],
        "STT should split long subtitle text at sentence boundaries",
    )
    long_sentence = "This is a very long subtitle line that has no punctuation and still should be split in a readable place"
    _assert(max(len(part.replace("\n", " ")) for part in split_stt_text(long_sentence)) <= 64, "STT should split long unpunctuated text")
    formatted_segments = split_stt_subtitle_segments(
        [SubtitleSegment(index=1, start_ms=1000, end_ms=9000, text=long_stt_text)]
    )
    _assert(len(formatted_segments) == 4, "STT should split one long segment into multiple SRT cues")
    _assert(formatted_segments[0].start_ms == 1000 and formatted_segments[-1].end_ms == 9000, "STT split should preserve segment bounds")
    _assert(all("\n" not in item.text for item in formatted_segments), "STT output should not use multiline cues")
    dense_text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen."
    dense_segments = split_stt_subtitle_segments(
        [SubtitleSegment(index=1, start_ms=1000, end_ms=3000, text=dense_text)],
        profile=ENGLISH_USA_SUBTITLE_PROFILE,
    )
    _assert(
        all(len(item.text.splitlines()) == 1 for item in dense_segments),
        "English STT profile should keep subtitles to one line",
    )
    high_cps_text = "One two three four five six seven eight nine ten."
    high_cps_segments = split_stt_subtitle_segments(
        [SubtitleSegment(index=1, start_ms=1000, end_ms=2000, text=high_cps_text)],
        profile=ENGLISH_USA_SUBTITLE_PROFILE,
    )
    _assert(len(high_cps_segments) == 1, "English STT profile should not split only because reading speed is high")
    messages.append("STT languages: OK")

    config_store = AppConfigStore(app_dir / "_self_test_config_filter.json")
    unsupported_path = app_dir / "_self_test_unsupported.bin"
    supported_path = app_dir / "_self_test_supported.srt"
    try:
        config_store.load()
        unsupported_path.write_text("", encoding="utf-8")
        supported_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n", encoding="utf-8")
        candidate_files = [str(unsupported_path), str(supported_path)]
        first_supported_dir = ""
        for file_path in candidate_files:
            path = Path(file_path)
            if path.is_file() and path.suffix.lower() in {*SUPPORTED_SUBTITLE_EXTENSIONS, ".mkv"}:
                first_supported_dir = str(path.resolve().parent)
                break
        if first_supported_dir:
            config_store.set_last_file_dir(first_supported_dir)
        _assert(config_store.last_file_dir() == str(app_dir.resolve()), "last file dir should use first supported file")
    finally:
        for path in (unsupported_path, supported_path, app_dir / "_self_test_config_filter.json"):
            try:
                path.unlink()
            except OSError:
                pass
    messages.append("config last dir filter: OK")

    return messages


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_start_module(path: Path):
    return _load_module(path, "_lektorai_start_self_test")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load module for self test: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", ";"):
            if separator in line:
                line = line.split(separator, 1)[0].strip()
        if line:
            names.add(line.lower().replace("_", "-"))
    return names


def _write_test_wav(path: Path, samples: list[int], sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(int(max(-32768, min(32767, sample))).to_bytes(2, "little", signed=True) for sample in samples))


def _wav_frame_count(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return int(wav.getnframes())


def _wav_sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return int(wav.getframerate())


def _wav_first_sample(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        data = wav.readframes(1)
    return int.from_bytes(data[:2], "little", signed=True) if data else 0


def _wav_sample_at(path: Path, frame_index: int) -> int:
    with wave.open(str(path), "rb") as wav:
        frame = max(0, min(int(frame_index), int(wav.getnframes()) - 1))
        wav.setpos(frame)
        data = wav.readframes(1)
    return int.from_bytes(data[:2], "little", signed=True) if data else 0


def _wav_max_abs_range(path: Path, start_frame: int, end_frame: int) -> int:
    with wave.open(str(path), "rb") as wav:
        start = max(0, min(int(start_frame), int(wav.getnframes())))
        end = max(start, min(int(end_frame), int(wav.getnframes())))
        wav.setpos(start)
        data = wav.readframes(end - start)
    values = array("h")
    values.frombytes(data)
    return max((abs(int(value)) for value in values), default=0)


def _wav_max_delta_range(path: Path, start_frame: int, end_frame: int) -> int:
    with wave.open(str(path), "rb") as wav:
        start = max(0, min(int(start_frame), int(wav.getnframes())))
        end = max(start, min(int(end_frame), int(wav.getnframes())))
        wav.setpos(start)
        data = wav.readframes(end - start)
    values = array("h")
    values.frombytes(data)
    return max((abs(int(values[index]) - int(values[index - 1])) for index in range(1, len(values))), default=0)


def _relative_app_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def _portable_scan_files(app_dir: Path) -> list[Path]:
    files = [app_dir / "START.py"]
    files.extend((app_dir / "app" / rel) for rel in sorted(_relative_app_files(app_dir / "app")))
    return [path for path in files if path.is_file()]


def _cleanup_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _cleanup_tree(child)
        else:
            try:
                child.unlink()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass




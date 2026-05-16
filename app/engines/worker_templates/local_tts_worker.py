from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import traceback
import unicodedata
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def configure_worker_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_worker_stdio()


_WHISPER_MODELS: dict[tuple[str, str, str, str, str], Any] = {}
_AUTO_DEVICE = ""
_OMNIVOICE_MODEL = None
_OMNIVOICE_MODEL_KEY = ""
_PIPER_VOICE = None
_PIPER_VOICE_KEY = ""
_COQUI_XTTS_MODEL = None
_COQUI_XTTS_MODEL_KEY = ""
DETERMINISTIC_RETRY_ENGINES = {"piper"}
PUNCTUATION_RETRY_ENGINES = {"piper"}
RETRY_TERMINAL_PUNCTUATION = ".,!?:;"


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: worker.py request.json result.json")
        return 2
    request_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    request: dict[str, Any] = {}
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = run_request(request)
    except Exception as exc:
        traceback.print_exc()
        result = {
            "engine_id": str(request.get("engine_id", "")) if isinstance(request, dict) else "",
            "job_id": str(request.get("job_id", "")) if isinstance(request, dict) else "",
            "ok": False,
            "segments": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.get("ok") else 1


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    engine_id = str(request.get("engine_id", ""))
    job_id = str(request.get("job_id", ""))
    settings = apply_engine_preset(engine_id, dict(request.get("settings", {}) or {}))
    segments = list(request.get("segments", []) or [])
    setup_cache(engine_id)

    results: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = int(segment.get("segment_id", 0))
        text = str(segment.get("text", "") or "")
        output_path = str(segment.get("output_path", "") or "")
        subtitle_ms = max(0, int(segment.get("end_ms", 0)) - int(segment.get("start_ms", 0)))
        try:
            generation = synthesize_with_retry(engine_id, text, output_path, settings, subtitle_ms, segment_id)
            results.append(
                {
                    "segment_id": segment_id,
                    "ok": True,
                    "output_path": output_path,
                    "error": "",
                    "duration_ms": media_duration_ms(output_path),
                    "retries": generation["retries"],
                    "qc_score": generation["qc_score"],
                    "attempts": generation["attempts"],
                    "selected_attempt": generation["selected_attempt"],
                    "qc_warnings": generation["qc_warnings"],
                    "whisper_text": generation.get("whisper_text", ""),
                    "whisper_similarity": generation.get("whisper_similarity"),
                    "attempt_details": generation.get("attempt_details", []),
                }
            )
            print(
                f"{engine_id}: segment {segment_id} OK, "
                f"proba={generation['selected_attempt']}/{generation['attempts']}, "
                f"qc={generation['qc_score']}"
            )
        except Exception as exc:
            traceback.print_exc()
            results.append(
                {
                    "segment_id": segment_id,
                    "ok": False,
                    "output_path": output_path,
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_ms": 0,
                    "retries": 0,
                    "qc_score": None,
                    "attempts": 0,
                    "selected_attempt": 0,
                    "qc_warnings": [],
                    "whisper_text": "",
                    "whisper_similarity": None,
                }
            )
            break

    ok = all(bool(item.get("ok")) for item in results) and len(results) == len(segments)
    return {"engine_id": engine_id, "job_id": job_id, "ok": ok, "segments": results, "error": "" if ok else "Worker failed"}


def synthesize_with_retry(
    engine_id: str,
    text: str,
    output_path: str,
    settings: dict[str, Any],
    subtitle_ms: int,
    segment_id: int,
) -> dict[str, Any]:
    audio_limit, whisper_limit, audio_enabled, whisper_enabled = effective_retry_settings(engine_id, settings)
    output = Path(output_path)
    candidate_dir = output.parent / "_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    attempt_details: list[dict[str, Any]] = []
    base_seed = _optional_int(settings.get("seed"))
    candidate_no = 0
    retry_texts = retry_text_variants(engine_id, text, max(1, audio_limit * whisper_limit))

    try:
        for whisper_attempt in range(1, whisper_limit + 1):
            if candidate_no >= len(retry_texts):
                break
            audio_best: dict[str, Any] | None = None
            for audio_attempt in range(1, audio_limit + 1):
                if candidate_no >= len(retry_texts):
                    break
                candidate_no += 1
                candidate_path = output if candidate_no == 1 else candidate_dir / f"{output.stem}_try{candidate_no}{output.suffix}"
                candidate_settings = dict(settings)
                if base_seed is not None:
                    candidate_settings["seed"] = int(base_seed) if candidate_no == 1 else int(base_seed) + (candidate_no * 10007) + int(segment_id)

                synthesize_once(engine_id, retry_texts[candidate_no - 1], str(candidate_path), candidate_settings)
                qc = audio_qc(candidate_path, subtitle_ms) if audio_enabled else audio_qc_ok(candidate_path)
                record = {"attempt": candidate_no, "path": candidate_path, "qc": qc}
                candidates.append(record)
                attempt_detail = {
                    "attempt": int(candidate_no),
                    "audio_attempt": int(audio_attempt),
                    "speech_attempt": int(whisper_attempt),
                    "text_variant": retry_variant_label(engine_id, retry_texts[candidate_no - 1]),
                    "audio_file": candidate_path.name,
                    "audio_qc_score": int(qc["score"]),
                    "audio_qc_warnings": list(qc["warnings"]),
                    "whisper_checked": False,
                    "whisper_score": None,
                    "whisper_similarity": None,
                    "whisper_text": "",
                    "final_score": int(qc["score"]),
                    "final_warnings": list(qc["warnings"]),
                    "selected": False,
                }
                record["attempt_detail"] = attempt_detail
                attempt_details.append(attempt_detail)
                print(
                    f"{engine_id}: segment {segment_id}, audio proba {audio_attempt}/{audio_limit}, "
                    f"qc={qc['score']}, ostrzezenia={','.join(qc['warnings']) or '-'}"
                )
                if audio_best is None or int(qc["score"]) < int(audio_best["qc"]["score"]):
                    audio_best = record
                if int(qc["score"]) == 0:
                    break

            if audio_best is None:
                continue
            qc = dict(audio_best["qc"])
            variant_label = retry_variant_label(engine_id, retry_texts[int(audio_best["attempt"]) - 1])
            if whisper_enabled:
                whisper = whisper_qc(Path(audio_best["path"]), text, settings)
                qc["score"] = int(qc["score"]) + int(whisper["score"])
                qc["warnings"] = list(qc["warnings"]) + list(whisper["warnings"])
                qc["whisper_text"] = whisper["text"]
                qc["whisper_similarity"] = whisper["similarity"]
                audio_best["qc"] = qc
                detail = audio_best.get("attempt_detail")
                if isinstance(detail, dict):
                    detail["whisper_checked"] = True
                    detail["whisper_score"] = int(whisper["score"])
                    detail["whisper_similarity"] = whisper["similarity"]
                    detail["whisper_text"] = str(whisper["text"])
                    detail["final_score"] = int(qc["score"])
                    detail["final_warnings"] = list(qc["warnings"])
            final_candidates.append(audio_best)
            print(
                f"{engine_id}: segment {segment_id}, mowa proba {whisper_attempt}/{whisper_limit}, "
                f"wariant={variant_label}, qc={qc['score']}, ostrzezenia={','.join(qc['warnings']) or '-'}"
            )
            if int(qc["score"]) == 0:
                break

        best = min(final_candidates or candidates, key=lambda item: int(item["qc"]["score"]))
        best_detail = best.get("attempt_detail")
        if isinstance(best_detail, dict):
            best_detail["selected"] = True
        if Path(best["path"]) != output:
            if output.exists():
                output.unlink()
            shutil.move(str(best["path"]), str(output))
        for candidate in candidates:
            path = Path(candidate["path"])
            if path != output and path.exists():
                path.unlink(missing_ok=True)
        return {
            "attempts": len(candidates),
            "selected_attempt": int(best["attempt"]),
            "retries": max(0, len(candidates) - 1),
            "qc_score": int(best["qc"]["score"]),
            "qc_warnings": list(best["qc"]["warnings"]),
            "whisper_text": str(best["qc"].get("whisper_text", "")),
            "whisper_similarity": best["qc"].get("whisper_similarity"),
            "attempt_details": attempt_details,
        }
    finally:
        try:
            if candidate_dir.exists() and not any(candidate_dir.iterdir()):
                candidate_dir.rmdir()
        except Exception:
            pass


def synthesize_once(engine_id: str, text: str, output_path: str, settings: dict[str, Any]) -> None:
    if engine_id == "chatterbox":
        synthesize_chatterbox(text, output_path, settings)
    elif engine_id == "omnivoice":
        synthesize_omnivoice(text, output_path, settings)
    elif engine_id == "piper":
        synthesize_piper(text, output_path, settings)
    elif engine_id == "coqui_xtts":
        synthesize_coqui_xtts(text, output_path, settings)
    else:
        raise RuntimeError(f"Unsupported engine: {engine_id}")


def default_retry_attempts(engine_id: str) -> int:
    if engine_id == "omnivoice":
        return 2
    if engine_id == "chatterbox":
        return 3
    return 1


def effective_retry_settings(engine_id: str, settings: dict[str, Any]) -> tuple[int, int, bool, bool]:
    audio_enabled = bool_setting(settings.get("audio_qc_enabled"), False)
    whisper_enabled = bool_setting(settings.get("whisper_qc_enabled"), False)
    audio_attempts = bounded_int(settings.get("audio_qc_retry_attempts", default_retry_attempts(engine_id)), 1, 5)
    whisper_attempts = bounded_int(settings.get("whisper_qc_retry_attempts", 1), 1, 5)
    audio_limit = audio_attempts if audio_enabled else 1
    whisper_limit = whisper_attempts if whisper_enabled else 1
    if engine_id in DETERMINISTIC_RETRY_ENGINES:
        if engine_id in PUNCTUATION_RETRY_ENGINES:
            return 1, whisper_limit, audio_enabled, whisper_enabled
        return 1, 1, audio_enabled, whisper_enabled
    return audio_limit, whisper_limit, audio_enabled, whisper_enabled


def effective_retry_limits(engine_id: str, settings: dict[str, Any]) -> tuple[int, int]:
    audio_limit, whisper_limit, _audio_enabled, _whisper_enabled = effective_retry_settings(engine_id, settings)
    return audio_limit, whisper_limit


def retry_text_for_attempt(engine_id: str, text: str, attempt: int) -> str:
    variants = retry_text_variants(engine_id, text, max(1, int(attempt)))
    index = min(max(1, int(attempt)) - 1, len(variants) - 1)
    return variants[index]


def retry_text_variants(engine_id: str, text: str, limit: int) -> list[str]:
    limit = max(1, int(limit))
    if engine_id not in PUNCTUATION_RETRY_ENGINES:
        return [str(text or "") for _ in range(limit)]
    stripped = str(text or "").strip()
    if not stripped:
        return [str(text or "")]
    base = strip_retry_terminal_punctuation(stripped)
    if not base:
        return [stripped]
    return [stripped, base + ".", base + ",", base + "!", base + "?"][:limit]


def strip_retry_terminal_punctuation(text: str) -> str:
    base = str(text or "").strip()
    while base and base[-1] in RETRY_TERMINAL_PUNCTUATION:
        base = base[:-1].rstrip()
    return base


def retry_variant_label(engine_id: str, text: str) -> str:
    if engine_id not in PUNCTUATION_RETRY_ENGINES:
        return "oryginal"
    stripped = str(text or "").strip()
    if stripped.endswith("."):
        return "kropka"
    if stripped.endswith(","):
        return "przecinek"
    if stripped.endswith("!"):
        return "wykrzyknik"
    if stripped.endswith("?"):
        return "pytajnik"
    return "oryginal"


def apply_engine_preset(engine_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    return dict(settings)


def setup_cache(engine_id: str) -> None:
    engine_dir = Path.cwd()
    cache_dir = engine_dir / "cache"
    hf_dir = cache_dir / "hf"
    transformers_dir = cache_dir / "transformers"
    for path in (cache_dir, hf_dir, transformers_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir)
    if engine_id == "chatterbox":
        pkuseg_dir = cache_dir / "pkuseg"
        pkuseg_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PKUSEG_HOME"] = str(pkuseg_dir)
        os.environ["HOME"] = str(cache_dir)
        os.environ["USERPROFILE"] = str(cache_dir)
    if engine_id == "omnivoice":
        os.environ["HF_HOME"] = str(cache_dir)
    if engine_id == "coqui_xtts":
        coqui_dir = cache_dir / "coqui"
        coqui_dir.mkdir(parents=True, exist_ok=True)
        os.environ["COQUI_TOS_AGREED"] = "1"
        os.environ["TTS_HOME"] = str(coqui_dir)
        os.environ["XDG_DATA_HOME"] = str(coqui_dir)


def has_local_model_cache() -> bool:
    cache_dir = Path.cwd() / "cache"
    if not cache_dir.exists():
        return False
    ignored_roots = {"whisper", "pkuseg"}
    for path in cache_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(cache_dir)
        except ValueError:
            continue
        if relative.parts and relative.parts[0].lower() in ignored_roots:
            continue
        return True
    return False


def media_duration_ms(path: str) -> int:
    try:
        import soundfile as sf

        info = sf.info(path)
        if int(info.samplerate) <= 0:
            return 0
        return int(round((int(info.frames) / int(info.samplerate)) * 1000))
    except Exception:
        return 0


def audio_qc(path: Path, subtitle_ms: int) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    data = np.asarray(wav, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration_ms = int(round((len(data) / int(sample_rate)) * 1000)) if int(sample_rate) > 0 else 0
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float32), dtype=np.float32))) if data.size else 0.0
    leading_db = window_db(data, sample_rate, 0, 250)
    trailing_db = window_db(data, sample_rate, max(0, duration_ms - 250), duration_ms)
    clipped_ratio = float(np.mean(np.abs(data) >= 0.999)) if data.size else 0.0

    warnings: list[str] = []
    if data.size == 0 or duration_ms <= 0:
        warnings.append("pusty segment")
    if data.size and db(rms) < -55.0:
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

    return {
        "duration_ms": duration_ms,
        "peak_db": db(peak),
        "rms_db": db(rms),
        "leading_db": leading_db,
        "trailing_db": trailing_db,
        "clipped_ratio": clipped_ratio,
        "warnings": warnings,
        "score": score_warnings(warnings),
    }


def audio_qc_ok(path: Path) -> dict[str, Any]:
    return {
        "duration_ms": media_duration_ms(str(path)),
        "peak_db": 0.0,
        "rms_db": 0.0,
        "leading_db": 0.0,
        "trailing_db": 0.0,
        "clipped_ratio": 0.0,
        "warnings": [],
        "score": 0,
    }


def window_db(data, sample_rate: int, start_ms: int, end_ms: int) -> float:
    import numpy as np

    if data.size == 0 or sample_rate <= 0 or end_ms <= start_ms:
        return -120.0
    start = max(0, int(start_ms * sample_rate / 1000))
    end = min(len(data), int(end_ms * sample_rate / 1000))
    if end <= start:
        return -120.0
    window = data[start:end]
    rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float32), dtype=np.float32))) if window.size else 0.0
    return db(rms)


def db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(max(value, 1e-12))


def score_warnings(warnings: list[str]) -> int:
    score = 0
    for warning in warnings:
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


def whisper_qc(path: Path, expected_text: str, settings: dict[str, Any]) -> dict[str, Any]:
    transcript = transcribe_with_faster_whisper(path, settings)
    similarity = text_similarity(expected_text, transcript)
    threshold = bounded_float(settings.get("whisper_qc_min_similarity", 0.62), 0.0, 1.0)
    penalty = whisper_similarity_penalty(similarity, threshold)
    warnings = []
    if penalty:
        warnings.append("whisper niezgodny")
    if not normalize_for_whisper_qc(transcript):
        warnings.append("whisper pusty")
        penalty = max(penalty, 85)
    return {
        "text": transcript,
        "similarity": round(float(similarity), 4),
        "score": int(penalty),
        "warnings": warnings,
    }


def transcribe_with_faster_whisper(path: Path, settings: dict[str, Any]) -> str:
    ensure_faster_whisper_packages_available()
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        if exc.name == "faster_whisper":
            raise RuntimeError("Whisper QC: brak silnika STT faster-whisper. Zainstaluj faster-whisper albo wylacz Whisper QC.") from exc
        raise RuntimeError(f"Whisper QC: brak zaleznosci {exc.name}. Zainstaluj faster-whisper albo wylacz Whisper QC.") from exc

    model_name = str(settings.get("whisper_qc_model", "small") or "small").strip() or "small"
    language = str(settings.get("whisper_qc_language", "pl") or "pl").strip() or "pl"
    device = str(settings.get("whisper_qc_device", "cpu") or "cpu").strip() or "cpu"
    compute_type = str(settings.get("whisper_qc_compute_type", "int8") or "int8").strip() or "int8"
    whisper_device, whisper_device_index = faster_whisper_device_args(device)
    cache_dir = common_whisper_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (model_name, whisper_device, str(whisper_device_index), compute_type, str(cache_dir))
    model = _WHISPER_MODELS.get(key)
    if model is None:
        device_label = faster_whisper_device_label(whisper_device, whisper_device_index)
        print(f"whisper qc: sprawdzanie modelu {model_name} na {device_label}")
        if has_whisper_model_cache(cache_dir, model_name):
            print(f"whisper qc: model w cache {model_name}")
            print(f"whisper qc: ladowanie modelu {model_name} na {device_label}")
        else:
            print(f"whisper qc: pobieranie modelu {model_name} na {device_label}")
        model = WhisperModel(
            model_name,
            device=whisper_device,
            device_index=whisper_device_index,
            compute_type=compute_type,
            download_root=str(cache_dir),
        )
        _WHISPER_MODELS[key] = model
    segments, _info = model.transcribe(
        str(path),
        language=language,
        task="transcribe",
        beam_size=1,
        vad_filter=False,
        word_timestamps=False,
        without_timestamps=True,
    )
    return " ".join(str(segment.text or "").strip() for segment in segments).strip()


def faster_whisper_device_args(device: str) -> tuple[str, int]:
    normalized = str(device or "cpu").strip().lower()
    match = re.fullmatch(r"cuda:(\d+)", normalized)
    if match:
        return "cuda", int(match.group(1))
    if normalized == "cuda":
        return "cuda", 0
    return "cpu", 0


def faster_whisper_device_label(device: str, device_index: int) -> str:
    if device == "cuda":
        return f"cuda:{int(device_index)}"
    return device


def text_similarity(expected_text: str, transcript: str) -> float:
    expected = normalize_for_whisper_qc(expected_text)
    actual = normalize_for_whisper_qc(transcript)
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    return round(float(SequenceMatcher(None, expected, actual).ratio()), 4)


def common_whisper_cache_dir() -> Path:
    override = str(os.environ.get("LEKTORAI_STT_FASTER_WHISPER_CACHE_DIR", "") or "").strip()
    if not override:
        override = str(os.environ.get("LEKTORAI_WHISPER_CACHE_DIR", "") or "").strip()
    if override:
        return Path(override)
    return Path.cwd() / "cache" / "whisper"


def ensure_faster_whisper_packages_available() -> None:
    raw_dirs = str(os.environ.get("LEKTORAI_STT_FASTER_WHISPER_PACKAGES_DIRS", "") or "").strip()
    package_dirs = [item for item in raw_dirs.split(os.pathsep) if item.strip()]
    legacy_dir = str(os.environ.get("LEKTORAI_APP_PACKAGES_DIR", "") or "").strip()
    if legacy_dir:
        package_dirs.append(legacy_dir)
    for packages_dir in reversed(package_dirs):
        packages_path = Path(packages_dir)
        if not packages_path.is_dir():
            continue
        packages_text = str(packages_path)
        if packages_text not in sys.path:
            sys.path.insert(0, packages_text)


def has_whisper_model_cache(cache_dir: Path, model_name: str) -> bool:
    if not cache_dir.is_dir():
        return False
    candidates = whisper_cache_name_candidates(model_name)
    for child in cache_dir.iterdir():
        if not child.is_dir():
            continue
        child_name = child.name.lower()
        if child_name in candidates or any(child_name.endswith(candidate) for candidate in candidates):
            if any(path.is_file() for path in child.rglob("*")):
                return True
    return False


def whisper_cache_name_candidates(model_name: str) -> set[str]:
    model = str(model_name or "").strip().lower().replace("/", "--")
    aliases = {model}
    if model == "large":
        aliases.add("large-v3")
    if model == "turbo":
        aliases.add("large-v3-turbo")
    candidates: set[str] = set()
    for alias in aliases:
        candidates.add(f"models--systran--faster-whisper-{alias}")
        candidates.add(f"models--mobiuslabsgmbh--faster-whisper-{alias}")
        candidates.add(f"models--openai--whisper-{alias}")
        candidates.add(f"--faster-whisper-{alias}")
        candidates.add(f"--whisper-{alias}")
    if "--" in model:
        candidates.add(f"models--{model}")
    return candidates


def whisper_similarity_penalty(similarity: float, threshold: float) -> int:
    similarity = bounded_float(similarity, 0.0, 1.0)
    threshold = bounded_float(threshold, 0.0, 1.0)
    if similarity >= threshold:
        return 0
    gap = threshold - similarity
    return max(15, min(95, int(round(gap * 140))))


def normalize_for_whisper_qc(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").casefold())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", without_marks)).strip()


def write_audio(path: str, wav, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(wav, "detach"):
        wav_np = wav.detach().float().cpu().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32)
    wav_np = np.asarray(wav_np, dtype=np.float32)
    if wav_np.ndim > 1:
        wav_np = wav_np.reshape(-1)
    sf.write(str(output_path), wav_np, int(sample_rate))


def best_device(requested: str = "auto") -> str:
    global _AUTO_DEVICE
    requested = (requested or "auto").strip().lower()
    if requested not in {"auto", "cuda"}:
        return requested
    if _AUTO_DEVICE:
        return _AUTO_DEVICE
    try:
        import torch

        if not torch.cuda.is_available():
            _AUTO_DEVICE = "cpu"
            return _AUTO_DEVICE
        best_idx = 0
        best_free = -1
        for idx in range(int(torch.cuda.device_count())):
            free_b, _total_b = torch.cuda.mem_get_info(idx)
            if int(free_b) > best_free:
                best_free = int(free_b)
                best_idx = idx
        _AUTO_DEVICE = f"cuda:{best_idx}"
        return _AUTO_DEVICE
    except Exception:
        _AUTO_DEVICE = "cpu"
        return _AUTO_DEVICE


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


_CHATTERBOX_MODEL = None
_CHATTERBOX_MODEL_KEY = ""


class _NoopWatermarker:
    def apply_watermark(self, wav, sample_rate=None):
        return wav


def _patch_chatterbox_watermarker() -> None:
    try:
        import chatterbox.mtl_tts as mtl_tts

        if not callable(getattr(mtl_tts.perth, "PerthImplicitWatermarker", None)):
            mtl_tts.perth.PerthImplicitWatermarker = _NoopWatermarker
    except Exception:
        pass


def synthesize_chatterbox(text: str, output_path: str, settings: dict[str, Any]) -> None:
    global _CHATTERBOX_MODEL, _CHATTERBOX_MODEL_KEY
    if not text.strip():
        raise RuntimeError("Empty text")
    import numpy as np
    import soundfile as sf
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = best_device(str(settings.get("device", "auto")))
    t3_model = str(settings.get("t3_model", "v2") or "v2").strip() or "v2"
    model_key = f"{device}|t3={t3_model}"
    if _CHATTERBOX_MODEL is None or _CHATTERBOX_MODEL_KEY != model_key:
        _patch_chatterbox_watermarker()
        print(f"chatterbox: sprawdzanie modelu t3={t3_model} na {device}")
        if has_local_model_cache():
            print(f"chatterbox: model w cache t3={t3_model}")
            print(f"chatterbox: ladowanie modelu t3={t3_model} na {device}")
        else:
            print(f"chatterbox: pobieranie modelu t3={t3_model} na {device}")
        _CHATTERBOX_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=t3_model)
        try:
            if getattr(_CHATTERBOX_MODEL, "watermarker", None) is not None:
                _CHATTERBOX_MODEL.watermarker = _NoopWatermarker()
        except Exception:
            pass
        _CHATTERBOX_MODEL_KEY = model_key

    seed_everything(_optional_int(settings.get("seed")))
    prompt = str(settings.get("audio_prompt_path", "") or "").strip()
    if prompt:
        if not Path(prompt).is_file():
            raise RuntimeError(f"Missing Chatterbox voice sample: {prompt}")
        _CHATTERBOX_MODEL.prepare_conditionals(prompt, exaggeration=float(settings.get("exaggeration", 0.5)))
    wav = _CHATTERBOX_MODEL.generate(
        text=text,
        language_id="pl",
        cfg_weight=float(settings.get("cfg_weight", 0.5)),
        exaggeration=float(settings.get("exaggeration", 0.5)),
        temperature=float(settings.get("temperature", 0.8)),
        repetition_penalty=float(settings.get("repetition_penalty", 1.2)),
        min_p=float(settings.get("min_p", 0.05)),
        top_p=float(settings.get("top_p", 1.0)),
    )
    sample_rate = int(getattr(_CHATTERBOX_MODEL, "sr", 24000))
    if hasattr(wav, "detach"):
        wav_np = wav.detach().float().cpu().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32)
    wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
    if bool(settings.get("trim_leading_silence", True)):
        wav_np = trim_leading_low_energy(
            wav_np,
            sample_rate,
            max_trim_ms=700,
            threshold_db=-40.0,
            pre_roll_ms=50,
            consecutive_frames=3,
        )
        wav_np = apply_fade_in(wav_np, sample_rate, fade_ms=10)
    wav_np = trim_tail(wav_np, sample_rate)
    sf.write(output_path, wav_np, sample_rate)


def synthesize_omnivoice(text: str, output_path: str, settings: dict[str, Any]) -> None:
    global _OMNIVOICE_MODEL, _OMNIVOICE_MODEL_KEY
    if not text.strip():
        raise RuntimeError("Empty text")
    import numpy as np
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    device = best_device(str(settings.get("device", "auto")))
    model_id = str(settings.get("model_id", "k2-fsa/OmniVoice") or "k2-fsa/OmniVoice").strip() or "k2-fsa/OmniVoice"
    model_key = f"{model_id}|{device}"
    if _OMNIVOICE_MODEL is None or _OMNIVOICE_MODEL_KEY != model_key:
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        print(f"omnivoice: sprawdzanie modelu {model_id} na {device}")
        if has_local_model_cache():
            print(f"omnivoice: model w cache {model_id}")
            print(f"omnivoice: ladowanie modelu {model_id} na {device}")
        else:
            print(f"omnivoice: pobieranie modelu {model_id} na {device}")
        _OMNIVOICE_MODEL = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
        _OMNIVOICE_MODEL_KEY = model_key

    ref_audio = str(settings.get("reference_audio_path", "") or "").strip()
    if not ref_audio:
        raise RuntimeError("OmniVoice wymaga probki glosu do trybu lektora.")
    if not Path(ref_audio).is_file():
        raise RuntimeError(f"Missing OmniVoice voice sample: {ref_audio}")
    ref_text = str(settings.get("reference_text", "") or "").strip() or None
    if ref_text is None:
        print("omnivoice: brak tekstu probki, model moze pobrac/uzyc ASR do transkrypcji probki")

    audio = _OMNIVOICE_MODEL.generate(
        text=text,
        language="pl",
        ref_audio=ref_audio,
        ref_text=ref_text,
        num_step=int(settings.get("num_step", 32)),
        guidance_scale=float(settings.get("guidance_scale", 2.0)),
        speed=float(settings.get("speed", 1.0)),
        denoise=bool(settings.get("denoise", True)),
        preprocess_prompt=False,
        postprocess_output=False,
    )
    wav_np = np.asarray(audio[0], dtype=np.float32).reshape(-1)
    sample_rate = int(getattr(_OMNIVOICE_MODEL, "sampling_rate", 24000) or 24000)
    if bool(settings.get("omnivoice_trim_edges", False)):
        wav_np = trim_omnivoice_silence_edges_np(wav_np, sample_rate)
    sf.write(output_path, wav_np, sample_rate)


def synthesize_piper(text: str, output_path: str, settings: dict[str, Any]) -> None:
    global _PIPER_VOICE, _PIPER_VOICE_KEY
    if not text.strip():
        raise RuntimeError("Empty text")
    from piper import PiperVoice, SynthesisConfig
    from piper.download_voices import download_voice

    voice_id = str(settings.get("voice", "pl_PL-gosia-medium") or "pl_PL-gosia-medium").strip() or "pl_PL-gosia-medium"
    models_dir = Path.cwd() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{voice_id}.onnx"
    config_path = models_dir / f"{voice_id}.onnx.json"
    if _PIPER_VOICE is None or _PIPER_VOICE_KEY != voice_id:
        print(f"piper: sprawdzanie modelu {voice_id}")
        if not model_path.exists() or not config_path.exists():
            print(f"piper: pobieranie modelu {voice_id}")
            download_voice(voice_id, models_dir)
        else:
            print(f"piper: model w cache {voice_id}")
        print(f"piper: ladowanie modelu {voice_id}")
        _PIPER_VOICE = PiperVoice.load(str(model_path), use_cuda=False)
        _PIPER_VOICE_KEY = voice_id

    syn_config = SynthesisConfig(
        speaker_id=bounded_int(settings.get("speaker_id", 0), 0, 999),
        length_scale=bounded_float(settings.get("length_scale", 1.0), 0.5, 2.0),
        noise_scale=bounded_float(settings.get("noise_scale", 0.667), 0.0, 1.5),
        noise_w_scale=bounded_float(settings.get("noise_w_scale", 0.8), 0.0, 1.5),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        _PIPER_VOICE.synthesize_wav(text, wav_file, syn_config=syn_config)


def synthesize_coqui_xtts(text: str, output_path: str, settings: dict[str, Any]) -> None:
    global _COQUI_XTTS_MODEL, _COQUI_XTTS_MODEL_KEY
    if not text.strip():
        raise RuntimeError("Empty text")
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS

    device = best_device(str(settings.get("device", "auto")))
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_key = f"{model_name}|{device}"
    if _COQUI_XTTS_MODEL is None or _COQUI_XTTS_MODEL_KEY != model_key:
        print(f"coqui_xtts: sprawdzanie modelu XTTS-v2 na {device}")
        if has_local_model_cache():
            print("coqui_xtts: model w cache XTTS-v2")
            print(f"coqui_xtts: ladowanie modelu XTTS-v2 na {device}")
        else:
            print(f"coqui_xtts: pobieranie modelu XTTS-v2 na {device}")
        _COQUI_XTTS_MODEL = TTS(model_name=model_name, progress_bar=True).to(device)
        _COQUI_XTTS_MODEL_KEY = model_key

    speaker_wav = str(settings.get("speaker_wav_path", "") or "").strip()
    speaker = str(settings.get("speaker", "Anna") or "Anna").strip() or "Anna"
    speed_key = "voice_sample_speed" if speaker_wav else "builtin_voice_speed"
    speed_default = 1.3 if speaker_wav else 1.6
    speed_value = settings.get(speed_key, settings.get("speed", speed_default))
    kwargs = {
        "text": text,
        "language": "pl",
        "file_path": str(output_path),
        "split_sentences": False,
        "temperature": bounded_float(settings.get("temperature", 0.1), 0.05, 1.5),
        "length_penalty": bounded_float(settings.get("length_penalty", 1.0), 0.0, 2.0),
        "repetition_penalty": bounded_float(settings.get("repetition_penalty", 9.0), 1.0, 12.0),
        "top_k": bounded_int(settings.get("top_k", 100), 0, 100),
        "top_p": bounded_float(settings.get("top_p", 1.0), 0.05, 1.0),
        "speed": bounded_float(speed_value, 0.5, 2.0),
    }
    if speaker_wav:
        if not Path(speaker_wav).is_file():
            raise RuntimeError(f"Missing Coqui XTTS voice sample: {speaker_wav}")
        kwargs["speaker_wav"] = speaker_wav
    else:
        kwargs["speaker"] = speaker
    _COQUI_XTTS_MODEL.tts_to_file(**kwargs)
    if bool(settings.get("xtts_trim_trailing_silence", True)):
        import soundfile as sf

        wav, sample_rate = sf.read(str(output_path), dtype="float32", always_2d=False)
        trimmed = trim_xtts_trailing_silence_np(wav, int(sample_rate))
        sf.write(str(output_path), trimmed, int(sample_rate))


def trim_xtts_trailing_silence_np(wav, sample_rate: int):
    import numpy as np

    data = np.asarray(wav, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.reshape(-1)
    if data.size <= 0 or sample_rate <= 0:
        return data

    frame_ms = 20
    frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    frame_count = data.size // frame_size
    if frame_count < 8:
        return data

    frames = data[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-9)
    threshold = -50.0
    active = (db >= threshold).astype(np.int32)
    smooth = np.convolve(active, np.ones(3, dtype=np.int32), mode="same") >= 2
    indexes = np.where(smooth)[0]
    if indexes.size <= 0:
        return data

    post_pad_ms = 120
    end = min(data.size, int(round(((int(indexes[-1]) + 1) * frame_ms + post_pad_ms) * sample_rate / 1000.0)))
    if end >= data.size - frame_size:
        return data
    if end < int(round(sample_rate * 0.25)):
        return data

    trimmed = np.array(data[:end], dtype=np.float32, copy=True)
    fade_ms = 12
    fade = min(max(0, int(round(sample_rate * fade_ms / 1000.0))), trimmed.size // 2)
    if fade > 1:
        trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return trimmed


def trim_omnivoice_silence_edges_np(wav, sample_rate: int):
    import numpy as np

    data = np.asarray(wav, dtype=np.float32).reshape(-1)
    if data.size <= 0 or sample_rate <= 0:
        return data

    frame_ms = 20
    frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    frame_count = data.size // frame_size
    if frame_count < 8:
        return data

    frames = data[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-9)
    floor = float(np.percentile(db, 20))

    # OmniVoice miewa glosny szum/wdech na poczatku, ale ciche koncowki mowy.
    # Dlatego start wykrywamy ostrzej, a koniec lagodniej i zawsze zostawiamy zapas.
    start_threshold = max(floor + 18.0, -52.0)
    end_threshold = max(floor + 10.0, -55.0)

    def active_span(threshold: float):
        active = (db > threshold).astype(np.int32)
        smooth = np.convolve(active, np.ones(5, dtype=np.int32), mode="same") >= 2
        indexes = np.where(smooth)[0]
        if indexes.size <= 0:
            return None
        return int(indexes[0]), int(indexes[-1])

    start_span = active_span(start_threshold)
    end_span = active_span(end_threshold)
    if start_span is None or end_span is None:
        return data

    pre_pad_ms = 50
    post_pad_ms = 100
    start = max(0, int(round((start_span[0] * frame_ms - pre_pad_ms) * sample_rate / 1000.0)))
    end = min(data.size, int(round(((end_span[1] + 1) * frame_ms + post_pad_ms) * sample_rate / 1000.0)))
    if end - start < int(round(sample_rate * 0.25)):
        return data

    trimmed = np.array(data[start:end], dtype=np.float32, copy=True)
    if trimmed.size <= 1:
        return trimmed
    fade_ms = 12
    fade = max(0, int(round(sample_rate * max(0, int(fade_ms)) / 1000.0)))
    fade = min(fade, trimmed.size // 2)
    if fade > 1:
        if start > 0:
            trimmed[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if end < data.size:
            trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return trimmed


def trim_tail(wav, sample_rate: int):
    import numpy as np

    if wav.size <= 0 or sample_rate <= 0:
        return wav
    frame_len = max(1, int(round(sample_rate * 0.01)))
    frames = int(np.ceil(len(wav) / frame_len))
    if frames <= 2:
        return wav
    voiced = []
    for i in range(frames):
        fr = wav[i * frame_len : min(len(wav), (i + 1) * frame_len)]
        rms = float(np.sqrt(np.mean(np.square(fr, dtype=np.float32), dtype=np.float32))) if fr.size else 0.0
        db = -120.0 if rms <= 0 else 20.0 * np.log10(max(rms, 1e-12))
        voiced.append(db >= -38.0)
    last = 0
    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            last = i
    cut = min(len(wav), (last + 4) * frame_len)
    return wav[:cut] if cut > 0 else wav


def audio_to_numpy(wav):
    import numpy as np

    if hasattr(wav, "detach"):
        wav_np = wav.detach().float().cpu().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32)
    return np.asarray(wav_np, dtype=np.float32).reshape(-1)


def cleanup_generated_edges(wav, sample_rate: int, trim_leading: bool = False):
    import numpy as np

    data = np.asarray(wav, dtype=np.float32).reshape(-1)
    if data.size <= 0 or sample_rate <= 0:
        return data
    if trim_leading:
        data = trim_leading_low_energy(data, sample_rate, max_trim_ms=280)
    data = trim_tail(data, sample_rate)
    data = apply_short_fades(data, sample_rate, fade_ms=18)
    return data


def trim_leading_low_energy(
    wav,
    sample_rate: int,
    max_trim_ms: int,
    threshold_db: float = -42.0,
    pre_roll_ms: int = 10,
    consecutive_frames: int = 1,
):
    import numpy as np

    if wav.size <= 0:
        return wav
    frame_len = max(1, int(round(sample_rate * 0.01)))
    max_frames = max(1, int(max_trim_ms / 10))
    search_end = min(wav.size, max_frames * frame_len)
    frame_count = int(np.ceil(search_end / frame_len))
    frame_db = []
    for index in range(frame_count):
        start = index * frame_len
        frame = wav[start : min(wav.size, start + frame_len)]
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32), dtype=np.float32))) if frame.size else 0.0
        frame_db.append(db(rms))
    first_voice = 0
    needed = max(1, int(consecutive_frames))
    for index in range(0, max(0, len(frame_db) - needed + 1)):
        if all(level >= float(threshold_db) for level in frame_db[index : index + needed]):
            start = index * frame_len
            pre_roll = int(round(sample_rate * max(0, int(pre_roll_ms)) / 1000))
            first_voice = max(0, start - pre_roll)
            break
    return wav[first_voice:] if first_voice > 0 else wav


def apply_fade_in(wav, sample_rate: int, fade_ms: int):
    import numpy as np

    data = np.asarray(wav, dtype=np.float32).copy()
    fade_len = min(data.size, max(1, int(round(sample_rate * fade_ms / 1000))))
    if fade_len <= 1:
        return data
    data[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    return data


def apply_short_fades(wav, sample_rate: int, fade_ms: int):
    import numpy as np

    data = np.asarray(wav, dtype=np.float32).copy()
    fade_len = min(data.size // 2, max(1, int(round(sample_rate * fade_ms / 1000))))
    if fade_len <= 1:
        return data
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    data[:fade_len] *= fade_in
    data[-fade_len:] *= fade_out
    return data


def _optional_int(value) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def bounded_int(value, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def bounded_float(value, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = minimum
    if not math.isfinite(number):
        number = minimum
    return max(minimum, min(maximum, number))


def bool_setting(value: Any, default: bool = False) -> bool:
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


if __name__ == "__main__":
    raise SystemExit(main())


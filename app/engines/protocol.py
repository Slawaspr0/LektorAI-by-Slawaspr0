from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentRequest:
    segment_id: int
    text: str
    start_ms: int
    end_ms: int
    output_path: str


@dataclass(frozen=True)
class EngineRequest:
    engine_id: str
    source_name: str
    job_id: str
    segments: list[SegmentRequest]
    settings: dict[str, Any] = field(default_factory=dict)
    dictionary: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentResult:
    segment_id: int
    ok: bool
    output_path: str = ""
    error: str = ""
    duration_ms: int = 0
    retries: int = 0
    qc_score: float | None = None
    attempts: int = 0
    selected_attempt: int = 0
    qc_warnings: tuple[str, ...] = ()
    whisper_text: str = ""
    whisper_similarity: float | None = None
    attempt_details: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EngineResult:
    engine_id: str
    job_id: str
    ok: bool
    segments: list[SegmentResult] = field(default_factory=list)
    error: str = ""


def write_request(path: Path, request: EngineRequest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(request)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_request(path: Path) -> EngineRequest:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        raw_segments = []
    segments = [_dataclass_from_dict(SegmentRequest, item) for item in raw_segments if isinstance(item, dict)]
    return EngineRequest(
        engine_id=str(data.get("engine_id", "")),
        source_name=str(data.get("source_name", "")),
        job_id=str(data.get("job_id", "")),
        segments=segments,
        settings=dict(data.get("settings", {}) or {}),
        dictionary=dict(data.get("dictionary", {}) or {}),
    )


def write_result(path: Path, result: EngineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_result(path: Path) -> EngineResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        raw_segments = []
    segments = [_dataclass_from_dict(SegmentResult, item) for item in raw_segments if isinstance(item, dict)]
    return EngineResult(
        engine_id=str(data.get("engine_id", "")),
        job_id=str(data.get("job_id", "")),
        ok=bool(data.get("ok", False)),
        segments=segments,
        error=str(data.get("error", "")),
    )


def _dataclass_from_dict(cls, data: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    normalized = {key: value for key, value in data.items() if key in allowed}
    if cls is SegmentResult and isinstance(normalized.get("qc_warnings"), list):
        normalized["qc_warnings"] = tuple(str(item) for item in normalized["qc_warnings"])
    if cls is SegmentResult and isinstance(normalized.get("attempt_details"), list):
        normalized["attempt_details"] = tuple(item for item in normalized["attempt_details"] if isinstance(item, dict))
    return cls(**normalized)

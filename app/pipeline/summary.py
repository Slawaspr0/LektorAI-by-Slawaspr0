from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_run_summary(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_run_error(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    payload.setdefault("status", "failed")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel_path(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)

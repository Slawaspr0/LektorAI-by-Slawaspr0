from __future__ import annotations

import json
import re
from pathlib import Path


def sanitize_dictionary(data: dict[str, str]) -> tuple[dict[str, str], int]:
    result: dict[str, str] = {}
    seen_keys: set[str] = set()
    skipped = 0
    for key, value in sorted((data or {}).items(), key=lambda item: str(item[0]).strip().lower()):
        clean_key = str(key).strip()
        clean_value = str(value).strip()
        normalized_key = clean_key.casefold()
        if (
            len(clean_key) < 2
            or not clean_value
            or not re.search(r"\w", clean_key, flags=re.UNICODE)
            or normalized_key in seen_keys
        ):
            skipped += 1
            continue
        seen_keys.add(normalized_key)
        result[clean_key] = clean_value
    return result, skipped


def load_dictionary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            sanitized, _ = sanitize_dictionary({str(k): str(v) for k, v in data.items()})
            return sanitized
    except Exception:
        return {}
    return {}


def save_dictionary(path: Path, data: dict[str, str]) -> tuple[int, int]:
    sanitized, skipped = sanitize_dictionary(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(sanitized), skipped

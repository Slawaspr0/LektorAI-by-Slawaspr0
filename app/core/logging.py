from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")


def safe_name(value: str, max_length: int = 80) -> str:
    ascii_value = _to_ascii_filename(value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value.strip())
    cleaned = cleaned.strip("._-")
    if max_length > 0 and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].strip("._-")
    return cleaned or "plik"


def _to_ascii_filename(value: str) -> str:
    polish = str.maketrans(
        {
            "\u0105": "a",
            "\u0107": "c",
            "\u0119": "e",
            "\u0142": "l",
            "\u0144": "n",
            "\u00f3": "o",
            "\u015b": "s",
            "\u017a": "z",
            "\u017c": "z",
            "\u0104": "A",
            "\u0106": "C",
            "\u0118": "E",
            "\u0141": "L",
            "\u0143": "N",
            "\u00d3": "O",
            "\u015a": "S",
            "\u0179": "Z",
            "\u017b": "Z",
        }
    )
    normalized = unicodedata.normalize("NFKD", str(value).translate(polish))
    return normalized.encode("ascii", "ignore").decode("ascii")


def setup_app_logger(logs_dir: Path) -> tuple[logging.Logger, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = app_log_path(logs_dir)

    logger = logging.getLogger("lektorai.app")
    logger.setLevel(logging.INFO)
    _close_logger_handlers(logger)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, log_path


def _close_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def app_log_path(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    return _unique_path(logs_dir, f"app_{timestamp()}", ".log")


def engine_log_path(engine_logs_dir: Path, engine_id: str, source_name: str) -> Path:
    engine_logs_dir.mkdir(parents=True, exist_ok=True)
    base = f"{safe_name(engine_id)}_{timestamp()}.{safe_name(source_name)}"
    return _unique_path(engine_logs_dir, base, ".log")


def broken_json_backup_path(path: Path) -> Path:
    return _unique_path(path.parent, f"{path.stem}.broken.{timestamp()}", path.suffix)


def _unique_path(directory: Path, base_name: str, suffix: str) -> Path:
    candidate = directory / f"{base_name}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{base_name}_{counter}{suffix}"
        counter += 1
    return candidate

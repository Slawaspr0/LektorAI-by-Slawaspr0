from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SMALL_DOWNLOAD_BYTES = 20 * 1024 * 1024
MEDIUM_DOWNLOAD_BYTES = 100 * 1024 * 1024
UNKNOWN_SIZE_REPORT_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class DownloadProgressPolicy:
    label: str = "Pobieranie"
    small_limit_bytes: int = SMALL_DOWNLOAD_BYTES
    medium_limit_bytes: int = MEDIUM_DOWNLOAD_BYTES
    unknown_step_bytes: int = UNKNOWN_SIZE_REPORT_BYTES


def download_file_with_progress(
    url: str,
    target: Path,
    *,
    label: str,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    timeout_s: float = 30.0,
) -> None:
    policy = DownloadProgressPolicy(label=str(label or "Pobieranie").strip() or "Pobieranie")
    request = urllib.request.Request(str(url), headers={"User-Agent": "LektorAI downloader"}, method="GET")
    target.parent.mkdir(parents=True, exist_ok=True)
    _emit(progress, policy.label)
    with urllib.request.urlopen(request, timeout=timeout_s) as response, target.open("wb") as output:
        total = _content_length(response)
        if total <= 0:
            _download_unknown_size(response, output, policy, progress, cancel_requested)
        else:
            _download_known_size(
                response,
                output,
                total,
                progress_percent_step_for_size(total, policy),
                policy,
                progress,
                cancel_requested,
            )
    _raise_if_cancelled(cancel_requested)
    _emit(progress, f"{policy.label} - pobrano")


def progress_percent_step_for_size(total_bytes: int, policy: DownloadProgressPolicy | None = None) -> int:
    policy = policy or DownloadProgressPolicy()
    try:
        total = int(total_bytes)
    except Exception:
        total = 0
    if total <= 0 or total < policy.small_limit_bytes:
        return 0
    if total < policy.medium_limit_bytes:
        return 20
    return 10


def _download_known_size(
    response,
    output,
    total: int,
    percent_step: int,
    policy: DownloadProgressPolicy,
    progress: Callable[[str], None] | None,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    downloaded = 0
    next_percent = percent_step if percent_step > 0 else 101
    while True:
        _raise_if_cancelled(cancel_requested)
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        output.write(chunk)
        downloaded += len(chunk)
        if percent_step <= 0:
            continue
        percent = int(downloaded * 100 / max(1, total))
        while next_percent <= 90 and percent >= next_percent:
            _emit(progress, f"{policy.label} - {next_percent}%")
            next_percent += percent_step


def _download_unknown_size(
    response,
    output,
    policy: DownloadProgressPolicy,
    progress: Callable[[str], None] | None,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    downloaded = 0
    next_report = max(1, int(policy.unknown_step_bytes))
    while True:
        _raise_if_cancelled(cancel_requested)
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        output.write(chunk)
        downloaded += len(chunk)
        if downloaded >= next_report:
            _emit(progress, f"{policy.label} - pobrano {_format_size(downloaded)}")
            next_report += max(1, int(policy.unknown_step_bytes))


def _content_length(response) -> int:
    try:
        return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _format_size(size_bytes: int) -> str:
    mib = float(size_bytes) / float(1024 * 1024)
    if mib < 1024:
        return f"{mib:.0f} MB"
    return f"{mib / 1024:.1f} GB"


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Przerwano przez uzytkownika")

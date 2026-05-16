from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.download import download_file_with_progress
from app.core.version import APP_NAME, APP_VERSION


UPDATE_INFO_URL = "https://raw.githubusercontent.com/Slawaspr0/LektorAI-by-Slawaspr0/main/update.json"
SOURCE_ZIP_URL = "https://github.com/Slawaspr0/LektorAI-by-Slawaspr0/archive/refs/heads/main.zip"
LOCAL_UPDATE_FILE = "update.json"

PROTECTED_ROOTS = {"engines", "stt", "cache", "logs", "temp", "packages", "__pycache__"}
PROTECTED_FILES = {"config.json", "ffmpeg.exe", "ffprobe.exe", "mkvmerge.exe", "ffmpeg", "ffprobe", "mkvmerge"}


@dataclass(frozen=True)
class UpdateInfo:
    app_name: str
    version: str
    build_id: str
    zip_url: str
    remove: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class UpdateCheckResult:
    ok: bool
    update_available: bool
    local: UpdateInfo
    remote: UpdateInfo | None
    message: str
    error: str = ""


def read_local_update_info(app_dir: Path) -> UpdateInfo:
    path = Path(app_dir) / LOCAL_UPDATE_FILE
    if path.is_file():
        try:
            return update_info_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return UpdateInfo(
        app_name=APP_NAME,
        version=APP_VERSION,
        build_id="",
        zip_url=SOURCE_ZIP_URL,
        remove=(),
        raw={},
    )


def update_info_from_dict(data: dict[str, Any]) -> UpdateInfo:
    if not isinstance(data, dict):
        data = {}
    remove_value = data.get("remove", ())
    if not isinstance(remove_value, list):
        remove_value = []
    remove = tuple(str(item).strip().replace("\\", "/") for item in remove_value if str(item).strip())
    return UpdateInfo(
        app_name=str(data.get("app_name", APP_NAME) or APP_NAME).strip(),
        version=str(data.get("version", APP_VERSION) or APP_VERSION).strip(),
        build_id=str(data.get("build_id", "") or "").strip(),
        zip_url=str(data.get("zip_url", SOURCE_ZIP_URL) or SOURCE_ZIP_URL).strip(),
        remove=remove,
        raw=dict(data),
    )


def check_for_updates(app_dir: Path, info_url: str = UPDATE_INFO_URL, timeout_s: float = 8.0) -> UpdateCheckResult:
    local = read_local_update_info(app_dir)
    try:
        remote = fetch_update_info(info_url, timeout_s=timeout_s)
    except Exception as exc:
        return UpdateCheckResult(
            ok=False,
            update_available=False,
            local=local,
            remote=None,
            message="Nie udalo sie sprawdzic aktualizacji.",
            error=str(exc),
        )
    available = is_update_available(local, remote)
    if available:
        message = f"Dostepna aktualizacja: {remote.version}"
    else:
        message = "Masz najnowsza wersje."
    return UpdateCheckResult(ok=True, update_available=available, local=local, remote=remote, message=message)


def fetch_update_info(info_url: str, timeout_s: float = 8.0) -> UpdateInfo:
    request = urllib.request.Request(
        str(info_url),
        headers={"User-Agent": f"{APP_NAME} updater"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read(1024 * 1024)
    return update_info_from_dict(json.loads(payload.decode("utf-8")))


def is_update_available(local: UpdateInfo, remote: UpdateInfo) -> bool:
    if not remote.version and not remote.build_id:
        return False
    if normalize_version(remote.version) != normalize_version(local.version):
        return True
    if remote.build_id and local.build_id and remote.build_id != local.build_id:
        return True
    if remote.build_id and not local.build_id:
        return True
    return False


def normalize_version(value: str) -> str:
    return str(value or "").strip().lower().lstrip("v")


def apply_update(
    app_dir: Path,
    parent_pid: int = 0,
    info_url: str = UPDATE_INFO_URL,
    restart: bool = True,
    timeout_s: float = 20.0,
) -> int:
    app_dir = Path(app_dir).resolve()
    log_path = update_log_path(app_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        def write(message: str) -> None:
            log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
            log.flush()

        write(f"Start aktualizacji: {app_dir}")
        result = check_for_updates(app_dir, info_url=info_url, timeout_s=timeout_s)
        if not result.ok:
            write(f"BLAD sprawdzania aktualizacji: {result.error}")
            return 2
        if not result.update_available:
            write("Brak aktualizacji.")
            return 0
        remote = result.remote
        if remote is None:
            write("BLAD: brak metadanych zdalnej aktualizacji.")
            return 2
        wait_for_pid_exit(parent_pid, timeout_s=60.0)
        with tempfile.TemporaryDirectory(prefix="lektorai_update_") as temp_name:
            temp_dir = Path(temp_name)
            zip_path = temp_dir / "update.zip"
            extracted_dir = temp_dir / "extracted"
            write(f"Pobieranie: {remote.zip_url}")
            download_file(remote.zip_url, zip_path, timeout_s=timeout_s, progress=write)
            write("Rozpakowywanie aktualizacji")
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(extracted_dir)
            source_root = find_update_source_root(extracted_dir)
            write(f"Kopiowanie plikow z: {source_root}")
            copy_update_tree(source_root, app_dir, write)
            remove_obsolete_paths(app_dir, remote.remove, write)
        write("Aktualizacja zakonczona.")
    if restart:
        restart_application(app_dir)
    return 0


def download_file(
    url: str,
    target: Path,
    timeout_s: float = 20.0,
    progress=None,
) -> None:
    download_file_with_progress(
        url,
        target,
        label="Aktualizacja: pobieranie plikow",
        progress=progress,
        timeout_s=timeout_s,
    )


def find_update_source_root(extracted_dir: Path) -> Path:
    candidates = [path for path in extracted_dir.iterdir() if path.is_dir()]
    for candidate in candidates:
        if (candidate / "START.py").is_file() and (candidate / "app").is_dir():
            return candidate
    if (extracted_dir / "START.py").is_file() and (extracted_dir / "app").is_dir():
        return extracted_dir
    raise RuntimeError("Paczka aktualizacji nie zawiera plikow aplikacji.")


def copy_update_tree(source_root: Path, app_dir: Path, log) -> None:
    for source in source_root.iterdir():
        relative = Path(source.name)
        if is_protected_update_path(relative):
            log(f"Pomijam dane uzytkownika: {relative.as_posix()}")
            continue
        copy_path(source, app_dir / source.name, relative, log)


def copy_path(source: Path, target: Path, relative: Path, log) -> None:
    if is_protected_update_path(relative):
        return
    if source.is_dir():
        if source.name == "__pycache__":
            return
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            copy_path(child, target / child.name, relative / child.name, log)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    log(f"Zaktualizowano: {relative.as_posix()}")


def remove_obsolete_paths(app_dir: Path, relative_paths: tuple[str, ...], log) -> None:
    for raw_relative in relative_paths:
        relative = safe_relative_path(raw_relative)
        if relative is None or is_protected_update_path(relative):
            log(f"Pomijam usuwanie chronionej sciezki: {raw_relative}")
            continue
        target = (app_dir / relative).resolve()
        try:
            target.relative_to(app_dir.resolve())
        except ValueError:
            log(f"Pomijam usuwanie poza folderem aplikacji: {raw_relative}")
            continue
        if target.is_dir():
            shutil.rmtree(target)
            log(f"Usunieto folder: {relative.as_posix()}")
        elif target.exists():
            target.unlink()
            log(f"Usunieto plik: {relative.as_posix()}")


def safe_relative_path(value: str) -> Path | None:
    text = str(value or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or ":" in text:
        return None
    path = Path(text)
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path


def is_protected_update_path(relative: Path) -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    if not parts:
        return True
    if parts[0] in PROTECTED_ROOTS:
        return True
    if len(parts) == 1 and parts[0] in PROTECTED_FILES:
        return True
    if any(part == "__pycache__" for part in parts):
        return True
    if parts[-1].endswith((".pyc", ".pyo")):
        return True
    return False


def wait_for_pid_exit(pid: int, timeout_s: float = 60.0) -> None:
    try:
        pid = int(pid)
    except Exception:
        return
    if pid <= 0 or pid == os.getpid():
        return
    if os.name == "nt":
        wait_for_pid_exit_windows(pid, timeout_s)
        return
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.25)


def wait_for_pid_exit_windows(pid: int, timeout_s: float) -> None:
    try:
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            return
        try:
            milliseconds = int(max(0.0, timeout_s) * 1000)
            ctypes.windll.kernel32.WaitForSingleObject(handle, milliseconds)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        time.sleep(min(max(0.0, timeout_s), 2.0))


def restart_application(app_dir: Path) -> None:
    start_py = Path(app_dir) / "START.py"
    if not start_py.is_file():
        return
    subprocess.Popen([sys.executable, str(start_py)], cwd=str(app_dir), close_fds=True)


def update_log_path(app_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    return Path(app_dir) / "logs" / f"update_{stamp}.log"


def run_updater_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aktualizator LektorAI")
    parser.add_argument("--app-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--info-url", default=UPDATE_INFO_URL)
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args(argv)
    return apply_update(Path(args.app_dir), parent_pid=args.pid, info_url=args.info_url, restart=not args.no_restart)


from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DeviceChoices:
    values: tuple[str, ...]
    labels: tuple[str, ...]


def build_device_choices(devices: Iterable[dict[str, Any]], include_auto: bool) -> DeviceChoices:
    values: list[str] = []
    labels: list[str] = []
    if include_auto:
        values.append("auto")
        labels.append("Auto")
    values.append("cpu")
    labels.append("CPU")
    for device in devices:
        try:
            index = int(device.get("index"))
        except Exception:
            continue
        if index < 0:
            continue
        name = str(device.get("name", "") or "").strip() or f"CUDA {index}"
        total_memory = _format_vram(device.get("total_memory"))
        suffix = f" ({total_memory})" if total_memory else ""
        values.append(f"cuda:{index}")
        labels.append(f"GPU {index}: {name}{suffix}")
    return DeviceChoices(tuple(values), tuple(labels))


def detect_torch_device_choices(
    python_executable: str | Path | None = None,
    *,
    include_auto: bool,
    timeout_s: int = 8,
) -> DeviceChoices:
    return build_device_choices(detect_cuda_devices(python_executable, timeout_s=timeout_s), include_auto=include_auto)


def detect_cuda_devices(
    python_executable: str | Path | None = None,
    *,
    timeout_s: int = 8,
    prefer_nvidia_smi: bool = False,
    torch_detector=None,
    smi_detector=None,
) -> tuple[dict[str, Any], ...]:
    torch_detector = torch_detector or detect_torch_cuda_devices
    smi_detector = smi_detector or detect_nvidia_smi_cuda_devices
    if prefer_nvidia_smi:
        devices = smi_detector(timeout_s=timeout_s)
        if devices:
            return devices
        return torch_detector(python_executable, timeout_s=timeout_s)
    devices = torch_detector(python_executable, timeout_s=timeout_s)
    if devices:
        return devices
    return smi_detector(timeout_s=timeout_s)


def detect_torch_cuda_devices(python_executable: str | Path | None = None, *, timeout_s: int = 8) -> tuple[dict[str, Any], ...]:
    python_path = Path(python_executable) if python_executable else Path(sys.executable)
    if not python_path.exists():
        return ()
    script = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    devices=[]\n"
        "    if torch.cuda.is_available():\n"
        "        for idx in range(int(torch.cuda.device_count())):\n"
        "            props = torch.cuda.get_device_properties(idx)\n"
        "            devices.append({'index': idx, 'name': torch.cuda.get_device_name(idx), 'total_memory': int(props.total_memory)})\n"
        "    print(json.dumps({'devices': devices}, ensure_ascii=True))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'devices': [], 'error': str(exc)}, ensure_ascii=True))\n"
    )
    try:
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            [str(python_path), "-B", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout_s)),
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
    except Exception:
        return ()
    if result.returncode != 0:
        return ()
    try:
        payload = json.loads((result.stdout or "").strip().splitlines()[-1])
    except Exception:
        return ()
    devices = payload.get("devices", [])
    if not isinstance(devices, list):
        return ()
    normalized: list[dict[str, Any]] = []
    for item in devices:
        if isinstance(item, dict):
            normalized.append(item)
    return tuple(normalized)


def detect_nvidia_smi_cuda_devices(*, timeout_s: int = 5) -> tuple[dict[str, Any], ...]:
    try:
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout_s)),
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
    except Exception:
        return ()
    if result.returncode != 0:
        return ()
    devices: list[dict[str, Any]] = []
    for raw_line in (result.stdout or "").splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            memory_mib = int(float(parts[2]))
        except Exception:
            continue
        devices.append({"index": index, "name": parts[1], "total_memory": memory_mib * 1024 * 1024})
    return tuple(devices)


def _format_vram(total_memory: Any) -> str:
    try:
        bytes_total = int(total_memory)
    except Exception:
        return ""
    if bytes_total <= 0:
        return ""
    gib = bytes_total / float(1024**3)
    return f"{gib:.1f} GB"

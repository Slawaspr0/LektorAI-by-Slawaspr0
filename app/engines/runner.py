from __future__ import annotations

import subprocess
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.logging import engine_log_path
from app.core.logging import timestamp
from app.core.paths import AppPaths
from app.engines.manager import EngineManager
from app.engines.protocol import EngineResult, read_result


DEFAULT_WORKER_TIMEOUT_S = 12 * 60 * 60


@dataclass(frozen=True)
class WorkerRun:
    request_path: Path
    result_path: Path
    log_path: Path


class EngineWorkerRunner:
    def __init__(self, paths: AppPaths, manager: EngineManager) -> None:
        self.paths = paths
        self.manager = manager

    def build_run_paths(self, engine_id: str, source_name: str, job_id: str) -> WorkerRun:
        engine_dir = self.paths.engine_dir(engine_id)
        temp_dir = engine_dir / "temp" / job_id
        log_path = engine_log_path(engine_dir / "logs", engine_id, source_name)
        return WorkerRun(
            request_path=temp_dir / "request.json",
            result_path=temp_dir / "result.json",
            log_path=log_path,
        )

    def run_worker(
        self,
        engine_id: str,
        worker_script: Path,
        run: WorkerRun,
        timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
        progress: Callable[[str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> EngineResult:
        engine_dir = self.paths.engine_dir(engine_id)
        python_path = self._venv_python(engine_dir)
        if not python_path.exists():
            raise FileNotFoundError(f"Brak Python venv dla {engine_id}: {python_path}")
        if not worker_script.exists():
            raise FileNotFoundError(f"Brak worker.py dla {engine_id}: {worker_script}")

        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        with run.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n=== {timestamp()} {engine_id} worker start ===\n")
            log_file.write(f"cwd: {engine_dir}\n")
            log_file.write(f"python: {python_path}\n")
            log_file.write(f"worker: {worker_script}\n")
            log_file.write(f"request: {run.request_path}\n")
            log_file.write(f"result: {run.result_path}\n\n")
            log_file.flush()
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            process = subprocess.Popen(
                [str(python_path), "-u", str(worker_script), str(run.request_path), str(run.result_path)],
                cwd=str(engine_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self.manager._worker_env(),
                creationflags=creationflags,
            )
            assert process.stdout is not None
            output_queue: queue.Queue[str] = queue.Queue()
            timed_out = False
            cancelled = False

            def read_output() -> None:
                try:
                    for output_line in process.stdout:
                        output_queue.put(output_line)
                finally:
                    try:
                        process.stdout.close()
                    except Exception:
                        pass

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            deadline = time.monotonic() + max(1, int(timeout_s))
            while process.poll() is None:
                self._drain_worker_output(output_queue, log_file, progress)
                if cancel_requested is not None and cancel_requested():
                    cancelled = True
                    log_file.write(f"\n=== {timestamp()} {engine_id} worker cancel requested ===\n")
                    log_file.flush()
                    self._terminate_process_tree(process)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    log_file.write(f"\n=== {timestamp()} {engine_id} worker timeout ===\n")
                    log_file.flush()
                    self._terminate_process_tree(process)
                    break
                time.sleep(0.2)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._terminate_process_tree(process)
                return_code = process.wait(timeout=10)
            reader.join(timeout=2)
            self._drain_worker_output(output_queue, log_file, progress)
            log_file.write(f"\n=== {timestamp()} {engine_id} worker exit: {return_code} ===\n")
            log_file.flush()
            if cancelled:
                raise RuntimeError(f"Worker {engine_id} przerwany przez uzytkownika")
            if timed_out:
                raise RuntimeError(f"Worker {engine_id} przekroczyl limit czasu")
        if return_code != 0:
            if run.result_path.exists():
                result = read_result(run.result_path)
                first_error = next((segment.error for segment in result.segments if not segment.ok and segment.error), "")
                raise RuntimeError(first_error or result.error or f"Worker {engine_id} zakonczyl sie kodem {return_code}")
            raise RuntimeError(f"Worker {engine_id} zakonczyl sie kodem {return_code}")
        if not run.result_path.exists():
            raise FileNotFoundError(f"Worker {engine_id} nie utworzyl result.json")
        return read_result(run.result_path)

    def _venv_python(self, engine_dir: Path) -> Path:
        return self.manager._venv_python(engine_dir)

    def _drain_worker_output(
        self,
        output_queue: queue.Queue[str],
        log_file,
        progress: Callable[[str], None] | None,
    ) -> None:
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                return
            log_file.write(line)
            log_file.flush()
            if progress is not None:
                progress(line.rstrip())

    def _terminate_process_tree(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if self._taskkill_process_tree(process.pid):
            return
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _taskkill_process_tree(self, pid: int) -> bool:
        if not hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            return False
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=20,
            )
        except Exception:
            return False
        return completed.returncode == 0

from __future__ import annotations

import json
import importlib.util
import math
import os
import stat
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Callable

from app.core.dictionary import sanitize_dictionary
from app.core.logging import broken_json_backup_path
from app.core.paths import AppPaths
from app.engines.config_schema import ConfigField, fields_for
from app.engines.install_specs import get_install_spec
from app.engines.registry import get_engine_definitions
from app.engines.schemas import EngineDefinition, EngineKind, EngineState, EngineStatus


ProgressFn = Callable[[str], None]


class EngineRemovalError(RuntimeError):
    pass


class EngineManager:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.definitions = {d.engine_id: d for d in get_engine_definitions()}
        self._package_check_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    def list_states(self) -> list[EngineState]:
        return [self.state_for(defn.engine_id) for defn in self.definitions.values()]

    def state_for(self, engine_id: str) -> EngineState:
        definition = self.definitions[engine_id]
        engine_dir = self.paths.engine_dir(engine_id)
        components = self._components_for(definition, engine_dir)

        if not definition.pipeline_ready:
            return EngineState(definition, EngineStatus.BROKEN, False, "pipeline niepodlaczony", components)

        if definition.built_in:
            missing_modules = self._missing_app_import_specs(definition)
            if missing_modules:
                return EngineState(
                    definition,
                    EngineStatus.BROKEN,
                    False,
                    f"brak pakietow aplikacji: {', '.join(missing_modules)}",
                    components,
                )
            if definition.requires_api_key and not self._has_api_key(engine_dir):
                return EngineState(definition, EngineStatus.REQUIRES_CONFIG, True, "brak klucza API", components)
            return EngineState(definition, EngineStatus.REQUIRES_INTERNET, True, "silnik internetowy", components)

        if not engine_dir.exists():
            return EngineState(definition, EngineStatus.NOT_INSTALLED, False, "brak folderu silnika", components)

        if definition.requires_venv and not self._venv_python(engine_dir).exists():
            return EngineState(definition, EngineStatus.NOT_INSTALLED, False, "brak venv", components)

        if definition.requires_venv and not (engine_dir / "worker.py").exists():
            return EngineState(definition, EngineStatus.BROKEN, False, "brak worker.py", components)

        missing_packages = self._missing_import_specs(definition, engine_dir)
        if missing_packages:
            return EngineState(
                definition,
                EngineStatus.BROKEN,
                False,
                f"brak pakietow: {', '.join(missing_packages)}",
                components,
            )

        if not (engine_dir / "config.json").exists():
            return EngineState(definition, EngineStatus.REQUIRES_CONFIG, True, "brak config.json", components)

        if definition.requires_model and not self._has_model_cache(engine_dir):
            return EngineState(
                definition,
                EngineStatus.INSTALLED_NO_MODEL,
                True,
                "model/cache pobierany przy pierwszym uzyciu",
                components,
            )

        return EngineState(definition, EngineStatus.READY, True, "", components)

    def ensure_engine_config(self, engine_id: str) -> Path:
        definition = self.definitions[engine_id]
        engine_dir = self.paths.engine_dir(engine_id)
        engine_dir.mkdir(parents=True, exist_ok=True)
        config_path = engine_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(
                json.dumps(definition.default_config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            self._merge_default_config(config_path, definition.default_config, engine_id)
        return config_path

    def ensure_engine_dictionary(self, engine_id: str) -> Path:
        engine_dir = self.paths.engine_dir(engine_id)
        engine_dir.mkdir(parents=True, exist_ok=True)
        dictionary_path = engine_dir / "dictionary.json"
        if not dictionary_path.exists():
            dictionary_path.write_text("{}\n", encoding="utf-8")
        else:
            try:
                data = json.loads(dictionary_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("dictionary root is not an object")
                sanitized, _ = sanitize_dictionary({str(key): str(value) for key, value in data.items()})
                normalized_text = json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n"
                if dictionary_path.read_text(encoding="utf-8") != normalized_text:
                    dictionary_path.write_text(normalized_text, encoding="utf-8")
            except Exception:
                self._quarantine_invalid_json(dictionary_path)
                dictionary_path.write_text("{}\n", encoding="utf-8")
        return dictionary_path

    def prepare_local_engine(self, engine_id: str) -> Path:
        definition = self.definitions[engine_id]
        if definition.kind != EngineKind.LOCAL:
            raise ValueError("Instalacja dotyczy tylko lokalnych silnikow TTS.")
        engine_dir = self.paths.engine_dir(engine_id)
        engine_dir.mkdir(parents=True, exist_ok=True)
        install_log = engine_dir / "install.log"
        install_log.write_text(
            "Przygotowano folder silnika.\n"
            "Uzyj instalacji TTS, aby utworzyc venv i zainstalowac zaleznosci.\n",
            encoding="utf-8",
        )
        return engine_dir

    def engine_dir_exists(self, engine_id: str) -> bool:
        return self.paths.engine_dir(engine_id).exists()

    def local_runtime_exists(self, engine_id: str) -> bool:
        definition = self.definitions[engine_id]
        if definition.kind != EngineKind.LOCAL:
            return False
        engine_dir = self.paths.engine_dir(engine_id)
        return engine_dir.exists() and self._venv_python(engine_dir).exists() and (engine_dir / "worker.py").exists()

    def removable_payload_exists(self, engine_id: str) -> bool:
        engine_dir = self.paths.engine_dir(engine_id)
        if not engine_dir.exists():
            return False
        keep_names = {"config.json", "dictionary.json"}
        return any(child.name not in keep_names for child in engine_dir.iterdir())

    def local_install_preview(self, engine_id: str) -> list[str]:
        definition = self.definitions[engine_id]
        if definition.kind != EngineKind.LOCAL:
            raise ValueError("Podglad instalacji dotyczy tylko lokalnych silnikow TTS.")
        spec = get_install_spec(engine_id)
        engine_dir = self.paths.engine_dir(engine_id)
        lines = [
            f"Silnik: {definition.display_name}",
            f"Folder silnika: {engine_dir}",
            f"Venv: {engine_dir / 'venv'}",
            "PyTorch CUDA:",
            f"  --index-url {spec.torch_index_url}",
        ]
        lines.extend(f"  {requirement}" for requirement in spec.torch_requirements)
        lines.extend(
            [
            f"Requirements: {engine_dir / 'requirements.txt'}",
            f"Constraints: {engine_dir / 'constraints.txt'}",
            ]
        )
        lines.extend(f"  {constraint}" for constraint in spec.constraints)
        lines.extend(
            [
            "Pakiety:",
            ]
        )
        lines.extend(f"  {requirement}" for requirement in spec.requirements)
        if spec.no_deps_requirements:
            lines.append("Pakiety bez zaleznosci:")
            lines.extend(f"  {requirement}" for requirement in spec.no_deps_requirements)
        if spec.import_checks:
            lines.append("Kontrola importu:")
            lines.extend(f"  {name}" for name in spec.import_checks)
        return lines

    def install_local_engine(self, engine_id: str, progress: ProgressFn | None = None) -> Path:
        definition = self.definitions[engine_id]
        if definition.kind != EngineKind.LOCAL:
            raise ValueError("Instalacja dotyczy tylko lokalnych silnikow TTS.")

        spec = get_install_spec(engine_id)
        engine_dir = self.paths.engine_dir(engine_id)
        engine_dir.mkdir(parents=True, exist_ok=True)

        requirements_path = engine_dir / "requirements.txt"
        no_deps_requirements_path = engine_dir / "requirements.no-deps.txt"
        constraints_path = engine_dir / "constraints.txt"
        requirements_path.write_text("\n".join(spec.requirements) + "\n", encoding="utf-8")
        if spec.no_deps_requirements:
            no_deps_requirements_path.write_text("\n".join(spec.no_deps_requirements) + "\n", encoding="utf-8")
        elif no_deps_requirements_path.exists():
            no_deps_requirements_path.unlink()
        constraints_path.write_text("\n".join(spec.constraints) + "\n", encoding="utf-8")
        install_log = engine_dir / "install.log"
        venv_dir = engine_dir / "venv"
        worker_path = engine_dir / "worker.py"

        self._emit(progress, f"TTS {engine_id}: instalacja rozpoczeta")
        with install_log.open("w", encoding="utf-8") as log:
            log.write(f"Engine: {engine_id}\n")
            log.write(f"Python: {sys.executable}\n")
            log.write(f"Venv: {venv_dir}\n\n")
            self._emit(progress, f"TTS {engine_id}: przygotowanie venv")
            self._ensure_venv(venv_dir, log)
            venv_python = self._venv_python(engine_dir)
            self._emit(progress, f"TTS {engine_id}: instalacja narzedzi build")
            self._run_logged_step(
                "Przygotowanie wheel/setuptools",
                self._build_tools_install_command(venv_python),
                log,
            )
            self._emit(progress, f"TTS {engine_id}: instalacja PyTorch")
            self._run_logged_step(
                "Instalacja PyTorch CUDA",
                self._torch_install_command(venv_python, spec, constraints_path),
                log,
            )
            self._emit(progress, f"TTS {engine_id}: instalacja pakietow TTS")
            self._run_logged_step(
                "Instalacja pakietow TTS",
                self._requirements_install_command(venv_python, requirements_path, constraints_path),
                log,
            )
            if spec.no_deps_requirements:
                self._emit(progress, f"TTS {engine_id}: instalacja pakietow dodatkowych")
                self._run_logged_step(
                    "Instalacja pakietow TTS bez zaleznosci",
                    self._no_deps_requirements_install_command(venv_python, no_deps_requirements_path),
                    log,
                )
            self._emit(progress, f"TTS {engine_id}: kontrola zaleznosci")
            self._run_pip_check(venv_python, spec, log)
            self._emit(progress, f"TTS {engine_id}: kontrola importow")
            self._run_import_checks(definition, venv_python, log)
            self.install_worker_script(engine_id)
            log.write(f"Worker: {worker_path}\n")

        self.clear_package_check_cache(engine_id)
        self._emit(progress, f"TTS {engine_id}: instalacja zakonczona")
        return engine_dir

    def install_worker_script(self, engine_id: str) -> Path:
        definition = self.definitions[engine_id]
        if definition.kind != EngineKind.LOCAL:
            raise ValueError("Worker dotyczy tylko lokalnych silnikow TTS.")
        engine_dir = self.paths.engine_dir(engine_id)
        engine_dir.mkdir(parents=True, exist_ok=True)
        worker_path = engine_dir / "worker.py"
        worker_template = Path(__file__).parent / "worker_templates" / "local_tts_worker.py"
        shutil.copy2(worker_template, worker_path)
        return worker_path

    def clear_package_check_cache(self, engine_id: str | None = None) -> None:
        if engine_id is None:
            self._package_check_cache.clear()
            return
        prefix = f"{engine_id}|"
        for key in list(self._package_check_cache):
            if key[0].startswith(prefix):
                self._package_check_cache.pop(key, None)

    def remove_engine_keep_user_settings(self, engine_id: str) -> None:
        engine_dir = self.paths.engine_dir(engine_id)
        if not engine_dir.exists():
            return
        keep_names = {"config.json", "dictionary.json"}
        for child in list(engine_dir.iterdir()):
            if child.name in keep_names and child.is_file():
                continue
            self._remove_path(child)
        self.clear_package_check_cache(engine_id)

    def remove_engine_completely(self, engine_id: str) -> None:
        engine_dir = self.paths.engine_dir(engine_id)
        if engine_dir.exists():
            self._remove_path(engine_dir)
        self.clear_package_check_cache(engine_id)

    def _venv_python(self, engine_dir: Path) -> Path:
        if os.name == "nt":
            return engine_dir / "venv" / "Scripts" / "python.exe"
        return engine_dir / "venv" / "bin" / "python"

    def _has_api_key(self, engine_dir: Path) -> bool:
        return bool(self._api_key_source(engine_dir))

    def _api_key_source(self, engine_dir: Path) -> str:
        if os.environ.get("OPENAI_API_KEY"):
            return "ENV"
        config_path = engine_dir / "config.json"
        if not config_path.exists():
            return ""
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if str(data.get("api_key", "")).strip():
                return "config"
        except Exception:
            return ""
        return ""

    def _merge_default_config(self, config_path: Path, defaults: dict, engine_id: str) -> None:
        if not defaults:
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self._quarantine_invalid_json(config_path)
                self._write_default_config(config_path, defaults)
                return
        except Exception:
            self._quarantine_invalid_json(config_path)
            self._write_default_config(config_path, defaults)
            return
        changed = False
        for key, value in defaults.items():
            if key not in data:
                data[key] = value
                changed = True
        normalized = self._normalize_known_config_values(engine_id, data, defaults)
        if normalized != data:
            data = normalized
            changed = True
        if changed:
            config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_default_config(self, config_path: Path, defaults: dict) -> None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(defaults, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _normalize_known_config_values(self, engine_id: str, data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(data)
        for field in fields_for(engine_id):
            if field.key not in normalized or field.key not in defaults:
                continue
            normalized[field.key] = self._coerce_config_value(field, normalized[field.key], defaults[field.key])
        return normalized

    def _coerce_config_value(self, field: ConfigField, value: Any, default: Any) -> Any:
        if field.field_type in {"str", "path"}:
            if isinstance(value, str):
                return value.strip()
            return default
        if field.field_type == "bool":
            coerced = self._coerce_bool(value)
            return default if coerced is None else coerced
        if field.field_type == "int":
            try:
                if isinstance(value, bool):
                    raise ValueError("bool is not int config")
                if isinstance(value, float):
                    if not value.is_integer():
                        raise ValueError("float is not whole int config")
                    number = int(value)
                else:
                    number = int(str(value).strip())
            except Exception:
                return default
            if field.minimum is not None and number < int(field.minimum):
                return default
            if field.maximum is not None and number > int(field.maximum):
                return default
            return number
        if field.field_type == "float":
            try:
                if isinstance(value, bool):
                    raise ValueError("bool is not float config")
                number = float(value)
            except Exception:
                return default
            if not math.isfinite(number):
                return default
            if field.minimum is not None and number < float(field.minimum):
                return default
            if field.maximum is not None and number > float(field.maximum):
                return default
            return number
        return value

    def _coerce_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "tak", "on"}:
                return True
            if normalized in {"0", "false", "no", "nie", "off"}:
                return False
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            if value in {0, 1}:
                return bool(value)
        return None

    def _quarantine_invalid_json(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        backup_path = broken_json_backup_path(path)
        path.replace(backup_path)
        return backup_path

    def _components_for(self, definition: EngineDefinition, engine_dir: Path) -> tuple[str, ...]:
        if definition.built_in:
            items = [
                self._app_packages_component_label(definition),
                self._component_label("config", (engine_dir / "config.json").exists()),
                self._component_label("slownik", (engine_dir / "dictionary.json").exists()),
            ]
            if definition.requires_api_key:
                items.append(self._api_key_component_label(engine_dir))
            return tuple(items)

        if not engine_dir.exists():
            return ("folder: brak",)
        return (
            "folder: OK",
            self._component_label("venv", self._venv_python(engine_dir).exists()),
            self._component_label("worker", (engine_dir / "worker.py").exists()),
            self._packages_component_label(definition, engine_dir),
            self._component_label("config", (engine_dir / "config.json").exists()),
            self._component_label("slownik", (engine_dir / "dictionary.json").exists()),
            self._component_label("cache", self._has_model_cache(engine_dir)),
        )

    def _component_label(self, name: str, ok: bool) -> str:
        return f"{name}: {'OK' if ok else 'brak'}"

    def _api_key_component_label(self, engine_dir: Path) -> str:
        source = self._api_key_source(engine_dir)
        if not source:
            return "api_key: brak"
        return f"api_key: {source}"

    def _packages_component_label(self, definition: EngineDefinition, engine_dir: Path) -> str:
        if not definition.requires_venv:
            return "pakiety: n/d"
        if not self._venv_python(engine_dir).exists():
            return "pakiety: brak venv"
        missing = self._missing_import_specs(definition, engine_dir)
        if not missing:
            return "pakiety: OK"
        return f"pakiety: brak ({', '.join(missing)})"

    def _app_packages_component_label(self, definition: EngineDefinition) -> str:
        if not definition.import_checks:
            return "pakiety: n/d"
        missing = self._missing_app_import_specs(definition)
        if not missing:
            return "pakiety: OK"
        return f"pakiety: brak ({', '.join(missing)})"

    def _has_model_cache(self, engine_dir: Path) -> bool:
        for root in (engine_dir / "cache", engine_dir / "models"):
            if not root.exists():
                continue
            if root.is_file():
                return True
            if root.is_dir() and any(self._is_model_cache_file(root, path) for path in root.rglob("*")):
                return True
        return False

    def _is_model_cache_file(self, root: Path, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            relative = path.relative_to(root)
        except ValueError:
            return False
        return not relative.parts or relative.parts[0].lower() != "whisper"

    def _missing_import_specs(self, definition: EngineDefinition, engine_dir: Path) -> tuple[str, ...]:
        if definition.kind != EngineKind.LOCAL or not definition.requires_venv:
            return ()
        python_path = self._venv_python(engine_dir)
        if not python_path.exists():
            return ()
        try:
            spec = get_install_spec(definition.engine_id)
        except Exception:
            return ()
        if not spec.import_checks:
            return ()
        cache_key = (
            f"{definition.engine_id}|{python_path}",
            "|".join(spec.import_checks),
        )
        if cache_key in self._package_check_cache:
            return self._package_check_cache[cache_key]
        script = self._import_missing_script(spec.import_checks)
        try:
            result = subprocess.run(
                [str(python_path), "-c", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._worker_env(),
                timeout=20,
            )
        except Exception:
            missing = ("nie mozna sprawdzic",)
            self._package_check_cache[cache_key] = missing
            return missing
        if result.returncode == 0:
            self._package_check_cache[cache_key] = ()
            return ()
        missing = tuple(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
        value = missing or ("nieznane",)
        self._package_check_cache[cache_key] = value
        return value

    def _missing_app_import_specs(self, definition: EngineDefinition) -> tuple[str, ...]:
        if not definition.import_checks:
            return ()
        return tuple(name for name in definition.import_checks if importlib.util.find_spec(name) is None)

    def _remove_path(self, path: Path) -> None:
        root = self.paths.runtime_engines_dir.resolve()
        target = path.resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Odmowa usuniecia poza folderem engines: {target}")
        try:
            if target.is_dir():
                shutil.rmtree(target, onexc=self._rmtree_onexc)
            elif target.exists():
                target.unlink()
        except PermissionError as exc:
            raise EngineRemovalError(
                "Nie moge usunac silnika, bo Windows blokuje plik w jego venv. "
                "Najczesciej oznacza to, ze nadal dziala proces generowania albo python z tego silnika. "
                "Zamknij generowanie/aplikacje albo zakoncz wiszacy proces python i sprobuj ponownie."
            ) from exc

    def _rmtree_onexc(self, func, path, exc) -> None:
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            func(path)
        except Exception:
            raise exc

    def _ensure_venv(self, venv_dir: Path, log) -> None:
        python_path = self._venv_python(venv_dir.parent)
        if python_path.exists():
            log.write("Venv istnieje, pomijam tworzenie.\n\n")
            return
        log.write("Tworzenie venv...\n")
        log.flush()
        temp_env = self._venv_creation_env(venv_dir)
        old_env = {name: os.environ.get(name) for name in ("TEMP", "TMP")}
        try:
            os.environ.update({name: temp_env[name] for name in ("TEMP", "TMP")})
            builder = venv.EnvBuilder(with_pip=False, clear=False)
            builder.create(venv_dir)
            self._bootstrap_venv_pip(venv_dir, temp_env, log)
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        log.write("Venv utworzony.\n\n")

    def _bootstrap_venv_pip(self, venv_dir: Path, env: dict[str, str], log) -> None:
        bundled_dir = Path(sys.base_prefix) / "Lib" / "ensurepip" / "_bundled"
        command = [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(venv_dir),
            "install",
            "--no-index",
            "--find-links",
            str(bundled_dir),
            "pip",
        ]
        log.write("> " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        log.write(result.stdout or "")
        log.write(f"\nKod wyjscia: {result.returncode}\n\n")
        log.flush()
        if result.returncode != 0:
            raise RuntimeError("Nie udalo sie zainstalowac pip w venv lokalnego TTS.")

    def _venv_creation_env(self, venv_dir: Path) -> dict[str, str]:
        env = dict(os.environ)
        temp_dir = venv_dir.parent / "temp" / "venv"
        temp_dir.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(temp_dir)
        env["TMP"] = str(temp_dir)
        return env

    def _run_logged(self, command: list[str], log, env: dict[str, str] | None = None) -> None:
        log.write("> " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.write(f"\nKod wyjscia: {return_code}\n\n")
        log.flush()
        if return_code != 0:
            raise RuntimeError(f"Instalacja TTS nie powiodla sie na komendzie: {' '.join(command)}")

    def _run_logged_step(self, label: str, command: list[str], log, env: dict[str, str] | None = None) -> None:
        log.write(f"Etap: {label}\n")
        log.flush()
        if env is None:
            self._run_logged(command, log)
        else:
            self._run_logged(command, log, env=env)
        log.write(f"Etap OK: {label}\n\n")
        log.flush()

    def _run_pip_check(self, python_path: Path, spec, log) -> None:
        command = [str(python_path), "-m", "pip", "check"]
        log.write("Etap: Kontrola spojnosci zaleznosci pip\n")
        log.write("> " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout or ""
        log.write(output)
        log.write(f"\nKod wyjscia: {result.returncode}\n")
        problems = [line.strip() for line in output.splitlines() if line.strip()]
        unexpected = [line for line in problems if not self._is_allowed_pip_check_line(spec, line)]
        if result.returncode != 0 and not unexpected:
            log.write("Pip check: tylko znane konflikty metadanych, zaakceptowane przez spec LektorAI.\n\n")
            log.flush()
            return
        if result.returncode == 0:
            log.write("Etap OK: Kontrola spojnosci zaleznosci pip\n\n")
            log.flush()
            return
        log.write("Pip check: wykryto nieoczekiwane konflikty zaleznosci.\n\n")
        log.flush()
        raise RuntimeError("Instalacja TTS nie powiodla sie na etapie pip check.")

    def _is_allowed_pip_check_line(self, spec, line: str) -> bool:
        return any(line.startswith(prefix) for prefix in spec.allowed_pip_check_prefixes)

    def _torch_install_command(self, python_path: Path, spec, constraints_path: Path) -> list[str]:
        return [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--index-url",
            spec.torch_index_url,
            *spec.torch_requirements,
            "-c",
            str(constraints_path),
        ]

    def _requirements_install_command(self, python_path: Path, requirements_path: Path, constraints_path: Path) -> list[str]:
        return [str(python_path), "-m", "pip", "install", "-r", str(requirements_path), "-c", str(constraints_path)]

    def _no_deps_requirements_install_command(self, python_path: Path, requirements_path: Path) -> list[str]:
        return [str(python_path), "-m", "pip", "install", "--no-deps", "-r", str(requirements_path)]

    def _build_tools_install_command(self, python_path: Path) -> list[str]:
        return [str(python_path), "-m", "pip", "install", "wheel", "setuptools"]

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        packages = str(self.paths.app_packages_dir)
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(part for part in (packages, current) if part)
        return env

    def _run_import_checks(self, definition: EngineDefinition, python_path: Path, log) -> None:
        try:
            spec = get_install_spec(definition.engine_id)
        except Exception:
            return
        if not spec.import_checks:
            return
        script = self._import_diagnostics_script(spec.import_checks)
        self._run_logged([str(python_path), "-c", script], log, env=self._worker_env())

    def _import_missing_script(self, names: tuple[str, ...]) -> str:
        return (
            "import importlib.util, sys; "
            f"names={list(names)!r}; "
            "missing=[n for n in names if importlib.util.find_spec(n) is None]; "
            "print('\\n'.join(missing)); "
            "sys.exit(1 if missing else 0)"
        )

    def _import_diagnostics_script(self, names: tuple[str, ...]) -> str:
        return (
            "import importlib, importlib.util, sys; "
            f"names={list(names)!r}; "
            "missing=[]; "
            "\nfor name in names:\n"
            "    spec=importlib.util.find_spec(name)\n"
            "    if spec is None:\n"
            "        print(f'IMPORT MISSING {name}')\n"
            "        missing.append(name)\n"
            "        continue\n"
            "    module=importlib.import_module(name)\n"
            "    origin=getattr(module, '__file__', None) or getattr(spec, 'origin', '') or 'built-in'\n"
            "    print(f'IMPORT OK {name} {origin}')\n"
            "sys.exit(1 if missing else 0)"
        )

    def _emit(self, progress: ProgressFn | None, message: str) -> None:
        if progress is not None:
            progress(message)

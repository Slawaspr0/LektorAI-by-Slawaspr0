from __future__ import annotations

import sys
from pathlib import Path


APP_NAME = "LektorAI by Slawaspr0"
APP_VERSION = "v1.4"
APP_DIR = Path(__file__).resolve().parent
APP_PACKAGES_DIR = APP_DIR / "packages"
if APP_PACKAGES_DIR.is_dir() and str(APP_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(APP_PACKAGES_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def main() -> int:
    if any(arg in {"--help", "-h", "/?"} for arg in sys.argv[1:]):
        print(_usage())
        return 0

    if "--version" in sys.argv[1:]:
        print(f"{APP_NAME} {APP_VERSION}", flush=True)
        return 0

    if any(arg in {"--list-engines", "--engines"} for arg in sys.argv[1:]):
        from app.core.paths import build_paths
        from app.engines.manager import EngineManager
        from app.engines.status import format_engine_state

        paths = build_paths(APP_DIR)
        manager = EngineManager(paths)
        for state in manager.list_states():
            components = ", ".join(state.components)
            suffix = f" | {components}" if components else ""
            print(f"{format_engine_state(state)}{suffix}")
        return 0

    if "--self-test" in sys.argv[1:]:
        from app.core.self_test import run_self_test

        for message in run_self_test(APP_DIR):
            print(message)
        print("self-test: OK")
        return 0

    if any(arg in {"--diagnose", "--diagnostics", "--diag"} for arg in sys.argv[1:]):
        from app.core.diagnostics import collect_diagnostics, format_diagnostics
        from app.core.paths import build_paths
        from app.engines.manager import EngineManager

        paths = build_paths(APP_DIR)
        print(format_diagnostics(collect_diagnostics(paths, EngineManager(paths))))
        return 0

    if "--preflight" in sys.argv[1:]:
        from app.core.paths import build_paths
        from app.core.preflight import build_preflight_report
        from app.engines.manager import EngineManager

        paths = build_paths(APP_DIR)
        ok, lines = build_preflight_report(paths, EngineManager(paths))
        print("\n".join(lines))
        return 0 if ok else 1

    engine_to_install = _arg_value("--install-engine")
    if engine_to_install:
        from app.core.paths import build_paths
        from app.engines.manager import EngineManager
        from app.engines.schemas import EngineKind

        paths = build_paths(APP_DIR)
        manager = EngineManager(paths)
        if engine_to_install not in manager.definitions:
            print(f"Nieznany silnik TTS: {engine_to_install}")
            return 2
        if manager.definitions[engine_to_install].kind != EngineKind.LOCAL:
            print(f"Silnik nie jest lokalnym TTS do instalacji: {engine_to_install}")
            return 2
        manager.install_local_engine(engine_to_install, print, torch_variant=_arg_value("--torch-cuda"))
        return 0

    engine_to_preview = _arg_value("--engine-install-plan")
    if engine_to_preview:
        from app.core.paths import build_paths
        from app.engines.manager import EngineManager
        from app.engines.schemas import EngineKind

        paths = build_paths(APP_DIR)
        manager = EngineManager(paths)
        if engine_to_preview not in manager.definitions:
            print(f"Nieznany silnik TTS: {engine_to_preview}")
            return 2
        if manager.definitions[engine_to_preview].kind != EngineKind.LOCAL:
            print(f"Silnik nie jest lokalnym TTS do instalacji: {engine_to_preview}")
            return 2
        print("\n".join(manager.local_install_preview(engine_to_preview, torch_variant=_arg_value("--torch-cuda"))))
        return 0

    engine_to_update = _arg_value("--update-worker")
    if engine_to_update:
        from app.core.paths import build_paths
        from app.engines.manager import EngineManager
        from app.engines.schemas import EngineKind

        paths = build_paths(APP_DIR)
        manager = EngineManager(paths)
        if engine_to_update not in manager.definitions:
            print(f"Nieznany silnik TTS: {engine_to_update}")
            return 2
        if manager.definitions[engine_to_update].kind != EngineKind.LOCAL:
            print(f"Silnik nie ma lokalnego workera: {engine_to_update}")
            return 2
        if not manager.local_runtime_exists(engine_to_update):
            print(f"Silnik nie jest jeszcze zainstalowany: {engine_to_update}")
            return 2
        worker_path = manager.install_worker_script(engine_to_update)
        print(f"Worker zaktualizowany: {worker_path}")
        return 0

    engine_to_remove_keep_settings = _arg_value("--remove-engine-keep-settings")
    if engine_to_remove_keep_settings:
        from app.cli.engine_commands import remove_engine_command
        from app.core.paths import build_paths
        from app.engines.manager import EngineManager

        paths = build_paths(APP_DIR)
        code, message = remove_engine_command(EngineManager(paths), engine_to_remove_keep_settings, True)
        print(message)
        return code

    engine_to_remove = _arg_value("--remove-engine")
    if engine_to_remove:
        from app.cli.engine_commands import remove_engine_command
        from app.core.paths import build_paths
        from app.engines.manager import EngineManager

        paths = build_paths(APP_DIR)
        code, message = remove_engine_command(EngineManager(paths), engine_to_remove, False)
        print(message)
        return code

    from app.ui.main_window import run_app

    return run_app(APP_DIR)


def _arg_value(name: str) -> str:
    prefix = f"{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix) :].strip()
    if name not in sys.argv:
        return ""
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"Brak wartosci dla {name}")
    return sys.argv[index + 1].strip()


def _usage() -> str:
    return (
        f"{APP_NAME}\n"
        "  START.py                         uruchamia GUI\n"
        "  START.py --version               wersja aplikacji\n"
        "  START.py --diagnose              diagnostyka tekstowa\n"
        "  START.py --preflight             sprawdza gotowosc do podstawowego testu\n"
        "  START.py --self-test             szybki test bez GUI i bez TTS\n"
        "  START.py --list-engines          lista silnikow TTS\n"
        "  START.py --engine-install-plan ID pokazuje plan instalacji lokalnego TTS\n"
        "  START.py --engine-install-plan ID --torch-cuda cu128 pokazuje plan dla wariantu PyTorch\n"
        "  START.py --install-engine ID     instaluje lokalny TTS: chatterbox, omnivoice, piper, coqui_xtts, supertonic\n"
        "  START.py --install-engine ID --torch-cuda cu128 instaluje wariant PyTorch dla nowszych kart\n"
        "  START.py --update-worker ID      aktualizuje worker.py lokalnego TTS\n"
        "  START.py --remove-engine ID      usuwa caly folder lokalnego TTS\n"
        "  START.py --remove-engine-keep-settings ID usuwa runtime lokalnego TTS, zostawia config i slownik\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())

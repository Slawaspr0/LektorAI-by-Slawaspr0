from __future__ import annotations

from app.engines.manager import EngineManager
from app.engines.schemas import EngineKind


def remove_engine_command(manager: EngineManager, engine_id: str, keep_settings: bool) -> tuple[int, str]:
    if engine_id not in manager.definitions:
        return 2, f"Nieznany silnik TTS: {engine_id}"
    if manager.definitions[engine_id].kind != EngineKind.LOCAL:
        return 2, f"Silnik wbudowany nie ma lokalnego runtime do odinstalowania: {engine_id}"
    if not manager.engine_dir_exists(engine_id):
        return 0, f"Silnik nie ma lokalnego runtime do usuniecia: {engine_id}"
    if keep_settings:
        if not manager.removable_payload_exists(engine_id):
            return 0, f"Silnik nie ma lokalnego runtime do usuniecia, ustawienia pozostaja: {engine_id}"
        manager.remove_engine_keep_user_settings(engine_id)
        return 0, f"Runtime silnika usuniety, ustawienia zachowane: {engine_id}"
    manager.remove_engine_completely(engine_id)
    return 0, f"Silnik usuniety calkowicie: {engine_id}"

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EngineKind(str, Enum):
    LOCAL = "local"
    INTERNET = "internet"


class EngineStatus(str, Enum):
    READY = "gotowy"
    NOT_INSTALLED = "niezainstalowany"
    INSTALLED_NO_MODEL = "zainstalowany, model niepobrany"
    REQUIRES_CONFIG = "wymaga konfiguracji"
    REQUIRES_INTERNET = "wymaga internetu"
    BROKEN = "wymaga naprawy"


@dataclass(frozen=True)
class EngineDefinition:
    engine_id: str
    display_name: str
    kind: EngineKind
    built_in: bool = False
    requires_venv: bool = False
    requires_model: bool = False
    requires_api_key: bool = False
    pipeline_ready: bool = True
    import_checks: tuple[str, ...] = ()
    default_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineState:
    definition: EngineDefinition
    status: EngineStatus
    selectable: bool
    reason: str = ""
    components: tuple[str, ...] = ()

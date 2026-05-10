from __future__ import annotations

from app.engines.schemas import EngineState


def format_engine_state(state: EngineState) -> str:
    text = f"{state.definition.display_name} - {state.status.value}"
    if state.reason:
        text += f" ({state.reason})"
    return text

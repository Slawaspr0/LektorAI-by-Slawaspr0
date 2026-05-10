from __future__ import annotations

from dataclasses import dataclass


TORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"
TORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"


@dataclass(frozen=True)
class EngineInstallSpec:
    engine_id: str
    torch_requirements: tuple[str, ...]
    requirements: tuple[str, ...]
    constraints: tuple[str, ...]
    no_deps_requirements: tuple[str, ...] = ()
    allowed_pip_check_prefixes: tuple[str, ...] = ()
    import_checks: tuple[str, ...] = ()
    torch_index_url: str = TORCH_CU126_INDEX


INSTALL_SPECS: dict[str, EngineInstallSpec] = {
    "chatterbox": EngineInstallSpec(
        engine_id="chatterbox",
        torch_requirements=("torch==2.6.0", "torchaudio==2.6.0"),
        requirements=(
            "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git",
            "soundfile",
        ),
        constraints=(
            "torch==2.6.0",
            "torchaudio==2.6.0",
        ),
        import_checks=("chatterbox", "soundfile"),
    ),
    "omnivoice": EngineInstallSpec(
        engine_id="omnivoice",
        torch_requirements=("torch==2.8.0+cu128", "torchaudio==2.8.0+cu128"),
        requirements=(
            "omnivoice==0.1.5",
            "soundfile",
        ),
        constraints=(
            "torch==2.8.0+cu128",
            "torchaudio==2.8.0+cu128",
        ),
        import_checks=("omnivoice", "soundfile"),
        torch_index_url=TORCH_CU128_INDEX,
    ),
}


def get_install_spec(engine_id: str) -> EngineInstallSpec:
    return INSTALL_SPECS[engine_id]

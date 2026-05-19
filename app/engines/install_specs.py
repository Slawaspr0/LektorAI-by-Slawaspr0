from __future__ import annotations

from dataclasses import dataclass


TORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"
TORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
CHATTERBOX_UPSTREAM_REQUIREMENT = "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git"
CHATTERBOX_RUNTIME_REQUIREMENTS = (
    "numpy>=1.24.0,<2.0.0; python_version < '3.13'",
    "numpy>=2.0.0; python_version >= '3.13'",
    "librosa==0.11.0",
    "s3tokenizer",
    "transformers==5.2.0",
    "diffusers==0.29.0",
    "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master",
    "conformer==0.3.2",
    "safetensors==0.5.3",
    "spacy-pkuseg",
    "pykakasi==2.3.0",
    "gradio==6.8.0",
    "pyloudnorm",
    "omegaconf",
    "soundfile",
)


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
    package_installer: str = "pip"
    local_repo_folder: str = ""
    editable_git_url: str = ""
    editable_extra: str = ""


@dataclass(frozen=True)
class EngineInstallVariant:
    variant_id: str
    label: str
    description: str
    spec: EngineInstallSpec


INSTALL_SPECS: dict[str, EngineInstallSpec] = {
    "chatterbox": EngineInstallSpec(
        engine_id="chatterbox",
        torch_requirements=("torch==2.6.0", "torchaudio==2.6.0"),
        requirements=(
            CHATTERBOX_UPSTREAM_REQUIREMENT,
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
    "piper": EngineInstallSpec(
        engine_id="piper",
        torch_requirements=(),
        requirements=(
            "piper-tts==1.4.2",
            "soundfile",
        ),
        constraints=(),
        import_checks=("piper", "soundfile"),
    ),
    "supertonic": EngineInstallSpec(
        engine_id="supertonic",
        torch_requirements=(),
        requirements=("supertonic",),
        constraints=(),
        import_checks=("supertonic",),
        package_installer="uv",
    ),
    "coqui_xtts": EngineInstallSpec(
        engine_id="coqui_xtts",
        torch_requirements=("torch==2.8.0+cu126", "torchaudio==2.8.0+cu126"),
        requirements=(
            "coqui-tts @ git+https://github.com/idiap/coqui-ai-TTS.git",
            "soundfile",
        ),
        constraints=(
            "torch==2.8.0+cu126",
            "torchaudio==2.8.0+cu126",
            "transformers>=4.57.6,<5",
            "huggingface-hub>=0.36.0,<1",
        ),
        import_checks=("TTS", "soundfile"),
        torch_index_url=TORCH_CU126_INDEX,
    ),
}


INSTALL_VARIANTS: dict[str, tuple[EngineInstallVariant, ...]] = {
    "chatterbox": (
        EngineInstallVariant(
            variant_id="cu126",
            label="PyTorch CU126 - zalecany",
            description="Standardowy wariant dla starszych i sprawdzonych kart NVIDIA.",
            spec=INSTALL_SPECS["chatterbox"],
        ),
        EngineInstallVariant(
            variant_id="cu128",
            label="PyTorch CU128 - wersja dla nowszych kart",
            description="Wariant dla nowszych kart NVIDIA. Podbija PyTorch do 2.8 CU128.",
            spec=EngineInstallSpec(
                engine_id="chatterbox",
                torch_requirements=("torch==2.8.0+cu128", "torchaudio==2.8.0+cu128"),
                requirements=CHATTERBOX_RUNTIME_REQUIREMENTS,
                no_deps_requirements=(CHATTERBOX_UPSTREAM_REQUIREMENT,),
                constraints=(
                    "torch==2.8.0+cu128",
                    "torchaudio==2.8.0+cu128",
                ),
                allowed_pip_check_prefixes=(
                    "chatterbox-tts 0.1.7 has requirement torch==2.6.0",
                    "chatterbox-tts 0.1.7 has requirement torchaudio==2.6.0",
                ),
                import_checks=("chatterbox", "soundfile"),
                torch_index_url=TORCH_CU128_INDEX,
            ),
        ),
    ),
    "coqui_xtts": (
        EngineInstallVariant(
            variant_id="cu126",
            label="PyTorch CU126 - zalecany",
            description="Standardowy wariant dla starszych i sprawdzonych kart NVIDIA.",
            spec=INSTALL_SPECS["coqui_xtts"],
        ),
        EngineInstallVariant(
            variant_id="cu128",
            label="PyTorch CU128 - wersja dla nowszych kart",
            description="Wariant dla nowszych kart NVIDIA.",
            spec=EngineInstallSpec(
                engine_id="coqui_xtts",
                torch_requirements=("torch==2.8.0+cu128", "torchaudio==2.8.0+cu128"),
                requirements=INSTALL_SPECS["coqui_xtts"].requirements,
                constraints=(
                    "torch==2.8.0+cu128",
                    "torchaudio==2.8.0+cu128",
                    "transformers>=4.57.6,<5",
                    "huggingface-hub>=0.36.0,<1",
                ),
                import_checks=INSTALL_SPECS["coqui_xtts"].import_checks,
                torch_index_url=TORCH_CU128_INDEX,
            ),
        ),
    ),
}


def list_install_variants(engine_id: str) -> tuple[EngineInstallVariant, ...]:
    return INSTALL_VARIANTS.get(engine_id, ())


def get_install_variant(engine_id: str, variant_id: str) -> EngineInstallVariant:
    for variant in list_install_variants(engine_id):
        if variant.variant_id == variant_id:
            return variant
    raise ValueError(f"Silnik {engine_id} nie ma wariantu instalacji: {variant_id}")


def get_install_spec(engine_id: str, variant_id: str | None = None) -> EngineInstallSpec:
    if variant_id:
        return get_install_variant(engine_id, variant_id).spec
    return INSTALL_SPECS[engine_id]

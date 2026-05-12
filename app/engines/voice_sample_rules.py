from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceSampleRule:
    engine_id: str
    config_key: str
    label: str
    recommended_min_seconds: float
    recommended_max_seconds: float
    max_seconds: float
    min_seconds: float = 1.0
    sample_rate: int = 24000
    optional: bool = True


VOICE_SAMPLE_RULES: dict[str, VoiceSampleRule] = {
    "chatterbox": VoiceSampleRule(
        engine_id="chatterbox",
        config_key="audio_prompt_path",
        label="probka glosu Chatterbox",
        recommended_min_seconds=3.0,
        recommended_max_seconds=10.0,
        max_seconds=30.0,
    ),
    "omnivoice": VoiceSampleRule(
        engine_id="omnivoice",
        config_key="reference_audio_path",
        label="probka glosu OmniVoice",
        recommended_min_seconds=3.0,
        recommended_max_seconds=10.0,
        max_seconds=10.0,
    ),
    "coqui_xtts": VoiceSampleRule(
        engine_id="coqui_xtts",
        config_key="speaker_wav_path",
        label="probka glosu Coqui XTTS",
        recommended_min_seconds=3.0,
        recommended_max_seconds=10.0,
        max_seconds=10.0,
    ),
}


def voice_sample_rule(engine_id: str) -> VoiceSampleRule | None:
    return VOICE_SAMPLE_RULES.get(str(engine_id or "").strip())


def voice_sample_config_key(engine_id: str) -> str:
    rule = voice_sample_rule(engine_id)
    return rule.config_key if rule is not None else ""


def voice_sample_duration_help(engine_id: str) -> str:
    rule = voice_sample_rule(engine_id)
    if rule is None:
        return ""
    optional_text = "Opcjonalna probka glosu" if rule.optional else "Probka glosu"
    return (
        f"{optional_text} WAV, MP3 albo FLAC. Zalecane {format_seconds_range(rule.recommended_min_seconds, rule.recommended_max_seconds)} "
        f"czystej polskiej mowy; maksymalnie "
        f"{format_seconds(rule.max_seconds)}. Program przygotuje roboczy WAV mono {rule.sample_rate // 1000} kHz."
    )


def validate_voice_sample_duration(engine_id: str, duration_seconds: float) -> list[str]:
    rule = voice_sample_rule(engine_id)
    if rule is None:
        return []
    try:
        duration = float(duration_seconds)
    except Exception:
        return [f"{rule.label}: nie mozna odczytac dlugosci pliku audio."]
    if duration <= 0:
        return [f"{rule.label}: plik audio ma nieprawidlowa dlugosc."]
    if duration < rule.min_seconds:
        return [
            f"{rule.label}: plik jest za krotki ({format_seconds(duration)}). "
            f"Daj przynajmniej {format_seconds(rule.min_seconds)} czystej mowy; zalecane "
            f"{format_seconds_range(rule.recommended_min_seconds, rule.recommended_max_seconds)}."
        ]
    if duration > rule.max_seconds:
        return [
            f"{rule.label}: probka jest za dluga ({format_seconds(duration)}). "
            f"Maksimum dla tego silnika to {format_seconds(rule.max_seconds)}. "
            f"Skroc probke do ok. {format_seconds_range(rule.recommended_min_seconds, rule.recommended_max_seconds)} czystej mowy."
        ]
    return []


def format_seconds(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return f"{int(number)}s"
    return f"{number:.1f}s".replace(".", ",")


def format_seconds_range(min_value: float, max_value: float) -> str:
    return f"{format_seconds(min_value).removesuffix('s')}-{format_seconds(max_value)}"

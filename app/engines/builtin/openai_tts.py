from __future__ import annotations

from pathlib import Path


def synthesize_openai_wav_sync(
    text: str,
    output_path: Path,
    model: str,
    voice: str,
    api_key: str,
    instructions: str = "",
) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Brak biblioteki openai. Zainstaluj requirements aplikacji.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key.strip() or None)
    request = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav",
    }
    if instructions.strip():
        request["instructions"] = instructions.strip()

    with client.audio.speech.with_streaming_response.create(**request) as response:
        response.stream_to_file(output_path)

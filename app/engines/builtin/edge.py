from __future__ import annotations

import asyncio
from pathlib import Path


async def synthesize_edge_mp3(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    pitch: str,
) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Brak biblioteki edge-tts. Zainstaluj requirements aplikacji.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            output_path.unlink(missing_ok=True)
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
            await communicate.save(str(output_path))
            return
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            await asyncio.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Edge TTS: nie udalo sie wygenerowac mowy po {max_attempts} probach: {last_error}") from last_error


def synthesize_edge_mp3_sync(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    pitch: str,
) -> None:
    asyncio.run(synthesize_edge_mp3(text, output_path, voice, rate, pitch))

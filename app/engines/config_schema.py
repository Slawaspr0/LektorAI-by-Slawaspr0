from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FieldType = Literal["str", "path", "int", "float", "bool", "choice", "percent_slider", "hz_slider"]


WHISPER_QC_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
)

DEVICE_OPTIONS: tuple[str, ...] = ("auto", "cpu", "cuda", "cuda:0", "cuda:1")

EDGE_POLISH_VOICES: tuple[str, ...] = ("pl-PL-MarekNeural", "pl-PL-ZofiaNeural")
EDGE_POLISH_VOICE_LABELS: tuple[str, ...] = ("Marek", "Zofia")
EDGE_RATE_MIN = -100
EDGE_RATE_MAX = 100
EDGE_PITCH_MIN = -50
EDGE_PITCH_MAX = 50

WHISPER_QC_MODEL_TOOLTIP = (
    "Model faster-whisper do kontroli, czy TTS wypowiedzial tekst z napisow. "
    "tiny/base: najszybsze, ale mniej dokladne; small: lekki domyslny kompromis; medium: dokladniejszy, wolniejszy; "
    "large-v1: starszy duzy model; large-v2: stabilniejszy od v1; large-v3/large: najlepsza jakosc, najciezsze; "
    "large-v3-turbo/turbo: szybsza wersja large-v3, dobry wybor przy mocnym GPU. "
    "Model pobiera sie dopiero przy pierwszym uzyciu Whisper QC."
)

DEVICE_TOOLTIP = "auto: wybiera najlepsze dostepne urzadzenie; cpu: procesor; cuda: domyslna karta NVIDIA; cuda:0/cuda:1: konkretna karta GPU."


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    field_type: FieldType
    tooltip: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    secret: bool = False
    visible: bool = True
    options: tuple[str, ...] = ()
    option_labels: tuple[str, ...] = ()
    show_help: bool = True


DIAGNOSTIC_REPORT_FIELD_KEYS: tuple[str, ...] = (
    "save_processed_subtitles",
    "save_run_reports",
)

DIAGNOSTIC_SEGMENT_FIELD_KEYS: tuple[str, ...] = (
    "save_lektor_segments",
)

DIAGNOSTIC_TRACK_FIELD_KEYS: tuple[str, ...] = (
    "save_lektor_track_before_normalization",
    "save_lektor_track_after_normalization",
)

DIAGNOSTIC_MIX_FIELD_KEYS: tuple[str, ...] = (
    "save_audio_mix_steps",
)

DIAGNOSTIC_FIELD_KEYS: tuple[str, ...] = (
    *DIAGNOSTIC_REPORT_FIELD_KEYS,
    *DIAGNOSTIC_SEGMENT_FIELD_KEYS,
    *DIAGNOSTIC_TRACK_FIELD_KEYS,
    *DIAGNOSTIC_MIX_FIELD_KEYS,
)

AUDIO_QC_FIELD_KEYS: tuple[str, ...] = (
    "audio_qc_enabled",
    "audio_qc_retry_attempts",
)

SPEECH_QC_FIELD_KEYS: tuple[str, ...] = (
    "whisper_qc_enabled",
    "whisper_qc_retry_attempts",
    "whisper_qc_model",
    "whisper_qc_min_similarity",
)


DIAGNOSTIC_OUTPUT_FIELDS: tuple[ConfigField, ...] = (
    ConfigField("save_processed_subtitles", "Napisy po obrobce", "bool", "Zachowuje plik SRT po oczyszczeniu napisow i slowniku, dokladnie ten uzyty do generowania lektora."),
    ConfigField("save_run_reports", "Raporty techniczne", "bool", "Zachowuje manifest segmentow, raport Audio QC, podsumowanie runu i pomocnicze raporty diagnostyczne."),
    ConfigField("save_lektor_segments", "Segmenty lektora", "bool", "Zachowuje pojedyncze pliki audio wygenerowane przez TTS dla kazdej kwestii."),
    ConfigField("save_lektor_track_before_normalization", "Sciezka przed normalizacja", "bool", "Zachowuje kompletna sciezke dzwiekowa lektora przed normalizacja glosnosci."),
    ConfigField("save_lektor_track_after_normalization", "Sciezka po normalizacji", "bool", "Zachowuje kompletna sciezke dzwiekowa lektora po normalizacji glosnosci."),
    ConfigField("save_audio_mix_steps", "Etapy miksowania audio", "bool", "Zachowuje robocze pliki audio z etapow miksowania: wyciagniete tlo oraz przygotowane sciezki PL 2.0/5.1."),
)


CONFIG_SCHEMAS: dict[str, tuple[ConfigField, ...]] = {
    "edge": (
        ConfigField("voice", "Glos lektora", "choice", "", options=EDGE_POLISH_VOICES, option_labels=EDGE_POLISH_VOICE_LABELS, show_help=False),
        ConfigField("rate", "Predkosc", "percent_slider", "Predkosc mowy lektora.", EDGE_RATE_MIN, EDGE_RATE_MAX, 1),
        ConfigField("pitch", "Barwa glosu", "hz_slider", "Barwa glosu lektora.", EDGE_PITCH_MIN, EDGE_PITCH_MAX, 1),
        ConfigField("edge_apply_segment_fade", "Przytnij i wygladz brzegi", "bool", "Po wlaczeniu przycina poczatek i koniec pliku lektora o podane przez uzytkownika wartosci."),
        ConfigField("edge_trim_start_ms", "Utnij poczatek (ms)", "int", "Ile milisekund uciac z poczatku pliku lektora.", 0, 1000, 1),
        ConfigField("edge_trim_end_ms", "Utnij koniec (ms)", "int", "Ile milisekund uciac z konca pliku lektora.", 0, 2000, 1),
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", "Ile razy modul bedzie powtarzal generowanie mowy w przypadku braku zgodnosci.", 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "openai": (
        ConfigField("api_key", "Klucz API", "str", "Klucz API OpenAI. Bez internetu i klucza ten silnik nie wygeneruje mowy.", secret=True),
        ConfigField("model", "Model", "str", "Nazwa modelu TTS OpenAI, domyslnie gpt-4o-mini-tts."),
        ConfigField("voice", "Glos", "str", "Nazwa glosu OpenAI, np. marin, cedar, coral, alloy, nova, onyx."),
        ConfigField("instructions", "Instrukcja glosu", "str", "Instrukcja stylu mowy dla modelu TTS, np. spokojny polski lektor."),
        ConfigField("audio_qc_enabled", "Wlacz kontrole audio", "bool", "Wlacza techniczna kontrole pliku audio: cisza, dlugosc segmentu, glosny poczatek lub koniec, clipping i podobne artefakty."),
        ConfigField("audio_qc_retry_attempts", "Liczba prob kontroli audio", "int", "Maksymalna liczba prob generowania segmentu, jesli kontrola audio wykryje podejrzany wynik.", 1, 5, 1),
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", "Ile razy modul bedzie powtarzal generowanie mowy w przypadku braku zgodnosci.", 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "chatterbox": (
        ConfigField("device", "Urzadzenie", "choice", DEVICE_TOOLTIP, options=DEVICE_OPTIONS),
        ConfigField(
            "t3_model",
            "Wersja modelu Chatterbox",
            "choice",
            "Wybiera wersje wielojezycznego modelu Chatterbox. v2 to domyslna wersja autora; v3 to nowszy wariant do testow jakosci.",
            options=("v2", "v3"),
            option_labels=("v2 - domyslna", "v3 - nowsza testowa"),
        ),
        ConfigField("audio_prompt_path", "Probka glosu", "path", "Opcjonalna probka glosu WAV, MP3 albo FLAC. Aplikacja przygotuje kopie robocza WAV mono 24 kHz z lekka normalizacja i czyszczeniem."),
        ConfigField(
            "trim_leading_silence",
            "Wycinanie poczatkowej ciszy",
            "bool",
            "Po wlaczeniu program wykrywa cisze na poczatku segmentu Chatterbox i usuwa ja z zapasem 50 ms przed mowa.",
        ),
        ConfigField(
            "cfg_weight",
            "Stabilnosc tekstu (CFG)",
            "float",
            "Zakres 0.0-2.0. Wyzej = model mocniej trzyma sie tekstu, ale glos moze brzmiec mniej naturalnie; nizej = wiecej naturalnosci, ale wieksze ryzyko odjazdow od tekstu.",
            0.0,
            2.0,
            0.05,
        ),
        ConfigField(
            "exaggeration",
            "Ekspresja glosu",
            "float",
            "Zakres 0.0-2.0. Wyzej = bardziej emocjonalny i ekspresyjny glos; nizej = spokojniejszy, bardziej neutralny lektor.",
            0.0,
            2.0,
            0.05,
        ),
        ConfigField("seed", "Seed", "int", "Opcjonalne ziarno losowosci. Stala wartosc pomaga powtarzalnosci, puste pole zostawia domyslne zachowanie.", 0, 2147483647, 1, visible=False),
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", "Ile razy modul bedzie powtarzal generowanie mowy w przypadku braku zgodnosci.", 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "omnivoice": (
        ConfigField("device", "Urzadzenie", "choice", DEVICE_TOOLTIP, options=DEVICE_OPTIONS),
        ConfigField("reference_audio_path", "Probka glosu", "path", "Probka glosu WAV, MP3 albo FLAC. Zalecane 3-10 sekund czystej polskiej mowy; program przygotuje roboczy WAV mono 24 kHz dla OmniVoice."),
        ConfigField("reference_text", "Tekst probki glosu", "str", "Opcjonalna transkrypcja probki glosu. Gdy ja podasz, OmniVoice nie musi zgadywac tekstu probki przez ASR; zwykle daje to stabilniejsze klonowanie i mniej pobierania dodatkowych modeli."),
        ConfigField(
            "num_step",
            "Kroki inferencji",
            "int",
            "Zakres 4-64, domyslnie 32. Wieksza liczba krokow zwykle poprawia jakosc i stabilnosc mowy, ale spowalnia generowanie; mniejsza przyspiesza kosztem jakosci.",
            4,
            64,
            1,
        ),
        ConfigField(
            "guidance_scale",
            "CFG",
            "float",
            "Zakres 0.0-4.0, domyslnie 2.0. Wyzej = model mocniej trzyma sie tekstu i probki glosu, ale moze brzmiec mniej naturalnie; nizej = wiecej swobody i naturalnosci, ale wieksze ryzyko odjazdow.",
            0.0,
            4.0,
            0.1,
        ),
        ConfigField(
            "speed",
            "Predkosc",
            "float",
            "Zakres 0.5-1.5, domyslnie 1.0. Wartosc 1.0 to normalne tempo; powyzej 1.0 lektor mowi szybciej, ponizej 1.0 wolniej.",
            0.5,
            1.5,
            0.05,
        ),
        ConfigField("denoise", "Denoise", "bool", "Wbudowany tryb czyszczenia OmniVoice. Po wlaczeniu model dostaje sygnal, aby generowac czystsza mowe; domyslnie wlaczone."),
        ConfigField("preprocess_prompt", "Przygotuj probke", "bool", "Ukryte ustawienie techniczne. W LektorAI domyslnie wylaczone, bo uzytkownik ma dostarczyc gotowa, dobra probke glosu.", visible=False),
        ConfigField("postprocess_output", "Wyczysc wynik", "bool", "Ukryte ustawienie techniczne. W LektorAI domyslnie wylaczone, bo fabryczne usuwanie ciszy OmniVoice potrafi ucinac koncowki slow.", visible=False),
        ConfigField("omnivoice_trim_edges", "Wycinanie ciszy na brzegach", "bool", "Po wlaczeniu program ostroznie wykrywa poczatek i koniec mowy, usuwa nadmiar ciszy/szumu tylko z brzegow segmentu i zostawia zapas, zeby nie uciac lektora."),
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", "Ile razy modul bedzie powtarzal generowanie mowy w przypadku braku zgodnosci.", 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
}


def fields_for(engine_id: str) -> tuple[ConfigField, ...]:
    return CONFIG_SCHEMAS.get(engine_id, ())


def visible_fields_for(engine_id: str) -> tuple[ConfigField, ...]:
    return tuple(field for field in fields_for(engine_id) if field.visible)


def is_diagnostic_field(field_or_key: ConfigField | str) -> bool:
    key = field_or_key.key if isinstance(field_or_key, ConfigField) else str(field_or_key)
    return key in DIAGNOSTIC_FIELD_KEYS


def is_audio_qc_field(field_or_key: ConfigField | str) -> bool:
    key = field_or_key.key if isinstance(field_or_key, ConfigField) else str(field_or_key)
    return key in AUDIO_QC_FIELD_KEYS


def is_speech_qc_field(field_or_key: ConfigField | str) -> bool:
    key = field_or_key.key if isinstance(field_or_key, ConfigField) else str(field_or_key)
    return key in SPEECH_QC_FIELD_KEYS


def normalize_config(engine_id: str, data: dict[str, Any]) -> dict[str, Any]:
    fields = fields_for(engine_id)
    if not fields:
        return dict(data)
    allowed = {field.key for field in fields}
    normalized = {key: value for key, value in data.items() if key in allowed}
    return normalized


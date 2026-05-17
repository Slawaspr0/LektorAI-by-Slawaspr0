from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.engines.voice_sample_rules import voice_sample_duration_help


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

DEVICE_OPTIONS: tuple[str, ...] = ("auto", "cpu")
WHISPER_QC_DEVICE_OPTIONS: tuple[str, ...] = ("cpu",)
WHISPER_QC_COMPUTE_TYPES: tuple[str, ...] = ("int8", "float16")
WHISPER_QC_COMPUTE_TYPE_LABELS: tuple[str, ...] = ("int8 - CPU\\GPU", "float16 - GPU")
WHISPER_QC_COMPUTE_TYPE_LABEL_BY_VALUE: dict[str, str] = dict(zip(WHISPER_QC_COMPUTE_TYPES, WHISPER_QC_COMPUTE_TYPE_LABELS))

EDGE_POLISH_VOICES: tuple[str, ...] = ("pl-PL-MarekNeural", "pl-PL-ZofiaNeural")
EDGE_POLISH_VOICE_LABELS: tuple[str, ...] = ("Marek", "Zofia")
PIPER_POLISH_VOICES: tuple[str, ...] = ("pl_PL-gosia-medium", "pl_PL-darkman-medium", "pl_PL-mc_speech-medium")
PIPER_POLISH_VOICE_LABELS: tuple[str, ...] = ("Gosia", "Darkman", "McSpeech")
COQUI_BUILTIN_SPEAKERS: tuple[str, ...] = ("Anna", "Craig Gutsy", "Ana Florence")
SUPERTONIC_VOICES: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")
SUPERTONIC_VOICE_LABELS: tuple[str, ...] = (
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
)
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

WHISPER_QC_RETRY_TOOLTIP = (
    "Liczba prob lacznie dla jednego segmentu. 1 = tylko oryginal, bez ponowien; "
    "5 = oryginal + maksymalnie 4 ponowienia."
)

DEVICE_TOOLTIP = "Auto wybiera najlepsze dostepne urzadzenie. Przy kilku kartach mozesz wybrac konkretna karte GPU."
WHISPER_QC_DEVICE_TOOLTIP = "Urzadzenie dla kontroli mowy. CPU jest najbezpieczniejsze. Przy kilku kartach mozesz ustawic kontrole mowy na innej karcie niz TTS."
WHISPER_QC_COMPUTE_TOOLTIP = "Tryb pracy kontroli mowy. int8 jest najbezpieczniejsze i zuzywa mniej pamieci; float16 moze byc szybsze na GPU, ale wymaga wiecej VRAM."


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
    "save_quality_report",
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
    "whisper_qc_device",
    "whisper_qc_compute_type",
    "whisper_qc_min_similarity",
)


DIAGNOSTIC_OUTPUT_FIELDS: tuple[ConfigField, ...] = (
    ConfigField("save_processed_subtitles", "Napisy po obrobce", "bool", "Zachowuje plik napisow po zastosowaniu slownika i poprawek programu. To ten tekst trafia do lektora."),
    ConfigField("save_quality_report", "Raport jakosci", "bool", "Zachowuje raport do porownywania ustawien TTS. Mozesz wkleic go do AI, zeby latwiej ocenic jakosc wyniku."),
    ConfigField("save_run_reports", "Raporty techniczne", "bool", "Zachowuje dodatkowe pliki z przebiegu pracy, przydatne przy szukaniu problemow z konkretnymi segmentami."),
    ConfigField("save_lektor_segments", "Segmenty lektora", "bool", "Zachowuje pojedyncze pliki audio dla kazdej kwestii."),
    ConfigField("save_lektor_track_before_normalization", "Sciezka przed normalizacja", "bool", "Zachowuje kompletna sciezke dzwiekowa lektora przed wyrownaniem glosnosci."),
    ConfigField("save_lektor_track_after_normalization", "Sciezka po normalizacji", "bool", "Zachowuje kompletna sciezke dzwiekowa lektora po wyrownaniu glosnosci."),
    ConfigField("save_audio_mix_steps", "Etapy miksowania audio", "bool", "Zachowuje posrednie pliki audio powstajace podczas laczenia lektora z tlem."),
)

OPEN_WORKSPACE_FIELD = ConfigField(
    "open_workspace_on_finish",
    "Otworz folder po pracy",
    "bool",
    "Po zakonczeniu otwiera folder roboczy LektorAI z wynikiem.",
)

TTS_TEXT_CLEANUP_FIELD = ConfigField(
    "normalize_tts_text",
    "Czyszczenie tekstu",
    "bool",
    "Zamienia nietypowe znaki z napisow na bezpieczne odpowiedniki przed generowaniem mowy.",
)


CONFIG_SCHEMAS: dict[str, tuple[ConfigField, ...]] = {
    "edge": (
        ConfigField("voice", "Glos lektora", "choice", "", options=EDGE_POLISH_VOICES, option_labels=EDGE_POLISH_VOICE_LABELS, show_help=False),
        ConfigField("rate", "Predkosc", "percent_slider", "Predkosc mowy lektora.", EDGE_RATE_MIN, EDGE_RATE_MAX, 1),
        ConfigField("pitch", "Barwa glosu", "hz_slider", "Barwa glosu lektora.", EDGE_PITCH_MIN, EDGE_PITCH_MAX, 1),
        ConfigField("edge_apply_segment_fade", "Przytnij i wygladz brzegi", "bool", "Po wlaczeniu przycina poczatek i koniec pliku lektora o podane przez uzytkownika wartosci."),
        ConfigField("edge_trim_start_ms", "Utnij poczatek (ms)", "int", "Ile milisekund uciac z poczatku pliku lektora.", 0, 1000, 1),
        ConfigField("edge_trim_end_ms", "Utnij koniec (ms)", "int", "Ile milisekund uciac z konca pliku lektora.", 0, 2000, 1),
        TTS_TEXT_CLEANUP_FIELD,
        OPEN_WORKSPACE_FIELD,
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", WHISPER_QC_RETRY_TOOLTIP, 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_device", "Urzadzenie", "choice", WHISPER_QC_DEVICE_TOOLTIP, options=WHISPER_QC_DEVICE_OPTIONS),
        ConfigField("whisper_qc_compute_type", "Tryb pracy", "choice", WHISPER_QC_COMPUTE_TOOLTIP, options=WHISPER_QC_COMPUTE_TYPES, option_labels=WHISPER_QC_COMPUTE_TYPE_LABELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "openai": (
        ConfigField("api_key", "Klucz API", "str", "Klucz API OpenAI. Bez internetu i klucza ten silnik nie wygeneruje mowy.", secret=True),
        ConfigField("model", "Model", "str", "Nazwa modelu TTS OpenAI, domyslnie gpt-4o-mini-tts."),
        ConfigField("voice", "Glos", "str", "Nazwa glosu OpenAI, np. marin, cedar, coral, alloy, nova, onyx."),
        ConfigField("instructions", "Instrukcja glosu", "str", "Instrukcja stylu mowy dla modelu TTS, np. spokojny polski lektor."),
        ConfigField("audio_qc_enabled", "Wlacz kontrole audio", "bool", "Wlacza techniczna kontrole pliku audio: cisza, dlugosc segmentu, glosny poczatek lub koniec, clipping i podobne artefakty."),
        ConfigField("audio_qc_retry_attempts", "Liczba prob kontroli audio", "int", "Maksymalna liczba prob generowania segmentu, jesli kontrola audio wykryje podejrzany wynik.", 1, 5, 1),
        TTS_TEXT_CLEANUP_FIELD,
        OPEN_WORKSPACE_FIELD,
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", WHISPER_QC_RETRY_TOOLTIP, 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_device", "Urzadzenie", "choice", WHISPER_QC_DEVICE_TOOLTIP, options=WHISPER_QC_DEVICE_OPTIONS),
        ConfigField("whisper_qc_compute_type", "Tryb pracy", "choice", WHISPER_QC_COMPUTE_TOOLTIP, options=WHISPER_QC_COMPUTE_TYPES, option_labels=WHISPER_QC_COMPUTE_TYPE_LABELS),
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
        ConfigField("audio_prompt_path", "Probka glosu", "path", voice_sample_duration_help("chatterbox")),
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
        TTS_TEXT_CLEANUP_FIELD,
        OPEN_WORKSPACE_FIELD,
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", WHISPER_QC_RETRY_TOOLTIP, 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_device", "Urzadzenie", "choice", WHISPER_QC_DEVICE_TOOLTIP, options=WHISPER_QC_DEVICE_OPTIONS),
        ConfigField("whisper_qc_compute_type", "Tryb pracy", "choice", WHISPER_QC_COMPUTE_TOOLTIP, options=WHISPER_QC_COMPUTE_TYPES, option_labels=WHISPER_QC_COMPUTE_TYPE_LABELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "omnivoice": (
        ConfigField("device", "Urzadzenie", "choice", DEVICE_TOOLTIP, options=DEVICE_OPTIONS),
        ConfigField("reference_audio_path", "Probka glosu", "path", voice_sample_duration_help("omnivoice")),
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
        TTS_TEXT_CLEANUP_FIELD,
        OPEN_WORKSPACE_FIELD,
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", WHISPER_QC_RETRY_TOOLTIP, 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_device", "Urzadzenie", "choice", WHISPER_QC_DEVICE_TOOLTIP, options=WHISPER_QC_DEVICE_OPTIONS),
        ConfigField("whisper_qc_compute_type", "Tryb pracy", "choice", WHISPER_QC_COMPUTE_TOOLTIP, options=WHISPER_QC_COMPUTE_TYPES, option_labels=WHISPER_QC_COMPUTE_TYPE_LABELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "piper": (
        ConfigField("voice", "Glos lektora", "choice", "Polski glos Piper. Model pobiera sie dopiero przy pierwszym uzyciu wybranego glosu.", options=PIPER_POLISH_VOICES, option_labels=PIPER_POLISH_VOICE_LABELS),
        ConfigField(
            "length_scale",
            "Tempo mowy",
            "float",
            "Zakres 0.5-2.0, domyslnie 1.1. W Piper wyzsza wartosc wydluza fonemy, czyli spowalnia mowe; nizsza przyspiesza.",
            0.5,
            2.0,
            0.05,
        ),
        ConfigField(
            "noise_scale",
            "Zmiennosc audio",
            "float",
            "Zakres 0.0-1.5, domyslnie 0.05. Wyzej = wieksza losowosc brzmienia; nizej = spokojniejszy, bardziej powtarzalny wynik.",
            0.0,
            1.5,
            0.01,
        ),
        ConfigField(
            "noise_w_scale",
            "Zmiennosc wymowy",
            "float",
            "Zakres 0.0-1.5, domyslnie 0.05. Wyzej = wieksza zmiennosc szerokosci fonemow i rytmu; nizej = bardziej przewidywalna wymowa.",
            0.0,
            1.5,
            0.01,
        ),
        ConfigField("speaker_id", "Speaker ID", "int", "Ukryte ustawienie techniczne dla modeli wielomowcowych. Polskie glosy Piper sa jednomowcowe, wiec zostaje 0.", 0, 999, 1, visible=False),
        TTS_TEXT_CLEANUP_FIELD,
        OPEN_WORKSPACE_FIELD,
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", WHISPER_QC_RETRY_TOOLTIP, 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_device", "Urzadzenie", "choice", WHISPER_QC_DEVICE_TOOLTIP, options=WHISPER_QC_DEVICE_OPTIONS),
        ConfigField("whisper_qc_compute_type", "Tryb pracy", "choice", WHISPER_QC_COMPUTE_TOOLTIP, options=WHISPER_QC_COMPUTE_TYPES, option_labels=WHISPER_QC_COMPUTE_TYPE_LABELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "supertonic": (
        ConfigField("voice", "Glos lektora", "choice", "Glos wbudowany Supertonic. Model pobiera sie dopiero przy pierwszym uzyciu.", options=SUPERTONIC_VOICES, option_labels=SUPERTONIC_VOICE_LABELS),
        ConfigField(
            "speed",
            "Predkosc",
            "float",
            "Zakres 0.5-2.0, domyslnie 1.05. Wyzsza wartosc przyspiesza mowe, nizsza spowalnia.",
            0.5,
            2.0,
            0.01,
        ),
        ConfigField(
            "total_steps",
            "Kroki jakosci",
            "int",
            "Zakres 2-12, domyslnie 8. Wiecej krokow zwykle daje lepsza jakosc, ale trwa dluzej.",
            2,
            12,
            1,
        ),
        ConfigField(
            "max_chunk_length",
            "Dlugosc fragmentu tekstu",
            "int",
            "Maksymalna dlugosc dluzszej kwestii przed podzialem na mniejsze fragmenty, podana w znakach tekstu.",
            80,
            500,
            10,
        ),
        ConfigField("silence_duration", "Pauza miedzy fragmentami", "float", "Ukryte ustawienie przerw przy dluzszych kwestiach.", 0.0, 1.0, 0.05, visible=False),
        ConfigField(
            "supertonic_trim_edges",
            "Wycinanie ciszy",
            "bool",
            "Po wlaczeniu program usuwa nadmiar ciszy z poczatku i konca segmentu Supertonic, zostawiajac krotki zapas przy mowie.",
        ),
        TTS_TEXT_CLEANUP_FIELD,
        OPEN_WORKSPACE_FIELD,
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", WHISPER_QC_RETRY_TOOLTIP, 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_device", "Urzadzenie", "choice", WHISPER_QC_DEVICE_TOOLTIP, options=WHISPER_QC_DEVICE_OPTIONS),
        ConfigField("whisper_qc_compute_type", "Tryb pracy", "choice", WHISPER_QC_COMPUTE_TOOLTIP, options=WHISPER_QC_COMPUTE_TYPES, option_labels=WHISPER_QC_COMPUTE_TYPE_LABELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
    "coqui_xtts": (
        ConfigField("device", "Urzadzenie", "choice", DEVICE_TOOLTIP, options=DEVICE_OPTIONS),
        ConfigField("speaker_wav_path", "Probka glosu", "path", voice_sample_duration_help("coqui_xtts") + " Jezeli pole jest puste, XTTS uzyje wybranego glosu wbudowanego."),
        ConfigField("speaker", "Glos wbudowany", "choice", "Glos wbudowany Coqui XTTS uzywany tylko wtedy, gdy nie podasz probki glosu.", options=COQUI_BUILTIN_SPEAKERS),
        ConfigField(
            "builtin_voice_speed",
            "Predkosc glosu wbudowanego",
            "float",
            "Zakres 0.5-2.0, domyslnie 1.6. Uzywana tylko wtedy, gdy pole probki glosu jest puste; powyzej 1.0 lektor mowi szybciej, ponizej 1.0 wolniej.",
            0.5,
            2.0,
            0.01,
        ),
        ConfigField(
            "voice_sample_speed",
            "Predkosc glosu z probki",
            "float",
            "Zakres 0.5-2.0, domyslnie 1.3. Uzywana tylko wtedy, gdy podasz probke glosu; powyzej 1.0 lektor mowi szybciej, ponizej 1.0 wolniej.",
            0.5,
            2.0,
            0.01,
        ),
        ConfigField(
            "temperature",
            "Temperatura",
            "float",
            "Zakres 0.05-1.5, domyslnie 0.1. Wyzej = bardziej kreatywny, ale mniej stabilny wynik; nizej = spokojniejszy i bardziej przewidywalny lektor.",
            0.05,
            1.5,
            0.01,
        ),
        ConfigField(
            "length_penalty",
            "Kara dlugosci",
            "float",
            "Zakres 0.0-2.0, domyslnie 1.0. Steruje preferencja dlugosci sekwencji generowanej przez model; wyzsze wartosci moga promowac dluzsze wypowiedzi.",
            0.0,
            2.0,
            0.05,
        ),
        ConfigField(
            "repetition_penalty",
            "Kara powtorzen",
            "float",
            "Zakres 1.0-12.0, domyslnie 9.0. Wyzej = mocniejsze ograniczanie powtorzen, ale zbyt duza wartosc moze pogorszyc naturalnosc.",
            1.0,
            12.0,
            0.1,
        ),
        ConfigField(
            "top_k",
            "Top K",
            "int",
            "Zakres 0-100, domyslnie 100. Ogranicza wybor tokenow do K najbardziej prawdopodobnych; nizsze wartosci zwiekszaja przewidywalnosc.",
            0,
            100,
            1,
        ),
        ConfigField(
            "top_p",
            "Top P",
            "float",
            "Zakres 0.05-1.0, domyslnie 1.0. Ogranicza losowanie do tokenow o lacznym prawdopodobienstwie top_p; nizsze wartosci sa stabilniejsze.",
            0.05,
            1.0,
            0.01,
        ),
        ConfigField("xtts_trim_trailing_silence", "Wycinanie koncowej ciszy", "bool", "Po wlaczeniu program usuwa dluga cisze z konca segmentu XTTS, zostawia okolo 120 ms zapasu i dodaje krotkie wygladzenie, zeby nie uciac lektora."),
        TTS_TEXT_CLEANUP_FIELD,
        OPEN_WORKSPACE_FIELD,
        *DIAGNOSTIC_OUTPUT_FIELDS,
        ConfigField("whisper_qc_enabled", "Wlacz kontrole mowy", "bool", "Program sprawdza czy lektor powiedzial to co znajduje sie w napisach."),
        ConfigField("whisper_qc_retry_attempts", "Liczba prob", "int", WHISPER_QC_RETRY_TOOLTIP, 1, 5, 1),
        ConfigField("whisper_qc_model", "Model", "choice", "Do wyboru rozne modele dla kontroli mowy.", options=WHISPER_QC_MODELS),
        ConfigField("whisper_qc_device", "Urzadzenie", "choice", WHISPER_QC_DEVICE_TOOLTIP, options=WHISPER_QC_DEVICE_OPTIONS),
        ConfigField("whisper_qc_compute_type", "Tryb pracy", "choice", WHISPER_QC_COMPUTE_TOOLTIP, options=WHISPER_QC_COMPUTE_TYPES, option_labels=WHISPER_QC_COMPUTE_TYPE_LABELS),
        ConfigField("whisper_qc_min_similarity", "Zgodnosc tekstu", "float", "Minimalna zgodnosc lektora z tekstem. 0,62 to inaczej minimalna zgodnosc 62%.", 0.0, 1.0, 0.01),
    ),
}


def fields_for(engine_id: str) -> tuple[ConfigField, ...]:
    return CONFIG_SCHEMAS.get(engine_id, ())


def visible_fields_for(engine_id: str) -> tuple[ConfigField, ...]:
    return tuple(field for field in fields_for(engine_id) if field.visible)


def whisper_qc_compute_type_options_for_device(device: str) -> tuple[str, ...]:
    normalized = str(device or "").strip().casefold()
    if not normalized or normalized == "cpu":
        return ("int8",)
    return WHISPER_QC_COMPUTE_TYPES


def whisper_qc_compute_type_labels_for_options(options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(WHISPER_QC_COMPUTE_TYPE_LABEL_BY_VALUE.get(option, option) for option in options)


def whisper_qc_effective_compute_type(device: str, compute_type: str) -> str:
    allowed = whisper_qc_compute_type_options_for_device(device)
    requested = str(compute_type or "").strip()
    return requested if requested in allowed else allowed[0]


def faster_whisper_device_kwargs(device: str) -> dict[str, object]:
    normalized = str(device or "").strip().lower() or "cpu"
    if normalized.startswith("cuda:"):
        index_text = normalized.split(":", 1)[1]
        if index_text.isdigit():
            return {"device": "cuda", "device_index": int(index_text)}
    return {"device": normalized}


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


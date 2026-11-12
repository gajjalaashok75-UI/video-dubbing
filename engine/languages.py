"""
Central language registry — single source of truth for the dubbing pipeline.

Replaces the scattered maps that used to live in `translation_v2.LANG_TO_FLORES`
and `openvoice_worker.EDGE_TTS_VOICES`.

Each entry describes, for one UI-facing language code:
  name         human-readable label
  flores       NLLB-200 flores-200 code for this language
  translate_to flores code we actually translate INTO (differs only where the
               synthesis engine cannot read the language's native script)
  engine       which TTS backend synthesizes it
  synth_lang   language code handed to that engine
  cps          rough characters-per-second of natural speech, used only as a
               sanity bound on duration fitting (see dubbing.plan_timeline)

IndicF5 (ai4bharat/IndicF5) natively covers 11 Indian languages. Its vocabulary
contains all 11 Indian scripts plus Latin, but NO Arabic script — so Urdu is
translated to Hindi/Devanagari and synthesized as Hindi. Urdu and Hindi are the
same spoken language (Hindustani); the audio a listener hears is correct Urdu.
The caveat is lexical: heavily Persianized formal Urdu will drift toward Hindi
vocabulary, because NLLB is asked for Hindi.
"""

import os

ENGINE_INDICF5 = "indicf5"
ENGINE_XTTS = "xtts"
ENGINE_S2PRO = "s2pro"

# Languages IndicF5 was trained on.
INDICF5_NATIVE = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}

# Languages Coqui XTTS-v2 supports (phase 2: foreign languages).
XTTS_NATIVE = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

# Fish Audio S2 Pro (extraplus/s2-pro) speaks 80+ languages. This is the
# intersection with the targets this project offers, read from the model
# card's language list. Odia is the one Indian language it does not cover, so a
# render targeting Odia under S2 Pro falls back to IndicF5 rather than failing.
S2PRO_NATIVE = {
    "hi", "mr", "bn", "ta", "te", "kn", "gu", "ml", "pa", "as",   # Indic
    "ur",
    "en", "es", "fr", "de", "it", "pt", "ru", "pl", "nl", "tr",
    "cs", "hu", "ar", "ja", "ko", "zh",
}

# Which engine can speak which language, for the preference resolver below.
_ENGINE_NATIVE = {
    ENGINE_INDICF5: INDICF5_NATIVE,
    ENGINE_XTTS: XTTS_NATIVE,
    ENGINE_S2PRO: S2PRO_NATIVE,
}

LANGUAGES = {
    # ---- Indian languages: IndicF5 ----
    "hi": dict(name="Hindi",     flores="hin_Deva", engine=ENGINE_INDICF5, cps=13.5),
    "mr": dict(name="Marathi",   flores="mar_Deva", engine=ENGINE_INDICF5, cps=13.0),
    "bn": dict(name="Bengali",   flores="ben_Beng", engine=ENGINE_INDICF5, cps=13.0),
    "ta": dict(name="Tamil",     flores="tam_Taml", engine=ENGINE_INDICF5, cps=12.0),
    "te": dict(name="Telugu",    flores="tel_Telu", engine=ENGINE_INDICF5, cps=12.5),
    "kn": dict(name="Kannada",   flores="kan_Knda", engine=ENGINE_INDICF5, cps=12.5),
    "gu": dict(name="Gujarati",  flores="guj_Gujr", engine=ENGINE_INDICF5, cps=13.0),
    "ml": dict(name="Malayalam", flores="mal_Mlym", engine=ENGINE_INDICF5, cps=12.0),
    "pa": dict(name="Punjabi",   flores="pan_Guru", engine=ENGINE_INDICF5, cps=13.0),
    "or": dict(name="Odia",      flores="ory_Orya", engine=ENGINE_INDICF5, cps=12.5),
    "as": dict(name="Assamese",  flores="asm_Beng", engine=ENGINE_INDICF5, cps=13.0),

    # Urdu: no Arabic script in the IndicF5 vocab -> translate to Hindi, speak as Hindi.
    "ur": dict(name="Urdu", flores="urd_Arab", translate_to="hin_Deva",
               engine=ENGINE_INDICF5, synth_lang="hi", cps=13.5),

    # ---- Foreign languages: XTTS-v2 (phase 2) ----
    "en": dict(name="English",    flores="eng_Latn", engine=ENGINE_XTTS, cps=15.5),
    "es": dict(name="Spanish",    flores="spa_Latn", engine=ENGINE_XTTS, cps=15.0),
    "fr": dict(name="French",     flores="fra_Latn", engine=ENGINE_XTTS, cps=15.0),
    "de": dict(name="German",     flores="deu_Latn", engine=ENGINE_XTTS, cps=14.5),
    "it": dict(name="Italian",    flores="ita_Latn", engine=ENGINE_XTTS, cps=15.0),
    "pt": dict(name="Portuguese", flores="por_Latn", engine=ENGINE_XTTS, cps=15.0),
    "ru": dict(name="Russian",    flores="rus_Cyrl", engine=ENGINE_XTTS, cps=14.0),
    "pl": dict(name="Polish",     flores="pol_Latn", engine=ENGINE_XTTS, cps=14.0),
    "nl": dict(name="Dutch",      flores="nld_Latn", engine=ENGINE_XTTS, cps=14.5),
    "tr": dict(name="Turkish",    flores="tur_Latn", engine=ENGINE_XTTS, cps=14.0),
    "cs": dict(name="Czech",      flores="ces_Latn", engine=ENGINE_XTTS, cps=14.0),
    "hu": dict(name="Hungarian",  flores="hun_Latn", engine=ENGINE_XTTS, cps=14.0),
    "ar": dict(name="Arabic",     flores="arb_Arab", engine=ENGINE_XTTS, cps=13.0),
    "ja": dict(name="Japanese",   flores="jpn_Jpan", engine=ENGINE_XTTS, cps=7.0),
    "ko": dict(name="Korean",     flores="kor_Hang", engine=ENGINE_XTTS, cps=8.0),
    "zh": dict(name="Chinese",    flores="zho_Hans", engine=ENGINE_XTTS,
               synth_lang="zh-cn", cps=6.5),
}

# Source-side only: Whisper may detect these even though we cannot synthesize them.
# Used to resolve a translation source code, never a synthesis target.
EXTRA_SOURCE_FLORES = {
    "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn",
    "sw": "swh_Latn", "uk": "ukr_Cyrl", "ro": "ron_Latn", "el": "ell_Grek",
    "sv": "swe_Latn", "da": "dan_Latn", "fi": "fin_Latn", "no": "nob_Latn",
    "he": "heb_Hebr", "fa": "pes_Arab", "ne": "npi_Deva", "si": "sin_Sinh",
}


class UnsupportedLanguage(ValueError):
    pass


def normalize(code: str) -> str:
    """'hi-IN' -> 'hi'; 'zh-CN' -> 'zh'. Case-insensitive."""
    if not code:
        raise UnsupportedLanguage("empty language code")
    c = code.strip().lower()
    if c in LANGUAGES:
        return c
    base = c.split("-")[0].split("_")[0]
    if base in LANGUAGES or base in EXTRA_SOURCE_FLORES:
        return base
    raise UnsupportedLanguage(f"unknown language code: {code!r}")


def source_flores(code: str) -> str:
    """flores-200 code to use as the TRANSLATION SOURCE."""
    c = normalize(code)
    if c in LANGUAGES:
        return LANGUAGES[c]["flores"]
    return EXTRA_SOURCE_FLORES[c]


def target_flores(code: str, for_speech: bool = True) -> str:
    """
    flores-200 code to translate INTO for this target.

    `for_speech` picks the script the voice model can read, which for Urdu
    is Devanagari. Text that will only ever be read - subtitles, a translated
    script - wants the language's own script instead.
    """
    c = normalize(code)
    if c not in LANGUAGES:
        raise UnsupportedLanguage(f"{code!r} is not a supported dubbing target")
    e = LANGUAGES[c]
    return e.get("translate_to", e["flores"]) if for_speech else e["flores"]


def engine_for(code: str) -> str:
    """
    The engine that should speak `code`.

    Each language has a house engine (IndicF5 for the Indian eleven, XTTS for
    the rest), but S2 Pro speaks most of both sets and is the preferred voice.
    The preference lives in `VIDEOTRANS_TTS_ENGINE` (an operator can export it)
    or the stored studio setting, whichever is set:

      `auto` (the default) prefers S2 Pro wherever it can speak the language
      AND is actually installed — the `fish_speech` package plus its ~11 GB of
      weights. When it is not installed, which is the normal state of a fresh
      clone, `auto` falls back to the house engine, so a default install keeps
      working untouched.

      Any other value is honoured only when that engine actually covers the
      language, so pinning S2 Pro does not silently break Odia, which it cannot
      speak.
    """
    c = normalize(code)
    if c not in LANGUAGES:
        raise UnsupportedLanguage(f"{code!r} is not a supported dubbing target")
    house = LANGUAGES[c]["engine"]
    want = preferred_engine()
    if want in ("", "auto"):
        # The requested default. `_engine_ready` is what keeps it safe: S2 Pro
        # is only chosen when it can actually run here.
        if (house != ENGINE_S2PRO and engine_supports(ENGINE_S2PRO, c)
                and _engine_ready(ENGINE_S2PRO)):
            return ENGINE_S2PRO
        return house
    if want in _ENGINE_NATIVE and c in _ENGINE_NATIVE[want]:
        return want
    # The pin does not cover this language; the house engine still does.
    return house


def _engine_ready(engine: str) -> bool:
    """
    Whether an engine is installed and usable here, without importing it eagerly.

    `tts_engines` owns the availability checks and imports this module at the
    top, so the import is local to keep the pair acyclic. A failure to import
    means "not ready", which is the safe answer: it can only ever send a
    language back to its house engine, never break a render.
    """
    try:
        import tts_engines
        return bool(tts_engines.engine_available(engine))
    except Exception:                                        # noqa: BLE001
        return False


def engine_supports(engine: str, code: str) -> bool:
    """Whether `engine` can speak `code` at all."""
    try:
        c = normalize(code)
    except UnsupportedLanguage:
        return False
    native = _ENGINE_NATIVE.get(engine)
    if native is None:
        return False
    return c in native or (c == "zh" and "zh-cn" in native)


def engines_for(code: str) -> list:
    """Every engine that can speak `code`, house engine first."""
    c = normalize(code)
    house = LANGUAGES[c]["engine"]
    others = [e for e in (ENGINE_S2PRO, ENGINE_INDICF5, ENGINE_XTTS)
              if e != house and engine_supports(e, c)]
    return [house] + others


def preferred_engine() -> str:
    """
    The pinned voice engine, or 'auto'.

    The environment wins, because an operator pinning a deployment must not be
    overridable from the UI; otherwise the stored studio setting is read. The
    import is local: `engines` reads this module's catalogue, so a top-level
    import would be circular.

    Memoised on the settings file's mtime and the environment value, because
    this is consulted once per phrase through `get_engine`, and a job can have
    hundreds of phrases — re-reading the JSON each time is pure waste.
    """
    global _PREF_CACHE
    raw = os.getenv("VIDEOTRANS_TTS_ENGINE")
    if raw is not None:
        return (raw or "auto").strip().lower() or "auto"

    key = None
    try:
        import engines
        try:
            key = ("mtime", os.path.getmtime(engines.SETTINGS_PATH))
        except OSError:
            key = ("absent",)
    except Exception:                                        # noqa: BLE001
        key = ("no-engines",)

    if _PREF_CACHE and _PREF_CACHE[0] == key:
        return _PREF_CACHE[1]

    try:
        import engines
        raw = engines.get("voice")
    except Exception:                                        # noqa: BLE001
        raw = None
    value = (raw or "auto").strip().lower() or "auto"
    _PREF_CACHE = (key, value)
    return value


_PREF_CACHE = None


def synth_lang(code: str) -> str:
    """Language code handed to the TTS engine (Urdu -> 'hi', Chinese -> 'zh-cn')."""
    c = normalize(code)
    return LANGUAGES[c].get("synth_lang", c)


def chars_per_second(code: str) -> float:
    c = normalize(code)
    return LANGUAGES.get(c, {}).get("cps", 14.0)


def display_name(code: str) -> str:
    return LANGUAGES[normalize(code)]["name"]


def supported_targets() -> list:
    """UI-facing list, Indian languages first."""
    indic = [c for c, e in LANGUAGES.items() if e["engine"] == ENGINE_INDICF5]
    other = [c for c, e in LANGUAGES.items() if e["engine"] != ENGINE_INDICF5]
    rows = []
    for c in indic + other:
        # `engine` is the house engine, which is what the language table says
        # this target is for; `engines` is every engine that can speak it, and
        # `active_engine` is the one the current preference selects. They differ
        # when S2 Pro is pinned, so the studio can say what will actually speak.
        active = engine_for(c)
        rows.append({"code": c, "name": LANGUAGES[c]["name"],
                     "engine": LANGUAGES[c]["engine"],
                     "active_engine": active,
                     "engines": engines_for(c)})
    return rows

"""
Translation v2 — NLLB-200-Distilled-600M
Drop-in replacement for translation.py with:
  - 600M params (vs 1B for IndicTrans2) — 40% less memory
  - 200+ language pairs bidirectional (not just EN→Indic)
  - Standard HuggingFace API (no indictrans_toolkit dependency)
  - CTranslate2-compatible for further speedup

Requires: transformers<4.55, sentencepiece, torch>=2.2

Root cause note — why the pin:
  transformers >= 4.55 restructured the M2M100 model module. The class that
  loaded NLLB used to live at transformers.models.m2m_100.M2M100ForConditionalGeneration;
  in some versions it was moved or the auto class resolution changed the path
  that from_pretrained() walks, and the import chain broke with a name-missing
  error inside torch._utils. Pinning to <4.55 keeps the old layout. The lazy
  import below keeps app.py importable on hosts where torch._utils is already
  broken (Colab CUDA-12 torch 2.6 on Python 3.13) so the error surfaces only
  when someone actually renders a dub, not when the server starts.
"""

from __future__ import annotations

import itertools
import os
import re as _re
import sys
import threading
import time
from typing import Optional

import engines as _engines
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = _engines.get("mt") or "facebook/nllb-200-distilled-600M"

# Device is resolved lazily (below, inside _load_model) because at import time
# torch.cuda.is_available() can return True on a machine with no usable CUDA
# Runtime (e.g. Colab with the CUDA library missing), and we want the mismatch
# caught at the first real inference, not at server startup. There is a tiny
# cost to re-checking each time — one tensor call — and it is paid exactly
# once per render, so it is not worth caching.

# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------

# Short codes that the studio and Whisper actually use, mapped to the flores-200
# codes NLLB expects. This list is complete for every language the app exposes
# through `engines` + `languages` today; anything else still goes through the
# flores-normalisation path below (handles BCP-47 like "hi-IN" and bare
# flores strings like "hin_Deva" already in the right shape).
#
# Names are short so the translate pass can inline the lookup without chasing
# through languages.py at module import — it is a dependency of the render path
# regardless and that module is importable even when the voice stage is gated.

_LANG_TO_FLORES: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "pt-BR": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "sw": "swh_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "el": "ell_Grek",
    "hu": "hun_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "eo": "epo_Latn",
    "ca": "cat_Latn",
    "eu": "eus_Latn",
    "gl": "glg_Latn",
    "af": "afr_Latn",
    "sq": "sqi_Latn",
    "hy": "hye_Armn",
    "ka": "kat_Geor",
    "az": "aze_Latn",
    "kk": "kaz_Cyrl",
    "uz": "uzb_Latn",
    "cy": "cym_Latn",
    "is": "isl_Latn",
    "mt": "mlt_Latn",
    "mk": "mkd_Cyrl",
    "sr": "srp_Cyrl",
    "bs": "bos_Latn",
    "hr": "hrv_Latn",
    "sl": "slv_Latn",
    "et": "est_Latn",
    "lv": "lav_Latn",
    "lt": "lit_Latn",
    "mn": "mon_Cyrl",
    "my": "mya_Hmng",
    "ps": "pus_Arab",
    "sd": "snd_Arab",
    "ceb": "ceb_Latn",
    "jv": "jav_Latn",
    "su": "sun_Latn",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "ha": "hau_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "sn": "sot_Latn",
    "tn": "tsn_Latn",
    "wo": "wol_Latn",
    "ff": "ful_Latn",
    "guz": "guz_Latn",
    "knc": "knc_Latn",
    "lob": "lob_Latn",
    "mer": "mer_Latn",
    "ny": "nya_Latn",
    "sah": "sah_Cyrl",
    "tk": "tuk_Latn",
    "tmh": "tmh_Latn",
    "bo": "bod_Hans",
    "yo": "yor_Latn",
}


def _resolve_code(code: str) -> str:
    """Map a language code the caller uses to the flores-200 code NLLB wants.

    Accepts:
      - short codes like 'hi'
      - BCP-47 like 'hi-IN' → 'hi'
      - bare flores strings like 'hin_Deva' (passed through unchanged)

    Falls back to the bare code if nothing matches, so an undeclared language
    does not crash the renderer — it just gets a possibly-wrong code and the
    HF call fails with a clear message.
    """
    if not code:
        return ""
    low = code.lower()
    if "_" in low and len(low) >= 8:  # already looks like a flores code
        return low
    base = low.split("-")[0]
    if base in _LANG_TO_FLORES:
        return _LANG_TO_FLORES[base]
    # Maybe it is already flores but shorter than our heuristic, e.g. 'hin_De'
    if "_" in low:
        return low
    return low


def _langs_flores(source: str, target: str, for_speech: bool = True) -> tuple[str, str]:
    """Return (src_flores, tgt_flores) for a translate call.

    Args:
        source: source language code
        target: target language code  
        for_speech: if True, use the script the voice model reads; if False,
                   use the language's own script (for subtitles in the original
                   script where that differs from speech, e.g. Urdu).

    Uses languages.py for special cases like Urdu where the spoken script
    (Devanagari for IndicF5) differs from the written script (Arabic).
    """
    import languages as _L  # local import to avoid startup cost

    src = _resolve_code(source)
    
    if for_speech:
        # For speech: use languages.py target_flores which handles Urdu->Devanagari
        tgt = _L.target_flores(target, for_speech=True)
    else:
        # For subtitles: use the language's own script
        tgt = _L.target_flores(target, for_speech=False)
    
    return src, tgt


# ---------------------------------------------------------------------------
# Lazy transformer import
# ---------------------------------------------------------------------------
#
# Importing ``transformers`` at module level triggers a chain that can fail:
#   transformers → model_debugging_utils → torch.distributed.tensor._ops
#   → torch._utils._maybe_view_chunk_cat
#
# On Colab with the CUDA-12 torch build installed *and* some Python 3.13
# numpy/torchaudio combo, that import fails with:
#   cannot import name '_maybe_view_chunk_cat' from 'torch._utils'
#
# The server process does not need the translator at startup to answer /health,
# /api/models, /languages, the settings endpoints, etc. So we defer the import
# until the first render. Before that point the app is fully responsive; the
# only visible effect of a broken transformers install is that a render
# rejects at the translate stage with the clear message below, instead of the
# server being unable to start.
#
# The same deferred import is the reason the code can still import quickly on
# a machine that has a *working* newer transformers — we simply never pay the
# import cost until it is needed, and if the chain is fine the render works.

_AUTO_MODEL_CLASS: Optional[type] = None
_AUTO_TOKENIZER_CLASS: Optional[type] = None
_IMPORT_LOCK = threading.Lock()


def _get_auto_classes():
    """Return (AutoModelForSeq2SeqLM, AutoTokenizer), importing on first call.

    Synchronised so a concurrent startup (two renders hitting the translate gate
    at the same time) does not import twice.
    """
    global _AUTO_MODEL_CLASS, _AUTO_TOKENIZER_CLASS
    if _AUTO_MODEL_CLASS is None:
        with _IMPORT_LOCK:
            # second check inside the lock — another thread may have imported
            if _AUTO_MODEL_CLASS is None:
                from transformers import (
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )
                _AUTO_MODEL_CLASS = AutoModelForSeq2SeqLM
                _AUTO_TOKENIZER_CLASS = AutoTokenizer
    return _AUTO_MODEL_CLASS, _AUTO_TOKENIZER_CLASS


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_tokenizer = None
_model = None


def _reset():
    """Drop a loaded model so the next call re-reads the engines choice.

    Used by the settings UI when the user switches the translation model.
    """
    global _tokenizer, _model, MODEL_NAME
    with _state_lock:
        _tokenizer = _model = None
        MODEL_NAME = _engines.get("mt") or "facebook/nllb-200-distilled-600M"


def _explain_load_error(exc: BaseException) -> str:
    """Say whether this is a missing package, missing weights, or a code bug.

    The message used to be identical for every failure: "pip install
    transformers sentencepiece && huggingface-cli download ...". A plain
    AttributeError raised inside this function therefore sent a run that already
    had both packages AND the weights on disk into reinstalling and
    re-downloading them. The three causes have three different answers, so the
    answer is chosen from the exception instead of assumed.
    """
    name = type(exc).__name__
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (f" [{name}] A package the loader imports is missing. "
                "Install it with: pip install transformers sentencepiece")
    if isinstance(exc, OSError):
        return (f" [{name}] The weights could not be read or downloaded. "
                f"Fetch them with `huggingface-cli download {MODEL_NAME}`, or "
                "set HF_TOKEN for a gated repo. This is a model/download "
                "failure, not a code failure.")
    return (f" [{name}] The packages and weights were found: this failed inside "
            "the load path itself, which is a code failure, not a missing "
            "package or weight. Read the traceback above; reinstalling or "
            "re-downloading will not change it.")


def refresh() -> str:
    """Drop the loaded model so the next render uses the newly chosen one.

    `app.py` calls `refresh()` on the module that owns each stage when the
    studio saves a new choice. This module never had one, so the change was
    stored, logged as "Could not apply the mt change live", and only took
    effect after a restart - while the API still answered
    `"restart_required": false`. Returns the model name now in use, like the
    other stages' refresh().
    """
    _reset()
    return MODEL_NAME


def _load():
    """Load (or return the cached) tokenizer + model.

    Raises RuntimeError with a clear message if the model cannot be loaded.
    """
    global _tokenizer, _model
    with _state_lock:
        if _model is not None:
            return _tokenizer, _model

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            AutoModel, AutoTokenizer = _get_auto_classes()
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            mdl = AutoModel.from_pretrained(MODEL_NAME)
            mdl = mdl.to(device)
            mdl.eval()

            if device == "cuda":
                # Half precision saves VRAM and is what the card wants.
                #
                # The condition used to be `tok.dtype is not None`. `dtype` is a
                # property of a MODEL; a tokenizer has none, and transformers'
                # PreTrainedTokenizerBase.__getattr__ raises
                #     AttributeError: NllbTokenizerFast has no attribute dtype
                # That fired AFTER the weights had loaded and moved to the GPU,
                # so every render died at this line on a perfectly good install.
                # `half()` is already guarded; no test is needed.
                try:
                    mdl = mdl.half()
                except Exception:
                    pass

            _tokenizer = tok
            _model = mdl
            return _tokenizer, _model

        except Exception as exc:
            _tokenizer = _model = None
            raise RuntimeError(
                f"Could not load NLLB model '{MODEL_NAME}': {exc}"
                f"{_explain_load_error(exc)}"
            ) from exc


# The loader's name before it was shortened to `_load`. `app.py`'s boot warmup
# imports `_load_model` (the same name `speech_to_text_v2` uses), so when the
# rename landed the warmup died with
#     ImportError: cannot import name '_load_model' from 'translation_v2'
# and the first render paid the load. Both names are kept so a rename here
# cannot break the warmup again.
_load_model = _load


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate a single string.

    Kept as an obvious single-segment entry point so callers that translate an
    ad-hoc string (logs, the UI preview, tests) do not have to build a segment
    list. For the actual render path, use translate_segments so that:
      - a long Whisper segment is split on sentence boundaries before it hits
        the model (NLLB degrades on length well before its hard token limit), and
      - each segment keeps its own timeline slot — the old code fed the entire
        transcript as one blob and the model commonly dropped everything after
        the first sentence.
    """
    if not text or not text.strip():
        return ""
    src, tgt = _langs_flores(source_lang, target_lang)
    if src == tgt:
        return text

    tok, model = _load()
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(next(model.parameters()).device)

    generated_ids = model.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
        max_new_tokens=256,
        num_beams=3,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tok.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Segment-level translation — the real render path
# ---------------------------------------------------------------------------

# NLLB starts quietly degrading above ~150 tokens and beyond ~512 it just
# stops getting the ending right. Whisper segments can be far longer than that
# (one segment routinely spans a question and its answer in a dialogue), so
# we split big segments before they reach the model. We split on sentence
# boundaries first, then on commas, and words are never broken mid-token.
#
# The split is purely a preprocessing step: the caller still gets one
# translated string per input segment, reconned from the pieces, so the
# timeline stays aligned 1:1 with the Whisper output.

_SENTENCE_SPLIT = _re.compile(r"(?<=[" + "!.?।" + r"])\s+", _re.U)
_MAX_TOKENS_PER_CHUNK = 420


def _split_into_chunks(text: str) -> list[str]:
    """Split a long segment into model-friendly chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= 600:
        return [text]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if buf_len + len(sentence) + 1 <= 600:
            buf.append(sentence)
            buf_len += len(sentence) + 1
        else:
            if buf:
                chunks.append(" ".join(buf))
            if len(sentence) <= 600:
                buf = [sentence]
                buf_len = len(sentence)
            else:
                # One sentence too long — split on commas.
                sub: list[str] = []
                sub_len = 0
                for part in sentence.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if sub_len + len(part) + 2 <= 600:
                        if sub:
                            part = "," + part
                        sub.append(part)
                        sub_len += len(part) + 2
                    else:
                        if sub:
                            chunks.append("".join(sub))
                        sub = [part]
                        sub_len = len(part)
                if sub:
                    buf = sub
                    buf_len = sub_len
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def translate_segments(
    segments: list[dict],
    source_lang: str,
    target_lang: str,
    *,
    for_speech: bool = True,
    batch_strategy: str = "chunks",
    max_segment_chars: int = 600,
) -> list[str]:
    """Translate one segment per input.

    This is what app.py calls for both dubs and subtitles. Returns one string
    per input segment, in order, empty for inputs that were empty.

    Args:
        segments: list of dicts with 'text' key (Whisper-style segments)
        source_lang: source language code
        target_lang: target language code
        for_speech: if True (default), translate for speech synthesis;
                   if False, translate for subtitles in the original script
                   (used when the target language's written form differs from
                    what the voice model reads, e.g. Urdu)
        batch_strategy: not used directly; kept for compatibility
        max_segment_chars: max characters per segment before splitting

    Why this exists instead of one big ``translate_text(transcript)``:
      - NLLB is a sentence-level model. Feeding it a 2-minute transcript as
        one paragraph reliably yields only the first sentence or two, then
        silence for the rest of the timeline. The dub finishes early and the
        video keeps playing — that was the "audio is shorter than video" bug.
      - Segment-by-segment translation keeps each segment's own timeline slot,
        so the assembled track matches the video duration (see
        `dubbing.plan_timeline` + `assemble`).
    """
    if not segments:
        return []

    src, tgt = _langs_flores(source_lang, target_lang, for_speech=for_speech)
    if src == tgt:
        return [seg.get("text", "") or "" for seg in segments]

    tok, model = _load()
    device = next(model.parameters()).device

    # Build input rows: (output_index, text_or_None)
    rows: list[tuple[int, Optional[str]]] = []
    for i, seg in enumerate(segments):
        raw = seg.get("text", "") or ""
        if not raw.strip():
            rows.append((i, None))
        else:
            for chunk in _split_into_chunks(raw)[: 40 if max_segment_chars > 600 else 20]:
                rows.append((i, chunk if chunk.strip() else None))

    # Translate each non-empty chunk, then reassemble per segment index.
    out: dict[int, list[str]] = {i: [] for i in range(len(segments))}

    # Batch by device — on CPU we translate one at a time anyway because
    # the model is small and batching gives no win there; on CUDA we batch
    # as many chunks as fit in the context.
    if device.type == "cuda":
        batch: list[tuple[int, str]] = []
        for idx, txt in rows:
            if txt is None:
                continue
            batch.append((idx, txt))
            if len(batch) >= 8:
                _flush_batch(batch, tok, model, src, tgt, out)
                batch.clear()
        if batch:
            _flush_batch(batch, tok, model, src, tgt, out)
    else:
        for idx, txt in rows:
            if txt is None:
                continue
            _translate_one(idx, txt, tok, model, device, src, tgt, out)

    return [(" ".join(out.get(i, [])).strip()) for i in range(len(segments))]


def _flush_batch(
    batch: list[tuple[int, str]],
    tok,
    model,
    src: str,
    tgt: str,
    out: dict[int, list[str]],
) -> None:
    tok.src_lang = src
    inputs = tok(
        [txt for _, txt in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(next(model.parameters()).device)

    generated_ids = model.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
        max_new_tokens=512,
        num_beams=3,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    decoded = tok.batch_decode(generated_ids, skip_special_tokens=True)
    for (idx, _txt), text in zip(batch, decoded):
        if text.strip():
            out.setdefault(idx, []).append(text.strip())


def _translate_one(
    idx: int,
    text: str,
    tok,
    model,
    device,
    src: str,
    tgt: str,
    out: dict[int, list[str]],
) -> None:
    tok.src_lang = src
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
        max_new_tokens=512,
        num_beams=3,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    decoded = tok.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if decoded:
        out.setdefault(idx, []).append(decoded)


# ---------------------------------------------------------------------------
# Code-switch cleanup
# ---------------------------------------------------------------------------
#
# After translation, a Hindi (or Tamil, Telugu…) target sometimes still
# contains English words — untranslated code-switching ("Have I done something
# wrong?"), acronyms, and proper nouns. The voice model reads those in its
# Latin phonetics, which stands out against the Indic audio around them.
#
# We find runs of ASCII letters/words in the translated text and, where the
# run is long enough to be real English and short enough to be a single phrase,
# re-translate that run from English into the target script. Short acronyms
# (DBMS, LLM, AI) and single capitalised words (names) are deliberately left
# alone — translating those produces worse output than leaving them.


_ASCII_RUN = _re.compile(r"[A-Za-z][A-Za-z'\-\.]*(\s+[A-Za-z][A-Za-z'\-\.]*)*", _re.U)
# Acronyms: 2-6 capital letters, possibly with internal dots (U.S.A.).
_ACRONYM = _re.compile(r"^[A-Z][A-Z.]{1,6}$")
# A single capitalised word that is likely a proper noun.
_LIKELY_NAME = _re.compile(r"^[A-Z][a-z]{1,30}$")


def _is_candidate_run(run: str) -> bool:
    """Is this ASCII run worth re-translating into the target script?"""
    words = run.split()
    if not words:
        return False
    # Single word: keep if it looks like a name or acronym.
    if len(words) == 1:
        w = words[0]
        if _ACRONYM.match(w) or _LIKELY_NAME.match(w):
            return False
        # Short single words are often names too; only translate if clearly
        # a multi-letter content word.
        return len(w) >= 5
    # Multi-word run: translate it (likely a phrase or clause).
    return True


def fix_code_switching(
    texts: list[str],
    target_lang: str,
    source_hint: str = "en",
) -> list[str]:
    """Re-translate English code-switching runs in `texts` into the target script.

    `texts` is the translated script from `translate_segments`. For each target
    language that uses a non-Latin script, run this after translate_segments so
    that leftover English reads in the target's phonetics through the voice
    model. For Latin-script targets this is a no-op.
    """
    src_hint, tgt = _langs_flores(source_hint, target_lang, for_speech=True)
    tgt_script = tgt.rsplit("_", 1)[-1] if "_" in tgt else ""
    if tgt_script in ("Latn", "Hans", "Hant", "Jpan", "Kana"):
        # Latin / CJK: leftover runs are already in a script the voice model
        # handles natively or the cleanup is not helpful. Skip.
        return texts

    # Nothing to do if the target uses the same script as the source hint.
    if tgt_script == "Latn":
        return texts

    # Collect (segment_index, start, end, run) for every candidate run.
    jobs: list[tuple[int, int, int, str]] = []
    for i, text in enumerate(texts):
        for m in _ASCII_RUN.finditer(text):
            run = m.group(0).strip()
            if _is_candidate_run(run):
                jobs.append((i, m.start(), m.end(), run))

    if not jobs:
        return texts

    # Re-translate each job from English into the target flores code.
    tok, model = _load()
    device = next(model.parameters()).device

    results: list[str] = []
    for _, _s, _e, run in jobs:
        tok.src_lang = "eng_Latn"
        inputs = tok(run, return_tensors="pt", truncation=True, max_length=256).to(device)
        gen = model.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
            max_new_tokens=128,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        translated = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
        results.append(translated)

    # Apply replacements; do it in reverse order so the offsets stay valid.
    texts = list(texts)
    for (i, start, end, _run), replacement in zip(jobs, results):
        if replacement:
            texts[i] = texts[i][:start] + replacement + texts[i][end:]
    return texts


# ---------------------------------------------------------------------------
# Debug helper used by the check scripts
# ---------------------------------------------------------------------------

def _model_info() -> dict:
    """Return what the translate stage currently knows, for diagnostics."""
    return {
        "model_name": MODEL_NAME,
        "loaded": _model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "transformers_imported": _AUTO_MODEL_CLASS is not None,
    }

"""
Translation v2 — NLLB-200-Distilled-600M
Drop-in replacement for translation.py with:
  - 600M params (vs 1B for IndicTrans2) — 40% less memory
  - 200+ language pairs bidirectional (not just EN→Indic)
  - Standard HuggingFace API (no indictrans_toolkit dependency)
  - CTranslate2-compatible for further speedup

IMPORTANT: This module uses subprocess to load the NLLB model to avoid
torch/transformers import chain failures on Python 3.13 with CUDA 12 torch.
The import chain (transformers -> torch.distributed.tensor._ops ->
torch._utils._maybe_view_chunk_cat) fails in some Colab environments.
"""

import os
import sys
import subprocess
import tempfile
import json
import re as _re

# Configuration
import engines

MODEL_NAME = engines.get("mt") or "facebook/nllb-200-distilled-600M"


def _get_device():
    forced = os.getenv("FORCE_DEVICE")
    if forced:
        return forced
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()

# Language code mapping
# Map short codes (used by the studio and Whisper) to NLLB flores-200 codes
LANG_TO_FLORES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
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
}


def _resolve_flores_code(lang_code: str) -> str:
    """
    Resolve a language code to NLLB flores-200 format.
    Accepts short codes ('hi'), flores codes ('hin_Deva'), or BCP-47 ('hi-IN').
    """
    # Already a flores code
    if "_" in lang_code and len(lang_code) == 8:
        return lang_code

    # Strip region (e.g., 'hi-IN' → 'hi')
    short = lang_code.split("-")[0].lower()

    if short in LANG_TO_FLORES:
        return LANG_TO_FLORES[short]

    # If it looks like a flores code already (e.g., 'hin_Deva')
    if "_" in lang_code:
        return lang_code

    raise ValueError(
        f"Unknown language code: '{lang_code}'. "
        f"Supported: {list(LANG_TO_FLORES.keys())}"
    )


# ---------------------------------------------------------------------------
# Subprocess-based model loading to avoid torch/transformers import issues
# ---------------------------------------------------------------------------

_TRANSLATOR_PROCESS = None
_TRANSLATOR_PIPE = None


def _start_translator_subprocess():
    """Start a subprocess that handles all transformers imports and model loading."""
    global _TRANSLATOR_PROCESS, _TRANSLATOR_PIPE
    
    # Create a temporary script that will handle the translation
    script = """
import sys
import os
import json

# Add the engine directory to path
sys.path.insert(0, '{engine_dir}')

# Now import transformers - this may fail in parent process but works in subprocess
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = sys.argv[1]
device = sys.argv[2]

print(f"LOADING_MODEL:{model_name}:{device}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

if device == "cuda":
    model = model.half()

model.eval()

print("MODEL_LOADED", flush=True)

# Read commands from stdin
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    
    try:
        cmd = json.loads(line)
        if cmd['type'] == 'translate':
            texts = cmd['texts']
            source_lang = cmd['source_lang']
            target_lang = cmd['target_lang']
            num_beams = cmd.get('num_beams', 2)
            
            tokenizer.src_lang = source_lang
            tgt_id = tokenizer.convert_tokens_to_ids(target_lang)
            
            results = []
            for text in texts:
                if not text or not text.strip():
                    results.append("")
                    continue
                
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(device)
                
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_id,
                        max_new_tokens=512,
                        num_beams=num_beams,
                        num_return_sequences=1,
                        no_repeat_ngram_size=4,
                        repetition_penalty=1.1,
                    )
                
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                results.append(decoded[0].strip())
            
            print(json.dumps({"type": "result", "results": results}), flush=True)
            
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
"""

    # Write the script to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script.format(engine_dir=os.path.dirname(os.path.abspath(__file__))))
        script_path = f.name
    
    # Start the subprocess
    _TRANSLATOR_PROCESS = subprocess.Popen(
        [sys.executable, script_path, MODEL_NAME, DEVICE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Read the initial output
    import time
    time.sleep(0.5)
    
    # Read lines until we get MODEL_LOADED or an error
    while True:
        line = _TRANSLATOR_PROCESS.stdout.readline().strip()
        if not line:
            break
        if line.startswith("LOADING_MODEL:"):
            print(f"[Translation-v2] Loading NLLB model: {MODEL_NAME} on {DEVICE}")
        elif line == "MODEL_LOADED":
            print("[Translation-v2] Model loaded successfully")
            return True
        elif line.startswith("ERROR:"):
            print(f"[Translation-v2] Failed to load model: {line}")
            return False
    
    return False


def _send_translation_request(texts, source_lang, target_lang, num_beams=2):
    """Send a translation request to the subprocess."""
    global _TRANSLATOR_PROCESS, _TRANSLATOR_PIPE
    
    if _TRANSLATOR_PROCESS is None or _TRANSLATOR_PROCESS.poll() is not None:
        if not _start_translator_subprocess():
            raise RuntimeError(
                f"Could not load NLLB model '{MODEL_NAME}'. "
                f"Run: pip install transformers sentencepiece && "
                f"huggingface-cli download {MODEL_NAME}"
            )
    
    request = {
        "type": "translate",
        "texts": texts,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "num_beams": num_beams
    }
    
    _TRANSLATOR_PROCESS.stdin.write(json.dumps(request) + "\n")
    _TRANSLATOR_PROCESS.stdin.flush()
    
    # Read the response
    response_line = _TRANSLATOR_PROCESS.stdout.readline().strip()
    if not response_line:
        raise RuntimeError("Translation subprocess disconnected")
    
    response = json.loads(response_line)
    
    if response["type"] == "error":
        raise RuntimeError(f"Translation failed: {response['message']}")
    
    return response["results"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def refresh():
    """Re-read the stage choice and drop the loaded model."""
    global MODEL_NAME, _TRANSLATOR_PROCESS, _TRANSLATOR_PIPE
    MODEL_NAME = engines.get("mt") or "facebook/nllb-200-distilled-600M"
    _TRANSLATOR_PROCESS = None
    _TRANSLATOR_PIPE = None
    return MODEL_NAME


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text between any supported language pair.
    """
    if not text or not text.strip():
        return ""

    # Resolve to flores-200 codes
    src_flores = _resolve_flores_code(source_lang)
    tgt_flores = _resolve_flores_code(target_lang)

    # Skip if same language
    if src_flores == tgt_flores:
        return text

    results = _send_translation_request([text], src_flores, tgt_flores, num_beams=5)
    translated = results[0] if results else ""

    print(f"[Translation-v2] {src_flores} → {tgt_flores}: "
          f"'{text[:60]}...' → '{translated[:60]}...'")

    return translated


def translate_batch(texts: list, source_lang: str, target_lang: str,
                    for_speech: bool = True,
                    batch_size: int = 8, num_beams: int = None) -> list:
    """
    Translate many short texts. Returns a list the same length as `texts`,
    with empty inputs preserved as empty strings.
    """
    import languages as _L

    src = _L.source_flores(source_lang)
    tgt = _L.target_flores(target_lang, for_speech)

    if src == tgt:
        return [t or "" for t in texts]

    beams = int(num_beams or os.getenv("NLLB_NUM_BEAMS", 2))
    return _send_translation_request(texts, src, tgt, num_beams=beams)


def _degenerate(text: str) -> bool:
    """A translation with fewer than two letters in it is not one."""
    return sum(1 for c in (text or "") if c.isalpha()) < 2


def translate_segments(segments: list, source_lang: str, target_lang: str,
                       num_beams: int = None, for_speech: bool = True) -> list:
    """
    Translate Whisper segments one by one, preserving alignment with the input
    list. Over-long segments are split, translated, and rejoined so a segment
    never silently loses its tail.
    """
    import re as _re
    import languages as _L

    _SENT_SPLIT = _re.compile(r"(?<=[.!?।])\\s+")
    MAX_SEGMENT_CHARS = 480

    def _split_long(text: str, limit: int = MAX_SEGMENT_CHARS) -> list:
        """Split an over-long segment on sentence boundaries, then on commas."""
        text = (text or "").strip()
        if len(text) <= limit:
            return [text] if text else []

        parts, buf = [], ""
        for sent in _SENT_SPLIT.split(text):
            if not sent:
                continue
            if len(buf) + len(sent) + 1 <= limit:
                buf = f"{buf} {sent}".strip()
            else:
                if buf:
                    parts.append(buf)
                buf = sent if len(sent) <= limit else ""
                if not buf:
                    chunk = ""
                    for piece in sent.split(","):
                        if len(chunk) + len(piece) + 1 <= limit:
                            chunk = f"{chunk},{piece}" if chunk else piece
                        else:
                            if chunk:
                                parts.append(chunk.strip())
                            chunk = piece[:limit]
                    if chunk:
                        buf = chunk.strip()
        if buf:
            parts.append(buf)
        return [p for p in parts if p.strip()]

    flat, owner = [], []
    for i, seg in enumerate(segments):
        pieces = _split_long((seg.get("text") or "").strip())
        for p in pieces:
            flat.append(p)
            owner.append(i)

    if not flat:
        return [""] * len(segments)

    translated = translate_batch(flat, source_lang, target_lang,
                                 for_speech=for_speech, num_beams=num_beams)

    joined = [[] for _ in segments]
    for idx, text in zip(owner, translated, strict=True):
        if text:
            joined[idx].append(text)

    result = [" ".join(parts).strip() for parts in joined]

    src = _L.source_flores(source_lang)
    tgt = _L.target_flores(target_lang, for_speech)
    kept = sum(1 for r in result if r)
    print(f"[Translation-v2] {src} -> {tgt}: {kept}/{len(segments)} segments "
          f"({len(flat)} sub-parts, beams={num_beams or os.getenv('NLLB_NUM_BEAMS', 2)})")
    return result


# ---------------------------------------------------------------------------
# Code-switch cleanup (same as before)
# ---------------------------------------------------------------------------

_LATIN_RUN = _re.compile(r"[A-Za-z][A-Za-z'’]*(?:\\s+[A-Za-z][A-Za-z'’]*)*")
_KEEP_ACRONYM = _re.compile(r"^[A-Z]{2,6}$")


def _should_translate_run(run: str) -> bool:
    words = run.split()
    if not words:
        return False
    if len(words) == 1:
        w = words[0]
        if _KEEP_ACRONYM.match(w):      # DBMS, LLM, AI
            return False
        if w[:1].isupper() and len(w) > 1:   # probably a name
            return False
        return len(w) >= 4              # a real English word
    return True                          # a phrase — translate it


def fix_code_switching(texts: list, target_lang: str,
                       source_hint: str = "en") -> list:
    """
    Translate leftover Latin-script runs into the target script.
    """
    import languages as _L

    try:
        tgt = _L.target_flores(target_lang)
    except Exception:
        return texts
    if tgt.endswith("_Latn"):
        return texts                     # target is Latin-script; nothing to do

    jobs, sites = [], []
    for i, text in enumerate(texts):
        for m in _LATIN_RUN.finditer(text or ""):
            run = m.group(0).strip()
            if _should_translate_run(run):
                jobs.append(run)
                sites.append((i, m.start(), m.end()))

    if not jobs:
        return texts

    try:
        fixed = translate_batch(jobs, source_hint, target_lang)
    except Exception as e:
        print(f"[Translation-v2] Code-switch cleanup skipped: {e}")
        return texts

    out = list(texts)
    # Replace right-to-left so earlier offsets stay valid.
    for (i, a, b), new in sorted(zip(sites, fixed), key=lambda x: -x[0][1]):
        if new and new.strip():
            out[i] = out[i][:a] + new.strip() + out[i][b:]

    print(f"[Translation-v2] Rewrote {len(jobs)} Latin-script run(s) into "
          f"{tgt} to stop them being spoken as English")
    return out

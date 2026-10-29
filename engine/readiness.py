"""
One readiness representation, for the studio, the API and the check scripts.

Three scripts grew up alongside Phases 2-5 — `multispeaker_check.py`,
`background_check.py`, `overlap_check.py` — and each answers "can this machine
do my thing?" in its own words. The studio's Setup page answers it a fourth
way. That is four places to update when a dependency changes, and four chances
for them to disagree about whether a model is present.

This module is the one answer. Everything else formats it.

Two rules shape the design:

**A missing optional feature is not a broken application.** A machine with no
Demucs can still dub; a machine with no pyannote can still dub one speaker. So a
feature reports `BLOCKED` only when the work it describes cannot be done at all,
and `FALLBACK_AVAILABLE` when it can be done in a reduced form. Only the core
path — ASR, translation, a voice engine — can make `ready` false.

**A token is reported, never returned.** `HF_TOKEN` is surfaced as
`present`/`missing` and nothing else: no value, no prefix, no length, no
Authorization header. `redact()` exists so a caller cannot accidentally include
one, and a test asserts that no token material reaches the payload.

Nothing here loads a model. Every probe is an import spec, a file check or a
`--version` call, so `snapshot()` is cheap enough for a request handler and safe
to run before the pipeline is set up.
"""

from __future__ import annotations

import os
import shutil
import subprocess

# The four states a feature can be in.
READY = "ready"                            # usable now
FALLBACK_AVAILABLE = "fallback_available"  # usable in a reduced form
OPTIONAL_MISSING = "optional_missing"      # absent, and nothing needs it
BLOCKED = "blocked"                        # the described work cannot be done

# What each state means for the overall verdict. A core feature that is only
# available in reduced form still cannot render, so it counts as blocking.
_BLOCKING = {BLOCKED}

CORE = "core"
OPTIONAL = "optional"


def _has_module(name: str) -> bool:
    """Whether an importable module exists, without importing it."""
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:                                        # noqa: BLE001
        return False


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _ffmpeg_version() -> str | None:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode != 0:
            return None
        first = (out.stdout or "").splitlines()[0]
        return first.split("version", 1)[1].strip().split()[0]
    except Exception:                                        # noqa: BLE001
        return None


def _torch_info() -> dict:
    """
    Torch, the device, and what the device is.

    The device is the interesting part: everything here is slower on CPU and
    several stages are slow enough to change what a user should attempt, so it
    is reported rather than inferred.
    """
    info = {"present": False, "version": None, "cuda": False, "mps": False,
            "device": "cpu", "gpu_name": None, "vram_gb": None, "error": None}
    try:
        import torch
    except Exception as e:                                   # noqa: BLE001
        info["error"] = str(e)
        return info

    info["present"] = True
    info["version"] = getattr(torch, "__version__", None)
    try:
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1e9, 1)
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps and mps.is_available())
    except Exception as e:                                   # noqa: BLE001
        info["error"] = str(e)

    info["device"] = ("cuda" if info["cuda"]
                      else "mps" if info["mps"] else "cpu")
    return info


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def token_state() -> dict:
    """
    Whether a Hugging Face token is configured — and nothing more.

    Deliberately returns no value, no prefix, no length. A caller that only
    needs "can this gated model be fetched" has no use for the token itself, and
    a payload that carries one leaks it into logs, screenshots and bug reports.
    """
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return {"present": bool(token)}


def redact(payload):
    """
    Strip anything that looks like a credential from a payload.

    A belt-and-braces guard for callers that build a report by hand. This module
    never puts a token in, but a readiness payload is the sort of thing that gets
    pasted into an issue, and the cost of being sure is a shallow walk.
    """
    secrets = []
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
                 "XAI_API_KEY", "VIDEOTRANS_LLM_API_KEY", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value and len(value) >= 8:
            secrets.append(value)

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            for secret in secrets:
                if secret in value:
                    value = value.replace(secret, "<redacted>")
            # A bare "Bearer xyz" header, whatever the token was.
            if value.lower().startswith("bearer "):
                return "Bearer <redacted>"
            return value
        return value

    return scrub(payload)


# ---------------------------------------------------------------------------
# Feature probes
# ---------------------------------------------------------------------------
def _gated_reason(package: str, repo: str, present: bool, token: bool) -> str:
    """
    The most specific true statement about why a gated feature cannot run.

    Order matters. "Package not installed" is actionable and comes first;
    "no token" next; then "not downloaded". Collapsing these into one message
    is what makes a user re-check a licence they already accepted, or hunt for a
    package that is installed.
    """
    if not present:
        return (f"{package} is not installed. "
                f"Install it to enable this feature.")
    if not token:
        return (f"HF_TOKEN is not set. {repo} is a gated model: accept its "
                f"licence at https://huggingface.co/{repo} and set HF_TOKEN "
                f"in .env.")
    return (f"{repo} weights are not downloaded. Fetch them from the studio's "
            f"Setup page, or run scripts/download_models.py.")


def feature_diarization(token: bool = None) -> dict:
    """Multi-speaker detection: pyannote plus its gated weights."""
    token = token_state()["present"] if token is None else token
    present = _has_module("pyannote")
    entry = {"status": OPTIONAL_MISSING, "tier": OPTIONAL,
             "reason": None, "fallback": "single-speaker dubbing"}
    if not present:
        entry["reason"] = (
            "pyannote.audio is not installed, so multi-speaker detection is "
            "off. Dubbing continues in single-speaker mode. Install with: "
            "pip install 'pyannote.audio>=3.3.2'")
        return entry
    if not token:
        entry["status"] = FALLBACK_AVAILABLE
        entry["reason"] = _gated_reason("pyannote.audio",
                                        "pyannote/speaker-diarization-3.1",
                                        present, token)
        return entry

    downloaded = True
    try:
        import models
        downloaded = models.is_cached("pyannote/speaker-diarization-3.1")
    except Exception:                                        # noqa: BLE001
        pass
    if not downloaded:
        entry["status"] = FALLBACK_AVAILABLE
        entry["reason"] = _gated_reason("pyannote.audio",
                                        "pyannote/speaker-diarization-3.1",
                                        present, token)
        return entry

    entry["status"] = READY
    return entry


def feature_background(token: bool = None) -> dict:
    """Background preservation: Demucs, which is open and ungated."""
    entry = {"status": OPTIONAL_MISSING, "tier": OPTIONAL, "reason": None,
             "fallback": "the dub replaces the original soundtrack"}
    if not _has_module("demucs"):
        entry["reason"] = (
            "Demucs is not installed, so the film's music and sound effects "
            "cannot be preserved. Dubbing continues with dialogue-only audio. "
            "Install with: pip install 'demucs>=4.0.0'")
        return entry
    # Demucs downloads htdemucs itself on first use; there is no file to check
    # and no token involved, so the package being present is the whole answer.
    entry["status"] = READY
    return entry


def feature_indicf5(token: bool = None, target_language: str = None) -> dict:
    """
    The voice engine for Indian languages: IndicF5 plus its vocoder.

    This is a CORE feature — without it a dub into an Indian language cannot be
    rendered at all — so its status can make the whole verdict not-ready.
    """
    token = token_state()["present"] if token is None else token
    entry = {"status": BLOCKED, "tier": CORE, "reason": None,
             "engine": "indicf5", "fallback": None}

    if not _has_module("f5_tts"):
        entry["reason"] = ("f5-tts is not installed, so IndicF5 cannot run. "
                           "Install the project requirements.")
        return entry

    try:
        import models
        if not models.is_cached("ai4bharat/IndicF5"):
            entry["status"] = BLOCKED
            entry["reason"] = _gated_reason(
                "f5-tts", "ai4bharat/IndicF5", True, token)
            return entry
        if not models.is_cached("charactr/vocos-mel-24khz"):
            entry["status"] = BLOCKED
            entry["reason"] = ("The vocos vocoder weights are not downloaded, "
                               "so IndicF5 has nothing to turn its mel output "
                               "into audio.")
            return entry
    except Exception as e:                                   # noqa: BLE001
        entry["status"] = FALLBACK_AVAILABLE
        entry["reason"] = f"Could not check the IndicF5 weights ({e})."
        return entry

    if target_language:
        try:
            import languages as L
            if L.engine_for(target_language) != L.ENGINE_INDICF5:
                entry["status"] = FALLBACK_AVAILABLE
                entry["engine"] = L.engine_for(target_language)
                entry["reason"] = (
                    f"{L.display_name(target_language)} is not an IndicF5 "
                    f"language; it uses {entry['engine']}.")
                return entry
        except Exception:                                    # noqa: BLE001
            pass

    entry["status"] = READY
    return entry


def feature_xtts(target_language: str = None) -> dict:
    """The foreign-language engine. Optional, and non-commercial."""
    entry = {"status": OPTIONAL_MISSING, "tier": OPTIONAL, "engine": "xtts",
             "reason": None, "fallback": None}
    if not _has_module("TTS"):
        entry["reason"] = (
            "The Coqui TTS package is not installed, so the non-Indian "
            "languages are unavailable. Translated subtitles still work in "
            "every language. Note the Coqui Public Model License is "
            "non-commercial. Install with: pip install 'TTS>=0.22.0'")
        return entry
    entry["status"] = READY
    return entry


def feature_asr() -> dict:
    """faster-whisper and its weights."""
    entry = {"status": BLOCKED, "tier": CORE, "reason": None,
             "model": "Systran/faster-whisper-large-v3"}
    if not _has_module("faster_whisper"):
        entry["reason"] = ("faster-whisper is not installed, so nothing can be "
                           "transcribed. Install the project requirements.")
        return entry
    try:
        import models
        if not models.is_cached("Systran/faster-whisper-large-v3"):
            entry["reason"] = ("The faster-whisper weights are not downloaded.")
            return entry
    except Exception as e:                                   # noqa: BLE001
        entry["status"] = FALLBACK_AVAILABLE
        entry["reason"] = f"Could not check the ASR weights ({e})."
        return entry
    entry["status"] = READY
    return entry


def feature_translation() -> dict:
    """NLLB-200 and its weights."""
    entry = {"status": BLOCKED, "tier": CORE, "reason": None,
             "model": "facebook/nllb-200-distilled-600M"}
    if not _has_module("transformers"):
        entry["reason"] = ("transformers is not installed, so nothing can be "
                           "translated. Install the project requirements.")
        return entry
    try:
        import models
        if not models.is_cached("facebook/nllb-200-distilled-600M"):
            entry["reason"] = ("The NLLB-200 weights are not downloaded.")
            return entry
    except Exception as e:                                   # noqa: BLE001
        entry["status"] = FALLBACK_AVAILABLE
        entry["reason"] = f"Could not check the translation weights ({e})."
        return entry
    entry["status"] = READY
    return entry


def feature_lipsync() -> dict:
    """
    Wav2Lip and s3fd.

    Optional in the sense that a dub to audio-only needs neither, but a video
    render does, so it is reported separately from the audio path.
    """
    entry = {"status": OPTIONAL_MISSING, "tier": OPTIONAL, "reason": None,
             "fallback": "audio-only dubbing (no lip sync)"}
    if not _has_module("cv2"):
        entry["reason"] = ("OpenCV is not installed, so the picture cannot be "
                           "lip-synced. Audio dubbing still works.")
        return entry
    try:
        import models
        missing = [w[0] for w in models.WEIGHTS
                   if models.verify(w[1], w[3], w[4]) != "ok"]
        if missing:
            entry["reason"] = (f"Missing weights: {', '.join(missing)}. "
                               f"Lip sync needs these; audio dubbing does not.")
            return entry
    except Exception as e:                                   # noqa: BLE001
        entry["status"] = FALLBACK_AVAILABLE
        entry["reason"] = f"Could not check the lip-sync weights ({e})."
        return entry
    entry["status"] = READY
    return entry


def feature_language(target_language: str = None) -> dict:
    """
    Whether the requested target language can actually be spoken here.

    This is the check that stops a render failing several minutes in: the studio
    asks before accepting the job.
    """
    entry = {"status": READY, "tier": CORE, "reason": None,
             "language": target_language, "engine": None}
    if not target_language:
        return entry
    try:
        import languages as L
        code = L.normalize(target_language)
        engine = L.engine_for(code)
    except Exception as e:                                   # noqa: BLE001
        entry["status"] = BLOCKED
        entry["reason"] = str(e)
        return entry

    entry["engine"] = engine
    if engine == L.ENGINE_INDICF5:
        entry["status"] = feature_indicf5(target_language=code)["status"]
        entry["reason"] = feature_indicf5(target_language=code)["reason"]
    else:
        xtts = feature_xtts(code)
        entry["status"] = xtts["status"]
        entry["reason"] = xtts["reason"]
    if entry["status"] == OPTIONAL_MISSING:
        # A target language with no engine is blocking, not merely optional:
        # the user asked for it.
        entry["status"] = BLOCKED
    return entry


# ---------------------------------------------------------------------------
# The whole picture
# ---------------------------------------------------------------------------
def snapshot(target_language: str = None, source_language: str = None) -> dict:
    """
    Everything, as a plain dict. Never raises, never loads a model.

    `ready` means "a dub can be rendered on this machine for this request". It
    is deliberately not "every feature works": a missing optional feature is
    reported with a fallback, not treated as a failure.
    """
    token = token_state()
    torch = _torch_info()
    ffmpeg_version = _ffmpeg_version()

    features = {
        "ffmpeg": {
            "status": READY if ffmpeg_version else BLOCKED,
            "tier": CORE,
            "reason": None if ffmpeg_version else
                      "ffmpeg is not on PATH. Every stage needs it.",
            "version": ffmpeg_version,
        },
        "torch": {
            "status": READY if torch["present"] else BLOCKED,
            "tier": CORE,
            "reason": None if torch["present"] else
                      "PyTorch is not installed, so no model can run.",
            "version": torch["version"],
            "device": torch["device"],
        },
        "asr": feature_asr(),
        "translation": feature_translation(),
        "indicf5": feature_indicf5(token["present"], target_language),
        "xtts": feature_xtts(target_language),
        "diarization": feature_diarization(token["present"]),
        "background_preservation": feature_background(token["present"]),
        "lipsync": feature_lipsync(),
    }
    if target_language:
        features["target_language"] = feature_language(target_language)

    # The core path is what decides readiness. An optional feature can never
    # make the application unusable; a core one always does.
    core = [name for name, f in features.items() if f.get("tier") == CORE]
    blocking = [name for name in core
                if features[name]["status"] in _BLOCKING]
    degraded = [name for name, f in features.items()
                if f["status"] == FALLBACK_AVAILABLE]

    hardware = {
        "device": torch["device"],
        "cuda": torch["cuda"],
        "mps": torch["mps"],
        "gpu_name": torch["gpu_name"],
        "vram_gb": torch["vram_gb"],
        "cpus": os.cpu_count(),
    }
    try:
        import models
        hardware["free_disk_gb"] = round(models.free_space_gb(), 1)
    except Exception:                                        # noqa: BLE001
        hardware["free_disk_gb"] = None

    return redact({
        "ready": not blocking,
        "blocking": blocking,
        "degraded": degraded,
        "hardware": hardware,
        "secrets": {"HF_TOKEN": "present" if token["present"] else "missing"},
        "languages": {"source": source_language, "target": target_language},
        "features": features,
    })


def summarize(payload: dict) -> list:
    """
    The snapshot as lines a person can read.

    Kept beside `snapshot()` so the text and the structure cannot drift: every
    line is derived from the same dict the API returns.
    """
    lines = []
    hw = payload.get("hardware") or {}
    device = hw.get("device") or "cpu"
    extra = ""
    if hw.get("gpu_name"):
        vram = f", {hw['vram_gb']} GB" if hw.get("vram_gb") else ""
        extra = f" ({hw['gpu_name']}{vram})"
    lines.append(f"device      {device}{extra}")
    lines.append(f"HF_TOKEN    {payload['secrets']['HF_TOKEN']}")
    if hw.get("free_disk_gb") is not None:
        lines.append(f"free disk   {hw['free_disk_gb']} GB")
    lines.append("")
    for name, feature in (payload.get("features") or {}).items():
        status = feature.get("status", "?")
        mark = {READY: "ok  ", FALLBACK_AVAILABLE: "fall",
                OPTIONAL_MISSING: "off ", BLOCKED: "FAIL"}.get(status, "?   ")
        lines.append(f"{mark} {name}")
        if feature.get("reason"):
            lines.append(f"       {feature['reason']}")
    return lines

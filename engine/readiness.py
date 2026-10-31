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
import sys

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
# Runtime compatibility: NumPy, which numba refuses above 2.2
# ---------------------------------------------------------------------------
#
# numba is not optional here: librosa loads audio through it, and both the voice
# engine and Wav2Lip load audio with librosa. A hosted run verified numpy 2.2.6
# and then started a server that reported "Numba needs NumPy 2.2 or less. Got
# NumPy 2.5" - two warmups failed, and the Setup page had already called both
# features ready. The version alone did not identify it, so the report names the
# interpreter and the numpy file too.

NUMPY_MAXIMUM = "2.2"


def _version_tuple(text: str) -> tuple:
    """(2, 2, 6) from "2.2.6": leading digits per part."""
    parts = []
    for chunk in str(text).split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def numpy_verdict(version: str | None,
                  numba_error: str | None = None) -> tuple:
    """Whether this NumPy can run numba, plus the sentence to say if it cannot.

    Pure on purpose: the failing case is a NumPy that is not installed here, so
    it can only be tested by passing the version in.
    """
    if not version:
        return False, "NumPy is not installed in this interpreter."
    if numba_error:
        first = (numba_error or "").strip().splitlines()
        return False, ("numba cannot be imported in this interpreter: "
                       + (first[0] if first else "unknown error"))
    if _version_tuple(version)[:2] > _version_tuple(NUMPY_MAXIMUM)[:2]:
        return False, (f"numba needs NumPy {NUMPY_MAXIMUM} or less and this "
                       f"interpreter has {version}. librosa loads audio through "
                       "numba, so the voice model and Wav2Lip cannot load while "
                       "every package and weight check still passes.")
    return True, None


def numpy_report() -> dict:
    """What NumPy this process has, and whether numba accepts it.

    `sys.executable` and `numpy.__file__` are the point: "Got NumPy 2.5" in a
    log does not say WHICH interpreter loaded it, so a session that verified
    2.2.6 reads like an impossible report.
    """
    report = {"executable": sys.executable, "version": None, "file": None,
              "numba": None, "maximum": NUMPY_MAXIMUM, "compatible": False,
              "reason": None}
    numba_error = None
    try:
        import numba
        report["numba"] = getattr(numba, "__version__", "?")
    except Exception as e:                                   # noqa: BLE001
        numba_error = str(e)
    try:
        import numpy
        report["version"] = getattr(numpy, "__version__", None)
        report["file"] = getattr(numpy, "__file__", None)
    except Exception as e:                                   # noqa: BLE001
        report["reason"] = f"NumPy is not importable ({e})."
        return report

    ok, reason = numpy_verdict(report["version"], numba_error)
    report["compatible"] = ok
    report["reason"] = reason
    return report


def require_compatible_runtime() -> dict:
    """Fail a server start that cannot run the pipeline, and say exactly why.

    Called from the app's lifespan, so the failure is one clear startup error
    carrying the interpreter, the NumPy file and the numba version - instead of
    two "warmup failed" lines while the Setup page advertises IndicF5 and lip
    sync. `VIDEOTRANS_ALLOW_INCOMPATIBLE_NUMPY=1` starts anyway for work that
    needs neither (transcription, translation, subtitles); readiness still
    reports both features as unavailable, so nothing claims to be ready.
    """
    report = numpy_report()
    if report["compatible"]:
        return report
    reason = report["reason"] or "NumPy is not usable in this interpreter."
    override = os.getenv("VIDEOTRANS_ALLOW_INCOMPATIBLE_NUMPY", "")
    if override not in ("", "0", "false", "no"):
        print("[APP] WARNING: " + reason)
        print("[APP] Continuing because VIDEOTRANS_ALLOW_INCOMPATIBLE_NUMPY is "
              "set; the voice model and lip sync will fail to load.")
        return report
    lines = [
        "Runtime check failed: " + reason,
        f"  executable : {report['executable']}",
        f"  numpy      : {report['version']} ({report['file']})",
        f"  numba      : {report['numba']}",
        f"  needed     : numpy <= {NUMPY_MAXIMUM}",
        "Install the pinned NumPy in THIS interpreter "
        "(pip install numpy==2.2.6) and start again. Set "
        "VIDEOTRANS_ALLOW_INCOMPATIBLE_NUMPY=1 to start without the voice model "
        "and lip sync.",
    ]
    print("\n".join(lines))
    raise RuntimeError(lines[0])


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


def _source_of(module_name: str) -> str | None:
    """The text of a local module's source file, or None if it is not there."""
    try:
        import importlib.util
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin or not os.path.isfile(spec.origin):
            return None
        with open(spec.origin, encoding="utf-8") as fh:
            return fh.read()
    except Exception:                                        # noqa: BLE001
        return None


def _defines(module_name: str, *names: str) -> list | None:
    """Which of these top-level names a module's source does NOT define.

    Parsed rather than imported: `translation_v2` imports torch, and a readiness
    probe has to stay cheap and work on a machine with no GPU. None means the
    module could not be found at all, which is a different problem from a name
    missing inside it.
    """
    source = _source_of(module_name)
    if source is None:
        return None
    try:
        import ast
        tree = ast.parse(source)
    except Exception:                                        # noqa: BLE001
        return None
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
    return [name for name in names if name not in defined]


def translation_api() -> str | None:
    """None when the translate stage exposes what the application calls.

    Two renames took this stage down while every package and weight check
    passed: the boot warmup imports `_load_model`, and the studio's model switch
    calls `refresh()`, and neither existed. Reading the names out of the source
    means the next rename shows up here instead of in a failed render.
    """
    missing = _defines("translation_v2", "translate_segments", "_load_model",
                       "refresh")
    if not missing:
        return None
    return ("The translate stage's code does not expose "
            f"{', '.join(missing)}, which the application calls at boot and "
            "when the model is changed, so a render would fail at translation. "
            "This is a code/install mismatch, not a missing weight.")


def diarization_api() -> str | None:
    """None when the diarization call path can authenticate to pyannote.

    pyannote 3.x takes `use_auth_token`; `token` arrived with the newer hub API.
    Passing the wrong name raised TypeError on every load, so the whole video
    was dubbed in one voice while this module still reported the feature ready.
    """
    source = _source_of("speaker_diarization")
    if source is None:
        return None
    if "use_auth_token" not in source:
        return ("The diarization call path does not pass pyannote's "
                "`use_auth_token`, so loading the pipeline raises TypeError and "
                "the whole video is dubbed in one voice. This is a code/version "
                "mismatch, not a missing weight.")


# Everything `python -m demucs.separate` imports before it does anything.
_DEMUCS_MODULES = ("demucs.separate", "lameenc", "julius", "openunmix", "dora",
                   "omegaconf")


def _demucs_missing() -> list:
    """Demucs' own import-time dependencies this runtime cannot resolve."""
    try:
        import source_separation as S
        return S.missing_deps()
    except Exception:                                        # noqa: BLE001
        return [name for name in _DEMUCS_MODULES if not _has_module(name)]


# Every Hub snapshot pyannote's 3.1 pipeline loads: the pipeline itself, the
# segmentation model that finds the turns, and the embedding model that tells
# the voices apart.
_DIARIZATION_REPOS = ("pyannote/speaker-diarization-3.1",
                      "pyannote/segmentation-3.0",
                      "pyannote/wespeaker-voxceleb-resnet34-LM")


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

    # ALL THREE repos, not just the pipeline one. pyannote 3.1's config names
    # `pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM`,
    # which it fetches on first use. Checking only the pipeline repo made the
    # Setup page say READY while a render still had to download, or -- offline --
    # quietly produced one voice for a two-hander.
    downloaded = True
    try:
        import models
        downloaded = all(models.is_cached(repo) for repo in _DIARIZATION_REPOS)
    except Exception:                                        # noqa: BLE001
        pass
    if not downloaded:
        entry["status"] = FALLBACK_AVAILABLE
        entry["reason"] = _gated_reason("pyannote.audio",
                                        "pyannote/speaker-diarization-3.1",
                                        present, token)
        return entry

    api = diarization_api()
    if api:
        entry["status"] = OPTIONAL_MISSING
        entry["reason"] = api
        return entry

    entry["status"] = READY
    return entry


def feature_background(token: bool = None) -> dict:
    """Background preservation: Demucs and the modules it imports at startup."""
    entry = {"status": OPTIONAL_MISSING, "tier": OPTIONAL, "reason": None,
             "fallback": "the dub replaces the original soundtrack"}
    if not _has_module("demucs"):
        entry["reason"] = (
            "Demucs is not installed, so the film's music and sound effects "
            "cannot be preserved. Dubbing continues with dialogue-only audio. "
            "Install with: pip install 'demucs>=4.0.0'")
        return entry

    # `find_spec("demucs")` alone answered ready for an install that could not
    # start: Demucs arrives with --no-deps, so lameenc, openunmix and
    # dora-search were absent and `python -m demucs.separate` died on import.
    # The probe asks Demucs' own import list instead.
    missing = _demucs_missing()
    if missing:
        entry["reason"] = (
            f"Demucs is installed but cannot start: {', '.join(missing)} is "
            "missing from this interpreter (Demucs is installed with --no-deps "
            "so it cannot rewrite torch or numpy). Background preservation is "
            "off; dubbing continues with dialogue-only audio. Install with: "
            f"pip install --no-deps {' '.join(missing)}")
        return entry

    # No weights to check: Demucs fetches htdemucs itself on first use, and no
    # token is involved.
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

    numpy_ok = numpy_report()
    if not numpy_ok["compatible"]:
        entry["status"] = BLOCKED
        entry["reason"] = numpy_ok["reason"]
        return entry

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
    api = translation_api()
    if api:
        entry["status"] = BLOCKED
        entry["reason"] = api
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
    numpy_ok = numpy_report()
    if not numpy_ok["compatible"]:
        entry["reason"] = numpy_ok["reason"]
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
        "runtime": numpy_report(),
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

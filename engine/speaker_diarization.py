"""
Speaker diarization: who spoke when, and which transcript words are theirs.

Two jobs, deliberately separated:

  `diarize()`          who spoke when — one backend call, no transcript needed
  `assign_speakers()`  which segment/word is whose — pure, model-free, testable

The split matters because the assignment logic is where the subtle bugs live
(mid-word speaker changes, unresolved words, genuine overlap) and it can be
tested exhaustively without a GPU, a model, or a network. Only `diarize()`
needs pyannote.

Backends sit behind `Diarizer` so a different engine can be added later without
touching the pipeline. pyannote is the default: one package, and its weights are
gated exactly like IndicF5's, so it reuses the token and the "is it downloaded"
machinery this project already has. DiariZen is *not* integrated — it needs a
fork of pyannote plus a torch pin that contradicts this repo's.

Every failure here is non-fatal. No pyannote, no token, no model, a crash, or
zero turns all fall back to one implicit `SPEAKER_00`, which is the pipeline
this project shipped before diarization existed. A dub must never fail because
the speaker detector had a bad day.
"""

from __future__ import annotations

import os
import threading

import speakers as SPK

# ---------------------------------------------------------------------------
# Version compatibility for pyannote 3.3.2
# ---------------------------------------------------------------------------
# pyannote 3.3.2 is pinned because pyannote 4.x would drag in torch>=2.8 and
# break the CUDA-12 torch 2.6 build. But 3.3.2 was written against
# huggingface_hub 0.x and torch 2.5, and two of those assumptions no longer
# hold, so the pipeline raises on load and the whole job silently drops to
# single-speaker mode — which is exactly why a two-hander comes back in one
# voice:
#
#   * `hf_hub_download(use_auth_token=...)` — huggingface_hub 2.0 removed the
#     `use_auth_token` keyword. pyannote still passes it, so every download
#     dies with `unexpected keyword argument 'use_auth_token'`.
#   * `torch.load` — torch 2.6 flipped the default of `weights_only` to True.
#     pyannote's checkpoints hold `pyannote.audio.core.task.Specifications` and
#     friends, which the safe unpickler refuses, so loading the weights raises
#     `UnpicklingError: Weights only load failed`.
#
# Both are repaired here, once, before any pyannote import, and only when the
# installed library actually needs it — on the versions pyannote was written
# for this is a no-op. The shims are narrow: they translate one keyword and
# restore one default, and touch nothing else.
_SHIMS_APPLIED = False
_SHIMS_LOCK = threading.Lock()


def _apply_compat_shims() -> None:
    """
    Make pyannote 3.3.2 load on huggingface_hub 2.x and torch 2.6.

    Idempotent and failure-safe: any problem here leaves the library untouched
    and the caller falls back to the single-speaker path exactly as before.
    """
    global _SHIMS_APPLIED
    with _SHIMS_LOCK:
        if not _SHIMS_APPLIED:
            _SHIMS_APPLIED = True
            _shim_torch_load()
        # The hub rebind runs on every call, not just the first: pyannote
        # imports `hf_hub_download` at module load, so whichever import order
        # the caller used, the already-imported pyannote modules must be
        # pointed at the shim. Cheap, and it closes the ordering hole where a
        # shim applied before pyannote was imported left pyannote holding the
        # original function.
        _shim_hf_hub_download()


def _shim_hf_hub_download() -> None:
    """
    Translate `use_auth_token=` to `token=` for callers that still pass it.

    huggingface_hub 2.0 renamed the argument. Only the download helpers
    pyannote imports are wrapped, so the rest of the hub API is untouched.
    """
    try:
        import inspect

        import huggingface_hub
        from huggingface_hub import hf_hub_download
    except Exception:                                        # noqa: BLE001
        return
    try:
        params = inspect.signature(hf_hub_download).parameters
    except (TypeError, ValueError):
        return
    # Nothing to do on the versions that still accept the old name, or on a
    # hub so old it has no `token` keyword to translate to.
    if "use_auth_token" in params or "token" not in params:
        return
    if getattr(hf_hub_download, "_vt_hub_shim", False):
        return

    real = hf_hub_download

    def shim(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return real(*args, **kwargs)

    shim._vt_hub_shim = True
    shim.__wrapped__ = real
    huggingface_hub.hf_hub_download = shim
    # pyannote does `from huggingface_hub import hf_hub_download` at module
    # import time, so patching the hub package alone is not enough — those
    # modules hold their own reference to the original function. Rebind every
    # already-imported pyannote module that captured it.
    import sys
    for name, module in list(sys.modules.items()):
        if not name.startswith("pyannote") or module is None:
            continue
        if getattr(module, "hf_hub_download", None) is real:
            try:
                module.hf_hub_download = shim
            except Exception:                                # noqa: BLE001
                pass
    print("[DIAR] Patched huggingface_hub for pyannote 3.3.2 "
          "(use_auth_token -> token)")


def _shim_torch_load() -> None:
    """
    Restore `weights_only=False` for pyannote's checkpoint loader.

    torch 2.6 made `weights_only=True` the default, which refuses the
    `Specifications`/`TorchVersion` objects pyannote serialises into its
    checkpoints. Lightning passes `weights_only` explicitly, so the value has
    to be forced rather than defaulted. These are the project's own pinned,
    licence-accepted checkpoints, so the trust assumption `weights_only=False`
    encodes is satisfied.
    """
    try:
        import inspect

        import torch
    except Exception:                                        # noqa: BLE001
        return
    try:
        params = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return
    if "weights_only" not in params:
        return
    if getattr(torch.load, "_vt_load_shim", False):
        return

    real = torch.load

    def shim(*args, **kwargs):
        kwargs["weights_only"] = False
        return real(*args, **kwargs)

    shim._vt_load_shim = True
    shim.__wrapped__ = real
    torch.load = shim
    print("[DIAR] Patched torch.load for pyannote 3.3.2 "
          "(weights_only=False)")

# The repo's default. pyannote 3.1 is what `video-dubbing-system` and
# `Ai-Vedio-Dubbing` both use, and it is the one whose output shape the
# reference implementations already handle.
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

# How far a word's midpoint may fall outside a turn and still count as that
# speaker's. pyannote's boundaries are not sample-accurate, and Whisper's word
# timings are interpolated for most Indian-language audio, so a strict test
# would leave a large fraction of words unassigned.
TOLERANCE = 0.20

# A turn shorter than this is a cough or a backchannel, not a line. Dropping
# them keeps a two-hander from being reported as a six-hander.
MIN_TURN = 0.20

# A detected speaker with less attributed speech than this is a blip rather than
# a cast member, and is folded into their nearest neighbour. Set generously low:
# a real speaker who says one short line still clears it, while a cough or a
# music sting does not. The operator tunable DIAR_MIN_SPEAKER_SPEECH overrides
# this; 0 disables the fold.
MIN_SPEAKER_SPEECH = 1.0

# ...but only if they are also a negligible SHARE of the speech. A short clip
# where two people split the audio has two speakers, not two blips, even though
# each falls under the 1s floor. A real speaker in a two-hander holds tens of
# percent of the audio; a cough holds single digits. A speaker must be under the
# absolute floor AND under this share of the total to be folded away.
MIN_SPEAKER_SHARE = 0.15

# When the top two speakers in a window are this close, the window is a real
# overlap and picking one is a coin flip. Refusing to guess is the point: a
# confidently wrong speaker is worse than an unresolved one, because it clones
# the wrong actor for the whole line.
OVERLAP_MARGIN = 0.15

# A window whose winning speaker holds less than this share of it is not
# convincingly anyone's.
MIN_DOMINANCE = 0.20

_TURNS = dict  # {"start": float, "end": float, "speaker": str}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Diarizer:
    """The interface every backend implements. See `PyannoteDiarizer`."""

    name = "none"

    def available(self) -> bool:
        """Whether this backend can run here, without loading anything heavy."""
        return False

    def diarize(self, audio_path: str, min_speakers: int = 1,
                max_speakers: int = None) -> list:
        raise NotImplementedError


class PyannoteDiarizer(Diarizer):
    """
    pyannote.audio, loaded once and held.

    The pipeline is a warm singleton for the same reason the voice engine is:
    loading it costs seconds, and a multi-speaker run calls it once per video,
    not once per phrase.
    """

    name = "pyannote"

    def __init__(self, model: str = None, device: str = None):
        self.model = model or os.getenv("DIARIZATION_MODEL") or DEFAULT_MODEL
        self._device = device
        self._pipeline = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("pyannote.audio") is not None
        except Exception:                                    # noqa: BLE001
            return False

    def _token(self):
        # Same resolution order as every other gated model in this project:
        # the environment wins, then .env, which `localenv` already loaded.
        return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            from pyannote.audio import Pipeline

            # Apply the pyannote-3.3.2 compatibility shims BEFORE the pipeline
            # touches huggingface_hub or torch.load. Without them the load
            # raises and the job silently dubs a conversation in one voice.
            _apply_compat_shims()

            token = self._token()
            if not token:
                raise RuntimeError(
                    "Diarization needs a Hugging Face token: "
                    f"{self.model} is gated. Accept the licence at "
                    f"https://huggingface.co/{self.model}, then set HF_TOKEN "
                    "in .env (see docs/GATED-MODELS.md).")
            print(f"[DIAR] Loading {self.model}...")
            try:
                pipeline = self._from_pretrained(token)
            except Exception as e:                           # noqa: BLE001
                raise RuntimeError(self._explain_load_failure(e)) from e
            device = self._device
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif getattr(torch.backends, "mps", None) and \
                            torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                except Exception:                            # noqa: BLE001
                    device = "cpu"
            if device != "cpu":
                try:
                    import torch
                    pipeline = pipeline.to(torch.device(device))
                except Exception as e:                       # noqa: BLE001
                    print(f"[DIAR] Could not move to {device} ({e}); using cpu")
                    device = "cpu"
            print(f"[DIAR] Ready on {device}")
            self._pipeline = pipeline
            return self._pipeline

    def _from_pretrained(self, token: str):
        """Load the pipeline with the auth keyword THIS pyannote accepts.

        The pinned 3.3.2 signature is `from_pretrained(checkpoint_path,
        hparams_file=None, use_auth_token=None, cache_dir=...)` - `token`
        arrived with the newer hub API. Passing the wrong name raised
            Pipeline.from_pretrained() got an unexpected keyword argument 'token'
        on every run, which the caller reports as "diarization unavailable" and
        then dubs a conversation in one voice. Try the pinned spelling first and
        fall back to the newer one, so neither version loses speaker detection.
        """
        from pyannote.audio import Pipeline

        try:
            return Pipeline.from_pretrained(self.model, use_auth_token=token)
        except TypeError as e:
            if "use_auth_token" not in str(e) and "token" not in str(e):
                raise
            # The shim above normally makes the old spelling work. If a hub
            # version still refuses both, the newer spelling is the last try;
            # if that also fails, re-raise the original so the real cause (a
            # gated-model 401/403, not the keyword) is what gets reported.
            try:
                return Pipeline.from_pretrained(self.model, token=token)
            except TypeError:
                raise e

    def _explain_load_failure(self, exc: Exception) -> str:
        """
        Say which of the four things went wrong, not just that one did.

        The brief's requirement. These have four different fixes and a single
        "could not load the model" sends the user to the wrong one:

          * the package is missing        -> install it
          * no token                      -> create one
          * a token but no access         -> accept the licence
          * access granted, weights gone  -> download them

        Hugging Face's own errors distinguish the middle two only by phrasing
        (401 vs 403, "Cannot access gated repo" vs "not in the authorized
        list"), so the text is matched rather than the exception type — the
        library wraps these in several different classes across versions.
        """
        text = str(exc)
        low = text.lower()
        model_url = f"https://huggingface.co/{self.model}"

        if "no module named" in low or "cannot import" in low:
            return ("pyannote.audio is not installed. "
                    "Install it with: pip install 'pyannote.audio>=3.3.2'")
        if "401" in text or "unauthorized" in low:
            return ("Hugging Face rejected the token (401). Check that HF_TOKEN "
                    "is valid and not expired.")
        if "403" in text or "gated" in low or "authorized" in low \
                or "access to model" in low:
            return (f"HF_TOKEN is set but access to {self.model} has not been "
                    f"granted. Accept the licence at {model_url} using the "
                    f"same account the token belongs to, then retry.")
        if "not found" in low or "404" in text:
            return (f"{self.model} weights are not available locally and could "
                    f"not be fetched. Download them from the studio's Setup "
                    f"page, or run scripts/download_models.py.")
        if "connection" in low or "timeout" in low or "temporary" in low:
            return (f"Could not reach Hugging Face to fetch {self.model}. "
                    f"Check the network, then retry.")
        # `splitlines()[0]` on an empty message is an IndexError, which would
        # turn "the model could not be loaded" into a crash inside the code
        # that explains why the model could not be loaded.
        first = text.splitlines()[0][:200] if text.strip() else "unknown error"
        return f"Could not load {self.model}: {first}"

    def diarize(self, audio_path: str, min_speakers: int = 1,
                max_speakers: int = None) -> list:
        pipeline = self._load()
        # Only pass the bounds that were actually set. pyannote's own default
        # is "find as many as you hear", so an unset ceiling is omitted rather
        # than guessed at — a ceiling is what turns a three-hander into a
        # two-voice dub.
        hints = {}
        if min_speakers is not None:
            hints["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            hints["max_speakers"] = int(max_speakers)
        try:
            output = pipeline(audio_path, **hints)
        except TypeError:
            # Older builds do not take the speaker-count hints.
            output = pipeline(audio_path)

        # pyannote 3.x returns an Annotation; 4.x returns a DiarizeOutput
        # wrapping one. `video-dubbing-system` handles the same split.
        annotation = getattr(output, "speaker_diarization", output)

        turns = []
        for turn, _, label in annotation.itertracks(yield_label=True):
            start, end = float(turn.start), float(turn.end)
            if end - start < MIN_TURN:
                continue
            turns.append({"start": start, "end": end, "speaker": str(label)})
        turns.sort(key=lambda t: t["start"])
        return turns


_BACKENDS = {"pyannote": PyannoteDiarizer}
_INSTANCES: dict = {}
_LOCK = threading.Lock()

# The reason the most recent `diarize()` returned nothing, or None when it
# succeeded. `diarize` must never raise (a dub cannot fail because the speaker
# detector did), so this is how the caller — and the per-step log — learns WHY
# a conversation collapsed to one voice instead of only that it did.
_LAST_ERROR: str | None = None


def _set_last_error(message: str | None) -> None:
    global _LAST_ERROR
    _LAST_ERROR = message


def last_error() -> str | None:
    """Why the last `diarize()` produced no turns, or None if it worked."""
    return _LAST_ERROR


def get_backend(name: str = None) -> Diarizer:
    """A warm backend instance. `name` defaults to $DIARIZATION_BACKEND."""
    name = (name or os.getenv("DIARIZATION_BACKEND") or "pyannote").strip().lower()
    cls = _BACKENDS.get(name)
    if cls is None:
        print(f"[DIAR] Unknown backend {name!r}; known: {sorted(_BACKENDS)}")
        return Diarizer()
    with _LOCK:
        if name not in _INSTANCES:
            _INSTANCES[name] = cls()
        return _INSTANCES[name]


def refresh() -> None:
    """Drop the warm backend so the next call picks up a new setting."""
    with _LOCK:
        _INSTANCES.clear()


def release() -> None:
    """
    Drop the loaded pipeline, keeping the backend object.

    Diarization and the voice model do not fit on a small GPU at the same time,
    and the diarizer is idle for the rest of the run once the turns are known.
    `Python-Autodub` frees it explicitly before F5-TTS loads for the same
    reason. Reloading it costs seconds; an out-of-memory render costs the whole
    job.
    """
    with _LOCK:
        for impl in _INSTANCES.values():
            pipeline = getattr(impl, "_pipeline", None)
            if pipeline is not None:
                try:
                    del impl._pipeline
                    impl._pipeline = None
                except Exception:                            # noqa: BLE001
                    pass
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[DIAR] Released the diarization model to free memory")
    except Exception:                                        # noqa: BLE001
        pass


def available(name: str = None) -> bool:
    """Whether diarization can run here at all, cheaply and without loading."""
    try:
        return get_backend(name).available()
    except Exception:                                        # noqa: BLE001
        return False


def _readable_audio(path: str):
    """
    `(audio_path, cleanup)` pyannote's loader can actually read.

    pyannote reads through libsndfile, which understands wav/flac/ogg but NOT
    an mp4/mkv container. Handed the source video it raises "Format not
    recognised", `diarize` swallows that and returns [], and the whole job
    silently falls back to a single voice — which is exactly how a two-hander
    gets dubbed entirely in the first speaker's voice. Callers are expected to
    pass extracted audio, but this is the last line of defence: if the path is
    not readable as sound, decode it to a temporary wav with ffmpeg and report
    it for deletion afterwards.

    `cleanup` is the temporary file to remove, or None when the input was
    already usable (the common case, so nothing is copied for it).

    A failed decode is NOT fatal: the original path is handed back so the
    backend can try it (a test backend needs no real file, and a real one will
    raise into `diarize`'s handler either way). This is a repair, not a gate.
    """
    try:
        import soundfile as sf
        sf.info(path)
        return path, None
    except Exception:                                        # noqa: BLE001
        pass

    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="diar_")
    os.close(fd)
    try:
        # 16 kHz mono is what pyannote resamples to internally anyway, so this
        # is not a quality compromise — it is the same signal, one decode
        # earlier, from a container ffmpeg reads and libsndfile does not.
        import av_sync
        av_sync.extract_audio(path, tmp, 16000, 1)
    except Exception:                                        # noqa: BLE001
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return path, None
    print(f"[DIAR] Decoded {os.path.basename(path)} to audio for diarization "
          f"(pyannote cannot read a video container directly)")
    return tmp, tmp


def diarize(audio_path: str, min_speakers: int = 1, max_speakers: int = None,
            backend: str = None) -> list:
    """
    Speaker turns for an audio file, or [] if anything at all goes wrong.

    Returning [] rather than raising is the contract: the caller treats "no
    turns" and "diarization is unavailable" identically, as one implicit
    speaker. That is what keeps a failed diarization from failing a render.
    """
    working = audio_path
    cleanup = None
    try:
        impl = get_backend(backend)
        if not impl.available():
            # The brief's requirement: name the cause and the consequence, so
            # the fallback is visible rather than silent.
            print("[DIAR] Speaker diarization unavailable:\n"
                  "       pyannote.audio is not installed.\n"
                  "       The job will continue in single-speaker mode.\n"
                  "       Install with:  pip install 'pyannote.audio>=3.3.2'")
            _set_last_error("pyannote.audio is not installed")
            return []
        working, cleanup = _readable_audio(audio_path)
        turns = impl.diarize(working, min_speakers=min_speakers,
                             max_speakers=max_speakers)
        print(f"[DIAR] {len(turns)} turns, "
              f"{len({t['speaker'] for t in turns})} speaker(s)")
        _set_last_error(None)
        return turns
    except Exception as e:                                   # noqa: BLE001
        # The message from `_load` already says which of the four cases it is
        # (package, token, weights, access). Passing it through rather than
        # replacing it with "failed" is the whole point.
        print(f"[DIAR] Speaker diarization unavailable:\n"
              f"       {e}\n"
              f"       The job will continue in single-speaker mode.")
        _set_last_error(str(e))
        return []
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Assignment — pure, model-free
# ---------------------------------------------------------------------------
def _overlaps(turn: dict, start: float, end: float) -> float:
    return max(0.0, min(end, turn["end"]) - max(start, turn["start"]))


def speaker_at(mid: float, turns: list, tolerance: float = TOLERANCE):
    """
    The speaker of the turn covering `mid`, or None.

    A tolerance widens each turn slightly at both ends. Without it, the gap
    between two adjacent turns — which pyannote does leave, by tens of
    milliseconds — drops the word between them.

    Ranking is by distance to the turn, not to its centre. Distance-to-centre
    lets a long turn beat a short one that actually contains the word, which is
    exactly backwards; distance-to-turn is zero for any turn that contains it,
    and only falls back to a boundary gap when nothing does.

    Ties go to the shorter turn, then the earlier one. Inside a genuine overlap
    the shorter turn is the more specific claim - a brief interjection inside a
    long speech - and an exact tie between two adjacent turns resolves to the
    one that started first, which is the same "carry the earlier speaker
    forward" instinct the rest of this module uses.
    """
    best, best_key = None, None
    for turn in turns:
        if turn["end"] < mid:
            distance = mid - turn["end"]
        elif mid < turn["start"]:
            distance = turn["start"] - mid
        else:
            distance = 0.0
        if distance > tolerance:
            continue
        key = (distance, turn["end"] - turn["start"], turn["start"])
        if best_key is None or key < best_key:
            best, best_key = turn["speaker"], key
    return best


def dominant_speaker(start: float, end: float, turns: list,
                     margin: float = OVERLAP_MARGIN,
                     min_dominance: float = MIN_DOMINANCE):
    """
    The speaker who owns a window, or None when the window is a real overlap.

    Returns `(label, confidence)` where confidence is the winner's share of the
    window and `label` is None when no speaker is clearly ahead. This is the
    `Python-Autodub` margin rule: if the runner-up is within `margin` of the
    winner, two people are talking at once and guessing would clone the wrong
    voice for the entire line.
    """
    span = end - start
    if span <= 0 or not turns:
        return None, 0.0

    shares: dict = {}
    for turn in turns:
        overlap = _overlaps(turn, start, end)
        if overlap > 0:
            shares[turn["speaker"]] = shares.get(turn["speaker"], 0.0) + overlap

    if not shares:
        return None, 0.0

    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top = ranked[0]
    top_ratio = min(1.0, top / span)
    runner_up = (ranked[1][1] / span) if len(ranked) > 1 else 0.0

    if top_ratio < min_dominance:
        return None, top_ratio
    if (top_ratio - runner_up) < margin:
        return None, top_ratio
    return top_label, top_ratio


def assign_words(segments: list, turns: list,
                 tolerance: float = TOLERANCE) -> list:
    """
    Label every word with a speaker, then roll the labels up to segments.

    Word level is the right granularity: a Whisper segment routinely spans a
    speaker change, and labelling the segment as a whole would give the whole
    exchange to whoever talked more. `Ai-Vedio-Dubbing` does the same and it is
    the most robust of the three reference schemes for word timings.

    Unresolved words carry the last known speaker forward. Without that, every
    inter-turn gap becomes an `UNKNOWN` third speaker — which is how a
    two-person scene turns into a diarization report with five voices.
    """
    if not turns:
        return []

    out = []
    last_known = None
    for seg in segments or []:
        words = seg.get("words") or []
        if not words:
            # No word timings (short clips, some languages): fall back to the
            # segment's own span, which is coarser but never absent.
            label, _ = dominant_speaker(float(seg.get("start", 0.0)),
                                        float(seg.get("end", 0.0)), turns)
            label = label or last_known
            if label:
                last_known = label
            out.append({**seg, "speaker": label})
            continue

        labelled = []
        for word in words:
            try:
                mid = (float(word["start"]) + float(word["end"])) / 2.0
            except (KeyError, TypeError, ValueError):
                labelled.append({**word, "speaker": last_known})
                continue
            label = speaker_at(mid, turns, tolerance) or last_known
            if label:
                last_known = label
            labelled.append({**word, "speaker": label})

        label = _majority([w.get("speaker") for w in labelled]) or last_known
        if label:
            last_known = label
        out.append({**seg, "words": labelled, "speaker": label})
    return out


def _majority(labels: list):
    """The most common non-None label, ties going to the first seen."""
    counts: dict = {}
    for label in labels:
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], -list(counts).index(kv[0])))[0]


# A run of words shorter than this on one side of a speaker change is a
# backchannel ("mhm", "yeah"), not a line. Splitting there would hand one word
# to the other actor and then speak it in their voice, which is the mismatch
# this function exists to remove.
MIN_SPLIT_SECONDS = 0.30


def _split_at_changes(segment: dict) -> list:
    """
    Split one labelled segment wherever the speaker changes.

    Whisper segments do not respect speaker boundaries: a three-second segment
    routinely starts with one actor's question and ends with the other's answer.
    Labelling the segment as a whole (which is what this used to do) gave the
    whole exchange to whoever talked more, so half of it was spoken in the wrong
    voice — audibly, male lines in the female voice.

    The words already carry per-word labels from `assign_words`, so the split is
    a groupby over them. A segment with one speaker — every segment of a
    single-speaker video — is returned unchanged, so that path is untouched.
    """
    words = segment.get("words") or []
    if len(words) < 2:
        return [_row(segment, segment.get("start"), segment.get("end"),
                     segment.get("text"), segment.get("speaker"), words)]

    labels = {w.get("speaker") for w in words}
    if len(labels) < 2 or None in labels:
        return [_row(segment, segment.get("start"), segment.get("end"),
                     segment.get("text"), segment.get("speaker"), words)]

    # Word timings are required: the split rows' spans come from them, and the
    # timeline schedules against the spans.
    if any(w.get("start") is None or w.get("end") is None for w in words):
        return [_row(segment, segment.get("start"), segment.get("end"),
                     segment.get("text"), segment.get("speaker"), words)]

    groups = []
    for word in words:
        if groups and groups[-1]["speaker"] == word["speaker"]:
            groups[-1]["words"].append(word)
        else:
            groups.append({"speaker": word["speaker"], "words": [word]})

    # More than a handful of changes inside one Whisper segment is a diarizer
    # flip-flopping, not a conversation. Splitting there would shred a line
    # into alternating one-word phrases spoken by different actors.
    if len(groups) > 4:
        return [_row(segment, segment.get("start"), segment.get("end"),
                     segment.get("text"), segment.get("speaker"), words)]

    # Fold a too-short run into the neighbour it is closest to, so a cough in
    # the middle of a line cannot become that speaker's one-word phrase.
    merged = []
    for group in groups:
        span = _span(group["words"])
        if merged and span < MIN_SPLIT_SECONDS:
            merged[-1]["words"].extend(group["words"])
        else:
            merged.append(group)
    if len(merged) > 1 and _span(merged[0]["words"]) < MIN_SPLIT_SECONDS:
        merged[1]["words"] = merged[0]["words"] + merged[1]["words"]
        merged = merged[1:]

    if len(merged) < 2:
        return [_row(segment, segment.get("start"), segment.get("end"),
                     segment.get("text"), segment.get("speaker"), words)]

    rows = []
    for group in merged:
        start, end = _span(group["words"], with_bounds=True)
        text = " ".join((w.get("word") or "").strip()
                        for w in group["words"]).strip()
        if not text:
            continue
        rows.append(_row(segment, start, end, text, group["speaker"],
                         group["words"]))
    return rows or [_row(segment, segment.get("start"), segment.get("end"),
                         segment.get("text"), segment.get("speaker"), words)]


def _span(words: list, with_bounds: bool = False):
    """The seconds a run of words occupies, or (start, end) when asked."""
    try:
        start = min(float(w["start"]) for w in words)
        end = max(float(w["end"]) for w in words)
    except (KeyError, TypeError, ValueError):
        return (0.0, 0.0) if with_bounds else 0.0
    return (start, end) if with_bounds else max(0.0, end - start)


def _row(segment: dict, start, end, text, speaker, words) -> dict:
    """One segment-shaped row, with the timings coerced the way callers read them."""
    return {"start": float(start if start is not None
                           else segment.get("start", 0.0)),
            "end": float(end if end is not None
                         else segment.get("end", 0.0)),
            "text": (text or "").strip(),
            "speaker": speaker,
            "words": words}


def assign_segments(segments: list, turns: list,
                    tolerance: float = TOLERANCE) -> list:
    """
    Speaker-aware segments: `[{start, end, text, speaker}, ...]`.

    Uses word-level assignment when the transcript has word timings, and the
    window-dominance rule otherwise. Segments are never dropped; an unresolved
    one carries the previous speaker forward and is only left unlabelled when
    nothing has been resolved at all (which the caller reads as "no
    diarization" and answers with one implicit speaker).
    """
    if not turns:
        return list(segments or [])

    with_words = any((seg.get("words") or []) for seg in segments or [])
    if with_words:
        labelled = assign_words(segments, turns, tolerance)
        rows = []
        for s in labelled:
            rows.extend(_split_at_changes(s))
        return rows

    out = []
    last_known = None
    for seg in segments or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        label, _ = dominant_speaker(start, end, turns)
        label = label or last_known
        if label:
            last_known = label
        out.append({"start": start, "end": end,
                    "text": (seg.get("text") or "").strip(),
                    "speaker": label,
                    "words": seg.get("words")})
    return out


def speaker_spans(segments: list) -> dict:
    """
    How much speech each label accounts for, and where their longest runs are.

    `segments` is the output of `assign_segments`. Used by Phase 3 to decide
    which speaker has enough clean audio to clone, and to pick their reference
    windows without re-reading the transcript.
    """
    out: dict = {}
    for seg in segments or []:
        label = SPK.label_of(seg)
        if not label:
            continue
        entry = out.setdefault(label, {"total_speech": 0.0, "segments": []})
        try:
            span = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        except (TypeError, ValueError):
            continue
        if span <= 0:
            continue
        entry["total_speech"] += span
        entry["segments"].append(seg)
    for entry in out.values():
        entry["total_speech"] = round(entry["total_speech"], 3)
    return out


def merge_spurious_speakers(segments: list, turns: list,
                            min_speech: float = MIN_SPEAKER_SPEECH,
                            keep_min: int = 1):
    """
    Fold speakers with almost no speech into their nearest real neighbour.

    A diarizer does not only miss speakers — it invents them. A cough, a door, a
    burst of music, or a breath between two lines comes back as its own label,
    and a monologue with one such blip is suddenly a "two-speaker" video: the
    multi-speaker path runs, the studio shows a phantom voice, and a handful of
    words are spoken in a voice that is not the speaker's. The blip itself is
    harmless (it is never cloned — `build_references` finds no usable window and
    marks it unresolved rather than borrowing another actor), but the
    classification flip and the phantom entry are not.

    Every label whose attributed speech is under `min_speech` seconds is a blip.
    Its segments and turns are reassigned to the temporally nearest KEPT
    speaker, so a stray syllable lands in the voice that was actually talking.
    `keep_min` is a floor on how many speakers survive — a genuinely quiet
    speaker is not deleted to satisfy the rule — so a clip where EVERY label is
    a blip keeps its dominant one rather than collapsing to nothing.

    Returns `(segments, turns, merged)` where `merged` maps each dropped label
    to the label that absorbed it, for the log.

    `min_speech <= 0` disables the fold entirely.
    """
    segments = list(segments or [])
    turns = list(turns or [])
    if min_speech <= 0 or not turns:
        return segments, turns, {}

    spans = speaker_spans(segments)
    if len(spans) <= max(1, int(keep_min)):
        return segments, turns, {}

    # A blip is small in absolute terms AND a negligible share of the speech.
    # The share test protects a short clip: on a 5s two-hander each speaker
    # holds ~50% of the audio, so neither is a blip even though both are under
    # the 1s floor, while a 0.3s cough in a 20s clip is both.
    total_speech = sum(e["total_speech"] for e in spans.values())

    def _is_blip(entry: dict) -> bool:
        seconds = entry["total_speech"]
        if seconds >= min_speech:
            return False
        share = (seconds / total_speech) if total_speech > 0 else 0.0
        return share < MIN_SPEAKER_SHARE

    # Order labels by speech, richest first, so the ones we KEEP are the real
    # speakers and the floor protects the richest when everything is a blip.
    ranked = sorted(spans.items(), key=lambda kv: kv[1]["total_speech"],
                    reverse=True)
    keep = max(1, int(keep_min))
    kept = [label for label, _ in ranked[:keep]]
    for label, entry in ranked[keep:]:
        if not _is_blip(entry):
            kept.append(label)
    kept_set = set(kept)

    dropped = [label for label, _ in ranked if label not in kept_set]
    if not dropped:
        return segments, turns, {}

    # Where each kept speaker sits in time, for nearest-neighbour resolution.
    centres: dict = {}
    for seg in segments:
        label = SPK.label_of(seg)
        if label in kept_set:
            try:
                mid = (float(seg.get("start", 0.0)) + float(seg.get("end", 0.0))) / 2.0
            except (TypeError, ValueError):
                continue
            centres.setdefault(label, []).append(mid)

    def _nearest(when: float) -> str | None:
        best, best_gap = None, None
        for label, mids in centres.items():
            gap = min(abs(when - m) for m in mids)
            if best_gap is None or gap < best_gap:
                best, best_gap = label, gap
        return best or (kept[0] if kept else None)

    merged: dict = {}
    for label in dropped:
        segs = spans.get(label, {}).get("segments") or []
        if not segs:
            continue
        mids = [(float(s.get("start", 0.0)) + float(s.get("end", 0.0))) / 2.0
                for s in segs]
        target = _nearest(sum(mids) / len(mids))
        if target:
            merged[label] = target

    if not merged:
        return segments, turns, {}

    out_segments = []
    for seg in segments:
        label = SPK.label_of(seg)
        if label in merged:
            out_segments.append({**seg, "speaker": merged[label]})
        else:
            out_segments.append(seg)

    out_turns = []
    for turn in turns:
        label = str(turn.get("speaker"))
        if label in merged:
            out_turns.append({**turn, "speaker": merged[label]})
        else:
            out_turns.append(turn)

    return out_segments, out_turns, merged


def summarize(segments: list, turns: list) -> str:
    """One log line: what was found, and how much of it is unresolved."""
    if not turns:
        return "[DIAR] No diarization; a single speaker is used"
    labels = {t["speaker"] for t in turns}
    unresolved = sum(1 for s in segments or [] if not SPK.label_of(s))
    speech = sum((float(s.get("end", 0)) - float(s.get("start", 0)))
                 for s in segments or []
                 if SPK.label_of(s) and
                 float(s.get("end", 0)) > float(s.get("start", 0)))
    return (f"[DIAR] {len(turns)} turns, {len(labels)} speaker(s) "
            f"{sorted(labels)}; {speech:.1f}s attributed, "
            f"{unresolved} segment(s) unresolved")

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
                max_speakers: int = 10) -> list:
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

            token = self._token()
            if not token:
                raise RuntimeError(
                    "Diarization needs a Hugging Face token: "
                    f"{self.model} is gated. Accept the licence at "
                    f"https://huggingface.co/{self.model}, then set HF_TOKEN "
                    "in .env (see docs/GATED-MODELS.md).")
            print(f"[DIAR] Loading {self.model}...")
            try:
                pipeline = Pipeline.from_pretrained(self.model, token=token)
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
                max_speakers: int = 10) -> list:
        pipeline = self._load()
        try:
            output = pipeline(audio_path, min_speakers=min_speakers,
                              max_speakers=max_speakers)
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


def diarize(audio_path: str, min_speakers: int = 1, max_speakers: int = 10,
            backend: str = None) -> list:
    """
    Speaker turns for an audio file, or [] if anything at all goes wrong.

    Returning [] rather than raising is the contract: the caller treats "no
    turns" and "diarization is unavailable" identically, as one implicit
    speaker. That is what keeps a failed diarization from failing a render.
    """
    try:
        impl = get_backend(backend)
        if not impl.available():
            # The brief's requirement: name the cause and the consequence, so
            # the fallback is visible rather than silent.
            print("[DIAR] Speaker diarization unavailable:\n"
                  "       pyannote.audio is not installed.\n"
                  "       The job will continue in single-speaker mode.\n"
                  "       Install with:  pip install 'pyannote.audio>=3.3.2'")
            return []
        turns = impl.diarize(audio_path, min_speakers=min_speakers,
                             max_speakers=max_speakers)
        print(f"[DIAR] {len(turns)} turns, "
              f"{len({t['speaker'] for t in turns})} speaker(s)")
        return turns
    except Exception as e:                                   # noqa: BLE001
        # The message from `_load` already says which of the four cases it is
        # (package, token, weights, access). Passing it through rather than
        # replacing it with "failed" is the whole point.
        print(f"[DIAR] Speaker diarization unavailable:\n"
              f"       {e}\n"
              f"       The job will continue in single-speaker mode.")
        return []


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

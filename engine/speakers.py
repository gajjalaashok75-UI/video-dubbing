"""
The speaker registry: one entry per detected voice, and the helpers that
resolve a unit to the reference it must be spoken in.

Before this module the pipeline had exactly one voice. Every phrase was
synthesized from the single `reference` dict that `reference_audio.build_reference`
returned, so footage with two people talking came back in one voice.

The registry is deliberately a plain list of dicts rather than a class
hierarchy: it is written into the project manifest as JSON and read back by the
studio, and a dict survives that round trip unchanged.

    {
      "label": "SPEAKER_01",
      "total_speech": 41.7,        # seconds of speech attributed to them
      "segment_count": 23,
      "reference": {"path", "text", "start", "end", "duration"},
      "reference_score": 3.42,     # reference_audio._score_window's score
      "fallback": False            # True when no clean same-speaker clip existed
    }

A single-speaker video gets exactly one entry, `SPEAKER_00`, whose reference is
the one the old pipeline would have picked. That is what makes this additive:
`implicit_speaker()` plus the global `reference` reproduces the previous
behaviour byte for byte.
"""

from __future__ import annotations

# The label every pipeline used before diarization existed. Kept as the
# fallback label so an unlabelled segment behaves exactly as it always did.
DEFAULT_LABEL = "SPEAKER_00"


def implicit_speaker(reference: dict = None) -> list:
    """
    The one-speaker registry: what a video with no diarization looks like.

    `reference` is the dict `reference_audio.build_reference()` returns. It may
    be None (a subtitle-only run never builds one), in which case the entry is
    still created so every segment has a label to carry.
    """
    entry = {
        "label": DEFAULT_LABEL,
        "total_speech": 0.0,
        "segment_count": 0,
        "reference": dict(reference) if reference else None,
        "reference_score": None,
        "fallback": False,
    }
    return [entry]


def normalize_speakers(value) -> list:
    """
    Coerce whatever is on a job into a usable registry.

    Jobs written before this module existed have no `speakers` key at all, and
    a manifest that has been through JSON may hold anything. Returning an empty
    list (rather than raising) keeps every caller on the "no diarization" path,
    which is the behaviour those jobs already had.
    """
    if not isinstance(value, list):
        return []
    out = []
    for entry in value:
        if not isinstance(entry, dict) or not entry.get("label"):
            continue
        ref = entry.get("reference")
        out.append({
            "label": str(entry["label"]),
            "total_speech": float(entry.get("total_speech") or 0.0),
            "segment_count": int(entry.get("segment_count") or 0),
            "reference": dict(ref) if isinstance(ref, dict) else None,
            "reference_score": entry.get("reference_score"),
            "fallback": bool(entry.get("fallback")),
        })
    return out


def labels(speakers: list) -> list:
    """The labels in registry order, deduplicated but order-preserving."""
    seen, out = set(), []
    for entry in speakers or []:
        label = (entry or {}).get("label")
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def label_of(segment: dict) -> str | None:
    """A segment's speaker, or None when it was never assigned one."""
    if not isinstance(segment, dict):
        return None
    label = segment.get("speaker")
    return str(label) if label else None


def ref_index_for(speakers: list, label: str | None) -> int | None:
    """
    Where `label` sits in the registry.

    Units store this alongside the label so a reordered or partially restored
    registry can still be matched by position, and so the studio has a stable
    integer identity to key on.
    """
    if not label:
        return None
    for i, entry in enumerate(speakers or []):
        if (entry or {}).get("label") == label:
            return i
    return None


def reference_for(unit: dict, speakers: list, fallback: dict = None) -> dict | None:
    """
    The reference profile a planned unit must be spoken in.

    Resolution order, most specific first:

      1. the registry entry at the unit's `ref_index`
      2. the registry entry whose label matches the unit's `speaker`
      3. `fallback` — the global reference the caller passed in

    Step 3 is what keeps a single-speaker job, and a job whose registry has been
    lost, on exactly the previous code path.
    """
    speakers = speakers or []

    idx = (unit or {}).get("ref_index")
    if isinstance(idx, int) and 0 <= idx < len(speakers):
        ref = (speakers[idx] or {}).get("reference")
        if isinstance(ref, dict) and ref.get("path"):
            return ref

    label = (unit or {}).get("speaker")
    if label:
        for entry in speakers:
            if (entry or {}).get("label") == label:
                ref = (entry or {}).get("reference")
                if isinstance(ref, dict) and ref.get("path"):
                    return ref

    return fallback


def tally(segments: list, label: str) -> dict:
    """How much speech a label accounts for: seconds and segment count."""
    total, count = 0.0, 0
    for seg in segments or []:
        if label_of(seg) != label:
            continue
        try:
            span = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        except (TypeError, ValueError):
            continue
        if span <= 0:
            continue
        total += span
        count += 1
    return {"total_speech": round(total, 3), "segment_count": count}


def build(segments: list, speakers: list) -> list:
    """
    Refresh the per-label tallies on a registry from the final segment list.

    Called once the segments carry their speakers, so the manifest reports real
    numbers rather than the zeros `implicit_speaker()` starts with.
    """
    out = []
    for entry in speakers or []:
        label = (entry or {}).get("label")
        if not label:
            continue
        counts = tally(segments, label)
        out.append({**entry, **counts})
    return out


# A same-speaker reference shorter than this clones unreliably. F5's speaker
# identity is unstable below roughly two seconds, and quality drops off well
# before the 6s the pipeline prefers.
SHORT_REFERENCE_SECONDS = 6.0
MIN_USABLE_SECONDS = 2.0

# The four states a speaker can be in. Deliberately an explicit vocabulary
# rather than booleans: the studio shows these words, the tests assert them, and
# the run manifest records them, so they have to mean one thing.
CLONED = "cloned"                      # a clean same-speaker reference
SHORT_REFERENCE = "short_reference"    # usable, but under the preferred length
GENERIC_FALLBACK = "generic_fallback"  # an explicit generic voice
UNRESOLVED = "unresolved"              # no usable reference at all


def status_of(entry: dict) -> str:
    """
    Which of the four states a registry entry is in.

    Derived from the entry rather than stored on it, so it cannot go stale when
    a reference is re-cut. `fallback` distinguishes the two no-own-voice cases:
    a generic voice is a deliberate substitution, an empty reference is not.
    """
    ref = (entry or {}).get("reference") or {}
    if not ref.get("path"):
        return UNRESOLVED
    if entry.get("fallback"):
        return GENERIC_FALLBACK
    try:
        duration = float(ref.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration < MIN_USABLE_SECONDS:
        return UNRESOLVED
    if duration < SHORT_REFERENCE_SECONDS:
        return SHORT_REFERENCE
    return CLONED


def warning_for(entry: dict) -> str | None:
    """
    What a person needs to know about this speaker, in one sentence, or None.

    The brief's requirement: do not silently hide a weak clone. A speaker that
    fell back is the single most likely reason a dub sounds wrong, and it is
    invisible in the output unless it is said out loud.
    """
    state = status_of(entry)
    label = (entry or {}).get("label") or "A speaker"
    ref = (entry or {}).get("reference") or {}
    try:
        duration = float(ref.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    if state == CLONED:
        return None
    if state == SHORT_REFERENCE:
        return (f"{label} has only {duration:.1f} seconds of usable reference "
                f"audio. Voice similarity may be reduced.")
    if state == GENERIC_FALLBACK:
        return (f"{label} could not obtain a reliable voice reference. "
                f"A fallback voice is being used.")
    return (f"{label} could not obtain a reliable voice reference, so their "
            f"lines are spoken in the video's default voice.")


def describe(entry: dict, aliases: dict = None) -> dict:
    """
    One speaker, as the studio and the run manifest want it.

    The internal `label` is always present and never replaced by an alias. A
    display name is an overlay: the stable id is what the cache, the plan and
    the manifest key on, and a rename must not move any of those.
    """
    label = (entry or {}).get("label") or ""
    ref = dict((entry or {}).get("reference") or {})
    state = status_of(entry)
    alias = ((aliases or {}).get(label) or "").strip() or None

    out = {
        "label": label,
        # `name` is what a person sees; `label` is what everything keys on.
        "name": alias or label,
        "alias": alias,
        "status": state,
        "warning": warning_for(entry),
        "fallback": bool((entry or {}).get("fallback")),
        "total_speech": round(float((entry or {}).get("total_speech") or 0.0), 2),
        "segment_count": int((entry or {}).get("segment_count") or 0),
        "reference_score": (entry or {}).get("reference_score"),
        "reference": {
            "path": ref.get("path"),
            "text": ref.get("text") or "",
            "start": ref.get("start"),
            "end": ref.get("end"),
            "duration": round(float(ref.get("duration") or 0.0), 2),
        } if ref.get("path") else None,
    }
    # How many phrases this speaker will actually speak. Filled in by the caller
    # once a plan exists; absent before then rather than reported as zero, which
    # would read as "this speaker says nothing".
    if "phrase_count" in (entry or {}):
        out["phrase_count"] = int(entry["phrase_count"])
    return out


def describe_all(speakers: list, aliases: dict = None,
                 plan: list = None) -> list:
    """The whole registry, described, with phrase counts from `plan` if given."""
    counts: dict = {}
    for unit in plan or []:
        label = unit.get("speaker") or DEFAULT_LABEL
        counts[label] = counts.get(label, 0) + 1

    out = []
    for entry in speakers or []:
        row = describe(entry, aliases)
        if plan is not None:
            row["phrase_count"] = counts.get(row["label"], 0)
        out.append(row)
    return out


def warnings(speakers: list) -> list:
    """Every speaker warning, in registry order. Empty when all are CLONED."""
    return [w for w in (warning_for(e) for e in speakers or []) if w]


def summarize(speakers: list, segments: list) -> str:
    """One line for the log: who was found and how much each of them said."""
    if not speakers:
        return "[SPK] No diarization; a single voice is used"
    parts = []
    for entry in speakers:
        label = entry.get("label")
        ref = entry.get("reference") or {}
        seconds = ref.get("duration")
        mark = " (fallback voice)" if entry.get("fallback") else ""
        speech = f"{entry.get('total_speech', 0.0):.1f}s speech"
        segs = f"{entry.get('segment_count', 0)} segments"
        if seconds:
            parts.append(f"{label}: {speech}, {segs}, "
                         f"{seconds:.1f}s reference{mark}")
        else:
            parts.append(f"{label}: {speech}, no reference{mark}")
    return f"[SPK] {len(speakers)} speaker(s) — " + "; ".join(parts)


# A generic voice is used only when a speaker has no usable same-speaker audio
# at all. It is deliberately an explicit opt-in file rather than a bundled
# clip: shipping a voice nobody chose is its own kind of impersonation, and the
# honest default is to mark the speaker unresolved.
GENERIC_DIR = "assets/generic"


def generic_voice_path() -> str | None:
    """
    The generic fallback voice, or None.

    Resolved from $VIDEOTRANS_GENERIC_VOICE first, then the first WAV in
    `engine/assets/generic/`. Nothing ships there by default, so out of the box
    a speaker with no usable audio is marked unresolved rather than given a
    voice nobody asked for.
    """
    import os
    forced = os.getenv("VIDEOTRANS_GENERIC_VOICE")
    if forced and os.path.exists(forced):
        return forced
    here = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(here, GENERIC_DIR)
    if not os.path.isdir(directory):
        return None
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith((".wav", ".flac", ".mp3")):
            return os.path.join(directory, name)
    return None


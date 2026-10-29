"""
What each artifact depends on, and what invalidates it.

Every stage of this pipeline caches something expensive. The caches grew up one
phase at a time and each decided its own reuse rule: the transcript by clip
length and source language, the stems by a hash of the file's size and mtime,
the phrase audio by index alone. Each rule is defensible on its own and there is
no single place that says what depends on what, which is how a change to one
input ends up reusing an artifact that no longer matches it.

This module is that single place. It does not move any bytes; it describes the
graph, and answers two questions:

    what does this artifact depend on?
    is the copy I have still valid for these inputs?

The rule the brief calls out is the important one: **file existence is not
validity**. A stems file left over from an untrimmed clip is present and wrong,
and reusing it puts the whole bed out of step with the picture. So validity is
always a comparison against the inputs the artifact was made from, never
`os.path.exists`.

Invalidation is deliberately narrow. A change to one input invalidates that
artifact and everything downstream of it, and nothing else — re-running Demucs
because a translation changed would be minutes of work for an unrelated answer.
"""

from __future__ import annotations

import os

# The artifact graph, in dependency order. `depends_on` lists the artifacts that
# must be current before this one is; `inputs` lists the job fields it is a
# function of. Together they say what a change invalidates.
ARTIFACTS = {
    # Audio extraction. Depends only on the clip and the trim.
    "stt_audio": {
        "inputs": ("input_path", "trim"),
        "depends_on": (),
        "cost": "seconds",
    },
    "hq_audio": {
        "inputs": ("input_path", "trim"),
        "depends_on": (),
        "cost": "seconds",
    },
    # The stems depend on the clip, the trim and the separation settings. A
    # different chunk size is a different result.
    "stems": {
        "inputs": ("input_path", "trim", "demucs_chunk_seconds", "demucs_model"),
        "depends_on": ("hq_audio",),
        "cost": "minutes",
    },
    # The transcript depends on the audio it was read from — which is the
    # vocals stem when separation ran, and the original otherwise. It therefore
    # depends on `stems`: re-separating with different settings produces
    # different audio, and reading a transcript off audio that has changed is
    # exactly the stale-cache case this graph exists to catch. ASR is expensive,
    # but it is genuinely downstream of separation, not merely adjacent to it.
    "transcript": {
        "inputs": ("stt_audio", "speech_source", "source_language", "asr_model"),
        "depends_on": ("stems",),
        "cost": "minutes",
    },
    # Diarization reads the original audio, not the stem, so it does not depend
    # on separation. Getting this wrong would re-run diarization every time
    # Demucs was toggled.
    "diarization": {
        "inputs": ("input_path", "trim", "diar_min_speakers",
                   "diar_max_speakers", "diarization_backend"),
        "depends_on": (),
        "cost": "minutes",
    },
    # Speaker assignments are a pure function of the transcript and the turns.
    "speakers": {
        "inputs": ("transcript", "diarization"),
        "depends_on": ("transcript", "diarization"),
        "cost": "seconds",
    },
    # References depend on the speaker assignment and the audio they are cut
    # from. Re-cloning ONE speaker invalidates only that speaker's entry, which
    # `reference_inputs` handles below rather than the whole artifact.
    "references": {
        "inputs": ("speakers", "speech_source", "reference_min_seconds",
                   "reference_max_seconds"),
        "depends_on": ("speakers",),
        "cost": "seconds",
    },
    # Translation depends on the transcript and the target language, and on
    # nothing about the audio.
    "translation": {
        "inputs": ("transcript", "source_language", "target_language",
                   "for_speech", "translation_model"),
        "depends_on": ("transcript",),
        "cost": "minutes",
    },
    # The timeline depends on the speakers and their references, because the
    # per-speaker tempo model is calibrated from the reference clip.
    "plan": {
        "inputs": ("speakers", "references", "translation", "video_duration"),
        "depends_on": ("speakers", "references", "translation"),
        "cost": "seconds",
    },
    # One cached WAV per phrase. Indexed, so an edit touches one entry.
    "phrase_audio": {
        "inputs": ("plan", "references"),
        "depends_on": ("plan", "references"),
        "cost": "minutes",
        "per_unit": True,
    },
    "dialogue_track": {
        "inputs": ("plan", "phrase_audio", "video_duration", "overlap_gain"),
        "depends_on": ("phrase_audio",),
        "cost": "seconds",
    },
    # The mix depends on the dialogue and the stems, and on the level settings.
    "mixed_audio": {
        "inputs": ("dialogue_track", "stems", "music_path", "music_gain",
                   "dialogue_lufs", "background_lufs", "background_duck",
                   "duck_db", "duck_ratio", "video_duration"),
        "depends_on": ("dialogue_track", "stems"),
        "cost": "seconds",
    },
    "output_video": {
        "inputs": ("mixed_audio", "dialogue_track", "input_path", "trim",
                   "output_crf", "output_preset"),
        "depends_on": ("dialogue_track",),
        "cost": "minutes",
    },
}


def inputs_for(name: str) -> tuple:
    """The job fields an artifact is a function of."""
    entry = ARTIFACTS.get(name)
    return tuple(entry["inputs"]) if entry else ()


def depends_on(name: str) -> tuple:
    """The artifacts that must be current before this one is."""
    entry = ARTIFACTS.get(name)
    return tuple(entry["depends_on"]) if entry else ()


def downstream(name: str) -> list:
    """
    Everything that becomes stale when `name` changes, transitively.

    The answer to "if I change this, what has to be redone?" — and the reason
    invalidation can be narrow instead of "wipe the workdir".
    """
    out, queue = [], [name]
    while queue:
        current = queue.pop()
        for candidate, entry in ARTIFACTS.items():
            if current in entry["depends_on"] and candidate not in out:
                out.append(candidate)
                queue.append(candidate)
    return out


def upstream(name: str) -> list:
    """Everything `name` needs, transitively."""
    out, queue = [], list(depends_on(name))
    while queue:
        current = queue.pop()
        if current in out:
            continue
        out.append(current)
        queue.extend(depends_on(current))
    return out


def invalidated_by(changed: list) -> list:
    """
    Which artifacts a set of changed inputs makes stale.

    `changed` is a list of artifact names or input field names. Input fields are
    matched against every artifact that names them, which is how a change to
    `target_language` reaches the translation and everything after it while
    leaving the stems alone.
    """
    stale = set()
    for item in changed or []:
        if item in ARTIFACTS:
            stale.add(item)
            stale.update(downstream(item))
            continue
        for name, entry in ARTIFACTS.items():
            if item in entry["inputs"]:
                stale.add(name)
                stale.update(downstream(name))
    return sorted(stale, key=_order)


def _order(name: str) -> int:
    """Dependency order, so a list reads top-to-bottom."""
    return list(ARTIFACTS).index(name) if name in ARTIFACTS else len(ARTIFACTS)


def cheapest_stale(stale: list) -> str | None:
    """
    The most expensive thing that has to be redone, for a warning.

    A user about to change a setting deserves to know it costs minutes rather
    than seconds, and the honest number is the worst one in the set.

    Ties go to the DEEPEST artifact — the last one in dependency order — because
    that is the one whose cost dominates: a change that re-runs both translation
    and phrase synthesis is minutes either way, but the phrase synthesis is what
    the user is actually waiting on. The choice is deterministic so a warning
    does not change wording between two identical runs.
    """
    rank = {"seconds": 0, "minutes": 1}
    worst, worst_key = None, (-1, -1)
    for name in stale or []:
        entry = ARTIFACTS.get(name)
        if not entry:
            continue
        key = (rank.get(entry.get("cost"), 0), _order(name))
        if key > worst_key:
            worst, worst_key = name, key
    return worst


def reference_inputs(label: str) -> tuple:
    """
    What one speaker's reference is a function of.

    Per-speaker, deliberately. A reference is cut from that speaker's own words
    in the source audio, so changing speaker B's window cannot invalidate
    speaker A's clip — which is what makes a per-speaker re-clone cheap.
    """
    return ("speakers", "speech_source", "reference_window:" + str(label))


def reference_invalidates(label: str) -> list:
    """
    What a change to ONE speaker's reference invalidates.

    Only that speaker's phrases, and the things built from them. Every other
    speaker's cached audio survives, which is requirement 10 of the brief and
    the reason `phrase_audio` is `per_unit`.
    """
    return ["references:" + str(label), "phrase_audio:" + str(label),
            "dialogue_track", "mixed_audio", "output_video"]


def fingerprint(job: dict) -> dict:
    """
    The inputs an artifact set was built from, as a comparable dict.

    Only fields that exist are recorded, so a manifest written before a setting
    was introduced compares equal on the fields it does have rather than reading
    as a change.
    """
    out = {}
    for name, entry in ARTIFACTS.items():
        for field in entry["inputs"]:
            if field in job:
                out[field] = job[field]
    return out


def describe() -> str:
    """The graph as text, for a log or a doc."""
    lines = ["Artifact dependency graph", ""]
    for name, entry in ARTIFACTS.items():
        deps = ", ".join(entry["depends_on"]) or "—"
        lines.append(f"{name:<18} depends on: {deps}")
        lines.append(f"{'':<18} inputs:     {', '.join(entry['inputs'])}")
        lines.append(f"{'':<18} cost:       {entry['cost']}"
                     + ("  (per phrase)" if entry.get("per_unit") else ""))
        lines.append("")
    return "\n".join(lines)


def stale_artifacts(job: dict, previous: dict) -> list:
    """
    What a re-run has to redo, given the job and the manifest it would reuse.

    `previous` is a stored fingerprint. Only fields present in both are
    compared, so a new setting does not read as a change to an old run.
    """
    if not previous:
        return sorted(ARTIFACTS, key=_order)
    changed = [key for key, value in previous.items()
               if key in job and job[key] != value]
    if not changed:
        return []
    return invalidated_by(changed)


def cache_root_for(workdir: str) -> str:
    """Where a job's cached artifacts live, for a report."""
    return os.path.join(workdir)

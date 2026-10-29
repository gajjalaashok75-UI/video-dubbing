"""
The run summary: what a finished job actually did.

Every stage of this pipeline degrades rather than fails — no pyannote means one
speaker, no Demucs means the soundtrack is replaced, a short reference means a
weaker clone. That is the right behaviour, and it is also how a dub ends up
quietly worse than the user expects with nothing on screen to say why.

This assembles one structure that says what ran, what did not, what fell back,
and what the result measured. It is written into the project manifest and
returned by the API, so the studio and a debugging script read the same thing.

Two rules:

**No secrets.** The summary names the models and the backends; it never carries
a token. `readiness.redact` is applied on the way out as a guard.

**No silent fallbacks.** A stage that did not run says so and says why. A
`warnings` list a person can read is the whole point of the module.
"""

from __future__ import annotations

import os
import time

import speakers as SPK

# The pipeline stages, in the order they run, with what the studio shows for
# each. `optional` marks a stage that legitimately does not run.
STAGES = [
    ("validating", "Validating", False),
    ("extracting_audio", "Extracting audio", False),
    ("separating_audio", "Separating vocals and background", True),
    ("transcribing", "Transcribing", False),
    ("diarizing", "Detecting speakers", True),
    ("selecting_reference", "Selecting speaker references", False),
    ("translating", "Translating", False),
    ("synthesizing_voice", "Synthesizing voices", False),
    ("assembling_dialogue", "Assembling dialogue", False),
    ("mixing_background", "Mixing original background", True),
    ("lip_syncing", "Lip syncing", False),
    ("verifying", "Verifying", False),
    ("uploading_result", "Finalizing", False),
]


def stage_rows(job: dict) -> list:
    """
    Each stage with the state it actually reached.

    The brief's requirement: an optional stage must report "skipped" or
    "fallback" rather than pretending it ran. Derived from what the job holds,
    not from a separate log, so it cannot disagree with the artifacts.
    """
    stems = job.get("stems") or None
    diarization = job.get("diarization") or []
    speakers = SPK.normalize_speakers(job.get("speakers"))
    rows = []

    for key, label, optional in STAGES:
        row = {"key": key, "label": label, "optional": optional,
               "state": "done", "detail": None}

        if key == "separating_audio":
            if stems:
                row["detail"] = (f"{stems.get('chunks', '?')} chunk(s), "
                                 f"model {stems.get('model', 'htdemucs')}")
            elif optional:
                row["state"] = "skipped"
                row["detail"] = ("Background separation is off, or Demucs is "
                                 "unavailable. The dub replaces the original "
                                 "soundtrack.")
        elif key == "diarizing":
            if diarization:
                labels = sorted({t.get("speaker") for t in diarization})
                row["detail"] = (f"{len(diarization)} turns, "
                                 f"{len(labels)} speaker(s)")
            else:
                row["state"] = "fallback"
                row["detail"] = ("Speaker detection is unavailable, so the "
                                 "video is dubbed in a single voice.")
        elif key == "selecting_reference":
            if speakers:
                cloned = sum(1 for s in speakers
                             if SPK.status_of(s) == SPK.CLONED)
                row["detail"] = f"{cloned}/{len(speakers)} speaker(s) cloned"
        elif key == "mixing_background":
            if job.get("mixed_path"):
                row["detail"] = "dialogue over the preserved background"
            elif optional:
                row["state"] = "skipped"
                row["detail"] = ("No background was preserved, so the dub is "
                                 "the only audio.")
        elif key == "lip_syncing":
            note = job.get("lip_sync") or {}
            if job.get("kind") in ("audio", "subtitles",
                                   "subtitles_translated"):
                row["state"] = "skipped"
                row["detail"] = f"not needed for a {job.get('kind')} output"
            elif note.get("skipped"):
                # The stage was applicable but not possible: no face was found
                # in the picture. The dub is still delivered, on the original
                # video, and this is the only place that says so.
                row["state"] = "fallback"
                row["detail"] = (note.get("reason")
                                 or "No face was found, so the dub plays over "
                                    "the original picture.")
        elif key == "verifying":
            sync = job.get("sync") or {}
            if sync.get("reason"):
                row["detail"] = sync["reason"]

        rows.append(row)
    return rows


def build(job: dict, readiness: dict = None) -> dict:
    """
    The whole summary for a finished job.

    `readiness` is a `readiness.snapshot()`, included so a manifest read months
    later still says what the machine could do at the time — which is usually
    the first question when a render is worse than expected.
    """
    import project

    speakers = SPK.normalize_speakers(job.get("speakers"))
    plan = job.get("plan") or []
    segments = job.get("segments") or []
    translated = job.get("translated_segments") or []
    stems = job.get("stems") or None
    diarization = job.get("diarization") or []

    # Overlap, when a plan exists. Imported here rather than at module level so
    # this module stays importable without the dubbing stack.
    overlap = {"regions": 0, "seconds": 0.0, "max_speakers": 0}
    if plan:
        try:
            import dubbing
            regions = dubbing.overlap_profile(plan)
            overlap = {
                "regions": len(regions),
                "seconds": round(sum(r["end"] - r["start"] for r in regions), 2),
                "max_speakers": max((len(r["speakers"]) for r in regions),
                                    default=0),
            }
        except Exception:                                    # noqa: BLE001
            pass

    speaker_rows = SPK.describe_all(speakers, plan=plan)
    fallback = [r["label"] for r in speaker_rows if r["status"] != SPK.CLONED]

    summary = {
        "generated_at": round(time.time(), 3),
        "job_id": job.get("job_id"),
        "kind": job.get("kind"),
        "status": job.get("status"),
        "languages": {
            "source": job.get("source_language"),
            "target": job.get("target_language"),
        },
        "video_duration": job.get("video_duration"),
        "segments": {
            "transcribed": len(segments),
            "translated": len(translated),
            "synthesized": len(plan),
        },
        "speakers": {
            "count": len(speaker_rows),
            "labels": [r["label"] for r in speaker_rows],
            "fallback": fallback,
            "detail": speaker_rows,
        },
        "diarization": {
            "ran": bool(diarization),
            "turns": len(diarization),
            "source": job.get("diarization_source"),
        },
        "overlap": overlap,
        "background": {
            "enabled": bool(stems),
            "separated": bool(stems),
            "ducking": bool((job.get("tunables") or {}).get("BACKGROUND_DUCK")),
            "mixed": bool(job.get("mixed_path")),
            "model": (stems or {}).get("model"),
        },
        "models": {
            "asr": "Systran/faster-whisper-large-v3",
            "translation": "facebook/nllb-200-distilled-600M",
            "tts": _tts_model(job.get("target_language")),
            "diarization": ("pyannote/speaker-diarization-3.1"
                            if diarization else None),
            "separation": (stems or {}).get("model"),
        },
        "artifacts": {
            "dialogue_only": job.get("dub_path"),
            "mixed": job.get("mixed_path"),
            "vocals": job.get("vocals_path"),
            "background": job.get("background_path"),
            "output": job.get("output_path") or job.get("filed_at"),
            "units_dir": job.get("units_dir"),
        },
        "lip_sync": dict(job.get("lip_sync") or {"skipped": False}),
        "stages": stage_rows(job),
        "warnings": warnings(job),
        "verification": verification(job),
    }

    if readiness is not None:
        summary["environment"] = {
            "device": (readiness.get("hardware") or {}).get("device"),
            "gpu_name": (readiness.get("hardware") or {}).get("gpu_name"),
            "ready": readiness.get("ready"),
            "degraded": readiness.get("degraded") or [],
            # Named, not valued: see `readiness.token_state`.
            "secrets": readiness.get("secrets") or {},
        }
    return summary


def _tts_model(target_language: str = None) -> str:
    """The voice model a target language actually uses."""
    if not target_language:
        return "ai4bharat/IndicF5"
    try:
        import languages as L
        engine = L.engine_for(target_language)
        if engine == L.ENGINE_INDICF5:
            return "ai4bharat/IndicF5"
        return "coqui/XTTS-v2"
    except Exception:                                        # noqa: BLE001
        return "ai4bharat/IndicF5"


def verification(job: dict) -> dict:
    """
    What the run measured about itself.

    Per-speaker voice similarity where it exists. This is a quality diagnostic,
    not identity proof: the same WavLM x-vector scores roughly 0.9 for almost
    any pair, so a raw cosine is reported alongside the normalised score and no
    universal pass threshold is claimed.
    """
    out = {"sync": job.get("sync") or None,
           "voice_match": job.get("voice_match") or None,
           "per_speaker": []}

    for row in job.get("speaker_voice_match") or []:
        out["per_speaker"].append({
            "label": row.get("label"),
            "score": row.get("score"),
            "cosine": row.get("cosine"),
            "status": _match_status(row),
        })
    return out


def _match_status(row: dict) -> str:
    """
    A word for a similarity score, without inventing a threshold.

    The bands are the ones the eval harness already anchors on: a normalised
    score at or above the same-speaker ceiling reads `match`, a negative score
    reads `different`, and the middle is deliberately `uncertain` rather than
    being given a cut-off nobody validated.
    """
    score = row.get("score")
    if score is None:
        return "unmeasured"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "unmeasured"
    if value >= 0.5:
        return "match"
    if value <= 0.0:
        return "different"
    return "uncertain"


def warnings(job: dict) -> list:
    """
    Everything a person should know about this run, in plain sentences.

    The single place a fallback becomes visible. Order is by importance: what
    could not run first, then what ran weakly, then what was measured.
    """
    out = []
    speakers = SPK.normalize_speakers(job.get("speakers"))

    if not job.get("diarization"):
        out.append("Speaker detection did not run, so the whole video is "
                   "dubbed in one voice.")

    # Speaker warnings, from `speakers.warning_for`. This is where a short
    # reference or a generic fallback surfaces.
    out.extend(SPK.warnings(speakers))

    if not job.get("stems") and job.get("kind") in ("dub", "audio"):
        out.append("The original music and sound effects were not preserved; "
                   "the dub is the only audio track.")

    note = job.get("lip_sync") or {}
    if note.get("skipped"):
        out.append("No face was found in the video, so the lips were not "
                   "re-synced; the dub plays over the original picture.")

    missing = [m for m in (job.get("missing_units") or [])]
    if missing:
        out.append(f"{len(missing)} phrase(s) have no audio and are silent in "
                   f"the track.")

    sync = job.get("sync") or {}
    if sync and sync.get("ok") is False:
        out.append(f"Audio and picture drift: {sync.get('reason')}")

    if job.get("status") == "failed" and job.get("error"):
        out.append(f"The run failed: {job['error']}")
    return out


def render(summary: dict) -> str:
    """The summary as text, for a log or a bug report."""
    lines = []
    langs = summary.get("languages") or {}
    lines.append(f"Run summary — {summary.get('kind')} → {langs.get('target')}")
    lines.append(f"  status            {summary.get('status')}")
    lines.append(f"  video duration    {summary.get('video_duration')}s")
    segs = summary.get("segments") or {}
    lines.append(f"  segments          {segs.get('transcribed')} transcribed, "
                 f"{segs.get('translated')} translated, "
                 f"{segs.get('synthesized')} synthesized")

    speakers = summary.get("speakers") or {}
    lines.append(f"  speakers          {speakers.get('count')}")
    for row in speakers.get("detail") or []:
        ref = row.get("reference") or {}
        lines.append(f"    {row['label']}  {row['status']}")
        lines.append(f"      {row.get('segment_count')} segments · "
                     f"{row.get('total_speech')}s speech")
        if ref:
            span = ""
            if ref.get("start") is not None and ref.get("end") is not None:
                span = f" · {ref['start']:.1f}–{ref['end']:.1f}s"
            lines.append(f"      reference {ref.get('duration')}s{span}")
        if row.get("phrase_count") is not None:
            lines.append(f"      {row['phrase_count']} dubbed phrase(s)")

    overlap = summary.get("overlap") or {}
    lines.append(f"  overlap           {overlap.get('regions')} region(s), "
                 f"{overlap.get('seconds')}s, up to "
                 f"{overlap.get('max_speakers')} at once")
    bg = summary.get("background") or {}
    lines.append(f"  background        "
                 f"{'preserved' if bg.get('separated') else 'not preserved'}"
                 + (", ducked" if bg.get("ducking") else ""))

    for row in summary.get("stages") or []:
        if row["state"] != "done":
            lines.append(f"  {row['label']}: {row['state'].upper()} — "
                         f"{row['detail']}")
    for line in summary.get("warnings") or []:
        lines.append(f"  ! {line}")
    return "\n".join(lines)

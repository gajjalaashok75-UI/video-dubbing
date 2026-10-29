"""
Project persistence.

Everything needed to reopen a finished dub already survives on disk: the
source clip, the reference, the per-phrase cache, the dub track and the
rendered video. What did not survive was the index — the plan, the reference
metadata and the paths lived in a module-level dict, so restarting the server
turned every editable project into a download link.

This writes that index next to the files it describes. A project is a folder
with a `project.json` in it; nothing else about the layout changes.
"""

from __future__ import annotations

import glob
import json
import os
import time
from datetime import datetime, timezone

MANIFEST = "project.json"
VERSION = 1

# Job keys worth persisting. Deliberately explicit: the job dict also holds
# transient things (cancel flags, progress strings) that must not come back.
PERSISTED = (
    "kind", "status", "step", "filename", "title", "project_id",
    "user_id", "video_id", "url", "output_path", "dub_path",
    "reference_path", "reference_text", "reference_seconds",
    "source_language", "target_language", "segment_count",
    "duration", "video_duration", "input_path", "workdir", "units_dir",
    "plan", "segments", "word_segments", "translated_segments",
    "translated_text", "original_text",
    "reference", "sync", "voice_match", "video_stale", "history",
    "filed_at", "filed_error", "subtitle_mode", "subtitle_language",
    "music_path", "music_gain", "voiceover_seconds",
    # The speaker registry and the diarization it came from. Persisted so a
    # reopened project still knows who spoke what, and so a sibling output of
    # the same clip can reuse the diarization instead of running it again.
    "speakers", "diarization", "diarization_source",
    # Phase 4: the separated stems, so a sibling output of the same clip
    # reuses them instead of spending minutes in Demucs again.
    "stems",
    # Phase 6: display names, and the per-speaker voice scores.
    "speaker_aliases", "speaker_voice_match",
)


def _plain(value):
    """
    Make a value JSON-safe.

    Whisper hands back numpy scalars inside its segment dicts, which json
    refuses. Converting on write keeps the manifest readable by anything.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    # numpy scalars and anything else with a python equivalent
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except Exception:                                    # noqa: BLE001
            pass
    return str(value)


def save(job_id: str, job: dict) -> str | None:
    """Write the manifest for a job. Never raises: losing the index is bad,
    but failing a finished render because of it would be worse."""
    workdir = job.get("workdir")
    if not workdir or not os.path.isdir(workdir):
        return None
    payload = {
        "version": VERSION,
        "job_id": job_id,
        "saved_at": time.time(),
        **{k: _plain(job[k]) for k in PERSISTED if k in job},
    }
    path = os.path.join(workdir, MANIFEST)
    try:
        # Write beside the target and rename, so an interrupted write cannot
        # leave a half-parsed manifest behind.
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        return path
    except Exception as e:                                   # noqa: BLE001
        print(f"[PROJECT] Could not save {job_id}: {e}")
        return None


def load(path: str) -> dict | None:
    """Read one manifest, or None if it is unreadable or its files are gone."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:                                   # noqa: BLE001
        print(f"[PROJECT] Skipping unreadable manifest {path}: {e}")
        return None

    if data.get("version") != VERSION:
        print(f"[PROJECT] Skipping {path}: manifest version "
              f"{data.get('version')}, expected {VERSION}")
        return None

    # A manifest describing files that no longer exist is worse than no
    # manifest, because it puts a project in the list that cannot be opened.
    for key in ("dub_path", "input_path"):
        target = data.get(key)
        if target and not os.path.exists(target):
            print(f"[PROJECT] Skipping {data.get('job_id')}: missing {key}")
            return None

    return data


def scan(root: str) -> dict:
    """
    Every project under `root`, keyed by job id.

    Only manifests are read; nothing here touches audio or video, so this stays
    cheap enough to run at startup.
    """
    found = {}
    for path in sorted(glob.glob(os.path.join(root, "*", MANIFEST))):
        data = load(path)
        if not data:
            continue
        job_id = data.get("job_id")
        if not job_id:
            continue
        data.pop("version", None)
        data.pop("job_id", None)
        # Kept as a creation date for projects the database never recorded.
        data["saved_at"] = data.pop("saved_at", None)
        # A project restored from disk is finished by definition; a run that
        # was interrupted mid-render has nothing to resume.
        if data.get("status") != "done":
            continue
        found[job_id] = data
    return found


def video_rows(jobs: dict, name_of=None) -> list:
    """
    One row per project, for the project list.

    A project used to appear only once it had a finished render, so a run in
    progress was invisible: start a long dub, go back to the list, and the
    work you just started is not there — and if the runtime dies there is
    nothing left to say it ever existed. Every project is listed now, with
    the status it is actually in.

    `name_of(job, job_id)` names a row; the caller supplies it because naming
    falls back to the source file, which is the app's concern, not this
    module's.
    """
    name_of = name_of or (lambda job, job_id: job.get("title") or job_id)

    by_project: dict = {}
    for job_id, job in jobs.items():
        if job.get("status") not in ("queued", "processing", "done", "failed"):
            continue
        pid = job.get("project_id", job_id)
        by_project.setdefault(pid, []).append((job_id, job))

    rows = []
    for pid, group in by_project.items():
        done = [(jid, j) for jid, j in group if j.get("status") == "done"]
        live = [(jid, j) for jid, j in group
                if j.get("status") in ("queued", "processing")]
        done.sort(key=lambda pair: pair[1].get("saved_at") or 0)
        if live:
            # While something is running, that is what the row is about,
            # even when the project already holds finished outputs.
            live.sort(key=lambda pair: pair[1].get("started_at") or 0,
                      reverse=True)
            job_id, job = live[0]
        elif done:
            job_id, job = done[0]
        else:
            job_id, job = group[0]

        status = job.get("status")
        created = job.get("saved_at") or job.get("started_at")
        rows.append({
            "id": pid,
            "job_id": job_id,
            "project_id": pid,
            "outputs": len(done),
            "openable": True,
            "title": name_of(job, job_id),
            "name": name_of(job, job_id),
            "original_language": (job.get("source_language")
                                 or job.get("original_language")),
            "target_language": job.get("target_language"),
            "duration": job.get("duration"),
            "processing_status": status,
            "dubbed_url": job.get("url"),
            "error_message": job.get("error") if status == "failed" else None,
            "created_at": (datetime.fromtimestamp(created, timezone.utc).isoformat()
                          if created else None),
        })

    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows

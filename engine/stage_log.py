"""
Per-step results, written to the runtime logs so a run can be read back.

A dub is a long chain of stages — probe, extract, transcribe, diarize,
separate, reference, translate, synthesize, assemble, lip sync, mix, verify —
and when the finished video is wrong the question is always "which stage
produced this?". The server log answers it only if you were watching, and it
interleaves every stage into one stream. This module gives each run its own
file with one JSON record per stage:

    <runtime logs>/stages/<job_id>.jsonl

One JSON object per line, appended as each stage finishes, so a run that is
still going can be tailed and a run that crashed has everything up to the
crash. A second, human-readable file beside it:

    <runtime logs>/stages/<job_id>.log

carries the same events as indented lines for reading in the notebook without
parsing JSON.

Nothing here may fail a render. Every write is wrapped: if the log directory
cannot be created or the disk is full, the record is dropped and the run
continues. The log is a diagnostic, never a dependency.
"""

from __future__ import annotations

import json
import os
import threading
import time

_LOCK = threading.Lock()
_DIR = None


def _log_dir() -> str | None:
    """
    Where the per-step logs live, resolved once.

    Prefers the deployment's own runtime layout (`deploy_helpers.runtime_dirs`,
    which the notebook configures before importing the app) so the logs land
    beside `server.log` where the user is already told to look. Falls back to a
    `logs` directory next to the upload directory, then to the upload directory
    itself, so the logs are still written on a machine that never ran the
    notebook setup.
    """
    global _DIR
    if _DIR is not None:
        return _DIR or None
    override = os.getenv("VIDEOTRANS_LOG_DIR")
    if override:
        _DIR = os.path.join(override, "stages")
    else:
        base = None
        try:
            import deploy_helpers
            base = str(deploy_helpers.runtime_dirs()["logs"])
        except Exception:                                    # noqa: BLE001
            pass
        if base is None:
            uploads = os.getenv("VIDEOTRANS_UPLOAD_DIR")
            if uploads:
                base = os.path.join(os.path.dirname(uploads), "logs")
        if base is None:
            base = os.path.join(os.getcwd(), "logs")
        _DIR = os.path.join(base, "stages")
    try:
        os.makedirs(_DIR, exist_ok=True)
    except Exception:                                        # noqa: BLE001
        _DIR = ""
    return _DIR or None


def _jsonable(value):
    """Best-effort JSON: a value that will not serialise becomes its repr."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def stage(job_id: str, name: str, **fields):
    """
    Record one stage's result. `name` is the stage, `fields` its findings.

    Call it once per stage with everything that stage decided — counts, chosen
    paths, detected speakers, measured durations, any warning. The point is
    that reading the file alone tells you what each stage produced, without
    re-running the pipeline.
    """
    record = {"t": round(time.time(), 3), "job": job_id, "stage": name}
    for key, value in fields.items():
        record[key] = _jsonable(value)
    _write(job_id, record)


def event(job_id: str, stage_name: str, message: str, **fields):
    """A sub-event inside a stage — a decision, a warning, a fallback."""
    record = {"t": round(time.time(), 3), "job": job_id, "stage": stage_name,
              "event": message}
    for key, value in fields.items():
        record[key] = _jsonable(value)
    _write(job_id, record)


def _write(job_id: str, record: dict) -> None:
    """Append one record to the run's JSONL and readable logs. Never raises."""
    directory = _log_dir()
    if not directory:
        return
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(job_id))
    jsonl = os.path.join(directory, f"{safe}.jsonl")
    human = os.path.join(directory, f"{safe}.log")
    try:
        with _LOCK:
            with open(jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open(human, "a", encoding="utf-8") as fh:
                fh.write(_human(record) + "\n")
    except Exception:                                        # noqa: BLE001
        pass


def _human(record: dict) -> str:
    """One record as an indented, greppable line."""
    stage_name = record.get("stage", "?")
    when = time.strftime("%H:%M:%S", time.localtime(record.get("t", 0)))
    head = f"[{when}] {stage_name}"
    if "event" in record:
        head += f" :: {record['event']}"
    extras = {k: v for k, v in record.items()
              if k not in ("t", "job", "stage", "event")}
    if not extras:
        return head
    lines = [head]
    for key, value in extras.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"      {key}: {value}")
    return "\n".join(lines)


def path_for(job_id: str) -> str | None:
    """The JSONL path for a run, for the API to report. None if unwritable."""
    directory = _log_dir()
    if not directory:
        return None
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(job_id))
    return os.path.join(directory, f"{safe}.jsonl")


def read(job_id: str) -> list:
    """The records written so far for a run, oldest first. [] if none."""
    target = path_for(job_id)
    if not target or not os.path.isfile(target):
        return []
    records = []
    try:
        with open(target, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    except Exception:                                        # noqa: BLE001
        return records
    return records

import asyncio
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure engine/ and the repository root are in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
for _p in (str(_CURRENT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Before anything reads an environment variable: a key in .env is worthless
# if it is loaded after the module that wanted it.
import localenv
_ENV_KEYS = localenv.load(str(_PROJECT_ROOT / ".env"))

from fastapi import (BackgroundTasks, Body, FastAPI, File, Form, HTTPException,
                     Query, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# There is no database and no object store. Everything a project needs lives in
# its own folder on this machine, indexed by the manifest project.py writes,
# and renders are served straight off disk. Nothing is uploaded anywhere.

# ---------------------------------------------------------------------------
# Pipeline modules
# ---------------------------------------------------------------------------
import numpy as np

import av_sync
import dubbing
import engines
import models
import llm
import project
import subtitles
import voiceover
import artifact_cache
import languages as L
import readiness
import reference_audio
import run_summary
import source_separation
import speaker_diarization
import speakers as SPK
import tts_engines
# Imported as modules as well as functions: applying a model change without a
# restart means calling refresh() on the module that owns each stage.
import lip_sync
import speech_to_text_v2
import translation_v2
from lip_sync import generate_lip_synced_video
from speech_to_text_v2 import transcribe_audio
from translation_v2 import translate_segments

jobs: dict = {}

# Where a run keeps its uploads, per-project folders and intermediate audio.
# The default sits beside the code, which is right for a desktop install; a
# hosted deployment (Colab, Kaggle, Docker) points VIDEOTRANS_UPLOAD_DIR at a
# writable volume instead, because the checkout is not durable there.
UPLOAD_DIR = os.path.abspath(
    os.getenv("VIDEOTRANS_UPLOAD_DIR") or os.path.join(_PROJECT_ROOT, "temp_uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# The heavy stages (TTS + Wav2Lip) share one GPU. Running two jobs through them
# at once thrashes MPS and makes both slower than running them back to back.
_HEAVY = threading.Semaphore(int(os.getenv("PIPELINE_CONCURRENCY", 1)))
BUSY_DETAIL = ("The engine is busy with a render. This edit needs the same "
               "model, so try again once the render finishes.")

KEEP_INTERMEDIATE = os.getenv("KEEP_INTERMEDIATE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Cleanup task
# ---------------------------------------------------------------------------
# How long an abandoned working directory survives. Saved projects are never
# swept: they carry a manifest and the user deletes them from the studio.


def sweep_workdirs() -> int:
    """
    Remove working directories left by runs that never became projects.

    Removing the cloud integration also removed the only thing that pruned this
    directory, so failed and cancelled runs accumulated with nothing to clear
    them. A directory is only swept when it has no manifest, is not a live job,
    and has not been touched recently.
    """
    if not os.path.isdir(UPLOAD_DIR):
        return 0
    cutoff = time.time() - engines.tunable("VIDEOTRANS_SWEEP_HOURS") * 3600
    live = {j.get("workdir") for j in jobs.values()}
    freed = 0

    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isdir(path):
                if not name.startswith("job_"):
                    continue
                # A manifest means this is a project someone can still open.
                if os.path.exists(os.path.join(path, project.MANIFEST)):
                    continue
                if path in live:
                    continue
                # An empty directory holds nothing and belongs to no project,
                # so it goes now rather than in two days. A task cancelled
                # while its files were being deleted can recreate one, and a
                # deleted project should not leave a trace behind it.
                if os.path.getmtime(path) > cutoff and os.listdir(path):
                    continue
                shutil.rmtree(path)
                freed += 1
            elif os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                # An upload whose job directory is gone has nothing left to
                # belong to.
                job_id = name.split("_", 1)[0]
                if job_id in jobs or os.path.isdir(os.path.join(UPLOAD_DIR, f"job_{job_id}")):
                    continue
                os.remove(path)
                freed += 1
        except OSError as e:
            print(f"[SWEEP] Could not remove {name}: {e}")

    if freed:
        print(f"[SWEEP] Removed {freed} abandoned item(s) from {UPLOAD_DIR}")
    return freed


async def sweep_loop():
    while True:
        try:
            sweep_workdirs()
        except Exception as e:                               # noqa: BLE001
            print(f"[SWEEP] Error: {e}")
        await asyncio.sleep(6 * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    restored = project.scan(UPLOAD_DIR)
    if restored:
        jobs.update(restored)
        print(f"[APP] Reopened {len(restored)} saved project(s)")
    sweeper = asyncio.create_task(sweep_loop())
    if os.getenv("WARMUP_ON_START", "1") not in ("0", "false", "no"):
        threading.Thread(target=_warmup, daemon=True).start()
    yield
    sweeper.cancel()


def _speaker_voice_matches(speakers: list, plan: list, units_dir: str,
                           reference_path: str, source_path: str,
                           workdir: str) -> list:
    """
    Per-speaker voice similarity, using the existing WavLM infrastructure.

    The old single score compared one reference against the whole dub, which
    stopped meaning anything the moment a video had two voices: a good clone of
    speaker A and a bad one of B average into a number that describes neither.

    So each speaker's reference is compared against a track built from THAT
    speaker's own generated phrases. Both sides are normalised the same way the
    single-voice score already was — against a same-speaker ceiling and an
    unrelated-speaker floor — because a raw WavLM cosine sits near 0.9 for
    almost any pair and is unreadable on its own.

    Deliberately reports a status word rather than a pass/fail: this is a
    quality diagnostic, not identity proof, and no threshold here has been
    validated across enough voices to claim one.

    Best effort throughout. A scoring failure must never fail a render that
    otherwise succeeded.
    """
    if os.getenv("VOICE_MATCH", "1") in ("0", "false", "no"):
        return []
    if not speakers or not plan:
        return []

    try:
        import soundfile as sf
        from eval_harness import SpeakerScorer
        scorer = SpeakerScorer()
    except Exception as e:                                   # noqa: BLE001
        print(f"[APP] Per-speaker voice match unavailable: {e}")
        return []

    # Group the cached phrases by speaker, and write one track per speaker.
    by_speaker: dict = {}
    for unit in plan:
        wave = dubbing.read_unit(units_dir, unit["index"])
        if wave is None or wave.size == 0:
            continue
        by_speaker.setdefault(unit.get("speaker") or SPK.DEFAULT_LABEL,
                              []).append(wave)
    if not by_speaker:
        return []

    tracks_dir = os.path.join(workdir, "speakers")
    os.makedirs(tracks_dir, exist_ok=True)

    results = []
    for entry in speakers:
        label = entry.get("label")
        chunks = by_speaker.get(label) or []
        ref = entry.get("reference") or {}
        row = {"label": label, "score": None, "cosine": None,
               "status": "unmeasured", "phrases": len(chunks),
               "track": None}
        if not chunks or not ref.get("path"):
            results.append(row)
            continue

        # The isolated track is written whether or not scoring works: it is
        # worth having for debugging either way.
        track = os.path.join(tracks_dir, f"{label}.wav")
        try:
            joined = np.concatenate(chunks)
            sf.write(track, joined, dubbing.SAMPLE_RATE, subtype="PCM_16")
            row["track"] = track
        except Exception as e:                               # noqa: BLE001
            print(f"[APP] Could not write the {label} track: {e}")
            results.append(row)
            continue

        try:
            ref_embedding = scorer.embed(ref["path"])
            dub_embedding = scorer.embed(track)
            if ref_embedding is None or dub_embedding is None:
                results.append(row)
                continue
            raw = scorer.cosine(ref_embedding, dub_embedding)
            ceiling = scorer.cosine(ref_embedding, scorer.embed(source_path))
            floor = scorer.floor(ref["path"])
            import math
            if any(math.isnan(v) for v in (raw, ceiling, floor)):
                results.append(row)
                continue
            row["cosine"] = round(raw, 4)
            if ceiling > floor:
                norm = (raw - floor) / (ceiling - floor)
                row["score"] = round(max(0.0, min(1.0, norm)), 3)
            # `run_summary` turns the score into a word; the vocabulary lives
            # there so the manifest and the studio cannot disagree.
            import run_summary
            row["status"] = run_summary._match_status(row)
            print(f"[APP] {label} voice match: "
                  f"{row['score'] if row['score'] is not None else 'n/a'} "
                  f"(cosine {row['cosine']}, {row['phrases']} phrases)")
        except Exception as e:                               # noqa: BLE001
            print(f"[APP] {label} voice match failed: {e}")
        results.append(row)

    return results


def _voice_match(ref_path: str, dub_path: str, source_path: str):
    """
    How close the dub sounds to the reference speaker, on a 0..1 scale.

    A bare WavLM cosine is not reportable: this model returns roughly 0.9 for
    almost any pair, so the raw number looks excellent even for a failed clone.
    eval_harness already solved this by anchoring against two references, and
    the same anchors are available here for free:

        ceiling  reference vs the speaker's own source audio
        floor    reference vs unrelated speakers (the IndicF5 prompt clips)

    The reported value is where the dub falls between them. Best effort: a
    scoring failure must never fail a render that otherwise succeeded.
    """
    if os.getenv("VOICE_MATCH", "1") in ("0", "false", "no"):
        return None
    try:
        from eval_harness import SpeakerScorer
        scorer = SpeakerScorer()
        ref = scorer.embed(ref_path)
        dub = scorer.embed(dub_path)
        if ref is None or dub is None:
            return None
        raw = scorer.cosine(ref, dub)
        ceiling = scorer.cosine(ref, scorer.embed(source_path))
        floor = scorer.floor(ref_path)
        import math
        if any(math.isnan(v) for v in (raw, ceiling, floor)) or ceiling <= floor:
            return None
        norm = (raw - floor) / (ceiling - floor)
        return {"score": round(max(0.0, min(1.0, norm)), 3),
                "cosine": round(raw, 4)}
    except Exception as e:                                   # noqa: BLE001
        print(f"[APP] Voice match unavailable: {e}")
        return None


# What the splash screen reports while the window is still closed. These are
# the real model loads, in the order they happen, not a progress animation.
BOOT = {"stage": "Starting", "index": 0, "total": 4, "ready": False, "notes": []}


def _boot(stage: str, index: int):
    BOOT["stage"] = stage
    BOOT["index"] = index
    print(f"[BOOT] {index}/{BOOT['total']} {stage}")


def _warmup():
    """
    Preload weights so the first render does not pay the load cost.

    Everything here used to load lazily on first use, which meant the first
    dub of a session was several minutes slower than the rest for reasons the
    user could not see. Loading it up front costs the same time somewhere the
    user is already waiting, and gives the splash something true to report.
    """
    steps = [
        ("Loading voice model", _warm_voice),
        ("Loading speech recognition", _warm_asr),
        ("Loading translator", _warm_translator),
        ("Loading lip sync", _warm_lipsync),
    ]
    BOOT["total"] = len(steps)
    for i, (label, fn) in enumerate(steps, 1):
        _boot(label, i)
        try:
            fn()
        except Exception as e:                               # noqa: BLE001
            # A component that will not load is not fatal at boot; the stage
            # that needs it will fail with a real message. Record it so the
            # splash can say so rather than claiming everything is ready.
            print(f"[BOOT] {label} failed: {e}")
            BOOT["notes"].append(_boot_note(label, e))
    BOOT["stage"] = "Ready"
    BOOT["ready"] = True
    print("[BOOT] Warmup complete")


def _boot_note(label: str, exc: Exception) -> str:
    """A note the splash can act on, not just the exception's first line."""
    text = str(exc)
    if "gated" in text.lower() or "401" in text:
        return ("Voice model not downloaded - it is gated. Accept the licence at "
                "https://huggingface.co/ai4bharat/IndicF5, set HF_TOKEN in .env, "
                "then download it (see docs/GATED-MODELS.md).")
    return f"{label} unavailable"


def _warm_voice():
    """
    Preload the voice model, and let a real failure reach the splash.

    tts_engines.warmup() swallows the exception and returns False, which is
    right for a render but wrong here: a gated model that was never downloaded
    is exactly the thing the splash has to say, and a bare False left BOOT
    claiming ready with no note. Only a genuine load failure is re-raised; a
    missing optional engine is not one.
    """
    import tts_engines
    try:
        tts_engines.get_engine(os.getenv("WARMUP_LANG", "hi")).load()
    except Exception as exc:                                 # noqa: BLE001
        if tts_engines._indicf5_present():
            raise
        raise RuntimeError(f"gated model not downloaded: {exc}") from exc


def _warm_asr():
    from speech_to_text_v2 import _load_model
    _load_model()


def _warm_translator():
    from translation_v2 import _load_model
    _load_model()


def _warm_lipsync():
    import lip_sync
    lip_sync.warmup()


app = FastAPI(title="VideoTrans.Ai — Indian-language Voice Cloning API",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


class JobCancelled(Exception):
    """Raised inside the pipeline when the user has asked it to stop."""


def _safe_name(name: str) -> str:
    """
    Reduce an uploaded filename to something that cannot leave UPLOAD_DIR.

    Only spaces were replaced before, so "../../../tmp/x.mp4" joined onto the
    uploads directory walked straight out of it: the job-id prefix does not
    help once a later component is "..". Take the basename, drop separators and
    leading dots, and keep the result to a length every filesystem accepts.
    """
    base = os.path.basename(str(name or "")).replace("\\", "_")
    base = re.sub(r"[/\x00]", "_", base)
    base = base.lstrip(". ").strip() or "upload"
    # Leave room for the job-id prefix and the suffixes the pipeline appends.
    root, ext = os.path.splitext(base)
    return (root[:80] or "upload") + ext[:12]


def _save(job_id: str) -> None:
    """Persist a job if it is still there. A project deleted mid-render is a
    reason to stop writing, not to raise."""
    job = jobs.get(job_id)
    if job is not None:
        project.save(job_id, job)


def _set(job_id: str, **kw):
    job = jobs.get(job_id)
    if job is None:
        return
    job.update(kw)
    # Every step report doubles as a cancellation checkpoint. The two longest
    # stages report per segment and per frame batch, so a cancel lands within
    # seconds instead of at the end of the run. Terminal updates carry a
    # `status` and must never raise.
    if "step" in kw and "status" not in kw and job.get("cancel_requested"):
        raise JobCancelled()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def process_video_task(job_id: str, input_path: str, original_language: str,
                       target_language: str, filename: str,
                       user_id: str = "anonymous",
                       trim_start: float = None, trim_end: float = None,
                       music_path: str = None, music_gain: float = -18.0,
                       kind: str = "dub", preserve_background: bool = None,
                       background_lufs: float = None,
                       background_duck: bool = None):
    """
    Segment-level dubbing pipeline.

    The dubbed track is built to exactly the video's duration (see `dubbing`),
    so lip sync receives matched streams and the output cannot drift, loop or
    truncate. There is no post-hoc tempo correction anywhere in this function.
    """
    workdir = os.path.join(UPLOAD_DIR, f"job_{job_id}")
    os.makedirs(workdir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(filename))[0]

    try:
        _set(job_id, status="processing", step="validating")

        if not os.path.exists(input_path) or os.path.getsize(input_path) < 1000:
            raise ValueError("Uploaded file is missing or too small")

        info = av_sync.probe(input_path)
        if not info["video"]:
            raise ValueError("Uploaded file has no video stream")
        video_dur = info["duration"]

        # In and out points, applied before anything else looks at the file, so
        # transcription, reference selection and the timeline all agree on what
        # "the clip" is.
        if trim_start is not None or trim_end is not None:
            a = max(0.0, float(trim_start or 0.0))
            b = min(video_dur, float(trim_end if trim_end is not None else video_dur))
            if b - a < 1.0:
                raise ValueError(
                    f"The trimmed range is only {max(0.0, b - a):.2f}s; "
                    f"at least a second is needed.")
            if a > 0.0 or b < video_dur - 0.01:
                trimmed = os.path.join(workdir, f"{stem}_trimmed.mp4")
                av_sync.trim(input_path, trimmed, a, b)
                input_path = trimmed
                video_dur = av_sync.duration(trimmed)
                print(f"[APP] Trimmed to {a:.2f}s-{b:.2f}s ({video_dur:.2f}s)")
                _set(job_id, duration=format_duration(video_dur))
        _set(job_id, duration=format_duration(video_dur))

        # Validate the target up front — a clear error beats bad audio.
        try:
            target_language = L.normalize(target_language)
            L.engine_for(target_language)
        except L.UnsupportedLanguage as e:
            raise ValueError(str(e))

        print(f"[APP] {filename}: {video_dur:.2f}s, "
              f"{info['video']['width']}x{info['video']['height']} @ "
              f"{info['video']['fps']:.3f}fps -> {L.display_name(target_language)}")

        # ── 1. Audio for STT (16 kHz mono is what Whisper wants) ──
        _set(job_id, step="extracting_audio")
        stt_audio = av_sync.extract_audio(
            input_path, os.path.join(workdir, f"{stem}_stt.wav"), 16000, 1)

        # ── 2. Transcribe with word timestamps ──
        # Or take a sibling's, if this project already transcribed this clip.
        sibling = _reusable_sibling(job_id, video_dur,
                                    original_language if original_language != "auto" else None)
        if sibling and sibling.get("word_segments"):
            segments = list(sibling["word_segments"])
            original_text = sibling.get("original_text") or " ".join(
                (sg.get("text") or "") for sg in segments).strip()
            _set(job_id, step="transcribing", original_text=original_text,
                 reused_from=sibling.get("job_id") or "sibling")
            print(f"[APP] Job {job_id} reused the transcript from a sibling "
                  f"({len(segments)} segments), skipping transcription")
        else:
            sibling = None
            _set(job_id, step="transcribing")
            result = transcribe_audio(stt_audio)
            segments = result.get("segments") or []
            original_text = result.get("text", "")
            _set(job_id, original_text=original_text)
            # A GPU that could not be used leaves the stage on the CPU. Record
            # it so the summary says why the render was slow, instead of the
            # speed-up coming back unexplained next time.
            if speech_to_text_v2.fallback_reason():
                _set(job_id, stt_fallback=speech_to_text_v2.fallback_reason())

        if not segments:
            raise RuntimeError(
                "No speech detected in the video — nothing to dub.")

        # Prefer what the user selected; fall back to detection. The old code
        # ignored the UI value entirely and trusted Whisper, which mis-detects
        # on short or accented audio and then mistranslates the whole video.
        # With a reused transcript there was no detection; the sibling's
        # language is the detection.
        detected = (sibling.get("source_language") if sibling
                    else result.get("language"))
        source_language = original_language or detected
        try:
            source_language = L.normalize(source_language)
        except L.UnsupportedLanguage:
            source_language = L.normalize(detected or "en")
        print(f"[APP] Source language: {source_language} "
              f"(detected {detected}, requested {original_language})")

        # ── 2b. Who is speaking ──
        # Runs on the ORIGINAL audio, not the 16 kHz mono ASR track: resampling
        # and the ASR pipeline's own filtering both smear the cues pyannote
        # keys on, and both `video-dubbing-system` and `Ai-Vedio-Dubbing` found
        # the full-band signal gives better turns. Best effort throughout — no
        # pyannote, no token, a crash, or zero turns all mean one implicit
        # speaker, which is exactly the pipeline this project shipped before
        # diarization existed.
        diarization, diar_source = [], None
        if kind in ("dub", "audio") and _diarization_enabled():
            if sibling and sibling.get("diarization"):
                diarization = list(sibling["diarization"])
                diar_source = "sibling"
                print(f"[APP] Job {job_id} reused {len(diarization)} "
                      f"diarization turn(s) from a sibling")
            else:
                _set(job_id, step="diarizing")
                # Diarization shares the one GPU with the voice model, so it
                # queues behind a render like every other heavy stage, and the
                # model is dropped again before TTS loads.
                with _HEAVY:
                    diarization = speaker_diarization.diarize(
                        input_path,
                        min_speakers=int(engines.tunable("DIAR_MIN_SPEAKERS")),
                        max_speakers=int(engines.tunable("DIAR_MAX_SPEAKERS")))
                speaker_diarization.release()
                diar_source = "detected" if diarization else None
            if diarization:
                segments = speaker_diarization.assign_segments(segments,
                                                               diarization)
                print(speaker_diarization.summarize(segments, diarization))

        seg_rows = [{"start": float(sg.get("start", 0.0)),
                     "end": float(sg.get("end", 0.0)),
                     "text": (sg.get("text") or "").strip(),
                     "speaker": SPK.label_of(sg)}
                    for sg in segments]
        common = dict(workdir=workdir, input_path=input_path,
                      video_duration=video_dur, source_language=source_language,
                      target_language=target_language, segment_count=len(segments),
                      segments=seg_rows, word_segments=segments,
                      # The registry itself is filled in below, once the
                      # per-speaker references have been cut. The diarization
                      # is known now and travels with the job so a sibling
                      # output of the same clip can reuse it.
                      speakers=[], diarization=diarization,
                      diarization_source=diar_source,
                      stems=None,
                      video_stale=False)

        if kind == "subtitles":
            _set(job_id, status="done", step="complete", **common)
            _save(job_id)
            print(f"[APP] Job {job_id} transcribed ({len(segments)} segments)")
            return

        # ── 2c. Separate the film's music and sound effects from its speech ──
        # Optional and failure-safe: no Demucs, no weights, a crash or a bad
        # stem all mean None here and the dialogue-only pipeline below, which is
        # what this project shipped before. The stems give us two things: a bed
        # to put the dub on top of, and a cleaner source for the voice
        # references, since music under a reference clip is contamination the
        # cloning model happily learns.
        #
        # It runs on the ORIGINAL video, not the 16 kHz mono ASR track:
        # separation needs the full band and the stereo image, and the two
        # extractions coexist because ASR still falls back to the 16 kHz one.
        stems = None
        speech_source = input_path
        if kind in ("dub", "audio") and _preserve_background(preserve_background):
            reuse = (sibling.get("stems") if sibling else None) or {}
            if reuse.get("vocals") and os.path.exists(reuse["vocals"]):
                stems = dict(reuse)
                speech_source = stems["vocals"]
                print(f"[APP] Job {job_id} reused the separated stems from a "
                      f"sibling")
            else:
                _set(job_id, step="separating_audio")
                stems = source_separation.separate(
                    input_path, workdir,
                    chunk_seconds=int(engines.tunable("DEMUCS_CHUNK_SECONDS")))
                source_separation.release()
                if stems:
                    speech_source = stems["vocals"]
                print(source_separation.summarize(stems))
        common["stems"] = stems
        common["speech_source"] = speech_source

        # ── 3. Reference clip for cloning ──
        # Only the outputs that actually speak need one. Subtitle jobs were
        # building a voice reference they never used: wasted work, and a way
        # for a subtitle run to fail inside voice-cloning code that has
        # nothing to do with it.
        reference = None
        if kind in ("dub", "audio"):
            _set(job_id, step="selecting_reference")
            # References are cut from the SEPARATED VOCALS when there are any.
            # A reference clip with music under it teaches the cloning model the
            # music, so `speech_source` is the vocals stem whenever separation
            # succeeded and the original video otherwise.
            ref_is_audio = bool(stems) and speech_source == stems.get("vocals")

            # The fallback: the one window the pipeline has always picked, used
            # when there is no diarization and as the last resort otherwise.
            reference = reference_audio.build_reference(
                speech_source, segments, video_dur,
                os.path.join(workdir, f"{stem}_ref.wav"), sample_rate=24000,
                is_audio=ref_is_audio)

            detected_labels = sorted({SPK.label_of(s) for s in segments
                                      if SPK.label_of(s)})
            multi = len(detected_labels) > 1
            if multi:
                # One reference per detected speaker, each cut from that
                # speaker's OWN cleanest windows. A speaker with no usable
                # audio is marked unresolved rather than given another actor's
                # voice.
                common["speakers"] = reference_audio.build_references(
                    speech_source, segments, video_dur,
                    os.path.join(workdir, "refs"), labels=detected_labels,
                    sample_rate=24000, is_audio=ref_is_audio,
                    generic_path=SPK.generic_voice_path())
            else:
                # One speaker, or none detected: the implicit registry, whose
                # single entry points at the global reference. Byte-identical to
                # the pipeline before diarization existed.
                common["speakers"] = SPK.build(
                    segments, SPK.implicit_speaker(reference))

            segments = _assign_implicit_speaker(segments)
            seg_rows = [{"start": float(sg.get("start", 0.0)),
                         "end": float(sg.get("end", 0.0)),
                         "text": (sg.get("text") or "").strip(),
                         "speaker": SPK.label_of(sg)}
                        for sg in segments]
            common["segments"] = seg_rows
            common["word_segments"] = segments
            common["speakers"] = SPK.build(seg_rows, common["speakers"])
            print(SPK.summarize(common["speakers"], seg_rows))

        # ── 4. Translate segment by segment ──
        # Text that is only read goes into the language's own script; text
        # that will be spoken goes into the one the voice model reads. The
        # two differ for Urdu, so a sibling's translation is only reused
        # when it was made for the same purpose.
        for_speech = kind in ("dub", "audio")
        if (sibling and sibling.get("target_language") == target_language
                and sibling.get("translated_segments")
                and len(sibling["translated_segments"]) == len(segments)
                and (L.target_flores(target_language, True)
                     == L.target_flores(target_language, False)
                     or (sibling.get("kind") in ("dub", "audio")) == for_speech)):
            translated = list(sibling["translated_segments"])
            _set(job_id, step="translating")
            print(f"[APP] Job {job_id} reused the {target_language} translation "
                  f"from the same sibling, skipping translation")
        else:
            _set(job_id, step="translating")
            translated = translate_segments(segments, source_language,
                                            target_language, for_speech=for_speech)
            from translation_v2 import fix_code_switching
            translated = fix_code_switching(translated, target_language)
        _set(job_id, translated_text=" ".join(t for t in translated if t).strip())

        if not any(t.strip() for t in translated):
            raise RuntimeError("Translation produced no text")

        if kind == "subtitles_translated":
            _set(job_id, status="done", step="complete",
                 translated_segments=list(translated), **common)
            _save(job_id)
            print(f"[APP] Job {job_id} translated ({len(segments)} segments)")
            return

        if kind == "subtitled":
            # Which language goes on the picture. Defaults to the language
            # being translated into, but subtitling a clip in the language it
            # is already spoken in is a normal thing to want, and that skips
            # translation entirely.
            sub_lang = ((jobs.get(job_id) or {}).get("subtitle_language")
                        or target_language)
            if sub_lang == (source_language or "").strip() or sub_lang == "source":
                sub_texts = [(sg.get("text") or "").strip() for sg in segments]
                sub_lang = source_language
            elif sub_lang != target_language:
                _set(job_id, step="translating")
                sub_texts = list(translate_segments(
                    segments, source_language, sub_lang, for_speech=False))
                from translation_v2 import fix_code_switching
                sub_texts = fix_code_switching(sub_texts, sub_lang)
            else:
                sub_texts = list(translated)
            # Same cues the .srt export would give, written onto the picture
            # so the file carries its own captions.
            _set(job_id, step="adding_subtitles")
            subbed = os.path.join(workdir, f"{stem}_subtitled.mp4")
            with _HEAVY:
                mode = subtitles.attach(
                    input_path,
                    subtitles.cues(segments, sub_texts),
                    subbed, workdir,
                    srt_text=subtitles.srt(segments, sub_texts),
                    language=sub_lang,
                    crf=engines.tunable("OUTPUT_CRF"),
                    preset=engines.tunable("OUTPUT_PRESET"))
            filed = _file_render(job_id, subbed, stem, sub_lang,
                                 suffix="_subtitled")
            _set(job_id, status="done", step="complete",
                 translated_segments=list(translated),
                 subtitle_mode=mode, subtitle_language=sub_lang,
                 output_path=subbed, url=f"/jobs/{job_id}/video",
                 filed_at=filed, **common)
            _save(job_id)
            print(f"[APP] Job {job_id} subtitled, {mode} "
                  f"({len(segments)} segments)")
            return

        with _HEAVY:
            # ── 5. Build the dubbed track (exact length by construction) ──
            _set(job_id, step="synthesizing_voice")

            def tts_progress(done, total):
                _set(job_id, step=f"synthesizing_voice ({done + 1}/{total})")

            mixed = None
            dub_path = os.path.join(workdir, f"{stem}_dub.wav")
            units_dir = os.path.join(workdir, "units")
            _set(job_id, step="assembling_dialogue")
            plan = dubbing.build_dubbed_track(
                segments, translated, reference, target_language, video_dur,
                dub_path,
                source_lang=source_language, progress=tts_progress,
                source_audio=stt_audio, cache_dir=units_dir,
                speakers=common["speakers"])

            if kind == "audio":
                # The bed belongs in the delivered file when there is no
                # picture to mux it into later.
                delivered = dub_path
                mixed = _mix_audio(job_id, workdir, stem, dub_path, video_dur,
                                   stems, music_path, music_gain,
                                   background_lufs, background_duck)
                if mixed:
                    delivered = mixed
                filed = _file_render(
                    job_id, delivered,
                    os.path.splitext(_safe_name(filename))[0].replace(f"{job_id}_", ""),
                    target_language)
                _set(job_id, status="done", step="complete",
                     dub_path=delivered, filed_at=filed,
                     plan=plan, units_dir=units_dir,
                     reference_path=reference["path"],
                     reference_text=reference.get("text", ""),
                     reference_seconds=round(reference.get("duration", 0.0), 2),
                     reference={"path": reference["path"],
                                "text": reference.get("text", ""),
                                "duration": reference.get("duration", 0.0),
                                "start": reference.get("start")},
                     music_path=music_path, music_gain=music_gain,
                     translated_segments=list(translated), **common)
                _save(job_id)
                print(f"[APP] Job {job_id} dubbed to audio")
                return

            # ── 6. Lip sync (single encode, audio muxed in the same pass) ──
            _set(job_id, step="lip_syncing")
            output_video_path = os.path.join(workdir, f"{stem}_dubbed.mp4")

            def w2l_progress(done, total):
                _set(job_id, step=f"lip_syncing ({done}/{total} frames)")

            # The dub track stays voice-only so it can still be A/B'd against
            # the reference; the bed is mixed into a separate file that the
            # picture is muxed with.
            audio_for_video = dub_path
            if stems or music_path:
                _set(job_id, step="mixing_background")
            mixed = _mix_audio(job_id, workdir, stem, dub_path, video_dur,
                               stems, music_path, music_gain,
                               background_lufs, background_duck)
            if mixed:
                audio_for_video = mixed

            lip_sync_note = None
            try:
                generate_lip_synced_video(
                    input_path, audio_for_video, output_video_path,
                    wav2lip_batch_size=engines.tunable("W2L_BATCH"),
                    face_det_batch_size=engines.tunable("W2L_DET_BATCH"),
                    crf=engines.tunable("OUTPUT_CRF"),
                    preset=engines.tunable("OUTPUT_PRESET"),
                    progress=w2l_progress)
            except lip_sync.NoFaceDetected as e:
                # Not a failure: there is nothing in this picture to move. Keep
                # the finished dub and deliver it on the original video.
                _set(job_id, step="lip_sync_skipped")
                lip_sync_note = _deliver_without_lip_sync(
                    input_path, audio_for_video, output_video_path, str(e),
                    no_face=True)
            except lip_sync.LipSyncError as e:
                # Any other lip-sync failure - a missing checkpoint, an audio
                # decoder the environment broke (a NumPy too new for Numba
                # did exactly that), an ffmpeg error - must not throw away a
                # finished dub either. The lips are a picture detail; the
                # soundtrack is the product. Reported as a fallback, with its
                # reason, so a broken install is still visible.
                print(f"[Wav2Lip] Lip sync failed, delivering without it: {e}")
                _set(job_id, step="lip_sync_skipped")
                lip_sync_note = _deliver_without_lip_sync(
                    input_path, audio_for_video, output_video_path, str(e))

        if not os.path.exists(output_video_path):
            raise RuntimeError("Lip sync produced no output file")
        _set(job_id, lip_sync=lip_sync_note)

        # ── 7. Verify audio and video actually agree ──
        _set(job_id, step="verifying")
        check = av_sync.verify_sync(output_video_path,
                                    tolerance=engines.tunable("SYNC_TOLERANCE"))
        print(f"[APP] Sync check: {check['reason']}")
        _set(job_id, sync=check)
        if not check["ok"]:
            print(f"[APP] WARNING — output failed sync check: {check['reason']}")

        match = _voice_match(reference["path"], dub_path, stt_audio)
        if match:
            print(f"[APP] Voice match: {match['score']:.3f} (cosine {match['cosine']})")
            _set(job_id, voice_match=match)

        # Per-speaker similarity, and one isolated track per speaker. The old
        # single score describes neither actor once there are two.
        per_speaker = _speaker_voice_matches(
            common["speakers"], plan, units_dir, reference["path"], stt_audio,
            workdir)
        if per_speaker:
            _set(job_id, speaker_voice_match=per_speaker)

        # ── 8. Publish ──
        # The render stays where it was written; this is the URL that serves it.
        _set(job_id, step="uploading_result")
        final_url = f"/jobs/{job_id}/video"
        # The original name, not the job-prefixed working one: this folder is
        # browsed by a person, and "dfd8ae5d-529_clip_hi.mp4" tells them nothing.
        filed_at = _file_render(
            job_id, output_video_path,
            os.path.splitext(_safe_name(filename))[0].replace(f"{job_id}_", ""),
            target_language)

        # The reference clip is what the voice was cloned FROM, and the dub is
        # what came out. Being able to hear both side by side is the only way a
        # user can judge clone quality, so keep them and expose them.
        _set(job_id, status="done", step="complete",
             url=final_url, output_path=output_video_path,
             filed_at=filed_at,
             video_stale=False,
             reference_path=reference["path"],
             reference_text=reference.get("text", ""),
             reference_seconds=round(reference.get("duration", 0.0), 2),
             dub_path=dub_path,
             # The dialogue-only track is kept as its own artifact even when a
             # bed was mixed in: it is what a voice-match score compares
             # against, what an A/B against the reference needs, and what a
             # remix can reuse without running IndicF5 again.
             mixed_path=(mixed or None),
             # `stems` itself, not just the two paths. `run_summary`,
             # `artifact_cache` and `_reusable_sibling` all resolve the
             # background from this key; without it a render that DID preserve
             # the soundtrack reported "not preserved" and a sibling re-ran
             # Demucs from scratch.
             stems=stems,
             background_path=(stems or {}).get("background"),
             vocals_path=(stems or {}).get("vocals"),
             source_language=source_language,
             segment_count=len(segments),
             # Whisper's per-segment timings and the translation aligned to
             # them. Both were computed and discarded; keeping them is what
             # makes transcript and subtitle export possible.
             plan=plan, units_dir=units_dir, workdir=workdir,
             music_path=music_path, music_gain=music_gain,
             word_segments=segments,
             input_path=input_path, video_duration=video_dur,
             reference={"path": reference["path"],
                        "text": reference.get("text", ""),
                        "duration": reference.get("duration", 0.0),
                        "start": reference.get("start")},
             # The registry the run actually spoke in, with the per-label
             # tallies filled in from the final segments.
             speakers=SPK.build(seg_rows, common["speakers"]),
             diarization=common["diarization"],
             diarization_source=common.get("diarization_source"),
             target_language=target_language,
             segments=[{"start": float(sg.get("start", 0.0)),
                        "end": float(sg.get("end", 0.0)),
                        "text": (sg.get("text") or "").strip(),
                        "speaker": SPK.label_of(sg)}
                       for sg in segments],
             translated_segments=list(translated))

        # The run summary, built last so it describes the finished artifact set.
        # Best effort: a reporting failure must not fail a completed render.
        try:
            job = jobs.get(job_id) or {}
            job["job_id"] = job_id
            summary = run_summary.build(job, readiness=readiness.snapshot(
                target_language=target_language, source_language=source_language))
            job["run_summary"] = summary
            print(run_summary.render(summary))
        except Exception as e:                               # noqa: BLE001
            print(f"[APP] Could not build the run summary: {e}")

        _save(job_id)
        print(f"[APP] Job {job_id} completed")

    except JobCancelled:
        print(f"[APP] Job {job_id} cancelled by the user")
        jobs.get(job_id, {}).update(
            {"status": "cancelled", "step": "cancelled",
             "error": "Cancelled before it finished."})
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="failed", step="error", error=str(e))

    finally:
        # A run that did not finish still gets a summary, so the stage that
        # stopped and the warnings so far survive into the manifest.
        try:
            job = jobs.get(job_id)
            if job and job.get("status") in ("failed", "cancelled"):
                job["job_id"] = job_id
                job["run_summary"] = run_summary.build(job)
        except Exception:                                    # noqa: BLE001
            pass
        if not KEEP_INTERMEDIATE:
            job = jobs.get(job_id, {})
            # Revising a finished dub needs more than the three output files:
            # the (possibly trimmed) source to re-run lip sync against, and the
            # per-phrase cache so one phrase can be rebuilt without redoing all
            # of them. Deleting these turned "edit a word" into a failed job.
            keep = {job.get("output_path"), job.get("reference_path"),
                    job.get("dub_path"), job.get("input_path")}
            for name in os.listdir(workdir) if os.path.isdir(workdir) else []:
                path = os.path.join(workdir, name)
                # The manifest is the index for everything kept above, not an
                # intermediate. Sweeping it away deleted the project the moment
                # it was saved.
                if name == project.MANIFEST:
                    continue
                if path not in keep and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
# The frontend used to be opened straight off disk, which is why profile.html
# had http://localhost:8000 hardcoded while workspace.html computed its own
# base URL. Serving it here gives one origin, kills the CORS/file:// problem,
# and makes every fetch a same-origin relative path.
_WEB_DIR = os.path.join(_PROJECT_ROOT, "web")
if os.path.isdir(_WEB_DIR):
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    # Hashed build assets are immutable, so they can be cached hard.
    app.mount("/app/assets",
              StaticFiles(directory=os.path.join(_WEB_DIR, "assets")),
              name="assets")

    @app.get("/")
    async def _root():
        return RedirectResponse("/app/")

    @app.get("/app")
    @app.get("/app/{path:path}")
    async def _spa(path: str = ""):
        """
        Serve the built single-page app.

        React Router owns /app/studio and /app/library, which do not exist on
        disk. Without this fallback a refresh or a direct link to either one
        would 404 against StaticFiles. Real files (favicon, etc.) are still
        served from disk; everything else returns index.html and lets the
        router resolve it.
        """
        candidate = os.path.normpath(os.path.join(_WEB_DIR, path))
        if (path and candidate.startswith(_WEB_DIR) and os.path.isfile(candidate)):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_WEB_DIR, "index.html"))

else:
    # The interface is a build artefact and is not in the repository, so a
    # fresh clone reaches /app/ before it exists. Saying so at startup beats
    # a 404 with nothing behind it.
    print("[APP] No web/ directory: the studio interface has not been built.")
    print("[APP] Build it with:  cd apps/studio && npm install && npm run build")
    print("[APP] The API itself is fine; only the browser interface is missing.")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.post("/process_video/")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    original_language: str = Query(..., description="Source language code"),
    target_language: str = Query(..., description="Target language code"),
    user_id: str = Query("anonymous", description="Owner of this job"),
    name: str = Query(None, description="What to call this project"),
    project_id: str = Query(None, description="An existing project this output "
                                              "joins. Omit to start a new one."),
    kind: str = Query("dub", description="dub, audio, subtitled, subtitles, "
                                         "or subtitles_translated"),
    subtitle_language: str = Query(
        None, description="Language for a subtitled video. Defaults to the "
                          "target; pass the source language to subtitle a clip "
                          "in the language it is already spoken in."),
    trim_start: float = Query(None, description="In point in seconds"),
    trim_end: float = Query(None, description="Out point in seconds"),
    music_gain: float = Query(-18.0, description="Music bed level in dB"),
    music: UploadFile = File(None),
    preserve_background: bool = Query(
        None, description="Keep the film's own music and sound effects under "
                          "the dub. Defaults to the PRESERVE_BACKGROUND "
                          "setting."),
    background_lufs: float = Query(
        None, description="Loudness the preserved background is normalised to. "
                          "Defaults to the BACKGROUND_LUFS setting."),
    background_duck: bool = Query(
        None, description="Lower the background while dialogue is present. "
                          "Defaults to the BACKGROUND_DUCK setting."),
):
    # A reopened project has its clip on disk already. Rendering another
    # output from it should not mean finding the file again, so with no
    # upload the project's own input is used.
    stored = None
    if file is None or not file.filename:
        stored = _project_input(project_id)
        if stored is None:
            raise HTTPException(
                status_code=400,
                detail="No file provided" if not project_id else
                       "This project's source clip is no longer on disk. "
                       "Import the clip to render something new from it.")

    try:
        target_language = L.normalize(target_language)
        L.engine_for(target_language)
        # A subtitle language is a language like any other, so a typo should
        # be refused here rather than mid-render.
        if subtitle_language and subtitle_language != "source":
            subtitle_language = L.normalize(subtitle_language)
    except L.UnsupportedLanguage as e:
        raise HTTPException(status_code=400, detail=str(e))

    if kind not in ("dub", "audio", "subtitles", "subtitles_translated",
                    "subtitled"):
        raise HTTPException(status_code=400, detail=f"Unknown output: {kind}")

    # Subtitles never reach a TTS engine, so they are fine in any language.
    if kind in ("dub", "audio"):
        _require_speakable(target_language)

    # The range still has to be checked against the real duration once the file
    # has been probed, but an out point at or before the in point is wrong on
    # its own terms. Catching it here fails in a second instead of accepting a
    # job that dies partway through transcription.
    if trim_start is not None and trim_start < 0:
        raise HTTPException(status_code=400, detail="The in point cannot be negative")
    if (trim_start is not None and trim_end is not None
            and trim_end - trim_start < 1.0):
        raise HTTPException(
            status_code=400,
            detail=f"The trimmed range is {trim_end - trim_start:.2f}s; "
                   f"at least a second is needed.")

    if stored is not None:
        with open(stored["input_path"], "rb") as f:
            content = f.read()
        filename = stored["filename"]
    else:
        content = await file.read()
        filename = file.filename
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 200MB)")
    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="File too small or empty")

    job_id = str(uuid.uuid4())[:12]
    safe_filename = f"{job_id}_{_safe_name(filename)}"
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(input_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "status": "queued",
        "step": "uploaded",
        "kind": kind,
        "filename": filename,
        "title": (name or "").strip() or None,
        # Several outputs from one clip are one project. A job that names no
        # project founds its own, so a lone render behaves as it always did.
        "project_id": _project_for(project_id, job_id),
        # Wall-clock start, so a project that is still rendering can be listed
        # and sorted while it has no manifest to date it.
        "started_at": time.time(),
        "original_language": original_language,
        "target_language": target_language,
        "subtitle_language": (subtitle_language or "").strip() or None,
        "user_id": user_id,
        # The per-request choices, or None to follow the stored setting.
        "preserve_background": preserve_background,
        "background_lufs": background_lufs,
        "background_duck": background_duck,
        "created_at": asyncio.get_event_loop().time(),
    }

    music_path = None
    if music is not None and music.filename:
        music_bytes = await music.read()
        if len(music_bytes) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Music file too large (max 50MB)")
        music_path = os.path.join(UPLOAD_DIR,
                                  f"{job_id}_music_{_safe_name(music.filename)}")
        with open(music_path, "wb") as f:
            f.write(music_bytes)

    background_tasks.add_task(process_video_task, job_id, input_path,
                             original_language, target_language, safe_filename,
                             user_id, trim_start, trim_end, music_path,
                             music_gain, kind, preserve_background,
                             background_lufs, background_duck)

    return JSONResponse({
        "status": "accepted",
        "job_id": job_id,
        "message": "Video processing started. Poll /jobs/{job_id}/status.",
    })


@app.get("/api/readiness")
async def get_readiness(target_language: str = Query(None),
                        source_language: str = Query(None)):
    """
    What this machine can actually do, structured.

    The studio asks before accepting a render so a missing model is a message
    on the setup screen rather than a failure minutes into a job. Cheap: every
    probe is an import spec or a file check, and nothing loads a model.

    Never returns a token — only whether one is present. See `readiness`.
    """
    return JSONResponse(readiness.snapshot(
        target_language=target_language, source_language=source_language))


@app.get("/jobs/{job_id}/summary")
async def get_run_summary(job_id: str):
    """
    The final report for a run: what ran, what fell back, what it measured.

    Available while a job is still running, so the studio can show progress and
    fallbacks as they happen rather than only at the end.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    summary = run_summary.build(job)
    return JSONResponse({"summary": summary,
                         "text": run_summary.render(summary)})


@app.patch("/jobs/{job_id}/speakers/{label}")
async def rename_speaker(job_id: str, label: str, body: dict = Body(...)):
    """
    Give a speaker a display name.

    The alias is an overlay and nothing else. `label` remains the stable id that
    the unit cache, the plan, the reference files and the manifest all key on,
    so renaming a speaker can never invalidate a phrase or move a reference. The
    studio shows the alias; the pipeline never sees it.
    """
    job = _require_editable(job_id)
    registry = SPK.normalize_speakers(job.get("speakers"))
    if not any(entry.get("label") == label for entry in registry):
        raise HTTPException(status_code=404,
                            detail=f"No speaker {label} in this project")

    alias = str(body.get("alias") or "").strip()
    aliases = dict(job.get("speaker_aliases") or {})
    if alias:
        aliases[label] = alias
    else:
        aliases.pop(label, None)
    job["speaker_aliases"] = aliases
    project.save(job_id, job)

    return JSONResponse({
        "label": label,
        "alias": aliases.get(label),
        "speakers": SPK.describe_all(registry, aliases=aliases,
                                     plan=job.get("plan")),
    })


@app.get("/jobs/{job_id}/speakers")
async def list_speakers(job_id: str):
    """
    The detected speakers, with status and warnings.

    A run that finished before diarization existed has no registry; it reports
    one implicit `SPEAKER_00` whose reference is the video-wide one, which is
    what those runs actually did.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    registry = SPK.normalize_speakers(job.get("speakers"))
    if not registry and job.get("reference"):
        registry = SPK.implicit_speaker(job["reference"])
    return JSONResponse({
        "speakers": SPK.describe_all(registry,
                                     aliases=job.get("speaker_aliases") or {},
                                     plan=job.get("plan")),
        "warnings": SPK.warnings(registry),
        "voice_match": job.get("speaker_voice_match") or [],
    })


@app.get("/jobs/{job_id}/artifact/{name}")
async def get_artifact(job_id: str, name: str):
    """
    Serve an intermediate artifact by name, for debugging.

    The point of Phase 6's artifact exposure: hearing the vocals stem, one
    speaker's isolated track, or the mix without the overlap gain is how you
    tell a scheduling bug from a bad clone. Only names on the allowlist are
    served, so this cannot be turned into a file-read primitive.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    allowed = {
        "dialogue": job.get("dub_path"),
        "mixed": job.get("mixed_path"),
        "vocals": job.get("vocals_path"),
        "background": job.get("background_path"),
        "output": job.get("output_path"),
    }
    # A per-speaker isolated track, when one was written.
    if name.startswith("speaker:") and name.count(":") == 1:
        label = name.split(":", 1)[1]
        workdir = job.get("workdir") or ""
        candidate = os.path.join(workdir, "speakers", f"{label}.wav")
        allowed[name] = candidate if os.path.exists(candidate) else None

    path = allowed.get(name)
    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"No artifact {name!r} for this job. Available: "
                   f"{sorted(k for k, v in allowed.items() if v)}")
    if not os.path.exists(path):
        raise HTTPException(status_code=410,
                            detail=f"The {name} artifact is no longer on disk.")
    return FileResponse(path, media_type="audio/wav",
                        filename=f"{job_id}_{name.replace(':', '_')}.wav")


@app.get("/api/artifacts")
async def get_artifact_graph():
    """
    The artifact dependency graph: what each stage's output depends on.

    Exposed so the studio can tell a user what a settings change will cost
    before they make it — "this re-runs Demucs (minutes)" rather than a
    surprise. Read-only; nothing here touches a job.
    """
    return JSONResponse({
        "artifacts": {
            name: {"depends_on": list(entry["depends_on"]),
                   "inputs": list(entry["inputs"]),
                   "cost": entry["cost"],
                   "per_unit": bool(entry.get("per_unit"))}
            for name, entry in artifact_cache.ARTIFACTS.items()
        },
        "text": artifact_cache.describe(),
    })


@app.get("/jobs/{job_id}/cache")
async def get_job_cache(job_id: str):
    """
    Which of this job's artifacts are present, and what a change would cost.

    `invalidated_by` answers the brief's requirement directly: a change to one
    input invalidates that artifact and its dependents, and nothing else.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    workdir = job.get("workdir") or ""
    units = job.get("units_dir") or ""
    stems = job.get("stems") or {}

    present = {
        "stems": bool(stems.get("background") and
                      os.path.exists(stems.get("background") or "")),
        "transcript": bool(job.get("word_segments")),
        "diarization": bool(job.get("diarization")),
        "speakers": bool(job.get("speakers")),
        "references": bool([e for e in SPK.normalize_speakers(job.get("speakers"))
                            if (e.get("reference") or {}).get("path")]),
        "translation": bool(job.get("translated_segments")),
        "plan": bool(job.get("plan")),
        "phrase_audio": bool(units and os.path.isdir(units)),
        "dialogue_track": bool(job.get("dub_path") and
                               os.path.exists(job.get("dub_path"))),
        "mixed_audio": bool(job.get("mixed_path") and
                            os.path.exists(job.get("mixed_path"))),
        "output_video": bool(job.get("output_path") and
                             os.path.exists(job.get("output_path"))),
    }

    costs = {}
    for field in ("trim", "target_language", "source_language",
                  "demucs_chunk_seconds", "music_gain", "background_duck",
                  "dialogue_lufs", "background_lufs", "output_crf"):
        stale = artifact_cache.invalidated_by([field])
        costs[field] = {
            "invalidates": stale,
            "worst": artifact_cache.cheapest_stale(stale),
        }

    return JSONResponse({
        "present": present,
        "workdir": workdir or None,
        "reclone": {
            label: artifact_cache.reference_invalidates(label)
            for label in SPK.labels(SPK.normalize_speakers(job.get("speakers")))
        },
        "change_costs": costs,
    })


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job_id, "status": job["status"],
                "step": job.get("step", "unknown"),
                "kind": job.get("kind", "dub"),
                "name": _project_name(job, job_id),
                # What this output was asked for, so the studio can tell a
                # request for the same thing from one for a different
                # language.
                "target_language": job.get("target_language"),
                "original_language": (job.get("original_language")
                                      or job.get("source_language")),
                "has_input": bool(job.get("input_path")
                                  and os.path.exists(job["input_path"]))}

    if job["status"] == "cancelled":
        response["error"] = job.get("error", "Cancelled.")

    if job["status"] == "done":
        response["url"] = job.get("url")
        response["translated_script"] = job.get("translated_text")
        response["original_text"] = job.get("original_text")
        response["sync"] = job.get("sync")
        response["source_language"] = job.get("source_language")
        response["segment_count"] = job.get("segment_count")
        response["reference_text"] = job.get("reference_text")
        response["reference_seconds"] = job.get("reference_seconds")
        response["voice_match"] = job.get("voice_match")
        response["voiceover_seconds"] = job.get("voiceover_seconds")
        response["has_transcript"] = bool(job.get("segments"))
        response["editable"] = bool(job.get("plan"))
        response["video_duration"] = job.get("video_duration")
        response["filed_at"] = job.get("filed_at")
        response["filed_error"] = job.get("filed_error")
        # "burned" or "muxed": whether the captions are drawn on the picture
        # or carried as a track the player switches on.
        response["subtitle_mode"] = job.get("subtitle_mode")
        response["subtitle_language"] = job.get("subtitle_language")
        response["project_id"] = job.get("project_id", job_id)
        response["video_stale"] = bool(job.get("video_stale"))
        if job.get("reference_path"):
            response["reference_audio"] = f"/jobs/{job_id}/audio/reference"
        if job.get("dub_path"):
            response["dub_audio"] = f"/jobs/{job_id}/audio/dub"
    if job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return JSONResponse(response)


@app.get("/jobs/{job_id}/video")
async def get_job_video(job_id: str, download: bool = False):
    """
    The finished render, served from where the pipeline wrote it.

    `download=1` asks the browser to save it rather than play it. Without the
    Content-Disposition header a plain `download` attribute does nothing in a
    webview, which is why the button appeared dead.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.get("output_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No render for this job")
    from fastapi.responses import FileResponse

    if not download:
        return FileResponse(path, media_type="video/mp4")

    # Name it after the project and language, not the job id.
    stem = _project_name(job, job_id).rsplit(".", 1)[0].strip() or "videotrans"
    stem = re.sub(r"[^\w\- ]+", "", stem).strip() or "videotrans"
    lang = job.get("target_language") or "dub"
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{stem}_{lang}.mp4")


@app.post("/jobs/{job_id}/reveal")
async def reveal_render(job_id: str):
    """
    Show the finished file in the desktop file manager.

    The render already exists on disk; on a local app the useful action is to
    point at it, not to send the bytes back over HTTP to be saved again.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    target = job.get("filed_at") or job.get("output_path")
    if not target or not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Nothing on disk to show yet")

    if sys.platform == "darwin":
        cmd = ["open", "-R", target]
    elif sys.platform.startswith("win"):
        cmd = ["explorer", "/select,", os.path.normpath(target)]
    else:
        cmd = ["xdg-open", os.path.dirname(target)]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"Could not open the file manager: {e}")
    return JSONResponse({"revealed": target})


@app.get("/jobs/{job_id}/audio/{which}")
async def get_job_audio(job_id: str, which: str, download: bool = False):
    """
    A finished job's reference clip or dubbed track.

    `download=1` saves it instead of playing it. A voice-over and an audio-only
    dub produce nothing but this, so without it those outputs could be heard
    and not kept.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    key = {"reference": "reference_path", "dub": "dub_path"}.get(which)
    if not key:
        raise HTTPException(status_code=404, detail="Unknown audio track")
    path = job.get(key)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Audio no longer on disk")
    from fastapi.responses import FileResponse
    if not download:
        return FileResponse(path, media_type="audio/wav")
    lang = job.get("target_language") or "audio"
    return FileResponse(path, media_type="audio/wav",
                        filename=f"{_download_name(job_id)}_{lang}.wav")


def voiceover_task(job_id: str, input_path: str, script: str,
                   language: str, filename: str, user_id: str):
    """
    Speak a typed script in the voice of an uploaded reference clip.

    Shares the front of the dubbing pipeline (extract, transcribe, choose the
    cleanest reference window) and then skips translation, timeline fitting and
    lip sync entirely, because the user supplied the words and there is no
    video to stay in sync with.
    """
    workdir = os.path.join(UPLOAD_DIR, f"job_{job_id}")
    os.makedirs(workdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(filename))[0]

    try:
        _set(job_id, status="processing", step="validating")
        if not os.path.exists(input_path) or os.path.getsize(input_path) < 1000:
            raise ValueError("Uploaded file is missing or too small")

        info = av_sync.probe(input_path)
        media_dur = info["duration"]

        _set(job_id, step="extracting_audio")
        stt_audio = av_sync.extract_audio(
            input_path, os.path.join(workdir, f"{stem}_stt.wav"), 16000, 1)

        _set(job_id, step="transcribing")
        result = transcribe_audio(stt_audio)
        segments = result.get("segments") or []
        if not segments:
            raise RuntimeError(
                "No speech found in the reference clip, so there is no voice to copy.")

        _set(job_id, step="selecting_reference")
        reference = reference_audio.build_reference(
            input_path, segments, media_dur,
            os.path.join(workdir, f"{stem}_ref.wav"), sample_rate=24000)

        with _HEAVY:
            _set(job_id, step="synthesizing_voice")

            def progress(done, total):
                _set(job_id, step=f"synthesizing_voice ({done + 1}/{total})")

            out_path = os.path.join(workdir, f"{stem}_voiceover.wav")
            result_info = voiceover.render(
                reference["path"], reference.get("text", ""), script,
                language, out_path, progress=progress)

        jobs[job_id].update({
            "status": "done", "step": "complete",
            "dub_path": result_info["path"],
            "reference_path": reference["path"],
            "reference_text": reference.get("text", ""),
            "reference_seconds": round(reference.get("duration", 0.0), 2),
            "segment_count": result_info["chunks"],
            "voiceover_seconds": result_info["duration"],
            "source_language": language,
            "translated_text": script,
        })
        _save(job_id)
        print(f"[APP] Voice-over {job_id} completed "
              f"({result_info['duration']}s, {result_info['chunks']} chunks)")

    except JobCancelled:
        print(f"[APP] Voice-over {job_id} cancelled by the user")
        jobs.get(job_id, {}).update(
            {"status": "cancelled", "step": "cancelled",
             "error": "Cancelled before it finished."})
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        jobs.get(job_id, {}).update(
            {"status": "failed", "step": "error", "error": str(e)})


def _source_text_at(segments: list, start: float, duration: float) -> str:
    """The source line a phrase overlaps most, for showing beside the edit box."""
    best, best_overlap = "", 0.0
    end = start + duration
    for seg in segments:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best_overlap, best = overlap, (seg.get("text") or "").strip()
    return best


def _diarization_enabled() -> bool:
    """
    Whether to look for multiple speakers at all.

    Off by default in two cases: a deployment that has no pyannote (checked
    cheaply, without loading anything), and an operator who set
    VIDEOTRANS_DIARIZE=off. Both mean one implicit speaker, which is the
    previous pipeline — so a machine that cannot diarize never pays for the
    attempt, and never fails because of it.
    """
    setting = os.getenv("VIDEOTRANS_DIARIZE", "").strip().lower()
    if setting in ("0", "false", "no", "off"):
        return False
    if setting in ("1", "true", "yes", "on"):
        return True
    return speaker_diarization.available()


def _preserve_background(requested: bool = None) -> bool:
    """
    Whether to run separation at all.

    Resolution order, most specific first:

      1. `requested` — the per-render choice the studio sends.
      2. `VIDEOTRANS_PRESERVE_BACKGROUND` — an operator turning it on or off for
         a whole deployment.
      3. the `PRESERVE_BACKGROUND` tunable, off by default.

    Demucs is minutes per film on CPU and a heavy optional dependency, so the
    default has to be the cheap path.

    The package is not checked here: `source_separation.separate` does that, so
    a machine with the setting on but Demucs missing gets one clear log line and
    the dialogue-only pipeline rather than a failure.
    """
    if requested is not None:
        return bool(requested)
    setting = os.getenv("VIDEOTRANS_PRESERVE_BACKGROUND")
    if setting is not None:
        return setting.strip().lower() in ("1", "true", "yes", "on")
    return bool(engines.tunable("PRESERVE_BACKGROUND"))


def _mix_audio(job_id: str, workdir: str, stem: str, dub_path: str,
               video_dur: float, stems: dict, music_path: str,
               music_gain: float, background_lufs: float = None,
               background_duck: bool = None):
    """
    Compose the deliverable audio: dialogue, plus whatever beds we have.

    Returns the path of the mixed file, or None when there is nothing to mix
    with — in which case the caller keeps the dialogue-only track it already
    has, which is the pre-Phase-4 behaviour.

    `background_lufs` and `background_duck` are per-render overrides; when they
    are None the stored setting is used, which is what keeps a render made
    without touching the studio identical to one made before these controls
    existed.

    Both beds are handled in one pass so the dialogue is normalised exactly
    once. Chaining two ffmpeg calls would re-normalise the output of the first,
    so the dialogue level would move depending on whether a music file happened
    to be attached — a coupling that makes a render unpredictable for no gain.

    Layer order is dialogue > preserved background > user music bed. The user's
    music does NOT replace the film's original score; both are kept.
    """
    background = (stems or {}).get("background")
    if background and not os.path.exists(background):
        background = None
    if not background and not music_path:
        return None

    # A level is clamped to a range that cannot produce obviously clipped or
    # inaudible output, so a mistyped value degrades rather than ruins a render.
    level = (float(engines.tunable("BACKGROUND_LUFS"))
             if background_lufs is None else float(background_lufs))
    level = max(-40.0, min(-6.0, level))
    duck = (bool(engines.tunable("BACKGROUND_DUCK"))
            if background_duck is None else bool(background_duck))

    out_path = os.path.join(workdir, f"{stem}_mixed.wav")
    try:
        return av_sync.mix_layers(
            dub_path, out_path, target=video_dur,
            background_path=background, music_path=music_path,
            music_gain=music_gain,
            dialogue_lufs=float(engines.tunable("DIALOGUE_LUFS")),
            background_lufs=level,
            duck=duck,
            duck_db=float(engines.tunable("DUCK_DB")),
            duck_ratio=float(engines.tunable("DUCK_RATIO")))
    except Exception as e:                                   # noqa: BLE001
        # A mix failure must not fail a dub that has already been synthesized.
        # The dialogue-only track is complete and correct; it just has no bed.
        print(f"[APP] Could not mix the background ({e}); delivering the "
              f"dialogue-only track")
        return None


def _deliver_without_lip_sync(input_path: str, audio_path: str,
                              out_path: str, reason: str,
                              no_face: bool = False) -> dict:
    """
    Attach the dub to the untouched picture when there is no face to sync.

    Lip sync is only applicable when a face is visible for most of the clip. A
    landscape, a screen recording, a title card, a product shot or footage from
    behind used to raise `LipSyncError`, which the job treated as a failure: the
    whole render was thrown away even though the dubbed audio was finished and
    correct. The face detector failing to find anything is a statement about the
    PICTURE, not about the audio, so the right answer is to skip the stage and
    deliver the original video with the new soundtrack.

    `av_sync.mux` copies the video stream untouched and encodes only the audio,
    so this is the same quality the source had. The reason string is kept on the
    job so the studio and the run summary can say the lips were not moved
    instead of implying they were.
    """
    av_sync.mux(input_path, audio_path, out_path)
    print(f"[Wav2Lip] Lip sync skipped — {reason}")
    return {"skipped": True, "reason": reason, "no_face": no_face}


def _assign_implicit_speaker(segments: list, label: str = None) -> list:
    """
    Stamp the one-speaker label onto segments that have none.

    A video with no diarization genuinely has one speaker, and saying so is
    better than leaving every segment unlabelled: the tallies in the registry
    are then real numbers, the studio can show who is speaking, and every unit
    still resolves to the same single reference it always did.

    Segments that already carry a label are left alone, so this never overwrites
    a real diarization result.
    """
    label = label or SPK.DEFAULT_LABEL
    out = []
    for seg in segments or []:
        if SPK.label_of(seg):
            out.append(seg)
        else:
            out.append({**seg, "speaker": label})
    return out


# How many phrase edits can be walked back. Each step keeps one WAV, so the
# ceiling is on disk use as much as on memory.
HISTORY_DEPTH = 20


def _push_history(job: dict, unit: dict) -> None:
    """
    Record a phrase's state before it is overwritten.

    The audio is copied rather than regenerated on undo: re-synthesizing would
    give a different take, because the engine is stochastic, so "undo" would
    not actually return what was there.
    """
    index = unit["index"]
    current = dubbing.segment_path(job["units_dir"], index)
    backup = None
    if os.path.exists(current):
        backup = os.path.join(job["units_dir"], f"undo_{index}_{uuid.uuid4().hex[:8]}.wav")
        try:
            shutil.copyfile(current, backup)
        except OSError as e:                                 # noqa: BLE001
            print(f"[APP] Could not snapshot phrase {index}: {e}")
            backup = None

    history = job.setdefault("history", [])
    history.append({"index": index, "text": unit["text"],
                    "seed": unit.get("seed"), "wav": backup,
                    "cleared": bool(unit.get("cleared"))})

    while len(history) > HISTORY_DEPTH:
        stale = history.pop(0)
        if stale.get("wav") and os.path.exists(stale["wav"]):
            try:
                os.remove(stale["wav"])
            except OSError:
                pass


def _download_name(job_id: str, stem: str = "") -> str:
    """A filename a person can recognise: the project, not the job id."""
    job = jobs.get(job_id) or {}
    pretty = (job.get("title") or "").strip()
    if not pretty:
        pretty = re.sub(r"^[0-9a-f]{8,}[-_]", "", stem or job_id)
    return re.sub(r"[^\w\- ]+", "", pretty).strip() or "videotrans"


def _file_render(job_id: str, source_path: str, stem: str, language: str,
                 suffix: str = "") -> str:
    """
    Copy a finished render into the user's output folder.

    The working copy stays where it is, because the studio serves playback and
    re-renders from there. This is the one a person goes looking for in Finder,
    so it gets a name that says what it is and never overwrites an earlier take.
    """
    try:
        folder = engines.output_dir()
        # Prefer what the person called the project. `stem` comes from the
        # stored upload name, which carries the job id, so renders were landing
        # as "a3f91c02-1b4_clip_hi.mp4" - unreadable, and the thing you scan
        # for in Finder.
        job = jobs.get(job_id) or {}
        pretty = (job.get("title") or "").strip()
        if not pretty:
            pretty = re.sub(r"^[0-9a-f]{8,}[-_]", "", stem)
        pretty = re.sub(r"[^\w\- ]+", "", pretty).strip() or stem or "videotrans"
        base = f"{pretty}_{language}{suffix}"
        target = os.path.join(folder, f"{base}.mp4")
        n = 2
        while os.path.exists(target):
            target = os.path.join(folder, f"{base}_{n}.mp4")
            n += 1
        shutil.copyfile(source_path, target)
        print(f"[APP] Filed {os.path.basename(target)} in {folder}")
        job = jobs.get(job_id)
        if job is not None:
            job.pop("filed_error", None)
        return target
    except Exception as e:                                   # noqa: BLE001
        # Filing is a convenience. A render that succeeded must not be
        # reported as failed because a folder was not writable - but it must
        # not be reported as filed either. Saying nothing sent people to an
        # output folder that never got the video.
        print(f"[APP] Could not file the render: {e}")
        job = jobs.get(job_id)
        if job is not None:
            job["filed_error"] = (
                f"The render is finished, but it could not be copied to "
                f"{engines.output_dir(create=False)}: {e}")
        return ""


def _require_editable(job_id: str) -> dict:
    """A job whose phrases can still be rebuilt: finished, with its plan kept."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("plan"):
        raise HTTPException(
            status_code=409,
            detail="This run has no editable timeline. Only dubs finished "
                   "since script editing was added can be revised.")
    return job


def _rebuild_track(job: dict) -> float:
    """
    Re-lay every cached phrase onto the track and rewrite the dub WAV.

    Only the edited phrase is ever re-synthesized; the rest are read back from
    the unit cache, so a one-word change costs one phrase and not a whole run.
    """
    plan = job["plan"]
    waves, missing = {}, []
    for unit in plan:
        wave = dubbing.read_unit(job["units_dir"], unit["index"])
        if wave is None:
            missing.append(unit["index"])
        else:
            waves[unit["index"]] = wave

    if missing and len(missing) == len(plan):
        raise HTTPException(
            status_code=409,
            detail="This project's phrase audio is gone, so the track cannot be "
                   "rebuilt. Re-render it.")
    if missing:
        # Not fatal - the track is still mostly right - but the caller has to
        # be able to say which lines fell silent instead of shipping them.
        print(f"[APP] Rebuilt without phrases {missing}: their audio is missing")
    job["missing_units"] = missing

    track = dubbing.assemble(plan, waves, job["video_duration"])
    import soundfile as sf
    sf.write(job["dub_path"], track, dubbing.SAMPLE_RATE, subtype="PCM_16")
    # The rendered video still carries the previous audio until it is redone.
    job["video_stale"] = True
    return round(len(track) / dubbing.SAMPLE_RATE, 3)


@app.get("/jobs/{job_id}/segments")
async def list_segments(job_id: str):
    """The editable phrase timeline: where each one sits and what it says."""
    job = _require_editable(job_id)
    source = job.get("segments") or []
    return JSONResponse({
        "video_duration": job.get("video_duration"),
        "video_stale": bool(job.get("video_stale")),
        "can_undo": len(job.get("history") or []),
        "target_language": job.get("target_language"),
        # The registry as the studio shows it: status, warning, reference span
        # and how many phrases each speaker actually speaks. `label` stays the
        # stable id; `name` is the alias a person may have given it.
        "speakers": SPK.describe_all(
            SPK.normalize_speakers(job.get("speakers")),
            aliases=job.get("speaker_aliases") or {},
            plan=job.get("plan")),
        "segments": [
            {
                "index": u["index"],
                "start": round(u["start"], 3),
                "duration": round(u["duration"], 3),
                "text": u["text"],
                # Units come from phrase splitting, so their index does not
                # line up with the source segments. Match on time instead: the
                # source line a phrase was spoken over.
                "source_text": _source_text_at(source, u["start"], u["duration"]),
                "seed": u.get("seed"),
                "cleared": bool(u.get("cleared")),
                # Who speaks this phrase. Absent on runs from before
                # diarization, which were all SPEAKER_00 in practice.
                "speaker": u.get("speaker") or SPK.DEFAULT_LABEL,
                "ref_index": u.get("ref_index"),
                # What this phrase actually takes to say, against the slot it
                # has. The single most useful thing to know about a dub, and
                # it used to take a click per phrase to find out.
                "spoken": dubbing.unit_seconds(job["units_dir"], u["index"]),
            }
            for u in job["plan"]
        ],
    })


@app.post("/jobs/{job_id}/segments/{index}")
async def revise_segment(job_id: str, index: int, body: dict = Body(default={})):
    """
    Rewrite one phrase, re-roll its delivery, or both.

    `text` replaces what the phrase says. `seed` changes the sampling draw,
    which is how you get a different take of the same words: IndicF5 is
    stochastic and the pipeline otherwise pins the seed for reproducibility.
    """
    job = _require_editable(job_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="This job is still rendering.")

    unit = next((u for u in job["plan"] if u["index"] == index), None)
    if unit is None:
        raise HTTPException(status_code=404, detail=f"No phrase {index} in this timeline")

    # Snapshot BEFORE the mutations below. Taking it afterwards records the
    # new state as the thing to restore, so undo returns the edit rather than
    # reversing it - which is exactly what it did until this was moved.
    previous = {"text": unit["text"], "seed": unit.get("seed")}

    text = body.get("text")
    if text is not None:
        text = str(text).strip()
        if not text:
            raise HTTPException(status_code=400, detail="A phrase cannot be empty")
        unit["text"] = text

    seed = body.get("seed")
    if seed is not None:
        try:
            # OverflowError is what infinity raises, and it is not a ValueError:
            # {"seed": 1e400} used to escape this handler as a 500.
            seed = int(seed)
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(status_code=400, detail="Seed must be a whole number")
        # The generators seeded downstream take a 32-bit value. A larger one
        # parses cleanly here and then fails inside synthesis instead.
        if not 0 <= seed < 2 ** 32:
            raise HTTPException(
                status_code=400,
                detail="Seed must be between 0 and 4294967295.")
        unit["seed"] = seed

    ref = job["reference"]
    try:
        # Interactive edits do not queue behind a render. Waiting silently for
        # the pipeline to free up ran past the client's timeout, so the studio
        # reported a failure for an edit that then quietly succeeded minutes
        # later. Saying the engine is busy is the truth and arrives at once.
        if not _HEAVY.acquire(timeout=2):
            unit.update(previous)
            raise HTTPException(status_code=409, detail=BUSY_DETAIL)
        try:
            # The phrase is re-spoken in ITS speaker's voice, not the video's
            # first voice: editing line 12 of a two-hander must not quietly
            # change who is talking.
            spoken_ref = SPK.reference_for(unit, job.get("speakers"), ref) or ref
            wave = dubbing.synth_unit(unit, spoken_ref["path"],
                                      spoken_ref.get("text", ""),
                                      float(spoken_ref.get("duration") or 0.0),
                                      job["target_language"],
                                      seed=unit.get("seed"))
        finally:
            _HEAVY.release()
    except HTTPException:
        raise
    except Exception as e:                                   # noqa: BLE001
        unit.update(previous)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not speak that phrase: {e}")

    if wave.size == 0 or float(np.max(np.abs(wave))) < 1e-4:
        # Same reason as above: keep the words and the audio telling one story.
        unit.update(previous)
        raise HTTPException(
            status_code=500,
            detail="That phrase came back silent. Try different wording or another seed.")

    _push_history(job, {"index": index, **previous})
    unit.pop("cleared", None)
    dubbing.write_unit(job["units_dir"], index, wave)
    duration = _rebuild_track(job)
    project.save(job_id, job)
    spoken = round(len(wave) / dubbing.SAMPLE_RATE, 3)
    print(f"[APP] Job {job_id} phrase {index} revised "
          f"({spoken}s in a {unit['duration']:.2f}s slot)")

    return JSONResponse({
        "index": index,
        "text": unit["text"],
        "seed": unit.get("seed"),
        "spoken_seconds": spoken,
        "slot_seconds": round(unit["duration"], 3),
        # Longer than its slot means the phrase was shortened to fit; that is
        # worth surfacing rather than hiding.
        "overruns": spoken >= round(unit["duration"], 3) - 0.02,
        "track_seconds": duration,
        "video_stale": True,
        "missing_units": job.get("missing_units") or [],
        "can_undo": len(job.get("history", [])),
    })


@app.delete("/jobs/{job_id}/segments/{index}")
async def clear_segment(job_id: str, index: int):
    """
    Silence a phrase, keeping its slot and its words.

    This is where Cut puts the audio. The text is kept rather than emptied:
    an empty phrase is rejected everywhere else, and seeing what used to be
    said is what makes the cut reversible by eye as well as by undo.
    """
    job = _require_editable(job_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="This job is still rendering.")

    unit = next((u for u in job["plan"] if u["index"] == index), None)
    if unit is None:
        raise HTTPException(status_code=404, detail=f"No phrase {index} in this timeline")
    if unit.get("cleared"):
        raise HTTPException(status_code=409, detail="That phrase is already silent")

    _push_history(job, unit)
    silence = np.zeros(int(float(unit["duration"]) * dubbing.SAMPLE_RATE),
                       dtype=np.float32)
    dubbing.write_unit(job["units_dir"], index, silence)
    unit["cleared"] = True

    duration = _rebuild_track(job)
    project.save(job_id, job)
    print(f"[APP] Job {job_id} phrase {index} cleared "
          f"({unit['duration']:.2f}s of silence)")

    return JSONResponse({
        "index": index,
        "text": unit["text"],
        "cleared": True,
        "slot_seconds": round(unit["duration"], 3),
        "track_seconds": duration,
        "video_stale": True,
        "can_undo": len(job.get("history", [])),
    })


@app.post("/jobs/{job_id}/undo")
async def undo_phrase(job_id: str):
    """
    Walk back the last phrase edit.

    Restores the previous text, seed and audio. The audio is restored from a
    copy rather than re-synthesized: the engine is stochastic, so regenerating
    would hand back a different take than the one being undone.
    """
    job = _require_editable(job_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="This job is still rendering.")

    history = job.get("history") or []
    if not history:
        raise HTTPException(status_code=409, detail="Nothing to undo.")

    entry = history.pop()
    unit = next((u for u in job["plan"] if u["index"] == entry["index"]), None)
    if unit is None:
        raise HTTPException(status_code=409,
                            detail="That phrase is no longer in the timeline.")

    unit["text"] = entry["text"]
    if entry.get("seed") is None:
        unit.pop("seed", None)
    else:
        unit["seed"] = entry["seed"]
    if entry.get("cleared"):
        unit["cleared"] = True
    else:
        unit.pop("cleared", None)

    restored_audio = False
    if entry.get("wav") and os.path.exists(entry["wav"]):
        try:
            shutil.copyfile(entry["wav"],
                            dubbing.segment_path(job["units_dir"], entry["index"]))
            os.remove(entry["wav"])
            restored_audio = True
        except OSError as e:                                 # noqa: BLE001
            print(f"[APP] Could not restore phrase {entry['index']}: {e}")

    duration = _rebuild_track(job)
    project.save(job_id, job)
    print(f"[APP] Job {job_id} undid phrase {entry['index']}")

    return JSONResponse({
        "index": entry["index"], "text": unit["text"],
        "seed": unit.get("seed"),
        # False means the text is back but the audio could not be restored, so
        # the phrase needs re-speaking. Saying so beats a silent mismatch.
        "audio_restored": restored_audio,
        "track_seconds": duration,
        "can_undo": len(history),
        "video_stale": True,
    })


@app.get("/jobs/{job_id}/segments/{index}/audio")
async def segment_audio(job_id: str, index: int):
    """Play back one phrase on its own, without rendering anything."""
    job = _require_editable(job_id)
    path = dubbing.segment_path(job["units_dir"], index)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="That phrase has no audio yet")
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="audio/wav")


@app.post("/jobs/{job_id}/rerender")
async def rerender_video(job_id: str, background_tasks: BackgroundTasks):
    """
    Re-run lip sync against the current audio.

    Editing phrases rewrites the audio immediately so it can be auditioned in
    seconds. The picture is only redone on request, because that is the part
    that costs minutes.
    """
    job = _require_editable(job_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="This job is already rendering.")

    job.update({"status": "processing", "step": "lip_syncing",
                "cancel_requested": False, "error": None})
    background_tasks.add_task(rerender_task, job_id)
    return JSONResponse({"job_id": job_id, "status": "processing"})


def rerender_task(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return
    try:
        def w2l_progress(done, total):
            _set(job_id, step=f"lip_syncing ({done}/{total} frames)")

        note = None
        with _HEAVY:
            try:
                generate_lip_synced_video(
                    job["input_path"], job["dub_path"], job["output_path"],
                    wav2lip_batch_size=engines.tunable("W2L_BATCH"),
                    face_det_batch_size=engines.tunable("W2L_DET_BATCH"),
                    crf=engines.tunable("OUTPUT_CRF"),
                    preset=engines.tunable("OUTPUT_PRESET"),
                    progress=w2l_progress)
            except lip_sync.NoFaceDetected as e:
                # Same rule as the first render: a clip with no face in it has
                # nothing to sync, so the edited audio is muxed onto it instead
                # of the re-render failing.
                _set(job_id, step="lip_sync_skipped")
                note = _deliver_without_lip_sync(
                    job["input_path"], job["dub_path"], job["output_path"],
                    str(e), no_face=True)
            except lip_sync.LipSyncError as e:
                # And the same rule for any other lip-sync failure: an edit must
                # not be lost to a stage that only moves mouths.
                print(f"[Wav2Lip] Lip sync failed, delivering without it: {e}")
                _set(job_id, step="lip_sync_skipped")
                note = _deliver_without_lip_sync(
                    job["input_path"], job["dub_path"], job["output_path"],
                    str(e))

        _set(job_id, step="verifying", lip_sync=note)
        check = av_sync.verify_sync(
            job["output_path"], tolerance=engines.tunable("SYNC_TOLERANCE"))

        final_url = f"/jobs/{job_id}/video"

        stem = os.path.splitext(os.path.basename(job.get("filename") or job_id))[0]
        filed_at = _file_render(job_id, job["output_path"], stem,
                                job.get("target_language", "out"))
        jobs[job_id].update({"status": "done", "step": "complete",
                             "sync": check, "url": final_url,
                             "filed_at": filed_at,
                             "video_stale": False})
        _save(job_id)
        print(f"[APP] Job {job_id} re-rendered")
    except JobCancelled:
        jobs.get(job_id, {}).update(
            {"status": "cancelled", "step": "cancelled",
             "error": "Cancelled before it finished."})
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        jobs.get(job_id, {}).update(
            {"status": "failed", "step": "error", "error": str(e)})


@app.get("/jobs/{job_id}/reference/candidates")
async def reference_candidates(job_id: str,
                               speaker: str = Query(
                                   None, description="Which speaker's windows "
                                                     "to list. Omit for the "
                                                     "single-voice list.")):
    """
    The best few windows the voice could be cloned from.

    Which window is chosen drives most of the cloning quality, and until now it
    was picked automatically with no way to disagree.
    """
    job = _require_editable(job_id)
    source = job.get("segments") or []
    if not source:
        raise HTTPException(status_code=409, detail="This run kept no transcript.")

    # candidate_windows needs word timings; the stored segments are sentence
    # level, so re-read the words from the transcript kept on the job.
    words = job.get("word_segments")
    if not words:
        raise HTTPException(
            status_code=409,
            detail="This run finished before reference picking was added. "
                   "Re-render to enable it.")

    current = job.get("reference", {})
    # A multi-speaker run has one reference per speaker, so the picker lists
    # windows for the speaker being asked about rather than for the video as a
    # whole. With no `speaker` the answer is the single-voice list it always was.
    speaker = (speaker or "").strip() or None
    registry = SPK.normalize_speakers(job.get("speakers"))
    # With a speaker selected, `current` is THAT speaker's reference, not the
    # video-wide one — otherwise the picker shows one actor's clip while you
    # are editing another's.
    chosen = current
    if speaker:
        for entry in registry:
            if entry.get("label") == speaker:
                chosen = entry.get("reference") or {}
                break

    return JSONResponse({
        "speaker": speaker,
        "current": {"start": chosen.get("start"),
                    "duration": chosen.get("duration"),
                    "text": chosen.get("text", "")},
        "speakers": SPK.describe_all(
            registry, aliases=job.get("speaker_aliases") or {},
            plan=job.get("plan")),
        "candidates": reference_audio.candidate_windows(
            words, job.get("video_duration") or 0.0, limit=5, speaker=speaker),
    })


@app.post("/jobs/{job_id}/reference")
async def choose_reference(job_id: str, background_tasks: BackgroundTasks,
                           body: dict = Body(...)):
    """
    Clone from a different window and speak the whole script again.

    Unlike a phrase edit this cannot reuse the unit cache: every phrase was
    spoken in the old voice window, so all of them have to be redone.
    """
    job = _require_editable(job_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="This job is already rendering.")

    try:
        start = float(body["start"])
        duration = float(body["duration"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Pass a numeric start and duration")
    if duration < 2.0:
        raise HTTPException(status_code=400,
                            detail="A reference under two seconds will not clone well")

    job.update({"status": "processing", "step": "selecting_reference",
                "cancel_requested": False, "error": None})
    background_tasks.add_task(reclone_task, job_id, start, duration,
                              (body.get("speaker") or None))
    return JSONResponse({"job_id": job_id, "status": "processing"})


def reclone_task(job_id: str, start: float, duration: float, speaker: str = None):
    """
    Re-cut a reference window and speak again in the new voice.

    With no `speaker` this is the original whole-run re-clone: the one reference
    every phrase used is replaced and the timeline is rebuilt.

    With a `speaker` it is the cheap, correct version: only that speaker's
    registry entry changes, and only THEIR phrases are re-synthesized. In a
    two-hander, re-cloning the second actor used to re-speak the first actor's
    lines as well — minutes of GPU work, and a fresh stochastic take on phrases
    nobody asked to change.
    """
    job = jobs.get(job_id)
    if not job:
        return
    try:
        ref_path = os.path.join(job["workdir"], f"ref_{int(start * 100)}.wav")
        reference_audio.extract_reference(
            job["input_path"], start, duration, ref_path, sample_rate=24000)

        # Measure what was actually written, never what was asked for. F5
        # derives the reference length from the real waveform, so if the two
        # disagree every generated phrase comes out the wrong length and gets
        # time-stretched to compensate, which is what sounds robotic.
        actual = duration
        try:
            actual = av_sync.duration(ref_path)
        except Exception as e:                               # noqa: BLE001
            print(f"[APP] Could not probe the new reference ({e}); "
                  f"using the requested {duration:.2f}s")

        # The window's words come from the transcript already on the job rather
        # than a second ASR pass, and must describe the same span of audio.
        words = job.get("word_segments") or []
        spoken = [w["word"] for seg in words for w in (seg.get("words") or [])
                  if w["start"] >= start and w["end"] <= start + actual]
        reference = {"path": ref_path, "text": " ".join(spoken).strip(),
                     "duration": actual, "start": start}
        if not reference["text"]:
            raise RuntimeError(
                "No transcribed words fall inside that window, so there is no "
                "reference text to pair with the audio.")

        def progress(done, total):
            _set(job_id, step=f"synthesizing_voice ({done + 1}/{total})")

        registry = SPK.normalize_speakers(job.get("speakers"))
        if speaker and registry:
            track = _respeak_speaker(job_id, job, speaker, reference, progress)
        else:
            # A whole-run re-clone means "every phrase now speaks in this
            # clip". Since Phase 3 each phrase is spoken in its OWN speaker's
            # reference, and `reference_for` prefers the registry over the
            # global one — so replacing only `job["reference"]` left every
            # unit resolving to the OLD file. The registry entries that were
            # resolving to it have to move with it, or the re-clone re-speaks
            # in the previous voice while claiming to have changed.
            rebound = _rebind_reference(registry, job.get("reference") or {},
                                        reference)
            if rebound:
                job["speakers"] = registry
            with _HEAVY:
                _set(job_id, step="synthesizing_voice")
                track = dubbing.synthesize_timeline(
                    job["plan"], reference["path"], reference["text"],
                    reference["duration"], job["target_language"],
                    job["video_duration"], progress, cache_dir=job["units_dir"],
                    speakers=registry)

        import soundfile as sf
        sf.write(job["dub_path"], track, dubbing.SAMPLE_RATE, subtype="PCM_16")

        update = {"status": "done", "step": "complete",
                  "video_stale": True}
        # A per-speaker re-clone changes THAT voice; the job-wide reference —
        # what the studio shows as "the clip the voice was cloned from" and
        # what unresolved speakers fall back to — must not be overwritten with
        # one actor's window.
        if not speaker or len(registry) <= 1:
            update.update({
                "reference": reference,
                "reference_path": reference["path"],
                "reference_text": reference["text"],
                "reference_seconds": round(reference["duration"], 2),
            })
        if speaker and registry:
            update["speakers"] = registry
        jobs[job_id].update(update)
        _save(job_id)
        who = f" for {speaker}" if speaker else ""
        print(f"[APP] Job {job_id} re-cloned{who} from {start:.2f}s "
              f"(+{reference['duration']:.2f}s)")
    except JobCancelled:
        jobs.get(job_id, {}).update(
            {"status": "cancelled", "step": "cancelled",
             "error": "Cancelled before it finished."})
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        jobs.get(job_id, {}).update(
            {"status": "failed", "step": "error", "error": str(e)})


def _rebind_reference(registry: list, old: dict, new: dict) -> int:
    """
    Point the speakers that were using `old` at `new`. Returns how many moved.

    Only entries with no reference of their own, or with exactly the old
    global clip, are touched. A speaker cloned from their own window keeps it:
    a whole-run re-clone must not collapse a two-hander back into one voice,
    which is the failure this whole phase exists to prevent.
    """
    old_path = (old or {}).get("path")
    rebound = 0
    for entry in registry or []:
        ref = entry.get("reference") or {}
        if ref.get("path") and not (old_path and ref.get("path") == old_path):
            continue
        entry["reference"] = dict(new)
        entry["reference_score"] = None
        entry["fallback"] = False
        rebound += 1
    if rebound:
        print(f"[APP] Re-clone rebound {rebound} speaker(s) to "
              f"{os.path.basename(new['path'])}")
    elif registry:
        print(f"[APP] Re-clone left {len(registry)} per-speaker reference(s) "
              f"unchanged; name a speaker to re-clone one of them")
    return rebound


def _respeak_speaker(job_id: str, job: dict, label: str, reference: dict,
                     progress=None):
    """
    Re-synthesize one speaker's phrases in a new reference and rebuild the track.

    Only units whose speaker is `label` are touched; every other unit keeps the
    audio already in the cache. The registry entry is updated in place so the
    manifest, the studio and any later edit all see the new voice.
    """
    registry = SPK.normalize_speakers(job.get("speakers"))
    for entry in registry:
        if entry.get("label") == label:
            entry["reference"] = dict(reference)
            entry["reference_score"] = None
            entry["fallback"] = False

    targets = [u for u in job["plan"]
               if (u.get("speaker") or SPK.DEFAULT_LABEL) == label]
    with _HEAVY:
        _set(job_id, step="synthesizing_voice")
        for n, unit in enumerate(targets):
            if progress:
                progress(n, len(targets))
            try:
                wave = dubbing.synth_unit(
                    unit, reference["path"], reference.get("text", ""),
                    reference["duration"], job["target_language"])
            except Exception as e:                           # noqa: BLE001
                print(f"[APP] Could not re-speak phrase {unit['index']} "
                      f"for {label}: {e}")
                continue
            if wave.size and float(np.max(np.abs(wave))) >= 1e-4:
                dubbing.write_unit(job["units_dir"], unit["index"], wave)

    job["speakers"] = registry
    return _rebuild_track(job)


@app.post("/jobs/{job_id}/segments/{index}/fit")
async def fit_segment(job_id: str, index: int, body: dict = Body(default={})):
    """
    Rewrite a phrase until it fits its slot when spoken.

    The pipeline's answer to an overlong line has always been to compress it,
    and compression past about 0.8x is what makes a dub sound robotic. This
    changes the words instead of the speed.

    The loop is measured, not hopeful: `natural_duration` mirrors F5's own
    internal length heuristic exactly and costs nothing to evaluate, so each
    rewrite is checked against the real target before anything is synthesized.
    """
    job = _require_editable(job_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="This job is still rendering.")
    if not llm.configured():
        raise HTTPException(
            status_code=503,
            detail="Script intelligence is off. Turn it on in Settings: run a "
                   "local model through Ollama, or add a key for a hosted "
                   "provider - a hosted one sends the script text off this "
                   "machine, unlike every other stage.")

    unit = next((u for u in job["plan"] if u["index"] == index), None)
    if unit is None:
        raise HTTPException(status_code=404, detail=f"No phrase {index} in this timeline")

    engine = tts_engines.get_engine(job["target_language"])
    if not hasattr(engine, "natural_duration"):
        raise HTTPException(
            status_code=409,
            detail="This voice engine cannot estimate spoken length, so a fit "
                   "cannot be checked.")

    ref = job["reference"]
    lang = L.synth_lang(job["target_language"])
    slot = float(unit["duration"])
    measure = lambda t: engine.natural_duration(t, ref["path"], ref["text"], lang=lang)

    # A little headroom: landing exactly on the slot still leaves the fitter
    # shaving the tail off the last word.
    target = slot * float(body.get("headroom", 0.96))
    original = unit["text"]
    natural = measure(original)

    attempts = [{"text": original, "seconds": round(natural, 3), "source": "original"}]
    if natural <= target:
        return JSONResponse({
            "index": index, "changed": False, "text": original,
            "slot_seconds": round(slot, 3), "spoken_seconds": round(natural, 3),
            "attempts": attempts,
            "detail": "This phrase already fits its slot.",
        })

    source_text = _source_text_at(job.get("segments") or [], unit["start"], slot)
    best_text, best_seconds = original, natural

    # Each attempt rewrites the ORIGINAL, not the previous attempt. Chaining
    # compounds drift: in testing, the third link in the chain had invented a
    # clause that was never in the source. Asking harder each round instead
    # keeps every candidate one step from the truth.
    failures = []
    for attempt in range(int(body.get("tries", 3))):
        try:
            candidate = llm.shorten(
                original, L.display_name(job["target_language"]),
                (target / natural) * (0.9 ** attempt), source_text=source_text)
        except llm.Unavailable as e:
            # One bad draw is not a broken model. Asking for an aggressive
            # ratio makes an empty or unusable reply likely enough that
            # treating the first one as fatal threw away rewrites that had
            # already succeeded on an earlier, gentler attempt.
            failures.append(str(e))
            attempts.append({"text": None, "seconds": None,
                             "source": "failed", "detail": str(e)[:160]})
            continue
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        seconds = measure(candidate)
        attempts.append({"text": candidate, "seconds": round(seconds, 3),
                         "source": "rewrite"})
        if seconds < best_seconds:
            best_text, best_seconds = candidate, seconds
        if seconds <= target:
            break

    if best_text == original:
        # Nothing usable came back at all, as opposed to rewrites that came
        # back but were no shorter. The difference decides whether the user
        # should look at their model or at their line.
        if len(failures) == len(attempts) - 1:
            raise HTTPException(status_code=503, detail=failures[-1])
        return JSONResponse({
            "index": index, "changed": False, "text": original,
            "slot_seconds": round(slot, 3), "spoken_seconds": round(natural, 3),
            "attempts": attempts,
            "detail": "No rewrite came out shorter than the original."
                      + (f" {len(failures)} attempt(s) returned nothing usable."
                         if failures else ""),
        })

    unit["text"] = best_text
    try:
        if not _HEAVY.acquire(timeout=2):
            unit["text"] = original
            raise HTTPException(status_code=409, detail=BUSY_DETAIL)
        try:
            spoken_ref = SPK.reference_for(unit, job.get("speakers"), ref) or ref
            wave = dubbing.synth_unit(unit, spoken_ref["path"],
                                      spoken_ref.get("text", ""),
                                      float(spoken_ref.get("duration") or 0.0),
                                      job["target_language"],
                                      seed=unit.get("seed"))
        finally:
            _HEAVY.release()
    except HTTPException:
        raise
    except Exception as e:                                   # noqa: BLE001
        unit["text"] = original
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not speak the rewrite: {e}")

    dubbing.write_unit(job["units_dir"], index, wave)
    _rebuild_track(job)
    project.save(job_id, job)
    print(f"[APP] Job {job_id} phrase {index} fitted "
          f"({natural:.2f}s -> {best_seconds:.2f}s for a {slot:.2f}s slot)")

    return JSONResponse({
        "index": index, "changed": True, "text": best_text,
        "slot_seconds": round(slot, 3),
        "spoken_seconds": round(best_seconds, 3),
        "was_seconds": round(natural, 3),
        "fits": best_seconds <= target,
        "attempts": attempts,
        # Words that may have been dropped rather than rephrased. Advisory:
        # a shorter synonym looks the same to a string comparison, so this is
        # for the person reading the line, not a blocking check.
        "check": llm.dropped_words(original, best_text)[:4],
        "video_stale": True,
        "can_undo": len(job.get("history", [])),
    })


EXPORTS = {
    # kind: (filename suffix, media type, builder)
    "transcript.txt": ("transcript", "text/plain",
                       lambda segs, tr: subtitles.plain(segs)),
    "translation.txt": ("translation", "text/plain",
                        lambda segs, tr: subtitles.plain(segs, tr)),
    "source.srt": ("source", "application/x-subrip",
                   lambda segs, tr: subtitles.srt(segs)),
    "source.vtt": ("source", "text/vtt",
                   lambda segs, tr: subtitles.vtt(segs)),
    "dub.srt": ("dub", "application/x-subrip",
                lambda segs, tr: subtitles.srt(segs, tr)),
    "dub.vtt": ("dub", "text/vtt",
                lambda segs, tr: subtitles.vtt(segs, tr)),
}


@app.get("/jobs/{job_id}/export/{kind}")
async def export_job_text(job_id: str, kind: str):
    """
    Transcript and subtitle files for a finished job.

    Everything served here is formatting over data the run already produced:
    Whisper's segment timings and the translation aligned to them. No model is
    invoked and nothing is recomputed.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    spec = EXPORTS.get(kind)
    if not spec:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown export. Available: {', '.join(sorted(EXPORTS))}")

    segments = job.get("segments")
    if not segments:
        raise HTTPException(
            status_code=409,
            detail="This job has no stored transcript. Only runs finished "
                   "since transcript export was added can be exported.")

    suffix, media_type, build = spec
    translated = job.get("translated_segments") or []
    if "dub" in kind or "translation" in kind:
        if not translated:
            raise HTTPException(status_code=409,
                                detail="This job has no stored translation.")

    body = build(segments, translated)
    ext = kind.rsplit(".", 1)[-1]
    name = f"{_download_name(job_id)}_{suffix}.{ext}"
    return Response(
        content=body,
        media_type=f"{media_type}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/boot")
async def boot_status():
    """What the engine is loading, for the splash screen to report."""
    return JSONResponse(BOOT)


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Ask a running job to stop.

    Cooperative: the flag is checked at every stage report, and the two long
    stages report per segment and per frame batch, so this lands within
    seconds. A job that has already finished is left alone.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in ("done", "failed", "cancelled"):
        return JSONResponse({"status": job["status"], "cancelled": False})
    job["cancel_requested"] = True
    return JSONResponse({"status": "cancelling", "cancelled": True})


@app.post("/voiceover/")
async def create_voiceover(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    script: str = Form(..., description="What the voice should say"),
    language: str = Query(..., description="Language of the script"),
    user_id: str = Query("anonymous"),
    name: str = Query(None, description="What to call this project"),
    project_id: str = Query(None, description="An existing project this output joins"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No reference clip provided")
    script = (script or "").strip()
    if not script:
        raise HTTPException(status_code=400, detail="The script is empty")
    if len(script) > 20000:
        raise HTTPException(status_code=400, detail="Script is too long (max 20000 characters)")

    try:
        language = L.normalize(language)
        L.engine_for(language)
    except L.UnsupportedLanguage as e:
        raise HTTPException(status_code=400, detail=str(e))

    # A voice-over is nothing but synthesis, so it always needs the engine.
    _require_speakable(language)

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Reference clip too large (max 200MB)")
    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="Reference clip too small or empty")

    job_id = str(uuid.uuid4())[:12]
    safe_filename = f"{job_id}_{_safe_name(file.filename)}"
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(input_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "status": "queued", "step": "uploaded", "kind": "voiceover",
        "filename": file.filename, "target_language": language,
        "title": (name or "").strip() or None,
        # Several outputs from one clip are one project. A job that names no
        # project founds its own, so a lone render behaves as it always did.
        "project_id": _project_for(project_id, job_id),
        # Wall-clock start, so a project that is still rendering can be listed
        # and sorted while it has no manifest to date it.
        "started_at": time.time(),
        "user_id": user_id,
        "created_at": asyncio.get_event_loop().time(),
    }
    background_tasks.add_task(voiceover_task, job_id, input_path, script,
                             language, safe_filename, user_id)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/api/settings/advanced")
async def get_advanced_settings():
    """
    The pipeline's knobs, and which of them apply to the chosen engines.

    A setting for a model nobody is using is noise, so each one says whether
    it is relevant rather than leaving the studio to guess.
    """
    return JSONResponse({
        "tunables": engines.tunables_state(),
        "workdir": UPLOAD_DIR,
        "workdir_bytes": _dir_bytes(UPLOAD_DIR),
        "output_dir": engines.output_dir(create=False),
    })


@app.post("/api/settings/advanced")
async def set_advanced_settings(body: dict = Body(...)):
    """Store knob values. An environment variable still wins over these."""
    try:
        state = engines.save_tunables(body or {})
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save: {e}")
    return JSONResponse({"tunables": state})


@app.post("/api/settings/sweep")
async def sweep_now():
    """Clear abandoned working folders on request rather than on the timer."""
    freed = sweep_workdirs()
    return JSONResponse({"removed": freed, "workdir_bytes": _dir_bytes(UPLOAD_DIR)})


def _dir_bytes(path: str) -> int:
    """How much a directory is holding. Best effort: a file that vanishes
    mid-walk is not worth failing a settings page over."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


@app.get("/api/models")
async def list_models():
    """
    Every model, its size, and whether it is already on this machine.

    Downloading several gigabytes should not require opening a terminal, so
    the studio drives the same downloader the command line does.
    """
    rows = models.inventory()
    return JSONResponse({
        "models": rows,
        "missing": sum(1 for r in rows if not r["present"]),
        "cache": models.cache_root(),
        "free_gb": round(models.free_space_gb(), 1),
        "progress": models.status(),
    })


@app.post("/api/models/download")
async def download_models(body: dict = Body(default={})):
    """
    Fetch the named models, or everything missing.

    Returns immediately; watch /api/models/progress. Refused while a run is
    already going, because two downloads writing one .part file would corrupt
    each other.
    """
    ids = body.get("ids") or None
    if ids is not None and not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="ids must be a list")
    if not models.run(ids):
        raise HTTPException(status_code=409,
                            detail="A download is already running.")
    return JSONResponse(models.status())


@app.get("/api/models/progress")
async def model_progress():
    """Which file, how far through, and whether anything went wrong."""
    return JSONResponse(models.status())


@app.post("/api/models/cancel")
async def cancel_models():
    """Stop after the file in flight. Partial files resume on the next run."""
    models.cancel()
    return JSONResponse(models.status())


@app.get("/api/settings/engines")
async def get_engine_settings():
    """
    Which model runs each stage, and which of them are already downloaded.

    `configured` is how the app knows whether to show setup on launch.
    """
    return JSONResponse({
        "configured": engines.configured(),
        "stages": engines.catalog(),
        "output_dir": engines.output_dir(create=False),
        "default_output_dir": engines.default_output_dir(),
    })


@app.post("/api/settings/engines")
async def set_engine_settings(body: dict = Body(...)):
    """
    Store stage choices and apply them to the running engine.

    Each stage used to read its model once at import and hold it for the life
    of the process, so changing one meant restarting the application. Every
    stage now re-reads its choice and drops its loaded weights on request, and
    the next job picks the new one up. A job already running keeps its own
    reference and finishes on what it started with, which is the behaviour you
    want mid-render.
    """
    try:
        chosen = engines.save(body or {})
    except ValueError as e:
        # A model that is not one of the stage's options.
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save settings: {e}")

    applied, busy = [], any(j.get("status") == "processing" for j in jobs.values())
    for stage, module in (("stt", speech_to_text_v2), ("mt", translation_v2),
                          ("lipsync", lip_sync), ("tts", tts_engines)):
        if stage not in (body or {}):
            continue
        try:
            module.refresh()
            applied.append(stage)
        except Exception as e:                               # noqa: BLE001
            print(f"[APP] Could not apply the {stage} change live: {e}")

    if applied:
        print(f"[APP] Applied without a restart: {', '.join(applied)}")

    return JSONResponse({
        "configured": True,
        "chosen": chosen,
        "applied": applied,
        # Nothing needs a restart any more. Kept so an older client that reads
        # this field is told the truth rather than a stale warning.
        "restart_required": False,
        # A change lands on the next job, so say so while one is in flight.
        "takes_effect_after_current_job": busy,
        "stages": engines.catalog(),
        "output_dir": engines.output_dir(create=False),
        "default_output_dir": engines.default_output_dir(),
    })


@app.get("/api/settings/llm")
async def get_llm_settings():
    """
    Which model does the script rewriting, and whether it can be reached.

    Never returns an API key, only whether one is stored.
    """
    return JSONResponse(llm.status())


@app.post("/api/settings/llm")
async def set_llm_settings(body: dict = Body(...)):
    """
    Choose a provider and model, and optionally store a key for it.

    Omitting `api_key` leaves any stored key untouched, so changing the model
    does not silently log you out; sending an empty one deletes it.
    """
    # str() first: a non-string provider used to reach .strip() and 500.
    provider = str(body.get("provider") or "").strip()
    if provider not in llm.PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider. Choose one of: {', '.join(llm.PROVIDERS)}")
    # Anything from the body may not be a string, and .strip() on an int is a
    # 500 rather than a bad request. Reject rather than coerce: str([]) is
    # "[]", which sailed through as a hostname and left the script model
    # pointing at nonsense until someone noticed it could not connect.
    def _text(field):
        value = body.get(field)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise HTTPException(
                status_code=400,
                detail=f"{field} must be text.")
        return value.strip()

    try:
        llm.save_settings(
            provider,
            _text("model"),
            key=body.get("api_key"),
            ollama_host=_text("ollama_host") or None,
            custom_url=body.get("custom_url"),
        )
    except OSError as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not save settings: {e}")
    return JSONResponse(llm.status())


@app.post("/api/settings/llm/test")
async def test_llm_settings():
    """
    Prove the current setting works, before someone relies on it mid-render.

    Deliberately a real completion rather than a reachability ping: a key can
    be valid and the model name wrong, and that only shows up on a real call.
    """
    try:
        reply = llm.complete(
            "Reply with exactly the word: ok",
            "Say ok.", max_tokens=16)
    except llm.Unavailable as e:
        raise HTTPException(status_code=502, detail=str(e))
    return JSONResponse({"ok": True, "reply": llm.clean_line(reply)[:80],
                         **llm.status()})


XTTS_HINT = ('Install the XTTS extra to speak this language: '
             'pip install "TTS>=0.22.0"')
GATED_HINT = ('Download the voice model first: its weights are gated. Accept '
              'the licence at https://huggingface.co/ai4bharat/IndicF5, put '
              'HF_TOKEN=hf_... in the .env file beside run.sh, then run '
              'scripts/download_models.py. See docs/GATED-MODELS.md.')


def _require_speakable(target_language: str) -> None:
    """
    Refuse a language nothing installed can speak, before any work starts.

    Two ways this can fail: XTTS is an optional extra - non-commercial
    licence, very large dependency tree - so a default install speaks the
    twelve Indian languages and not the other sixteen; and IndicF5's weights
    are gated, so they may simply not be downloaded yet. Both are caught here
    rather than as a ModuleNotFoundError or a 401 after transcription and
    translation have already run.
    """
    if tts_engines.language_available(target_language):
        return
    engine = L.engine_for(target_language)
    if engine == L.ENGINE_INDICF5:
        detail = (f"{L.display_name(target_language)} is spoken by IndicF5, "
                  f"whose weights are not on this machine yet. {GATED_HINT}")
    else:
        detail = (f"{L.display_name(target_language)} is spoken by XTTS, "
                  f"which is not installed. {XTTS_HINT}. The Indian "
                  f"languages need nothing extra.")
    raise HTTPException(status_code=400, detail=detail)


@app.get("/languages")
async def get_languages():
    """
    Supported dubbing targets, Indian languages first.

    Each says whether the engine that speaks it is installed, so the studio
    can show what this machine can actually do rather than advertising
    sixteen languages it would fail on.
    """
    rows = []
    for entry in L.supported_targets():
        row = dict(entry)
        row["available"] = tts_engines.engine_available(entry.get("engine", ""))
        if not row["available"]:
            row["needs"] = (GATED_HINT if entry.get("engine") == L.ENGINE_INDICF5
                            else XTTS_HINT)
        rows.append(row)
    return JSONResponse(rows)


@app.get("/api/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "projects": len([j for j in jobs.values() if j.get("status") == "done"]),
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
        # Off unless a key is configured. It is the only stage that sends text
        # off this machine, so the client says so before using it.
        "script_intelligence": llm.configured(),
        "llm": {k: v for k, v in llm.status().items()
                if k in ("provider", "model", "local", "has_key")},
    })


def _project_name(job: dict, job_id: str) -> str:
    """What to call this project: what the user named it, else its source
    file, else the id. The id is the last resort, not the label."""
    name = (job.get("title") or "").strip()
    if name:
        return name
    return job.get("filename") or job_id


def _reusable_sibling(job_id: str, video_dur: float, source_language: str):
    """
    A finished job in the same project that already transcribed this clip.

    A dub and a subtitled video of one clip used to run Whisper and NLLB
    twice each - minutes of work for the same answer, and two chances for
    the transcripts to disagree. The second output takes the first one's
    transcript, and its translation too if the target language matches.

    The match is deliberately strict: same project, same clip length to the
    hundredth, same source language. A trim changes the length and so is a
    different clip.
    """
    me = jobs.get(job_id) or {}
    pid = me.get("project_id", job_id)
    best = None
    for jid, j in jobs.items():
        if jid == job_id or j.get("project_id", jid) != pid:
            continue
        if j.get("status") != "done" or not j.get("segments"):
            continue
        if abs(float(j.get("video_duration") or 0) - float(video_dur)) > 0.01:
            continue
        if source_language and j.get("source_language") not in (None, source_language):
            continue
        if best is None or (j.get("saved_at") or 0) > (best.get("saved_at") or 0):
            best = j
    return best


def _project_input(project_id) -> dict | None:
    """
    The clip a project was built from, if it is still on disk.

    Every output keeps its own copy of the input, so the newest finished one
    is the freshest. Returns the path and the name it was uploaded under.
    """
    pid = str(project_id or "").strip()
    if not pid:
        return None
    best = None
    for jid, j in jobs.items():
        if j.get("project_id", jid) != pid:
            continue
        # A voice-over's input is the clip whose voice it copied, not a
        # video to dub or subtitle.
        if j.get("kind") == "voiceover":
            continue
        path = j.get("input_path")
        if not path or not os.path.exists(path):
            continue
        if best is None or (j.get("saved_at") or 0) > (best.get("saved_at") or 0):
            best = j
    if best is None:
        return None
    return {"input_path": best["input_path"],
            "filename": best.get("filename") or os.path.basename(best["input_path"])}


def _project_for(requested, job_id: str) -> str:
    """The project a new job joins, or its own id if it starts one."""
    pid = str(requested or "").strip()
    if pid and any(j.get("project_id") == pid or jid == pid
                   for jid, j in jobs.items()):
        return pid
    return job_id


@app.patch("/jobs/{job_id}")
async def rename_project(job_id: str, body: dict = Body(...)):
    """
    Name an output, and optionally say where it lives.

    Naming and filing are the same act from the person's side: "call this X
    and put it here". A new `output_dir` re-files the finished render there
    and points `filed_at` at the copy; the working copy is untouched, because
    playback and re-renders come from it.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No such job")
    name = body.get("name")
    if name is not None:
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="name must be text")
        name = name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="A project needs a name")
        if len(name) > 120:
            raise HTTPException(status_code=400, detail="That name is too long")
        job["title"] = name

    folder = body.get("output_dir")
    if folder is not None:
        if not isinstance(folder, str):
            raise HTTPException(status_code=400, detail="output_dir must be text")
        folder = os.path.expanduser(folder.strip())
        src = job.get("output_path")
        if folder and src and os.path.isfile(src):
            try:
                os.makedirs(folder, exist_ok=True)
                stem = _download_name(job_id)
                lang = job.get("target_language") or "out"
                suffix = {"subtitled": "_subtitled", "audio": "_audio"}.get(
                    job.get("kind"), "")
                ext = os.path.splitext(src)[1] or ".mp4"
                target = os.path.join(folder, f"{stem}_{lang}{suffix}{ext}")
                n = 2
                while os.path.exists(target) and not os.path.samefile(target, src):
                    target = os.path.join(folder, f"{stem}_{lang}{suffix}_{n}{ext}")
                    n += 1
                if not os.path.exists(target):
                    shutil.copyfile(src, target)
                job["filed_at"] = target
                job.pop("filed_error", None)
            except OSError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not save there: {e}")

    project.save(job_id, job)
    return JSONResponse({"job_id": job_id, "name": job.get("title"),
                         "filed_at": job.get("filed_at")})


@app.get("/projects/{project_id}/outputs")
async def project_outputs(project_id: str):
    """
    Everything a project has produced, for the media pool.

    Opening a project used to show the one job it was opened on and nothing
    of the others made from the same clip. This is the list that fixes that.
    """
    rows = []
    for jid, j in jobs.items():
        if j.get("project_id", jid) != project_id or j.get("status") != "done":
            continue
        rows.append({
            "job_id": jid,
            "kind": j.get("kind") or "dub",
            "name": _project_name(j, jid),
            "url": j.get("url"),
            "audio": f"/jobs/{jid}/audio/dub" if j.get("dub_path") else None,
            "filed_at": j.get("filed_at"),
            "target_language": j.get("target_language"),
            "subtitle_language": j.get("subtitle_language"),
            "saved_at": j.get("saved_at"),
        })
    rows.sort(key=lambda r: r.get("saved_at") or 0)
    return JSONResponse({"project_id": project_id, "outputs": rows})


@app.get("/videos/")
async def get_videos(user_id: str = "anonymous"):
    """
    Every project on this machine, including the one being rendered right now.

    The manifest is the only record. There is no remote index to fall out of
    sync with, and no outage that can hide someone's work. A run in progress
    has no manifest yet, so it is listed from the live job registry instead —
    a project used to disappear from this list the moment you started working
    on it, and never came back if the runtime died mid-render.

    `user_id` is recorded on each project but deliberately not filtered on.
    It is a leftover from the hosted version, where accounts existed; here
    there is one person and no sign-in, and the id it filtered by was a random
    UUID in localStorage. Clearing site data, or opening the studio in another
    browser, minted a new one and made every existing project vanish from this
    list while its files sat untouched on disk. The parameter stays accepted so
    older clients keep working.
    """
    # One row per project, not per output. Opening the row opens the job that
    # founded the project, or the one running now; the studio then lists every
    # output the project holds and follows the live one to completion.
    return JSONResponse(project.video_rows(jobs, _project_name))


@app.delete("/videos/{job_id}")
async def delete_project(job_id: str):
    """
    Delete a project and everything it produced.

    Removes the whole working folder: source, reference, phrase cache, dub,
    render and manifest. Irreversible, which is why the studio asks twice.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Project not found")
    if job.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="This project is still rendering. Cancel it first, then "
                   "delete it.")
    jobs.pop(job_id, None)

    uploads = os.path.abspath(UPLOAD_DIR) + os.sep

    # Refuse to touch anything outside the uploads directory, so a corrupted
    # manifest cannot aim this at somewhere else on the disk.
    workdir = job.get("workdir")
    if workdir and os.path.isdir(workdir) and os.path.abspath(workdir).startswith(uploads):
        try:
            shutil.rmtree(workdir)
        except OSError as e:
            raise HTTPException(status_code=500,
                                detail=f"Could not delete the project files: {e}")

    source = job.get("input_path")
    if source and os.path.isfile(source) and os.path.abspath(source).startswith(uploads):
        try:
            os.remove(source)
        except OSError:
            pass

    print(f"[APP] Deleted project {job_id}")
    return JSONResponse({"status": "deleted", "job_id": job_id})

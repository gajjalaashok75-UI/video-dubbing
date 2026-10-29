"""
Overlap checkpoint: hear two actors talking at the same time.

The Phase 5 counterpart to `multispeaker_check.py`. It answers one question:

    when two people talk over each other, do they both stay in time, and does
    each keep their own voice?

That is hard to judge from a finished render — a pushed-back line sounds like a
bad dub, not like a scheduling bug. So this prints the overlap timeline as a
table, writes one isolated track per speaker, and lets you listen to them
separately and together.

Usage
-----
    python scripts/overlap_check.py plan  clip.mp4
    python scripts/overlap_check.py check clip.mp4 --language hi
    python scripts/overlap_check.py check clip.mp4 --language hi --render

`plan` needs no models: it reports what would run and whether the machine is
ready. `check` runs the pipeline far enough to produce the dialogue tracks.

Artifacts in `overlap_check/<clip>/`:

    dialogue/SPEAKER_00.wav   that actor's generated lines, alone
    dialogue/SPEAKER_01.wav
    dialogue/combined.wav     all speakers assembled, overlap gain applied
    dialogue/no_overlap_gain.wav   the same placement without the gain, for A/B
    background.wav            the preserved soundtrack (Phase 4), if enabled
    mixed.wav                 dialogue over the background
    dubbed.mp4                the finished video (--render)
    timeline.txt              the active-speaker table
    report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "engine"))

SR = 24000


def _load():
    import av_sync
    import dubbing
    import speaker_diarization
    import speakers as SPK
    return av_sync, dubbing, speaker_diarization, SPK


def _load_pipeline():
    import reference_audio
    import speech_to_text_v2
    import translation_v2
    return reference_audio, speech_to_text_v2, translation_v2


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def _speaker_track(plan: list, waves: dict, label: str,
                   video_duration: float):
    """
    One speaker's lines, on a track of the full video length.

    Built by the same `assemble` the pipeline uses, restricted to that speaker's
    units — so what you hear is exactly that actor's contribution, in place.
    """
    import dubbing
    mine = [u for u in plan
            if (u.get("speaker") or "SPEAKER_00") == label]
    # Overlap gain is a property of the MIX, not of one voice, so it is off for
    # an isolated track: a single speaker's own audio is never attenuated.
    return dubbing.assemble(mine, waves, video_duration, overlap_gain=False)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args) -> int:
    av_sync, dubbing, diar, SPK = _load()

    print("=" * 72)
    print("OVERLAP CHECKPOINT — PLAN")
    print("=" * 72)

    if not os.path.exists(args.clip):
        print(f"FAIL  clip not found: {args.clip}")
        return 2
    info = av_sync.probe(args.clip)
    if not info.get("video"):
        print(f"FAIL  no video stream in {args.clip}")
        return 2
    print(f"clip      {args.clip}")
    print(f"duration  {info['duration']:.2f}s")

    print()
    print("Scheduling is per speaker:")
    print("  * a different actor may start while another is still talking")
    print("  * the SAME actor still cannot talk over themselves")
    print("  * with one speaker there is one cursor, so nothing changes")
    print()
    print("Assembly keeps additive placement and manages level locally:")
    print("  * 1 voice -> 1.000, 2 -> 0.707 each, 3 -> 0.577 each")
    print("  * a local ceiling takes any remaining excess INSIDE the overlap")
    print("  * solo audio is never attenuated by a distant overlap")

    print()
    print("Backends:")
    ready = True
    print(f"  diarization  {'available' if diar.available() else 'ABSENT'}"
          + ("" if diar.available()
             else "   -> one speaker, so no overlap to check"))
    if not diar.available():
        print("               install with: pip install 'pyannote.audio>=3.3.2'")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"  HF_TOKEN     {'set' if token else 'NOT SET'}")
    try:
        import torch
        print(f"  torch        {torch.__version__}")
    except ImportError:
        print("  torch        ABSENT -> IndicF5 cannot run")
        ready = False
    try:
        import models
        missing = [m["id"] for m in models.missing(core_only=True)]
        print(f"  models       {'all core weights present' if not missing else 'MISSING: ' + ', '.join(missing)}")
        ready = ready and not missing
    except Exception as e:                                   # noqa: BLE001
        print(f"  models       could not check ({e})")

    print()
    if ready:
        print("READY. Run:")
        print(f"  python scripts/overlap_check.py check {args.clip} "
              f"--language {args.language}")
    else:
        print("NOT READY for a real run; see the notes above.")
        print("The model-free tests still cover the scheduling and gain rules:")
        print("  ./venv/bin/python -m unittest tests.test_overlap -v")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    av_sync, dubbing, diar, SPK = _load()
    reference_audio, stt, translation = _load_pipeline()

    clip = os.path.abspath(args.clip)
    if not os.path.exists(clip):
        print(f"clip not found: {clip}")
        return 2

    stem_name = os.path.splitext(os.path.basename(clip))[0]
    out_dir = os.path.join(ROOT, "overlap_check", stem_name)
    if os.path.isdir(out_dir) and not args.keep:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    workdir = os.path.join(out_dir, "work")
    os.makedirs(workdir, exist_ok=True)
    dialogue_dir = os.path.join(out_dir, "dialogue")
    os.makedirs(dialogue_dir, exist_ok=True)

    video_dur = av_sync.duration(clip)
    report = {"clip": clip, "language": args.language, "duration": video_dur}

    # 1. Transcribe and diarize.
    print("[1/4] Transcribing and diarizing...")
    stt_audio = os.path.join(workdir, "stt.wav")
    av_sync.extract_audio(clip, stt_audio, 16000, 1)
    result = stt.transcribe_audio(stt_audio)
    segments = result.get("segments") or []
    if not segments:
        print("      no speech detected; nothing to check")
        return 1
    source_lang = result.get("language") or "en"
    print(f"      {len(segments)} segments, language={source_lang}")

    turns = diar.diarize(clip, min_speakers=args.min_speakers,
                         max_speakers=args.max_speakers)
    if not turns:
        print("      no diarization; a single speaker means no overlap exists")
        return 1
    segments = diar.assign_segments(segments, turns)
    print("      " + diar.summarize(segments, turns))
    labels = sorted({SPK.label_of(s) for s in segments if SPK.label_of(s)})
    if len(labels) < 2:
        print(f"      only one speaker resolved ({labels}); there is nothing "
              f"to overlap. Try a clip with a conversation in it.")
    report["labels"] = labels
    report["turns"] = turns

    # 2. Per-speaker references, cut from the vocals stem when one exists.
    print("[2/4] Cutting one reference per speaker...")
    stems = None
    if args.preserve_background:
        import source_separation
        stems = source_separation.separate(clip, workdir, log=print,
                                           chunk_seconds=args.chunk_seconds)
        source_separation.release()
    speech_source = stems["vocals"] if stems else clip

    registry = reference_audio.build_references(
        speech_source, segments, video_dur, os.path.join(out_dir, "references"),
        labels=labels, sample_rate=SR, is_audio=bool(stems),
        generic_path=SPK.generic_voice_path())
    for entry in registry:
        ref = entry.get("reference") or {}
        print(f"      {entry['label']}: {entry.get('total_speech', 0.0):.1f}s "
              f"speech -> {ref.get('path')}")

    # 3. Translate and synthesize per speaker.
    print("[3/4] Translating and synthesizing each speaker...")
    translated = translation.translate_segments(
        segments, source_lang, args.language, for_speech=True)
    translated = translation.fix_code_switching(translated, args.language)

    units_dir = os.path.join(workdir, "units")
    dialogue_path = os.path.join(dialogue_dir, "combined.wav")
    plan = dubbing.build_dubbed_track(
        segments, translated, registry[0]["reference"] or {}, args.language,
        video_dur, dialogue_path, source_lang=source_lang,
        source_audio=stt_audio, cache_dir=units_dir, speakers=registry)

    # 4. Report and artifacts.
    print("[4/4] Building the timeline and artifacts...")
    waves = {}
    for unit in plan:
        wave = dubbing.read_unit(units_dir, unit["index"])
        if wave is not None:
            waves[unit["index"]] = wave

    regions = dubbing.overlap_profile(plan)
    table = _timeline_table(plan, video_dur)
    with open(os.path.join(out_dir, "timeline.txt"), "w", encoding="utf-8") as f:
        f.write(table)
    print()
    print(table)

    # One isolated track per speaker, plus the placement without gain for A/B.
    per_speaker = {}
    for label in labels:
        track = _speaker_track(plan, waves, label, video_dur)
        path = os.path.join(dialogue_dir, f"{label}.wav")
        _write(path, track)
        per_speaker[label] = path

    no_gain = os.path.join(dialogue_dir, "no_overlap_gain.wav")
    _write(no_gain, dubbing.assemble(plan, waves, video_dur,
                                     overlap_gain=False))
    report["artifacts"] = {"combined": dialogue_path, "no_gain": no_gain,
                           "per_speaker": per_speaker}

    if stems:
        bg = os.path.join(out_dir, "background.wav")
        av_sync.extract_audio(stems["background"], bg, SR, 1)
        mixed = os.path.join(out_dir, "mixed.wav")
        av_sync.mix_layers(dialogue_path, mixed, target=video_dur,
                           background_path=stems["background"],
                           dialogue_lufs=float(_tunable("DIALOGUE_LUFS", -12.0)),
                           background_lufs=float(_tunable("BACKGROUND_LUFS", -26.0)),
                           duck=bool(_tunable("BACKGROUND_DUCK", False)))
        report["artifacts"]["background"] = bg
        report["artifacts"]["mixed"] = mixed

    if args.render:
        from lip_sync import generate_lip_synced_video
        dubbed = os.path.join(out_dir, "dubbed.mp4")
        audio = report["artifacts"].get("mixed", dialogue_path)
        generate_lip_synced_video(clip, audio, dubbed)
        report["artifacts"]["dubbed"] = dubbed

    # Checks.
    import numpy as np
    dialogue = _read(dialogue_path)
    checks = [
        ("dialogue is exactly the video length",
         abs(len(dialogue) / SR - video_dur) < 0.05),
        ("no invalid samples",
         not np.isnan(dialogue).any() and not np.isinf(dialogue).any()),
        ("the track does not clip",
         float(np.abs(dialogue).max()) <= 1.0 + 1e-6),
        ("at least one overlap region was found",
         bool(regions) or len(labels) < 2),
    ]
    print()
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    report["checks"] = {label: ok for label, ok in checks}
    report["overlap_regions"] = regions

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"Artifacts in {out_dir} — listen in this order:")
    for label in labels:
        print(f"  dialogue/{label}.wav        {label} alone, in place")
    print("  dialogue/no_overlap_gain.wav  the same placement, no gain applied")
    print("  dialogue/combined.wav         all speakers, overlap gain applied")
    if stems:
        print("  background.wav                the preserved soundtrack")
        print("  mixed.wav                     dialogue over the background")
    if args.render:
        print("  dubbed.mp4                    the finished video")
    print("  timeline.txt                  the active-speaker table")
    print()
    print("If two speakers overlap and BOTH keep their own voice and their own")
    print("start time, the scheduler is right. Compare combined.wav against")
    print("no_overlap_gain.wav to hear what the local gain is doing.")
    return 0


def _timeline_table(plan: list, video_duration: float) -> str:
    """
    The active-speaker table, as the brief specifies:

        time        active speakers
        00:10.20    SPEAKER_00
        00:11.45    SPEAKER_00 + SPEAKER_01
        00:12.80    SPEAKER_01
    """
    lines = [f"{'time':<12} active speakers", "-" * 52]
    edges = set()
    for unit in plan or []:
        start = float(unit.get("start", 0.0))
        end = start + float(unit.get("duration", 0.0))
        if end > start:
            edges.add(start)
            edges.add(end)
    points = sorted(edges)
    for a, b in zip(points, points[1:]):
        if b - a <= 1e-9:
            continue
        mid = (a + b) / 2.0
        active = sorted({
            (u.get("speaker") or "SPEAKER_00")
            for u in plan or []
            if float(u.get("start", 0.0)) <= mid
            < float(u.get("start", 0.0)) + float(u.get("duration", 0.0))
        })
        if not active:
            continue
        mark = "   <-- overlap" if len(active) > 1 else ""
        lines.append(f"{_clock(a):<12} {' + '.join(active)}{mark}")
    return "\n".join(lines) + "\n"


def _write(path: str, track) -> None:
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, track, SR, subtype="PCM_16")


def _read(path: str):
    import soundfile as sf
    wave, _ = sf.read(path, dtype="float32")
    return wave


def _tunable(key: str, default):
    try:
        import engines
        return engines.tunable(key)
    except Exception:                                        # noqa: BLE001
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify overlapping speakers stay in time and in voice")
    parser.add_argument("mode", choices=("plan", "check"),
                        help="plan: report readiness. check: run the check.")
    parser.add_argument("clip", help="a clip with two people talking over "
                                     "each other")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--preserve-background", action="store_true",
                        help="also run Demucs, to check the mix under overlap")
    parser.add_argument("--render", action="store_true",
                        help="also run lip sync (needs the Wav2Lip weights)")
    parser.add_argument("--chunk-seconds", type=int, default=600)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    try:
        return cmd_plan(args) if args.mode == "plan" else cmd_check(args)
    except SystemExit:
        raise
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        print(f"\nThe check could not complete: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

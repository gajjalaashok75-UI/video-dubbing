"""
The canonical final validation: one command, the whole pipeline.

Phases 2-5 each grew a check script — `multispeaker_check.py`,
`background_check.py`, `overlap_check.py` — and each is good at its own stage.
This is the one to run when you want to know whether a clip dubs correctly end
to end, and it is the one to paste into a bug report.

    python scripts/final_check.py plan clip.mp4
        Readiness only. Loads no model, touches no GPU.

    python scripts/final_check.py check clip.mp4 --target-language hi
        The full audio pipeline, as far as this machine can go.

    python scripts/final_check.py check clip.mp4 --target-language hi \
        --preserve-background --render
        The whole thing, including Demucs and Wav2Lip.

Everything it reports comes from `readiness` and `run_summary`, so what you see
here is what the studio shows and what the manifest records. There is no third
opinion to disagree with.

If a stage cannot run on this machine it says BLOCKED and why, and carries on to
the next one rather than stopping — the point of the harness is to find out how
far this environment gets, not to fail at the first missing package.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "engine"))

SR = 24000


def _engine():
    """The light modules, which import without torch."""
    import av_sync
    import readiness
    import run_summary
    return av_sync, readiness, run_summary


def _heavy():
    """
    The modules that pull in torch, imported only when a run actually starts.

    `plan` has to work on a machine that has not installed the model stack yet —
    that is the machine `plan` is for — and several of these import torch at
    module level. Importing them eagerly turned "here is what you are missing"
    into a traceback about the thing that is missing.
    """
    import dubbing
    import speaker_diarization
    import speakers as SPK
    import translation_v2
    return dubbing, speaker_diarization, SPK, translation_v2


def _banner(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def _status_line(name: str, ok, detail: str = "") -> None:
    mark = {True: "ok  ", False: "FAIL", None: "skip"}[ok]
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args) -> int:
    av_sync, readiness, _summary = _engine()

    _banner("FINAL CHECK — PLAN")
    print("Readiness, as the studio and the API see it. No model is loaded.")

    snap = readiness.snapshot(target_language=args.target_language,
                              source_language=args.source_language)
    print()
    for line in readiness.summarize(snap):
        print(line)

    print()
    print("What each optional feature costs if it is missing:")
    print("  multispeaker           -> one voice for the whole video")
    print("  background preservation-> the dub replaces the soundtrack")
    print("  lip sync               -> audio-only output")

    if args.clip:
        if not os.path.exists(args.clip):
            print(f"\nclip not found: {args.clip}")
            return 2
        info = av_sync.probe(args.clip)
        print()
        print(f"clip      {args.clip}")
        print(f"duration  {info['duration']:.2f}s")
        print(f"size      {info['video']['width']}x{info['video']['height']} "
              f"@ {info['video']['fps']:.3f}fps")
        audio = info.get("audio") or {}
        print(f"audio     {audio.get('codec')}, "
              f"{'stereo' if (audio.get('channels') or 1) > 1 else 'mono'}")

    print()
    if snap["ready"]:
        print("READY for a real run.")
        if snap.get("degraded"):
            print(f"  (reduced: {', '.join(snap['degraded'])})")
        print()
        print("Run:")
        cmd = f"  python scripts/final_check.py check {args.clip or 'clip.mp4'}"
        print(cmd + f" --target-language {args.target_language}")
    else:
        print("NOT READY for a real run. Blocking:")
        for name in snap.get("blocking") or []:
            feature = snap["features"].get(name) or {}
            print(f"  {name}: {feature.get('reason')}")
        print()
        print("The model-free test suite still runs without any of these:")
        print("  ./venv/bin/python -m unittest discover -s tests")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    av_sync, readiness, run_summary = _engine()

    clip = os.path.abspath(args.clip)
    if not os.path.exists(clip):
        print(f"clip not found: {clip}")
        return 2

    _banner("FINAL CHECK — RUN")

    # The readiness gate runs BEFORE the heavy imports. Checking readiness and
    # then importing torch would report "PyTorch is not installed" as a
    # traceback about PyTorch not being installed, which is the one thing this
    # harness exists to avoid.
    snap = readiness.snapshot(target_language=args.target_language)
    print(f"device      {(snap['hardware'] or {}).get('device')}")
    print(f"HF_TOKEN    {snap['secrets']['HF_TOKEN']}")
    if not snap["ready"]:
        print()
        print("BLOCKED before starting:")
        for name in snap.get("blocking") or []:
            print(f"  {name}: {(snap['features'].get(name) or {}).get('reason')}")
        print()
        print("This is an environment limitation, not a code failure. The")
        print("model-free suite covers the same logic:")
        print("  ./venv/bin/python -m unittest discover -s tests")
        return 1

    dubbing, diar, SPK, translation = _heavy()

    stem_name = os.path.splitext(os.path.basename(clip))[0]
    out_dir = os.path.join(ROOT, "final_check", stem_name)
    if os.path.isdir(out_dir) and not args.keep:
        shutil.rmtree(out_dir)
    workdir = os.path.join(out_dir, "work")
    os.makedirs(workdir, exist_ok=True)

    # The job dict this builds is the same shape `process_video_task` produces,
    # so `run_summary.build` reads it unchanged.
    job = {"job_id": f"final_check_{stem_name}", "kind": "dub",
           "status": "processing", "input_path": clip,
           "target_language": args.target_language,
           "source_language": args.source_language,
           "workdir": workdir}
    video_dur = av_sync.duration(clip)
    job["video_duration"] = video_dur

    steps = []

    def step(name):
        steps.append(name)
        print(f"\n[{len(steps)}] {name}")

    try:
        # 1. Stems.
        stems = None
        speech_source = clip
        if args.preserve_background:
            step("Separating the soundtrack (Demucs)")
            import source_separation
            stems = source_separation.separate(clip, workdir, log=print,
                                               chunk_seconds=args.chunk_seconds)
            source_separation.release()
            if stems:
                speech_source = stems["vocals"]
                job["stems"] = stems
                job["vocals_path"] = stems["vocals"]
                job["background_path"] = stems["background"]
                print(source_separation.summarize(stems))
        else:
            print("\n[skip] Background separation (--preserve-background not given)")

        # 2. ASR.
        step("Transcribing")
        stt_audio = os.path.join(workdir, "stt.wav")
        av_sync.extract_audio(speech_source, stt_audio, 16000, 1)
        import speech_to_text_v2
        result = speech_to_text_v2.transcribe_audio(stt_audio)
        segments = result.get("segments") or []
        if not segments:
            print("      no speech detected in the clip")
            return 1
        source_lang = job["source_language"] = (result.get("language") or "en")
        print(f"      {len(segments)} segments, language={source_lang}")
        job["word_segments"] = segments

        # 3. Diarization.
        step("Detecting speakers")
        turns = diar.diarize(clip, min_speakers=args.min_speakers,
                             max_speakers=args.max_speakers)
        job["diarization"] = turns
        job["diarization_source"] = "detected" if turns else None
        if turns:
            segments = diar.assign_segments(segments, turns)
            print("      " + diar.summarize(segments, turns))

        labels = sorted({SPK.label_of(s) for s in segments if SPK.label_of(s)})
        if not labels:
            labels = [SPK.DEFAULT_LABEL]
        segments = [{**s, "speaker": SPK.label_of(s) or labels[0]}
                    for s in segments]
        job["segments"] = [{"start": float(s["start"]), "end": float(s["end"]),
                            "text": s.get("text", ""),
                            "speaker": SPK.label_of(s)} for s in segments]

        # 4. References.
        step("Selecting speaker references")
        import reference_audio
        refs = reference_audio.build_references(
            speech_source, segments, video_dur,
            os.path.join(out_dir, "references"), labels=labels,
            sample_rate=SR, is_audio=bool(stems),
            generic_path=SPK.generic_voice_path())
        job["speakers"] = refs
        for entry in refs:
            row = SPK.describe(entry)
            print(f"      {row['label']}: {row['status']} — "
                  f"{row['segment_count']} segments, "
                  f"{(row['reference'] or {}).get('duration') or 0.0}s reference")
            if row["warning"]:
                print(f"        ! {row['warning']}")

        # 5. Translation.
        step("Translating")
        translated = translation.translate_segments(
            segments, source_lang, args.target_language, for_speech=True)
        translated = translation.fix_code_switching(translated,
                                                    args.target_language)
        job["translated_segments"] = list(translated)
        print(f"      {len(translated)} translated segments")

        # 6. Synthesis.
        step("Synthesizing voices")
        units_dir = os.path.join(workdir, "units")
        dialogue = os.path.join(out_dir, "dialogue.wav")
        plan = dubbing.build_dubbed_track(
            segments, translated, refs[0].get("reference") or {},
            args.target_language, video_dur, dialogue, source_lang=source_lang,
            source_audio=stt_audio, cache_dir=units_dir, speakers=refs)
        job["plan"] = plan
        job["dub_path"] = dialogue
        job["units_dir"] = units_dir

        # 7. Mix.
        if stems:
            step("Mixing the original background")
            mixed = os.path.join(out_dir, "mixed.wav")
            av_sync.mix_layers(
                dialogue, mixed, target=video_dur,
                background_path=stems["background"],
                dialogue_lufs=float(_tunable("DIALOGUE_LUFS", -12.0)),
                background_lufs=float(_tunable("BACKGROUND_LUFS", -26.0)),
                duck=bool(_tunable("BACKGROUND_DUCK", False)))
            job["mixed_path"] = mixed
        else:
            print("\n[skip] Background mixing")

        # 8. Optional render.
        if args.render:
            step("Lip syncing")
            from lip_sync import generate_lip_synced_video
            dubbed = os.path.join(out_dir, "dubbed.mp4")
            generate_lip_synced_video(clip, job.get("mixed_path") or dialogue,
                                      dubbed)
            job["output_path"] = dubbed
        else:
            print("\n[skip] Lip sync (--render not given)")

        job["status"] = "done"

    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"\nThe run stopped: {e}")

    # Report, from the same structures the studio reads.
    summary = run_summary.build(job, readiness=snap)
    job["run_summary"] = summary

    _banner("RESULT")
    print(run_summary.render(summary))

    _banner("ARTIFACTS")
    artifacts = summary.get("artifacts") or {}
    for name, path in artifacts.items():
        if path and os.path.exists(path):
            print(f"  {name:<16} {path}")
    for row in (summary.get("speakers") or {}).get("detail") or []:
        ref = row.get("reference") or {}
        if ref.get("path") and os.path.exists(ref["path"]):
            print(f"  reference:{row['label']:<6} {ref['path']}")

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "text": run_summary.render(summary)},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  report.json      {os.path.join(out_dir, 'report.json')}")

    # Exit non-zero only when the run genuinely failed, not when an optional
    # stage was unavailable. A degraded run is a result, not an error.
    return 0 if job["status"] == "done" else 1


def _tunable(key: str, default):
    try:
        import engines
        return engines.tunable(key)
    except Exception:                                        # noqa: BLE001
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One canonical end-to-end validation for a clip")
    parser.add_argument("mode", choices=("plan", "check"),
                        help="plan: readiness only. check: run the pipeline.")
    parser.add_argument("clip", nargs="?", default=None,
                        help="a short clip; optional for `plan`")
    parser.add_argument("--target-language", default="hi",
                        help="the language to dub into (default hi)")
    parser.add_argument("--source-language", default=None,
                        help="the language spoken in the clip; detected if omitted")
    parser.add_argument("--preserve-background", action="store_true",
                        help="keep the original music and sound effects (Demucs)")
    parser.add_argument("--render", action="store_true",
                        help="also run Wav2Lip to produce a finished video")
    parser.add_argument("--chunk-seconds", type=int, default=600)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--keep", action="store_true",
                        help="do not clear a previous run's output folder")
    args = parser.parse_args()

    if args.mode == "check" and not args.clip:
        parser.error("`check` needs a clip")

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

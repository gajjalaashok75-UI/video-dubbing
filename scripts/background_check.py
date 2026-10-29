"""
Background-preservation checkpoint: hear each layer of the mix on its own.

The Phase 4 counterpart to `multispeaker_check.py`. It answers one question for
a short BGM/SFX-heavy clip:

    did we keep the film's music and sound effects, and is the new dialogue
    still intelligible over them?

Six things come out, in the order you want to listen to them:

    1. original.wav      what the clip sounded like
    2. vocals.wav        Demucs' speech stem (what ASR and the references see)
    3. background.wav    Demucs' no_vocals stem — the preserved score, SFX,
                         ambience. If this sounds like the original minus the
                         talking, separation worked.
    4. dialogue.wav      the IndicF5 dub on its own, no bed
    5. mixed.wav         dialogue over the preserved background: the deliverable
    6. dubbed.mp4        the finished video (only with --render)

Listening to 2 and 3 is the actual test. A duration check cannot tell you that
the score survived; two seconds of 3 will.

Usage
-----
    python scripts/background_check.py plan clip.mp4
    python scripts/background_check.py check clip.mp4 --language hi
    python scripts/background_check.py check clip.mp4 --language hi --render

`--render` runs the full pipeline including lip sync, which needs the Wav2Lip
weights and takes minutes. Without it you get all five audio artifacts, which is
enough to judge the mix.
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
    import source_separation
    return av_sync, source_separation


def _load_pipeline():
    import dubbing
    import reference_audio
    import speakers as SPK
    import speech_to_text_v2
    import translation_v2
    return dubbing, reference_audio, SPK, speech_to_text_v2, translation_v2


def _band_energy(path: str, lo: float, hi: float, sr: int = SR) -> float:
    """Mean power in a frequency band. A crude but real presence check."""
    import numpy as np
    import soundfile as sf
    try:
        wave, rate = sf.read(path, dtype="float32")
    except Exception:                                        # noqa: BLE001
        return 0.0
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if wave.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(wave)) ** 2
    freqs = np.fft.rfftfreq(len(wave), 1 / rate)
    mask = (freqs >= lo) & (freqs < hi)
    return float(spec[mask].sum() / mask.sum()) if mask.any() else 0.0


def _rms(path: str) -> float:
    import numpy as np
    import soundfile as sf
    try:
        wave, _ = sf.read(path, dtype="float32")
    except Exception:                                        # noqa: BLE001
        return 0.0
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return float(np.sqrt(np.mean(wave ** 2))) if wave.size else 0.0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args) -> int:
    av_sync, sep = _load()

    print("=" * 72)
    print("BACKGROUND PRESERVATION CHECKPOINT — PLAN")
    print("=" * 72)

    if not os.path.exists(args.clip):
        print(f"FAIL  clip not found: {args.clip}")
        return 2
    info = av_sync.probe(args.clip)
    if not info.get("video"):
        print(f"FAIL  no video stream in {args.clip}")
        return 2
    audio = info.get("audio") or {}
    print(f"clip      {args.clip}")
    print(f"duration  {info['duration']:.2f}s")
    print(f"audio     {audio.get('codec')}, "
          f"{info['video']['width']}x{info['video']['height']}")
    if info["duration"] > 1800:
        print("          NOTE: this is a long clip. Demucs on CPU is roughly")
        print("          realtime-to-slower; use a short excerpt to iterate.")

    print()
    print("What will run:")
    print("  1. extract_audio        16 kHz mono, for ASR (unchanged)")
    print("  2. separating_audio     HQ stereo -> chunked Demucs -> stems/")
    print("  3. transcribing         faster-whisper on the VOCALS stem")
    print("  4. selecting_reference  one window per speaker, from the stem")
    print("  5. synthesizing_voice   IndicF5, one voice per speaker")
    print("  6. mixing               dialogue + background -> mixed.wav")

    print()
    print("Backends:")
    ready = True

    have_demucs = sep.available()
    print(f"  demucs       {'present' if have_demucs else 'ABSENT'}"
          + ("" if have_demucs else "   -> the soundtrack is replaced"))
    if not have_demucs:
        print("               install with: pip install 'demucs>=4.0.0'")

    try:
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu")
        print(f"  torch        {torch.__version__} on {device}")
    except ImportError:
        print("  torch        ABSENT -> IndicF5 cannot run")
        ready = False

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"  HF_TOKEN     {'set' if token else 'NOT SET'}"
          + ("" if token else "   -> IndicF5 is gated"))
    if not token:
        print("               https://huggingface.co/ai4bharat/IndicF5")

    try:
        import models
        missing = [m["id"] for m in models.missing(core_only=True)]
        if missing:
            print(f"  models       MISSING: {', '.join(missing)}")
            ready = False
        else:
            print("  models       all core weights present")
    except Exception as e:                                   # noqa: BLE001
        print(f"  models       could not check ({e})")

    print()
    print("Layer order in the mix, loudest first:")
    print("  1. dialogue             the new IndicF5 speech")
    print("  2. preserved background the film's own score, SFX and ambience")
    print("  3. user music bed       only if one is uploaded; it goes UNDER")
    print("                          the preserved background, never instead")
    print("                          of it")
    print("Ducking is off by default.")

    print()
    if ready and have_demucs:
        print("READY. Run:")
        print(f"  python scripts/background_check.py check {args.clip} "
              f"--language {args.language}")
    else:
        print("NOT READY for a real run; see the notes above.")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    av_sync, sep = _load()
    dubbing, reference_audio, SPK, stt, translation = _load_pipeline()

    clip = os.path.abspath(args.clip)
    if not os.path.exists(clip):
        print(f"clip not found: {clip}")
        return 2

    stem_name = os.path.splitext(os.path.basename(clip))[0]
    out_dir = os.path.join(ROOT, "background_check", stem_name)
    if os.path.isdir(out_dir) and not args.keep:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    workdir = os.path.join(out_dir, "work")
    os.makedirs(workdir, exist_ok=True)

    report = {"clip": clip, "language": args.language, "layers": {}}
    video_dur = av_sync.duration(clip)

    def out(name):
        return os.path.join(out_dir, name)

    # 1. Original.
    print("[1/6] Extracting the original audio...")
    original = out("1_original.wav")
    av_sync.extract_audio(clip, original, SR, 1)
    report["layers"]["original"] = original

    # 2/3. Stems.
    print("[2/6] Separating the soundtrack (this is the slow part)...")
    stems = sep.separate(clip, workdir, log=print,
                         chunk_seconds=args.chunk_seconds)
    if not stems:
        print()
        print("Separation did not produce stems, so there is no preserved")
        print("background to check. The dialogue-only pipeline still works.")
        print("Install Demucs to enable this:  pip install 'demucs>=4.0.0'")
        return 1

    vocals = out("2_vocals.wav")
    background = out("3_background.wav")
    av_sync.extract_audio(stems["vocals"], vocals, SR, 1)
    av_sync.extract_audio(stems["background"], background, SR, 1)
    report["layers"]["vocals"] = vocals
    report["layers"]["background"] = background
    report["stems"] = {k: v for k, v in stems.items() if k != "cached"}

    print()
    print(f"      vocals      {av_sync.duration(vocals):.2f}s  "
          f"-> {vocals}")
    print(f"      background  {av_sync.duration(background):.2f}s  "
          f"-> {background}")

    # 4. Transcribe the vocals, then diarize + reference + translate + synth.
    print("[3/6] Transcribing the vocals stem...")
    stt_audio = os.path.join(workdir, "stt.wav")
    av_sync.extract_audio(stems["vocals"], stt_audio, 16000, 1)
    result = stt.transcribe_audio(stt_audio)
    segments = result.get("segments") or []
    if not segments:
        print("      no speech detected in the vocals stem; stopping here")
        return 1
    print(f"      {len(segments)} segments, language={result.get('language')}")

    print("[4/6] Diarizing and cutting references from the vocals stem...")
    import speaker_diarization as diar
    turns = diar.diarize(clip, min_speakers=args.min_speakers,
                         max_speakers=args.max_speakers)
    if turns:
        segments = diar.assign_segments(segments, turns)
        print("      " + diar.summarize(segments, turns))
    labels = sorted({SPK.label_of(s) for s in segments if SPK.label_of(s)})
    if not labels:
        labels = [SPK.DEFAULT_LABEL]
        segments = [{**s, "speaker": SPK.DEFAULT_LABEL} for s in segments]

    refs_dir = os.path.join(out_dir, "references")
    registry = reference_audio.build_references(
        stems["vocals"], segments, video_dur, refs_dir, labels=labels,
        sample_rate=SR, is_audio=True, generic_path=SPK.generic_voice_path())
    for entry in registry:
        ref = entry.get("reference") or {}
        print(f"      {entry['label']}: {entry.get('total_speech', 0.0):.1f}s "
              f"speech -> {ref.get('path')}")
    report["speakers"] = registry

    print("[5/6] Translating and synthesizing the dialogue...")
    source_lang = result.get("language") or "en"
    translated = translation.translate_segments(
        segments, source_lang, args.language, for_speech=True)
    translated = translation.fix_code_switching(translated, args.language)

    dialogue = out("4_dialogue.wav")
    units_dir = os.path.join(workdir, "units")
    plan = dubbing.build_dubbed_track(
        segments, translated, registry[0]["reference"] or {}, args.language,
        video_dur, dialogue, source_lang=source_lang,
        source_audio=stt_audio, cache_dir=units_dir, speakers=registry)
    report["layers"]["dialogue"] = dialogue
    report["plan"] = len(plan)
    print(f"      dialogue    {av_sync.duration(dialogue):.2f}s  "
          f"-> {dialogue}")

    # 5. Mix.
    print("[6/6] Mixing the dialogue over the preserved background...")
    mixed = out("5_mixed.wav")
    av_sync.mix_layers(
        dialogue, mixed, target=video_dur,
        background_path=stems["background"],
        dialogue_lufs=float(_tunable("DIALOGUE_LUFS", -12.0)),
        background_lufs=float(_tunable("BACKGROUND_LUFS", -26.0)),
        duck=bool(_tunable("BACKGROUND_DUCK", False)))
    report["layers"]["mixed"] = mixed

    # 6. Optional full render.
    if args.render:
        print("[6b] Rendering the video (lip sync; this takes minutes)...")
        from lip_sync import generate_lip_synced_video
        dubbed = out("6_dubbed.mp4")
        generate_lip_synced_video(clip, mixed, dubbed)
        report["layers"]["dubbed"] = dubbed

    # Report.
    print()
    print("=" * 72)
    print("LAYER CHECK")
    print("=" * 72)
    rows = [
        ("original", original),
        ("vocals", vocals),
        ("background", background),
        ("dialogue", dialogue),
        ("mixed", mixed),
    ]
    print(f"{'layer':<12} {'seconds':>9} {'RMS':>9} {'low band':>12} "
          f"{'mid band':>12}")
    for name, path in rows:
        print(f"{name:<12} {av_sync.duration(path):>9.2f} "
              f"{_rms(path):>9.5f} {_band_energy(path, 40, 160):>12.3f} "
              f"{_band_energy(path, 200, 3500):>12.3f}")

    print()
    checks = []
    checks.append((
        "mixed is exactly the video length",
        abs(av_sync.duration(mixed) - video_dur) < 0.05))
    checks.append((
        "the background stem is not silent",
        _rms(background) > 1e-4))
    checks.append((
        "the dialogue is present in the mix",
        _band_energy(mixed, 200, 3500) > 1e-6))
    checks.append((
        "the background is present in the mix",
        _band_energy(mixed, 40, 160) > 1e-7))
    checks.append((
        "the mix is louder than the background alone",
        _rms(mixed) > _rms(background)))
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    report["checks"] = {label: ok for label, ok in checks}

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"Artifacts in {out_dir} — listen in this order:")
    print("  1_original.wav    the clip as it was")
    print("  2_vocals.wav      Demucs' speech stem")
    print("  3_background.wav  the preserved score and SFX  <-- the key one")
    print("  4_dialogue.wav    the IndicF5 dub on its own")
    print("  5_mixed.wav       dialogue over the preserved background")
    if args.render:
        print("  6_dubbed.mp4      the finished video")
    print()
    print("If 3 sounds like the original with the talking removed, separation")
    print("worked. If 5 keeps that while staying intelligible, the mix works.")
    return 0


def _tunable(key: str, default):
    """The stored setting, or the default. Tolerant of engines being absent."""
    try:
        import engines
        return engines.tunable(key)
    except Exception:                                        # noqa: BLE001
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the original soundtrack survives the dub")
    parser.add_argument("mode", choices=("plan", "check"),
                        help="plan: report readiness. check: run the check.")
    parser.add_argument("clip", help="a short BGM/SFX-heavy clip")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--render", action="store_true",
                        help="also run lip sync (needs the Wav2Lip weights)")
    parser.add_argument("--chunk-seconds", type=int, default=600,
                        help="Demucs chunk size; smaller is easier to debug")
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--keep", action="store_true",
                        help="do not clear a previous run's output folder")
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

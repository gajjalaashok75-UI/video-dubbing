"""
Multi-speaker checkpoint: prove each actor's phrases are spoken in THEIR voice.

This is the manual verification procedure for Phases 1-3. It answers one
question, for a short 2-3 speaker clip:

    SPEAKER_00 -> reference 00 -> IndicF5 output voice 00
    SPEAKER_01 -> reference 01 -> IndicF5 output voice 01
    SPEAKER_02 -> reference 02 -> IndicF5 output voice 02

It deliberately stops short of a full dub. Listening to three short reference
clips and three short phrases takes a minute; listening to a finished render
and guessing which voice is which takes much longer, and cannot tell "wrong
speaker" apart from "bad clone".

Two modes:

  plan    Say what will be done, and check the machine can do it. No GPU work.
  check   Run diarization, cut the references, synthesize ONE phrase per
          speaker, and report a similarity matrix.

The similarity matrix is the actual evidence. `mkl` on the diagonal means each
phrase sounds most like its own speaker's reference. A high off-diagonal value
means two speakers were cloned from the same voice - the exact failure the
fallback ladder in Phase 3 exists to prevent.

Usage
-----
    # What would happen, and is the machine ready?
    python scripts/multispeaker_check.py plan clip.mp4

    # Do it, and write the artifacts somewhere you can listen to them
    python scripts/multispeaker_check.py check clip.mp4 --language hi

    # Compare against the single-voice baseline, which is what a regression
    # looks like: every phrase matching the same reference.
    python scripts/multispeaker_check.py check clip.mp4 --no-diarization

Artifacts land in `multispeaker_check/<clip>/`:

    refs/speaker_SPEAKER_00.wav     listen to this first
    refs/speaker_SPEAKER_00.txt     the transcript it was paired with
    ...
    phrases/SPEAKER_00.wav          one synthesized phrase, in that voice
    ...
    report.json                     the matrix, and every decision made
    report.txt                      the same, for reading
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


def _load():
    """Import the pipeline modules, or explain what is missing."""
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)}. Install the project requirements "
            f"first (./run.sh --setup --yes, or pip install -r requirements.txt).")

    import av_sync
    import reference_audio
    import speaker_diarization
    import speakers as SPK
    # `speech_to_text_v2` imports torch at module level, so it is loaded only
    # when it is actually needed. `plan` has to work on a machine that has not
    # installed the torch stack yet - that is the machine `plan` is for.
    stt = None
    return av_sync, reference_audio, speaker_diarization, SPK, stt


def _load_stt():
    import speech_to_text_v2
    return speech_to_text_v2


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args) -> int:
    av_sync, reference_audio, diar, SPK, stt = _load()

    print("=" * 72)
    print("MULTI-SPEAKER CHECKPOINT — PLAN")
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
    print(f"size      {info['video']['width']}x{info['video']['height']} "
          f"@ {info['video']['fps']:.3f}fps")

    print()
    print("Stages that will run:")
    print("  1. extract_audio      16 kHz mono, for ASR")
    print("  2. transcribing       faster-whisper large-v3, with word timings")
    print("  3. diarizing          pyannote, on the ORIGINAL audio")
    print("  4. selecting_reference one window per detected speaker")
    print("  5. (synthesis)        one phrase per speaker, to compare voices")

    print()
    print("Backends:")
    ready = True

    have_pyannote = diar.available()
    print(f"  diarization  {'pyannote' if have_pyannote else 'ABSENT'}"
          + ("" if have_pyannote else "   -> the run falls back to one speaker"))
    if not have_pyannote:
        print("               install with: pip install 'pyannote.audio>=3.3.2'")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"  HF_TOKEN     {'set' if token else 'NOT SET'}")
    if not token:
        print("               pyannote and IndicF5 are both gated. Accept the")
        print("               licences and set HF_TOKEN in .env:")
        print("                 https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("                 https://huggingface.co/pyannote/segmentation-3.0")
        print("                 https://huggingface.co/ai4bharat/IndicF5")

    try:
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu")
        print(f"  torch        {torch.__version__} on {device}")
        if device == "cpu":
            print("               synthesis on CPU is slow; a short clip is "
                  "strongly recommended")
    except ImportError:
        print("  torch        ABSENT -> IndicF5 cannot run")
        ready = False

    try:
        import models
        missing = [m["id"] for m in models.missing(core_only=True)]
        if missing:
            print(f"  models       MISSING: {', '.join(missing)}")
            print("               download them from the studio's Setup page, or")
            print("               ./venv/bin/python scripts/download_models.py")
            ready = False
        else:
            print("  models       all core weights present")
    except Exception as e:                                   # noqa: BLE001
        print(f"  models       could not check ({e})")

    print()
    if ready and have_pyannote and token:
        print("READY. Run:")
        print(f"  python scripts/multispeaker_check.py check {args.clip} "
              f"--language {args.language}")
    else:
        print("NOT READY for a real run. The single-speaker path still works;")
        print("see the notes above for what is missing.")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def _similarity_matrix(paths: dict, labels: list):
    """
    Cosine similarity between every synthesized phrase and every reference.

    This is the evidence. `wavlm-base-plus-sv` is the same speaker model the
    eval harness and the studio's voice-match score already use, so the numbers
    are comparable with the rest of the project.
    """
    try:
        from eval_harness import SpeakerScorer
    except Exception as e:                                   # noqa: BLE001
        return None, f"speaker scorer unavailable ({e})"

    scorer = SpeakerScorer()
    embeddings = {}
    for label in labels:
        path = paths.get(label)
        if not path or not os.path.exists(path):
            continue
        emb = scorer.embed(path)
        if emb is not None:
            embeddings[label] = emb
    if len(embeddings) < 2:
        return None, "not enough embeddings to compare"

    matrix = {}
    for a in labels:
        if a not in embeddings:
            continue
        row = {}
        for b in labels:
            if b not in embeddings:
                continue
            row[b] = round(float(scorer.cosine(embeddings[a], embeddings[b])), 4)
        matrix[a] = row
    return matrix, None


def _verdict(matrix: dict, labels: list) -> list:
    """
    Read the matrix: does each phrase match its own reference best?

    A `diagonal` result is the pass. Anything else names the pair that matched
    instead, because "SPEAKER_01 sounds like SPEAKER_00's reference" is a
    precise, actionable failure and "FAIL" is not.
    """
    lines = []
    for label in labels:
        row = matrix.get(label) or {}
        if not row:
            continue
        best = max(row.items(), key=lambda kv: kv[1])
        if best[0] == label:
            lines.append(f"  PASS  {label}: its own reference scores highest "
                         f"({best[1]:.3f})")
        else:
            lines.append(f"  FAIL  {label}: matches {best[0]}'s reference "
                         f"({best[1]:.3f}) better than its own "
                         f"({row.get(label, float('nan')):.3f})")
    return lines


def cmd_check(args) -> int:
    av_sync, reference_audio, diar, SPK, _stt = _load()
    stt = _load_stt()

    clip = os.path.abspath(args.clip)
    if not os.path.exists(clip):
        print(f"clip not found: {clip}")
        return 2

    stem = os.path.splitext(os.path.basename(clip))[0]
    out_dir = os.path.join(ROOT, "multispeaker_check", stem)
    if os.path.isdir(out_dir) and not args.keep:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    refs_dir = os.path.join(out_dir, "refs")
    phrase_dir = os.path.join(out_dir, "phrases")
    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(phrase_dir, exist_ok=True)

    report = {"clip": clip, "language": args.language, "steps": []}

    # ── 1. audio + transcript ──
    print("[1/5] Extracting audio...")
    stt_audio = os.path.join(out_dir, "stt.wav")
    av_sync.extract_audio(clip, stt_audio, 16000, 1)

    print("[2/5] Transcribing...")
    result = stt.transcribe_audio(stt_audio)
    segments = result.get("segments") or []
    if not segments:
        print("      no speech detected; nothing to check")
        return 1
    print(f"      {len(segments)} segments, language={result.get('language')}")
    report["segments"] = len(segments)
    report["detected_language"] = result.get("language")

    # ── 2. diarization ──
    turns = []
    if args.no_diarization:
        print("[3/5] Diarization SKIPPED (--no-diarization): the baseline")
    else:
        print("[3/5] Diarizing...")
        turns = diar.diarize(clip,
                             min_speakers=args.min_speakers,
                             max_speakers=args.max_speakers)
        print(f"      {len(turns)} turns")
    report["turns"] = turns
    report["diarization"] = bool(turns)

    if turns:
        segments = diar.assign_segments(segments, turns)
        print("      " + diar.summarize(segments, turns))

    labels = sorted({SPK.label_of(s) for s in segments if SPK.label_of(s)})
    if not labels:
        labels = [SPK.DEFAULT_LABEL]
        segments = [{**s, "speaker": SPK.DEFAULT_LABEL} for s in segments]
        print(f"      no speakers resolved; treating the clip as {labels[0]}")
    report["labels"] = labels
    if len(labels) < 2 and not args.no_diarization:
        print("      NOTE: only one speaker was found. The voice check needs at")
        print("      least two to say anything. Try a clip with a conversation")
        print("      in it, or lower --min-speakers.")

    # ── 3. references ──
    print("[4/5] Cutting one reference per speaker...")
    registry = reference_audio.build_references(
        clip, segments, av_sync.duration(clip), refs_dir, labels=labels,
        sample_rate=24000, generic_path=SPK.generic_voice_path())
    report["speakers"] = registry

    ref_paths = {}
    for entry in registry:
        label = entry["label"]
        ref = entry.get("reference") or {}
        ref_paths[label] = ref.get("path")
        mark = "  <-- FALLBACK, not this speaker's own voice" if entry.get("fallback") else ""
        print(f"      {label}: {entry.get('total_speech', 0.0):.1f}s speech, "
              f"{entry.get('segment_count', 0)} segments, "
              f"reference {ref.get('duration') or 0.0:.2f}s{mark}")
        if ref.get("path"):
            print(f"        listen: {ref['path']}")

    # ── 4. one phrase per speaker ──
    print("[5/5] Synthesizing one phrase per speaker...")
    import dubbing
    phrase_paths = {}
    for entry in registry:
        label = entry["label"]
        ref = entry.get("reference")
        if not ref or not ref.get("path"):
            print(f"      {label}: no reference, skipping")
            continue
        phrase = _pick_phrase(segments, label)
        if phrase is None:
            print(f"      {label}: no segment to speak, skipping")
            continue
        text = phrase["text"] or "This is a test of the cloned voice."
        unit = {"index": 0, "text": text, "start": phrase["start"],
                "duration": max(0.5, phrase["end"] - phrase["start"]),
                "speaker": label, "ref_index": None}
        try:
            wave = dubbing.synth_unit(unit, ref["path"], ref.get("text", ""),
                                      float(ref.get("duration") or 0.0),
                                      args.language)
        except Exception as e:                               # noqa: BLE001
            print(f"      {label}: synthesis failed ({e})")
            report.setdefault("errors", []).append({label: str(e)})
            continue
        if wave is None or getattr(wave, "size", 0) == 0:
            print(f"      {label}: synthesis returned no audio")
            continue
        import soundfile as sf
        path = os.path.join(phrase_dir, f"{label}.wav")
        sf.write(path, wave, dubbing.SAMPLE_RATE, subtype="PCM_16")
        phrase_paths[label] = path
        print(f"      {label}: spoke {len(wave) / dubbing.SAMPLE_RATE:.2f}s "
              f"in {ref['path'].split(os.sep)[-1]} -> {path}")

    # ── 5. the matrix ──
    print()
    print("=" * 72)
    print("VOICE CHECK")
    print("=" * 72)
    matrix, why = _similarity_matrix(phrase_paths, labels)
    report["phrase_paths"] = phrase_paths
    report["matrix"] = matrix
    if matrix is None:
        print(f"Could not score the voices: {why}")
    else:
        header = "phrase \\ reference".ljust(22) + "".join(
            l.ljust(14) for l in labels)
        print(header)
        for a in labels:
            if a not in matrix:
                continue
            print(a.ljust(22) + "".join(
                f"{matrix[a].get(b, float('nan')):.3f}".ljust(14)
                for b in labels))
        print()
        for line in _verdict(matrix, labels):
            print(line)
        report["verdict"] = _verdict(matrix, labels)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"Artifacts in {out_dir}")
    print(f"  refs/     listen to each speaker's reference clip first")
    print(f"  phrases/  then each synthesized phrase")
    print(f"  report.json / report.txt")

    # A human-readable copy of the same thing.
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Multi-speaker check: {clip}\n")
        f.write(f"language: {args.language}   "
                f"diarization: {'on' if turns else 'off'}\n\n")
        for entry in registry:
            ref = entry.get("reference") or {}
            f.write(f"{entry['label']}\n")
            f.write(f"  speech      {entry.get('total_speech', 0.0):.1f}s in "
                    f"{entry.get('segment_count', 0)} segments\n")
            f.write(f"  reference   {ref.get('path')}\n")
            f.write(f"  duration    {ref.get('duration')}\n")
            f.write(f"  text        {(ref.get('text') or '')[:120]}\n")
            f.write(f"  fallback    {entry.get('fallback')}\n")
            f.write(f"  phrase      {phrase_paths.get(entry['label'])}\n\n")
        if matrix:
            f.write("similarity (phrase x reference)\n")
            for a in labels:
                if a in matrix:
                    f.write(f"  {a}: " + ", ".join(
                        f"{b}={matrix[a][b]:.3f}" for b in labels
                        if b in matrix[a]) + "\n")
    return 0


def _pick_phrase(segments: list, label: str):
    """
    The longest segment for a speaker, which gives the voice most to work with.

    Deliberately not the segment the reference came from: speaking back the
    reference text would prove nothing about cloning a NEW line.
    """
    best = None
    for seg in segments or []:
        if seg.get("speaker") != label:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        span = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        if best is None or span > best[0]:
            best = (span, {"start": float(seg.get("start", 0.0)),
                           "end": float(seg.get("end", 0.0)), "text": text})
    return best[1] if best else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify each speaker is cloned from their own voice")
    parser.add_argument("mode", choices=("plan", "check"),
                        help="plan: report readiness. check: run the check.")
    parser.add_argument("clip", help="a short 2-3 speaker video")
    parser.add_argument("--language", default="hi",
                        help="target language for the test phrase (default hi)")
    parser.add_argument("--no-diarization", action="store_true",
                        help="skip diarization, to see the single-voice baseline")
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

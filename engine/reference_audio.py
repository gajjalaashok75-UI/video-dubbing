"""
Reference-audio selection for voice cloning.

The old pipeline took a blind `-t 15` from t=0, which in real footage is usually
intro music, a title card, silence, or a second speaker. Cloning quality then
varied purely with how the video happened to open — the single biggest source of
run-to-run inconsistency.

This module instead uses the word-level timestamps Whisper already produces to
find the cleanest contiguous run of confident speech, and extracts it at 24 kHz
(not 16 kHz — everything above 8 kHz is sibilance and breathiness, which is most
of what makes a voice identifiable to a cloning model).

It also returns the TRANSCRIPT of the chosen window, which IndicF5 requires as
its `ref_text` conditioning input.
"""

import os
import subprocess

# A prompt shorter than this is too little for a stable speaker embedding;
# longer than ~15s and F5 starts clipping it internally anyway.
MIN_REF_SECONDS = 6.0
MAX_REF_SECONDS = 12.0

# Words separated by more than this are treated as a pause that breaks a window.
MAX_INTERNAL_GAP = 0.6

# Openings are disproportionately likely to be music/titles/intros.
INTRO_PENALTY_BEFORE = 2.0


def _flatten_words(segments: list, speaker: str = None) -> list:
    """
    Collect word-level timestamps across all segments.

    When `speaker` is given, only that speaker's words are collected. This is
    what stops the old failure mode where the "cleanest window" in a two-hander
    belonged to the other actor: the window is now scored over one person's
    speech, so a reference can only ever be cut from its own speaker.
    """
    words = []
    for seg in segments or []:
        if speaker is not None:
            seg_speaker = seg.get("speaker")
            # A word is attributed by its own label when it has one, falling
            # back to the segment's. `assign_words` writes both.
            for w in seg.get("words") or []:
                label = w.get("speaker") or seg_speaker
                if label != speaker:
                    continue
                if w.get("start") is None or w.get("end") is None:
                    continue
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                words.append({
                    "word": text,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "p": float(w.get("probability", 1.0)),
                })
            continue

        for w in seg.get("words") or []:
            if w.get("start") is None or w.get("end") is None:
                continue
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({
                "word": text,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "p": float(w.get("probability", 1.0)),
            })
    words.sort(key=lambda w: w["start"])
    return words


def _score_window(window: list) -> float:
    """
    Higher is better. Rewards confident, continuous, well-filled speech.
    """
    if not window:
        return -1e9

    dur = window[-1]["end"] - window[0]["start"]
    if dur <= 0:
        return -1e9

    mean_conf = sum(w["p"] for w in window) / len(window)

    # Fraction of the window actually occupied by speech (vs. internal pauses).
    voiced = sum(w["end"] - w["start"] for w in window)
    density = min(1.0, voiced / dur)

    # Largest internal gap — a big one means we straddled a pause or a cut.
    max_gap = 0.0
    for a, b in zip(window, window[1:]):
        max_gap = max(max_gap, b["start"] - a["end"])

    score = (mean_conf * 2.0) + (density * 1.5) - (max_gap * 1.0)

    if window[0]["start"] < INTRO_PENALTY_BEFORE:
        score -= 0.5

    # Mild preference for longer prompts, saturating at MAX_REF_SECONDS.
    score += 0.3 * min(dur, MAX_REF_SECONDS) / MAX_REF_SECONDS

    return score


def find_best_window(segments: list, total_duration: float,
                     speaker: str = None) -> tuple:
    """
    Return (start, end, text) for the best reference window, or None if the
    transcript has no usable word timings.

    With `speaker` given, only that speaker's words are considered, so the
    window is one person talking rather than a mix of two.
    """
    words = _flatten_words(segments, speaker)
    if not words:
        return None

    # Footage with long pauses (speeches, dramatic delivery) can leave every
    # gap-bounded window under MIN_REF_SECONDS. A short reference is not a
    # cosmetic problem: it throws off the seconds-per-byte calibration the
    # generator uses, and a 3.8s reference produced the worst output in the
    # whole corpus. So retry with a progressively more permissive gap before
    # settling for a short window.
    for max_gap in (MAX_INTERNAL_GAP, MAX_INTERNAL_GAP * 2, MAX_INTERNAL_GAP * 4):
        best = _best_window_for_gap(words, max_gap)
        if best and (best[-1]["end"] - best[0]["start"]) >= MIN_REF_SECONDS:
            break

    if best is None:
        best = _longest_run(words)

    if not best:
        return None

    start = max(0.0, best[0]["start"] - 0.10)
    end = min(total_duration, best[-1]["end"] + 0.10)
    text = " ".join(w["word"] for w in best).strip()
    return start, end, text


def candidate_windows(segments: list, total_duration: float, limit: int = 5,
                      speaker: str = None,
                      min_seconds: float = MIN_REF_SECONDS) -> list:
    """
    The best few reference windows, not just the winner.

    Reference selection drives most of the cloning quality and has always been
    automatic and invisible. The scoring already ranks every candidate; this
    returns the top handful so a person can listen and disagree.

    Windows that overlap an already-chosen one by more than half are dropped,
    since a one-word shift is not a real alternative.

    With `speaker` given, only that speaker's windows are returned, which is
    what the studio's per-speaker reference picker shows.

    `min_seconds` is the shortest window worth returning. It defaults to the
    preferred 6s, but `build_references` lowers it so a speaker with only short
    clean runs still has candidates to stitch or to use as a short reference.
    Filtering at 6s here would make those fallbacks unreachable.
    """
    words = _flatten_words(segments, speaker)
    if not words:
        return []

    scored = []
    for max_gap in (MAX_INTERNAL_GAP, MAX_INTERNAL_GAP * 2, MAX_INTERNAL_GAP * 4):
        for i in range(len(words)):
            window = []
            for j in range(i, len(words)):
                if window and words[j]["start"] - window[-1]["end"] > max_gap:
                    break
                window.append(words[j])
                dur = window[-1]["end"] - window[0]["start"]
                if dur < min_seconds:
                    continue
                if dur > MAX_REF_SECONDS:
                    break
                scored.append((_score_window(window), list(window)))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    chosen = []
    for score, window in scored:
        start = max(0.0, window[0]["start"] - 0.10)
        end = min(total_duration, window[-1]["end"] + 0.10)
        if end - start <= 0:
            continue
        overlapping = any(
            min(end, c["start"] + c["duration"]) - max(start, c["start"])
            > 0.5 * min(end - start, c["duration"])
            for c in chosen
        )
        if overlapping:
            continue
        chosen.append({
            "start": round(start, 2),
            "duration": round(end - start, 2),
            "score": round(float(score), 3),
            "text": " ".join(w["word"] for w in window).strip(),
        })
        if len(chosen) >= limit:
            break

    return chosen


def _best_window_for_gap(words: list, max_gap: float):
    """Highest-scoring window whose internal gaps never exceed `max_gap`."""
    best, best_score = None, -1e9

    for i in range(len(words)):
        window = []
        for j in range(i, len(words)):
            if window and words[j]["start"] - window[-1]["end"] > max_gap:
                break
            window.append(words[j])
            dur = window[-1]["end"] - window[0]["start"]
            if dur < MIN_REF_SECONDS:
                continue
            if dur > MAX_REF_SECONDS:
                break
            s = _score_window(window)
            if s > best_score:
                best_score = s
                best = list(window)

    return best


def _longest_run(words: list):
    """Longest gap-free run, however short — last resort."""
    run, longest = [], []
    for w in words:
        if run and w["start"] - run[-1]["end"] > MAX_INTERNAL_GAP * 4:
            if (run[-1]["end"] - run[0]["start"]) > (
                longest[-1]["end"] - longest[0]["start"] if longest else 0
            ):
                longest = run
            run = []
        run.append(w)
    if run and (not longest or (run[-1]["end"] - run[0]["start"]) >
                (longest[-1]["end"] - longest[0]["start"])):
        longest = run
    return longest


def extract_reference(source_path: str, start: float, duration: float,
                      out_path: str, sample_rate: int = 24000,
                      denoise: bool = False, is_audio: bool = False) -> str:
    """
    Cut the reference window and condition it for speaker-embedding quality:
    trim edge silence, remove low-frequency rumble, normalize loudness.

    Deliberately does NOT apply aggressive denoise by default - spectral
    denoisers smear exactly the timbre detail a cloning model keys on.

    `source_path` is normally the video. Pass an audio file (a separated vocals
    stem, say) with `is_audio=True` and the `-vn` is dropped: a stem has no
    picture, and asking ffmpeg to ignore a stream that is not there is a
    needless failure mode.
    """
    # ONLY timing-preserving filters here. `silenceremove` used to be in this
    # chain and it was catastrophic: it strips INTERNAL pauses, not just the
    # edges, and cut a 7.08s window down to 2.58s — 64% of the audio gone.
    #
    # That breaks F5 in two ways at once. The prompt drops below the ~5s it
    # needs for a stable speaker identity, and — far worse — the audio no longer
    # corresponds to `ref_text`, which still transcribes the whole window. The
    # model is handed 2.6s of audio and told it says 7 seconds' worth of words,
    # so its duration model is wrecked and the output comes out rushed and
    # robotic.
    #
    # Anything added here must leave the timeline alone, or `ref_text` has to be
    # re-derived to match.
    chain = ["highpass=f=60", "loudnorm=I=-18:TP=-2:LRA=11"]
    if denoise:
        chain.insert(1, "afftdn=nr=8:nf=-30")

    cmd = ["ffmpeg", "-y",
           "-ss", f"{start:.3f}",
           "-t", f"{duration:.3f}",
           "-i", source_path]
    if not is_audio:
        cmd.append("-vn")
    cmd += [
        "-af", ",".join(chain),
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def stitch_reference(parts: list, out_path: str, sample_rate: int = 24000,
                     gap: float = 0.15) -> str:
    """
    Join several short same-speaker windows into one longer reference.

    Step 2 of the fallback ladder: when a speaker has no single 6-second run but
    does have three clean 3-second ones, three 3-second runs are a far better
    reference than one of them alone - and are honest, because they are all the
    same person.

    A short silence is inserted between parts so the model hears a boundary
    rather than a splice. This is not the same as padding one clip: the parts
    are non-overlapping samples of one speaker, chosen by the same scorer that
    picks a single window.
    """
    import numpy as np
    import soundfile as sf

    waves = []
    for path in parts or []:
        try:
            wave, sr = sf.read(path, dtype="float32")
        except Exception as e:                               # noqa: BLE001
            print(f"[REF] Could not read {path} for stitching: {e}")
            continue
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if wave.size == 0:
            continue
        if sr != sample_rate:
            try:
                import librosa
                wave = librosa.resample(wave, orig_sr=sr, target_sr=sample_rate)
            except Exception:                                # noqa: BLE001
                continue
        waves.append(wave)

    if not waves:
        raise RuntimeError("No usable parts to stitch a reference from")

    silence = np.zeros(int(gap * sample_rate), dtype=np.float32)
    pieces = []
    for i, wave in enumerate(waves):
        if i:
            pieces.append(silence)
        pieces.append(wave)
    joined = np.concatenate(pieces)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, joined, sample_rate, subtype="PCM_16")
    return out_path


def build_reference(video_path: str, segments: list, total_duration: float,
                    out_path: str, sample_rate: int = 24000,
                    is_audio: bool = False) -> dict:
    """
    Pick and extract the best voice-cloning reference from a video.

    Returns {"path", "text", "start", "end", "duration"}.
    `text` is the transcript of the window, required by IndicF5 as `ref_text`.

    `is_audio=True` means `video_path` is already audio (a separated vocals
    stem), so ffmpeg is not told to drop a video stream that is not there.
    """
    picked = find_best_window(segments, total_duration)

    if picked is None:
        # No usable word timings at all — fall back to a mid-video window,
        # which still beats t=0 for avoiding intros.
        start = min(max(0.0, total_duration * 0.25), max(0.0, total_duration - MIN_REF_SECONDS))
        end = min(total_duration, start + MAX_REF_SECONDS)
        text = ""
        print("[REF] No word timings available; falling back to a mid-video window")
    else:
        start, end, text = picked

    requested = max(0.5, end - start)
    extract_reference(video_path, start, requested, out_path, sample_rate,
                      is_audio=is_audio)

    # Report the duration of the file we ACTUALLY wrote, never the duration we
    # asked for. Callers use this to compute F5's `fix_duration`, and F5 derives
    # the reference length from the real waveform — so if the two disagree, every
    # generated segment comes out the wrong length and then gets time-stretched
    # to compensate, which sounds robotic. Encoder padding alone can shift this
    # by tens of milliseconds even with no filtering.
    actual = requested
    try:
        import av_sync
        actual = av_sync.duration(out_path)
    except Exception as e:
        print(f"[REF] Could not probe reference duration ({e}); "
              f"falling back to the requested {requested:.2f}s")

    if abs(actual - requested) > 0.05:
        print(f"[REF] NOTE: extracted {actual:.2f}s from a {requested:.2f}s "
              f"window — using the measured value")

    print(f"[REF] Reference window {start:.2f}s–{end:.2f}s ({actual:.2f}s) "
          f"@{sample_rate}Hz -> {os.path.basename(out_path)}")
    if actual < MIN_REF_SECONDS - 0.5:
        print(f"[REF] WARNING: only {actual:.2f}s of reference audio; "
              f"voice cloning is unreliable below ~{MIN_REF_SECONDS:.0f}s")
    if text:
        print(f"[REF] Reference text: {text[:90]}{'...' if len(text) > 90 else ''}")

    return {"path": out_path, "text": text, "start": start,
            "end": end, "duration": actual}


# A same-speaker reference shorter than this is not worth cloning from: F5's
# speaker identity is unstable below roughly two seconds.
MIN_USABLE_SECONDS = 2.0

# How many short same-speaker windows may be stitched together for step 2 of the
# fallback ladder. Bounded so the joined reference stays inside the range F5
# handles well, rather than becoming an hour of montage.
MAX_STITCH_PARTS = 4


def _measured(path: str, requested: float) -> float:
    """
    The real duration of a file we just wrote.

    Never the duration we asked for. F5 derives the reference length from the
    actual waveform, so if the two disagree every generated phrase comes out the
    wrong length and is time-stretched to compensate - which is what sounds
    robotic. Encoder padding alone shifts this by tens of milliseconds.
    """
    try:
        import av_sync
        return av_sync.duration(path)
    except Exception as e:                                   # noqa: BLE001
        print(f"[REF] Could not probe {os.path.basename(path)} ({e}); "
              f"using the requested {requested:.2f}s")
        return requested


def _windows_for(segments: list, total_duration: float, speaker: str,
                 limit: int = 5) -> list:
    """
    Ranked candidate windows for one speaker, longest-and-cleanest first.

    `candidate_windows` already ranks by confidence, density and gap; this adds
    the tie-break the multi-speaker case needs - prefer the window closest to
    the middle of the 6-12s band, because a 6.0s reference is better than a
    12.0s one only when the 12.0s one is noisier.
    """
    windows = candidate_windows(segments, total_duration, limit=max(limit, 5),
                               speaker=speaker,
                               min_seconds=MIN_USABLE_SECONDS)
    if not windows:
        return []

    def rank(w):
        dur = float(w.get("duration") or 0.0)
        # Distance from the middle of the preferred band, as a penalty.
        mid = (MIN_REF_SECONDS + MAX_REF_SECONDS) / 2.0
        band = (MAX_REF_SECONDS - MIN_REF_SECONDS) / 2.0
        off = abs(dur - mid) / band if band else 0.0
        return (float(w.get("score") or 0.0) - 0.25 * off)

    return sorted(windows, key=rank, reverse=True)


def _extract_from(source_path: str, window: dict, out_path: str,
                  sample_rate: int, is_audio: bool) -> dict | None:
    """Cut one window to `out_path`. Returns the reference dict, or None."""
    start = max(0.0, float(window.get("start") or 0.0))
    duration = max(0.5, float(window.get("duration") or 0.0))
    try:
        extract_reference(source_path, start, duration, out_path, sample_rate,
                          is_audio=is_audio)
    except Exception as e:                                   # noqa: BLE001
        print(f"[REF] Could not cut {start:.2f}s+{duration:.2f}s: {e}")
        return None
    actual = _measured(out_path, duration)
    if actual <= 0.05:
        return None
    return {"path": out_path, "text": (window.get("text") or "").strip(),
            "start": round(start, 3), "end": round(start + actual, 3),
            "duration": round(actual, 3)}


def build_references(source_path: str, segments: list, total_duration: float,
                     out_dir: str, labels: list = None,
                     sample_rate: int = 24000, is_audio: bool = False,
                     generic_path: str = None) -> list:
    """
    One reference profile per detected speaker.

    Returns the registry `speakers.normalize_speakers` understands:

        [{"label", "total_speech", "segment_count", "reference",
          "reference_score", "fallback"}, ...]

    `source_path` is normally the video; pass a separated vocals stem with
    `is_audio=True` to cut references from speech that has no music under it.

    The fallback order is deliberate and is the whole point of this function:

      1. the best clean SAME-SPEAKER window of 6-12s;
      2. failing that, several clean non-overlapping SAME-SPEAKER windows
         stitched together, if together they reach the preferred length;
      3. failing that, 2-6s of same-speaker speech, used with a warning;
      4. failing that, an explicit generic voice (`generic_path`) if one is
         available, with `fallback=True`;
      5. failing that, an empty reference and `fallback=True`.

    At no point does it substitute ANOTHER DETECTED ACTOR's voice. Impersonating
    a real person in the cast is worse than an obviously generic voice, and it is
    the one outcome a listener cannot detect.
    """
    import speakers as SPK

    labels = list(labels or [])
    if not labels:
        labels = sorted({SPK.label_of(s) for s in segments or []
                         if SPK.label_of(s)})
    if not labels:
        return []

    os.makedirs(out_dir, exist_ok=True)
    registry = []

    for label in labels:
        entry = {"label": label, "total_speech": 0.0, "segment_count": 0,
                 "reference": None, "reference_score": None, "fallback": False}
        tally = SPK.tally(segments, label)
        entry.update(tally)

        windows = _windows_for(segments, total_duration, label)
        out_path = os.path.join(out_dir, f"speaker_{label}.wav")
        out_txt = os.path.join(out_dir, f"speaker_{label}.txt")

        # ── 1. One clean window in the preferred band ──
        long_enough = [w for w in windows
                       if float(w.get("duration") or 0.0) >= MIN_REF_SECONDS]
        chosen = None
        if long_enough:
            chosen = _extract_from(source_path, long_enough[0], out_path,
                                   sample_rate, is_audio)
            if chosen:
                entry["reference_score"] = long_enough[0].get("score")

        # ── 2. Stitch several clean same-speaker windows ──
        if chosen is None and len(windows) >= 2:
            parts = []
            texts = []
            total = 0.0
            for w in windows:
                if len(parts) >= MAX_STITCH_PARTS:
                    break
                dur = float(w.get("duration") or 0.0)
                if dur <= 0.5:
                    continue
                part = os.path.join(out_dir, f".part_{label}_{len(parts)}.wav")
                got = _extract_from(source_path, w, part, sample_rate, is_audio)
                if got is None:
                    continue
                parts.append(part)
                texts.append(got["text"])
                total += got["duration"]
                if total >= MIN_REF_SECONDS:
                    break
            if parts:
                try:
                    stitch_reference(parts, out_path, sample_rate)
                    actual = _measured(out_path, total)
                    if actual >= MIN_USABLE_SECONDS:
                        chosen = {"path": out_path,
                                  "text": " ".join(t for t in texts if t).strip(),
                                  "start": None, "end": None,
                                  "duration": round(actual, 3)}
                        entry["reference_score"] = None
                        print(f"[REF] {label}: stitched {len(parts)} windows "
                              f"into a {actual:.2f}s reference")
                except Exception as e:                       # noqa: BLE001
                    print(f"[REF] {label}: could not stitch a reference ({e})")
                finally:
                    for part in parts:
                        try:
                            os.remove(part)
                        except OSError:
                            pass

        # ── 3. Short same-speaker speech (2-6s), used with a warning ──
        if chosen is None and windows:
            short = [w for w in windows
                     if float(w.get("duration") or 0.0) >= MIN_USABLE_SECONDS]
            if short:
                chosen = _extract_from(source_path, short[0], out_path,
                                       sample_rate, is_audio)
                if chosen:
                    entry["reference_score"] = short[0].get("score")
                    print(f"[REF] {label}: only {chosen['duration']:.2f}s of "
                          f"clean speech; cloning quality will be limited")

        # ── 4. An explicit generic voice ──
        if chosen is None and generic_path and os.path.exists(generic_path):
            chosen = {"path": generic_path, "text": "", "start": None,
                      "end": None, "duration": round(_measured(generic_path, 0.0), 3)}
            entry["fallback"] = True
            print(f"[REF] {label}: no usable same-speaker audio; using the "
                  f"generic fallback voice")

        # ── 5. Nothing usable ──
        if chosen is None:
            entry["fallback"] = True
            print(f"[REF] {label}: {entry['total_speech']:.1f}s of speech but no "
                  f"usable reference; this speaker is unresolved")

        entry["reference"] = chosen
        if chosen and chosen.get("text"):
            try:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(chosen["text"])
            except OSError:
                pass
        registry.append(entry)

    print(SPK.summarize(registry, segments))
    return registry


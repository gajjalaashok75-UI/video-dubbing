"""
Speech/silence detection over the source audio, and phrase splitting.

Why this module exists
----------------------
Whisper segments are not speech units. A single segment routinely spans several
phrases with real pauses inside it, and the dubbing timeline used to fill each
segment's whole slot with ONE continuous utterance. Every pause inside that
segment was therefore replaced by words.

That is audible and severe. Measured on a 20.27s clip: 11.9% of the audio is
silence — 2.41s spread over 16 separate pauses, every one of them between 50ms
and 150ms. Those short pauses are what makes speech parseable; removing them
crams 2.41s of extra syllables into the same running time, so the dub sounds
rushed and the words run into each other even when every segment is nominally
playing at 1.00x speed.

Whisper's own word timestamps cannot be used to find them. On the same clip its
word boundaries were contiguous — it reported 1.1% pause against the waveform's
11.9% — because word-level timing for Hindi at `medium` is interpolated rather
than measured. So the pauses are detected from the waveform directly.
"""

import numpy as np

# Frame step for the energy envelope.
HOP_SECONDS = 0.010

# Level below the clip's peak at which a frame counts as silence.
SILENCE_DB = -35.0

# Ignore anything shorter than this: it is a stop consonant or a glottal gap,
# not a pause, and splitting on it would shred words.
MIN_PAUSE = 0.08

# A phrase shorter than this is not worth synthesizing on its own; it gets
# merged into its neighbour.
MIN_PHRASE = 0.35

# A phrase has to carry enough text to be worth its own slot. Splitting on
# detected pauses alone can produce more phrases than the translation has words,
# and the splitter then guarantees each one a word - which is how a sentence
# ended up spoken as "कर", "रहे", "हैं।", three separate one-word utterances
# with pauses between them.
MIN_PHRASE_WORDS = 2
# Used only to cap how many phrases a segment can have, never to judge whether
# a produced phrase is too thin: in Devanagari a perfectly good two-word phrase
# is often under ten codepoints.
MIN_PHRASE_CHARS = 10


def detect_speech_runs(audio_path: str, silence_db: float = SILENCE_DB,
                       min_pause: float = MIN_PAUSE) -> list:
    """
    Return [(start, end), ...] of speech regions, in seconds.

    Silences shorter than `min_pause` are treated as part of the surrounding
    speech rather than as boundaries.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16000)
    if y.size == 0:
        return []

    hop = max(1, int(sr * HOP_SECONDS))
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    if rms.max() <= 0:
        return []

    db = librosa.amplitude_to_db(rms, ref=np.max)
    voiced = db > silence_db

    # Bridge silences that are too short to count as pauses.
    bridge = int(round(min_pause / HOP_SECONDS))
    idx = 0
    while idx < len(voiced):
        if not voiced[idx]:
            start = idx
            while idx < len(voiced) and not voiced[idx]:
                idx += 1
            if start > 0 and idx < len(voiced) and (idx - start) < bridge:
                voiced[start:idx] = True
        else:
            idx += 1

    runs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start * HOP_SECONDS, i * HOP_SECONDS))
            start = None
    if start is not None:
        runs.append((start * HOP_SECONDS, len(voiced) * HOP_SECONDS))

    total = len(y) / sr
    return [(max(0.0, a), min(total, b)) for a, b in runs if b > a]


def _split_text_by_weights(text: str, weights: list) -> list:
    """
    Split `text` into len(weights) parts at word boundaries, with each part's
    character count roughly proportional to its weight.

    Used to distribute a segment's translation across the phrases inside it.
    Approximate by nature — we do not know which translated word corresponds to
    which source phrase — but preserving the rhythm matters more than getting
    the split exactly on the right word.
    """
    words = (text or "").split()
    n = len(weights)
    if n <= 1 or len(words) <= 1:
        return [text.strip()] + [""] * (n - 1)

    total_w = float(sum(weights)) or 1.0
    total_c = sum(len(w) + 1 for w in words)

    parts, i = [], 0
    used = 0.0
    for k, w in enumerate(weights):
        if k == n - 1:
            parts.append(" ".join(words[i:]))
            break
        used += w
        want = total_c * (used / total_w)
        acc, j = 0, i
        while j < len(words) and acc < want:
            acc += len(words[j]) + 1
            j += 1
        # Always leave at least one word for each remaining part.
        j = max(i + 1, min(j, len(words) - (n - k - 1)))
        parts.append(" ".join(words[i:j]))
        i = j

    return [p.strip() for p in parts]


def _merge_thin(units: list) -> list:
    """
    Fold any unit too thin to be spoken on its own into its neighbour.

    A one-word unit is not a phrase; given its own slot and its own pause on
    either side, it is delivered as an isolated utterance. Merging joins the
    time spans too, so the pause it used to own becomes part of the phrase.
    """
    if len(units) <= 1:
        return units

    out = []
    for unit in units:
        thin = len(unit["target"].split()) < MIN_PHRASE_WORDS
        if thin and out:
            prev = out[-1]
            prev["end"] = unit["end"]
            prev["target"] = f"{prev['target']} {unit['target']}".strip()
            prev["text"] = f"{prev['text']} {unit['text']}".strip()
        else:
            out.append(dict(unit))

    # A thin FIRST unit has no predecessor, so it folds forward instead.
    if len(out) > 1 and len(out[0]["target"].split()) < MIN_PHRASE_WORDS:
        head, nxt = out[0], out[1]
        nxt["start"] = head["start"]
        nxt["target"] = f"{head['target']} {nxt['target']}".strip()
        nxt["text"] = f"{head['text']} {nxt['text']}".strip()
        out = out[1:]
    return out


def _cap_by_text(runs: list, text: str) -> list:
    """
    Reduce `runs` until there is enough text to give each one a real phrase.

    Merging is done by repeatedly joining the pair separated by the shortest
    silence, so the longest pauses - the ones a listener actually hears as
    phrasing - are the last to go.
    """
    words = len((text or "").split())
    if words == 0:
        return runs
    cap = max(1, min(words // MIN_PHRASE_WORDS, len(text) // MIN_PHRASE_CHARS))
    if len(runs) <= cap:
        return runs

    runs = [list(r) for r in runs]
    while len(runs) > cap:
        # Gap between consecutive runs; the smallest is the least missed.
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        i = gaps.index(min(gaps))
        runs[i][1] = runs[i + 1][1]
        del runs[i + 1]
    return runs


def build_phrase_units(segments: list, translated: list, speech_runs: list,
                       min_phrase: float = MIN_PHRASE) -> list:
    """
    Turn Whisper segments into finer phrase units aligned to real speech runs.

    Each unit is {"start", "end", "text", "target", "segment"}, where `text` is
    the SOURCE chunk and `target` the translated chunk — matching the convention
    plan_timeline expects (segment carries source text, translations arrive as a
    parallel list). Both are split the same way so the duration model still sees
    a meaningful target/source length ratio per unit.

    The gaps BETWEEN units are the speaker's actual pauses; the timeline leaves
    them as silence, which is what restores the natural rhythm.

    A segment containing no detected pause yields exactly one unit, so this
    degrades gracefully to the previous whole-segment behaviour.
    """
    units = []

    # strict: the timeline must cover every segment, not however many
    # translations happened to come back.
    for seg_i, (seg, text) in enumerate(zip(segments, translated, strict=True)):
        text = (text or "").strip()
        source_text = (seg.get("text") or "").strip()
        if not text:
            continue

        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", s0))
        if s1 <= s0:
            continue

        # Speech runs overlapping this segment, clipped to it.
        inside = []
        for a, b in speech_runs:
            lo, hi = max(a, s0), min(b, s1)
            if hi - lo > 0.05:
                inside.append([lo, hi])

        if not inside:
            units.append({"start": s0, "end": s1, "text": source_text,
                          "target": text, "segment": seg_i})
            continue

        # Merge runs that are too short to stand alone as a phrase.
        merged = [inside[0]]
        for lo, hi in inside[1:]:
            if (hi - lo) < min_phrase or (merged[-1][1] - merged[-1][0]) < min_phrase:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])

        # Cap the phrase count by what the text can actually fill. When merging
        # is forced, absorb the SMALLEST gap first so the pauses that carry the
        # most rhythm are the ones that survive.
        merged = _cap_by_text(merged, text)

        weights = [hi - lo for lo, hi in merged]
        tgt_chunks = _split_text_by_weights(text, weights)
        src_chunks = _split_text_by_weights(source_text, weights)

        made = [{"start": lo, "end": hi,
                 "text": src.strip() or tgt.strip(),
                 "target": tgt.strip(), "segment": seg_i}
                for (lo, hi), tgt, src in zip(merged, tgt_chunks, src_chunks,
                                              strict=True)
                if tgt.strip()]
        # The cap bounds the phrase count on average, but a weighted split can
        # still starve the last part. This is where the guarantee has to hold.
        units.extend(_merge_thin(made))

    units.sort(key=lambda u: u["start"])
    return units


# Whisper's word timings are interpolated and clip syllables; pad before use.
WORD_PAD = 0.12
# Ordinary inter-word gap; close anything shorter so a phrase is one interval.
WORD_MERGE_GAP = 0.30
# Below this many seconds of word timings the transcript is not dense enough to
# be trusted as a mask, and the energy gate is used on its own (the old
# behaviour). A clip that genuinely has no transcript keeps its report.
MIN_WORD_SECONDS = 1.0


def _spoken_intervals(words: list, pad: float = WORD_PAD,
                      merge_gap: float = WORD_MERGE_GAP) -> list:
    """
    Union of the word timings, padded and merged, as [(start, end), ...].

    Whisper's word timings are interpolated and can clip a syllable, so each is
    padded before merging rather than used as-is; the merge then closes the
    ordinary inter-word gaps so a phrase reads as one interval rather than a
    picket fence.
    """
    spans = []
    for w in words or []:
        # Whisper's segment output nests words as {"word","start","end"}; the
        # plain pair form is what the coverage helper is tested with. Accept
        # both so the caller does not have to reshape the transcript.
        if isinstance(w, dict):
            a, b = w.get("start"), w.get("end")
        else:
            try:
                a, b = w[0], w[1]
            except (TypeError, IndexError):
                continue
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if b < a:
            a, b = b, a
        spans.append([max(0.0, a - pad), b + pad])
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _intersect_runs(runs: list, spoken: list) -> list:
    """The parts of `runs` that lie inside `spoken`, as [(start, end), ...]."""
    out = []
    for a, b in runs:
        for c, d in spoken:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                out.append((lo, hi))
    out.sort()
    return out


def uncovered_speech(runs: list, plan: list, min_hole: float = 0.5,
                     words: list = None) -> dict:
    """
    Where the source speaks but no dubbed phrase was placed.

    `runs` is `detect_speech_runs` output; `plan` is the dubbing timeline, each
    item carrying `start` and `duration`. A dropped clause does not announce
    itself — the assembled track is still exactly the video's length, so the
    length check passes and the failure is only audible. Measured against the
    plan, though, it is plain: a stretch of the original where someone was
    talking and no phrase covers it is a hole.

    A hole shorter than `min_hole` is normal breath between lines, not a dropped
    clause, so it is not named — naming those would drown the signal.

    `words` is the transcript's word timings, as `[[start, end], ...]` or the
    `{"start","end"}` dicts Whisper emits. When given, a run is only counted as
    speech where the transcript actually has words: a sustained music bed sits
    well above the energy gate and used to be reported as speech, so a scored
    trailer showed one 25s "dropped dialogue" hole that was really the score.
    Requiring both signals removes that false alarm while keeping the real one —
    dropped dialogue is speech the transcript knows about and the plan does not
    cover. A transcript too sparse to be trusted is ignored, so the previous
    behaviour is preserved for a clip with no words.

    Pure, so it is testable without librosa or a rendered dub.
    """
    covered = sorted(
        (float(u.get("start", 0.0)),
         float(u.get("start", 0.0)) + float(u.get("duration", 0.0)))
        for u in (plan or []) if float(u.get("duration", 0.0)) > 0.0)

    def covered_seconds(a: float, b: float) -> float:
        total = 0.0
        for s, e in covered:
            total += max(0.0, min(b, e) - max(a, s))
        return total

    spoken = _spoken_intervals(words)
    masked = bool(spoken) and sum(b - a for a, b in spoken) >= MIN_WORD_SECONDS
    measured = _intersect_runs(runs, spoken) if masked else list(runs)

    uncovered = []
    for a, b in measured:
        gap = (b - a) - covered_seconds(a, b)
        if gap >= min_hole:
            uncovered.append({"start": round(a, 2), "end": round(b, 2),
                              "gap_s": round(gap, 2)})

    return {
        "source_speech_s": round(sum(b - a for a, b in measured), 2),
        "dubbed_s": round(sum(e - s for s, e in covered), 2),
        "uncovered_s": round(sum(u["gap_s"] for u in uncovered), 2),
        "uncovered": uncovered,
    }


def summarize(units: list, segments: list, total_duration: float) -> str:
    speech = sum(u["end"] - u["start"] for u in units)
    return (f"[VAD] {len(segments)} segments -> {len(units)} phrase units; "
            f"{speech:.2f}s speech, {total_duration - speech:.2f}s pause "
            f"({100 * (total_duration - speech) / max(total_duration, 0.01):.1f}%)")


# ---------------------------------------------------------------------------
# Prosody: how loudly each line is delivered, relative to the rest of the clip
# ---------------------------------------------------------------------------
# S2 Pro takes free-form inline control tags ("[loud]", "[low volume]", ...),
# and a trailer's whole character is the contrast between a whispered line and
# a shouted one. The tags are only worth emitting when they are measured rather
# than guessed, so the level is read from the source waveform and compared with
# the clip's own median speech level: a line is "loud" when it is loud for THIS
# speaker in THIS clip, which is the comparison a listener actually makes.

# How far above/below the clip's median speech a line must sit to be tagged.
LOUD_DB = 4.0
QUIET_DB = -5.0


def speech_levels(audio_path: str, spans: list) -> list:
    """
    Median speech level in dBFS for each (start, end) span, or None for a span.

    One load of the file and one RMS envelope, then a slice per span: measuring
    each line separately would decode the whole file once per line. Spans with
    no audible speech return None rather than a floor, so they are never tagged.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16000)
    if y.size == 0:
        return [None] * len(spans)

    hop = max(1, int(sr * HOP_SECONDS))
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    if rms.max() <= 0:
        return [None] * len(spans)
    db = librosa.amplitude_to_db(rms, ref=1.0)
    # Frames that are effectively digital silence carry no level information.
    floor = float(np.percentile(db, 5))
    peak = float(db.max())
    voiced_cut = max(floor, peak - 40.0)

    levels = []
    for span in spans:
        try:
            a, b = float(span[0]), float(span[1])
        except (TypeError, ValueError):
            levels.append(None)
            continue
        i0 = max(0, int(a / HOP_SECONDS))
        i1 = min(len(db), max(i0 + 1, int(b / HOP_SECONDS)))
        window = db[i0:i1]
        window = window[window > voiced_cut]
        levels.append(float(np.median(window)) if window.size else None)
    return levels


def prosody_tags(levels: list, loud_db: float = LOUD_DB,
                 quiet_db: float = QUIET_DB) -> list:
    """
    Turn measured levels into S2 Pro control tags, one per span.

    "" means "say it normally" and is the right answer for the median line; only
    a clear deviation earns a tag. The median is taken over the non-None levels,
    so a clip that is uniformly loud is not tagged loud all the way through.
    """
    present = [v for v in levels if v is not None]
    if len(present) < 3:
        # Too few lines to know what "normal" is for this clip. Tagging from a
        # two-line sample would be a coin flip dressed as analysis.
        return [""] * len(levels)
    baseline = float(np.median(present))

    tags = []
    for level in levels:
        if level is None:
            tags.append("")
        elif level - baseline >= loud_db:
            tags.append("[loud]")
        elif level - baseline <= quiet_db:
            tags.append("[low volume]")
        else:
            tags.append("")
    return tags

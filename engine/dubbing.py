"""
Segment-level dubbing and timeline assembly.

This is the module that makes gaps structurally impossible.

The old pipeline translated the entire transcript as one string, synthesized it
as one blob, and then tried to rescue the timing afterwards with a global tempo
stretch. Two things went wrong: the stretch was applied in the wrong direction,
and its output was never actually fed to Wav2Lip. Whatever length the TTS
happened to produce became the length of the video, so the render either looped
back to the start or cut off mid-shot.

Here the dubbed track is *constructed* to be exactly as long as the video:

  1. Each Whisper segment keeps its own time slot.
  2. Each segment is synthesized to fill that slot, using IndicF5's `fix_duration`
     (verified accurate to ~10 ms) rather than stretching afterwards.
  3. Speaking-rate change is clamped to a natural band; text that will not fit
     borrows from the following pause instead of being crushed.
  4. The assembled track is padded/trimmed to exactly the video duration.

Because step 4 is an invariant rather than a check, Wav2Lip always receives an
audio track the same length as its video, and cannot loop or truncate.

Every unit also carries the speaker it belongs to, and is spoken in that
speaker's reference voice. A video with no diarization has one implicit
`SPEAKER_00` whose reference is the global one, so the single-voice path is
unchanged; see `speakers`.
"""

import os
from contextlib import contextmanager

import numpy as np

import languages as L
import speakers as SPK
import tts_engines

SAMPLE_RATE = 24000

# How far the speaking rate may deviate from the reference speaker's natural
# rate. r = natural_duration / allotted_duration; r > 1 means talking faster.
MAX_SPEEDUP = 1.35     # never crush speech below ~74% of its natural length
MAX_SLOWDOWN = 0.75    # never drawl beyond ~133% of natural length

# Fraction of the pause after a segment that the segment may borrow.
BORROW_FRACTION = 0.85

# Below this, don't bother correcting the length.
LENGTH_TOLERANCE = 0.02

# Hard limits on post-hoc time stretching. The generator is asked for the right
# length up front (F5's fix_duration), so anything outside this band signals a
# bug rather than a legitimate fit, and a phase vocoder stops being transparent
# well before the edges.
MAX_STRETCH_DOWN = 0.80
MAX_STRETCH_UP = 1.25

# Fade length applied to each segment's edges before it is laid onto the track.
EDGE_FADE_MS = 8.0


def _chars(text: str) -> int:
    return max(1, len((text or "").strip()))


class DurationModel:
    """
    Estimates how long a piece of text takes to say *in this speaker's voice*.

    Deliberately counts CHARACTERS, not bytes. F5's own internal heuristic is
    byte-based, which is fine when the reference and the target are the same
    language but breaks badly cross-lingually: a Devanagari character is 3 UTF-8
    bytes against 1 for Latin, so calibrating an English reference in bytes and
    applying it to Hindi over-estimates every duration by roughly 3x. English to
    Hindi is our main path, so that error is not survivable.

    Instead we separate the two effects:
      * script density  — characters per second typical for the TARGET language
                          (languages.chars_per_second)
      * speaker tempo   — how fast THIS speaker talks relative to typical for
                          the SOURCE language, measured from the reference clip
    """

    # Guards against a wild factor from a mistimed or mistranscribed reference.
    MIN_SPEAKER_FACTOR = 0.6
    MAX_SPEAKER_FACTOR = 1.7

    def __init__(self, ref_duration: float, ref_text: str,
                 source_lang: str = None, target_lang: str = "hi",
                 speaker_factor: float = None):
        self.source_cps = L.chars_per_second(source_lang) if source_lang else None
        self.target_cps = L.chars_per_second(target_lang)

        # A caller that has already measured this speaker's tempo — from their
        # own reference window — passes it in, because a per-speaker model is
        # the whole point of a multi-speaker run: two actors do not talk at the
        # same rate, and calibrating both against one of them makes the slower
        # one sound rushed.
        if speaker_factor is not None:
            self.speaker_factor = min(self.MAX_SPEAKER_FACTOR,
                                      max(self.MIN_SPEAKER_FACTOR,
                                          float(speaker_factor)))
            return

        factor = 1.0
        if ref_duration > 0.5 and ref_text and len(ref_text.strip()) > 8:
            ref_cps = _chars(ref_text) / ref_duration
            typical = self.source_cps or ref_cps
            if ref_cps > 0.1:
                factor = typical / ref_cps
        self.speaker_factor = min(self.MAX_SPEAKER_FACTOR,
                                  max(self.MIN_SPEAKER_FACTOR, factor))

    def tempo_factor(self) -> float:
        """
        This speaker's tempo, for handing to another `DurationModel`.

        Exposed so `build_dubbed_track` can build one model per speaker without
        re-deriving the ratio from the same reference twice.
        """
        return self.speaker_factor

    def estimate(self, text: str, source_text: str = None,
                 slot: float = None) -> float:
        """
        How long `text` should take to say.

        When we know what the ORIGINAL speaker said in this slot and how long
        they took, that is far better evidence than any characters-per-second
        table: it is this speaker, this sentence, this recording. So scale the
        measured slot by how much longer the translation is, correcting for the
        two scripts packing different amounts of speech into a character.

            natural = slot x (target_chars / source_chars)
                           x (source_cps / target_cps)

        This is self-calibrating and, importantly, degrades to `natural == slot`
        when source and target are the same language — a same-language re-voice
        then needs no speed change at all.

        The global cps table is only a fallback for when segment-level evidence
        is missing. Relying on it alone was over-estimating durations by ~41% on
        real footage, which pinned whole runs of segments at maximum speedup.
        """
        if (source_text and slot and slot > 0.05
                and len(source_text.strip()) > 1 and self.source_cps):
            ratio = _chars(text) / _chars(source_text)
            density = self.source_cps / self.target_cps
            return slot * ratio * density

        return (_chars(text) / self.target_cps) * self.speaker_factor


def _trim_silence(wave: np.ndarray, floor_db: float = -40.0,
                  keep_ms: float = 30.0) -> np.ndarray:
    """
    Strip leading/trailing near-silence, keeping a short natural tail.

    Only touches the ends; internal pauses inside an utterance are the model's
    own phrasing and must survive.
    """
    if wave.size < int(SAMPLE_RATE * 0.05):
        return wave

    peak = float(np.max(np.abs(wave)))
    if peak <= 0:
        return wave

    win = max(1, int(SAMPLE_RATE * 0.01))
    n = wave.size // win
    if n < 3:
        return wave

    frames = wave[: n * win].reshape(n, win)
    env = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + 1e-12
    db = 20.0 * np.log10(env / peak)
    loud = np.where(db > floor_db)[0]
    if loud.size == 0:
        return wave

    keep = int(SAMPLE_RATE * keep_ms / 1000.0)
    start = max(0, loud[0] * win - keep)
    end = min(wave.size, (loud[-1] + 1) * win + keep)
    return wave[start:end] if end > start else wave


def _edge_fade(wave: np.ndarray, ms: float) -> np.ndarray:
    """Apply a short linear fade at both ends so placement can't click."""
    n = int(SAMPLE_RATE * ms / 1000.0)
    if wave.size < 2 * n or n < 2:
        return wave
    out = wave.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _time_stretch(wave: np.ndarray, factor: float) -> np.ndarray:
    """
    Stretch `wave` by `factor` (>1 = longer) preserving pitch.
    Only ever used for sub-5% residual corrections.
    """
    if wave.size == 0 or abs(factor - 1.0) < 0.005:
        return wave
    try:
        import librosa
        # librosa's `rate` is playback rate: rate>1 shortens.
        return librosa.effects.time_stretch(wave, rate=1.0 / factor).astype(np.float32)
    except Exception:
        return wave


def _fit_length(wave: np.ndarray, target_samples: int) -> np.ndarray:
    """Force `wave` to exactly `target_samples`, gently."""
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    n = wave.size
    if n == 0:
        return np.zeros(target_samples, dtype=np.float32)

    drift = abs(n - target_samples) / float(target_samples)
    if drift > LENGTH_TOLERANCE:
        factor = target_samples / float(n)
        # A phase vocoder is only transparent for small corrections. Anything
        # beyond this band means the generator returned a wildly wrong length,
        # which is a bug upstream — squeezing e.g. 0.4x turns speech into the
        # classic robotic warble. Clamp it and say so, rather than quietly
        # destroying the audio to satisfy a duration target.
        if not (MAX_STRETCH_DOWN <= factor <= MAX_STRETCH_UP):
            print(f"[DUB] WARNING: segment wants a {factor:.2f}x time stretch "
                  f"({n / SAMPLE_RATE:.2f}s -> {target_samples / SAMPLE_RATE:.2f}s). "
                  f"Clamping — check that the reference duration is correct.")
            factor = min(MAX_STRETCH_UP, max(MAX_STRETCH_DOWN, factor))
        wave = _time_stretch(wave, factor)
        n = wave.size

    if n > target_samples:
        out = wave[:target_samples].copy()
        # 5 ms fade so a hard cut never clicks
        f = min(120, target_samples)
        if f > 1:
            out[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        return out
    return np.pad(wave, (0, target_samples - n))


def prepare_units(segments: list, translated_texts: list,
                  source_audio: str = None, video_duration: float = 0.0):
    """
    Split Whisper segments at the speaker's REAL pauses, returning
    (units, translated_texts) ready for `plan_timeline`.

    Without this, each segment's whole slot is filled with one continuous
    utterance and every pause inside it is replaced by words. On real footage
    that meant losing 2.41s of micro-pauses (16 of them, each 50-150ms) from a
    20s clip — which is what made the dub sound rushed and run its words
    together even with every segment nominally at 1.00x speed.

    Falls back to whole-segment units if the audio is unavailable or VAD fails,
    so this can only improve on the previous behaviour, never break it.

    Each returned unit inherits the SPEAKER of the segment it was split from,
    so a phrase is always spoken in the voice that segment was assigned.
    """
    if not source_audio:
        # The whole-segment path: the "units" ARE the segments, so the
        # translations are passed through untouched rather than read off a
        # `target` key that only phrase units carry.
        units = _with_speakers(segments, segments, translated_texts)
        return units, list(translated_texts)

    try:
        import speech_activity as SA
        runs = SA.detect_speech_runs(source_audio)
        units = SA.build_phrase_units(segments, translated_texts, runs)
        if not units:
            units = _with_speakers(segments, segments, translated_texts)
            return units, list(translated_texts)
        print(SA.summarize(units, segments, video_duration))
        units = _with_speakers(segments, units, [u["target"] for u in units])
        return units, [u["target"] for u in units]
    except Exception as e:
        print(f"[DUB] Pause detection unavailable ({e}); "
              f"falling back to whole-segment units")
        units = _with_speakers(segments, segments, translated_texts)
        return units, list(translated_texts)


def _with_speakers(segments: list, units: list, texts: list):
    """
    Copy each unit's speaker across from the segment it came from.

    `speech_activity.build_phrase_units` splits a segment into several phrase
    units and records the source segment index on each, so the label travels
    with the phrase rather than being re-derived from the clock. Units that
    already carry a speaker are left alone, so this is safe to call twice.
    """
    out = []
    for i, unit in enumerate(units):
        label = unit.get("speaker")
        if label is None:
            src_i = unit.get("segment")
            if isinstance(src_i, int) and 0 <= src_i < len(segments):
                label = SPK.label_of(segments[src_i])
            elif i < len(segments):
                label = SPK.label_of(segments[i])
        out.append({**unit, "speaker": label})
    return out


def plan_timeline(segments: list, texts: list, duration_model: DurationModel,
                  video_duration: float, natural_fn=None, speakers: list = None,
                  duration_models: dict = None, natural_for=None) -> list:
    """
    Decide the start time and allotted duration for every segment.

    Returns a list of dicts:
    {index, text, start, duration, natural, rate, speaker, ref_index}.
    `rate` is the speaking-rate multiplier actually applied, for logging.

    `speakers` is the registry the unit's label is resolved against; when it is
    omitted or empty every unit is `SPEAKER_00` with `ref_index` None, which is
    exactly what a single-speaker job produces.

    `duration_models` optionally maps a label to its own `DurationModel`, so two
    actors are timed against their own measured tempo rather than one shared
    average.

    `natural_fn(text)` and `natural_for(text, seg, label)` both ask the generator
    for its own length estimate. `natural_for` wins when both are given: it can
    see which speaker the phrase belongs to, so the estimate is measured against
    the reference that phrase will actually be spoken in. Either way the estimate
    is taken here, before the closing squeeze, so the length invariant holds.

    Scheduling is per speaker, not global. A single cursor serialised the whole
    video, so in a conversation

        SPEAKER_00  10.0 ───────── 13.0
        SPEAKER_01        11.5 ───────── 14.0

    the second line was pushed to 13.0-15.5 because the first had not finished.
    That is not what a conversation sounds like, and it desynchronises the
    speaker from the picture. Each speaker now keeps their own cursor, so a
    different actor may start while another is still talking, which is the
    normal case in film. The same speaker still cannot overlap themselves: their
    own cursor advances, so a second line waits for the first to finish.

    With one speaker there is one cursor, which is the previous behaviour
    exactly — the single-speaker plan is byte-for-byte what it always was.
    """
    plan = []
    speakers = speakers or []
    duration_models = duration_models or {}

    # One cursor per speaker. `cursors` is the whole scheduling state; there is
    # deliberately no global cursor, because a global one is the bug.
    cursors: dict = {}

    def next_start_for(label: str, from_index: int, seg_end: float) -> float:
        """
        When this speaker next speaks, which is how long they have to finish.

        Deliberately the next segment belonging to the SAME speaker, not simply
        the next segment in the list. In a conversation the following segment is
        usually the other actor, and treating their start as this speaker's
        deadline would cap a phrase by someone else's turn. For a single speaker
        the two are the same list position, so this is the previous behaviour.
        """
        for j in range(from_index + 1, len(segments)):
            if SPK.label_of(segments[j]) == label:
                return float(segments[j].get("start", seg_end))
        return video_duration

    # strict: one translation per segment. A short list used to truncate
    # the plan silently, which loses the end of the video with no error.
    for i, (seg, text) in enumerate(zip(segments, texts, strict=True)):
        text = (text or "").strip()
        if not text:
            continue

        label = SPK.label_of(seg) or (speakers[0]["label"] if speakers
                                      else SPK.DEFAULT_LABEL)

        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        slot = max(0.05, seg_end - seg_start)

        # Pause between this segment and this speaker's next (or the tail of
        # the video).
        next_start = next_start_for(label, i, seg_end)
        gap = max(0.0, next_start - seg_end)
        available = slot + gap * BORROW_FRACTION

        # Give the estimator this segment's own evidence: what the original
        # speaker said here and how long they took over it.
        # Prefer the generator's OWN duration estimate when it exposes one:
        # asking F5 for exactly what it would produce anyway means it never has
        # to internally compress or pad. Fall back to the text-ratio model.
        natural = None
        if natural_for is not None:
            try:
                natural = natural_for(text, seg, label)
            except Exception:
                natural = None
        if not natural or natural <= 0.05:
            if natural_fn is not None:
                try:
                    natural = natural_fn(text)
                except Exception:
                    natural = None
        if not natural or natural <= 0.05:
            model = duration_models.get(label, duration_model)
            natural = model.estimate(
                text, source_text=(seg.get("text") or "").strip(), slot=slot)

        # Give the generator the length it actually wants.
        #
        # This used to force every unit into its own slot, capped at `available`.
        # That converts a mild, uniform mismatch into violent per-unit
        # distortion: on real footage the overall need was only 1.15x the
        # available time, but individual units ranged from 0.85x to 1.76x, so
        # some phrases were crushed to unintelligibility while others were
        # padded with silence. Uneven compression is what reads as "the voice is
        # inconsistent" and "the words crash into each other".
        #
        # Instead each unit gets its natural length, overruns cascade into the
        # following pause, and any residual overflow is taken out globally at
        # the end of this function — one gentle scale applied evenly rather than
        # a different violent one per phrase.
        target = natural

        # Safety bounds only. With `natural` as the target these should not
        # normally bind; they exist so a bad estimate cannot produce absurdity.
        rate = natural / max(target, 1e-3)
        if target > available * 2.5 and available > 0.2:
            target = available * 2.5
            rate = natural / target

        # Never start before THIS SPEAKER's previous line finished. Another
        # speaker's position is not a constraint.
        cursor = cursors.get(label, 0.0)
        start = max(seg_start, cursor)
        plan.append({
            "index": i, "text": text, "start": start,
            "duration": target, "natural": natural, "rate": rate,
            "speaker": label,
            "ref_index": SPK.ref_index_for(speakers, label),
        })
        cursors[label] = start + target

    # If cascading pushed us past the end of the video, compress everything by
    # the (small) residual so the last word still lands inside the picture.
    # The end of the plan is the latest any speaker runs, not a single cursor.
    end = max(cursors.values()) if cursors else 0.0
    if plan and end > video_duration > 0:
        squeeze = video_duration / end
        print(f"[DUB] Timeline overflows by {end - video_duration:.2f}s; "
              f"compressing by {1 / squeeze:.3f}x")
        # The re-serialisation below has to be per speaker too. Applying one
        # running cursor here would undo the whole point of the change: it would
        # push every overlapping speaker back into a queue after the squeeze.
        tightened: dict = {}
        for p in plan:
            label = p.get("speaker") or SPK.DEFAULT_LABEL
            c = tightened.get(label, 0.0)
            p["start"] *= squeeze
            p["duration"] *= squeeze
            p["start"] = max(p["start"], c)
            tightened[label] = p["start"] + p["duration"]

    return plan


def overlap_profile(plan: list) -> list:
    """
    Where more than one generated voice is speaking at once.

    Returns `[{"start", "end", "speakers"}]` for each region with two or more
    active speakers, which is what the manual check reports and what the log
    line below is built from. Purely diagnostic: nothing in the pipeline
    branches on it.
    """
    edges = set()
    for unit in plan or []:
        start = float(unit.get("start", 0.0))
        end = start + float(unit.get("duration", 0.0))
        if end > start:
            edges.add(start)
            edges.add(end)
    points = sorted(edges)
    regions = []
    for a, b in zip(points, points[1:]):
        if b - a <= 1e-9:
            continue
        mid = (a + b) / 2.0
        active = sorted({
            (u.get("speaker") or SPK.DEFAULT_LABEL)
            for u in plan or []
            if float(u.get("start", 0.0)) <= mid
            < float(u.get("start", 0.0)) + float(u.get("duration", 0.0))
        })
        if len(active) > 1:
            regions.append({"start": round(a, 3), "end": round(b, 3),
                            "speakers": active})
    return regions


@contextmanager
def _with_seed(engine, seed):
    """
    Temporarily override the engine's sampling seed.

    IndicF5 is stochastic and the engine is a warm singleton, so re-rolling one
    phrase means swapping the seed for exactly one call and putting it back.
    """
    if seed is None or not hasattr(engine, "seed"):
        yield
        return
    previous = engine.seed
    engine.seed = int(seed)
    try:
        yield
    finally:
        engine.seed = previous


def synth_unit(unit: dict, ref_path: str, ref_text: str, ref_duration: float,
               target_lang: str, seed=None) -> np.ndarray:
    """
    Synthesize one planned unit and apply the conditioning every unit gets:
    trim the model's trailing padding, shorten if it overruns its slot, fade
    both edges. Returns the wave ready to be laid onto a track.

    Raises on failure so the caller decides whether a gap or an error is right.
    """
    engine = tts_engines.get_engine(target_lang)
    lang = L.synth_lang(target_lang)
    use_fix = isinstance(engine, tts_engines.IndicF5Engine)

    with _with_seed(engine, seed):
        if use_fix:
            # fix_duration is the TOTAL (reference + generated) length.
            wave, sr = _synth_fixed(engine, unit["text"], ref_path, ref_text,
                                    lang, ref_duration + unit["duration"])
        else:
            wave, sr = engine.synthesize(unit["text"], ref_path, ref_text, lang=lang)

    if sr != SAMPLE_RATE and wave.size:
        import librosa
        wave = librosa.resample(wave, orig_sr=sr, target_sr=SAMPLE_RATE)
    wave = np.asarray(wave, dtype=np.float32)

    # Trim the silence F5 pads onto the end of a unit.
    #
    # The model fills only ~73% of whatever duration it is given with actual
    # speech and pads the remainder. Left in place, that padding sits INSIDE
    # the unit, so the dub pauses at arbitrary points rather than where the
    # speaker paused. Trimming it puts the speech at the unit's start and lets
    # the gap to the next unit — which came from detecting the speaker's real
    # pauses — carry the rhythm instead.
    wave = _trim_silence(wave)

    want = int(round(unit["duration"] * SAMPLE_RATE))
    # Never stretch a unit UP to fill its slot: that would re-introduce the
    # padding just removed. Only shorten if it genuinely overruns.
    if wave.size > want:
        wave = _fit_length(wave, want)

    # Segments are dropped into a track of digital zeros, so without these a
    # segment starting or ending on a non-zero sample steps straight from
    # silence to full amplitude and clicks at every phrase boundary.
    return _edge_fade(wave, EDGE_FADE_MS)


def _overlap_gain(track: np.ndarray, plan: list, waves: dict,
                  fade_ms: float = EDGE_FADE_MS,
                  ceiling: float = 1.0) -> np.ndarray:
    """
    Scale down only where several generated voices are simultaneously active.

    Additive placement is right — two actors really do occupy the same samples —
    but summing them can exceed full scale, and the pre-existing guard (`if peak
    > 1: track /= peak`) is the wrong answer to that: one loud 300 ms overlap
    would attenuate the entire film, which is audible everywhere the overlap is
    not.

    Two stages, both local:

    1. **Power-preserving gain, `1/sqrt(n)`** for `n` voices active at a sample.
       Power sums linearly, so `n` equal uncorrelated voices scaled by
       `1/sqrt(n)` keep the total power constant:

           1 voice  -> 1.000
           2 voices -> 0.707 each
           3 voices -> 0.577 each

    2. **A local ceiling, inside the overlap regions only.** Stage 1 preserves
       *power*, which is the right model for uncorrelated speech but not a
       bound: two near-full-scale voices summing in phase give `2/sqrt(2)` =
       1.41, which still clips. Rather than let the global limiter answer that
       by scaling the whole film, the excess is taken out of the overlap region
       itself. Solo audio is never touched.

    Crucially both stages are *local curves*, not per-phrase decisions. A phrase
    that overlaps another for 300 ms is scaled for those 300 ms and left alone
    for the rest, rather than being quietened throughout. They are also built
    from the same plan `assemble` places from, so the two cannot disagree about
    where the overlap is.

    **With one speaker both stages are exactly 1.0 everywhere**, so this is a
    no-op and a single-speaker track is byte-for-byte what it always was. That is
    not a coincidence — it is why the single-speaker regression tests still pass.

    Edges are smoothed over `fade_ms` so a gain step cannot click. The window is
    a box filter applied twice, which is cheap and has no ringing; a sharp step
    at a phrase boundary would be audible as a tick.
    """
    total = len(track)
    if total == 0:
        return np.ones(0, dtype=np.float32)

    # 1. Count, per sample, how many generated voices are active.
    active = np.zeros(total + 1, dtype=np.int32)
    for unit in plan or []:
        wave = waves.get(unit.get("index"))
        if wave is None or wave.size == 0:
            continue
        at = int(round(float(unit.get("start", 0.0)) * SAMPLE_RATE))
        if at >= total:
            continue
        end = min(total, at + wave.size)
        if end <= at:
            continue
        active[at] += 1
        active[end] -= 1
    counts = np.cumsum(active[:total])

    if counts.max(initial=0) <= 1:
        return np.ones(total, dtype=np.float32)

    def smooth(curve: np.ndarray) -> np.ndarray:
        n = max(1, int(SAMPLE_RATE * fade_ms / 1000.0))
        if n > 1 and curve.size > n:
            kernel = np.ones(n, dtype=np.float32) / n
            curve = np.convolve(curve, kernel, mode="same").astype(np.float32)
            curve = np.convolve(curve, kernel, mode="same").astype(np.float32)
        return curve

    # 2. Stage 1: the power-preserving curve.
    gain = smooth((1.0 / np.sqrt(np.maximum(1, counts))).astype(np.float32))

    # 3. Stage 2: the local ceiling.
    #
    #    Measured over the WHOLE track but applied only where the mask is set, so
    #    the reduction lands in the overlap and never in a solo passage. The mask
    #    is the same smoothed curve the gain uses, which matters: a single pass
    #    is not enough, because at the ramp into an overlap the mask is partial,
    #    so the reduction there is partial too and the peak survives at the
    #    boundary. Each iteration takes the remaining excess out of the masked
    #    region and leaves solo audio untouched, and the sequence converges
    #    geometrically — measured, it needs 16 iterations to bring a 1.27 peak
    #    under 1.0 while a solo passage stays at exactly 0.5. The loop is bounded
    #    so a pathological input cannot spin, and it exits as soon as the peak is
    #    inside the ceiling, which is the common case.
    mask = smooth((counts > 1).astype(np.float32))
    if mask.size and float(mask.max()) > 0.0:
        for _ in range(24):
            peak = float(np.max(np.abs(track * gain))) if track.size else 0.0
            if peak <= ceiling:
                break
            scale = ceiling / peak
            gain = (gain * (1.0 + (scale - 1.0) * mask)).astype(np.float32)

    return gain


def assemble(plan: list, waves: dict, video_duration: float,
             overlap_gain: bool = True) -> np.ndarray:
    """
    Lay already-synthesized unit waves onto a track of exactly
    `video_duration` seconds. `waves` maps plan index to wave; a missing entry
    leaves silence in that slot.

    Placement is additive, so two speakers may occupy the same samples — which
    is the whole point of Phase 5. Where they do, `_overlap_gain` scales the sum
    locally so the overlap cannot clip and the rest of the track is untouched.

    `overlap_gain=False` reproduces the pre-Phase-5 mix exactly; it exists so the
    difference is testable rather than asserted.
    """
    total_samples = int(round(video_duration * SAMPLE_RATE))
    track = np.zeros(total_samples, dtype=np.float32)

    for unit in plan:
        wave = waves.get(unit["index"])
        if wave is None or wave.size == 0:
            continue
        at = int(round(unit["start"] * SAMPLE_RATE))
        if at >= total_samples:
            break
        end = min(total_samples, at + wave.size)
        track[at:end] += wave[: end - at]

    # Local overlap management, applied BEFORE the peak check so the check sees
    # what will actually be written. With one speaker the curve is 1.0 and this
    # is a no-op.
    if overlap_gain and track.size:
        track = track * _overlap_gain(track, plan, waves)

    peak = float(np.max(np.abs(track))) if track.size else 0.0

    # The guard that was missing. A track of exactly the right length made
    # entirely of zeros passed every duration check in this pipeline and was
    # reported as a clean run — that is how a completely untrained model went
    # unnoticed. Length is not evidence of content.
    if peak < 1e-4:
        raise RuntimeError(
            "Synthesized track is silent (peak %.2e). The TTS model produced no "
            "audio — check that its weights actually loaded." % peak)

    # Last-resort safety only. `_overlap_gain` has already handled the overlap
    # case; anything still above full scale is three or more voices summing in
    # phase, or a unit that came back louder than it should have. This is the
    # guard, not the strategy: scaling the whole film for one loud moment is
    # exactly what the local curve exists to avoid.
    if peak > 1.0:
        print(f"[DUB] Peak {peak:.3f} above full scale after overlap gain; "
              f"scaling the track by {1 / peak:.3f}")
        track /= peak
    return track


def segment_path(cache_dir: str, index) -> str:
    return os.path.join(cache_dir, f"unit_{index}.wav")


def write_unit(cache_dir: str, index, wave: np.ndarray) -> str:
    import soundfile as sf
    os.makedirs(cache_dir, exist_ok=True)
    path = segment_path(cache_dir, index)
    sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")
    return path


def read_unit(cache_dir: str, index):
    import soundfile as sf
    path = segment_path(cache_dir, index)
    if not os.path.exists(path):
        return None
    wave, _ = sf.read(path, dtype="float32")
    return np.asarray(wave, dtype=np.float32).reshape(-1)


def unit_seconds(cache_dir: str, index) -> float | None:
    """
    How long a phrase's cached audio actually is.

    Reads the header rather than the samples: this is asked for every phrase
    every time the timeline is listed, and the answer is in the first few
    bytes. Reflects the current take, so it stays right after an edit.
    """
    import soundfile as sf
    path = segment_path(cache_dir, index)
    if not os.path.exists(path):
        return None
    try:
        info = sf.info(path)
        return round(info.frames / float(info.samplerate or SAMPLE_RATE), 3)
    except Exception:                                        # noqa: BLE001
        return None


def synthesize_timeline(plan: list, ref_path: str, ref_text: str, ref_duration: float,
                        target_lang: str, video_duration: float,
                        progress=None, cache_dir: str = None,
                        speakers: list = None) -> np.ndarray:
    """
    Synthesize every planned segment and lay it onto a track of exactly
    `video_duration` seconds.

    When `cache_dir` is given, each unit's conditioned wave is written there so
    a later edit can rebuild one phrase instead of the whole track.

    Each unit is spoken in ITS OWN speaker's reference when `speakers` is given;
    the `ref_path`/`ref_text`/`ref_duration` arguments are the single-voice
    fallback and are what a job with no registry uses, so the previous
    behaviour is reproduced exactly.
    """
    waves = {}
    failures = []
    silent = []
    global_ref = {"path": ref_path, "text": ref_text, "duration": ref_duration}

    for n, unit in enumerate(plan):
        if progress:
            progress(n, len(plan))
        ref = SPK.reference_for(unit, speakers, global_ref) or global_ref
        try:
            wave = synth_unit(unit, ref.get("path") or ref_path,
                              ref.get("text") or ref_text,
                              float(ref.get("duration") or ref_duration),
                              target_lang)
        except Exception as e:
            print(f"[DUB] Segment {unit['index']} failed ({e}); leaving silence")
            failures.append((unit["index"], str(e)))
            continue

        # A segment that comes back silent is a failure too — it just does not
        # raise. Track it, or a silent model produces a silent dub that looks
        # like a completely successful run.
        if wave.size == 0 or float(np.max(np.abs(wave))) < 1e-4:
            silent.append(unit["index"])

        waves[unit["index"]] = wave
        if cache_dir:
            write_unit(cache_dir, unit["index"], wave)

    if failures and len(failures) == len(plan):
        raise RuntimeError(
            f"Every one of {len(plan)} segments failed to synthesize. "
            f"First error: {failures[0][1]}")
    if failures:
        print(f"[DUB] WARNING: {len(failures)}/{len(plan)} segments failed")
    if silent:
        print(f"[DUB] WARNING: {len(silent)}/{len(plan)} segments came back "
              f"silent: {silent[:8]}")

    return assemble(plan, waves, video_duration)


def _synth_fixed(engine, text, ref_path, ref_text, lang, fix_duration):
    """Call IndicF5 with an explicit total duration."""
    from f5_tts.infer.utils_infer import infer_batch_process

    model = engine.load()
    if hasattr(engine, "_seed_rng"):
        engine._seed_rng()
    audio_t, sr = engine._load_ref(ref_path)
    rtext = engine._normalize_ref_text(ref_text)

    # The fork's infer_batch_process returns (wave, sr, spec) directly; the pip
    # release returns a generator. tts_engines pins the fork on sys.path.
    wave, out_sr, _ = infer_batch_process(
        (audio_t, sr), rtext, [text.strip()],
        model.ema_model, model.vocoder,
        mel_spec_type="vocos", device=engine.device,
        nfe_step=engine.nfe_step, cross_fade_duration=0.0,
        fix_duration=fix_duration, progress=None,
    )
    return np.asarray(wave, dtype=np.float32).reshape(-1), out_sr


def _speaker_duration_models(speakers: list, reference: dict,
                             source_lang: str, target_lang: str,
                             baseline: DurationModel) -> dict:
    """
    One `DurationModel` per speaker, each calibrated on that speaker's own
    reference clip.

    A shared model measures one actor's tempo and applies it to everyone. Two
    people in one scene rarely talk at the same rate, so the slower one gets
    compressed and the faster one gets padded — the same "the voice is
    inconsistent" complaint the per-unit fix in `plan_timeline` addressed,
    arriving from a different direction.

    A speaker whose reference is missing or too short keeps the baseline, which
    is the single-voice behaviour.
    """
    models = {}
    for entry in speakers or []:
        label = entry.get("label")
        ref = entry.get("reference") or {}
        duration = float(ref.get("duration") or 0.0)
        text = ref.get("text") or ""
        if not label:
            continue
        if ref is reference or (duration <= 0.0 and not text):
            models[label] = baseline
            continue
        models[label] = DurationModel(duration, text,
                                      source_lang=source_lang,
                                      target_lang=target_lang)
    return models


def build_dubbed_track(segments: list, translated_texts: list, reference: dict,
                       target_lang: str, video_duration: float, out_path: str,
                       source_lang: str = None, progress=None,
                       source_audio: str = None, cache_dir: str = None,
                       speakers: list = None) -> list:
    """
    Full segment-level dub. Writes a WAV of exactly `video_duration` seconds.

    `reference` is the dict returned by reference_audio.build_reference(): the
    single-voice fallback, and the only voice a job without a speaker registry
    ever uses.

    `speakers` is the optional registry from `reference_audio.build_references`.
    When it is present each phrase is spoken in its own speaker's reference and
    timed against that speaker's tempo; when it is absent or empty the run is
    the single-speaker pipeline, unchanged.
    """
    speakers = speakers or []
    dm = DurationModel(reference["duration"], reference["text"],
                       source_lang=source_lang, target_lang=target_lang)
    models = _speaker_duration_models(speakers, reference, source_lang,
                                      target_lang, dm)

    units, texts = prepare_units(segments, translated_texts,
                                 source_audio, video_duration)

    engine = tts_engines.get_engine(target_lang)
    natural_fn = None
    natural_for = None
    if hasattr(engine, "natural_duration"):
        _lang = L.synth_lang(target_lang)

        def natural_for(text, seg=None, label=None):
            # F5's own estimate is measured against whichever reference the
            # phrase will actually be spoken in, so the per-speaker call passes
            # that speaker's clip rather than the global one. Resolving by label
            # rather than by a plan unit is deliberate: this runs while the plan
            # is still being built, so no unit exists yet.
            ref = reference
            for entry in speakers:
                if (entry.get("label") == label
                        and (entry.get("reference") or {}).get("path")):
                    ref = entry["reference"]
                    break
            return engine.natural_duration(text, ref["path"], ref.get("text", ""),
                                           lang=_lang)

    plan = plan_timeline(units, texts, dm, video_duration, natural_fn=natural_fn,
                         speakers=speakers, duration_models=models,
                         natural_for=natural_for)

    if not plan:
        raise RuntimeError("Nothing to synthesize: no non-empty translated segments")

    speech = sum(p["duration"] for p in plan)
    print(f"[DUB] {len(plan)} segments, {speech:.1f}s speech in a "
          f"{video_duration:.1f}s video ({L.display_name(target_lang)})")
    if speakers:
        print(SPK.summarize(speakers, segments))

    # Where two actors talk over each other. Purely informational, but it is
    # the one thing about a multi-speaker dub that is invisible in the numbers:
    # a total speech time that exceeds the video is normal, not a bug.
    regions = overlap_profile(plan)
    if regions:
        total = sum(r["end"] - r["start"] for r in regions)
        print(f"[DUB] {len(regions)} overlap region(s), {total:.1f}s of "
              f"simultaneous dialogue "
              f"(e.g. {regions[0]['speakers']} at {regions[0]['start']:.1f}s)")

    track = synthesize_timeline(plan, reference["path"], reference["text"],
                                reference["duration"], target_lang,
                                video_duration, progress, cache_dir=cache_dir,
                                speakers=speakers)

    import soundfile as sf
    sf.write(out_path, track, SAMPLE_RATE, subtype="PCM_16")

    actual = len(track) / SAMPLE_RATE
    print(f"[DUB] Wrote {out_path} — {actual:.3f}s "
          f"(video {video_duration:.3f}s, delta {abs(actual - video_duration) * 1000:.0f}ms)")
    # The plan goes back to the caller so a later edit knows where every phrase
    # sits and what it was asked to say.
    return plan

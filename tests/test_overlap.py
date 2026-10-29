"""
Phase 5: overlapping speakers, and overlap-safe assembly.

The bug this phase fixes is one line long and easy to miss: a single global
cursor meant that in a conversation the second actor could not start until the
first had finished. `SPEAKER_01` at 11.5s was pushed to 13.0s because
`SPEAKER_00` was still talking. That destroys conversational timing and puts the
voice on the wrong side of the cut.

Two halves are pinned here, and they fail independently:

**Scheduling** — a different speaker may start while another is talking; the
SAME speaker still may not overlap themselves. Single-speaker scheduling must be
bit-for-bit what it was, which is the reason the per-speaker cursor is safe to
introduce at all.

**Assembly** — additive placement is kept, and the level is managed *locally*
where voices coincide. The gain curve is `1/sqrt(n)`, and it is exactly 1.0 when
only one voice is active, so a single-speaker track is unchanged byte for byte.

No models, no network, no GPU.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import dubbing  # noqa: E402
import speaker_diarization as SD  # noqa: E402
import speakers as SPK  # noqa: E402

SR = dubbing.SAMPLE_RATE


def seg(start, end, text, speaker):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def registry(*labels, duration=8.0):
    return [{"label": label, "total_speech": 20.0, "segment_count": 4,
             "reference": {"path": f"/tmp/refs/{label}.wav",
                           "text": f"reference for {label}",
                           "start": 0.0, "end": duration,
                           "duration": duration},
             "reference_score": 1.0, "fallback": False}
            for label in labels]


def model():
    return dubbing.DurationModel(8.0, "a reference sentence here",
                                 source_lang="en", target_lang="hi")


def plan_for(rows, video_duration, speakers=None, texts=None):
    """
    Plan `rows`, defaulting to a translation proportionate to each source line.

    The duration model scales a phrase by the ratio of translation length to
    source length, so a translation far longer than its source legitimately
    produces a phrase far longer than its slot — and then the closing squeeze
    moves everything, which would make these tests about the squeeze instead of
    about overlap. A proportionate default keeps each phrase near its own slot,
    which is the situation the overlap rules are about.
    """
    if texts is None:
        texts = [_matching(row.get("text", "")) for row in rows]
    return dubbing.plan_timeline(rows, texts, model(), video_duration,
                                 speakers=speakers or [])


def _matching(source: str) -> str:
    """A stand-in translation with a comparable character count."""
    n = max(4, len(source or ""))
    return ("बोली " * ((n // 4) + 1))[:n]


def span(unit):
    """(start, end) of a planned unit."""
    return (round(unit["start"], 4),
            round(unit["start"] + unit["duration"], 4))


def tone(seconds, freq=300.0, amp=0.3):
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# The headline requirement
# ---------------------------------------------------------------------------
class DifferentSpeakerOverlapTests(unittest.TestCase):
    """A different actor may start while another is still talking."""

    def test_the_headline_case_is_not_serialised(self):
        """
        The requirement, as a test.

            SPEAKER_00  10.0 ──────── 13.0
            SPEAKER_01        11.5 ──────── 14.0

        must stay that way. It must NOT become SPEAKER_01 at 13.0-15.5.
        """
        rows = [seg(10.0, 13.0, "first line", "SPEAKER_00"),
                seg(11.5, 14.0, "second line", "SPEAKER_01")]
        plan = plan_for(rows, 20.0, registry("SPEAKER_00", "SPEAKER_01"))

        a = next(u for u in plan if u["speaker"] == "SPEAKER_00")
        b = next(u for u in plan if u["speaker"] == "SPEAKER_01")

        # A starts where the original speaker did.
        self.assertAlmostEqual(a["start"], 10.0, places=2)
        # B starts near its own original start, NOT after A finishes.
        self.assertAlmostEqual(b["start"], 11.5, places=2)
        self.assertLess(b["start"], 13.0,
                        "SPEAKER_01 was pushed behind SPEAKER_00's line; the "
                        "global cursor is back")
        # And they genuinely coincide.
        self.assertLess(b["start"], a["start"] + a["duration"])

    def test_the_old_behaviour_would_have_failed_this(self):
        # A guard on the guard: with one shared cursor B lands at 13.0. This
        # documents what the bug looked like, so a regression is recognisable.
        rows = [seg(10.0, 13.0, "first", "SPEAKER_00"),
                seg(11.5, 14.0, "second", "SPEAKER_01")]
        plan = plan_for(rows, 20.0, registry("SPEAKER_00", "SPEAKER_01"))
        b = next(u for u in plan if u["speaker"] == "SPEAKER_01")
        self.assertNotAlmostEqual(b["start"], 13.0, places=1)

    def test_two_speakers_each_keep_their_own_reference(self):
        rows = [seg(1.0, 4.0, "first speaker line here", "SPEAKER_00"),
                seg(2.0, 5.0, "second speaker line here", "SPEAKER_01")]
        reg = registry("SPEAKER_00", "SPEAKER_01")
        plan = plan_for(rows, 10.0, reg)
        a = next(u for u in plan if u["speaker"] == "SPEAKER_00")
        b = next(u for u in plan if u["speaker"] == "SPEAKER_01")
        self.assertEqual(a["ref_index"], 0)
        self.assertEqual(b["ref_index"], 1)
        self.assertEqual(SPK.reference_for(a, reg)["path"],
                         "/tmp/refs/SPEAKER_00.wav")
        self.assertEqual(SPK.reference_for(b, reg)["path"],
                         "/tmp/refs/SPEAKER_01.wav")

    def test_basic_two_speaker_overlap_from_the_brief(self):
        # A: 1.0-4.0, B: 2.0-5.0
        rows = [seg(1.0, 4.0, "first speaker line here", "SPEAKER_00"),
                seg(2.0, 5.0, "second speaker line here", "SPEAKER_01")]
        plan = plan_for(rows, 10.0, registry("SPEAKER_00", "SPEAKER_01"))
        a = next(u for u in plan if u["speaker"] == "SPEAKER_00")
        b = next(u for u in plan if u["speaker"] == "SPEAKER_01")
        self.assertAlmostEqual(a["start"], 1.0, places=2)
        self.assertAlmostEqual(b["start"], 2.0, places=2)
        self.assertLess(b["start"], 4.0, "B was pushed past A's end")
        # Both are active somewhere in 2.0-4.0.
        regions = dubbing.overlap_profile(plan)
        self.assertTrue(regions)
        self.assertTrue(any(set(r["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
                            for r in regions))

    def test_three_speakers_all_overlap(self):
        rows = [seg(1.0, 5.0, "first speaker line here", "SPEAKER_00"),
                seg(2.0, 6.0, "second speaker line here", "SPEAKER_01"),
                seg(3.0, 7.0, "third speaker line here", "SPEAKER_02")]
        reg = registry("SPEAKER_00", "SPEAKER_01", "SPEAKER_02")
        plan = plan_for(rows, 12.0, reg)
        starts = {u["speaker"]: round(u["start"], 2) for u in plan}
        self.assertAlmostEqual(starts["SPEAKER_00"], 1.0, places=1)
        self.assertAlmostEqual(starts["SPEAKER_01"], 2.0, places=1)
        self.assertAlmostEqual(starts["SPEAKER_02"], 3.0, places=1)
        # Each keeps its own reference.
        self.assertEqual([u["ref_index"] for u in plan], [0, 1, 2])

    def test_three_speakers_produce_a_three_way_overlap_region(self):
        rows = [seg(1.0, 6.0, "first speaker line here", "SPEAKER_00"),
                seg(2.0, 6.0, "second speaker line here", "SPEAKER_01"),
                seg(3.0, 6.0, "third speaker line here", "SPEAKER_02")]
        plan = plan_for(rows, 12.0,
                        registry("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"))
        three = [r for r in dubbing.overlap_profile(plan)
                 if len(r["speakers"]) == 3]
        self.assertTrue(three, "expected a region with all three active")

    def test_a_long_line_does_not_steal_the_other_speakers_space(self):
        # The brief: "A long SPEAKER_00 phrase must not steal timing space from
        # SPEAKER_01." A's overrun must not move B.
        rows = [seg(0.0, 2.0, "a very long line indeed " * 6, "SPEAKER_00"),
                seg(2.0, 4.0, "second speaker line here", "SPEAKER_01")]
        plan = plan_for(rows, 20.0, registry("SPEAKER_00", "SPEAKER_01"))
        a = next(u for u in plan if u["speaker"] == "SPEAKER_00")
        b = next(u for u in plan if u["speaker"] == "SPEAKER_01")
        self.assertAlmostEqual(b["start"], 2.0, places=2,
                               msg="SPEAKER_01 was displaced by SPEAKER_00's "
                                   "overrun")
        # A may well run long and overlap B; that is the point.
        self.assertGreater(a["start"] + a["duration"], 2.0)


# ---------------------------------------------------------------------------
# Same speaker must still be serialised
# ---------------------------------------------------------------------------
class SameSpeakerSerialisationTests(unittest.TestCase):
    """One actor cannot talk over themselves."""

    def test_the_same_speaker_is_still_serialised(self):
        # A1: 1.0-4.0, A2: 2.0-5.0. Deterministic: A2 waits for A1.
        rows = [seg(1.0, 4.0, "first line from one actor", "SPEAKER_00"),
                seg(2.0, 5.0, "second line from one actor", "SPEAKER_00")]
        plan = plan_for(rows, 12.0, registry("SPEAKER_00"))
        first = plan[0]
        second = plan[1]
        self.assertGreaterEqual(
            round(second["start"], 4),
            round(first["start"] + first["duration"], 4) - 1e-6,
            "a speaker overlapped themselves")

    def test_same_speaker_overlap_never_appears_in_the_profile(self):
        rows = [seg(1.0, 4.0, "first line from one actor", "SPEAKER_00"),
                seg(2.0, 5.0, "second line from one actor", "SPEAKER_00"),
                seg(3.0, 6.0, "three", "SPEAKER_00")]
        plan = plan_for(rows, 15.0, registry("SPEAKER_00"))
        self.assertEqual(dubbing.overlap_profile(plan), [],
                         "one speaker cannot produce an overlap region")

    def test_serialisation_is_deterministic_across_runs(self):
        rows = [seg(1.0, 4.0, "first line from one actor", "SPEAKER_00"),
                seg(2.0, 5.0, "second line from one actor", "SPEAKER_00"),
                seg(2.5, 6.0, "three", "SPEAKER_00")]
        first = plan_for(rows, 15.0, registry("SPEAKER_00"))
        second = plan_for(rows, 15.0, registry("SPEAKER_00"))
        self.assertEqual([span(u) for u in first], [span(u) for u in second])

    def test_mixed_speakers_serialise_only_within_each_speaker(self):
        # A, B, A, B — each speaker's own lines queue; the two speakers
        # interleave freely.
        rows = [seg(1.0, 3.0, "first line from the first actor", "SPEAKER_00"),
                seg(1.5, 3.5, "first line from the second actor", "SPEAKER_01"),
                seg(2.0, 4.0, "second line from the first actor", "SPEAKER_00"),
                seg(2.5, 4.5, "second line from the second actor", "SPEAKER_01")]
        plan = plan_for(rows, 12.0, registry("SPEAKER_00", "SPEAKER_01"))
        by_speaker = {}
        for unit in plan:
            by_speaker.setdefault(unit["speaker"], []).append(unit)
        for label, units in by_speaker.items():
            for a, b in zip(units, units[1:]):
                self.assertGreaterEqual(
                    round(b["start"], 4),
                    round(a["start"] + a["duration"], 4) - 1e-6,
                    f"{label} overlapped themselves")


# ---------------------------------------------------------------------------
# Single-speaker regression
# ---------------------------------------------------------------------------
class SingleSpeakerRegressionTests(unittest.TestCase):
    """The reason per-speaker cursors are safe to introduce."""

    def test_one_speaker_scheduling_is_unchanged(self):
        rows = [seg(0.0, 2.0, "one two three", "SPEAKER_00"),
                seg(2.5, 4.5, "four five six", "SPEAKER_00"),
                seg(5.0, 7.0, "seven eight nine", "SPEAKER_00")]
        texts = ["एक दो तीन", "चार पाँच छह", "सात आठ नौ"]
        # No registry: exactly the pre-Phase-1 call.
        before = dubbing.plan_timeline(rows, texts, model(), 7.0)
        # The implicit single-speaker registry a real run builds.
        after = dubbing.plan_timeline(rows, texts, model(), 7.0,
                                      speakers=SPK.implicit_speaker(
                                          {"path": "/tmp/r.wav",
                                           "text": "a reference sentence here",
                                           "duration": 8.0}))
        self.assertEqual(len(before), len(after))
        for b, a in zip(before, after):
            for key in ("index", "text", "start", "duration", "natural", "rate"):
                self.assertEqual(b[key], a[key],
                                 f"{key} moved for phrase {b['index']}")

    def test_one_speaker_never_produces_an_overlap_region(self):
        rows = [seg(0.0, 2.0, "first line from one actor", "SPEAKER_00"),
                seg(1.0, 3.0, "second line from one actor", "SPEAKER_00")]
        plan = plan_for(rows, 8.0, registry("SPEAKER_00"))
        self.assertEqual(dubbing.overlap_profile(plan), [])

    def test_a_label_less_run_is_treated_as_one_speaker(self):
        rows = [{"start": 0.0, "end": 2.0, "text": "a"},
                {"start": 1.0, "end": 3.0, "text": "b"}]
        plan = plan_for(rows, 8.0)
        self.assertTrue(all(u["speaker"] == SPK.DEFAULT_LABEL for u in plan))
        self.assertEqual(dubbing.overlap_profile(plan), [])

    def test_borrowing_still_uses_the_speakers_own_next_line(self):
        # For one speaker, "next line of mine" and "next line in the list" are
        # the same thing, so the borrow window is unchanged.
        rows = [seg(0.0, 2.0, "short", "SPEAKER_00"),
                seg(3.0, 5.0, "next", "SPEAKER_00")]
        plan = plan_for(rows, 10.0, registry("SPEAKER_00"))
        self.assertAlmostEqual(plan[0]["start"], 0.0, places=3)


# ---------------------------------------------------------------------------
# Assembly and local gain
# ---------------------------------------------------------------------------
class OverlapAssemblyTests(unittest.TestCase):
    """`assemble` stays additive, with the level managed locally."""

    def test_the_track_is_still_exactly_the_video_length(self):
        plan = [{"index": 0, "start": 1.0, "duration": 3.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 2.0, "duration": 3.0,
                 "speaker": "SPEAKER_01"}]
        waves = {0: tone(3.0), 1: tone(3.0, 500.0)}
        track = dubbing.assemble(plan, waves, 7.0)
        self.assertEqual(len(track), int(7.0 * SR))

    def test_a_non_integer_video_duration_is_honoured(self):
        plan = [{"index": 0, "start": 0.5, "duration": 2.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 1.0, "duration": 2.0,
                 "speaker": "SPEAKER_01"}]
        waves = {0: tone(2.0), 1: tone(2.0, 500.0)}
        track = dubbing.assemble(plan, waves, 3.777)
        self.assertEqual(len(track), round(3.777 * SR))

    def test_both_speakers_are_present_in_the_overlap(self):
        plan = [{"index": 0, "start": 0.0, "duration": 2.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 1.0, "duration": 2.0,
                 "speaker": "SPEAKER_01"}]
        # Distinct frequencies, so each can be measured independently.
        waves = {0: tone(2.0, 300.0), 1: tone(2.0, 3000.0)}
        track = dubbing.assemble(plan, waves, 4.0)
        spec = np.abs(np.fft.rfft(track)) ** 2
        freqs = np.fft.rfftfreq(len(track), 1 / SR)

        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(spec[m].sum())

        self.assertGreater(band(200, 400), 0.0, "SPEAKER_00 is missing")
        self.assertGreater(band(2900, 3100), 0.0, "SPEAKER_01 is missing")

    def test_the_gain_curve_is_one_everywhere_for_a_single_voice(self):
        plan = [{"index": 0, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 2.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"}]
        waves = {0: tone(1.0), 1: tone(1.0)}
        track = np.zeros(int(4.0 * SR), dtype=np.float32)
        gain = dubbing._overlap_gain(track, plan, waves)
        self.assertTrue(np.allclose(gain, 1.0),
                        "a single speaker must not be attenuated at all")

    def test_a_single_speaker_track_is_byte_identical_to_the_pre_phase5_mix(self):
        """
        The regression lock for assembly.

        With one voice the overlap curve is exactly 1.0, so the gain-managed
        track must equal the plain additive sum sample for sample.
        """
        plan = [{"index": 0, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 1.5, "duration": 1.0,
                 "speaker": "SPEAKER_00"}]
        waves = {0: tone(1.0), 1: tone(1.0, 440.0)}
        managed = dubbing.assemble(plan, waves, 3.0, overlap_gain=True)
        plain = dubbing.assemble(plan, waves, 3.0, overlap_gain=False)
        np.testing.assert_array_equal(managed, plain)

    def test_overlap_gain_matches_one_over_sqrt_n(self):
        # 2 voices -> 0.707, 3 voices -> 0.577, at the centre of each region.
        two = [{"index": 0, "start": 0.0, "duration": 2.0,
                "speaker": "SPEAKER_00"},
               {"index": 1, "start": 0.0, "duration": 2.0,
                "speaker": "SPEAKER_01"}]
        waves = {0: tone(2.0), 1: tone(2.0, 500.0)}
        gain = dubbing._overlap_gain(np.zeros(int(2.0 * SR), dtype=np.float32),
                                     two, waves)
        self.assertAlmostEqual(float(gain[len(gain) // 2]), 1 / np.sqrt(2),
                               places=2)

        three = two + [{"index": 2, "start": 0.0, "duration": 2.0,
                        "speaker": "SPEAKER_02"}]
        waves3 = {**waves, 2: tone(2.0, 900.0)}
        gain3 = dubbing._overlap_gain(np.zeros(int(2.0 * SR), dtype=np.float32),
                                      three, waves3)
        self.assertAlmostEqual(float(gain3[len(gain3) // 2]), 1 / np.sqrt(3),
                               places=2)

    def test_non_overlapping_parts_are_not_attenuated(self):
        """
        The brief's requirement: do not lower an entire phrase because part of
        it overlaps. The solo portion of a phrase keeps full level.
        """
        plan = [{"index": 0, "start": 0.0, "duration": 4.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 3.0, "duration": 1.0,
                 "speaker": "SPEAKER_01"}]
        waves = {0: tone(4.0, 300.0), 1: tone(1.0, 900.0)}
        gain = dubbing._overlap_gain(np.zeros(int(4.0 * SR), dtype=np.float32),
                                     plan, waves)
        # Well inside SPEAKER_00's solo stretch.
        self.assertAlmostEqual(float(gain[int(0.5 * SR)]), 1.0, places=3)
        # Inside the overlap.
        self.assertLess(float(gain[int(3.4 * SR)]), 0.95)

    def test_a_loud_overlap_does_not_attenuate_the_whole_track(self):
        """
        The brief's other requirement: do not let one loud moment scale the
        film. A full-scale phrase outside the overlap must survive at level.
        """
        quiet = np.full(int(1.0 * SR), 0.5, dtype=np.float32)
        loud = np.full(int(1.0 * SR), 0.9, dtype=np.float32)
        plan = [{"index": 0, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 1.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                {"index": 2, "start": 2.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                # A loud pair at 3.0 that would clip if simply summed: 0.9 + 0.9
                # = 1.8, and 1/sqrt(2) only brings that to 1.27.
                {"index": 3, "start": 3.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                {"index": 4, "start": 3.0, "duration": 1.0,
                 "speaker": "SPEAKER_01"}]
        waves = {0: quiet, 1: quiet, 2: quiet, 3: loud, 4: loud}
        track = dubbing.assemble(plan, waves, 4.0)
        # The solo half-second is untouched, despite the loud overlap later.
        solo = float(np.max(np.abs(track[int(0.1 * SR):int(0.9 * SR)])))
        self.assertAlmostEqual(solo, 0.5, places=2,
                               msg="a solo phrase was attenuated by a distant "
                                   "overlap")
        # Nothing is out of range anywhere.
        self.assertLessEqual(float(np.max(np.abs(track))), 1.0 + 1e-6)

    def test_the_output_never_contains_invalid_samples(self):
        plan = [{"index": 0, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_01"},
                {"index": 2, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_02"}]
        waves = {i: np.full(int(SR), 0.9, dtype=np.float32) for i in range(3)}
        track = dubbing.assemble(plan, waves, 1.0)
        self.assertFalse(np.isnan(track).any())
        self.assertFalse(np.isinf(track).any())
        self.assertLessEqual(float(np.max(np.abs(track))), 1.0 + 1e-6)

    def test_three_in_phase_full_scale_voices_are_brought_under_full_scale(self):
        """
        Three voices at full scale sum to 3.0. `1/sqrt(3)` gives 1.73, which
        still clips, so the local ceiling has to take the rest — and it must do
        so without the global limiter firing, because that would scale
        everything.
        """
        plan = [{"index": i, "start": 0.0, "duration": 1.0,
                 "speaker": f"SPEAKER_{i:02d}"} for i in range(3)]
        waves = {i: np.full(int(SR), 1.0, dtype=np.float32) for i in range(3)}
        track = dubbing.assemble(plan, waves, 1.0)
        peak = float(np.max(np.abs(track)))
        self.assertLessEqual(peak, 1.0 + 1e-6,
                             "the track still clips after overlap gain")
        # Brought under full scale, but not crushed: the local ceiling takes
        # only the excess, so the result sits just below 1.0 rather than at
        # 1/sqrt(3) = 0.577.
        self.assertGreater(peak, 0.9,
                           "the overlap was over-attenuated; the local ceiling "
                           "should take only the excess above full scale")

    def test_the_silent_track_guard_still_fires(self):
        plan = [{"index": 0, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_01"}]
        waves = {0: np.zeros(SR, dtype=np.float32),
                 1: np.zeros(SR, dtype=np.float32)}
        with self.assertRaises(RuntimeError):
            dubbing.assemble(plan, waves, 2.0)


class BoundaryCaseTests(unittest.TestCase):
    """Overlaps at the awkward sizes and positions."""

    def _track(self, a_start, a_dur, b_start, b_dur, video=10.0):
        plan = [{"index": 0, "start": a_start, "duration": a_dur,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": b_start, "duration": b_dur,
                 "speaker": "SPEAKER_01"}]
        waves = {0: np.full(int(a_dur * SR), 0.6, dtype=np.float32),
                 1: np.full(int(b_dur * SR), 0.6, dtype=np.float32)}
        return plan, waves, dubbing.assemble(plan, waves, video)

    def test_a_one_sample_overlap(self):
        # Two samples short of an exact abutment.
        plan, waves, track = self._track(0.0, 1.0, 1.0 - 1 / SR, 1.0)
        self.assertEqual(len(track), int(10.0 * SR))
        self.assertFalse(np.isnan(track).any())

    def test_a_ten_millisecond_overlap(self):
        plan, _, track = self._track(0.0, 1.0, 0.990, 1.0)
        self.assertEqual(len(track), int(10.0 * SR))
        self.assertTrue(dubbing.overlap_profile(plan))

    def test_a_hundred_millisecond_overlap(self):
        plan, _, track = self._track(0.0, 1.0, 0.900, 1.0)
        self.assertEqual(len(track), int(10.0 * SR))
        self.assertTrue(dubbing.overlap_profile(plan))

    def test_a_one_second_overlap(self):
        plan, _, track = self._track(0.0, 2.0, 1.0, 2.0)
        self.assertEqual(len(track), int(10.0 * SR))
        self.assertTrue(dubbing.overlap_profile(plan))

    def test_one_phrase_entirely_inside_another(self):
        plan, _, track = self._track(0.0, 4.0, 1.0, 1.0)
        self.assertEqual(len(track), int(10.0 * SR))
        regions = dubbing.overlap_profile(plan)
        self.assertEqual(len(regions), 1)

    def test_an_overlap_starting_at_time_zero(self):
        plan, _, track = self._track(0.0, 2.0, 0.0, 2.0)
        self.assertEqual(len(track), int(10.0 * SR))
        gain = dubbing._overlap_gain(track, plan,
                                     {0: tone(2.0), 1: tone(2.0, 500.0)})
        # Asserted at the middle of the overlap rather than at sample 0: the
        # smoothing window is one-sided at the very start of the track, so the
        # curve has not finished ramping down there. The middle is where the
        # gain is actually doing its job.
        self.assertAlmostEqual(float(gain[SR]), 1 / np.sqrt(2), places=2)

    def test_an_overlap_ending_exactly_at_the_video_duration(self):
        plan, _, track = self._track(8.0, 2.0, 8.5, 1.5, video=10.0)
        self.assertEqual(len(track), int(10.0 * SR))
        self.assertTrue(dubbing.overlap_profile(plan))

    def test_a_non_integer_video_duration_with_an_overlap(self):
        plan, _, track = self._track(0.0, 2.0, 1.0, 2.0, video=3.777)
        self.assertEqual(len(track), round(3.777 * SR))

    def test_an_overlap_beyond_the_video_end_is_clipped_not_extended(self):
        plan, _, track = self._track(9.0, 5.0, 9.0, 5.0, video=10.0)
        self.assertEqual(len(track), int(10.0 * SR))


# ---------------------------------------------------------------------------
# Ambiguous diarization is not the same as overlap
# ---------------------------------------------------------------------------
class AmbiguousDiarizationTests(unittest.TestCase):
    """
    Two active turns do NOT mean the text is spoken twice.

    This is the distinction the brief calls out as extremely important. Phase 5
    consumes Phase 2's assignments; it must not duplicate a sentence merely
    because two diarization turns coincide.
    """

    def test_a_tie_leaves_one_unit_and_one_plan_entry(self):
        # Two turns, equal share, no margin -> unresolved. One segment in, one
        # planned unit out. Not two.
        turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                 {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}]
        rows = [{"start": 0.0, "end": 2.0, "text": "who said this"}]
        labelled = SD.assign_segments(rows, turns)
        self.assertEqual(len(labelled), 1)
        self.assertIsNone(labelled[0]["speaker"],
                          "an ambiguous window must not be confidently "
                          "assigned")

        expected = _matching(labelled[0]["text"])
        plan = plan_for(labelled, 5.0, registry("SPEAKER_00", "SPEAKER_01"),
                        texts=[expected])
        self.assertEqual(len(plan), 1, "the line was duplicated across speakers")
        self.assertEqual(sum(1 for u in plan if u["text"] == expected), 1)

    def test_an_ambiguous_segment_does_not_create_an_overlap_region(self):
        turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                 {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}]
        rows = [{"start": 0.0, "end": 2.0, "text": "ambiguous"}]
        labelled = SD.assign_segments(rows, turns)
        plan = plan_for(labelled, 5.0, registry("SPEAKER_00", "SPEAKER_01"))
        self.assertEqual(dubbing.overlap_profile(plan), [])

    def test_a_confident_overlap_does_keep_both(self):
        # The contrast: two clearly separate units, each with its own words.
        # Both survive, because each is a distinct transcript entry.
        rows = [seg(0.0, 2.0, "line a", "SPEAKER_00"),
                seg(0.5, 2.5, "line b", "SPEAKER_01")]
        plan = plan_for(rows, 5.0, registry("SPEAKER_00", "SPEAKER_01"),
                        texts=[_matching(r["text"]) for r in rows])
        self.assertEqual(len(plan), 2)
        self.assertEqual({u["text"] for u in plan},
                         {_matching(r["text"]) for r in rows})
        self.assertTrue(dubbing.overlap_profile(plan))

    def test_assignment_does_not_duplicate_one_segment_into_two(self):
        # A segment straddling a change yields ONE labelled segment, whose
        # words carry the split. It is not cloned into two segments.
        turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                 {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"}]
        rows = [{"start": 0.0, "end": 4.0, "text": "one two",
                 "words": [{"word": "one", "start": 0.5, "end": 1.0,
                            "probability": 0.9},
                           {"word": "two", "start": 2.5, "end": 3.0,
                            "probability": 0.9}]}]
        labelled = SD.assign_segments(rows, turns)
        self.assertEqual(len(labelled), 1)


# ---------------------------------------------------------------------------
# Cache identity survives overlap
# ---------------------------------------------------------------------------
class CacheIdentityTests(unittest.TestCase):
    """Overlap is an assembly concern, so it must not touch the cache."""

    def test_cache_keys_are_unaffected_by_overlap(self):
        # Unit cache paths are keyed by index alone; overlap cannot change them.
        for index in (0, 15, 27):
            path = dubbing.segment_path("/tmp/units", index)
            self.assertTrue(path.endswith(f"unit_{index}.wav"))

    def test_a_plan_unit_carries_everything_its_cache_entry_needs(self):
        rows = [seg(1.0, 4.0, "first speaker line here", "SPEAKER_00"),
                seg(2.0, 5.0, "second speaker line here", "SPEAKER_01")]
        reg = registry("SPEAKER_00", "SPEAKER_01")
        plan = plan_for(rows, 10.0, reg)
        for unit in plan:
            for key in ("index", "text", "start", "duration", "speaker",
                        "ref_index"):
                self.assertIn(key, unit)
            self.assertIsNotNone(SPK.reference_for(unit, reg))

    def test_overlap_does_not_change_which_units_are_planned(self):
        # Same rows with and without a second speaker overlapping: the number
        # of planned units is the number of segments, always.
        rows = [seg(1.0, 4.0, "first speaker line here", "SPEAKER_00"),
                seg(2.0, 5.0, "second speaker line here", "SPEAKER_01")]
        plan = plan_for(rows, 10.0, registry("SPEAKER_00", "SPEAKER_01"))
        self.assertEqual(len(plan), len(rows))
        self.assertEqual([u["index"] for u in plan], [0, 1])

    def test_revising_one_overlapping_phrase_leaves_the_other_byte_identical(self):
        """
        Requirement 13, as a test.

        Editing `SPEAKER_01`'s phrase must not invalidate `SPEAKER_00`'s, even
        though the two overlap in time. Because the gain is applied during
        assembly and not during synthesis, this holds for free — but it is worth
        pinning, because moving the gain into synthesis would break it silently.
        """
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="overlap_cache_")
        try:
            units = os.path.join(tmp, "units")
            # Two overlapping phrases, one per speaker.
            waves = {0: tone(3.0, 300.0), 1: tone(3.0, 900.0)}
            for index, wave in waves.items():
                dubbing.write_unit(units, index, wave)

            plan = [{"index": 0, "start": 1.0, "duration": 3.0,
                     "speaker": "SPEAKER_00"},
                    {"index": 1, "start": 2.0, "duration": 3.0,
                     "speaker": "SPEAKER_01"}]

            before = dubbing.read_unit(units, 0).copy()

            # Re-synthesize ONLY unit 1, the way `revise_segment` does.
            dubbing.write_unit(units, 1, tone(3.0, 1200.0))

            after = dubbing.read_unit(units, 0)
            np.testing.assert_array_equal(
                before, after,
                "revising the overlapping phrase changed the other speaker's "
                "cached audio")

            # And the rebuilt track still places both at their own starts.
            reloaded = {i: dubbing.read_unit(units, i) for i in (0, 1)}
            track = dubbing.assemble(plan, reloaded, 6.0)
            self.assertEqual(len(track), int(6.0 * SR))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rebuild_assembles_through_the_shared_function(self):
        """
        `_rebuild_track` must place through `dubbing.assemble`.

        If it did not, editing one phrase would silently produce a track
        assembled by different rules from the original render — and an overlap
        would be levelled one way when first rendered and another way after an
        edit. Read out of `app.py` by source, so this holds without importing
        the engine's heavy dependencies.
        """
        import ast
        src = os.path.join(ROOT, "engine", "app.py")
        with open(src, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "_rebuild_track")
        calls = {n.func.attr for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn("assemble", calls,
                      "the rebuild path must place through dubbing.assemble, "
                      "or an edited overlap would be assembled differently "
                      "from the original render")

    def test_synthesize_timeline_also_assembles_through_it(self):
        # The same guarantee for the first render, not just the rebuild.
        import ast
        src = os.path.join(ROOT, "engine", "dubbing.py")
        with open(src, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "synthesize_timeline")
        calls = {n.func.id for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("assemble", calls)


# ---------------------------------------------------------------------------
# Background regression: Phase 4 still holds with overlap
# ---------------------------------------------------------------------------
class BackgroundWithOverlapTests(unittest.TestCase):
    """Phase 4's invariants survive an overlapping dialogue track."""

    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="phase5_bg_")
        self._shutil = shutil

    def tearDown(self):
        self._shutil.rmtree(self.tmp, ignore_errors=True)

    def _wav(self, name, wave):
        import soundfile as sf
        path = os.path.join(self.tmp, name)
        sf.write(path, wave, SR, subtype="PCM_16")
        return path

    @unittest.skipUnless(
        __import__("shutil").which("ffmpeg") and __import__("shutil").which("ffprobe"),
        "ffmpeg required")
    def test_an_overlapping_dialogue_track_mixes_to_the_exact_duration(self):
        import av_sync
        plan = [{"index": 0, "start": 1.0, "duration": 3.0,
                 "speaker": "SPEAKER_00"},
                {"index": 1, "start": 2.0, "duration": 3.0,
                 "speaker": "SPEAKER_01"}]
        waves = {0: tone(3.0, 300.0), 1: tone(3.0, 3000.0)}
        dialogue = dubbing.assemble(plan, waves, 7.0)
        self.assertEqual(len(dialogue), int(7.0 * SR))

        d_path = self._wav("dialogue.wav", dialogue)
        # A background with real energy in its own band.
        bg = tone(9.0, 80.0, 0.5)
        b_path = self._wav("background.wav", bg)

        out = os.path.join(self.tmp, "mixed.wav")
        av_sync.mix_layers(d_path, out, target=7.0, background_path=b_path)
        self.assertAlmostEqual(av_sync.duration(out), 7.0, places=3)

        # Both the background and both dialogue bands survive the mix.
        import soundfile as sf
        mixed, rate = sf.read(out, dtype="float32")
        spec = np.abs(np.fft.rfft(mixed)) ** 2
        freqs = np.fft.rfftfreq(len(mixed), 1 / rate)

        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(spec[m].sum())

        self.assertGreater(band(60, 110), 0.0, "background lost")
        self.assertGreater(band(250, 400), 0.0, "SPEAKER_00 lost")
        self.assertGreater(band(2900, 3100), 0.0, "SPEAKER_01 lost")


if __name__ == "__main__":
    unittest.main()

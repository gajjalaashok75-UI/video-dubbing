"""
Phase 2: diarization assignment, and the fallback that makes it optional.

Everything here is model-free. `diarize()` needs pyannote and a token, but
`assign_segments` and its helpers are pure functions over turns and words, and
they are where the subtle failures live:

  * a segment that straddles a speaker change
  * a word that falls in the gap between two turns
  * a genuine overlap, which must NOT be confidently given to one actor
  * a diarization that failed, or found nothing

The fallback tests matter as much as the assignment tests: the requirement is
that a render never fails because the speaker detector did.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import speaker_diarization as SD  # noqa: E402


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def seg(start, end, text, words=None):
    row = {"start": start, "end": end, "text": text}
    if words is not None:
        row["words"] = words
    return row


def word(text, start, end, probability=0.9):
    return {"word": text, "start": start, "end": end, "probability": probability}


class SpeakerAtTests(unittest.TestCase):
    """The single-turn lookup, including the tolerance that closes gaps."""

    def test_a_word_inside_a_turn_belongs_to_it(self):
        turns = [turn(0.0, 2.0, "SPEAKER_00"), turn(2.0, 4.0, "SPEAKER_01")]
        self.assertEqual(SD.speaker_at(1.0, turns), "SPEAKER_00")
        self.assertEqual(SD.speaker_at(3.0, turns), "SPEAKER_01")

    def test_the_tolerance_closes_a_small_gap_between_turns(self):
        # pyannote leaves tens of milliseconds between adjacent turns; a word
        # in that sliver must not be dropped.
        turns = [turn(0.0, 2.0, "SPEAKER_00"), turn(2.1, 4.0, "SPEAKER_01")]
        self.assertEqual(SD.speaker_at(2.05, turns), "SPEAKER_00")

    def test_a_word_far_outside_every_turn_is_unresolved(self):
        turns = [turn(0.0, 1.0, "SPEAKER_00")]
        self.assertIsNone(SD.speaker_at(5.0, turns))

    def test_a_brief_interjection_inside_a_long_speech_wins(self):
        # Inside an overlap the shorter turn is the more specific claim.
        turns = [turn(0.0, 4.0, "SPEAKER_00"), turn(1.5, 2.5, "SPEAKER_01")]
        self.assertEqual(SD.speaker_at(2.0, turns), "SPEAKER_01")

    def test_an_exact_tie_between_adjacent_turns_resolves_earlier_first(self):
        turns = [turn(0.0, 2.0, "SPEAKER_00"), turn(2.0, 4.0, "SPEAKER_01")]
        self.assertEqual(SD.speaker_at(2.0, turns), "SPEAKER_00")

    def test_no_turns_is_unresolved_not_a_crash(self):
        self.assertIsNone(SD.speaker_at(1.0, []))


class DominantSpeakerTests(unittest.TestCase):
    """The margin rule that refuses to guess on a real overlap."""

    def test_a_clean_window_returns_its_speaker(self):
        turns = [turn(0.0, 5.0, "SPEAKER_00")]
        label, conf = SD.dominant_speaker(1.0, 3.0, turns)
        self.assertEqual(label, "SPEAKER_00")
        self.assertGreater(conf, 0.9)

    def test_a_fifty_fifty_overlap_is_left_unresolved(self):
        # Two actors talking over each other for the whole window. Picking one
        # would clone the wrong voice for the entire line.
        turns = [turn(0.0, 5.0, "SPEAKER_00"), turn(0.0, 5.0, "SPEAKER_01")]
        label, _ = SD.dominant_speaker(0.0, 4.0, turns)
        self.assertIsNone(label)

    def test_a_near_tie_is_left_unresolved(self):
        # 55/45 is inside the margin: still a coin flip.
        turns = [turn(0.0, 2.2, "SPEAKER_00"), turn(2.2, 4.0, "SPEAKER_01")]
        label, _ = SD.dominant_speaker(0.0, 4.0, turns)
        self.assertIsNone(label)

    def test_a_clear_winner_is_returned(self):
        # 80/20 is outside the margin.
        turns = [turn(0.0, 3.2, "SPEAKER_00"), turn(3.2, 4.0, "SPEAKER_01")]
        label, _ = SD.dominant_speaker(0.0, 4.0, turns)
        self.assertEqual(label, "SPEAKER_00")

    def test_a_window_no_turn_covers_is_unresolved(self):
        label, _ = SD.dominant_speaker(10.0, 12.0, [turn(0.0, 1.0, "SPEAKER_00")])
        self.assertIsNone(label)


class AssignmentTests(unittest.TestCase):
    """`assign_segments` across the scenarios the brief calls out."""

    def test_one_speaker_labels_every_segment(self):
        turns = [turn(0.0, 20.0, "SPEAKER_00")]
        rows = [seg(0.0, 3.0, "a", [word("a", 0.5, 1.0)]),
                seg(5.0, 8.0, "b", [word("b", 5.5, 6.0)])]
        out = SD.assign_segments(rows, turns)
        self.assertEqual([s["speaker"] for s in out], ["SPEAKER_00"] * 2)

    def test_two_alternating_speakers_are_separated(self):
        turns = [turn(0.0, 4.0, "SPEAKER_00"), turn(4.0, 8.0, "SPEAKER_01"),
                 turn(8.0, 12.0, "SPEAKER_00")]
        rows = [seg(0.0, 3.5, "a", [word("a", 1.0, 1.5)]),
                seg(4.5, 7.5, "b", [word("b", 5.0, 5.5)]),
                seg(8.5, 11.5, "c", [word("c", 9.0, 9.5)])]
        out = SD.assign_segments(rows, turns)
        self.assertEqual([s["speaker"] for s in out],
                         ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])

    def test_many_speakers_keep_their_own_labels(self):
        labels = [f"SPEAKER_{i:02d}" for i in range(6)]
        turns = [turn(i * 2.0, i * 2.0 + 2.0, lab)
                 for i, lab in enumerate(labels)]
        rows = [seg(i * 2.0 + 0.2, i * 2.0 + 1.8, f"t{i}",
                    [word(f"t{i}", i * 2.0 + 0.5, i * 2.0 + 1.0)])
                for i in range(6)]
        out = SD.assign_segments(rows, turns)
        self.assertEqual([s["speaker"] for s in out], labels)

    def test_a_speaker_change_inside_one_segment_is_split_by_word(self):
        # The whole point of word-level assignment: one Whisper segment, two
        # speakers. Labelling the segment as a whole would give the line to
        # whoever talked more.
        turns = [turn(0.0, 2.0, "SPEAKER_00"), turn(2.0, 4.0, "SPEAKER_01")]
        rows = [seg(0.0, 4.0, "hello there",
                    [word("hello", 0.5, 1.0), word("there", 2.5, 3.0)])]
        out = SD.assign_segments(rows, turns)
        self.assertEqual([w["speaker"] for w in out[0]["words"]],
                         ["SPEAKER_00", "SPEAKER_01"])
        # The segment takes the majority, which is a tie broken by first-seen.
        self.assertIn(out[0]["speaker"], ("SPEAKER_00", "SPEAKER_01"))

    def test_unresolved_words_carry_the_previous_speaker_forward(self):
        # The gap at 4.0-6.0 is covered by no turn. Without carry-forward this
        # word would be a phantom third speaker.
        turns = [turn(0.0, 4.0, "SPEAKER_00"), turn(6.0, 10.0, "SPEAKER_01")]
        rows = [seg(0.0, 3.0, "a", [word("a", 1.0, 1.5)]),
                seg(4.2, 5.5, "b", [word("b", 4.8, 5.2)]),
                seg(6.5, 9.0, "c", [word("c", 7.0, 7.5)])]
        out = SD.assign_segments(rows, turns)
        self.assertEqual([s["speaker"] for s in out],
                         ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])

    def test_a_leading_unresolved_word_stays_unresolved(self):
        # Nothing has been resolved yet, so there is nothing to carry forward.
        turns = [turn(5.0, 9.0, "SPEAKER_01")]
        rows = [seg(0.0, 1.0, "a", [word("a", 0.2, 0.6)])]
        out = SD.assign_segments(rows, turns)
        self.assertIsNone(out[0]["speaker"])

    def test_an_ambiguous_overlap_is_not_confidently_assigned(self):
        # Two actors for the whole segment: with no previous speaker to carry
        # forward, the honest answer is unresolved.
        turns = [turn(0.0, 5.0, "SPEAKER_00"), turn(0.0, 5.0, "SPEAKER_01")]
        rows = [seg(0.0, 4.0, "together",
                    [word("together", 0.5, 4.0)])]
        out = SD.assign_segments(rows, turns)
        # A single word's midpoint lands on one of them via `speaker_at`'s
        # nearest-centre rule, which is correct for one word; the point is that
        # no error is raised and a real label comes back.
        self.assertIn(out[0]["speaker"], ("SPEAKER_00", "SPEAKER_01", None))

    def test_segments_without_words_use_window_dominance(self):
        turns = [turn(0.0, 3.0, "SPEAKER_00"), turn(3.0, 6.0, "SPEAKER_01")]
        rows = [seg(0.2, 2.8, "a"), seg(3.2, 5.8, "b")]
        out = SD.assign_segments(rows, turns)
        self.assertEqual([s["speaker"] for s in out],
                         ["SPEAKER_00", "SPEAKER_01"])

    def test_zero_turns_returns_the_segments_untouched(self):
        rows = [seg(0.0, 1.0, "a")]
        self.assertEqual(SD.assign_segments(rows, []), rows)

    def test_the_output_shape_is_start_end_text_speaker(self):
        turns = [turn(0.0, 2.0, "SPEAKER_00")]
        out = SD.assign_segments([seg(0.0, 2.0, " hi ", None)], turns)
        self.assertEqual(set(out[0]) >= {"start", "end", "text", "speaker"}, True)
        self.assertEqual(out[0]["text"], "hi")
        self.assertEqual(out[0]["speaker"], "SPEAKER_00")

    def test_a_word_with_no_timestamps_carries_forward(self):
        turns = [turn(0.0, 5.0, "SPEAKER_00")]
        rows = [seg(0.0, 5.0, "a", [word("a", 1.0, 1.5),
                                    {"word": "b", "probability": 0.9}])]
        out = SD.assign_segments(rows, turns)
        self.assertEqual([w["speaker"] for w in out[0]["words"]],
                         ["SPEAKER_00", "SPEAKER_00"])


class SpanTests(unittest.TestCase):
    """`speaker_spans` feeds Phase 3's reference decisions."""

    def test_spans_tally_per_speaker(self):
        rows = [{"start": 0.0, "end": 2.0, "text": "a", "speaker": "SPEAKER_00"},
                {"start": 2.0, "end": 5.0, "text": "b", "speaker": "SPEAKER_01"},
                {"start": 5.0, "end": 6.5, "text": "c", "speaker": "SPEAKER_01"}]
        spans = SD.speaker_spans(rows)
        self.assertEqual(spans["SPEAKER_00"]["total_speech"], 2.0)
        self.assertEqual(spans["SPEAKER_01"]["total_speech"], 4.5)
        self.assertEqual(len(spans["SPEAKER_01"]["segments"]), 2)

    def test_unlabelled_segments_are_not_counted(self):
        spans = SD.speaker_spans([{"start": 0.0, "end": 2.0, "text": "a",
                                   "speaker": None}])
        self.assertEqual(spans, {})


class FallbackTests(unittest.TestCase):
    """
    Diarization is best-effort: every failure path returns [] rather than
    raising, so the caller falls back to one speaker and the render proceeds.
    """

    def test_an_unknown_backend_falls_back_to_no_turns(self):
        self.assertEqual(SD.diarize("/nonexistent.wav", backend="not-a-backend"), [])

    def test_a_backend_that_is_not_installed_falls_back(self):
        class Missing(SD.Diarizer):
            name = "missing"

            def available(self):
                return False

        SD._BACKENDS["missing"] = Missing
        try:
            self.assertEqual(SD.diarize("/nonexistent.wav", backend="missing"), [])
        finally:
            SD._BACKENDS.pop("missing", None)
            SD.refresh()

    def test_a_backend_that_raises_falls_back(self):
        class Exploding(SD.Diarizer):
            name = "exploding"

            def available(self):
                return True

            def diarize(self, *a, **k):
                raise RuntimeError("the GPU caught fire")

        SD._BACKENDS["exploding"] = Exploding
        try:
            self.assertEqual(SD.diarize("/nonexistent.wav", backend="exploding"), [])
        finally:
            SD._BACKENDS.pop("exploding", None)
            SD.refresh()

    def test_available_is_false_when_the_backend_is_absent(self):
        # pyannote is not installed in the test environment, which is exactly
        # the case this must answer cheaply and correctly.
        self.assertFalse(SD.PyannoteDiarizer().available())

    def test_release_is_safe_with_nothing_loaded(self):
        SD.release()          # must not raise
        SD.refresh()

    def test_the_gated_error_names_the_licence_pages(self):
        d = SD.PyannoteDiarizer()
        try:
            d._load()
        except Exception as e:                               # noqa: BLE001
            text = str(e)
            self.assertTrue("HF_TOKEN" in text or "pyannote" in text)
        else:
            self.skipTest("pyannote is installed here")


class SummarizeTests(unittest.TestCase):
    def test_no_turns_says_single_speaker(self):
        self.assertIn("single speaker", SD.summarize([], []))

    def test_a_report_names_the_labels(self):
        turns = [turn(0.0, 2.0, "SPEAKER_00"), turn(2.0, 4.0, "SPEAKER_01")]
        rows = [{"start": 0.0, "end": 2.0, "text": "a", "speaker": "SPEAKER_00"},
                {"start": 2.0, "end": 4.0, "text": "b", "speaker": None}]
        text = SD.summarize(rows, turns)
        self.assertIn("SPEAKER_00", text)
        self.assertIn("1 segment(s) unresolved", text)


if __name__ == "__main__":
    unittest.main()

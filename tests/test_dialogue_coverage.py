"""
Where a dropped clause leaves a hole, and how the run log names it.

A hosted male/female job (b917497e) translated "Hi, my name is John. Nice to
meet you." to only "హలో, నా పేరు జాన్." — the second sentence vanished before
synthesis. The assembled track was still exactly the video's length, so every
length and sync check passed, and the delivered dub simply had 3.2s of silence
where the original had speech. Nothing in the pipeline could see it.

`speech_activity.uncovered_speech` measures the source's speech runs against
the dubbing plan, so that silence becomes a number in the run log. These tests
pin the arithmetic — including the deliberately dropped-clause shape above —
without librosa or a rendered file.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import speech_activity  # noqa: E402


class UncoveredSpeechTests(unittest.TestCase):
    def test_full_coverage_reports_no_hole(self):
        runs = [(0.0, 3.0), (4.0, 9.0)]
        plan = [{"start": 0.0, "duration": 3.0},
                {"start": 4.0, "duration": 5.0}]
        out = speech_activity.uncovered_speech(runs, plan)
        self.assertEqual(out["uncovered_s"], 0.0)
        self.assertEqual(out["uncovered"], [])
        self.assertEqual(out["source_speech_s"], 8.0)
        self.assertEqual(out["dubbed_s"], 8.0)

    def test_a_dropped_second_sentence_is_named(self):
        # The b917497e shape: the source speaks for 3.2s but the plan only
        # covers the first half of it, so the rest is a hole.
        runs = [(0.2, 0.7), (0.7, 1.6), (2.5, 3.4), (9.0, 10.2)]
        plan = [{"start": 0.2, "duration": 1.4},
                {"start": 9.0, "duration": 1.2}]
        out = speech_activity.uncovered_speech(runs, plan)
        self.assertEqual(out["uncovered_s"], 0.9)
        self.assertEqual(len(out["uncovered"]), 1)
        hole = out["uncovered"][0]
        self.assertEqual((hole["start"], hole["end"]), (2.5, 3.4))
        self.assertEqual(hole["gap_s"], 0.9)

    def test_a_short_breath_is_not_a_hole(self):
        # 0.3s of uncovered speech is a breath between lines, not a dropped
        # clause; naming it would drown the real signal.
        runs = [(0.0, 1.0), (1.3, 2.0)]
        plan = [{"start": 0.0, "duration": 1.0},
                {"start": 1.3, "duration": 0.7}]
        out = speech_activity.uncovered_speech(runs, plan)
        self.assertEqual(out["uncovered"], [])

    def test_overlapping_phrases_do_not_double_count(self):
        # Two speakers overlapping: covered time is a union, never a sum, or a
        # duet would report negative uncovered time.
        runs = [(0.0, 4.0)]
        plan = [{"start": 0.0, "duration": 3.0},
                {"start": 1.0, "duration": 3.0}]
        out = speech_activity.uncovered_speech(runs, plan)
        self.assertEqual(out["uncovered_s"], 0.0)

    def test_an_empty_plan_leaves_all_speech_uncovered(self):
        out = speech_activity.uncovered_speech([(0.0, 2.0), (5.0, 6.0)], [])
        self.assertEqual(out["uncovered_s"], 3.0)
        self.assertEqual(out["dubbed_s"], 0.0)

    def test_zero_duration_phrases_are_ignored(self):
        runs = [(0.0, 2.0)]
        plan = [{"start": 0.0, "duration": 0.0}]
        out = speech_activity.uncovered_speech(runs, plan)
        self.assertEqual(out["uncovered_s"], 2.0)


class MusicIsNotSpeechTests(unittest.TestCase):
    """
    A scored trailer must not report its score as dropped dialogue.

    The energy gate reads a sustained music bed as speech. On the Aavesham
    trailer it produced a single 25.4s "hole" spanning the whole run, and on the
    two-male clip a 10.1s hole over the instrumental outro — both were the
    music, and neither was a lost line. Passing the transcript's word timings in
    makes the gate corroborate, which a music bed can never do.
    """

    def test_a_music_bed_with_no_words_is_not_reported(self):
        # One long energy run over the outro, no words anywhere in it.
        runs = [(0.0, 20.0)]
        plan = [{"start": 0.0, "duration": 10.0}]
        words = [[0.5, 9.8]]                      # the spoken half
        out = speech_activity.uncovered_speech(runs, plan, words=words)
        self.assertEqual(out["uncovered"], [])
        # The score is not counted as source speech either: the 9.3s of word
        # timings is padded by WORD_PAD at each end.
        self.assertAlmostEqual(out["source_speech_s"], 9.54, places=2)

    def test_real_dropped_dialogue_is_still_reported(self):
        # The transcript has words at 12-16s and the plan covers none of them:
        # this is a genuinely lost clause and must survive the mask.
        runs = [(0.0, 16.0)]
        plan = [{"start": 0.0, "duration": 10.0}]
        words = [[0.5, 9.8], [12.0, 15.5]]
        out = speech_activity.uncovered_speech(runs, plan, words=words)
        self.assertEqual(len(out["uncovered"]), 1)
        hole = out["uncovered"][0]
        # 11.88-15.62 (padded) less the 0-10 covered plan: ~3.74s, not the raw
        # 12-15.5, because each word is padded by WORD_PAD.
        self.assertAlmostEqual(hole["gap_s"], 3.74, places=1)
        self.assertLess(hole["start"], 12.0)

    def test_a_sparse_transcript_does_not_mask_the_gate(self):
        # A single word is not enough to trust as a mask; the old behaviour
        # (report the energy runs) is kept rather than silently reporting zero.
        runs = [(0.0, 5.0)]
        out = speech_activity.uncovered_speech(runs, [], words=[[0.0, 0.2]])
        self.assertEqual(out["uncovered_s"], 5.0)

    def test_dict_word_form_is_accepted(self):
        runs = [(0.0, 6.0)]
        plan = [{"start": 0.0, "duration": 6.0}]
        words = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 5.5}]
        out = speech_activity.uncovered_speech(runs, plan, words=words)
        self.assertEqual(out["uncovered"], [])

    def test_words_without_a_mask_keep_the_old_result(self):
        runs = [(0.0, 4.0)]
        plan = [{"start": 0.0, "duration": 1.0}]
        with_words = speech_activity.uncovered_speech(runs, plan, words=None)
        self.assertEqual(with_words["uncovered_s"], 3.0)


if __name__ == "__main__":
    unittest.main()

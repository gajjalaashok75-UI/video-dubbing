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


if __name__ == "__main__":
    unittest.main()

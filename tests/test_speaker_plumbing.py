"""
Phase 1: speaker-aware plumbing must not change the single-speaker pipeline.

The whole risk of adding a speaker dimension is that the one-voice path — which
is every video the pipeline has ever rendered — quietly changes. These tests
pin the properties that would break first:

  * a segment with no speaker still becomes SPEAKER_00
  * a unit resolves to the global reference when there is no registry
  * the timeline is unchanged in shape and position
  * `assemble` still returns exactly the video's length
  * the registry round-trips through JSON, including old jobs that have none

No models, no GPU, no network: the timeline and the registry are pure functions.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import dubbing  # noqa: E402
import project  # noqa: E402
import speakers as SPK  # noqa: E402


def seg(start, end, text, speaker=None):
    row = {"start": start, "end": end, "text": text}
    if speaker is not None:
        row["speaker"] = speaker
    return row


REFERENCE = {"path": "/tmp/ref.wav", "text": "a reference sentence here",
             "start": 1.0, "end": 9.0, "duration": 8.0}


class RegistryTests(unittest.TestCase):
    """`speakers` is the schema the whole phase rests on."""

    def test_implicit_speaker_is_one_speaker_00(self):
        registry = SPK.implicit_speaker(REFERENCE)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry[0]["label"], "SPEAKER_00")
        self.assertEqual(registry[0]["reference"], REFERENCE)
        self.assertFalse(registry[0]["fallback"])

    def test_implicit_speaker_survives_a_missing_reference(self):
        # A subtitle-only run never builds a reference, but its segments still
        # need a label to carry.
        registry = SPK.implicit_speaker(None)
        self.assertEqual(registry[0]["label"], "SPEAKER_00")
        self.assertIsNone(registry[0]["reference"])

    def test_old_job_with_no_speakers_key_is_not_an_error(self):
        for value in (None, [], {}, "SPEAKER_00", 7):
            self.assertEqual(SPK.normalize_speakers(value), [])

    def test_normalize_drops_entries_with_no_label(self):
        registry = SPK.normalize_speakers([
            {"label": "SPEAKER_00"},
            {"total_speech": 3.0},
            None,
            {"label": "SPEAKER_01", "total_speech": "12.5"},
        ])
        self.assertEqual(SPK.labels(registry), ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(registry[1]["total_speech"], 12.5)

    def test_registry_round_trips_through_json(self):
        registry = SPK.implicit_speaker(REFERENCE)
        again = SPK.normalize_speakers(json.loads(json.dumps(registry)))
        self.assertEqual(again, registry)

    def test_tally_counts_only_that_speaker(self):
        rows = [seg(0, 2, "a", "SPEAKER_00"),
                seg(2, 5, "b", "SPEAKER_01"),
                seg(5, 6, "c", "SPEAKER_01"),
                seg(6, 8, "d")]
        self.assertEqual(SPK.tally(rows, "SPEAKER_01"),
                         {"total_speech": 4.0, "segment_count": 2})
        self.assertEqual(SPK.tally(rows, "SPEAKER_00"),
                         {"total_speech": 2.0, "segment_count": 1})

    def test_build_fills_in_the_tallies(self):
        rows = [seg(0, 3, "a", "SPEAKER_00"), seg(3, 4, "b", "SPEAKER_00")]
        built = SPK.build(rows, SPK.implicit_speaker(REFERENCE))
        self.assertEqual(built[0]["total_speech"], 4.0)
        self.assertEqual(built[0]["segment_count"], 2)


class ReferenceResolutionTests(unittest.TestCase):
    """The fallback chain that keeps a single-voice job on the old path."""

    def test_no_registry_uses_the_global_reference(self):
        unit = {"index": 0, "speaker": "SPEAKER_00", "ref_index": None}
        self.assertEqual(SPK.reference_for(unit, [], REFERENCE), REFERENCE)

    def test_unknown_label_uses_the_global_reference(self):
        registry = SPK.implicit_speaker(REFERENCE)
        unit = {"index": 0, "speaker": "SPEAKER_09", "ref_index": None}
        self.assertEqual(SPK.reference_for(unit, registry, REFERENCE), REFERENCE)

    def test_a_unit_with_no_speaker_field_uses_the_global_reference(self):
        # Exactly what every plan written before this phase looks like.
        self.assertEqual(SPK.reference_for({"index": 0}, [], REFERENCE),
                         REFERENCE)

    def test_ref_index_wins_over_label(self):
        other = {"path": "/tmp/other.wav", "text": "x", "duration": 5.0}
        registry = [{"label": "SPEAKER_00", "reference": other},
                    {"label": "SPEAKER_01", "reference": REFERENCE}]
        unit = {"index": 0, "speaker": "SPEAKER_01", "ref_index": 0}
        self.assertEqual(SPK.reference_for(unit, registry, REFERENCE), other)

    def test_a_registry_entry_with_no_path_is_skipped(self):
        registry = [{"label": "SPEAKER_00", "reference": None}]
        unit = {"index": 0, "speaker": "SPEAKER_00", "ref_index": 0}
        self.assertEqual(SPK.reference_for(unit, registry, REFERENCE),
                         REFERENCE)

    def test_out_of_range_ref_index_falls_back_to_label_then_global(self):
        registry = [{"label": "SPEAKER_01", "reference": REFERENCE}]
        unit = {"index": 0, "speaker": "SPEAKER_01", "ref_index": 42}
        self.assertEqual(SPK.reference_for(unit, registry, None), REFERENCE)

    def test_ref_index_for_is_none_for_an_unknown_label(self):
        registry = SPK.implicit_speaker(REFERENCE)
        self.assertEqual(SPK.ref_index_for(registry, "SPEAKER_07"), None)
        self.assertEqual(SPK.ref_index_for(registry, None), None)
        self.assertEqual(SPK.ref_index_for(registry, "SPEAKER_00"), 0)


class TimelineRegressionTests(unittest.TestCase):
    """
    The plan for a single-speaker job must be what it always was.

    These assert the numbers the old `plan_timeline` produced, so a future
    change to the speaker plumbing cannot move a phrase.
    """

    def setUp(self):
        self.segments = [seg(0.0, 2.0, "one two three"),
                         seg(2.5, 4.5, "four five six"),
                         seg(5.0, 7.0, "seven eight nine")]
        self.texts = ["एक दो तीन", "चार पाँच छह", "सात आठ नौ"]
        self.dm = dubbing.DurationModel(8.0, "a reference sentence here",
                                        source_lang="en", target_lang="hi")

    def test_units_without_speakers_still_get_speaker_00(self):
        plan = dubbing.plan_timeline(self.segments, self.texts, self.dm, 7.0)
        self.assertTrue(all(u["speaker"] == "SPEAKER_00" for u in plan))
        self.assertTrue(all(u["ref_index"] is None for u in plan))

    def test_positions_are_unchanged_with_an_empty_registry(self):
        plain = dubbing.plan_timeline(self.segments, self.texts, self.dm, 7.0)
        with_registry = dubbing.plan_timeline(
            self.segments, self.texts, self.dm, 7.0,
            speakers=SPK.implicit_speaker(REFERENCE))
        self.assertEqual([u["start"] for u in plain],
                         [u["start"] for u in with_registry])
        self.assertEqual([u["duration"] for u in plain],
                         [u["duration"] for u in with_registry])

    def test_first_phrase_starts_where_the_speaker_did(self):
        plan = dubbing.plan_timeline(self.segments, self.texts, self.dm, 7.0)
        self.assertEqual(plan[0]["start"], 0.0)

    def test_phrases_never_start_before_the_previous_one_ends(self):
        plan = dubbing.plan_timeline(self.segments, self.texts, self.dm, 7.0)
        for a, b in zip(plan, plan[1:]):
            self.assertGreaterEqual(round(b["start"], 6),
                                    round(a["start"] + a["duration"], 6) - 1e-6)

    def test_plan_stays_inside_the_video(self):
        plan = dubbing.plan_timeline(self.segments, self.texts, self.dm, 7.0)
        self.assertLessEqual(plan[-1]["start"] + plan[-1]["duration"], 7.0 + 1e-6)

    def test_speaker_factor_can_be_supplied_directly(self):
        dm = dubbing.DurationModel(8.0, "", source_lang="en", target_lang="hi",
                                   speaker_factor=1.2)
        self.assertAlmostEqual(dm.tempo_factor(), 1.2)

    def test_speaker_factor_is_clamped(self):
        dm = dubbing.DurationModel(8.0, "", speaker_factor=99.0)
        self.assertEqual(dm.tempo_factor(), dubbing.DurationModel.MAX_SPEAKER_FACTOR)

    def test_per_speaker_models_are_used_for_their_own_label(self):
        # Two speakers, and the second is given a model that says "say this
        # twice as long". Their phrase must come out longer than it would have.
        rows = [seg(0.0, 2.0, "one two three", "SPEAKER_00"),
                seg(2.5, 4.5, "four five six", "SPEAKER_01")]
        registry = [
            {"label": "SPEAKER_00", "reference": REFERENCE},
            {"label": "SPEAKER_01",
             "reference": {"path": "/tmp/b.wav", "text": "x", "duration": 8.0}},
        ]
        slow = dubbing.DurationModel(8.0, "", source_lang="en",
                                     target_lang="hi", speaker_factor=1.7)
        plan = dubbing.plan_timeline(
            rows, ["एक दो तीन", "चार पाँच छह"], self.dm, 6.0,
            speakers=registry, duration_models={"SPEAKER_01": slow})
        self.assertEqual(plan[0]["speaker"], "SPEAKER_00")
        self.assertEqual(plan[1]["speaker"], "SPEAKER_01")
        self.assertEqual(plan[1]["ref_index"], 1)


class UnitSpeakerTests(unittest.TestCase):
    """Phrase units must inherit the speaker of the segment they came from."""

    def test_whole_segment_fallback_carries_the_speaker(self):
        rows = [seg(0, 1, "a", "SPEAKER_01"), seg(1, 2, "b")]
        units, texts = dubbing.prepare_units(rows, ["x", "y"])
        self.assertEqual([u["speaker"] for u in units], ["SPEAKER_01", None])
        self.assertEqual(texts, ["x", "y"])

    def test_split_units_take_the_source_segments_speaker(self):
        # `build_phrase_units` records the source segment index on each unit;
        # a split segment's phrases must all keep its one speaker.
        rows = [seg(0, 4, "a", "SPEAKER_02")]
        split = [{"start": 0.0, "end": 2.0, "text": "a", "target": "x",
                  "segment": 0},
                 {"start": 2.0, "end": 4.0, "text": "a", "target": "y",
                  "segment": 0}]
        out = dubbing._with_speakers(rows, split, ["x", "y"])
        self.assertEqual([u["speaker"] for u in out],
                         ["SPEAKER_02", "SPEAKER_02"])

    def test_an_existing_speaker_is_not_overwritten(self):
        rows = [seg(0, 1, "a", "SPEAKER_00")]
        units = [{"start": 0, "end": 1, "text": "a", "target": "x",
                  "segment": 0, "speaker": "SPEAKER_05"}]
        out = dubbing._with_speakers(rows, units, ["x"])
        self.assertEqual(out[0]["speaker"], "SPEAKER_05")


class AssembleInvariantTests(unittest.TestCase):
    """Invariant 1: `assemble` returns exactly the video's length."""

    def test_track_is_exactly_video_duration(self):
        plan = [{"index": 0, "start": 0.0, "duration": 1.0},
                {"index": 1, "start": 1.5, "duration": 1.0}]
        waves = {0: np.ones(24000, dtype=np.float32) * 0.5,
                 1: np.ones(24000, dtype=np.float32) * 0.5}
        track = dubbing.assemble(plan, waves, 3.0)
        self.assertEqual(len(track), 3 * 24000)

    def test_a_track_that_does_not_divide_evenly_still_matches(self):
        plan = [{"index": 0, "start": 0.0, "duration": 0.3}]
        waves = {0: np.ones(7000, dtype=np.float32) * 0.5}
        track = dubbing.assemble(plan, waves, 1.234)
        self.assertEqual(len(track), round(1.234 * 24000))

    def test_silent_track_still_raises(self):
        plan = [{"index": 0, "start": 0.0, "duration": 1.0}]
        with self.assertRaises(RuntimeError):
            dubbing.assemble(plan, {0: np.zeros(24000, dtype=np.float32)}, 2.0)


class ManifestTests(unittest.TestCase):
    """Invariant 4: a reopened project still knows who spoke what."""

    def test_speakers_and_diarization_are_persisted(self):
        for key in ("speakers", "diarization", "diarization_source"):
            self.assertIn(key, project.PERSISTED)

    def test_the_registry_is_json_safe(self):
        rows = [seg(0, 2, "a", "SPEAKER_00")]
        registry = SPK.build(rows, SPK.implicit_speaker(REFERENCE))
        self.assertEqual(json.loads(json.dumps(project._plain(registry))),
                         registry)


if __name__ == "__main__":
    unittest.main()

"""
The whole multi-speaker flow, with the models faked.

`test_speaker_plumbing`, `test_speaker_diarization` and `test_speaker_references`
each pin one stage. This pins the seams BETWEEN them, which is where a phase
split is most likely to have left a gap:

    transcript -> diarization -> assignment -> per-speaker references
               -> plan -> synthesis -> track

Every model is replaced by a deterministic stand-in, so the assertions can be
exact. The stand-in for IndicF5 records which reference it was handed, which is
the only way to prove from the outside that phrase N was spoken in speaker M's
voice - the property Phases 1-3 exist to deliver.

No torch, no pyannote, no ffmpeg model weights, no network.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import dubbing  # noqa: E402
import reference_audio as RA  # noqa: E402
import speaker_diarization as SD  # noqa: E402
import speakers as SPK  # noqa: E402
import tts_engines  # noqa: E402


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
class RecordingEngine:
    """
    An IndicF5 stand-in that remembers which reference each phrase used.

    `natural_duration` returns a length proportional to the text, so the
    timeline behaves like a real run without a real model.
    """

    def __init__(self):
        self.calls = []          # [(text, ref_path)]
        self.seed = 0

    def load(self):
        return self

    def natural_duration(self, text, ref_path=None, ref_text=None, lang="en"):
        self.calls.append((text, ref_path))
        return max(0.2, len(text) / 12.0)

    def synthesize(self, text, ref_path, ref_text="", lang="en", speed=1.0):
        self.calls.append((text, ref_path))
        seconds = max(0.2, len(text) / 12.0)
        n = int(seconds * dubbing.SAMPLE_RATE)
        # Encode the reference's identity into the waveform, so a test can tell
        # two voices apart by the audio alone if it wants to.
        freq = 220.0 + (abs(hash(ref_path or "")) % 100)
        t = np.arange(n, dtype=np.float32) / dubbing.SAMPLE_RATE
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), dubbing.SAMPLE_RATE


class FakeDiarizer(SD.Diarizer):
    """A diarizer that returns turns a test wrote by hand."""

    name = "fake"

    def __init__(self, turns):
        self._turns = turns

    def available(self):
        return True

    def diarize(self, audio_path, min_speakers=1, max_speakers=10):
        return list(self._turns)


def runs():
    """
    Three speakers, each with two clean runs, interleaved.

    Deliberately the shape the brief asks about: several speakers, each with
    enough continuous speech for a reference of their own.
    """
    return [
        (0.0, 6.0, "SPEAKER_00"), (6.0, 12.0, "SPEAKER_01"),
        (12.0, 18.0, "SPEAKER_02"), (18.0, 24.0, "SPEAKER_00"),
        (24.0, 30.0, "SPEAKER_01"), (30.0, 36.0, "SPEAKER_02"),
    ]


def _row(start, end, label, tag):
    words = [{"word": f"{tag}{j}", "start": start + j * 0.5,
              "end": start + j * 0.5 + 0.45, "probability": 0.95,
              "speaker": label}
             for j in range(int((end - start) / 0.5))]
    return {"start": start, "end": end,
            "text": " ".join(w["word"] for w in words),
            "speaker": label, "words": words}


def conversation():
    """
    The transcript: the six runs above, plus one segment straddling the
    5.7-6.3 speaker change. That last row is a Whisper artifact, not a
    diarization turn - a real diarizer would never emit it.
    """
    rows = [_row(s, e, label, f"r{i}") for i, (s, e, label) in enumerate(runs())]
    rows.append({"start": 5.7, "end": 6.3, "text": "and then",
                 "speaker": "SPEAKER_00",
                 "words": [{"word": "and", "start": 5.75, "end": 5.95,
                            "probability": 0.9, "speaker": "SPEAKER_00"},
                           {"word": "then", "start": 6.05, "end": 6.25,
                            "probability": 0.9, "speaker": "SPEAKER_01"}]})
    rows.sort(key=lambda r: r["start"])
    return rows


def turns_for(_rows=None):
    """Diarization turns: the runs only, never the straddling transcript row."""
    return [{"start": s, "end": e, "speaker": label}
            for s, e, label in runs()]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="multi_integration_")
        self.engine = RecordingEngine()
        self._real_get_engine = tts_engines.get_engine
        tts_engines.get_engine = lambda lang: self.engine
        dubbing.tts_engines.get_engine = lambda lang: self.engine

    def tearDown(self):
        tts_engines.get_engine = self._real_get_engine
        dubbing.tts_engines.get_engine = self._real_get_engine
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- the seams ---------------------------------------------------------

    def test_transcript_to_assignment_to_registry(self):
        rows = conversation()
        turns = turns_for([r for r in rows if r.get("speaker")])
        labelled = SD.assign_segments(rows, turns)

        labels = sorted({SPK.label_of(s) for s in labelled if SPK.label_of(s)})
        self.assertEqual(labels,
                         ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])

        registry = RA.build_references(
            self._wav(40.0), labelled, 40.0, os.path.join(self.tmp, "refs"),
            labels=labels, sample_rate=24000, is_audio=True)
        self.assertEqual([e["label"] for e in registry], labels)
        for entry in registry:
            self.assertIsNotNone(entry["reference"],
                                 f"{entry['label']} should have a reference")
            self.assertFalse(entry["fallback"])
        # Three DISTINCT reference files: the whole point.
        paths = {e["reference"]["path"] for e in registry}
        self.assertEqual(len(paths), 3)

    def test_the_straddling_segment_splits_by_word(self):
        rows = conversation()
        turns = turns_for([r for r in rows if r.get("speaker")])
        labelled = SD.assign_segments(rows, turns)
        straddle = next(s for s in labelled if abs(s["start"] - 5.7) < 1e-9)
        by_word = {w["word"]: w["speaker"] for w in straddle["words"]}
        self.assertEqual(by_word["and"], "SPEAKER_00")
        self.assertEqual(by_word["then"], "SPEAKER_01")

    def test_each_phrase_is_synthesized_in_its_own_speakers_voice(self):
        """
        The requirement, end to end.

        Build the plan from a three-speaker transcript and check that every
        phrase's synthesis call used that phrase's speaker's reference path.
        """
        rows = conversation()
        turns = turns_for([r for r in rows if r.get("speaker")])
        labelled = SD.assign_segments(rows, turns)
        labels = sorted({SPK.label_of(s) for s in labelled if SPK.label_of(s)})

        registry = RA.build_references(
            self._wav(40.0), labelled, 40.0, os.path.join(self.tmp, "refs"),
            labels=labels, sample_rate=24000, is_audio=True)
        by_label = {e["label"]: e for e in registry}

        dm = dubbing.DurationModel(8.0, "a reference", source_lang="en",
                                   target_lang="hi")
        texts = ["एक", "दो", "तीन", "चार", "पाँच", "छह", "सात"]
        plan = dubbing.plan_timeline(labelled, texts, dm, 40.0,
                                     speakers=registry)
        self.assertTrue(plan)

        # No global reference at all: the registry is the only source, so a
        # phrase that failed to resolve would raise rather than silently use
        # one voice for everything.
        track = dubbing.synthesize_timeline(
            plan, "/nonexistent/global.wav", "", 0.0, "hi", 40.0,
            cache_dir=os.path.join(self.tmp, "units"), speakers=registry)

        self.assertEqual(len(track), int(40.0 * dubbing.SAMPLE_RATE))
        self.assertTrue(self.engine.calls, "nothing was synthesized")

        for text, ref_path in self.engine.calls:
            unit = next(u for u in plan if u["text"] == text)
            expected = by_label[unit["speaker"]]["reference"]["path"]
            self.assertEqual(
                ref_path, expected,
                f"{text!r} ({unit['speaker']}) was spoken in {ref_path}, "
                f"expected {expected}")

    def test_a_speaker_with_no_reference_falls_back_to_the_global_one(self):
        # A registry entry with no reference must not crash synthesis; it
        # resolves to the global reference, which is the pre-Phase-3 behaviour.
        registry = [{"label": "SPEAKER_00", "reference": None,
                     "total_speech": 1.0, "segment_count": 1, "fallback": True},
                    {"label": "SPEAKER_01",
                     "reference": {"path": "/tmp/b.wav", "text": "b",
                                   "duration": 8.0},
                     "total_speech": 10.0, "segment_count": 2, "fallback": False}]
        plan = [{"index": 0, "text": "hello", "start": 0.0, "duration": 1.0,
                 "speaker": "SPEAKER_00", "ref_index": 0},
                {"index": 1, "text": "world", "start": 1.0, "duration": 1.0,
                 "speaker": "SPEAKER_01", "ref_index": 1}]
        dubbing.synthesize_timeline(
            plan, "/tmp/global.wav", "g", 7.0, "hi", 2.0,
            cache_dir=os.path.join(self.tmp, "units2"), speakers=registry)
        used = dict(self.engine.calls)
        self.assertEqual(used["hello"], "/tmp/global.wav")
        self.assertEqual(used["world"], "/tmp/b.wav")

    def test_the_track_length_invariant_survives_three_speakers(self):
        rows = conversation()
        turns = turns_for([r for r in rows if r.get("speaker")])
        labelled = SD.assign_segments(rows, turns)
        labels = sorted({SPK.label_of(s) for s in labelled if SPK.label_of(s)})
        registry = RA.build_references(
            self._wav(40.0), labelled, 40.0, os.path.join(self.tmp, "refs"),
            labels=labels, sample_rate=24000, is_audio=True)
        dm = dubbing.DurationModel(8.0, "a reference", source_lang="en",
                                   target_lang="hi")
        texts = ["x"] * len(labelled)
        plan = dubbing.plan_timeline(labelled, texts, dm, 40.0,
                                     speakers=registry)
        track = dubbing.synthesize_timeline(
            plan, "/g.wav", "", 0.0, "hi", 40.0,
            cache_dir=os.path.join(self.tmp, "u3"), speakers=registry)
        self.assertEqual(len(track), int(40.0 * dubbing.SAMPLE_RATE))

    def test_per_phrase_caching_still_works_per_speaker(self):
        # Invariant 3: editing one phrase re-synthesizes one phrase. With three
        # speakers, each unit's cached WAV must still be its own.
        rows = conversation()
        turns = turns_for([r for r in rows if r.get("speaker")])
        labelled = SD.assign_segments(rows, turns)
        labels = sorted({SPK.label_of(s) for s in labelled if SPK.label_of(s)})
        registry = RA.build_references(
            self._wav(40.0), labelled, 40.0, os.path.join(self.tmp, "refs"),
            labels=labels, sample_rate=24000, is_audio=True)
        dm = dubbing.DurationModel(8.0, "a reference", source_lang="en",
                                   target_lang="hi")
        cache = os.path.join(self.tmp, "units4")
        plan = dubbing.plan_timeline(labelled, ["y"] * len(labelled), dm, 40.0,
                                     speakers=registry)
        dubbing.synthesize_timeline(plan, "/g.wav", "", 0.0, "hi", 40.0,
                                    cache_dir=cache, speakers=registry)
        for unit in plan:
            self.assertIsNotNone(dubbing.read_unit(cache, unit["index"]),
                                 f"phrase {unit['index']} was not cached")
        # Re-reading one phrase does not need the engine again.
        before = len(self.engine.calls)
        wave = dubbing.read_unit(cache, plan[0]["index"])
        self.assertIsNotNone(wave)
        self.assertEqual(len(self.engine.calls), before)

    def test_a_reclone_of_one_speaker_only_re_speaks_that_speaker(self):
        # The app-level behaviour, exercised through the same helpers the
        # endpoint uses.
        rows = conversation()
        turns = turns_for([r for r in rows if r.get("speaker")])
        labelled = SD.assign_segments(rows, turns)
        labels = sorted({SPK.label_of(s) for s in labelled if SPK.label_of(s)})
        registry = RA.build_references(
            self._wav(40.0), labelled, 40.0, os.path.join(self.tmp, "refs"),
            labels=labels, sample_rate=24000, is_audio=True)
        dm = dubbing.DurationModel(8.0, "a reference", source_lang="en",
                                   target_lang="hi")
        cache = os.path.join(self.tmp, "units5")
        plan = dubbing.plan_timeline(labelled, ["z"] * len(labelled), dm, 40.0,
                                     speakers=registry)
        dubbing.synthesize_timeline(plan, "/g.wav", "", 0.0, "hi", 40.0,
                                    cache_dir=cache, speakers=registry)

        targets = [u for u in plan if u["speaker"] == "SPEAKER_01"]
        others = [u for u in plan if u["speaker"] != "SPEAKER_01"]
        self.assertTrue(targets and others)

        # Fingerprint the other speakers' cached audio.
        before = {u["index"]: dubbing.read_unit(cache, u["index"]).copy()
                  for u in others}

        self.engine.calls.clear()
        for unit in targets:
            wave = dubbing.synth_unit(unit, "/tmp/new_b.wav", "b", 8.0, "hi")
            dubbing.write_unit(cache, unit["index"], wave)

        # Only SPEAKER_01's phrases were re-spoken.
        self.assertEqual(len(self.engine.calls), len(targets))
        for _, ref in self.engine.calls:
            self.assertEqual(ref, "/tmp/new_b.wav")
        # And the others are byte-identical.
        for u in others:
            after = dubbing.read_unit(cache, u["index"])
            np.testing.assert_array_equal(before[u["index"]], after)

    # -- fallbacks ---------------------------------------------------------

    def test_diarization_failure_leaves_the_single_speaker_path_intact(self):
        rows = conversation()
        # A diarizer that returns nothing is exactly what a missing pyannote,
        # a missing token, or a crash all produce.
        labelled = SD.assign_segments(rows, [])
        self.assertEqual(labelled, rows)

        registry = SPK.implicit_speaker({"path": "/tmp/g.wav", "text": "g",
                                         "duration": 8.0})
        dm = dubbing.DurationModel(8.0, "g", source_lang="en", target_lang="hi")
        plan = dubbing.plan_timeline(
            [{**r, "speaker": SPK.DEFAULT_LABEL} for r in rows],
            ["a"] * len(rows), dm, 40.0, speakers=registry)
        self.assertTrue(all(p["speaker"] == SPK.DEFAULT_LABEL for p in plan))
        self.assertTrue(all(p["ref_index"] == 0 for p in plan))
        for unit in plan:
            self.assertEqual(
                SPK.reference_for(unit, registry)["path"], "/tmp/g.wav")

    def test_a_fake_backend_drives_diarize_end_to_end(self):
        rows = conversation()
        turns = turns_for([r for r in rows if r.get("speaker")])
        SD._BACKENDS["fake"] = FakeDiarizer
        try:
            SD.refresh()
            SD._INSTANCES["fake"] = FakeDiarizer(turns)
            got = SD.diarize("/does/not/matter.wav", backend="fake")
            self.assertEqual(got, turns)
        finally:
            SD._BACKENDS.pop("fake", None)
            SD.refresh()

    # -- helper ------------------------------------------------------------

    def _wav(self, seconds: float) -> str:
        import soundfile as sf
        path = os.path.join(self.tmp, f"src_{seconds}.wav")
        t = np.arange(int(seconds * 24000), dtype=np.float32) / 24000
        sf.write(path, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
                 path and 24000, subtype="PCM_16")
        return path


if __name__ == "__main__":
    unittest.main()

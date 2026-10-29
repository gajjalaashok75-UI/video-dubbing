"""
Phase 3: one reference profile per speaker, and the fallback ladder.

The requirement that matters most here is the negative one: a speaker must never
be given ANOTHER DETECTED ACTOR's voice. Every test in `FallbackLadderTests`
exists to hold that line, because it is the one failure a listener cannot hear.

`build_references` shells out to ffmpeg, so the extraction tests drive it
against a real generated WAV through the actual code path rather than mocking
it. ffmpeg is the tool the pipeline itself depends on, and a test that stubs it
would not catch a broken filter chain.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import reference_audio as RA  # noqa: E402
import speakers as SPK  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def word(text, start, end, probability=0.95, speaker=None):
    row = {"word": text, "start": start, "end": end, "probability": probability}
    if speaker is not None:
        row["speaker"] = speaker
    return row


def seg(start, end, text, speaker, words):
    return {"start": start, "end": end, "text": text, "speaker": speaker,
            "words": words}


def two_speakers():
    """
    A believable two-hander: A speaks 0-8s, B speaks 8-16s, A speaks 16-24s.

    Each speaker gets enough continuous speech for a 6-12s reference window.
    """
    rows = []
    # Speaker A, two long runs.
    for i, (s, e) in enumerate([(0.0, 7.5), (16.0, 23.5)]):
        words = [word(f"a{i}{j}", s + j * 0.5, s + j * 0.5 + 0.45,
                      speaker="SPEAKER_00") for j in range(14)]
        rows.append(seg(s, e, " ".join(w["word"] for w in words),
                        "SPEAKER_00", words))
    # Speaker B, one long run.
    words = [word(f"b{j}", 8.0 + j * 0.5, 8.0 + j * 0.5 + 0.45,
                  speaker="SPEAKER_01") for j in range(14)]
    rows.append(seg(8.0, 15.5, " ".join(w["word"] for w in words),
                    "SPEAKER_01", words))
    return rows


def write_wav(path, seconds, sample_rate=24000, freq=220.0):
    """A real WAV with a real duration, so ffprobe and ffmpeg both work."""
    import soundfile as sf
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    wave = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, wave, sample_rate, subtype="PCM_16")
    return path


class WindowFilterTests(unittest.TestCase):
    """A speaker's windows are built only from that speaker's words."""

    def test_windows_are_restricted_to_the_speaker(self):
        rows = two_speakers()
        a = RA._flatten_words(rows, "SPEAKER_00")
        b = RA._flatten_words(rows, "SPEAKER_01")
        self.assertTrue(a and b)
        # A's words are only in A's two runs; B's only in B's one.
        self.assertTrue(all(w["start"] < 8.0 or w["start"] >= 16.0 for w in a))
        self.assertTrue(all(8.0 <= w["start"] < 16.0 for w in b))
        self.assertEqual(len(a) + len(b), 42)

    def test_no_speaker_filter_still_collects_everything(self):
        self.assertEqual(len(RA._flatten_words(two_speakers())), 42)

    def test_a_speakers_best_window_never_spans_another_speaker(self):
        rows = two_speakers()
        picked = RA.find_best_window(rows, 24.0, speaker="SPEAKER_01")
        self.assertIsNotNone(picked)
        start, end, _ = picked
        # B only speaks 8-15.5s, so their window must sit inside it.
        self.assertGreaterEqual(start, 7.5)
        self.assertLessEqual(end, 16.0)

    def test_word_labels_beat_the_segment_label(self):
        # A segment labelled A containing a word labelled B: the word wins, so
        # a straddling segment cannot leak the other speaker into a window.
        rows = [seg(0.0, 4.0, "x", "SPEAKER_00",
                    [word("x", 0.5, 1.0, speaker="SPEAKER_01")])]
        self.assertEqual(RA._flatten_words(rows, "SPEAKER_00"), [])
        self.assertEqual(len(RA._flatten_words(rows, "SPEAKER_01")), 1)

    def test_candidate_windows_are_per_speaker(self):
        rows = two_speakers()
        a = RA.candidate_windows(rows, 24.0, limit=5, speaker="SPEAKER_00")
        b = RA.candidate_windows(rows, 24.0, limit=5, speaker="SPEAKER_01")
        self.assertTrue(a and b)
        for w in b:
            self.assertGreaterEqual(w["start"], 7.5)
            self.assertLessEqual(w["start"] + w["duration"], 16.0)


class RankingTests(unittest.TestCase):
    """`_windows_for` prefers the middle of the 6-12s band."""

    def test_a_band_middle_window_outranks_a_marginal_one(self):
        # Two windows, same score: the one nearer 9s should win.
        windows = [
            {"start": 0.0, "duration": 6.1, "score": 3.0, "text": "a"},
            {"start": 20.0, "duration": 9.0, "score": 3.0, "text": "b"},
        ]
        original = RA.candidate_windows
        RA.candidate_windows = lambda *a, **k: windows
        try:
            ranked = RA._windows_for([], 30.0, "SPEAKER_00")
        finally:
            RA.candidate_windows = original
        self.assertEqual(ranked[0]["text"], "b")

    def test_a_clearly_better_score_still_wins(self):
        windows = [
            {"start": 0.0, "duration": 6.0, "score": 1.0, "text": "weak"},
            {"start": 20.0, "duration": 9.0, "score": 5.0, "text": "strong"},
        ]
        original = RA.candidate_windows
        RA.candidate_windows = lambda *a, **k: windows
        try:
            ranked = RA._windows_for([], 30.0, "SPEAKER_00")
        finally:
            RA.candidate_windows = original
        self.assertEqual(ranked[0]["text"], "strong")


class RegistryShapeTests(unittest.TestCase):
    """The registry must be exactly what the pipeline consumes."""

    def test_labels_are_discovered_from_the_segments(self):
        rows = two_speakers()
        labels = sorted({SPK.label_of(s) for s in rows})
        self.assertEqual(labels, ["SPEAKER_00", "SPEAKER_01"])

    def test_tallies_match_the_segments(self):
        rows = two_speakers()
        self.assertEqual(SPK.tally(rows, "SPEAKER_00")["segment_count"], 2)
        self.assertEqual(SPK.tally(rows, "SPEAKER_01")["segment_count"], 1)
        self.assertAlmostEqual(SPK.tally(rows, "SPEAKER_00")["total_speech"], 15.0)

    def test_every_entry_has_the_agreed_fields(self):
        # What the manifest, the studio and `reference_for` all read.
        entry = SPK.implicit_speaker({"path": "/tmp/a.wav", "text": "t",
                                      "duration": 8.0})[0]
        for key in ("label", "total_speech", "segment_count", "reference",
                    "reference_score", "fallback"):
            self.assertIn(key, entry)


class StitchTests(unittest.TestCase):
    """Step 2 of the ladder: several short same-speaker windows, joined."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stitch_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parts_are_joined_with_a_gap(self):
        if not HAVE_FFMPEG:
            self.skipTest("soundfile only")
        import soundfile as sf
        parts = [write_wav(os.path.join(self.tmp, f"p{i}.wav"), 1.0)
                 for i in range(3)]
        out = os.path.join(self.tmp, "joined.wav")
        RA.stitch_reference(parts, out, sample_rate=24000, gap=0.15)
        wave, sr = sf.read(out)
        # 3 seconds of audio plus two 0.15s gaps.
        self.assertAlmostEqual(len(wave) / sr, 3.3, places=1)

    def test_an_empty_part_list_raises(self):
        with self.assertRaises(RuntimeError):
            RA.stitch_reference([], os.path.join(self.tmp, "x.wav"))

    def test_unreadable_parts_are_skipped_not_fatal(self):
        import soundfile as sf
        good = write_wav(os.path.join(self.tmp, "good.wav"), 1.0)
        out = os.path.join(self.tmp, "joined.wav")
        RA.stitch_reference([os.path.join(self.tmp, "missing.wav"), good],
                            out, sample_rate=24000)
        wave, sr = sf.read(out)
        self.assertAlmostEqual(len(wave) / sr, 1.0, places=1)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is required for reference extraction")
class FallbackLadderTests(unittest.TestCase):
    """
    The five rungs, in order, and the line that must never be crossed.

    These run `build_references` for real against a generated WAV.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="refs_test_")
        self.source = write_wav(os.path.join(self.tmp, "source.wav"), 24.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, rows, **kw):
        return RA.build_references(
            self.source, rows, 24.0, os.path.join(self.tmp, "refs"),
            sample_rate=24000, is_audio=True, **kw)

    def test_rung_1_a_long_clean_same_speaker_window(self):
        registry = self._build(two_speakers())
        by_label = {e["label"]: e for e in registry}
        for label in ("SPEAKER_00", "SPEAKER_01"):
            ref = by_label[label]["reference"]
            self.assertIsNotNone(ref, f"{label} should have a reference")
            self.assertGreaterEqual(ref["duration"], RA.MIN_REF_SECONDS)
            self.assertFalse(by_label[label]["fallback"])
        # Each reference file is named for its speaker, and holds that
        # speaker's own transcript.
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "refs", "speaker_SPEAKER_00.wav")))
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "refs", "speaker_SPEAKER_01.txt")))

    def test_rung_1_references_do_not_overlap_between_speakers(self):
        # The strongest statement of "no cross-contamination" that the output
        # supports: A's reference is cut from A's time, B's from B's.
        registry = self._build(two_speakers())
        by_label = {e["label"]: e for e in registry}
        a = by_label["SPEAKER_00"]["reference"]
        b = by_label["SPEAKER_01"]["reference"]
        self.assertFalse(a["end"] > 8.0 and a["start"] < 15.5,
                         "A's reference must not sit inside B's speech")
        self.assertGreaterEqual(b["start"], 7.5)
        self.assertLessEqual(b["end"], 16.0)

    def test_rung_2_short_windows_are_stitched(self):
        # No single window reaches 6s: 3 short runs of 2.5s each. Step 2 should
        # join them rather than settle for one.
        rows = []
        for i, s in enumerate([0.0, 6.0, 12.0]):
            words = [word(f"w{i}{j}", s + j * 0.5, s + j * 0.5 + 0.45,
                          speaker="SPEAKER_00") for j in range(5)]
            rows.append(seg(s, s + 2.5, "x", "SPEAKER_00", words))
        registry = self._build(rows)
        entry = registry[0]
        self.assertIsNotNone(entry["reference"])
        self.assertGreaterEqual(entry["reference"]["duration"],
                                RA.MIN_USABLE_SECONDS)
        self.assertFalse(entry["fallback"])

    def test_rung_3_short_same_speaker_speech_is_used_with_a_warning(self):
        # 4s of speech in one run: below the 6s preference, above the 2s floor.
        words = [word(f"w{j}", j * 0.5, j * 0.5 + 0.45, speaker="SPEAKER_00")
                 for j in range(8)]
        rows = [seg(0.0, 4.0, "x", "SPEAKER_00", words)]
        registry = self._build(rows)
        entry = registry[0]
        self.assertIsNotNone(entry["reference"])
        self.assertLess(entry["reference"]["duration"], RA.MIN_REF_SECONDS)
        self.assertGreaterEqual(entry["reference"]["duration"],
                                RA.MIN_USABLE_SECONDS)
        self.assertFalse(entry["fallback"])

    def test_rung_4_a_generic_voice_is_used_and_marked(self):
        generic = write_wav(os.path.join(self.tmp, "generic.wav"), 8.0, freq=330.0)
        # 1s of speech: below the 2s floor, so nothing usable is left.
        rows = [seg(0.0, 1.0, "x", "SPEAKER_00",
                    [word("w", 0.1, 0.6, speaker="SPEAKER_00")])]
        registry = self._build(rows, generic_path=generic)
        entry = registry[0]
        self.assertIsNotNone(entry["reference"])
        self.assertEqual(entry["reference"]["path"], generic)
        self.assertTrue(entry["fallback"])

    def test_rung_5_no_usable_audio_marks_the_speaker_unresolved(self):
        rows = [seg(0.0, 1.0, "x", "SPEAKER_00",
                    [word("w", 0.1, 0.6, speaker="SPEAKER_00")])]
        registry = self._build(rows)
        entry = registry[0]
        self.assertIsNone(entry["reference"])
        self.assertTrue(entry["fallback"])

    def test_never_impersonates_another_detected_actor(self):
        """
        The requirement, stated as a test.

        SPEAKER_01 has plenty of clean audio; SPEAKER_00 has almost none. The
        old behaviour would have handed SPEAKER_00's lines to SPEAKER_01's
        voice. It must not.
        """
        rows = two_speakers()
        # Strip A's speech down to a fragment, leaving B untouched.
        rows = [r for r in rows if r["speaker"] != "SPEAKER_00"]
        rows.append(seg(0.0, 0.8, "hi", "SPEAKER_00",
                        [word("hi", 0.1, 0.5, speaker="SPEAKER_00")]))

        registry = self._build(rows)
        by_label = {e["label"]: e for e in registry}
        b_path = by_label["SPEAKER_01"]["reference"]["path"]
        a_ref = by_label["SPEAKER_00"]["reference"]

        self.assertIsNotNone(b_path)
        if a_ref is not None:
            self.assertNotEqual(
                a_ref["path"], b_path,
                "SPEAKER_00 must never be cloned from SPEAKER_01's voice")
        self.assertTrue(by_label["SPEAKER_00"]["fallback"])

    def test_the_generic_voice_is_preferred_to_another_actor(self):
        # Same setup, but a generic voice exists: it must be chosen over B's.
        generic = write_wav(os.path.join(self.tmp, "generic.wav"), 8.0, freq=440.0)
        rows = [r for r in two_speakers() if r["speaker"] != "SPEAKER_00"]
        rows.append(seg(0.0, 0.8, "hi", "SPEAKER_00",
                        [word("hi", 0.1, 0.5, speaker="SPEAKER_00")]))
        registry = self._build(rows, generic_path=generic)
        by_label = {e["label"]: e for e in registry}
        self.assertEqual(by_label["SPEAKER_00"]["reference"]["path"], generic)
        self.assertTrue(by_label["SPEAKER_00"]["fallback"])

    def test_every_speaker_appears_exactly_once(self):
        registry = self._build(two_speakers())
        self.assertEqual([e["label"] for e in registry],
                         ["SPEAKER_00", "SPEAKER_01"])

    def test_no_labels_yields_an_empty_registry(self):
        self.assertEqual(self._build([]), [])

    def test_references_are_cut_from_the_source_we_handed_it(self):
        # `is_audio=True` against a WAV must work: no `-vn` against a stream
        # that does not exist.
        registry = self._build(two_speakers())
        for entry in registry:
            self.assertTrue(os.path.exists(entry["reference"]["path"]))

    def test_the_reference_transcript_is_written_beside_the_wav(self):
        registry = self._build(two_speakers())
        for entry in registry:
            txt = os.path.join(self.tmp, "refs",
                               f"speaker_{entry['label']}.txt")
            self.assertTrue(os.path.exists(txt))
            with open(txt, encoding="utf-8") as f:
                self.assertTrue(f.read().strip())


class GenericVoiceResolutionTests(unittest.TestCase):
    """`generic_voice_path` is opt-in, and None by default."""

    def test_nothing_shipped_means_no_generic_voice(self):
        # engine/assets/generic/ does not exist in the repo, so the honest
        # default is "no generic voice" -> the speaker is marked unresolved.
        previous = os.environ.pop("VIDEOTRANS_GENERIC_VOICE", None)
        try:
            self.assertIsNone(SPK.generic_voice_path())
        finally:
            if previous is not None:
                os.environ["VIDEOTRANS_GENERIC_VOICE"] = previous

    def test_the_environment_variable_wins(self):
        tmp = tempfile.mkdtemp(prefix="generic_test_")
        try:
            path = write_wav(os.path.join(tmp, "v.wav"), 3.0)
            os.environ["VIDEOTRANS_GENERIC_VOICE"] = path
            self.assertEqual(SPK.generic_voice_path(), path)
        finally:
            os.environ.pop("VIDEOTRANS_GENERIC_VOICE", None)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_missing_environment_path_is_ignored(self):
        os.environ["VIDEOTRANS_GENERIC_VOICE"] = "/nope/nothing.wav"
        try:
            self.assertIsNone(SPK.generic_voice_path())
        finally:
            os.environ.pop("VIDEOTRANS_GENERIC_VOICE", None)


class EndToEndReferenceTests(unittest.TestCase):
    """
    The property the whole phase exists for: each speaker's phrases resolve to
    that speaker's own reference, all the way to synthesis.
    """

    def test_each_unit_resolves_to_its_own_speaker_reference(self):
        import dubbing
        registry = [
            {"label": "SPEAKER_00",
             "reference": {"path": "/tmp/a.wav", "text": "a", "duration": 8.0}},
            {"label": "SPEAKER_01",
             "reference": {"path": "/tmp/b.wav", "text": "b", "duration": 8.0}},
        ]
        rows = [seg(0.0, 2.0, "one", "SPEAKER_00", None),
                seg(2.5, 4.5, "two", "SPEAKER_01", None)]
        dm = dubbing.DurationModel(8.0, "a reference", source_lang="en",
                                   target_lang="hi")
        plan = dubbing.plan_timeline(rows, ["x", "y"], dm, 5.0,
                                     speakers=registry)
        self.assertEqual(plan[0]["ref_index"], 0)
        self.assertEqual(plan[1]["ref_index"], 1)
        self.assertEqual(SPK.reference_for(plan[0], registry)["path"],
                         "/tmp/a.wav")
        self.assertEqual(SPK.reference_for(plan[1], registry)["path"],
                         "/tmp/b.wav")

    def test_a_single_speaker_registry_keeps_the_global_reference(self):
        import dubbing
        global_ref = {"path": "/tmp/global.wav", "text": "g", "duration": 7.0}
        registry = SPK.implicit_speaker(global_ref)
        rows = [seg(0.0, 2.0, "one", "SPEAKER_00", None)]
        dm = dubbing.DurationModel(7.0, "a reference", source_lang="en",
                                   target_lang="hi")
        plan = dubbing.plan_timeline(rows, ["x"], dm, 3.0, speakers=registry)
        self.assertEqual(SPK.reference_for(plan[0], registry, global_ref),
                         global_ref)

    def test_the_single_speaker_plan_is_identical_with_and_without_a_registry(self):
        """
        The regression lock for Phase 1.

        A one-speaker video must produce the same timeline it always did. This
        compares the whole plan - start, duration, rate, natural - with no
        registry at all (the pre-Phase-1 call signature) against the implicit
        registry a real single-speaker run now builds.
        """
        import dubbing
        global_ref = {"path": "/tmp/g.wav", "text": "a reference here",
                      "duration": 8.0}
        registry = SPK.implicit_speaker(global_ref)
        rows = [seg(0.0, 2.0, "one two three", "SPEAKER_00", None),
                seg(2.5, 4.5, "four five six", "SPEAKER_00", None),
                seg(5.0, 7.0, "seven eight nine", "SPEAKER_00", None)]
        texts = ["एक दो तीन", "चार पाँच छह", "सात आठ नौ"]
        dm = dubbing.DurationModel(8.0, "a reference here", source_lang="en",
                                   target_lang="hi")

        # Exactly the call the pre-Phase-1 pipeline made.
        before = dubbing.plan_timeline(rows, texts, dm, 7.0)
        # What a single-speaker run makes now.
        after = dubbing.plan_timeline(rows, texts, dm, 7.0, speakers=registry)

        self.assertEqual(len(before), len(after))
        for b, a in zip(before, after):
            for key in ("index", "text", "start", "duration", "natural", "rate"):
                self.assertEqual(b[key], a[key],
                                 f"{key} moved for phrase {b['index']}")

    def test_every_single_speaker_unit_resolves_to_the_global_reference(self):
        import dubbing
        global_ref = {"path": "/tmp/g.wav", "text": "a reference here",
                      "duration": 8.0}
        registry = SPK.implicit_speaker(global_ref)
        rows = [seg(0.0, 2.0, "one two three", "SPEAKER_00", None)]
        dm = dubbing.DurationModel(8.0, "a reference here", source_lang="en",
                                   target_lang="hi")
        plan = dubbing.plan_timeline(rows, ["x"], dm, 3.0, speakers=registry)
        for unit in plan:
            resolved = SPK.reference_for(unit, registry, global_ref)
            self.assertEqual(resolved["path"], global_ref["path"])
            self.assertEqual(resolved["text"], global_ref["text"])


if __name__ == "__main__":
    unittest.main()

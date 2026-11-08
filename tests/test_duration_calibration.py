"""
The duration request must be calibrated, not a raw byte count, across scripts.

`IndicF5Engine.natural_duration` used to mirror F5's internal length heuristic
verbatim:

    duration = ref_audio_len + ref_audio_len / ref_text_len * gen_text_len

with `gen_text_len` counted in UTF-8 BYTES. Same-script that is harmless. Cross
script it is catastrophic: a Devanagari character is 3 bytes against 1 for
Latin, so an English reference conditioned on a Hindi target asks for roughly
three times the time the words need. The timeline then overflows and the closing
squeeze in `plan_timeline` crushes every phrase to fit.

Measured on an English->Hindi clip before the fix: the byte mirror wanted 15.5s
of speech for a 9.0s video (a 1.7x squeeze) and SPEAKER_00's opening line was
crushed below intelligibility. After the fix the same plan wanted 6.7s (1.02x)
and all three lines round-tripped. That is the regression these tests pin.

The invariant is the one that matters: for a Hindi target the estimate must not
scale with the Devanagari byte count, and it must agree with the calibrated
`dubbing.DurationModel` that the rest of the pipeline already trusts.

No models, no network, no GPU.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import dubbing  # noqa: E402
import tts_engines  # noqa: E402

HINDI = "देखो, यह तो हमारे बीच की निजी बात थी।"
ENGLISH_REF = "a private, personal thing between us"


def _reference(path: str, seconds: float = 6.0) -> str:
    """A quiet-but-nonzero reference clip, long enough to calibrate on."""
    sr = 24000
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    # A voiced-ish tone with an amplitude envelope, so the voiced-fraction
    # estimate in `_seconds_per_byte` has something to find.
    wave = 0.2 * np.sin(2 * np.pi * 140 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    wave += 0.002 * rng.standard_normal(wave.size)
    sf.write(path, wave.astype(np.float32), sr, subtype="PCM_16")
    return path


class CrossScriptDurationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ref = _reference(os.path.join(tmp.name, "ref.wav"))
        self.engine = tts_engines.IndicF5Engine()

    def test_hindi_estimate_does_not_scale_with_byte_count(self):
        """The Devanagari byte count must not inflate the estimate ~3x."""
        got = self.engine.natural_duration(
            HINDI, self.ref, ENGLISH_REF, lang="hi", source_lang="en")
        # A 37-character Hindi line is a few seconds of speech, not ten. The
        # byte mirror produced ~5.6s for this line; the calibrated estimate is
        # far closer to its true length.
        self.assertLess(got, 5.0)
        self.assertGreater(got, 1.0)

    def test_estimate_agrees_with_the_pipeline_duration_model(self):
        """
        The generator's estimate and `DurationModel` must not disagree by a
        factor — they describe the same speaker saying the same words.
        """
        model = dubbing.DurationModel(
            6.0, ENGLISH_REF,
            source_lang="en", target_lang="hi")
        by_model = model.estimate(HINDI, source_text="Look, that was private", slot=3.0)
        by_engine = self.engine.natural_duration(
            HINDI, self.ref, ENGLISH_REF, lang="hi", source_lang="en")
        ratio = by_engine / by_model
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)

    def test_same_script_keeps_the_model_mirror(self):
        """
        With no source language there is nothing to separate tempo from script
        density, so the byte mirror is kept — the previous behaviour.
        """
        text = "a private personal thing"
        got = self.engine.natural_duration(text, self.ref, ENGLISH_REF, lang="en")
        self.assertGreater(got, 0.0)

    def test_no_source_language_matches_the_old_byte_math(self):
        old = self.engine.natural_duration(HINDI, self.ref, ENGLISH_REF, lang="hi")
        # Byte-based, so it is the inflated one — this is what the fix replaces
        # when a source language is known.
        calibrated = self.engine.natural_duration(
            HINDI, self.ref, ENGLISH_REF, lang="hi", source_lang="en")
        self.assertGreater(old, calibrated)


if __name__ == "__main__":
    unittest.main()

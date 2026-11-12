"""
The Fish Audio S2 Pro engine, as an optional second voice model.

S2 Pro (extraplus/s2-pro) is a Fish Speech model that covers most of this
project's targets — Indic and foreign alike — from one checkpoint. It is
selected with VIDEOTRANS_TTS_ENGINE=s2pro, or the "Voice engine" setting in the
studio. Two things must hold for it to be usable: the `fish_speech` package,
and its own ~11 GB of weights. Neither is in the default install, so the checks
below are mostly about refusing S2 Pro cleanly rather than about synthesis.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import engines as E  # noqa: E402
import languages as L  # noqa: E402
import models as M  # noqa: E402
import readiness as R  # noqa: E402
import tts_engines as T  # noqa: E402


class _NoEnv(unittest.TestCase):
    """Clear the pin before each test; it decides which engine speaks."""

    def setUp(self):
        self._saved = os.environ.pop("VIDEOTRANS_TTS_ENGINE", None)
        L._PREF_CACHE = None
        # The engine availability cache outlives a test and would leak "S2 Pro
        # is installed" into the next one.
        T._AVAILABLE.clear()

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("VIDEOTRANS_TTS_ENGINE", None)
        else:
            os.environ["VIDEOTRANS_TTS_ENGINE"] = self._saved
        L._PREF_CACHE = None
        T._AVAILABLE.clear()


class RegistrationTests(_NoEnv):
    def test_the_engine_id_is_s2pro(self):
        self.assertEqual(L.ENGINE_S2PRO, "s2pro")

    def test_weights_are_in_the_manifest(self):
        ids = [e["id"] for e in M.MODELS]
        self.assertIn("extraplus/s2-pro", ids)

    def test_the_weights_are_optional(self):
        # A default install must not require 11 GB it may never use.
        entry = next(e for e in M.MODELS if e["id"] == "extraplus/s2-pro")
        self.assertFalse(entry["required"])

    def test_the_weights_are_ungated(self):
        # Unlike IndicF5, S2 Pro needs no licence acceptance to fetch.
        entry = next(e for e in M.MODELS if e["id"] == "extraplus/s2-pro")
        self.assertFalse(entry.get("gated", False))

    def test_the_catalogue_offers_a_voice_stage(self):
        self.assertIn("voice", E.CATALOG)
        ids = {o["id"] for o in E.CATALOG["voice"]["options"]}
        self.assertEqual(ids, {"auto", "s2pro", "indicf5", "xtts"})

    def test_the_voice_stage_defaults_to_auto(self):
        self.assertEqual(E.CATALOG["voice"]["default"], "auto")

    def test_the_voice_stage_reads_the_documented_variable(self):
        self.assertEqual(E.CATALOG["voice"]["env"], "VIDEOTRANS_TTS_ENGINE")


class PreferenceTests(_NoEnv):
    def test_auto_keeps_the_house_engine(self):
        self.assertEqual(L.preferred_engine(), "auto")
        self.assertEqual(L.engine_for("hi"), "indicf5")
        self.assertEqual(L.engine_for("en"), "xtts")

    def test_auto_falls_back_to_the_house_engine_without_the_weights(self):
        # The reported request was "set S2 Pro as the default". It is the
        # preferred voice, but a fresh clone has neither the fish_speech
        # package nor the 11 GB of weights, so auto must not send every render
        # to an engine that cannot load. Here nothing is installed, so every
        # language stays on its house engine.
        T._AVAILABLE.clear()
        for code in ("hi", "te", "en", "ja"):
            self.assertNotEqual(L.engine_for(code), "s2pro")

    def test_auto_prefers_s2pro_once_it_is_installed(self):
        # With the package and weights present, auto becomes S2 Pro for every
        # language it covers - the "default" the request asked for.
        T._AVAILABLE["s2pro"] = True
        try:
            self.assertEqual(L.engine_for("hi"), "s2pro")
            self.assertEqual(L.engine_for("en"), "s2pro")
            # Odia is still IndicF5: S2 Pro cannot speak it, so preferring it
            # must never take Odia's voice away.
            self.assertEqual(L.engine_for("or"), "indicf5")
        finally:
            T._AVAILABLE.clear()

    def test_a_pin_beats_the_installed_check(self):
        # Explicitly pinning IndicF5 must win even when S2 Pro is installed.
        T._AVAILABLE["s2pro"] = True
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "indicf5"
        L._PREF_CACHE = None
        try:
            self.assertEqual(L.engine_for("hi"), "indicf5")
            self.assertEqual(L.engine_for("en"), "xtts")
        finally:
            T._AVAILABLE.clear()

    def test_s2pro_covers_an_indic_language(self):
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "s2pro"
        L._PREF_CACHE = None
        self.assertEqual(L.engine_for("hi"), "s2pro")
        self.assertEqual(L.engine_for("te"), "s2pro")

    def test_s2pro_covers_a_foreign_language(self):
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "s2pro"
        L._PREF_CACHE = None
        self.assertEqual(L.engine_for("en"), "s2pro")
        self.assertEqual(L.engine_for("ja"), "s2pro")

    def test_s2pro_falls_back_for_a_language_it_cannot_speak(self):
        # Odia is the one Indian language S2 Pro does not cover. Pinning it
        # must not take Odia's voice away; it stays on IndicF5.
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "s2pro"
        L._PREF_CACHE = None
        self.assertFalse(L.engine_supports("s2pro", "or"))
        self.assertEqual(L.engine_for("or"), "indicf5")

    def test_an_unknown_pin_does_not_break_the_house_engine(self):
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "no-such-engine"
        L._PREF_CACHE = None
        self.assertEqual(L.engine_for("hi"), "indicf5")
        self.assertEqual(L.engine_for("en"), "xtts")

    def test_engines_for_lists_every_engine_that_can_speak_a_language(self):
        # House engine first, then the alternatives. Hindi is also in XTTS's
        # set, so all three can speak it.
        self.assertEqual(L.engines_for("hi"), ["indicf5", "s2pro", "xtts"])
        self.assertEqual(L.engines_for("en"), ["xtts", "s2pro"])
        # Odia is IndicF5 only.
        self.assertEqual(L.engines_for("or"), ["indicf5"])

    def test_supported_targets_report_the_active_engine(self):
        rows = {r["code"]: r for r in L.supported_targets()}
        self.assertEqual(rows["hi"]["engine"], "indicf5")
        self.assertEqual(rows["hi"]["active_engine"], "indicf5")
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "s2pro"
        L._PREF_CACHE = None
        rows = {r["code"]: r for r in L.supported_targets()}
        self.assertEqual(rows["hi"]["active_engine"], "s2pro")
        self.assertEqual(rows["or"]["active_engine"], "indicf5")


class AvailabilityTests(_NoEnv):
    def setUp(self):
        super().setUp()
        T._AVAILABLE.clear()

    def test_s2pro_is_unavailable_without_the_package(self):
        # fish_speech is not installed in this environment, so the honest
        # answer is "no" - the studio must not offer an engine that will fail
        # several minutes into a render.
        import importlib.util
        if importlib.util.find_spec("fish_speech") is not None:
            self.skipTest("fish_speech is installed here")
        self.assertFalse(T.engine_available("s2pro"))

    def test_the_present_check_requires_the_package(self):
        import importlib.util
        if importlib.util.find_spec("fish_speech") is not None:
            self.skipTest("fish_speech is installed here")
        self.assertFalse(T._s2pro_present())

    def test_get_engine_returns_the_s2pro_class_when_pinned(self):
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "s2pro"
        L._PREF_CACHE = None
        T._ENGINES.clear()
        try:
            engine = T.get_engine("hi")
            self.assertIsInstance(engine, T.S2ProEngine)
        finally:
            T._ENGINES.clear()

    def test_s2pro_reports_the_codec_sample_rate(self):
        # The DAC codec in this checkpoint is 44.1 kHz; the pipeline resamples
        # to its own 24 kHz, so the number here only has to be the truth.
        self.assertEqual(T.S2ProEngine.SAMPLE_RATE, 44100)

    def test_s2pro_has_no_nfe_step(self):
        # It is not an f5_tts model, so synth_unit must take the non-IndicF5
        # branch. If this ever grows an nfe_step, that branch has to be revisited.
        self.assertFalse(hasattr(T.S2ProEngine(), "nfe_step"))

    def test_s2pro_has_a_natural_duration_estimate(self):
        engine = T.S2ProEngine()
        long_text = engine.natural_duration("a longer line of text here", lang="en")
        short_text = engine.natural_duration("hi", lang="en")
        self.assertGreater(long_text, short_text)
        self.assertGreater(short_text, 0.0)

    def test_empty_text_synthesizes_nothing(self):
        # Must not touch the model: an empty unit is common at plan edges.
        wave, sr = T.S2ProEngine().synthesize("", None, "")
        self.assertEqual(wave.size, 0)
        self.assertEqual(sr, T.S2ProEngine.SAMPLE_RATE)


class ReadinessTests(_NoEnv):
    def test_the_feature_is_reported_as_optional(self):
        entry = R.feature_s2pro()
        self.assertEqual(entry["tier"], R.OPTIONAL)

    def test_the_reason_names_the_package_when_it_is_missing(self):
        import importlib.util
        if importlib.util.find_spec("fish_speech") is not None:
            self.skipTest("fish_speech is installed here")
        entry = R.feature_s2pro()
        self.assertEqual(entry["status"], R.OPTIONAL_MISSING)
        self.assertIn("fish_speech", entry["reason"])

    def test_it_appears_in_the_snapshot(self):
        snap = R.snapshot()
        self.assertIn("s2pro", snap["features"])

    def test_a_pinned_language_blocks_when_s2pro_is_missing(self):
        import importlib.util
        if importlib.util.find_spec("fish_speech") is not None:
            self.skipTest("fish_speech is installed here")
        os.environ["VIDEOTRANS_TTS_ENGINE"] = "s2pro"
        L._PREF_CACHE = None
        entry = R.feature_language("hi")
        # The user asked for Hindi and pinned an engine that is not ready, so
        # this is blocking rather than merely optional.
        self.assertEqual(entry["engine"], "s2pro")
        self.assertEqual(entry["status"], R.BLOCKED)


class GracefulFailureTests(_NoEnv):
    def test_loading_without_the_package_raises_a_synthesis_error(self):
        import importlib.util
        if importlib.util.find_spec("fish_speech") is not None:
            self.skipTest("fish_speech is installed here")
        with self.assertRaises(T.SynthesisError) as ctx:
            T.S2ProEngine().load()
        self.assertIn("fish_speech", str(ctx.exception))

    def test_the_message_names_the_install_command(self):
        import importlib.util
        if importlib.util.find_spec("fish_speech") is not None:
            self.skipTest("fish_speech is installed here")
        with self.assertRaises(T.SynthesisError) as ctx:
            T.S2ProEngine().load()
        self.assertIn("fish-speech", str(ctx.exception))


class NotebookTests(unittest.TestCase):
    def test_the_install_cell_installs_fish_speech_without_deps(self):
        # Installing it WITH deps would pull torch 2.8 over the CUDA-12 torch
        # 2.6 build and break CTranslate2 with libcublas.so.12.
        src = (ROOT / "deploy" / "notebooks" / "build_notebooks.py").read_text(
            encoding="utf-8")
        self.assertIn("fish-speech.git", src)
        block = src.split("S2PRO_DEPS")[0]
        self.assertIn("--no-deps", block)

    def test_the_requirements_file_documents_the_extra(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("extraplus/s2-pro", text)
        self.assertIn("fish-speech", text)


if __name__ == "__main__":
    unittest.main()

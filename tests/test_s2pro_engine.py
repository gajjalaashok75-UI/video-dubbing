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
from unittest import mock

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

    def test_the_weights_are_not_in_the_default_download_set(self):
        # 11 GB beside the other ~7.5 GB overflows a Kaggle working volume, and
        # nothing loads it unless the user pins the engine. `extra` keeps it out
        # of a default fetch and out of `missing(core_only=True)`.
        entry = next(e for e in M.MODELS if e["id"] == "extraplus/s2-pro")
        self.assertTrue(entry.get("extra"))
        core = [r["id"] for r in M.missing(core_only=True)]
        self.assertNotIn("extraplus/s2-pro", core)
        full = [r["id"] for r in M.missing(core_only=False)]
        self.assertIn("extraplus/s2-pro", full)

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

    def test_auto_never_selects_s2pro_even_when_it_is_installed(self):
        # S2 Pro is a ~5B model whose weights fill a free T4 and starve Whisper
        # and Wav2Lip. It must stay OPT-IN: `auto` keeps the house engine even
        # when the package and the 11 GB of weights are present. Only an
        # explicit pin may choose it.
        T._AVAILABLE["s2pro"] = True
        try:
            for code in ("hi", "te", "en", "ja", "or"):
                self.assertNotEqual(L.engine_for(code), "s2pro",
                                    "auto must not implicitly boot S2 Pro")
            self.assertEqual(L.engine_for("hi"), "indicf5")
            self.assertEqual(L.engine_for("en"), "xtts")
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


class VramGuardTests(_NoEnv):
    """
    A load that cannot fit must be refused, not left to the allocator.

    S2 Pro's bf16 weights are ~11 GB. Loading it on a 14.56 GiB T4 with the
    other models resident does not just fail - the allocator takes the whole
    kernel down, which the user sees as the runtime dying. The guard reads the
    free memory first and raises a named error instead.
    """

    def test_a_tight_gpu_is_refused(self):
        with mock.patch.object(T, "_free_vram_gb", lambda: 4.0):
            with self.assertRaises(T.SynthesisError) as ctx:
                T._require_room_for_s2pro("cuda")
        self.assertIn("12 GB", str(ctx.exception))
        self.assertIn("IndicF5", str(ctx.exception))

    def test_a_roomy_gpu_is_allowed(self):
        with mock.patch.object(T, "_free_vram_gb", lambda: 15.0):
            T._require_room_for_s2pro("cuda")     # does not raise

    def test_an_unknown_free_memory_does_not_block(self):
        # `_free_vram_gb` returns None when the query is unavailable (CPU, MPS,
        # a driver hiccup). That must not read as "zero free" and refuse a load
        # that would in fact have worked.
        with mock.patch.object(T, "_free_vram_gb", lambda: None):
            T._require_room_for_s2pro("cuda")     # does not raise

    def test_a_non_cuda_device_is_never_blocked(self):
        with mock.patch.object(T, "_free_vram_gb",
                               side_effect=AssertionError("must not query")):
            T._require_room_for_s2pro("cpu")
            T._require_room_for_s2pro("mps")

    def test_the_free_memory_reads_the_device(self):
        # A real read, not a mock: on a machine without CUDA it must say so
        # rather than invent a number.
        free = T._free_vram_gb()
        if _cuda():
            self.assertIsInstance(free, float)
            self.assertGreater(free, 0)
        else:
            self.assertIsNone(free)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                                        # noqa: BLE001
        return False


class PipInstallTests(unittest.TestCase):
    """
    The shipped `pip_install` must survive one bad package in a batch.

    This is the exact failure that broke S2 Pro on Colab/Kaggle: `pyaudio`
    cannot build (no PortAudio headers), pip aborts the whole `pip install a b
    c ...`, and every other package in the batch - including `audiotools`, which
    `import fish_speech` needs - is silently dropped. The helper retries each
    package alone so the survivors still land.
    """

    def _load(self):
        import ast
        import json

        nb = json.loads((ROOT / "deploy" / "notebooks" / "colab.ipynb").read_text(
            encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            tree = ast.parse(src)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "pip_install":
                    return ast.get_source_segment(src, node)
        self.fail("pip_install is not defined in the install cell")

    def _packages(self, cmd):
        """The package arguments of a `pip install` command, ignoring flags."""
        out, skip_next = [], False
        for arg in cmd:
            if skip_next:
                skip_next = False
                continue
            if arg in ("-c", "-r", "-t", "--target"):
                skip_next = True
                continue
            if arg.startswith("-"):
                continue
            out.append(arg)
        return out[3:]  # drop the "<python> -m pip install" prefix

    def _namespace(self, fake_run):
        import sys as _sys
        import types

        subprocess_stub = types.SimpleNamespace(run=fake_run)
        ns = {
            "sys": _sys,
            "subprocess": subprocess_stub,
            "CONSTRAINTS": "/tmp/constraints.txt",
        }
        exec(self._load(), ns)
        return ns["pip_install"]

    def test_a_bad_package_does_not_sink_the_others(self):
        calls = []

        def fake_run(cmd, **kwargs):
            pkgs = self._packages(cmd)
            # A lone pyaudio fails, and so does any batch that contains it, as
            # real pip does; everything else installs.
            calls.append(list(pkgs))
            if "pyaudio" in pkgs:
                return type("P", (), {"returncode": 1, "stdout": "",
                                      "stderr": "portaudio.h: No such file"})()
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        pip_install = self._namespace(fake_run)
        failed = pip_install(["audiotools", "pyaudio", "loguru"])
        self.assertEqual(failed, ["pyaudio"])
        # The batch was tried once, then each package alone.
        self.assertTrue(any(len(c) > 1 for c in calls))
        self.assertIn(["audiotools"], calls)
        self.assertIn(["loguru"], calls)

    def test_a_clean_batch_is_a_single_call(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        pip_install = self._namespace(fake_run)
        self.assertEqual(pip_install(["a", "b", "c"]), [])
        self.assertEqual(len(calls), 1)


class NotebookTests(unittest.TestCase):
    def test_the_install_cell_installs_fish_speech_without_deps(self):
        # Installing it WITH deps would pull torch 2.8 over the CUDA-12 torch
        # 2.6 build and break CTranslate2 with libcublas.so.12.
        src = (ROOT / "deploy" / "notebooks" / "build_notebooks.py").read_text(
            encoding="utf-8")
        self.assertIn("fish-speech.git", src)
        block = src.split("S2PRO_PACKAGES")[0]
        self.assertIn("--no-deps", block)

    def test_pyaudio_is_not_installed(self):
        # pyaudio cannot build without PortAudio headers, and one failing
        # package aborts the whole `pip install` batch - which is how
        # `audiotools` went missing and S2 Pro failed to import. Nothing on the
        # TTS path imports it, so it must not be in the install list.
        src = (ROOT / "deploy" / "notebooks" / "build_notebooks.py").read_text(
            encoding="utf-8")
        block = src.split("S2PRO_PACKAGES")[1].split("]")[0]
        self.assertNotIn("pyaudio", block)

    def test_the_dependency_group_is_installed_tolerantly(self):
        # The group must go through `pip_install` (per-package salvage), not a
        # single all-or-nothing `pip install`.
        src = (ROOT / "deploy" / "notebooks" / "build_notebooks.py").read_text(
            encoding="utf-8")
        self.assertIn("failed = pip_install(", src)

    def test_the_import_check_names_the_modules_the_engine_loads(self):
        # `import fish_speech` passes on a broken install; the check must ask
        # for the modules `S2ProEngine.load()` actually imports.
        src = (ROOT / "deploy" / "notebooks" / "build_notebooks.py").read_text(
            encoding="utf-8")
        self.assertIn("S2PRO_IMPORT", src)
        self.assertIn("fish_speech.models.dac.inference import load_model", src)
        self.assertIn("launch_thread_safe_queue", src)

    def test_a_pyrootutils_marker_is_written(self):
        # `fish_speech/models/dac/inference.py` calls
        # `pyrootutils.setup_root(..., indicator=".project-root")` at import
        # time; without a marker above it the import raises FileNotFoundError.
        src = (ROOT / "deploy" / "notebooks" / "build_notebooks.py").read_text(
            encoding="utf-8")
        self.assertIn(".project-root", src)

    def test_the_requirements_file_documents_the_extra(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("extraplus/s2-pro", text)
        self.assertIn("fish-speech", text)


if __name__ == "__main__":
    unittest.main()

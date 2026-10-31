"""The load-path bugs that a package-and-weights check cannot see.

One hosted session had every dependency verified and every weight downloaded,
and still failed twice in the same way:

    [BOOT] Loading translator failed: cannot import name '_load_model'
    AttributeError: NllbTokenizerFast has no attribute dtype
    [DIAR] ... Pipeline.from_pretrained() got an unexpected keyword argument 'token'
    [APP] Could not apply the mt change live: no attribute 'refresh'

Four separate code/API mismatches, none of which any `find_spec` or
`models.is_cached()` can notice. These tests pin each one, plus the readiness
probes that now report them instead of claiming ready.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import readiness  # noqa: E402


class TranslationLoaderTests(unittest.TestCase):
    def test_the_names_app_calls_are_defined(self):
        # app.py imports `_load_model` at boot and calls `refresh()` when the
        # studio changes the model. Both were missing after a rename.
        self.assertEqual(
            readiness._defines("translation_v2", "translate_segments",
                               "_load_model", "refresh"),
            [])

    def test_the_load_path_never_reads_a_dtype_off_the_tokenizer(self):
        # Parsed, not grepped: the comment explaining the old line quotes
        # `tok.dtype`, and only executable code may be judged.
        source = readiness._source_of("translation_v2")
        self.assertIsNotNone(source, "translation_v2.py was not found")
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Attribute) and node.attr == "dtype"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("tok", "tokenizer")):
                offenders.append(node.lineno)
        self.assertEqual(offenders, [],
                         "the tokenizer is asked for .dtype; only a model has one")

    def test_the_load_error_distinguishes_packages_weights_and_code(self):
        try:
            import translation_v2
        except Exception as e:                               # noqa: BLE001
            self.skipTest(f"translation_v2 is not importable here ({e})")

        code = translation_v2._explain_load_error(AttributeError("no dtype"))
        self.assertIn("code failure", code)
        self.assertNotIn("pip install", code)

        package = translation_v2._explain_load_error(
            ModuleNotFoundError("No module named 'sentencepiece'"))
        self.assertIn("pip install", package)

        weights = translation_v2._explain_load_error(
            OSError("We couldn't connect to huggingface.co"))
        self.assertIn("download", weights)


class DiarizerAuthTests(unittest.TestCase):
    """The keyword has to match the pinned pyannote, not the newest hub API."""

    def _modules(self, accepted: set, calls: list, error: str = None):
        class Pipeline:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                calls.append(kwargs)
                extra = set(kwargs) - accepted
                if extra:
                    name = sorted(extra)[0]
                    raise TypeError(
                        "Pipeline.from_pretrained() got an unexpected keyword "
                        f"argument '{name}'")
                return "pipeline"

        module = types.ModuleType("pyannote.audio")
        module.Pipeline = Pipeline
        return {"pyannote": types.ModuleType("pyannote"),
                "pyannote.audio": module}

    def test_the_pinned_pyannote_gets_use_auth_token(self):
        calls: list = []
        with mock.patch.dict(sys.modules,
                             self._modules({"use_auth_token"}, calls)):
            import speaker_diarization
            result = speaker_diarization.PyannoteDiarizer()._from_pretrained("hf_x")
        self.assertEqual(result, "pipeline")
        self.assertEqual(list(calls[0]), ["use_auth_token"])

    def test_a_newer_pyannote_falls_back_to_token(self):
        calls: list = []
        with mock.patch.dict(sys.modules, self._modules({"token"}, calls)):
            import speaker_diarization
            result = speaker_diarization.PyannoteDiarizer()._from_pretrained("hf_x")
        self.assertEqual(result, "pipeline")
        self.assertEqual(list(calls[-1]), ["token"])

    def test_an_unrelated_type_error_is_not_masked(self):
        class Pipeline:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                raise TypeError("something else broke")

        module = types.ModuleType("pyannote.audio")
        module.Pipeline = Pipeline
        with mock.patch.dict(sys.modules,
                             {"pyannote": types.ModuleType("pyannote"),
                              "pyannote.audio": module}):
            import speaker_diarization
            with self.assertRaises(TypeError) as caught:
                speaker_diarization.PyannoteDiarizer()._from_pretrained("hf_x")
        self.assertIn("something else", str(caught.exception))


class DemucsProbeTests(unittest.TestCase):
    """`find_spec("demucs")` is not the question; its imports are."""

    def test_the_modules_demucs_imports_are_required(self):
        import source_separation
        for name in ("lameenc", "julius", "openunmix", "dora", "omegaconf"):
            with self.subTest(module=name):
                self.assertIn(name, source_separation.REQUIRED_MODULES)

    def test_a_missing_dependency_makes_it_unavailable(self):
        import source_separation
        present = {"demucs.separate", "julius", "openunmix", "dora",
                   "omegaconf", "tqdm"}
        with mock.patch("importlib.util.find_spec",
                        side_effect=lambda name, *a, **k:
                        object() if name in present else None):
            self.assertEqual(source_separation.missing_deps(), ["lameenc"])
            self.assertFalse(source_separation.available())

    def test_a_complete_install_is_available(self):
        import source_separation
        with mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(source_separation.missing_deps(), [])
            self.assertTrue(source_separation.available())

    def test_readiness_does_not_call_it_ready_without_its_dependencies(self):
        with mock.patch.object(readiness, "_has_module", return_value=True), \
                mock.patch.object(readiness, "_demucs_missing",
                                  return_value=["lameenc", "dora"]):
            entry = readiness.feature_background(True)
        self.assertNotEqual(entry["status"], readiness.READY)
        self.assertIn("lameenc", entry["reason"])


class ReadinessApiProbeTests(unittest.TestCase):
    """Readiness must test the paths that broke, not just package presence."""

    def test_an_absent_module_is_not_reported_as_a_code_bug(self):
        self.assertIsNone(readiness._defines("no_such_module_xyz", "x"))

    def test_translation_is_blocked_when_an_entry_point_is_missing(self):
        fake_models = types.SimpleNamespace(is_cached=lambda repo: True)
        with mock.patch.object(readiness, "_defines",
                               return_value=["_load_model", "refresh"]), \
                mock.patch.object(readiness, "_has_module", return_value=True), \
                mock.patch.dict(sys.modules, {"models": fake_models}):
            entry = readiness.feature_translation()
        self.assertEqual(entry["status"], readiness.BLOCKED)
        self.assertIn("_load_model", entry["reason"])

    def test_diarization_probe_flags_a_missing_auth_keyword(self):
        with mock.patch.object(readiness, "_source_of",
                               return_value="def refresh(): pass"):
            self.assertIsNotNone(readiness.diarization_api())
        with mock.patch.object(readiness, "_source_of", return_value=None):
            self.assertIsNone(readiness.diarization_api())

    def test_the_shipped_code_passes_both_probes(self):
        self.assertIsNone(readiness.translation_api())
        self.assertIsNone(readiness.diarization_api())


if __name__ == "__main__":
    unittest.main()

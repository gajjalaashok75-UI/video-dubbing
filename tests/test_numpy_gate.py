"""Startup must refuse a NumPy that numba cannot run.

A hosted run verified `numpy==2.2.6` and then started a server that reported

    [BOOT] Loading voice model failed: Numba needs NumPy 2.2 or less. Got NumPy 2.5
    [Wav2Lip] Warmup failed: Numba needs NumPy 2.2 or less. Got NumPy 2.5

while the Setup page still said IndicF5 and lip sync were ready. The versions
themselves were correct; nothing compared them against what the interpreter
could actually run, and the message named neither the interpreter nor the numpy
file, so the mismatch looked impossible.

These tests pin the verdict, the report's contents, and the two places the
verdict is enforced: the server start and the readiness payload.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import readiness  # noqa: E402


def _report(**overrides) -> dict:
    report = {
        "executable": "/usr/bin/python3",
        "version": "2.5.1",
        "file": "/usr/lib/python3/dist-packages/numpy/__init__.py",
        "numba": "0.61.2",
        "maximum": readiness.NUMPY_MAXIMUM,
        "compatible": False,
        "reason": "numba needs NumPy 2.2 or less and this interpreter has 2.5.1.",
    }
    report.update(overrides)
    return report


class VerdictTests(unittest.TestCase):
    def test_the_pinned_numpy_is_accepted(self):
        ok, reason = readiness.numpy_verdict("2.2.6")
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_the_numpy_that_broke_the_voice_model_is_rejected(self):
        # 2.5 is the version the hosted log reported; the others are the same
        # boundary, so none of them may pass.
        for version in ("2.3.0", "2.4.2", "2.5.0", "2.5.1", "3.0.0"):
            with self.subTest(version=version):
                ok, reason = readiness.numpy_verdict(version)
                self.assertFalse(ok, f"{version} was accepted")
                self.assertIn(readiness.NUMPY_MAXIMUM, reason)

    def test_an_older_numpy_still_passes(self):
        for version in ("1.26.4", "2.0.2", "2.1.3", "2.2.0"):
            with self.subTest(version=version):
                self.assertTrue(readiness.numpy_verdict(version)[0])

    def test_a_missing_numpy_is_rejected(self):
        ok, reason = readiness.numpy_verdict(None)
        self.assertFalse(ok)
        self.assertIn("not installed", reason)

    def test_an_unimportable_numba_is_rejected(self):
        ok, reason = readiness.numpy_verdict(
            "2.2.6",
            numba_error="Numba needs NumPy 2.2 or less. Got NumPy 2.5.")
        self.assertFalse(ok)
        self.assertIn("numba", reason)

    def test_a_version_suffix_does_not_confuse_the_comparison(self):
        self.assertTrue(readiness.numpy_verdict("2.2.6+local")[0])
        self.assertFalse(readiness.numpy_verdict("2.3.0rc1")[0])


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_interpreter_version_file_and_numba(self):
        report = readiness.numpy_report()
        for key in ("executable", "version", "file", "numba", "maximum",
                    "compatible", "reason"):
            self.assertIn(key, report)
        self.assertTrue(report["executable"])
        self.assertEqual(report["maximum"], readiness.NUMPY_MAXIMUM)

    def test_compatibility_agrees_with_the_verdict(self):
        report = readiness.numpy_report()
        expected, _ = readiness.numpy_verdict(report["version"])
        self.assertEqual(report["compatible"], expected)


class StartupGateTests(unittest.TestCase):
    def test_an_incompatible_runtime_stops_the_start_and_names_the_paths(self):
        out = io.StringIO()
        with mock.patch.object(readiness, "numpy_report",
                               return_value=_report()), \
                mock.patch.dict(os.environ, {}, clear=False), \
                contextlib.redirect_stdout(out):
            os.environ.pop("VIDEOTRANS_ALLOW_INCOMPATIBLE_NUMPY", None)
            with self.assertRaises(RuntimeError) as caught:
                readiness.require_compatible_runtime()

        self.assertIn("2.5.1", str(caught.exception))
        printed = out.getvalue()
        # The whole point of the report: which interpreter, which file.
        self.assertIn("/usr/bin/python3", printed)
        self.assertIn("numpy/__init__.py", printed)
        self.assertIn("numpy==2.2.6", printed)

    def test_a_compatible_runtime_starts(self):
        with mock.patch.object(readiness, "numpy_report",
                               return_value=_report(compatible=True,
                                                    reason=None,
                                                    version="2.2.6")):
            report = readiness.require_compatible_runtime()
        self.assertTrue(report["compatible"])

    def test_the_documented_override_starts_with_a_warning(self):
        out = io.StringIO()
        with mock.patch.object(readiness, "numpy_report",
                               return_value=_report()), \
                mock.patch.dict(os.environ,
                                {"VIDEOTRANS_ALLOW_INCOMPATIBLE_NUMPY": "1"}), \
                contextlib.redirect_stdout(out):
            report = readiness.require_compatible_runtime()
        self.assertFalse(report["compatible"])
        self.assertIn("WARNING", out.getvalue())


class ReadinessGateTests(unittest.TestCase):
    """The payload may not call a feature ready that cannot load."""

    def test_the_voice_engine_is_not_ready_on_an_incompatible_numpy(self):
        fake_models = types.SimpleNamespace(is_cached=lambda repo: True)
        with mock.patch.object(readiness, "_has_module", return_value=True), \
                mock.patch.object(readiness, "numpy_report",
                                  return_value=_report()), \
                mock.patch.dict(sys.modules, {"models": fake_models}):
            entry = readiness.feature_indicf5(True, "hi")
        self.assertEqual(entry["status"], readiness.BLOCKED)
        self.assertIn("numba", entry["reason"])

    def test_lip_sync_is_not_ready_on_an_incompatible_numpy(self):
        fake_models = types.SimpleNamespace(is_cached=lambda repo: True,
                                            WEIGHTS=[])
        with mock.patch.object(readiness, "_has_module", return_value=True), \
                mock.patch.object(readiness, "numpy_report",
                                  return_value=_report()), \
                mock.patch.dict(sys.modules, {"models": fake_models}):
            entry = readiness.feature_lipsync()
        self.assertNotEqual(entry["status"], readiness.READY)
        self.assertIn("numba", entry["reason"])

    def test_the_snapshot_carries_the_runtime_report(self):
        snap = readiness.snapshot()
        self.assertIn("runtime", snap)
        self.assertIn("executable", snap["runtime"])
        self.assertIn("numpy", json_dumps(snap["runtime"]).lower())


def json_dumps(value) -> str:
    import json
    return json.dumps(value)


if __name__ == "__main__":
    unittest.main()

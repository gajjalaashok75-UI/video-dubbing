"""
The voice stage needs IndicF5's f5_tts fork, and nothing used to fetch it.

`engine/vendor/f5_tts` is not on PyPI and was never committed, so every fresh
clone ran with the pip `f5-tts` release installed and the guard in
`tts_engines._use_vendored_f5_tts` refused the voice stage:

    [BOOT] Loading voice model failed: f5_tts resolved to the pip release, not
    IndicF5's vendored fork ... Expected engine/vendor/f5_tts to be first on
    sys.path.

`scripts/vendor_f5_tts.py` closes that gap. Its marker check must stay identical
to the guard's, or setup can "succeed" while synthesis still refuses to run.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import vendor_f5_tts as vendor  # noqa: E402

DIT = Path("model") / "backbones" / "dit.py"


class MarkerTests(unittest.TestCase):
    def _tree(self, body: str | None) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        if body is not None:
            dit = root / DIT
            dit.parent.mkdir(parents=True)
            dit.write_text(body, encoding="utf-8")
        return root

    def test_fork_is_recognised(self):
        root = self._tree("class AdaLayerNormZero_Final:\n    pass\n")
        self.assertTrue(vendor.is_vendored(root))

    def test_pip_release_is_rejected(self):
        once = self._tree("class AdaLayerNorm_Final:\n    pass\n")
        self.assertFalse(vendor.is_vendored(once))

    def test_missing_dit_is_rejected(self):
        empty = self._tree(None)
        self.assertFalse(vendor.is_vendored(empty))

    def test_missing_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(vendor.is_vendored(Path(tmp) / "nope"))


class GuardAgreementTests(unittest.TestCase):
    def test_script_checks_the_marker_the_engine_guard_checks(self):
        """
        The guard decides with `hasattr(_dit, MARKER)`; setup must verify the
        same name, or a "successful" fetch still fails at render time.
        """
        source = (ROOT / "engine" / "tts_engines.py").read_text(encoding="utf-8")
        self.assertIn(f'"{vendor.MARKER}"', source,
                      "the vendoring script verifies a different marker than "
                      "_use_vendored_f5_tts does")


class CheckModeTests(unittest.TestCase):
    def test_check_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vendor, "VENDOR_DIR", Path(tmp) / "f5_tts"):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = vendor.main(["--check"])
                self.assertEqual(code, 1)
                self.assertIn("MISSING", out.getvalue())

    def test_check_reports_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "f5_tts"
            dit = root / DIT
            dit.parent.mkdir(parents=True)
            dit.write_text("class AdaLayerNormZero_Final:\n    pass\n", encoding="utf-8")
            with mock.patch.object(vendor, "VENDOR_DIR", root):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = vendor.main(["--check"])
                self.assertEqual(code, 0)
                self.assertIn("present", out.getvalue())

    def test_check_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "f5_tts"
            with mock.patch.object(vendor, "VENDOR_DIR", root):
                with contextlib.redirect_stdout(io.StringIO()):
                    vendor.main(["--check"])
            self.assertFalse(root.exists(), "--check must not create anything")

    def test_install_is_idempotent_when_already_vendored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "f5_tts"
            dit = root / DIT
            dit.parent.mkdir(parents=True)
            dit.write_text("class AdaLayerNormZero_Final:\n    pass\n", encoding="utf-8")
            with mock.patch.object(vendor, "VENDOR_DIR", root), \
                 mock.patch.object(vendor, "_clone") as clone:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(vendor.install())
            clone.assert_not_called()


if __name__ == "__main__":
    unittest.main()

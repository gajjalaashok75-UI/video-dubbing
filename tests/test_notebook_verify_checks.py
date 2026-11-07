"""The Verify cell's assertions must agree with the pins in requirements.txt.

The cell is a subprocess that fails the whole install when an assertion does not
hold, so an assertion that contradicts a pin blocks a session that is actually
correct - which is exactly what `av_major < 18` did while the pin allowed
`av<19` and 18.1.0 was the version pip installed.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "deploy" / "notebooks" / "build_notebooks.py"
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def verify_source() -> str:
    """The VERIFY cell's Python, read out of the generator.

    The checks live inside an `r\"\"\"...\"\"\"` block of the install cell, so the
    generator is read as text and the block between the delimiters is returned.
    """
    lines = GENERATOR.read_text(encoding="utf-8").splitlines()
    for start, line in enumerate(lines):
        if line.strip() == 'VERIFY = r"""':
            break
    else:
        raise AssertionError("VERIFY cell not found in the generator")
    for end in range(start + 1, len(lines)):
        if lines[end].strip() == '"""':
            return "\n".join(lines[start + 1:end])
    raise AssertionError("VERIFY cell is not closed")


def function_source(name: str) -> str:
    """Source of one `def` inside the VERIFY cell."""
    source = verify_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in the VERIFY cell")


def run_check(name: str, modules: dict) -> str:
    """Run one check function against stub modules; return its report string."""
    namespace: dict = {}
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update({k: v for k, v in modules.items()})
    try:
        exec(function_source(name), namespace)  # noqa: S102
        return namespace[name]()
    finally:
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous


def module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def whisper_modules(av_version: str) -> dict:
    return {
        "faster_whisper": module("faster_whisper", __version__="1.2.1"),
        "ctranslate2": module("ctranslate2", __version__="4.8.5"),
        "av": module("av", __version__=av_version),
    }


def numpy_modules(numpy_version: str) -> dict:
    return {
        "numpy": module("numpy", __version__=numpy_version),
        "numba": module("numba", __version__="0.64.0"),
    }


class FasterWhisperCheckTests(unittest.TestCase):
    def test_the_last_good_pyav_passes(self):
        # 18.1.0 is what `av>=11,<19` resolves to; it must not fail the cell.
        report = run_check("check_whisper", whisper_modules("18.1.0"))
        self.assertIn("av=18.1.0", report)

    def test_the_bad_pyav_still_fails(self):
        # 19.0.0 removed the `metadata_errors` kwarg faster-whisper passes.
        with self.assertRaises(AssertionError):
            run_check("check_whisper", whisper_modules("19.0.0"))

    def test_an_older_pyav_passes(self):
        run_check("check_whisper", whisper_modules("11.0.0"))


class NumpyCheckTests(unittest.TestCase):
    def test_the_pinned_version_passes(self):
        report = run_check("check_numpy", numpy_modules("2.2.6"))
        self.assertIn("numpy=2.2.6", report)

    def test_another_2_2_patch_passes(self):
        # The pin is `<2.3`, not `==2.2.6`, so the next patch must pass too.
        run_check("check_numpy", numpy_modules("2.2.9"))

    def test_the_numba_cap_still_fails(self):
        with self.assertRaises(AssertionError):
            run_check("check_numpy", numpy_modules("2.3.0"))

    def test_a_pre_release_of_the_cap_fails(self):
        with self.assertRaises(AssertionError):
            run_check("check_numpy", numpy_modules("2.3.0rc1"))

    def test_the_floor_still_fails(self):
        with self.assertRaises(AssertionError):
            run_check("check_numpy", numpy_modules("1.25.2"))


class PinAgreementTests(unittest.TestCase):
    def test_the_numpy_pin_matches_what_the_check_allows(self):
        text = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("numpy>=1.26.0,<2.3", text)

    def test_the_av_pin_matches_what_the_check_allows(self):
        text = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("av>=11,<19", text)


if __name__ == "__main__":
    unittest.main()

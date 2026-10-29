"""
The dependency contract whose absence killed every transcription.

A hosted run (Colab, 2026-10-28) resolved faster-whisper 1.2.1 alongside PyAV
19.0.2, and every render died before decoding a single frame:

    File ".../faster_whisper/audio.py", line 46, in decode_audio
        with av.open(input_file, mode="r", metadata_errors="ignore") as container:
    TypeError: open() got an unexpected keyword argument 'metadata_errors'

faster-whisper declares `av>=11` with no upper bound, and PyAV 19.0.0 removed
that argument. No code path can catch this - the import succeeds and the model
loads fine - so the ceiling itself is what has to be asserted.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIREMENTS = ROOT / "requirements.txt"
DOCKER_REQUIREMENTS = ROOT / "deploy" / "docker" / "requirements.docker.txt"

# PyAV 19.0.0 removed `metadata_errors`; 18.x still has it.
FIRST_BAD_PYAV = "19.0.0"
LAST_GOOD_PYAV = "18.1.0"

_AV_LINE = re.compile(r"^\s*av\s*([<>=!][^#\s]*)", re.MULTILINE)
_OPS = ("==", ">=", "<=", "!=", ">", "<")


def av_spec(text: str) -> str | None:
    """The version specifier this requirements file puts on `av`, if any."""
    match = _AV_LINE.search(text)
    return match.group(1) if match else None


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def satisfies(version: str, spec: str) -> bool:
    """Does `version` meet a comma-separated specifier? Only the ops used here."""
    got = _version(version)
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for op in _OPS:
            if chunk.startswith(op):
                want = _version(chunk[len(op):])
                width = max(len(got), len(want))
                a = got + (0,) * (width - len(got))
                b = want + (0,) * (width - len(want))
                if not {
                    "==": a == b, ">=": a >= b, "<=": a <= b,
                    "!=": a != b, ">": a > b, "<": a < b,
                }[op]:
                    return False
                break
        else:
            raise AssertionError(f"unsupported version specifier: {chunk!r}")
    return True


class OptionalExtraTests(unittest.TestCase):
    """
    The optional, gated extras must stay optional and must stay in step.

    pyannote is the multi-speaker extra. It is heavy and gated, so it is
    commented out of both files: an install that cannot reach Hugging Face, or
    that does not want the weight, still gets a working single-speaker
    pipeline. What must not happen is the two files disagreeing about it, or a
    stray uncommented line quietly making every install pay for it.
    """

    def _lines(self, path):
        return path.read_text(encoding="utf-8").splitlines()

    def test_pyannote_is_not_installed_by_default(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                for line in self._lines(path):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    self.assertFalse(
                        stripped.startswith("pyannote"),
                        f"{path} installs pyannote unconditionally; it is a "
                        "gated, heavy extra and must stay commented out")

    def test_both_files_mention_pyannote_the_same_way(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("pyannote.audio", text,
                              f"{path} should document how to enable the "
                              "multi-speaker extra")

    def test_demucs_is_not_installed_by_default(self):
        # Phase 4's extra. Same rule as pyannote: heavy, optional, and off
        # unless someone asks for it.
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                for line in self._lines(path):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    self.assertFalse(
                        stripped.startswith("demucs"),
                        f"{path} installs demucs unconditionally")

    def test_both_files_document_the_background_extra(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("demucs>=4.0.0", text,
                              f"{path} should say how to enable background "
                              "preservation")

    def test_the_gated_model_is_documented_where_the_licence_is(self):
        # The token and the licence URL belong next to the pin, so the person
        # reading requirements.txt knows what to accept before installing.
        text = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("speaker-diarization", text)
        self.assertIn("HF_TOKEN", text)


class AvPinTests(unittest.TestCase):
    def test_both_requirement_files_pin_av(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                spec = av_spec(path.read_text(encoding="utf-8"))
                self.assertIsNotNone(
                    spec, f"{path} does not pin av; faster-whisper's bare `av>=11` "
                          "resolves to a release that dropped metadata_errors")

    def test_pin_excludes_the_pyav_that_broke_decode_audio(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                spec = av_spec(path.read_text(encoding="utf-8"))
                self.assertFalse(
                    satisfies(FIRST_BAD_PYAV, spec),
                    f"av spec {spec!r} allows PyAV {FIRST_BAD_PYAV}, which removed "
                    "the metadata_errors argument faster-whisper passes")

    def test_pin_still_allows_a_pyav_that_has_the_argument(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                spec = av_spec(path.read_text(encoding="utf-8"))
                self.assertTrue(
                    satisfies(LAST_GOOD_PYAV, spec),
                    f"av spec {spec!r} rejects PyAV {LAST_GOOD_PYAV}, which still "
                    "accepts metadata_errors")

    def test_pin_meets_faster_whispers_own_requirement(self):
        # faster-whisper asks for `av>=11`; the ceiling must not contradict it.
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                spec = av_spec(text)
                self.assertTrue(satisfies(LAST_GOOD_PYAV, ">=11"))
                self.assertTrue(satisfies(LAST_GOOD_PYAV, spec))

    def test_requirement_files_agree(self):
        self.assertEqual(
            av_spec(REQUIREMENTS.read_text(encoding="utf-8")),
            av_spec(DOCKER_REQUIREMENTS.read_text(encoding="utf-8")),
            "the av pin drifted between requirements.txt and the Docker copy",
        )

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV is not installed here")
    def test_installed_pyav_accepts_metadata_errors(self):
        """Behavioural check: the installed PyAV must still take the argument."""
        import av  # noqa: PLC0415 - only when installed

        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "not-a-real.wav")
            try:
                av.open(missing, mode="r", metadata_errors="ignore")
            except TypeError as e:
                self.fail(f"PyAV {av.__version__} rejects metadata_errors: {e}")
            except Exception:  # noqa: BLE001 - accepted the kwarg, then failed on the file
                pass


if __name__ == "__main__":
    unittest.main()

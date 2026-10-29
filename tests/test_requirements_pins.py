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


def _bare_line(text: str, package: str) -> str | None:
    """The uncommented line for `package`, with or without a version spec."""
    pattern = re.compile(rf"^\s*{re.escape(package)}\s*([<>=!].*)?$",
                         re.MULTILINE)
    match = pattern.search(text)
    return match.group(0) if match else None


def spec_for(text: str, package: str) -> str | None:
    """The version specifier on an uncommented `package` line, if any."""
    pattern = re.compile(rf"^\s*{re.escape(package)}\s*([<>=!][^#\s]*)",
                         re.MULTILINE)
    match = pattern.search(text)
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


class NumpyPinTests(unittest.TestCase):
    """
    Numba's NumPy ceiling, which no code path can catch.

    A hosted run (Colab, 2026-10-29) had NumPy 2.5 alongside numba 0.61, which
    librosa pulls in for audio loading. The voice model then failed to LOAD:

        [BOOT] Loading voice model failed: Numba needs NumPy 2.2 or less.
        Got NumPy 2.5.

    and Wav2Lip's warmup failed with the same message. The package that has to
    be held back is NumPy itself, so the ceiling is what is asserted.
    """

    FIRST_BAD = "2.3"
    LAST_GOOD = "2.2.6"

    def test_both_files_cap_numpy(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                spec = spec_for(path.read_text(encoding="utf-8"), "numpy")
                self.assertIsNotNone(
                    spec, f"{path} does not pin numpy; a fresh install takes "
                          "the newest release and numba refuses it")

    def test_the_cap_excludes_the_numpy_that_broke_the_voice_model(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                spec = spec_for(path.read_text(encoding="utf-8"), "numpy")
                self.assertFalse(satisfies(self.FIRST_BAD, spec),
                                 f"numpy spec {spec!r} allows 2.3+, which "
                                 "numba refuses")

    def test_the_cap_still_allows_a_numpy_numba_accepts(self):
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                spec = spec_for(path.read_text(encoding="utf-8"), "numpy")
                self.assertTrue(satisfies(self.LAST_GOOD, spec),
                                f"numpy spec {spec!r} rejects {self.LAST_GOOD}")

    def test_the_two_files_agree(self):
        self.assertEqual(
            spec_for(REQUIREMENTS.read_text(encoding="utf-8"), "numpy"),
            spec_for(DOCKER_REQUIREMENTS.read_text(encoding="utf-8"), "numpy"),
            "the numpy pin drifted between requirements.txt and the Docker copy")


class TorchPinTests(unittest.TestCase):
    """
    The CUDA-12 torch line, which CTranslate2's libcublas.so.12 depends on.

    A hosted run had torch 2.11.0+cu130 installed, which satisfies a bare
    `torch>=2.2.0`, so the cu124 install cell left it alone. CTranslate2 then
    died on the first encode with "Library libcublas.so.12 is not found or
    cannot be loaded" and every render failed ~6 seconds in.
    """

    FIRST_BAD = "2.7"
    LAST_GOOD = "2.6.0"

    def test_numpy_is_not_mentioned_before_its_pin(self):
        # Guard against a stray extra `numpy` line that would re-open the range.
        for path in (REQUIREMENTS, DOCKER_REQUIREMENTS):
            with self.subTest(path=path.name):
                lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                         if ln.strip().startswith("numpy")]
                self.assertEqual(len(lines), 1, f"{path} names numpy {len(lines)} "
                                 "times; one pin only")

    def test_the_torch_cap_excludes_the_cuda_13_wheels(self):
        spec = spec_for(REQUIREMENTS.read_text(encoding="utf-8"), "torch")
        self.assertIsNotNone(spec)
        self.assertFalse(satisfies(self.FIRST_BAD, spec),
                         f"torch spec {spec!r} allows 2.7+, whose CUDA 13 "
                         "cuBLAS is libcublas.so.13")

    def test_the_torch_cap_allows_the_last_cu124_release(self):
        spec = spec_for(REQUIREMENTS.read_text(encoding="utf-8"), "torch")
        self.assertTrue(satisfies(self.LAST_GOOD, spec),
                        f"torch spec {spec!r} rejects {self.LAST_GOOD}, the "
                        "last build published for cu124")

    def test_the_companion_packages_share_the_cap(self):
        # torchaudio/torchvision are versioned with torch; a mismatch here is
        # its own pip error at install time.
        text = REQUIREMENTS.read_text(encoding="utf-8")
        for package in ("torchaudio", "torchvision"):
            with self.subTest(package=package):
                self.assertIsNotNone(spec_for(text, package),
                                     f"{package} needs a ceiling too")

    def test_the_docker_file_keeps_torch_out(self):
        # The Dockerfile installs torch from the cu124 index in its own layer;
        # naming it here would pull the wrong wheels from PyPI.
        text = DOCKER_REQUIREMENTS.read_text(encoding="utf-8")
        for package in ("torch", "torchaudio", "torchvision"):
            with self.subTest(package=package):
                self.assertIsNone(
                    _bare_line(text, package),
                    f"requirements.docker.txt must not install {package}; "
                    "the Dockerfile does it from the cu124 index")

    def test_the_dockerfile_pins_the_cu124_build(self):
        dockerfile = (ROOT / "deploy" / "docker" / "Dockerfile").read_text(
            encoding="utf-8")
        self.assertIn("torch==2.6.0+cu124", dockerfile,
                      "the Dockerfile must pin the CUDA 12 build, not a range")
        self.assertIn("torch.version.cuda", dockerfile,
                      "the build should assert the CUDA major it got")


class NotebookTorchTests(unittest.TestCase):
    """The Colab/Kaggle install cell must force the CUDA 12 build."""

    def setUp(self):
        self.generator = (ROOT / "deploy" / "notebooks" /
                          "build_notebooks.py").read_text(encoding="utf-8")

    def test_the_install_cell_pins_the_cu124_build(self):
        self.assertIn("torch==2.6.0+cu124", self.generator)

    def test_the_install_cell_forces_the_install(self):
        # A hosted image already has a newer torch; a plain `pip install`
        # considers the requirement satisfied and leaves the wrong CUDA level
        # in place, which is exactly how the broken run happened.
        self.assertIn("--force-reinstall", self.generator)

    def test_the_install_cell_checks_the_cuda_major(self):
        self.assertIn("torch.version.cuda", self.generator)

    def test_the_install_cell_re_asserts_the_numpy_cap(self):
        self.assertIn("numpy>=1.26.0,<2.3", self.generator)


if __name__ == "__main__":
    unittest.main()

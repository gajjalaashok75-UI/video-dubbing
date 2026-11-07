"""
The optional-engine installs must not be allowed to move torch or NumPy.

A hosted run (Colab, 2026-11) had the CUDA 12.4 torch stack in place and then
installed the multi-speaker engine:

    pip install pyannote.audio==3.3.2

pyannote pulls `speechbrain` and `pytorch-metric-learning`, and the versions
already on the image asked for a NEWER torch, so the resolver replaced the
CUDA 12.4 build with a CUDA 13 one. Two things broke at once:

    ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume
    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

The CUDA 13 wheels left their `nvidia-*-cu13` twins beside the CUDA 12 ones, so
`import torch` resolved a symbol from the wrong NCCL, and CTranslate2 could no
longer find the cuBLAS it was built against. The torch probe then failed on
every subsequent run, the install cell reinstalled torch alone (which cannot
clear the stray `-cu13` wheels), and the notebook looped forever asking for a
restart.

Two fixes are pinned here: the `-c deploy/constraints.txt` flag on every install
that could move torch, and the loop-breaker in the install cell.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import deploy_helpers as runtime  # noqa: E402

CONSTRAINTS = ROOT / "deploy" / "constraints.txt"
GENERATOR = ROOT / "deploy" / "notebooks" / "build_notebooks.py"
NOTEBOOKS = (ROOT / "deploy" / "notebooks" / "colab.ipynb",
             ROOT / "deploy" / "notebooks" / "kaggle.ipynb")


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


class ConstraintsFileTests(unittest.TestCase):
    """`deploy/constraints.txt` holds the two packages a resolver may not move."""

    def setUp(self):
        self.text = CONSTRAINTS.read_text(encoding="utf-8")

    def test_the_file_exists_where_the_notebook_looks_for_it(self):
        self.assertTrue(CONSTRAINTS.is_file(),
                        "the notebook passes -c deploy/constraints.txt; a "
                        "missing file makes every optional install fail")
        self.assertEqual(runtime.constraints_path(ROOT), CONSTRAINTS)

    def test_torch_is_pinned_to_the_cu124_release(self):
        # `==2.6.0` matches the `2.6.0+cu124` local build (PEP 440 ignores the
        # local segment when the specifier has none) and excludes the CUDA 13
        # wheels whose cuBLAS is libcublas.so.13.
        self.assertIn("torch==2.6.0", self.text)
        self.assertIn("torchaudio==2.6.0", self.text)
        self.assertIn("torchvision==0.21.0", self.text)

    def test_numpy_stays_under_numbas_cap(self):
        self.assertIn("numpy>=1.26.0,<2.3", self.text)

    def test_the_hub_pair_is_held_together(self):
        self.assertIn("transformers==4.57.6", self.text)
        self.assertIn("huggingface-hub==0.36.2", self.text)


class NotebookConstraintsTests(unittest.TestCase):
    """Every install that could move torch passes the constraints file."""

    def setUp(self):
        self.generator = GENERATOR.read_text(encoding="utf-8")

    def test_the_generator_defines_the_constraints_path(self):
        self.assertIn('os.path.join(REPO_DIR, "deploy", "constraints.txt")',
                      self.generator)

    def test_the_requirements_install_is_constrained(self):
        self.assertIn('"-c",\n        CONSTRAINTS,\n        "-r",\n'
                      '        "requirements.txt",', self.generator)

    def test_the_pyannote_install_is_constrained(self):
        # The exact install whose chain moved torch.
        self.assertIn('"-c",\n        CONSTRAINTS,\n        PYANNOTE,',
                      self.generator)

    def test_the_demucs_install_is_constrained(self):
        self.assertIn('"-c",\n        CONSTRAINTS,\n        "--no-deps",',
                      self.generator)

    def test_both_notebooks_carry_the_constraints_flag(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source = notebook_source(path)
                self.assertIn("CONSTRAINTS", source)
                self.assertGreaterEqual(source.count('"-c"'), 3,
                                        "requirements, pyannote and demucs "
                                        "installs all need the flag")


class TorchLoopBreakerTests(unittest.TestCase):
    """The install cell must not ask for a restart it cannot make useful."""

    def setUp(self):
        self.generator = GENERATOR.read_text(encoding="utf-8")

    def test_stray_cuda13_wheels_are_uninstalled(self):
        self.assertIn("nvidia-nccl-cu13", self.generator)
        self.assertIn("Removing the CUDA 13 wheels", self.generator)

    def test_the_torch_install_is_re_probed(self):
        # A replacement that did not take must surface the real error, not
        # another restart prompt.
        self.assertIn("after = subprocess.run", self.generator)
        self.assertIn("still does not import after a forced", self.generator)

    def test_the_restart_is_gated_on_a_stale_kernel(self):
        # The old cell raised SystemExit unconditionally, so a run that
        # replaced torch looped forever. Now it restarts only when the kernel
        # actually holds the old build.
        self.assertIn("stale_torch", self.generator)
        self.assertIn("runtime.stale_modules()", self.generator)

    def test_both_notebooks_break_the_loop(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source = notebook_source(path)
                self.assertIn("stale_torch", source)
                self.assertIn("nvidia-nccl-cu13", source)


class StrayCuda13Tests(unittest.TestCase):
    """`stray_cuda13_packages` names the wheels that break `import torch`."""
    def test_nothing_is_reported_on_a_clean_environment(self):
        with mock.patch("importlib.metadata.version", side_effect=ImportError):
            self.assertEqual(runtime.stray_cuda13_packages(), [])

    def test_the_cuda13_nccl_is_reported(self):
        real = __import__("importlib.metadata", fromlist=["version"]).version

        def version(name):
            if name == "nvidia-nccl-cu13":
                return "2.29.7"
            raise __import__("importlib.metadata",
                             fromlist=["PackageNotFoundError"]
                             ).PackageNotFoundError(name)

        with mock.patch("importlib.metadata.version", side_effect=version):
            self.assertEqual(runtime.stray_cuda13_packages(),
                             ["nvidia-nccl-cu13"])

    def test_the_cuda12_wheels_are_not_reported(self):
        def version(name):
            raise __import__("importlib.metadata",
                             fromlist=["PackageNotFoundError"]
                             ).PackageNotFoundError(name)

        with mock.patch("importlib.metadata.version", side_effect=version):
            self.assertEqual(runtime.stray_cuda13_packages(), [])


class CheckoutCellTests(unittest.TestCase):
    """A zip checkout must survive the first cell, which used to re-clone over it."""

    def setUp(self):
        self.generator = GENERATOR.read_text(encoding="utf-8")

    def test_a_checkout_is_recognised_by_requirements_txt(self):
        # git archive (the delivered zip) has no `.git`, so keying on `.git`
        # made the cell clone the remote over the user's copy.
        self.assertIn("def _is_checkout(path):", self.generator)
        self.assertIn('os.path.isfile(os.path.join(path, "requirements.txt"))',
                      self.generator)
        self.assertIn("HAVE_CODE = _is_checkout(REPO_DIR)", self.generator)

    def test_an_unzipped_copy_away_from_repo_dir_is_still_found(self):
        # The user may unzip into a differently named folder; walking up from
        # the working directory finds it before the clone fallback.
        self.assertIn("while True:", self.generator)
        self.assertIn("REPO_DIR = here", self.generator)

    def test_an_existing_checkout_is_not_re_cloned(self):
        self.assertIn("Using the repository already present", self.generator)

    def test_a_non_empty_clone_target_is_cleared(self):
        # `git clone` refuses a non-empty directory; a half-finished run left
        # one behind and the clone failed outright.
        self.assertIn("shutil.rmtree(REPO_DIR)", self.generator)

    def test_both_notebooks_use_an_existing_checkout(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source = notebook_source(path)
                self.assertIn("HAVE_CODE", source)
                self.assertIn("requirements.txt", source)


if __name__ == "__main__":
    unittest.main()

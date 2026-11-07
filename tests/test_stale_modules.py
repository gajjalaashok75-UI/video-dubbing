"""
A notebook kernel must not start the server on a package it imported before the
install replaced it.

A hosted session verified `numpy==2.2.6` in the install cell's subprocess and
then started a server that reported NumPy 2.5:

    Runtime check failed: numba needs NumPy 2.2 or less and this interpreter
    has 2.4.6 ... numpy: 2.4.6 (/usr/local/lib/python3.13/dist-packages/...)

The install cell ran pip in a subprocess, so its probe saw the new files; the
kernel had already imported the image's NumPy (the Check cell calls
`hardware_report` and `readiness.snapshot`, both of which import torch and
numpy) and kept the old module object in `sys.modules`. Python cannot swap an
imported extension module, so the server - a thread of that kernel - inherited
the stale NumPy and numba refused it.

These tests pin the guard that catches it, and the two notebook changes that
make the clean path work: the Check cell probes in a subprocess, and the install
cell refuses to continue on a stale kernel.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import deploy_helpers as runtime  # noqa: E402

NOTEBOOKS = (ROOT / "deploy" / "notebooks" / "colab.ipynb",
             ROOT / "deploy" / "notebooks" / "kaggle.ipynb")


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def _fake_module(version: str) -> types.ModuleType:
    module = types.ModuleType("fake")
    module.__version__ = version
    return module


class StaleModuleTests(unittest.TestCase):
    def test_an_unimported_package_is_not_stale(self):
        # Nothing in sys.modules for it, so there is nothing the kernel can
        # hand a later importer.
        with mock.patch.object(runtime, "_WATCHED", ("numpy",)):
            self.assertEqual(runtime.stale_modules(), {})

    def test_a_module_matching_its_distribution_is_not_stale(self):
        with mock.patch.object(runtime, "_WATCHED", ("numpy",)), \
                mock.patch.dict(sys.modules, {"numpy": _fake_module("2.2.6")}), \
                mock.patch.object(importlib.metadata, "version",
                                  return_value="2.2.6"):
            self.assertEqual(runtime.stale_modules(), {})

    def test_the_imported_old_version_is_reported(self):
        # The hosted failure: kernel holds 2.5.4, disk now has 2.2.6.
        with mock.patch.object(runtime, "_WATCHED", ("numpy",)), \
                mock.patch.dict(sys.modules, {"numpy": _fake_module("2.5.4")}), \
                mock.patch.object(importlib.metadata, "version",
                                  return_value="2.2.6"):
            stale = runtime.stale_modules()
        self.assertEqual(stale, {"numpy": ("2.5.4", "2.2.6")})

    def test_torch_is_watched_too(self):
        with mock.patch.object(runtime, "_WATCHED", ("torch",)), \
                mock.patch.dict(sys.modules,
                                {"torch": _fake_module("2.11.0+cu130")}), \
                mock.patch.object(importlib.metadata, "version",
                                  return_value="2.6.0"):
            self.assertIn("torch", runtime.stale_modules())

    def test_a_module_with_no_distribution_is_ignored(self):
        # A vendored or path-injected module has no metadata; it cannot be
        # compared, and it must not be reported as stale.
        with mock.patch.object(runtime, "_WATCHED", ("torch",)), \
                mock.patch.dict(sys.modules, {"torch": _fake_module("1.0")}), \
                mock.patch.object(
                    importlib.metadata, "version",
                    side_effect=importlib.metadata.PackageNotFoundError):
            self.assertEqual(runtime.stale_modules(), {})

    def test_a_module_without_a_version_attribute_is_ignored(self):
        module = types.ModuleType("numpy")  # no __version__
        with mock.patch.object(runtime, "_WATCHED", ("numpy",)), \
                mock.patch.dict(sys.modules, {"numpy": module}), \
                mock.patch.object(importlib.metadata, "version",
                                  return_value="2.2.6"):
            self.assertEqual(runtime.stale_modules(), {})


class NotebookGuardTests(unittest.TestCase):
    def test_the_check_cell_probes_in_a_subprocess(self):
        # Importing torch/numpy in the kernel is what poisons it; the probe
        # must run in a child process so the kernel stays clean.
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn('subprocess.run([sys.executable, "-c", PROBE]', src,
                              "the machine check must run out of process, or the "
                              "kernel keeps the image's NumPy before the install")

    def test_the_install_cell_guards_against_a_stale_kernel(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("stale_modules()", src)
                self.assertIn("RESTART REQUIRED", src)

    def test_the_troubleshooting_table_names_the_restart(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                self.assertIn("RESTART REQUIRED", notebook_source(path))


if __name__ == "__main__":
    unittest.main()

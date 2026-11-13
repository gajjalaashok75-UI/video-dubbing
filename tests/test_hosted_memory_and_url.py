"""
The hosted-runtime fixes: memory, the S2 Pro install gate, and the public URL.

A Colab session ran every cell, printed a working URL, and then died minutes
later with "your session crashed after using all available RAM". The growth was
monotonic - 0.1, 4.6, 9.6, ~12.5 GB - which is a load, not a leak: the four
models boot at once, each allocating large scratch buffers that are never
released, and the caching allocator keeps every freed block reserved. The
12.5 GB line is S2 Pro's ~11 GB of weights landing on top of a pipeline that was
already resident.

Four things are pinned here:

* the boot sequence releases memory between stages and asks torch to hand
  freed blocks back (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`);
* the S2 Pro install is opt-in, so a default Colab/Kaggle run never pulls the
  ~11 GB model that the machine cannot hold;
* the notebook repairs the protobuf that `audiotools` pins stale, since that
  pin is what broke `import` after an otherwise successful install;
* the public-URL cell falls back to a Cloudflare tunnel, so Kaggle - which has
  no kernel proxy - still gets a browser-reachable URL, and Colab's proxy
  failing no longer prints an unreachable 127.0.0.1 link.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import deploy_helpers as runtime  # noqa: E402

GENERATOR = ROOT / "deploy" / "notebooks" / "build_notebooks.py"
NOTEBOOKS = (ROOT / "deploy" / "notebooks" / "colab.ipynb",
             ROOT / "deploy" / "notebooks" / "kaggle.ipynb")


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


class MemoryHygieneTests(unittest.TestCase):
    """Boot must return scratch memory before the next model loads."""

    def setUp(self):
        self.generator = GENERATOR.read_text(encoding="utf-8")

    def test_boot_releases_memory_after_every_stage(self):
        # The `finally` matters: a stage that fails still allocated, and the
        # next stage would otherwise start on top of it.
        app = (ROOT / "engine" / "app.py").read_text(encoding="utf-8")
        self.assertIn("def _release_memory():", app)
        self.assertIn("_release_memory()", app)
        self.assertIn("finally:", app)
        # Both halves: the Python heap and torch's caching allocator.
        self.assertIn("gc.collect()", app)
        self.assertIn("torch.cuda.empty_cache()", app)

    def test_the_synth_loop_collects_periodically(self):
        # The synth loop is the longest-running part of a render; without a
        # periodic collection the Python-side footprint only grows.
        dubbing = (ROOT / "engine" / "dubbing.py").read_text(encoding="utf-8")
        self.assertIn("gc.collect()", dubbing)
        self.assertIn("_RELEASE_EVERY_UNITS", dubbing)

    def test_the_allocator_is_told_to_release_blocks(self):
        # Without this, `empty_cache` cannot return the load-time peaks and the
        # reserved total climbs to the sum of every stage's high-water mark.
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {}, clear=True):
            runtime.configure_runtime(tmp)
            self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                             "expandable_segments:True")

    def test_an_operator_setting_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ,
                                {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"},
                                clear=True):
            runtime.configure_runtime(tmp)
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                             "max_split_size_mb:64")


class S2ProOptInTests(unittest.TestCase):
    """A default run must not install or load the ~11 GB S2 Pro model."""

    def setUp(self):
        self.generator = GENERATOR.read_text(encoding="utf-8")

    def test_the_install_is_behind_a_flag(self):
        self.assertIn("VIDEOTRANS_INSTALL_S2PRO", self.generator)
        # The heavy install must be conditional, not run for every user.
        self.assertIn("if INSTALL_S2PRO:", self.generator)

    def test_the_flag_defaults_to_off(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn(
                    'os.environ.get("VIDEOTRANS_INSTALL_S2PRO", "")', src,
                    "a default run must not install S2 Pro")

    def test_the_notebook_says_how_to_opt_in(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("VIDEOTRANS_INSTALL_S2PRO", src)
                self.assertIn("~11 GB", src)

    def test_the_house_engine_is_the_default(self):
        # S2 Pro is not the default voice engine; a default run uses IndicF5 /
        # XTTS, which are the models the RAM budget is sized for.
        languages = (ROOT / "engine" / "languages.py").read_text(encoding="utf-8")
        self.assertIn("def engine_for(", languages)
        self.assertNotIn("_engine_ready", languages)

    def test_the_model_cell_fetches_extras_only_when_opted_in(self):
        # The 11 GB of S2 Pro weights must not land on a default volume; the
        # model cell passes --extras only when the install cell was opted in.
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("--extras", src)
                self.assertIn(
                    'os.environ.get("VIDEOTRANS_INSTALL_S2PRO", "")', src)


class ProtobufRepairTests(unittest.TestCase):
    """The install repairs the protobuf that `audiotools` pins stale."""

    def setUp(self):
        self.generator = GENERATOR.read_text(encoding="utf-8")

    def test_the_notebook_forces_a_protobuf_that_can_import(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("protobuf==5.29.5", src)
                self.assertIn("--no-deps", src)

    def test_the_repair_is_not_undone_by_a_later_install(self):
        # The repair has to be the last word on protobuf; a plain install after
        # it would drag 3.19 back in. Pin that it is the audiotools repair that
        # carries `--no-deps`.
        self.assertIn("audiotools", self.generator)
        self.assertIn("protobuf", self.generator)


class PublicUrlTests(unittest.TestCase):
    """Both platforms must end with a browser-reachable URL."""

    def setUp(self):
        self.generator = GENERATOR.read_text(encoding="utf-8")

    def tearDown(self):
        # `cloudflared_public_url` records the host in the environment and the
        # process in a module global; leaving either behind changes what a later
        # test's `notebook_public_url` returns.
        os.environ.pop("PUBLIC_HOST", None)
        runtime._TUNNEL_PROC = None

    def test_both_notebooks_share_the_url_cell(self):
        # The tunnel used to be Kaggle-only; Colab users whose proxy failed were
        # told to open 127.0.0.1. One cell now serves both.
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("runtime.cloudflared_public_url(", src)

    def test_the_stop_cell_ends_the_tunnel(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("runtime.stop_tunnel()", src)

    def test_the_helper_returns_none_instead_of_raising(self):
        # A missing wget, a dead binary, a timeout: all must return None so the
        # cell can fall back to the loopback URL and say so.
        with mock.patch("subprocess.run",
                        side_effect=FileNotFoundError("wget")):
            self.assertIsNone(runtime.cloudflared_public_url(8000))

    def test_the_helper_reads_the_url_out_of_the_log(self):
        class FakeProc:
            def poll(self):
                return None

            def terminate(self):
                pass

        log_text = ("INF Registered tunnel connection\n"
                    "INF https://snowy-otter-1234.trycloudflare.com\n")

        def fake_popen(argv, stdout=None, stderr=None):
            stdout.write(log_text)
            stdout.flush()
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            # Pre-place the binary so the helper skips the download and goes
            # straight to spawning the tunnel.
            tools = Path(tmp) / "tools"
            tools.mkdir()
            (tools / "cloudflared").write_text("#!/bin/sh\n")
            with mock.patch.object(runtime, "runtime_dirs",
                                   return_value={"logs": Path(tmp) / "logs"}), \
                    mock.patch.object(runtime.subprocess, "Popen",
                                      side_effect=fake_popen), \
                    mock.patch("time.sleep"):
                url = runtime.cloudflared_public_url(8000, timeout=5)
        self.assertEqual(url, "https://snowy-otter-1234.trycloudflare.com")

    def test_stop_tunnel_reports_whether_one_ran(self):
        self.assertFalse(runtime.stop_tunnel())


if __name__ == "__main__":
    unittest.main()

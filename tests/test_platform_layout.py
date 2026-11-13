"""
The checkout and the runtime tree must agree on what platform this is.

`detect_platform()` returns "colab" for anything with a /content mount, checked
before /kaggle. The notebook's repo-dir expression tested /kaggle first, so a
Colab instance that also had a /kaggle directory cloned the checkout to
/kaggle/working/video-dubbing while the runtime tree - uploads, renders, and the
server log the user is told to read - went to /content/videotrans-runtime:

    File "/kaggle/working/video-dubbing/engine/app.py", line 429, in process_video_task
    Server log: /content/videotrans-runtime/logs/server.log

Two volumes for one session. These tests pin the rule, including the generated
notebooks, since those are what actually run.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "deploy" / "notebooks"))

import build_notebooks  # noqa: E402
import deploy_helpers as runtime  # noqa: E402

NOTEBOOKS = (ROOT / "deploy" / "notebooks" / "colab.ipynb",
             ROOT / "deploy" / "notebooks" / "kaggle.ipynb")


class _FakePath:
    def __init__(self, dirs: set[str]) -> None:
        self._dirs = dirs

    def isdir(self, path: str) -> bool:
        return path in self._dirs


class _FakeOs:
    """Just enough of `os` to evaluate the notebook's expression."""

    def __init__(self, dirs: set[str], cwd: str = "/somewhere",
                 env: dict | None = None) -> None:
        self.path = _FakePath(dirs)
        self._cwd = cwd
        self.environ = dict(env or {})

    def getcwd(self) -> str:
        return self._cwd


def repo_dir_from_notebook(dirs: set[str], env: dict | None = None) -> str:
    return eval(build_notebooks.REPO_DIR_EXPR,          # noqa: S307
                {"os": _FakeOs(dirs, env=env)})


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


class RepoDirRuleTests(unittest.TestCase):
    def test_repo_and_runtime_tree_share_a_parent(self):
        for platform in ("colab", "kaggle"):
            with self.subTest(platform=platform):
                self.assertEqual(
                    runtime.repo_dir(platform).parent,
                    runtime.default_runtime_root(platform).parent,
                    f"on {platform} the checkout and the runtime tree are on "
                    "different volumes",
                )

    def test_notebook_rule_picks_kaggle_when_both_directories_exist(self):
        # The regression: a Kaggle image also mounts /content, so the two
        # markers are not mutually exclusive and order decides. Kaggle is tested
        # first, matching `detect_platform`, so the checkout and the runtime tree
        # land on the same volume.
        self.assertEqual(repo_dir_from_notebook({"/content", "/kaggle"}),
                         "/kaggle/working/video-dubbing")

    def test_the_platform_environment_variables_decide(self):
        # The marker each platform actually sets is definitive; the directories
        # are only a fallback for an image that sets neither.
        self.assertEqual(
            repo_dir_from_notebook(set(), {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}),
            "/kaggle/working/video-dubbing")
        self.assertEqual(
            repo_dir_from_notebook(set(), {"COLAB_GPU": "1"}),
            "/content/video-dubbing")

    def test_notebook_rule_on_each_platform(self):
        self.assertEqual(repo_dir_from_notebook({"/content"}), "/content/video-dubbing")
        self.assertEqual(repo_dir_from_notebook({"/kaggle"}),
                         "/kaggle/working/video-dubbing")

    def test_notebook_rule_falls_back_to_the_working_directory(self):
        self.assertEqual(repo_dir_from_notebook(set()), "/somewhere")

    def test_notebook_rule_agrees_with_repo_dir(self):
        # as_posix(): the notebooks run on Linux, where the expression is
        # evaluated, so compare the path text it would produce there.
        pairs = (("colab", {"/content"}), ("kaggle", {"/kaggle"}))
        for platform, dirs in pairs:
            with self.subTest(platform=platform):
                self.assertEqual(repo_dir_from_notebook(dirs),
                                 runtime.repo_dir(platform).as_posix())

    def test_notebook_rule_and_detect_platform_agree_when_both_exist(self):
        # The two must not disagree: the bug was detect_platform saying "colab"
        # (runtime tree under /content) while the checkout went to /kaggle.
        for dirs in ({"/content", "/kaggle"}, {"/content"}, {"/kaggle"}, set()):
            with self.subTest(dirs=dirs):
                chosen = repo_dir_from_notebook(dirs)
                fake = _FakeOs(dirs)
                if fake.environ.get("KAGGLE_KERNEL_RUN_TYPE") or "/kaggle" in dirs:
                    platform = "kaggle"
                elif fake.environ.get("COLAB_GPU") or "/content" in dirs:
                    platform = "colab"
                else:
                    platform = None
                if platform is not None:
                    self.assertEqual(chosen, runtime.repo_dir(platform).as_posix())


class DetectPlatformTests(unittest.TestCase):
    """Kaggle must win over Colab when both markers are present.

    A Kaggle image also mounts /content, so a `/content`-first test called a
    Kaggle session "colab": the runtime tree went to /content/videotrans-runtime
    while the checkout was under /kaggle/working. The environment variables come
    first because they are definitive; the directories are only a fallback.
    """

    def _detect(self, dirs, env):
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("os.path.isdir", lambda p: p in dirs), \
                mock.patch("os.path.exists", lambda p: p == "/.dockerenv"
                           and "/.dockerenv" in dirs):
            return runtime.detect_platform()

    def test_kaggle_env_var_beats_a_content_mount(self):
        self.assertEqual(
            self._detect({"/content", "/kaggle"},
                         {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}),
            "kaggle")

    def test_kaggle_directory_beats_a_content_mount(self):
        self.assertEqual(self._detect({"/content", "/kaggle"}, {}), "kaggle")

    def test_colab_env_var(self):
        self.assertEqual(self._detect(set(), {"COLAB_GPU": "1"}), "colab")

    def test_colab_directory(self):
        self.assertEqual(self._detect({"/content"}, {}), "colab")

    def test_docker(self):
        self.assertEqual(self._detect({"/.dockerenv"}, {}), "docker")

    def test_local(self):
        self.assertEqual(self._detect(set(), {}), "local")

    def test_agrees_with_repo_dir_for_kaggle_when_both_exist(self):
        # detect_platform and repo_dir must pick the same platform.
        platform = self._detect({"/content", "/kaggle"}, {})
        self.assertEqual(platform, "kaggle")
        self.assertEqual(runtime.repo_dir(platform).as_posix(),
                         "/kaggle/working/video-dubbing")


class GeneratedNotebookTests(unittest.TestCase):
    """The committed notebooks must match the generator's current cells."""

    def test_both_notebooks_use_the_tee_safe_follower(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("runtime.follow_log(", src,
                              "regenerate: python deploy/notebooks/build_notebooks.py")
                self.assertNotIn("fh.readline()", src,
                                 "the old print-a-read-line loop is back")

    def test_both_notebooks_fetch_the_vendored_fork(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                self.assertIn("scripts/vendor_f5_tts.py", notebook_source(path),
                              "the Indian-language voice needs the f5_tts fork "
                              "fetched during setup")

    def test_both_notebooks_use_the_shared_repo_dir_rule(self):
        # One resolver, decided once, in the first cell. The cells used to
        # recompute REPO_DIR independently and disagree, so a Kaggle session
        # whose checkout was at /kaggle/working/video-dubbing had a later cell
        # `os.chdir` into /content/video-dubbing and raise FileNotFoundError.
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn("def _resolve_repo_dir():", src)
                self.assertIn('os.environ["VIDEOTRANS_REPO_DIR"] = REPO_DIR', src)
                self.assertIn('os.environ.get("VIDEOTRANS_REPO_DIR")', src)
                self.assertIn('"/kaggle/working/video-dubbing"', src)
                self.assertIn('"/content/video-dubbing"', src)

    def test_both_notebooks_read_the_platform_secret(self):
        # A secret added after the kernel started is not in os.environ; the
        # token cell has to fetch it through the platform SDK and export it,
        # or the download step never sees it.
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                src = notebook_source(path)
                self.assertIn('userdata.get("HF_TOKEN")', src)
                self.assertIn('get_secret("HF_TOKEN")', src)

    def test_notebooks_have_cell_ids(self):
        # nbformat 4.5 requires an id per cell; hand-edited cells lose them.
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(all("id" in cell for cell in notebook["cells"]))


if __name__ == "__main__":
    unittest.main()

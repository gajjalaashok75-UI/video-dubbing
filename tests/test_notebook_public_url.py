"""
The server cell must print a URL the user's browser can actually open.

On Colab that is the kernel proxy, obtained with `google.colab.output.eval_js`.
`eval_js` returns `None` when the frontend is busy - and the cell calls it right
after `start_server`, with a multi-gigabyte model loading - so a single
unvalidated call hands the user `127.0.0.1:8000`, which is inside the VM and
unreachable from the browser. That is what a real session printed.

These tests pin the retry-and-validate policy against a fake `google.colab`
module, so a regression is caught here instead of in someone's run.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import deploy_helpers as runtime  # noqa: E402

PROXY = "https://8000-abc123.prod.colab.dev"


class _FakeOutput:
    """A stand-in for `google.colab.output`, scripted with replies."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.scripts: list[str] = []
        self.kwargs: list[dict] = []

    def eval_js(self, script, **kwargs):
        self.scripts.append(script)
        self.kwargs.append(kwargs)
        reply = self.replies.pop(0) if self.replies else None
        if isinstance(reply, Exception):
            raise reply
        return reply


class _FakeOutputNoTimeout(_FakeOutput):
    """An older colabtools whose `eval_js` has no `timeout_sec` parameter."""

    def eval_js(self, script):
        return super().eval_js(script)


class _Colab:
    """Install a fake `google.colab` for the duration of a test."""

    def __init__(self, output_module) -> None:
        self.output = output_module
        self.saved: dict = {}

    def __enter__(self):
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.output = self.output  # type: ignore[attr-defined]
        google.colab = colab  # type: ignore[attr-defined]
        for name, module in (("google", google), ("google.colab", colab),
                             ("google.colab.output", self.output)):
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = module
        return self

    def __exit__(self, *exc):
        for name, previous in self.saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        return False


def _no_sleep():
    return mock.patch.object(runtime.time, "sleep", lambda *_: None)


class ColabProxyUrlTests(unittest.TestCase):
    def test_a_proxy_url_is_returned(self):
        output = _FakeOutput([PROXY])
        with _Colab(output), _no_sleep():
            self.assertEqual(runtime._colab_proxy_url(8000), PROXY)
        self.assertEqual(output.scripts, ["google.colab.kernel.proxyPort(8000)"])

    def test_a_trailing_slash_is_removed(self):
        output = _FakeOutput([PROXY + "/"])
        with _Colab(output), _no_sleep():
            self.assertEqual(runtime._colab_proxy_url(8000), PROXY)

    def test_the_reply_is_awaited_with_a_timeout(self):
        output = _FakeOutput([PROXY])
        with _Colab(output), _no_sleep():
            runtime._colab_proxy_url(8000)
        self.assertEqual(output.kwargs, [{"timeout_sec": 10}])

    def test_a_none_reply_is_retried(self):
        output = _FakeOutput([None, PROXY])
        with _Colab(output), _no_sleep():
            self.assertEqual(runtime._colab_proxy_url(8000), PROXY)
        self.assertEqual(len(output.scripts), 2)

    def test_a_loopback_reply_is_rejected(self):
        output = _FakeOutput(["http://127.0.0.1:8000"])
        with _Colab(output), _no_sleep():
            self.assertIsNone(runtime._colab_proxy_url(8000))

    def test_a_reply_without_the_port_is_rejected(self):
        output = _FakeOutput(["https://example.com/"])
        with _Colab(output), _no_sleep():
            self.assertIsNone(runtime._colab_proxy_url(8000))

    def test_an_exception_is_retried(self):
        output = _FakeOutput([RuntimeError("frontend busy"), PROXY])
        with _Colab(output), _no_sleep():
            self.assertEqual(runtime._colab_proxy_url(8000), PROXY)
        self.assertEqual(len(output.scripts), 2)

    def test_the_attempts_are_bounded(self):
        output = _FakeOutput([])  # every reply is None
        with _Colab(output), _no_sleep():
            self.assertIsNone(runtime._colab_proxy_url(8000, attempts=3))
        self.assertEqual(len(output.scripts), 3)

    def test_an_eval_js_without_timeout_sec_still_works(self):
        output = _FakeOutputNoTimeout([PROXY])
        with _Colab(output), _no_sleep():
            self.assertEqual(runtime._colab_proxy_url(8000), PROXY)
        self.assertEqual(output.kwargs, [{}])  # retried without the keyword

    def test_no_google_colab_module_returns_none(self):
        saved = sys.modules.get("google.colab")
        sys.modules["google.colab"] = None  # makes the import raise
        try:
            self.assertIsNone(runtime._colab_proxy_url(8000))
        finally:
            if saved is None:
                sys.modules.pop("google.colab", None)
            else:
                sys.modules["google.colab"] = saved


class NotebookPublicUrlTests(unittest.TestCase):
    def test_colab_uses_the_proxy(self):
        output = _FakeOutput([PROXY])
        with _Colab(output), _no_sleep(), \
                mock.patch.object(runtime, "detect_platform", lambda: "colab"):
            self.assertEqual(runtime.notebook_public_url(8000), PROXY)

    def test_colab_falls_back_to_loopback_when_the_proxy_fails(self):
        output = _FakeOutput([None, None, None])
        with _Colab(output), _no_sleep(), \
                mock.patch.object(runtime, "detect_platform", lambda: "colab"):
            self.assertEqual(runtime.notebook_public_url(8000),
                             "http://127.0.0.1:8000")

    def test_kaggle_uses_public_host_when_a_tunnel_set_it(self):
        with mock.patch.object(runtime, "detect_platform", lambda: "kaggle"), \
                mock.patch.dict(os.environ, {"PUBLIC_HOST": "x.trycloudflare.com"}):
            self.assertEqual(runtime.notebook_public_url(8000),
                             "https://x.trycloudflare.com")

    def test_a_wildcard_bind_is_shown_as_loopback(self):
        with mock.patch.object(runtime, "detect_platform", lambda: "local"), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime.notebook_public_url(8000),
                             "http://127.0.0.1:8000")


class ServerCellTests(unittest.TestCase):
    """The cell must tell the user when the printed URL is not reachable."""

    def _server_cell(self, notebook: str) -> str:
        import json

        path = ROOT / "deploy" / "notebooks" / notebook
        data = json.loads(path.read_text(encoding="utf-8"))
        for cell in data["cells"]:
            source = "".join(cell["source"])
            if "VideoTrans is running" in source:
                return source
        raise AssertionError(f"server cell not found in {notebook}")

    def test_both_notebooks_warn_on_a_loopback_url(self):
        for notebook in ("colab.ipynb", "kaggle.ipynb"):
            with self.subTest(notebook=notebook):
                source = self._server_cell(notebook)
                self.assertIn("loopback address", source)
                self.assertIn("notebook_public_url", source)


if __name__ == "__main__":
    unittest.main()

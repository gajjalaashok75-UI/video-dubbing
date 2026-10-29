"""
Re-cloning a reference must actually change the voice that is spoken.

The Phase 3 registry made every phrase resolve its own speaker's reference
(`speakers.reference_for`, which prefers the registry over the global one). The
re-clone endpoint kept replacing only `job["reference"]`, so:

  * a single-speaker job re-cloned to a new clip still spoke in the OLD voice,
    because `SPEAKER_00`'s registry entry still pointed at the old file — and
    then stored the new clip as the one it had used;
  * a per-speaker re-clone overwrote the job-wide reference with one actor's
    window, so the studio showed the wrong clip as "what the voice came from"
    and every unresolved speaker fell back to that actor's voice.

`_rebind_reference` is the rule that fixes the first without collapsing the
second. Model-free.
"""

from __future__ import annotations

import ast
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import speakers as SPK  # noqa: E402

APP_PATH = os.path.join(ROOT, "engine", "app.py")


def _app_functions(names: set, **namespace) -> types.ModuleType:
    with open(APP_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise AssertionError(
            f"expected {sorted(names)} in app.py, found "
            f"{sorted(n.name for n in body)}")
    module = types.ModuleType("app_reclone_probe")
    module.__dict__.update(namespace)
    module.__dict__.setdefault("print", print)
    module.__dict__.setdefault("os", os)
    exec(compile(ast.Module(body=body, type_ignores=[]), APP_PATH, "exec"),
         module.__dict__)
    return module


def ref(path, duration=8.0, text="hello"):
    return {"path": path, "text": text, "duration": duration}


def unit(label, index=0):
    return {"index": index, "text": "x", "speaker": label, "ref_index": None}


class RebindTests(unittest.TestCase):
    def setUp(self):
        self.module = _app_functions({"_rebind_reference"})

    def test_a_single_speaker_registry_moves_to_the_new_clip(self):
        old = ref("/tmp/old.wav")
        registry = SPK.implicit_speaker(old)
        self.assertEqual(registry[0]["reference"]["path"], "/tmp/old.wav")

        moved = self.module._rebind_reference(registry, old, ref("/tmp/new.wav"))

        self.assertEqual(moved, 1)
        self.assertEqual(registry[0]["reference"]["path"], "/tmp/new.wav")
        # And that IS what the plan resolves, which is the whole point.
        resolved = SPK.reference_for(unit("SPEAKER_00"), registry, old)
        self.assertEqual(resolved["path"], "/tmp/new.wav")

    def test_another_actors_reference_is_never_replaced(self):
        # A two-hander: each actor has their own window. A whole-run re-clone
        # must not collapse them into one voice.
        old = ref("/tmp/global.wav")
        registry = [
            {"label": "SPEAKER_00", "reference": ref("/tmp/a.wav",
                                                     text="a"), "fallback": False},
            {"label": "SPEAKER_01", "reference": ref("/tmp/b.wav",
                                                     text="b"), "fallback": False},
        ]
        moved = self.module._rebind_reference(registry, old, ref("/tmp/new.wav"))

        self.assertEqual(moved, 0)
        self.assertEqual([e["reference"]["path"] for e in registry],
                         ["/tmp/a.wav", "/tmp/b.wav"])

    def test_an_unresolved_speaker_picks_up_the_new_clip(self):
        registry = [{"label": "SPEAKER_00", "reference": None, "fallback": True},
                    {"label": "SPEAKER_01", "reference": ref("/tmp/b.wav"),
                     "fallback": False}]
        moved = self.module._rebind_reference(registry, ref("/tmp/global.wav"),
                                             ref("/tmp/new.wav"))
        self.assertEqual(moved, 1)
        self.assertEqual(registry[0]["reference"]["path"], "/tmp/new.wav")
        self.assertFalse(registry[0]["fallback"])
        self.assertEqual(registry[1]["reference"]["path"], "/tmp/b.wav")

    def test_an_entry_pointing_at_the_old_global_clip_moves(self):
        old = ref("/tmp/global.wav")
        registry = [
            {"label": "SPEAKER_00", "reference": dict(old), "fallback": False},
            {"label": "SPEAKER_01", "reference": ref("/tmp/b.wav"),
             "fallback": False},
        ]
        moved = self.module._rebind_reference(registry, old, ref("/tmp/new.wav"))
        self.assertEqual(moved, 1)
        self.assertEqual(registry[0]["reference"]["path"], "/tmp/new.wav")

    def test_no_registry_is_not_an_error(self):
        self.assertEqual(
            self.module._rebind_reference([], ref("/tmp/old.wav"),
                                          ref("/tmp/new.wav")), 0)


if __name__ == "__main__":
    unittest.main()

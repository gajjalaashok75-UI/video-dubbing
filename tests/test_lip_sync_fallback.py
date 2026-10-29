"""
Phase 7: a clip with no face in it is dubbed, not destroyed.

Lip sync used to abort the whole job when the face detector found nothing:
`LipSyncError` is a plain failure, so the finished dub was thrown away and the
project came back `failed`. The picture is the only thing missing a face — the
translated audio is complete and correct — so the stage is not applicable
rather than broken, and the right answer is to deliver the dub over the
untouched video and say so.

What is pinned here:

  * the detector's "nothing anywhere" case raises `NoFaceDetected`, and that is
    still a `LipSyncError` so every existing caller keeps its behaviour;
  * the delivery helper muxes the picture with the dubbed audio, at the source
    picture's quality (stream copy);
  * the run summary reports the stage as a fallback with a readable reason, and
    raises a warning, instead of claiming it ran.

Model-free. The one test that needs ffmpeg is skipped without it, because it
actually produces a file and measures it.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import av_sync  # noqa: E402
import lip_sync  # noqa: E402
import run_summary as RS  # noqa: E402

HAVE_FFMPEG = (shutil.which("ffmpeg") is not None
               and shutil.which("ffprobe") is not None)

APP_PATH = os.path.join(ROOT, "engine", "app.py")


def _app_functions(names: set, **namespace) -> types.ModuleType:
    """
    Compile a few functions out of app.py.

    Importing `app` loads OpenCV, torch and every model wrapper through
    `lip_sync`; these tests are about two pure helpers, so they are compiled
    into a throwaway module with only the names they use. Same approach the
    mix tests take.
    """
    with open(APP_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise AssertionError(
            f"expected {sorted(names)} in app.py, found "
            f"{sorted(n.name for n in body)}")
    module = types.ModuleType("app_fallback_probe")
    module.__dict__.update(namespace)
    module.__dict__.setdefault("print", print)
    module.__dict__.setdefault("os", os)
    exec(compile(ast.Module(body=body, type_ignores=[]), APP_PATH, "exec"),
         module.__dict__)
    return module


class NoFaceSignalTests(unittest.TestCase):
    """The detector's answer must be distinguishable from a broken render."""

    def test_no_face_is_a_lip_sync_error(self):
        # Existing callers catch LipSyncError; the new type must not escape
        # them, or a render that used to fail cleanly would now crash.
        self.assertTrue(issubclass(lip_sync.NoFaceDetected, lip_sync.LipSyncError))
        self.assertTrue(lip_sync.NoFaceDetected("x").no_face)

    def test_a_plain_error_is_not_a_no_face_error(self):
        self.assertFalse(getattr(lip_sync.LipSyncError("x"), "no_face", False))

    def test_a_clip_with_no_face_anywhere_raises_the_new_type(self):
        runner = lip_sync.Wav2LipRunner()
        raw = [None, None, None]
        with self.assertRaises(lip_sync.NoFaceDetected):
            runner._boxes_from_detections(raw, [0, 1, 2], 3, (720, 1280),
                                          scale=1.0, pads=(0, 12, 0, 0))

    def test_a_clip_with_one_face_does_not(self):
        # The whole-clip case is the only one that skips: a single detected
        # frame is enough for the track to be synced, as before.
        runner = lip_sync.Wav2LipRunner()
        raw = [None, (10, 10, 60, 60), None]
        boxes = runner._boxes_from_detections(raw, [0, 1, 2], 3, (720, 1280),
                                              scale=1.0, pads=(0, 12, 0, 0))
        self.assertEqual(len(boxes), 3)


class DeliveryTests(unittest.TestCase):
    """The fallback must attach the dub, not re-encode or drop anything."""

    def test_the_helper_muxes_the_picture_with_the_dub(self):
        calls = []
        fake = types.SimpleNamespace(
            mux=lambda video, audio, out: calls.append((video, audio, out)))
        module = _app_functions({"_deliver_without_lip_sync"}, av_sync=fake)

        note = module._deliver_without_lip_sync(
            "/in/clip.mp4", "/in/dub.wav", "/out/dubbed.mp4",
            "No face detected anywhere in the video.")

        self.assertEqual(calls, [("/in/clip.mp4", "/in/dub.wav",
                                  "/out/dubbed.mp4")])
        self.assertTrue(note["skipped"])
        self.assertIn("No face", note["reason"])

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
    def test_the_delivered_file_keeps_the_picture_and_has_the_dub(self):
        tmp = tempfile.mkdtemp(prefix="no_face_")
        self.addCleanup(shutil.rmtree, tmp, True)
        source = os.path.join(tmp, "source.mp4")
        audio = os.path.join(tmp, "dub.wav")
        out = os.path.join(tmp, "dubbed.mp4")

        # One second of a still picture, and one second of tone: enough for
        # probe/verify to measure both streams.
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=gray:s=160x120:d=1:r=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", source], check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=660:duration=1",
             "-c:a", "pcm_s16le", audio], check=True, capture_output=True)

        module = _app_functions({"_deliver_without_lip_sync"}, av_sync=av_sync)
        note = module._deliver_without_lip_sync(source, audio, out, "no face")

        self.assertTrue(os.path.exists(out))
        self.assertTrue(note["skipped"])
        info = av_sync.probe(out)
        self.assertIsNotNone(info["video"], "the picture was dropped")
        self.assertIsNotNone(info["audio"], "the dub was dropped")
        self.assertAlmostEqual(info["duration"], 1.0, delta=0.15)
        self.assertTrue(av_sync.verify_sync(out)["ok"])


class _Job(dict):
    """A job with the minimum a summary needs."""


def _job(**kw):
    base = {
        "job_id": "job", "kind": "dub", "status": "done",
        "source_language": "en", "target_language": "hi",
        "video_duration": 12.0, "segments": [], "translated_segments": [],
        "plan": [], "speakers": [], "diarization": [],
        "stems": {"background": "/tmp/bg.wav", "model": "htdemucs"},
        "mixed_path": "/tmp/mixed.wav",
    }
    base.update(kw)
    job = _Job(base)
    return job


class ReportingTests(unittest.TestCase):
    """A skipped stage must be visible, in both the rows and the warnings."""

    def test_the_lip_sync_stage_reports_a_fallback_with_the_reason(self):
        job = _job(lip_sync={"skipped": True,
                             "reason": "No face detected anywhere in the video."})
        row = next(r for r in RS.stage_rows(job) if r["key"] == "lip_syncing")
        self.assertEqual(row["state"], "fallback")
        self.assertIn("No face", row["detail"])

    def test_a_normal_render_still_reports_the_stage_as_done(self):
        job = _job(lip_sync={"skipped": False})
        row = next(r for r in RS.stage_rows(job) if r["key"] == "lip_syncing")
        self.assertEqual(row["state"], "done")

    def test_the_skip_is_warned_about_in_word(self):
        job = _job(lip_sync={"skipped": True, "no_face": True,
                             "reason": "No face detected anywhere in the video."})
        joined = " ".join(RS.warnings(job))
        self.assertIn("No face was found", joined)

    def test_a_broken_lip_sync_is_a_fallback_with_its_own_reason(self):
        # A failure that is NOT "no face" - a missing checkpoint, an audio
        # decoder the environment broke - delays nothing that matters: the dub
        # is finished. It must still be reported as a fallback, and reported
        # with the real reason rather than "no face was found".
        job = _job(lip_sync={"skipped": True, "no_face": False,
                             "reason": "LipSyncError: Numba needs NumPy 2.2"})
        row = next(r for r in RS.stage_rows(job) if r["key"] == "lip_syncing")
        self.assertEqual(row["state"], "fallback")
        joined = " ".join(RS.warnings(job))
        self.assertIn("Lip sync did not run", joined)
        self.assertIn("Numba", joined)
        self.assertNotIn("No face was found", joined)

    def test_the_delivery_helper_defaults_to_not_a_no_face_skip(self):
        fake = types.SimpleNamespace(mux=lambda *a: None)
        module = _app_functions({"_deliver_without_lip_sync"}, av_sync=fake)
        note = module._deliver_without_lip_sync("/v.mp4", "/d.wav", "/o.mp4",
                                                "ffmpeg encode failed")
        self.assertFalse(note["no_face"])
        self.assertEqual(note["reason"], "ffmpeg encode failed")

    def test_the_summary_carries_the_note_for_the_manifest(self):
        job = _job(lip_sync={"skipped": True, "reason": "no face"})
        self.assertEqual(RS.build(job)["lip_sync"]["reason"], "no face")

    def test_an_audio_only_run_is_still_skipped_not_fallback(self):
        job = _job(kind="audio", lip_sync={"skipped": True, "reason": "x"})
        row = next(r for r in RS.stage_rows(job) if r["key"] == "lip_syncing")
        self.assertEqual(row["state"], "skipped")


if __name__ == "__main__":
    unittest.main()

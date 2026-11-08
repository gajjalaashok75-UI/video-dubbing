"""
Diarization must be handed AUDIO, not the source video.

The bug this pins: `app.py` called `speaker_diarization.diarize(input_path, ...)`
with the uploaded `.mp4`. pyannote reads through libsndfile, which cannot open an
mp4 container, so it raised

    Error opening '..._male-female.mp4': Format not recognised.

`diarize` swallowed that, returned [], and the render silently fell back to
single-speaker mode — a male/female clip came back entirely in one (female)
voice, with no error shown. `diarize` now decodes an unreadable path to a
temporary wav before handing it to pyannote, and `app.py` passes the 16 kHz wav
it already extracted for Whisper.

`av_sync.extract_audio` shells out to ffmpeg, which is not available in this
test environment, so the decode is stubbed and this pins the ORCHESTRATION:
that an unreadable input is decoded, that the DECODED path reaches the backend,
and that the temporary file is removed. The real decode on Colab is ffmpeg's,
already exercised by the pipeline. Everything else is the real code path.

No models, no network, no GPU.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import speaker_diarization as SD  # noqa: E402


class _RecordingDiarizer(SD.Diarizer):
    """A backend that records the path it was asked about."""

    name = "recording"

    def __init__(self):
        self.seen = []

    def available(self):
        return True

    def diarize(self, audio_path, min_speakers=1, max_speakers=10):
        self.seen.append(audio_path)
        return [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
                {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"}]


def _wav(path: str, seconds: float = 1.0) -> str:
    sr = 16000
    import numpy as np
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path


class ReadableAudioTests(unittest.TestCase):
    def test_a_wav_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = _wav(os.path.join(tmp, "a.wav"))
            got, cleanup = SD._readable_audio(wav)
            self.assertEqual(got, wav)
            self.assertIsNone(cleanup)

    def test_a_video_container_is_decoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "clip.mp4")
            with open(video, "wb") as fh:      # not sound: sf.info will refuse
                fh.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
            decoded = _wav(os.path.join(tmp, "decoded.wav"))

            calls = []

            def fake_extract(src, out, sr, ch):
                calls.append((src, out, sr, ch))
                _wav(out)

            with mock.patch("av_sync.extract_audio", side_effect=fake_extract):
                got, cleanup = SD._readable_audio(video)

            self.assertEqual(calls and calls[0][0], video)   # decoded the video
            self.assertEqual(calls[0][2], 16000)             # 16 kHz mono
            self.assertEqual(calls[0][3], 1)
            self.assertEqual(got, cleanup)                   # temp, to be removed
            self.assertTrue(os.path.exists(got))
            os.unlink(got)


class DiarizeOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = _RecordingDiarizer()
        patcher = mock.patch.object(SD, "get_backend", return_value=self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_audio_reaches_the_backend_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = _wav(os.path.join(tmp, "a.wav"))
            turns = SD.diarize(wav)
            self.assertEqual(len(turns), 2)
            self.assertEqual(self.backend.seen, [wav])

    def test_a_video_is_decoded_then_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "clip.mp4")
            with open(video, "wb") as fh:
                fh.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)

            def fake_extract(src, out, sr, ch):
                _wav(out)

            with mock.patch("av_sync.extract_audio", side_effect=fake_extract):
                turns = SD.diarize(video)

            self.assertEqual(len(turns), 2)
            handed = self.backend.seen[0]
            self.assertNotEqual(handed, video)      # never handed the container
            self.assertTrue(handed.endswith(".wav"))
            self.assertFalse(os.path.exists(handed))  # temp removed afterwards

    def test_a_decode_failure_passes_through_without_raising(self):
        """
        A decode failure is not fatal: the original path is handed to the
        backend, and if the backend then fails (as pyannote would on the same
        container) `diarize` still returns [] rather than raising, so the render
        falls back to one speaker instead of dying.
        """
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "clip.mp4")
            with open(video, "wb") as fh:
                fh.write(b"\x00" * 64)

            def boom(*a, **k):
                raise RuntimeError("ffmpeg not available")

            with mock.patch("av_sync.extract_audio", side_effect=boom):
                turns = SD.diarize(video)

            # Tolerated: the fake backend gets the original path and answers.
            self.assertEqual(self.backend.seen, [video])
            self.assertEqual(len(turns), 2)

    def test_a_backend_that_rejects_the_container_falls_back(self):
        """The real case: pass-through reaches pyannote, which rejects it."""
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "clip.mp4")
            with open(video, "wb") as fh:
                fh.write(b"\x00" * 64)

            class _Picky(_RecordingDiarizer):
                def diarize(self, audio_path, min_speakers=1, max_speakers=10):
                    raise RuntimeError("Format not recognised.")

            picky = _Picky()
            with mock.patch.object(SD, "get_backend", return_value=picky), \
                 mock.patch("av_sync.extract_audio",
                            side_effect=RuntimeError("no ffmpeg")):
                turns = SD.diarize(video)

            self.assertEqual(turns, [])          # graceful single-speaker


if __name__ == "__main__":
    unittest.main()

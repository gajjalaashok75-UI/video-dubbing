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


class DiarizationFailureIsReportedTests(unittest.TestCase):
    """
    A collapse to one voice must be explainable after the fact.

    `diarize` never raises, so the only way a caller (or the per-step log) can
    learn WHY a conversation came back single-speaker is `last_error()`. The
    reason has to survive the return and clear on the next success.
    """

    def setUp(self):
        SD._set_last_error(None)

    def test_a_missing_backend_records_a_reason(self):
        class _Absent(_RecordingDiarizer):
            def available(self):
                return False

        with mock.patch.object(SD, "get_backend", return_value=_Absent()):
            turns = SD.diarize("/tmp/whatever.wav")
        self.assertEqual(turns, [])
        self.assertIn("not installed", SD.last_error())

    def test_a_backend_exception_records_its_message(self):
        class _Broken(_RecordingDiarizer):
            def diarize(self, audio_path, min_speakers=1, max_speakers=None):
                raise RuntimeError("Gated repo: accept the licence")

        with mock.patch.object(SD, "get_backend", return_value=_Broken()):
            turns = SD.diarize("/tmp/whatever.wav")
        self.assertEqual(turns, [])
        self.assertIn("Gated repo", SD.last_error())

    def test_a_success_clears_the_previous_reason(self):
        SD._set_last_error("stale failure")
        with mock.patch.object(SD, "get_backend",
                               return_value=_RecordingDiarizer()), \
             mock.patch.object(SD, "_readable_audio",
                               return_value=("/tmp/a.wav", None)):
            turns = SD.diarize("/tmp/a.wav")
        self.assertEqual(len(turns), 2)
        self.assertIsNone(SD.last_error())


class SpeakerCeilingIsOptionalTests(unittest.TestCase):
    """
    No ceiling by default: the diarizer finds as many speakers as it hears.

    A wrong `max_speakers` merges extra actors into the kept ones, which is a
    silent wrong number of voices. The default must therefore be "unset", and
    an unset ceiling must not be sent to pyannote as `max_speakers=None`.
    """

    def _pipeline(self, record):
        class _Annotation:
            def itertracks(self, yield_label=True):
                return iter(())

        class _Pipeline:
            def __call__(self, path, **kwargs):
                record.update(kwargs)
                return _Annotation()

        return _Pipeline()

    def test_an_unset_ceiling_is_not_passed(self):
        import speaker_diarization as sd
        record = {}
        d = sd.PyannoteDiarizer()
        with mock.patch.object(d, "_load",
                               return_value=self._pipeline(record)):
            d.diarize("/tmp/a.wav", min_speakers=1, max_speakers=None)
        self.assertEqual(record, {"min_speakers": 1})

    def test_a_set_ceiling_is_passed(self):
        import speaker_diarization as sd
        record = {}
        d = sd.PyannoteDiarizer()
        with mock.patch.object(d, "_load",
                               return_value=self._pipeline(record)):
            d.diarize("/tmp/a.wav", min_speakers=2, max_speakers=4)
        self.assertEqual(record, {"min_speakers": 2, "max_speakers": 4})

    def test_the_tunable_default_is_no_ceiling(self):
        import engines
        self.assertIsNone(engines.TUNABLES["DIAR_MAX_SPEAKERS"][0])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DIAR_MAX_SPEAKERS", None)
            self.assertIsNone(engines.tunable("DIAR_MAX_SPEAKERS"))


if __name__ == "__main__":
    unittest.main()

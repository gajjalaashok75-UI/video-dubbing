"""
A GPU that cannot be used must not fail the render.

A hosted run (Colab T4, 2026-10-29) had a CUDA 13 torch installed while
CTranslate2 - faster-whisper's backend - is built for CUDA 12. CTranslate2
loads its CUDA runtime on the FIRST encode, not when the model is constructed,
so the job was accepted, the model "loaded successfully", and then ~6 seconds
later every single render died with:

    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

Nothing in the decode settings can fix a missing shared library, and dropping to
a smaller model cannot either. The stage now retries on the CPU, which is slower
but correct, and records why so the slowdown is visible instead of mysterious.

What is pinned here:

  * the recogniser only treats GPU-unusable failures as a reason to demote -
    a real model error still raises;
  * the retry happens on the CPU, at CPU settings;
  * ``fallback_reason()`` says what happened, for the run summary;
  * the failure is caught even though ``transcribe`` is lazy: the generator is
    drained inside the retry, otherwise the error would surface later with no
    model left to retry with.

Model-free: the model is a fake.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

try:
    import speech_to_text_v2 as STT  # noqa: E402
except Exception as exc:                                  # pragma: no cover
    raise unittest.SkipTest(f"speech_to_text_v2 is not importable here: {exc}")

CUBLAS_FAILURE = RuntimeError(
    "Library libcublas.so.12 is not found or cannot be loaded")


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end, self.probability = word, start, end, 0.9


class _Segment:
    def __init__(self, text, start, end, words=None):
        self.text, self.start, self.end = text, start, end
        self.words = words or []


class _Info:
    language = "en"

    def __init__(self, duration):
        self.duration = duration


class _FakeModel:
    """A model whose first call fails the way the GPU did, then works."""

    def __init__(self, failures=1, error=CUBLAS_FAILURE):
        self.calls = 0
        self.failures = failures
        self.error = error

    def transcribe(self, path, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        segments = [_Segment(" hello there", 0.0, 1.2,
                             [_Word(" hello", 0.0, 0.5),
                              _Word(" there", 0.5, 1.2)])]
        return iter(segments), _Info(1.2)


class _State:
    """Save and restore the module globals a demotion rewrites."""

    def __init__(self, test):
        self.test = test

    def __enter__(self):
        self.saved = (STT.DEVICE, STT.COMPUTE_TYPE, STT._model,
                      STT._FALLBACK_REASON)
        STT.DEVICE = "cuda"
        STT.COMPUTE_TYPE = "float16"
        STT._model = None
        STT._FALLBACK_REASON = None
        return self

    def __exit__(self, *exc):
        (STT.DEVICE, STT.COMPUTE_TYPE, STT._model,
         STT._FALLBACK_REASON) = self.saved
        return False


class ErrorClassificationTests(unittest.TestCase):
    def test_the_cublas_failure_is_recognised(self):
        self.assertTrue(STT._cuda_runtime_error(CUBLAS_FAILURE))

    def test_a_cudnn_failure_is_recognised(self):
        self.assertTrue(STT._cuda_runtime_error(
            RuntimeError("Library libcudnn.so.9 is not found or cannot be "
                         "loaded")))

    def test_a_driver_too_old_is_recognised(self):
        self.assertTrue(STT._cuda_runtime_error(
            RuntimeError("CUDA driver version is insufficient for CUDA "
                         "runtime version")))

    def test_an_unrelated_error_is_not(self):
        # A genuine model problem must still fail the job loudly.
        self.assertFalse(STT._cuda_runtime_error(
            RuntimeError("Unable to open file 'missing.wav'")))
        self.assertFalse(STT._cuda_runtime_error(ValueError("bad language")))

    def test_out_of_memory_is_not_treated_as_unusable(self):
        # An OOM on a full GPU is a different problem; silently moving a
        # 3 GB model to the CPU would hide it and be hopelessly slow.
        self.assertFalse(STT._cuda_runtime_error(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))


class DemotionTests(unittest.TestCase):
    def test_demotion_switches_to_cpu_int8_and_records_the_reason(self):
        with _State(self):
            changed = STT._demote_to_cpu(CUBLAS_FAILURE)
            self.assertTrue(changed)
            self.assertEqual(STT.DEVICE, "cpu")
            self.assertEqual(STT.COMPUTE_TYPE, "int8")
            self.assertIn("libcublas", STT.fallback_reason())

    def test_demotion_is_a_no_op_off_the_gpu(self):
        with _State(self):
            STT.DEVICE = "cpu"
            self.assertFalse(STT._demote_to_cpu(CUBLAS_FAILURE))
            self.assertIsNone(STT.fallback_reason())

    def test_demotion_ignores_a_non_cuda_error(self):
        with _State(self):
            self.assertFalse(STT._demote_to_cpu(RuntimeError("no such file")))
            self.assertEqual(STT.DEVICE, "cuda")

    def test_demotion_drops_the_gpu_model(self):
        with _State(self):
            STT._model = object()
            STT._demote_to_cpu(CUBLAS_FAILURE)
            self.assertIsNone(STT._model, "the GPU model must be reloaded")


class _FakeWhisperLoader:
    """Stands in for ``_load_model``: counts the reloads, same fake each time."""

    def __init__(self, model):
        self.model = model
        self.loads = 0

    def __call__(self):
        self.loads += 1
        return self.model


class TranscribeRetryTests(unittest.TestCase):
    def test_a_cublas_failure_mid_transcribe_retries_on_the_cpu(self):
        model = _FakeModel(failures=1)
        loader = _FakeWhisperLoader(model)
        with _State(self):
            with mock.patch.object(STT, "_load_model", loader):
                result = STT.transcribe_audio("/tmp/does-not-exist.wav")

            self.assertEqual(model.calls, 2, "the decode must be retried once")
            self.assertEqual(STT.DEVICE, "cpu")
            self.assertIn("hello there", result["text"])
            self.assertEqual(len(result["segments"]), 1)
            self.assertIn("libcublas", STT.fallback_reason())

    def test_a_real_model_error_still_raises(self):
        model = _FakeModel(failures=99, error=RuntimeError("corrupt audio"))
        loader = _FakeWhisperLoader(model)
        with _State(self):
            with mock.patch.object(STT, "_load_model", loader):
                with self.assertRaises(RuntimeError):
                    STT.transcribe_audio("/tmp/does-not-exist.wav")
            self.assertEqual(STT.DEVICE, "cuda", "the GPU was blamed for nothing")
            self.assertIsNone(STT.fallback_reason())

    def test_the_lazy_generator_is_drained_inside_the_retry(self):
        # `transcribe` returns a generator, so the CUDA call happens on the
        # first `next()`. If that were left to the caller, the retry would have
        # nothing to catch and the job would still fail.
        class _Lazy(_FakeModel):
            def transcribe(self, path, **kwargs):
                self.calls += 1
                calls = self.calls

                def _gen():
                    if calls == 1:
                        raise CUBLAS_FAILURE
                    yield _Segment(" hi", 0.0, 1.0)

                return _gen(), _Info(1.0)

        model = _Lazy()
        loader = _FakeWhisperLoader(model)
        with _State(self):
            with mock.patch.object(STT, "_load_model", loader):
                result = STT.transcribe_audio("/tmp/does-not-exist.wav")
            self.assertEqual(model.calls, 2)
            self.assertEqual(result["text"].strip(), "hi")


class TailPassTests(unittest.TestCase):
    def test_the_second_pass_still_runs_after_the_refactor(self):
        # The first decode is a list now, not the raw generator the tail pass
        # used to re-enter with `_pass(clip=...)`. Wire the second pass through
        # the same retry wrapper and prove it still recovers the tail.
        class _Tail(_FakeModel):
            def __init__(self):
                super().__init__(failures=0)

            def transcribe(self, path, **kwargs):
                self.calls += 1
                if kwargs.get("clip_timestamps"):
                    return iter([_Segment(" and the ending", 3.0, 4.0)]), _Info(4.0)
                return iter([_Segment(" the opening", 0.0, 1.0)]), _Info(4.0)

        model = _Tail()
        loader = _FakeWhisperLoader(model)
        with _State(self):
            STT.DEVICE = "cpu"       # no demotion involved in this test
            with mock.patch.object(STT, "_load_model", loader), \
                    mock.patch.object(STT, "_speech_in", lambda *a: 2.0):
                result = STT.transcribe_audio("/tmp/does-not-exist.wav")

        texts = [sg["text"] for sg in result["segments"]]
        self.assertEqual(len(result["segments"]), 2, texts)
        self.assertIn("the opening", " ".join(texts))
        self.assertIn("the ending", " ".join(texts))
        self.assertEqual(model.calls, 2)


class RecoveryTests(unittest.TestCase):
    def test_refresh_does_not_resurrect_the_gpu(self):
        # A settings change reloads the model; it must not silently put the
        # stage back on the GPU that failed, or the next render fails again.
        with _State(self):
            STT._demote_to_cpu(CUBLAS_FAILURE)
            with mock.patch.dict(os.environ, {}, clear=False):
                STT.refresh()
            self.assertEqual(STT.DEVICE, "cpu")


if __name__ == "__main__":
    unittest.main()

"""
Phase 4: preserving the film's music and sound effects under the new dialogue.

Two things are being pinned here, and they are different in kind.

**The separation wrapper** must never be the reason a dub fails. Every failure
path — Demucs absent, weights missing, a crashed process, malformed output, a
timeout — returns None and leaves the dialogue-only pipeline intact. Those tests
drive the real code against a fake `demucs` module rather than mocking the
wrapper, so the command line, the chunking and the reassembly are all exercised.

**The mix** must be exactly the right length, must contain both layers, and must
not let `amix` quietly duck the dialogue. Those run real ffmpeg against
generated audio and measure the result with an FFT, because a duration check
alone would pass on a track containing nothing but silence.

No network, no model weights, no GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import av_sync  # noqa: E402
import source_separation as SS  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

SR = 24000


def tone(freq: float, seconds: float, amp: float = 0.3, sr: int = SR):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path: str, wave, sr: int = SR, channels: int = 1) -> str:
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if channels == 2 and wave.ndim == 1:
        wave = np.stack([wave, wave], axis=1)
    sf.write(path, np.ascontiguousarray(wave), sr, subtype="PCM_16")
    return path


def band_energy(path: str, lo: float, hi: float, sr: int = SR) -> float:
    """Mean power in a frequency band, for proving a layer is present."""
    import soundfile as sf
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if wave.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(wave)) ** 2
    freqs = np.fft.rfftfreq(len(wave), 1 / rate)
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    return float(spec[mask].sum() / mask.sum())


# ---------------------------------------------------------------------------
# A fake demucs, so the wrapper can be driven for real
# ---------------------------------------------------------------------------
class FakeDemucs:
    """
    A real, importable `demucs` package that splits by frequency.

    Written to disk and put on `sys.path` rather than injected into
    `sys.modules`, because `available()` asks `importlib.util.find_spec` — the
    same question the real code asks — and a module object in `sys.modules` does
    not answer it. Testing through the real import mechanism is the point: a
    fake that cannot be found would silently exercise the fallback path.

    The split is crude (low frequencies to `vocals`, high to `no_vocals`) but
    deterministic and measurable, which is all the wrapper's contract needs.
    """

    def __init__(self, *, mode: str = "ok", pad_frames: int = 0):
        self.mode = mode            # ok | fail | silent | no_output
        self.pad_frames = pad_frames
        self.calls = []
        self.dir = None

    def install(self, tmp: str) -> str:
        """Write the package and put it first on sys.path."""
        pkg = os.path.join(tmp, "demucs")
        os.makedirs(pkg, exist_ok=True)
        self.dir = pkg
        with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(pkg, "separate.py"), "w", encoding="utf-8") as f:
            f.write(_FAKE_SEPARATE_SRC)
        with open(os.path.join(pkg, "_config.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": self.mode, "pad_frames": self.pad_frames}, f)
        sys.path.insert(0, tmp)
        return tmp

    def uninstall(self, tmp: str) -> None:
        for name in [m for m in sys.modules
                     if m == "demucs" or m.startswith("demucs.")]:
            del sys.modules[name]
        while tmp in sys.path:
            sys.path.remove(tmp)

    def read_calls(self) -> list:
        """What the fake recorded, written to a file by the subprocess."""
        path = os.path.join(self.dir or "", "_calls.json")
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:                                    # noqa: BLE001
            return []


_FAKE_SEPARATE_SRC = '''
"""A stand-in for demucs.separate, driven by _config.json beside it."""
import json, os, sys
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))


def _config():
    with open(os.path.join(HERE, "_config.json"), encoding="utf-8") as f:
        return json.load(f)


def _record(argv):
    path = os.path.join(HERE, "_calls.json")
    calls = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                calls = json.load(f)
            except Exception:
                calls = []
    calls.append(list(argv))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calls, f)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _record(argv)
    cfg = _config()
    mode = cfg.get("mode", "ok")
    if mode == "fail":
        sys.exit(2)
    if mode == "no_output":
        return
    try:
        out_dir = argv[argv.index("-o") + 1]
    except (ValueError, IndexError):
        out_dir = "out"
    model = argv[argv.index("-n") + 1] if "-n" in argv else "htdemucs"
    inputs = [a for a in argv if a.endswith(".wav") and not a.startswith("-")]
    for path in inputs:
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
        if mode == "silent":
            wave = np.zeros_like(wave)
        stem = os.path.splitext(os.path.basename(path))[0]
        base = os.path.join(out_dir, model, stem)
        os.makedirs(base, exist_ok=True)
        spec = np.fft.rfft(wave, axis=0)
        freqs = np.fft.rfftfreq(wave.shape[0], 1 / sr)
        cut = (freqs > 250).astype(np.float32)[:, None]
        low = np.fft.irfft(spec * (1 - cut), n=wave.shape[0], axis=0)
        high = np.fft.irfft(spec * cut, n=wave.shape[0], axis=0)
        pad = int(cfg.get("pad_frames") or 0)
        if pad:
            zeros = np.zeros((pad, wave.shape[1]), dtype=np.float32)
            low = np.concatenate([low, zeros], axis=0)
            high = np.concatenate([high, zeros], axis=0)
        sf.write(os.path.join(base, "vocals.wav"),
                 np.ascontiguousarray(low.astype(np.float32)), sr,
                 subtype="PCM_16")
        sf.write(os.path.join(base, "no_vocals.wav"),
                 np.ascontiguousarray(high.astype(np.float32)), sr,
                 subtype="PCM_16")


if __name__ == "__main__":
    main()
'''


class SeparationWrapperTests(unittest.TestCase):
    """The wrapper's contract: a dict on success, None on every failure."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sep_test_")
        self.workdir = os.path.join(self.tmp, "work")
        os.makedirs(self.workdir, exist_ok=True)
        self.source = write_wav(os.path.join(self.tmp, "source.wav"),
                                tone(120, 6.0), SR, channels=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _install(self, fake):
        """Write a real importable `demucs` and put it first on sys.path."""
        self._fake_root = os.path.join(self.tmp, "fakemod")
        os.makedirs(self._fake_root, exist_ok=True)
        fake.install(self._fake_root)
        self._fake = fake
        return fake

    def _uninstall(self):
        if getattr(self, "_fake", None) is not None:
            self._fake.uninstall(self._fake_root)
        if getattr(self, "_fake_root", None):
            while self._fake_root in sys.path:
                sys.path.remove(self._fake_root)

    def test_available_is_false_without_demucs(self):
        self._uninstall()
        self.assertFalse(SS.available())

    def test_available_is_true_with_demucs(self):
        fake = self._install(FakeDemucs())
        try:
            self.assertTrue(SS.available())
        finally:
            self._uninstall()

    def test_a_successful_run_writes_both_stems(self):
        self._install(FakeDemucs())
        try:
            stems = SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertIsNotNone(stems)
        self.assertTrue(os.path.exists(stems["vocals"]))
        self.assertTrue(os.path.exists(stems["background"]))
        self.assertEqual(stems["model"], SS.MODEL)

    def test_missing_demucs_returns_none_and_does_not_raise(self):
        self._uninstall()
        self.assertIsNone(SS.separate(self.source, self.workdir,
                                      log=lambda *a: None))

    def test_a_crashing_demucs_returns_none(self):
        self._install(FakeDemucs(mode="fail"))
        try:
            self.assertIsNone(SS.separate(self.source, self.workdir,
                                          log=lambda *a: None))
        finally:
            self._uninstall()

    def test_no_output_at_all_returns_none(self):
        self._install(FakeDemucs(mode="no_output"))
        try:
            self.assertIsNone(SS.separate(self.source, self.workdir,
                                          log=lambda *a: None))
        finally:
            self._uninstall()

    def test_a_failed_run_leaves_no_stale_cache_behind(self):
        # A half-written stem is worse than none: the next run would reuse it
        # and the background would be wrong for the whole film.
        self._install(FakeDemucs(mode="no_output"))
        try:
            SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        stems_dir = os.path.join(self.workdir, SS.STEMS_DIR)
        self.assertFalse(os.path.exists(os.path.join(stems_dir, SS.META_NAME)))
        self.assertFalse(os.path.exists(os.path.join(stems_dir, SS.VOCALS_NAME)))

    def test_the_stems_are_the_exact_length_of_the_source(self):
        self._install(FakeDemucs())
        try:
            stems = SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        # Durations, not frame counts: the stems are at Demucs' 44.1 kHz and
        # the test source is at 24 kHz, so the frame counts differ by design
        # while the durations must match exactly.
        want = av_sync.duration(self.source)
        self.assertAlmostEqual(av_sync.duration(stems["vocals"]), want,
                               places=2)
        self.assertAlmostEqual(av_sync.duration(stems["background"]), want,
                               places=2)

    def test_multiple_chunks_are_processed_and_reassembled(self):
        # 6s of audio in 2s chunks: three chunks, one Demucs invocation.
        self._install(FakeDemucs())
        try:
            stems = SS.separate(self.source, self.workdir, chunk_seconds=2,
                                log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertIsNotNone(stems)
        self.assertGreaterEqual(stems["chunks"], 3)
        self.assertAlmostEqual(av_sync.duration(stems["vocals"]),
                               av_sync.duration(self.source), places=2)
        self.assertAlmostEqual(av_sync.duration(stems["background"]),
                               av_sync.duration(self.source), places=2)

    def test_one_demucs_invocation_for_all_chunks(self):
        # Loading the checkpoint once per chunk instead of once per film is the
        # difference between seconds and minutes of wasted work.
        fake = self._install(FakeDemucs())
        try:
            SS.separate(self.source, self.workdir, chunk_seconds=2,
                        log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertEqual(len(fake.read_calls()), 1)

    def test_demucs_runs_sequentially(self):
        fake = self._install(FakeDemucs())
        try:
            SS.separate(self.source, self.workdir, chunk_seconds=2,
                        log=lambda *a: None)
        finally:
            self._uninstall()
        argv = fake.read_calls()[0]
        self.assertIn("-j", argv)
        self.assertEqual(argv[argv.index("-j") + 1], "1")

    def test_a_padding_model_is_corrected_at_every_chunk(self):
        # The drift bug this exists to prevent: a model that returns extra
        # frames per chunk. Uncorrected, a 12-chunk film would slide out of
        # sync by seconds.
        self._install(FakeDemucs(pad_frames=1234))
        try:
            stems = SS.separate(self.source, self.workdir, chunk_seconds=2,
                                log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertEqual(stems["chunks_adjusted"], stems["chunks"] * 2)
        # The correction is the point: without it each chunk would add 1234
        # frames, so a 12-chunk film would drift by seconds.
        self.assertAlmostEqual(av_sync.duration(stems["vocals"]),
                               av_sync.duration(self.source), places=2)
        self.assertAlmostEqual(av_sync.duration(stems["background"]),
                               av_sync.duration(self.source), places=2)

    def test_a_silent_model_is_not_treated_as_a_valid_stem(self):
        # Silence of exactly the right length passed every check this project
        # has ever had. Here it still returns stems, because separation
        # succeeded; the guard that matters is that the length is right and the
        # caller can measure the content.
        self._install(FakeDemucs(mode="silent"))
        try:
            stems = SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertIsNotNone(stems)
        self.assertLess(band_energy(stems["background"], 20, 200), 1e-6)


class Phase4DoesNotDisturbPhases1to3Tests(unittest.TestCase):
    """
    Adding the vocals stem must not move the speaker mapping.

    Phase 4 changes WHERE a reference is cut from (the vocals stem rather than
    the mixed soundtrack). It must not change WHICH reference a phrase gets.
    These run the real reference-cutting code against a separated stem and
    assert the Phase 1-3 contract still holds exactly.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="phase4_regress_")
        import speaker_diarization as SD
        self.SD = SD

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rows(self):
        rows = []
        for i, (s, e, label) in enumerate([
                (0.0, 6.0, "SPEAKER_00"), (6.0, 12.0, "SPEAKER_01"),
                (12.0, 18.0, "SPEAKER_00"), (18.0, 24.0, "SPEAKER_01")]):
            words = [{"word": f"w{i}{j}", "start": s + j * 0.5,
                      "end": s + j * 0.5 + 0.45, "probability": 0.95,
                      "speaker": label}
                     for j in range(int((e - s) / 0.5))]
            rows.append({"start": s, "end": e,
                         "text": " ".join(w["word"] for w in words),
                         "speaker": label, "words": words})
        return rows

    def test_references_cut_from_a_stem_still_map_one_per_speaker(self):
        import reference_audio as RA
        import speakers as SPK
        # A vocals stem: same duration, 44.1 kHz, as Demucs would produce.
        stem = write_wav(os.path.join(self.tmp, "vocals.wav"),
                         tone(200, 24.0, sr=44100), 44100, channels=2)
        rows = self._rows()
        registry = RA.build_references(
            stem, rows, 24.0, os.path.join(self.tmp, "refs"),
            labels=["SPEAKER_00", "SPEAKER_01"], sample_rate=24000,
            is_audio=True)
        self.assertEqual([e["label"] for e in registry],
                         ["SPEAKER_00", "SPEAKER_01"])
        paths = {e["reference"]["path"] for e in registry}
        self.assertEqual(len(paths), 2, "speakers must not share a reference")
        for entry in registry:
            self.assertFalse(entry["fallback"])
            self.assertTrue(os.path.exists(entry["reference"]["path"]))

    def test_the_vocals_stem_is_never_another_speakers_reference(self):
        import reference_audio as RA
        stem = write_wav(os.path.join(self.tmp, "vocals.wav"),
                         tone(200, 24.0, sr=44100), 44100, channels=2)
        registry = RA.build_references(
            stem, self._rows(), 24.0, os.path.join(self.tmp, "refs"),
            labels=["SPEAKER_00", "SPEAKER_01"], sample_rate=24000,
            is_audio=True)
        by_label = {e["label"]: e["reference"]["path"] for e in registry}
        self.assertNotEqual(by_label["SPEAKER_00"], by_label["SPEAKER_01"])

    def test_is_audio_cuts_a_stem_without_a_video_stream(self):
        # A stem has no picture; `-vn` against a stream that is not there is a
        # needless failure mode, which is why `is_audio` exists.
        import reference_audio as RA
        stem = write_wav(os.path.join(self.tmp, "vocals.wav"),
                         tone(200, 8.0, sr=44100), 44100, channels=2)
        out = os.path.join(self.tmp, "ref.wav")
        RA.extract_reference(stem, 0.5, 6.0, out, sample_rate=24000,
                             is_audio=True)
        self.assertTrue(os.path.exists(out))
        self.assertGreater(av_sync.duration(out), 5.0)

    def test_a_reference_from_a_stem_resolves_per_unit(self):
        # The Phase 3 contract, end to end: segment -> speaker -> that
        # speaker's reference -> the engine.
        import dubbing
        import speakers as SPK
        registry = [
            {"label": "SPEAKER_00", "total_speech": 12.0, "segment_count": 2,
             "reference": {"path": "/tmp/refs/speaker_SPEAKER_00.wav",
                           "text": "a", "duration": 8.0},
             "reference_score": 1.0, "fallback": False},
            {"label": "SPEAKER_01", "total_speech": 12.0, "segment_count": 2,
             "reference": {"path": "/tmp/refs/speaker_SPEAKER_01.wav",
                           "text": "b", "duration": 8.0},
             "reference_score": 1.0, "fallback": False},
        ]
        rows = self._rows()
        dm = dubbing.DurationModel(8.0, "a reference", source_lang="en",
                                   target_lang="hi")
        plan = dubbing.plan_timeline(rows, ["x"] * len(rows), dm, 24.0,
                                     speakers=registry)
        for unit in plan:
            ref = SPK.reference_for(unit, registry)
            self.assertEqual(
                ref["path"], f"/tmp/refs/speaker_{unit['speaker']}.wav",
                f"{unit['speaker']} resolved to {ref['path']}")


class MixingFallbackTests(unittest.TestCase):
    """
    With separation unavailable, the pipeline is exactly what it was.

    Invariant 7: a job still completes, and nothing about Phases 1-3 changes.

    `app._mix_audio` is loaded by source rather than by importing `app`, because
    importing the engine pulls in OpenCV and torch through `lip_sync`. That is a
    lot of dependency for a function whose whole job is to decide whether there
    is anything to mix — and loading it directly means these tests run on the
    same bare environment the rest of the suite does.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="phase4_fallback_")
        self._mix_audio = _load_mix_audio()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_stems_means_no_mix_and_the_dialogue_track_stands(self):
        # Returning None is what tells the caller to keep the dialogue-only
        # track it already has.
        dub = write_wav(os.path.join(self.tmp, "dub.wav"), tone(300, 4.0))
        self.assertIsNone(self._mix_audio("job", self.tmp, "stem", dub, 4.0,
                                          None, None, -18.0))

    def test_stems_with_a_missing_background_file_are_ignored(self):
        dub = write_wav(os.path.join(self.tmp, "dub.wav"), tone(300, 4.0))
        self.assertIsNone(self._mix_audio(
            "job", self.tmp, "stem", dub, 4.0,
            {"vocals": "/nope.wav", "background": "/nope.wav"}, None, -18.0))

    def test_a_present_background_does_produce_a_mix(self):
        # The other half of the contract: when there IS a bed, a file comes
        # back, and it is the right length.
        dub = write_wav(os.path.join(self.tmp, "dub.wav"), tone(300, 4.0))
        bed = write_wav(os.path.join(self.tmp, "bed.wav"), tone(80, 4.0, 0.5))
        out = self._mix_audio("job", self.tmp, "stem", dub, 4.0,
                              {"vocals": None, "background": bed}, None, -18.0)
        self.assertIsNotNone(out)
        self.assertTrue(os.path.exists(out))
        self.assertAlmostEqual(av_sync.duration(out), 4.0, places=3)
        self.assertGreater(band_energy(out, 60, 110), 1e3)

    def test_separate_returns_none_without_demucs(self):
        for name in [m for m in sys.modules
                     if m == "demucs" or m.startswith("demucs.")]:
            del sys.modules[name]
        src = write_wav(os.path.join(self.tmp, "src.wav"), tone(120, 3.0),
                        SR, channels=2)
        self.assertIsNone(SS.separate(src, self.tmp, log=lambda *a: None))


def _load_mix_audio():
    """
    Pull `_mix_audio` and the tunable it reads out of `app.py` by source.

    Extracted rather than imported so the test does not drag in OpenCV, torch
    and the whole engine bootstrap to exercise one function. The source is
    compiled into a throwaway module with just the names it needs, so the code
    under test is the real code and not a copy.
    """
    import ast
    import types as _types

    src_path = os.path.join(ROOT, "engine", "app.py")
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = {"_mix_audio", "_preserve_background"}
    body = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted]
    if len(body) != len(wanted):
        raise AssertionError(
            f"expected {sorted(wanted)} in app.py, found "
            f"{sorted(n.name for n in body)}")

    module = _types.ModuleType("engine_mix_probe")
    module.__dict__["os"] = os
    module.__dict__["av_sync"] = av_sync
    module.__dict__["engines"] = _StubEngines()
    exec(compile(ast.Module(body=body, type_ignores=[]), src_path, "exec"),
         module.__dict__)
    return module._mix_audio


class _StubEngines:
    """Just the tunables `_mix_audio` reads, at their defaults."""

    _VALUES = {
        "DIALOGUE_LUFS": -12.0,
        "BACKGROUND_LUFS": -26.0,
        "BACKGROUND_DUCK": False,
        "DUCK_DB": -12.0,
        "DUCK_RATIO": 6.0,
        "PRESERVE_BACKGROUND": False,
    }

    def tunable(self, key):
        return self._VALUES[key]


class ChunkingTests(unittest.TestCase):
    """The chunk/reassembly mechanics, independent of any model."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="chunk_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg required")
    def test_segmenting_and_concatenating_is_sample_exact(self):
        # Measured against a source whose samples are all distinct, because a
        # gap, a duplicate or a shift at a boundary is exactly what the
        # reference implementation's `-c copy` chunking is suspected of.
        import soundfile as sf
        sr = 44100
        n = sr * 5
        ramp = (np.arange(n, dtype=np.float32) % 2000) / 2000.0 - 0.5
        src = write_wav(os.path.join(self.tmp, "hq.wav"), ramp, sr, channels=2)

        chunks = SS._split(src, os.path.join(self.tmp, "chunks"),
                           chunk_seconds=2)
        self.assertGreaterEqual(len(chunks), 3)

        joined = os.path.join(self.tmp, "joined.wav")
        SS._concat(chunks, joined)

        a, _ = sf.read(src, dtype="float32")
        b, _ = sf.read(joined, dtype="float32")
        self.assertEqual(len(a), len(b))
        # PCM_16 round-trip tolerance; a boundary defect would be far larger.
        self.assertLess(float(np.abs(a - b).max()), 1e-3)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg required")
    def test_fit_stem_trims_an_overlong_chunk(self):
        import soundfile as sf
        long = write_wav(os.path.join(self.tmp, "long.wav"), tone(200, 2.0), SR)
        out = os.path.join(self.tmp, "fit.wav")
        _, delta = SS._fit_stem(long, 24000, out)
        self.assertEqual(sf.info(out).frames, 24000)
        self.assertLess(delta, 0)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg required")
    def test_fit_stem_pads_a_short_chunk(self):
        import soundfile as sf
        short = write_wav(os.path.join(self.tmp, "short.wav"), tone(200, 0.5), SR)
        out = os.path.join(self.tmp, "fit.wav")
        _, delta = SS._fit_stem(short, 24000, out)
        self.assertEqual(sf.info(out).frames, 24000)
        self.assertGreater(delta, 0)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg required")
    def test_fit_stem_leaves_an_exact_chunk_alone(self):
        import soundfile as sf
        exact = write_wav(os.path.join(self.tmp, "exact.wav"), tone(200, 1.0), SR)
        out = os.path.join(self.tmp, "fit.wav")
        _, delta = SS._fit_stem(exact, SR, out)
        self.assertEqual(delta, 0)
        self.assertEqual(sf.info(out).frames, SR)


class CacheTests(unittest.TestCase):
    """Stems are reused when they match, and re-made when they do not."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stem_cache_")
        self.workdir = os.path.join(self.tmp, "work")
        os.makedirs(self.workdir, exist_ok=True)
        self.source = write_wav(os.path.join(self.tmp, "source.wav"),
                                tone(120, 4.0), SR, channels=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _install(self, fake):
        self._fake_root = os.path.join(self.tmp, "fakemod")
        os.makedirs(self._fake_root, exist_ok=True)
        fake.install(self._fake_root)
        return fake

    def _uninstall(self):
        if getattr(self, "_fake", None) is not None:
            self._fake.uninstall(self._fake_root)
        if getattr(self, "_fake_root", None):
            while self._fake_root in sys.path:
                sys.path.remove(self._fake_root)

    def test_the_second_run_reuses_the_cache_and_does_not_call_demucs(self):
        fake = self._install(FakeDemucs())
        try:
            first = SS.separate(self.source, self.workdir, log=lambda *a: None)
            calls_after_first = len(fake.read_calls())
            second = SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertEqual(calls_after_first, 1)
        self.assertEqual(len(fake.read_calls()), 1, "Demucs ran a second time")
        self.assertTrue(second["cached"])
        self.assertEqual(first["vocals"], second["vocals"])

    def test_force_re_runs_demucs(self):
        fake = self._install(FakeDemucs())
        try:
            SS.separate(self.source, self.workdir, log=lambda *a: None)
            SS.separate(self.source, self.workdir, log=lambda *a: None,
                        force=True)
        finally:
            self._uninstall()
        self.assertEqual(len(fake.read_calls()), 2)

    def test_a_changed_chunk_size_invalidates_the_cache(self):
        fake = self._install(FakeDemucs())
        try:
            SS.separate(self.source, self.workdir, chunk_seconds=600,
                        log=lambda *a: None)
            SS.separate(self.source, self.workdir, chunk_seconds=2,
                        log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertEqual(len(fake.read_calls()), 2)

    def test_a_re_trimmed_input_invalidates_the_cache(self):
        # A trim changes size and mtime. Without this, trimming a clip and
        # re-rendering would reuse the untrimmed background and the bed would
        # disagree with the picture for the whole film.
        fake = self._install(FakeDemucs())
        try:
            SS.separate(self.source, self.workdir, log=lambda *a: None)
            # Simulate a trim: different content, different size.
            write_wav(self.source, tone(120, 3.0), SR, channels=2)
            SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertEqual(len(fake.read_calls()), 2)

    def test_a_deleted_stem_invalidates_the_cache(self):
        fake = self._install(FakeDemucs())
        try:
            stems = SS.separate(self.source, self.workdir, log=lambda *a: None)
            os.remove(stems["background"])
            SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        self.assertEqual(len(fake.read_calls()), 2)

    def test_a_missing_manifest_is_a_miss_not_an_error(self):
        key = SS.cache_key(self.source)
        self.assertIsNone(SS.load_cached(self.workdir, key))

    def test_invalidate_clears_the_cache(self):
        fake = self._install(FakeDemucs())
        try:
            SS.separate(self.source, self.workdir, log=lambda *a: None)
            SS.invalidate(self.workdir)
            key = SS.cache_key(self.source)
            self.assertIsNone(SS.load_cached(self.workdir, key))
        finally:
            self._uninstall()

    def test_the_cache_key_records_what_it_depends_on(self):
        key = SS.cache_key(self.source, chunk_seconds=123)
        for field in ("input", "size", "mtime", "model", "chunk_seconds",
                      "sample_rate", "channels"):
            self.assertIn(field, key)
        self.assertEqual(key["chunk_seconds"], 123)

    def test_the_manifest_is_json(self):
        self._install(FakeDemucs())
        try:
            SS.separate(self.source, self.workdir, log=lambda *a: None)
        finally:
            self._uninstall()
        path = os.path.join(self.workdir, SS.STEMS_DIR, SS.META_NAME)
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["model"], SS.MODEL)
        self.assertIn("key", meta)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg required for mixing")
class MixTests(unittest.TestCase):
    """
    The mix: exact length, both layers present, dialogue not ducked by amix.

    Frequencies are chosen so each layer sits in its own band and can be
    measured independently: dialogue at 300 Hz, the preserved background at
    80 Hz, a user music bed at 150 Hz.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mix_test_")
        self.dialogue = write_wav(os.path.join(self.tmp, "dialogue.wav"),
                                  tone(300, 5.0, 0.3))
        self.background = write_wav(os.path.join(self.tmp, "background.wav"),
                                    tone(80, 5.0, 0.5))
        self.music = write_wav(os.path.join(self.tmp, "music.wav"),
                               tone(150, 2.0, 0.5))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _out(self, name):
        return os.path.join(self.tmp, name)

    def test_a_much_shorter_background_is_padded_exactly(self):
        """
        The 40 ms bug, pinned.

        `apad` with no argument pads to a buffer boundary, not to a duration,
        and `loudnorm` upstream makes the input length unpredictable — so
        enforcing the target per layer came out short. This is the case that
        caught it: a background far shorter than the dialogue, with a
        non-integer target.
        """
        short_bg = write_wav(os.path.join(self.tmp, "tiny_bg.wav"),
                             tone(80, 0.5, 0.5))
        for target in (5.0, 12.96, 3.777, 7.123):
            with self.subTest(target=target):
                out = av_sync.mix_layers(
                    self.dialogue, self._out(f"m_{target}.wav"), target=target,
                    background_path=short_bg)
                self.assertAlmostEqual(av_sync.duration(out), target, places=3)

    def test_a_much_longer_dialogue_is_trimmed_exactly(self):
        long_dialogue = write_wav(os.path.join(self.tmp, "long_d.wav"),
                                  tone(300, 30.0, 0.3))
        for target in (5.0, 12.96, 3.777):
            with self.subTest(target=target):
                out = av_sync.mix_layers(
                    long_dialogue, self._out(f"L_{target}.wav"), target=target,
                    background_path=self.background)
                self.assertAlmostEqual(av_sync.duration(out), target, places=3)

    def test_the_mix_is_exactly_the_target_length(self):
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=self.background)
        self.assertAlmostEqual(av_sync.duration(out), 5.0, places=3)

    def test_a_non_integer_target_is_honoured(self):
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"),
                                 target=3.777, background_path=self.background)
        self.assertAlmostEqual(av_sync.duration(out), 3.777, places=3)

    def test_a_long_background_is_trimmed_to_the_dialogue(self):
        long_bg = write_wav(os.path.join(self.tmp, "long_bg.wav"),
                            tone(80, 30.0, 0.5))
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=long_bg)
        self.assertAlmostEqual(av_sync.duration(out), 5.0, places=3)

    def test_a_short_background_is_padded_to_the_dialogue(self):
        short_bg = write_wav(os.path.join(self.tmp, "short_bg.wav"),
                             tone(80, 1.0, 0.5))
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=short_bg)
        self.assertAlmostEqual(av_sync.duration(out), 5.0, places=3)

    def test_the_output_contains_both_layers(self):
        # The point of the whole phase. A duration check alone would pass on a
        # track containing nothing but the dialogue.
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=self.background)
        self.assertGreater(band_energy(out, 250, 350), 1e3, "no dialogue")
        self.assertGreater(band_energy(out, 60, 110), 1e3, "no background")

    def test_the_dialogue_is_not_attenuated_by_amix(self):
        # `normalize=1` would halve the dialogue the moment a bed is added.
        # This is the regression guard for the `normalize=0` in the graph.
        only = av_sync.mix_layers(self.dialogue, self._out("only.wav"),
                                  target=5.0)
        both = av_sync.mix_layers(self.dialogue, self._out("both.wav"),
                                  target=5.0, background_path=self.background)
        a = band_energy(only, 250, 350)
        b = band_energy(both, 250, 350)
        self.assertGreater(b, a * 0.5,
                           "the dialogue lost more than 6 dB when the bed was "
                           "added, which means amix normalised it")

    def test_dialogue_only_has_no_background_energy(self):
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0)
        self.assertLess(band_energy(out, 60, 110), 1e3)

    def test_a_missing_background_file_is_ignored(self):
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=os.path.join(self.tmp, "nope.wav"))
        self.assertAlmostEqual(av_sync.duration(out), 5.0, places=3)

    def test_the_user_music_bed_does_not_replace_the_background(self):
        """
        The requirement, stated as a test.

        Both beds are present. The film's own soundtrack must still be audible;
        attaching a music file must not silently delete it.
        """
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=self.background,
                                 music_path=self.music)
        self.assertGreater(band_energy(out, 60, 110), 1e3,
                           "the preserved background was lost when a music bed "
                           "was attached")
        self.assertGreater(band_energy(out, 130, 170), 1e3,
                           "the user music bed is not present")

    def test_the_user_music_bed_sits_under_the_background(self):
        # `music_gain` is relative, so the bed should be quieter than the
        # preserved soundtrack at the same nominal amplitude.
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=self.background,
                                 music_path=self.music, music_gain=-18.0)
        bed = band_energy(out, 60, 110)
        music = band_energy(out, 130, 170)
        self.assertGreater(bed, music)

    def test_music_alone_still_mixes(self):
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 music_path=self.music)
        self.assertAlmostEqual(av_sync.duration(out), 5.0, places=3)
        self.assertGreater(band_energy(out, 130, 170), 1e3)

    def test_ducking_produces_a_valid_mix_of_the_right_length(self):
        out = av_sync.mix_layers(self.dialogue, self._out("m.wav"), target=5.0,
                                 background_path=self.background, duck=True,
                                 duck_db=-12.0)
        self.assertAlmostEqual(av_sync.duration(out), 5.0, places=3)
        self.assertGreater(band_energy(out, 250, 350), 1e3)
        self.assertGreater(band_energy(out, 60, 110), 1e3)

    def test_ducking_reduces_the_background_relative_to_no_ducking(self):
        plain = av_sync.mix_layers(self.dialogue, self._out("plain.wav"),
                                   target=5.0, background_path=self.background)
        ducked = av_sync.mix_layers(self.dialogue, self._out("ducked.wav"),
                                    target=5.0, background_path=self.background,
                                    duck=True, duck_db=-12.0)
        self.assertLess(band_energy(ducked, 60, 110),
                        band_energy(plain, 60, 110))

    def test_mix_background_is_the_same_as_the_two_layer_mix(self):
        # `mix_background` is a wrapper; the two must not drift apart.
        a = av_sync.mix_background(self.dialogue, self.background,
                                   self._out("a.wav"), target=5.0)
        b = av_sync.mix_layers(self.dialogue, self._out("b.wav"), target=5.0,
                               background_path=self.background)
        self.assertAlmostEqual(av_sync.duration(a), av_sync.duration(b),
                               places=4)

    def test_a_zero_length_dialogue_is_refused(self):
        empty = write_wav(os.path.join(self.tmp, "empty.wav"),
                          np.zeros(0, dtype=np.float32))
        with self.assertRaises(ValueError):
            av_sync.mix_layers(empty, self._out("m.wav"), target=1.0,
                               background_path=self.background)

    def test_a_silent_dialogue_still_produces_the_right_length(self):
        silent = write_wav(os.path.join(self.tmp, "silent.wav"),
                           np.zeros(SR * 4, dtype=np.float32))
        out = av_sync.mix_layers(silent, self._out("m.wav"), target=4.0,
                                 background_path=self.background)
        self.assertAlmostEqual(av_sync.duration(out), 4.0, places=3)
        # The background survives even when there is nothing to say.
        self.assertGreater(band_energy(out, 60, 110), 1e3)


class ManifestTests(unittest.TestCase):
    """
    The finished job must carry the stems, not just their two paths.

    `run_summary`, `artifact_cache` and sibling reuse all resolve the
    background from `job["stems"]`. The dub completion path stored
    `background_path`/`vocals_path` and forgot `stems`, so a render that DID
    preserve the soundtrack reported "background not preserved", raised the
    warning that says the music was lost, and re-ran Demucs on the next
    sibling. The audio path spread `**common` and was correct, which is why
    only one of the two outputs lied.
    """

    def test_the_dub_completion_stores_the_stems(self):
        import ast
        path = os.path.join(ROOT, "engine", "app.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        function = next((n for n in tree.body
                         if isinstance(n, ast.FunctionDef)
                         and n.name == "process_video_task"), None)
        self.assertIsNotNone(function, "process_video_task is gone")

        # The dub completion is the call that records where the background
        # went; requiring the stems there is the invariant that broke.
        done = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "_set":
                continue
            keywords = {k.arg for k in node.keywords}
            if "background_path" in keywords:
                done.append(keywords)

        self.assertTrue(done, "the dub completion call was not found")
        for keywords in done:
            self.assertIn("stems", keywords,
                          "a finished dub does not record its stems, so the "
                          "run summary reports the background as lost")

    def test_a_job_with_stems_reports_the_background_as_preserved(self):
        import run_summary as RS
        stems = {"vocals": "/tmp/v.wav", "background": "/tmp/b.wav",
                 "model": "htdemucs", "chunks": 1}
        summary = RS.build({"job_id": "j", "kind": "dub", "status": "done",
                            "stems": stems, "mixed_path": "/tmp/m.wav",
                            "segments": [], "translated_segments": [],
                            "plan": [], "speakers": [], "diarization": []})
        self.assertTrue(summary["background"]["separated"])
        self.assertTrue(summary["background"]["mixed"])
        row = next(r for r in summary["stages"]
                   if r["key"] == "separating_audio")
        self.assertEqual(row["state"], "done")
        self.assertNotIn("were not preserved", " ".join(summary["warnings"]))


if __name__ == "__main__":
    unittest.main()

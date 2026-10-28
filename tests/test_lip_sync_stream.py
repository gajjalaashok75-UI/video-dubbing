"""
`lip_sync` must never hold the whole clip.

A 60-second 1080x1920 dub is 1,799 frames — 10.4 GiB decoded — which is more
memory than a hosted runtime has, so the render was killed and the session
with it. These tests pin the streaming behaviour, and they prove the change
did not alter the picture: the bytes written are identical to the in-memory
algorithm it replaces.

No models, GPU or network; nobody needs wav2lip_gan.pth to run this.
"""

import os
import sys
import tempfile
import unittest
import weakref

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

try:
    import cv2
except ImportError:                                          # pragma: no cover
    cv2 = None

try:
    import torch
except ImportError:                                          # pragma: no cover
    torch = None

if cv2 is not None:
    import lip_sync

W, H = 160, 120
PADS = (0, 12, 0, 0)
BATCH = 16


def make_clip(path: str, n_frames: int, fps: float = 10.0) -> None:
    """A small clip whose frames are all different, so a mix-up is visible."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    if not writer.isOpened():                                # pragma: no cover
        raise unittest.SkipTest("no MJPG encoder available")
    for i in range(n_frames):
        frame = np.full((H, W, 3), (i * 7) % 255, dtype=np.uint8)
        frame[30:70, 40:100] = ((i * 11) % 255, 120, 60)
        writer.write(frame)
    writer.release()


class FakeDetector:
    """s3fd, replaced by a fixed box in the downscaled frame's coordinates."""

    RECT = (40, 30, 100, 70)

    def __init__(self):
        self.frames = 0

    def get_detections_for_batch(self, batch):
        self.frames += len(batch)
        return [self.RECT for _ in range(len(batch))]


class FakeModel:
    """Wav2Lip's generator: anything deterministic will do."""

    def __call__(self, mel, img):
        n = img.shape[0]
        return torch.full((n, 3, lip_sync.IMG_SIZE, lip_sync.IMG_SIZE),
                          255.0, dtype=torch.float32)


class FakeWriter:
    """Stands in for the ffmpeg pipe, keeping the raw frames for comparison."""

    def __init__(self):
        self.frames = []
        self.stdin = self
        self.closed = False

    def write(self, payload):
        self.frames.append(payload)

    def close(self):
        self.closed = True

    def wait(self):
        return 0

    def poll(self):
        return 0

    def kill(self):                                          # pragma: no cover
        pass


class TrackingCapture:
    """A VideoCapture that remembers (weakly) every frame it handed out."""

    def __init__(self, cap):
        self._cap = cap
        self.alive = []

    def read(self):
        ok, frame = self._cap.read()
        if ok:
            self.alive.append(weakref.ref(frame))
        return ok, frame

    def get(self, prop):
        return self._cap.get(prop)

    def isOpened(self):
        return self._cap.isOpened()

    def release(self):
        self._cap.release()


@unittest.skipIf(cv2 is None or torch is None,
                 "needs OpenCV and torch (neither model is used)")
class StreamingTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clip = os.path.join(self.tmp.name, "clip.avi")
        self.out = os.path.join(self.tmp.name, "out.mp4")
        self.audio = os.path.join(self.tmp.name, "dub.wav")

    # -- harness ------------------------------------------------------------
    def runner(self):
        runner = lip_sync.Wav2LipRunner()
        # Bypass load(): the weights are not part of this test.
        runner._model = FakeModel()
        runner.device = "cpu"
        runner._detector = FakeDetector()
        # The mel comes from the vendored Wav2Lip audio module, which needs
        # librosa; the frame pipeline only needs the right shape.
        runner._mel_chunks = lambda wav, n, fps: [
            np.zeros((80, lip_sync.MEL_STEP_SIZE), dtype=np.float32)
            for _ in range(n)]
        return runner

    def captured_writer(self):
        writer = FakeWriter()
        original = lip_sync._open_writer
        lip_sync._open_writer = lambda *a, **k: writer
        self.addCleanup(setattr, lip_sync, "_open_writer", original)
        return writer

    def tracking(self):
        real = cv2.VideoCapture
        made = []

        def factory(path, *a, **k):
            cap = TrackingCapture(real(path, *a, **k))
            made.append(cap)
            return cap

        cv2.VideoCapture = factory
        self.addCleanup(setattr, cv2, "VideoCapture", real)
        return made

    def boxes(self, runner, n_frames):
        n, size, scale, _fps, idx, raw = runner._scan_and_detect(
            self.clip, PADS, BATCH, lip_sync.DETECT_EVERY)
        self.assertEqual(n, n_frames)
        return runner._boxes_from_detections(raw, idx, n, size, scale, PADS)

    # -- the old algorithm, kept as an oracle -------------------------------
    @staticmethod
    def in_memory(path, boxes, mel_chunks, blend, batch):
        """What `run` did before: decode everything, then batch over the list."""
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        out = []
        for i in range(0, len(frames), batch):
            sl = slice(i, i + batch)
            batch_frames, batch_boxes, batch_mels = frames[sl], boxes[sl], mel_chunks[sl]

            faces = []
            for frame, (y1, y2, x1, x2) in zip(batch_frames, batch_boxes, strict=True):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = frame
                faces.append(cv2.resize(crop, (lip_sync.IMG_SIZE, lip_sync.IMG_SIZE)))

            img_batch = np.asarray(faces, dtype=np.uint8)
            masked = img_batch.copy()
            masked[:, lip_sync.IMG_SIZE // 2:] = 0
            net_in = np.concatenate((masked, img_batch), axis=3) / 255.0
            mel_in = np.asarray(batch_mels)[..., np.newaxis]

            t_img = torch.FloatTensor(np.transpose(net_in, (0, 3, 1, 2)))
            t_mel = torch.FloatTensor(np.transpose(mel_in, (0, 3, 1, 2)))
            with torch.no_grad():
                pred = FakeModel()(t_mel, t_img)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for p, frame, (y1, y2, x1, x2) in zip(pred, batch_frames, batch_boxes,
                                                  strict=True):
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    out.append(frame.tobytes())
                    continue
                mouth = cv2.resize(p.astype(np.uint8), (bw, bh))
                mask = cv2.resize(blend, (bw, bh))[..., None]
                region = frame[y1:y2, x1:x2].astype(np.float32)
                frame[y1:y2, x1:x2] = (
                    mouth.astype(np.float32) * mask + region * (1.0 - mask)
                ).astype(np.uint8)
                out.append(frame.tobytes())
        return out

    # -- tests --------------------------------------------------------------
    def test_output_is_identical_to_holding_every_frame(self):
        make_clip(self.clip, 40)
        writer = self.captured_writer()
        runner = self.runner()

        boxes = self.boxes(runner, 40)
        runner.run(self.clip, self.audio, self.out,
                   wav2lip_batch_size=BATCH, face_det_batch_size=BATCH)

        mel = runner._mel_chunks(self.audio, 40, 10.0)
        blend = lip_sync._edge_mask(lip_sync.IMG_SIZE, 10)
        expected = self.in_memory(self.clip, boxes, mel, blend, BATCH)

        self.assertEqual(len(writer.frames), 40)
        self.assertEqual(writer.frames, expected)

    def test_frames_are_released_batch_by_batch(self):
        make_clip(self.clip, 120)
        self.captured_writer()
        runner = self.runner()
        captures = self.tracking()

        runner.run(self.clip, self.audio, self.out,
                   wav2lip_batch_size=BATCH, face_det_batch_size=BATCH)

        render = captures[-1]
        self.assertGreaterEqual(len(render.alive), 120)
        alive = sum(1 for ref in render.alive if ref() is not None)
        # Holding the clip would leave all 120 alive; a batch at a time leaves
        # the batch being rendered and little else.
        self.assertLess(alive, 3 * BATCH,
                        f"{alive} of 120 decoded frames are still alive")

    def test_scan_samples_every_third_frame_and_the_last(self):
        make_clip(self.clip, 40)
        runner = self.runner()
        n, (h, w), scale, _fps, sampled, raw = runner._scan_and_detect(
            self.clip, PADS, BATCH, lip_sync.DETECT_EVERY)

        self.assertEqual(n, 40)
        self.assertEqual((h, w), (H, W))
        self.assertEqual(len(raw), len(sampled))
        self.assertEqual(sampled[-1], n - 1)          # 39 is not a multiple of 3
        self.assertEqual(sampled[:-1], list(range(0, n - 1, 3)))
        self.assertEqual(len(sampled), 14)

    def test_a_clip_with_no_face_is_refused(self):
        make_clip(self.clip, 12)
        runner = self.runner()
        runner._detector = type("NoFaces", (), {
            "get_detections_for_batch": lambda self, batch: [None] * len(batch)})()

        with self.assertRaises(lip_sync.LipSyncError) as caught:
            runner._render(self.clip, self.audio, self.out, PADS, BATCH, BATCH,
                           lip_sync.DETECT_EVERY, 18, "medium", 10, None)
        self.assertIn("No face detected", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

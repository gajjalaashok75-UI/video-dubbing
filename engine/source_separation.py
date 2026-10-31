"""
Source separation: keep the film's music and sound effects, replace the speech.

The dub used to be the only audio in the output, so a dubbed film lost its
score, its ambience and every sound effect the moment it was rendered. This
module runs Demucs' two-stem separation to split the original soundtrack into
`vocals` and `no_vocals`, and the `no_vocals` stem becomes the bed the new
dialogue is laid over.

Three things here are deliberate and each was measured rather than assumed:

1. **The input is not the 16 kHz mono ASR track.** Demucs is trained on
   full-band stereo music, and the repo's own `reference_audio` docstring
   already argues that everything above 8 kHz is sibilance and breathiness.
   Feeding it a band-limited mono downmix throws away most of what it uses to
   decide what is a voice. A separate HQ extraction is made for it.

2. **A movie is processed in chunks.** One two-hour file held whole is what
   OOMs a machine, and this project has already been bitten by exactly that
   (see the 10.4 GiB frame-list note in AGENTS.md). 10-minute chunks keep peak
   memory flat.

3. **Chunk boundaries are verified, not trusted.** `ffmpeg -f segment -c copy`
   on PCM WAV was measured to concatenate back sample-exactly (0 mismatched
   samples over a 12s ramp at 44.1 kHz stereo), so the *segmentation* is safe.
   Demucs is the risk: a model that pads its input to a multiple of some window
   returns a different number of frames per chunk, and concatenating those
   produces a drift that grows with the number of chunks — audible as the
   dialogue sliding out of sync over a long film. Every chunk's output is
   therefore checked against the chunk's input length, and a mismatch is
   corrected by trimming or padding the *stem*, never by changing the timeline.

Everything in here is optional and failure-safe. No Demucs, no weights, a
crash, a timeout, or malformed output all return None, and the caller falls back
to the dialogue-only pipeline this project shipped before. A dub must never fail
because separation failed.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Demucs is slow, and on CPU it is very slow. 10 minutes is the chunk size the
# reference implementation settled on, and it is a reasonable compromise: large
# enough that the per-chunk overhead disappears, small enough that a single
# chunk's tensors fit comfortably in RAM.
CHUNK_SECONDS = 600

# The model. `htdemucs` is the hybrid transformer Demucs, and it is what both
# `video-dubbing-system` and `Python-Autodub` use. The default `--two-stems
# vocals` split is what produces the `no_vocals` bed.
MODEL = "htdemucs"

# Where the stems land inside a job's workdir.
STEMS_DIR = "stems"
VOCALS_NAME = "vocals.wav"
BACKGROUND_NAME = "background.wav"
META_NAME = "stems.json"

# The HQ extraction Demucs is fed. 44.1 kHz stereo is the rate Demucs' own
# training data uses; matching it means no resampling inside the model.
HQ_SAMPLE_RATE = 44100
HQ_CHANNELS = 2

# A separation that produces less than this share of its input is malformed.
MIN_DURATION_RATIO = 0.9

# How long Demucs may run before it is killed. A full-length film on CPU is
# genuinely slow, so this is generous; it exists to stop a wedged process from
# holding a job forever, not to bound normal work.
TIMEOUT_SECONDS = int(os.getenv("DEMUCS_TIMEOUT", str(6 * 60 * 60)))

_STEMS = dict  # {"vocals": path, "background": path, "sample_rate": int, ...}


# What `python -m demucs.separate` imports before it does any work:
# separate.py (dora.log, torch, torchaudio), audio.py (lameenc, julius, numpy),
# states.py (omegaconf), pretrained.py (dora.log), apply.py (tqdm).
# `dora-search` provides the `dora` module; lameenc and julius are separate
# wheels. Demucs itself is installed with `--no-deps` so its old tree cannot
# rewrite torch or numpy, which is exactly why these have to be named here.
REQUIRED_MODULES = ("demucs.separate", "lameenc", "julius", "openunmix", "dora",
                    "omegaconf", "tqdm")


def missing_deps() -> list:
    """The separation command's imports this interpreter cannot resolve."""
    missing = []
    for name in REQUIRED_MODULES:
        try:
            import importlib.util
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except Exception:                                    # noqa: BLE001
            missing.append(name)
    return missing


def available() -> bool:
    """
    Whether Demucs can actually run here, cheaply and without loading anything.

    Checked by import spec rather than by running `demucs --help`, because this
    is called on the request path and shelling out to answer a yes/no question
    is a needless way to make a dub feel slow.

    But one spec was not enough: `find_spec("demucs")` answered yes for an
    install whose `python -m demucs.separate` died on `import lameenc`, and the
    Setup page called background preservation ready while every render dropped
    the soundtrack. The question asked here is whether the command's imports
    resolve, not whether one package directory exists.
    """
    return not missing_deps()


def _run(cmd: list, timeout: int = None) -> subprocess.CompletedProcess:
    """
    Run a command, capturing output, and raise a useful error on failure.

    `PYTHONPATH` is forwarded explicitly. Demucs is invoked as
    `python -m demucs.separate`, and a subprocess does not inherit the parent's
    `sys.path` — so anything that put a package there at runtime (a vendored
    copy, a test fixture, a deployment that ships its own wheel) would be found
    by `available()` and then missing in the child. That mismatch produced a
    confusing "not installed" error from a process that had just confirmed it
    was.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    paths = [p for p in sys.path if p]
    env["PYTHONPATH"] = os.pathsep.join(
        paths + ([existing] if existing else []))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Demucs timed out after {timeout}s. Raise DEMUCS_TIMEOUT if this "
            f"machine is simply slow.") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{cmd[0]} not found. ffmpeg is required for separation.") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): "
                           + " | ".join(tail))
    return proc


# ---------------------------------------------------------------------------
# HQ extraction
# ---------------------------------------------------------------------------
def extract_hq(input_path: str, out_path: str,
               sample_rate: int = HQ_SAMPLE_RATE,
               channels: int = HQ_CHANNELS) -> str:
    """
    Full-band stereo PCM for Demucs.

    Deliberately not `av_sync.extract_audio`, which produces the 16 kHz mono
    track Whisper wants. Those settings are wrong for a separation model: the
    bandwidth is gone and the stereo image it uses to place a voice in the mix
    has been thrown away. The two extractions coexist; the 16 kHz one is still
    made, because ASR falls back to it whenever separation is unavailable.
    """
    _run(["ffmpeg", "-y", "-nostats", "-loglevel", "error",
          "-i", input_path,
          "-vn",
          "-acodec", "pcm_s16le",
          "-ar", str(sample_rate),
          "-ac", str(channels),
          out_path])
    return out_path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _split(hq_path: str, chunk_dir: str,
           chunk_seconds: int = CHUNK_SECONDS) -> list:
    """
    Slice the HQ audio into fixed-length WAV chunks.

    Stream-copied rather than re-encoded, so the samples are bit-identical to
    the source. Measured: concatenating these chunks back reproduces the input
    sample-for-sample, so the segmentation itself introduces no drift.
    """
    os.makedirs(chunk_dir, exist_ok=True)
    _run(["ffmpeg", "-y", "-nostats", "-loglevel", "error",
          "-i", hq_path,
          "-f", "segment",
          "-segment_time", str(chunk_seconds),
          "-c", "copy",
          os.path.join(chunk_dir, "chunk_%04d.wav")])
    chunks = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.wav")))
    if not chunks:
        raise RuntimeError("ffmpeg produced no chunks")
    return chunks


def _frames(path: str) -> int:
    """Frame count of an audio file, from its header."""
    import soundfile as sf
    return int(sf.info(path).frames)


def _fit_stem(path: str, want_frames: int, out_path: str) -> tuple:
    """
    Force one Demucs output to the exact frame count of its input chunk.

    Returns `(out_path, delta)` where delta is frames removed (negative) or
    added (positive), so the caller can report a real number.

    Demucs normally returns the same length it was given, but "normally" is not
    a guarantee, and the failure is silent: a chunk that comes back 2,048 frames
    long at 44.1 kHz is 46 ms, and across a 12-chunk film that is half a second
    of accumulating drift between the new dialogue and the picture. Correcting
    each chunk against its own input keeps the error bounded at zero no matter
    how many chunks there are.
    """
    import numpy as np
    import soundfile as sf

    # The destination directory is created up front. It used to be created only
    # on the write path, so an exact-length chunk — the common case — took the
    # copy path into a directory that did not exist yet and failed with a
    # bare Errno 2.
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    have = wave.shape[0]
    delta = have - want_frames

    if have == want_frames:
        if os.path.abspath(path) != os.path.abspath(out_path):
            shutil.copyfile(path, out_path)
        return out_path, 0

    if have > want_frames:
        wave = wave[:want_frames]
    else:
        pad = np.zeros((want_frames - have, wave.shape[1]), dtype=np.float32)
        wave = np.concatenate([wave, pad], axis=0)

    sf.write(out_path, np.ascontiguousarray(wave), sr, subtype="PCM_16")
    # Negative means frames were removed, positive means they were added, so
    # the sign describes what was done to the stem rather than the raw
    # arithmetic difference. `delta` is `have - want`, which has the opposite
    # sign in both cases.
    return out_path, -delta


def _concat(parts: list, out_path: str) -> str:
    """
    Join stem chunks losslessly, with an explicit concat list.

    The `concat` demuxer with `-c copy` is what the reference implementation
    uses and it is correct here: these are all PCM WAV at one rate and channel
    count, so the join is a straight sample splice.
    """
    listing = out_path + ".concat.txt"
    with open(listing, "w", encoding="utf-8") as f:
        for part in parts:
            # ffmpeg's concat demuxer treats a single quote as an escape, so a
            # path containing one has to be quoted the way it expects. This is
            # the classic concat-list bug and it only shows up on somebody
            # else's machine.
            safe = os.path.abspath(part).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    try:
        _run(["ffmpeg", "-y", "-nostats", "-loglevel", "error",
              "-f", "concat", "-safe", "0",
              "-i", listing,
              "-c", "copy",
              out_path])
    finally:
        try:
            os.remove(listing)
        except OSError:
            pass
    return out_path


# ---------------------------------------------------------------------------
# The separation itself
# ---------------------------------------------------------------------------
def _separate_chunks(chunks: list, out_dir: str, device: str,
                     log=print) -> tuple:
    """
    Run Demucs over every chunk in ONE invocation and collect the stems.

    One invocation rather than one per chunk: Demucs loads its model once and
    then streams the files, so this is the difference between loading a
    ~300 MB checkpoint once and loading it twelve times.

    `-j 1` keeps it sequential, which is what bounds peak memory — the parallel
    workers each hold their own tensors, and on a machine with 15 GB of RAM
    that is the difference between finishing and being killed.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs.separate",
           "-n", MODEL,
           "--two-stems", "vocals",
           "-j", "1",
           "-o", out_dir]
    if device:
        cmd += ["-d", device]
    cmd += chunks

    log(f"[STEM] demucs {' '.join(cmd[2:])}")
    _run(cmd, timeout=TIMEOUT_SECONDS)

    vocals, background = [], []
    for chunk in chunks:
        stem = os.path.splitext(os.path.basename(chunk))[0]
        base = os.path.join(out_dir, MODEL, stem)
        v = os.path.join(base, "vocals.wav")
        b = os.path.join(base, "no_vocals.wav")
        if not os.path.exists(v):
            found = glob.glob(os.path.join(out_dir, MODEL, "**", "vocals.wav"),
                              recursive=True)
            raise RuntimeError(
                f"Demucs produced no vocals stem for {stem} "
                f"(found {len(found)} vocals.wav under {out_dir}/{MODEL})")
        vocals.append(v)
        background.append(b if os.path.exists(b) else None)
    return vocals, background


def separate(input_path: str, workdir: str, device: str = None,
             chunk_seconds: int = CHUNK_SECONDS, log=print,
             force: bool = False) -> dict | None:
    """
    Split `input_path` into a vocals stem and a background stem.

    Returns `{"vocals", "background", "sample_rate", "duration", "chunks",
    "model", "chunk_seconds", "key"}` or None when separation is unavailable or
    fails. Never raises: every failure is a warning and a None, so the caller's
    dialogue-only path is always still available.

    The stems are written to `<workdir>/stems/` and cached there. `force`
    re-runs regardless; otherwise a cache that matches the requested settings is
    reused and Demucs is not started at all.
    """
    stems_dir = os.path.join(workdir, STEMS_DIR)
    key = cache_key(input_path, chunk_seconds)
    meta_path = os.path.join(stems_dir, META_NAME)

    if not force:
        cached = load_cached(workdir, key)
        if cached:
            log(f"[STEM] Reusing cached stems for {os.path.basename(input_path)}")
            return cached

    if not available():
        log("[STEM] Background preservation unavailable:\n"
            "       Demucs is not installed.\n"
            "       The job will continue with dialogue-only audio.\n"
            "       Install with:  pip install 'demucs>=4.0.0'")
        return None

    try:
        os.makedirs(stems_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="demucs_", dir=workdir) as tmp:
            hq = os.path.join(tmp, "hq.wav")
            extract_hq(input_path, hq)
            total_frames = _frames(hq)
            log(f"[STEM] HQ source: {total_frames} frames @ "
                f"{HQ_SAMPLE_RATE} Hz, {HQ_CHANNELS}ch")

            chunk_dir = os.path.join(tmp, "chunks")
            chunks = _split(hq, chunk_dir, chunk_seconds)
            log(f"[STEM] {len(chunks)} chunk(s) of up to {chunk_seconds}s")

            vocals, background = _separate_chunks(
                chunks, os.path.join(tmp, "out"), device, log)

            # Fit every chunk back to its own input length BEFORE joining, so
            # a model that padded cannot shift everything after it.
            fit_dir = os.path.join(tmp, "fit")
            v_parts, b_parts, adjusted = [], [], 0
            for i, (chunk, v, b) in enumerate(zip(chunks, vocals, background,
                                                  strict=True)):
                want = _frames(chunk)
                v_out = os.path.join(fit_dir, f"v{i:04d}.wav")
                _, dv = _fit_stem(v, want, v_out)
                v_parts.append(v_out)
                if dv:
                    adjusted += 1
                    log(f"[STEM] chunk {i}: vocals off by {dv:+d} frames, "
                        f"corrected")
                if b:
                    b_out = os.path.join(fit_dir, f"b{i:04d}.wav")
                    _, db = _fit_stem(b, want, b_out)
                    b_parts.append(b_out)
                    if db:
                        adjusted += 1
                        log(f"[STEM] chunk {i}: background off by {db:+d} "
                            f"frames, corrected")
            if adjusted == 0:
                log("[STEM] every chunk came back the exact length it went in")

            vocals_path = os.path.join(stems_dir, VOCALS_NAME)
            background_path = os.path.join(stems_dir, BACKGROUND_NAME)
            _concat(v_parts, vocals_path)
            _concat(b_parts, background_path)

            v_frames, b_frames = _frames(vocals_path), _frames(background_path)
            if v_frames != total_frames or b_frames != total_frames:
                raise RuntimeError(
                    f"reassembled stems do not match the source length "
                    f"(vocals {v_frames}, background {b_frames}, "
                    f"source {total_frames})")
            log(f"[STEM] stems reassembled at exactly {total_frames} frames "
                f"({total_frames / HQ_SAMPLE_RATE:.3f}s)")

            meta = {
                "key": key,
                "model": MODEL,
                "chunk_seconds": chunk_seconds,
                "sample_rate": HQ_SAMPLE_RATE,
                "channels": HQ_CHANNELS,
                "frames": total_frames,
                "duration": round(total_frames / HQ_SAMPLE_RATE, 6),
                "chunks": len(chunks),
                "chunks_adjusted": adjusted,
                "vocals": vocals_path,
                "background": background_path,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            log(f"[STEM] background stem kept: {background_path}")
            return {**meta, "cached": False}
    except Exception as e:                                   # noqa: BLE001
        log(f"[STEM] Background preservation unavailable:\n"
            f"       {e}\n"
            f"       The job will continue with dialogue-only audio.")
        # A half-written stem is worse than none: the next run would reuse it.
        for name in (VOCALS_NAME, BACKGROUND_NAME, META_NAME):
            try:
                os.remove(os.path.join(stems_dir, name))
            except OSError:
                pass
        return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def cache_key(input_path: str, chunk_seconds: int = CHUNK_SECONDS) -> dict:
    """
    What the stems correspond to.

    Stored with them so a stale cache can be told apart from a valid one. A
    trim changes the file's size and mtime, so a re-cut clip invalidates its
    stems automatically — without this, trimming a video and re-rendering would
    silently reuse the untrimmed background and the picture and the bed would
    disagree for the whole film.
    """
    try:
        st = os.stat(input_path)
        size, mtime = st.st_size, round(st.st_mtime, 3)
    except OSError:
        size, mtime = None, None
    return {
        "input": os.path.abspath(input_path),
        "size": size,
        "mtime": mtime,
        "model": MODEL,
        "chunk_seconds": int(chunk_seconds),
        "sample_rate": HQ_SAMPLE_RATE,
        "channels": HQ_CHANNELS,
    }


def load_cached(workdir: str, key: dict) -> dict | None:
    """
    The stems for this exact input, or None.

    Valid means: the metadata matches the key on every field, and both stem
    files are on disk. Anything else — a different model, a different chunk
    size, a re-trimmed input — is a miss and Demucs runs again.
    """
    stems_dir = os.path.join(workdir, STEMS_DIR)
    meta_path = os.path.join(stems_dir, META_NAME)
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:                                        # noqa: BLE001
        return None

    stored = meta.get("key") or {}
    for field, want in key.items():
        if stored.get(field) != want:
            return None
    for field in ("vocals", "background"):
        path = meta.get(field)
        if not path or not os.path.exists(path):
            return None
    return {**meta, "cached": True}


def invalidate(workdir: str) -> None:
    """Forget the cached stems, so the next separation runs Demucs again."""
    stems_dir = os.path.join(workdir, STEMS_DIR)
    for name in (VOCALS_NAME, BACKGROUND_NAME, META_NAME):
        try:
            os.remove(os.path.join(stems_dir, name))
        except OSError:
            pass


def release() -> None:
    """
    Free whatever Demucs left behind.

    Demucs runs as a subprocess, so its memory goes with it; this exists for the
    symmetry with `speaker_diarization.release()` and to drop any cached CUDA
    context a library-level call may have left in this process.
    """
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        pass


def summarize(stems: dict) -> str:
    """One line for the log."""
    if not stems:
        return "[STEM] No separation; the dub is the only audio"
    return (f"[STEM] {stems.get('chunks')} chunk(s), model {stems.get('model')}, "
            f"{stems.get('duration', 0.0):.2f}s background preserved"
            + (" (cached)" if stems.get("cached") else ""))

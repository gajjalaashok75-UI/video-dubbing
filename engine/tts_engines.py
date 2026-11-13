"""
TTS engine layer — warm, in-process, local.

The old design spawned a fresh Python subprocess per request, which reloaded
torch, the OpenVoice converter, and (via OpenVoice's se_extractor) an entire
extra Whisper model on every single job. It also called edge-tts, which is a
network round-trip to Azure and therefore neither local nor reliable.

This module replaces that with warm singletons held in the API process:

  IndicF5Engine  ai4bharat/IndicF5 — 11 Indian languages, zero-shot cloning
                 directly from a reference clip + its transcript. Single stage:
                 no stock voice to repaint, so the speaker identity comes from
                 the reference rather than from a tone-color transfer.

  XTTSEngine     Coqui XTTS-v2 — cross-lingual cloning for foreign languages
                 (phase 2). Loaded lazily; never touched for Indian targets.
                 NOTE: Coqui Public Model License is non-commercial.

Both are pure-local: weights come from the HF cache, nothing hits the network
at inference time.
"""

import os
import sys
import threading

import numpy as np

import languages as L

# Put IndicF5's own f5_tts fork ahead of the pip `f5-tts` release, at import
# time so nothing can bind the site-packages copy first. See
# _use_vendored_f5_tts for why the two are not interchangeable.
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(os.path.join(_VENDOR_DIR, "f5_tts")):
    sys.path.insert(0, _VENDOR_DIR)

_MPS_FALLBACK_SET = False


def _pick_device() -> str:
    forced = os.getenv("FORCE_DEVICE")
    if forced:
        return forced
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_mps_fallback():
    """Some F5/vocos ops have no MPS kernel; let torch fall back silently."""
    global _MPS_FALLBACK_SET
    if not _MPS_FALLBACK_SET:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        _MPS_FALLBACK_SET = True


def _free_vram_gb() -> float | None:
    """
    Free device memory in GB, or None when it cannot be read.

    Used to refuse a load that would not fit rather than let the allocator OOM
    the whole kernel. Returns None on CPU/MPS or when the driver query fails, so
    the caller treats "unknown" as "do not block".
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:                                        # noqa: BLE001
        return None


class SynthesisError(RuntimeError):
    pass


# S2 Pro's bf16 weights are ~11 GB and the load peak is higher still. Below
# this much free device memory the load cannot finish without starving the
# models already resident.
_S2PRO_MIN_FREE_GB = 12.0


def _require_room_for_s2pro(device: str) -> None:
    """
    Raise `SynthesisError` when S2 Pro cannot fit on `device`.

    On a free T4 (14.56 GiB) a load that leaves no headroom takes Whisper and
    Wav2Lip down with it - the whole session dies with an allocator error that
    reads like a platform crash. Refusing up front, with a message that names
    the fix, turns that into a clean per-engine failure. A device that is not
    CUDA, or whose free memory cannot be read, is never blocked.
    """
    if device != "cuda":
        return
    free = _free_vram_gb()
    if free is not None and free < _S2PRO_MIN_FREE_GB:
        raise SynthesisError(
            f"S2 Pro needs about {_S2PRO_MIN_FREE_GB:.0f} GB of free GPU memory "
            f"but only {free:.1f} GB is free. It will not fit beside the other "
            f"models on this GPU. Use a larger GPU, or set the voice engine to "
            f"Automatic (IndicF5) for the Indian languages."
        )


_TORCHAUDIO_PATCHED = False


def _patch_torchaudio_load():
    """
    Give `torchaudio.load` a soundfile-backed implementation.

    torchaudio 2.11 dropped its own decoding backends and routes `load` through
    TorchCodec, which is not installed here — so any library calling it dies
    with "TorchCodec is required for load_with_torchcodec". IndicF5 could be
    routed around it (we hand `infer_batch_process` a tensor directly), but XTTS
    calls it deep inside its own inference path, so the only options are to
    install torchcodec or to supply the function torchaudio used to have.

    soundfile is already a dependency and handles every format we feed it.
    """
    global _TORCHAUDIO_PATCHED
    if _TORCHAUDIO_PATCHED:
        return

    import torch
    import torchaudio

    try:
        torchaudio.load(__file__)  # cheap probe: fails either way, but how?
    except Exception as e:
        if "TorchCodec" not in str(e):
            _TORCHAUDIO_PATCHED = True   # a working backend exists
            return

    import soundfile as sf

    def _load(uri, frame_offset=0, num_frames=-1, normalize=True,
              channels_first=True, **_):
        data, sr = sf.read(str(uri), dtype="float32", always_2d=True,
                           start=int(frame_offset),
                           frames=(-1 if num_frames in (-1, None) else int(num_frames)))
        t = torch.from_numpy(data)                    # (frames, channels)
        return (t.T.contiguous() if channels_first else t), sr

    torchaudio.load = _load
    _TORCHAUDIO_PATCHED = True
    print("[TTS] Patched torchaudio.load to use soundfile (no torchcodec)")


# ---------------------------------------------------------------------------
# IndicF5
# ---------------------------------------------------------------------------
def _use_vendored_f5_tts():
    """
    Fail fast if the pip `f5-tts` release shadowed IndicF5's own fork.

    The two are NOT interchangeable, even though both export `CFM`/`DiT` and
    accept the checkpoint's 447 tensors with zero missing and zero unexpected
    keys. The parameter *names* match; the forward math does not:

      * `DiT.norm_out` is `AdaLayerNormZero_Final` in the fork but
        `AdaLayerNorm_Final` in the release (same weights, different layer), and
        the fork's `TextEmbedding` pads text without the release's `mask_padding`
        zeroing.
      * `CFM.sample` derives `duration` from text length in the fork; the
        release folds it into the transformer call and adds an EPSS timestep
        schedule and a cond/uncond cache.
      * `ConvolutionalPositionalEmbedding.forward` masks after each conv in the
        release but once at the end in the fork.

    Loading the checkpoint into the release therefore does not error — it
    produces fluent-sounding audio that is the wrong words. Measured on the
    model's own reference prompt and text, the release transcribed as
    "कर दो कर दो कर दो" while the fork reproduced the sentence exactly.

    The vendor directory is placed on sys.path at module import, so by the time
    this runs the fork should already have won. If something imported the pip
    release first, evict it and re-bind from the vendor copy; if that still
    yields the release, raise rather than silently generating the wrong words.
    """
    import f5_tts.model.backbones.dit as _dit
    if hasattr(_dit, "AdaLayerNormZero_Final"):
        return

    for name in [m for m in sys.modules if m == "f5_tts" or m.startswith("f5_tts.")]:
        del sys.modules[name]
    import f5_tts.model.backbones.dit as _dit
    if not hasattr(_dit, "AdaLayerNormZero_Final"):
        raise SynthesisError(
            "f5_tts resolved to the pip release, not IndicF5's vendored fork "
            f"({_dit.__file__}). The release loads the checkpoint without error "
            "but generates the wrong words. Expected engine/vendor/f5_tts to be "
            "first on sys.path. Fetch it with 'python scripts/vendor_f5_tts.py' "
            "from the repo root, then restart the app; run.sh and the "
            "notebooks do this in their setup step.")


def _build_indicf5_container(device: str = "cpu"):
    """
    Instantiate IndicF5's two modules from the vendored f5_tts fork directly.

    The repository's own `model.py` cannot be used here, for three reasons that
    each stop it dead on this stack:

      1. `AutoModel.from_pretrained(..., trust_remote_code=True)` now builds the
         model inside an `init_empty_weights()` context (transformers >= 4.55),
         so every parameter starts on the meta device. The remote `__init__`
         then calls `load_vocoder(...).to(device)` on real tensors and dies with
         "Cannot copy out of meta tensor".
      2. Its `__init__` calls `load_model(...)` without the `ckpt_path` that
         f5-tts 1.1.10 requires, raising TypeError before any weights load.
      3. It wraps both modules in `torch.compile`, which is useless for a
         single-shot load and renames every state_dict key with `_orig_mod.`.

    Building the same two modules by hand sidesteps all three and lets the
    checkpoint map cleanly. The architecture below is copied from the repo's
    `model.py` and its `config.json` (DiT dim=1024, depth=22, heads=16).
    """
    import torch.nn as nn
    from f5_tts.infer import utils_infer as U
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer
    from huggingface_hub import hf_hub_download

    vocab = hf_hub_download(IndicF5Engine.REPO_ID, "checkpoints/vocab.txt")
    vocab_map, vocab_size = get_tokenizer(vocab, "custom")

    class _Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.ema_model = CFM(
                transformer=DiT(
                    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                    conv_layers=4, text_num_embeds=vocab_size,
                    mel_dim=U.n_mel_channels),
                mel_spec_kwargs=dict(
                    n_fft=U.n_fft, hop_length=U.hop_length,
                    win_length=U.win_length, n_mel_channels=U.n_mel_channels,
                    target_sample_rate=U.target_sample_rate,
                    mel_spec_type=U.mel_spec_type),
                odeint_kwargs=dict(method=U.ode_method),
                vocab_char_map=vocab_map,
            )
            self.vocoder = U.load_vocoder(vocoder_name="vocos", is_local=False,
                                          device=device)

    return _Container()


class IndicF5Engine:
    """
    Warm wrapper around ai4bharat/IndicF5.

    Bypasses the model's own `forward()` for three reasons:
      1. it hardcodes device="cpu"
      2. it strips silence and re-normalizes loudness, which fights duration control
      3. it routes through `infer_process`, whose `torchaudio.load` now demands
         torchcodec (absent here) — we hand `infer_batch_process` a tensor instead
    """

    SAMPLE_RATE = 24000
    REPO_ID = "ai4bharat/IndicF5"

    # nfe_step is the number of flow-matching steps; cost is linear in it.
    # Measured on this machine (Apple M4, 4s of Hindi audio, warm model):
    #   mps nfe=32  47.6s  (12.0x realtime)      cpu nfe=32  73.3s (18.4x)
    #   mps nfe=16  24.7s  ( 6.2x realtime)      cpu nfe=16  36.9s ( 9.3x)
    # MPS beats CPU by ~1.5x, and batching four utterances into one call did
    # NOT help (6.8x realtime), so the cost is genuinely the DiT sampling
    # rather than per-call overhead — there is nothing to amortize.
    # 16 is the default here because latency matters; raise it for final renders.

    def __init__(self, nfe_step: int = None, seed: int = None):
        self._model = None
        self._lock = threading.Lock()
        self._ref_cache = {}
        self._spb_cache = {}
        self.device = None
        import engines
        self.nfe_step = int(nfe_step or engines.get("tts") or 16)

        # Flow matching starts from random noise, so the SAME inputs give
        # different audio on every run — and the difference is not cosmetic.
        # Measured: two runs of one clip with byte-identical reference, units,
        # text and pacing scored CER 0.088 and 0.258. Without a fixed seed the
        # evaluation harness cannot attribute a change to a code change rather
        # than to sampling luck, which makes A/B comparison meaningless.
        # Set INDICF5_SEED= (empty) to opt back into stochastic sampling.
        env_seed = os.getenv("INDICF5_SEED", "1234")
        self.seed = (int(seed) if seed is not None
                     else (int(env_seed) if env_seed.strip() else None))

    # -- loading ------------------------------------------------------------
    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            _ensure_mps_fallback()
            _use_vendored_f5_tts()
            self.device = _pick_device()
            print(f"[IndicF5] Loading {self.REPO_ID} on {self.device}...")

            model = _build_indicf5_container(self.device)
            self._load_real_weights(model)

            model.ema_model.to(self.device)
            model.vocoder.to(self.device)
            model.ema_model.eval()
            model.vocoder.eval()
            model.runtime_device = self.device

            self._model = model
            print(f"[IndicF5] Ready on {self.device}")
            return self._model

    @staticmethod
    def _load_real_weights(model):
        """
        Load the checkpoint by hand, stripping the `_orig_mod.` prefix.

        Every one of the 447 tensors in ai4bharat/IndicF5's model.safetensors is
        named `ema_model._orig_mod.transformer.…` / `vocoder._orig_mod.…` — the
        checkpoint was saved from a torch.compile()-wrapped module, and compile
        inserts `_orig_mod` into the module path. The instantiated model has no
        such level, so `from_pretrained` matches ZERO weights and silently leaves
        both the DiT and the vocoder randomly initialized (in practice NaN).

        The only sign is a "weights were not used … You should probably TRAIN
        this model" warning buried in transformers' startup output, and the
        result is audio of exactly the right length containing pure silence.

        Verified: 0/447 keys match raw, 447/447 match after stripping, with no
        missing and no unexpected keys.
        """
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(IndicF5Engine.REPO_ID, "model.safetensors")
        raw = load_file(path)
        state = {k.replace("._orig_mod.", "."): v for k, v in raw.items()}

        result = model.load_state_dict(state, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if missing or unexpected:
            raise SynthesisError(
                f"IndicF5 checkpoint did not map cleanly onto the model "
                f"({len(missing)} missing, {len(unexpected)} unexpected). "
                f"First missing: {missing[:3]}")

        # A NaN here means the weights never landed — fail loudly rather than
        # emit silence that looks like a successful run.
        with torch.no_grad():
            probe = model.ema_model.transformer.proj_out.weight
            bad = bool(torch.isnan(probe).any()) or float(probe.std()) == 0.0
        if bad:
            raise SynthesisError(
                "IndicF5 weights are NaN/zero after loading — the model would "
                "generate silence.")

        print(f"[IndicF5] Loaded {len(state)} tensors "
              f"(stripped '_orig_mod.' prefix)")

    # -- reference handling -------------------------------------------------
    def _load_ref(self, ref_path: str):
        """
        Load + cache the reference waveform as a (1, N) float32 tensor at 24 kHz.
        Cached so a 30-segment video resamples the reference once, not 30 times.
        """
        key = (ref_path, os.path.getmtime(ref_path))
        if key in self._ref_cache:
            return self._ref_cache[key]

        import torch
        import soundfile as sf

        data, sr = sf.read(ref_path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != self.SAMPLE_RATE:
            import torchaudio
            t = torch.from_numpy(data).unsqueeze(0)
            t = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(t)
            sr = self.SAMPLE_RATE
        else:
            t = torch.from_numpy(data).unsqueeze(0)

        self._ref_cache.clear()          # only ever one reference per job
        self._ref_cache[key] = (t, sr)
        return t, sr

    def natural_duration(self, text: str, ref_path: str, ref_text: str,
                         lang: str = "hi", source_lang: str = None) -> float:
        """
        How long F5 will make this utterance if left alone, in seconds.

        Mirrors the model's own internal heuristic (utils_infer:
        `duration = ref_audio_len + ref_audio_len / ref_text_len * gen_text_len`,
        byte-counted) when no source language is known, because then the byte
        count is the only signal available and it is the previous behaviour.

        When the source language IS known it is replaced by a calibrated
        estimate, because the byte mirror is only meaningful same-script and
        actively harmful cross-script. The model scales generation by the
        TARGET's UTF-8 byte count, so a Devanagari target against a Latin
        reference is asked for ~3x the time its words need (3 bytes/char against
        1). The timeline then overflows and the closing squeeze crushes every
        phrase to fit — audible as a dub where words are mangled and whole lines
        vanish. Measured on an English->Hindi clip: the byte mirror wanted 15.5s
        of speech for a 9.0s video (a 1.7x squeeze); the calibrated estimate
        wanted 6.7s (1.02x).

        The calibrated form separates the two effects the byte count conflates,
        the same way `dubbing.DurationModel` does: script density from the
        target language's characters-per-second, speaker tempo from how fast
        THIS reference talks relative to typical for the source language. For a
        same-script pair the two agree to within the reference's own bytes-per-
        character, so this does not regress a same-language re-voice.
        """
        rt = self._normalize_ref_text(ref_text)
        ref_bytes = max(1, len(rt.encode("utf-8")))
        gen_text = (text or "").strip()
        spb = self._seconds_per_byte(ref_path, ref_bytes)
        spb = self._sanity_clamp(spb, rt, lang)

        # No source language: the speaker's pace cannot be told apart from the
        # script's density, so keep the model's own byte mirror unchanged.
        if not source_lang:
            return spb * len(gen_text.encode("utf-8"))

        ref_chars = max(1, len(rt.strip()))
        ref_cps = ref_chars / max(1e-6, spb * ref_bytes)
        tempo = L.chars_per_second(source_lang) / ref_cps if ref_cps > 0 else 1.0
        tempo = min(1.7, max(0.6, tempo))
        return (max(1, len(gen_text)) / L.chars_per_second(lang)) * tempo

    @staticmethod
    def _sanity_clamp(spb: float, ref_text: str, lang: str) -> float:
        """
        Keep the seconds-per-byte calibration inside a physically plausible band.

        A short or badly-transcribed reference produces a wild rate, and every
        duration downstream inherits it. Observed on a clip whose reference came
        out at 3.8s: 0.0138 s/byte against a normal ~0.028 — so every phrase was
        asked for in half the time it needs, and F5 responds to an impossible
        duration by garbling and repeating words rather than by speaking faster.
        That single bad reference produced the worst output in the whole corpus
        (CER 3.39, the dub looping "पानी कर लिया है" four times).

        The expected rate is derivable: characters-per-second for the language,
        times the bytes-per-character of its script. Anything far from that is a
        broken reference, not an unusual speaker.
        """
        try:
            cps = L.chars_per_second(lang)
        except Exception:
            return spb
        chars = max(1, len(ref_text.strip()))
        bytes_per_char = len(ref_text.encode("utf-8")) / chars
        expected = 1.0 / max(1e-6, cps * bytes_per_char)

        lo, hi = expected * 0.65, expected * 1.60
        if spb < lo or spb > hi:
            clamped = min(hi, max(lo, spb))
            print(f"[IndicF5] Reference implies {spb * 1000:.1f} ms/byte, "
                  f"outside the plausible {lo * 1000:.1f}-{hi * 1000:.1f} for "
                  f"'{lang}' — clamping to {clamped * 1000:.1f}. "
                  f"The reference clip is probably too short or mistranscribed.")
            return clamped
        return spb

    def _seconds_per_byte(self, ref_path: str, ref_bytes: int) -> float:
        """
        Seconds of SPEECH per byte of text, from the reference clip.

        Calibrates on the reference's voiced time, not its wall time. The clip
        contains the speaker's own pauses (~13% of it), and counting those makes
        every byte look slower than it is: measured on real footage the wall
        clock gave 32.0 ms/byte against an actual speaking rate of 26.0 ms/byte,
        a 23% over-estimate. F5 was therefore handed more time than the words
        needed and padded the remainder with silence — audible as gaps, while
        the speech inside each unit still ran at its own pace.
        """
        key = (ref_path, os.path.getmtime(ref_path), int(ref_bytes))
        if key in self._spb_cache:
            return self._spb_cache[key]

        wave, sr = self._load_ref(ref_path)
        y = wave.reshape(-1).cpu().numpy() if hasattr(wave, "cpu") else np.asarray(wave).reshape(-1)
        voiced = wave.shape[-1] / float(sr)
        try:
            import librosa
            rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.02),
                                      hop_length=int(sr * 0.01))[0]
            if rms.max() > 0:
                db = librosa.amplitude_to_db(rms, ref=np.max)
                frac = float((db > -35.0).mean())
                if 0.3 < frac <= 1.0:
                    voiced = (wave.shape[-1] / float(sr)) * frac
        except Exception:
            pass

        self._spb_cache.clear()
        self._spb_cache[key] = voiced / float(max(1, ref_bytes))
        return self._spb_cache[key]

    def _seed_rng(self):
        """Reset torch RNG so a given input always yields the same audio."""
        if self.seed is None:
            return
        import torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        try:
            torch.mps.manual_seed(self.seed)
        except Exception:
            pass

    @staticmethod
    def _normalize_ref_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "."
        if text[-1] not in ".?!।,":
            text += "."
        return text + " "

    # -- synthesis ----------------------------------------------------------
    def synthesize(self, text: str, ref_path: str, ref_text: str,
                   lang: str = "hi", speed: float = 1.0):
        """
        Generate one utterance. Returns (np.float32 waveform, sample_rate).

        `speed` is F5's own duration prior. We keep it near 1.0 and do precise
        duration fitting downstream in `dubbing`, because F5's `fix_duration`
        does not track the requested length closely enough to rely on.
        """
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self.SAMPLE_RATE

        model = self.load()
        from f5_tts.infer.utils_infer import infer_batch_process

        self._seed_rng()
        audio_t, sr = self._load_ref(ref_path)
        rtext = self._normalize_ref_text(ref_text)

        try:
            # The fork's infer_batch_process returns (wave, sr, spec) directly,
            # unlike the pip release which returns a generator. See
            # _use_vendored_f5_tts for why we run the fork.
            wave, out_sr, _ = infer_batch_process(
                (audio_t, sr),
                rtext,
                [text],
                model.ema_model,
                model.vocoder,
                mel_spec_type="vocos",
                device=self.device,
                nfe_step=self.nfe_step,
                cross_fade_duration=0.0,   # single batch, nothing to cross-fade
                speed=speed,
                progress=None,
            )
        except Exception as e:
            raise SynthesisError(f"IndicF5 synthesis failed: {e}") from e

        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        return wave, out_sr


# ---------------------------------------------------------------------------
# XTTS-v2 (foreign languages, phase 2)
# ---------------------------------------------------------------------------
class XTTSEngine:
    """Lazily-loaded Coqui XTTS-v2. Non-commercial license — see module docstring."""

    SAMPLE_RATE = 24000
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self):
        self._tts = None
        self._lock = threading.Lock()
        self.device = None

    def load(self):
        if self._tts is not None:
            return self._tts
        with self._lock:
            if self._tts is not None:
                return self._tts
            _ensure_mps_fallback()
            self._allow_xtts_globals()
            _patch_torchaudio_load()
            from TTS.api import TTS
            # XTTS has incomplete MPS kernel coverage; CPU is the reliable path.
            self.device = "cuda" if _pick_device() == "cuda" else "cpu"
            print(f"[XTTS] Loading {self.MODEL_NAME} on {self.device}...")
            self._tts = TTS(self.MODEL_NAME).to(self.device)
            print("[XTTS] Ready")
            return self._tts

    @staticmethod
    def _allow_xtts_globals():
        """
        Allowlist Coqui's config classes for torch.load.

        PyTorch 2.6 flipped `torch.load`'s `weights_only` default to True, and
        the XTTS checkpoint pickles `XttsConfig` and friends — so on torch 2.13
        loading dies with `UnsupportedGlobal: XttsConfig`. These are Coqui's own
        dataclasses from a checkpoint we already trust (it is in the local HF
        cache), so allowlisting them is the narrow fix; flipping weights_only
        off globally would disable the protection for every other load too.
        """
        import torch
        if not hasattr(torch.serialization, "add_safe_globals"):
            return  # older torch: weights_only was not the default
        allowed = []
        for module, names in (
            ("TTS.tts.configs.xtts_config", ["XttsConfig"]),
            ("TTS.tts.models.xtts", ["XttsAudioConfig", "XttsArgs"]),
            ("TTS.config.shared_configs", ["BaseDatasetConfig", "BaseAudioConfig"]),
        ):
            try:
                mod = __import__(module, fromlist=names)
                allowed += [getattr(mod, n) for n in names if hasattr(mod, n)]
            except Exception:
                continue
        if allowed:
            torch.serialization.add_safe_globals(allowed)

    def natural_duration(self, text: str, ref_path: str = None,
                         ref_text: str = None, lang: str = "en",
                         source_lang: str = None) -> float:
        """
        Estimated spoken length, so the planner treats XTTS like IndicF5.

        XTTS has no `fix_duration` equivalent — it speaks at its own pace and
        the timeline absorbs the difference. Without an estimate here the
        planner fell back to the generic characters-per-second model, whose
        targets could be far enough off to force overruns and cascading.
        XTTS's rate is stable enough that the language table is a fair proxy.

        `source_lang` is accepted for interface parity with IndicF5 and ignored:
        this estimate already counts TARGET characters against the target
        language's rate, so it carries none of the cross-script byte error that
        parameter exists to correct.
        """
        try:
            cps = L.chars_per_second(lang)
        except Exception:
            cps = 15.0
        return max(0.2, len((text or "").strip()) / cps)

    def synthesize(self, text: str, ref_path: str, ref_text: str = "",
                   lang: str = "en", speed: float = 1.0):
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self.SAMPLE_RATE
        tts = self.load()
        try:
            wave = tts.tts(text=text, speaker_wav=ref_path, language=lang, speed=speed)
        except TypeError:
            wave = tts.tts(text=text, speaker_wav=ref_path, language=lang)
        except Exception as e:
            raise SynthesisError(f"XTTS synthesis failed: {e}") from e
        return np.asarray(wave, dtype=np.float32).reshape(-1), self.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Fish Audio S2 Pro (extraplus/s2-pro)
# ---------------------------------------------------------------------------
class S2ProEngine:
    """
    Warm wrapper around Fish Audio S2 Pro, via the `fish_speech` package.

    S2 Pro is a Fish Speech model (the `fish_qwen3_omni` architecture) with
    zero-shot cloning from a reference clip and its transcript, the same
    interface IndicF5 exposes — so it slots into `synth_unit` unchanged. It
    covers most of this project's targets, Indic and foreign alike.

    Unlike IndicF5 it is not an f5_tts flow-matching model: there is no
    `nfe_step` and no `fix_duration`, so `synth_unit` takes the non-IndicF5
    branch and lets the timeline absorb the model's own pace, exactly as it
    does for XTTS.

    Weights are `extraplus/s2-pro` on the Hub (ungated, but under the Fish
    Audio Research License — research and non-commercial use). `fish_speech`
    is an optional extra: a default install does not carry it and this engine
    reports itself unavailable rather than failing a render several minutes in.
    """

    REPO_ID = "extraplus/s2-pro"
    # The DAC codec in this checkpoint runs at 44.1 kHz.
    SAMPLE_RATE = 44100
    MODEL_FAMILY = "s2"

    def __init__(self, seed: int = None):
        self._engine = None
        self._lock = threading.Lock()
        self._ref_cache = {}
        self.device = None
        env_seed = os.getenv("S2PRO_SEED", "1234")
        self.seed = (int(seed) if seed is not None
                     else (int(env_seed) if env_seed.strip() else None))

    # -- loading ------------------------------------------------------------
    def load(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine

            # Resolve the package BEFORE anything expensive. A missing
            # fish_speech used to be discovered after a ~11 GB snapshot
            # download and a torch import, which is a long wait for a failure
            # that a spec check answers instantly.
            try:
                from fish_speech.inference_engine import TTSInferenceEngine
                from fish_speech.models.dac.inference import (
                    load_model as load_decoder_model,
                )
                from fish_speech.models.text2semantic.inference import (
                    launch_thread_safe_queue,
                )
            except Exception as e:                            # noqa: BLE001
                raise SynthesisError(
                    f"fish_speech is not installed, so S2 Pro cannot speak. "
                    f"Install it with `pip install "
                    f"git+https://github.com/fishaudio/fish-speech.git`. ({e})"
                ) from e

            import torch
            _ensure_mps_fallback()
            self.device = _pick_device()
            # Refuse a load that cannot fit before it starts (see the helper).
            _require_room_for_s2pro(self.device)
            checkpoint = _s2pro_checkpoint_dir()
            # bfloat16 is the upstream server's default (`--half` opts into
            # fp16). It is the safer of the two across Ampere and older cards,
            # and the model card's own examples run it this way.
            precision = torch.bfloat16
            print(f"[S2Pro] Loading {self.REPO_ID} on {self.device}...")

            llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint,
                device=self.device,
                precision=precision,
                compile=False,
            )
            # `resolve_codec_path` also finds the sibling s2-pro/codec.pth when
            # the checkpoint directory does not carry one, which is the case for
            # fine-tuned copies of this model.
            try:
                from fish_speech.models.text2semantic.inference import (
                    resolve_codec_path,
                )
                codec_path = str(resolve_codec_path(checkpoint))
            except Exception:                                    # noqa: BLE001
                codec_path = os.path.join(checkpoint, "codec.pth")
            decoder = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=codec_path,
                device=self.device,
            )
            self._engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder,
                precision=precision,
                compile=False,
                model_family=self.MODEL_FAMILY,
            )
            print(f"[S2Pro] Ready on {self.device}")
            return self._engine

    # -- reference handling -------------------------------------------------
    def _reference(self, ref_path: str, ref_text: str):
        """
        Read the reference clip once per job and cache its bytes.

        S2 Pro takes the reference audio inline in the request, so the bytes are
        what get cached rather than a decoded tensor. Cached because a
        thirty-segment video would otherwise re-read the same file thirty times.
        """
        key = (ref_path, os.path.getmtime(ref_path))
        if key in self._ref_cache:
            return self._ref_cache[key]
        with open(ref_path, "rb") as f:
            audio = f.read()
        self._ref_cache.clear()          # only ever one reference per job
        self._ref_cache[key] = (audio, ref_text or "")
        return audio, ref_text or ""

    # -- synthesis ----------------------------------------------------------
    def natural_duration(self, text: str, ref_path: str = None,
                         ref_text: str = None, lang: str = "en",
                         source_lang: str = None) -> float:
        """
        Estimated spoken length, so the planner treats S2 Pro like the others.

        S2 Pro has no `fix_duration`; it speaks at its own pace and the timeline
        absorbs the difference, so this is the generic characters-per-second
        model — the same fallback XTTS uses.
        """
        try:
            cps = L.chars_per_second(lang)
        except Exception:
            cps = 14.0
        return max(0.2, len((text or "").strip()) / cps)

    def synthesize(self, text: str, ref_path: str, ref_text: str = "",
                   lang: str = "en", speed: float = 1.0, prosody: str = ""):
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self.SAMPLE_RATE

        # S2 Pro reads inline control tags ("[loud]", "[low volume]", ...) as
        # free-form delivery instructions. The tag is prepended rather than
        # spliced into the line so it always governs the whole phrase; an empty
        # tag leaves the text exactly as the others would speak it.
        tag = (prosody or "").strip()
        if tag:
            text = f"{tag} {text}"

        engine = self.load()
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        audio, transcript = self._reference(ref_path, ref_text)
        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=audio, text=transcript)],
            format="wav",
            seed=self.seed,
            # The reference is the same speaker clip for every phrase in a job,
            # so let the engine keep its encoded prompt tokens instead of
            # re-encoding ~10s of audio on each of a hundred units.
            use_memory_cache="on",
        )

        try:
            result = None
            for chunk in engine.inference(req):
                if chunk.code == "error":
                    raise SynthesisError(
                        f"S2 Pro synthesis failed: {chunk.error}")
                if chunk.code == "final":
                    result = chunk
            if result is None or result.audio is None:
                raise SynthesisError("S2 Pro returned no audio")
            sr, wave = result.audio
        except SynthesisError:
            raise
        except Exception as e:                                # noqa: BLE001
            raise SynthesisError(f"S2 Pro synthesis failed: {e}") from e

        # The codec decodes to float32 in [-1, 1], which is the range the rest
        # of the pipeline works in. Normalise defensively in case a future
        # codec hands back int16 PCM instead.
        arr = np.asarray(wave)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        else:
            arr = arr.astype(np.float32)
        return arr.reshape(-1), int(sr)


def _s2pro_checkpoint_dir() -> str:
    """
    The local snapshot directory for extraplus/s2-pro, fetched if absent.

    `from_pretrained` wants a directory, not a repo id, so the snapshot is
    resolved here. `local_files_only` is not set: an un-downloaded model
    fetches on first use, which is the behaviour the settings page promises.
    """
    from huggingface_hub import snapshot_download
    return snapshot_download(S2ProEngine.REPO_ID)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_ENGINES = {}
_REG_LOCK = threading.Lock()

_AVAILABLE: dict = {}


def refresh() -> None:
    """
    Drop the warm voice engines so the next phrase picks up a new setting.

    The nfe_step ("how much work per phrase") is read when an engine is
    constructed, so changing it has to discard the instance, not just the
    stored value.
    """
    with _REG_LOCK:
        _ENGINES.clear()
    _AVAILABLE.clear()


def _indicf5_present() -> bool:
    """
    Whether IndicF5's weights are actually on this machine.

    The package is always importable; the 1.4 GB of weights are not. IndicF5
    is a gated repo, so an unauthenticated fetch leaves a README behind and
    nothing else - which used to be enough for the studio to advertise eleven
    Indian languages it could not speak.
    """
    try:
        import models
    except ImportError:
        return True              # cannot tell; do not block on a guess
    for entry in models.MODELS:
        if entry["id"] == IndicF5Engine.REPO_ID:
            return models.is_cached(entry["id"], entry["files"])
    return True


def _s2pro_present() -> bool:
    """
    Whether S2 Pro can actually be run here: package installed and weights present.

    Two separate conditions, and both must hold. `fish_speech` is an optional
    extra a default install does not carry; the ~11 GB of weights are a
    separate download. Advertising S2 Pro when either is missing is the same
    failure this file already fixed for XTTS and IndicF5.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("fish_speech") is None:
            return False
    except Exception:                                        # noqa: BLE001
        return False
    try:
        import models
    except ImportError:
        return True
    for entry in models.MODELS:
        if entry["id"] == S2ProEngine.REPO_ID:
            return models.is_cached(entry["id"], entry["files"])
    return True


def engine_available(name: str) -> bool:
    """
    Whether the model behind an engine is actually usable here.

    XTTS is an optional extra: its licence is non-commercial and its
    dependency tree is enormous, so requirements.txt leaves it out. Without
    this check the app offered sixteen languages it could not speak, and the
    failure arrived as a ModuleNotFoundError deep in a render the person had
    already waited several minutes for. IndicF5 is the mirror image: the
    package is always present, but its gated weights may not be. S2 Pro needs
    both the `fish_speech` package and its own weights.
    """
    if name in _AVAILABLE:
        return _AVAILABLE[name]
    ok = True
    if name == L.ENGINE_XTTS:
        try:
            import importlib.util
            ok = importlib.util.find_spec("TTS") is not None
        except Exception:                                    # noqa: BLE001
            ok = False
    elif name == L.ENGINE_INDICF5:
        ok = _indicf5_present()
    elif name == L.ENGINE_S2PRO:
        ok = _s2pro_present()
    _AVAILABLE[name] = ok
    return ok


def language_available(target_lang: str) -> bool:
    """Whether this language can actually be spoken on this machine."""
    return engine_available(L.engine_for(target_lang))


def get_engine(target_lang: str):
    """Return the warm engine instance responsible for `target_lang`."""
    name = L.engine_for(target_lang)
    with _REG_LOCK:
        if name not in _ENGINES:
            if name == L.ENGINE_INDICF5:
                _ENGINES[name] = IndicF5Engine()
            elif name == L.ENGINE_S2PRO:
                _ENGINES[name] = S2ProEngine()
            else:
                _ENGINES[name] = XTTSEngine()
        return _ENGINES[name]


def warmup(target_lang: str = "hi"):
    """Preload weights so the first request doesn't pay the load cost."""
    try:
        get_engine(target_lang).load()
        return True
    except Exception as e:
        print(f"[TTS] Warmup failed for {target_lang}: {e}")
        return False


def synthesize(text: str, ref_path: str, ref_text: str, target_lang: str,
               speed: float = 1.0):
    """Synthesize `text` in `target_lang` in the reference speaker's voice."""
    engine = get_engine(target_lang)
    return engine.synthesize(text, ref_path, ref_text,
                             lang=L.synth_lang(target_lang), speed=speed)

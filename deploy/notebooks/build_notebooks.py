#!/usr/bin/env python3
"""
Regenerate the hosted-runner notebooks.

`colab.ipynb` and `kaggle.ipynb` are generated from one cell list, so the two
cannot drift: Kaggle differs only by the tunnel step, because it has no port
proxy like Colab's. Edit the cells here and re-run this script rather than
editing the notebooks by hand.

    python deploy/notebooks/build_notebooks.py

The generated files are committed, so a user never needs to run this.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO_URL = "https://github.com/gajjalaashok75-UI/video-dubbing"

# Colab mounts /content; Kaggle mounts /kaggle/working. Both are writable for
# the life of the session, which is all a runtime tree needs. Written as one
# parenthesised expression so it can be pasted into a cell as `REPO_DIR = ...`,
# and self-contained because the clone cell runs before the checkout it is
# creating can be imported.
#
# The tests are in the SAME ORDER as `deploy_helpers.detect_platform`, and that
# is not cosmetic: that function returns "colab" on any machine with /content,
# so testing /kaggle first put the checkout at /kaggle/working/video-dubbing
# while the runtime tree (and the server log the user is told to read) went to
# /content/videotrans-runtime - one session split across two volumes.
# `tests/test_platform_layout.py` keeps this expression and `repo_dir()` equal.
REPO_DIR_EXPR = ('(\n'
                 '    "/content/video-dubbing" if os.path.isdir("/content")\n'
                 '    else "/kaggle/working/video-dubbing" if os.path.isdir("/kaggle")\n'
                 '    else os.getcwd()\n'
                 ')')


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _src(src)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(src)}


def _assign_ids(cells: list[dict]) -> None:
    """Give every cell a stable id, as nbformat 4.5 requires.

    Derived from the position and a hash of the source, so regenerating the
    notebooks does not churn every id and produce a meaningless diff.
    """
    for i, cell in enumerate(cells):
        digest = hashlib.sha1("".join(cell["source"]).encode()).hexdigest()[:8]
        cell["id"] = f"{cell['cell_type'][:2]}-{i:02d}-{digest}"


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

TITLE = md("""
# VideoTrans - run the whole pipeline on a free GPU

Dub a video into another language - detect who is speaking, clone **each**
speaker's voice, translate, lay the new dialogue over the film's own music and
sound effects, lip-sync - on the machine this notebook is attached to, not on
your laptop.

**This needs a GPU runtime.** In Colab: **Runtime -> Change runtime type -> T4
GPU**. In Kaggle: **Settings -> Accelerator -> GPU T4 x2**, and turn
**Internet** on. Then **Run all**.

> The first run takes 10-20 minutes: the install pulls PyTorch and the ML
> stack, then ~7.5 GB of model weights. Later runs in the same session are
> much faster.
>
> On a GPU a 7-second clip renders in well under a minute. On a CPU the same
> clip takes several minutes. That difference is the point of this notebook:
> the engine already prefers CUDA when it sees it, so the only thing to fix is
> giving it a GPU.
""")

GET_CODE_MD = md("""
## 1. Get the code

Colab and Kaggle both give a writable working directory. We clone the
repository there and install FFmpeg, which every render shells out to.
""")

GET_CODE = code(f"""
import os
import shutil
import subprocess

REPO_URL = "{REPO_URL}"
REPO_DIR = {REPO_DIR_EXPR}

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Cloning the repository...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    print("Repository already present; updating...")
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", "main"],
                   check=False)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "FETCH_HEAD"], check=False)

if not shutil.which("ffmpeg"):
    print("Installing FFmpeg...")
    subprocess.run(["apt-get", "-qq", "update"], check=False)
    subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)

print("FFmpeg:", shutil.which("ffmpeg"))
print("Ready: ", REPO_DIR)
""")

CHECK_MD = md("""
## 2. Check the machine

Confirms the GPU is visible, reports how much VRAM it has, and picks the
quality preset. This cell is quick - it downloads nothing.
""")

CHECK = code(f"""
import os
import sys

REPO_DIR = {REPO_DIR_EXPR}
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from engine import deploy_helpers as runtime

print("Platform:", runtime.detect_platform())
print("Python:  ", sys.version.split()[0])

report = runtime.hardware_report()
if report["cuda_available"]:
    print(f"GPU:      {{report['gpu_name']}} ({{report['vram_gb']}} GB VRAM)")
else:
    print("GPU:      none - renders run on CPU and are slow")
print("Device:  ", report["device"])
print("Preset:  ", report["recommended_preset"])
for note in report["notes"]:
    print("  note:", note)

print("FFmpeg:  ", "ready" if runtime.ensure_ffmpeg() else "NOT available")

# The same readiness the studio shows. Every probe is an import spec or a file
# check, so this is instant and loads no model.
sys.path.insert(0, "engine")
import readiness  # noqa: E402

snapshot = readiness.snapshot()
print()
print("Features:")
for line in readiness.summarize(snapshot):
    print(" ", line)
print()
print("A missing OPTIONAL feature is a reduced dub, not a broken install.")
print("Run the Verify cell after the install to see the full picture.")
""")

RUNTIME_MD = md("""
## 3. Point the runtime at a writable directory

Uploads, finished renders, the model cache and the server log are all written
at run time. This cell puts them under one runtime root, instead of inside the
checkout or the home directory.

It also sets `HF_HOME`, so the ~7.5 GB of weights land on the same writable
volume as the renders rather than in a cache a runtime restart discards.

**It must run before the server is imported**, because the engine reads the
upload directory at import time.
""")

RUNTIME = code("""
dirs = runtime.configure_runtime()

print("Runtime root:", next(iter(dirs.values())).parent)
for name, path in dirs.items():
    print(f"  {name:8} -> {path}")

# Finished videos land here; remembered for later cells.
OUTPUT_DIR = dirs["output"]
""")

TOKEN_MD = md("""
## 4. Set your Hugging Face token (do this before the models)

The Indian-language voice model, `ai4bharat/IndicF5`, is **gated**. It needs a
token, and it needs one *before* the fetch in the next step - which is why this
cell sits here and not at the end.

1. Accept the licence at <https://huggingface.co/ai4bharat/IndicF5>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Put it in the platform's **Secrets** panel as `HF_TOKEN` (recommended, so it
   is not saved in the notebook), or uncomment the line below.

Without a token the other models still download and transcription,
translation, subtitles and lip sync all work. Only the 11 Indian languages
cannot be spoken, because their voice weights are the gated ones.
""")

TOKEN = code("""
import os

# Prefer the platform secret (Colab: the key icon in the left sidebar;
# Kaggle: Add-ons -> Secrets). Both inject a secret into os.environ at kernel
# START, so one added after the runtime came up is only reachable through
# their SDK. Read it here too and export it: the download and the server both
# look at the environment, and without this the cell reported "No HF_TOKEN"
# while the secret was sitting right there in the panel.
if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata  # Colab
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        try:
            from kaggle_secrets import UserSecretsClient  # Kaggle
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass

# Uncomment the next line only if you must hardcode it.
# os.environ["HF_TOKEN"] = "hf_..."

if os.environ.get("HF_TOKEN"):
    print("HF_TOKEN is set - the gated voice model can be fetched.")
else:
    print("No HF_TOKEN. The Indian-language voice model will be skipped;")
    print("everything else downloads and works. Set it, re-run this cell, then")
    print("re-run the model cell if you want those languages.")
""")

INSTALL_MD = md("""
## 5. Install the application

Two installs: the CUDA PyTorch wheels first, then the rest of the pinned
dependencies. This is the slow cell - expect 5-10 minutes on the first run.

Installing PyTorch on its own from the `cu124` index is what makes the GPU get
used. Installing the plain `requirements.txt` alone would fetch the CPU-only
wheels and every render would silently fall back to the CPU.
""")

INSTALL = code('''
import os
import subprocess
import sys

# ============================================================
# Locate repository
# ============================================================

REPO_DIR = ''' + REPO_DIR_EXPR + '''

os.chdir(REPO_DIR)

print("Repository:", REPO_DIR)
print("Python:", sys.version)
print()


# ============================================================
# Helpers
# ============================================================

def run(cmd, *, check=True, capture=False):
    print("+", " ".join(cmd))

    if capture:
        return subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=check,
        )

    return subprocess.run(cmd, check=check)


def stream(cmd):
    print("+", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="")

    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code %d: %s"
            % (proc.returncode, " ".join(cmd))
        )


# ============================================================
# Required Torch stack
# ============================================================
#
# A CUDA build is not enough - it has to be a CUDA 12 one. CTranslate2
# (faster-whisper's backend) dlopens libcublas.so.12 on the FIRST encode, and
# torch >= 2.7 ships CUDA 13 wheels whose cuBLAS is libcublas.so.13. 2.6.0 is
# the last release published for cu124.

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

TORCH_PACKAGES = [
    "torch==2.6.0+cu124",
    "torchaudio==2.6.0+cu124",
    "torchvision==0.21.0+cu124",
]


# ============================================================
# Check the COMPLETE Torch stack
# ============================================================

TORCH_PROBE = r"""
import sys

try:
    import torch
    import torchaudio
    import torchvision

    print("torch      :", torch.__version__)
    print("torchaudio :", torchaudio.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA       :", torch.version.cuda)
    print("GPU        :", torch.cuda.is_available())

    ok = (
        torch.__version__.startswith("2.6.0")
        and torchaudio.__version__.startswith("2.6.0")
        and torchvision.__version__.startswith("0.21.0")
        and (torch.version.cuda or "").startswith("12.4")
        and torch.cuda.is_available()
    )

    sys.exit(0 if ok else 1)

except Exception as exc:
    print("Torch stack probe failed:", repr(exc))
    sys.exit(1)
"""


print("=" * 70)
print("CHECKING PYTORCH / CUDA STACK")
print("=" * 70)

probe = subprocess.run(
    [sys.executable, "-c", TORCH_PROBE],
    text=True,
    capture_output=True,
)

print(probe.stdout)

if probe.stderr:
    print(probe.stderr)


if probe.returncode != 0:

    print()
    print("Installing the exact CUDA 12.4 PyTorch stack...")

    # Remove potentially mismatched binaries first.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchaudio",
            "torchvision",
        ],
        check=False,
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            *TORCH_PACKAGES,
            "--index-url",
            TORCH_INDEX,
        ]
    )

    print()
    print("=" * 70)
    print("PYTORCH WAS REPLACED")
    print("=" * 70)
    print()
    print(
        "IMPORTANT: restart the Colab/Kaggle Python runtime NOW, "
        "then run this cell again."
    )

    raise SystemExit(
        "PyTorch binaries changed. Runtime restart required before continuing."
    )


print("Exact PyTorch 2.6.0 / CUDA 12.4 stack is already usable.")
print()


# ============================================================
# Install main application requirements
# ============================================================

print("=" * 70)
print("INSTALLING APPLICATION REQUIREMENTS")
print("=" * 70)

stream(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-r",
        "requirements.txt",
    ]
)


# ============================================================
# IndicF5 vendored implementation
# ============================================================

print()
print("=" * 70)
print("FETCHING INDICF5 F5-TTS FORK")
print("=" * 70)

result = subprocess.run(
    [sys.executable, "scripts/vendor_f5_tts.py"]
)

if result.returncode != 0:
    raise RuntimeError(
        "Failed to fetch IndicF5's vendored f5_tts implementation."
    )


# ============================================================
# Demucs
# ============================================================
#
# pyannote is NOT installed here anymore. It is pinned in requirements.txt:
#
#     pyannote.audio==3.3.2
#
# so pip resolves it together with the core environment, and a job never ends
# up with pyannote 4.x pulling a newer torch.
#
# Demucs is intentionally installed without dependencies because its old
# dependency tree must not rewrite Torch, NumPy, Hydra, OmegaConf, etc.

print()
print("=" * 70)
print("INSTALLING OPTIONAL BACKGROUND ENGINE")
print("=" * 70)

# Demucs' own import chain, installed alongside it with --no-deps for the same
# reason: `pip check` reported dora-search, lameenc and openunmix missing, which
# is why "demucs installed" printed ok while the separation command could not
# start. dora-search's own small dependencies are named too, since --no-deps
# skips them.
DEMUCS_DEPS = [
    "lameenc",
    "julius",
    "openunmix",
    "dora-search",
    "omegaconf",
    "retrying",
    "submitit",
    "treetable",
]

demucs = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "demucs==4.0.1",
    ],
    text=True,
    capture_output=True,
)

if demucs.returncode == 0:
    print("ok   demucs 4.0.1 installed")
    print("     installing the modules demucs imports, also with --no-deps:")
    deps = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            *DEMUCS_DEPS,
        ],
        text=True,
        capture_output=True,
    )
    if deps.returncode != 0:
        print("--   some Demucs dependencies did not install")
        for line in (deps.stderr or deps.stdout or "").strip().splitlines()[-3:]:
            print("       ", line)
else:
    print("--   demucs installation failed")
    print(demucs.stdout)
    print(demucs.stderr)

# The only check that means anything. `find_spec("demucs")` answered yes for an
# install whose `python -m demucs.separate` died on `import lameenc`, so the
# separation command is asked to come up once instead.
demucs_ready = subprocess.run(
    [sys.executable, "-m", "demucs.separate", "--help"],
    text=True,
    capture_output=True,
)
if demucs_ready.returncode == 0:
    print("ok   demucs can start (background preservation is usable)")
else:
    print("--   demucs is installed but cannot start; background preservation is off")
    for line in (demucs_ready.stderr or demucs_ready.stdout or "").strip().splitlines()[-3:]:
        print("       ", line)


# ============================================================
# Restore the pins a dependency can move
# ============================================================
#
# requirements.txt pins these, but a resolver can still move them:
#
#   NumPy      - numba (librosa's audio loader, used by the voice engine AND
#                by Wav2Lip) refuses anything newer than 2.2.
#   transformers/hub - 5.x and hub 1.x cannot live next to each other, and the
#                symptom is "cannot import name 'GenerationMixin'".
#
# These are only repaired when the probe says they moved, so a clean install
# pays nothing for them.

PROBE = (
    "import sys, numpy, numba, transformers, huggingface_hub;"
    "print(numpy.__version__);"
    "print(transformers.__version__);"
    "print(huggingface_hub.__version__);"
    "ok = (numpy.__version__.startswith('2.2') or numpy.__version__.startswith('2.1')"
    " or numpy.__version__.startswith('2.0') or numpy.__version__.startswith('1.'))"
    " and transformers.__version__.startswith('4.')"
    " and huggingface_hub.__version__.startswith('0.');"
    "print('OK' if ok else 'FAIL'); sys.exit(0 if ok else 1)"
)
REPAIR = [
    "numpy>=1.26.0,<2.3",
    "transformers==4.57.6",
    "huggingface-hub==0.36.2",
]

probe = subprocess.run([sys.executable, "-c", PROBE],
                       capture_output=True, text=True)

if probe.returncode != 0:
    print("  NumPy / transformers is out of range; restoring the pins...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REPAIR],
                   check=False)
    probe = subprocess.run([sys.executable, "-c", PROBE],
                           capture_output=True, text=True)

print("  numpy", (probe.stdout or "?").strip(),
      "(the voice model needs it at 2.2 or less)")


# ============================================================
# Final dependency verification
# ============================================================

print()
print("=" * 70)
print("VERIFYING CRITICAL DEPENDENCIES")
print("=" * 70)

VERIFY = r"""
import sys

errors = []


def check(name, fn):
    try:
        value = fn()
        print(f"ok   {name:<22} {value}")
    except Exception as exc:
        print(f"FAIL {name:<22} {exc!r}")
        errors.append((name, exc))


# ------------------------------------------------------------
# Torch
# ------------------------------------------------------------

def check_torch():
    import torch

    assert torch.__version__.startswith("2.6.0"), torch.__version__
    assert (torch.version.cuda or "").startswith("12.4"), torch.version.cuda
    assert torch.cuda.is_available()

    return f"{torch.__version__} CUDA={torch.version.cuda}"


check("torch", check_torch)


# ------------------------------------------------------------
# Torchaudio
# ------------------------------------------------------------

def check_torchaudio():
    import torchaudio

    assert torchaudio.__version__.startswith("2.6.0"), torchaudio.__version__

    return torchaudio.__version__


check("torchaudio", check_torchaudio)


# ------------------------------------------------------------
# Torchvision
# ------------------------------------------------------------

def check_torchvision():
    import torchvision

    assert torchvision.__version__.startswith("0.21.0"), torchvision.__version__

    return torchvision.__version__


check("torchvision", check_torchvision)


# ------------------------------------------------------------
# NumPy / Numba
# ------------------------------------------------------------

def check_numpy():
    import numpy
    import numba

    assert numpy.__version__ == "2.2.6", numpy.__version__

    return f"numpy={numpy.__version__}, numba={numba.__version__}"


check("numpy/numba", check_numpy)


# ------------------------------------------------------------
# Transformers / NLLB imports
# ------------------------------------------------------------

def check_transformers():
    import transformers
    import huggingface_hub

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    assert transformers.__version__ == "4.57.6", transformers.__version__
    assert huggingface_hub.__version__ == "0.36.2", huggingface_hub.__version__

    return (
        f"transformers={transformers.__version__}, "
        f"hub={huggingface_hub.__version__}"
    )


check("NLLB/transformers", check_transformers)


# ------------------------------------------------------------
# SentencePiece
# ------------------------------------------------------------

def check_sentencepiece():
    import sentencepiece

    return sentencepiece.__version__


check("sentencepiece", check_sentencepiece)


# ------------------------------------------------------------
# faster-whisper
# ------------------------------------------------------------

def check_whisper():
    import faster_whisper
    import ctranslate2
    import av

    av_major = int(av.__version__.split(".")[0])

    assert av_major < 18, av.__version__

    return (
        f"faster-whisper={getattr(faster_whisper, '__version__', '?')}, "
        f"ctranslate2={ctranslate2.__version__}, "
        f"av={av.__version__}"
    )


check("faster-whisper", check_whisper)


# ------------------------------------------------------------
# Pyannote
# ------------------------------------------------------------

def check_pyannote():
    import pyannote.audio

    from pyannote.audio import Pipeline

    version = getattr(pyannote.audio, "__version__", "?")

    assert version.startswith("3.3.2"), version

    return version


check("pyannote", check_pyannote)


# ------------------------------------------------------------
# IndicF5 dependencies
# ------------------------------------------------------------

def check_f5():
    import vocos
    import torchdiffeq
    import x_transformers

    return "imports successful"


check("IndicF5 dependencies", check_f5)


# ------------------------------------------------------------
# Librosa
# ------------------------------------------------------------

def check_librosa():
    import librosa

    return librosa.__version__


check("librosa", check_librosa)


print()
if errors:
    print("=" * 70)
    print("ENVIRONMENT VERIFICATION FAILED")
    print("=" * 70)

    for name, exc in errors:
        print(f"{name}: {exc!r}")

    sys.exit(1)


print("=" * 70)
print("CORE ENVIRONMENT VERIFIED")
print("=" * 70)
"""

verify = subprocess.run(
    [sys.executable, "-c", VERIFY],
    text=True,
    capture_output=True,
)

print(verify.stdout)

if verify.stderr:
    print(verify.stderr)

if verify.returncode != 0:
    raise SystemExit(
        "Dependency verification failed. Do NOT start the server."
    )


# ============================================================
# pip check
# ============================================================

print()
print("=" * 70)
print("PIP DEPENDENCY CHECK")
print("=" * 70)

pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    text=True,
    capture_output=True,
)

print(pip_check.stdout)

if pip_check.stderr:
    print(pip_check.stderr)

# Colab itself contains many unrelated preinstalled packages. Therefore
# pip check may report conflicts belonging to the base notebook
# environment. Critical VideoTrans packages were already validated
# explicitly above.

if pip_check.returncode != 0:
    print()
    print(
        "WARNING: pip check found conflicts. Review them above. "
        "If they only belong to unrelated Colab packages, they can "
        "usually be ignored. VideoTrans critical packages passed "
        "their dedicated verification."
    )


print()
print("=" * 70)
print("INSTALL COMPLETE")
print("=" * 70)
print()
print("Torch / CUDA       : verified")
print("TorchAudio ABI     : verified")
print("NumPy / Numba      : verified")
print("NLLB / Transformers: verified")
print("faster-whisper     : verified")
print("pyannote 3.3.2     : verified")
print("IndicF5 deps       : verified")
print()
print("You can now run the Verify/Download Models cell.")
''')

MODELS_MD = md("""
## 6. Fetch the models

About 7.5 GB, resumable, and only once per runtime. This cell streams the
downloader's output and then **verifies** what arrived, so a skipped or failed
fetch is reported here instead of surfacing as a broken render later.
""")

MODELS = code(f"""
import os

# Locate the checkout and work from it, so this cell does not depend on an
# earlier cell having run in this session.
REPO_DIR = {REPO_DIR_EXPR}
os.chdir(REPO_DIR)

import subprocess
import sys

proc = subprocess.Popen(
    [sys.executable, "scripts/download_models.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()

# Ask the engine what is still missing rather than trusting the exit code:
# a gated model that could not be fetched still exits 0.
sys.path.insert(0, "engine")
import models  # noqa: E402

missing = [r for r in models.inventory() if not r["present"]]
required = [r for r in missing if r["required"]]

print()
print("=" * 62)
print("  Model cache:", models.cache_root())
for r in models.inventory():
    mark = "present" if r["present"] else "MISSING"
    print(f"  [{{mark:>7}}] {{r['label']:<20}} {{r['size']:>7}}")
print("=" * 62)

# A gated model that was skipped is a warning, not a failure: without it the
# Indian languages are unavailable but every other language still renders. A
# non-gated required model missing is fatal, because the pipeline cannot run.
blocking = [r for r in required if not r["gated"]]
gated = [r for r in required if r["gated"]]

if gated:
    names = ", ".join(r["label"] for r in gated)
    print()
    print(f"WARNING: {{names}} was skipped (gated, no HF_TOKEN).")
    print("The Indian languages will report as unavailable; everything else works.")
    print("To enable them: set HF_TOKEN in the cell above, then re-run this cell.")

if blocking:
    names = ", ".join(r["label"] for r in blocking)
    raise RuntimeError(
        f"Still missing {{len(blocking)}} required model(s): {{names}}.\\n"
        "Renders will fail without them. Re-run this cell - the download is "
        "resumable. If it keeps failing, read the downloader output above."
    )

print()
print("Ready. Start the server in the next cell.")
""")

VERIFY_MD = md("""
## 6a. Verify what actually works

One cell, the whole picture: which stages this runtime can run, which are
reduced, and which are blocked. It is the same `readiness` report the studio
shows on its Setup page, and the same one `scripts/final_check.py plan` prints.

Nothing here loads a model, so it takes a second. **Run this before you start a
render** - it is much cheaper than discovering a missing package ten minutes
into a job.

A missing *optional* feature is a reduced dub, not a broken install. The two
extras are:

- **pyannote** - multi-speaker detection, so each actor gets their own cloned
  voice. Without it the whole video is dubbed in one voice.
- **Demucs** - keeps the film's own music, sound effects and ambience under the
  new dialogue. Without it the dub replaces the soundtrack.

Neither being absent stops a dub.
""")

VERIFY = code(f"""
import os
import sys

REPO_DIR = {REPO_DIR_EXPR}
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import json
import readiness

# Ask about a language too, so the answer covers the actual voice engine a
# render would use rather than the general case.
TARGET = os.environ.get("VIDEOTRANS_TARGET_LANG", "hi")
snapshot = readiness.snapshot(target_language=TARGET)

print("=" * 66)
print("  READINESS")
print("=" * 66)
for line in readiness.summarize(snapshot):
    print(line)

print()
if snapshot["ready"]:
    print(f"READY. A dub into {{TARGET}} can be rendered here.")
else:
    print("NOT READY. Blocking:")
    for name in snapshot["blocking"]:
        feature = snapshot["features"].get(name) or {{}}
        print(f"  {{name}}: {{feature.get('reason')}}")
    print()
    print("These are environment limits, not code failures. Transcription,")
    print("translation and subtitles still work for the features that are ready.")

# Structured, so it can be diffed between runtimes.
print()
print("features =", json.dumps(
    {{n: f["status"] for n, f in snapshot["features"].items()}}, indent=2))

# The optional extras, stated in terms of what a render will actually do.
print()
print("What a render will do here:")
multi = snapshot["features"]["diarization"]["status"]
bg = snapshot["features"]["background_preservation"]["status"]
print("  speakers      :", "one per actor" if multi == "ready"
      else "a single voice for the whole video")
print("  soundtrack    :", "the film's music and SFX are kept"
      if bg == "ready" else "replaced by the dub")
print("  lip sync      :", "on" if snapshot["features"]["lipsync"]["status"]
      == "ready" else "unavailable (audio dubbing still works)")
""")

SERVER_MD = md("""
## 7. Start the server

Starts the API **and** the built studio interface (the repo ships `web/`, so
no Node build is needed) in a background thread of this kernel. The cell prints
the public URL to open.

Re-running the cell stops the previous server first, so you never end up with
two processes fighting over the port.

Everything the server writes is mirrored to `server.log` under the runtime
root. The cell prints the path; read it if a render fails.
""")

SERVER = code("""
# Stop any server a previous run of this cell left behind.
_previous = globals().get("server")
if _previous is not None:
    _previous.stop()
    print("Stopped the previous server.")

PORT = 8000
server = runtime.start_server(host="0.0.0.0", port=PORT, ready_timeout=600)

url = runtime.notebook_public_url(PORT)
print()
print("=" * 62)
print("  VideoTrans is running")
print(f"  Open:  {url}/app/")
print("=" * 62)
print()
print("Keep this notebook tab open while you use the app.")
if server.log_file is not None:
    print("Server log:", server.log_file)
""")

LOG_MD = md("""
## 7a. Watch the server log (optional)

Prints the tail of `server.log` and follows it, so a render's progress and any
traceback appear here as they happen. Run it when something looks wrong.

Stop the cell with the notebook's stop button when you are done; the server
keeps running.
""")

LOG = code("""
if server.log_file is None:
    print("No log file (logging was off when the server started).")
else:
    print("Following", server.log_file, "- press the stop button to end this cell.\\n")
    try:
        # NOT a plain print loop: `start_server` mirrors stdout into this same
        # file, so printing a line it just read appends a copy that the next
        # read picks up, forever. `follow_log` suspends the mirror for its own
        # writes; that loop once grew server.log to 548 MB.
        runtime.follow_log(server.log_file)
    except KeyboardInterrupt:
        print("\\nStopped following. The server is still running.")
""")

USE_MD = md("""
## Use it

1. Open the URL printed above (append `/app/`).
2. On the **Dub** page, drop in a talking-head video and pick the source and
   target languages.
3. Press **Render** and watch the progress. Renders are written to the output
   directory from step 3.

**Pick a language whose voice model is installed.** The language list marks the
ones that are not, and the reason. The 11 Indian languages need `HF_TOKEN`; the
other languages need the XTTS extra, which is not in the default install
(`pip install "TTS>=0.22.0"`). Choosing one that is not installed is refused
before any work starts, with that reason.

### Multi-speaker

When pyannote installed successfully, a conversation is dubbed with **one voice
per actor**: each speaker is cloned from their own cleanest 6-12 s of speech,
and their lines are spoken in that voice throughout.

The finished project lists every speaker with its status:

| Status | Meaning |
|---|---|
| **Cloned** | a clean reference of their own was found |
| **Short reference** | under 6 s of usable audio; the clone will be weaker |
| **Fallback voice** | no usable audio at all; a generic voice is used |
| **Unresolved** | no reference, so their lines use the video's default voice |

**A speaker is never given another actor's voice.** A weak clone is reported
rather than disguised.

Each speaker has a **Change** button that lists only that speaker's own
candidate windows. Picking one re-speaks that speaker's phrases and leaves the
others byte-identical - it does not re-run transcription, translation or
Demucs.

### Overlapping dialogue

Two actors talking at once keep their own start times; neither is pushed behind
the other. The level is lowered only where they actually coincide, so a solo
line is never attenuated by a distant overlap. One speaker still cannot talk
over themselves.

### Keeping the soundtrack

Turn on **Keep original** on the Dub page to preserve the film's music, sound
effects and ambience under the new dialogue. That needs Demucs, and it is
**slow on CPU** - minutes per minute of video.

The layer order is dialogue, then the preserved soundtrack, then any music bed
you upload. An uploaded bed goes *under* the film's own score; it never
replaces it.
""")

TUNNEL_MD = md("""
## 7b. Open a public URL (Kaggle only)

Kaggle has no built-in port proxy like Colab's, so to open the interface from
your own browser we start a Cloudflare quick tunnel. It needs no account and
prints a throwaway `https://...trycloudflare.com` URL.

Run this cell, copy the URL it prints, and open it with `/app/` on the end. It
is valid only while this notebook is running.
""")

TUNNEL = code("""
import os
import re
import subprocess
import time

CLOUDFLARED = "/kaggle/working/cloudflared"
if not os.path.exists(CLOUDFLARED):
    print("Downloading cloudflared...")
    subprocess.run(
        ["wget", "-q", "-O", CLOUDFLARED,
         "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
        check=True,
    )
    os.chmod(CLOUDFLARED, 0o755)

log_path = "/kaggle/working/cloudflared.log"
log = open(log_path, "w+")
tunnel = subprocess.Popen(
    [CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
    stdout=log, stderr=subprocess.STDOUT,
)

public = None
for _ in range(60):
    time.sleep(1)
    log.flush()
    log.seek(0)
    match = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", log.read())
    if match:
        public = match.group(0)
        break

if public:
    os.environ["PUBLIC_HOST"] = public.replace("https://", "")
    print("=" * 62)
    print("  Public URL:", public + "/app/")
    print("=" * 62)
else:
    print("The tunnel did not come up. See", log_path)
""")

STOP_MD = md("""
## 8. Stop the server

Frees the port and the GPU memory. Run this when you are done, or just close
the notebook - the server is a daemon thread and dies with the kernel.
""")

STOP = code("""
server.stop()
print("Server stopped:", not server.is_running())

# Also stop the tunnel if one was started in the tunnel cell.
_tunnel = globals().get("tunnel")
if _tunnel is not None:
    _tunnel.terminate()
    print("Tunnel stopped.")
""")

TROUBLE = md("""
## Troubleshooting

| Symptom | Fix |
|---|---|
| `The server did not become ready ...` | Re-run the install cell; a partial install is the usual cause. Read `server.log` (path printed by the server cell), or run the log cell. |
| `ModuleNotFoundError: No module named 'torch'` | The CUDA install cell did not finish. Re-run it and read its output. |
| `Library libcublas.so.12 is not found or cannot be loaded` | torch is a CUDA 13 build. CTranslate2 (faster-whisper) needs CUDA 12. Re-run the install cell: it force-installs `torch==2.6.0+cu124` whenever the torch it finds is not a CUDA 12 one. |
| `Numba needs NumPy 2.2 or less. Got NumPy 2.5.` | Something installed a newer NumPy after the install cell (a manual `pip install -U transformers` does it). Re-run the install cell; it restores the `numpy<2.3` pin. |
| `cannot import name 'GenerationMixin'` | transformers 5.x was installed over the pinned 4.x. Re-run the install cell; it restores the `transformers<5` pin. Do not `pip install -U transformers` in this runtime. |
| `GPU: none - renders run on CPU` | The runtime is not a GPU one. Change it (see the top) and re-run from cell 1. |
| `Still missing N required model(s)` | The model cell did not finish. Re-run it; it is resumable. Check the disk-space line it prints. |
| A model says MISSING after it downloaded | The download was interrupted. Re-run the model cell. |
| `WARNING: ... was skipped (gated, no HF_TOKEN)` | Expected without a token. Indian languages are unavailable; everything else works. |
| A language is refused when you press Render | That language's voice model is not installed. The language list says which and why. Indian languages need `HF_TOKEN`; the others need `pip install "TTS>=0.22.0"`. |
| `401`/`403` when fetching models | The gated voice model needs a token. Accept the licence at <https://huggingface.co/ai4bharat/IndicF5>, set `HF_TOKEN`, re-run the token and model cells. |
| Renders are still slow | Confirm the GPU line in cell 2. A CPU render is correct, just slow. |
| Out of memory on a small GPU | Use a smaller Whisper: `os.environ["WHISPER_MODEL"] = "medium"` before starting the server. |
| The LLM test returns 502 | Expected with no LLM configured. Script Intelligence is optional and off by default; nothing else needs it. |
| The public URL stops working | The runtime recycled. Re-run from cell 1; the URL is valid only while the notebook runs. |
| Every speaker came out as one voice | pyannote did not install, so there is no speaker detection. Check the install cell for the `--  pyannote` line. |
| A speaker says "Fallback voice" | That actor had no usable clean audio, so a generic voice is used. Their reference window can be changed from the Speakers panel. |
| A speaker says "Short reference" | Under 6 s of clean speech was found for them. The clone still works but is weaker; try changing their reference window. |
| The soundtrack was not preserved | Demucs did not install or the *Keep original* toggle was off. Check the install cell for the `--  demucs` line. |
| Background preservation is very slow | Demucs is roughly realtime on CPU and much faster on a GPU. Check the device line in the Verify cell. |
| A render says "reduced" | That is a fallback, not a failure - the Verify cell lists exactly which stage. The run summary on the finished project says the same thing. |
""")

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

SHARED = [TITLE, GET_CODE_MD, GET_CODE, CHECK_MD, CHECK, RUNTIME_MD, RUNTIME,
          TOKEN_MD, TOKEN, INSTALL_MD, INSTALL, MODELS_MD, MODELS,
          VERIFY_MD, VERIFY,
          SERVER_MD, SERVER, LOG_MD, LOG, USE_MD]

COLAB = SHARED + [STOP_MD, STOP, TROUBLE]
KAGGLE = SHARED + [TUNNEL_MD, TUNNEL, STOP_MD, STOP, TROUBLE]


def write(path: str, cells: list[dict]) -> None:
    _assign_ids(cells)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3",
                           "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"{path}: {len(cells)} cells ({n_code} code)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    write(os.path.join(here, "colab.ipynb"), COLAB)
    write(os.path.join(here, "kaggle.ipynb"), KAGGLE)

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
# parenthesised expression so it can be pasted into a cell as `REPO_DIR = ...`.
REPO_DIR_EXPR = ('(\n'
                 '    "/kaggle/working/video-dubbing" if os.path.isdir("/kaggle")\n'
                 '    else "/content/video-dubbing"\n'
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
# Multiva - run the whole pipeline on a free GPU

Dub a talking-head video into another language - transcribe, translate, clone
the speaker's voice, lip-sync - on the machine this notebook is attached to,
not on your laptop.

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
# Kaggle: Add-ons -> Secrets). Uncomment the next line only if you must
# hardcode it.
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

INSTALL = code(f"""
import os

# Locate the checkout and work from it, so this cell does not depend on an
# earlier cell having run in this session.
REPO_DIR = {REPO_DIR_EXPR}
os.chdir(REPO_DIR)

import subprocess
import sys

print("Installing the CUDA PyTorch build (the slow part)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch>=2.2.0", "torchaudio>=2.2.0", "torchvision>=0.17.0",
     "--index-url", "https://download.pytorch.org/whl/cu124"],
    check=True,
)

# The CUDA build must have won, or every render falls back to the CPU.
import torch  # noqa: E402
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())

cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
print("Installing:", " ".join(cmd))

# Stream pip to the notebook so a failure shows which package died.
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
proc.wait()

if proc.returncode != 0:
    print()
    print("=" * 62)
    print("INSTALL FAILED. The real cause is in the pip output above.")
    print("=" * 62)
    raise SystemExit("The install failed; see the pip output above this line.")

print("\\nInstall complete.")
""")

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
print("  Multiva is running")
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
import time

if server.log_file is None:
    print("No log file (logging was off when the server started).")
else:
    path = server.log_file
    print("Following", path, "- press the stop button to end this cell.\\n")
    with open(path, encoding="utf-8", errors="replace") as fh:
        # Start at the top of the current log, then follow the tail.
        print(fh.read(), end="")
        try:
            while True:
                line = fh.readline()
                if line:
                    print(line, end="")
                else:
                    time.sleep(0.5)
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
""")

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

SHARED = [TITLE, GET_CODE_MD, GET_CODE, CHECK_MD, CHECK, RUNTIME_MD, RUNTIME,
          TOKEN_MD, TOKEN, INSTALL_MD, INSTALL, MODELS_MD, MODELS,
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

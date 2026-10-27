# deploy/

Three ways to run Multiva on a GPU instead of a CPU. The engine detects its
device on its own — CUDA, then Apple silicon, then CPU — so nothing in the code
changes between them; only where it runs does.

| Path | Use it when | Entry point |
|---|---|---|
| [`docker/`](docker/) | You have a GPU box, or Docker with the NVIDIA Container Toolkit | `docker compose -f deploy/docker/docker-compose.yml up --build` |
| [`notebooks/colab.ipynb`](notebooks/colab.ipynb) | You want a free GPU and a browser, with no local install | Open in Colab, pick a GPU runtime, Run all |
| [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb) | Same, on Kaggle | Open in Kaggle, pick a GPU accelerator + Internet, Run all |

All three share `engine/deploy_helpers.py`: platform detection, the writable
runtime tree, the GPU report, starting the server in a background thread, and
building the public URL. That is deliberate — the Colab and Kaggle notebooks are
thin wrappers over the same helper, so they cannot drift apart.

## Why this exists

Voice synthesis is a diffusion transformer sampled once per phrase, so it
dominates the render. On CPU the pipeline is correct but slow; on a GPU the same
clip renders many times faster. Nothing else about the code is different.

## What each one does, in order

1. Get the code (clone the repo).
2. Check the machine (GPU visible? how much VRAM? which preset?).
3. Point the runtime at a writable directory — uploads, renders, model cache.
4. Install the **CUDA PyTorch build first**, then the rest of
   `requirements.txt`. Installing `requirements.txt` alone would fetch the
   CPU-only wheels and every render would silently fall back to the CPU.
5. Fetch the models (~7.5 GB, resumable).
6. Start the server — the API and the built studio interface.
7. Print the URL to open.
8. Stop the server.

## The gated voice model

`ai4bharat/IndicF5` — the voice that speaks the 11 Indian languages — is a
**gated** repository. Accept its licence on the model page, create a read token,
and set `HF_TOKEN`. Without it that one model is skipped and those languages are
marked unavailable; transcription, translation, subtitles and lip sync all still
work. Steps: [`docs/GATED-MODELS.md`](../docs/GATED-MODELS.md).

- Docker: put `HF_TOKEN=hf_...` in a `.env` beside `docker-compose.yml`.
- Colab/Kaggle: set it in the notebook's secrets panel, or `os.environ` before
  the model cell.

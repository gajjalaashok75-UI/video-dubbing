# Working notes for agents

## Layout

- `engine/` — FastAPI app and pipeline. `app.py` holds the API and the boot
  warmup; `models.py` owns the model catalogue and the Hugging Face cache
  checks; `subtitles.py` burns captions; `tts_engines.py` wraps the voices.
- `apps/studio/` — React + Tauri desktop studio. `src/` is the UI; the built
  output is committed to `web/` and served by the engine.
- `scripts/` — `selftest.py` (end-to-end), `download_models.py`, `fit_bench.py`.
- `docs/` — architecture, testing, audit, gated-model instructions.

## Environment on this machine

- Engine is started with `../venv/bin/python -m uvicorn app:app --host 0.0.0.0
  --port 12000` from `engine/`. Port **12000** is the public one; `selftest.py`
  defaults to 8000, so pass `MULTIVA_URL=http://127.0.0.1:12000`.
- `run.sh` is the single entry point; `--setup --yes` installs everything
  (venv, models, frontend, desktop window) and exits.
- Desktop build: `cd apps/studio && npx tauri build`. Needs Rust plus
  `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`.
  A headless container cannot open a GTK window without a display; use
  `xvfb-run -a --server-args="-screen 0 1600x1000x24"` to verify it launches.

## Models

- Only one model is gated: `ai4bharat/IndicF5` (the voice). It needs
  `HF_TOKEN` in `.env`; without it the 11 Indian languages are marked
  unavailable and that one download is skipped — everything else still works.
  `models.is_cached()` checks the files a model actually loads, not just the
  folder, because a failed gated fetch leaves only a `README.md` behind.
- On this machine the gated voice model is absent, so a real dub cannot run.
  Subtitle renders (`kind=subtitles_translated`) exercise transcription,
  translation and font rendering end to end without it.
- `engine/vendor/f5_tts/` is IndicF5's OWN fork of f5-tts, not the pip release.
  The checkpoint loads into both with zero missing keys, but only the fork
  reproduces the reference text; the pip release returns fluent audio that is
  the wrong words ("कर दो कर दो कर दो"). `tts_engines.py` puts the vendor dir
  first on `sys.path` and `_use_vendored_f5_tts()` raises if the release won.
  Do not "clean up" the vendored copy or replace it with the pip package.
  Note the fork's `infer_batch_process` returns `(wave, sr, spec)` directly,
  while the release returns a generator.

## Lint and test

- Python: the project enforces only correctness rules
  (`ruff check --select E9,F63,F7,F82`). Full `ruff check` reports ~260
  pre-existing style findings; do not treat those as regressions.
- Frontend: `npx tsc -b && vite build` from `apps/studio`; `npx oxlint src`
  for warnings.
- End-to-end: `MULTIVA_URL=http://127.0.0.1:12000 ./venv/bin/python
  scripts/selftest.py [clip.mp4]`. With no clip it runs 38 checks; with a clip
  it adds the render, editor, export and voice-over checks.

## Deployment (deploy/)

- `engine/deploy_helpers.py` is the shared runtime for Colab, Kaggle and Docker:
  `detect_platform`, `configure_runtime`, `hardware_report`, `start_server`,
  `notebook_public_url`. Keep the notebooks thin — the logic belongs here.
- `configure_runtime()` sets `MULTIVA_UPLOAD_DIR` / `MULTIVA_OUTPUT_DIR` and MUST
  run before `app` is imported: `engine/app.py` snapshots `UPLOAD_DIR` at import.
- The device is chosen automatically (`_pick_device` in `tts_engines.py`,
  `lip_sync.py`, `translation_v2.py`). Do NOT add GPU-only code paths; the fix
  for "slow on CPU" is to run on a GPU, not to special-case one.
- In the notebooks and the Dockerfile the CUDA PyTorch wheels are installed
  FIRST, from the cu124 index. Installing `requirements.txt` alone pulls the
  CPU-only wheels and every render silently falls back to the CPU.
- `deploy/docker/entrypoint.sh` subcommands take their own name out of `$@`
  (`"${@:2}"`); a bare `"$@"` forwards `models` as a flag and the script rejects it.
- Docker build context is the repo root. `.dockerignore` (root) and
  `deploy/docker/Dockerfile.dockerignore` both trim it; keep them in sync with
  `.gitignore`. The build was verified: `docker build -f deploy/docker/Dockerfile .`
  then `docker run` answers `/api/health` and serves `/app/`.

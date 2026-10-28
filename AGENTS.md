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
  defaults to 8000, so pass `VIDEOTRANS_URL=http://127.0.0.1:12000`.
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
- `MODELS` entries name the exact files a repo loads, and those names must match
  the repo's `main` revision. `facebook/nllb-200-distilled-600M` and
  `microsoft/wavlm-base-plus-sv` ship `pytorch_model.bin`, not
  `model.safetensors`. Naming a file a repo does not publish makes the model
  read as MISSING forever, so it is re-downloaded on every run and the setup
  page never stops calling it absent. When adding or editing an entry, confirm
  the filenames against the repo's file list first.
- `engines._hf_cached()` delegates to `models.is_cached()` on purpose, so the
  studio catalogue and the setup page cannot disagree about what is present.
  The catalogue also offers models the `MODELS` manifest does not name (the
  lighter Whisper and NLLB alternatives). For those `is_cached()` accepts any
  real weight file, because there is no exact filename to check; without that
  fallback a downloaded alternative reads as absent forever and is re-fetched
  every run. A failed gated fetch still leaves only a README, so it is still
  rejected. When you add a catalogue option that is not in `MODELS`, expect
  readiness to come from this fallback rather than an exact-file check.
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
- That fork is NOT committed and NOT on PyPI, so it is fetched at setup time by
  `scripts/vendor_f5_tts.py` (idempotent; `--check` reports, changes nothing).
  `run.sh` and both notebooks call it. Without it every fresh clone lost the
  voice stage to the guard above. It verifies `AdaLayerNormZero_Final`, the same
  marker the guard tests, so setup cannot pass while synthesis still refuses.

## Lint and test

- Python: the project enforces only correctness rules
  (`ruff check --select E9,F63,F7,F82`). Full `ruff check` reports ~260
  pre-existing style findings; do not treat those as regressions.
- Frontend: `npx tsc -b && vite build` from `apps/studio`; `npx oxlint src`
  for warnings.
- Unit tests: `./venv/bin/python -m unittest discover -s tests -v`. Seconds,
  no GPU, no models, no network. They pin the things that broke a hosted run:
  the PyAV ceiling that keeps `metadata_errors` available, the log follower not
  feeding its own output back into `server.log`, the checkout and runtime tree
  agreeing on the platform, and the vendored f5_tts fork.
- End-to-end: `VIDEOTRANS_URL=http://127.0.0.1:12000 ./venv/bin/python
  scripts/selftest.py [clip.mp4]`. With no clip it runs 38 checks; with a clip
  it adds the render, editor, export and voice-over checks.

## Deployment (deploy/)

- `engine/deploy_helpers.py` is the shared runtime for Colab, Kaggle and Docker:
  `detect_platform`, `configure_runtime`, `hardware_report`, `start_server`,
  `notebook_public_url`. Keep the notebooks thin — the logic belongs here.
- `configure_runtime()` sets `VIDEOTRANS_UPLOAD_DIR` / `VIDEOTRANS_OUTPUT_DIR` and MUST
  run before `app` is imported: `engine/app.py` snapshots `UPLOAD_DIR` at import.
  It also pins `HF_HOME` to `<runtime>/hf` via `setdefault`, so weights land on
  the writable volume instead of `~/.cache/huggingface`, which hosted runtimes
  discard between sessions.
- `start_server()` mirrors stdout/stderr to `<runtime>/logs/server.log` (see
  `server_log_path()`); `ServerHandle.tail()` reads it back. The tee is installed
  once per process (`_install_log_tee`) — the engine's logging handler captures
  `sys.stderr` at creation, so re-teeing on every start would write each line
  twice. Do not move the tee into `stop()`.
- `deploy/notebooks/build_notebooks.py` generates `colab.ipynb` and
  `kaggle.ipynb` from one source. Edit the generator, not the notebooks, then
  run it: `./venv/bin/python deploy/notebooks/build_notebooks.py`. It assigns
  each cell a stable id (position + source hash), which nbformat 4.5 requires;
  editing a notebook by hand instead leaves cells without ids and the
  validator warns.
- The notebooks' first cell runs `git fetch` + `git reset --hard FETCH_HEAD`
  when the checkout already exists. Never run that cell against this working
  copy: it discards uncommitted work.
- Script intelligence (`engine/llm.py`) is a SEPARATE, OPTIONAL stage from the
  model downloads. Picking a provider there never replaces, skips or affects a
  downloaded model: it only rewrites one over-long line when the user presses
  Fit on a segment (`POST /jobs/{id}/segments/{i}/fit`). No render/worker path
  calls `llm.*`, so an unreachable provider cannot break a dub.
- Provider config lives at `~/.videotrans/llm.json` (mode 0600, holds keys) and is
  overlaid by the environment, which wins. Env names: `VIDEOTRANS_LLM_PROVIDER`,
  `VIDEOTRANS_LLM_MODEL`, `VIDEOTRANS_LLM_URL`/`OLLAMA_HOST`, `VIDEOTRANS_LLM_CUSTOM_URL`
  and `VIDEOTRANS_LLM_API_KEY` (generic key for whichever provider is chosen — the
  only way to configure the OpenAI-compatible `custom` provider headlessly).
  Per-vendor keys: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`,
  `GROQ_API_KEY`, `XAI_API_KEY`. `VIDEOTRANS_LLM=off` disables the stage.
- `llm.status()` must never return a key, only `has_key`. Keep it that way: the
  response is served to the studio and appears in `/api/health`.
- Provider model ids drift and the "Test" button makes a real call, so a stale
  default fails the first time someone tries it. Confirm defaults against the
  provider's own docs before changing them (xAI moved `grok-2-latest` on).
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

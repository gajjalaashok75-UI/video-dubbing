# Working notes for agents

## Layout

- `engine/` — FastAPI app and pipeline. `app.py` holds the API and the boot
  warmup; `models.py` owns the model catalogue and the Hugging Face cache
  checks; `subtitles.py` burns captions; `tts_engines.py` wraps the voices.
  `lip_sync.py` must never hold the whole clip: it reads the file twice (count
  and detect, then render) and keeps one batch. A clip with NO face anywhere
  raises `lip_sync.NoFaceDetected` (a `LipSyncError` subclass); that is not a
  failure — `process_video_task` and `rerender_task` catch it and mux the
  finished dub onto the untouched picture via `av_sync.mux`, recording
  `job["lip_sync"] = {"skipped", "reason"}` for the run summary. Do not turn it
  back into a failed job: a landscape or a screen recording is a normal input.
  Decoding a 60-second 1080x1920 clip into a list is 1,799 frames, 10.4 GiB —
  more than a hosted runtime has, so the render is killed and takes the session
  with it.
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

## Multi-speaker + background preservation (planned)

- `docs/INTEGRATION-multispeaker-background.md` is the exploration report for
  adding multi-speaker dubbing and original-background preservation. Nothing in
  it is implemented yet; read it before touching `dubbing.py`, `app.py` or
  `reference_audio.py`.
- `references/` vendors four read-only reference repos (Python-Autodub,
  video-dubbing-system, Ai-Vedio-Dubbing, PolyDub). Their `.git` dirs were
  removed so they commit as plain files. They are inputs to the report, not
  build dependencies, and nothing imports from them.
- The core chain is fixed: faster-whisper → NLLB-200 → IndicF5 → Wav2Lip.
  IndicF5 stays the Indian-language voice model; do not swap in F5-TTS or XTTS
  when porting techniques from the references, which mostly use F5-TTS/XTTS.
- Two invariants must survive any port: `dubbing.assemble` returns exactly
  `video_duration` seconds, and per-phrase audio stays cached in
  `<workdir>/units/` so one edit re-synthesizes one phrase.
- `translation_v2.translate_segments` translates ONE SENTENCE PER CALL, then
  retries any chunk that came back empty, then falls back to translating the
  whole segment. NLLB handed two sentences can translate the first and drop the
  rest — that is how "Nice to meet you." vanished from "Hi, my name is John.
  Nice to meet you.", leaving a hole in the dub. Do not batch a segment's
  sentences into one call to save time; that reintroduces the dropped clause.
  The `translate` stage log records `dropped_translations` and `dropped_segments`
  so a regression is visible without listening to the output.

## Multi-speaker dubbing (Phases 1-3 done)

- `engine/speakers.py` owns the registry: one entry per detected voice, with
  `label`, `total_speech`, `segment_count`, `reference`, `reference_score`,
  `fallback`. `reference_for(unit, speakers, fallback)` is the resolution chain
  every synthesis path goes through. A job with no registry resolves to the
  global reference, which is the pre-diarization behaviour.
- `speaker_diarization.assign_segments` SPLITS a segment where the speaker
  changes, using the per-word labels. One Whisper segment routinely spans a
  question and its answer; labelling the segment as a whole spoke half the
  exchange in the other actor's cloned voice. Runs shorter than
  `MIN_SPLIT_SECONDS` fold into their neighbour, and more than four changes in
  one segment is treated as diarizer noise and left whole, so the text is never
  spoken twice and no single word is given to the wrong actor.
- `engine/speaker_diarization.py` is the optional stage. pyannote is the only
  backend; `Diarizer` is the interface a second one would implement. `diarize()`
  returns `[]` rather than raising on EVERY failure path - no pyannote, no
  token, a crash, zero turns - so a render never fails because the speaker
  detector did. `release()` drops the model before IndicF5 loads; they share one
  GPU and the diarizer is idle once the turns are known.
- `speaker_diarization.merge_spurious_speakers(segments, turns, min_speech,
  keep_min)` folds a label with almost no speech into its temporally nearest
  real speaker. A diarizer INVENTS speakers as readily as it misses them: a
  cough, a door, a music sting becomes a label, and a monologue with one such
  blip becomes a "two-speaker" video with a phantom voice in the studio. A label
  is a blip only when it is under BOTH the absolute floor (1s,
  `DIAR_MIN_SPEAKER_SPEECH`) and `MIN_SPEAKER_SHARE` (5%) of the speech, so a
  short 50/50 two-hander keeps both speakers. `keep_min` (wired from
  `DIAR_MIN_SPEAKERS`) is a floor on survivors, so a clip where everything is a
  blip keeps its richest rather than collapsing to nothing. Wired in `app.py`
  right after `assign_segments`, before the count is treated as a cast list.
- `reference_audio.build_references()` cuts one reference per speaker. The
  fallback ladder is 6-12s same-speaker -> stitched short windows -> 2-6s with a
  warning -> generic voice -> unresolved. It must NEVER use another detected
  actor's voice; `tests/test_speaker_references.py` holds that line as a test.
  `candidate_windows(min_seconds=...)` is lowered to the 2s floor by
  `build_references`, because filtering at 6s makes rungs 2-5 unreachable.
- `dubbing.plan_timeline` emits `speaker` and `ref_index`; `synth_unit` and
  `synthesize_timeline` resolve each unit's own reference; `build_dubbed_track`
  builds one `DurationModel` per speaker. `natural_for(text, seg, label)` is how
  a per-speaker generator estimate gets taken BEFORE the closing squeeze in
  `plan_timeline` - doing it afterwards moves every phrase and breaks the
  length invariant.
- Verify a real 2-3 speaker clip with
  `python scripts/multispeaker_check.py plan|check clip.mp4`. `check` prints a
  phrase-vs-reference similarity matrix; the diagonal is the pass.
- A re-clone has to move the references the PLAN resolves, not just
  `job["reference"]`: `_rebind_reference` repoints the registry entries that
  were using the old global clip, and leaves a two-hander's per-speaker clips
  alone. A per-speaker re-clone must not overwrite the job-wide reference.
- `docs/INTEGRATION-multispeaker-background.md` is the design. All six
  phases are implemented; see the sections below.

## Background preservation (Phase 4 done)

- `engine/source_separation.py` splits the original soundtrack with Demucs
  `htdemucs --two-stems vocals`. The `no_vocals` stem is the bed; the `vocals`
  stem is what ASR and the voice references are read from. On by default, so a
  dub keeps the film's score unless the user turns it off (`PRESERVE_BACKGROUND`
  tunable, `VIDEOTRANS_PRESERVE_BACKGROUND`, or the per-render
  `preserve_background` query param — most specific wins). Every failure — no
  Demucs, no weights, a crash, bad output, a timeout — returns `None` and
  leaves the dialogue-only pipeline intact.
- The separation input is HQ stereo 44.1 kHz, NOT the 16 kHz mono ASR track.
  Demucs needs the full band and the stereo image; the two extractions coexist
  because ASR still falls back to the 16 kHz one.
- Chunking is 10 minutes at a time, one Demucs invocation for all chunks
  (`-j 1`). `ffmpeg -f segment -c copy` was measured to reassemble
  sample-exactly, so the segmentation is safe; Demucs is not, so every chunk's
  output is fitted back to its own input length before joining. A model that
  pads each chunk would otherwise drift by seconds over a film.
- The finished dub stores `stems` ITSELF on the job, not just
  `background_path`/`vocals_path`. `run_summary`, `artifact_cache` and sibling
  reuse all read `job["stems"]`; when the dub path omitted it, a render that
  did preserve the soundtrack reported "background not preserved" and warned
  that the music was lost.
- Stems cache in `<workdir>/stems/` with `stems.json` recording the input's
  size, mtime, model, chunk size and rate. A re-trim changes size and mtime, so
  it invalidates the cache automatically. `stems` is persisted on the manifest
  so a sibling output of the same clip reuses them.
- `av_sync.mix_layers` is the mixer: one ffmpeg pass, layer order dialogue >
  preserved background > user music bed, `normalize=0`, and length enforced
  ONCE at the end as `atrim` then `apad=whole_dur=<target>`. Do not move that
  length fix-up per layer — `apad` with no argument pads to a buffer boundary,
  not a duration, and `loudnorm` upstream makes the input length unpredictable;
  per-layer enforcement came out 40 ms short on a 12.96s clip.
- A user-uploaded music bed goes UNDER the preserved background. It never
  replaces it; both are kept and both are audible.
- `_mix_audio` returns `None` when there is nothing to mix with, which is what
  tells the caller to keep the dialogue-only track. That track is kept as its
  own artifact either way.
- Verify a real BGM-heavy clip with
  `python scripts/background_check.py plan|check clip.mp4`. It writes six
  listenable artifacts; `3_background.wav` is the one that matters.
- Phase 6 (polish) is NOT implemented; Phase 5 is done, see below.

## Overlapping dialogue (Phase 5 done)

- `plan_timeline` schedules with ONE CURSOR PER SPEAKER, not one global cursor.
  A global cursor serialised the whole video, so in a conversation the second
  actor could not start until the first had finished — `SPEAKER_01` at 11.5s was
  pushed to 13.0s because `SPEAKER_00` was still talking. Do not reintroduce a
  single `cursor` variable. The closing squeeze is per speaker for the same
  reason: one running cursor there would re-serialise everything after a squeeze.
- `next_start_for` looks ahead to the speaker's OWN next line, not the next line
  in the list. The following segment is usually the other actor, and treating
  their start as this speaker's deadline would cap a phrase by someone else's
  turn. For one speaker the two are the same position, so single-speaker
  borrowing is unchanged.
- `assemble(plan, waves, video_duration, overlap_gain=True)` keeps additive
  placement and multiplies by `_overlap_gain(...)`. That curve is `1/sqrt(n)` for
  `n` active voices, then a LOCAL CEILING applied only inside the overlap. The
  ceiling iterates (bounded at 24) because a single pass leaves the peak alive on
  the smoothed ramp into an overlap; measured, 16 iterations take a 1.27 peak
  under 1.0 while a solo passage stays at exactly 0.5.
- The global `if peak > 1: track /= peak` is a LAST-RESORT guard, never the
  strategy. If it fires on an overlap it scales the whole film, which is audible
  everywhere the overlap is not. It now prints when it fires.
- With one speaker the gain curve is exactly 1.0 everywhere, so a single-speaker
  track is byte-for-byte unchanged. `tests/test_overlap.py` locks that, and
  `overlap_gain=False` reproduces the pre-Phase-5 mix for A/B.
- Overlap is an ASSEMBLY concern, so cache identity is untouched: editing one
  overlapping phrase leaves the other speaker's cached WAV byte-identical. Both
  `_rebuild_track` and `synthesize_timeline` go through `assemble`, so an edit
  cannot be levelled differently from the original render.
- Two diarization turns overlapping does NOT mean the text is spoken twice.
  Phase 5 consumes Phase 2's assignments; an ambiguous window stays unresolved
  and yields one unit, not two.
- `dubbing.overlap_profile(plan)` reports the regions where two or more voices
  are active. Purely diagnostic; nothing branches on it.
- Verify a real clip with `python scripts/overlap_check.py plan|check clip.mp4`.
  It prints the active-speaker table and writes one isolated track per speaker.
- Phases 1-6 are all done; see the sections below.

## Reporting layer (Phase 6 done)

- `engine/readiness.py` is the ONE readiness answer. The studio, the API and
  every check script read it; nothing else formats its own opinion. A feature is
  `ready` / `fallback_available` / `optional_missing` / `blocked`, and only a
  `tier == "core"` feature can make `ready` false. Never load a model in a probe.
- `readiness.token_state()` returns `{"present": bool}` and NOTHING else — no
  value, no prefix, no length. `readiness.redact()` exists so a hand-built
  payload cannot leak one. `tests/test_phase6_reporting.py` asserts no token
  material reaches `snapshot()`, `run_summary.build()`, or any endpoint.
- `engine/run_summary.py` is what a finished job did: `stage_rows()` reports each
  stage as done/skipped/fallback with a reason, `warnings()` is the plain-
  sentence list where a fallback becomes visible, and `_match_status()` is the
  only place a voice score becomes a word. A degraded run is a RESULT, not an
  error — do not make an optional stage failing turn a render into a failure.
- `engine/artifact_cache.py` is the dependency graph. `invalidated_by(["field"])`
  answers what a change costs; `reference_invalidates(label)` is why a per-speaker
  re-clone is cheap. Validity is a comparison, never `os.path.exists`.
  The transcript depends on `stems` because it is read from the vocals stem —
  that is a real dependency, not an adjacency.
- Speaker status lives in `speakers.status_of()` and is DERIVED, not stored, so
  it cannot go stale when a reference is re-cut. `describe()` always emits the
  stable `label`; `name` is the alias and never replaces it.
- Aliases are an overlay only. `label` is what the unit cache, the plan, the
  reference files and the manifest key on.
- The per-speaker voice score is a DIAGNOSTIC, not identity proof. `match` /
  `uncertain` / `different` / `unmeasured`; no universal pass threshold is
  claimed because none has been validated.
- `scripts/final_check.py` is the canonical end-to-end validation. `plan` must
  work with no torch installed — that is the machine `plan` is for — so the
  heavy imports are behind `_heavy()` and the readiness gate runs BEFORE them.
- `docs/VALIDATION-MATRIX.md` is the real-video matrix. Every case is BLOCKED
  here. Do not mark a case PASS without actually running it.

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
  agreeing on the platform, the vendored f5_tts fork, lip sync streaming (a
  fake model and detector render a synthetic clip and the bytes are compared
  with the old in-memory algorithm), and `/videos/` listing a run in progress.
- The Verify cell's assertions must agree with `requirements.txt`. A check that
  contradicts a pin blocks a session that is actually correct, which is how
  `assert av_major < 18` failed every run: the pin is `av>=11,<19` and 18.1.0 is
  the last release with `metadata_errors`, so the check must accept it and
  reject only 19+. Same for NumPy: assert the range `>=1.26,<2.3`, not one patch
  release, or the next 2.2.x that pip resolves fails the cell.
- End-to-end: `VIDEOTRANS_URL=http://127.0.0.1:12000 ./venv/bin/python
  scripts/selftest.py [clip.mp4]`. With no clip it runs 38 checks; with a clip
  it adds the render, editor, export and voice-over checks.

## Deployment (deploy/)

- `engine/deploy_helpers.py` is the shared runtime for Colab, Kaggle and Docker:
  `detect_platform`, `configure_runtime`, `hardware_report`, `start_server`,
  `notebook_public_url`. Keep the notebooks thin — the logic belongs here.
- `notebook_public_url()` prefers Colab's port proxy (`_colab_proxy_url`, via
  `google.colab.output.eval_js`), then `PUBLIC_HOST` (the Kaggle cloudflared
  tunnel), then loopback. The proxy call is retried and validated: `eval_js`
  returns `None` when the frontend is busy — which it is right after
  `start_server`, with a model loading — and a single unvalidated call printed
  `http://127.0.0.1:8000`, unreachable from the user's browser. Only an
  `https://` reply that embeds the port counts as a proxy. The server cell also
  warns when it can only show loopback, so the failure is visible.
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
- The notebooks' first cell updates an existing checkout with `git fetch` +
  `git reset --hard FETCH_HEAD`, but only when the directory is a git checkout.
  A copy delivered as a zip has no `.git`, so the cell keys on `requirements.txt`
  instead: without that it treated the copy as absent and cloned the remote over
  it, silently discarding the code the user was given. Never run that cell
  against this working copy: it discards uncommitted work.
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
- It must be a **CUDA 12** torch, not just a CUDA one. CTranslate2
  (faster-whisper) dlopens `libcublas.so.12` on the first encode; torch >= 2.7
  ships CUDA 13 wheels whose cuBLAS is `libcublas.so.13`. A hosted image that
  already had torch 2.11+cu130 satisfied the old `torch>=2.2.0`, so the cu124
  install did nothing and every render died ~6s in (`Library libcublas.so.12 is
  not found or cannot be loaded`). `requirements.txt` caps the torch stack at
  `<2.7`, the Dockerfile pins `2.6.0+cu124` and asserts `torch.version.cuda`,
  and the notebook cell force-installs that build whenever the torch it finds
  is not CUDA 12. Keep the three in step.
- `speech_to_text_v2` retries on the CPU when the GPU cannot be used at all
  (`_demote_to_cpu`, `fallback_reason()`). A missing CUDA library is not
  fixable by a smaller model or a decode setting, and ctranslate2 only touches
  the GPU when the first segment is pulled - hence `_decode` drains the
  generator inside the retry. Do not let a CUDA failure fail a job: the render
  is slower, not broken, and the summary says why via `stt_fallback`.
- NumPy is capped at `<2.3` because numba (through librosa, which the voice
  engine and Wav2Lip both load audio with) refuses anything newer: "Numba needs
  NumPy 2.2 or less". Unpinned, IndicF5 fails to load at boot and Wav2Lip's
  warmup fails with it. The notebook re-asserts the pin after the optional
  extras, since `pip install -U transformers` can pull a newer NumPy in.
- The pin only works if the KERNEL actually runs the new NumPy. The Check cell
  imports torch and numpy to print the machine report, and the install cell then
  `pip install`s the pinned NumPy in a SUBPROCESS. The subprocess sees 2.2.6, but
  the kernel keeps the image's 2.5.4 in `sys.modules`, and the server (a thread
  of that kernel) inherits it — numba refuses it and the voice model plus lip
  sync fail at boot while every check above passed. So the Check cell's probe
  runs in a SUBPROCESS and never imports torch/numpy into the kernel, and the
  install cell calls `deploy_helpers.stale_modules()` and raises `RESTART
  REQUIRED` if the kernel is holding a version that is no longer on disk. Do not
  move those imports back into the kernel: a compiled extension cannot be swapped
  once imported, so the only correct answer is a restart.
- The optional engines (pyannote for multi-speaker, demucs for background) are
  installed with `-c deploy/constraints.txt`, never bare. `pyannote.audio==3.3.2`
  pulls speechbrain and pytorch-metric-learning, and the versions already on a
  hosted image ask for a NEWER torch; pip then replaces the CUDA 12.4 build with
  a CUDA 13 one. That breaks CTranslate2 (`libcublas.so.12` is gone) and leaves
  the CUDA 13 `nvidia-*-cu13` wheels beside the CUDA 12 ones, so `import torch`
  dies with `libtorch_cuda.so: undefined symbol: ncclCommResume`. The constraints
  file holds `torch==2.6.0` (`==` ignores the local `+cu124` segment), NumPy under
  numba's cap, and the transformers/huggingface-hub pair.
- The torch probe's install branch PURGES the `nvidia-*-cu13` wheels before it
  reinstalls, then RE-PROBES. Without the purge, `pip install torch` cannot clear
  the stray CUDA 13 wheels and the probe fails again on the next run; the old
  cell then asked for a restart every time, an infinite loop. And the restart is
  gated on `deploy_helpers.stale_modules()` for torch only: if the kernel never
  imported torch, replacing it on disk needs no restart at all, so the cell
  continues straight through.
- Lip sync is non-fatal in both render paths: `NoFaceDetected` and any other
  `LipSyncError` mux the dub onto the untouched picture via
  `_deliver_without_lip_sync`. `no_face` in the job's `lip_sync` note is what
  makes the run summary say "no face" rather than "lip sync did not run".
- `deploy/docker/entrypoint.sh` subcommands take their own name out of `$@`
  (`"${@:2}"`); a bare `"$@"` forwards `models` as a flag and the script rejects it.
- Docker build context is the repo root. `.dockerignore` (root) and
  `deploy/docker/Dockerfile.dockerignore` both trim it; keep them in sync with
  `.gitignore`. The build was verified: `docker build -f deploy/docker/Dockerfile .`
  then `docker run` answers `/api/health` and serves `/app/`.
- The duration request to IndicF5 must be CALIBRATED across scripts, not a raw
  byte count. F5's own heuristic (`ref_audio_len + ref_audio_len / ref_text_len *
  gen_text_len`) counts `gen_text_len` in UTF-8 BYTES, so an English reference
  conditioned on a Devanagari target is asked for ~3x the time its words need (3
  bytes/char vs 1). The timeline then overflows and `plan_timeline`'s closing
  squeeze compresses every phrase to fit — words mangle and whole lines vanish.
  `IndicF5Engine.natural_duration` therefore takes `source_lang` and, when it is
  known, splits the estimate the way `DurationModel` does: target
  characters-per-second for script density, and the reference's own tempo
  relative to the source language. Measured on an English->Hindi 9.0s clip:
  before, 17.2s requested -> 1.59x squeeze and the opening line disappeared;
  after, 8.0s -> 1.03x and all three lines round-trip. Keep the byte mirror only
  as the no-`source_lang` fallback. Pinned by `tests/test_duration_calibration.py`.
- pyannote 3.3.2 must load under the pinned huggingface_hub 2.x / torch 2.6, or
  diarization silently fails and every job drops to single-speaker (a two-hander
  comes back in one voice). `speaker_diarization._apply_compat_shims` translates
  `use_auth_token=` to `token=` and restores `torch.load(weights_only=False)` for
  pyannote's own checkpoint classes; both are narrow, idempotent, failure-safe,
  and applied before any pyannote import. On the versions pyannote expects they
  are no-ops.
- Demucs' `--no-deps` install must name `sphn`, `einops`, and `pyyaml` as well
  as the dora-search chain. `demucs.api` imports `sphn` at module load and it is
  NOT implied by `find_spec("demucs")`, so a Colab install reported background
  preservation ready while `python -m demucs.separate` died on `import sphn` and
  the render silently dropped the soundtrack. Keep `REQUIRED_MODULES` in
  `engine/source_separation.py` and `DEMUCS_DEPS` in
  `deploy/notebooks/build_notebooks.py` in sync.
- Diarization must be given AUDIO, never the uploaded video. pyannote reads
  through libsndfile, which opens wav/flac/ogg but not an mp4/mkv container, so
  `diarize(input_path, ...)` on the raw upload raised "Format not recognised",
  `diarize` swallowed it (it must never fail a render) and returned [], and the
  job silently fell back to ONE voice — a male/female clip came back entirely in
  the first speaker's voice, and the frontend showed a single reference lane and
  no timeline because there was only one speaker. `app.py` now passes the 16 kHz
  mono `stt_audio` it already extracted for Whisper, and
  `speaker_diarization._readable_audio` decodes any unreadable path to a
  temporary wav as a last-resort repair (a failed decode passes the original
  path through, so test/mock backends that never read the file are unaffected).
  Pinned by `tests/test_diarization_input.py`.
- The diarizer's speaker CEILING must default to unset, not 10. pyannote
  treats `max_speakers=N` as "merge anything beyond N into the N it keeps", so a
  wrong ceiling is a silently wrong number of voices; leaving it unset lets the
  pipeline find as many speakers as it hears. `DIAR_MAX_SPEAKERS` is now `None`
  by default and `PyannoteDiarizer.diarize` omits any bound that was not set
  rather than sending `max_speakers=None` through. Set the tunable only to stop
  background chatter becoming characters.
- Every stage's result must be written to the runtime logs, one JSON record per
  stage, so a wrong dub is diagnosable after the fact instead of only audible.
  `engine/stage_log.py` writes `<runtime logs>/stages/<job_id>.jsonl` (plus a
  human-readable `.log`) and is wired into `app.py` for run_start, probe,
  extract_audio, transcribe, diarize, separate_audio, select_reference,
  translate, synthesize, verify, source_vs_final, failed and cancelled. `verify`
  plus `source_vs_final` are what answer "is this still my film?": they record
  the sync check, the voice match, and the delivered vs original duration,
  loudness and background. Read them in the notebook's "7b. Per-step results"
  cell or at `GET /jobs/{job_id}/stages`. Never let a log write fail a render.
- A dropped clause is invisible to every length and sync check: the assembled
  track is still exactly the video's length, so `verify` passes and the hole is
  only audible. `speech_activity.uncovered_speech(runs, plan)` measures the
  source's speech runs against the dubbing plan and names every stretch the plan
  does not cover (holes under 0.5s are breath, not a dropped clause).
  `app._dialogue_coverage` reads the audio and calls it; `_stage_source_vs_final`
  writes the result into the `source_vs_final` log record, the job stores it as
  `dialogue_coverage`, `GET /jobs/{id}` returns it, and `run_summary.warnings`
  turns it into a sentence with the timestamps. Pinned by
  `tests/test_dialogue_coverage.py`. Translation is the stage that drops the
  clause (NLLB, given two sentences, translates the first and silently discards
  the rest), so `translation_v2._split_into_chunks` splits EVERY segment on
  sentence boundaries, not only ones above the old 600-char cap, and retries any
  chunk that comes back empty. The `translate` log records
  `dropped_translations`/`dropped_segments` so the drop is visible even when the
  retry cannot recover it.
- The `run_start` log must record the RESOLVED `preserve_background`, not just
  the request. The studio sends its own value, but the API/headless path sends
  `null`, and a log that shows only `null` reads as "off" when the default
  (on) is what actually ran. `preserve_background_resolved` is
  `_preserve_background(preserve_background)` and is what to read.
- The mix must be LIMITED, not just each layer. `av_sync.mix_layers` and
  `mix_music` sum layers that are each loudnorm'd to their own target with a
  true-peak ceiling, then hand the sum to the encoder with `amix=normalize=0`,
  which does no peak management. Where a loud line lands on a loud downbeat the
  sum overshoots: the five hosted clips measured -0.2, -0.9, 0.0, +0.1 and 0.0
  dBFS true peak, three of five clipping. A single `alimiter`
  (`TRUE_PEAK_LIMIT` = 0.891, -1 dBFS) at the end of each chain fixes it, and
  `latency=1` compensates the lookahead so the dub does not drift out of sync.
  Pinned by `MixDoesNotClipTests`; the first test proves the fixture clips
  unlimited, so the others cannot pass for the wrong reason. Measure with the
  largest absolute sample, not `volumedetect` (rounds to 0.1 dB) or ebur128's
  true-peak meter (over-reads on synthetic broadband noise — it put +5 dBFS on
  a file whose samples never leave +/-1.0).
- The dialogue-coverage gate must NOT be fooled by a music bed. The energy
  detector cannot tell a sustained score from a voice, so reading the mixed
  original reported phantom "dropped dialogue": the two-male clip showed a 10s
  hole over its instrumental outro and the Aavesham trailer, scored throughout,
  one 25s hole across the whole run. `_dialogue_coverage` now reads the
  SEPARATED VOCALS where separation ran, and passes the transcript's word
  timings (`segments`) through so the gate must ALSO see a transcript word in
  the interval — the one thing a music bed never has. `_spoken_intervals` pads
  Whisper's interpolated word timings (`WORD_PAD`) and merges the ordinary
  inter-word gaps (`WORD_MERGE_GAP`); a transcript with less than
  `MIN_WORD_SECONDS` of words is too sparse to trust as a mask and falls back
  to the energy gate alone.

## Installing the optional extras on Colab/Kaggle

Both extras (Demucs, S2 Pro) fight the pinned stack, and pip's failure modes
make that fight silent. What actually bites:

- pip is all-or-nothing per invocation. One package that cannot build aborts
  the whole `pip install a b c ...` and drops every other package in the batch,
  with no error the caller sees unless it reads the log. `pyaudio` (no PortAudio
  headers on these runtimes) did exactly this and left `audiotools` uninstalled,
  so `import fish_speech` died with `No module named 'audiotools'` while the
  install cell printed "ok". The notebooks now install the group through
  `pip_install`, which retries each package alone so the survivors land, and
  `pyaudio` is dropped outright - nothing on the TTS path imports it.
- `fish_speech/models/dac/inference.py` runs
  `pyrootutils.setup_root(__file__, indicator=".project-root")` at import time.
  No `.project-root` exists above the installed package, so the import raises
  `FileNotFoundError: Project root directory not found` even though the package
  is installed. The install cell writes an empty marker into the installed
  `fish_speech/` directory. The repo root has one for the same reason.
- Install the extras' dependency groups WITH resolution under
  `deploy/constraints.txt`, not with `--no-deps`. `--no-deps` looks safer but it
  means a package like `descript-audio-codec` cannot pull `descript-audiotools`
  (the `audiotools` import), and the import fails later. The constraints file
  holds torch/NumPy/transformers at their pins so resolution cannot move them.
- `fish-speech` metadata pins torch==2.8.0 / transformers==5.19.0. Install the
  package itself with `--no-deps` (it is the one that would drag torch 2.8 over
  the CUDA-12 torch 2.6 build and break CTranslate2); install its imports
  separately, as above.
- A readiness probe must import the exact modules `S2ProEngine.load()` uses, not
  just `import fish_speech`: the package root imports cleanly while the heavy
  modules (and the pyrootutils call) sit deeper. The notebooks' import check
  names them.
- `scripts/download_models.py` prefetches the Demucs `htdemucs` checkpoint
  (mirror list, verified by SHA-256, installed under the name torch.hub looks
  for). `--no-background` skips it. Without the prefetch the first separation
  hits the Meta host whose TLS certificate expired, and a failed fetch disables
  background preservation silently - the render just loses its music.
- `web/` is committed and CI fails if it drifts from `apps/studio/src`. After
  any change under `apps/studio/src`, run `cd apps/studio && npm run build` and
  commit `web/` in the same change; commit 48f2486 changed the source and left
  the committed bundle stale.

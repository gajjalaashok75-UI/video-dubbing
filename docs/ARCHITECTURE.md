# Backend architecture

Everything runs locally. There is no inference API call anywhere in this
pipeline and no cloud storage behind it. The one optional outbound call is the
script model, when a hosted provider is chosen over Ollama, and it carries a
single line of already-translated text.

## Entry points

| File | What it is |
|---|---|
| `app.py` | The FastAPI service. Owns the job store, the pipeline and every route. |
| `gradio_app.py` | A harness for trying synthesis without the studio. Imported by `eval_harness`. |
| `eval_harness.py` | Measures speaker similarity and intelligibility across a corpus. |

## Models

| Purpose | Model | Where |
|---|---|---|
| Speech recognition | `Systran/faster-whisper-large-v3` | `speech_to_text_v2.py` |
| Translation | `facebook/nllb-200-distilled-600M` | `translation_v2.py` |
| Voice cloning | `ai4bharat/IndicF5` + `charactr/vocos-mel-24khz` | `tts_engines.py` |
| Voice cloning (non-Indic) | XTTS-v2, non-commercial licence | `tts_engines.py` |
| Lip sync | Wav2Lip + s3fd | `lip_sync.py` |
| Speaker similarity | `microsoft/wavlm-base-plus-sv` | `eval_harness.py`, and the voice-match score in `app.py` |

All are preloaded at boot by `_warmup()` and reported through `/api/boot`, which
is what the splash screen displays.

## The dub pipeline

`process_video_task` in `app.py`, one stage per `_set(job_id, step=...)`:

1. **validating** — probe the file, apply in/out points (`av_sync.trim`)
2. **extracting_audio** — 16 kHz mono for ASR (`av_sync.extract_audio`)
3. **transcribing** — segments and word timings (`speech_to_text_v2`)
4. **diarizing** — who spoke when, on the original audio (`speaker_diarization`).
   Optional: skipped when pyannote is absent, and non-fatal on failure. Without
   it every segment is one implicit `SPEAKER_00`.
5. **separating_audio** — Demucs splits the original soundtrack into speech and
   everything else (`source_separation`). Optional: skipped unless asked for,
   and non-fatal on failure. The `no_vocals` stem becomes the bed; the `vocals`
   stem is what ASR and the voice references are read from.
6. **selecting_reference** — score the cleanest continuous window per speaker
   (`reference_audio`), cut from the vocals stem when there is one. One speaker
   gets one window, exactly as before.
7. **translating** — per segment, then repair code-switching (`translation_v2`)
8. **synthesizing_voice** — phrase-level, laid onto a fixed-length track (`dubbing`).
   Each phrase is spoken in its own speaker's reference voice. Scheduling is
   per speaker, so a different actor may start while another is still talking;
   the same actor still cannot overlap themselves. Assembly is additive, with
   the level managed locally where voices coincide (`1/sqrt(n)`, plus a local
   ceiling inside the overlap only).
9. **mixing_background** — dialogue over the preserved background, at exactly
   the video's length (`av_sync.mix_layers`). Layer order is dialogue >
   preserved background > user music bed.
10. **lip_syncing** — mouth region only, single encode (`lip_sync`)
11. **verifying** — A/V drift and voice match
12. **filing** — the finished render is copied to the project's output folder

The dialogue-only track is kept as its own artifact even when a bed is mixed in:
it is what the voice-match score compares against, what an A/B against the
reference needs, and what a remix can reuse without running IndicF5 again.

Two stages report a real counter (`synthesizing_voice (3/8)`,
`lip_syncing (240/512 frames)`), which is what drives the progress bar and
bounds how quickly a cancel lands.

## The voice-over pipeline

`voiceover_task` shares stages 1-4, then speaks a typed script (`voiceover.py`).
It skips translation, timeline fitting and lip sync: the user supplied the
words and there is no picture to stay in sync with, so every phrase is spoken
at its natural length with no time-stretching.

## Revising a finished dub

`dubbing.synth_unit` conditions one phrase; `dubbing.assemble` lays cached
phrases onto a track. Each phrase's audio is cached in `<workdir>/units/`, so
editing one line re-synthesizes one phrase rather than the whole run.

- `POST /jobs/{id}/segments/{i}` — new text and/or a new sampling seed
- `POST /jobs/{id}/reference` — clone from a different window (re-speaks everything)
- `POST /jobs/{id}/rerender` — re-run lip sync against the current audio

Audio rebuilds in seconds; the picture is only redone on request, because that
is the part that costs minutes.

## Projects

`project.py` writes a `project.json` beside the files a job already keeps, and
the server scans for manifests at startup. Without it, restarting turned every
finished dub back into a download link. `/videos/` builds one row per project,
however many outputs it holds, from the manifests **and** the live job
registry: a run that has not finished has no manifest yet, and listing only
finished work made a project disappear the moment you started rendering it.

## Readiness, the run summary, and the artifact graph

Three modules carry everything the studio, the API and the check scripts report.
None of them touches audio.

`readiness.py` answers "can this machine do it", as a structure rather than
prose. Every probe is an import spec or a file check, so it is cheap enough for a
request handler. A feature is `ready`, `fallback_available`, `optional_missing`
or `blocked`, and only a **core** feature can make the whole verdict not-ready —
a missing Demucs is a reduced dub, not a broken application. It reports
`HF_TOKEN` as present or missing and never returns any part of it.

`run_summary.py` is what a finished job actually did: stages with the state each
reached, per-speaker status, overlap regions, which models ran, and a `warnings`
list in plain sentences. Every stage of this pipeline degrades rather than
fails, and that is exactly how a dub ends up quietly worse than the user expects
with nothing on screen to say why. The summary is where a fallback becomes
visible. It is written into the project manifest and returned by the API.

`artifact_cache.py` is the dependency graph, in one place. Each cache grew up in
its own phase with its own reuse rule; this says what depends on what, so a
change invalidates that artifact and its dependents and nothing else. Validity is
always a comparison against the inputs — never `os.path.exists`, because a stems
file left over from an untrimmed clip is present and wrong.

## Cancellation

Cooperative. `_set()` raises `JobCancelled` whenever a step is reported while
the cancel flag is set, which makes every stage boundary a checkpoint for free.
A cancel lands within roughly one phrase, because a TTS forward pass is atomic
and cannot be interrupted mid-call.

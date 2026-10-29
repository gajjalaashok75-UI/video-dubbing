# Integration report — multi-speaker dubbing + background preservation

Status: **exploration only. No production code has been modified.**

Scope: port multi-speaker dubbing and original-background preservation into the
existing Multiva / VideoTrans pipeline, keeping the core model chain unchanged:

```
faster-whisper  →  NLLB-200  →  IndicF5  →  Wav2Lip
```

IndicF5 stays the voice model. XTTS is not proposed as a replacement; it remains
what it already is (the optional foreign-language engine). F5-TTS is not
proposed either, even though three of the four reference repos use it — the
reference code is used for *technique*, not for its TTS choice.

Reference repos inspected (vendored under `references/`, read-only):

| Repo | Licence | What is worth taking |
|---|---|---|
| `Daniel-McLarty/Python-Autodub` | MPL-2.0 | Chunked Demucs, DiariZen, overlap-margin speaker assignment, per-speaker clone audit + human-in-the-loop, LUFS bed/dialog mix, input-hash resume |
| `sun-kic/video-dubbing-system` | none stated | pyannote service shape, speaker/voice-profile data model, `_align_segments`, per-speaker ref extraction from the *vocals* stem, pydub bed mix |
| `DjebrilSVN/Ai-Vedio-Dubbing` | MIT (README badge) | Word-midpoint speaker merge with carry-forward, per-speaker ref capped at 6 s, gap-based duration fit, `afftdn` clean-up, bed ducking |
| `himanishpuri/PolyDub` | none stated (CPML note) | Clean optional-Demucs step, elastic timeline with a speed cap, `tpad`/`apad` merge with no data loss, per-step `skip_existing`, run summary |

---

## 1. Current architecture and exact files to modify

### 1.1 What runs today

`engine/app.py` owns everything. The dub pipeline is `process_video_task`
(`engine/app.py:349`), one `_set(job_id, step=...)` per stage:

| # | step | code | notes |
|---|---|---|---|
| 1 | `validating` | `av_sync.probe`, `av_sync.trim` | in/out points applied first |
| 2 | `extracting_audio` | `av_sync.extract_audio` → 16 kHz mono | for ASR only |
| 3 | `transcribing` | `speech_to_text_v2.transcribe_audio` | segments + word timings; may be reused from a sibling |
| 4 | `selecting_reference` | `reference_audio.build_reference` | **one** 6–12 s window for the whole video |
| 5 | `translating` | `translation_v2.translate_segments` + `fix_code_switching` | per segment |
| 6 | `synthesizing_voice` | `dubbing.build_dubbed_track` | phrase units → `plan_timeline` → `synth_unit` per phrase → `assemble` |
| 7 | `lip_syncing` | `lip_sync.generate_lip_synced_video` | frame count authoritative; audio muxed in the same encode |
| 8 | `verifying` | `av_sync.verify_sync`, `_voice_match` | A/V drift + WavLM similarity |
| 9 | `uploading_result` | `_file_render` | files into the project output folder |

The two properties the whole design rests on, and which the port must not break:

* **The dubbed track is exactly `video_duration` seconds by construction**
  (`dubbing.assemble` allocates `int(round(video_duration * SAMPLE_RATE))` and
  raises if the result is silent). Wav2Lip therefore can never loop or truncate.
* **Per-phrase units are cached** in `<workdir>/units/unit_<index>.wav` and
  re-laid by `_rebuild_track` (`engine/app.py:1222`), so editing one line
  re-synthesizes one phrase.

### 1.2 The two limitations, located precisely

**Limitation A — one voice per video.**

* `reference_audio.build_reference` (`engine/reference_audio.py:258`) returns a
  single `{"path","text","start","end","duration"}` dict. `find_best_window`
  (`:86`) and `candidate_windows` (`:118`) score windows over *all* words from
  *all* speakers.
* `dubbing.build_dubbed_track` (`engine/dubbing.py:558`) takes that one
  `reference` and passes `reference["path"]/["text"]/["duration"]` to
  `synthesize_timeline` (`:489`), which passes them to `synth_unit` (`:366`)
  for every phrase.
* `seg_rows` built at `engine/app.py:452` carry only `{start,end,text}` — there
  is no speaker field anywhere in the job, the plan, the unit cache, or the
  manifest written by `project.save`.
* `README.md` states the limitation outright ("One voice per video. Whisper does
  not diarize… the largest capability gap") and `docs/TESTING.md:258` repeats it.

**Limitation B — no background preservation.**

* There is no source separation anywhere in the repo (verified: no `demucs`,
  no `htdemucs`, no `no_vocals` in any non-vendor source file).
* The rendered video's only audio stream is the synthesized dialogue:
  `generate_lip_synced_video(input_path, audio_for_video, ...)` at
  `engine/app.py:580` muxes the dub track (or the dub + a user-supplied music
  bed) as the sole audio. The original music/SFX is discarded.
* The only mixing that exists is `av_sync.mix_music` (`engine/av_sync.py:181`)
  for an **optional user-uploaded** bed (`music_path` / `music_gain`, default
  −18 dB). That is a replacement bed, not the original soundtrack.

### 1.3 Exact files that need modification

| File | Change | Why |
|---|---|---|
| `engine/speaker_diarization.py` | **new** | backend registry (pyannote default, DiariZen optional), `diarize()` → `[{start,end,speaker}]` |
| `engine/source_separation.py` | **new** | chunked Demucs `--two-stems vocals`; returns `(vocals, background)` |
| `engine/app.py` | edit | new stages `separating_audio` + `diarizing`; per-speaker reference build; pass speaker→reference map into `dubbing`; mix background before lip sync; store `speakers`/`diarization`/`stems` on the job; new endpoints; per-speaker `reclone_task`; `_reusable_sibling` reuses diarization; `voiceover_task` untouched or single-speaker |
| `engine/reference_audio.py` | edit | `build_references()` (plural) — per-speaker windows; `extract_reference` gains a `source_audio` override so refs are cut from the **vocals** stem |
| `engine/dubbing.py` | edit | `DurationModel` per speaker; `prepare_units` carries `speaker`; `plan_timeline` overlap-aware; `synth_unit`/`synthesize_timeline`/`build_dubbed_track` take a **per-unit reference** |
| `engine/speech_activity.py` | edit | phrase units inherit the segment's speaker |
| `engine/av_sync.py` | edit | `mix_background()` (LUFS balance + optional ducking); keep `mix_music` |
| `engine/engines.py` | edit | tunables: diarization backend, Demucs on/off, bed gain, ducking depth |
| `engine/models.py` | edit | catalogue entries for the diarization weights (gated → reuses the existing `HF_TOKEN` path) and the Demucs checkpoint |
| `engine/project.py` | edit | persist `speakers` + `diarization` + stem paths in `project.json` |
| `apps/studio/src/lib/api.ts`, `apps/studio/src/pages/Studio.tsx`, `apps/studio/src/lib/pipeline.ts` | edit | speaker list + per-speaker reference preview, "preserve background" toggle, new stage labels |
| `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `README.md` | edit | document the two new stages and remove the "one voice" / "no bed" limits |

Unchanged on purpose: `tts_engines.py` (IndicF5 wrapper), `lip_sync.py`,
`translation_v2.py`, `speech_to_text_v2.py` (ASR itself), `languages.py`.

---

## 2. Relevant files and functions per reference repo

### 2.1 Python-Autodub (`references/Python-Autodub`)

| Location | Function / block | Port candidate |
|---|---|---|
| `src/pipeline.py:123-190` | Chunked Demucs | **Yes.** `ffmpeg -f segment -segment_time 600 -c copy` → one `demucs.separate -n htdemucs -d <dev> --two-stems vocals -j 1` over all chunks → lossless `concat` demuxer reassembly. This is the RAM answer for full-length movies. |
| `src/pipeline.py:210-263` | DiariZen + WhisperX alignment | Technique only. `DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")`; `whisperx.assign_word_speakers(df, aligned)`. |
| `src/pipeline.py:266-300` | Per-speaker clone extraction | **Yes.** Group turns by speaker; window `min_frames=2s`, `max_frames=7s` (hybrid) or `12s`; prefer the longest valid chunk, else the shortest above the floor. |
| `src/pipeline.py:316-360` | Clone audit + `CloneReviewUI` | **Yes, later phase.** Per-speaker playback synced to the video, with manual wav/txt override. Maps onto the studio's existing reference picker. |
| `src/pipeline.py:392-441` | Assembly | Partial. One `float32` master array; `librosa.effects.time_stretch` capped at 2.0 then hard cut. The repo's `plan_timeline` is better; take only the "cap then spill/cut" idea as a guard. |
| `src/pipeline.py:448-460` | LUFS mix | **Yes.** `[0:a]loudnorm=I=-26:TP=-2.0:LRA=11[bg]; [1:a]loudnorm=I=-12:TP=-1.0:LRA=11[dialog]; [bg][dialog]amix=inputs=2:duration=longest`. |
| `src/pipeline.py:47-84` | `manage_temp_state` | **Yes.** md5 of video + both SRTs → `run_hashes.json`; purge temp on mismatch. Adapt to `project.json`. |
| `src/worker.py:63-86` | Overlap-margin speaker pick | **Yes.** Sum overlap per speaker in the window; require `top_ratio − runner_up_ratio ≥ conf_thresh`, else unassigned. |
| `src/worker.py:118-165` | Hybrid cloning | Optional. Concat base clone + scene audio, `ref_text = base_text + scene_text`. |
| `src/worker.py:166-208` | Validation retry loop | Optional. Language-ID killswitch, ASR pre-speech trim, `difflib` similarity ≥ 0.45. |
| `src/worker.py:98-116` | Generic voice fallback | **Yes.** When no usable clone, fall back to a generic voice rather than a wrong clone. |
| `src/utils.py:18-24, 26-48` | `get_file_hash`, `estimate_gender_from_pitch` | `get_file_hash` yes; pitch helper only if a generic-voice fallback is built. |

### 2.2 video-dubbing-system (`references/video-dubbing-system`)

| Location | Function | Port candidate |
|---|---|---|
| `backend/services/speaker_diarization.py:10-66` | `SpeakerDiarizationService.diarize` | **Yes — the shape.** Class-level singleton + lock; `Pipeline.from_pretrained(model, token=...)`; handles pyannote 3.x `Annotation` **and** 4.x `DiarizeOutput.speaker_diarization`; `min_speakers=1, max_speakers=10`. |
| `backend/workers/tasks.py:27-41` | `_find_best_speaker` | **Yes.** Max time-overlap assignment. |
| `backend/workers/tasks.py:45-58` | `_align_segments` | **Yes.** Produces `DubbingSegment{start,end,speaker,text}`. |
| `backend/workers/tasks.py:61-92` | `_auto_extract_speaker_refs` | **Yes.** Per speaker: longest segment, target 10 s, floor 3 s, cut from the **vocals** track. |
| `backend/workers/tasks.py:154-175` | ASR on vocals, diarization on full audio | **Yes.** The comment ("Diarization on full audio for better results") matches Autodub's intent and is the right default. |
| `backend/services/video_processor.py:120-152` | `separate_vocals` | **Yes.** `demucs --two-stems vocals -o out`; outputs `htdemucs/<stem>/{vocals,no_vocals}.wav`. |
| `backend/services/video_processor.py:154-208` | `mix_dubbed_with_background` | Partial. Silent canvas + `overlay` per TTS at `start_ms`; `background + bg_volume_db` (default −8 dB) then overlay. Simple, but a static gain; prefer the LUFS version. |
| `backend/models/speaker.py:10-62` | `Speaker`, `VoiceProfile` | **Yes — data model.** `label`, `display_name`, `color`, `voice_profile_id`, `total_speech_time`; `reference_audio_path`, `embedding_path`, `embedding_vector`, `tts_engine`, `is_cloned`, `clone_quality_score`. |
| `backend/models/schemas.py:26-45` | `TranscriptSegment`, `DiarizationSegment`, `DubbingSegment` | **Yes — the segment schema.** |

### 2.3 Ai-Vedio-Dubbing (`references/Ai-Vedio-Dubbing`)

| Location | Function | Port candidate |
|---|---|---|
| `app.py:154-193` | Demucs + pyannote + per-speaker refs | **Yes, with care.** `pipeline.instantiate({"clustering": {"threshold": 0.60}})`; drop turns < 0.2 s; ref = longest turn capped at 6000 ms. |
| `app.py:207-238` | Word-midpoint speaker merge | **Yes.** For each word, find the turn containing its midpoint within ±0.2 s; carry `last_known_speaker` forward for `UNKNOWN`. This is the most robust of the three assignment schemes for word-level data the repo already has. |
| `app.py:413-447` | Per-speaker ref choice | Yes — pick the segment whose duration is closest to 6 s. Simpler than Autodub's; useful as a tie-break. |
| `app.py:449-495` | Gap-based duration fit | Partial. `max_safe_duration_ms = next_start − start`; `atempo` capped at `max_atempo`; fade in/out 30 ms. The repo already does this better with `fix_duration`. |
| `app.py:501-518` | Bed re-separation + final mix | Yes. Separate the **original stereo** audio → `no_vocals` → `[1:a][2:a]amix=inputs=2:duration=first`. |
| `NoteBooks…/p3.ipynb` | "Ducking background volume (−6 dB)"; `afftdn` de-hiss | **Yes, as options.** A static −6 dB duck and an optional `afftdn` clean-up on generated speech. |

### 2.4 PolyDub (`references/PolyDub`)

| Location | Function | Port candidate |
|---|---|---|
| `src/polydub/steps/separate.py:31-103` | `_extract_hq_stereo` + Demucs | **Yes.** HQ 44.1 kHz **stereo** extraction for Demucs (not the 16 kHz mono ASR track), subprocess, take `no_vocals.wav`. Optional-dependency guard (`find_spec("demucs") is None` → warn, continue). |
| `src/polydub/steps/align.py:46-64` | `_atempo_filter` | Yes. Chains `atempo` for ratios outside [0.5, 2.0] — a cleaner equivalent of the repo's `_time_stretch` clamp. |
| `src/polydub/steps/align.py:101-151` | `_place_segments` | Partial. `place_start = max(orig_start, cursor)`; natural pace when it fits the room to the next clip; else compress up to `max_speed_adjust` then spill. **Adopt the "never start before the cursor" rule per speaker; do not adopt spilling past the video length** (the repo's length invariant is stronger). |
| `src/polydub/steps/merge.py:42-158` | `_build_audio_filter` + merge | Yes for the bed path: `[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]`. The `tpad`/`apad` "no data loss" logic is not needed once the track is exactly video-length. |
| `src/polydub/pipeline.py:110-160` | `select_steps` | **Yes.** Demucs runs only when requested — keeps the default run fast, which matters because Demucs is minutes per video on CPU. |
| `src/polydub/pipeline.py:250-320` | `run_pipeline` + `write_summary` | **Yes.** Per-step `skip_existing` by output existence, and a `run.json` summary the UI can read. |
| `src/polydub/steps/asr.py:74-96` | ASR cache by `sha256(audio)+model` | Yes — mirrors the repo's existing sibling reuse. |

---

## 3. Proposed architecture

Two new stages inserted into `process_video_task`, both optional and both
degrading to today's behaviour when unavailable:

```
validating
extracting_audio
  ├── separating_audio   (NEW, optional)  Demucs two-stem on HQ stereo
  │        ├─ stems/vocals.wav      → ASR + per-speaker reference cutting
  │        └─ stems/background.wav  → kept for the final mix
transcribing              (on vocals when available, else the 16 kHz mono)
  ├── diarizing          (NEW, optional)  pyannote/DiariZen → speaker turns
selecting_reference       (NOW: one reference per speaker)
translating
synthesizing_voice        (per unit → its speaker's reference)
mixing_background         (NEW, optional)  dub + background → audio_for_video
lip_syncing
verifying                 (A/V drift + per-speaker voice match)
uploading_result
```

Design rules carried over unchanged:

1. `dubbing.assemble` still returns exactly `video_duration` seconds.
2. Per-phrase caching stays; only the cache key gains a speaker dimension.
3. The heavy stages stay serialized behind the existing `_HEAVY` semaphore, and
   the diarization model is freed before TTS loads (Autodub's `clear_vram`).
4. Every new stage is best-effort: a failure logs and falls back, it does not
   fail a render that would otherwise succeed.

---

## 4. Proposed data model for speaker-aware segments

Extend the existing plain dicts (the repo deliberately avoids an ORM; the
reference `video-dubbing-system` uses SQLAlchemy, which is *not* ported).

```python
# engine/speaker_diarization.py
SpeakerTurn = dict          # {"start": float, "end": float, "speaker": str}

# segment row stored on the job (extends engine/app.py:452 seg_rows)
Segment = dict              # {"start", "end", "text", "speaker"}
                            #  speaker: "SPEAKER_00"… or None (unresolved)

# one entry per speaker, stored on the job as job["speakers"]
Speaker = dict              # {
                            #   "label": "SPEAKER_00",
                            #   "total_speech": 41.7,      # seconds
                            #   "segment_count": 23,
                            #   "reference": {             # like today's dict
                            #       "path", "text", "start", "end", "duration"
                            #   },
                            #   "reference_score": 3.42,   # from _score_window
                            #   "fallback": False          # generic voice used
                            # }

# a planned unit (extends dubbing.plan_timeline output)
Unit = dict                 # {"index", "text", "start", "duration", "natural",
                            #  "rate", "speaker", "ref_index"}   # ref_index → speakers
```

Backward compatibility:

* Jobs with no `speaker` on any segment behave exactly as today: one implicit
  `SPEAKER_00` whose reference is the current `build_reference` result.
* `job["speakers"]` is absent on old manifests; `_require_editable` and the
  studio both already tolerate missing optional fields.
* `reference` (singular) stays populated for the single-speaker case so the
  existing studio reference picker, `_voice_match`, and `/jobs/{id}/reference`
  keep working.

---

## 5. Recommended diarization backend

**Default: `pyannote/speaker-diarization-3.1` via `pyannote.audio>=3.3.2`.
Optional: DiariZen (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`) behind the same
interface.**

Justification:

* **Dependency weight.** pyannote.audio is one package. DiariZen requires the
  DiariZen git repo **plus** a *fork of pyannote-audio from the same git repo*,
  plus `lightning`, `speechbrain`, `torchmetrics`, `torch-audiomentations`,
  `torch-pitch-shift`, `diffq`, `torchcodec` and `python-mpv`, pinned to
  `torch==2.8.0`. The repo pins `torch>=2.2` and `av<19` and already patches
  `torchaudio.load` to avoid torchcodec; DiariZen's tree directly contradicts
  that.
* **Account story.** pyannote 3.1 is gated, but the repo **already** has the
  entire gated-model apparatus — `HF_TOKEN` in `.env`, `models.is_cached()`
  file-level checks, `docs/GATED-MODELS.md`, and a UI that marks a gated model
  unavailable up front. DiariZen's "no account" advantage is real but buys
  nothing that is not already built.
* **Output shape.** Both return turns via `itertracks(yield_label=True)`, and
  the repo needs exactly `[{start, end, speaker}]`. pyannote's 3.x/4.x output
  difference is already handled by the reference implementation
  (`speaker_diarization.py:48-53`) and is trivial to reproduce.
* **Why still offer DiariZen.** For an offline deployment with no HF account,
  and because the user asked for the comparison. It is the better *quality*
  option on overlapping and far-field audio, and Python-Autodub proves it
  works end to end. Behind a `Diarizer` protocol the choice is one setting.

Interface (mirrors the reference singleton, no ORM):

```python
# engine/speaker_diarization.py
class Diarizer:                       # protocol
    def diarize(self, audio_path: str,
                min_speakers: int = 1,
                max_speakers: int = 10) -> list[SpeakerTurn]: ...

class PyannoteDiarizer: ...           # default
class DiariZenDiarizer: ...           # optional, imported lazily

def get_diarizer(name: str | None = None) -> Diarizer: ...
def available() -> bool: ...          # mirrors tts_engines.engine_available
```

Diarization runs on the **full original audio**, not the vocals stem (both
`video-dubbing-system` and `Ai-Vedio-Dubbing` do this and say why). ASR runs on
the **vocals** stem when separation succeeded, which improves transcription.

---

## 6. Demucs integration design

Port `PolyDub`'s clean step + `Python-Autodub`'s chunking.

```
separate_stems(input_video, workdir, device="cuda"|"cpu", chunk_seconds=600)
    → (vocals_path, background_path) | None
```

1. **Extract HQ audio** — `ffmpeg -i <video> -ac 2 -ar 44100 -vn hq.wav`.
   Deliberately *not* the 16 kHz mono ASR track: Demucs needs the full band, and
   the repo's `reference_audio` docstring already argues that everything above
   8 kHz is sibilance and breathiness, which is most of what makes a voice
   identifiable.
2. **Chunk** — `ffmpeg -f segment -segment_time 600 -c copy chunk_%03d.wav`.
   One 2-hour movie held whole is what OOMs the machine; the repo already has a
   precedent for caring about this (`lip_sync`'s 10.4 GiB frame-list note in
   `AGENTS.md`).
3. **Separate** — one `python -m demucs.separate -n htdemucs -d <device>
   --two-stems vocals -j 1 -o <out> chunk_*.wav`. `-j 1` keeps peak RAM flat.
4. **Reassemble losslessly** — `ffmpeg -f concat -safe 0 -i list.txt -c copy`
   for `vocals` and `no_vocals` separately.
5. **Cache** — `<workdir>/stems/{vocals,background}.wav`. Never re-run for the
   same clip; the sibling/project mechanism already knows when the clip is the
   same.
6. **Optional dependency.** `find_spec("demucs") is None` → log, return `None`,
   and the pipeline behaves exactly as today (audio replace). Guarded by a
   tunable so a default run stays fast.

The vocals stem is also the **source for per-speaker reference extraction** —
all three diarizing reference repos cut references from vocals, and it is the
single biggest quality win for cloning.

---

## 7. Per-speaker reference-selection design

Reuse the repo's existing scoring (`_score_window`, `MAX_INTERNAL_GAP`,
`MIN_REF_SECONDS=6`, `MAX_REF_SECONDS=12`) but restrict it to one speaker's
words.

```
build_references(input_path, segments, speakers, total_duration, out_dir,
                 source_audio=None) -> dict[label, reference]
```

1. **Filter words by speaker.** `_flatten_words(segments)` already walks
   word-level timings; add a `speaker` predicate so a speaker's candidate
   windows are built only from words assigned to that speaker. This removes the
   current failure mode where the "cleanest window" is a *different* speaker.
2. **Cut from the vocals stem** (`source_audio` override on
   `reference_audio.extract_reference`) so the reference is not polluted by
   music. Keep the existing timing-preserving filter chain
   (`highpass=f=60, loudnorm=I=-18:TP=-2:LRA=11`) and the "never add
   `silenceremove`" warning.
3. **Duration policy.** Keep 6–12 s. Reconcile with the references: Autodub uses
   2 s floor / 7–12 s target; `video-dubbing-system` uses a 3 s floor / 10 s
   target; `Ai-Vedio-Dubbing` caps at 6 s. The repo's 6–12 s is the widest safe
   band and matches its own note that a 3.8 s reference produced the worst
   output in the corpus.
4. **Per-speaker fallback ladder** (Autodub's, adapted):
   * best window ≥ 6 s → use it;
   * shorter but ≥ 2 s → use it and warn (the repo already warns below
     `MIN_REF_SECONDS − 0.5`);
   * < 2 s or no words → reuse the highest-`total_speech` speaker's reference,
     mark `fallback=True`, and log;
   * no speaker at all → today's `build_reference`, labelled `SPEAKER_00`.
5. **Cache** — `<workdir>/refs/speaker_<label>.wav` + `.txt`. Keyed by label,
   and re-cut only when the label set or window changes.
6. **Studio parity** — `candidate_windows` gains a per-speaker variant so the
   existing reference picker (`/jobs/{id}/reference/candidates`) lists windows
   *for the selected speaker*, and `/jobs/{id}/reference` gains an optional
   `speaker` field. The human-in-the-loop idea from Autodub's `CloneReviewUI`
   maps onto this without a Tkinter/MPV dependency.

---

## 8. Background / SFX mixing strategy

Replace the "dub is the only audio" assumption with an explicit bed mix.

```
mix_background(dialogue_path, background_path, out_path,
               bed_lufs=-26.0, dialogue_lufs=-12.0,
               duck_db=0.0, sample_rate=24000) -> str
```

* **Source** — the Demucs `no_vocals` stem, full-quality stereo, from the same
  input the picture came from. When the user also uploads a music bed, that bed
  is layered *under* the preserved background (the two are independent; the
  existing `mix_music` path stays for the bed-only case).
* **Balance** — adopt Autodub's LUFS approach rather than a static dB cut:

  ```
  [0:a]loudnorm=I=-26:TP=-2.0:LRA=11[bg];
  [1:a]loudnorm=I=-12:TP=-1.0:LRA=11[dialog];
  [bg][dialog]amix=inputs=2:duration=longest:normalize=0[out]
  ```

  `normalize=0` is essential and is already the repo's stated reasoning in
  `mix_music` ("`amix` would duck both inputs by its own normalisation, so the
  sum is explicit instead"). LUFS balancing beats `video-dubbing-system`'s flat
  −8 dB and `Ai-Vedio-Dubbing`'s flat −6 dB, both of which are level-blind.
* **Optional ducking** — for BGM-heavy material, `sidechaincompress` keyed on
  the dialogue (true ducking), or `Ai-Vedio-Dubbing`'s static −6 dB as the
  cheap fallback. Exposed as a tunable; off by default so it cannot surprise a
  user who wants the bed untouched.
* **Length** — the dialogue track is exactly `video_duration`; the background
  is trimmed or `apad`-ed to the same length, so the mix needs no `-shortest`
  (matching `av_sync.mux`'s deliberate rejection of `-shortest`).
* **Ordering** — mixing happens *before* lip sync, so Wav2Lip muxes the final
  mix in its single encode. The voice-only `dub_path` is still kept separately
  so the studio can A/B the clone, exactly as today.
* **Deliverable for `kind="audio"`** — the mix is written into the delivered
  file, mirroring the existing `music_path` branch at `engine/app.py:534-561`.

---

## 9. Overlap handling

Overlapping dialogue is the case where naive ports break, so it needs an
explicit rule. Today `plan_timeline` serialises strictly (`start = max(seg_start,
cursor)`), which is correct for one voice and *wrong* for two people talking at
once — it would push the second speaker's line after the first and destroy sync.

Rules:

1. **Assignment is per word, not per segment.** Use `Ai-Vedio-Dubbing`'s
   midpoint test (±0.2 s) to label each word, then let a segment inherit the
   majority label. This handles a segment that straddles a speaker change.
2. **Unresolved words carry forward** (`last_known_speaker`), which is what
   stops `UNKNOWN` from becoming a phantom third speaker.
3. **Confidence gate for genuine overlap.** Adopt Autodub's margin rule: if the
   top and runner-up overlap ratios within a window are within `conf_thresh`,
   do not guess — either carry forward the previous speaker or, if the two
   speakers are both clearly present, treat the segment as a **duet** and
   synthesize both, summing them in `assemble` (which already does
   `track[at:end] += wave`).
4. **Serialisation becomes per speaker, not global.** The cursor advances per
   speaker track. Two different speakers may therefore occupy the same wall
   time; the same speaker may not. This preserves lip-sync for each and keeps
   the "no clip is cut off by the next one" property that `PolyDub`'s
   `_place_segments` argues for.
5. **Gain safety on overlap.** Summing two voices can clip. Normalise the sum
   per overlap region, or scale each speaker by `1/sqrt(n)` where `n` speakers
   are simultaneously active; `assemble`'s existing `peak > 1.0` divide is the
   last-resort guard, not the strategy.
6. **Single-speaker videos must be unaffected** — with one label the per-speaker
   cursor is identical to today's global cursor, so the plan is byte-identical
   to the current output.

---

## 10. Duration / alignment strategy

Keep the repo's approach; it is strictly better than all four references.

* **Keep** `fix_duration`-first synthesis (IndicF5 is asked for the right length
  up front), `DurationModel`'s self-calibrating `slot × (target_chars /
  source_chars) × (source_cps / target_cps)`, `BORROW_FRACTION`, and the single
  global squeeze at the end of `plan_timeline`. This is what makes the track
  exactly video-length by construction.
* **Add per-speaker tempo.** `DurationModel` currently derives `speaker_factor`
  from the one global reference. With per-speaker references, build one
  `DurationModel` per speaker (or one shared model plus a per-speaker factor),
  and pass the unit's own reference into `natural_fn`.
* **Do not adopt** PolyDub's spilling past the video length (it changes the
  picture length and breaks the length invariant), nor Autodub's hard cut of
  overrunning speech (it loses words). The repo's bounded `_time_stretch`
  (0.80×–1.25×) with a loud warning is the right compromise and already exists.
* **Borrow one idea:** `PolyDub`'s `_atempo_filter` chaining is a cleaner way to
  express out-of-range ratios than clamping, if the residual stretch ever needs
  to exceed the current band.

---

## 11. Caching / resume implications

| Artefact | Cache key today | Change |
|---|---|---|
| ASR transcript | reused from a sibling by project + clip length + source language (`_reusable_sibling`) | unchanged |
| Diarization | none | **new:** store on the job/manifest; reuse from a sibling exactly like the transcript. Diarization is deterministic for a fixed clip, so a sibling must not re-run it. |
| Demucs stems | none | **new:** `<workdir>/stems/`, plus a hash of the (possibly trimmed) input in the manifest so a re-trim invalidates them |
| Per-speaker references | one `<stem>_ref.wav` | **new:** `refs/speaker_<label>.wav`; invalidate when the label set changes |
| Phrase units | `<workdir>/units/unit_<index>.wav` | **keep the flat key** (index is already globally unique) but record `speaker` and `ref_index` on each plan unit, so `_rebuild_track` and `revise_segment` know which voice to use when re-synthesizing one phrase |
| Reference re-clone | `/jobs/{id}/reference` re-speaks everything | extend with `speaker`; only that speaker's phrases need re-speaking, which is a strict improvement |
| Manifest | `project.save` | add `speakers`, `diarization`, `stems` |

Adopt Autodub's `manage_temp_state` idea as a manifest field: hash the input
video, the trim points and the language, and purge only the dependent caches
when any changes. The repo already has the "same clip to the hundredth of a
second" strictness in `_reusable_sibling` to build on.

Add `PolyDub`'s per-step `skip_existing` semantics for the two new stages
(output existence), and write the extra stage outcomes into the existing
project manifest so the studio can show them.

---

## 12. Dependencies and VRAM / RAM requirements

New optional dependencies:

| Package | Purpose | Notes |
|---|---|---|
| `demucs>=4` | source separation | ~300 MB checkpoint; optional extra, mirroring PolyDub's `[separate]` |
| `pyannote.audio>=3.3.2` | default diarization | gated weights; reuses the existing `HF_TOKEN` path |
| DiariZen (git) + `lightning`, `speechbrain`, `torchmetrics`, `torch-audiomentations`, `torch-pitch-shift`, `diffq` | optional diarization | **conflicts with the repo's `torch`/`av` pins; keep strictly optional** |

Both new stages must be added to `requirements.txt`, `deploy/docker/requirements.docker.txt`
and the `models.MODELS` catalogue, and `tests/test_requirements_pins.py` should
be extended so the Demucs/pyannote pins cannot silently break the resolve job.

Resource envelope (measured expectations, to be confirmed on target hardware):

* **VRAM:** IndicF5 ≈ 2–3 GB, Wav2Lip + s3fd ≈ 1.5–2 GB, pyannote ≈ 1.5–2 GB,
  Demucs ≈ 2–3 GB. Sum exceeds a small GPU, so the existing `_HEAVY` semaphore
  must cover diarization and Demucs too, and the diarization model must be
  dropped before IndicF5 loads (Autodub's explicit `del` + `clear_vram`).
* **RAM:** Demucs on a full-length movie is the risk; the 10-minute chunking is
  what keeps peak RAM flat. The current sandbox has **15 GB RAM, 4 CPUs and no
  GPU**, so a real multi-speaker run here is slow — tests must stay model-free.
* **Runtime:** Diarization and Demucs are both minutes-scale on CPU, which is
  why both are opt-in (`select_steps` semantics from PolyDub) and why the
  studio needs a real stage label rather than a spinner.

---

## 13. Failure / fallback behaviour

Every failure degrades to exactly today's behaviour:

| Failure | Fallback |
|---|---|
| Demucs not installed / fails | no stems; ASR on the 16 kHz mono; straight audio replace (today's output) |
| Diarization backend not installed / fails | one implicit `SPEAKER_00`; today's single-reference pipeline |
| Diarization returns zero turns | same as above |
| Speaker's best window < 2 s | reuse the longest-speaking speaker's reference, `fallback=True`, warn |
| Speaker unresolved for a segment | carry the previous speaker forward; if none, the global reference |
| Reference cut produces no words in the window | raise per-speaker and fall back to that speaker's next-best window, then to the global reference |
| Bed mix fails | deliver the dialogue-only track and say so (never fail the render) |
| A phrase comes back silent | unchanged: keep the silence, log, and `assemble` still raises only if the *whole* track is silent |
| Lip sync | unchanged — it is downstream of the mix and already has its own fallbacks |

New hard errors are deliberately *not* introduced. The existing guards
(`assemble`'s silent-track raise, the "every segment failed" raise, the gated
model up-front refusal) all stay.

---

## 14. Implementation phases

Each phase is independently shippable and leaves the pipeline working.

**Phase 1 — plumbing, no behaviour change.**
Add `speaker: None` to segment rows and `speaker`/`ref_index` to plan units.
Add a `speakers` list to the job (empty ⇒ single implicit speaker) and to the
manifest. Extend `synth_unit`/`synthesize_timeline`/`build_dubbed_track` to
accept a *per-unit* reference while still defaulting to the single global one.
Deliverable: identical output, new schema in place.

**Phase 2 — diarization.**
`engine/speaker_diarization.py` (pyannote default, DiariZen optional), the
`diarizing` stage, word-level assignment, `SPEAKER_00` fallback. Verify against
a known two-speaker clip that assignment is correct before any synthesis change.

**Phase 3 — per-speaker references.**
`build_references()` + per-speaker candidate windows + per-speaker `DurationModel`
+ the fallback ladder. Deliverable: each speaker cloned from their own voice.

**Phase 4 — Demucs + background.**
`engine/source_separation.py`, the `separating_audio` stage, `av_sync.mix_background`,
the `mixing_background` stage, tunables, the "preserve background" toggle in the
studio. Deliverable: original music/SFX audible under the dub.

**Phase 5 — overlap.**
Per-speaker cursors, duet synthesis, overlap gain safety. Deliverable: two
speakers talking at once no longer collide or drift.

**Phase 6 — polish.**
Per-speaker voice-match reporting, per-speaker reference picker in the studio,
clone audit UI, per-step skip/resume, docs and README limit removal.

---

## 15. Tests required

Following the repo's conventions: model-free `unittest` in `tests/`, no GPU, no
network, synthetic fixtures. These pin behaviour that a metric-less run would
not catch.

**Fixtures.** A small generator that builds a WAV with N distinct tones/noise
bursts at known times, and a matching segment list — no real speech needed to
test assignment, planning, and assembly.

| Scenario | Test | Asserts |
|---|---|---|
| **Single speaker** | one speaker, one reference | output is **identical** to the current pipeline (regression lock); one `SPEAKER_00`; no stems requested |
| **Two speakers** | two alternating speakers | two references extracted; every plan unit's `ref_index` matches its segment's speaker; no unit uses the other speaker's reference |
| **Many speakers** | 5–8 speakers, one with < 2 s of speech | the short speaker falls back and is marked `fallback=True`; the others clone from their own windows; no phantom speaker is created |
| **Overlapping dialogue** | two speakers overlapping by ~1 s | both lines are placed at (or near) their original starts; the overlap region contains both voices; no global serialisation pushes the second speaker out of sync; the track is still exactly `video_duration` |
| **BGM-heavy** | loud music + sparse speech | `separating_audio` produces both stems; `mix_background` output contains the bed energy; LUFS balance leaves dialogue dominant; with Demucs disabled the output is the straight-replace path |
| **Fallbacks** | Demucs absent; diarizer absent; diarizer returns `[]` | each falls back to today's behaviour without raising, and the job still completes |
| **Determinism / resume** | run twice with the same input | diarization and stems are reused, not recomputed; a changed trim invalidates them |
| **Cache keying** | revise one phrase | only that phrase is re-synthesized; it uses its own speaker's reference |

Extend the existing suites rather than adding a parallel one:
`tests/test_requirements_pins.py` (new Demucs/pyannote pins),
and a new `tests/test_speaker_segments.py`, `tests/test_background_mix.py`,
`tests/test_overlap_plan.py` in the same style as `test_lip_sync_stream.py`
(fake model + synthetic input + byte comparison).

The end-to-end `scripts/selftest.py` gains the two new stage names and a
"preserve background" run; the full model-dependent walkthrough in
`docs/TESTING.md` gains the same.

---

## 16. Summary of recommendations

1. **Diarization:** pyannote 3.1 default, DiariZen optional, behind one protocol.
2. **Assignment:** word-midpoint with carry-forward, plus Autodub's margin gate
   for genuine overlap.
3. **References:** per-speaker windows scored by the repo's existing
   `_score_window`, cut from the Demucs vocals stem, cached per label, with a
   documented fallback ladder.
4. **Background:** chunked Demucs, LUFS-balanced mix with `normalize=0`, optional
   ducking, mixed before lip sync.
5. **Overlap:** per-speaker cursors and duet summing; single-speaker output stays
   byte-identical.
6. **Duration:** keep `fix_duration` + `DurationModel`, add per-speaker tempo;
   reject the references' spilling and hard-cut strategies.
7. **Everything optional:** both features degrade to today's pipeline, and the
   model chain (faster-whisper → NLLB-200 → IndicF5 → Wav2Lip) is untouched.

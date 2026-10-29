# Real-video validation matrix

Status: **not yet run against real models.** Every case below is `BLOCKED` on
this machine, for the reasons at the bottom of this file.

The matrix exists so that when the models are available, the work is a checklist
rather than an investigation. Each case names the clip, what it proves, and how
to read the result.

Nothing here has been marked PASS. A case becomes PASS only when the command was
actually run against real models and the named evidence was actually observed.

---

## How to run a case

```bash
# 1. Confirm the machine can do it at all. Cheap, no model loaded.
python scripts/final_check.py plan clip.mp4 --target-language hi

# 2. Run the audio pipeline. Add --render for the finished video.
python scripts/final_check.py check clip.mp4 --target-language hi \
    --preserve-background

# 3. Read the result. The summary names every speaker, its status, its
#    reference, the overlap regions and the warnings.
```

Artifacts land in `final_check/<clip>/`. Listen to them in this order:

| Artifact | What it tells you |
|---|---|
| `references/*.wav` | What each voice was cloned from |
| `dialogue.wav` | The dub on its own, no bed |
| `mixed.wav` | The dub over the preserved soundtrack |
| `dubbed.mp4` | The finished picture |

For one stage at a time, `multispeaker_check.py`, `background_check.py` and
`overlap_check.py` each isolate their own concern.

---

## The cases

### Case A — single speaker

**Proves:** nothing regressed. This is the path every earlier version took, and
it must produce the same timing and the same voice.

**Clip:** one person talking to camera, 20–60 s, some music underneath.

**Evidence to look for:**

- the summary lists exactly one speaker, `SPEAKER_00`, status `cloned`
- no overlap regions reported
- the dialogue track is exactly the video's length
- `references/SPEAKER_00.wav` sounds like the speaker

**Status:** `BLOCKED`

---

### Case B — two alternating speakers

**Proves:** reference separation. Each actor is cloned from their own voice, not
from whichever voice happened to be cleanest overall.

**Clip:** two people in conversation, each with at least one continuous 6–12 s
turn.

**Evidence to look for:**

- two speakers, both `cloned`
- two *different* files under `references/`
- the similarity matrix from `multispeaker_check.py` has a high diagonal
- `dialogue/SPEAKER_00.wav` and `dialogue/SPEAKER_01.wav` sound like different
  people, and each sounds like its own reference

**Status:** `BLOCKED`

---

### Case C — three or more speakers

**Proves:** speaker consistency. A voice does not drift between its lines.

**Clip:** a scene with three or more distinct voices, several minutes.

**Evidence to look for:**

- every speaker has its own reference and its own status
- each speaker's isolated track is consistent end to end
- no speaker is marked `generic_fallback` unless it genuinely lacks clean audio

**Status:** `BLOCKED`

---

### Case D — overlapping dialogue

**Proves:** the Phase 5 scheduler. Two actors talking at once keep their own
start times.

**Clip:** an argument or a scene where people interrupt each other.

**Evidence to look for:**

- `overlap_check.py` prints the active-speaker table with `+` rows
- the second speaker's start is near its own original start, not after the first
  speaker's line ends
- each speaker keeps their own voice through the overlap
- `dialogue/no_overlap_gain.wav` and `dialogue/combined.wav` differ only in level

**Status:** `BLOCKED`

---

### Case E — BGM-heavy scene

**Proves:** the Phase 4 mix. The film's music survives under the new dialogue.

**Clip:** a montage or a scored scene with little or no dialogue.

**Evidence to look for:**

- the summary says `background: preserved`
- `3_background.wav` from `background_check.py` sounds like the original with
  the talking removed
- `5_mixed.wav` contains both the score and the dialogue, at exactly the video's
  length

**Status:** `BLOCKED`

---

### Case F — dialogue plus loud sound effects

**Proves:** the mix does not lose transient content. An explosion, a door slam
or a car horn should still land.

**Clip:** an action scene with dialogue over loud effects.

**Evidence to look for:**

- the effects are audible in `3_background.wav`
- they are still audible in the final mix, not ducked into nothing
- with ducking **off** (the default), the effects are not pulled down at all

**Note:** this is the case most likely to need a ducking adjustment. The
default is deliberately conservative, so a case where the effects are lost is
evidence for turning ducking *off*, not on.

**Status:** `BLOCKED`

---

### Case G — short-reference actor

**Proves:** the fallback ladder and its warnings. An actor with almost no clean
audio is reported, not silently mis-cloned.

**Clip:** a scene where one person speaks only briefly, ideally under 4 s.

**Evidence to look for:**

- that speaker's status is `short_reference`, or `generic_fallback` if under 2 s
- the warning appears in the summary and in the studio
- the speaker is **never** given another actor's reference — compare the files
  under `references/`, they must differ

**Status:** `BLOCKED`

---

### Case H — code-mixed speech

**Proves:** translation and the code-switching repair. Hinglish, or a film that
mixes two languages.

**Clip:** dialogue that switches language mid-sentence.

**Evidence to look for:**

- the transcript keeps the mixed text
- the translation is coherent rather than transliterated
- the dub speaks the repaired text, and the phrases fit their slots

**Note:** NLLB is a sentence-level model, so heavy code-switching is a known
weakness. A poor result here is a limitation, not a regression.

**Status:** `BLOCKED`

---

## Environment blockers

Every case above is blocked on this machine by the same set of absences. None is
a code failure.

| Requirement | State here | Effect |
|---|---|---|
| `HF_TOKEN` | not available to the process | IndicF5 and pyannote are both gated, so neither can be fetched |
| PyTorch | not installed | no model can run at all |
| faster-whisper | not installed | nothing can be transcribed |
| transformers | not installed | nothing can be translated |
| f5-tts | not installed | no voice can be synthesised |
| pyannote | not installed | no multi-speaker detection |
| Demucs | not installed | no background preservation |
| OpenCV | not installed | no lip sync |
| Wav2Lip, s3fd weights | not downloaded | no lip sync |
| GPU | none (4 CPUs, 15 GB RAM) | not blocking, but synthesis would take hours |

`python scripts/final_check.py plan clip.mp4` reports all of this directly, and
`readiness.snapshot()` returns it as structured data.

## What *is* covered without the models

The model-free suite covers the logic every case above depends on: speaker
assignment, the reference fallback ladder, per-speaker scheduling, overlap gain,
Demucs chunking and reassembly, the stem cache, the LUFS mix and its exact
length, readiness, the run summary, and secret redaction.

```bash
./venv/bin/python -m unittest discover -s tests
```

That is not a substitute for the matrix — it cannot tell you whether a clone
sounds like the actor. It does mean that when the models arrive, a failure is
more likely to be a model or an environment problem than a logic one.

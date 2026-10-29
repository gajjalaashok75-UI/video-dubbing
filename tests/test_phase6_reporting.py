"""
Phase 6: readiness, the run summary, and the API surface around them.

Phase 6 adds no DSP. What it adds is the reporting layer — and the reporting
layer is where a pipeline lies to its user. A stage that silently fell back, a
manifest that omits the warning, a readiness check that says "ready" on a
machine that cannot render, a payload that carries a token: none of these
produce a wrong audio file, and all of them cost somebody an afternoon.

So the tests here are about honesty rather than arithmetic:

  * a missing optional feature must not make the application unusable
  * a fallback must be visible in the summary, not just in a log line
  * no token material may reach any payload
  * the same input change must invalidate the same things every time

No models, no network, no GPU.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import artifact_cache as AC  # noqa: E402
import readiness  # noqa: E402
import run_summary as RS  # noqa: E402
import speakers as SPK  # noqa: E402


def tone(seconds, freq=300.0, amp=0.3, sr=24000):
    """A real waveform, for the tests that drive real ffmpeg."""
    import numpy as np
    t = np.arange(int(seconds * sr), dtype="float32") / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def _load_mix_audio():
    """
    `app._mix_audio` and the gate it reads, loaded by source.

    Importing `app` pulls in OpenCV and torch through `lip_sync`; these tests
    are about the mix parameters, not the models, so the two functions are
    compiled into a throwaway module with only the names they need.
    """
    import ast
    import types as _types
    import av_sync
    import engines

    src_path = os.path.join(ROOT, "engine", "app.py")
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = {"_mix_audio", "_preserve_background"}
    body = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in wanted]
    if len(body) != len(wanted):
        raise AssertionError(
            f"expected {sorted(wanted)} in app.py, found "
            f"{sorted(n.name for n in body)}")
    module = _types.ModuleType("engine_mix_probe")
    module.__dict__["os"] = os
    module.__dict__["av_sync"] = av_sync
    module.__dict__["engines"] = engines
    exec(compile(ast.Module(body=body, type_ignores=[]), src_path, "exec"),
         module.__dict__)
    return module._mix_audio


def speaker(label, duration=9.3, fallback=False, speech=41.7, segments=23):
    return {
        "label": label,
        "total_speech": speech,
        "segment_count": segments,
        "fallback": fallback,
        "reference": {"path": f"/tmp/refs/{label}.wav", "text": f"text for {label}",
                      "start": 18.2, "end": 18.2 + duration,
                      "duration": duration},
        "reference_score": 1.0,
    }


def job(**overrides):
    base = {
        "job_id": "job1", "kind": "dub", "status": "done",
        "source_language": "en", "target_language": "hi",
        "video_duration": 60.0,
        "segments": [{"start": 0.0, "end": 2.0, "text": "a",
                      "speaker": "SPEAKER_00"}],
        "translated_segments": ["x"],
        "word_segments": [{"start": 0.0, "end": 2.0, "text": "a"}],
        "speakers": [speaker("SPEAKER_00"), speaker("SPEAKER_01", duration=7.5)],
        "diarization": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                        {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_01"}],
        "diarization_source": "detected",
        "plan": [
            {"index": 0, "text": "x", "start": 0.0, "duration": 2.0,
             "speaker": "SPEAKER_00", "ref_index": 0},
            {"index": 1, "text": "y", "start": 1.0, "duration": 2.0,
             "speaker": "SPEAKER_01", "ref_index": 1},
        ],
        "stems": {"vocals": "/tmp/v.wav", "background": "/tmp/b.wav",
                  "model": "htdemucs", "chunks": 2},
        "dub_path": "/tmp/dub.wav", "mixed_path": "/tmp/mixed.wav",
        "units_dir": "/tmp/units", "workdir": "/tmp/work",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------
class ReadinessShapeTests(unittest.TestCase):
    """The structured contract the studio and the scripts both read."""

    def test_snapshot_returns_the_agreed_top_level_keys(self):
        snap = readiness.snapshot()
        for key in ("ready", "blocking", "degraded", "hardware", "secrets",
                    "features"):
            self.assertIn(key, snap)

    def test_every_feature_has_a_status_and_a_tier(self):
        snap = readiness.snapshot()
        for name, feature in snap["features"].items():
            self.assertIn(feature["status"], {
                readiness.READY, readiness.FALLBACK_AVAILABLE,
                readiness.OPTIONAL_MISSING, readiness.BLOCKED,
            }, f"{name} has an unknown status")
            self.assertIn(feature.get("tier"), (readiness.CORE,
                                                readiness.OPTIONAL),
                          f"{name} has no tier")

    def test_snapshot_is_json_serialisable(self):
        # It goes straight out of a JSONResponse.
        json.dumps(readiness.snapshot(target_language="hi"))

    def test_a_missing_optional_feature_does_not_block(self):
        # The brief's requirement: an optional feature being absent must not
        # make the application unusable.
        snap = readiness.snapshot()
        optional_blocked = [
            name for name, f in snap["features"].items()
            if f.get("tier") == readiness.OPTIONAL
            and f["status"] == readiness.BLOCKED]
        self.assertEqual(optional_blocked, [],
                         "an optional feature reported BLOCKED")
        for name, feature in snap["features"].items():
            if feature.get("tier") == readiness.OPTIONAL:
                self.assertNotIn(name, snap["blocking"],
                                 f"{name} is optional but is blocking")

    def test_blocking_only_contains_core_features(self):
        snap = readiness.snapshot(target_language="hi")
        for name in snap["blocking"]:
            self.assertEqual(snap["features"][name].get("tier"),
                             readiness.CORE)

    def test_a_missing_optional_feature_offers_a_fallback(self):
        # Every optional feature says what happens without it, so the studio
        # can reassure rather than just warn.
        for name in ("diarization", "background_preservation", "lipsync"):
            feature = readiness.snapshot()["features"][name]
            if feature["status"] != readiness.READY:
                self.assertTrue(feature.get("fallback"),
                                f"{name} is unavailable with no stated fallback")

    def test_target_language_is_reported_when_asked_about(self):
        snap = readiness.snapshot(target_language="hi")
        self.assertIn("target_language", snap["features"])
        self.assertEqual(snap["features"]["target_language"]["language"], "hi")

    def test_an_unknown_target_language_is_blocked_not_crashing(self):
        snap = readiness.snapshot(target_language="not-a-language")
        self.assertFalse(snap["ready"])
        self.assertEqual(snap["features"]["target_language"]["status"],
                         readiness.BLOCKED)

    def test_hardware_reports_a_device(self):
        hw = readiness.snapshot()["hardware"]
        self.assertIn(hw["device"], ("cpu", "cuda", "mps"))
        self.assertIn("cpus", hw)


class ReadinessNeverLeaksSecretsTests(unittest.TestCase):
    """The brief's rule, as tests. No value, no prefix, no length."""

    SECRET = "hf_THIS_IS_A_FAKE_TOKEN_1234567890"

    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        os.environ["HF_TOKEN"] = self.SECRET

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_token_state_returns_only_presence(self):
        state = readiness.token_state()
        self.assertEqual(state, {"present": True})
        self.assertNotIn("token", state)
        self.assertNotIn("value", state)

    def test_snapshot_never_contains_the_token(self):
        blob = json.dumps(readiness.snapshot(target_language="hi"))
        self.assertNotIn(self.SECRET, blob)
        # Not a prefix or a suffix either.
        self.assertNotIn(self.SECRET[:12], blob)
        self.assertNotIn(self.SECRET[-12:], blob)

    def test_the_summary_never_contains_the_token(self):
        blob = json.dumps(RS.build(job()))
        self.assertNotIn(self.SECRET, blob)

    def test_the_secrets_field_names_the_variable_not_its_value(self):
        self.assertEqual(readiness.snapshot()["secrets"],
                         {"HF_TOKEN": "present"})

    def test_redact_removes_a_token_from_a_hand_built_payload(self):
        payload = {"a": f"Bearer {self.SECRET}",
                   "b": {"nested": f"token={self.SECRET}"},
                   "c": [self.SECRET, "safe"],
                   "d": 5}
        out = readiness.redact(payload)
        blob = json.dumps(out)
        self.assertNotIn(self.SECRET, blob)
        self.assertEqual(out["d"], 5)
        self.assertIn("safe", out["c"])

    def test_redact_handles_a_bearer_header_whatever_the_token(self):
        out = readiness.redact({"header": "Bearer something-else-entirely"})
        self.assertEqual(out["header"], "Bearer <redacted>")

    def test_redact_leaves_ordinary_text_alone(self):
        payload = {"reason": "HF_TOKEN is not set.", "n": 3}
        self.assertEqual(readiness.redact(payload), payload)

    def test_a_short_value_is_not_treated_as_a_secret(self):
        # A one-character environment variable is not a credential, and
        # replacing every "a" in a report would be worse than useless.
        os.environ["OPENAI_API_KEY"] = "x"
        try:
            out = readiness.redact({"text": "a normal sentence"})
            self.assertEqual(out["text"], "a normal sentence")
        finally:
            os.environ.pop("OPENAI_API_KEY", None)


class GatedGuidanceTests(unittest.TestCase):
    """
    Four different problems, four different fixes.

    Collapsing them into "could not load the model" sends a user to re-accept a
    licence they already accepted, or to install a package they already have.
    """

    def test_no_token_says_how_to_get_one(self):
        reason = readiness._gated_reason(
            "pyannote.audio", "pyannote/speaker-diarization-3.1",
            present=True, token=False)
        self.assertIn("HF_TOKEN", reason)
        self.assertIn("huggingface.co/pyannote/speaker-diarization-3.1", reason)
        self.assertIn("accept", reason.lower())

    def test_missing_package_says_how_to_install_it(self):
        reason = readiness._gated_reason(
            "pyannote.audio", "pyannote/speaker-diarization-3.1",
            present=False, token=False)
        self.assertIn("not installed", reason)
        self.assertNotIn("HF_TOKEN", reason)

    def test_package_and_token_present_but_no_weights_says_download(self):
        reason = readiness._gated_reason(
            "pyannote.audio", "pyannote/speaker-diarization-3.1",
            present=True, token=True)
        self.assertIn("weights are not downloaded", reason)
        self.assertNotIn("not installed", reason)

    def test_the_three_reasons_are_all_different(self):
        reasons = {
            readiness._gated_reason("p", "r", True, False),
            readiness._gated_reason("p", "r", False, False),
            readiness._gated_reason("p", "r", True, True),
        }
        self.assertEqual(len(reasons), 3)

    def test_the_diarizer_classifies_each_failure_differently(self):
        import speaker_diarization as SD
        d = SD.PyannoteDiarizer()
        cases = {
            "ModuleNotFoundError: No module named 'pyannote'": "not installed",
            "401 Client Error: Unauthorized for url": "401",
            "403 Client Error: Cannot access gated repo": "not been granted",
            "EntryNotFoundError: 404 not found": "weights are not available",
            "ConnectionError: connection timed out": "network",
        }
        for message, needle in cases.items():
            with self.subTest(message=message):
                out = d._explain_load_failure(RuntimeError(message))
                self.assertIn(needle.lower(), out.lower(),
                              f"{message!r} produced {out!r}")

    def test_the_classifier_never_raises_on_an_odd_exception(self):
        import speaker_diarization as SD
        d = SD.PyannoteDiarizer()
        for exc in (RuntimeError(""), ValueError("x" * 5000), KeyError("k")):
            self.assertTrue(d._explain_load_failure(exc))


# ---------------------------------------------------------------------------
# Speaker status and warnings
# ---------------------------------------------------------------------------
class SpeakerStatusTests(unittest.TestCase):
    """The four states, and the warning each produces."""

    def test_a_long_clean_reference_is_cloned(self):
        self.assertEqual(SPK.status_of(speaker("SPEAKER_00", duration=9.3)),
                         SPK.CLONED)
        self.assertIsNone(SPK.warning_for(speaker("SPEAKER_00", duration=9.3)))

    def test_a_short_reference_is_flagged_and_warned_about(self):
        entry = speaker("SPEAKER_03", duration=3.1)
        self.assertEqual(SPK.status_of(entry), SPK.SHORT_REFERENCE)
        warning = SPK.warning_for(entry)
        self.assertIn("3.1 seconds", warning)
        self.assertIn("SPEAKER_03", warning)
        self.assertIn("may be reduced", warning)

    def test_a_generic_fallback_is_named(self):
        entry = speaker("SPEAKER_05", duration=8.0, fallback=True)
        self.assertEqual(SPK.status_of(entry), SPK.GENERIC_FALLBACK)
        self.assertIn("fallback voice", SPK.warning_for(entry))

    def test_no_reference_is_unresolved(self):
        entry = speaker("SPEAKER_07")
        entry["reference"] = None
        self.assertEqual(SPK.status_of(entry), SPK.UNRESOLVED)
        self.assertIsNotNone(SPK.warning_for(entry))

    def test_a_reference_below_the_usable_floor_is_unresolved(self):
        entry = speaker("SPEAKER_08", duration=1.0)
        self.assertEqual(SPK.status_of(entry), SPK.UNRESOLVED)

    def test_the_boundary_is_exactly_the_preferred_length(self):
        self.assertEqual(SPK.status_of(speaker("A", duration=6.0)), SPK.CLONED)
        self.assertEqual(SPK.status_of(speaker("A", duration=5.99)),
                         SPK.SHORT_REFERENCE)

    def test_warnings_lists_only_the_flagged_speakers(self):
        speakers = [speaker("SPEAKER_00"),
                    speaker("SPEAKER_03", duration=3.1),
                    speaker("SPEAKER_05", duration=8.0, fallback=True)]
        self.assertEqual(len(SPK.warnings(speakers)), 2)

    def test_status_is_derived_not_stored(self):
        # Changing the reference must change the status without anything having
        # to remember to update a field.
        entry = speaker("A", duration=9.3)
        self.assertEqual(SPK.status_of(entry), SPK.CLONED)
        entry["reference"]["duration"] = 1.0
        self.assertEqual(SPK.status_of(entry), SPK.UNRESOLVED)


class SpeakerDescribeTests(unittest.TestCase):
    """The shape the studio reads, and the alias rule."""

    def test_describe_carries_everything_the_studio_shows(self):
        row = SPK.describe(speaker("SPEAKER_00"))
        for key in ("label", "name", "status", "warning", "total_speech",
                    "segment_count", "reference", "reference_score"):
            self.assertIn(key, row)
        for key in ("path", "text", "start", "end", "duration"):
            self.assertIn(key, row["reference"])

    def test_an_alias_becomes_the_name_but_never_the_label(self):
        row = SPK.describe(speaker("SPEAKER_00"), aliases={"SPEAKER_00": "Narrator"})
        self.assertEqual(row["name"], "Narrator")
        self.assertEqual(row["alias"], "Narrator")
        # The stable id must survive: the cache, the plan and the reference
        # files all key on it.
        self.assertEqual(row["label"], "SPEAKER_00")

    def test_without_an_alias_the_name_is_the_label(self):
        row = SPK.describe(speaker("SPEAKER_01"))
        self.assertEqual(row["name"], "SPEAKER_01")
        self.assertIsNone(row["alias"])

    def test_an_empty_alias_falls_back_to_the_label(self):
        row = SPK.describe(speaker("SPEAKER_00"), aliases={"SPEAKER_00": "   "})
        self.assertEqual(row["name"], "SPEAKER_00")

    def test_describe_all_counts_phrases_per_speaker(self):
        plan = [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_00"},
                {"speaker": "SPEAKER_01"}]
        rows = SPK.describe_all([speaker("SPEAKER_00"), speaker("SPEAKER_01")],
                                plan=plan)
        self.assertEqual(rows[0]["phrase_count"], 2)
        self.assertEqual(rows[1]["phrase_count"], 1)

    def test_describe_is_json_safe(self):
        json.dumps(SPK.describe_all([speaker("SPEAKER_00")]))


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------
class RunSummaryTests(unittest.TestCase):
    """What the manifest records, and what it refuses to hide."""

    def test_the_summary_has_the_agreed_sections(self):
        summary = RS.build(job())
        for key in ("languages", "video_duration", "segments", "speakers",
                    "diarization", "overlap", "background", "models",
                    "artifacts", "stages", "warnings", "verification"):
            self.assertIn(key, summary)

    def test_it_is_json_safe(self):
        json.dumps(RS.build(job()))

    def test_the_overlap_count_matches_the_plan(self):
        summary = RS.build(job())
        # The two planned units overlap from 1.0 to 2.0.
        self.assertEqual(summary["overlap"]["regions"], 1)
        self.assertEqual(summary["overlap"]["max_speakers"], 2)

    def test_a_single_speaker_plan_reports_no_overlap(self):
        summary = RS.build(job(plan=[
            {"index": 0, "text": "x", "start": 0.0, "duration": 1.0,
             "speaker": "SPEAKER_00"}]))
        self.assertEqual(summary["overlap"]["regions"], 0)

    def test_segment_counts_come_from_the_job(self):
        summary = RS.build(job())
        self.assertEqual(summary["segments"]["transcribed"], 1)
        self.assertEqual(summary["segments"]["translated"], 1)
        self.assertEqual(summary["segments"]["synthesized"], 2)

    def test_background_state_is_reported(self):
        summary = RS.build(job())
        self.assertTrue(summary["background"]["separated"])
        self.assertEqual(summary["background"]["model"], "htdemucs")
        self.assertTrue(summary["background"]["mixed"])

    def test_background_absent_is_reported_as_absent(self):
        summary = RS.build(job(stems=None, mixed_path=None))
        self.assertFalse(summary["background"]["separated"])
        self.assertFalse(summary["background"]["mixed"])

    def test_the_models_actually_used_are_named(self):
        models = RS.build(job())["models"]
        self.assertEqual(models["asr"], "Systran/faster-whisper-large-v3")
        self.assertEqual(models["translation"],
                         "facebook/nllb-200-distilled-600M")
        self.assertEqual(models["tts"], "ai4bharat/IndicF5")
        self.assertEqual(models["separation"], "htdemucs")

    def test_the_diarization_backend_is_named_only_when_it_ran(self):
        self.assertIsNotNone(RS.build(job())["models"]["diarization"])
        self.assertIsNone(
            RS.build(job(diarization=[], speakers=[
                speaker("SPEAKER_00")]))["models"]["diarization"])

    def test_environment_is_included_when_a_snapshot_is_given(self):
        snap = readiness.snapshot()
        summary = RS.build(job(), readiness=snap)
        self.assertIn("environment", summary)
        # Whether a token is configured is a property of the machine running
        # the test, not something this test can assume: asserting "missing"
        # made it fail on every machine that HAS one. What must hold is that
        # the summary carries a word and never the value.
        present = bool(os.getenv("HF_TOKEN"))
        self.assertEqual(summary["environment"]["secrets"],
                         {"HF_TOKEN": "present" if present else "missing"})
        self.assertIn("device", summary["environment"])


class RunSummaryHonestyTests(unittest.TestCase):
    """
    A fallback must be visible. This is the module's reason to exist.
    """

    def test_a_missing_diarization_is_warned_about(self):
        summary = RS.build(job(diarization=[], diarization_source=None))
        joined = " ".join(summary["warnings"])
        self.assertIn("one voice", joined)

    def test_a_short_reference_warning_reaches_the_summary(self):
        summary = RS.build(job(speakers=[speaker("SPEAKER_03", duration=3.1)]))
        joined = " ".join(summary["warnings"])
        self.assertIn("3.1 seconds", joined)

    def test_a_fallback_speaker_warning_reaches_the_summary(self):
        summary = RS.build(job(speakers=[
            speaker("SPEAKER_05", duration=8.0, fallback=True)]))
        joined = " ".join(summary["warnings"])
        self.assertIn("fallback voice", joined)

    def test_unpreserved_background_is_warned_about(self):
        summary = RS.build(job(stems=None))
        joined = " ".join(summary["warnings"])
        self.assertIn("not preserved", joined)

    def test_silent_phrases_are_warned_about(self):
        summary = RS.build(job(missing_units=[3, 7]))
        self.assertIn("2 phrase(s)", " ".join(summary["warnings"]))

    def test_a_failed_sync_check_is_warned_about(self):
        summary = RS.build(job(sync={"ok": False, "reason": "audio 0.4s longer"}))
        self.assertIn("0.4s longer", " ".join(summary["warnings"]))

    def test_a_failed_run_records_its_reason(self):
        summary = RS.build(job(status="failed", error="no speech detected"))
        self.assertIn("no speech detected", " ".join(summary["warnings"]))

    def test_a_clean_run_has_no_warnings(self):
        # The control: warnings must mean something.
        clean = job(speakers=[speaker("SPEAKER_00")],
                    diarization=[{"start": 0.0, "end": 2.0,
                                  "speaker": "SPEAKER_00"}],
                    plan=[{"index": 0, "text": "x", "start": 0.0,
                           "duration": 1.0, "speaker": "SPEAKER_00"}])
        self.assertEqual(RS.build(clean)["warnings"], [])

    def test_the_rendered_text_shows_every_warning(self):
        summary = RS.build(job(speakers=[speaker("SPEAKER_03", duration=3.1)]))
        text = RS.render(summary)
        self.assertIn("SPEAKER_03", text)
        self.assertIn("3.1 seconds", text)


class StageReportingTests(unittest.TestCase):
    """Optional stages must say they were skipped, not look successful."""

    def test_every_stage_has_a_key_label_and_state(self):
        for row in RS.stage_rows(job()):
            self.assertIn("key", row)
            self.assertIn("label", row)
            self.assertIn(row["state"], ("done", "skipped", "fallback"))

    def test_the_stage_order_matches_the_pipeline(self):
        keys = [row["key"] for row in RS.stage_rows(job())]
        self.assertEqual(keys, [key for key, _, _ in RS.STAGES])
        for expected in ("separating_audio", "diarizing", "assembling_dialogue",
                         "mixing_background", "lip_syncing"):
            self.assertIn(expected, keys)

    def test_absent_separation_is_skipped_with_a_reason(self):
        rows = {r["key"]: r for r in RS.stage_rows(job(stems=None))}
        self.assertEqual(rows["separating_audio"]["state"], "skipped")
        self.assertIn("Demucs", rows["separating_audio"]["detail"])

    def test_failed_diarization_is_a_fallback_not_a_failure(self):
        rows = {r["key"]: r for r in RS.stage_rows(job(diarization=[]))}
        self.assertEqual(rows["diarizing"]["state"], "fallback")
        self.assertIn("single voice", rows["diarizing"]["detail"])

    def test_a_present_stage_reports_its_detail(self):
        rows = {r["key"]: r for r in RS.stage_rows(job())}
        self.assertEqual(rows["diarizing"]["state"], "done")
        self.assertIn("2 turns", rows["diarizing"]["detail"])
        self.assertIn("2 chunk", rows["separating_audio"]["detail"])

    def test_an_audio_only_output_skips_lip_sync(self):
        rows = {r["key"]: r for r in RS.stage_rows(job(kind="audio"))}
        self.assertEqual(rows["lip_syncing"]["state"], "skipped")


class VoiceVerificationTests(unittest.TestCase):
    """A diagnostic, not identity proof."""

    def test_scores_are_reported_per_speaker(self):
        summary = RS.build(job(speaker_voice_match=[
            {"label": "SPEAKER_00", "score": 0.72, "cosine": 0.91},
            {"label": "SPEAKER_01", "score": 0.31, "cosine": 0.88}]))
        rows = summary["verification"]["per_speaker"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["label"], "SPEAKER_00")

    def test_a_high_score_reads_as_a_match(self):
        self.assertEqual(RS._match_status({"score": 0.8}), "match")

    def test_a_negative_score_reads_as_different(self):
        self.assertEqual(RS._match_status({"score": -0.2}), "different")

    def test_the_middle_reads_as_uncertain_rather_than_pass(self):
        # No threshold here has been validated across enough voices to claim
        # one, so the middle band is deliberately not called a pass.
        self.assertEqual(RS._match_status({"score": 0.3}), "uncertain")

    def test_an_unmeasured_speaker_says_so(self):
        self.assertEqual(RS._match_status({"score": None}), "unmeasured")
        self.assertEqual(RS._match_status({}), "unmeasured")

    def test_a_non_numeric_score_does_not_raise(self):
        self.assertEqual(RS._match_status({"score": "abc"}), "unmeasured")


# ---------------------------------------------------------------------------
# Artifact graph / cache invalidation
# ---------------------------------------------------------------------------
class ArtifactGraphTests(unittest.TestCase):
    """What depends on what, and therefore what a change costs."""

    def test_every_artifact_declares_inputs_and_a_cost(self):
        for name, entry in AC.ARTIFACTS.items():
            self.assertTrue(entry["inputs"], f"{name} has no inputs")
            self.assertIn(entry["cost"], ("seconds", "minutes"))

    def test_every_dependency_names_a_real_artifact(self):
        for name, entry in AC.ARTIFACTS.items():
            for dep in entry["depends_on"]:
                self.assertIn(dep, AC.ARTIFACTS,
                              f"{name} depends on unknown {dep}")

    def test_the_graph_is_acyclic(self):
        for name in AC.ARTIFACTS:
            self.assertNotIn(name, AC.downstream(name),
                             f"{name} is downstream of itself")

    def test_a_target_language_change_does_not_touch_the_stems(self):
        # The point of a narrow invalidation: re-running Demucs because the
        # translation changed would be minutes for an unrelated answer.
        stale = AC.invalidated_by(["target_language"])
        self.assertIn("translation", stale)
        self.assertIn("output_video", stale)
        self.assertNotIn("stems", stale)
        self.assertNotIn("transcript", stale)
        self.assertNotIn("diarization", stale)

    def test_a_music_gain_change_invalidates_only_the_mix(self):
        self.assertEqual(AC.invalidated_by(["music_gain"]), ["mixed_audio"])

    def test_a_ducking_toggle_invalidates_only_the_mix(self):
        self.assertEqual(AC.invalidated_by(["background_duck"]), ["mixed_audio"])

    def test_a_chunk_size_change_invalidates_separation_and_its_dependents(self):
        stale = AC.invalidated_by(["demucs_chunk_seconds"])
        self.assertIn("stems", stale)
        # The transcript is read from the vocals stem, so it genuinely is
        # downstream of separation.
        self.assertIn("transcript", stale)
        self.assertIn("output_video", stale)

    def test_a_trim_change_invalidates_everything(self):
        stale = AC.invalidated_by(["trim"])
        for name in ("stt_audio", "hq_audio", "stems", "transcript",
                     "diarization", "output_video"):
            self.assertIn(name, stale)

    def test_an_unknown_field_invalidates_nothing(self):
        self.assertEqual(AC.invalidated_by(["not_a_field"]), [])

    def test_dependents_are_returned_in_dependency_order(self):
        stale = AC.invalidated_by(["transcript"])
        self.assertLess(stale.index("translation"), stale.index("phrase_audio"))
        self.assertLess(stale.index("phrase_audio"), stale.index("mixed_audio"))

    def test_the_worst_cost_is_reported(self):
        # A target-language change re-runs translation, then every phrase, then
        # the render. The honest worst case is the render: it is minutes, and
        # it is the thing the user waits for last.
        stale = AC.invalidated_by(["target_language"])
        self.assertEqual(AC.cheapest_stale(stale), "output_video")

    def test_a_cheap_change_reports_a_cheap_worst_case(self):
        self.assertEqual(AC.cheapest_stale(AC.invalidated_by(["music_gain"])),
                         "mixed_audio")
        # And it is genuinely seconds, not minutes.
        self.assertEqual(AC.ARTIFACTS["mixed_audio"]["cost"], "seconds")

    def test_describe_is_readable(self):
        text = AC.describe()
        self.assertIn("Artifact dependency graph", text)
        self.assertIn("stems", text)


class PerSpeakerInvalidationTests(unittest.TestCase):
    """
    Requirement 10: changing one speaker's reference must not re-synthesize
    another speaker's phrases.
    """

    def test_a_reclone_is_scoped_to_one_speaker(self):
        stale = AC.reference_invalidates("SPEAKER_01")
        self.assertIn("phrase_audio:SPEAKER_01", stale)
        self.assertNotIn("phrase_audio:SPEAKER_00", stale)

    def test_a_reclone_never_invalidates_the_transcript(self):
        # The expensive stages must survive: re-cloning is a voice change, not
        # a re-transcription.
        stale = AC.reference_invalidates("SPEAKER_02")
        for preserved in ("transcript", "diarization", "translation", "stems"):
            self.assertNotIn(preserved, stale)

    def test_a_reclone_still_invalidates_the_downstream_outputs(self):
        stale = AC.reference_invalidates("SPEAKER_00")
        for downstream in ("dialogue_track", "mixed_audio", "output_video"):
            self.assertIn(downstream, stale)

    def test_two_speakers_produce_two_different_sets(self):
        a = set(AC.reference_invalidates("SPEAKER_00"))
        b = set(AC.reference_invalidates("SPEAKER_01"))
        self.assertNotEqual(a, b)
        self.assertEqual(a & b, {"dialogue_track", "mixed_audio",
                                 "output_video"})


class StaleDetectionTests(unittest.TestCase):
    """Validity is a comparison, never a file check."""

    def test_no_previous_fingerprint_means_everything_is_stale(self):
        self.assertEqual(len(AC.stale_artifacts({}, None)), len(AC.ARTIFACTS))

    def test_an_unchanged_fingerprint_is_not_stale(self):
        current = {"target_language": "hi", "source_language": "en"}
        self.assertEqual(AC.stale_artifacts(current, dict(current)), [])

    def test_a_changed_field_makes_its_dependents_stale(self):
        previous = {"target_language": "hi"}
        current = {"target_language": "ta"}
        stale = AC.stale_artifacts(current, previous)
        self.assertIn("translation", stale)
        self.assertNotIn("stems", stale)

    def test_a_field_absent_from_the_old_manifest_is_not_a_change(self):
        # A run stored before a setting existed must not read as changed on
        # that setting, or every old project re-renders for no reason.
        previous = {"target_language": "hi"}
        current = {"target_language": "hi", "background_duck": True}
        self.assertEqual(AC.stale_artifacts(current, previous), [])

    def test_fingerprint_records_only_known_inputs(self):
        fp = AC.fingerprint({"target_language": "hi", "nonsense": 1,
                             "video_duration": 30.0})
        self.assertIn("target_language", fp)
        self.assertNotIn("nonsense", fp)


# ---------------------------------------------------------------------------
# Manifest backward compatibility
# ---------------------------------------------------------------------------
class ManifestCompatibilityTests(unittest.TestCase):
    """Old projects must keep loading."""

    def test_a_job_with_none_of_the_new_keys_still_builds(self):
        old = {"job_id": "old", "kind": "dub", "status": "done",
               "target_language": "hi", "video_duration": 10.0,
               "segments": [], "translated_segments": []}
        summary = RS.build(old)
        self.assertEqual(summary["speakers"]["count"], 0)
        self.assertEqual(summary["overlap"]["regions"], 0)
        self.assertFalse(summary["background"]["separated"])

    def test_a_job_with_no_speakers_falls_back_to_one_implicit_speaker(self):
        old = {"job_id": "old", "kind": "dub", "status": "done",
               "reference": {"path": "/tmp/r.wav", "text": "t",
                             "duration": 8.0}}
        registry = SPK.normalize_speakers(old.get("speakers")) \
            or SPK.implicit_speaker(old["reference"])
        rows = SPK.describe_all(registry)
        self.assertEqual(rows[0]["label"], "SPEAKER_00")
        self.assertEqual(rows[0]["status"], SPK.CLONED)

    def test_new_keys_are_in_the_persisted_set(self):
        import project
        for key in ("speaker_aliases", "speaker_voice_match", "stems",
                    "speakers", "diarization", "diarization_source"):
            self.assertIn(key, project.PERSISTED)

    def test_an_alias_survives_a_manifest_round_trip(self):
        import project
        payload = project._plain({"speaker_aliases": {"SPEAKER_00": "Narrator"},
                                  "speaker_voice_match": [
                                      {"label": "SPEAKER_00", "score": 0.5}]})
        again = json.loads(json.dumps(payload))
        self.assertEqual(again["speaker_aliases"]["SPEAKER_00"], "Narrator")

    def test_a_manifest_with_aliases_still_describes_speakers(self):
        rows = SPK.describe_all(
            [speaker("SPEAKER_00")],
            aliases={"SPEAKER_00": "Narrator"})
        self.assertEqual(rows[0]["name"], "Narrator")
        self.assertEqual(rows[0]["label"], "SPEAKER_00")


# ---------------------------------------------------------------------------
# The API surface, tested without importing the engine's heavy dependencies
# ---------------------------------------------------------------------------
def _app_route_paths() -> set:
    """
    Every route path declared in `app.py`, read from the source.

    Read rather than imported: importing the engine pulls in OpenCV and torch
    through `lip_sync`, and these tests are about which endpoints exist and what
    shape they return, not about the models behind them.
    """
    import ast
    src = os.path.join(ROOT, "engine", "app.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    paths = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "app"):
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                paths.add(decorator.args[0].value)
    return paths


class ApiSurfaceTests(unittest.TestCase):
    """The endpoints Phase 6 adds, and that the old ones survive."""

    def setUp(self):
        self.paths = _app_route_paths()

    def test_the_phase_6_endpoints_exist(self):
        for path in ("/api/readiness", "/jobs/{job_id}/summary",
                     "/jobs/{job_id}/speakers",
                     "/jobs/{job_id}/speakers/{label}",
                     "/jobs/{job_id}/artifact/{name}",
                     "/api/artifacts", "/jobs/{job_id}/cache"):
            self.assertIn(path, self.paths, f"{path} is missing")

    def test_the_earlier_phase_endpoints_survive(self):
        for path in ("/process_video/", "/jobs/{job_id}/status",
                     "/jobs/{job_id}/segments", "/jobs/{job_id}/reference",
                     "/jobs/{job_id}/reference/candidates",
                     "/jobs/{job_id}/rerender", "/languages", "/api/health"):
            self.assertIn(path, self.paths, f"{path} was removed")

    def test_the_readiness_endpoint_takes_language_parameters(self):
        import ast
        src = os.path.join(ROOT, "engine", "app.py")
        with open(src, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "get_readiness")
        params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        self.assertIn("target_language", params)
        self.assertIn("source_language", params)

    def test_no_endpoint_returns_a_token(self):
        # The readiness and summary endpoints are the two that could plausibly
        # leak one, and both go through `readiness.redact` / `token_state`.
        blob = json.dumps(readiness.snapshot(target_language="hi"))
        self.assertNotIn("hf_", blob)
        self.assertNotIn("Bearer", blob)
        self.assertNotIn("Authorization", blob)


class ProgressStageNameTests(unittest.TestCase):
    """The studio's stage keys must match what the backend reports."""

    def test_every_backend_stage_key_has_a_studio_entry(self):
        studio = os.path.join(ROOT, "apps", "studio", "src", "lib",
                              "pipeline.ts")
        with open(studio, encoding="utf-8") as f:
            text = f.read()
        for key, _label, _optional in RS.STAGES:
            # `assembling_dialogue` is part of synthesizing_voice on the
            # backend; every other key is reported verbatim.
            if key == "assembling_dialogue":
                continue
            self.assertIn(f'key: "{key}"', text,
                          f"the studio has no progress entry for {key}")

    def test_the_phase_6_stage_names_are_present(self):
        labels = " ".join(label for _k, label, _o in RS.STAGES)
        for phrase in ("Separating", "Detecting speakers",
                       "Selecting speaker references", "Assembling dialogue",
                       "Mixing original background"):
            self.assertIn(phrase, labels)


class MixOverrideTests(unittest.TestCase):
    """
    The per-render background controls.

    The defaults must reproduce the tested Phase 4 behaviour exactly, or a
    render made without touching the new controls would differ from one made
    before they existed.
    """

    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="mix_override_")
        self._shutil = shutil
        self.mix = _load_mix_audio()

    def tearDown(self):
        self._shutil.rmtree(self.tmp, ignore_errors=True)

    def _wav(self, name, wave):
        import soundfile as sf
        path = os.path.join(self.tmp, name)
        sf.write(path, wave, 24000, subtype="PCM_16")
        return path

    def _stems(self):
        import numpy as np
        bed = (0.5 * np.sin(2 * np.pi * 80 *
                           np.arange(24000 * 4) / 24000)).astype("float32")
        return {"vocals": None, "background": self._wav("bed.wav", bed)}

    def test_no_override_uses_the_stored_default(self):
        # Omitting both is the pre-Phase-6 call, and must still work.
        dub = self._wav("dub.wav", tone(300, 4.0))
        out = self.mix("job", self.tmp, "stem", dub, 4.0, self._stems(),
                       None, -18.0)
        self.assertIsNotNone(out)
        self.assertTrue(os.path.exists(out))

    def test_an_explicit_level_is_accepted(self):
        dub = self._wav("dub.wav", tone(300, 4.0))
        out = self.mix("job", self.tmp, "stem", dub, 4.0, self._stems(),
                       None, -18.0, background_lufs=-30.0)
        self.assertTrue(os.path.exists(out))

    def test_a_quieter_background_really_is_quieter(self):
        import numpy as np
        dub = self._wav("dub.wav", tone(300, 4.0))
        loud = self.mix("a", self.tmp, "loud", dub, 4.0, self._stems(),
                        None, -18.0, background_lufs=-14.0)
        soft = self.mix("b", self.tmp, "soft", dub, 4.0, self._stems(),
                        None, -18.0, background_lufs=-38.0)

        def bed_energy(path):
            import soundfile as sf
            wave, rate = sf.read(path, dtype="float32")
            spec = np.abs(np.fft.rfft(wave)) ** 2
            freqs = np.fft.rfftfreq(len(wave), 1 / rate)
            mask = (freqs >= 60) & (freqs < 110)
            return float(spec[mask].sum())

        self.assertGreater(bed_energy(loud), bed_energy(soft),
                           "a lower background level did not produce a quieter "
                           "bed")

    def test_an_out_of_range_level_is_clamped_not_rejected(self):
        # A mistyped value should degrade, not ruin the render or crash it.
        dub = self._wav("dub.wav", tone(300, 4.0))
        for level in (-500.0, 40.0):
            with self.subTest(level=level):
                out = self.mix("job", self.tmp, f"c{int(level)}", dub, 4.0,
                               self._stems(), None, -18.0,
                               background_lufs=level)
                self.assertTrue(os.path.exists(out))
                # And the result is still a valid, non-clipping track.
                import soundfile as sf
                wave, _ = sf.read(out, dtype="float32")
                self.assertLessEqual(float(abs(wave).max()), 1.0 + 1e-6)

    def test_ducking_can_be_turned_on_per_render(self):
        dub = self._wav("dub.wav", tone(300, 4.0))
        out = self.mix("job", self.tmp, "duck", dub, 4.0, self._stems(),
                       None, -18.0, background_duck=True)
        self.assertTrue(os.path.exists(out))

    def test_the_result_is_still_exactly_the_video_length(self):
        # The Phase 4 invariant must survive the new parameters.
        import av_sync
        dub = self._wav("dub.wav", tone(300, 4.0))
        for duck in (False, True):
            with self.subTest(duck=duck):
                out = self.mix("job", self.tmp, f"len{int(duck)}", dub, 4.0,
                               self._stems(), None, -18.0,
                               background_lufs=-24.0, background_duck=duck)
                self.assertAlmostEqual(av_sync.duration(out), 4.0, places=3)


class BackgroundControlParamTests(unittest.TestCase):
    """The query parameters exist and reach the task."""

    def setUp(self):
        src = os.path.join(ROOT, "engine", "app.py")
        with open(src, encoding="utf-8") as f:
            self.text = f.read()

    def test_the_route_accepts_both_parameters(self):
        for name in ("background_lufs", "background_duck"):
            self.assertIn(name, self.text,
                          f"the render route does not accept {name}")

    def test_they_are_recorded_on_the_job(self):
        self.assertIn('"background_lufs": background_lufs', self.text)
        self.assertIn('"background_duck": background_duck', self.text)

    def test_they_reach_the_worker(self):
        self.assertIn("background_lufs, background_duck)", self.text)

    def test_the_defaults_are_the_tested_phase_4_values(self):
        import engines
        self.assertEqual(engines.TUNABLES["BACKGROUND_LUFS"][0], -26.0)
        self.assertEqual(engines.TUNABLES["DIALOGUE_LUFS"][0], -12.0)
        self.assertIs(engines.TUNABLES["BACKGROUND_DUCK"][0], False)


if __name__ == "__main__":
    unittest.main()

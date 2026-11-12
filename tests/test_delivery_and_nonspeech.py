"""
Two things the dub used to get wrong about HOW a line is delivered.

1. Whisper labels non-speech with bracketed cues and note glyphs. In a film
   trailer almost every line carries one, and the pipeline translated and then
   SPOKE them: the dub narrated its own score ("[Music]") in every gap and sang
   the lyrics of the song. They are descriptions of the audio, not dialogue.

2. Every line arrived at the same level. A trailer is built on the contrast
   between a whispered line and a shouted one, and flattening it is what makes
   a dub sound like a read-through. S2 Pro accepts inline delivery tags, so the
   source level is measured and the tag chosen from it - but only when it is a
   clear deviation from the clip's own median, so a uniformly loud clip is not
   tagged loud all the way through.

Model-free: the STT helpers are pure text, and the prosody tests build their
own level arrays rather than decoding audio.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

# speech_to_text_v2 imports torch at module scope for device selection, but
# degrades to CPU when it is absent, so these helpers need no stub.
import speech_activity as SA  # noqa: E402
import speech_to_text_v2 as STT  # noqa: E402


class NonSpeechStrippingTests(unittest.TestCase):
    def test_a_music_cue_is_removed_entirely(self):
        self.assertEqual(STT.strip_non_speech("[Music]"), "")

    def test_a_cue_around_speech_leaves_the_speech(self):
        self.assertEqual(STT.strip_non_speech("[Music] Hello there."),
                         "Hello there.")
        self.assertEqual(STT.strip_non_speech("[Applause] You did it!"),
                         "You did it!")

    def test_note_glyphs_are_removed(self):
        self.assertEqual(STT.strip_non_speech("\u266a\u266a la la \u266a\u266a"),
                         "la la")
        self.assertEqual(STT.strip_non_speech("\u266a\u266a\u266a"), "")

    def test_a_bare_stage_direction_line_is_removed(self):
        for cue in ("(laughs)", "Music", "[BLANK_AUDIO]", "(sighs)"):
            self.assertEqual(STT.strip_non_speech(cue), "", cue)

    def test_ordinary_prose_keeps_the_word_music(self):
        # The removal must not eat dialogue that happens to contain a cue word.
        self.assertEqual(STT.strip_non_speech("I love this music."),
                         "I love this music.")
        self.assertEqual(STT.strip_non_speech("Hello [inaudible] world"),
                         "Hello world")

    def test_empty_and_none_are_safe(self):
        self.assertEqual(STT.strip_non_speech(""), "")
        self.assertEqual(STT.strip_non_speech(None), "")

    def test_a_line_that_is_only_annotation_is_flagged_as_not_speech(self):
        for cue in ("[Music]", "\u266a\u266a\u266a", "(laughs)", ""):
            self.assertTrue(STT._is_sung_or_annotated(cue), cue)
        self.assertFalse(STT._is_sung_or_annotated("Hello world"))
        self.assertFalse(STT._is_sung_or_annotated("[Music] Hello"))


class ProsodyTagTests(unittest.TestCase):
    def test_a_uniform_clip_is_not_tagged(self):
        # Every line at the same level: there is no contrast to carry, so no
        # line should be marked loud or quiet.
        self.assertEqual(SA.prosody_tags([-20.0] * 6), [""] * 6)

    def test_a_clear_outlier_is_tagged(self):
        levels = [-24.0, -24.5, -24.0, -23.5, -12.0, -24.0]
        tags = SA.prosody_tags(levels)
        self.assertEqual(tags[4], "[loud]")
        self.assertEqual(tags[:4], [""] * 4)

    def test_a_whispered_line_is_tagged_quiet(self):
        levels = [-20.0, -20.5, -20.0, -19.5, -30.0]
        tags = SA.prosody_tags(levels)
        self.assertEqual(tags[4], "[low volume]")

    def test_a_short_clip_is_left_alone(self):
        # Two lines are not a baseline; tagging from them would be a coin flip.
        self.assertEqual(SA.prosody_tags([-20.0, -5.0]), ["", ""])

    def test_a_missing_measurement_is_never_tagged(self):
        levels = [-20.0, None, -20.0, -6.0, -20.5]
        tags = SA.prosody_tags(levels)
        self.assertEqual(tags[1], "")
        self.assertEqual(tags[3], "[loud]")


class S2ProTagPlumbingTests(unittest.TestCase):
    """The tag reaches the request text, with the model itself stubbed out."""

    def _run(self, prosody):
        import tts_engines as T

        seen = {}

        class _Req:
            def __init__(self, **kw):
                seen.update(kw)

        class _RefAudio:
            def __init__(self, **kw):
                pass

        class _Chunk:
            code = "final"
            error = None
            audio = (44100, [0.1, 0.2, 0.1])

        class _Model:
            def inference(self, req):
                yield _Chunk()

        # fish_speech is an optional dependency, so the test supplies just the
        # two names synthesize imports from it.
        schema = types.ModuleType("fish_speech.utils.schema")
        schema.ServeTTSRequest = _Req
        schema.ServeReferenceAudio = _RefAudio
        pkg = types.ModuleType("fish_speech")
        utils = types.ModuleType("fish_speech.utils")
        saved = {k: sys.modules.get(k) for k in
                 ("fish_speech", "fish_speech.utils", "fish_speech.utils.schema")}
        sys.modules["fish_speech"] = pkg
        sys.modules["fish_speech.utils"] = utils
        sys.modules["fish_speech.utils.schema"] = schema
        try:
            eng = T.S2ProEngine.__new__(T.S2ProEngine)
            eng.seed = 0
            eng.load = lambda: _Model()
            eng._reference = lambda path, text: (b"", "ref")
            eng.synthesize("Hello", "ref.wav", "ref", prosody=prosody)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return seen.get("text", "")

    def test_the_tag_is_prepended_to_the_spoken_text(self):
        self.assertEqual(self._run("[loud]"), "[loud] Hello")

    def test_no_tag_leaves_the_text_untouched(self):
        self.assertEqual(self._run(""), "Hello")


if __name__ == "__main__":
    unittest.main()

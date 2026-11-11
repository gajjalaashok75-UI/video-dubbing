"""The dropped-clause bug in translation, and the chunk-aware safety net.

A hosted job dubbed "Hi, my name is John. Nice to meet you." and the second
sentence never reached the audio: NLLB, handed two sentences in one call,
translated the first and returned nothing for the rest. The old code split a
segment only above 600 characters, so a short two-sentence line went in whole
and lost its ending. Nothing downstream could tell — an empty chunk and a
genuinely short sentence look identical once the text is joined.

These tests drive `translate_segments` with a fake tokenizer and model, so the
model's habit of dropping a chunk is reproduced exactly and the retry that
recovers it is exercised, without loading 2.5 GB of weights.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))


def _install_torch_stub() -> None:
    """A minimal torch, enough to import translation_v2 without CUDA."""
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.cuda = _Cuda()
    sys.modules["torch"] = torch


_install_torch_stub()

import translation_v2  # noqa: E402


class FakeParam:
    device = types.SimpleNamespace(type="cpu")


class FakeModel:
    """
    A model that returns nothing for a marked chunk the first time it sees it.

    This is the failure in miniature: a chunk goes in and no text comes back,
    which the old code turned into a silent hole in the dub. The retry path is
    what makes the second attempt succeed — so the marked chunk must come back
    empty once and non-empty after, or the safety net would not be exercised.
    """

    def __init__(self, droppable: str):
        self.droppable = droppable
        self.calls: list[str] = []
        self._inputs: list[str] = []
        self._dropped: set[str] = set()

    def parameters(self):
        return iter([FakeParam()])

    def generate(self, **kwargs):
        out = []
        for text in self._inputs:
            self.calls.append(text)
            if self.droppable in text and text not in self._dropped:
                # First attempt at this chunk: the model returns nothing.
                self._dropped.add(text)
                out.append([])
            else:
                out.append([1])
        return out


class FakeInputs(dict):
    """What `tok(...)` returns: a mapping the model is called with, with .to()."""

    def to(self, _device):
        return self


class FakeTok:
    def __init__(self, model: FakeModel):
        self._model = model
        self.src_lang = None

    def __call__(self, text, return_tensors=None, padding=False,
                 truncation=False, max_length=None):
        texts = [text] if isinstance(text, str) else list(text)
        self._model._inputs = texts
        return FakeInputs(input_ids=texts)

    def convert_tokens_to_ids(self, token):
        return 0

    def batch_decode(self, generated, skip_special_tokens=True):
        # `generated` is the list of id-lists the model returned. An empty list
        # is the model returning nothing — decode it as the empty string, as a
        # real tokenizer would, so a dropped chunk stays dropped.
        return ["TR:" + text if ids else ""
                for text, ids in zip(self._model._inputs, generated)]


class ChunkingTests(unittest.TestCase):
    def test_a_short_two_sentence_segment_is_split(self):
        # The regression: this used to return one chunk, so the model saw both
        # sentences at once and could drop the second.
        self.assertEqual(
            translation_v2._split_into_chunks("Hi, my name is John. Nice to meet you."),
            ["Hi, my name is John.", "Nice to meet you."])

    def test_every_sentence_gets_its_own_chunk(self):
        text = "One. Two! Three? Four."
        self.assertEqual(translation_v2._split_into_chunks(text),
                         ["One.", "Two!", "Three?", "Four."])

    def test_an_over_long_sentence_is_split_on_commas(self):
        sentence = ", ".join(f"clause number {n}" for n in range(80)) + "."
        chunks = translation_v2._split_into_chunks(sentence)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), translation_v2._MAX_CHUNK_CHARS + 1)

    def test_a_many_sentence_segment_is_not_truncated(self):
        # The guard must not bind on real content: truncating here would drop
        # the tail of the line, the very failure being fixed.
        text = " ".join(f"Sentence number {n}." for n in range(120))
        chunks = translation_v2._split_into_chunks(text)
        self.assertEqual(len(chunks), 120)

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(translation_v2._split_into_chunks("   "), [])


class DroppedChunkTests(unittest.TestCase):
    def _run(self, segments, droppable):
        model = FakeModel(droppable)
        tok = FakeTok(model)
        translation_v2._tokenizer = tok
        translation_v2._model = model
        # Skip the language-code lookup; identity codes keep src != tgt.
        original = translation_v2._langs_flores
        translation_v2._langs_flores = lambda s, t, for_speech=True: ("eng_Latn", "hin_Deva")
        try:
            return translation_v2.translate_segments(segments, "en", "hi"), model
        finally:
            translation_v2._langs_flores = original
            translation_v2._tokenizer = None
            translation_v2._model = None

    def test_a_dropped_second_sentence_is_recovered(self):
        # Both sentences must appear in the output, in order.
        segments = [{"text": "Hi, my name is John. Nice to meet you."}]
        out, model = self._run(segments, droppable="Nice to meet you")
        self.assertEqual(len(out), 1)
        self.assertIn("Hi, my name is John.", out[0])
        self.assertIn("Nice to meet you.", out[0])
        # The retry actually happened: the dropped sentence was re-sent alone
        # after the first call returned nothing for it.
        self.assertEqual(model.calls.count("Nice to meet you."), 2)

    def test_output_stays_aligned_with_the_segments(self):
        segments = [{"text": "First line."}, {"text": ""},
                    {"text": "Second. Third."}]
        out, _ = self._run(segments, droppable="nothing-matches")
        self.assertEqual(len(out), len(segments))
        self.assertEqual(out[1], "")
        self.assertIn("Second.", out[2])
        self.assertIn("Third.", out[2])

    def test_a_segment_the_model_refuses_in_pieces_is_translated_whole(self):
        # Every chunk comes back empty; the whole-segment fallback must fire so
        # the line is not left silent.
        model = FakeModel("__never__")

        class AlwaysEmpty(FakeTok):
            def batch_decode(self, generated, skip_special_tokens=True):
                return [""] * len(generated)

        tok = AlwaysEmpty(model)
        translation_v2._tokenizer = tok
        translation_v2._model = model
        original = translation_v2._langs_flores
        translation_v2._langs_flores = lambda s, t, for_speech=True: ("eng_Latn", "hin_Deva")
        try:
            out = translation_v2.translate_segments(
                [{"text": "A line."}], "en", "hi")
        finally:
            translation_v2._langs_flores = original
            translation_v2._tokenizer = None
            translation_v2._model = None
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()

from dataclasses import fields, is_dataclass

import pytest

from tts.data import text as text_mod
from tts.data.text import Phonemizer, Text, get_phonemizer


@pytest.fixture(scope="module")
def phonemizer() -> Phonemizer:
    return get_phonemizer()


def test_text_is_dataclass_with_expected_fields() -> None:
    assert is_dataclass(Text)
    names = {f.name for f in fields(Text)}
    assert names == {"text"}


def test_get_phonemizer_is_cached() -> None:
    assert get_phonemizer() is get_phonemizer()


def test_phonemizer_returns_str_and_list(phonemizer: Phonemizer) -> None:
    phonemes, tokens = phonemizer.phonemize("hello world")
    assert isinstance(phonemes, str)
    assert phonemes
    assert isinstance(tokens, list)
    assert tokens


def test_phonemizer_call_matches_phonemize(phonemizer: Phonemizer) -> None:
    assert phonemizer("hello") == phonemizer.phonemize("hello")


def test_phonemizer_is_deterministic(phonemizer: Phonemizer) -> None:
    a, _ = phonemizer.phonemize("the quick brown fox")
    b, _ = phonemizer.phonemize("the quick brown fox")
    assert a == b


def test_text_phonemes_returns_non_empty_string(phonemizer: Phonemizer) -> None:
    t = Text(text="hello")
    assert isinstance(t.phonemes, str)
    assert t.phonemes


def test_text_tokens_returns_list(phonemizer: Phonemizer) -> None:
    t = Text(text="hello world")
    assert isinstance(t.tokens, list)
    assert len(t.tokens) > 0


def test_text_phonemes_matches_phonemizer(phonemizer: Phonemizer) -> None:
    t = Text(text="hello")
    expected_phonemes, _ = phonemizer.phonemize("hello")
    assert t.phonemes == expected_phonemes


def test_text_phonemes_tokens_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class FakePhonemizer:
        def phonemize(self, text: str):
            calls["n"] += 1
            return ("FAKE", [text])

    monkeypatch.setattr(text_mod, "get_phonemizer", lambda: FakePhonemizer())

    t = Text(text="hello")
    assert t.phonemes == "FAKE"
    assert t.tokens == ["hello"]
    assert t.phonemes == "FAKE"
    assert calls["n"] == 1


def test_text_instances_have_independent_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhonemizer:
        def phonemize(self, text: str):
            return (text.upper(), list(text))

    monkeypatch.setattr(text_mod, "get_phonemizer", lambda: FakePhonemizer())

    t1 = Text(text="abc")
    t2 = Text(text="xyz")
    assert t1.phonemes == "ABC"
    assert t2.phonemes == "XYZ"
    assert t1.tokens == ["a", "b", "c"]
    assert t2.tokens == ["x", "y", "z"]

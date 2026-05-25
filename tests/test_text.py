import json
from dataclasses import fields, is_dataclass

import pytest
from misaki.token import MToken

from jwt.data import text as text_mod
from jwt.data.text import Phonemizer, Text, Tokenizer, Vocabulary, get_phonemizer


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


def test_text_word_tokens_returns_list_of_mtokens(phonemizer: Phonemizer) -> None:
    t = Text(text="hello world")
    assert isinstance(t.word_tokens, list)
    assert t.word_tokens
    assert all(isinstance(tok, MToken) for tok in t.word_tokens)


def test_text_word_tokens_matches_phonemizer(phonemizer: Phonemizer) -> None:
    t = Text(text="hello")
    _, expected_tokens = phonemizer.phonemize("hello")
    assert t.word_tokens == expected_tokens


def test_text_tokens_uses_tokenizer_to_encode_text() -> None:
    vocab = Vocabulary({"h": 0, "i": 1})
    tokenizer = Tokenizer(vocab)
    t = Text(text="hi", tokenizer=tokenizer)
    assert t.tokens == [0, 1]


def test_text_tokens_raises_value_error_when_tokenizer_is_none() -> None:
    t = Text(text="hello")
    with pytest.raises(ValueError, match="tokenizer should not be None"):
        _ = t.tokens


def test_vocabulary_is_dict_subclass() -> None:
    v = Vocabulary({"a": 1, "b": 2})
    assert isinstance(v, dict)
    assert v["a"] == 1
    assert v["b"] == 2
    assert len(v) == 2


def test_vocabulary_to_json_writes_mapping(tmp_path) -> None:
    path = tmp_path / "vocab.json"
    Vocabulary({"a": 1, "b": 2}).to_json(str(path))
    with open(path) as f:
        assert json.load(f) == {"a": 1, "b": 2}


def test_vocabulary_from_json_loads_mapping(tmp_path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"x": 0, "y": 1}))
    v = Vocabulary.from_json(str(path))
    assert isinstance(v, Vocabulary)
    assert v == {"x": 0, "y": 1}


def test_vocabulary_json_roundtrip_preserves_unicode_phonemes(tmp_path) -> None:
    path = tmp_path / "vocab.json"
    original = Vocabulary({"ə": 0, "ʃ": 1, "θ": 2})
    original.to_json(str(path))
    loaded = Vocabulary.from_json(str(path))
    assert loaded == original


def test_tokenizer_encode_maps_characters_to_ids() -> None:
    tokenizer = Tokenizer(Vocabulary({"a": 0, "b": 1, "c": 2}))
    assert tokenizer.encode("abc") == [0, 1, 2]
    assert tokenizer.encode("cab") == [2, 0, 1]


def test_tokenizer_call_matches_encode() -> None:
    tokenizer = Tokenizer(Vocabulary({"a": 0, "b": 1}))
    assert tokenizer("ab") == tokenizer.encode("ab")


def test_tokenizer_encode_empty_input_returns_empty_list() -> None:
    tokenizer = Tokenizer(Vocabulary({"a": 0}))
    assert tokenizer.encode("") == []


def test_tokenizer_encode_unknown_symbol_raises_key_error() -> None:
    tokenizer = Tokenizer(Vocabulary({"a": 0}))
    with pytest.raises(KeyError):
        tokenizer.encode("z")

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest
import torch

from jwt.data.source import (
    ArrowTTSSource,
    LJTTSSource,
    Word,
    check_same_sample_rate,
)
from jwt.data.text import Tokenizer, Vocabulary

TOKENIZER = Tokenizer(Vocabulary({"a": 0, "b": 1, " ": 2}))


def test_arrow_source_concatenates_shards(make_prepared: Callable[..., Path]) -> None:
    """All shards form one source with its light columns in memory."""
    folder = make_prepared(
        "d",
        [
            ("u0", "d/a", "s", 0, 2),
            ("u1", "d/a", "s", 1, 3),
            ("u2", "d/b", None, None, 4),
        ],
    )
    source = ArrowTTSSource(str(folder), TOKENIZER, patch_size=10)
    assert len(source) == 3
    assert source.utt_ids == ["u0", "u1", "u2"]
    assert source.speakers == ["d/a", "d/a", "d/b"]
    assert source.sessions == ["s", "s", None]
    assert source.session_idxs == [0, 1, None]
    assert source.durations == [1.0, 1.5, 2.0]
    assert (source.sample_rate, source.hop_length) == (100, 10)
    # u1 has 3 words, one every 0.5 s: 0.3 s spoken, "word " and "ab " each.
    assert source.words(1) == [
        Word(start=0.0, end=pytest.approx(0.3), text_start=0, phoneme_start=0),
        Word(start=0.5, end=pytest.approx(0.8), text_start=5, phoneme_start=3),
        Word(start=1.0, end=pytest.approx(1.3), text_start=10, phoneme_start=6),
    ]


def test_arrow_source_item_uses_stored_phonemes(
    make_prepared: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Items use stored phonemes and patch the waveform into frames."""

    def boom() -> NoReturn:
        raise AssertionError("G2P must not run")

    monkeypatch.setattr("jwt.data.text.text.get_phonemizer", boom)
    utts = [
        ("u0", "d/a", "s", 0, 1),
        ("u1", "d/a", "s", 1, 2),
        ("u2", "d/a", "s", 2, 3),
    ]
    audio, text = ArrowTTSSource(
        str(make_prepared("d", utts)), TOKENIZER, patch_size=10
    )[2]
    assert text.tokens == [0, 1, 2, 0, 1, 2, 0, 1]
    assert audio.acoustic is not None and audio.acoustic.shape == (1, 10, 15)
    assert audio.waveform.shape == (1, 150)
    assert audio.waveform[0, 7].item() == 7 / 32768.0
    assert torch.equal(audio.acoustic[0, :, 1], audio.waveform[0, 10:20])


def test_arrow_source_requires_shards(make_prepared: Callable[..., Path]) -> None:
    """A directory without shards fails at construction."""
    folder = make_prepared("d", [("u0", "d/a", None, None, 1)])
    for path in folder.glob("*.arrow"):
        path.unlink()
    with pytest.raises(ValueError, match="no arrow shards"):
        ArrowTTSSource(str(folder), tokenizer=None, patch_size=10)


def test_check_same_sample_rate_rejects_mismatch(
    make_prepared: Callable[..., Path],
) -> None:
    """Mixing sample rates fails and names the offending dataset."""
    a = ArrowTTSSource(str(make_prepared("a", [("u", "a/x", None, None, 1)])), None, 10)
    b = ArrowTTSSource(str(make_prepared("b", [("u", "b/x", None, None, 1)])), None, 10)
    check_same_sample_rate([a, b], ["path_a", "path_b"])
    b.meta.sample_rate = 200
    with pytest.raises(ValueError, match="path_b"):
        check_same_sample_rate([a, b], ["path_a", "path_b"])


def test_arrow_source_pads_last_patch(make_prepared: Callable[..., Path]) -> None:
    """Patches come from reshaping the waveform; a partial last patch is zero-padded."""
    source = ArrowTTSSource(
        str(make_prepared("d", [("u", "d/x", None, None, 1)])), None, 20
    )
    audio, _ = source[0]
    assert audio.acoustic is not None and audio.acoustic.shape == (1, 20, 3)
    assert (
        audio.acoustic[0, :, 2].tolist()
        == [i / 32768.0 for i in range(40, 50)] + [0.0] * 10
    )


def test_lj_source_works_without_a_tokenizer(tmp_path: Path) -> None:
    """The LJSpeech reader for create_arrow.py uses LJTTSSource before any
    vocabulary exists — construction and item access must not require one."""
    (tmp_path / "metadata.csv").write_text(
        "LJ001-0001|raw text|normalized text\n", encoding="utf-8"
    )
    source = LJTTSSource(str(tmp_path))
    assert len(source) == 1
    _, text = source[0]
    assert text.tokenizer is None
    assert text.text == "normalized text"


def test_lj_source_threads_sample_rate_to_audio(tmp_path: Path) -> None:
    (tmp_path / "metadata.csv").write_text(
        "LJ001-0001|raw text|normalized text\n", encoding="utf-8"
    )
    source = LJTTSSource(str(tmp_path), sample_rate=24000)
    audio, _ = source[0]
    assert audio.sample_rate == 24000

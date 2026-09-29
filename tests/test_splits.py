import random

import pytest

from jwt.data.splits import split_indices


def speakers_and_ids(
    n_speakers: int = 6, per_speaker: int = 10
) -> tuple[list[str], list[str]]:
    speakers = [f"d/s{i // per_speaker}" for i in range(n_speakers * per_speaker)]
    utt_ids = [f"u{i}" for i in range(len(speakers))]
    return speakers, utt_ids


def test_split_sizes_and_disjoint() -> None:
    """Splits have the requested sizes and unseen speakers never train."""
    speakers, utt_ids = speakers_and_ids()
    train, valid, unseen = split_indices(
        speakers, utt_ids, n_valid=5, n_valid_speakers=2
    )
    assert (len(train), len(valid), len(unseen)) == (35, 5, 20)
    assert not set(train) & set(valid) and not set(train) & set(unseen)
    assert not {speakers[i] for i in unseen} & {speakers[i] for i in train + valid}


def test_split_independent_of_row_order() -> None:
    """The split depends on ids, not on row order."""
    speakers, utt_ids = speakers_and_ids()
    order = list(range(len(speakers)))
    random.Random(0).shuffle(order)
    speakers2, utt_ids2 = [speakers[i] for i in order], [utt_ids[i] for i in order]
    a = split_indices(speakers, utt_ids, 5, 2)
    b = split_indices(speakers2, utt_ids2, 5, 2)
    for x, y in zip(a, b, strict=True):
        assert {utt_ids[i] for i in x} == {utt_ids2[i] for i in y}


def test_split_rejects_too_many_valid_speakers() -> None:
    """Holding out every speaker fails with a clear error."""
    speakers, utt_ids = speakers_and_ids(n_speakers=1)
    with pytest.raises(ValueError, match="n_valid_speakers"):
        split_indices(speakers, utt_ids, n_valid=2, n_valid_speakers=1)


def test_split_rejects_empty_train() -> None:
    """Holding out every utterance fails with a clear error."""
    speakers, utt_ids = speakers_and_ids(n_speakers=1, per_speaker=3)
    with pytest.raises(ValueError, match="no training"):
        split_indices(speakers, utt_ids, n_valid=3, n_valid_speakers=0)

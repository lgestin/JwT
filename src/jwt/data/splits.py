"""Deterministic per-dataset train / valid / valid_unseen splits."""

import zlib


def crc(s: str) -> int:
    return zlib.crc32(s.encode())


def split_indices(
    speakers: list[str], utt_ids: list[str], n_valid: int, n_valid_speakers: int
) -> tuple[list[int], list[int], list[int]]:
    """(train, valid, valid_unseen) indices, chosen by crc32 of speaker / utt_id."""
    distinct = sorted(set(speakers), key=crc)
    if n_valid_speakers and n_valid_speakers >= len(distinct):
        raise ValueError(
            f"n_valid_speakers={n_valid_speakers} leaves no training speaker "
            f"({len(distinct)} speakers)"
        )
    unseen = set(distinct[:n_valid_speakers])
    valid_unseen = [i for i, s in enumerate(speakers) if s in unseen]
    seen = sorted(
        (i for i, s in enumerate(speakers) if s not in unseen),
        key=lambda i: crc(utt_ids[i]),
    )
    valid, train = sorted(seen[:n_valid]), sorted(seen[n_valid:])
    if not train:
        raise ValueError(f"no training utterances left after n_valid={n_valid}")
    return train, valid, valid_unseen

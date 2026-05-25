import torch
import torch.nn.functional as F

from jwt.data.dataset import Batch, Sample


def pad_sequences_longest(
    sequences: list[torch.Tensor], mode="constant"
) -> tuple[list[torch.Tensor], list[int]]:
    lengths = [seq.size(-1) for seq in sequences]
    max_length = max(lengths)
    padded_sequences = [F.pad(seq, (0, max_length - seq.size(-1)), mode=mode) for seq in sequences]
    return padded_sequences, lengths


def mask_from_lengths(lengths: list[int]) -> torch.BoolTensor:
    arange = torch.arange(max(lengths)).unsqueeze(0).repeat(len(lengths), 1)
    mask = arange < torch.tensor(lengths).unsqueeze(1)
    return mask  # ty: ignore[invalid-return-type]


def collate(samples: list[Sample]) -> Batch:
    idxs = [sample.idx for sample in samples]
    audios = [sample.audio for sample in samples]
    acoustics = [audio.acoustic for audio in audios]
    padded_acoustics, acoustic_lengths = pad_sequences_longest(acoustics)  # ty: ignore[invalid-argument-type]
    stacked_acoustic = torch.stack(padded_acoustics)
    acoustic_mask = mask_from_lengths(acoustic_lengths)
    tokens = [torch.tensor(sample.text.tokens) for sample in samples]
    padded_tokens, tokens_lengths = pad_sequences_longest(tokens)
    stacked_tokens = torch.stack(padded_tokens).long()
    tokens_mask = mask_from_lengths(tokens_lengths)
    return Batch(
        idxs=idxs,
        audios=audios,
        acoustic=stacked_acoustic,  # ty: ignore[invalid-argument-type]
        acoustic_mask=acoustic_mask,
        tokens=stacked_tokens,  # ty: ignore[invalid-argument-type]
        tokens_mask=tokens_mask,
    )

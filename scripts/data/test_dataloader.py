"""Smoke test: load a few batches of a prepared dataset (argv[1], raw512)."""

import sys

from torch.utils.data import DataLoader

from jwt.data.collate import collate
from jwt.data.dataset import AudioDataset
from jwt.data.source import ArrowTTSSource
from jwt.data.text import Tokenizer, Vocabulary


def main() -> None:
    vocab = Vocabulary.from_json("data/vocabulary.json")
    tokenizer = Tokenizer(vocab)
    source = ArrowTTSSource(sys.argv[1], tokenizer, patch_size=512)
    print(f"Source size: {len(source)}")

    dataset = AudioDataset(tts_source=source, sample_rate=source.sample_rate)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    for i, batch in enumerate(loader):
        print(
            f"batch {i}: idxs={batch.idxs} "
            f"acoustic={tuple(batch.acoustic.shape)} "
            f"tokens={tuple(batch.tokens.shape)}"
        )
        if i >= 2:
            break


if __name__ == "__main__":
    main()

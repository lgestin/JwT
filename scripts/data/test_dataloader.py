"""Smoke test: load a few batches of LJSpeech via AudioDataset + DataLoader."""

from torch.utils.data import DataLoader

from jwt.data.collate import collate
from jwt.data.dataset import AudioDataset
from jwt.data.source import ArrowTTSSource
from jwt.data.text import Tokenizer, Vocabulary


def main() -> None:
    vocab = Vocabulary.from_json("data/vocabulary.json")
    tokenizer = Tokenizer(vocab)
    source = ArrowTTSSource("data/ljspeech_24khz.arrow", tokenizer=tokenizer)
    print(f"Source size: {len(source)}")

    dataset = AudioDataset(tts_source=source, sample_rate=24000)
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
            f"mels={tuple(batch.mels.shape)} dtype={batch.mels.dtype} "
            f"tokens={tuple(batch.tokens.shape)} dtype={batch.tokens.dtype}"
        )
        if i >= 2:
            break


if __name__ == "__main__":
    main()

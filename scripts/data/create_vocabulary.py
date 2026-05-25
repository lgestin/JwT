"""Compute a phoneme vocabulary from a TTS source and save it as JSON."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rich.progress import track
from simple_parsing import ArgumentParser

from jwt.data.source import LJTTSSource
from jwt.data.text import Vocabulary


@dataclass
class Args:
    source_path: Path  # Root folder of the TTS source (e.g. data/LJSpeech-1.1).
    output: Path  # Destination JSON path for the vocabulary.


def main(args: Args) -> None:
    source = LJTTSSource(str(args.source_path))
    print(f"Loaded {len(source)} items from {args.source_path}")

    counter: Counter[str] = Counter()
    for i in track(range(len(source)), description="Phonemizing"):
        _, text = source[i]
        counter.update(text.phonemes)

    symbols = sorted(counter)
    vocab = Vocabulary({s: i for i, s in enumerate(symbols)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocab.to_json(str(args.output))

    print(f"Wrote {len(vocab)} symbols to {args.output}")
    for s, c in counter.most_common():
        print(f"  {s!r}: {c}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)

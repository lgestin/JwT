"""Compute the phoneme vocabulary of prepared datasets and save it as JSON."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from simple_parsing import ArgumentParser

from jwt.data.prepared import shard_paths
from jwt.data.text import Vocabulary


@dataclass
class Args:
    dataset_paths: list[Path]  # Prepared dataset directories (create_arrow.py).
    output: Path  # Destination JSON path for the vocabulary.


def main(args: Args) -> Vocabulary:
    counter: Counter[str] = Counter()
    for folder in args.dataset_paths:
        for path in shard_paths(folder):
            table = pa.ipc.open_file(pa.memory_map(str(path), "r")).read_all()
            for phonemes in table["phonemes"].to_pylist():
                counter.update(phonemes)

    vocab = Vocabulary({s: i for i, s in enumerate(sorted(counter))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocab.to_json(str(args.output))

    print(f"Wrote {len(vocab)} symbols to {args.output}")
    for s, c in counter.most_common():
        print(f"  {s!r}: {c}")
    return vocab


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)

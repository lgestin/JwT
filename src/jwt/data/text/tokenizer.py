import json
from typing import Self


class Vocabulary(dict[str, int]):
    @classmethod
    def from_json(cls, path: str) -> Self:
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self, f)
        return


class Tokenizer:
    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary

    def encode(self, phonemes: str) -> list[int]:
        return [self.vocabulary[p] for p in phonemes]

    def __call__(self, phonemes: str) -> list[int]:
        return self.encode(phonemes)

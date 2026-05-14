from dataclasses import dataclass
from functools import cache, cached_property

from misaki import en, espeak
from misaki.token import MToken


@dataclass
class Text:
    text: str

    @cached_property
    def phonemes_tokens(self) -> tuple[str, list[MToken]]:
        return get_phonemizer().phonemize(self.text)

    @property
    def phonemes(self) -> str:
        return self.phonemes_tokens[0]

    @property
    def word_tokens(self) -> list[MToken]:
        return self.phonemes_tokens[1]


class Phonemizer:
    def __init__(self):

        fallback = espeak.EspeakFallback(british=False)
        self.g2p = en.G2P(trf=False, british=False, fallback=fallback)

    def phonemize(self, text: str):
        return self.g2p(text)

    def __call__(self, text: str):
        return self.phonemize(text)


@cache
def get_phonemizer() -> Phonemizer:
    phonemizer = Phonemizer()
    return phonemizer

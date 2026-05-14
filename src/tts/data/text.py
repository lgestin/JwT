from dataclasses import dataclass
from functools import cache, cached_property


@dataclass
class Text:
    text: str

    @cached_property
    def phonemes_tokens(self) -> tuple[str, list[int]]:
        return get_phonemizer().phonemize(self.text)

    @property
    def phonemes(self):
        return self.phonemes_tokens[0]

    @property
    def tokens(self):
        return self.phonemes_tokens[1]


class Phonemizer:
    def __init__(self):
        from misaki import en, espeak

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

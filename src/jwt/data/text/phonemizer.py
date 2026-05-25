from functools import cache

from misaki import en, espeak
from misaki.token import MToken


class Phonemizer:
    def __init__(self):

        fallback = espeak.EspeakFallback(british=False)
        self.g2p = en.G2P(trf=False, british=False, fallback=fallback)

    def phonemize(self, text: str) -> tuple[str, list[MToken]]:
        return self.g2p(text)

    def __call__(self, text: str):
        return self.phonemize(text)


@cache
def get_phonemizer() -> Phonemizer:
    phonemizer = Phonemizer()
    return phonemizer

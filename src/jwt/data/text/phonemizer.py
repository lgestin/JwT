import threading
from functools import cache

from misaki import en, espeak
from misaki.token import MToken


class Phonemizer:
    def __init__(self) -> None:

        fallback = espeak.EspeakFallback(british=False)
        self.g2p = en.G2P(trf=False, british=False, fallback=fallback)
        # One G2P call at a time: the espeak fallback wraps a single, non-thread-
        # safe C library, so concurrent calls swap phonemes between texts silently
        # or raise a line-count mismatch. G2P is GIL-bound, so this costs little.
        self.lock = threading.Lock()

    def phonemize(self, text: str) -> tuple[str, list[MToken]]:
        with self.lock:
            return self.g2p(text)

    def __call__(self, text: str) -> tuple[str, list[MToken]]:
        return self.phonemize(text)


@cache
def get_phonemizer() -> Phonemizer:
    phonemizer = Phonemizer()
    return phonemizer

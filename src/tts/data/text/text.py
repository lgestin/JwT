from dataclasses import dataclass
from functools import cached_property

from misaki.token import MToken

from tts.data.text.phonemizer import get_phonemizer
from tts.data.text.tokenizer import Tokenizer


@dataclass
class Text:
    text: str
    tokenizer: Tokenizer | None = None

    @cached_property
    def phonemes_tokens(self) -> tuple[str, list[MToken]]:
        return get_phonemizer().phonemize(self.text)

    @property
    def phonemes(self) -> str:
        return self.phonemes_tokens[0]

    @property
    def word_tokens(self) -> list[MToken]:
        return self.phonemes_tokens[1]

    @property
    def tokens(self) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("self.tokenizer should not be None")
        return self.tokenizer.encode(self.phonemes)

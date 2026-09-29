"""Word alignment of misaki word tokens; aligners are selected via `Aligners`."""

import re
from enum import StrEnum
from itertools import accumulate, chain, pairwise
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
import torchaudio
from misaki.token import MToken
from torchaudio.models import Wav2Vec2Model

MMS_SAMPLE_RATE = 16_000


@runtime_checkable
class Aligner(Protocol):
    def align(
        self,
        waveforms: list[torch.Tensor],
        sample_rate: int,
        tokens: list[list[MToken]],
    ) -> list[list[tuple[float, float]] | None]:
        """Per utterance, (start, end) seconds per token; None if unalignable."""
        ...


def piece_starts(pieces: list[str], full: str) -> list[int] | None:
    """Start offset of each piece in `full`; None if the pieces don't rebuild it."""
    if "".join(pieces) != full:
        return None
    return list(accumulate((len(p) for p in pieces), initial=0))[:-1]


def word_phoneme_starts(tokens: list[MToken], phonemes: str) -> list[int] | None:
    """Start index of each token in `phonemes`; None if tokens don't rebuild it."""
    return piece_starts(
        [(tok.phonemes or "") + tok.whitespace for tok in tokens], phonemes
    )


def word_text_starts(tokens: list[MToken], text: str) -> list[int] | None:
    """Start index of each token in `text`; None if tokens don't rebuild it."""
    return piece_starts([tok.text + tok.whitespace for tok in tokens], text)


def mms_word(token_text: str) -> str | None:
    """MMS_FA word: letters/apostrophes, star for digits, None for punctuation."""
    word = re.sub(r"[^a-z']", "", token_text.lower())
    if word:
        return word
    return "*" if re.search(r"\d", token_text) else None


class MMSFAligner(Aligner):
    """torchaudio MMS_FA (wav2vec2 CTC) forced alignment; provisional baseline.

    torchaudio's wrapper only handles batch size 1, so batches call the inner
    model directly: rows are normalized one by one and the star column is added
    by hand.
    """

    def __init__(self, device: str = "cuda") -> None:
        bundle = torchaudio.pipelines.MMS_FA
        self.device = torch.device(device)
        model = bundle.get_model(with_star=True).model  # inner model, no wrapper
        assert isinstance(model, Wav2Vec2Model)
        self.model = model.to(self.device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    @torch.inference_mode()
    def align(
        self,
        waveforms: list[torch.Tensor],
        sample_rate: int,
        tokens: list[list[MToken]],
    ) -> list[list[tuple[float, float]] | None]:
        """Punctuation tokens get a zero-length span at the previous word's end."""
        wavs = [
            torchaudio.functional.resample(w[0].float(), sample_rate, MMS_SAMPLE_RATE)
            for w in waveforms
        ]
        lengths = torch.tensor([len(w) for w in wavs])
        batch = torch.zeros(len(wavs), int(lengths.max()))
        for i, w in enumerate(wavs):
            batch[i, : len(w)] = F.layer_norm(w, w.shape)
        cuda = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=cuda):
            emission, frames = self.model(
                batch.to(self.device), lengths.to(self.device)
            )
        assert frames is not None
        emission = emission.float().log_softmax(-1).cpu()  # CPU forced_align is faster
        emission = torch.cat((emission, torch.zeros_like(emission[..., :1])), dim=-1)
        rows = zip(wavs, frames.tolist(), tokens, strict=True)
        all_spans = []
        for i, (wav, n, toks) in enumerate(rows):
            sec_per_frame = len(wav) / n / MMS_SAMPLE_RATE
            all_spans.append(self.token_spans(emission[i, :n], sec_per_frame, toks))
        return all_spans

    def token_spans(
        self, emission: torch.Tensor, sec_per_frame: float, tokens: list[MToken]
    ) -> list[tuple[float, float]] | None:
        words = [mms_word(tok.text) for tok in tokens]
        spoken = [word for word in words if word is not None]
        if not spoken:
            return None
        token_ids = self.tokenizer(spoken)  # ids, despite torchaudio's str annotation
        ids = list(chain.from_iterable(token_ids))
        # CTC needs a frame per token, plus a blank between repeated tokens.
        if len(ids) + sum(a == b for a, b in pairwise(ids)) > len(emission):
            return None
        word_spans = iter(self.aligner(emission, token_ids))  # ty: ignore[invalid-argument-type]
        spans, end = [], 0.0
        for word in words:
            start = end
            if word is not None:
                span = next(word_spans)
                start, end = span[0].start * sec_per_frame, span[-1].end * sec_per_frame
            spans.append((start, end))
        return spans


class Aligners(StrEnum):
    MMS_FA = "MMS_FA"

    def aligner(self, device: str) -> Aligner:
        match self:
            case Aligners.MMS_FA:
                return MMSFAligner(device)

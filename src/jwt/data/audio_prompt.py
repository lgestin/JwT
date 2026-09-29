"""Audio prompt (voice reference) selection from a source's light columns."""

import random
from collections import defaultdict
from dataclasses import dataclass

import torch

from jwt.data.audio import Audio
from jwt.data.source import ArrowTTSSource
from jwt.data.text import Text


@dataclass
class AudioPromptConfig:
    p_drop: float = 0.2  # no audio prompt (unconditional branch for CFG)
    p_other: float = 0.5  # prompt from another utterance, else a same-utterance cut
    min_prompt_s: float = 1.0
    max_prompt_s: float = 5.0
    min_target_s: float = 1.0
    max_trim_s: float = 0.5  # random trim off the end of a cut prompt
    min_pause_s: float = 0.15  # cut only where the target starts after a pause


@dataclass(frozen=True)
class Cut:
    """Prompt is this utterance before `word`; the target starts at `word`."""

    word: int


@dataclass(frozen=True)
class OtherUtterance:
    """Prompt is a window of row `ref_idx`, another clip of the same speaker."""

    ref_idx: int


class AudioPromptSampler:
    """Chooses no prompt, a word-boundary cut, or another utterance of the speaker."""

    def __init__(
        self,
        source: ArrowTTSSource,
        cfg: AudioPromptConfig,
        indices: list[int] | None = None,
    ):
        """indices: the split to draw other-utterance prompts from (default: all)."""
        self.source = source
        self.cfg = cfg
        self.by_speaker: dict[str, list[int]] = defaultdict(list)
        # (index, position in session) per (speaker, session)
        self.by_session: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(
            list
        )
        for i in range(len(source)) if indices is None else indices:
            speaker, session = source.speakers[i], source.sessions[i]
            pos = source.session_idxs[i]
            self.by_speaker[speaker].append(i)
            if session is not None and pos is not None:
                self.by_session[(speaker, session)].append((i, pos))

    def __call__(self, idx: int, rng: random.Random) -> Cut | OtherUtterance | None:
        if rng.random() < self.cfg.p_drop:
            return None
        if rng.random() < self.cfg.p_other:
            ref = self.other_utterance(idx, rng)
            if ref is not None:
                return OtherUtterance(ref)
        return self.word_cut(idx, rng)

    def other_utterance(self, idx: int, rng: random.Random) -> int | None:
        """Same session away from the neighbours, else any other speaker utterance."""
        speaker, session = self.source.speakers[idx], self.source.sessions[idx]
        pos = self.source.session_idxs[idx]
        if session is not None and pos is not None:
            far = [
                j for j, p in self.by_session[(speaker, session)] if abs(p - pos) > 1
            ]
            if far:
                return rng.choice(far)
        pool = self.by_speaker[speaker]
        if len(pool) < 2:
            return None
        while (j := rng.choice(pool)) == idx:
            pass
        return j

    def word_cut(self, idx: int, rng: random.Random) -> Cut | None:
        cfg = self.cfg
        words = self.source.words(idx)
        duration = self.source.durations[idx]

        def is_valid_cut(k: int) -> bool:
            start = words[k].start
            return (
                cfg.min_prompt_s <= start <= cfg.max_prompt_s
                and duration - start >= cfg.min_target_s
                and start - words[k - 1].end >= cfg.min_pause_s
            )

        candidates = list(filter(is_valid_cut, range(1, len(words))))
        return Cut(rng.choice(candidates)) if candidates else None

    def apply(
        self, idx: int, audio: Audio, text: Text, rng: random.Random
    ) -> tuple[Audio, Text, torch.Tensor]:
        """Audio prompt frames for `idx`, with audio and text cut to the target."""
        acoustic = audio.acoustic
        assert acoustic is not None
        choice = self(idx, rng)
        if isinstance(choice, OtherUtterance):
            ref = self.source[choice.ref_idx][0].acoustic
            assert ref is not None
            n = self.frames(self.cfg.max_prompt_s, audio.sample_rate)
            start = rng.randint(0, max(0, ref.shape[-1] - n))
            return audio, text, ref[..., start : start + n]
        if isinstance(choice, Cut):
            return self.split_at(idx, audio, text, choice.word, rng)
        return audio, text, acoustic[..., :0]

    def frames(self, seconds: float, sample_rate: int) -> int:
        return int(seconds * sample_rate / self.source.hop_length)

    def split_at(
        self, idx: int, audio: Audio, text: Text, word: int, rng: random.Random
    ) -> tuple[Audio, Text, torch.Tensor]:
        """Prompt before `word` (minus a random trim), target from `word` on."""
        acoustic = audio.acoustic
        assert acoustic is not None
        first = self.source.words(idx)[word]  # first word of the target
        hop = self.source.hop_length
        f = int(first.start * audio.sample_rate / hop)
        trim = rng.randint(0, self.frames(self.cfg.max_trim_s, audio.sample_rate))
        target = Audio(
            waveform=audio.waveform[..., f * hop :],  # ty: ignore[invalid-argument-type]
            sample_rate=audio.sample_rate,
            loudness=audio.loudness,
            acoustic=acoustic[..., f:],  # ty: ignore[invalid-argument-type]
        )
        text = Text(
            text=text.text[first.text_start :],
            tokenizer=text.tokenizer,
            stored_phonemes=text.phonemes[first.phoneme_start :],
        )
        return target, text, acoustic[..., : max(f - trim, 0)]

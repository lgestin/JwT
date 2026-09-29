import random
from dataclasses import dataclass, fields
from typing import Self

import torch
from torch.utils.data import Dataset

from jwt.data.audio import Audio, AudioFile
from jwt.data.audio_prompt import AudioPromptConfig, AudioPromptSampler
from jwt.data.source import ArrowTTSSource, TTSSource
from jwt.data.text import Text


@dataclass
class Sample:
    idx: int
    audio: Audio
    text: Text
    audio_prompt: torch.Tensor | None = None


@dataclass
class Batch:
    idxs: list[int]
    audios: list[Audio]
    acoustic: torch.FloatTensor
    acoustic_mask: torch.BoolTensor
    tokens: torch.LongTensor
    tokens_mask: torch.BoolTensor
    audio_prompt: torch.Tensor | None = None
    audio_prompt_mask: torch.BoolTensor | None = None

    def to(self, device: str | torch.device, non_blocking: bool = False) -> Self:
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.to(device, non_blocking=non_blocking)
                setattr(self, field.name, value)
        return self

    def pin_memory(self) -> Self:
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                setattr(self, field.name, value.pin_memory())
        return self


@dataclass
class FlowMatchingBatch:
    timestep: torch.Tensor
    x_0: torch.Tensor
    x_1: torch.Tensor

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.timestep.record_stream(stream)
        self.x_0.record_stream(stream)
        self.x_1.record_stream(stream)


class AudioDataset(Dataset):
    """A split of a source; audio prompts are drawn from the same split."""

    def __init__(
        self,
        tts_source: TTSSource,
        sample_rate: int,
        indices: list[int] | None = None,
        audio_prompt: AudioPromptConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """indices: source rows in this split (default: all).
        seed: fixed per-item RNG (validation); None draws fresh randomness."""
        self.tts_source = tts_source
        self.sample_rate = sample_rate
        self.indices = list(range(len(tts_source))) if indices is None else indices
        self.seed = seed
        self.audio_prompts = None
        if audio_prompt is not None:
            if not isinstance(tts_source, ArrowTTSSource):
                raise TypeError(
                    f"audio prompts need an ArrowTTSSource (speaker and word "
                    f"columns), got {type(tts_source).__name__}"
                )
            self.audio_prompts = AudioPromptSampler(
                tts_source, audio_prompt, self.indices
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        row = self.indices[index]
        audio, text = self.tts_source[row]
        if isinstance(audio, AudioFile):
            audio = audio.resample(self.sample_rate).normalize(-24.0).audio
        if self.audio_prompts is None:
            return Sample(idx=row, audio=audio, text=text)
        rng = random.Random(None if self.seed is None else self.seed + row)
        audio, text, prompt = self.audio_prompts.apply(row, audio, text, rng)
        return Sample(idx=row, audio=audio, text=text, audio_prompt=prompt)

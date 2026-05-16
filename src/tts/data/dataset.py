from dataclasses import dataclass, fields

import torch
from torch.utils.data import Dataset

from tts.data.audio import Audio, AudioFile
from tts.data.source import TTSSource
from tts.data.text import Text


@dataclass
class Sample:
    idx: int
    audio: Audio
    text: Text


@dataclass
class Batch:
    idxs: list[int]
    audios: list[Audio]
    texts: list[Text]
    mels: torch.FloatTensor
    mels_mask: torch.BoolTensor
    tokens: torch.LongTensor
    tokens_mask: torch.BoolTensor

    def to(self, device: str | torch.device, non_blocking: bool = False):
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.to(device, non_blocking=non_blocking)
                setattr(self, field.name, value)
        return self

    def pin_memory(self):
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

    def record_stream(self, stream: torch.cuda.Stream):
        self.timestep.record_stream(stream)
        self.x_0.record_stream(stream)
        self.x_1.record_stream(stream)


class AudioDataset(Dataset):
    def __init__(self, tts_source: TTSSource, sample_rate: int):
        self.tts_source = tts_source
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.tts_source)

    def __getitem__(self, index: int) -> Sample:
        audio, text = self.tts_source[index]
        if isinstance(audio, AudioFile):
            audio = audio.resample(self.sample_rate).normalize(-24.0).audio
        return Sample(idx=index, audio=audio, text=text)

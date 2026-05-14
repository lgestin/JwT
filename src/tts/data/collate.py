import torch

from tts.data.dataset import Batch, Sample


def collate(samples: list[Sample]) -> Batch:
    idxs = [sample.idx for sample in samples]
    audios = [sample.audio for sample in samples]
    waveforms = torch.stack([audio.waveform for audio in audios])
    texts = [sample.text for sample in samples]
    tokens = torch.stack([torch.tensor(text.tokens) for text in texts]).long()
    return Batch(
        idxs=idxs,
        audios=audios,
        waveforms=waveforms,
        texts=texts,
        tokens=tokens,
    )

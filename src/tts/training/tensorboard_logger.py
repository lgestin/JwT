from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: Path | str):
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def log_audio(
        self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int
    ) -> None:
        wav = waveform.detach().cpu().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        self.writer.add_audio(tag, wav, step, sample_rate=sample_rate)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        self.writer.add_image(tag, image, step)

    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", float(v), step)

    def set_description(self, description: str) -> None:
        pass

    def update_progress(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        self.writer.close()

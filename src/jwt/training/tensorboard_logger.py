from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from jwt.training.plots import render_curve


def _flatten_for_hparams(config: object) -> dict[str, int | float | str | bool]:
    """Flatten a nested dataclass/dict to the (str, int, float, bool) scalars
    that `add_hparams` accepts, joining nested keys with dots."""
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    out: dict[str, int | float | str | bool] = {}

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)  # ty: ignore[invalid-argument-type]
        elif isinstance(value, Enum):
            out[prefix] = str(value.value)
        elif isinstance(value, bool | int | float | str):
            out[prefix] = value
        elif value is None:
            out[prefix] = "None"
        else:
            out[prefix] = str(value)

    walk("", config)
    return out


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

    def log_diagnostics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        for k, v in metrics.items():
            fv = float(v)
            if fv == fv:  # skip NaN — e.g. an empty timestep bin
                self.writer.add_scalar(f"{prefix}/{k}", fv, step)

    def log_curve(
        self,
        tag: str,
        x: list[float],
        y: list[float],
        step: int,
        xlabel: str = "t",
    ) -> None:
        # Rasterized into the Images tab rather than sent through
        # `add_histogram_raw`. The histogram widget is built for count
        # distributions: it resamples the stored bins onto its own display
        # buckets and *sums* whatever lands in each one, so a mean-per-bin
        # curve on a non-uniformly populated grid grows step artefacts wherever
        # a display bucket happens to swallow more bins than its neighbour. A
        # plot stores exactly what it displays, and gaps stay gaps.
        image = render_curve(x, y, xlabel=xlabel, ylabel=tag.rsplit("/", 1)[-1])
        self.writer.add_image(tag, image, step)

    def log_config(self, config: object, step: int = 0) -> None:
        # `add_hparams` requires only flat scalar values; nested dataclasses are
        # joined with dots so e.g. `model.transformer_config.dim` becomes one
        # hparam. The empty metric_dict means the run shows up in the HParams
        # tab without any paired metrics.
        flat = _flatten_for_hparams(config)
        self.writer.add_hparams(flat, {}, global_step=step)

    def set_description(self, description: str) -> None:
        pass

    def update_progress(self, n: int = 1) -> None:
        pass

    def set_progress(self, completed: int) -> None:
        pass

    def close(self) -> None:
        self.writer.close()

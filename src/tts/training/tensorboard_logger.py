from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def _flatten_for_hparams(config: object) -> dict[str, int | float | str | bool]:
    """Flatten a nested dataclass/dict to the (str, int, float, bool) scalars
    that `add_hparams` accepts, joining nested keys with dots."""
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    out: dict[str, int | float | str | bool] = {}

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(value, Enum):
            out[prefix] = str(value.value)
        elif isinstance(value, bool):
            out[prefix] = value
        elif isinstance(value, (int, float, str)):
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

    def log_histogram(
        self, tag: str, bin_edges: list[float], bin_values: list[float], step: int
    ) -> None:
        # Render a precomputed histogram (e.g. FM loss bucketed by timestep) in
        # the Histograms tab. `bin_edges` are the outer edges, so the right edge
        # of bucket i is `bin_edges[i + 1]`. The summary statistics are
        # placeholders: this is a precomputed function of the bin axis, not a
        # sample of values, so the Distributions-tab stats are not meaningful.
        self.writer.add_histogram_raw(
            tag=tag,
            min=bin_edges[0],
            max=bin_edges[-1],
            num=len(bin_values),
            sum=0.0,
            sum_squares=0.0,
            bucket_limits=bin_edges[1:],
            bucket_counts=bin_values,
            global_step=step,
        )

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

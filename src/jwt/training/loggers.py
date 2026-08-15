from abc import abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Protocol

import torch
from matplotlib import colormaps

from jwt.data.audio.audio import Audio


def flatten_config(config: object) -> dict[str, int | float | str | bool]:
    """Flatten a nested dataclass/dict to (str, int, float, bool) scalars,
    joining nested keys with dots — the shape both TensorBoard's `add_hparams`
    and wandb's `config` accept."""
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


def metric_tag(prefix: str, key: str) -> str:
    """Map a (split, metric) pair to its question-oriented section tag.

    Sections group panels by the question they answer: `loss/` (is it
    optimizing?), `optim/` (is optimization healthy?), `quality_tf/`
    (teacher-forced quality), `quality_gen/` (free-run quality and stopping
    health). Backends group panels by the prefix before the last "/".
    """
    if key == "loss" or key.endswith("_loss") or key == "logmel_l1":
        name = prefix if key == "loss" else f"{prefix}_{key.removesuffix('_loss')}"
        return f"loss/{name}"
    if key.endswith("_norm"):
        return f"optim/{key}"
    if prefix == "sampled":
        return f"quality_gen/{key}"
    return f"quality_tf/{prefix}_{key}"


@dataclass
class SampleRecord:
    """Per-sample evaluation media and metrics, presentation-agnostic.

    Backends choose the rendering: wandb builds a table, TensorBoard logs each
    entry individually, the console writes wav files. Sparse records are fine —
    absent keys simply don't render.
    """

    index: int
    audio: dict[str, Audio] = field(default_factory=dict)
    images: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class Logger(Protocol):
    """Base class for all loggers."""

    def log_scalar(self, tag: str, value: float, step: int): ...
    def log_audio(
        self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int
    ): ...
    def log_image(self, tag: str, image: torch.Tensor, step: int): ...
    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ): ...
    def log_diagnostics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ):
        """Low-frequency, high-cardinality diagnostics (e.g. binned loss).

        Human-facing loggers (console / progress bar) should no-op this; only
        machine-readable backends (TensorBoard) implement it.
        """
        ...

    def log_curve(
        self,
        tag: str,
        x: list[float],
        y: list[float],
        step: int,
        xlabel: str = "t",
        history: bool = True,
    ):
        """Log `y` sampled on the `x` grid as a line plot (equal lengths).

        NaN entries in `y` mark points with no data and are not drawn.
        `history=False` marks a config-fixed curve logged once — backends with
        a step-history view (wandb's heatmap) skip it. Curves deliberately do
        not go through TensorBoard's histogram summary: that widget is built
        for count distributions and re-buckets what it is given (see
        `TensorBoardLogger.log_curve`). Human-facing loggers no-op it.
        """
        ...

    def log_samples(
        self,
        section: str,
        records: list[SampleRecord],
        step: int,
        join: str | None = None,
    ):
        """Log per-sample evaluation records under `section`.

        Each backend renders them its own way (table, individual entries, wav
        files) — see `SampleRecord`. `join` names another section whose latest
        records share this section's indices (e.g. static references for
        generated samples); table backends show them side by side without
        re-uploading, others ignore the hint since they logged that section
        already.
        """
        ...

    def log_config(self, config: object, step: int = 0):
        """Log the resolved run config as hparams. Human-facing loggers no-op
        it; TensorBoard sends it through `add_hparams`.
        """
        ...

    @abstractmethod
    def set_description(self, description: str):
        """Set current status description (for progress bars)."""
        pass

    @abstractmethod
    def update_progress(self, n: int = 1):
        """Update progress counter."""
        pass

    @abstractmethod
    def set_progress(self, completed: int):
        """Set the progress counter to an absolute value (e.g. on resume)."""
        pass

    @abstractmethod
    def close(self):
        """Close the logger and clean up resources."""
        pass


def colorize(img: torch.Tensor, cmap: str = "magma") -> torch.Tensor:
    """(H, W) or (1, H, W) grayscale in [0, 1] -> (3, H, W) RGB through a
    perceptually uniform matplotlib colormap."""
    m = img.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    rgba = colormaps[cmap](m.numpy())  # (H, W, 4)
    return torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).float()


def mel_image(mel: torch.Tensor) -> torch.Tensor:
    """Render a log-mel spectrogram as a (3, n_mels, T) magma-colored image in
    [0, 1], low frequencies at the bottom.

    Accepts (n_mels, T) or (B, n_mels, T) — only the first batch element is used.
    """
    m = mel.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    mn = m.min()
    mx = m.max()
    m = (m - mn) / (mx - mn).clamp(min=1e-9)
    # Low frequencies at the bottom for display.
    m = torch.flip(m, dims=[0])
    return colorize(m)


class MultiLogger:
    """Fan out every call to a list of loggers."""

    def __init__(self, *loggers: Logger):
        self.loggers = loggers

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for lg in self.loggers:
            lg.log_scalar(tag, value, step)

    def log_audio(
        self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int
    ) -> None:
        for lg in self.loggers:
            lg.log_audio(tag, waveform, step, sample_rate)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        for lg in self.loggers:
            lg.log_image(tag, image, step)

    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        for lg in self.loggers:
            lg.log_metrics(metrics, step, prefix)

    def log_diagnostics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        for lg in self.loggers:
            lg.log_diagnostics(metrics, step, prefix)

    def log_curve(
        self,
        tag: str,
        x: list[float],
        y: list[float],
        step: int,
        xlabel: str = "t",
        history: bool = True,
    ) -> None:
        for lg in self.loggers:
            lg.log_curve(tag, x, y, step, xlabel, history)

    def log_samples(
        self,
        section: str,
        records: list[SampleRecord],
        step: int,
        join: str | None = None,
    ) -> None:
        for lg in self.loggers:
            lg.log_samples(section, records, step, join)

    def log_config(self, config: object, step: int = 0) -> None:
        for lg in self.loggers:
            lg.log_config(config, step)

    def set_description(self, description: str) -> None:
        for lg in self.loggers:
            lg.set_description(description)

    def update_progress(self, n: int = 1) -> None:
        for lg in self.loggers:
            lg.update_progress(n)

    def set_progress(self, completed: int) -> None:
        for lg in self.loggers:
            lg.set_progress(completed)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()

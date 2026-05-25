from abc import abstractmethod
from typing import Protocol

import torch


class Logger(Protocol):
    """Base class for all loggers."""

    def log_scalar(self, tag: str, value: float, step: int): ...
    def log_audio(self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int): ...
    def log_image(self, tag: str, image: torch.Tensor, step: int): ...
    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train"): ...
    def log_diagnostics(self, metrics: dict[str, float], step: int, prefix: str = "train"):
        """Low-frequency, high-cardinality diagnostics (e.g. binned loss).

        Human-facing loggers (console / progress bar) should no-op this; only
        machine-readable backends (TensorBoard) implement it.
        """
        ...

    def log_histogram(self, tag: str, bin_edges: list[float], bin_values: list[float], step: int):
        """Log a precomputed histogram — `bin_edges` holds the outer edges, so
        it has one more entry than `bin_values`. Human-facing loggers no-op it.
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


def log_mel(logger: Logger, tag: str, mel: torch.Tensor, step: int) -> None:
    """Render a log-mel spectrogram as a grayscale image and log it via `log_image`.

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
    logger.log_image(tag, m.unsqueeze(0), step)


class MultiLogger:
    """Fan out every call to a list of loggers."""

    def __init__(self, *loggers: Logger):
        self.loggers = loggers

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for lg in self.loggers:
            lg.log_scalar(tag, value, step)

    def log_audio(self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int) -> None:
        for lg in self.loggers:
            lg.log_audio(tag, waveform, step, sample_rate)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        for lg in self.loggers:
            lg.log_image(tag, image, step)

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train") -> None:
        for lg in self.loggers:
            lg.log_metrics(metrics, step, prefix)

    def log_diagnostics(self, metrics: dict[str, float], step: int, prefix: str = "train") -> None:
        for lg in self.loggers:
            lg.log_diagnostics(metrics, step, prefix)

    def log_histogram(
        self, tag: str, bin_edges: list[float], bin_values: list[float], step: int
    ) -> None:
        for lg in self.loggers:
            lg.log_histogram(tag, bin_edges, bin_values, step)

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

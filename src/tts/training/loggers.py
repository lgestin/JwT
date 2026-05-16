from abc import abstractmethod
from typing import Protocol

import torch


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

    @abstractmethod
    def set_description(self, description: str):
        """Set current status description (for progress bars)."""
        pass

    @abstractmethod
    def update_progress(self, n: int = 1):
        """Update progress counter."""
        pass

    @abstractmethod
    def close(self):
        """Close the logger and clean up resources."""
        pass


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

    def set_description(self, description: str) -> None:
        for lg in self.loggers:
            lg.set_description(description)

    def update_progress(self, n: int = 1) -> None:
        for lg in self.loggers:
            lg.update_progress(n)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()

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

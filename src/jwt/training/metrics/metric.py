from typing import Any, Protocol

import torch


class AbsoluteMetric(Protocol):
    def score(
        pred: torch.Tensor,
        mask: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]: ...


class ComparativeMetric(Protocol):
    def score(
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]: ...

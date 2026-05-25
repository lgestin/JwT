from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch
import torch.nn.functional as F


class FlowParametrizations(StrEnum):
    """Enum of available flow matching parametrizations."""

    JWT = "jwt"
    RECTIFIED_FLOW = "rectified_flow"

    @property
    def parametrization(self):
        match self:
            case FlowParametrizations.RECTIFIED_FLOW:
                return RectifiedFlowParametrization
            case FlowParametrizations.JWT:
                return JustWaveformTransformersParametrization


@dataclass(frozen=True)
class ParametrizationLossOutput:
    loss: torch.Tensor
    x_pred: torch.Tensor | None = None
    x_0: torch.Tensor | None = None
    x_1: torch.Tensor | None = None
    x_t: torch.Tensor | None = None


class FlowParametrization(Protocol):
    @staticmethod
    def prepare_x_t(
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor: ...

    @staticmethod
    def loss(
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        pred: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ) -> ParametrizationLossOutput: ...

    @staticmethod
    def step(
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor: ...


class RectifiedFlowParametrization(FlowParametrization):
    """Standard Rectified Flow parametrization with velocity prediction.

    The model directly predicts the velocity v_t = x_1 - x_0.
    This is the default flow matching formulation from:
    "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    """

    @staticmethod
    def prepare_x_t(
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        x_t = (1 - timestep) * x_0 + timestep * x_1
        return x_t

    @staticmethod
    def loss(
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        pred: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ) -> ParametrizationLossOutput:
        v_t = x_1 - x_0
        v_pred = pred
        loss = loss_fn(v_pred, v_t)
        # Recover x_1 prediction from velocity: x_1 = x_t + (1 - t) * v
        x_pred = x_t + (1 - timestep) * v_pred
        return ParametrizationLossOutput(loss=loss, x_pred=x_pred)

    @staticmethod
    def step(
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        v_pred = pred
        x_t = x_t + dt * v_pred
        return x_t


class JustWaveformTransformersParametrization(FlowParametrization):
    """
    Adapted from JIT https://arxiv.org/html/2511.13720
    """

    @staticmethod
    def prepare_x_t(
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        x_t = (1 - timestep) * x_0 + timestep * x_1
        return x_t

    @staticmethod
    def loss(
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        pred: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.l1_loss,
    ) -> ParametrizationLossOutput:
        v_t = (x_1 - x_t) / (1 - timestep).clip(min=0.05)
        x_pred = pred
        v_pred = (x_pred - x_t) / (1 - timestep).clip(min=0.05)
        loss = loss_fn(v_t, v_pred)
        return ParametrizationLossOutput(loss=loss, x_pred=x_pred)

    @staticmethod
    def step(
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        x_pred = pred
        v_pred = (x_pred - x_t) / (1 - timestep)
        x_t = x_t + dt * v_pred
        return x_t

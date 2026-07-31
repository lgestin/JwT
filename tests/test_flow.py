from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn.functional as F

from jwt.model.flow import (
    FlowParametrizations,
    JustWaveformTransformersParametrization,
    ParametrizationLossOutput,
    RectifiedFlowParametrization,
)

PARAMETRIZATIONS = [
    RectifiedFlowParametrization,
    JustWaveformTransformersParametrization,
]


@pytest.fixture(params=[(2, 4, 8), (1, 1, 16), (3, 7, 5)])
def batch(request) -> tuple[torch.Tensor, torch.Tensor]:
    shape = request.param
    g = torch.Generator().manual_seed(0)
    x_0 = torch.randn(shape, generator=g)
    x_1 = torch.randn(shape, generator=g)
    return x_0, x_1


def _timestep(batch_size: int, value: float) -> torch.Tensor:
    return torch.full((batch_size, 1, 1), value)


def test_enum_values() -> None:
    assert FlowParametrizations.RECTIFIED_FLOW.value == "rectified_flow"
    assert FlowParametrizations.JWT.value == "jwt"


def test_enum_parametrization_mapping() -> None:
    assert (
        FlowParametrizations.RECTIFIED_FLOW.parametrization
        is RectifiedFlowParametrization
    )
    assert (
        FlowParametrizations.JWT.parametrization
        is JustWaveformTransformersParametrization
    )


def test_loss_output_is_frozen() -> None:
    out = ParametrizationLossOutput(loss=torch.zeros(()))
    with pytest.raises(FrozenInstanceError):
        out.loss = torch.ones(())


def test_loss_output_defaults_to_none() -> None:
    out = ParametrizationLossOutput(loss=torch.zeros(()))
    assert out.x_pred is None
    assert out.x_0 is None
    assert out.x_1 is None
    assert out.x_t is None


@pytest.mark.parametrize("param", PARAMETRIZATIONS)
def test_prepare_x_t_at_t_zero_returns_x_0(param, batch) -> None:
    x_0, x_1 = batch
    x_t = param.prepare_x_t(x_0, x_1, _timestep(x_0.shape[0], 0.0))
    torch.testing.assert_close(x_t, x_0)


@pytest.mark.parametrize("param", PARAMETRIZATIONS)
def test_prepare_x_t_at_t_one_returns_x_1(param, batch) -> None:
    x_0, x_1 = batch
    x_t = param.prepare_x_t(x_0, x_1, _timestep(x_0.shape[0], 1.0))
    torch.testing.assert_close(x_t, x_1)


@pytest.mark.parametrize("param", PARAMETRIZATIONS)
def test_prepare_x_t_is_linear_at_midpoint(param, batch) -> None:
    x_0, x_1 = batch
    x_t = param.prepare_x_t(x_0, x_1, _timestep(x_0.shape[0], 0.5))
    torch.testing.assert_close(x_t, 0.5 * x_0 + 0.5 * x_1)


@pytest.mark.parametrize("param", PARAMETRIZATIONS)
def test_loss_returns_output_dataclass(param, batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.3)
    x_t = param.prepare_x_t(x_0, x_1, t)
    pred = torch.randn_like(x_0)
    out = param.loss(x_t, t, pred, x_0, x_1)
    assert isinstance(out, ParametrizationLossOutput)
    assert out.loss.ndim == 0
    assert out.x_pred.shape == x_0.shape


@pytest.mark.parametrize("param", PARAMETRIZATIONS)
def test_loss_is_differentiable(param, batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.3)
    x_t = param.prepare_x_t(x_0, x_1, t)
    pred = torch.randn_like(x_0).requires_grad_(True)
    out = param.loss(x_t, t, pred, x_0, x_1)
    out.loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


@pytest.mark.parametrize("param", PARAMETRIZATIONS)
def test_loss_uses_custom_loss_fn(param, batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.3)
    x_t = param.prepare_x_t(x_0, x_1, t)
    pred = torch.randn_like(x_0)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls.append((a, b))
        return F.l1_loss(a, b)

    param.loss(x_t, t, pred, x_0, x_1, loss_fn=fake_loss)
    assert len(calls) == 1


def test_rectified_flow_loss_zero_for_perfect_velocity(batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.3)
    x_t = RectifiedFlowParametrization.prepare_x_t(x_0, x_1, t)
    v_pred = x_1 - x_0
    out = RectifiedFlowParametrization.loss(x_t, t, v_pred, x_0, x_1)
    assert out.loss.item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(out.x_pred, x_1)


def test_rectified_flow_step_one_shot_reaches_x_1(batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.0)
    v_pred = x_1 - x_0
    x_final = RectifiedFlowParametrization.step(x_0, t, v_pred, dt=1.0)
    torch.testing.assert_close(x_final, x_1)


def test_jwt_loss_zero_for_perfect_x_1_prediction(batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.3)
    x_t = JustWaveformTransformersParametrization.prepare_x_t(x_0, x_1, t)
    out = JustWaveformTransformersParametrization.loss(x_t, t, x_1, x_0, x_1)
    assert out.loss.item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(out.x_pred, x_1)


def test_jwt_step_one_shot_reaches_x_1(batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.0)
    x_final = JustWaveformTransformersParametrization.step(x_0, t, x_1, dt=1.0)
    torch.testing.assert_close(x_final, x_1)


def test_jwt_step_partial_from_midpoint_reaches_x_1(batch) -> None:
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 0.5)
    x_t = JustWaveformTransformersParametrization.prepare_x_t(x_0, x_1, t)
    x_final = JustWaveformTransformersParametrization.step(x_t, t, x_1, dt=0.5)
    torch.testing.assert_close(x_final, x_1)


def test_jwt_loss_is_finite_at_t_equals_one(batch) -> None:
    # The clip(min=0.05) inside JWT loss guards the (1-t) divisor — at t=1
    # the loss must remain finite for the parametrization to be trainable.
    x_0, x_1 = batch
    t = _timestep(x_0.shape[0], 1.0)
    x_t = x_1
    x_pred = x_1 + 0.1
    out = JustWaveformTransformersParametrization.loss(x_t, t, x_pred, x_0, x_1)
    assert torch.isfinite(out.loss).item()


@pytest.mark.parametrize("param", PARAMETRIZATIONS)
def test_integration_with_perfect_prediction_reaches_x_1(param, batch) -> None:
    x_0, x_1 = batch
    n_steps = 10
    dt = 1.0 / n_steps
    x = x_0.clone()
    for i in range(n_steps):
        t = _timestep(x_0.shape[0], i * dt)
        pred = (x_1 - x_0) if param is RectifiedFlowParametrization else x_1
        x = param.step(x, t, pred, dt=dt)
    torch.testing.assert_close(x, x_1, atol=1e-5, rtol=1e-5)

import pytest
import torch
from torch import nn

from jwt.training.ema import EMA, EMAConfig


def _const_model(value: float) -> nn.Module:
    """A 2x2 linear layer with every parameter filled with ``value``."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)
    return model


def test_config_defaults() -> None:
    """EMA is enabled by default with the agreed decay."""
    cfg = EMAConfig()
    assert cfg.enabled is True
    assert cfg.decay == 0.9995


def test_effective_decay_warmup() -> None:
    """The decay-warmup ramps from low values up to the configured decay."""
    ema = EMA(_const_model(1.0), decay=0.9995)
    assert ema._effective_decay(0) == pytest.approx(1 / 10)
    assert ema._effective_decay(90) == pytest.approx(91 / 100)
    # Far enough along, the warmup saturates at the configured decay.
    assert ema._effective_decay(1_000_000) == pytest.approx(0.9995)


def test_update_moves_shadow_toward_live() -> None:
    """update() blends shadow toward live weights by (1 - effective_decay)."""
    model = _const_model(1.0)
    ema = EMA(model, decay=0.5)
    # Move the live weights, then update. At step 1000 the warmup has
    # saturated, so the effective decay is exactly 0.5.
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(3.0)
    ema.update(model, step=1000)
    # shadow = 0.5 * 1.0 + 0.5 * 3.0 = 2.0
    for tensor in ema._shadow.values():
        assert torch.allclose(tensor, torch.full_like(tensor, 2.0))


def test_swapped_installs_and_restores() -> None:
    """swapped() installs EMA weights, then restores the originals on exit."""
    model = _const_model(3.0)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for tensor in ema._shadow.values():
            tensor.fill_(2.0)

    with ema.swapped(model):
        for param in model.parameters():
            assert torch.allclose(param, torch.full_like(param, 2.0))

    for param in model.parameters():
        assert torch.allclose(param, torch.full_like(param, 3.0))


def test_swapped_restores_on_exception() -> None:
    """An exception inside the block still restores the original weights."""
    model = _const_model(3.0)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for tensor in ema._shadow.values():
            tensor.fill_(2.0)

    with pytest.raises(RuntimeError, match="boom"), ema.swapped(model):
        raise RuntimeError("boom")

    for param in model.parameters():
        assert torch.allclose(param, torch.full_like(param, 3.0))


def test_state_dict_round_trip() -> None:
    """load_state_dict restores decay and shadow weights from state_dict."""
    model = _const_model(1.0)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(3.0)
    ema.update(model, step=1000)  # shadow now 2.0

    restored = EMA(_const_model(1.0), decay=0.9995)
    restored.load_state_dict(ema.state_dict())

    assert restored.decay == 0.5
    for name, tensor in ema._shadow.items():
        assert torch.equal(restored._shadow[name], tensor)

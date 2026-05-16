import torch

from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from tts.model.transformer import TransformerConfig


def _make_model(n: int = 4) -> RollingFlowSpeaker:
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        mel_dim=16,
        n_denoising_steps=n,
    )
    return RollingFlowSpeaker(cfg).eval()


def _make_text(B: int, T_text: int) -> MaskedTensor:
    return MaskedTensor(
        values=torch.randint(0, 20, (B, 1, T_text)),
        mask=torch.ones(B, T_text, dtype=torch.bool),
    )


def test_speak_shape_and_mask() -> None:
    torch.manual_seed(0)
    model = _make_model()
    text = _make_text(2, 4)
    mel_lens = torch.tensor([5, 8], dtype=torch.long)
    out = model.speak(text, mel_lens)
    assert out.values.shape == (2, 16, 8)
    assert out.mask.shape == (2, 8)
    assert out.mask.sum(-1).tolist() == [5, 8]


def test_speak_outputs_are_finite() -> None:
    torch.manual_seed(0)
    model = _make_model()
    text = _make_text(2, 4)
    out = model.speak(text, torch.tensor([6, 6]))
    assert torch.isfinite(out.values).all()


def test_speak_is_deterministic_with_pinned_x0() -> None:
    torch.manual_seed(0)
    model = _make_model()
    text = _make_text(1, 4)
    mel_lens = torch.tensor([6])
    x_0 = torch.randn(1, 16, 6)
    a = model.speak(text, mel_lens, x_0=x_0)
    b = model.speak(text, mel_lens, x_0=x_0)
    assert torch.equal(a.values, b.values)


def test_speak_actually_updates_positions() -> None:
    """Output should differ from the initial noise — speak must integrate."""
    torch.manual_seed(0)
    model = _make_model()
    text = _make_text(1, 4)
    mel_lens = torch.tensor([6])
    x_0 = torch.randn(1, 16, 6)
    out = model.speak(text, mel_lens, x_0=x_0)
    assert not torch.allclose(out.values, x_0)

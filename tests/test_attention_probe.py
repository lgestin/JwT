import pytest
import torch

from tts.model.attention import SDPAAttention, TorchAttention
from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from tts.model.transformer import TransformerConfig
from tts.training.attention_probe import capture_attention, log_attention_maps


def _model(num_layers: int = 3) -> RollingFlowSpeaker:
    torch.manual_seed(0)
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(
            dim=32, num_heads=4, num_layers=num_layers
        ),
        vocabulary_size=20,
        acoustic_dim=8,
        n_denoising_steps=4,
        eos_n_frames=2,
    )
    return RollingFlowSpeaker(cfg).eval()


def _inputs(
    model: RollingFlowSpeaker, B: int = 2, t_text: int = 4, t_ac: int = 6
) -> tuple[MaskedTensor, MaskedTensor]:
    text = MaskedTensor(
        values=torch.randint(0, model.cfg.vocabulary_size, (B, 1, t_text)),
        mask=torch.ones(B, t_text, dtype=torch.bool),
    )
    acoustic = MaskedTensor(
        values=torch.randn(B, model.cfg.acoustic_dim, t_ac),
        mask=torch.ones(B, t_ac, dtype=torch.bool),
    )
    return text, acoustic


def test_capture_attention_collects_layer_averaged_map() -> None:
    model = _model(num_layers=3)
    text, acoustic = _inputs(model, B=2, t_text=4, t_ac=6)
    with capture_attention(model) as collector:
        model.training_step(
            text, acoustic, attention_implementation=TorchAttention
        )
    attn = collector.map
    assert attn.shape == (2, 4 + 6, 4 + 6)
    # Averaging head-/layer-wise over softmax rows keeps each query a
    # distribution over the visible keys.
    rows = attn.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-4)


def test_capture_attention_records_nothing_for_sdpa() -> None:
    """The fused backend exposes no weights — the collector stays empty."""
    model = _model(num_layers=2)
    text, acoustic = _inputs(model)
    with capture_attention(model) as collector:
        model.training_step(
            text, acoustic, attention_implementation=SDPAAttention
        )
    with pytest.raises(RuntimeError):
        _ = collector.map


def test_capture_attention_removes_hooks_on_exit() -> None:
    model = _model(num_layers=2)
    text, acoustic = _inputs(model)
    with capture_attention(model) as collector:
        model.training_step(
            text, acoustic, attention_implementation=TorchAttention
        )
    for block in model.transformer.blocks:
        assert len(block.attn._forward_hooks) == 0
    # A second probe still works — hooks were cleanly re-registered.
    with capture_attention(model) as collector2:
        model.training_step(
            text, acoustic, attention_implementation=TorchAttention
        )
    assert collector2.map.shape == collector.map.shape


def test_log_attention_maps_slices_text_to_audio_block() -> None:
    logged: dict[str, torch.Tensor] = {}

    class FakeLogger:
        def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
            logged[tag] = image

    attn = torch.rand(2, 10, 10)
    text_lens = torch.tensor([4, 3])
    acoustic_lens = torch.tensor([6, 5])
    log_attention_maps(FakeLogger(), attn, text_lens, acoustic_lens, step=7)

    assert set(logged) == {"valid/attention/0", "valid/attention/1"}
    # (C=1, text rows, audio columns).
    assert logged["valid/attention/0"].shape == (1, 4, 6)
    assert logged["valid/attention/1"].shape == (1, 3, 5)
    # Normalized to [0, 1].
    for img in logged.values():
        assert img.min() >= 0.0 and img.max() <= 1.0

import pytest
import torch

from jwt.model.attention import SDPAAttention, TorchAttention
from jwt.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from jwt.model.transformer import TransformerConfig
from jwt.training.attention_probe import attention_images, capture_attention


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
        model.training_step(text, acoustic, attention_implementation=TorchAttention)
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
        model.training_step(text, acoustic, attention_implementation=SDPAAttention)
    with pytest.raises(RuntimeError):
        _ = collector.map


def test_capture_attention_removes_hooks_on_exit() -> None:
    model = _model(num_layers=2)
    text, acoustic = _inputs(model)
    with capture_attention(model) as collector:
        model.training_step(text, acoustic, attention_implementation=TorchAttention)
    for block in model.transformer.blocks:
        assert len(block.attn._forward_hooks) == 0
    # A second probe still works — hooks were cleanly re-registered.
    with capture_attention(model) as collector2:
        model.training_step(text, acoustic, attention_implementation=TorchAttention)
    assert collector2.map.shape == collector.map.shape


def test_attention_images_slices_text_to_audio_block() -> None:
    attn = torch.rand(3, 10, 10)
    text_lens = torch.tensor([4, 3, 0])
    acoustic_lens = torch.tensor([6, 5, 4])
    images = attention_images(attn, text_lens, acoustic_lens)

    # Sample 2 has no text — skipped.
    assert set(images) == {0, 1}
    # (RGB, text rows, audio columns) — viridis-colorized, nearest-neighbor
    # upscaled by an integer factor so viewers can't blur the cells.
    k0 = -(-256 // 4)
    k1 = -(-256 // 3)
    assert images[0].shape == (3, 4 * k0, 6 * k0)
    assert images[1].shape == (3, 3 * k1, 5 * k1)
    # Each source cell is a constant k x k block (no interpolation).
    assert torch.equal(images[0][:, 0, 0], images[0][:, k0 - 1, k0 - 1])
    # Normalized to [0, 1].
    for img in images.values():
        assert img.min() >= 0.0 and img.max() <= 1.0

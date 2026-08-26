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


def _model(num_layers: int = 3, n_registers: int = 0) -> RollingFlowSpeaker:
    torch.manual_seed(0)
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(
            dim=32, num_heads=4, num_layers=num_layers, n_registers=n_registers
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
    attn = collector.maps
    assert attn.shape == (2, 4 + 6, 4 + 6)
    # Averaging head-/layer-wise over softmax rows keeps each query a
    # distribution over the visible keys.
    rows = attn.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-4)


def test_capture_attention_strips_registers() -> None:
    """Registers sit ahead of the packed sequence; the map must drop them so
    `attention_images` keeps indexing in [text | audio | pad] coordinates."""
    model = _model(num_layers=2, n_registers=8)
    text, acoustic = _inputs(model, B=2, t_text=4, t_ac=6)
    with capture_attention(model) as collector:
        model.training_step(text, acoustic, attention_implementation=TorchAttention)
    attn = collector.maps
    assert attn.shape == (2, 4 + 6, 4 + 6)
    rows = attn.sum(dim=-1)
    assert bool((rows <= 1 + 1e-4).all())
    assert bool((rows < 1 - 1e-4).any()), "some mass should land on registers"


def test_capture_attention_records_nothing_for_sdpa() -> None:
    """The fused backend exposes no weights — the collector stays empty."""
    model = _model(num_layers=2)
    text, acoustic = _inputs(model)
    with capture_attention(model) as collector:
        model.training_step(text, acoustic, attention_implementation=SDPAAttention)
    with pytest.raises(RuntimeError):
        _ = collector.maps


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
    assert collector2.maps.shape == collector.maps.shape


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


def test_collector_strips_register_prefix_before_averaging() -> None:
    from jwt.training.attention_probe import AttentionCollector

    # Every layer carries the same 2-token register prefix: (B, H, T + n, T + n).
    ext1 = torch.rand(2, 4, 8, 8)
    ext2 = torch.rand(2, 4, 8, 8)
    collector = AttentionCollector(n_registers=2)
    collector.record(ext1)
    collector.record(ext2)
    expected = (ext1[:, :, 2:, 2:].mean(1) + ext2[:, :, 2:, 2:].mean(1)) / 2
    assert collector.maps.shape == (2, 6, 6)
    assert torch.allclose(collector.maps, expected)


def test_collector_register_maps_directions() -> None:
    from jwt.training.attention_probe import AttentionCollector

    ext1 = torch.rand(2, 4, 8, 8)  # registers at positions [0, 2)
    ext2 = torch.rand(2, 4, 8, 8)
    collector = AttentionCollector(n_registers=2)
    for m in (ext1, ext2):
        collector.record(m)
    reg_to_seq = collector.registers_to_seq_maps
    seq_to_reg = collector.seq_to_registers_maps
    # Rows are queries: registers_to_seq is register queries over real keys,
    # seq_to_registers is real queries into register keys. Averaged over
    # heads and layers.
    exp_r2s = (ext1[:, :, :2, 2:].mean(1) + ext2[:, :, :2, 2:].mean(1)) / 2
    exp_s2r = (ext1[:, :, 2:, :2].mean(1) + ext2[:, :, 2:, :2].mean(1)) / 2
    assert reg_to_seq.shape == (2, 2, 6) and seq_to_reg.shape == (2, 6, 2)
    assert torch.allclose(reg_to_seq, exp_r2s)
    assert torch.allclose(seq_to_reg, exp_s2r)


def test_collector_without_registers_has_no_register_outputs() -> None:
    from jwt.training.attention_probe import AttentionCollector

    collector = AttentionCollector(n_registers=0)
    collector.record(torch.rand(2, 4, 6, 6))
    text_lens, acoustic_lens = torch.tensor([2, 3]), torch.tensor([4, 3])
    assert collector.scalars(text_lens, acoustic_lens) == {}
    images = collector.images(text_lens, acoustic_lens)
    assert set(images) == {0, 1}
    assert all(set(imgs) == {"attention"} for imgs in images.values())


def test_collector_images_include_register_maps() -> None:
    from jwt.training.attention_probe import AttentionCollector

    collector = AttentionCollector(n_registers=2)
    collector.record(torch.rand(2, 4, 8, 8))
    images = collector.images(torch.tensor([2, 3]), torch.tensor([4, 3]))
    assert all(
        set(imgs) == {"attention", "registers_to_seq", "registers_from_seq"}
        for imgs in images.values()
    )


def test_capture_attention_records_the_models_seq_mask() -> None:
    """The hook reads the widened bool mask off `SelfAttention`'s args and
    crops the register prefix, so `seq_mask` is `(B, T)` in packed coords and
    marks exactly the queries the model treats as real: every text token, and
    audio frames up to and including the first `t = 0` (the rolling frontier);
    the pure-noise frames past it are masked."""
    model = _model(num_layers=2, n_registers=3)
    B, t_text, t_ac = 2, 4, 6
    text, acoustic = _inputs(model, B=B, t_text=t_text, t_ac=t_ac)
    with capture_attention(model) as collector:
        # Pinned fronts: sample 0 has two noise frames past its frontier,
        # sample 1 has none — so both mask shapes are exercised.
        out = model.training_step(
            text,
            acoustic,
            acoustic_front=torch.tensor([0, 2]),
            attention_implementation=TorchAttention,
        )
    seq_mask = collector.seq_mask
    assert seq_mask is not None
    assert seq_mask.dtype == torch.bool and seq_mask.shape == (B, t_text + t_ac)

    first_zero = (out.t == 0.0).int().argmax(-1)  # (B,), acoustic coords
    assert first_zero.tolist() == [3, 5]
    audio_pos = torch.arange(t_ac).unsqueeze(0)
    expected = torch.cat(
        (torch.ones(B, t_text, dtype=torch.bool), audio_pos <= first_zero[:, None]),
        dim=1,
    )
    assert torch.equal(seq_mask, expected)

    scalars = collector.scalars(text.mask.sum(-1), acoustic.mask.sum(-1))
    assert set(scalars) == {
        "register_mass",
        "register_mass_text",
        "register_mass_audio",
    }
    parked = collector.seq_to_registers_maps.sum(-1)  # (B, T)
    assert torch.allclose(scalars["register_mass"], parked[expected].mean())


def test_registers_mass_prefers_seq_mask_over_lengths() -> None:
    from jwt.training.attention_probe import registers_mass

    seq_to_reg = torch.rand(1, 6, 2)
    text_lens, acoustic_lens = torch.tensor([2]), torch.tensor([4])
    seq_mask = torch.tensor([[True, True, True, False, False, False]])
    mass = seq_to_reg.sum(-1)[0]

    by_lens = registers_mass(seq_to_reg, text_lens, acoustic_lens)
    by_mask = registers_mass(seq_to_reg, text_lens, acoustic_lens, seq_mask)
    assert torch.allclose(by_lens["register_mass_audio"], mass[2:6].mean())
    assert torch.allclose(by_mask["register_mass_audio"], mass[2:3].mean())
    assert torch.allclose(by_mask["register_mass_text"], mass[:2].mean())
    assert torch.allclose(by_mask["register_mass"], mass[:3].mean())


def test_register_mass_is_the_complement_of_the_seq_row_sum() -> None:
    """Every real query's mass splits between the sequence keys (`maps`) and
    the register keys (`register_mass`); a softmax row makes them sum to 1."""
    from jwt.training.attention_probe import AttentionCollector

    n, T = 2, 6
    logits = torch.randn(2, 4, n + T, n + T)
    collector = AttentionCollector(n_registers=n)
    collector.record(logits.softmax(-1))
    collector.record(logits.roll(1, dims=0).softmax(-1))
    text_lens, acoustic_lens = torch.tensor([2, 3]), torch.tensor([4, 2])
    scalars = collector.scalars(text_lens, acoustic_lens)
    assert set(scalars) == {
        "register_mass",
        "register_mass_text",
        "register_mass_audio",
    }

    seq_row_sum = collector.maps.sum(-1)  # (B, T)
    parked = collector.seq_to_registers_maps.sum(-1)  # (B, T)
    assert torch.allclose(seq_row_sum + parked, torch.ones_like(parked), atol=1e-6)

    pos = torch.arange(T).unsqueeze(0)
    in_text = pos < text_lens.unsqueeze(1)
    in_audio = (pos >= text_lens.unsqueeze(1)) & (
        pos < (text_lens + acoustic_lens).unsqueeze(1)
    )
    assert not bool((in_text | in_audio)[1, 5])
    assert torch.allclose(scalars["register_mass_text"], parked[in_text].mean())
    assert torch.allclose(scalars["register_mass_audio"], parked[in_audio].mean())
    assert torch.allclose(scalars["register_mass"], parked[in_text | in_audio].mean())


def test_register_attention_images_shapes() -> None:
    from jwt.training.attention_probe import registers_attention_images

    reg_to_seq = torch.rand(2, 3, 10)  # (B, n_registers, T)
    seq_to_reg = torch.rand(2, 10, 3)  # (B, T, n_registers)
    text_lens = torch.tensor([4, 3])
    acoustic_lens = torch.tensor([5, 4])
    images = registers_attention_images(
        reg_to_seq, seq_to_reg, text_lens, acoustic_lens
    )
    # Two images per sample, registers (rows) x real sequence (columns),
    # colorized and integer-upscaled like `attention_images`.
    assert set(images) == {0, 1}
    assert set(images[0]) == {"registers_to_seq", "registers_from_seq"}
    # Only the register axis (rows) is upscaled; frames stay one pixel wide.
    k0 = -(-256 // 3)
    assert images[0]["registers_to_seq"].shape == (3, 3 * k0, 9)
    assert images[0]["registers_from_seq"].shape == (3, 3 * k0, 9)
    assert images[1]["registers_to_seq"].shape == (3, 3 * k0, 7)
    for imgs in images.values():
        for img in imgs.values():
            assert img.min() >= 0.0 and img.max() <= 1.0

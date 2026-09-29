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


def make_model(num_layers: int = 3, n_registers: int = 0) -> RollingFlowSpeaker:
    torch.manual_seed(0)
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(
            dim=32, num_heads=4, num_layers=num_layers, n_registers=n_registers
        ),
        vocabulary_size=20,
        acoustic_dim=8,
        n_denoising_steps=4,
        eos_n_frames=4,
    )
    return RollingFlowSpeaker(cfg).eval()


def make_lens(
    text: MaskedTensor, acoustic: MaskedTensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return text.mask.sum(-1), acoustic.mask.sum(-1)


def make_inputs(
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
    model = make_model(num_layers=3)
    text, acoustic = make_inputs(model, B=2, t_text=4, t_ac=6)
    with capture_attention(model, *make_lens(text, acoustic)) as collector:
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
    model = make_model(num_layers=2, n_registers=8)
    text, acoustic = make_inputs(model, B=2, t_text=4, t_ac=6)
    with capture_attention(model, *make_lens(text, acoustic)) as collector:
        model.training_step(text, acoustic, attention_implementation=TorchAttention)
    attn = collector.maps
    assert attn.shape == (2, 4 + 6, 4 + 6)
    rows = attn.sum(dim=-1)
    assert bool((rows <= 1 + 1e-4).all())
    assert bool((rows < 1 - 1e-4).any()), "some mass should land on registers"


def test_capture_attention_records_nothing_for_sdpa() -> None:
    """The fused backend exposes no weights — the collector stays empty."""
    model = make_model(num_layers=2)
    text, acoustic = make_inputs(model)
    with capture_attention(model, *make_lens(text, acoustic)) as collector:
        model.training_step(text, acoustic, attention_implementation=SDPAAttention)
    with pytest.raises(RuntimeError):
        _ = collector.maps


def test_capture_attention_removes_hooks_on_exit() -> None:
    model = make_model(num_layers=2)
    text, acoustic = make_inputs(model)
    with capture_attention(model, *make_lens(text, acoustic)) as collector:
        model.training_step(text, acoustic, attention_implementation=TorchAttention)
    for block in model.transformer.blocks:
        assert len(block.attn._forward_hooks) == 0
    # A second probe still works — hooks were cleanly re-registered.
    with capture_attention(model, *make_lens(text, acoustic)) as collector2:
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
    collector = AttentionCollector(
        torch.tensor([2, 3]), torch.tensor([4, 3]), n_registers=2
    )
    collector.record(ext1)
    collector.record(ext2)
    expected = (ext1[:, :, 2:, 2:].mean(1) + ext2[:, :, 2:, 2:].mean(1)) / 2
    assert collector.maps.shape == (2, 6, 6)
    assert torch.allclose(collector.maps, expected)


def test_collector_register_maps_directions() -> None:
    from jwt.training.attention_probe import AttentionCollector

    ext1 = torch.rand(2, 4, 8, 8)  # registers at positions [0, 2)
    ext2 = torch.rand(2, 4, 8, 8)
    collector = AttentionCollector(
        torch.tensor([2, 3]), torch.tensor([4, 3]), n_registers=2
    )
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

    collector = AttentionCollector(torch.tensor([2, 3]), torch.tensor([4, 3]))
    collector.record(torch.rand(2, 4, 6, 6))
    scalars = collector.metrics
    assert not any(k.startswith("register") for k in scalars)
    images = collector.images
    assert set(images) == {0, 1}
    assert all(set(imgs) == {"attention"} for imgs in images.values())


def test_collector_images_include_register_maps() -> None:
    from jwt.training.attention_probe import AttentionCollector

    collector = AttentionCollector(
        torch.tensor([2, 3]), torch.tensor([4, 3]), n_registers=2
    )
    collector.record(torch.rand(2, 4, 8, 8))
    images = collector.images
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
    model = make_model(num_layers=2, n_registers=3)
    B, t_text, t_ac = 2, 4, 6
    text, acoustic = make_inputs(model, B=B, t_text=t_text, t_ac=t_ac)
    with capture_attention(model, *make_lens(text, acoustic)) as collector:
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

    scalars = collector.metrics
    assert {"register_mass", "register_mass_text", "register_mass_audio"} <= set(
        scalars
    )
    parked = collector.seq_to_registers_maps.sum(-1)  # (B, T)
    per_utt = torch.stack([parked[b, expected[b]].mean() for b in range(B)])
    assert torch.allclose(scalars["register_mass"], per_utt.mean())


def test_registers_mass_prefers_seq_mask_over_lengths() -> None:
    """The collector's masks use the recorded seq_mask when present, else the
    lengths — checked through register_mass over hand-built weights."""
    from jwt.training.attention_probe import AttentionCollector

    n, T = 2, 6
    attn = torch.rand(1, 3, n + T, n + T).softmax(-1)
    text_lens, acoustic_lens = torch.tensor([2]), torch.tensor([4])
    seq_mask = torch.tensor([[True, True, True, False, False, False]])

    by_lens = AttentionCollector(text_lens, acoustic_lens, n_registers=n)
    by_lens.record(attn)
    by_mask = AttentionCollector(text_lens, acoustic_lens, n_registers=n)
    by_mask.record(attn, torch.cat((torch.ones(1, n, dtype=torch.bool), seq_mask), 1))
    mass = by_lens.seq_to_registers_maps.sum(-1)[0]

    assert torch.allclose(by_lens.metrics["register_mass_audio"], mass[2:6].mean())
    assert torch.allclose(by_mask.metrics["register_mass_audio"], mass[2:3].mean())
    assert torch.allclose(by_mask.metrics["register_mass_text"], mass[:2].mean())
    assert torch.allclose(by_mask.metrics["register_mass"], mass[:3].mean())


def test_register_mass_is_the_complement_of_the_seq_row_sum() -> None:
    """Every real query's mass splits between the sequence keys (`maps`) and
    the register keys (`register_mass`); a softmax row makes them sum to 1."""
    from jwt.training.attention_probe import AttentionCollector

    n, T = 2, 6
    logits = torch.randn(2, 4, n + T, n + T)
    text_lens, acoustic_lens = torch.tensor([2, 3]), torch.tensor([4, 2])
    collector = AttentionCollector(text_lens, acoustic_lens, n_registers=n)
    collector.record(logits.softmax(-1))
    collector.record(logits.roll(1, dims=0).softmax(-1))
    scalars = collector.metrics
    assert {"register_mass", "register_mass_text", "register_mass_audio"} <= set(
        scalars
    )

    seq_row_sum = collector.maps.sum(-1)  # (B, T)
    parked = collector.seq_to_registers_maps.sum(-1)  # (B, T)
    assert torch.allclose(seq_row_sum + parked, torch.ones_like(parked), atol=1e-6)

    pos = torch.arange(T).unsqueeze(0)
    in_text = pos < text_lens.unsqueeze(1)
    in_audio = (pos >= text_lens.unsqueeze(1)) & (
        pos < (text_lens + acoustic_lens).unsqueeze(1)
    )
    assert not bool((in_text | in_audio)[1, 5])

    def per_utt(mask: torch.Tensor) -> torch.Tensor:
        return torch.stack([parked[b, mask[b]].mean() for b in range(2)]).mean()

    assert torch.allclose(scalars["register_mass_text"], per_utt(in_text))
    assert torch.allclose(scalars["register_mass_audio"], per_utt(in_audio))
    assert torch.allclose(scalars["register_mass"], per_utt(in_text | in_audio))


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


def uniform_rows(B: int, H: int, T: int) -> torch.Tensor:
    return torch.full((B, H, T, T), 1.0 / T)


def test_attention_entropy_uniform_is_one_and_one_hot_is_zero() -> None:
    from jwt.training.attention_probe import AttentionCollector

    T = 6
    text_lens, acoustic_lens = torch.tensor([2, 3]), torch.tensor([4, 3])
    uniform = AttentionCollector(text_lens, acoustic_lens)
    uniform.record(uniform_rows(2, 4, T))
    s = uniform.metrics
    for k in ("attn_entropy", "attn_entropy_min_head", "attn_entropy_max_head"):
        assert torch.allclose(s[k], torch.tensor(1.0)), k
        assert torch.allclose(s[k + "_text"], torch.tensor(1.0)), k
        assert torch.allclose(s[k + "_audio"], torch.tensor(1.0)), k
    assert torch.allclose(s["attn_head_jsd"], torch.tensor(0.0), atol=1e-6)

    onehot = AttentionCollector(text_lens, acoustic_lens)
    onehot.record(torch.eye(T).expand(2, 4, T, T).clone())
    s = onehot.metrics
    assert torch.allclose(s["attn_entropy"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(s["attn_head_jsd"], torch.tensor(0.0), atol=1e-6)


def test_attention_entropy_min_max_over_heads_and_jsd() -> None:
    """Head 0 one-hot, head 1 uniform: min is 0, max is 1, mean 0.5, and the
    head-averaged row has entropy above the mean of the heads' — that gap is
    the head JSD."""
    from jwt.training.attention_probe import AttentionCollector

    T = 4
    attn = torch.stack((torch.eye(T), torch.full((T, T), 1 / T))).unsqueeze(0)
    collector = AttentionCollector(torch.tensor([1]), torch.tensor([3]))
    collector.record(attn)  # (1, 2, T, T)
    s = collector.metrics
    assert torch.allclose(s["attn_entropy_min_head"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(s["attn_entropy_max_head"], torch.tensor(1.0))
    assert torch.allclose(s["attn_entropy"], torch.tensor(0.5))
    # Mixed row: 5/8 on the diagonal, 1/8 elsewhere.
    p = torch.tensor([5 / 8, 1 / 8, 1 / 8, 1 / 8])
    h_mix = -(p * p.log()).sum() / torch.log(torch.tensor(float(T)))
    assert torch.allclose(s["attn_head_jsd"], h_mix - 0.5)


def test_attention_entropy_normalizes_by_real_keys_and_registers() -> None:
    """Entropy is scaled by log(n_real_keys + n_registers) per sample, so a
    uniform row over exactly the visible keys scores 1 whatever the length."""
    from jwt.training.attention_probe import AttentionCollector

    n, T = 2, 6
    text_lens, acoustic_lens = torch.tensor([2, 1]), torch.tensor([4, 2])
    attn = torch.zeros(2, 3, n + T, n + T)
    attn[0, :, :, : n + 6] = 1 / (n + 6)
    attn[1, :, :, : n + 3] = 1 / (n + 3)
    collector = AttentionCollector(text_lens, acoustic_lens, n_registers=n)
    collector.record(attn)
    s = collector.metrics
    assert torch.allclose(s["attn_entropy"], torch.tensor(1.0))
    assert torch.allclose(s["attn_entropy_audio"], torch.tensor(1.0))


def test_attention_entropy_respects_seq_mask() -> None:
    """Queries past the rolling frontier are excluded, and the key count used
    for normalization is the masked one."""
    from jwt.training.attention_probe import AttentionCollector

    T = 6
    attn = torch.zeros(1, 2, T, T)
    attn[:, :, :3, :3] = 1 / 3  # rows 0..2 uniform over the 3 real keys
    attn[:, :, 3:, 3] = 1.0  # noise rows: one-hot, would drag entropy down
    seq_mask = torch.tensor([[True, True, True, False, False, False]])
    collector = AttentionCollector(torch.tensor([2]), torch.tensor([4]))
    collector.record(attn, seq_mask)
    s = collector.metrics
    assert torch.allclose(s["attn_entropy"], torch.tensor(1.0))
    assert torch.allclose(s["attn_entropy_audio"], torch.tensor(1.0))


def test_cross_modal_mass() -> None:
    from jwt.training.attention_probe import AttentionCollector

    # [text(2) | audio(3) | pad(1)]; audio rows put 0.7 on text, 0.3 on audio;
    # text rows put 0.4 on text, 0.6 on audio.
    attn = torch.zeros(1, 1, 6, 6)
    attn[0, 0, :2, :2] = 0.2
    attn[0, 0, :2, 2:5] = 0.2
    attn[0, 0, 2:5, :2] = 0.35
    attn[0, 0, 2:5, 2:5] = 0.1
    collector = AttentionCollector(torch.tensor([2]), torch.tensor([3]))
    collector.record(attn)
    s = collector.metrics
    assert torch.allclose(s["attn_mass_audio_to_text"], torch.tensor(0.7))
    assert torch.allclose(s["attn_mass_audio_to_audio"], torch.tensor(0.3))
    assert torch.allclose(s["attn_mass_text_to_text"], torch.tensor(0.4))
    assert torch.allclose(s["attn_mass_text_to_audio"], torch.tensor(0.6))


def test_alignment_picks_best_monotonic_and_covering_head() -> None:
    """Head 0 traces the text (monotonic, full coverage); head 1 parks every
    frame on token 0 (monotonic but no coverage); head 2 reads backwards
    (coverage without monotonicity). The best head by monotonic x coverage is
    reported."""
    from jwt.training.attention_probe import AttentionCollector

    tl, al = 3, 6  # [text(3) | audio(6)]
    T = tl + al
    attn = torch.full((1, 3, T, T), 1e-3)
    audio_q = torch.arange(tl, T)
    attn[0, 0, audio_q, torch.tensor([0, 0, 1, 1, 2, 2])] = 1.0
    attn[0, 1, audio_q, 0] = 1.0
    attn[0, 2, audio_q, torch.tensor([2, 2, 1, 1, 0, 0])] = 1.0
    collector = AttentionCollector(torch.tensor([tl]), torch.tensor([al]))
    collector.record(attn)
    s = collector.metrics
    assert torch.allclose(s["attn_align_monotonic"], torch.tensor(1.0))
    assert torch.allclose(s["attn_align_coverage"], torch.tensor(1.0))

    # Without head 0 the winner is head 2 (3/5 monotonic steps, coverage 1)
    # over head 1 (1.0 x 1/3).
    collector = AttentionCollector(torch.tensor([tl]), torch.tensor([al]))
    collector.record(attn[:, 1:])
    s = collector.metrics
    assert torch.allclose(s["attn_align_monotonic"], torch.tensor(3 / 5))
    assert torch.allclose(s["attn_align_coverage"], torch.tensor(1.0))


def test_scalars_on_empty_collector_raises_the_informative_error() -> None:
    from jwt.training.attention_probe import AttentionCollector

    collector = AttentionCollector(torch.tensor([2]), torch.tensor([4]))
    with pytest.raises(RuntimeError, match="TorchAttention"):
        _ = collector.metrics


def test_scalars_omit_empty_modality_instead_of_nan() -> None:
    """A batch with no audio queries drops the `_audio` splits and the
    audio-query masses rather than logging NaN."""
    from jwt.training.attention_probe import AttentionCollector

    collector = AttentionCollector(torch.tensor([4]), torch.tensor([0]))
    collector.record(torch.rand(1, 2, 4, 4).softmax(-1))
    s = collector.metrics
    assert "attn_entropy" in s and "attn_entropy_text" in s
    assert "attn_entropy_audio" not in s
    assert "attn_mass_audio_to_text" not in s and "attn_mass_audio_to_audio" not in s
    # Text-query metrics stay: text's mass on (empty) audio keys is just 0.
    assert torch.allclose(s["attn_mass_text_to_audio"], torch.tensor(0.0))
    assert "attn_align_monotonic" not in s
    assert all(torch.isfinite(v) for v in s.values())


def test_capture_attention_logs_full_scalar_set() -> None:
    model = make_model(num_layers=2, n_registers=3)
    text, acoustic = make_inputs(model, B=2, t_text=4, t_ac=6)
    with capture_attention(model, *make_lens(text, acoustic)) as collector:
        model.training_step(text, acoustic, attention_implementation=TorchAttention)
    s = collector.metrics
    assert {"register_mass", "attn_entropy", "attn_align_monotonic"} <= set(s)
    assert all(torch.isfinite(v) for v in s.values())
    assert 0.0 <= float(s["attn_entropy_min_head"]) <= float(s["attn_entropy"])
    assert float(s["attn_entropy"]) <= float(s["attn_entropy_max_head"]) <= 1.0
    assert float(s["attn_head_jsd"]) >= -1e-6


def test_sample_metrics_per_sample_values() -> None:
    """Per-sample headline metrics: entropy over the sample's own queries,
    audio->text mass, and the alignment pair whose uniform mean is exactly the
    batch scalar."""
    from jwt.training.attention_probe import AttentionCollector

    tl, al, T = 2, 4, 6
    attn = torch.rand(2, 3, T, T).softmax(-1)
    collector = AttentionCollector(torch.tensor([tl, tl]), torch.tensor([al, al]))
    collector.record(attn)
    per = collector.utterance_metrics
    assert set(per) == {0, 1}
    assert {
        "attn_entropy",
        "attn_mass_audio_to_text",
        "attn_align_monotonic",
        "attn_align_coverage",
    } <= set(per[0])

    # Entropy: normalized mean-head entropy over sample 0's real queries.
    p = attn.float()
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean(1) / torch.log(
        torch.tensor(float(T))
    )
    assert abs(per[0]["attn_entropy"] - ent[0].mean().item()) < 1e-5
    # Audio->text mass over sample 1's audio queries.
    mass = attn.mean(1)[1, tl:, :tl].sum(-1).mean()
    assert abs(per[1]["attn_mass_audio_to_text"] - mass.item()) < 1e-5
    # The batch alignment scalar is the uniform mean of the per-sample values.
    s = collector.metrics
    mean_mono = (per[0]["attn_align_monotonic"] + per[1]["attn_align_monotonic"]) / 2
    assert abs(float(s["attn_align_monotonic"]) - mean_mono) < 1e-5


def test_sample_metrics_include_register_mass_and_skip_empty_samples() -> None:
    from jwt.training.attention_probe import AttentionCollector

    n = 2
    attn = torch.rand(2, 3, n + 6, n + 6).softmax(-1)
    # Sample 1 has no real positions at all.
    collector = AttentionCollector(
        torch.tensor([2, 0]), torch.tensor([4, 0]), n_registers=n
    )
    collector.record(attn)
    per = collector.utterance_metrics
    assert set(per) == {0}
    parked = collector.seq_to_registers_maps.sum(-1)
    assert abs(per[0]["register_mass"] - parked[0].mean().item()) < 1e-5


def test_position_metrics_shapes_and_reduction_chain() -> None:
    """`position_metrics` is dense `(B, T)`; `utterance_metrics` reduces the
    time axis under the masks; `metrics` is their uniform batch mean."""
    from jwt.training.attention_probe import AttentionCollector

    n, T = 2, 6
    attn = torch.rand(2, 3, n + T, n + T).softmax(-1)
    text_lens, acoustic_lens = torch.tensor([2, 3]), torch.tensor([4, 3])
    collector = AttentionCollector(text_lens, acoustic_lens, n_registers=n)
    collector.record(attn)
    dense = collector.position_metrics
    assert set(dense) == {
        "attn_entropy",
        "attn_entropy_min_head",
        "attn_entropy_max_head",
        "attn_head_jsd",
        "attn_mass_to_text",
        "attn_mass_to_audio",
        "register_mass",
    }
    assert all(v.shape == (2, T) for v in dense.values())

    per = collector.utterance_metrics
    ent0 = dense["attn_entropy"][0, :6].mean()  # sample 0 real = all 6
    assert abs(per[0]["attn_entropy"] - float(ent0)) < 1e-6
    assert (
        abs(
            per[1]["attn_mass_audio_to_text"]
            - float(dense["attn_mass_to_text"][1, 3:6].mean())
        )
        < 1e-6
    )

    m = collector.metrics
    assert torch.allclose(
        m["attn_entropy"],
        torch.tensor((per[0]["attn_entropy"] + per[1]["attn_entropy"]) / 2),
    )

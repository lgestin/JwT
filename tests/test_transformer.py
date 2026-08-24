import pytest
import torch
import torch.nn as nn

from jwt.model.transformer import (
    AdaLN,
    RMSNorm,
    Transformer,
    TransformerBlock,
    TransformerConfig,
    precompute_freqs_cis,
)


def test_transformer_shape() -> None:
    model = Transformer(TransformerConfig(dim=64, num_heads=4, num_layers=2))
    x = torch.randn(2, 16, 64)
    t = torch.rand(2, 16)
    y = model(x, t)
    assert y.shape == (2, 16, 64)


def test_transformer_backward() -> None:
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    x = torch.randn(1, 8, 32, requires_grad=True)
    t = torch.rand(1, 8)
    model(x, t).sum().backward()
    assert x.grad is not None


def test_transformer_block_shape() -> None:
    block = TransformerBlock(dim=32, num_heads=4)
    freqs_cis = precompute_freqs_cis(seq_len=8, dim=8, theta=10000.0)
    x = torch.randn(2, 8, 32)
    t_emb = torch.randn(2, 8, 32)
    y = block(x, freqs_cis, t_emb)
    assert y.shape == x.shape


def test_config_roundtrip() -> None:
    cfg = TransformerConfig(dim=64, num_heads=4, num_layers=2, mlp_ratio=2.0)
    assert TransformerConfig.from_dict(cfg.to_dict()) == cfg
    assert TransformerConfig.loads_json(cfg.dumps_json()) == cfg


def test_seq_mask_changes_output() -> None:
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    # AdaLN is zero-init, which gates the attention residual to 0 and makes the
    # mask irrelevant. Break the init so attention contributes to the output.
    for block in model.blocks:
        nn.init.normal_(block.adaLN.linear.weight, std=0.02)
    x = torch.randn(1, 8, 32)
    t = torch.rand(1, 8)
    seq_mask = torch.arange(8)[None] < 4  # hide the last 4 keys
    y_masked = model(x, t, seq_mask=seq_mask)
    y_unmasked = model(x, t)
    assert y_masked.shape == y_unmasked.shape
    assert not torch.allclose(y_masked, y_unmasked)


def test_zero_init_adaLN_is_identity_path() -> None:
    """At init the AdaLN linear is zero, so gates are 0 and residual
    blocks are no-ops."""
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    x = torch.randn(1, 8, 32)
    t = torch.rand(1, 8)
    y = model(x, t)
    assert torch.allclose(y, model.final_norm(x), atol=1e-5)


def test_adaln_returns_raw_unclamped_chunks() -> None:
    """AdaLN is a plain projection — it does not squash anything. Clamping the
    gates is the residual block's job, since only the block knows which chunks
    are gates."""
    torch.manual_seed(0)
    dim = 32
    adaln = AdaLN(dim)
    nn.init.normal_(adaln.linear.weight, std=10.0)
    nn.init.normal_(adaln.linear.bias, std=10.0)

    chunks = adaln(torch.randn(4, dim) * 5.0)

    assert len(chunks) == 6
    for chunk in chunks:
        assert chunk.abs().max() > 1.0


def test_transformer_block_clamps_gates() -> None:
    """Gates are tanh-clamped inside the block, so a block's contribution to the
    residual stream cannot grow without bound as adaLN's pre-activations grow."""
    dim = 32
    freqs_cis = precompute_freqs_cis(seq_len=8, dim=8, theta=10000.0)
    torch.manual_seed(0)
    x = torch.randn(2, 8, dim)
    t_emb = torch.randn(2, 8, dim)

    def residual_delta(gate_bias: float) -> float:
        torch.manual_seed(0)
        block = TransformerBlock(dim=dim, num_heads=4)
        # adaLN weights stay zero-init, so shift/scale are exactly 0 and the
        # gate chunks are exactly `gate_bias` — isolating the gate path.
        with torch.no_grad():
            block.adaLN.linear.bias[2 * dim : 3 * dim] = gate_bias
            block.adaLN.linear.bias[5 * dim : 6 * dim] = gate_bias
        return (block(x, freqs_cis, t_emb) - x).abs().max().item()

    small, huge = residual_delta(10.0), residual_delta(1000.0)

    assert small > 0.0  # the gate path is actually doing something
    # tanh(10) and tanh(1000) both saturate at 1.0; unclamped this would be 100x.
    assert huge == pytest.approx(small, rel=1e-3)


def test_timesteps_change_output() -> None:
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    # Break the zero-init so modulation has an effect.
    for block in model.blocks:
        nn.init.normal_(block.adaLN.linear.weight, std=0.02)
    x = torch.randn(1, 8, 32)
    t1 = torch.zeros(1, 8)
    t2 = torch.ones(1, 8)
    y1 = model(x, t1)
    y2 = model(x, t2)
    assert not torch.allclose(y1, y2)


# --- low-rank AdaLN ---------------------------------------------------------
#
# AdaLN's input is `silu(t_emb)`, and `t` is drawn from a grid of exactly
# `n_denoising_steps` values, so the modulation it produces spans at most
# `n_denoising_steps` dimensions no matter how wide the projection is.
# `adaln_rank` factorizes `Linear(dim, 6*dim)` into `dim -> rank -> 6*dim`.


def test_adaln_rank_none_keeps_a_single_full_rank_linear() -> None:
    adaln = AdaLN(dim=64)
    assert isinstance(adaln.linear, nn.Linear)
    assert adaln.linear.weight.shape == (6 * 64, 64)


def test_adaln_low_rank_parameter_count() -> None:
    dim, rank = 64, 8
    full = sum(p.numel() for p in AdaLN(dim).parameters())
    low = sum(p.numel() for p in AdaLN(dim, rank=rank).parameters())
    # down: dim*rank (no bias, it would be absorbed by up) + up: rank*6*dim + 6*dim
    assert full == dim * 6 * dim + 6 * dim
    assert low == dim * rank + rank * 6 * dim + 6 * dim
    assert low < full


def test_adaln_low_rank_produces_zero_modulation_at_init() -> None:
    """Zero-init must survive factorization: blocks still start as identity."""
    adaln = AdaLN(dim=32, rank=8)
    outs = adaln(torch.randn(4, 32))
    for chunk in outs:
        assert torch.all(chunk == 0.0)


def test_adaln_low_rank_also_returns_raw_unclamped_chunks() -> None:
    """Factorizing must not change the contract: still a plain projection."""
    torch.manual_seed(0)
    adaln = AdaLN(dim=32, rank=8)
    nn.init.normal_(adaln.linear[-1].weight, std=10.0)
    nn.init.normal_(adaln.linear[-1].bias, std=10.0)

    chunks = adaln(torch.randn(4, 32) * 5.0)

    assert len(chunks) == 6
    for chunk in chunks:
        assert chunk.abs().max() > 1.0


def test_adaln_low_rank_modulation_is_bounded_by_rank() -> None:
    """The point of the feature: shift/scale span at most `rank` dimensions."""
    torch.manual_seed(0)
    dim, rank, n_inputs = 64, 8, 40
    t_emb = torch.randn(n_inputs, dim)

    def modulation_rank(adaln: AdaLN) -> int:
        shift_a, scale_a, _, shift_m, scale_m, _ = adaln(t_emb)
        out = torch.cat([shift_a, scale_a, shift_m, scale_m], dim=-1)
        return int(torch.linalg.matrix_rank(out - out.mean(0)).item())

    low = AdaLN(dim, rank=rank)
    nn.init.normal_(low.linear[-1].weight, std=0.5)
    assert modulation_rank(low) <= rank

    full = AdaLN(dim)
    nn.init.normal_(full.linear.weight, std=0.5)
    assert modulation_rank(full) > rank


def test_adaln_low_rank_trains_both_factors() -> None:
    """`up` is zero-init so `down` starts with no gradient (LoRA convention);
    once `up` moves off zero, gradient must reach `down` too."""
    torch.manual_seed(0)
    adaln = AdaLN(dim=32, rank=8)
    t_emb = torch.randn(4, 32)

    adaln(t_emb)[0].sum().backward()
    assert (
        adaln.linear[-1].weight.grad is not None
        and adaln.linear[-1].weight.grad.abs().sum() > 0
    )
    assert adaln.linear[0].weight.grad is not None
    assert adaln.linear[0].weight.grad.abs().sum() == 0  # up is still zero

    adaln.zero_grad()
    nn.init.normal_(adaln.linear[-1].weight, std=0.5)
    adaln(t_emb)[0].sum().backward()
    assert (
        adaln.linear[0].weight.grad is not None
        and adaln.linear[0].weight.grad.abs().sum() > 0
    )


def test_adaln_rank_must_be_positive() -> None:
    for bad in (0, -1):
        try:
            AdaLN(dim=32, rank=bad)
        except AssertionError:
            continue
        raise AssertionError(f"AdaLN accepted rank={bad}")


def test_transformer_adaln_rank_reduces_parameters() -> None:
    kwargs = dict(dim=64, num_heads=4, num_layers=3)
    full = Transformer(TransformerConfig(**kwargs))
    low = Transformer(TransformerConfig(**kwargs, adaln_rank=8))

    def adaln_params(m: Transformer) -> int:
        return sum(p.numel() for n, p in m.named_parameters() if "adaLN" in n)

    assert adaln_params(low) < adaln_params(full)
    assert adaln_params(full) == 3 * (64 * 6 * 64 + 6 * 64)
    assert adaln_params(low) == 3 * (64 * 8 + 8 * 6 * 64 + 6 * 64)


def test_transformer_adaln_rank_forward_and_backward() -> None:
    model = Transformer(
        TransformerConfig(dim=32, num_heads=4, num_layers=2, adaln_rank=4)
    )
    x = torch.randn(2, 16, 32, requires_grad=True)
    t = torch.rand(2, 16)
    y = model(x, t)
    assert y.shape == (2, 16, 32)
    y.sum().backward()
    assert x.grad is not None


def test_transformer_adaln_rank_still_starts_as_identity_path() -> None:
    torch.manual_seed(0)
    model = Transformer(
        TransformerConfig(dim=32, num_heads=4, num_layers=2, adaln_rank=4)
    )
    x = torch.randn(1, 8, 32)
    y = model(x, torch.rand(1, 8))
    assert torch.allclose(y, model.final_norm(x), atol=1e-5)


def test_config_roundtrip_with_adaln_rank() -> None:
    cfg = TransformerConfig(dim=64, num_heads=4, num_layers=2, adaln_rank=16)
    assert TransformerConfig.from_dict(cfg.to_dict()) == cfg
    assert TransformerConfig.loads_json(cfg.dumps_json()) == cfg
    assert TransformerConfig().adaln_rank is None


# --- final layer: norm + timestep modulation --------------------------------
#
# DiT's FinalLayer is norm -> adaLN(shift, scale) -> linear. The linear lives in
# RollingFlowSpeaker (it owns acoustic_dim); the norm and modulation live here.
# Without the modulation, RMSNorm pins the output magnitude to a t-independent
# value while the target magnitude varies ~10x across the schedule.


def test_final_norm_has_no_affine_scale() -> None:
    """The learned scale is subsumed by the modulation's scale."""
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    assert model.final_norm.scale is None


def test_final_modulation_emits_two_unclamped_chunks() -> None:
    """shift/scale, not (shift, scale, gate) x 2 — there is no residual to gate,
    so neither chunk may be tanh-clamped."""
    torch.manual_seed(0)
    adaln = AdaLN(dim=32, n_chunks=2)
    nn.init.normal_(adaln.linear.weight, std=10.0)
    nn.init.normal_(adaln.linear.bias, std=10.0)

    out = adaln(torch.randn(4, 32) * 5.0)

    assert len(out) == 2
    assert all(chunk.shape == (4, 32) for chunk in out)
    assert max(chunk.abs().max().item() for chunk in out) > 1.0


def test_final_modulation_is_identity_at_init() -> None:
    """Zero-init means the whole final layer starts bit-identical to a bare norm."""
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    x = torch.randn(1, 8, 32)
    y = model(x, torch.rand(1, 8))
    assert torch.allclose(y, model.final_norm(x), atol=1e-5)


def test_final_modulation_alone_makes_output_timestep_dependent() -> None:
    """With every block's adaLN left at zero-init the blocks are identity, so any
    t-dependence in the output must come from the final modulation."""
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    x = torch.randn(1, 8, 32)
    t1, t2 = torch.zeros(1, 8), torch.ones(1, 8)

    assert torch.allclose(model(x, t1), model(x, t2), atol=1e-6)

    nn.init.normal_(model.final_modulation.linear.weight, std=0.05)
    assert not torch.allclose(model(x, t1), model(x, t2), atol=1e-6)


def test_final_modulation_honours_adaln_rank() -> None:
    model = Transformer(
        TransformerConfig(dim=64, num_heads=4, num_layers=1, adaln_rank=8)
    )
    assert model.final_modulation.linear[0].weight.shape == (8, 64)
    assert model.final_modulation.linear[-1].weight.shape == (2 * 64, 8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("affine", [True, False])
def test_rmsnorm_output_dtype_matches_input(dtype: torch.dtype, affine: bool) -> None:
    """Output dtype must depend on the input alone, never on `affine`.

    The normalization runs in fp32 for stability; applying the fp32 `scale`
    before casting back keeps that an implementation detail. Multiplying after
    the cast promoted a bf16 activation to fp32 and leaked it downstream.
    """
    x = torch.randn(4, 64, dtype=dtype)
    assert RMSNorm(64, affine=affine)(x).dtype == dtype

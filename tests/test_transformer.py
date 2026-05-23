import torch
import torch.nn as nn

from tts.model.transformer import (
    AdaLN,
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
    freqs_cis = precompute_freqs_cis(seq_len=8, head_dim=8)
    x = torch.randn(2, 8, 32)
    t_emb = torch.randn(2, 8, 32)
    y = block(x, freqs_cis, t_emb)
    assert y.shape == x.shape


def test_config_roundtrip() -> None:
    cfg = TransformerConfig(dim=64, num_heads=4, num_layers=2, mlp_ratio=2.0)
    assert TransformerConfig.from_dict(cfg.to_dict()) == cfg
    assert TransformerConfig.loads_json(cfg.dumps_json()) == cfg


def test_causal_mask_changes_output() -> None:
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    # AdaLN is zero-init, which gates the attention residual to 0 and makes the
    # mask irrelevant. Break the init so attention contributes to the output.
    for block in model.blocks:
        nn.init.normal_(block.adaLN.linear.weight, std=0.02)
    x = torch.randn(1, 8, 32)
    t = torch.rand(1, 8)
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1).logical_not()
    y_masked = model(x, t, mask=mask)
    y_unmasked = model(x, t)
    assert y_masked.shape == y_unmasked.shape
    assert not torch.allclose(y_masked, y_unmasked)


def test_zero_init_adaLN_is_identity_path() -> None:
    """At init the AdaLN linear is zero, so gates are 0 and residual blocks are no-ops."""
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(dim=32, num_heads=4, num_layers=2))
    x = torch.randn(1, 8, 32)
    t = torch.rand(1, 8)
    y = model(x, t)
    assert torch.allclose(y, model.norm(x), atol=1e-5)


def test_adaln_gates_are_tanh_clamped() -> None:
    """Even when AdaLN's linear produces large pre-activations, the two gate
    chunks must come out in [-1, 1]. Shift/scale are intentionally not clamped."""
    torch.manual_seed(0)
    dim = 32
    adaln = AdaLN(dim)
    # Blow the linear weights up so the raw chunks land well outside [-1, 1].
    nn.init.normal_(adaln.linear.weight, std=10.0)
    nn.init.normal_(adaln.linear.bias, std=10.0)

    t_emb = torch.randn(4, dim) * 5.0
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = adaln(t_emb)

    assert torch.all(gate_a.abs() <= 1.0)
    assert torch.all(gate_m.abs() <= 1.0)
    # Sanity-check: with such large weights the raw chunks would be massive,
    # so this asserts the clamp is actually doing work (and isn't a no-op via
    # everything being small to start with).
    assert gate_a.abs().max() > 0.5
    assert gate_m.abs().max() > 0.5
    # Shift/scale stay free.
    assert shift_a.abs().max() > 1.0 or scale_a.abs().max() > 1.0
    assert shift_m.abs().max() > 1.0 or scale_m.abs().max() > 1.0


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

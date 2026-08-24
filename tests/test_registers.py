import pytest
import torch
import torch.nn as nn

from jwt.model.attention import (
    FlashVarlenAttention,
    SDPAAttention,
    flash_attn_varlen_func,  # type: ignore[attr-defined]
)
from jwt.model.registers import Registers
from jwt.model.transformer import (
    Transformer,
    TransformerConfig,
    precompute_freqs_cis,
)


def _skip_unless_cuda_flash() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if flash_attn_varlen_func is None:
        pytest.skip("flash-attn not installed")


def _seq_mask(lens: list[int], T: int, device: str = "cpu") -> torch.Tensor:
    seq_mask = torch.zeros(len(lens), T, dtype=torch.bool, device=device)
    for i, L in enumerate(lens):
        seq_mask[i, :L] = True
    return seq_mask


def _model(n: int = 4, **kwargs) -> Transformer:
    """A small Transformer with registers and *open* adaLN gates.

    AdaLN is zero-init, which gates every residual to 0 and makes registers
    (and masks, and attention generally) unobservable at the output. Every
    test that inspects behaviour rather than shape needs the gates open.
    """
    torch.manual_seed(0)
    cfg = TransformerConfig(
        dim=32,
        num_heads=4,
        num_layers=4,
        n_registers=n,
        **kwargs,
    )
    model = Transformer(cfg)
    for block in model.blocks:
        nn.init.normal_(block.adaLN.linear.weight, std=0.02)
        nn.init.normal_(block.adaLN.linear.bias, std=0.02)
    return model


# --- config -----------------------------------------------------------------


def test_config_roundtrip() -> None:
    cfg = TransformerConfig(dim=64, num_heads=4, num_layers=4, n_registers=8)
    back = TransformerConfig.from_dict(cfg.to_dict())
    assert back == cfg
    assert TransformerConfig.loads_json(cfg.dumps_json()) == cfg


def test_zero_registers_builds_no_module() -> None:
    model = Transformer(
        TransformerConfig(dim=32, num_heads=4, num_layers=2, n_registers=0)
    )
    assert model.registers is None


# --- prepend ----------------------------------------------------------------


def test_prepend_shapes_and_values() -> None:
    n, B, T, dim, head_dim = 4, 2, 6, 32, 8
    registers = Registers(n=n, dim=dim)
    x = torch.randn(B, T, dim)
    t_emb = torch.randn(B, T, dim)
    freqs_cis = precompute_freqs_cis(seq_len=T, dim=head_dim, theta=10000.0)
    seq_mask = _seq_mask([3, 6], T=T)

    x_p, t_emb_p, mask_p, freqs_p = registers.prepend(x, t_emb, seq_mask, freqs_cis)

    assert x_p.shape == (B, T + n, dim)
    assert t_emb_p.shape == (B, T + n, dim)
    assert mask_p is not None and mask_p.shape == (B, T + n)
    assert freqs_p.shape == (T + n, head_dim // 2)

    # The real sequence is carried through untouched, only shifted right by n.
    assert torch.equal(x_p[:, n:], x)
    assert torch.equal(t_emb_p[:, n:], t_emb)
    assert torch.equal(freqs_p[n:], freqs_cis)
    assert mask_p is not None and torch.equal(mask_p[:, n:], seq_mask)

    # Registers are the learned parameter, shared across the batch.
    for b in range(B):
        assert torch.equal(x_p[b, :n], registers.registers)


def test_prepend_none_mask_stays_none() -> None:
    registers = Registers(n=4, dim=32)
    _, _, mask_p, _ = registers.prepend(
        torch.randn(1, 6, 32),
        torch.randn(1, 6, 32),
        None,
        precompute_freqs_cis(6, 8, 10000.0),
    )
    assert mask_p is None


def test_prepend_registers_are_always_visible() -> None:
    """Registers are n leading True keys in every row, so any backend's
    `build_mask` sees them as valid tokens."""
    n, T = 4, 6
    registers = Registers(n=n, dim=32)
    seq_mask = _seq_mask([3, 6], T=T)
    _, _, mask_p, _ = registers.prepend(
        torch.randn(2, T, 32),
        torch.randn(2, T, 32),
        seq_mask,
        precompute_freqs_cis(T, 8, 10000.0),
    )
    assert mask_p is not None and mask_p.dtype == torch.bool
    assert bool(mask_p[:, :n].all())
    assert torch.equal(mask_p[:, n:], seq_mask)


def test_prepend_registers_get_identity_rope() -> None:
    """Registers are position-free: their RoPE factor is exactly 1+0j, so
    `apply_rope` leaves them unrotated and the real tokens keep the absolute
    phases they had before insertion."""
    n, T, head_dim = 4, 6, 8
    registers = Registers(n=n, dim=32)
    freqs_cis = precompute_freqs_cis(seq_len=T, dim=head_dim, theta=10000.0)
    _, _, _, freqs_p = registers.prepend(
        torch.randn(1, T, 32), torch.randn(1, T, 32), None, freqs_cis
    )
    assert freqs_p.dtype == freqs_cis.dtype
    assert torch.equal(freqs_p[:n], torch.ones_like(freqs_p[:n]))
    assert torch.equal(freqs_p[n:], freqs_cis)


def test_prepend_registers_get_zero_t_emb() -> None:
    """Registers have no timestep. A zero `t_emb` makes silu(0) = 0, so their
    adaLN modulation is exactly each block's (learned) bias."""
    n, T, dim = 4, 6, 32
    registers = Registers(n=n, dim=dim)
    t_emb = torch.randn(2, T, dim)
    _, t_emb_p, _, _ = registers.prepend(
        torch.randn(2, T, dim), t_emb, None, precompute_freqs_cis(T, 8, 10000.0)
    )
    assert torch.equal(t_emb_p[:, :n], torch.zeros_like(t_emb_p[:, :n]))


def test_prepend_follows_input_dtype() -> None:
    """Under autocast `x` is half precision while the parameter stays fp32;
    `cat` would raise if the registers were not cast to match."""
    n, T, dim = 4, 6, 32
    registers = Registers(n=n, dim=dim)
    x = torch.randn(2, T, dim, dtype=torch.bfloat16)
    x_p, _, _, _ = registers.prepend(
        x, torch.randn(2, T, dim), None, precompute_freqs_cis(T, 8, 10000.0)
    )
    assert x_p.dtype == torch.bfloat16


# --- integration with Transformer -------------------------------------------


def test_registers_are_stripped_from_output() -> None:
    model = _model(n=8)
    x = torch.randn(2, 16, 32)
    t = torch.rand(2, 16)
    assert model(x, t).shape == x.shape


def test_registers_receive_gradient() -> None:
    model = _model(n=4)
    model(torch.randn(2, 12, 32), torch.rand(2, 12)).sum().backward()
    grad = model.registers.registers.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_registers_change_the_output() -> None:
    """Guards against a silent no-op: the same weights with and without
    registers must not agree."""
    torch.manual_seed(0)
    with_registers = _model(n=8)
    without = Transformer(
        TransformerConfig(dim=32, num_heads=4, num_layers=4, n_registers=0)
    )
    without.load_state_dict(with_registers.state_dict(), strict=False)

    x, t = torch.randn(2, 12, 32), torch.rand(2, 12)
    assert not torch.allclose(with_registers(x, t), without(x, t), atol=1e-4)


def test_output_is_invariant_to_extra_padding() -> None:
    """Registers must not leak masked positions into the valid ones: growing
    the pad must leave every real position bit-comparable."""
    model = _model(n=4)
    T, pad = 12, 5
    seq_mask = _seq_mask([12, 7, 4], T=T)
    x, t = torch.randn(3, T, 32), torch.rand(3, T)

    out = model(x, t, seq_mask)
    out_padded = model(
        torch.cat((x, torch.randn(3, pad, 32)), dim=1),
        torch.cat((t, torch.rand(3, pad)), dim=1),
        _seq_mask([12, 7, 4], T=T + pad),
    )

    for b, L in enumerate([12, 7, 4]):
        torch.testing.assert_close(out[b, :L], out_padded[b, :L], atol=1e-5, rtol=1e-4)


def test_registers_match_across_backends() -> None:
    """End-to-end: the varlen packed layout built from the widened `seq_mask`
    must give the same hidden states as the dense SDPA mask at every valid
    position."""
    _skip_unless_cuda_flash()
    model = _model(n=8).to("cuda").eval()

    B, T = 3, 32
    seq_mask = _seq_mask([32, 11, 23], T=T, device="cuda")
    x = torch.randn(B, T, 32, device="cuda")
    t = torch.rand(B, T, device="cuda")

    outs = {}
    for impl in (SDPAAttention, FlashVarlenAttention):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs[impl.__name__] = model(x, t, seq_mask, impl).float()

    valid = seq_mask.unsqueeze(-1).expand_as(outs["SDPAAttention"])
    diff = (outs["SDPAAttention"] - outs["FlashVarlenAttention"])[valid].abs().max()
    assert diff.item() < 5e-2, f"flash varlen diverged from SDPA: {diff.item():.3e}"

import pytest
import torch
import torch.nn as nn

from jwt.model.attention import (
    FlashVarlenAttention,
    SDPAAttention,
    flash_attn_varlen_func,  # type: ignore[attr-defined]
)
from jwt.model.registers import Registers, RegistersConfig
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


def _widen(seq_mask: torch.Tensor, n: int) -> torch.Tensor:
    """The same (B, T) mask with n always-valid positions prepended."""
    registers = torch.ones(
        seq_mask.size(0), n, dtype=seq_mask.dtype, device=seq_mask.device
    )
    return torch.cat((registers, seq_mask), dim=1)


def _model(n: int = 4, starts_layer: int = 2, **kwargs) -> Transformer:
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
        registers=RegistersConfig(n=n, starts_layer=starts_layer),
        **kwargs,
    )
    model = Transformer(cfg)
    for block in model.blocks:
        nn.init.normal_(block.adaLN.linear.weight, std=0.02)
        nn.init.normal_(block.adaLN.linear.bias, std=0.02)
    return model


# --- config -----------------------------------------------------------------


def test_config_roundtrip() -> None:
    cfg = TransformerConfig(
        dim=64,
        num_heads=4,
        num_layers=4,
        registers=RegistersConfig(n=8, starts_layer=2),
    )
    back = TransformerConfig.from_dict(cfg.to_dict())
    assert back == cfg
    assert isinstance(back.registers, RegistersConfig)
    assert TransformerConfig.loads_json(cfg.dumps_json()) == cfg


def test_starts_layer_past_last_block_is_rejected() -> None:
    """Registers that never get inserted would still be stripped from the
    output, silently shortening it — so the config must refuse them."""
    for starts_layer in (4, 5):
        with pytest.raises(ValueError, match="starts_layer"):
            TransformerConfig(
                num_layers=4, registers=RegistersConfig(n=4, starts_layer=starts_layer)
            )


def test_starts_layer_on_last_block_is_accepted() -> None:
    cfg = TransformerConfig(
        num_layers=4, registers=RegistersConfig(n=4, starts_layer=3)
    )
    assert cfg.registers is not None and cfg.registers.starts_layer == 3


# --- prepend_mask -----------------------------------------------------------


def test_prepend_mask_none_stays_none() -> None:
    registers = Registers(dim=32, config=RegistersConfig(n=4, starts_layer=0))
    assert registers.prepend_mask(None) is None


def test_prepend_mask_rejects_unknown_type() -> None:
    registers = Registers(dim=32, config=RegistersConfig(n=4, starts_layer=0))
    with pytest.raises(TypeError, match="unsupported attention mask type"):
        registers.prepend_mask("not a mask")  # type: ignore[arg-type]


def test_prepend_mask_tensor_widens_key_axis() -> None:
    n = 4
    seq_mask = _seq_mask([3, 6], T=6)
    registers = Registers(dim=32, config=RegistersConfig(n=n, starts_layer=0))
    widened = registers.prepend_mask(SDPAAttention.build_mask(seq_mask))

    assert isinstance(widened, torch.Tensor)
    # (B, 1, 1, T) -> (B, 1, 1, T + n): only the key axis grows.
    assert widened.shape == (2, 1, 1, 6 + n)
    assert widened.dtype == torch.bool
    assert bool(widened[..., :n].all()), "registers must be visible to every query"
    assert torch.equal(widened, SDPAAttention.build_mask(_widen(seq_mask, n)))


def test_prepend_mask_varlen_explicit_layout() -> None:
    """Hand-computed packed layout, independent of `build_mask`.

    B=2, T=6, n=2. Sample 0 has 3 valid tokens, sample 1 has 6, so rows in the
    padded (2, 8) grid start at 0 and 8; each row is [2 registers | tokens].
    """
    n = 2
    seq_mask = _seq_mask([3, 6], T=6)
    registers = Registers(dim=32, config=RegistersConfig(n=n, starts_layer=0))
    mask = registers.prepend_mask(FlashVarlenAttention.build_mask(seq_mask))

    assert mask is not None and not isinstance(mask, torch.Tensor)
    assert mask.cu_seqlens.tolist() == [0, 5, 13]  # (3 + 2), then (6 + 2)
    assert mask.max_seqlen == 8  # 6 + 2
    assert mask.indices.tolist() == [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]
    assert mask.B == 2 and mask.T == 8


@pytest.mark.parametrize("n", [1, 4, 16])
@pytest.mark.parametrize("lens", [[3, 6, 6], [1, 1, 1], [7, 7, 7]])
def test_prepend_mask_varlen_matches_build_mask(n: int, lens: list[int]) -> None:
    """Widening a built varlen mask must equal building it from the already
    widened `seq_mask` — including `indices` *order*, which the kernel reads
    as the packed layout and would silently misattend if permuted."""
    seq_mask = _seq_mask(lens, T=7)
    registers = Registers(dim=32, config=RegistersConfig(n=n, starts_layer=0))

    got = registers.prepend_mask(FlashVarlenAttention.build_mask(seq_mask))
    want = FlashVarlenAttention.build_mask(_widen(seq_mask, n))

    assert got is not None and not isinstance(got, torch.Tensor)
    assert torch.equal(got.cu_seqlens, want.cu_seqlens)
    assert got.cu_seqlens.dtype == want.cu_seqlens.dtype
    assert got.max_seqlen == want.max_seqlen
    assert torch.equal(got.indices, want.indices)
    assert got.indices.dtype == want.indices.dtype
    assert (got.B, got.T) == (want.B, want.T)


def test_prepend_mask_varlen_segments_are_contiguous_and_sorted() -> None:
    """`flash_attn_varlen_func` reads `indices` as B contiguous segments
    delimited by `cu_seqlens`; each segment must stay inside its own row."""
    n, T = 3, 9
    seq_mask = _seq_mask([2, 9, 5], T=T)
    registers = Registers(dim=32, config=RegistersConfig(n=n, starts_layer=0))
    mask = registers.prepend_mask(FlashVarlenAttention.build_mask(seq_mask))
    assert mask is not None and not isinstance(mask, torch.Tensor)

    assert int(mask.cu_seqlens[-1]) == mask.indices.numel()
    for b in range(mask.B):
        segment = mask.indices[mask.cu_seqlens[b] : mask.cu_seqlens[b + 1]]
        assert bool((segment.diff() > 0).all()), "segment must be strictly increasing"
        assert bool((segment // mask.T == b).all()), "segment must stay in row b"
        # The row's n registers come first, then its real tokens.
        assert segment[:n].tolist() == [b * mask.T + i for i in range(n)]


# --- prepend ----------------------------------------------------------------


def test_prepend_shapes_and_values() -> None:
    n, B, T, dim, head_dim = 4, 2, 6, 32, 8
    registers = Registers(dim=dim, config=RegistersConfig(n=n, starts_layer=0))
    x = torch.randn(B, T, dim)
    t_emb = torch.randn(B, T, dim)
    freqs_cis = precompute_freqs_cis(seq_len=T, dim=head_dim, theta=10000.0)
    mask = SDPAAttention.build_mask(_seq_mask([3, 6], T=T))

    x_p, t_emb_p, mask_p, freqs_p = registers.prepend(x, t_emb, mask, freqs_cis)

    assert x_p.shape == (B, T + n, dim)
    assert t_emb_p.shape == (B, T + n, dim)
    assert isinstance(mask_p, torch.Tensor) and mask_p.shape == (B, 1, 1, T + n)
    assert freqs_p.shape == (T + n, head_dim // 2)

    # The real sequence is carried through untouched, only shifted right by n.
    assert torch.equal(x_p[:, n:], x)
    assert torch.equal(t_emb_p[:, n:], t_emb)
    assert torch.equal(freqs_p[n:], freqs_cis)

    # Registers are the learned parameter, shared across the batch.
    for b in range(B):
        assert torch.equal(x_p[b, :n], registers.registers)


def test_prepend_registers_get_identity_rope() -> None:
    """Registers are position-free: their RoPE factor is exactly 1+0j, so
    `apply_rope` leaves them unrotated and the real tokens keep the absolute
    phases they had before insertion."""
    n, T, head_dim = 4, 6, 8
    registers = Registers(dim=32, config=RegistersConfig(n=n, starts_layer=0))
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
    registers = Registers(dim=dim, config=RegistersConfig(n=n, starts_layer=0))
    t_emb = torch.randn(2, T, dim)
    _, t_emb_p, _, _ = registers.prepend(
        torch.randn(2, T, dim), t_emb, None, precompute_freqs_cis(T, 8, 10000.0)
    )
    assert torch.equal(t_emb_p[:, :n], torch.zeros_like(t_emb_p[:, :n]))


def test_prepend_follows_input_dtype() -> None:
    """Under autocast `x` is half precision while the parameter stays fp32;
    `cat` would raise if the registers were not cast to match."""
    n, T, dim = 4, 6, 32
    registers = Registers(dim=dim, config=RegistersConfig(n=n, starts_layer=0))
    x = torch.randn(2, T, dim, dtype=torch.bfloat16)
    x_p, _, _, _ = registers.prepend(
        x, torch.randn(2, T, dim), None, precompute_freqs_cis(T, 8, 10000.0)
    )
    assert x_p.dtype == torch.bfloat16


# --- integration with Transformer -------------------------------------------


def test_registers_are_stripped_from_output() -> None:
    model = _model(n=8, starts_layer=2)
    x = torch.randn(2, 16, 32)
    t = torch.rand(2, 16)
    assert model(x, t).shape == x.shape


def test_registers_receive_gradient() -> None:
    model = _model(n=4, starts_layer=1)
    model(torch.randn(2, 12, 32), torch.rand(2, 12)).sum().backward()
    grad = model.registers.registers.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_registers_change_the_output() -> None:
    """Guards against a silent no-op: the same weights with and without
    registers must not agree."""
    torch.manual_seed(0)
    with_registers = _model(n=8, starts_layer=1)
    without = Transformer(
        TransformerConfig(dim=32, num_heads=4, num_layers=4, registers=None)
    )
    without.load_state_dict(with_registers.state_dict(), strict=False)

    x, t = torch.randn(2, 12, 32), torch.rand(2, 12)
    assert not torch.allclose(with_registers(x, t), without(x, t), atol=1e-4)


def test_starts_layer_changes_the_output() -> None:
    x, t = torch.randn(2, 12, 32), torch.rand(2, 12)
    early, late = _model(n=4, starts_layer=0), _model(n=4, starts_layer=3)
    late.load_state_dict(early.state_dict())
    assert not torch.allclose(early(x, t), late(x, t), atol=1e-4)


def test_output_is_invariant_to_extra_padding() -> None:
    """Registers must not leak masked positions into the valid ones: growing
    the pad must leave every real position bit-comparable."""
    model = _model(n=4, starts_layer=1)
    T, pad = 12, 5
    seq_mask = _seq_mask([12, 7, 4], T=T)
    x, t = torch.randn(3, T, 32), torch.rand(3, T)

    out = model(x, t, SDPAAttention.build_mask(seq_mask))
    out_padded = model(
        torch.cat((x, torch.randn(3, pad, 32)), dim=1),
        torch.cat((t, torch.rand(3, pad)), dim=1),
        SDPAAttention.build_mask(_seq_mask([12, 7, 4], T=T + pad)),
    )

    for b, L in enumerate([12, 7, 4]):
        torch.testing.assert_close(out[b, :L], out_padded[b, :L], atol=1e-5, rtol=1e-4)


def test_registers_match_across_backends() -> None:
    """End-to-end: the varlen packed layout built by `prepend_mask` must give
    the same hidden states as the dense SDPA mask at every valid position."""
    _skip_unless_cuda_flash()
    model = _model(n=8, starts_layer=1).to("cuda").eval()

    B, T = 3, 32
    seq_mask = _seq_mask([32, 11, 23], T=T, device="cuda")
    x = torch.randn(B, T, 32, device="cuda")
    t = torch.rand(B, T, device="cuda")

    outs = {}
    for impl in (SDPAAttention, FlashVarlenAttention):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs[impl.__name__] = model(x, t, impl.build_mask(seq_mask), impl).float()

    valid = seq_mask.unsqueeze(-1).expand_as(outs["SDPAAttention"])
    diff = (outs["SDPAAttention"] - outs["FlashVarlenAttention"])[valid].abs().max()
    assert diff.item() < 5e-2, f"flash varlen diverged from SDPA: {diff.item():.3e}"

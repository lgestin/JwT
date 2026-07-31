import pytest
import torch
import torch.nn as nn

from jwt.model.attention import (
    AttentionImplementations,
    FlashVarlenAttention,
    SDPAAttention,
    TorchAttention,
    flash_attn_varlen_func,  # type: ignore[attr-defined]
)
from jwt.model.transformer import Transformer, TransformerConfig


def _skip_unless_cuda_flash() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if flash_attn_varlen_func is None:
        pytest.skip("flash-attn not installed")


def _qkv(B=2, H=4, T=6, D=8):
    torch.manual_seed(0)
    return (
        torch.randn(B, H, T, D),
        torch.randn(B, H, T, D),
        torch.randn(B, H, T, D),
    )


def test_build_mask_shape() -> None:
    attn_keys = torch.ones(2, 6, dtype=torch.bool)
    for impl in (SDPAAttention, TorchAttention):
        mask = impl.build_mask(attn_keys)
        assert mask.shape == (2, 1, 1, 6)
        assert mask.dtype == torch.bool


def test_sdpa_returns_no_weights() -> None:
    q, k, v = _qkv()
    out, attn_weights = SDPAAttention.attention(q, k, v, mask=None)
    assert out.shape == q.shape
    assert attn_weights is None


def test_torch_returns_normalized_weights() -> None:
    q, k, v = _qkv()
    out, attn_weights = TorchAttention.attention(q, k, v, mask=None)
    assert out.shape == q.shape
    assert attn_weights is not None
    assert attn_weights.shape == (q.shape[0], q.shape[1], q.shape[2], q.shape[2])
    # Each query row is a probability distribution over keys.
    rows = attn_weights.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


def test_torch_matches_sdpa_output() -> None:
    """The explicit backend must produce the same context vectors as the
    fused kernel — only the exposed weights differ."""
    q, k, v = _qkv()
    out_sdpa, _ = SDPAAttention.attention(q, k, v, mask=None)
    out_torch, _ = TorchAttention.attention(q, k, v, mask=None)
    assert torch.allclose(out_sdpa, out_torch, atol=1e-5)


def test_torch_matches_sdpa_with_mask() -> None:
    q, k, v = _qkv()
    attn_keys = torch.tensor(
        [[True, True, True, False, False, False], [True, True, True, True, True, False]]
    )
    mask = TorchAttention.build_mask(attn_keys)
    out_sdpa, _ = SDPAAttention.attention(q, k, v, mask)
    out_torch, attn_weights = TorchAttention.attention(q, k, v, mask)
    assert torch.allclose(out_sdpa, out_torch, atol=1e-5)
    # Masked-out keys receive exactly zero probability.
    assert attn_weights is not None
    masked = attn_weights[~mask.expand_as(attn_weights)]
    assert torch.all(masked == 0.0)


def test_enum_resolves_implementation() -> None:
    assert AttentionImplementations.SDPA.implementation is SDPAAttention
    assert AttentionImplementations.TORCH.implementation is TorchAttention
    assert AttentionImplementations.FLASH_VARLEN.implementation is FlashVarlenAttention


def test_flash_varlen_build_mask_structure() -> None:
    _skip_unless_cuda_flash()
    attn_keys = torch.zeros(2, 6, dtype=torch.bool, device="cuda")
    attn_keys[0, :3] = True
    attn_keys[1, :6] = True
    mask = FlashVarlenAttention.build_mask(attn_keys)
    assert mask.cu_seqlens.tolist() == [0, 3, 9]
    assert mask.max_seqlen == 6
    assert mask.indices.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]
    assert mask.B == 2 and mask.T == 6


def test_flash_varlen_matches_sdpa_attention() -> None:
    """Kernel-level: FlashVarlen and SDPA give the same context vectors at
    valid positions, within bf16 reduction-order noise. Masked positions are
    undefined for varlen (it never writes them), so they're excluded."""
    _skip_unless_cuda_flash()
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 32, 16
    device, dtype = "cuda", torch.bfloat16

    attn_keys = torch.zeros(B, T, dtype=torch.bool, device=device)
    attn_keys[0, :20] = True
    attn_keys[1, :32] = True

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    sdpa_out, _ = SDPAAttention.attention(q, k, v, SDPAAttention.build_mask(attn_keys))
    flash_out, _ = FlashVarlenAttention.attention(
        q, k, v, FlashVarlenAttention.build_mask(attn_keys)
    )

    valid = attn_keys[:, None, :, None].expand_as(sdpa_out)
    assert torch.allclose(sdpa_out[valid], flash_out[valid], atol=5e-3)


def test_transformer_outputs_match_across_backends() -> None:
    """End-to-end: a small Transformer produces equivalent hidden states at
    valid positions regardless of attention backend, within bf16 noise."""
    _skip_unless_cuda_flash()
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    model = (
        Transformer(TransformerConfig(dim=64, num_heads=4, num_layers=2))
        .to(device)
        .eval()
    )
    # AdaLN zero-init gates the attention residual to 0 (see
    # test_zero_init_adaLN_is_identity_path in test_transformer.py), which
    # would make any mask/backend irrelevant. Randomize so attention matters.
    for block in model.blocks:
        nn.init.normal_(block.adaLN.linear.weight, std=0.02)

    B, T = 3, 24
    x = torch.randn(B, T, 64, device=device)
    t = torch.rand(B, T, device=device)
    attn_keys = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i, L in enumerate([12, 18, 24]):
        attn_keys[i, :L] = True

    outs: dict[str, torch.Tensor] = {}
    for impl in (TorchAttention, SDPAAttention, FlashVarlenAttention):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            out = model(
                x,
                t,
                mask=impl.build_mask(attn_keys),
                attention_implementation=impl,
            )
        outs[impl.__name__] = out.float()

    valid = attn_keys.unsqueeze(-1).expand_as(outs["SDPAAttention"])
    ref = outs["SDPAAttention"]
    for name in ("TorchAttention", "FlashVarlenAttention"):
        diff = (ref - outs[name])[valid].abs().max().item()
        assert diff < 5e-2, f"{name} diverged from SDPA: max abs diff {diff:.3e}"

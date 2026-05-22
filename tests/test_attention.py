import torch

from tts.model.attention import (
    AttentionImplementations,
    SDPAAttention,
    TorchAttention,
)


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
        [[True, True, True, False, False, False],
         [True, True, True, True, True, False]]
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

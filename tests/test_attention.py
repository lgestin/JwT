import pytest
import torch
from torch import nn

from jwt.model.attention import (
    AttentionImplementations,
    FlexAttention,
    SDPAAttention,
    TorchAttention,
)
from jwt.model.transformer import Transformer, TransformerConfig

INF = float("inf")


def skip_unless_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def uncommitted(seq_mask: torch.Tensor) -> torch.Tensor:
    """All-inf commit indices: every query sees every visible key."""
    return torch.full(seq_mask.shape, INF, device=seq_mask.device)


def make_qkv(
    B: int = 2, H: int = 4, T: int = 6, D: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return (
        torch.randn(B, H, T, D),
        torch.randn(B, H, T, D),
        torch.randn(B, H, T, D),
    )


def test_build_mask_shape() -> None:
    seq_mask = torch.ones(2, 6, dtype=torch.bool)
    commit_index = uncommitted(seq_mask)
    for impl in (SDPAAttention, TorchAttention):
        mask = impl.build_mask(seq_mask, commit_index, commit_index)
        assert mask.shape == (2, 1, 6, 6)
        assert mask.dtype == torch.bool


def test_sdpa_returns_no_weights() -> None:
    q, k, v = make_qkv()
    out, attn_weights = SDPAAttention.attention(q, k, v, mask=None)
    assert out.shape == q.shape
    assert attn_weights is None


def test_torch_returns_normalized_weights() -> None:
    q, k, v = make_qkv()
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
    q, k, v = make_qkv()
    out_sdpa, _ = SDPAAttention.attention(q, k, v, mask=None)
    out_torch, _ = TorchAttention.attention(q, k, v, mask=None)
    assert torch.allclose(out_sdpa, out_torch, atol=1e-5)


def test_torch_matches_sdpa_with_mask() -> None:
    q, k, v = make_qkv()
    seq_mask = torch.tensor(
        [[True, True, True, False, False, False], [True, True, True, True, True, False]]
    )
    commit_index = uncommitted(seq_mask)
    mask = TorchAttention.build_mask(seq_mask, commit_index, commit_index)
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
    assert AttentionImplementations.FLEX.implementation is FlexAttention
    with pytest.raises(ValueError, match="FLASH_VARLEN"):
        _ = AttentionImplementations.FLASH_VARLEN.implementation


# --- commit-index (block-causal) masks ---------------------------------------


def make_staircase() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Toy packed layout: 2 prefix | 2 clean | 2 window | 1 pad, and the
    expected (Tq, Tk) visibility under `commit_rule`."""
    seq_mask = torch.tensor([[True] * 6 + [False]])
    commit_index = torch.tensor([[0.0, 0.0, 1.0, 2.0, INF, INF, INF]])
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    return seq_mask, commit_index, expected


def test_dense_mask_follows_commit_rule() -> None:
    """The dense mask is the block-causal staircase of the commit rule."""
    seq_mask, commit_index, expected = make_staircase()
    for impl in (SDPAAttention, TorchAttention):
        mask = impl.build_mask(seq_mask, commit_index, commit_index)
        assert mask.shape == (1, 1, 7, 7)
        assert torch.equal(mask[0, 0], expected)


def test_dense_mask_of_uncommitted_tokens_is_key_only() -> None:
    """With every token uncommitted each query sees exactly the visible keys."""
    seq_mask, _, _ = make_staircase()
    commit_index = uncommitted(seq_mask)
    mask = SDPAAttention.build_mask(seq_mask, commit_index, commit_index)
    assert torch.equal(mask[0, 0], seq_mask.expand(7, 7))


def test_dense_mask_with_cached_keys() -> None:
    """Queries may be a suffix of the keys: 2 new tokens (one clean, one
    window) against 4 cached keys plus themselves."""
    valid_k = torch.tensor([[True, True, False, True, True, True]])
    commit_index_k = torch.tensor([[0.0, 0.0, INF, 1.0, 2.0, INF]])
    commit_index_q = commit_index_k[:, -2:]
    mask = SDPAAttention.build_mask(valid_k, commit_index_k, commit_index_q)
    assert mask.shape == (1, 1, 2, 6)
    assert mask[0, 0].tolist() == [
        [True, True, False, True, True, False],
        [True, True, False, True, True, True],
    ]


# --- FlexAttention -------------------------------------------------------------


def make_layout(
    B: int, T: int, spans: list[tuple[int, int, int]], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample (prefix, clean, window) lengths -> seq_mask and commit."""
    seq_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    commit_index = torch.full((B, T), INF, device=device)
    for b, (P, C, W) in enumerate(spans):
        seq_mask[b, : P + C + W] = True
        commit_index[b, :P] = 0.0
        commit_index[b, P : P + C] = torch.arange(1, C + 1, device=device).float()
    return seq_mask, commit_index


def test_flex_rejects_foreign_mask() -> None:
    """FlexAttention refuses a dense tensor mask."""
    q, k, v = make_qkv()
    seq_mask = torch.ones(2, 6, dtype=torch.bool)
    commit_index = uncommitted(seq_mask)
    mask = SDPAAttention.build_mask(seq_mask, commit_index, commit_index)
    with pytest.raises(TypeError):
        FlexAttention.attention(q, k, v, mask)


def test_flex_matches_sdpa_with_key_mask() -> None:
    """Flex and SDPA agree under a plain key mask."""
    skip_unless_cuda()
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 32, 16
    device, dtype = "cuda", torch.bfloat16
    seq_mask, _ = make_layout(B, T, [(20, 0, 0), (32, 0, 0)], device)
    commit_index = uncommitted(seq_mask)
    q, k, v = (torch.randn(B, H, T, D, device=device, dtype=dtype) for _ in range(3))

    sdpa_out, _ = SDPAAttention.attention(
        q, k, v, SDPAAttention.build_mask(seq_mask, commit_index, commit_index)
    )
    flex_out, weights = FlexAttention.attention(
        q, k, v, FlexAttention.build_mask(seq_mask, commit_index, commit_index)
    )
    assert weights is None
    valid = seq_mask[:, None, :, None].expand_as(sdpa_out)
    assert torch.allclose(sdpa_out[valid], flex_out[valid], atol=5e-3)


def test_flex_matches_sdpa_under_commit_rule() -> None:
    """Flex and SDPA agree under the commit rule."""
    skip_unless_cuda()
    torch.manual_seed(0)
    B, H, T, D = 3, 4, 40, 16
    device, dtype = "cuda", torch.bfloat16
    seq_mask, commit_index = make_layout(
        B, T, [(6, 10, 8), (9, 0, 12), (4, 20, 16)], device
    )
    q, k, v = (torch.randn(B, H, T, D, device=device, dtype=dtype) for _ in range(3))

    sdpa_out, _ = SDPAAttention.attention(
        q, k, v, SDPAAttention.build_mask(seq_mask, commit_index, commit_index)
    )
    flex_out, _ = FlexAttention.attention(
        q, k, v, FlexAttention.build_mask(seq_mask, commit_index, commit_index)
    )
    valid = seq_mask[:, None, :, None].expand_as(sdpa_out)
    assert torch.allclose(sdpa_out[valid], flex_out[valid], atol=5e-3)


def test_flex_matches_sdpa_with_cached_keys() -> None:
    """Queries are the last Tq tokens of the keys, as in the cached sampler."""
    skip_unless_cuda()
    torch.manual_seed(0)
    B, H, Tk, Tq, D = 2, 4, 50, 8, 16
    device, dtype = "cuda", torch.bfloat16
    valid_k = torch.ones(B, Tk, dtype=torch.bool, device=device)
    valid_k[1, 10:14] = False  # padded text in the cache
    commit_index_k = torch.cat(
        [
            torch.zeros(B, 20, device=device),
            torch.arange(1, 23, device=device).float().expand(B, 22),
            torch.full((B, Tq), INF, device=device),
        ],
        dim=1,
    )
    commit_index_k[:, 42] = 23.0  # the window's first frame just reached t=1
    commit_index_q = commit_index_k[:, -Tq:]
    q = torch.randn(B, H, Tq, D, device=device, dtype=dtype)
    k, v = (torch.randn(B, H, Tk, D, device=device, dtype=dtype) for _ in range(2))

    sdpa_out, _ = SDPAAttention.attention(
        q, k, v, SDPAAttention.build_mask(valid_k, commit_index_k, commit_index_q)
    )
    flex_out, _ = FlexAttention.attention(
        q, k, v, FlexAttention.build_mask(valid_k, commit_index_k, commit_index_q)
    )
    assert torch.allclose(sdpa_out, flex_out, atol=5e-3)


def test_transformer_outputs_match_across_backends() -> None:
    """End-to-end: a small Transformer produces equivalent hidden states at
    valid positions regardless of attention backend, within bf16 noise."""
    skip_unless_cuda()
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
    seq_mask, commit_index = make_layout(
        B, T, [(4, 4, 4), (6, 0, 12), (2, 10, 12)], device
    )

    outs: dict[str, torch.Tensor] = {}
    for impl in (TorchAttention, SDPAAttention, FlexAttention):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            out = model(
                x,
                t,
                torch.arange(T, device=device).float().expand(B, T),
                impl,
                seq_mask=seq_mask,
                commit_index=commit_index,
            )
        outs[impl.__name__] = out.float()

    valid = seq_mask.unsqueeze(-1).expand_as(outs["SDPAAttention"])
    ref = outs["SDPAAttention"]
    for name in ("TorchAttention", "FlexAttention"):
        diff = (ref - outs[name])[valid].abs().max().item()
        assert diff < 5e-2, f"{name} diverged from SDPA: max abs diff {diff:.3e}"

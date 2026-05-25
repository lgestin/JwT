from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # ty: ignore[invalid-assignment]


@dataclass
class FlashAttentionVarlenMask:
    """Metadata for `flash_attn_varlen_func`: which `(b, t)` positions are
    valid plus the cumulative-seqlen layout flash_attn expects."""

    cu_seqlens: torch.Tensor  # (B+1,) int32, prefix-sum of per-sample seqlens
    max_seqlen: int
    indices: torch.Tensor  # int64 flat positions of valid tokens in (B*T,)
    B: int
    T: int


type AttentionMask = torch.Tensor | FlashAttentionVarlenMask


class AttentionImplementation(Protocol):
    """A pluggable attention backend.

    Backends are stateless: both methods are static, so the implementation can
    be passed around as a bare class and selected per `forward` call.
    """

    @staticmethod
    def build_mask(attn_keys: torch.Tensor) -> AttentionMask:
        """`attn_keys`: (B, T) bool, True = visible key. Returns the mask
        object consumed by `attention`, or None for an unmasked backend."""
        ...

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: AttentionMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`q`, `k`, `v`: (B, H, T, D). Returns `(out, attn_weights)`.

        `out` is (B, H, T, D). `attn_weights` is the (B, H, T, T) softmax
        matrix when the backend can expose it, else `None` (the fused SDPA
        kernel never materializes it).
        """
        ...


class SDPAAttention:
    """Attention via `F.scaled_dot_product_attention` — the fused kernel.

    Fast and memory-efficient, but the softmax matrix is never materialized,
    so `attention` returns `attn_weights=None`.
    """

    @staticmethod
    def build_mask(attn_keys: torch.Tensor) -> torch.Tensor:
        return attn_keys[:, None, None, :]

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        assert mask is None or isinstance(mask, torch.Tensor)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None


class TorchAttention:
    """Explicit attention in plain torch ops.

    Slower and materializes the (B, H, T, T) softmax matrix, but returns it as
    `attn_weights` so callers can inspect the attention distribution. Scores
    are computed in fp32 for a clean, autocast-independent map.
    """

    @staticmethod
    def build_mask(attn_keys: torch.Tensor) -> torch.Tensor:
        return attn_keys[:, None, None, :]

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert mask is None or isinstance(mask, torch.Tensor)
        scale = q.shape[-1] ** -0.5
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        out = (attn_weights @ v.float()).to(v.dtype)
        return out, attn_weights


class FlashVarlenAttention:
    """Attention via `flash_attn.flash_attn_varlen_func`.

    Unpads `q`/`k`/`v` to a flat `(total_valid, H, D)` layout, runs the
    FlashAttention varlen kernel, then scatters back. Skips all wasted compute
    on padded positions, but pays a per-layer gather/scatter — best when the
    mask is sparse enough that the skipped flops outweigh that cost.

    Constraints: requires CUDA + fp16/bf16 `q`/`k`/`v`. `attn_weights` is
    never materialized (returns `None`). Requires `flash-attn` installed.
    """

    @staticmethod
    def build_mask(attn_keys: torch.Tensor) -> FlashAttentionVarlenMask:
        B, T = attn_keys.shape
        seqlens = attn_keys.sum(dim=-1, dtype=torch.int32)  # (B,)
        cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
        indices = attn_keys.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        return FlashAttentionVarlenMask(
            cu_seqlens=cu_seqlens,
            max_seqlen=int(seqlens.max().item()),
            indices=indices,
            B=B,
            T=T,
        )

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: FlashAttentionVarlenMask | None,
    ) -> tuple[torch.Tensor, None]:
        assert mask is not None, "FlashVarlenAttention requires a non-None mask"
        B, H, T, D = q.shape

        def pack(x: torch.Tensor) -> torch.Tensor:
            # (B, H, T, D) -> (total_valid, H, D)
            return x.transpose(1, 2).reshape(B * T, H, D).index_select(0, mask.indices)

        out = flash_attn_varlen_func(
            pack(q),
            pack(k),
            pack(v),
            cu_seqlens_q=mask.cu_seqlens,
            cu_seqlens_k=mask.cu_seqlens,
            max_seqlen_q=mask.max_seqlen,
            max_seqlen_k=mask.max_seqlen,
        )  # (total_valid, H, D)

        out_padded = q.new_zeros(B * T, H, D)
        out_padded.index_copy_(0, mask.indices, out)
        return out_padded.reshape(B, T, H, D).transpose(1, 2), None


class AttentionImplementations(StrEnum):
    """Config-selectable attention backend. `.implementation` returns the class."""

    TORCH = "torch"
    SDPA = "sdpa"
    FLASH_VARLEN = "flash_varlen"

    @property
    def implementation(self) -> type[AttentionImplementation]:
        match self:
            case AttentionImplementations.TORCH:
                return TorchAttention
            case AttentionImplementations.SDPA:
                return SDPAAttention
            case AttentionImplementations.FLASH_VARLEN:
                if flash_attn_varlen_func is None:
                    raise ImportError("flash-attn is not installed")
                return FlashVarlenAttention

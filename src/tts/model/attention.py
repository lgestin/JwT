from enum import StrEnum
from typing import Protocol

import torch
import torch.nn.functional as F


class AttentionImplementation(Protocol):
    """A pluggable attention backend.

    Backends are stateless: both methods are static, so the implementation can
    be passed around as a bare class and selected per `forward` call.
    """

    @staticmethod
    def build_mask(attn_keys: torch.Tensor) -> torch.Tensor | None:
        """`attn_keys`: (B, T) bool, True = visible key. Returns the mask
        tensor consumed by `attention`, or None for an unmasked backend."""
        ...

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
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
        scale = q.shape[-1] ** -0.5
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        out = (attn_weights @ v.float()).to(v.dtype)
        return out, attn_weights


class AttentionImplementations(StrEnum):
    """Config-selectable attention backend. `.implementation` returns the class."""

    TORCH = "torch"
    SDPA = "sdpa"

    @property
    def implementation(self) -> type[AttentionImplementation]:
        match self:
            case AttentionImplementations.TORCH:
                return TorchAttention
            case AttentionImplementations.SDPA:
                return SDPAAttention

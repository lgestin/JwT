from enum import StrEnum
from typing import Protocol

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


def commit_rule(
    seq_mask: torch.Tensor,
    commit_index_k: torch.Tensor,
    commit_index_q: torch.Tensor,
) -> torch.Tensor:
    """Dense (B, Tq, Tk) visibility of key `k` from query `q`.

    `seq_mask[k] & (commit_index[k] <= commit_index[q])`, where `commit_index`
    is when a token freezes: 0 for the text prefix, `i + 1` for a clean
    acoustic frame `i`, `inf` for window frames and padding. A token therefore
    only sees tokens frozen no later than itself, which is what makes the
    frozen part of the sequence cacheable.
    """
    return seq_mask[:, None, :] & (
        commit_index_k[:, None, :] <= commit_index_q[:, :, None]
    )


type AttentionMask = torch.Tensor | BlockMask


class AttentionImplementation(Protocol):
    """A pluggable attention backend.

    Backends are stateless: both methods are static, so the implementation can
    be passed around as a bare class and selected per `forward` call.
    """

    @staticmethod
    def build_mask(
        seq_mask: torch.Tensor,
        commit_index_k: torch.Tensor,
        commit_index_q: torch.Tensor,
    ) -> AttentionMask:
        """`seq_mask`: (B, Tk) bool, True = visible key; `commit_index_k` (B, Tk)
        and `commit_index_q` (B, Tq) float, see `commit_rule`. Returns the mask
        object consumed by `attention`."""
        ...

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: AttentionMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`q`: (B, H, Tq, D), `k`, `v`: (B, H, Tk, D). Returns `(out, attn_weights)`.

        `out` is (B, H, Tq, D). `attn_weights` is the (B, H, Tq, Tk) softmax
        matrix when the backend can expose it, else `None` (the fused SDPA
        kernel never materializes it).
        """
        ...


class SDPAAttention(AttentionImplementation):
    """Attention via `F.scaled_dot_product_attention` — the fused kernel.

    Fast and memory-efficient, but the softmax matrix is never materialized,
    so `attention` returns `attn_weights=None`.
    """

    @staticmethod
    def build_mask(
        seq_mask: torch.Tensor,
        commit_index_k: torch.Tensor,
        commit_index_q: torch.Tensor,
    ) -> torch.Tensor:
        return commit_rule(seq_mask, commit_index_k, commit_index_q).unsqueeze(1)

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: AttentionMask | None,
    ) -> tuple[torch.Tensor, None]:
        assert mask is None or isinstance(mask, torch.Tensor)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None


class TorchAttention(AttentionImplementation):
    """Explicit attention in plain torch ops.

    Slower and materializes the (B, H, T, T) softmax matrix, but returns it as
    `attn_weights` so callers can inspect the attention distribution. Scores
    are computed in fp32 for a clean, autocast-independent map.
    """

    @staticmethod
    def build_mask(
        seq_mask: torch.Tensor,
        commit_index_k: torch.Tensor,
        commit_index_q: torch.Tensor,
    ) -> torch.Tensor:
        return commit_rule(seq_mask, commit_index_k, commit_index_q).unsqueeze(1)

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: AttentionMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert mask is None or isinstance(mask, torch.Tensor)
        scale = q.shape[-1] ** -0.5
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        out = (attn_weights @ v.float()).to(v.dtype)
        return out, attn_weights


# Compiled once per process; both are slow reference paths in eager.
compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
compiled_create_block_mask = torch.compile(create_block_mask)


class FlexAttention(AttentionImplementation):
    """Attention via `torch.nn.attention.flex_attention`.

    The commit rule is a `mask_mod` closure over the mask tensors; the block
    mask skips fully masked tiles, so one fused call covers the whole
    block-causal layout. `attn_weights` is never materialized (returns
    `None`). Requires CUDA and a head dim >= 16.
    """

    @staticmethod
    @torch.compiler.disable  # built outside any outer compiled graph
    def build_mask(
        seq_mask: torch.Tensor,
        commit_index_k: torch.Tensor,
        commit_index_q: torch.Tensor,
    ) -> BlockMask:
        B, Tk = seq_mask.shape
        Tq = commit_index_q.shape[1]

        def mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return seq_mask[b, kv_idx] & (
                commit_index_k[b, kv_idx] <= commit_index_q[b, q_idx]
            )

        return compiled_create_block_mask(
            mask_mod, B, None, Tq, Tk, device=seq_mask.device
        )

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: AttentionMask | None,
    ) -> tuple[torch.Tensor, None]:
        if mask is not None and not isinstance(mask, BlockMask):
            raise TypeError("FlexAttention requires a BlockMask")
        out = compiled_flex_attention(q, k, v, block_mask=mask)
        assert isinstance(out, torch.Tensor)
        return out, None


class AttentionImplementations(StrEnum):
    """Config-selectable attention backend. `.implementation` returns the class."""

    TORCH = "torch"
    SDPA = "sdpa"
    FLASH_VARLEN = "flash_varlen"
    FLEX = "flex"

    @property
    def implementation(self) -> type[AttentionImplementation]:
        match self:
            case AttentionImplementations.TORCH:
                return TorchAttention
            case AttentionImplementations.SDPA:
                return SDPAAttention
            case AttentionImplementations.FLASH_VARLEN:
                raise ValueError(
                    "FLASH_VARLEN is not available on this branch: flash-attn "
                    "cannot express the commit-rule mask, use FLEX"
                )
            case AttentionImplementations.FLEX:
                return FlexAttention

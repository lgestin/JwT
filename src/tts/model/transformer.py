import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from simple_parsing import Serializable

from tts.model.attention import AttentionImplementation, SDPAAttention


@dataclass
class TransformerConfig(Serializable):
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 10
    mlp_ratio: float = 4.0
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    time_freq_embed_dim: int = 256


def precompute_freqs_cis(
    seq_len: int, head_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """RoPE cos/sin table of shape (1, 1, seq_len, head_dim // 2, 2)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len)
    freqs = torch.outer(positions, freqs)
    freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cis.unsqueeze(0).unsqueeze(0)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate (q, k) of shape (B, H, L, D) by freqs_cis (1, 1, L, D // 2, 2)."""
    q_ = q.float().reshape(*q.shape[:-1], -1, 2)
    k_ = k.float().reshape(*k.shape[:-1], -1, 2)
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
    q_out = torch.stack(
        [cos * q_[..., 0] - sin * q_[..., 1], sin * q_[..., 0] + cos * q_[..., 1]],
        dim=-1,
    )
    k_out = torch.stack(
        [cos * k_[..., 0] - sin * k_[..., 1], sin * k_[..., 0] + cos * k_[..., 1]],
        dim=-1,
    )
    return q_out.flatten(-2).type_as(q), k_out.flatten(-2).type_as(k)


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Sinusoidal embedding for scalar timesteps. t (..., ) -> (..., dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.freq_embed_dim = freq_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.freq_embed_dim)
        return self.mlp(emb)


class AdaLN(nn.Module):
    """DiT-style modulation: silu -> linear producing (shift, scale, gate) x 2."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.linear(F.silu(t_emb)).chunk(6, dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x * rrms).to(dtype)
        return x * self.scale if self.scale is not None else x


class QKNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = RMSNorm(head_dim)
        self.key_norm = RMSNorm(head_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q).to(q), self.key_norm(k).to(k)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.qk_norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns `(out, attn_weights)`. `attn_weights` is the (B, H, T, T)
        softmax matrix for backends that expose it, else `None`."""
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.qk_norm(q, k)
        q, k = apply_rope(q, k, freqs_cis)
        out, attn_weights = attention_implementation.attention(q, k, v, mask)
        out = rearrange(out, "B H L D -> B L (H D)")
        return self.proj(out), attn_weights


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, affine=False)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim, affine=False)
        self.ff = FeedForward(dim, mlp_ratio)
        self.adaLN = AdaLN(dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        t_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(t_emb)
        attn_out, _ = self.attn(
            self.norm1(x) * (1 + scale_a) + shift_a,
            freqs_cis,
            mask,
            attention_implementation,
        )
        x = x + gate_a * attn_out
        x = x + gate_m * self.ff(self.norm2(x) * (1 + scale_m) + shift_m)
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.dim % config.num_heads == 0, "dim must be divisible by num_heads"
        self.config = config
        self.time_embedder = TimestepEmbedder(config.dim, config.time_freq_embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.dim, config.num_heads, config.mlp_ratio)
                for _ in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.dim)
        freqs_cis = precompute_freqs_cis(
            config.max_seq_len, config.dim // config.num_heads, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> torch.Tensor:
        """x: (B, L, D), t: (B, L) per-position timestep, mask: optional attn mask."""
        freqs_cis = self.freqs_cis[:, :, : x.shape[1]]
        t_emb = self.time_embedder(t)
        for block in self.blocks:
            x = block(x, freqs_cis, t_emb, mask, attention_implementation)
        return self.norm(x)

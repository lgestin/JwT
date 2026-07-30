import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from simple_parsing import Serializable

from jwt.model.attention import AttentionImplementation, SDPAAttention


@dataclass
class TransformerConfig(Serializable):
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 10
    mlp_ratio: float = 2.67
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    time_freq_embed_dim: int = 256


def precompute_freqs_cis(seq_len: int, dim: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,  # complex64
) -> tuple[torch.Tensor, torch.Tensor]:
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(-2)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


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
        for m in (self.mlp[0], self.mlp[2]):
            assert isinstance(m, nn.Linear)
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.freq_embed_dim)
        return self.mlp(emb)


class AdaLN(nn.Module):
    """DiT-style modulation: silu -> linear producing (shift, scale, gate) x 2.

    Gates are tanh-clamped to [-1, 1] so a single block's contribution can never
    dominate the residual stream.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.linear(
            F.silu(t_emb)
        ).chunk(6, dim=-1)
        return shift_a, scale_a, gate_a.tanh(), shift_m, scale_m, gate_m.tanh()


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


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, affine=False)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim, affine=False)
        self.ff = SwiGLUFFN(dim, mlp_ratio)
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
    freqs_cis: torch.Tensor

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
        freqs_cis = self.freqs_cis[: x.shape[1]]
        t_emb = self.time_embedder(t)
        for block in self.blocks:
            x = block(x, freqs_cis, t_emb, mask, attention_implementation)
        return self.norm(x)

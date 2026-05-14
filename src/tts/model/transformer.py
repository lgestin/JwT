from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from simple_parsing import Serializable


@dataclass
class TransformerConfig(Serializable):
    dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float = 4.0
    max_seq_len: int = 8192
    rope_theta: float = 10000.0


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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rrms).to(dtype) * self.scale


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
    ) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.qk_norm(q, k)
        q, k = apply_rope(q, k, freqs_cis)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = rearrange(x, "B H L D -> B L (H D)")
        return self.proj(x)


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
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.dim % config.num_heads == 0, "dim must be divisible by num_heads"
        self.config = config
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
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        freqs_cis = self.freqs_cis[:, :, : x.shape[1]]
        for block in self.blocks:
            x = block(x, freqs_cis, mask)
        return self.norm(x)

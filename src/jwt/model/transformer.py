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
    t: torch.Tensor,
    dim: int,
    resolution: float = 2 * math.pi / 1000,
    bandwidth: float = 10000.0,
) -> torch.Tensor:
    """Sinusoidal (Fourier) features for a scalar timestep.

    Each t is projected onto a bank of `dim // 2` sinusoids whose angular
    frequencies are geometrically spaced:

        f_max = 2 * pi / resolution                     # fastest channel
        f_k   = f_max * bandwidth ** (-k / half)        # k = 0 .. half-1
        f_min ~= f_max / bandwidth = 2 * pi / (resolution * bandwidth)

    Equivalently: the fastest channel completes one cycle every `resolution`
    units of t, and the slowest one every `resolution * bandwidth` units.

    Args:
        t: (...) tensor of timesteps, any shape — the feature axis is appended,
            so (B,) and the per-position (B, L) both work. `resolution` and
            `bandwidth` are interpreted in the *same units as t* (steps,
            seconds, [0,1], ...).
        dim: output width; must be even.
        resolution: period of the fastest channel = the fine end of the scale.
            Two timesteps much closer together than this map to almost the same
            vector, so this sets the finest distinction the embedding can carry.
            Set it to ~4-10x the smallest Delta t you need to distinguish (going
            all the way down to Delta t itself aliases: the top channel wraps a
            full cycle between neighbours and becomes constant).
        bandwidth: ratio f_max / f_min = the number of scales spanned, i.e. how
            far the coarse end sits above the fine end. `resolution * bandwidth`
            is the span over which the embedding is an unambiguous, monotone
            "where am I in the schedule" coordinate; beyond it every channel has
            wrapped at least once. Set bandwidth >= a few x (range of t) /
            resolution.

    Returns:
        (..., dim) tensor in [-1, 1], laid out as [cos | sin].

    Note:
        Frequency coverage is `half / log2(bandwidth)` channels per octave, so
        raising `bandwidth` without raising `dim` thins out the bank. The
        endpoint is exclusive, so the true f_min is bandwidth ** (1 / half)
        above f_max / bandwidth.

    Example:
        Defaults target continuous t in [0, 1] with ~1e-3 granularity:
        resolution ~= 6.3e-3 (about 6x the target Delta t, so no aliasing) and a
        coarsest period of 63, ~60x the range of t, so the low channels act as a
        smooth global ramp. For integer t in 0..999, use resolution ~= 5,
        bandwidth ~= 1e4.
    """
    assert dim % 2 == 0, f"timestep_embedding dim must be even, got {dim}"
    half = dim // 2
    f_max = 2 * math.pi / resolution
    i = torch.arange(half, dtype=torch.float32, device=t.device) / half
    freqs = f_max * torch.exp(-math.log(bandwidth) * i)
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        assert freq_embed_dim % 2 == 0, "freq_embed_dim must be even"
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
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.linear(F.silu(t_emb)).chunk(
            6, dim=-1
        )
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

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

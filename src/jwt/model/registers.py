from dataclasses import dataclass

import torch
import torch.nn as nn
from simple_parsing import Serializable

from jwt.model.attention import AttentionMask, FlashAttentionVarlenMask


@dataclass
class RegistersConfig(Serializable):
    n: int = 16
    starts_layer: int = 0


class Registers(nn.Module):
    """n learned tokens prepended to every sequence from `starts_layer` onward.

    The tokens carry no content of their own: they are scratch space attention
    can park on, so the real tokens are not forced to hoard global information
    in their own residual stream. They sit at the front, before position 0 of
    the real sequence, and `Transformer` strips them again before the output
    head, so the block stack is the only part of the model that sees them.
    """

    def __init__(self, dim: int, config: RegistersConfig):
        super().__init__()
        self.config = config
        self.starts_layer = config.starts_layer
        self.n = config.n
        self.registers = nn.Parameter(torch.randn(config.n, dim) * 0.02)

    def prepend(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        mask: AttentionMask | None,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMask | None, torch.Tensor]:
        n = self.config.n
        x_reg = self.registers.expand(x.size(0), -1, -1).to(x)
        x = torch.cat((x_reg, x), dim=1)
        t_emb_reg = t_emb.new_zeros((t_emb.size(0), n, t_emb.size(-1)))
        t_emb = torch.cat((t_emb_reg, t_emb), dim=1)
        mask = self.prepend_mask(mask)
        freqs_cis_reg = freqs_cis.new_ones((n, freqs_cis.size(1)))
        freqs_cis = torch.cat((freqs_cis_reg, freqs_cis), dim=0)
        return x, t_emb, mask, freqs_cis

    def prepend_mask(self, mask: AttentionMask | None) -> AttentionMask | None:
        n = self.config.n
        if mask is None:
            return None

        elif isinstance(mask, torch.Tensor):
            registers_mask = mask.new_ones((*mask.shape[:-1], n))
            return torch.cat((registers_mask, mask), dim=-1)

        elif isinstance(mask, FlashAttentionVarlenMask):
            B, T = mask.B, mask.T
            device = mask.indices.device
            # Every sample gains n valid tokens, so its packed segment grows by
            # n and sample b's segment start shifts by b * n.
            cu_seqlens = mask.cu_seqlens + n * torch.arange(
                B + 1, device=device, dtype=mask.cu_seqlens.dtype
            )
            # A valid position b * T + t in the padded (B, T) grid lands in the
            # (B, T + n) grid past the n registers now heading its own row.
            b, t = mask.indices // T, mask.indices % T
            tokens = b * (T + n) + n + t
            rows = torch.arange(B, device=device, dtype=mask.indices.dtype) * (T + n)
            registers = rows[:, None] + torch.arange(
                n, device=device, dtype=mask.indices.dtype
            )
            # Each row's registers sort ahead of that row's own tokens and
            # behind the previous row's, so sorting the union restores the
            # per-sample contiguous order `flash_attn_varlen_func` expects.
            indices = torch.cat((registers.reshape(-1), tokens)).sort().values
            return FlashAttentionVarlenMask(
                cu_seqlens=cu_seqlens,
                max_seqlen=mask.max_seqlen + n,
                indices=indices,
                B=B,
                T=T + n,
            )
        raise TypeError(f"unsupported attention mask type: {type(mask)}")

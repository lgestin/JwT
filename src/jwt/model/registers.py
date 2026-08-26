import torch
import torch.nn as nn
import torch.nn.functional as F


class Registers(nn.Module):
    """n learned tokens prepended to every sequence before the block stack.

    The tokens carry no content of their own: they are scratch space attention
    can park on, so the real tokens are not forced to hoard global information
    in their own residual stream. They sit at the front, before position 0 of
    the real sequence, and `Transformer` strips them again before the output
    head, so the block stack is the only part of the model that sees them.
    """

    def __init__(self, n: int, dim: int):
        super().__init__()
        self.n = n
        self.registers = nn.Parameter(torch.randn(n, dim) * 0.02)

    @torch.compiler.disable
    def prepend(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        seq_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        x_reg = self.registers.expand(x.size(0), -1, -1).to(x)
        x = torch.cat((x_reg, x), dim=1)
        t_emb_reg = t_emb.new_zeros((t_emb.size(0), self.n, t_emb.size(-1)))
        t_emb = torch.cat((t_emb_reg, t_emb), dim=1)
        if seq_mask is not None:
            seq_mask = F.pad(seq_mask, (self.n, 0), value=True)
        freqs_cis_reg = freqs_cis.new_ones((self.n, freqs_cis.size(1)))
        freqs_cis = torch.cat((freqs_cis_reg, freqs_cis), dim=0)
        return x, t_emb, seq_mask, freqs_cis

    @torch.compiler.disable
    def strip(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.n :], t_emb[:, self.n :]

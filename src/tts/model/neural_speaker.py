from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.model.transformer import Transformer, TransformerConfig


class NeuralSpeaker(Protocol):
    def speak(self, text: torch.LongTensor) -> torch.FloatTensor: ...


@dataclass
class RollingFlowConfig:
    transformer_config: TransformerConfig
    vocabulary_size: int
    mel_dim: int
    n_denoising_steps: int


@dataclass
class MaskedTensor:
    values: torch.Tensor
    mask: torch.BoolTensor

    def __post_init__(self):
        assert self.values.shape[-1] == self.mask.shape[-1]
        assert self.values.ndim == self.mask.ndim + 1

    @property
    def shape(self):
        return self.values.shape

    @property
    def masked_shape(self):
        return self.mask.sum(-1)


class RollingFlowSpeaker(NeuralSpeaker, nn.Module):
    def __init__(self, cfg: RollingFlowConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        dim = cfg.transformer_config.dim
        self.text_in = nn.Embedding(cfg.vocabulary_size, dim)
        self.mel_in = nn.Linear(cfg.mel_dim, dim)
        self.mel_out = nn.Linear(dim, cfg.mel_dim)
        self.text_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.mel_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.transformer = Transformer(cfg.transformer_config)

    def forward(
        self,
        text: MaskedTensor,
        mels: MaskedTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity at every mel position given text + (possibly noisy) mels.

        text.values: (B, 1, T_text)       text.mask: (B, T_text)
        mels.values: (B, mel_dim, T_mel)  mels.mask: (B, T_mel)
        t:           (B, T_mel)           per-mel-position timestep in [0, 1]
        returns:     (B, T_mel, mel_dim)  predicted velocity
        """
        B, mel_dim, T_mel = mels.values.shape
        text_ids = text.values.squeeze(-2)  # (B, T_text)
        T_text = text_ids.shape[-1]
        T = T_text + T_mel
        device = mels.values.device

        text_lens = text.mask.sum(-1)

        # Project both modalities into the transformer's hidden dim and tag them.
        text_lat = self.text_in(text_ids) + self.text_modality
        mel_lat = self.mel_in(mels.values.transpose(1, 2)) + self.mel_modality
        dim = text_lat.shape[-1]

        # Pack [real text | real mels | trailing pad] per sample.
        arange = torch.arange(T, device=device).expand(B, T)
        in_text = F.pad(text.mask, (0, T_mel))
        in_mels = F.pad(mels.mask, (T_text, 0))
        pack_idx = torch.where(
            in_text,
            arange,
            T_text + (arange - text_lens.unsqueeze(1)),
        ).clamp(min=0, max=T - 1)

        x_concat = torch.cat([text_lat, mel_lat], dim=1)
        x_packed = torch.gather(x_concat, 1, pack_idx.unsqueeze(-1).expand(B, T, dim))

        # Pack per-position t: text positions are always clean (t=1).
        t_concat = torch.cat(
            [torch.ones(B, T_text, device=device, dtype=t.dtype), t], dim=1
        )
        t_packed = torch.gather(t_concat, 1, pack_idx)

        attn_mask = (in_text | in_mels).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        out_packed = self.transformer(x_packed, t_packed, attn_mask)
        v_pred_packed = self.mel_out(out_packed)  # (B, T, mel_dim)

        # Unpack: mel position i in sample b lives at packed position text_lens[b] + i.
        mel_idx = torch.arange(T_mel, device=device).expand(B, T_mel)
        unpack_idx = (text_lens.unsqueeze(1) + mel_idx).clamp(max=T - 1)
        v_pred = torch.gather(
            v_pred_packed, 1, unpack_idx.unsqueeze(-1).expand(B, T_mel, mel_dim)
        )
        return v_pred

    def training_step(
        self,
        text: MaskedTensor,
        mels: MaskedTensor,
        *,
        mel_front: torch.LongTensor | None = None,
        x_0: torch.Tensor | None = None,
        n: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """Sample a rolling front + noise, run forward, and return loss inputs.

        Optional args let callers pin the random choices for reproducibility:
        - mel_front: (B,) long, where each sample's denoising front lands
        - x_0:       (B, T_mel, mel_dim), the noise tensor mixed with x_1
        - n:         override for cfg.n_denoising_steps

        Returns (v_pred, target, loss_mask) all aligned in (B, T_mel, *) layout.
        """
        B, mel_dim, T_mel = mels.values.shape
        device = mels.values.device
        n = n if n is not None else self.cfg.n_denoising_steps
        mel_lens = mels.mask.sum(-1)

        if mel_front is None:
            u = torch.rand(B, device=device)
            mel_front = (u * mel_lens.float()).long()
        x_1 = mels.values.transpose(1, 2)  # (B, T_mel, mel_dim)
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        mel_idx = torch.arange(T_mel, device=device).expand(B, T_mel)
        t = torch.clamp(
            1.0 - (mel_idx - mel_front.unsqueeze(1)).float() / (n - 1),
            0.0,
            1.0,
        )  # (B, T_mel)

        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        target = x_1 - x_0

        noisy_mels = MaskedTensor(values=x_t.transpose(1, 2), mask=mels.mask)
        v_pred = self.forward(text, noisy_mels, t)

        # Supervise only the active rolling window: ramp (t < 1) + first t=0 position.
        loss_mask = (
            mels.mask
            & (mel_idx > mel_front.unsqueeze(1))
            & (mel_idx <= mel_front.unsqueeze(1) + n)
        )
        return v_pred, target, loss_mask

    @torch.no_grad()
    def speak(
        self,
        text: MaskedTensor,
        mel_lens: torch.LongTensor,
        *,
        x_0: torch.Tensor | None = None,
    ) -> MaskedTensor:
        """Generate mels for each sample by rolling-Euler integration of the flow.

        text:     MaskedTensor — values (B, 1, T_text), mask (B, T_text)
        mel_lens: (B,) long, desired mel length per sample
        x_0:      optional (B, mel_dim, T_max) noise override for reproducibility

        Returns a MaskedTensor with values (B, mel_dim, T_max) and mask (B, T_max)
        where T_max = mel_lens.max(). Padding positions are False in the mask.
        """
        B = text.values.shape[0]
        T_max = int(mel_lens.max().item())
        device = text.values.device
        n = self.cfg.n_denoising_steps
        dt = 1.0 / (n - 1)

        if x_0 is None:
            x_0 = torch.randn(B, self.cfg.mel_dim, T_max, device=device)
        mels_values = x_0.clone()

        mel_idx = torch.arange(T_max, device=device).expand(B, T_max)
        mels_mask = mel_idx < mel_lens.unsqueeze(1)

        for k in range(T_max + n - 2):
            front = k - (n - 1)
            t = torch.clamp(
                1.0 - (mel_idx - front).float() / (n - 1),
                0.0,
                1.0,
            )  # (B, T_max)

            mels_mt = MaskedTensor(values=mels_values, mask=mels_mask)
            v_pred = self.forward(text, mels_mt, t)  # (B, T_max, mel_dim)

            update = (v_pred * dt) * (t < 1.0).unsqueeze(-1)
            mels_values = mels_values + update.transpose(1, 2)

        return MaskedTensor(values=mels_values, mask=mels_mask)

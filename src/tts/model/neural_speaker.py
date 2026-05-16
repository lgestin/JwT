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

    def forward(self, text: MaskedTensor, mels: MaskedTensor):
        # text.values: (B, 1, T_text)      text.mask: (B, T_text)
        # mels.values: (B, mel_dim, T_mel) mels.mask: (B, T_mel)
        B, mel_dim, T_mel = mels.values.shape
        text_ids = text.values.squeeze(-2)  # (B, T_text)
        T_text = text_ids.shape[-1]
        T = T_text + T_mel
        device = mels.values.device
        n = self.cfg.n_denoising_steps

        text_lens = text.mask.sum(-1)
        mel_lens = mels.mask.sum(-1)

        # Mel-space flow on the original (unpacked) mel layout.
        x_1 = mels.values.transpose(1, 2)  # (B, T_mel, mel_dim)
        x_0 = torch.randn_like(x_1)

        u = torch.rand(B, device=device)
        mel_front = (u * mel_lens.float()).long()  # (B,) front index in mel layout

        mel_idx = torch.arange(T_mel, device=device).expand(B, T_mel)
        t_mel = torch.clamp(
            1.0 - (mel_idx - mel_front.unsqueeze(1)).float() / (n - 1),
            0.0,
            1.0,
        )  # (B, T_mel)
        x_t_mel = (1 - t_mel.unsqueeze(-1)) * x_0 + t_mel.unsqueeze(-1) * x_1
        target = x_1 - x_0  # (B, T_mel, mel_dim)

        # Project both modalities into the transformer's hidden dim and tag them.
        text_lat = self.text_in(text_ids) + self.text_modality  # (B, T_text, dim)
        mel_lat = self.mel_in(x_t_mel) + self.mel_modality  # (B, T_mel, dim)
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

        x_concat = torch.cat([text_lat, mel_lat], dim=1)  # (B, T, dim)
        x_packed = torch.gather(x_concat, 1, pack_idx.unsqueeze(-1).expand(B, T, dim))

        # Per-position timesteps in packed layout. Text positions sit before the
        # front and clamp to t=1 (clean) automatically.
        front_packed = text_lens + mel_front
        t_packed = torch.clamp(
            1.0 - (arange - front_packed.unsqueeze(1)).float() / (n - 1),
            0.0,
            1.0,
        )  # (B, T)

        attn_mask = (in_text | in_mels).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        out_packed = self.transformer(x_packed, t_packed, attn_mask)  # (B, T, dim)
        v_pred_packed = self.mel_out(out_packed)  # (B, T, mel_dim)

        # Unpack: mel position i in sample b lives at packed position text_lens[b] + i.
        unpack_idx = (text_lens.unsqueeze(1) + mel_idx).clamp(max=T - 1)  # (B, T_mel)
        v_pred = torch.gather(
            v_pred_packed, 1, unpack_idx.unsqueeze(-1).expand(B, T_mel, mel_dim)
        )  # (B, T_mel, mel_dim)

        return v_pred, target, mels.mask

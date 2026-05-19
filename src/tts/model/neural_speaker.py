from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tts.model.transformer import (
    Transformer,
    TransformerConfig,
)


class NeuralSpeaker(Protocol):
    def speak(self, text: "MaskedTensor") -> "MaskedTensor": ...


@dataclass
class RollingFlowConfig:
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    vocabulary_size: int = 0
    mel_dim: int = 100
    n_denoising_steps: int = 32
    max_mel_len: int = 2048
    eos_n_frames: int = 3
    eos_mel_value: float = -15.0
    eos_detect_threshold: float = -11.0


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


_LJSPEECH_LOG_MEL_MEAN = -5.896610
_LJSPEECH_LOG_MEL_STD = 2.226763


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
        # Global log-mel stats (LJSpeech 24 kHz, BigVGAN log-mels). The model
        # operates in normalized space; speak() denormalizes before returning.
        self.register_buffer(
            "mel_mean", torch.tensor(_LJSPEECH_LOG_MEL_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "mel_std", torch.tensor(_LJSPEECH_LOG_MEL_STD, dtype=torch.float32)
        )

    def forward(
        self,
        text: MaskedTensor,
        mels: MaskedTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity at every mel position.

        text.values: (B, 1, T_text)       text.mask: (B, T_text)
        mels.values: (B, mel_dim, T_mel)  mels.mask: (B, T_mel)
        t:           (B, T_mel)           per-mel-position timestep in [0, 1]
        returns:
            v_pred: (B, T_mel, mel_dim)   predicted velocity
        """
        B, mel_dim, T_mel = mels.values.shape
        text_ids = text.values.squeeze(-2)  # (B, T_text)
        T_text = text_ids.shape[-1]
        T = T_text + T_mel
        device = mels.values.device

        text_lens = text.mask.sum(-1)
        mel_lens = mels.mask.sum(-1)
        total_lens = text_lens + mel_lens

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

        # Keep masks in packed coords. in_text/in_mels above are in original
        # coords and silently misalign the attention mask when text is padded.
        in_real_packed = arange < total_lens.unsqueeze(1)
        in_mels_packed = (arange >= text_lens.unsqueeze(1)) & in_real_packed

        # Attention keys: visible up to and including the first real t=0 (the
        # "next frontier"). Pure-noise positions beyond it carry no signal
        # and would only distract attention.
        is_zero_real = (t_packed == 0.0) & in_mels_packed
        keep_first_zero = is_zero_real.cumsum(-1) <= 1
        attn_keys = in_real_packed & keep_first_zero  # (B, T)
        attn_mask = attn_keys.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.BoolTensor,
        torch.Tensor,
    ]:
        """Append EOS sentinel frames, sample a rolling front, run forward, return loss inputs.

        Optional args let callers pin the random choices for reproducibility:
        - mel_front: (B,) long, where each sample's denoising front lands;
          defaults to a uniform sample in [-(n-1), mel_lens+eos_n_frames) so
          negative values reproduce the inference warm-up distribution
        - x_0:       (B, T_mel+eos_n_frames, mel_dim), the noise tensor
        - n:         override for cfg.n_denoising_steps

        Returns (v_pred, v_target, v_mask, t):
        - v_pred:   (B, T_mel+eos_n_frames, mel_dim)  predicted velocity
        - v_target: (B, T_mel+eos_n_frames, mel_dim)  flow-matching velocity target
        - v_mask:   (B, T_mel+eos_n_frames)           supervision mask for the ramp
        - t:        (B, T_mel+eos_n_frames)           per-position rolling timestep
        """
        B, mel_dim, T_mel = mels.values.shape
        device = mels.values.device
        n = n if n is not None else self.cfg.n_denoising_steps
        mel_lens = mels.mask.sum(-1)

        eos_n = self.cfg.eos_n_frames
        eos_val = self.cfg.eos_mel_value
        T_mel_ext = T_mel + eos_n

        # Build extended values: real frames followed by sentinel, then padding.
        # values_ext starts as the padded real frames extended by eos_n zero columns.
        values_ext = F.pad(mels.values, (0, eos_n))  # (B, mel_dim, T_mel_ext)
        mel_idx_ext = torch.arange(T_mel_ext, device=device).unsqueeze(0)  # (1, T_mel_ext)
        in_sentinel = (mel_idx_ext >= mel_lens.unsqueeze(1)) & (
            mel_idx_ext < (mel_lens + eos_n).unsqueeze(1)
        )  # (B, T_mel_ext)
        values_ext = torch.where(
            in_sentinel.unsqueeze(1),
            torch.full_like(values_ext, eos_val),
            values_ext,
        )

        in_real = mel_idx_ext < mel_lens.unsqueeze(1)  # (B, T_mel_ext)
        mask_ext = in_real | in_sentinel  # (B, T_mel_ext)

        mel_lens_ext = mel_lens + eos_n

        if mel_front is None:
            u = torch.rand(B, device=device)
            mel_front = (u * (mel_lens_ext.float() + (n - 1)) - (n - 1)).long()

        # Normalize to ~N(0,1) so noise and signal share scale.
        x_1 = (values_ext.transpose(1, 2) - self.mel_mean) / self.mel_std
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        mel_idx = torch.arange(T_mel_ext, device=device).expand(B, T_mel_ext)
        t = torch.clamp(
            1.0 - (mel_idx - mel_front.unsqueeze(1)).float() / (n - 1),
            0.0,
            1.0,
        )  # (B, T_mel_ext)

        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        v_target = x_1 - x_0

        noisy_mels = MaskedTensor(values=x_t.transpose(1, 2), mask=mask_ext)
        v_pred = self.forward(text, noisy_mels, t)

        v_mask = (
            mask_ext
            & (mel_idx > mel_front.unsqueeze(1))
            & (mel_idx < mel_front.unsqueeze(1) + n)
        )

        return v_pred, v_target, v_mask, t

    @torch.no_grad()
    def speak(
        self,
        text: MaskedTensor,
        *,
        x_0: torch.Tensor | None = None,
    ) -> MaskedTensor:
        """Generate mels via rolling-Euler integration, stopping on the EOS sentinel.

        Each generated frame is checked once it is fully denoised (t → 1). When
        its denormalized mean drops below cfg.eos_detect_threshold the loop marks
        that sample done and records the trim position. The loop exits once all
        samples are done or cfg.max_mel_len frames have been added.

        text: MaskedTensor — values (B, 1, T_text), mask (B, T_text)
        x_0:  optional (B, mel_dim, max_mel_len) noise override for reproducibility

        Returns a MaskedTensor with values (B, mel_dim, T_out) where T_out is the
        longest trim across the batch. Each sample's mask sums to its trim length.
        """
        B = text.values.shape[0]
        device = text.values.device
        n = self.cfg.n_denoising_steps
        dt = 1.0 / (n - 1)
        mel_dim = self.cfg.mel_dim
        max_T = self.cfg.max_mel_len

        if x_0 is None:
            x_0 = torch.randn(B, mel_dim, max_T, device=device)
        else:
            assert x_0.shape[-1] >= max_T, (
                f"x_0 must have at least cfg.max_mel_len ({max_T}) frames, "
                f"got {x_0.shape[-1]}"
            )

        mels_values = torch.empty(B, mel_dim, 0, device=device, dtype=x_0.dtype)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        trim = torch.full((B,), -1, dtype=torch.long, device=device)

        for k in range(max_T + n - 1):
            if k < max_T:
                mels_values = torch.cat([mels_values, x_0[..., k : k + 1]], dim=-1)

            L = mels_values.shape[-1]
            mel_idx = torch.arange(L, device=device).expand(B, L)
            buffer_mask = torch.ones(B, L, dtype=torch.bool, device=device)

            t = torch.clamp((k - mel_idx).float() / (n - 1), 0.0, 1.0)

            mels_mt = MaskedTensor(values=mels_values, mask=buffer_mask)
            v_pred = self.forward(text, mels_mt, t)

            in_window = (t < 1.0) & buffer_mask
            update = (v_pred * dt) * in_window.unsqueeze(-1)
            mels_values = mels_values + update.transpose(1, 2)

            # Check the frame that just reached t=1 for the EOS sentinel.
            if k >= n - 1:
                p = k - (n - 1)
                frame_raw = mels_values[:, :, p] * self.mel_std + self.mel_mean
                triggered = (~done) & (frame_raw.mean(dim=-1) < self.cfg.eos_detect_threshold)
                trim[triggered] = p
                done |= triggered

            if done.all():
                break

        # Samples that hit max_T without triggering: scan for first below-threshold frame.
        if not done.all():
            frames_raw = mels_values * self.mel_std + self.mel_mean  # (B, mel_dim, L)
            frame_means = frames_raw.mean(dim=1)  # (B, L)
            L = mels_values.shape[-1]
            for b in range(B):
                if trim[b] == -1:
                    below = (frame_means[b] < self.cfg.eos_detect_threshold).nonzero(
                        as_tuple=True
                    )[0]
                    trim[b] = int(below[0].item()) if len(below) > 0 else L

        trim = trim.clamp(min=0, max=max_T)
        T_out = int(trim.max().item())
        T_out = max(T_out, 1)

        mels_out = mels_values[..., :T_out]
        mel_idx_out = torch.arange(T_out, device=device).expand(B, T_out)
        mask = mel_idx_out < trim.unsqueeze(1)

        mels_out = mels_out * self.mel_std + self.mel_mean
        return MaskedTensor(values=mels_out, mask=mask)

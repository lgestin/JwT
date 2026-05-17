from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.model.transformer import Transformer, TransformerConfig


class NeuralSpeaker(Protocol):
    def speak(self, text: "MaskedTensor") -> "MaskedTensor": ...


@dataclass
class RollingFlowConfig:
    transformer_config: TransformerConfig
    vocabulary_size: int
    mel_dim: int
    n_denoising_steps: int
    remaining_loss_weight: float = 1.0
    max_mel_len: int = 2048
    stop_threshold: float = 1.0


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
        self.frames_remaining_head = nn.Linear(dim, 1)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity and log1p(remaining frames) at every mel position.

        text.values: (B, 1, T_text)       text.mask: (B, T_text)
        mels.values: (B, mel_dim, T_mel)  mels.mask: (B, T_mel)
        t:           (B, T_mel)           per-mel-position timestep in [0, 1]
        returns:
            v_pred: (B, T_mel, mel_dim)   predicted velocity
            r_pred: (B, T_mel)            predicted log1p(remaining frames)
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

        # Attention keys: visible up to and including the first real t=0 (the
        # "next frontier"). Pure-noise positions beyond it carry no signal
        # and would only distract attention.
        is_zero_real = (t_packed == 0.0) & in_mels
        keep_first_zero = is_zero_real.cumsum(-1) <= 1
        attn_keys = (in_text | in_mels) & keep_first_zero  # (B, T)
        attn_mask = attn_keys.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        out_packed = self.transformer(x_packed, t_packed, attn_mask)
        v_pred_packed = self.mel_out(out_packed)  # (B, T, mel_dim)
        r_pred_packed = self.frames_remaining_head(out_packed).squeeze(-1)  # (B, T)

        # Unpack: mel position i in sample b lives at packed position text_lens[b] + i.
        mel_idx = torch.arange(T_mel, device=device).expand(B, T_mel)
        unpack_idx = (text_lens.unsqueeze(1) + mel_idx).clamp(max=T - 1)
        v_pred = torch.gather(
            v_pred_packed, 1, unpack_idx.unsqueeze(-1).expand(B, T_mel, mel_dim)
        )
        r_pred = torch.gather(r_pred_packed, 1, unpack_idx)
        return v_pred, r_pred

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
        torch.Tensor,
        torch.BoolTensor,
        torch.Tensor,
    ]:
        """Sample a rolling front + noise, run forward, and return loss inputs.

        Optional args let callers pin the random choices for reproducibility:
        - mel_front: (B,) long, where each sample's denoising front lands
        - x_0:       (B, T_mel, mel_dim), the noise tensor mixed with x_1
        - n:         override for cfg.n_denoising_steps

        Returns (v_pred, v_target, v_mask, r_pred, r_target, r_mask, t):
        - v_pred:   (B, T_mel, mel_dim)  predicted velocity
        - v_target: (B, T_mel, mel_dim)  flow-matching velocity target (x_1 - x_0)
        - v_mask:   (B, T_mel)           supervision mask for the velocity ramp
        - r_pred:   (B, T_mel)           predicted log1p(remaining frames)
        - r_target: (B, T_mel)           log1p(remaining frames) ground truth
        - r_mask:   (B, T_mel)           supervision mask: real frames with t=1
        - t:        (B, T_mel)           per-position rolling timestep in [0, 1]
        """
        B, mel_dim, T_mel = mels.values.shape
        device = mels.values.device
        n = n if n is not None else self.cfg.n_denoising_steps
        mel_lens = mels.mask.sum(-1)

        if mel_front is None:
            u = torch.rand(B, device=device)
            mel_front = (u * mel_lens.float()).long()
        # Normalize log-mels to roughly N(0, 1) so noise and signal share scale.
        x_1 = (mels.values.transpose(1, 2) - self.mel_mean) / self.mel_std
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        mel_idx = torch.arange(T_mel, device=device).expand(B, T_mel)
        t = torch.clamp(
            1.0 - (mel_idx - mel_front.unsqueeze(1)).float() / (n - 1),
            0.0,
            1.0,
        )  # (B, T_mel)

        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        v_target = x_1 - x_0

        noisy_mels = MaskedTensor(values=x_t.transpose(1, 2), mask=mels.mask)
        v_pred, r_pred = self.forward(text, noisy_mels, t)

        # Supervise the active rolling window: ramp (0 < t < 1) + first t=0.
        # Positions at relative-n from the front (the second t=0) are never
        # touched by the inference Euler loop, so supervising them trains a
        # behavior the model would never use.
        v_mask = (
            mels.mask
            & (mel_idx > mel_front.unsqueeze(1))
            & (mel_idx < mel_front.unsqueeze(1) + n)
        )

        # Remaining-frames head: supervise only at clean (t=1) positions —
        # the trailing edge of the rolling front, which is exactly where
        # speak() reads the head. Noisy positions carry no useful "remaining"
        # signal and would train a behavior inference never uses.
        remaining = (mel_lens.unsqueeze(1) - mel_idx - 1).clamp(min=0).float()
        r_target = torch.log1p(remaining)
        r_mask = mels.mask & (t == 1.0)

        return v_pred, v_target, v_mask, r_pred, r_target, r_mask, t

    @torch.no_grad()
    def speak(
        self,
        text: MaskedTensor,
        *,
        x_0: torch.Tensor | None = None,
    ) -> MaskedTensor:
        """Generate mels by rolling-Euler integration with autonomous stopping.

        Per-sample length is decided by the frames-remaining head: once a
        sample's predicted remaining (read at the trailing clean edge of the
        rolling front) drops below cfg.stop_threshold, that sample's length
        is frozen. The batched buffer keeps growing up to the longest decided
        length (or cfg.max_mel_len) to keep attention context consistent.

        text:     MaskedTensor — values (B, 1, T_text), mask (B, T_text)
        x_0:      optional (B, mel_dim, max_mel_len) noise override for
                  reproducibility (sliced one frame per step as the buffer grows)

        Returns a MaskedTensor with values (B, mel_dim, T_out) and mask
        (B, T_out) where T_out = decided_len.max(). Each sample's mask sums
        to its predicted length; positions beyond that are False.
        """
        B = text.values.shape[0]
        device = text.values.device
        n = self.cfg.n_denoising_steps
        dt = 1.0 / (n - 1)
        mel_dim = self.cfg.mel_dim
        max_T = self.cfg.max_mel_len
        stop_threshold = self.cfg.stop_threshold

        if x_0 is None:
            x_0 = torch.randn(B, mel_dim, max_T, device=device)
        else:
            assert x_0.shape[-1] >= max_T, (
                f"x_0 must have at least cfg.max_mel_len ({max_T}) frames along "
                f"the time axis, got {x_0.shape[-1]}"
            )

        # decided_len: per-sample predicted length. Sentinel max_T means undecided.
        decided_len = torch.full((B,), max_T, dtype=torch.long, device=device)

        mels_values = torch.empty(B, mel_dim, 0, device=device, dtype=x_0.dtype)

        for k in range(max_T + n - 2):
            # Grow the buffer by one new noise frame — the new t=0 frontier.
            if k < max_T:
                mels_values = torch.cat([mels_values, x_0[..., k : k + 1]], dim=-1)

            L = mels_values.shape[-1]
            mel_idx = torch.arange(L, device=device).expand(B, L)
            # Once a sample's stop is decided, drop positions past it so
            # in_mels matches the training distribution (true length).
            buffer_mask = mel_idx < decided_len.clamp(max=L).unsqueeze(1)

            # t per position = (steps since it was added) / (n - 1), clamped.
            t = torch.clamp((k - mel_idx).float() / (n - 1), 0.0, 1.0)

            mels_mt = MaskedTensor(values=mels_values, mask=buffer_mask)
            v_pred, r_pred = self.forward(text, mels_mt, t)

            in_window = (t < 1.0) & buffer_mask
            update = (v_pred * dt) * in_window.unsqueeze(-1)
            mels_values = mels_values + update.transpose(1, 2)

            # Read remaining only at the trailing clean (t=1) edge of the
            # rolling front — matches the training supervision mask. During
            # warmup (k < n-1) no position is fully clean yet, so skip.
            if k < n - 1:
                continue
            p = k - (n - 1)
            r_at_p = r_pred[:, p]  # (B,)
            predicted_remaining = torch.expm1(r_at_p)
            newly_decided = (decided_len == max_T) & (
                predicted_remaining < stop_threshold
            )
            decided_len = torch.where(
                newly_decided,
                torch.full_like(decided_len, p + 1),
                decided_len,
            )

            if bool((decided_len < max_T).all()):
                break

        T_out = int(decided_len.max().item())
        mels_values = mels_values[..., :T_out]
        mel_idx = torch.arange(T_out, device=device).expand(B, T_out)
        mask = mel_idx < decided_len.unsqueeze(1)

        # Denormalize from N(0, 1) space to log-mel scale for downstream consumers.
        mels_values = mels_values * self.mel_std + self.mel_mean
        return MaskedTensor(values=mels_values, mask=mask)

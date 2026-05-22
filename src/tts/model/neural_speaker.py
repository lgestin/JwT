from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.data.audio.codecs import Codec, Codecs
from tts.model.attention import AttentionImplementation, SDPAAttention
from tts.model.flow import (
    FlowParametrizations,
    JustWaveformTransformersParametrization,
    RectifiedFlowParametrization,
)
from tts.model.transformer import (
    Transformer,
    TransformerConfig,
)
from tts.training.timestep_schedules import TimestepSchedules


class NeuralSpeaker(Protocol):
    def speak(self, text: "MaskedTensor", codec: Codec) -> "MaskedTensor": ...


@dataclass
class RollingFlowConfig:
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    vocabulary_size: int = 0
    codec: Codecs = Codecs.BIGVGAN
    parametrization: FlowParametrizations = FlowParametrizations.RECTIFIED_FLOW
    timestep_schedule: TimestepSchedules = TimestepSchedules.LINEAR
    acoustic_dim: int = 100
    n_denoising_steps: int = 32
    max_acoustic_len: int = 2048
    eos_n_frames: int = 3
    noise_scale: float = 1.0


def _per_pos_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-position MSE: reduces only over the last (acoustic_dim) axis."""
    return (a - b).pow(2).mean(-1)


def _per_pos_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-position L1: reduces only over the last (acoustic_dim) axis."""
    return (a - b).abs().mean(-1)


# Mirror flow.py's default loss_fn per parametrization, but per-position so the
# trainer can apply the rolling-window v_mask before reducing.
_PER_POS_LOSS_FN = {
    RectifiedFlowParametrization: _per_pos_mse,
    JustWaveformTransformersParametrization: _per_pos_l1,
}


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


@dataclass
class TrainingStepOutput:
    """Result of `RollingFlowSpeaker.training_step` — all fields are GPU tensors.

    `per_pos_loss` is the parametrization's per-position loss *before* the
    rolling-window mask is applied; the trainer reuses it to bin the loss by
    timestep without recomputing anything.
    """

    loss: torch.Tensor          # scalar, masked-mean of per_pos_loss
    x_pred: torch.Tensor        # (B, T_ext, acoustic_dim) — recovered x_1
    v_mask: torch.Tensor        # (B, T_ext) bool — rolling-window supervision mask
    t: torch.Tensor             # (B, T_ext) — per-position timestep in [0, 1]
    per_pos_loss: torch.Tensor  # (B, T_ext) — per-position loss before masking


class RollingFlowSpeaker(NeuralSpeaker, nn.Module):
    def __init__(self, cfg: RollingFlowConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        # Resolve the parametrization class once — it's a static dispatch table,
        # not an instance, so this stays free of training state.
        self.param = cfg.parametrization.parametrization
        self._per_pos_loss_fn = _PER_POS_LOSS_FN[self.param]
        self.schedule = cfg.timestep_schedule.schedule
        dim = cfg.transformer_config.dim
        self.text_in = nn.Embedding(cfg.vocabulary_size, dim)
        self.acoustic_in = nn.Linear(cfg.acoustic_dim, dim)
        self.acoustic_out = nn.Linear(dim, cfg.acoustic_dim)
        self.text_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.acoustic_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.transformer = Transformer(cfg.transformer_config)

    def forward(
        self,
        text: MaskedTensor,
        acoustic: MaskedTensor,
        t: torch.Tensor,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> torch.Tensor:
        """Run a forward pass and return the raw model output.

        The semantic meaning of the returned tensor depends on the configured
        parametrization (velocity for RectifiedFlow, x_1 for JWT). The model
        itself is parametrization-agnostic; the parametrization class converts
        this tensor into a loss (during training) or a velocity (during
        sampling).

        text.values:     (B, 1, T_text)             text.mask: (B, T_text)
        acoustic.values: (B, acoustic_dim, T_ac)    acoustic.mask: (B, T_ac)
        t:               (B, T_ac)                  per-acoustic-position timestep in [0, 1]
        returns:
            pred: (B, T_ac, acoustic_dim)            raw model output
        """
        B, acoustic_dim, T_ac = acoustic.values.shape
        text_ids = text.values.squeeze(-2)  # (B, T_text)
        T_text = text_ids.shape[-1]
        T = T_text + T_ac
        device = acoustic.values.device

        text_lens = text.mask.sum(-1)
        acoustic_lens = acoustic.mask.sum(-1)
        total_lens = text_lens + acoustic_lens

        # Project both modalities into the transformer's hidden dim and tag them.
        text_lat = self.text_in(text_ids) + self.text_modality
        acoustic_lat = (
            self.acoustic_in(acoustic.values.transpose(1, 2)) + self.acoustic_modality
        )
        dim = text_lat.shape[-1]

        # Pack [real text | real acoustic | trailing pad] per sample.
        arange = torch.arange(T, device=device).expand(B, T)
        in_text = F.pad(text.mask, (0, T_ac))
        in_ac = F.pad(acoustic.mask, (T_text, 0))
        pack_idx = torch.where(
            in_text,
            arange,
            T_text + (arange - text_lens.unsqueeze(1)),
        ).clamp(min=0, max=T - 1)

        x_concat = torch.cat([text_lat, acoustic_lat], dim=1)
        x_packed = torch.gather(x_concat, 1, pack_idx.unsqueeze(-1).expand(B, T, dim))

        # Pack per-position t: text positions are always clean (t=1).
        t_concat = torch.cat(
            [torch.ones(B, T_text, device=device, dtype=t.dtype), t], dim=1
        )
        t_packed = torch.gather(t_concat, 1, pack_idx)

        # Keep masks in packed coords. in_text/in_ac above are in original
        # coords and silently misalign the attention mask when text is padded.
        in_real_packed = arange < total_lens.unsqueeze(1)
        in_ac_packed = (arange >= text_lens.unsqueeze(1)) & in_real_packed

        # Attention keys: visible up to and including the first real t=0 (the
        # "next frontier"). Pure-noise positions beyond it carry no signal
        # and would only distract attention.
        is_zero_real = (t_packed == 0.0) & in_ac_packed
        keep_first_zero = is_zero_real.cumsum(-1) <= 1
        attn_keys = in_real_packed & keep_first_zero  # (B, T)
        attn_mask = attention_implementation.build_mask(attn_keys)  # (B, 1, 1, T)
        out_packed = self.transformer(
            x_packed, t_packed, attn_mask, attention_implementation
        )
        pred_packed = self.acoustic_out(out_packed)  # (B, T, acoustic_dim)

        # Unpack: acoustic position i in sample b lives at packed position text_lens[b] + i.
        ac_idx = torch.arange(T_ac, device=device).expand(B, T_ac)
        unpack_idx = (text_lens.unsqueeze(1) + ac_idx).clamp(max=T - 1)
        pred = torch.gather(
            pred_packed, 1, unpack_idx.unsqueeze(-1).expand(B, T_ac, acoustic_dim)
        )
        return pred

    def _sample_noise(
        self,
        shape: tuple[int, ...] | torch.Size,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw the x_0 prior — a Gaussian scaled by `cfg.noise_scale`.

        Shared by `training_step` and `speak` so the corrupting noise has the
        same distribution at train and inference time.
        """
        return self.cfg.noise_scale * torch.randn(shape, device=device, dtype=dtype)

    def training_step(
        self,
        text: MaskedTensor,
        acoustic: MaskedTensor,
        *,
        acoustic_front: torch.LongTensor | None = None,
        x_0: torch.Tensor | None = None,
        n: int | None = None,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> TrainingStepOutput:
        """Sample a rolling front, run forward, return masked loss + x_pred.

        The trainer is expected to have already appended EOS sentinel frames and
        normalized the values. `acoustic.values` is therefore (B, acoustic_dim,
        T_ext) in normalized space with `acoustic.mask` covering real + sentinel.

        Optional args let callers pin the random choices for reproducibility:
        - acoustic_front: (B,) long, where each sample's denoising front lands;
          defaults to a uniform sample in [-(n-1), acoustic_lens_ext) so
          negative values reproduce the inference warm-up distribution
        - x_0: (B, T_ext, acoustic_dim), the noise tensor
        - n: override for cfg.n_denoising_steps

        `attention_implementation` selects the attention backend (default fused
        SDPA); pass `TorchAttention` to expose attention weights for probing.

        Returns a `TrainingStepOutput`:
        - loss:         scalar — masked-mean of the parametrization's per-position loss
        - x_pred:       (B, T_ext, acoustic_dim) — predicted x_1 (normalized)
                        recovered by the parametrization; used for codec-agnostic
                        monitoring
        - v_mask:       (B, T_ext) — supervision mask for the rolling window
        - t:            (B, T_ext) — per-position rolling timestep
        - per_pos_loss: (B, T_ext) — per-position loss before masking
        """
        B, acoustic_dim, T_ext = acoustic.values.shape
        device = acoustic.values.device
        n = n if n is not None else self.cfg.n_denoising_steps
        acoustic_lens_ext = acoustic.mask.sum(-1)

        if acoustic_front is None:
            u = torch.rand(B, device=device)
            acoustic_front = (
                u * (acoustic_lens_ext.float() + (n - 1)) - (n - 1)
            ).long()

        x_1 = acoustic.values.transpose(1, 2)  # (B, T_ext, acoustic_dim), normalized
        if x_0 is None:
            x_0 = self._sample_noise(x_1.shape, device=x_1.device, dtype=x_1.dtype)

        ac_idx = torch.arange(T_ext, device=device).expand(B, T_ext)
        progress = torch.clamp(
            1.0 - (ac_idx - acoustic_front.unsqueeze(1)).float() / (n - 1),
            0.0,
            1.0,
        )  # (B, T_ext) — fraction of the n-step denoising trajectory completed
        t = self.schedule.timestep(progress)  # (B, T_ext) — warped timestep
        t_b = t.unsqueeze(-1)  # (B, T_ext, 1) for broadcasting along acoustic_dim

        x_t = self.param.prepare_x_t(x_0, x_1, t_b)

        noisy = MaskedTensor(values=x_t.transpose(1, 2), mask=acoustic.mask)
        pred = self.forward(text, noisy, t, attention_implementation)

        v_mask = (
            acoustic.mask
            & (ac_idx > acoustic_front.unsqueeze(1))
            & (ac_idx < acoustic_front.unsqueeze(1) + n)
        )

        # Per-position loss (B, T_ext) + recovered x_1 prediction (B, T_ext, D).
        loss_out = self.param.loss(
            x_t=x_t, timestep=t_b, pred=pred, x_0=x_0, x_1=x_1,
            loss_fn=self._per_pos_loss_fn,
        )
        loss = (loss_out.loss * v_mask).sum() / v_mask.sum().clamp(min=1)
        return TrainingStepOutput(
            loss=loss,
            x_pred=loss_out.x_pred,
            v_mask=v_mask,
            t=t,
            per_pos_loss=loss_out.loss,
        )

    @torch.no_grad()
    def speak(
        self,
        text: MaskedTensor,
        codec: Codec,
        *,
        x_0: torch.Tensor | None = None,
    ) -> MaskedTensor:
        """Generate acoustic features via rolling-Euler integration, stopping on EOS.

        Each generated frame is checked once it is fully denoised (t → 1). When
        codec.is_eos fires on its unnormalized form the loop marks that sample
        done and records the trim position. The loop exits once all samples are
        done or cfg.max_acoustic_len frames have been added.

        text:  MaskedTensor — values (B, 1, T_text), mask (B, T_text)
        codec: Codec used for unnormalize + EOS detection. Must match the codec
               type that the model config was instantiated with.
        x_0:   optional (B, acoustic_dim, max_acoustic_len) noise override.

        Returns a MaskedTensor with values (B, acoustic_dim, T_out) in **normalized**
        space — callers are expected to call codec.unnormalize before codec.decode.

        The model's `cfg.codec` enum records which codec the model was trained
        with; the loader is responsible for instantiating the matching codec.
        Here we only sanity-check that the duck implements the Codec protocol
        and the acoustic_dim matches.
        """
        assert isinstance(codec, Codec), (
            f"speak() received {type(codec).__name__}, which does not implement Codec"
        )
        assert codec.acoustic_dim == self.cfg.acoustic_dim, (
            f"codec.acoustic_dim={codec.acoustic_dim} but model "
            f"cfg.acoustic_dim={self.cfg.acoustic_dim}"
        )

        B = text.values.shape[0]
        device = text.values.device
        n = self.cfg.n_denoising_steps
        acoustic_dim = self.cfg.acoustic_dim
        max_T = self.cfg.max_acoustic_len

        if x_0 is None:
            x_0 = self._sample_noise((B, acoustic_dim, max_T), device=device)
        else:
            assert x_0.shape[-1] >= max_T, (
                f"x_0 must have at least cfg.max_acoustic_len ({max_T}) frames, "
                f"got {x_0.shape[-1]}"
            )

        values = torch.empty(B, acoustic_dim, 0, device=device, dtype=x_0.dtype)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        trim = torch.full((B,), -1, dtype=torch.long, device=device)

        for k in range(max_T + n - 1):
            if k < max_T:
                values = torch.cat([values, x_0[..., k : k + 1]], dim=-1)

            L = values.shape[-1]
            ac_idx = torch.arange(L, device=device).expand(B, L)
            buffer_mask = torch.ones(B, L, dtype=torch.bool, device=device)

            progress = torch.clamp((k - ac_idx).float() / (n - 1), 0.0, 1.0)
            t = self.schedule.timestep(progress)

            mt = MaskedTensor(values=values, mask=buffer_mask)
            pred = self.forward(text, mt, t)

            # Take a rolling-Euler step through the parametrization. For RF this
            # adds dt*pred; for JWT it divides by (1-t), so the t=1 column may
            # contain inf/nan — torch.where below zeroes those positions before
            # they touch `values`.
            x_t_BLD = values.transpose(1, 2)  # (B, L, dim)
            dt = self.schedule.dt(progress, n)  # (B, L) — per-position step size
            x_t_new = self.param.step(x_t_BLD, t.unsqueeze(-1), pred, dt.unsqueeze(-1))

            in_window = (t < 1.0) & buffer_mask
            update = torch.where(
                in_window.unsqueeze(-1),
                x_t_new - x_t_BLD,
                x_t_BLD.new_zeros(()),
            )
            values = values + update.transpose(1, 2)

            # Check the frame that just reached t=1 for the EOS sentinel.
            if k >= n - 1:
                p = k - (n - 1)
                frame_raw = codec.unnormalize(values[:, :, p])
                triggered = (~done) & codec.is_eos(frame_raw)
                trim[triggered] = p
                done |= triggered

            if done.all():
                break

        # Samples that hit max_T without triggering: scan for first below-threshold frame.
        if not done.all():
            frames_raw = codec.unnormalize(values)  # (B, acoustic_dim, L)
            # codec.is_eos expects (..., acoustic_dim); transpose so last dim is acoustic_dim.
            is_eos_per_frame = codec.is_eos(frames_raw.transpose(1, 2))  # (B, L)
            L = values.shape[-1]
            for b in range(B):
                if trim[b] == -1:
                    below = is_eos_per_frame[b].nonzero(as_tuple=True)[0]
                    trim[b] = int(below[0].item()) if len(below) > 0 else L

        trim = trim.clamp(min=0, max=max_T)
        T_out = int(trim.max().item())
        T_out = max(T_out, 1)

        out = values[..., :T_out]
        ac_idx_out = torch.arange(T_out, device=device).expand(B, T_out)
        mask = ac_idx_out < trim.unsqueeze(1)
        return MaskedTensor(values=out, mask=mask)

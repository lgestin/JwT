"""Mel-spectrogram inversion with pure (non-rolling) flow matching.

The model receives a noisy raw-audio-patch sequence ``x_t`` with its clip's
log-mel spectrogram concatenated along the feature dim (mel hop == patch size,
so frames align 1:1) and predicts the clean patches. One timestep per sample;
every position is denoised in parallel with full bidirectional attention — no
text, no rolling front, no EOS: the mel determines the output length.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from jwt.data.audio.codecs import Codecs, RawAudioPatcher
from jwt.data.audio.stft import MelSpectrogram
from jwt.model.attention import AttentionImplementation, SDPAAttention
from jwt.model.flow import FlowParametrizations
from jwt.model.loss import LossFn, LossFns
from jwt.model.neural_speaker import MaskedTensor, TrainingStepOutput
from jwt.model.transformer import Transformer, TransformerConfig
from jwt.training.timestep_schedules import TimestepSchedules


@dataclass
class SpectrogramInverterConfig:
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    # Records the patch codec the weights were trained with; also instantiated
    # by the training script. Must be a RawAudio* variant (patch == mel hop).
    codec: Codecs = Codecs.RAWAUDIO_256
    parametrization: FlowParametrizations = FlowParametrizations.JWT
    timestep_schedule: TimestepSchedules = TimestepSchedules.LOG_NORM
    acoustic_dim: int = 256
    n_denoising_steps: int = 128
    noise_scale: float = 1.0
    # Mel conditioning, computed from the audio itself. hop must equal the
    # patch size so mel frames and patches align 1:1.
    n_mels: int = 80
    mel_n_fft: int = 1024
    mel_hop_length: int = 256
    sample_rate: int = 22050  # overwritten from the arrow by the training script
    # Fixed affine on clamp(1e-5).log() mels -> roughly unit range. Defaults
    # mirror the LJSpeech log-mel stats in codecs.py (-5.8966 / 2.2268).
    mel_shift: float = -5.9
    mel_scale: float = 2.2


class SpectrogramInverter(nn.Module):
    def __init__(self, cfg: SpectrogramInverterConfig):
        super().__init__()
        assert cfg.mel_hop_length == cfg.acoustic_dim, (
            f"mel hop ({cfg.mel_hop_length}) must equal the patch size "
            f"({cfg.acoustic_dim}) so mel frames align 1:1 with patches"
        )
        self.cfg = cfg
        # Resolve the parametrization class once — it's a static dispatch table,
        # not an instance, so this stays free of training state.
        self.param = cfg.parametrization.parametrization
        self.schedule = cfg.timestep_schedule.schedule
        rank = cfg.transformer_config.adaln_rank
        if rank is not None and rank < cfg.n_denoising_steps:
            print(
                f"adaln_rank={rank} is below n_denoising_steps={cfg.n_denoising_steps}"
            )
        dim = cfg.transformer_config.dim
        self.acoustic_in = nn.Linear(cfg.acoustic_dim + cfg.n_mels, dim)
        self.acoustic_out = nn.Linear(dim, cfg.acoustic_dim)
        nn.init.zeros_(self.acoustic_out.weight)
        nn.init.zeros_(self.acoustic_out.bias)
        self.transformer = Transformer(cfg.transformer_config)
        # Buffers are persistent=False, so the state_dict and EMA are unaffected.
        self.mel = MelSpectrogram(
            n_fft=cfg.mel_n_fft,
            hop_length=cfg.mel_hop_length,
            n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
            window="hann",
            center=False,
            mel_scale="slaney",
        )

    def encode_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, S) unnormalized waveform -> (B, n_mels, S/hop) normalized log-mel.

        Callers are expected to run this in fp32 outside autocast (STFT
        stability). With center=False the STFT reflect-pads (n_fft - hop) // 2
        per side, so S divisible by hop yields exactly S/hop frames.
        """
        logmel = self.mel(waveform).clamp(min=1e-5).log()
        return (logmel - self.cfg.mel_shift) / self.cfg.mel_scale

    def forward(
        self,
        acoustic: MaskedTensor,
        mel: MaskedTensor,
        t: torch.Tensor,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> torch.Tensor:
        """Run a forward pass and return the raw model output.

        The semantic meaning of the returned tensor depends on the configured
        parametrization (velocity for RectifiedFlow, x_1 for JWT).

        acoustic.values: (B, acoustic_dim, T)  x_t     acoustic.mask: (B, T)
        mel.values:      (B, n_mels, T)  normalized log-mel, same frame grid
        t:               (B, T)          per-position timestep in [0, 1]
        returns:
            pred: (B, T, acoustic_dim)   raw model output
        """
        x = torch.cat([acoustic.values, mel.values], dim=1).transpose(1, 2)
        h = self.acoustic_in(x)
        attn_mask = attention_implementation.build_mask(acoustic.mask)
        out = self.transformer(h, t, attn_mask, attention_implementation)
        return self.acoustic_out(out)

    def _sample_noise(
        self,
        shape: tuple[int, ...] | torch.Size,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw the x_0 prior — a Gaussian scaled by `cfg.noise_scale`.

        Shared by `training_step` and `invert` so the corrupting noise has the
        same distribution at train and inference time.
        """
        return self.cfg.noise_scale * torch.randn(shape, device=device, dtype=dtype)

    def training_step(
        self,
        mel: MaskedTensor,
        acoustic: MaskedTensor,
        *,
        t: torch.Tensor | None = None,
        x_0: torch.Tensor | None = None,
        loss_fn: LossFn | None = None,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> TrainingStepOutput:
        """Sample a per-clip timestep, run forward, return the masked loss.

        Pure flow matching: one t per sample, drawn by warping u ~ U(0, 1)
        through the timestep schedule (the stochastic analogue of the
        schedule's deterministic inference grid), broadcast over all positions.
        Every real position is supervised (`v_mask == acoustic.mask`).

        Optional args pin the random choices for reproducibility:
        - t:   (B,) per-sample timesteps in [0, 1]
        - x_0: (B, T, acoustic_dim), the noise tensor
        - loss_fn: override for the trainer's loss; must return an un-reduced
          (B, T, D) elementwise loss tensor
        """
        B, _acoustic_dim, T = acoustic.values.shape
        device = acoustic.values.device
        # Default is MSE; the trainer overrides via TrainerConfig.loss_fn.
        loss_fn = loss_fn if loss_fn is not None else LossFns.MSE.fn

        if t is None:
            t = self.schedule.timestep(torch.rand(B, device=device))
        t = t.unsqueeze(-1).expand(B, T)  # (B, T), constant per sample
        t_b = t.unsqueeze(-1)  # (B, T, 1) for broadcasting along acoustic_dim

        x_1 = acoustic.values.transpose(1, 2)  # (B, T, acoustic_dim), normalized
        if x_0 is None:
            x_0 = self._sample_noise(x_1.shape, device=x_1.device, dtype=x_1.dtype)

        x_t = self.param.prepare_x_t(x_0, x_1, t_b)

        noisy = MaskedTensor(values=x_t.transpose(1, 2), mask=acoustic.mask)
        pred = self.forward(noisy, mel, t, attention_implementation)

        v_mask = acoustic.mask

        # Elementwise loss (B, T, D) + recovered x_1 prediction (B, T, D).
        loss_out = self.param.loss(
            x_t=x_t,
            timestep=t_b,
            pred=pred,
            x_0=x_0,
            x_1=x_1,
            loss_fn=loss_fn,
        )
        per_pos_loss = loss_out.loss.mean(-1)  # (B, T) — reduce the feature dim
        loss = (per_pos_loss * v_mask).sum() / v_mask.sum().clamp(min=1)
        return TrainingStepOutput(
            loss=loss,
            x_pred=loss_out.x_pred,
            v_mask=v_mask,
            t=t,
            per_pos_loss=per_pos_loss,
        )

    @torch.no_grad()
    def invert(
        self,
        mel: MaskedTensor,
        *,
        x_0: torch.Tensor | None = None,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> MaskedTensor:
        """Generate acoustic patches from a mel via parallel Euler integration.

        All positions share the timestep and step together over the schedule's
        n-step grid — n - 1 forwards total. The mel fixes the output length, so
        there is no stopping heuristic. Progress stays strictly below 1 inside
        the loop (the last step's dt lands exactly on t = 1), so JWT's
        1 / (1 - t) step never divides by zero.

        mel: MaskedTensor — values (B, n_mels, T) normalized log-mel.
        x_0: optional (B, acoustic_dim, T) noise override.

        Returns a MaskedTensor with values (B, acoustic_dim, T) in
        **normalized** space — callers apply codec.unnormalize before decode.
        """
        B, _, T = mel.values.shape
        device = mel.values.device
        n = self.cfg.n_denoising_steps

        if x_0 is None:
            x_0 = self._sample_noise((B, self.cfg.acoustic_dim, T), device=device)
        values = x_0

        for k in range(n - 1):
            progress = torch.full(
                (B, T), k / (n - 1), device=device, dtype=values.dtype
            )
            t = self.schedule.timestep(progress)
            noisy = MaskedTensor(values=values, mask=mel.mask)
            pred = self.forward(noisy, mel, t, attention_implementation)

            x_t = values.transpose(1, 2)  # (B, T, acoustic_dim)
            dt = self.schedule.dt(progress, n)
            x_t = self.param.step(x_t, t.unsqueeze(-1), pred, dt.unsqueeze(-1))
            values = x_t.transpose(1, 2)

        return MaskedTensor(values=values, mask=mel.mask)


class SpectrogramInverterCodec(nn.Module):
    """A trained `SpectrogramInverter` exposed through the `Codec` protocol.

    encode is the mel front-end, decode is flow-matching denoising — so the
    inverter can act as a neural vocoder codec (e.g. for a TTS model that
    predicts mels). Native (unnormalized) space is the plain log-mel;
    normalize/unnormalize apply the model's fixed affine.
    """

    # Codec protocol attributes; the mel front-end is built for one rate.
    required_sample_rate: int | None

    def __init__(self, model: SpectrogramInverter):
        super().__init__()
        self.model = model
        self.patcher = RawAudioPatcher(patch_size=model.cfg.acoustic_dim)
        self.acoustic_dim = model.cfg.n_mels
        self.hop_length = model.cfg.mel_hop_length
        self.required_sample_rate = model.cfg.sample_rate
        # log(1e-5) is the mel clamp floor; frames this close to it are silence.
        self.eos_threshold: float = math.log(1e-5) + 0.1

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, S) or (B, 1, S) waveform -> (B, n_mels, T) log-mel (native)."""
        if waveform.dim() == 3:
            assert waveform.shape[1] == 1, "expected mono input (B, 1, S) or (B, S)"
            waveform = waveform.squeeze(1)
        pad = (-waveform.shape[-1]) % self.hop_length
        if pad:
            waveform = F.pad(waveform, (0, pad))
        return self.model.mel(waveform).clamp(min=1e-5).log()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, T) log-mel (native) -> (B, T * patch_size) waveform."""
        mel = MaskedTensor(
            values=self.normalize(z),
            mask=torch.ones(  # ty: ignore[invalid-argument-type]
                z.shape[0], z.shape[-1], dtype=torch.bool, device=z.device
            ),
        )
        patches = self.model.invert(mel)
        return self.patcher.decode(self.patcher.unnormalize(patches.values))

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.model.cfg.mel_shift) / self.model.cfg.mel_scale

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.model.cfg.mel_scale + self.model.cfg.mel_shift

    def eos_frames(
        self,
        n: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.full(
            (self.acoustic_dim, n), math.log(1e-5), device=device, dtype=dtype
        )

    def is_eos(self, frame: torch.Tensor) -> torch.BoolTensor:
        return frame.mean(dim=-1) < self.eos_threshold  # ty: ignore[invalid-return-type]

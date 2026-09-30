import math
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from jwt.data.audio.codecs import Codec, Codecs
from jwt.model.attention import AttentionImplementation, SDPAAttention
from jwt.model.flow import FlowParametrizations
from jwt.model.kvcache import KVCache
from jwt.model.loss import LossFn, LossFns
from jwt.model.transformer import (
    Transformer,
    TransformerConfig,
)
from jwt.training.timestep_schedules import TimestepSchedules


class NeuralSpeaker(Protocol):
    def speak(
        self,
        text: "MaskedTensor",
        codec: Codec,
        *,
        prompt: "MaskedTensor | None" = None,
    ) -> "MaskedTensor": ...


@dataclass
class RollingFlowConfig:
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    vocabulary_size: int = 0
    codec: Codecs = Codecs.RAWAUDIO_512
    parametrization: FlowParametrizations = FlowParametrizations.JWT
    timestep_schedule: TimestepSchedules = TimestepSchedules.LOG_NORM_SD3
    acoustic_dim: int = 100
    n_denoising_steps: int = 32
    max_acoustic_len: int = 2048
    eos_n_frames: int = 32  # must equal n_denoising_steps, see __post_init__
    noise_scale: float = 1.0
    # Phonemes per audio patch: text tokens sit on the audio clock at
    # j / phoneme_per_audio_patch and acoustic frame i at i, so a frame and its
    # aligned phoneme share a RoPE position. It is 15.25 phonemes/s (measured
    # on LJSpeech) over the codec's patch rate, so it tracks both hop size and
    # sample rate: the default matches this config's default codec
    # (RAWAUDIO_512 at 22.05 kHz) and every config in `configs/` sets its own.
    phoneme_per_audio_patch: float = 0.354

    def __post_init__(self) -> None:
        if self.eos_n_frames != self.n_denoising_steps:
            raise ValueError(
                f"eos_n_frames={self.eos_n_frames} must equal "
                f"n_denoising_steps={self.n_denoising_steps}. A clean frame attends "
                "to at most n_denoising_steps frames ahead, so with a shorter EOS "
                "span the only frames whose window is cut short are the ones inside "
                "the sentinel: the EOS probe learns that mask-boundary shortcut "
                "(teacher-forced AUC 1.0) and never fires free-running, where the "
                "window is never cut."
            )


@dataclass
class MaskedTensor:
    values: torch.Tensor
    mask: torch.BoolTensor

    def __post_init__(self) -> None:
        assert self.values.shape[-1] == self.mask.shape[-1]
        assert self.values.ndim == self.mask.ndim + 1

    @property
    def shape(self) -> torch.Size:
        return self.values.shape

    @property
    def masked_shape(self) -> torch.Tensor:
        return self.mask.sum(-1)


@dataclass
class TrainingStepOutput:
    """Result of `RollingFlowSpeaker.training_step` — all fields are GPU tensors.

    `per_pos_loss` is the parametrization's per-position loss *before* the
    rolling-window mask is applied; the trainer reuses it to bin the loss by
    timestep without recomputing anything.
    """

    loss: torch.Tensor  # scalar, masked-mean of per_pos_loss
    x_pred: torch.Tensor  # (B, T_ext, acoustic_dim) — recovered x_1
    v_mask: torch.Tensor  # (B, T_ext) bool — rolling-window supervision mask
    t: torch.Tensor  # (B, T_ext) — per-position timestep in [0, 1]
    per_pos_loss: torch.Tensor  # (B, T_ext) — per-position loss before masking


class RollingFlowSpeaker(NeuralSpeaker, nn.Module):
    phoneme_per_audio_patch: torch.Tensor

    def __init__(self, cfg: RollingFlowConfig) -> None:
        nn.Module.__init__(self)
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
        self.text_in = nn.Embedding(cfg.vocabulary_size, dim)
        self.acoustic_in = nn.Linear(cfg.acoustic_dim, dim)
        self.acoustic_out = nn.Linear(dim, cfg.acoustic_dim)
        nn.init.zeros_(self.acoustic_out.weight)
        nn.init.zeros_(self.acoustic_out.bias)
        self.text_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.acoustic_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.speaker_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.transformer = Transformer(cfg.transformer_config)
        self.register_buffer(
            "phoneme_per_audio_patch",
            torch.tensor(cfg.phoneme_per_audio_patch, dtype=torch.float32),
        )

    def forward(
        self,
        text: MaskedTensor,
        acoustic: MaskedTensor,
        t: torch.Tensor,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
        *,
        prompt: MaskedTensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass and return the raw model output.

        The semantic meaning of the returned tensor depends on the configured
        parametrization (velocity for RectifiedFlow, x_1 for JWT). The model
        itself is parametrization-agnostic; the parametrization class converts
        this tensor into a loss (during training) or a velocity (during
        sampling).

        text.values:     (B, 1, T_text)             text.mask: (B, T_text)
        acoustic.values: (B, acoustic_dim, T_acoustic)    acoustic.mask: (B, T_acoustic)
        t:               (B, T_acoustic)               per-acoustic-position t in [0, 1]
        prompt.values:   (B, acoustic_dim, P)  normalized speaker prompt frames,
                         prompt.mask: (B, P); a zero-length prompt is unconditional
        returns:
            pred: (B, T_acoustic, acoustic_dim)            raw model output
        """
        B, acoustic_dim, T_acoustic = acoustic.values.shape
        text_ids = text.values.squeeze(-2)  # (B, T_text)
        T_text = text_ids.shape[-1]
        T = T_text + T_acoustic
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
        in_text = F.pad(text.mask, (0, T_acoustic))
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

        # Keep masks in packed coords. in_text above is in original coords
        # and would silently misalign the attention mask when text is padded.
        in_real_packed = arange < total_lens.unsqueeze(1)
        in_acoustic_packed = (arange >= text_lens.unsqueeze(1)) & in_real_packed

        # Sequence mask (True = visible key): visible up to and including the
        # first real t=0 (the "next frontier"). Pure-noise positions beyond it
        # carry no signal and would only distract attention.
        is_zero_real = (t_packed == 0.0) & in_acoustic_packed
        keep_first_zero = is_zero_real.cumsum(-1) <= 1
        seq_mask = in_real_packed & keep_first_zero  # (B, T)

        in_text_packed = arange < text_lens.unsqueeze(1)
        acoustic_pos = (arange - text_lens.unsqueeze(1)).float()
        positions = torch.where(
            in_text_packed, arange / self.phoneme_per_audio_patch, acoustic_pos
        )
        # Text commits first (0); padding never (inf), so no query row is empty.
        acoustic_commit_index = self.acoustic_commit_index(t_packed, acoustic_pos)
        commit_index = torch.where(
            in_text_packed,
            0.0,
            torch.where(in_acoustic_packed, acoustic_commit_index, math.inf),
        )

        # Prepended after packing, so every packed index above stays valid and
        # the output only needs the P prompt positions dropped.
        P = 0
        if prompt is not None:
            P = prompt.mask.shape[-1]
            x_packed, t_packed, positions, seq_mask, commit_index = self.prepend_prompt(
                prompt, x_packed, t_packed, positions, seq_mask, commit_index
            )

        out_packed = self.transformer(
            x_packed,
            t_packed,
            positions,
            attention_implementation,
            seq_mask=seq_mask,
            commit_index=commit_index,
        )
        pred_packed = self.acoustic_out(out_packed[:, P:])  # (B, T, acoustic_dim)

        # Unpack: acoustic position i in sample b lives at packed
        # position text_lens[b] + i.
        acoustic_idx = torch.arange(T_acoustic, device=device).expand(B, T_acoustic)
        unpack_idx = (text_lens.unsqueeze(1) + acoustic_idx).clamp(max=T - 1)
        pred = torch.gather(
            pred_packed, 1, unpack_idx.unsqueeze(-1).expand(B, T_acoustic, acoustic_dim)
        )
        return pred

    @staticmethod
    def acoustic_commit_index(
        t: torch.Tensor, acoustic_idx: torch.Tensor
    ) -> torch.Tensor:
        """Commit index of acoustic frames for `attention.commit_rule`:
        `i + 1` once frame `i` is clean (t=1), inf while it is in the window."""
        # Exact comparison: every schedule maps progress 1 to exactly t=1.
        return torch.where(t == 1.0, acoustic_idx + 1.0, math.inf)

    def prepend_prompt(
        self,
        prompt: MaskedTensor,
        x: torch.Tensor,
        t: torch.Tensor,
        positions: torch.Tensor,
        seq_mask: torch.Tensor,
        commit_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Put the prompt frames in front of the sequence: clean (t=1), frozen with
        the text (commit index 0), at audio-clock positions [-P_b, 0) so the
        prompt ends where the target starts."""
        P = prompt.values.shape[-1]
        prompt_lat = (
            self.acoustic_in(prompt.values.transpose(1, 2)) + self.speaker_modality
        )
        prompt_pos = torch.arange(P, device=x.device) - prompt.mask.sum(
            -1, keepdim=True
        )
        return (
            torch.cat([prompt_lat, x], dim=1),
            F.pad(t, (P, 0), value=1.0),
            torch.cat([prompt_pos.to(positions.dtype), positions], dim=1),
            torch.cat([prompt.mask, seq_mask], dim=1),
            F.pad(commit_index, (P, 0), value=0.0),
        )

    def prefill(
        self, text: MaskedTensor, cache: KVCache, prompt: MaskedTensor | None = None
    ) -> None:
        """Commit registers, prompt and text to `cache`."""
        B, T_text = text.mask.shape
        device = text.values.device
        x = self.text_in(text.values.squeeze(-2)) + self.text_modality
        t = torch.ones(B, T_text, device=device)
        positions = torch.arange(T_text, device=device).expand(B, T_text)
        positions = positions / self.phoneme_per_audio_patch
        seq_mask = text.mask
        commit_index = torch.zeros(B, T_text, device=device)
        if prompt is not None:
            x, t, positions, seq_mask, commit_index = self.prepend_prompt(
                prompt, x, t, positions, seq_mask, commit_index
            )
        # Padding keeps index 0 with a False mask, so every sample commits
        # every token and the cache stays aligned across the batch.
        self.transformer(
            x,
            t,
            positions,
            seq_mask=seq_mask,
            commit_index=commit_index,
            cache=cache,
        )

    def predict_window(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        acoustic_idx: torch.Tensor,
        cache: KVCache,
    ) -> torch.Tensor:
        """Model output for the window `x_t` (B, W, acoustic_dim) at float
        acoustic indices `acoustic_idx` (B, W), attending through `cache`; the frame
        at t=1, if any, is committed to it."""
        out = self.transformer(
            self.acoustic_in(x_t) + self.acoustic_modality,
            t,
            acoustic_idx,
            commit_index=self.acoustic_commit_index(t, acoustic_idx),
            cache=cache,
        )
        return self.acoustic_out(out)

    def sample_noise(
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
        prompt: MaskedTensor | None = None,
        acoustic_front: torch.LongTensor | None = None,
        x_0: torch.Tensor | None = None,
        n: int | None = None,
        loss_fn: LossFn | None = None,
        attention_implementation: type[AttentionImplementation] = SDPAAttention,
    ) -> TrainingStepOutput:
        """Sample a rolling front, run forward, return masked loss + x_pred.

        The trainer is expected to have already appended EOS sentinel frames and
        normalized the values. `acoustic.values` is therefore (B, acoustic_dim,
        T_ext) in normalized space with `acoustic.mask` covering real + sentinel.

        `prompt` is the normalized speaker prompt, see `forward`.

        Optional args let callers pin the random choices for reproducibility:
        - acoustic_front: (B,) long, where each sample's denoising front lands;
          defaults to a uniform sample in [-(n-1), acoustic_lens_ext) so
          negative values reproduce the inference warm-up distribution
        - x_0: (B, T_ext, acoustic_dim), the noise tensor
        - n: override for cfg.n_denoising_steps
        - loss_fn: override for cfg.loss_fn; must return an un-reduced (B, T, D)
          elementwise loss tensor

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
        B, _, T_ext = acoustic.values.shape
        device = acoustic.values.device
        n = n if n is not None else self.cfg.n_denoising_steps
        # Default is MSE; the trainer overrides via TrainerConfig.loss_fn.
        loss_fn = loss_fn if loss_fn is not None else LossFns.MSE.fn
        acoustic_lens_ext = acoustic.mask.sum(-1)

        if acoustic_front is None:
            u = torch.rand(B, device=device)
            acoustic_front = (
                u * (acoustic_lens_ext.float() + (n - 1)) - (n - 1)
            ).long()  # ty: ignore[invalid-assignment]

        x_1 = acoustic.values.transpose(1, 2)  # (B, T_ext, acoustic_dim), normalized
        if x_0 is None:
            x_0 = self.sample_noise(x_1.shape, device=x_1.device, dtype=x_1.dtype)

        acoustic_idx = torch.arange(T_ext, device=device).expand(B, T_ext)
        progress = torch.clamp(
            1.0 - (acoustic_idx - acoustic_front.unsqueeze(1)).float() / (n - 1),  # ty: ignore[unresolved-attribute]
            0.0,
            1.0,
        )  # (B, T_ext) — fraction of the n-step denoising trajectory completed
        t = self.schedule.timestep(progress)  # (B, T_ext) — warped timestep
        t_b = t.unsqueeze(-1)  # (B, T_ext, 1) for broadcasting along acoustic_dim

        x_t = self.param.prepare_x_t(x_0, x_1, t_b)

        noisy = MaskedTensor(values=x_t.transpose(1, 2), mask=acoustic.mask)
        pred = self.forward(text, noisy, t, attention_implementation, prompt=prompt)

        v_mask = (
            acoustic.mask
            & (acoustic_idx > acoustic_front.unsqueeze(1))  # ty: ignore[unresolved-attribute]
            & (acoustic_idx < acoustic_front.unsqueeze(1) + n)  # ty: ignore[unresolved-attribute]
        )

        # Elementwise loss (B, T_ext, D) + recovered x_1 prediction (B, T_ext, D).
        loss_out = self.param.loss(
            x_t=x_t,
            timestep=t_b,
            pred=pred,
            x_0=x_0,
            x_1=x_1,
            loss_fn=loss_fn,
        )
        per_pos_loss = loss_out.loss.mean(-1)  # (B, T_ext) — reduce the feature dim
        loss = (per_pos_loss * v_mask).sum() / v_mask.sum().clamp(min=1)
        return TrainingStepOutput(
            loss=loss,
            x_pred=loss_out.x_pred,
            v_mask=v_mask,
            t=t,
            per_pos_loss=per_pos_loss,
        )

    @torch.no_grad()
    def speak(
        self,
        text: MaskedTensor,
        codec: Codec,
        *,
        prompt: MaskedTensor | None = None,
        x_0: torch.Tensor | None = None,
        use_kv_cache: bool = True,
    ) -> MaskedTensor:
        """Generate acoustic features via rolling-Euler integration, stopping on EOS.

        At step k the window is frames [k-n+1, k]: its first frame has just
        reached t=1 and is checked for EOS, the rest take one Euler step. With
        `use_kv_cache` text is prefilled into a `KVCache` and every step runs the
        model on the window alone, committing the t=1 frame; without it the
        whole buffer is recomputed each step. The two give the same frames,
        since the cache is exact under the commit-rule mask. The loop exits
        once all samples are done or cfg.max_acoustic_len frames have been
        generated.

        text:  MaskedTensor — values (B, 1, T_text), mask (B, T_text)
        codec: Codec used for unnormalize + EOS detection. Must match the codec
               type that the model config was instantiated with.
        prompt: optional normalized speaker prompt, see `forward`.
        x_0:   optional (B, acoustic_dim, max_acoustic_len) noise override.
        use_kv_cache: False recomputes the full buffer, as a reference.

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
        B = text.mask.shape[0]
        device = text.values.device
        n = self.cfg.n_denoising_steps
        acoustic_dim = self.cfg.acoustic_dim
        max_T = self.cfg.max_acoustic_len

        if x_0 is None:
            x_0 = self.sample_noise((B, acoustic_dim, max_T), device=device)
        else:
            assert x_0.shape[-1] >= max_T, (
                f"x_0 must have at least cfg.max_acoustic_len ({max_T}) "
                f"frames, got {x_0.shape[-1]}"
            )

        values = x_0[..., :max_T].clone()  # (B, acoustic_dim, max_T)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        trim = torch.full((B,), -1, dtype=torch.long, device=device)

        cache = KVCache() if use_kv_cache else None
        if cache is not None:
            self.prefill(text, cache, prompt)

        for k in range(max_T + n - 1):
            lo, hi = max(k - (n - 1), 0), min(k + 1, max_T)
            acoustic_idx = torch.arange(lo, hi, device=device, dtype=torch.float32)
            acoustic_idx = acoustic_idx.expand(B, hi - lo)
            progress = torch.clamp((k - acoustic_idx) / (n - 1), 0.0, 1.0)
            t = self.schedule.timestep(progress)  # (B, W)
            x_t = values[..., lo:hi].transpose(1, 2)  # (B, W, acoustic_dim)

            if cache is not None:
                pred = self.predict_window(x_t, t, acoustic_idx, cache)
            else:
                # Frames before the window are clean.
                t_all = F.pad(t, (lo, 0), value=1.0)
                buffer_mask = torch.ones(B, hi, dtype=torch.bool, device=device)
                buffer = MaskedTensor(values=values[..., :hi], mask=buffer_mask)  # ty: ignore[invalid-argument-type]
                pred = self.forward(text, buffer, t_all, prompt=prompt)[:, lo:]

            # Rolling-Euler step through the parametrization. For RF this adds
            # dt*pred; for JWT it divides by (1-t), so the t=1 frame may come
            # out inf/nan — torch.where keeps its old value.
            dt = self.schedule.dt(progress, n)  # (B, W) — per-position step size
            x_new = self.param.step(x_t, t.unsqueeze(-1), pred, dt.unsqueeze(-1))
            x_t = torch.where((t < 1.0).unsqueeze(-1), x_new, x_t)
            values[..., lo:hi] = x_t.transpose(1, 2)

            # Check the frame that just reached t=1 for the EOS sentinel.
            if k >= n - 1:
                frame_raw = codec.unnormalize(values[:, :, lo])
                triggered = (~done) & codec.is_eos(frame_raw)
                trim[triggered] = lo
                done |= triggered

            if done.all():
                break

        # Samples that hit max_T without triggering: scan for the first
        # below-threshold frame.
        if not done.all():
            frames_raw = codec.unnormalize(values)  # (B, acoustic_dim, L)
            # codec.is_eos expects (..., acoustic_dim); transpose so the
            # last dim is acoustic_dim.
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
        acoustic_idx_out = torch.arange(T_out, device=device).expand(B, T_out)
        mask = acoustic_idx_out < trim.unsqueeze(1)
        return MaskedTensor(values=out, mask=mask)  # ty: ignore[invalid-argument-type]

import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from jwt.data.audio.codecs import Codec
from jwt.data.audio.stft import MelSpectrogram
from jwt.data.dataset import Batch
from jwt.model.attention import AttentionImplementations, TorchAttention
from jwt.model.discriminator import MultiResolutionSTFTDiscriminator
from jwt.model.loss import LossFns
from jwt.model.neural_speaker import (
    MaskedTensor,
    RollingFlowSpeaker,
    TrainingStepOutput,
)
from jwt.training.adversarial_loss import AdversarialLoss
from jwt.training.attention_probe import capture_attention, log_attention_maps
from jwt.training.checkpoint_manager import CheckpointManager
from jwt.training.ema import EMA
from jwt.training.loggers import Logger, log_mel
from jwt.training.loss import MelAuxLoss
from jwt.training.metrics import (
    binned_loss_stats,
    masked_mean_std,
    per_pos_l1_error,
)


@dataclass
class TrainerState:
    step: int
    best_loss: float = float("inf")


class AMPDtype(StrEnum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"

    @property
    def dtype(self):
        if self == AMPDtype.FP16:
            return torch.float16
        elif self == AMPDtype.BF16:
            return torch.bfloat16
        elif self == AMPDtype.FP32:
            return torch.float32


@dataclass
class TrainerConfig:
    clip_grad_norm: float | None = 1.0
    device: str = "cuda"
    amp_dtype: AMPDtype = AMPDtype.BF16
    smp_steps: int = 2_500
    valid_steps: int = 1_000
    checkpoint_steps: int = 5_000
    max_steps: int = 200_001
    n_smp: int = 16
    grad_accum_steps: int = 1
    loss_fn: LossFns = LossFns.L1
    attention_implementation: AttentionImplementations = AttentionImplementations.FLASH_VARLEN
    # Auxiliary log-mel L1 loss weight. 0 = monitor only (no gradient signal);
    aux_mel_weight: float = 0.0
    # Adversarial (multi-resolution STFT GAN) loss. 0 = disabled (no
    # discriminator is built). Raw-audio codecs only — see Args.__post_init__.
    adv_weight: float = 0.0
    # L1 feature-matching loss weight (companion to the adversarial term).
    feat_match_weight: float = 0.0
    # Discriminator optimizer learning rate.
    disc_lr: float = 2e-4
    # STFT resolutions (n_fft) for the multi-resolution discriminator.
    disc_n_fft: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    # Crop the discriminator to the K leading (highest-t, about-to-be-committed)
    # frames of each rolling window. 0 = discriminate the whole sequence.
    # Requires disc_window_frames * codec.hop_length >= max(disc_n_fft).
    disc_window_frames: int = 0
    # Weight the generator adversarial hinge per frame by t ** adv_t_weight_pow,
    # down-weighting noisy low-t frames. 0 = uniform (no t weighting).
    adv_t_weight_pow: float = 0.0
    # Warm up the generator on FM (+mel) before the GAN turns on: adversarial
    # and discriminator updates only run once step >= adv_start_step.
    adv_start_step: int = 0
    # Scalar diagnostics (throughput, memory, normalization stats) and the
    # binned loss-by-t histogram are accumulated on-GPU and flushed to
    # TensorBoard every `log_steps` steps.
    log_steps: int = 100
    n_loss_bins: int = 10


class Trainer:
    def __init__(self, config: TrainerConfig, state: TrainerState | None = None):
        self.config = config
        self.state = state or TrainerState(step=0)
        self._device = torch.device(config.device)

    @property
    def device(self):
        return self._device

    @property
    def amp_dtype(self):
        return self.config.amp_dtype.dtype

    @property
    def smp_steps(self):
        return self.config.smp_steps

    @property
    def valid_steps(self):
        return self.config.valid_steps

    @property
    def checkpoint_steps(self):
        return self.config.checkpoint_steps

    @property
    def max_steps(self):
        return self.config.max_steps

    @property
    def noamp(self):
        return self._device.type != "cuda"

    @property
    def step(self):
        return self.state.step

    @property
    def best_loss(self):
        return self.state.best_loss


# Viz mel for tensorboard — codec-agnostic, lets us inspect decoded waveforms
# regardless of which acoustic representation the model actually trained on.
_VIZ_MEL_N_MELS = 80
_VIZ_MEL_N_FFT = 1024
_VIZ_MEL_HOP = 256


def prepare_acoustic_batch(batch: Batch, codec: Codec, eos_n: int) -> MaskedTensor:
    """Append codec EOS sentinel frames, normalize, return a MaskedTensor.

    Shared by the trainer and lr_finder so both go through the same EOS+normalize
    contract that the model expects.
    """
    values = batch.acoustic
    if values.ndim == 4 and values.shape[1] == 1:
        values = values.squeeze(1)
    mask = batch.acoustic_mask
    B, acoustic_dim, T = values.shape

    lens = mask.sum(-1)
    T_ext = T + eos_n

    values_ext = torch.nn.functional.pad(values, (0, eos_n))
    ac_idx_ext = torch.arange(T_ext, device=values.device).unsqueeze(0)
    in_sentinel = (ac_idx_ext >= lens.unsqueeze(1)) & (ac_idx_ext < (lens + eos_n).unsqueeze(1))
    eos = codec.eos_frames(eos_n, device=values.device, dtype=values.dtype)
    eos_grid = eos.unsqueeze(0).expand(B, acoustic_dim, eos_n)
    eos_padded = torch.nn.functional.pad(eos_grid, (T, 0))
    values_ext = torch.where(in_sentinel.unsqueeze(1), eos_padded, values_ext)

    in_real = ac_idx_ext < lens.unsqueeze(1)
    mask_ext = in_real | in_sentinel

    values_norm = codec.normalize(values_ext)
    return MaskedTensor(values=values_norm, mask=mask_ext)  # ty: ignore[invalid-argument-type]


class TTSRollingFlowMatchingTrainer(Trainer):
    def __init__(
        self,
        config: TrainerConfig,
        codec: Codec,
        sample_rate: int,
        model: RollingFlowSpeaker,
        optimizer: Optimizer,
        scaler: GradScaler | None,
        logger: Logger,
        train_dloader: DataLoader,
        valid_dloader: DataLoader,
        smp_dloader: DataLoader,
        state: TrainerState | None,
        checkpoint_manager: CheckpointManager | None,
        ema: EMA | None = None,
        discriminator: MultiResolutionSTFTDiscriminator | None = None,
        disc_optimizer: Optimizer | None = None,
    ):
        super().__init__(config=config, state=state)
        self.codec = codec
        # Sample rate is a property of the dataset, not the codec — the training
        # script reads it from the arrow file and passes it in.
        self.sample_rate = sample_rate
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.logger = logger
        self.train_dloader = train_dloader
        self.valid_dloader = valid_dloader
        self.smp_dloader = smp_dloader
        self.checkpoint_manager = checkpoint_manager
        self.ema = ema
        self.viz_mel = MelSpectrogram(
            n_fft=_VIZ_MEL_N_FFT,
            hop_length=_VIZ_MEL_HOP,
            n_mels=_VIZ_MEL_N_MELS,
            sample_rate=sample_rate,
            window="hann",
            center=False,
            mel_scale="slaney",
        ).to(self._device)
        # Auxiliary mel loss, only when weighted in. Config validation forbids
        # aux_mel_weight > 0 for BigVGAN, so this is always a raw-audio codec.
        self.mel_aux_loss: MelAuxLoss | None = (
            MelAuxLoss(sample_rate=sample_rate, hop_length=codec.hop_length).to(self._device)
            if config.aux_mel_weight > 0
            else None
        )
        # Adversarial loss + its discriminator/optimizer, only when weighted in.
        # Config validation forbids adv_weight > 0 for BigVGAN, so the
        # discriminator always supervises a real decoded waveform.
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.adv_loss: AdversarialLoss | None = (
            AdversarialLoss(
                discriminator,
                hop_length=codec.hop_length,
                window_frames=config.disc_window_frames,
                t_weight_pow=config.adv_t_weight_pow,
            ).to(self._device)
            if config.adv_weight > 0 and discriminator is not None
            else None
        )

    def _prepare_acoustic(self, batch: Batch) -> MaskedTensor:
        return prepare_acoustic_batch(batch, self.codec, self.model.cfg.eos_n_frames)

    def _ema_weights(self):
        """EMA weights installed for the block, or a no-op when EMA is off."""
        return self.ema.swapped(self.model) if self.ema is not None else nullcontext()

    def train(self):
        self._log_initial_samples()
        self._log_timestep_schedule()
        self.optimizer.zero_grad()
        # Reflect the resumed step on the progress bar (no-op on a fresh run).
        self.logger.set_progress(self.step)
        grad_accum_steps = self.config.grad_accum_steps

        micro = 0
        accum: dict[str, float] = {}
        diag_accum: dict[str, torch.Tensor] = {}
        diag_micro = 0
        bin_accum: dict[str, torch.Tensor] = {}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        last_log_step = self.step
        last_log_time = time.perf_counter()

        while self.step < self.max_steps:
            for batch in self.train_dloader:
                if micro == 0:
                    if self.step % self.smp_steps == 0:
                        with self._ema_weights():
                            self._log_samples()
                    if self.step % self.valid_steps == 0:
                        with self._ema_weights():
                            self.validation()
                    if (
                        self.checkpoint_manager is not None
                        and self.step % self.checkpoint_steps == 0
                        and self.step > 0
                    ):
                        self.checkpoint_manager.save(
                            step=self.step,
                            model=self.model,
                            optimizer=self.optimizer,
                            scaler=self.scaler,
                            best_loss=self.best_loss,
                            additional_state=(
                                {"ema": self.ema.state_dict()} if self.ema is not None else None
                            ),
                            discriminator=self.discriminator,
                            disc_optimizer=self.disc_optimizer,
                        )
                        self.checkpoint_manager.cleanup_old_checkpoints()

                if hasattr(self.optimizer, "train"):
                    self.optimizer.train()  # ty: ignore[call-non-callable]
                self.model.train()

                metrics, scalars, bins = self.training_step(batch)
                for k, v in metrics.items():
                    accum[k] = accum.get(k, 0.0) + float(v)
                self._accumulate_diagnostics(diag_accum, scalars)
                self._accumulate_diagnostics(bin_accum, bins)
                micro += 1
                diag_micro += 1

                if micro >= grad_accum_steps:
                    opt_metrics = self._optimizer_step()
                    avg = {k: v / micro for k, v in accum.items()}
                    avg.update({k: float(v) for k, v in opt_metrics.items()})
                    self.logger.log_metrics(avg, self.step, prefix="train")
                    self.logger.update_progress(1)
                    self.state.step += 1
                    micro = 0
                    accum = {}

                    if self.step % self.config.log_steps == 0:
                        self._log_train_diagnostics(
                            diag_accum, diag_micro, last_log_step, last_log_time
                        )
                        self._emit_loss_histogram(bin_accum, self.step, "train")
                        diag_accum = {}
                        diag_micro = 0
                        bin_accum = {}
                        last_log_step = self.step
                        last_log_time = time.perf_counter()

                    if self.step >= self.max_steps:
                        self.logger.close()
                        return

    def training_step(
        self, batch: Batch
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Single micro-step: forward + (under training) scaled backward.

        Loss is divided by grad_accum_steps before backward so accumulated
        gradients average across the window. Returns `(metrics, scalars,
        bins)`: `metrics` are the headline scalars (unscaled loss); `scalars`
        are on-GPU diagnostics averaged over the logging window; `bins` are the
        loss-by-t (sum, count) tensors accumulated for the histogram.
        """
        batch = batch.to(self.device, non_blocking=True)
        text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)
        acoustic = self._prepare_acoustic(batch)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            # The trainer owns the loss form (`TrainerConfig.loss_fn`) —
            # model.training_step applies it per-element and returns it masked.
            # x_pred is x_1 recovered in normalized space, decoded below to a
            # waveform for the auxiliary mel loss when it is enabled.
            out = self.model.training_step(
                text,
                acoustic,
                loss_fn=self.config.loss_fn.fn,
                attention_implementation=self.config.attention_implementation.implementation,
            )
            fm_loss, x_pred, v_mask = out.loss, out.x_pred, out.v_mask

        loss = fm_loss
        metrics: dict[str, torch.Tensor] = {"fm_loss": fm_loss.detach()}

        # Adversarial supervision is gated on a warmup so FM (+mel) shapes the
        # generator before the GAN turns on. It and the mel aux loss both need
        # the decoded waveform, so the decode runs once when either is active.
        adv_active = (
            self.adv_loss is not None
            and self.model.training
            and self.step >= self.config.adv_start_step
        )
        need_wav = self.mel_aux_loss is not None or adv_active

        # Decoded waveforms (fp32 outside autocast for STFT stability), kept for
        # the discriminator update after the generator backward.
        pred_wav: torch.Tensor | None = None
        target_wav: torch.Tensor | None = None

        # Auxiliary log-mel L1 — a perceptual loss term for codecs whose
        # FM-space loss isn't itself perceptual (raw-audio patches). `mel_aux_loss`
        # is None for BigVGAN, whose FM loss is already mel-space.
        logmel_diff: torch.Tensor | None = None
        if need_wav:
            with torch.autocast(device_type=self.device.type, enabled=False):
                x_1_target_norm = acoustic.values.float()
                x_1_pred_norm = x_pred.float().transpose(1, 2)
                pred_wav = self.codec.decode(self.codec.unnormalize(x_1_pred_norm))
                target_wav = self.codec.decode(self.codec.unnormalize(x_1_target_norm))

                if self.mel_aux_loss is not None:
                    logmel_l1, logmel_diff = self.mel_aux_loss(pred_wav, target_wav, v_mask)
                    loss = loss + self.config.aux_mel_weight * logmel_l1
                    metrics["logmel_l1"] = logmel_l1.detach()

                if adv_active:
                    assert self.adv_loss is not None
                    # Generator side: freeze D so the adversarial term backprops
                    # through the discriminator to the model without writing grads
                    # on the D params — accumulation-safe, no mid-window zeroing.
                    self._set_disc_requires_grad(False)
                    adv_g, feat = self.adv_loss.generator_loss(pred_wav, target_wav, v_mask, out.t)
                    loss = loss + self.config.adv_weight * adv_g
                    loss = loss + self.config.feat_match_weight * feat
                    metrics["adv_g"] = adv_g.detach()
                    metrics["feat_match"] = feat.detach()

        metrics["loss"] = loss.detach()
        scalars, bins = self._step_diagnostics(out, text, acoustic, logmel_diff)

        if self.model.training:
            scaled = loss / self.config.grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

        # Discriminator update: real target vs. detached prediction. Runs after
        # the generator backward on a fresh graph through the detached fake.
        if adv_active:
            assert self.adv_loss is not None and pred_wav is not None and target_wav is not None
            self._set_disc_requires_grad(True)
            with torch.autocast(device_type=self.device.type, enabled=False):
                disc_loss = self.adv_loss.discriminator_loss(pred_wav, target_wav, v_mask, out.t)
            (disc_loss / self.config.grad_accum_steps).backward()
            metrics["disc_loss"] = disc_loss.detach()

        return metrics, scalars, bins

    def _set_disc_requires_grad(self, flag: bool) -> None:
        """Toggle discriminator param grads for the G vs. D phase of the step."""
        if self.discriminator is not None:
            for p in self.discriminator.parameters():
                p.requires_grad_(flag)

    def _step_diagnostics(
        self,
        out: TrainingStepOutput,
        text: MaskedTensor,
        acoustic: MaskedTensor,
        logmel_diff: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-micro-step diagnostics — all kept on-GPU (no host sync).

        Returns `(scalars, bins)`: `scalars` are 0-dim tensors the caller
        averages over the logging window; `bins` holds the loss-by-t
        `(sum, count)` tensors the caller accumulates for the histogram.
        `logmel_diff` is the `(B, T)` per-frame auxiliary log-mel L1, present
        only when the aux mel loss is active (raw-audio codecs).
        """
        bin_sums, bin_counts = binned_loss_stats(
            out.per_pos_loss, out.t, out.v_mask, self.config.n_loss_bins
        )
        # Un-reweighted x_1-prediction error, binned the same way. Reading it
        # next to `bin_sums` separates genuine denoising difficulty from the
        # parametrization's loss reweighting (e.g. JWT's 1/(1-t) factor, which
        # inflates the FM loss near t=1 regardless of accuracy). Same t/v_mask,
        # so its bin counts equal `bin_counts` and aren't re-stored.
        x1_err = per_pos_l1_error(out.x_pred, acoustic.values.transpose(1, 2))
        x1_err_sums, _ = binned_loss_stats(x1_err, out.t, out.v_mask, self.config.n_loss_bins)
        ac_mean, ac_std = masked_mean_std(acoustic.values, out.v_mask)
        scalars = {
            "vmask_fill": out.v_mask.float().mean(),
            "ac_len_mean": acoustic.mask.sum(-1).float().mean(),
            "text_len_mean": text.mask.sum(-1).float().mean(),
            "ac_target_mean": ac_mean,
            "ac_target_std": ac_std,
        }
        bins = {
            "bin_sums": bin_sums,
            "bin_counts": bin_counts,
            "x1_err_sums": x1_err_sums,
        }
        # Auxiliary log-mel L1, binned by the same t/v_mask — the *perceptual*
        # analogue of `x1_err_by_t` (mel space rather than normalized FM space).
        # Same bins as `bin_counts`, so its counts aren't re-stored.
        if logmel_diff is not None:
            logmel_l1_sums, _ = binned_loss_stats(
                logmel_diff, out.t, out.v_mask, self.config.n_loss_bins
            )
            bins["logmel_l1_sums"] = logmel_l1_sums
        return scalars, bins

    @staticmethod
    def _accumulate_diagnostics(acc: dict[str, torch.Tensor], new: dict[str, torch.Tensor]) -> None:
        """In-place GPU accumulation — no host sync."""
        for k, v in new.items():
            acc[k] = v if k not in acc else acc[k] + v

    def _reduce_diagnostics(
        self, acc: dict[str, torch.Tensor], micro: int
    ) -> dict[str, torch.Tensor]:
        """Window-average the accumulated data diagnostics (0-dim GPU tensors).

        These describe the batches being fed (lengths, target stats, supervision
        fill), so they are logged under the `data/` section rather than `train/`.
        The binned FM loss is emitted separately by `_emit_loss_histogram`.
        """
        return {
            k: acc[k] / micro
            for k in (
                "vmask_fill",
                "ac_len_mean",
                "text_len_mean",
                "ac_target_mean",
                "ac_target_std",
            )
        }

    def _emit_loss_histogram(
        self, diag_accum: dict[str, torch.Tensor], step: int, prefix: str
    ) -> None:
        """Emit the timestep-binned loss histograms.

        `fm_loss_by_t` is the parametrization's (possibly reweighted) loss;
        `x1_err_by_t` is the un-reweighted `|x_1 - x_pred|` error. When the
        auxiliary mel loss is active a third series, `logmel_l1_by_t`, reports
        that error in perceptual mel space. Comparing them isolates reweighting
        artefacts from genuine denoising difficulty. Each bin mean is
        `sum / count`, exact over the window; an empty bin (count 0 -> NaN) is
        zero-filled. One host transfer covers every series.
        """
        series = [
            ("fm_loss_by_t", diag_accum["bin_sums"]),
            ("x1_err_by_t", diag_accum["x1_err_sums"]),
        ]
        if "logmel_l1_sums" in diag_accum:
            series.append(("logmel_l1_by_t", diag_accum["logmel_l1_sums"]))
        counts = diag_accum["bin_counts"]
        means = torch.stack([sums / counts for _, sums in series])
        vals = torch.nan_to_num(means.detach().float(), nan=0.0).cpu().tolist()
        n = self.config.n_loss_bins
        edges = [i / n for i in range(n + 1)]
        for (tag, _), bin_values in zip(series, vals, strict=True):
            self.logger.log_histogram(f"{prefix}/{tag}", edges, bin_values, step)

    def _log_timestep_schedule(self) -> None:
        """Log the timestep schedule curve — t = schedule.timestep(progress)
        sampled on the n-step denoising grid. Config-fixed (a pure function of
        progress), so it is logged once at startup rather than per window."""
        n = self.model.cfg.n_denoising_steps
        timesteps = self.model.schedule.timesteps(n).detach().float().tolist()
        edges = [i / n for i in range(n + 1)]
        self.logger.log_histogram("schedule/timesteps", edges, timesteps, self.step)

    def _emit_diagnostics(
        self,
        gpu_metrics: dict[str, torch.Tensor],
        step: int,
        prefix: str,
        host_metrics: dict[str, float] | None = None,
    ) -> None:
        """Single batched host transfer, then hand off to the logger."""
        keys = list(gpu_metrics)
        stacked = torch.stack([gpu_metrics[k].detach().float().reshape(()) for k in keys])
        merged = dict(zip(keys, stacked.cpu().tolist(), strict=True))  # one D2H transfer
        if host_metrics:
            merged.update(host_metrics)
        self.logger.log_diagnostics(merged, step, prefix=prefix)

    def _log_train_diagnostics(
        self,
        diag_accum: dict[str, torch.Tensor],
        diag_micro: int,
        last_step: int,
        last_time: float,
    ) -> None:
        # Batch-shape / target diagnostics describe the data, not the
        # optimization — they go under `data/train`. Optimization and hardware
        # diagnostics (param norm, throughput, memory) live under `system` so
        # they don't crowd the `train/` loss curves.
        data_metrics = self._reduce_diagnostics(diag_accum, diag_micro)
        system_metrics: dict[str, torch.Tensor] = {
            "param_norm": torch.stack([p.detach().norm() for p in self.model.parameters()]).norm()
        }

        now = time.perf_counter()
        elapsed = max(now - last_time, 1e-6)
        steps = self.step - last_step
        batch_size = self.train_dloader.batch_size or 1
        host: dict[str, float] = {
            "steps_per_sec": steps / elapsed,
            "samples_per_sec": steps * batch_size * self.config.grad_accum_steps / elapsed,
        }
        if self.device.type == "cuda":
            host["peak_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            torch.cuda.reset_peak_memory_stats(self.device)

        self._emit_diagnostics(data_metrics, self.step, "data/train")
        self._emit_diagnostics(system_metrics, self.step, "system", host_metrics=host)

    def _optimizer_step(self) -> dict[str, torch.Tensor]:
        """Clip + step + zero_grad. Called once per accumulation window."""
        metrics: dict[str, torch.Tensor] = {}
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.config.clip_grad_norm is not None:
            metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.clip_grad_norm,
            )
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model, self.step)
        self.optimizer.zero_grad()

        # Discriminator step — only once the GAN is active, so its optimizer
        # state isn't perturbed by weight decay during the warmup. EMA tracks
        # the generator only.
        if self.disc_optimizer is not None and self.step >= self.config.adv_start_step:
            if self.config.clip_grad_norm is not None and self.discriminator is not None:
                metrics["disc_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    max_norm=self.config.clip_grad_norm,
                )
            self.disc_optimizer.step()
            self.disc_optimizer.zero_grad()
        return metrics

    @torch.inference_mode()
    def validation(self):
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()  # ty: ignore[call-non-callable]

        sums: dict[str, float] = {}
        diag_accum: dict[str, torch.Tensor] = {}
        bin_accum: dict[str, torch.Tensor] = {}
        count = 0
        for vbatch in self.valid_dloader:
            metrics, scalars, bins = self.training_step(vbatch)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            self._accumulate_diagnostics(diag_accum, scalars)
            self._accumulate_diagnostics(bin_accum, bins)
            count += 1

        if count == 0:
            return

        val_metrics = {k: v / count for k, v in sums.items()}
        self.logger.log_metrics(val_metrics, self.step, prefix="valid")
        # Validation emits only data diagnostics — route them to `data/valid`
        # to mirror the `data/train` split.
        self._emit_diagnostics(self._reduce_diagnostics(diag_accum, count), self.step, "data/valid")
        self._emit_loss_histogram(bin_accum, self.step, "valid")
        self._log_attention_maps()

        loss_val = val_metrics.get("loss", float("inf"))
        if loss_val < self.best_loss:
            self.state.best_loss = loss_val

    @torch.inference_mode()
    def _log_attention_maps(self) -> None:
        """Log per-sample text->audio attention heatmaps for the first `n_smp`
        validation samples.

        Runs one extra eager forward with the weight-exposing `TorchAttention`
        backend (the fused SDPA kernel cannot surface attention weights), behind
        hooks that collect every layer's map. The logged map is averaged over
        heads and layers — see `jwt.training.attention_probe`.
        """
        batch = next(iter(self.valid_dloader)).to(self.device, non_blocking=True)
        n = min(self.config.n_smp, batch.tokens.shape[0])
        text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)
        acoustic = self._prepare_acoustic(batch)
        text_n = MaskedTensor(values=text.values[:n], mask=text.mask[:n])  # ty: ignore[invalid-argument-type]
        acoustic_n = MaskedTensor(values=acoustic.values[:n], mask=acoustic.mask[:n])  # ty: ignore[invalid-argument-type]

        with (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=not self.noamp,
            ),
            capture_attention(self.model) as collector,
        ):
            self.model.training_step(
                text_n,
                acoustic_n,
                loss_fn=self.config.loss_fn.fn,
                attention_implementation=TorchAttention,
            )

        log_attention_maps(
            self.logger,
            collector.map,
            text_n.mask.sum(-1),
            acoustic_n.mask.sum(-1),
            self.step,
        )

    @torch.inference_mode()
    def _log_samples(self):
        self.model.eval()
        batch = next(iter(self.smp_dloader)).to(self.device)
        n = min(len(batch.audios), self.config.n_smp)

        text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            acoustic_pred = self.model.speak(text, codec=self.codec)

        for i in range(n):
            L = int(acoustic_pred.mask[i].sum().item())
            if L == 0:
                continue
            ac_i = acoustic_pred.values[i : i + 1, :, :L]  # (1, acoustic_dim, L), normalized
            wav = self.codec.decode(self.codec.unnormalize(ac_i))[0]
            self.logger.log_audio(f"sampled/{i}", wav, self.step, self.sample_rate)
            # The model picks its own length via the EOS sentinel; early in
            # training it can emit a near-empty generation. viz_mel's STFT
            # (center=False) reflection-pads by (n_fft - hop) // 2 and crashes
            # when the waveform is shorter than n_fft, so skip the mel for it.
            if wav.shape[-1] < self.viz_mel.n_fft:
                warnings.warn(
                    f"sample {i}: generation too short for mel viz "
                    f"({wav.shape[-1]} samples < n_fft={self.viz_mel.n_fft}, "
                    f"{L} acoustic frame(s)); skipping spectrogram at step "
                    f"{self.step}",
                    stacklevel=2,
                )
                continue
            viz_mel = self.viz_mel(wav.unsqueeze(0))[0].clamp(min=1e-5).log()
            log_mel(self.logger, f"sampled/{i}/mel", viz_mel, self.step)

    def _log_initial_samples(self):
        smp_batch = next(iter(self.smp_dloader))
        n = min(len(smp_batch.audios), self.config.n_smp)

        for i, audio in enumerate(smp_batch.audios[:n]):
            self.logger.log_audio(f"{i}/clean", audio.waveform, 0, audio.sample_rate)

            waveform = audio.waveform.to(self.device)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=not self.noamp,
                ),
            ):
                reconstructed = self.codec.reconstruct(waveform[None])[0]
                clean_viz = self.viz_mel(waveform).clamp(min=1e-5).log()
                recon_viz = self.viz_mel(reconstructed.unsqueeze(0))[0].clamp(min=1e-5).log()
            self.logger.log_audio(
                f"{i}/reconstructed",
                reconstructed,
                0,
                self.sample_rate,
            )
            log_mel(self.logger, f"{i}/clean/mel", clean_viz, 0)
            log_mel(self.logger, f"{i}/reconstructed/mel", recon_viz, 0)

import itertools
import math
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from jwt.data.audio.audio import Audio
from jwt.data.audio.codecs import Codec
from jwt.data.audio.stft import MelSpectrogram
from jwt.data.dataset import Batch
from jwt.model.attention import AttentionImplementations, TorchAttention
from jwt.model.loss import LossFns
from jwt.model.neural_speaker import (
    MaskedTensor,
    RollingFlowSpeaker,
    TrainingStepOutput,
)
from jwt.training.attention_probe import attention_images, capture_attention
from jwt.training.checkpoint_manager import CheckpointManager
from jwt.training.ema import EMA
from jwt.training.loggers import Logger, SampleRecord, mel_image
from jwt.training.loss import masked_mean_reduction
from jwt.training.metrics.nisqa import NISQA
from jwt.training.metrics.pesq import PESQ
from jwt.training.metrics.snr import mag_snr, si_snr, snr
from jwt.training.metrics.stoi import STOI
from jwt.training.metrics.utils import (
    binned_loss_stats,
    masked_mean_std,
    per_pos_l1_error,
    sampled_generation_stats,
)
from jwt.training.metrics.utmos import UTMOS

# Free generations shorter than this are zero-padded up to it before MOS
# scoring — the predictors need a minimum of signal, and padding (unlike
# exclusion) keeps the scored population fixed across steps and runs.
MIN_SCORED_SECONDS = 0.5
# Fixed x_0 seed for the sampled-metrics pass: identical latents every
# evaluation, so the curves track the model rather than the noise draw.
SAMPLED_NOISE_SEED = 20_260_814


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
    attention_implementation: AttentionImplementations = (
        AttentionImplementations.FLASH_VARLEN
    )
    # Auxiliary log-mel L1 loss weight. 0 = monitor only (no gradient signal);
    aux_mel_weight: float = 0.0
    # Scalar diagnostics (throughput, memory, normalization stats) and the
    log_steps: int = 1_000
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
    in_sentinel = (ac_idx_ext >= lens.unsqueeze(1)) & (
        ac_idx_ext < (lens + eos_n).unsqueeze(1)
    )
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
        self.mel_spectrogram = MelSpectrogram(
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            sample_rate=sample_rate,
            window="hann",
            center=False,
            mel_scale="slaney",
            n_mfcc=13,
        ).to(self._device)
        self.utmos = UTMOS().to(self._device)
        self.nisqa = NISQA().to(self._device)
        self.stoi = STOI().to(self._device)
        self.pesq = PESQ().to(self._device)

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
                            self._log_sampled_metrics()
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
                                {"ema": self.ema.state_dict()}
                                if self.ema is not None
                                else None
                            ),
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
                        self._emit_loss_curves(bin_accum, self.step, "train")
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
        loss-by-t (sum, count) tensors accumulated for the by-t curves.
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

        metrics: dict[str, torch.Tensor] = {"fm_loss": fm_loss.detach()}
        with torch.autocast(device_type=self.device.type, enabled=False):
            x_1_target_norm = acoustic.values.float()
            x_1_pred_norm = x_pred.float().transpose(1, 2)
            pred_wav = self.codec.decode(self.codec.unnormalize(x_1_pred_norm))
            target_wav = self.codec.decode(self.codec.unnormalize(x_1_target_norm))
            repeats = self.codec.hop_length // self.mel_spectrogram.hop_length
            mel_mask = v_mask.repeat_interleave(repeats=repeats, dim=1)
            logmel_l1 = self.mel_spectrogram.logmel_l1(pred_wav, target_wav)
            logmel_l1 = masked_mean_reduction(logmel_l1, mel_mask).mean(0)
            self._reconstruction_metrics(metrics, pred_wav, target_wav, v_mask)

        loss = fm_loss + self.config.aux_mel_weight * logmel_l1
        metrics["logmel_l1"] = logmel_l1.detach()
        metrics["loss"] = loss.detach()
        scalars, bins = self._step_diagnostics(out, text, acoustic)

        if self.model.training:
            scaled = loss / self.config.grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

        return metrics, scalars, bins

    def _reconstruction_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        pred_wav: torch.Tensor,  # (B, S)
        trgt_wav: torch.Tensor,  # (B, S)
        mask: torch.Tensor,  # (B, T)
    ) -> None:
        """Masked GPU metrics on the teacher-forced reconstruction, in place.

        si_snr/snr every step; the pricier spectral pair (lsd/mcd) only under
        validation (eval mode), averaged over the full valid set.
        """
        hop = self.codec.hop_length
        if pred_wav.dim() == 3:
            pred_wav = pred_wav.squeeze(1)
        if trgt_wav.dim() == 3:
            trgt_wav = trgt_wav.squeeze(1)
        wav_mask = mask.repeat_interleave(hop, dim=-1)
        metrics["si_snr"] = si_snr(pred_wav, trgt_wav, wav_mask).mean()
        metrics["snr"] = snr(pred_wav, trgt_wav, wav_mask).mean()
        metrics["mag_snr"] = mag_snr(
            pred_wav, trgt_wav, wav_mask, self.mel_spectrogram
        ).mean()
        # magnitude good + waveform bad ⇒ the error is phase-attributable
        metrics["phase_snr_gap"] = metrics["mag_snr"] - metrics["snr"]
        if not self.model.training:
            repeats = self.codec.hop_length // self.mel_spectrogram.hop_length
            mel_mask = mask.repeat_interleave(repeats=repeats, dim=1)

            logstft_l1 = self.mel_spectrogram.logstft_l1(pred_wav, trgt_wav)
            logstft_l1 = masked_mean_reduction(logstft_l1, mel_mask).mean()
            metrics["logstft_l1"] = logstft_l1.detach()

            mcd = self.mel_spectrogram.mel_cepstral_distortion(pred_wav, trgt_wav)
            mcd = masked_mean_reduction(mcd, mel_mask).mean()
            metrics["mel_cepstral_distortion"] = mcd.detach()

    def _step_diagnostics(
        self,
        out: TrainingStepOutput,
        text: MaskedTensor,
        acoustic: MaskedTensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-micro-step diagnostics — all kept on-GPU (no host sync).

        Returns `(scalars, bins)`: `scalars` are 0-dim tensors the caller
        averages over the logging window; `bins` holds the loss-by-t
        `(sum, count)` tensors the caller accumulates for the by-t curves.
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
        x1_err_sums, _ = binned_loss_stats(
            x1_err, out.t, out.v_mask, self.config.n_loss_bins
        )
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
        return scalars, bins

    @staticmethod
    def _accumulate_diagnostics(
        acc: dict[str, torch.Tensor], new: dict[str, torch.Tensor]
    ) -> None:
        """In-place GPU accumulation — no host sync."""
        for k, v in new.items():
            acc[k] = v if k not in acc else acc[k] + v

    def _reduce_diagnostics(
        self, acc: dict[str, torch.Tensor], micro: int
    ) -> dict[str, torch.Tensor]:
        """Window-average the accumulated data diagnostics (0-dim GPU tensors).

        These describe the batches being fed (lengths, target stats, supervision
        fill), so they are logged under the `data/` section rather than `train/`.
        The binned FM loss is emitted separately by `_emit_loss_curves`.
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

    def _emit_loss_curves(
        self, diag_accum: dict[str, torch.Tensor], step: int, prefix: str
    ) -> None:
        """Emit the timestep-binned loss curves.

        `fm_loss_by_t` is the parametrization's (possibly reweighted) loss;
        `x1_err_by_t` is the un-reweighted `|x_1 - x_pred|` error. Comparing
        them isolates reweighting artefacts from genuine denoising difficulty.
        Each bin mean is `sum / count`, exact over the window, plotted at its
        bin centre; an empty bin (count 0 -> NaN) stays NaN so the curve gaps
        there instead of dipping to zero. One host transfer covers every
        series.
        """
        series = [
            ("fm_loss", diag_accum["bin_sums"]),
            ("x1_err", diag_accum["x1_err_sums"]),
        ]
        counts = diag_accum["bin_counts"]
        means = torch.stack([sums / counts for _, sums in series])
        vals = means.detach().float().cpu().tolist()
        n = self.config.n_loss_bins
        centers = [(i + 0.5) / n for i in range(n)]
        for (tag, _), curve in zip(series, vals, strict=True):
            self.logger.log_curve(f"by_t/{prefix}_{tag}", centers, curve, step)

    def _log_timestep_schedule(self) -> None:
        """Log the timestep schedule curve — t = schedule.timestep(progress)
        sampled on the n-step denoising grid. Config-fixed (a pure function of
        progress), so it is logged once at startup rather than per window."""
        n = self.model.cfg.n_denoising_steps
        timesteps = self.model.schedule.timesteps(n).detach().float().tolist()
        # `timesteps` warps the uniform progress grid, so x is that same grid.
        progress = torch.linspace(0.0, 1.0, n).tolist()
        self.logger.log_curve(
            "schedule/timesteps",
            progress,
            timesteps,
            self.step,
            xlabel="progress",
            history=False,  # config-fixed, logged once
        )

    def _emit_diagnostics(
        self,
        gpu_metrics: dict[str, torch.Tensor],
        step: int,
        prefix: str,
        host_metrics: dict[str, float] | None = None,
    ) -> None:
        """Single batched host transfer, then hand off to the logger."""
        merged: dict[str, float] = {}
        if gpu_metrics:
            keys = list(gpu_metrics)
            stacked = torch.stack(
                [gpu_metrics[k].detach().float().reshape(()) for k in keys]
            )
            merged = dict(
                zip(keys, stacked.cpu().tolist(), strict=True)
            )  # one D2H transfer
        if host_metrics:
            merged.update(host_metrics)
        if merged:
            self.logger.log_diagnostics(merged, step, prefix=prefix)

    def _log_train_diagnostics(
        self,
        diag_accum: dict[str, torch.Tensor],
        diag_micro: int,
        last_step: int,
        last_time: float,
    ) -> None:
        # Batch-shape / target diagnostics describe the data, not the
        # optimization — they go under `data/train`. Param norm sits with the
        # optimization-health panels (`optim`); throughput and memory get
        # their own section so nothing crowds the loss curves.
        data_metrics = self._reduce_diagnostics(diag_accum, diag_micro)
        optim_metrics: dict[str, torch.Tensor] = {
            "param_norm": torch.stack(
                [p.detach().norm() for p in self.model.parameters()]
            ).norm()
        }

        now = time.perf_counter()
        elapsed = max(now - last_time, 1e-6)
        steps = self.step - last_step
        batch_size = self.train_dloader.batch_size or 1
        host: dict[str, float] = {
            "steps_per_sec": steps / elapsed,
            "samples_per_sec": steps
            * batch_size
            * self.config.grad_accum_steps
            / elapsed,
        }
        if self.device.type == "cuda":
            host["peak_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            torch.cuda.reset_peak_memory_stats(self.device)

        self._emit_diagnostics(data_metrics, self.step, "data/train")
        self._emit_diagnostics(optim_metrics, self.step, "optim")
        self._emit_diagnostics({}, self.step, "throughput", host_metrics=host)

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
        self._emit_diagnostics(
            self._reduce_diagnostics(diag_accum, count), self.step, "data/valid"
        )
        self._emit_loss_curves(bin_accum, self.step, "valid")
        self._log_audio_metrics(self._attention_images())

        loss_val = val_metrics.get("loss", float("inf"))
        if loss_val < self.best_loss:
            self.state.best_loss = loss_val

    @torch.inference_mode()
    def _probe_attention(
        self, text: MaskedTensor, acoustic: MaskedTensor
    ) -> dict[int, torch.Tensor]:
        """Per-sample text->audio attention heatmaps, keyed by sample index.

        Runs one extra eager forward with the weight-exposing `TorchAttention`
        backend (the fused SDPA kernel cannot surface attention weights), behind
        hooks that collect every layer's map. The map is averaged over heads
        and layers — see `jwt.training.attention_probe`.
        """
        with (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=not self.noamp,
            ),
            capture_attention(self.model) as collector,
        ):
            self.model.training_step(
                text,
                acoustic,
                loss_fn=self.config.loss_fn.fn,
                attention_implementation=TorchAttention,
            )

        return attention_images(collector.map, text.mask.sum(-1), acoustic.mask.sum(-1))

    @torch.inference_mode()
    def _attention_images(self) -> dict[int, torch.Tensor]:
        """Teacher-forced attention probe over the first `n_smp` validation
        samples."""
        batch = next(iter(self.valid_dloader)).to(self.device, non_blocking=True)
        n = min(self.config.n_smp, batch.tokens.shape[0])
        text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)
        acoustic = self._prepare_acoustic(batch)
        text_n = MaskedTensor(values=text.values[:n], mask=text.mask[:n])  # ty: ignore[invalid-argument-type]
        acoustic_n = MaskedTensor(values=acoustic.values[:n], mask=acoustic.mask[:n])  # ty: ignore[invalid-argument-type]
        return self._probe_attention(text_n, acoustic_n)

    @torch.inference_mode()
    def _log_audio_metrics(
        self, attention: dict[int, torch.Tensor] | None = None
    ) -> None:
        """SI-SDR, SDR, PESQ, ESTOI, NISQA on teacher-forced reconstructions
        of the first `n_smp` validation samples.

        Runs one extra forward and scores only full (n-1)-frame supervised
        windows: uniform length keeps the metrics comparable across steps and
        allows a single gather + batched metric calls. Clipped warm-up/tail
        windows are dropped. `attention` carries the per-sample heatmaps for
        the same batch, keyed by sample index.
        """
        from torchmetrics.functional.audio import (
            scale_invariant_signal_distortion_ratio,
            signal_distortion_ratio,
        )

        batch = next(iter(self.valid_dloader)).to(self.device, non_blocking=True)
        n = min(self.config.n_smp, batch.tokens.shape[0])
        text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)
        acoustic = self._prepare_acoustic(batch)
        text_n = MaskedTensor(values=text.values[:n], mask=text.mask[:n])  # ty: ignore[invalid-argument-type]
        acoustic_n = MaskedTensor(values=acoustic.values[:n], mask=acoustic.mask[:n])  # ty: ignore[invalid-argument-type]

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            out = self.model.training_step(
                text_n,
                acoustic_n,
                loss_fn=self.config.loss_fn.fn,
                attention_implementation=self.config.attention_implementation.implementation,
            )

        x_1_pred_norm = out.x_pred.float().transpose(1, 2)
        pred_wav = self.codec.decode(self.codec.unnormalize(x_1_pred_norm))
        trgt_wav = self.codec.decode(self.codec.unnormalize(acoustic_n.values.float()))
        if pred_wav.ndim == 3:
            pred_wav = pred_wav.squeeze(1)
        if trgt_wav.ndim == 3:
            trgt_wav = trgt_wav.squeeze(1)

        hop = self.codec.hop_length
        max_s = min(pred_wav.shape[-1], trgt_wav.shape[-1])
        mask = out.v_mask
        n_win = self.model.cfg.n_denoising_steps - 1
        L = n_win * hop
        s0 = mask.long().argmax(-1) * hop
        keep = (mask.sum(-1) == n_win) & (s0 + L <= max_s)
        if not bool(keep.any()):
            return
        idx = s0[keep, None] + torch.arange(L, device=s0.device)
        pred_wav = pred_wav[keep].gather(1, idx)
        trgt_wav = trgt_wav[keep].gather(1, idx)

        rows = (
            torch.isfinite(pred_wav).all(-1)
            & (pred_wav.pow(2).mean(-1) > 1e-8)
            & (trgt_wav.pow(2).mean(-1) > 1e-8)
        )
        if not bool(rows.any()):
            return
        pred_wav = pred_wav[rows]
        trgt_wav = trgt_wav[rows]

        per_sample: dict[str, torch.Tensor] = {
            "si_sdr": scale_invariant_signal_distortion_ratio(pred_wav, trgt_wav),
            "sdr": signal_distortion_ratio(pred_wav, trgt_wav, load_diag=1e-6),
        }
        scores = {
            **self.utmos.score(pred_wav, sample_rate=self.sample_rate),
            **self.nisqa.score(pred_wav, sample_rate=self.sample_rate),
            **self.stoi.score(pred_wav, trgt_wav, sample_rate=self.sample_rate),
            **self.pesq.score(pred_wav, trgt_wav, sample_rate=self.sample_rate),
        }
        per_sample.update({k: v.reshape(-1) for k, v in scores.items()})
        metrics = {k: v.mean().item() for k, v in per_sample.items()}
        self.logger.log_metrics(metrics, self.step, prefix="valid")

        # Per-sample records under the sample's index in the validation batch
        # (stable across steps despite the window filters above); only metrics
        # scored per sample fit.
        n_rows = pred_wav.shape[0]
        keys = [k for k, v in per_sample.items() if v.numel() == n_rows]
        sample_idx = torch.arange(keep.shape[0], device=keep.device)[keep][rows]
        attention = attention or {}
        records = [
            SampleRecord(
                index=si,
                audio={
                    "pred": Audio(pred_wav[i], self.sample_rate),  # ty: ignore[invalid-argument-type]
                    "target": Audio(trgt_wav[i], self.sample_rate),  # ty: ignore[invalid-argument-type]
                },
                images=({"attention": attention[si]} if si in attention else {}),
                metrics={k: float(per_sample[k][i]) for k in keys},
            )
            for i, si in enumerate(int(s) for s in sample_idx.tolist())
        ]
        self.logger.log_samples("valid_audio", records, self.step)

    @torch.inference_mode()
    def _log_sampled_metrics(self) -> None:
        """UTMOS/NISQA and termination stats over free generations of the smp
        and validation prompts — the quantitative counterpart of `_log_samples`.

        Every generation is scored at its full length, terminated or not —
        short ones are zero-padded to `MIN_SCORED_SECONDS` — so the scored
        population is all prompts, always, comparable across steps and runs;
        stopping-criterion health is reported separately (`eos_rate`,
        `len_ratio`) instead of contaminating the MOS numbers.
        """
        self.model.eval()
        max_len = self.model.cfg.max_acoustic_len
        min_samples = math.ceil(MIN_SCORED_SECONDS * self.sample_rate)

        gen_lens_all: list[torch.Tensor] = []
        ref_lens_all: list[torch.Tensor] = []
        wavs: list[torch.Tensor] = []
        batches = itertools.chain(
            [next(iter(self.smp_dloader))], iter(self.valid_dloader)
        )
        for i, batch in enumerate(batches):
            batch = batch.to(self.device, non_blocking=True)
            text = MaskedTensor(
                values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask
            )
            B = batch.tokens.shape[0]
            g = torch.Generator().manual_seed(SAMPLED_NOISE_SEED + i)
            # CPU draw keeps the latents device-independent; speak() applies no
            # scaling to a provided x_0, so the prior's noise_scale goes here.
            x_0 = (
                torch.randn(B, self.model.cfg.acoustic_dim, max_len, generator=g)
                * self.model.cfg.noise_scale
            ).to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=not self.noamp,
            ):
                acoustic_pred = self.model.speak(text, codec=self.codec, x_0=x_0)

            gen_lens = acoustic_pred.mask.sum(-1)
            gen_lens_all.append(gen_lens)
            ref_lens_all.append(batch.acoustic_mask.sum(-1))
            for b in range(B):
                L = int(gen_lens[b])
                if L > 0:
                    ac_b = acoustic_pred.values[b : b + 1, :, :L].float()
                    wav = self.codec.decode(self.codec.unnormalize(ac_b))[0]
                    wav = wav.reshape(-1)
                else:
                    wav = torch.zeros(0, device=self.device)
                if wav.shape[-1] < min_samples:
                    wav = torch.nn.functional.pad(wav, (0, min_samples - wav.shape[-1]))
                wavs.append(wav)

        stats = sampled_generation_stats(
            torch.cat(gen_lens_all), torch.cat(ref_lens_all), max_len
        )
        metrics = {k: float(v) for k, v in stats.items()}
        per_utt: dict[str, list[float]] = defaultdict(list)
        for wav in wavs:
            scored = self.utmos.score(wav, sample_rate=self.sample_rate) | (
                self.nisqa.score(wav, sample_rate=self.sample_rate)
            )
            for k, v in scored.items():
                per_utt[k].append(float(v.mean()))
        metrics.update({k: sum(v) / len(v) for k, v in per_utt.items()})
        self.logger.log_metrics(metrics, self.step, prefix="sampled")

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

        # Self-forced probe: alignment read back from the generated frames.
        att: dict[int, torch.Tensor] = {}
        if bool(acoustic_pred.mask[:n].any()):
            att = self._probe_attention(
                MaskedTensor(values=text.values[:n], mask=text.mask[:n]),  # ty: ignore[invalid-argument-type]
                MaskedTensor(
                    values=acoustic_pred.values[:n],
                    mask=acoustic_pred.mask[:n],  # ty: ignore[invalid-argument-type]
                ),
            )

        records: list[SampleRecord] = []
        for i in range(n):
            L = int(acoustic_pred.mask[i].sum().item())
            if L == 0:
                continue
            ac_i = acoustic_pred.values[
                i : i + 1, :, :L
            ]  # (1, acoustic_dim, L), normalized
            wav = self.codec.decode(self.codec.unnormalize(ac_i))[0]
            record = SampleRecord(
                index=i,
                audio={"audio": Audio(wav, self.sample_rate)},  # ty: ignore[invalid-argument-type]
            )
            if i in att:
                record.images["attention"] = att[i]
            records.append(record)
            if wav.shape[-1] < self.mel_spectrogram.n_fft:
                warnings.warn(
                    f"sample {i}: generation too short for mel viz "
                    f"({wav.shape[-1]} samples < n_fft={self.mel_spectrogram.n_fft}, "
                    f"{L} acoustic frame(s)); skipping spectrogram at step "
                    f"{self.step}",
                    stacklevel=2,
                )
                continue
            mel_spectrogram = (
                self.mel_spectrogram(wav.unsqueeze(0))[0].clamp(min=1e-5).log()
            )
            record.images["mel"] = mel_image(mel_spectrogram)
            utmos = self.utmos(wav, sample_rate=self.sample_rate)["utmos"]
            record.metrics["utmos"] = utmos.item()
        if records:
            # References share the smp batch, so indices align for the join.
            self.logger.log_samples("samples", records, self.step, join="references")

    def _log_initial_samples(self):
        smp_batch = next(iter(self.smp_dloader))
        n = min(len(smp_batch.audios), self.config.n_smp)

        records: list[SampleRecord] = []
        for i, audio in enumerate(smp_batch.audios[:n]):
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
                clean_viz = self.mel_spectrogram(waveform).clamp(min=1e-5).log()
                recon_viz = (
                    self.mel_spectrogram(reconstructed.unsqueeze(0))[0]
                    .clamp(min=1e-5)
                    .log()
                )
            records.append(
                SampleRecord(
                    index=i,
                    audio={
                        "clean": Audio(audio.waveform, audio.sample_rate),
                        "reconstructed": Audio(reconstructed, self.sample_rate),  # ty: ignore[invalid-argument-type]
                    },
                    images={
                        "clean_mel": mel_image(clean_viz),
                        "reconstructed_mel": mel_image(recon_viz),
                    },
                )
            )
        self.logger.log_samples("references", records, 0)

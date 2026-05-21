import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tts.data.audio.codecs import Codec
from tts.data.audio.stft import MelSpectrogram
from tts.data.dataset import Batch
from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowSpeaker,
    TrainingStepOutput,
)
from tts.training.checkpoint_manager import CheckpointManager
from tts.training.ema import EMA
from tts.training.loggers import Logger, log_mel
from tts.training.loss import MelAuxLoss
from tts.training.metrics import binned_loss_stats, masked_mean_std


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
    # Auxiliary log-mel L1 loss weight. 0 = monitor only (no gradient signal);
    # ~0.01 matches the gated_dit_block recipe — useful for codecs whose FM
    # space isn't itself perceptual (e.g. RawAudioPatcher patches).
    aux_mel_weight: float = 0.0
    # Scalar diagnostics (throughput, memory, normalization stats) are
    # accumulated on-GPU and flushed to TensorBoard every `log_steps` steps.
    log_steps: int = 100
    # The binned loss-by-t histogram is coarser: emitted every `hist_steps`,
    # each one aggregating the full window. Independent of `log_steps`.
    hist_steps: int = 500
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


def prepare_acoustic_batch(
    batch: Batch, codec: Codec, eos_n: int
) -> MaskedTensor:
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
    return MaskedTensor(values=values_norm, mask=mask_ext)


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
        self.viz_mel = MelSpectrogram(
            n_fft=_VIZ_MEL_N_FFT,
            hop_length=_VIZ_MEL_HOP,
            n_mels=_VIZ_MEL_N_MELS,
            sample_rate=sample_rate,
            window="hann",
            center=False,
            log_eps=1e-5,
            mel_scale="slaney",
        ).to(self._device)
        # Auxiliary mel loss, only when weighted in. Config validation forbids
        # aux_mel_weight > 0 for BigVGAN, so this is always a raw-audio codec.
        self.mel_aux_loss: MelAuxLoss | None = (
            MelAuxLoss(
                sample_rate=sample_rate, hop_length=codec.hop_length
            ).to(self._device)
            if config.aux_mel_weight > 0
            else None
        )

    def _prepare_acoustic(self, batch: Batch) -> MaskedTensor:
        return prepare_acoustic_batch(batch, self.codec, self.model.cfg.eos_n_frames)

    def _ema_weights(self):
        """EMA weights installed for the block, or a no-op when EMA is off."""
        return (
            self.ema.swapped(self.model)
            if self.ema is not None
            else nullcontext()
        )

    def train(self):
        self._log_initial_samples()
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
                                {"ema": self.ema.state_dict()}
                                if self.ema is not None
                                else None
                            ),
                        )

                if hasattr(self.optimizer, "train"):
                    self.optimizer.train()
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
                        diag_accum = {}
                        diag_micro = 0
                        last_log_step = self.step
                        last_log_time = time.perf_counter()

                    if self.step % self.config.hist_steps == 0:
                        self._emit_loss_histogram(bin_accum, self.step, "train")
                        bin_accum = {}

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
            # The parametrization owns the loss form (MSE for RectifiedFlow,
            # reweighted L1 for JWT) — model.training_step returns it masked.
            # x_pred is x_1 recovered in normalized space, decoded below to a
            # waveform for the auxiliary mel loss when it is enabled.
            out = self.model.training_step(text, acoustic)
            fm_loss, x_pred, v_mask = out.loss, out.x_pred, out.v_mask

        loss = fm_loss
        metrics: dict[str, torch.Tensor] = {"fm_loss": fm_loss.detach()}

        # Auxiliary log-mel L1 — a perceptual loss term for codecs whose
        # FM-space loss isn't itself perceptual (raw-audio patches). Decodes the
        # predicted and target features to waveforms and compares them in mel
        # space. fp32 outside autocast for STFT stability. `mel_aux_loss` is
        # None for BigVGAN, whose FM loss is already mel-space.
        if self.mel_aux_loss is not None:
            with torch.autocast(device_type=self.device.type, enabled=False):
                x_1_target_norm = acoustic.values.float()
                x_1_pred_norm = x_pred.float().transpose(1, 2)
                pred_wav = self.codec.decode(self.codec.unnormalize(x_1_pred_norm))
                target_wav = self.codec.decode(
                    self.codec.unnormalize(x_1_target_norm)
                )
                logmel_l1 = self.mel_aux_loss(pred_wav, target_wav, v_mask)
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

    def _step_diagnostics(
        self, out: TrainingStepOutput, text: MaskedTensor, acoustic: MaskedTensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-micro-step diagnostics — all kept on-GPU (no host sync).

        Returns `(scalars, bins)`: `scalars` are 0-dim tensors the caller
        averages over the logging window; `bins` holds the loss-by-t
        `(sum, count)` tensors the caller accumulates for the histogram.
        """
        bin_sums, bin_counts = binned_loss_stats(
            out.per_pos_loss, out.t, out.v_mask, self.config.n_loss_bins
        )
        ac_mean, ac_std = masked_mean_std(acoustic.values, out.v_mask)
        scalars = {
            "vmask_fill": out.v_mask.float().mean(),
            "ac_len_mean": acoustic.mask.sum(-1).float().mean(),
            "text_len_mean": text.mask.sum(-1).float().mean(),
            "ac_target_mean": ac_mean,
            "ac_target_std": ac_std,
        }
        bins = {"bin_sums": bin_sums, "bin_counts": bin_counts}
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
        """Emit the FM loss binned by timestep t as a precomputed histogram.

        `bin_means = bin_sums / bin_counts` is exact over the window; an empty
        bin (count 0 -> NaN) is zero-filled. One host transfer.
        """
        bin_means = diag_accum["bin_sums"] / diag_accum["bin_counts"]
        values = torch.nan_to_num(bin_means.detach().float(), nan=0.0).cpu().tolist()
        n = self.config.n_loss_bins
        edges = [i / n for i in range(n + 1)]
        self.logger.log_histogram(f"{prefix}/fm_loss_by_t", edges, values, step)

    def _emit_diagnostics(
        self,
        gpu_metrics: dict[str, torch.Tensor],
        step: int,
        prefix: str,
        host_metrics: dict[str, float] | None = None,
    ) -> None:
        """Single batched host transfer, then hand off to the logger."""
        keys = list(gpu_metrics)
        stacked = torch.stack(
            [gpu_metrics[k].detach().float().reshape(()) for k in keys]
        )
        merged = dict(zip(keys, stacked.cpu().tolist()))  # one D2H transfer
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
        # diagnostics (param norm, throughput, memory) stay under `train`.
        data_metrics = self._reduce_diagnostics(diag_accum, diag_micro)
        train_metrics: dict[str, torch.Tensor] = {
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
            host["peak_mem_gb"] = (
                torch.cuda.max_memory_allocated(self.device) / 1e9
            )
            torch.cuda.reset_peak_memory_stats(self.device)

        self._emit_diagnostics(data_metrics, self.step, "data/train")
        self._emit_diagnostics(train_metrics, self.step, "train", host_metrics=host)

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
            self.optimizer.eval()

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
        self._emit_loss_histogram(bin_accum, self.step, "valid")

        loss_val = val_metrics.get("loss", float("inf"))
        if loss_val < self.best_loss:
            self.state.best_loss = loss_val

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
            self.logger.log_audio(
                f"sampled/{i}", wav, self.step, self.sample_rate
            )
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
            viz_mel = self.viz_mel(wav.unsqueeze(0))[0]
            log_mel(self.logger, f"sampled/{i}/mel", viz_mel, self.step)

    def _log_initial_samples(self):
        smp_batch = next(iter(self.smp_dloader))
        n = min(len(smp_batch.audios), self.config.n_smp)

        for i, audio in enumerate(smp_batch.audios[:n]):
            self.logger.log_audio(
                f"{i}/clean", audio.waveform, 0, audio.sample_rate
            )

            waveform = audio.waveform.to(self.device)
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=not self.noamp,
            ):
                reconstructed = self.codec.reconstruct(waveform[None])[0]
                clean_viz = self.viz_mel(waveform)
                recon_viz = self.viz_mel(reconstructed.unsqueeze(0))[0]
            self.logger.log_audio(
                f"{i}/reconstructed",
                reconstructed,
                0,
                self.sample_rate,
            )
            log_mel(self.logger, f"{i}/clean/mel", clean_viz, 0)
            log_mel(self.logger, f"{i}/reconstructed/mel", recon_viz, 0)

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tts.data.audio.codecs import Codec
from tts.data.dataset import Batch
from tts.model.neural_speaker import MaskedTensor, RollingFlowSpeaker
from tts.training.checkpoint_manager import CheckpointManager
from tts.training.loggers import Logger


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
    clip_grad_norm: float | None
    device: torch.device
    amp_dtype: AMPDtype
    smp_steps: int
    valid_steps: int
    checkpoint_steps: int
    max_steps: int
    noamp: bool
    n_smp: int = 8
    grad_accum_steps: int = 1


class Trainer:
    def __init__(self, config: TrainerConfig, state: TrainerState | None = None):
        self.config = config
        self.state = state or TrainerState(step=0)

    @property
    def device(self):
        return self.config.device

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
        return self.config.noamp

    @property
    def step(self):
        return self.state.step

    @property
    def best_loss(self):
        return self.state.best_loss


class TTSRollingFlowMatchingTrainer(Trainer):
    def __init__(
        self,
        config: TrainerConfig,
        codec: Codec | None,
        model: RollingFlowSpeaker,
        optimizer: Optimizer,
        scaler: GradScaler | None,
        logger: Logger,
        train_dloader: DataLoader,
        valid_dloader: DataLoader,
        smp_dloader: DataLoader,
        state: TrainerState | None,
        checkpoint_manager: CheckpointManager | None,
    ):
        super().__init__(config=config, state=state)
        self.codec = codec
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.logger = logger
        self.train_dloader = train_dloader
        self.valid_dloader = valid_dloader
        self.smp_dloader = smp_dloader
        self.checkpoint_manager = checkpoint_manager

    def train(self):
        self._log_initial_samples()
        self.optimizer.zero_grad()
        grad_accum_steps = self.config.grad_accum_steps

        micro = 0
        accum: dict[str, float] = {}

        while self.step < self.max_steps:
            for batch in self.train_dloader:
                if micro == 0:
                    if self.step % self.smp_steps == 0:
                        self._log_samples()
                    if self.step % self.valid_steps == 0:
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
                        )

                if hasattr(self.optimizer, "train"):
                    self.optimizer.train()
                self.model.train()

                metrics = self.training_step(batch)
                for k, v in metrics.items():
                    accum[k] = accum.get(k, 0.0) + float(v)
                micro += 1

                if micro >= grad_accum_steps:
                    opt_metrics = self._optimizer_step()
                    avg = {k: v / micro for k, v in accum.items()}
                    avg.update({k: float(v) for k, v in opt_metrics.items()})
                    self.logger.log_metrics(avg, self.step, prefix="train")
                    self.logger.update_progress(1)
                    self.state.step += 1
                    micro = 0
                    accum = {}

                    if self.step >= self.max_steps:
                        self.logger.close()
                        return

    def training_step(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Single micro-step: forward + (under training) scaled backward.

        Loss is divided by grad_accum_steps before backward so accumulated
        gradients average across the window. The returned metric is the
        unscaled loss.
        """
        batch = batch.to(self.device)
        mels_values = batch.mels
        if mels_values.ndim == 4 and mels_values.shape[1] == 1:
            mels_values = mels_values.squeeze(1)
        text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)
        mels = MaskedTensor(values=mels_values, mask=batch.mels_mask)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            v_pred, target, loss_mask = self.model.training_step(text, mels)
            per_pos = (v_pred - target).pow(2).mean(-1)  # (B, T_mel)
            loss = (per_pos * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        metrics: dict[str, torch.Tensor] = {"loss": loss.detach()}

        if self.model.training:
            scaled = loss / self.config.grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

        return metrics

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
        self.optimizer.zero_grad()
        return metrics

    @torch.inference_mode()
    def validation(self):
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        sums: dict[str, float] = {}
        count = 0
        for vbatch in self.valid_dloader:
            metrics = self.training_step(vbatch)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            count += 1

        if count == 0:
            return

        val_metrics = {k: v / count for k, v in sums.items()}
        self.logger.log_metrics(val_metrics, self.step, prefix="valid")

        loss_val = val_metrics.get("loss", float("inf"))
        if loss_val < self.best_loss:
            self.state.best_loss = loss_val

    @torch.inference_mode()
    def _log_samples(self):
        if self.codec is None:
            return

        self.model.eval()
        batch = next(iter(self.smp_dloader)).to(self.device)
        n = min(len(batch.audios), self.config.n_smp)

        text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)
        mel_lens = batch.mels_mask.sum(-1).clamp(min=1)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            mels_pred = self.model.speak(text, mel_lens)
            wavs = self.codec.decode(mels_pred.values[:n])

        for i, wav in enumerate(wavs):
            self.logger.log_audio(
                f"sampled/{i}", wav, self.step, self.codec.sample_rate
            )

    def _log_initial_samples(self):
        smp_batch = next(iter(self.smp_dloader))
        n = min(len(smp_batch.audios), self.config.n_smp)

        for i, audio in enumerate(smp_batch.audios[:n]):
            self.logger.log_audio(
                f"{i}/clean", audio.waveform, 0, audio.sample_rate
            )

            if self.codec is not None:
                waveform = audio.waveform.to(self.device)
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=not self.noamp,
                ):
                    reconstructed = self.codec.reconstruct(waveform[None])[0]
                self.logger.log_audio(
                    f"{i}/reconstructed",
                    reconstructed,
                    0,
                    self.codec.sample_rate,
                )

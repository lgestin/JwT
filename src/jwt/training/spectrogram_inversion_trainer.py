"""Trainer for mel-spectrogram inversion (`SpectrogramInverter`).

Subclasses the TTS trainer to inherit its loop, optimizer step, EMA,
checkpointing, validation and histogram machinery, and swaps the data path:
batches are fixed-length `WindowBatch` waveforms, the mel conditioning is
computed on-the-fly from them, and there is no text and no EOS.
"""

import torch

from jwt.data.window_dataset import WindowBatch
from jwt.model.neural_speaker import MaskedTensor
from jwt.model.spectrogram_inverter import SpectrogramInverter
from jwt.training.loggers import log_mel
from jwt.training.trainer import TTSRollingFlowMatchingTrainer


class SpectrogramInversionTrainer(TTSRollingFlowMatchingTrainer):
    model: SpectrogramInverter

    def _prepare_batch(self, batch: WindowBatch) -> tuple[MaskedTensor, MaskedTensor]:
        """Build the (acoustic, mel) MaskedTensor pair from window waveforms.

        Windows are uniform-length so both masks are all-ones; they exist to
        satisfy the model/diagnostics interfaces. The mel is computed in fp32
        outside autocast (STFT stability); hop == patch size guarantees the
        frame grids align.
        """
        wav = batch.waveform  # (B, S), unnormalized
        patches = self.codec.encode(wav)  # (B, patch_size, T) — a pure rearrange
        mask = torch.ones(
            patches.shape[0], patches.shape[-1], dtype=torch.bool, device=wav.device
        )
        acoustic = MaskedTensor(values=self.codec.normalize(patches), mask=mask)  # ty: ignore[invalid-argument-type]
        logmel = self.model.encode_mel(wav.float())
        assert logmel.shape[-1] == patches.shape[-1], (
            f"mel frames ({logmel.shape[-1]}) != patches ({patches.shape[-1]})"
        )
        mel = MaskedTensor(values=logmel, mask=mask)  # ty: ignore[invalid-argument-type]
        return acoustic, mel

    def training_step(  # ty: ignore[invalid-method-override]  # WindowBatch, not Batch
        self, batch: WindowBatch
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Single micro-step: forward + (under training) scaled backward.

        Mirrors the parent's contract: returns `(metrics, scalars, bins)` with
        a `"loss"` key in `metrics` and the histogram tensors in `bins`.
        """
        batch = batch.to(self.device, non_blocking=True)
        acoustic, mel = self._prepare_batch(batch)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            out = self.model.training_step(
                mel,
                acoustic,
                loss_fn=self.config.loss_fn.fn,
                attention_implementation=self.config.attention_implementation.implementation,
            )
            fm_loss, x_pred, v_mask = out.loss, out.x_pred, out.v_mask

        loss = fm_loss
        metrics: dict[str, torch.Tensor] = {"fm_loss": fm_loss.detach()}

        # Auxiliary log-mel L1, identical to the TTS trainer. Note the target
        # mel is also the model's conditioning, so this loss is partially
        # redundant here — set aux_mel_weight to 0 to disable it.
        logmel_diff: torch.Tensor | None = None
        if self.mel_aux_loss is not None:
            with torch.autocast(device_type=self.device.type, enabled=False):
                x_1_target_norm = acoustic.values.float()
                x_1_pred_norm = x_pred.float().transpose(1, 2)
                pred_wav = self.codec.decode(self.codec.unnormalize(x_1_pred_norm))
                target_wav = self.codec.decode(self.codec.unnormalize(x_1_target_norm))
                logmel_l1, logmel_diff = self.mel_aux_loss(pred_wav, target_wav, v_mask)
            loss = fm_loss + self.config.aux_mel_weight * logmel_l1
            metrics["logmel_l1"] = logmel_l1.detach()

        metrics["loss"] = loss.detach()
        # mel rides in the text slot: _step_diagnostics only reads its
        # .mask.sum(-1), so data/*/text_len_mean reports the (constant) window
        # length in frames.
        scalars, bins = self._step_diagnostics(out, mel, acoustic, logmel_diff)

        if self.model.training:
            scaled = loss / self.config.grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

        return metrics, scalars, bins

    @torch.inference_mode()
    def _log_attention_maps(self) -> None:
        """No-op: the text->audio attention probe has no meaning here."""

    @torch.inference_mode()
    def _log_samples(self):
        self.model.eval()
        batch = next(iter(self.smp_dloader)).to(self.device)
        n = min(len(batch.idxs), self.config.n_smp)

        _, mel = self._prepare_batch(batch)
        mel_n = MaskedTensor(values=mel.values[:n], mask=mel.mask[:n])  # ty: ignore[invalid-argument-type]

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            acoustic_pred = self.model.invert(mel_n)

        for i in range(n):
            ac_i = acoustic_pred.values[i : i + 1]  # (1, acoustic_dim, T), normalized
            wav = self.codec.decode(self.codec.unnormalize(ac_i))[0]
            self.logger.log_audio(f"sampled/{i}", wav, self.step, self.sample_rate)
            viz_mel = self.viz_mel(wav.unsqueeze(0))[0].clamp(min=1e-5).log()
            log_mel(self.logger, f"sampled/{i}/mel", viz_mel, self.step)

    def _log_initial_samples(self):
        """Log the clean reference windows once at startup.

        No codec-reconstruction pair: patch encode/decode is an exact identity
        for raw-audio windows.
        """
        batch = next(iter(self.smp_dloader))
        n = min(len(batch.idxs), self.config.n_smp)

        for i in range(n):
            waveform = batch.waveform[i]
            self.logger.log_audio(f"{i}/clean", waveform, 0, self.sample_rate)
            with torch.no_grad():
                clean_viz = (
                    self.viz_mel(waveform.unsqueeze(0).to(self.device))[0]
                    .clamp(min=1e-5)
                    .log()
                )
            log_mel(self.logger, f"{i}/clean/mel", clean_viz, 0)

"""Training loss components that are not owned by the model or the codec.

`MultiResComplexSTFTAuxLoss` is the codec-agnostic auxiliary loss: it compares
a predicted and a target waveform in complex STFT space at several window
sizes, taking L1 on stacked (real, imag). Unlike a magnitude / mel loss, this
directly supervises phase, since equal magnitudes with mismatched phases
produce different complex coefficients.

`MelL1Monitor` is the same log-mel L1 that used to be the auxiliary loss term,
now kept only as a metric — the trainer evaluates it without a gradient so
training runs stay comparable to the older mel-only baseline by inspection.

The trainer decodes the codec features to waveforms and passes them here, so
this module knows nothing about any particular codec.
"""

import torch
import torch.nn as nn

from jwt.data.audio.stft import STFT, MelSpectrogram


class MultiResComplexSTFTAuxLoss(nn.Module):
    """v_mask-weighted L1 on (real, imag) of the complex STFT, summed over scales.

    Every scale's hop is fixed to the codec's hop_length and `center=False` is
    used with internal reflect padding, so each scale's time axis equals T_ac
    frame-for-frame. That alignment is what lets the per-position diagnostic
    and the v_mask weighting be honest.
    """

    def __init__(
        self,
        hop_length: int,
        n_ffts: tuple[int, ...] = (512, 1024, 2048),
    ):
        super().__init__()
        assert all(n >= hop_length for n in n_ffts), (
            f"every n_fft must be >= hop_length ({hop_length}); got {n_ffts}"
        )
        self.stfts = nn.ModuleList(
            [
                STFT(n_fft=n, hop_length=hop_length, window="hann", center=False)
                for n in n_ffts
            ]
        )

    def forward(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """pred_wav/target_wav: (B, S). v_mask: (B, T).

        Returns ``(loss, per_pos)``. ``loss`` is the 0-dim v_mask-weighted mean
        averaged across scales; ``per_pos`` is the detached ``(B, T)`` per-frame
        L1 from the *first* scale, used as a phase-binning diagnostic and the
        ``aux_stft_l1_by_t`` series.
        """
        per_pos_diag: torch.Tensor | None = None
        loss = pred_wav.new_zeros(())
        for stft_mod in self.stfts:
            assert isinstance(stft_mod, STFT)
            s_pred = stft_mod.stft(pred_wav)  # (B, F, T) complex, T == v_mask.shape[-1]
            s_target = stft_mod.stft(target_wav)
            assert s_pred.shape[-1] == v_mask.shape[-1], (
                f"stft time axis ({s_pred.shape[-1]}) must match v_mask "
                f"({v_mask.shape[-1]}) so the supervision window aligns"
            )
            d = s_pred - s_target
            per_pos = (d.real.abs() + d.imag.abs()).mean(dim=1)  # (B, T)
            loss = loss + (per_pos * v_mask).sum() / v_mask.sum().clamp(min=1)
            if per_pos_diag is None:
                per_pos_diag = per_pos.detach()
        assert per_pos_diag is not None
        return loss / len(self.stfts), per_pos_diag


class MelL1Monitor(nn.Module):
    """v_mask-weighted log-mel L1 between predicted and target waveforms.

    Same shape contract and time-axis alignment as ``MultiResComplexSTFTAuxLoss``:
    a waveform of ``T * hop_length`` samples maps to exactly ``T`` mel frames so
    the v_mask aligns frame-for-frame.

    Intended as a *monitor* rather than a training term — the trainer calls it
    under ``torch.no_grad`` so it contributes no gradient. Kept around as a
    metric so runs using the new complex-STFT loss stay comparable, by
    inspection, to the older mel-only baseline.
    """

    def __init__(self, sample_rate: int, hop_length: int, n_mels: int = 80):
        super().__init__()
        self.mel = MelSpectrogram(
            n_fft=4 * hop_length,
            hop_length=hop_length,
            n_mels=n_mels,
            sample_rate=sample_rate,
            window="hann",
            center=False,
            mel_scale="slaney",
        )

    def forward(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """pred_wav/target_wav: (B, S). v_mask: (B, T).

        Returns ``(loss, per_pos)`` matching the STFT loss's contract. Caller
        is responsible for running this outside the autograd graph (e.g. inside
        ``torch.no_grad``) if it shouldn't contribute to training.
        """
        logmel_pred = self.mel(pred_wav).clamp(min=1e-5).log()
        logmel_target = self.mel(target_wav).clamp(min=1e-5).log()
        assert logmel_pred.shape == logmel_target.shape, (
            f"pred/target mels differ in shape: {logmel_pred.shape} vs {logmel_target.shape}"
        )
        assert logmel_pred.shape[-1] == v_mask.shape[-1], (
            f"mel time axis ({logmel_pred.shape[-1]}) must match v_mask "
            f"({v_mask.shape[-1]}) so the supervision window aligns"
        )
        logmel_diff = (logmel_pred - logmel_target).abs().mean(dim=1)  # (B, T)
        loss = (logmel_diff * v_mask).sum() / v_mask.sum().clamp(min=1)
        return loss, logmel_diff.detach()

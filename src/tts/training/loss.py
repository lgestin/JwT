"""Training loss components that are not owned by the model or the codec.

`MelAuxLoss` is the codec-agnostic auxiliary log-mel L1 loss: it compares a
predicted and a target waveform in log-mel space. The trainer decodes the codec
features to waveforms and passes them here, so this module knows nothing about
any particular codec.
"""

import torch
import torch.nn as nn

from tts.data.audio.stft import MelSpectrogram


class MelAuxLoss(nn.Module):
    """v_mask-weighted log-mel L1 between a predicted and a target waveform.

    The mel `hop_length` is the codec's `hop_length`, so a waveform of
    `T * hop_length` samples maps to exactly `T` mel frames and a v_mask in
    acoustic time aligns frame-for-frame with the mel time axis.
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
    ) -> torch.Tensor:
        """pred_wav/target_wav: (B, S). v_mask: (B, T). Returns a 0-dim tensor."""
        logmel_pred = self.mel(pred_wav).clamp(min=1e-5).log()
        logmel_target = self.mel(target_wav).clamp(min=1e-5).log()
        assert logmel_pred.shape == logmel_target.shape, (
            f"pred/target mels differ in shape: "
            f"{logmel_pred.shape} vs {logmel_target.shape}"
        )
        assert logmel_pred.shape[-1] == v_mask.shape[-1], (
            f"mel time axis ({logmel_pred.shape[-1]}) must match v_mask "
            f"({v_mask.shape[-1]}) so the supervision window aligns"
        )
        logmel_diff = (logmel_pred - logmel_target).abs().mean(dim=1)  # (B, T)
        return (logmel_diff * v_mask).sum() / v_mask.sum().clamp(min=1)

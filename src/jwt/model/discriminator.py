"""Multi-resolution STFT discriminator for adversarial waveform training.

A Voxtral / EnCodec-style discriminator: one sub-discriminator per STFT
resolution, each a small 2D-conv stack over the complex spectrogram (real and
imaginary parts stacked as two channels). Each sub-discriminator returns a logit
map and its per-layer feature maps; the trainer uses the logits for the hinge
GAN loss and the feature maps for the L1 feature-matching loss.

The discriminator consumes the *decoded waveform* the trainer already produces
for the auxiliary mel loss, so it is codec-agnostic — it only sees `(B, S)`
waveforms. STFT runs in fp32 (outside autocast at the call site) for stability,
matching `MelAuxLoss`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

# A per-resolution discriminator output: (logit map, list of layer feature maps).
DiscOutput = tuple[torch.Tensor, list[torch.Tensor]]


class STFTDiscriminator(nn.Module):
    """A single-resolution complex-STFT discriminator.

    Computes a complex STFT, stacks (real, imag) as two channels, and runs a
    strided 2D-conv stack over the (frequency, time) plane. `forward` returns the
    final logit map and every intermediate activation (for feature matching).
    """

    window: torch.Tensor

    def __init__(self, n_fft: int, hop_length: int, channels: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.register_buffer("window", torch.hann_window(n_fft))

        ch = channels
        # 2 input channels (real, imag). Stride in frequency progressively shrinks
        # the spectral axis; time stride stays modest so logits keep temporal
        # resolution. weight_norm follows the standard GAN-vocoder convention.
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, ch, (3, 3), padding=(1, 1))),
                weight_norm(nn.Conv2d(ch, ch, (3, 3), stride=(2, 1), padding=(1, 1))),
                weight_norm(nn.Conv2d(ch, ch, (3, 3), stride=(2, 2), padding=(1, 1))),
                weight_norm(nn.Conv2d(ch, ch, (3, 3), stride=(2, 1), padding=(1, 1))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def _spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, S) waveform -> (B, 2, F, T) real/imag stacked complex STFT."""
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (B, F, T) complex
        spec = torch.view_as_real(spec)  # (B, F, T, 2)
        return spec.permute(0, 3, 1, 2).contiguous()  # (B, 2, F, T)

    def forward(self, wav: torch.Tensor) -> DiscOutput:
        x = self._spectrogram(wav)
        features: list[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        logits = self.conv_post(x)
        features.append(logits)
        return logits, features


class MultiResolutionSTFTDiscriminator(nn.Module):
    """A bank of complex-STFT discriminators at several FFT resolutions.

    `forward` returns one `(logits, features)` per resolution. The default
    resolutions are a generic 5-scale set; the exact Voxtral set is 24 kHz
    specific, so the resolutions are configurable via `TrainerConfig.disc_n_fft`.
    """

    def __init__(
        self,
        n_ffts: tuple[int, ...] = (2048, 1024, 512, 256, 128),
        channels: int = 32,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                STFTDiscriminator(n_fft=n, hop_length=n // 4, channels=channels)
                for n in n_ffts
            ]
        )

    def forward(self, wav: torch.Tensor) -> list[DiscOutput]:
        return [d(wav) for d in self.discriminators]

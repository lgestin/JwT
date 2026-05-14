from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: Literal["hann", "hamming"] = "hamming",
        center: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        if window == "hann":
            window_tensor = torch.hann_window(n_fft)
        elif window == "hamming":
            window_tensor = torch.hamming_window(n_fft)
        else:
            raise ValueError(f"Unknown window type: {window!r}")
        self.register_buffer("window", window_tensor, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.magnitudes(x)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        self.window = self.window.to(x.device)
        if self.center:
            x = x.squeeze(1) if x.dim() == 3 else x
        else:
            p = (self.n_fft - self.hop_length) // 2
            x = F.pad(x, (p, p), "reflect").squeeze(1)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            normalized=False,
            center=self.center,
            return_complex=True,
        )
        return stft

    def istft(self, stft: torch.Tensor, length: int | None = None) -> torch.Tensor:
        self.window = self.window.to(stft.device)
        if self.center:
            waveform = torch.istft(
                stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                normalized=False,
                center=True,
                return_complex=False,
                length=length,
            )
        else:
            p = (self.n_fft - self.hop_length) // 2
            target_length = None if length is None else length + 2 * p
            waveform = torch.istft(
                stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                normalized=False,
                center=False,
                return_complex=False,
                length=target_length,
            )[..., p:-p]
        return waveform

    def magnitudes(self, x: torch.Tensor) -> torch.Tensor:
        stft = self.stft(x)
        magnitudes = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-9)
        return magnitudes


class MelSpectrogram(STFT):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        sample_rate: int,
        f_min: float = 0.0,
        f_max: float = torch.inf,
        window: Literal["hann", "hamming"] = "hamming",
        center: bool = False,
        log_eps: float | None = None,
        mel_scale: Literal["htk", "slaney"] = "htk",
    ):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, window=window, center=center
        )
        self.log_eps = log_eps
        melscale_fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=max(f_min, 0.0),
            f_max=min(f_max, sample_rate // 2),
            n_mels=n_mels,
            norm="slaney",
            mel_scale=mel_scale,
            sample_rate=sample_rate,
        )
        self.register_buffer("melscale_fbanks", melscale_fbanks, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitudes = super().magnitudes(x)
        self.melscale_fbanks = self.melscale_fbanks.to(magnitudes.device)
        mels = self.melscale_fbanks.T @ magnitudes
        if self.log_eps is not None:
            mels = torch.log(mels.clamp(min=self.log_eps))
        return mels

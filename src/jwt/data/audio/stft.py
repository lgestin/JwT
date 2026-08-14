import math
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
        if not self.center:
            p = (self.n_fft - self.hop_length) // 2
            if length is not None:
                length = length + 2 * p
        waveform = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            normalized=False,
            center=self.center,
            return_complex=False,
            length=length,
        )
        if not self.center and p > 0:
            waveform = waveform[..., p:-p]
        return waveform

    def magnitudes(self, x: torch.Tensor) -> torch.Tensor:
        stft = self.stft(x)
        magnitudes = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-9)
        return magnitudes

    def log_magnitudes(
        self,
        x: torch.Tensor,
        eps: float = 1e-8,
        log_base: float = math.e,
    ) -> torch.Tensor:
        x_logstft = self.magnitudes(x).clamp(min=eps).log() / math.log(log_base)
        return x_logstft

    def logstft_l1(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        eps: float = 1e-8,
        log_base: float = math.e,
    ) -> torch.Tensor:
        pred_logstft = self.log_magnitudes(pred, eps=eps, log_base=log_base)
        trgt_logstft = self.log_magnitudes(trgt, eps=eps, log_base=log_base)
        logstft_l1 = F.l1_loss(pred_logstft, trgt_logstft, reduction="none").mean(1)
        return logstft_l1


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
        mel_scale: Literal["htk", "slaney"] = "htk",
        n_mfcc: int | None = None,
    ):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, window=window, center=center
        )
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

        if n_mfcc is not None:
            dct = torchaudio.functional.create_dct(
                n_mfcc=n_mfcc + 1,
                n_mels=n_mels,
                norm="ortho",
            )
            self.register_buffer("dct", dct, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mel(x)

    def mel(self, x: torch.Tensor) -> torch.Tensor:
        magnitudes = super().magnitudes(x)
        self.melscale_fbanks = self.melscale_fbanks.to(magnitudes.device)
        mels = self.melscale_fbanks.T @ magnitudes
        return mels

    def logmel(
        self,
        x: torch.Tensor,
        eps: float = 1e-8,
        log_base: float = math.e,
    ) -> torch.Tensor:
        return self.mel(x).clamp(min=eps).log() / math.log(log_base)

    def logmel_l1(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor | None = None,
        eps: float = 1e-8,
        log_base: float = math.e,
    ) -> torch.Tensor:
        pred_logmel = self.logmel(pred, eps=eps, log_base=log_base)
        trgt_logmel = self.logmel(trgt, eps=eps, log_base=log_base)
        logmel_l1 = F.l1_loss(pred_logmel, trgt_logmel, reduction="none").mean(1)
        return logmel_l1

    def mel_cepstral_distortion(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        eps: float = 1e-8,
    ):
        pred_logmel = self.logmel(pred, eps=eps, log_base=math.e)
        trgt_logmel = self.logmel(trgt, eps=eps, log_base=math.e)
        pred_cep = torch.einsum("bmt,mk->bkt", pred_logmel, self.dct)[:, 1:]
        trgt_cep = torch.einsum("bmt,mk->bkt", trgt_logmel, self.dct)[:, 1:]
        cep_l2 = (
            20 / math.log(10) * F.mse_loss(pred_cep, trgt_cep, reduction="none").mean(1)
        )
        return cep_l2

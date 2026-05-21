from enum import StrEnum
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tts.data.audio.stft import MelSpectrogram


# Patch sizes for the RawAudio codec variants. Each variant is its own codec
# identity: str(variant).lower() names the acoustic_{name} arrow column, so the
# patch size baked into the precomputed dataset and the runtime codec can't drift.
_RAWAUDIO_PATCH = {
    "RawAudio32": 32,
    "RawAudio64": 64,
    "RawAudio128": 128,
    "RawAudio256": 256,
}


class Codecs(StrEnum):
    BIGVGAN = "BigVGAN"
    RAWAUDIO_32 = "RawAudio32"
    RAWAUDIO_64 = "RawAudio64"
    RAWAUDIO_128 = "RawAudio128"
    RAWAUDIO_256 = "RawAudio256"

    @property
    def codec_class(self) -> type:
        match self:
            case Codecs.BIGVGAN:
                return BigVGAN
            case _:
                return RawAudioPatcher

    @property
    def codec(self):
        match self:
            case Codecs.BIGVGAN:
                return BigVGAN()
            case _:
                return RawAudioPatcher(patch_size=_RAWAUDIO_PATCH[self.value])


@runtime_checkable
class Codec(Protocol):
    """Audio representation interface.

    encode/decode/eos_frames/is_eos all operate in the codec's native
    (unnormalized) space. The model operates on normalize()'d features.
    Shape contract: (B, acoustic_dim, T).
    """

    sample_rate: int
    acoustic_dim: int
    hop_length: int

    def encode(self, waveform: torch.Tensor) -> torch.Tensor: ...
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor: ...
    def eos_frames(
        self, n: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Unnormalized sentinel frames, shape (acoustic_dim, n)."""
        ...

    def is_eos(self, frame: torch.Tensor) -> torch.BoolTensor:
        """Test unnormalized frames, shape (..., acoustic_dim) -> (...,)."""
        ...

    def to_logmel(self, features: torch.Tensor) -> torch.Tensor:
        """Map unnormalized features to a log-mel spectrogram for monitoring.

        Input shape: (B, acoustic_dim, T). Output shape: (B, n_mels, T_mel).
        The output time axis must align frame-for-frame with the input so a
        v_mask in acoustic-time can be reused as a mel-time mask. For codecs
        where features are already log-mel, this is identity.
        """
        ...

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))


class BigVGANVersions(StrEnum):
    V2_24KHz_100MEL_256X = "nvidia/bigvgan_v2_24khz_100band_256x"


# Global log-mel stats fit on LJSpeech at 24 kHz with this BigVGAN's mel front-end.
_LJSPEECH_LOG_MEL_MEAN = -5.896610
_LJSPEECH_LOG_MEL_STD = 2.226763


class BigVGAN(nn.Module):
    def __init__(self, version: BigVGANVersions = BigVGANVersions.V2_24KHz_100MEL_256X):
        nn.Module.__init__(self)
        # local: avoid heavy bigvgan import at module load
        from bigvgan.bigvgan import BigVGAN as _BigVGAN
        from bigvgan.bigvgan import load_hparams_from_json
        from huggingface_hub import hf_hub_download

        model_id = str(version)
        config_file = hf_hub_download(repo_id=model_id, filename="config.json")
        weights_file = hf_hub_download(
            repo_id=model_id, filename="bigvgan_generator.pt"
        )

        h = load_hparams_from_json(config_file)
        self.decoder = _BigVGAN(h, use_cuda_kernel=False)
        # The published checkpoint already has weight norm stripped; load_state_dict succeeds directly.
        checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)
        self.decoder.load_state_dict(checkpoint["generator"])
        self.decoder.remove_weight_norm()
        self.decoder.eval()

        self.mel_spectrogram = MelSpectrogram(
            n_fft=h.n_fft,
            hop_length=h.hop_size,
            n_mels=h.num_mels,
            sample_rate=h.sampling_rate,
            f_min=h.fmin,
            f_max=torch.inf,
            window="hann",
            center=False,
            log_eps=1e-5,
            mel_scale="slaney",
        )
        self.sample_rate = int(h.sampling_rate)
        self.acoustic_dim = int(h.num_mels)
        self.hop_length = int(h.hop_size)

        self.register_buffer(
            "mel_mean", torch.tensor(_LJSPEECH_LOG_MEL_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "mel_std", torch.tensor(_LJSPEECH_LOG_MEL_STD, dtype=torch.float32)
        )

        # Sentinel level chosen well below the noise floor of normalized log-mels
        # so denoising drives unused tail to it.
        self.eos_value: float = -15.0
        # Frame-mean threshold for declaring a frame an EOS sentinel.
        self.eos_threshold: float = -11.0

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel_spectrogram(waveform)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mel_mean) / self.mel_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mel_std + self.mel_mean

    def to_logmel(self, features: torch.Tensor) -> torch.Tensor:
        # Features are already log-mel; identity preserves the (B, n_mels, T) layout.
        return features

    def eos_frames(
        self,
        n: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.full(
            (self.acoustic_dim, n), self.eos_value, device=device, dtype=dtype
        )

    def is_eos(self, frame: torch.Tensor) -> torch.BoolTensor:
        return frame.mean(dim=-1) < self.eos_threshold


class RawAudioPatcher(nn.Module):
    """Codec that turns a 1D waveform into a sequence of non-overlapping patches.

    encode reshapes (B, S) (or (B, 1, S)) into (B, patch_size, n_patches).
    decode does the inverse. Both are deterministic — the model's input/output
    projection is responsible for learning a useful per-patch embedding.
    """

    def __init__(self, patch_size: int = 256, sample_rate: int = 24000):
        nn.Module.__init__(self)
        self.patch_size = patch_size
        self.acoustic_dim = patch_size
        self.hop_length = patch_size
        self.sample_rate = sample_rate
        self.eos_threshold: float = 1e-4

        # In __init__: waveform is RMS-normalized to -24 dBFS upstream
        # (create_arrow_ljspeech.py target_loudness), so per-sample std ~ 10**(-24/20).
        wav_std = 10 ** (-24.0 / 20)  # ~0.0631
        self.register_buffer("wav_std", torch.tensor(wav_std, dtype=torch.float32))

        # Internal mel for the codec-agnostic monitoring metric. hop_length is
        # tied to patch_size so to_logmel preserves the acoustic-time axis and
        # the trainer's v_mask aligns frame-for-frame.
        self._monitor_mel = MelSpectrogram(
            n_fft=4 * patch_size,
            hop_length=patch_size,
            n_mels=80,
            sample_rate=sample_rate,
            window="hann",
            center=False,
            log_eps=1e-5,
            mel_scale="slaney",
        )

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            assert waveform.shape[1] == 1, "expected mono input (B, 1, S) or (B, S)"
            waveform = waveform.squeeze(1)
        S = waveform.shape[-1]
        pad = (-S) % self.patch_size
        if pad:
            waveform = F.pad(waveform, (0, pad))
        return rearrange(waveform, "b (t p) -> b p t", p=self.patch_size)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return rearrange(z, "b p t -> b (t p)")

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.wav_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.wav_std

    def to_logmel(self, features: torch.Tensor) -> torch.Tensor:
        # Inline the rearrange (instead of calling self.decode, which is
        # @torch.no_grad) so to_logmel stays differentiable — used as the
        # auxiliary log-mel loss in the trainer. STFT(center=False) with
        # hop=patch_size gives T_mel == T_acoustic, so v_mask carries over.
        wav = rearrange(features, "b p t -> b (t p)")
        return self._monitor_mel(wav)

    def eos_frames(
        self,
        n: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(self.acoustic_dim, n, device=device, dtype=dtype)

    def is_eos(self, frame: torch.Tensor) -> torch.BoolTensor:
        return frame.pow(2).mean(dim=-1).sqrt() < self.eos_threshold

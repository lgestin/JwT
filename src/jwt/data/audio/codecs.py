from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from jwt.data.audio.stft import MelSpectrogram

# Patch sizes for the RawAudio codec variants. Each variant is its own codec
# identity: str(variant).lower() names the acoustic_{name} arrow column, so the
# patch size baked into the precomputed dataset and the runtime codec can't drift.
_RAWAUDIO_PATCH = {
    "RawAudio32": 32,
    "RawAudio64": 64,
    "RawAudio128": 128,
    "RawAudio256": 256,
    "RawAudio512": 512,
}


class Codecs(StrEnum):
    BIGVGAN = "BigVGAN"
    RAWAUDIO_32 = "RawAudio32"
    RAWAUDIO_64 = "RawAudio64"
    RAWAUDIO_128 = "RawAudio128"
    RAWAUDIO_256 = "RawAudio256"
    RAWAUDIO_512 = "RawAudio512"

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

    # The sample rate encode/decode is locked to, or None if the codec works at
    # any rate. The data's sample rate is a property of the dataset, not the
    # codec — this only expresses a hard constraint (see check_sample_rate).
    required_sample_rate: int | None
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

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))


def check_sample_rate(codec: Codec, sample_rate: int) -> None:
    """Raise if `sample_rate` is incompatible with `codec`.

    A codec whose `required_sample_rate` is None accepts any rate; otherwise the
    data must match it exactly (e.g. BigVGAN's vocoder is locked to 24 kHz).
    """
    required = codec.required_sample_rate
    if required is not None and required != sample_rate:
        raise ValueError(
            f"{type(codec).__name__} requires {required} Hz audio, but the "
            f"data sample rate is {sample_rate} Hz"
        )


class BigVGANVersions(StrEnum):
    V2_24KHz_100MEL_256X = "nvidia/bigvgan_v2_24khz_100band_256x"


# Global log-mel stats fit on LJSpeech at 24 kHz with this BigVGAN's mel front-end.
_LJSPEECH_LOG_MEL_MEAN = -5.896610
_LJSPEECH_LOG_MEL_STD = 2.226763


class BigVGAN(nn.Module):
    # The bundled vocoder and its mel front-end are trained at 24 kHz.
    required_sample_rate: int | None = 24000
    mel_mean: torch.Tensor
    mel_std: torch.Tensor

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

        h: Any = load_hparams_from_json(config_file)
        self.decoder = _BigVGAN(h, use_cuda_kernel=False)
        # Published checkpoint has weight norm stripped; load_state_dict
        # succeeds directly.
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
            mel_scale="slaney",
        )
        assert int(h.sampling_rate) == self.required_sample_rate, (
            f"BigVGAN checkpoint sample rate {h.sampling_rate} != "
            f"required_sample_rate {self.required_sample_rate}"
        )
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
        return self.mel_spectrogram(waveform).clamp(min=1e-5).log()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mel_mean) / self.mel_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mel_std + self.mel_mean

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
        return frame.mean(dim=-1) < self.eos_threshold  # ty: ignore[invalid-return-type]


class RawAudioPatcher(nn.Module):
    """Codec that turns a 1D waveform into a sequence of non-overlapping patches.

    encode reshapes (B, S) (or (B, 1, S)) into (B, patch_size, n_patches).
    decode does the inverse. Both are deterministic — the model's input/output
    projection is responsible for learning a useful per-patch embedding.
    """

    # Patching raw samples is sample-rate agnostic.
    required_sample_rate: int | None = None
    wav_std: torch.Tensor

    def __init__(self, patch_size: int = 256):
        nn.Module.__init__(self)
        self.patch_size = patch_size
        self.acoustic_dim = patch_size
        self.hop_length = patch_size
        self.eos_threshold: float = 1e-4

        # In __init__: waveform is RMS-normalized to -24 dBFS upstream
        # (create_arrow_ljspeech.py target_loudness), so per-sample std ~ 10**(-24/20).
        wav_std = 10 ** (-24.0 / 20)  # ~0.0631
        self.register_buffer("wav_std", torch.tensor(wav_std, dtype=torch.float32))

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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Not @torch.no_grad: the auxiliary mel loss decodes predicted features
        # to a waveform and backprops through this rearrange.
        return rearrange(z, "b p t -> b (t p)")

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.wav_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.wav_std

    def eos_frames(
        self,
        n: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(self.acoustic_dim, n, device=device, dtype=dtype)

    def is_eos(self, frame: torch.Tensor) -> torch.BoolTensor:
        return frame.pow(2).mean(dim=-1).sqrt() < self.eos_threshold  # ty: ignore[invalid-return-type]

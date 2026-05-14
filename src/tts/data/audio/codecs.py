from enum import StrEnum
from typing import Protocol

import torch
import torch.nn as nn

from tts.data.audio.stft import MelSpectrogram


class Codecs(StrEnum):
    BIGVGAN = "BigVGAN"

    @property
    def codec(self):
        match self:
            case Codecs.BIGVGAN:
                return BigVGAN()


class Codec(Protocol):
    def encode(self, waveform: torch.Tensor): ...
    def decode(self, z: torch.Tensor): ...
    def reconstruct(self, waveform: torch.Tensor):
        return self.decode(self.encode(waveform))


class BigVGANVersions(StrEnum):
    V2_24KHz_100MEL_256X = "nvidia/bigvgan_v2_24khz_100band_256x"


class BigVGAN(Codec, nn.Module):
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

        f_max = h.fmax if h.fmax is not None else h.sampling_rate // 2
        self.mel_spectrogram = MelSpectrogram(
            n_fft=h.n_fft,
            hop_length=h.hop_size,
            n_mels=h.num_mels,
            sample_rate=h.sampling_rate,
            f_min=h.fmin,
            f_max=f_max,
            window="hann",
            center=True,
            log_eps=1e-5,
        )

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel_spectrogram(waveform)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

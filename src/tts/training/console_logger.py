from pathlib import Path

import torch
import torchaudio


class ConsoleLogger:
    """Minimal Logger: prints to stdout, writes audio as wav files."""

    def __init__(self, audio_dir: Path | str | None = None):
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        if self.audio_dir is not None:
            self.audio_dir.mkdir(parents=True, exist_ok=True)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        print(f"[step {step:>7}] {tag}: {float(value):.6f}")

    def log_audio(
        self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int
    ) -> None:
        if self.audio_dir is None:
            return
        wav = waveform.detach().cpu().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        path = self.audio_dir / f"{tag.replace('/', '_')}_step{step:07d}.wav"
        torchaudio.save(str(path), wav, sample_rate)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        pass

    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        items = " | ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
        print(f"[step {step:>7}] [{prefix}] {items}")

    def set_description(self, description: str) -> None:
        print(f"[ {description} ]")

    def update_progress(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

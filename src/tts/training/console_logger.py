from pathlib import Path

import torch
import torchaudio
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class _IterSpeedColumn(ProgressColumn):
    """Render the current iteration speed: 'X.XX it/s' or 'X.XX s/it'."""

    def render(self, task: Task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("?.?? it/s", style="progress.data.speed")
        if speed >= 1.0:
            return Text(f"{speed:.2f} it/s", style="progress.data.speed")
        return Text(f"{1.0 / speed:.2f} s/it", style="progress.data.speed")


class ConsoleLogger:
    """Logger with a rich progress bar; writes audio as wav files."""

    def __init__(
        self,
        total: int | None = None,
        audio_dir: Path | str | None = None,
    ):
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        if self.audio_dir is not None:
            self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.console = Console()
        self.progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            _IterSpeedColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self.progress.start()
        self.task = self.progress.add_task("training", total=total)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.console.print(
            f"[step {step:>7}] {tag}: {float(value):.6f}", markup=False, highlight=False
        )

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
        items = " ".join(f"{k}={float(v):.3f}" for k, v in metrics.items())
        if prefix == "train":
            # High-frequency train metrics ride on the progress bar description
            # so they update in place instead of scrolling the console.
            self.progress.update(self.task, description=f"train {items}")
        else:
            self.console.print(
                f"[step {step:>7}] [{prefix}] {items}",
                markup=False,
                highlight=False,
            )

    def log_diagnostics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        # High-cardinality diagnostics never touch the console / progress bar.
        pass

    def log_histogram(
        self, tag: str, bin_edges: list[float], bin_values: list[float], step: int
    ) -> None:
        # Histograms are a TensorBoard-only artifact.
        pass

    def log_config(self, config: object, step: int = 0) -> None:
        # The resolved config is also dumped to output_dir/config.yaml — no
        # need to flood the console with it.
        pass

    def set_description(self, description: str) -> None:
        self.progress.update(self.task, description=description)

    def update_progress(self, n: int = 1) -> None:
        self.progress.update(self.task, advance=n)

    def set_progress(self, completed: int) -> None:
        # `reset` clears the speed-sample history too, so iter/s and ETA are
        # estimated fresh from the resumed step rather than skewed by a jump.
        self.progress.reset(self.task, completed=completed)

    def close(self) -> None:
        self.progress.stop()

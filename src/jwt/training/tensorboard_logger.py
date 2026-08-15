from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from jwt.training.loggers import SampleRecord, flatten_config, metric_tag
from jwt.training.plots import render_curve


class TensorBoardLogger:
    def __init__(self, log_dir: Path | str):
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def log_audio(
        self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int
    ) -> None:
        wav = waveform.detach().cpu().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        self.writer.add_audio(tag, wav, step, sample_rate=sample_rate)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        self.writer.add_image(tag, image, step)

    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(metric_tag(prefix, k), float(v), step)

    def log_diagnostics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        for k, v in metrics.items():
            fv = float(v)
            if fv == fv:  # skip NaN — e.g. an empty timestep bin
                self.writer.add_scalar(f"{prefix}/{k}", fv, step)

    def log_curve(
        self,
        tag: str,
        x: list[float],
        y: list[float],
        step: int,
        xlabel: str = "t",
        history: bool = True,
    ) -> None:
        # Rasterized into the Images tab rather than sent through
        # `add_histogram_raw`. The histogram widget is built for count
        # distributions: it resamples the stored bins onto its own display
        # buckets and *sums* whatever lands in each one, so a mean-per-bin
        # curve on a non-uniformly populated grid grows step artefacts wherever
        # a display bucket happens to swallow more bins than its neighbour. A
        # plot stores exactly what it displays, and gaps stay gaps.
        image = render_curve(x, y, xlabel=xlabel, ylabel=tag.rsplit("/", 1)[-1])
        self.writer.add_image(tag, image, step)

    _HPARAM_METRICS = (
        "loss/valid",
        "loss/valid_fm",
        "quality_tf/valid_si_snr",
        "quality_tf/valid_snr",
        "quality_tf/valid_logstft_l1",
        "quality_tf/valid_mel_cepstral_distortion",
        "quality_tf/valid_si_sdr",
        "quality_tf/valid_sdr",
        "quality_tf/valid_pesq",
        "quality_tf/valid_stoi",
        "quality_tf/valid_nisqa_mos",
        "quality_tf/valid_utmos",
    )

    def log_samples(
        self,
        section: str,
        records: list[SampleRecord],
        step: int,
        join: str | None = None,
    ) -> None:
        # No table widget — unpack each record into individual entries.
        for r in records:
            for name, a in r.audio.items():
                self.log_audio(
                    f"{section}/{r.index}_{name}", a.waveform, step, a.sample_rate
                )
            for name, img in r.images.items():
                self.log_image(f"{section}/{r.index}_{name}", img, step)
            for name, v in r.metrics.items():
                self.log_scalar(f"{section}/{r.index}_{name}", v, step)

    def log_config(self, config: object, step: int = 0) -> None:
        flat = flatten_config(config)
        exp, ssi, sei = hparams(flat, dict.fromkeys(self._HPARAM_METRICS, 0.0))
        file_writer = self.writer._get_file_writer()  # re-opens after close()
        for summary in (exp, ssi, sei):
            file_writer.add_summary(summary, global_step=step)

    def set_description(self, description: str) -> None:
        pass

    def update_progress(self, n: int = 1) -> None:
        pass

    def set_progress(self, completed: int) -> None:
        pass

    def close(self) -> None:
        self.writer.close()

"""Weights & Biases logging backend.

Every ``wandb.log`` call carries a ``trainer/step`` field instead of wandb's
own monotonic step (``define_metric("*", step_metric="trainer/step")``), so
out-of-order logging — reference audio at step 0, resumed runs — just works
and every panel plots against the trainer step.
"""

import contextlib
import itertools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb

from jwt.training.loggers import SampleRecord, flatten_config, metric_tag

STEP_METRIC = "trainer/step"

# Best-so-far summaries for the runs table (min for losses, max for quality).
_SUMMARY_MIN = (
    "loss/valid",
    "loss/valid_fm",
    "quality_tf/valid_logstft_l1",
    "quality_tf/valid_mel_cepstral_distortion",
)
_SUMMARY_MAX = (
    "quality_tf/valid_si_snr",
    "quality_tf/valid_snr",
    "quality_tf/valid_si_sdr",
    "quality_tf/valid_sdr",
    "quality_tf/valid_pesq",
    "quality_tf/valid_stoi",
    "quality_tf/valid_nisqa_mos",
    "quality_tf/valid_utmos",
    "quality_gen/nisqa_mos",
    "quality_gen/utmos",
)


def _audio_array(waveform: torch.Tensor) -> np.ndarray:
    """To the (T,) or (T, C) float array `wandb.Audio` expects."""
    wav = waveform.detach().cpu().float().squeeze()
    if wav.ndim == 2:
        wav = wav.transpose(0, 1)
    return wav.numpy()


def _image_array(image: torch.Tensor) -> np.ndarray:
    """(H, W) / (C, H, W) float in [0, 1] to the uint8 HWC array `wandb.Image`
    renders without rescaling."""
    img = image.detach().cpu().float()
    if img.ndim == 3:
        img = img.permute(1, 2, 0).squeeze(-1)
    return (img * 255).clamp(0, 255).byte().numpy()


def _curve_rows(x: list[float], y: list[float]) -> list[list[float]]:
    """(x, y) pairs with NaN points dropped — gaps stay gaps."""
    return [[xi, yi] for xi, yi in zip(x, y, strict=True) if yi == yi]


def _bin_edges(x: list[float]) -> list[float]:
    """Bin edges bracketing the monotone grid of centers `x`: interior edges at
    midpoints, end edges extrapolated half a bin out."""
    mids = [(a + b) / 2 for a, b in itertools.pairwise(x)]
    first = x[0] - (x[1] - x[0]) / 2
    last = x[-1] + (x[-1] - x[-2]) / 2
    return [first, *mids, last]


def _record_cells(record: SampleRecord) -> dict[str, Any]:
    """A record's cells converted to wandb media/values, insertion-ordered."""
    cells: dict[str, Any] = {}
    for k, a in record.audio.items():
        cells[k] = wandb.Audio(_audio_array(a.waveform), sample_rate=a.sample_rate)
    for k, img in record.images.items():
        cells[k] = wandb.Image(_image_array(img))
    cells.update(record.metrics)
    return cells


class WandbLogger:
    def __init__(
        self,
        log_dir: Path | str,
        run_name: str | None = None,
        project: str = "JwT",
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
    ):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            dir=str(log_dir),
            mode=mode,  # ty: ignore[invalid-argument-type]
        )
        self.run.define_metric(STEP_METRIC)
        self.run.define_metric("*", step_metric=STEP_METRIC)
        for k in _SUMMARY_MIN:
            self.run.define_metric(k, step_metric=STEP_METRIC, summary="min")
        for k in _SUMMARY_MAX:
            self.run.define_metric(k, step_metric=STEP_METRIC, summary="max")
        self._alerted: set[str] = set()
        # Latest raw records per section, for `join`. Converted fresh on every
        # log — reused bound media objects don't render in the table UI, and
        # wandb's content-addressed storage dedupes identical bytes anyway.
        self._section_records: dict[str, dict[int, SampleRecord]] = {}

    def _log(self, data: Mapping[str, object], step: int) -> None:
        self.run.log({**data, STEP_METRIC: step})

    def _alert_if_diverged(self, tag: str, value: float, step: int) -> None:
        if "loss" not in tag or math.isfinite(value) or tag in self._alerted:
            return
        self._alerted.add(tag)
        # Alerts are unsupported offline — never kill a run over one.
        with contextlib.suppress(Exception):
            self.run.alert(
                title=f"{tag} diverged",
                text=f"{tag}={value} at step {step} ({self.run.name})",
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        value = float(value)
        self._alert_if_diverged(tag, value, step)
        self._log({tag: value}, step)

    def log_audio(
        self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int
    ) -> None:
        audio = wandb.Audio(_audio_array(waveform), sample_rate=sample_rate)
        self._log({tag: audio}, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        self._log({tag: wandb.Image(_image_array(image))}, step)

    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        data = {metric_tag(prefix, k): float(v) for k, v in metrics.items()}
        for tag, v in data.items():
            self._alert_if_diverged(tag, v, step)
        self._log(data, step)

    def log_diagnostics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        data = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
        finite = {k: v for k, v in data.items() if v == v}  # skip NaN bins
        if finite:
            self._log(finite, step)

    def log_curve(
        self,
        tag: str,
        x: list[float],
        y: list[float],
        step: int,
        xlabel: str = "t",
        history: bool = True,
    ) -> None:
        ylabel = tag.rsplit("/", 1)[-1]
        # Latest curve only — step history lives in the `_hist` heatmap.
        rows: list[Any] = _curve_rows(x, y)
        table = wandb.Table(columns=[xlabel, ylabel], data=rows)
        self._log({tag: wandb.plot.line(table, xlabel, ylabel, title=tag)}, step)
        if history and len(x) >= 2:
            # Heatmap history: successive histograms render as a value-over-
            # steps heatmap. "Counts" are the per-bin curve values (NaN -> 0).
            vals = [yi if yi == yi else 0.0 for yi in y]
            hist = wandb.Histogram(np_histogram=(vals, _bin_edges(x)))  # ty: ignore[invalid-argument-type]
            self._log({f"{tag}_hist": hist}, step)

    def log_samples(
        self,
        section: str,
        records: list[SampleRecord],
        step: int,
        join: str | None = None,
    ) -> None:
        self._section_records[section] = {r.index: r for r in records}
        own = {r.index: _record_cells(r) for r in records}
        joined = (
            {
                i: _record_cells(r)
                for i, r in self._section_records.get(join, {}).items()
            }
            if join is not None
            else {}
        )
        columns: dict[str, None] = {}
        for cells in (*joined.values(), *own.values()):
            for k in cells:
                columns.setdefault(k, None)
        data: list[Any] = [
            [
                r.index,
                *(
                    ({**joined.get(r.index, {}), **own[r.index]}).get(k)
                    for k in columns
                ),
            ]
            for r in records
        ]
        cols: list[Any] = ["idx", *columns]
        self._log({f"tables/{section}": wandb.Table(columns=cols, data=data)}, step)

    def log_config(self, config: object, step: int = 0) -> None:
        self.run.config.update(flatten_config(config), allow_val_change=True)

    def set_description(self, description: str) -> None:
        pass

    def update_progress(self, n: int = 1) -> None:
        pass

    def set_progress(self, completed: int) -> None:
        pass

    def close(self) -> None:
        self.run.finish()

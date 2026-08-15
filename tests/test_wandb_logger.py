from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from jwt.data.audio.audio import Audio
from jwt.training.loggers import MultiLogger, SampleRecord, flatten_config
from jwt.training.wandb_logger import (
    WandbLogger,
    _audio_array,
    _curve_rows,
    _image_array,
)


def test_audio_array_returns_1d_or_time_by_channel() -> None:
    assert _audio_array(torch.zeros(100)).shape == (100,)
    assert _audio_array(torch.zeros(1, 100)).shape == (100,)
    assert _audio_array(torch.zeros(2, 100)).shape == (100, 2)


def test_image_array_converts_chw_float_to_hwc_uint8() -> None:
    gray = _image_array(torch.full((1, 4, 6), 0.5))
    assert gray.shape == (4, 6)
    assert gray.dtype.name == "uint8"
    assert int(gray[0, 0]) == 127

    rgb = _image_array(torch.rand(3, 4, 6))
    assert rgb.shape == (4, 6, 3)
    assert rgb.dtype.name == "uint8"


def test_mel_image_is_colorized_and_flipped() -> None:
    from jwt.training.loggers import mel_image

    mel = torch.arange(6.0).reshape(2, 3)  # low freqs in row 0
    img = mel_image(mel)
    assert img.shape == (3, 2, 3)  # RGB via the magma colormap
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    # Not grayscale: channels differ somewhere.
    assert not torch.allclose(img[0], img[1])
    # Low frequencies (originally row 0, lowest values) end up at the bottom:
    # the bottom-left pixel is the darkest, top-right the brightest.
    luma = img.mean(0)
    assert luma[1, 0] == luma.min()
    assert luma[0, 2] == luma.max()


def test_colorize_maps_grayscale_to_rgb() -> None:
    from jwt.training.loggers import colorize

    rgb = colorize(torch.linspace(0, 1, 12).reshape(1, 3, 4), cmap="viridis")
    assert rgb.shape == (3, 3, 4)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0
    assert not torch.allclose(rgb[0], rgb[2])


def test_bin_edges_bracket_centers() -> None:
    from jwt.training.wandb_logger import _bin_edges

    # Uniform grid: centers 0.25/0.75 -> edges 0/0.5/1.
    assert _bin_edges([0.25, 0.75]) == pytest.approx([0.0, 0.5, 1.0])
    # Non-uniform grid: interior edges at midpoints, ends extrapolated.
    assert _bin_edges([0.1, 0.2, 0.6]) == pytest.approx([0.05, 0.15, 0.4, 0.8])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_log_curve_history_flag_controls_hist(tmp_path: Path) -> None:
    logger = WandbLogger(log_dir=tmp_path, run_name="hist-test", mode="offline")
    try:
        logged: dict[str, object] = {}
        logger._log = lambda data, step: logged.update(data)  # ty: ignore[invalid-assignment]
        logger.log_curve("by_t/train_fm_loss", [0.25, 0.75], [1.0, 2.0], step=1)
        assert "by_t/train_fm_loss_hist" in logged
        logger.log_curve(
            "schedule/timesteps", [0.0, 1.0], [0.0, 1.0], step=1, history=False
        )
        assert "schedule/timesteps" in logged
        assert "schedule/timesteps_hist" not in logged
    finally:
        logger.close()


def test_curve_rows_drop_nan_points() -> None:
    rows = _curve_rows([0.1, 0.5, 0.9], [1.0, float("nan"), 3.0])
    assert rows == [[0.1, 1.0], [0.9, 3.0]]


@dataclass
class _Inner:
    dim: int = 8


@dataclass
class _Cfg:
    lr: float = 1e-3
    inner: _Inner = field(default_factory=_Inner)


def test_flatten_config_joins_nested_keys_with_dots() -> None:
    assert flatten_config(_Cfg()) == {"lr": 1e-3, "inner.dim": 8}


def test_metric_tag_routes_by_question() -> None:
    from jwt.training.loggers import metric_tag

    # Losses land together, split in the name.
    assert metric_tag("train", "loss") == "loss/train"
    assert metric_tag("valid", "loss") == "loss/valid"
    assert metric_tag("train", "fm_loss") == "loss/train_fm"
    assert metric_tag("valid", "logmel_l1") == "loss/valid_logmel_l1"
    # Optimization health.
    assert metric_tag("train", "grad_norm") == "optim/grad_norm"
    # Teacher-forced quality for train/valid splits.
    assert metric_tag("train", "si_snr") == "quality_tf/train_si_snr"
    assert metric_tag("valid", "pesq") == "quality_tf/valid_pesq"
    assert metric_tag("valid", "mel_cepstral_distortion") == (
        "quality_tf/valid_mel_cepstral_distortion"
    )
    # Free-run generation quality and stopping health.
    assert metric_tag("sampled", "utmos") == "quality_gen/utmos"
    assert metric_tag("sampled", "eos_rate") == "quality_gen/eos_rate"


def _records() -> list[SampleRecord]:
    return [
        SampleRecord(
            index=0,
            audio={"pred": Audio(torch.zeros(800), 8000)},
            images={"mel": torch.rand(1, 8, 12)},
            metrics={"utmos": 3.5},
        ),
        # Missing image and metric — backends must tolerate sparse records.
        SampleRecord(index=1, audio={"pred": Audio(torch.zeros(800), 8000)}),
    ]


class _Recorder:
    def __init__(self) -> None:
        self.sections: list[str] = []

    def log_samples(
        self,
        section: str,
        records: list[SampleRecord],
        step: int,
        join: str | None = None,
    ) -> None:
        self.sections.append(section)

    def __getattr__(self, name: str):
        return lambda *a, **k: None


def test_multilogger_fans_out_log_samples() -> None:
    a, b = _Recorder(), _Recorder()
    MultiLogger(a, b).log_samples("samples", _records(), step=1)  # ty: ignore[invalid-argument-type]
    assert a.sections == ["samples"]
    assert b.sections == ["samples"]


def test_tensorboard_logger_unpacks_log_samples(tmp_path: Path) -> None:
    from jwt.training.tensorboard_logger import TensorBoardLogger

    logger = TensorBoardLogger(log_dir=tmp_path)
    logger.log_samples("samples", _records(), step=1)
    logger.close()
    assert any(tmp_path.iterdir()), "no event file written"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_log_samples_join_merges_columns_with_fresh_media(tmp_path: Path) -> None:
    """Joined tables get the reference columns, converted to fresh media
    objects on every log — reused bound objects don't render in the table UI,
    and content-addressed storage dedupes the identical bytes anyway."""
    import wandb

    logger = WandbLogger(log_dir=tmp_path, run_name="join-test", mode="offline")
    try:
        tables: list[object] = []
        logger._log = lambda data, step: tables.extend(  # ty: ignore[invalid-assignment]
            v for k, v in data.items() if k.startswith("tables/samples")
        )
        refs = [SampleRecord(index=0, audio={"clean": Audio(torch.zeros(800), 8000)})]
        logger.log_samples("references", refs, 0)
        for step in (10, 20):
            gen = [
                SampleRecord(
                    index=0,
                    audio={"audio": Audio(torch.full((800,), 0.01 * step), 8000)},
                    metrics={"utmos": 3.0},
                )
            ]
            logger.log_samples("samples", gen, step, join="references")
        assert [t.columns for t in tables] == [["idx", "clean", "audio", "utmos"]] * 2  # ty: ignore[unresolved-attribute]
        first, second = (t.data[0][1] for t in tables)  # ty: ignore[unresolved-attribute]
        assert isinstance(first, wandb.Audio) and isinstance(second, wandb.Audio)
        assert first is not second  # fresh conversion per log
    finally:
        logger.close()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_wandb_logger_offline_end_to_end(tmp_path: Path) -> None:
    """Exercise every protocol method offline: no network, nothing raises."""
    logger = WandbLogger(log_dir=tmp_path, run_name="test-run", mode="offline")
    try:
        logger.log_scalar("train/loss", 1.0, step=5)
        logger.log_metrics({"loss": 0.5, "si_snr": 3.0}, step=5, prefix="valid")
        logger.log_metrics({"loss": float("nan")}, step=6, prefix="train")  # alert
        logger.log_diagnostics({"ok": 1.0, "empty_bin": float("nan")}, step=6)
        logger.log_audio("samples/0", torch.zeros(1, 800), step=6, sample_rate=8000)
        # Out-of-order step: reference audio logged at step 0 mid-run.
        logger.log_audio(
            "references/0_clean", torch.zeros(800), step=0, sample_rate=8000
        )
        logger.log_image("references/0_clean_mel", torch.rand(1, 8, 12), step=6)
        logger.log_curve(
            "valid/fm_loss_by_t", [0.25, 0.75], [1.0, float("nan")], step=6
        )
        logger.log_config(_Cfg())
        logger.log_samples("valid_audio", _records(), step=6)
        logger.set_description("desc")
        logger.update_progress()
        logger.set_progress(6)
    finally:
        logger.close()
    assert any((tmp_path / "wandb").glob("offline-run-*")), "no offline run written"

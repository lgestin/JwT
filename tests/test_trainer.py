import math

import pytest
import torch

from jwt.data.audio.codecs import RawAudioPatcher
from jwt.data.audio.stft import MelSpectrogram
from jwt.model.neural_speaker import MaskedTensor, TrainingStepOutput
from jwt.training.trainer import TrainerConfig, TTSRollingFlowMatchingTrainer


class _RecordingLogger:
    """Captures `log_curve` calls so tests can assert on emitted data."""

    def __init__(self) -> None:
        self.curves: dict[str, list[float]] = {}
        self.grids: dict[str, list[float]] = {}

    def log_curve(self, tag, x, y, step, xlabel="t") -> None:
        assert len(x) == len(y)
        self.curves[tag] = y
        self.grids[tag] = x


def test_diagnostics_emit_unreweighted_x1_error_curve() -> None:
    """`x1_err_by_t` reports |x_1 - x_pred| with no reweighting, alongside the
    parametrization's `fm_loss_by_t` — a t-bin can show a high FM loss yet a
    small genuine prediction error (the JWT 1/(1-t) confound)."""
    # Bypass the heavyweight constructor: these diagnostics depend only on
    # `config` and `logger`.
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu", n_loss_bins=2)
    logger = _RecordingLogger()
    trainer.logger = logger  # type: ignore[assignment]

    # Bin 0 (t=0.25): big prediction error, small reweighted FM loss.
    # Bin 1 (t=0.75): tiny prediction error, big reweighted FM loss.
    out = TrainingStepOutput(
        loss=torch.tensor(0.5),
        x_pred=torch.tensor([[[2.0], [5.05]]]),  # (B=1, T=2, D=1)
        v_mask=torch.tensor([[True, True]]),
        t=torch.tensor([[0.25, 0.75]]),
        per_pos_loss=torch.tensor([[0.1, 0.9]]),
    )
    text = MaskedTensor(
        values=torch.zeros(1, 1, 3), mask=torch.ones(1, 3, dtype=torch.bool)
    )
    acoustic = MaskedTensor(
        values=torch.tensor([[[5.0, 5.0]]]),  # (B=1, D=1, T=2)
        mask=torch.ones(1, 2, dtype=torch.bool),
    )

    _, bins = trainer._step_diagnostics(out, text, acoustic)
    trainer._emit_loss_curves(bins, step=0, prefix="train")

    assert logger.curves["train/fm_loss_by_t"] == pytest.approx([0.1, 0.9], abs=1e-4)
    # |2 - 5| = 3.0 in bin 0; |5.05 - 5| = 0.05 in bin 1 — the reverse ranking.
    assert logger.curves["train/x1_err_by_t"] == pytest.approx([3.0, 0.05], abs=1e-4)


def _diag_inputs() -> tuple[TrainingStepOutput, MaskedTensor, MaskedTensor]:
    """Shared `_step_diagnostics` inputs: 2 frames, t-bins 0 (t=0.25) and 1."""
    out = TrainingStepOutput(
        loss=torch.tensor(0.5),
        x_pred=torch.tensor([[[2.0], [5.05]]]),  # (B=1, T=2, D=1)
        v_mask=torch.tensor([[True, True]]),
        t=torch.tensor([[0.25, 0.75]]),
        per_pos_loss=torch.tensor([[0.1, 0.9]]),
    )
    text = MaskedTensor(
        values=torch.zeros(1, 1, 3), mask=torch.ones(1, 3, dtype=torch.bool)
    )
    acoustic = MaskedTensor(
        values=torch.tensor([[[5.0, 5.0]]]),  # (B=1, D=1, T=2)
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    return out, text, acoustic


def test_loss_curves_are_plotted_on_the_bin_centre_grid() -> None:
    """The x grid carries one point per bin, at the bin's centre — not the
    n+1 outer edges a histogram summary wants. Sending a mean-per-bin curve
    through a histogram let TensorBoard re-bucket and sum it, which invented
    step artefacts wherever a display bucket swallowed more bins than its
    neighbour."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu", n_loss_bins=4)
    logger = _RecordingLogger()
    trainer.logger = logger  # type: ignore[assignment]

    _, bins = trainer._step_diagnostics(*_diag_inputs())
    trainer._emit_loss_curves(bins, step=0, prefix="train")

    assert logger.grids["train/fm_loss_by_t"] == pytest.approx(
        [0.125, 0.375, 0.625, 0.875]
    )


def test_empty_t_bins_stay_nan_so_the_curve_gaps() -> None:
    """A bin no sample landed in has no mean, and zero-filling it would draw a
    dip to zero that the model never produced. It stays NaN; `render_curve`
    breaks the line there instead."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu", n_loss_bins=4)
    logger = _RecordingLogger()
    trainer.logger = logger  # type: ignore[assignment]

    # Samples land at t=0.25 and t=0.75 only — bins 0 and 2 stay empty.
    _, bins = trainer._step_diagnostics(*_diag_inputs())
    trainer._emit_loss_curves(bins, step=0, prefix="train")

    curve = logger.curves["train/fm_loss_by_t"]
    assert [math.isnan(v) for v in curve] == [True, False, True, False]
    assert curve[1] == pytest.approx(0.1, abs=1e-4)
    assert curve[3] == pytest.approx(0.9, abs=1e-4)


_SPECTRAL_KEYS = {"logstft_l1", "mel_cepstral_distortion"}
_WAVEFORM_KEYS = {"si_snr", "snr", "mag_snr", "phase_snr_gap"}


def _metrics_trainer(hop: int = 256) -> TTSRollingFlowMatchingTrainer:
    """Trainer stub with just enough state for `_reconstruction_metrics`."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu")
    trainer.codec = RawAudioPatcher(patch_size=hop)
    trainer.model = torch.nn.Linear(1, 1)  # type: ignore[assignment] — only .training is read
    trainer.mel_spectrogram = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        sample_rate=24000,
        window="hann",
        center=False,
        mel_scale="slaney",
        n_mfcc=13,
    )
    return trainer


def test_reconstruction_metrics_cheap_pair_in_train_mode() -> None:
    """Train mode computes only the cheap time-domain pair."""
    trainer = _metrics_trainer()
    trainer.model.train()

    g = torch.Generator().manual_seed(0)
    target = torch.randn(2, 16 * 256, generator=g)
    pred = target + 0.01 * torch.randn(2, 16 * 256, generator=g)
    v_mask = torch.ones(2, 16, dtype=torch.bool)

    metrics: dict[str, torch.Tensor] = {}
    trainer._reconstruction_metrics(metrics, pred, target, v_mask)

    assert set(metrics) == _WAVEFORM_KEYS
    for v in metrics.values():
        assert v.shape == () and torch.isfinite(v)


def test_reconstruction_metrics_squeezes_channel_dim() -> None:
    """(B, 1, S) waveforms (vocoder-shaped decodes) are squeezed to (B, S)."""
    trainer = _metrics_trainer()
    trainer.model.eval()

    g = torch.Generator().manual_seed(0)
    target = torch.randn(2, 1, 16 * 256, generator=g)
    pred = target + 0.01 * torch.randn(2, 1, 16 * 256, generator=g)
    v_mask = torch.ones(2, 16, dtype=torch.bool)

    metrics: dict[str, torch.Tensor] = {}
    trainer._reconstruction_metrics(metrics, pred, target, v_mask)

    assert set(metrics) == _WAVEFORM_KEYS | _SPECTRAL_KEYS
    for v in metrics.values():
        assert v.shape == () and torch.isfinite(v)


def test_reconstruction_metrics_adds_spectral_pair_in_eval_mode() -> None:
    """Eval mode additionally computes the spectral pair."""
    trainer = _metrics_trainer()
    trainer.model.eval()

    g = torch.Generator().manual_seed(0)
    target = torch.randn(2, 16 * 256, generator=g)
    pred = target + 0.01 * torch.randn(2, 16 * 256, generator=g)
    v_mask = torch.ones(2, 16, dtype=torch.bool)

    metrics: dict[str, torch.Tensor] = {}
    trainer._reconstruction_metrics(metrics, pred, target, v_mask)

    assert set(metrics) == _WAVEFORM_KEYS | _SPECTRAL_KEYS
    for v in metrics.values():
        assert v.shape == () and torch.isfinite(v)


def test_reconstruction_metrics_respect_the_mask() -> None:
    """Frames masked out of v_mask must not contribute: with the second half
    corrupted, masking to the clean half improves every metric."""
    trainer = _metrics_trainer()
    trainer.model.eval()

    g = torch.Generator().manual_seed(0)
    T = 16
    target = torch.randn(2, T * 256, generator=g)
    pred = target + 0.01 * torch.randn(2, T * 256, generator=g)
    pred[:, T * 128 :] = torch.randn(2, T * 128, generator=g)

    full = torch.ones(2, T, dtype=torch.bool)
    first_half = full.clone()
    first_half[:, T // 2 :] = False

    m_full: dict[str, torch.Tensor] = {}
    m_half: dict[str, torch.Tensor] = {}
    trainer._reconstruction_metrics(m_full, pred, target, full)
    trainer._reconstruction_metrics(m_half, pred, target, first_half)

    assert m_half["si_snr"] > m_full["si_snr"]
    assert m_half["snr"] > m_full["snr"]
    assert m_half["mag_snr"] > m_full["mag_snr"]
    for key in _SPECTRAL_KEYS:
        assert m_half[key] < m_full[key]

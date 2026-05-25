import pytest
import torch

from jwt.model.neural_speaker import MaskedTensor, TrainingStepOutput
from jwt.training.trainer import TrainerConfig, TTSRollingFlowMatchingTrainer


class _RecordingLogger:
    """Captures `log_histogram` calls so tests can assert on emitted data."""

    def __init__(self) -> None:
        self.histograms: dict[str, list[float]] = {}

    def log_histogram(self, tag, bin_edges, bin_values, step) -> None:
        self.histograms[tag] = bin_values


def test_diagnostics_emit_unreweighted_x1_error_histogram() -> None:
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
    text = MaskedTensor(values=torch.zeros(1, 1, 3), mask=torch.ones(1, 3, dtype=torch.bool))
    acoustic = MaskedTensor(
        values=torch.tensor([[[5.0, 5.0]]]),  # (B=1, D=1, T=2)
        mask=torch.ones(1, 2, dtype=torch.bool),
    )

    _, bins = trainer._step_diagnostics(out, text, acoustic)
    trainer._emit_loss_histogram(bins, step=0, prefix="train")

    assert logger.histograms["train/fm_loss_by_t"] == pytest.approx([0.1, 0.9], abs=1e-4)
    # |2 - 5| = 3.0 in bin 0; |5.05 - 5| = 0.05 in bin 1 — the reverse ranking.
    assert logger.histograms["train/x1_err_by_t"] == pytest.approx([3.0, 0.05], abs=1e-4)
    # No aux mel loss in play, so no perceptual histogram is emitted.
    assert "train/logmel_l1_by_t" not in logger.histograms


def _diag_inputs() -> tuple[TrainingStepOutput, MaskedTensor, MaskedTensor]:
    """Shared `_step_diagnostics` inputs: 2 frames, t-bins 0 (t=0.25) and 1."""
    out = TrainingStepOutput(
        loss=torch.tensor(0.5),
        x_pred=torch.tensor([[[2.0], [5.05]]]),  # (B=1, T=2, D=1)
        v_mask=torch.tensor([[True, True]]),
        t=torch.tensor([[0.25, 0.75]]),
        per_pos_loss=torch.tensor([[0.1, 0.9]]),
    )
    text = MaskedTensor(values=torch.zeros(1, 1, 3), mask=torch.ones(1, 3, dtype=torch.bool))
    acoustic = MaskedTensor(
        values=torch.tensor([[[5.0, 5.0]]]),  # (B=1, D=1, T=2)
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    return out, text, acoustic


def test_diagnostics_emit_logmel_l1_by_t_when_aux_loss_active() -> None:
    """When the aux mel loss is active, `_step_diagnostics` bins the per-frame
    log-mel L1 by timestep into a `logmel_l1_by_t` histogram — the perceptual
    (mel-space) companion to `x1_err_by_t`."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu", n_loss_bins=2)
    logger = _RecordingLogger()
    trainer.logger = logger  # type: ignore[assignment]

    out, text, acoustic = _diag_inputs()
    # Per-frame log-mel L1: 0.4 in bin 0 (t=0.25), 0.6 in bin 1 (t=0.75).
    logmel_diff = torch.tensor([[0.4, 0.6]])

    _, bins = trainer._step_diagnostics(out, text, acoustic, logmel_diff)
    trainer._emit_loss_histogram(bins, step=0, prefix="train")

    assert logger.histograms["train/logmel_l1_by_t"] == pytest.approx([0.4, 0.6], abs=1e-4)
    # The base histograms are still emitted alongside it.
    assert "train/fm_loss_by_t" in logger.histograms
    assert "train/x1_err_by_t" in logger.histograms

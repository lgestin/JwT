import math

import pytest
import torch

from jwt.data.audio.audio import Audio
from jwt.data.audio.codecs import RawAudioPatcher
from jwt.data.audio.stft import MelSpectrogram
from jwt.data.dataset import Batch
from jwt.model.neural_speaker import MaskedTensor, TrainingStepOutput
from jwt.training.trainer import (
    TrainerConfig,
    TrainerState,
    TTSRollingFlowMatchingTrainer,
)


class RecordingLogger:
    """Captures `log_curve` calls so tests can assert on emitted data."""

    def __init__(self) -> None:
        self.curves: dict[str, list[float]] = {}
        self.grids: dict[str, list[float]] = {}

    def log_curve(
        self, tag: str, x: list[float], y: list[float], step: int, xlabel: str = "t"
    ) -> None:
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
    logger = RecordingLogger()
    trainer.logger = logger  # ty: ignore[invalid-assignment]

    # Bin 0 (t=0.25): big prediction error, small reweighted FM loss.
    # Bin 1 (t=0.75): tiny prediction error, big reweighted FM loss.
    out = TrainingStepOutput(
        loss=torch.tensor(0.5),
        fm_loss=torch.tensor(0.5),
        eos_loss=torch.tensor(0.0),
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

    _, bins = trainer.step_diagnostics(out, text, acoustic)
    trainer.emit_loss_curves(bins, step=0, prefix="train")

    assert logger.curves["by_t/train_fm_loss"] == pytest.approx([0.1, 0.9], abs=1e-4)
    # |2 - 5| = 3.0 in bin 0; |5.05 - 5| = 0.05 in bin 1 — the reverse ranking.
    assert logger.curves["by_t/train_x1_err"] == pytest.approx([3.0, 0.05], abs=1e-4)


def diag_inputs() -> tuple[TrainingStepOutput, MaskedTensor, MaskedTensor]:
    """Shared `step_diagnostics` inputs: 2 frames, t-bins 0 (t=0.25) and 1."""
    out = TrainingStepOutput(
        loss=torch.tensor(0.5),
        fm_loss=torch.tensor(0.5),
        eos_loss=torch.tensor(0.0),
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
    logger = RecordingLogger()
    trainer.logger = logger  # ty: ignore[invalid-assignment]

    _, bins = trainer.step_diagnostics(*diag_inputs())
    trainer.emit_loss_curves(bins, step=0, prefix="train")

    assert logger.grids["by_t/train_fm_loss"] == pytest.approx(
        [0.125, 0.375, 0.625, 0.875]
    )


def test_empty_t_bins_stay_nan_so_the_curve_gaps() -> None:
    """A bin no sample landed in has no mean, and zero-filling it would draw a
    dip to zero that the model never produced. It stays NaN; `render_curve`
    breaks the line there instead."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu", n_loss_bins=4)
    logger = RecordingLogger()
    trainer.logger = logger  # ty: ignore[invalid-assignment]

    # Samples land at t=0.25 and t=0.75 only — bins 0 and 2 stay empty.
    _, bins = trainer.step_diagnostics(*diag_inputs())
    trainer.emit_loss_curves(bins, step=0, prefix="train")

    curve = logger.curves["by_t/train_fm_loss"]
    assert [math.isnan(v) for v in curve] == [True, False, True, False]
    assert curve[1] == pytest.approx(0.1, abs=1e-4)
    assert curve[3] == pytest.approx(0.9, abs=1e-4)


SPECTRAL_KEYS = {"logstft_l1", "mel_cepstral_distortion"}
WAVEFORM_KEYS = {"si_snr", "snr", "mag_snr", "phase_snr_gap"}


def metrics_trainer(hop: int = 256) -> TTSRollingFlowMatchingTrainer:
    """Trainer stub with just enough state for `reconstruction_metrics`."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu")
    trainer.codec = RawAudioPatcher(patch_size=hop)
    trainer.model = torch.nn.Linear(1, 1)  # ty: ignore[invalid-assignment] — only .training is read
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
    trainer = metrics_trainer()
    trainer.model.train()

    g = torch.Generator().manual_seed(0)
    target = torch.randn(2, 16 * 256, generator=g)
    pred = target + 0.01 * torch.randn(2, 16 * 256, generator=g)
    v_mask = torch.ones(2, 16, dtype=torch.bool)

    metrics: dict[str, torch.Tensor] = {}
    trainer.reconstruction_metrics(metrics, pred, target, v_mask)

    assert set(metrics) == WAVEFORM_KEYS
    for v in metrics.values():
        assert v.shape == () and torch.isfinite(v)


def test_reconstruction_metrics_squeezes_channel_dim() -> None:
    """(B, 1, S) waveforms (vocoder-shaped decodes) are squeezed to (B, S)."""
    trainer = metrics_trainer()
    trainer.model.eval()

    g = torch.Generator().manual_seed(0)
    target = torch.randn(2, 1, 16 * 256, generator=g)
    pred = target + 0.01 * torch.randn(2, 1, 16 * 256, generator=g)
    v_mask = torch.ones(2, 16, dtype=torch.bool)

    metrics: dict[str, torch.Tensor] = {}
    trainer.reconstruction_metrics(metrics, pred, target, v_mask)

    assert set(metrics) == WAVEFORM_KEYS | SPECTRAL_KEYS
    for v in metrics.values():
        assert v.shape == () and torch.isfinite(v)


def test_reconstruction_metrics_adds_spectral_pair_in_eval_mode() -> None:
    """Eval mode additionally computes the spectral pair."""
    trainer = metrics_trainer()
    trainer.model.eval()

    g = torch.Generator().manual_seed(0)
    target = torch.randn(2, 16 * 256, generator=g)
    pred = target + 0.01 * torch.randn(2, 16 * 256, generator=g)
    v_mask = torch.ones(2, 16, dtype=torch.bool)

    metrics: dict[str, torch.Tensor] = {}
    trainer.reconstruction_metrics(metrics, pred, target, v_mask)

    assert set(metrics) == WAVEFORM_KEYS | SPECTRAL_KEYS
    for v in metrics.values():
        assert v.shape == () and torch.isfinite(v)


def test_reconstruction_metrics_respect_the_mask() -> None:
    """Frames masked out of v_mask must not contribute: with the second half
    corrupted, masking to the clean half improves every metric."""
    trainer = metrics_trainer()
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
    trainer.reconstruction_metrics(m_full, pred, target, full)
    trainer.reconstruction_metrics(m_half, pred, target, first_half)

    assert m_half["si_snr"] > m_full["si_snr"]
    assert m_half["snr"] > m_full["snr"]
    assert m_half["mag_snr"] > m_full["mag_snr"]
    for key in SPECTRAL_KEYS:
        assert m_half[key] < m_full[key]


class MetricsLogger:
    def __init__(self) -> None:
        self.logged: list[tuple[str, dict[str, float]]] = []

    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        self.logged.append((prefix, metrics))


def test_valid_unseen_logs_mean_loss_under_its_own_prefix() -> None:
    """Held-out speakers get their own `valid_unseen` loss, averaged over batches."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu")
    trainer.state = TrainerState(step=5)
    logger = MetricsLogger()
    trainer.logger = logger  # ty: ignore[invalid-assignment]
    losses = iter([1.0, 3.0])
    trainer.training_step = lambda batch: (  # ty: ignore[invalid-assignment]
        {"loss": torch.tensor(next(losses))},
        {},
        {},
    )
    trainer.valid_unseen_dloader = [object(), object()]  # ty: ignore[invalid-assignment]
    trainer.log_valid_unseen()
    assert logger.logged == [("valid_unseen", {"loss": 2.0})]


class RecordingCheckpointManager:
    """Captures `save` kwargs and counts cleanups."""

    def __init__(self) -> None:
        self.saves: list[dict] = []
        self.cleanups = 0

    def save(self, **kwargs: object) -> None:
        self.saves.append(kwargs)

    def cleanup_old_checkpoints(self) -> None:
        self.cleanups += 1


def test_save_checkpoint_tags_the_last_validation_loss() -> None:
    """Checkpoints carry the latest validation loss, not the best-so-far."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.state = TrainerState(step=300, best_loss=0.3, last_val_loss=0.4)
    trainer.model = torch.nn.Linear(2, 2)  # ty: ignore[invalid-assignment]
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.scaler = None
    trainer.ema = None
    manager = RecordingCheckpointManager()
    trainer.checkpoint_manager = manager  # ty: ignore[invalid-assignment]

    trainer.save_checkpoint()

    (saved,) = manager.saves
    assert saved["step"] == 300
    assert saved["val_loss"] == pytest.approx(0.4)
    assert saved["best_loss"] == pytest.approx(0.3)
    assert manager.cleanups == 1


def prompt_batch(prompt_lens: list[int], hop: int = 4) -> Batch:
    """Two-sample batch with raw prompts collated to `(B, 1, hop, P)`."""
    B, P = len(prompt_lens), max(prompt_lens)
    mask = torch.arange(P).expand(B, P) < torch.tensor(prompt_lens)[:, None]
    return Batch(
        idxs=list(range(B)),
        audios=[Audio(torch.randn(1, 8 * hop), 16000) for _ in range(B)],
        acoustic=torch.randn(B, 1, hop, 8),  # ty: ignore[invalid-argument-type]
        acoustic_mask=torch.ones(B, 8, dtype=torch.bool),  # ty: ignore[invalid-argument-type]
        tokens=torch.zeros(B, 3, dtype=torch.long),  # ty: ignore[invalid-argument-type]
        tokens_mask=torch.ones(B, 3, dtype=torch.bool),  # ty: ignore[invalid-argument-type]
        audio_prompt=torch.randn(B, 1, hop, P) * mask[:, None, None, :],
        audio_prompt_mask=mask,  # ty: ignore[invalid-argument-type]
    )


def test_prepare_prompt_squeezes_and_normalizes() -> None:
    """The collated `(B, 1, D, P)` prompt becomes a normalized `(B, D, P)`
    MaskedTensor; `n` keeps the first samples; no prompt gives `None`."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.codec = RawAudioPatcher(patch_size=4)
    batch = prompt_batch([3, 1])
    prompt = trainer.prepare_prompt(batch)
    assert prompt is not None
    assert torch.equal(prompt.values, trainer.codec.normalize(batch.audio_prompt[:, 0]))
    assert torch.equal(prompt.mask, batch.audio_prompt_mask)
    head = trainer.prepare_prompt(batch, n=1)
    assert head is not None and head.values.shape == (1, 4, 3)
    batch.audio_prompt = batch.audio_prompt_mask = None
    assert trainer.prepare_prompt(batch) is None


class SamplesLogger:
    def __init__(self) -> None:
        self.records: dict[str, list] = {}

    def log_samples(
        self, section: str, records: list, step: int, join: str | None = None
    ) -> None:
        self.records[section] = records


def test_references_include_the_prompt_audio() -> None:
    """Each reference row carries its prompt's audio (its real frames only) and
    mel; a row whose prompt was dropped has neither."""
    trainer = TTSRollingFlowMatchingTrainer.__new__(TTSRollingFlowMatchingTrainer)
    trainer.config = TrainerConfig(device="cpu", n_smp=2)
    trainer.device = torch.device("cpu")
    trainer.codec = RawAudioPatcher(patch_size=256)
    trainer.sample_rate = 16000
    trainer.mel_spectrogram = MelSpectrogram(
        n_fft=1024, hop_length=256, n_mels=80, sample_rate=16000
    )
    logger = SamplesLogger()
    trainer.logger = logger  # ty: ignore[invalid-assignment]
    batch = prompt_batch([8, 0], hop=256)
    trainer.smp_dloader = [batch]  # ty: ignore[invalid-assignment]
    trainer.log_initial_samples()
    with_prompt, without = logger.records["references"]
    wav = with_prompt.audio["prompt"].waveform
    assert torch.equal(wav.reshape(-1), batch.audio_prompt[0, 0].T.reshape(-1))
    assert "prompt_mel" in with_prompt.images
    assert "prompt" not in without.audio and "prompt_mel" not in without.images

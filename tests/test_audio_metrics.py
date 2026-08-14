import pytest
import torch

from jwt.training.metrics.nisqa import NISQA
from jwt.training.metrics.pesq import PESQ
from jwt.training.metrics.snr import si_snr, snr
from jwt.training.metrics.stoi import STOI
from jwt.training.metrics.utmos import UTMOS

SEED = 42
SAMPLE_RATE = 24000
S = 16 * 256  # short signals for the time-domain pair
S_PERC = SAMPLE_RATE  # 1 s — PESQ/STOI need at least ~1/4 s of audio


def _wavs(batch: int = 2, n_samples: int = S, seed: int = SEED):
    """Seeded (pred, target) pair, pred = target + small noise. Shapes (B, S)."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(batch, n_samples, generator=g) * 0.063
    pred = target + 0.01 * torch.randn(batch, n_samples, generator=g)
    return pred, target


def _full_mask(batch: int = 2, n_samples: int = S) -> torch.Tensor:
    return torch.ones(batch, n_samples, dtype=torch.bool)


def test_si_snr_is_scale_invariant() -> None:
    pred, target = _wavs()
    mask = _full_mask()
    base = si_snr(pred, target, mask)
    scaled = si_snr(0.5 * pred, target, mask)
    assert torch.allclose(base, scaled, atol=1e-4)


def test_snr_is_scale_sensitive() -> None:
    pred, target = _wavs()
    mask = _full_mask()
    base = snr(pred, target, mask)
    scaled = snr(0.5 * pred, target, mask)
    assert not torch.allclose(base, scaled, atol=1.0)


def test_si_snr_identical_signals_is_large_and_finite() -> None:
    pred, target = _wavs()
    mask = _full_mask()
    out = si_snr(target, target.clone(), mask)
    assert torch.isfinite(out).all()
    assert (out > 40).all()


def test_time_metrics_ignore_samples_outside_mask() -> None:
    """Corrupting the masked-out half must not change the score."""
    pred, target = _wavs()
    mask = _full_mask()
    mask[:, S // 2 :] = False

    clean = si_snr(pred, target, mask), snr(pred, target, mask)
    pred_corrupt = pred.clone()
    pred_corrupt[:, S // 2 :] = 10.0
    corrupt = si_snr(pred_corrupt, target, mask), snr(pred_corrupt, target, mask)

    assert torch.allclose(clean[0], corrupt[0], atol=1e-4)
    assert torch.allclose(clean[1], corrupt[1], atol=1e-4)


def test_si_snr_and_snr_match_torchmetrics_on_unmasked_input() -> None:
    from torchmetrics.functional.audio import (
        scale_invariant_signal_distortion_ratio,
        signal_noise_ratio,
    )

    pred, target = _wavs()
    mask = _full_mask()
    assert torch.allclose(
        si_snr(pred, target, mask),
        scale_invariant_signal_distortion_ratio(pred, target, zero_mean=True),
        atol=1e-3,
    )
    assert torch.allclose(
        snr(pred, target, mask),
        signal_noise_ratio(pred, target, zero_mean=False),
        atol=1e-3,
    )


def test_masked_metrics_stay_on_device_and_detached() -> None:
    pred, target = _wavs()
    pred = pred.requires_grad_(True)
    mask = _full_mask()
    for out in (si_snr(pred, target, mask), snr(pred, target, mask)):
        assert out.device == pred.device
        assert not out.requires_grad


def test_pesq_scores_and_resamples_non_16khz_input() -> None:
    """24 kHz input exercises the internal resample; identical signals score
    near PESQ's 4.64 ceiling, noisy ones lower."""
    pred, target = _wavs(n_samples=S_PERC)
    out = PESQ().score(target, target.clone(), sample_rate=SAMPLE_RATE)
    assert set(out) == {"pesq"}
    assert out["pesq"].shape == (2,)
    assert (out["pesq"] > 4.0).all()

    noisy = PESQ().score(pred, target, sample_rate=SAMPLE_RATE)
    assert (noisy["pesq"] < out["pesq"]).all()


def test_stoi_scores_identical_signals_at_one() -> None:
    pred, target = _wavs(n_samples=S_PERC)
    out = STOI().score(target, target.clone(), sample_rate=SAMPLE_RATE)
    assert set(out) == {"stoi"}
    assert out["stoi"].shape == (2,)
    assert torch.allclose(out["stoi"].float(), torch.ones(2), atol=1e-3)


def test_nisqa_returns_all_five_dimensions() -> None:
    pred, _ = _wavs(n_samples=S_PERC)
    out = NISQA().score(pred, sample_rate=SAMPLE_RATE)
    assert set(out) == {f"nisqa_{dim}" for dim in NISQA.DIMS}
    for v in out.values():
        assert v.shape == (2,)
        assert torch.isfinite(v).all()


def test_utmos_scores_with_stubbed_hub(monkeypatch) -> None:
    """UTMOS wiring (resample + dict contract) without the torch.hub download."""

    class _StubMOS(torch.nn.Module):
        def forward(self, waveforms: torch.Tensor, sample_rate: int) -> torch.Tensor:
            assert sample_rate == 16_000
            return waveforms.abs().mean(-1)

    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: _StubMOS())
    pred, _ = _wavs(n_samples=S_PERC)
    out = UTMOS().score(pred, sample_rate=SAMPLE_RATE)
    assert set(out) == {"utmos"}
    assert out["utmos"].shape == (2,)


def test_metric_classes_reject_masks(monkeypatch) -> None:
    """Masked scoring is deliberately unsupported — a mask must raise, not be
    silently ignored."""
    pred, target = _wavs()
    mask = _full_mask()
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: torch.nn.Identity())
    for call in (
        lambda: PESQ().score(pred, target, mask, sample_rate=SAMPLE_RATE),
        lambda: STOI().score(pred, target, mask, sample_rate=SAMPLE_RATE),
        lambda: NISQA().score(pred, mask, sample_rate=SAMPLE_RATE),
        lambda: UTMOS().score(pred, mask, sample_rate=SAMPLE_RATE),
    ):
        with pytest.raises(NotImplementedError):
            call()

import pytest
import torch

from jwt.training.loss import masked_mean_reduction


def test_masked_mean_reduction_matches_manual_mean() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    out = masked_mean_reduction(x, mask)
    assert out.tolist() == pytest.approx([1.5, 6.0])


def test_masked_mean_reduction_excludes_masked_positions() -> None:
    x = torch.tensor([[1.0, 100.0]])
    mask = torch.tensor([[True, False]])
    assert masked_mean_reduction(x, mask).item() == pytest.approx(1.0)


def test_masked_mean_reduction_zero_mask_is_zero_without_nan() -> None:
    """An all-zero row must yield 0, not NaN — this feeds the training loss."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[False, False], [True, True]])
    out = masked_mean_reduction(x, mask)
    assert torch.isfinite(out).all()
    assert out.tolist() == pytest.approx([0.0, 3.5])

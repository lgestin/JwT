import torch

from tts.training.metrics import (
    binned_loss_stats,
    masked_mean_std,
    per_pos_l1_error,
)


def test_binned_loss_stats_clamps_edges() -> None:
    """t == 0.0 lands in the first bin; t == 1.0 folds into the last bin."""
    per_pos = torch.ones(1, 2)
    t = torch.tensor([[0.0, 1.0]])
    v_mask = torch.ones(1, 2, dtype=torch.bool)

    sums, counts = binned_loss_stats(per_pos, t, v_mask, n_bins=10)

    assert counts[0].item() == 1.0
    assert counts[9].item() == 1.0
    assert counts[1:9].sum().item() == 0.0


def test_binned_loss_stats_means() -> None:
    """Per-bin mean (sum/count) matches a hand-built input."""
    per_pos = torch.tensor([[2.0, 4.0, 9.0]])
    t = torch.tensor([[0.02, 0.05, 0.55]])  # bins 0, 0, 5
    v_mask = torch.ones(1, 3, dtype=torch.bool)

    sums, counts = binned_loss_stats(per_pos, t, v_mask, n_bins=10)
    means = sums / counts

    assert means[0].item() == 3.0  # mean(2, 4)
    assert means[5].item() == 9.0


def test_binned_loss_stats_respects_mask() -> None:
    """Masked-out positions contribute to neither the sum nor the count."""
    per_pos = torch.tensor([[2.0, 100.0]])
    t = torch.tensor([[0.05, 0.05]])
    v_mask = torch.tensor([[True, False]])

    sums, counts = binned_loss_stats(per_pos, t, v_mask, n_bins=10)

    assert counts[0].item() == 1.0
    assert sums[0].item() == 2.0


def test_binned_loss_stats_empty_bins_are_nan() -> None:
    """A fully-masked input yields zero counts, so bin means are NaN (0/0)."""
    per_pos = torch.ones(1, 2)
    t = torch.tensor([[0.05, 0.05]])
    v_mask = torch.zeros(1, 2, dtype=torch.bool)

    sums, counts = binned_loss_stats(per_pos, t, v_mask, n_bins=10)

    assert counts.sum().item() == 0.0
    assert torch.isnan(sums / counts).all()


def test_per_pos_l1_error_averages_over_feature_dim() -> None:
    """Returns (B, T) mean-absolute error, averaged over the feature dim."""
    pred = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
    target = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])

    err = per_pos_l1_error(pred, target)

    assert err.shape == (1, 2)
    assert err[0, 0].item() == 0.0  # exact match
    assert err[0, 1].item() == 5.0  # mean(|4|, |5|, |6|)


def test_per_pos_l1_error_detaches() -> None:
    """The diagnostic must not retain the autograd graph in the hot loop."""
    pred = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.zeros(1, 2, 3)

    err = per_pos_l1_error(pred, target)

    assert not err.requires_grad


def test_masked_mean_std() -> None:
    """Mean/std are computed over the selected frames only."""
    # (B=1, D=2, T=3); the third frame (value 99) is masked out.
    values = torch.tensor([[[1.0, 3.0, 99.0], [1.0, 3.0, 99.0]]])
    frame_mask = torch.tensor([[True, True, False]])

    mean, std = masked_mean_std(values, frame_mask)

    assert mean.item() == 2.0  # mean of [1, 3, 1, 3]
    assert torch.isclose(std, torch.tensor(1.0))  # population std of [1, 3, 1, 3]


def test_masked_mean_std_all_false_is_nan() -> None:
    values = torch.randn(1, 2, 3)
    frame_mask = torch.zeros(1, 3, dtype=torch.bool)

    mean, std = masked_mean_std(values, frame_mask)

    assert torch.isnan(mean)
    assert torch.isnan(std)

import pytest
import torch

from jwt.training.metrics.pesq import PESQ
from jwt.training.metrics.utils import (
    binned_loss_stats,
    masked_mean_std,
    per_pos_l1_error,
    sampled_generation_stats,
)


def test_binned_loss_stats_clamps_edges() -> None:
    """t == 0.0 lands in the first bin; t == 1.0 folds into the last bin."""
    per_pos = torch.ones(1, 2)
    t = torch.tensor([[0.0, 1.0]])
    v_mask = torch.ones(1, 2, dtype=torch.bool)

    _, counts = binned_loss_stats(per_pos, t, v_mask, n_bins=10)

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


def test_sampled_generation_stats_terminated_and_ratio() -> None:
    """eos_rate counts stops before max_len; len_ratio is over stopped rows
    only."""
    gen_lens = torch.tensor([100, 600, 300])  # middle row hit max_len
    ref_lens = torch.tensor([200, 300, 300])

    stats = sampled_generation_stats(gen_lens, ref_lens, max_len=600)

    assert stats["eos_rate"].item() == pytest.approx(2 / 3)
    # mean of 100/200 and 300/300 — the max_len row is excluded.
    assert stats["len_ratio"].item() == pytest.approx(0.75)


def test_sampled_generation_stats_nothing_stopped_omits_len_ratio() -> None:
    """With no terminated generation there is no len_ratio — the curve gaps
    rather than reporting a made-up value."""
    gen_lens = torch.tensor([600, 600])
    ref_lens = torch.tensor([300, 300])

    stats = sampled_generation_stats(gen_lens, ref_lens, max_len=600)

    assert stats["eos_rate"].item() == 0.0
    assert "len_ratio" not in stats


def test_pesq_marks_unscorable_rows_nan_and_counts_the_rest() -> None:
    """An unscorable row stays in place as NaN, so it keeps its index in the
    batch; `pesq_scored` reports how many actually scored."""
    torch.manual_seed(0)
    target = torch.randn(3, 16_000) * 0.1
    pred = target.clone()
    pred[0] = 0.0  # silence — no utterance for PESQ to score

    out = PESQ().score(pred, target, sample_rate=16_000)

    assert out["pesq"].shape == (3,)  # row-aligned with the batch
    assert torch.isnan(out["pesq"][0])
    assert torch.isfinite(out["pesq"][1:]).all()
    assert out["pesq_scored"].item() == 2.0
    assert torch.isfinite(out["pesq"].nanmean())  # what log_audio_metrics logs


@pytest.mark.parametrize(
    "pred, target",
    [
        (torch.zeros(2, 16_000), torch.zeros(2, 16_000)),  # -7, no utterances
        (torch.randn(2, 300) * 0.1, torch.randn(2, 300) * 0.1),  # -6, too short
    ],
    ids=["no_utterances", "too_short"],
)
def test_pesq_error_codes_become_nan_not_scores(
    pred: torch.Tensor, target: torch.Tensor
) -> None:
    """PESQ signals failure as a negative code as well as NaN; averaging a -7
    in would be worse than the crash this replaced."""
    out = PESQ().score(pred, target, sample_rate=16_000)

    assert torch.isnan(out["pesq"]).all()
    assert out["pesq_scored"].item() == 0.0
    assert torch.isnan(out["pesq"].nanmean())  # nothing scored, nothing to log


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pesq_accepts_cuda_inputs() -> None:
    torch.manual_seed(0)
    target = (torch.randn(2, 16_000) * 0.1).cuda()
    pred = target.clone()
    pred[0] = 0.0

    out = PESQ().score(pred, target, sample_rate=16_000)

    assert out["pesq"].shape == (2,)
    assert torch.isnan(out["pesq"][0])
    assert torch.isfinite(out["pesq"][1])
    assert out["pesq_scored"].item() == 1.0

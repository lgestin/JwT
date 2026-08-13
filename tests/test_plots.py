import torch

from jwt.training.plots import render_curve


def test_render_curve_returns_a_uint8_rgb_image() -> None:
    """`log_image` hands the tensor straight to TensorBoard, which passes uint8
    through unscaled — so the renderer owes it (3, H, W) uint8, not floats."""
    image = render_curve([0.0, 0.5, 1.0], [1.0, 0.5, 0.25], xlabel="t", ylabel="loss")

    assert image.dtype == torch.uint8
    assert image.ndim == 3
    assert image.shape[0] == 3
    assert image.shape[1] > 0 and image.shape[2] > 0
    # Not a blank canvas: something was actually drawn on it.
    assert image.min() < image.max()


def test_render_curve_draws_through_nan_gaps() -> None:
    """Empty t-bins arrive as NaN. Matplotlib must break the line there rather
    than raise or draw a dip, so a sparse grid renders honestly."""
    y = [1.0, float("nan"), 0.5, float("nan"), 0.25]
    image = render_curve([0.0, 0.25, 0.5, 0.75, 1.0], y, xlabel="t", ylabel="loss")

    assert image.dtype == torch.uint8
    assert image.min() < image.max()


def test_render_curve_handles_an_all_nan_series() -> None:
    """A logging window can close before any sample lands in a bin. The plot is
    empty, but rendering it must not blow up the training loop."""
    n = 4
    image = render_curve(
        [i / n for i in range(n)], [float("nan")] * n, xlabel="t", ylabel="loss"
    )

    assert image.dtype == torch.uint8
    assert image.shape[0] == 3

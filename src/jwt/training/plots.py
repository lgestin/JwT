"""Rasterized diagnostic plots for the image-based logging backends."""

from collections.abc import Sequence

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_FIGSIZE = (5.0, 3.0)
_DPI = 110


def render_curve(
    x: Sequence[float], y: Sequence[float], *, xlabel: str, ylabel: str
) -> torch.Tensor:
    """Rasterize a line plot of `y` against `x` as a uint8 `(3, H, W)` RGB tensor.

    Uses the Agg canvas directly rather than `pyplot`, so no global figure state
    is created and nothing has to be closed. NaN entries in `y` break the line
    instead of being drawn, so gaps in a sparsely populated grid read as missing
    rather than as zeros.
    """
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.plot(x, y, marker=".", markersize=3.0, linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    canvas.draw()
    # `buffer_rgba` is a view onto the canvas; copy before handing it to torch.
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1)

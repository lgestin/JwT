from enum import StrEnum
from typing import Protocol

import torch


class TimestepSchedule(Protocol):
    """Warp from uniform denoising progress to the flow-matching timestep."""

    def timestep(self, progress: torch.Tensor) -> torch.Tensor:
        """Map denoising progress in ``[0, 1]`` to a timestep in ``[0, 1]``.

        Must be monotonic non-decreasing and fix the endpoints: ``0 -> 0`` and
        ``1 -> 1``. Applied elementwise, shape-preserving.
        """
        ...

    def timesteps(self, n_steps: int) -> torch.Tensor:
        """The full timestep grid of an ``n_steps`` denoising trajectory.

        ``timestep`` applied to the uniform progress grid
        ``{0, 1/(n_steps-1), ..., 1}``; returns a ``(n_steps,)`` tensor. Every
        schedule shares this warp-on-the-uniform-grid construction, so it is
        derived here from ``timestep`` rather than reimplemented per schedule.
        """
        return self.timestep(torch.linspace(0.0, 1.0, n_steps))

    def dt(self, progress: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Per-position Euler step size taken from each ``progress``.

        Equals ``timestep(progress + h) - timestep(progress)`` with
        ``h = 1 / (n_steps - 1)`` — the timestep advance of one rolling step.
        Same shape as ``progress``.
        """
        ...
        return _euler_dt(self, progress, n_steps)


def _euler_dt(
    scheduler: TimestepSchedule, progress: torch.Tensor, n_steps: int
) -> torch.Tensor:
    """Generic Euler step size — ``t(progress + h) - t(progress)`` by finite
    difference, for schedules whose step size is not constant.

    ``progress + h`` is clamped to 1 so the final rolling step lands exactly on
    ``t = 1`` instead of reading the schedule past its domain.
    """
    h = 1.0 / (n_steps - 1)
    nxt = (progress + h).clamp(max=1.0)
    return scheduler.timestep(nxt) - scheduler.timestep(progress)


class LinearTimestepSchedule(TimestepSchedule):
    """Identity warp: ``t == progress``, i.e. uniform timestep spacing.

    This reproduces the schedule the rolling model originally hard-coded.
    """

    def timestep(self, progress: torch.Tensor) -> torch.Tensor:
        return progress


class LogNormTimestepSchedule(TimestepSchedule):
    """Logit-normal warp: ``t = sigmoid(mean + std * Phi_inv(progress))``.

    ``Phi_inv`` is the standard-normal quantile function, so the uniform
    progress grid is placed at the quantiles of a logit-normal(mean, std)
    distribution — the deterministic, rolling-grid analogue of drawing
    timesteps from logit-normal(mean, std) (the SD3 sampling distribution).

    ``mean < 0`` concentrates the schedule toward small ``t`` (the noisy,
    generative timesteps); ``std`` controls how peaked the concentration is.

    ``eps`` trims that fraction of probability mass from each tail before
    warping: the uniform progress grid is squeezed into ``[eps, 1 - eps]`` and
    the resulting timesteps are rescaled back onto ``[0, 1]``. The logit-normal
    quantile function has unbounded slope at progress 0 and 1, which makes the
    first and (especially) last Euler steps disproportionately large; trimming
    the tails bounds those steps without flattening the early-emphasis bump.
    ``eps = 0`` is the untrimmed logit-normal.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 0.0):
        assert std > 0, "std must be positive"
        assert 0.0 <= eps < 0.5, "eps must be in [0, 0.5)"
        self.mean = mean
        self.std = std
        self.eps = eps
        self._span = 1.0 - 2.0 * eps
        # Timesteps the trimmed window's endpoints warp to, before rescaling.
        # Built with the same expression timestep() uses, so the rescale is
        # exact at progress 0 and 1. eps == 0 gives _t_lo == 0, _t_hi == 1
        # (Phi_inv(0) == -inf, sigmoid(-inf) == 0): an identity rescale.
        bounds_pp = self.eps + self._span * torch.tensor([0.0, 1.0])
        bounds = torch.sigmoid(self.mean + self.std * torch.special.ndtri(bounds_pp))
        self._t_lo = bounds[0].item()
        self._t_hi = bounds[1].item()

    def timestep(self, progress: torch.Tensor) -> torch.Tensor:
        pp = self.eps + self._span * progress
        raw = torch.sigmoid(self.mean + self.std * torch.special.ndtri(pp))
        t = (raw - self._t_lo) / (self._t_hi - self._t_lo)
        # Pin the endpoints: across devices/dtypes the rescale is only
        # approximate at the bounds, and speak() freezes a finished frame only
        # when timestep(1) == 1 exactly. Make 0 -> 0 and 1 -> 1 exact.
        t = torch.where(progress <= 0.0, torch.zeros_like(t), t)
        return torch.where(progress >= 1.0, torch.ones_like(t), t)

    def dt(self, progress: torch.Tensor, n_steps: int) -> torch.Tensor:
        return _euler_dt(self, progress, n_steps)


class TimestepSchedules(StrEnum):
    """Available timestep schedules, selectable from config."""

    LINEAR = "linear"
    LOG_NORM = "log_norm"
    LOG_NORM_TRIMMED = "log_norm_trimmed"

    @property
    def schedule(self) -> TimestepSchedule:
        match self:
            case TimestepSchedules.LINEAR:
                return LinearTimestepSchedule()
            case TimestepSchedules.LOG_NORM:
                # JiT logit-normal (arxiv 2511.13720, "Back to Basics"):
                # P_mean=-0.8, P_std=0.8 — skewed toward small t (noisier,
                # harder-to-denoise timesteps).
                return LogNormTimestepSchedule(mean=-0.8, std=0.8)
            case TimestepSchedules.LOG_NORM_TRIMMED:
                # Same logit-normal warp, eps=0.025 trimmed from each tail so
                # the final Euler step is not a t≈0.66→1 cliff (~34% of the
                # trajectory at n=32). See the tail-trim design doc.
                return LogNormTimestepSchedule(mean=-0.8, std=0.8, eps=0.025)

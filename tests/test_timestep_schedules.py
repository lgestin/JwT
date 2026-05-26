import pytest
import torch

from jwt.model.neural_speaker import RollingFlowConfig, RollingFlowSpeaker
from jwt.training.timestep_schedules import (
    LinearTimestepSchedule,
    LogNormTimestepSchedule,
    TimestepSchedules,
)

# --- LinearTimestepSchedule -------------------------------------------------


def test_linear_timestep_is_identity() -> None:
    sched = LinearTimestepSchedule()
    progress = torch.linspace(0.0, 1.0, 11)
    torch.testing.assert_close(sched.timestep(progress), progress)


def test_linear_timestep_fixes_endpoints() -> None:
    sched = LinearTimestepSchedule()
    endpoints = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(sched.timestep(endpoints), endpoints)


def test_linear_dt_is_constant_within_window() -> None:
    """Identity warp: every in-window step (progress < 1) advances t by exactly h.

    Linear now inherits the generic ``_euler_dt`` default, which tapers to 0 at
    progress == 1 — so the constant-h property is asserted over in-window
    progress values only (the frozen progress == 1 frame is unsupervised)."""
    sched = LinearTimestepSchedule()
    n = 32
    h = 1.0 / (n - 1)
    progress = torch.tensor([j * h for j in range(n - 1)])  # 0 .. 1 - h
    dt = sched.dt(progress, n_steps=n)
    assert dt.shape == progress.shape
    torch.testing.assert_close(dt, torch.full_like(progress, h))


def test_linear_dt_matches_one_step_timestep_advance() -> None:
    """dt must equal t(progress + h) - t(progress), the one rolling-step advance."""
    sched = LinearTimestepSchedule()
    n = 16
    h = 1.0 / (n - 1)
    progress = torch.tensor([0.0, h, 5 * h, 1.0 - h])
    expected = sched.timestep(progress + h) - sched.timestep(progress)
    torch.testing.assert_close(sched.dt(progress, n_steps=n), expected)


# --- LogNormTimestepSchedule ------------------------------------------------


def test_lognorm_timestep_fixes_endpoints() -> None:
    sched = LogNormTimestepSchedule(mean=0.0, std=1.0)
    endpoints = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(sched.timestep(endpoints), endpoints)


def test_lognorm_timestep_monotonic_and_in_unit_range() -> None:
    sched = LogNormTimestepSchedule(mean=0.3, std=1.5)
    t = sched.timestep(torch.linspace(0.0, 1.0, 50))
    assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
    assert torch.all(t[1:] >= t[:-1])  # non-decreasing


def test_lognorm_is_symmetric_at_mean_zero() -> None:
    # mean=0: t = sigmoid(std * Phi_inv(progress)); progress 0.5 -> t 0.5.
    sched = LogNormTimestepSchedule(mean=0.0, std=1.0)
    torch.testing.assert_close(sched.timestep(torch.tensor([0.5])), torch.tensor([0.5]))


def test_lognorm_negative_mean_concentrates_toward_small_t() -> None:
    # mean<0 shifts mass to small t: the grid midpoint maps below 0.5.
    sched = LogNormTimestepSchedule(mean=-1.0, std=1.0)
    assert sched.timestep(torch.tensor([0.5])).item() < 0.5


def test_lognorm_dt_matches_one_step_timestep_advance() -> None:
    sched = LogNormTimestepSchedule(mean=0.2, std=1.0)
    n = 16
    h = 1.0 / (n - 1)
    progress = torch.tensor([0.0, h, 7 * h, 1.0 - h, 1.0])
    expected = sched.timestep((progress + h).clamp(max=1.0)) - sched.timestep(progress)
    torch.testing.assert_close(sched.dt(progress, n_steps=n), expected)


def test_lognorm_dt_is_finite_and_nonnegative() -> None:
    sched = LogNormTimestepSchedule(mean=-0.5, std=1.2)
    dt = sched.dt(torch.linspace(0.0, 1.0, 32), n_steps=32)
    assert torch.isfinite(dt).all()
    assert torch.all(dt >= 0.0)


# --- timesteps(n_steps): the full denoising t-grid ---------------------------


def test_linear_timesteps_grid_is_uniform() -> None:
    n = 9
    torch.testing.assert_close(LinearTimestepSchedule().timesteps(n), torch.linspace(0.0, 1.0, n))


def test_timesteps_grid_is_timestep_on_uniform_progress() -> None:
    n = 12
    uniform = torch.linspace(0.0, 1.0, n)
    for sched in (
        LinearTimestepSchedule(),
        LogNormTimestepSchedule(mean=-0.4, std=1.3),
    ):
        grid = sched.timesteps(n)
        assert grid.shape == (n,)
        torch.testing.assert_close(grid, sched.timestep(uniform))
        # endpoints are pinned to the full [0, 1] range
        torch.testing.assert_close(grid[[0, -1]], torch.tensor([0.0, 1.0]))


# --- enum / config wiring ----------------------------------------------------


def test_enum_resolves_each_variant() -> None:
    assert isinstance(TimestepSchedules.LINEAR.schedule, LinearTimestepSchedule)
    assert isinstance(TimestepSchedules.LOG_NORM.schedule, LogNormTimestepSchedule)


def test_lognorm_enum_uses_jit_paper_values() -> None:
    # JiT (arxiv 2511.13720, "Back to Basics"): logit-normal P_mean=-0.8, P_std=0.8.
    sched = TimestepSchedules.LOG_NORM.schedule
    assert isinstance(sched, LogNormTimestepSchedule)
    assert sched.mean == -0.8
    assert sched.std == 0.8


def test_config_defaults_to_lognorm_and_model_resolves_it() -> None:
    cfg = RollingFlowConfig(vocabulary_size=8)
    assert cfg.timestep_schedule is TimestepSchedules.LOG_NORM
    model = RollingFlowSpeaker(cfg)
    assert isinstance(model.schedule, LogNormTimestepSchedule)


def test_model_resolves_lognorm_schedule_from_config() -> None:
    cfg = RollingFlowConfig(vocabulary_size=8, timestep_schedule=TimestepSchedules.LOG_NORM)
    model = RollingFlowSpeaker(cfg)
    assert isinstance(model.schedule, LogNormTimestepSchedule)


# --- LogNormTimestepSchedule: tail trimming (eps) ---------------------------


def test_lognorm_eps_default_is_zero() -> None:
    """Omitting eps leaves the class behaviour unchanged (backward compatible)."""
    assert LogNormTimestepSchedule(mean=-0.8, std=0.8).eps == 0.0


def test_lognorm_eps_zero_matches_untrimmed_warp() -> None:
    """eps=0 must reproduce the plain logit-normal warp on interior points."""
    mean, std = -0.8, 0.8
    sched = LogNormTimestepSchedule(mean=mean, std=std, eps=0.0)
    progress = torch.linspace(0.1, 0.9, 25)  # interior points only
    expected = torch.sigmoid(mean + std * torch.special.ndtri(progress))
    torch.testing.assert_close(sched.timestep(progress), expected)


def test_lognorm_trimmed_fixes_endpoints_exactly() -> None:
    """speak() freezes a frame only when timestep(1) == 1, so the endpoints
    must be exact, not merely close."""
    sched = LogNormTimestepSchedule(mean=-0.8, std=0.8, eps=0.025)
    assert sched.timestep(torch.tensor([0.0])).item() == 0.0
    assert sched.timestep(torch.tensor([1.0])).item() == 1.0


def test_lognorm_trimmed_is_monotonic_in_unit_range() -> None:
    sched = LogNormTimestepSchedule(mean=-0.8, std=0.8, eps=0.025)
    t = sched.timestep(torch.linspace(0.0, 1.0, 200))
    assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
    assert torch.all(t[1:] >= t[:-1])  # non-decreasing


def test_lognorm_trimming_bounds_the_final_step() -> None:
    """The point of trimming: the t=1 cliff shrinks well below the untrimmed
    schedule's final Euler step."""
    n = 32
    untrimmed = LogNormTimestepSchedule(mean=-0.8, std=0.8, eps=0.0).timesteps(n)
    trimmed = LogNormTimestepSchedule(mean=-0.8, std=0.8, eps=0.025).timesteps(n)
    last_untrimmed = (untrimmed[-1] - untrimmed[-2]).item()
    last_trimmed = (trimmed[-1] - trimmed[-2]).item()
    assert last_trimmed < 0.15
    assert last_trimmed < 0.5 * last_untrimmed


def test_lognorm_trimming_keeps_the_early_emphasis_bump() -> None:
    """Trimming tames the tails but the interior still has finer steps than
    either end -- the early-emphasis bump is preserved."""
    n = 32
    grid = LogNormTimestepSchedule(mean=-0.8, std=0.8, eps=0.025).timesteps(n)
    dt = grid[1:] - grid[:-1]
    assert dt.min() < dt[0]  # finer than the noisy-end step
    assert dt.min() < dt[-1]  # finer than the clean-end step


def test_lognorm_eps_out_of_range_raises() -> None:
    with pytest.raises(AssertionError):
        LogNormTimestepSchedule(mean=0.0, std=1.0, eps=-0.01)
    with pytest.raises(AssertionError):
        LogNormTimestepSchedule(mean=0.0, std=1.0, eps=0.5)


# --- enum: trimmed variant ---------------------------------------------------


def test_enum_resolves_trimmed_variant() -> None:
    sched = TimestepSchedules.LOG_NORM_TRIMMED.schedule
    assert isinstance(sched, LogNormTimestepSchedule)
    assert sched.mean == -0.8
    assert sched.std == 0.8
    assert sched.eps == 0.025


def test_enum_log_norm_stays_untrimmed() -> None:
    """LOG_NORM must remain bit-identical so in-flight runs are unaffected."""
    sched = TimestepSchedules.LOG_NORM.schedule
    assert isinstance(sched, LogNormTimestepSchedule)
    assert sched.eps == 0.0

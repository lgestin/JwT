import pytest
import torch
import torch.nn as nn

from jwt.training.muon import NorMuonScheduleFree, zeropower_via_newtonschulz5
from jwt.training.optimizer import (
    OptimizerConfig,
    Optimizers,
    warmup_scale,
)


@pytest.mark.parametrize("shape", [(8, 8), (16, 4), (4, 16)])
def test_newtonschulz_orthogonalizes(shape) -> None:
    """The polar factor has near-unit singular values, tall or wide."""
    torch.manual_seed(0)
    g = torch.randn(*shape)
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(g).float())
    # The quintic iteration lands in ~[0.5, 1.5] by design, not exactly 1.
    assert s.min() > 0.5 and s.max() < 1.5
    assert zeropower_via_newtonschulz5(g).shape == g.shape


def _muon_group(p: nn.Parameter, **kw):
    return [dict(params=[p], use_muon=True, **kw)]


def _drive(opt, p, grads) -> None:
    for g in grads:
        p.grad = g.clone()
        opt.step()


def _reload(opt, p, tmp_path):
    """Round-trip the state through disk, as the trainer does (no aliasing)."""
    path = tmp_path / "opt.pt"
    torch.save(opt.state_dict(), path)
    group = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
    fresh = NorMuonScheduleFree([dict(params=[p], **group)])
    fresh.load_state_dict(torch.load(path, weights_only=False))
    return fresh


def test_train_eval_round_trip_is_identity() -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(4, 4))
    opt = NorMuonScheduleFree(_muon_group(p, lr=0.1))
    _drive(opt, p, [torch.randn(4, 4) for _ in range(3)])
    y = p.detach().clone()
    opt.eval()
    x = p.detach().clone()
    opt.train()
    assert torch.allclose(p.detach(), y, atol=1e-5)
    assert not torch.allclose(x, y, atol=1e-4)


def test_eval_iterate_is_the_running_average() -> None:
    """x is the lr^2-weighted average of z (uniform at constant lr)."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(4, 4))
    opt = NorMuonScheduleFree(_muon_group(p, lr=0.1))
    zs = []
    for _ in range(5):
        p.grad = torch.randn(4, 4)
        opt.step()
        zs.append(opt.state[p]["z"].clone())
    opt.eval()
    assert torch.allclose(p.detach(), torch.stack(zs).mean(0), atol=1e-5)


def test_step_descends_a_quadratic() -> None:
    """Both branches reduce a convex objective, measured at the averaged iterate."""
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(8, 8))
    b = nn.Parameter(torch.randn(8))
    target = torch.randn(8, 8)
    opt = NorMuonScheduleFree(
        [
            dict(params=[w], use_muon=True, lr=0.05),
            dict(params=[b], use_muon=False, lr=0.05),
        ]
    )

    def loss_at_eval() -> float:
        opt.eval()
        with torch.no_grad():
            out = float(((w - target) ** 2).sum() + (b**2).sum())
        opt.train()
        return out

    before = loss_at_eval()
    for _ in range(50):
        loss = ((w - target) ** 2).sum() + (b**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss_at_eval() < before


def test_warmup_scale_ramps_linearly_then_holds() -> None:
    assert warmup_scale(0, 10) == pytest.approx(0.1)
    assert warmup_scale(9, 10) == 1.0
    assert warmup_scale(50, 10) == 1.0
    assert warmup_scale(0, 0) == 1.0


def test_step_rejects_the_eval_iterate() -> None:
    p = nn.Parameter(torch.randn(4, 4))
    opt = NorMuonScheduleFree(_muon_group(p, lr=0.1))
    _drive(opt, p, [torch.randn(4, 4)])
    opt.eval()
    p.grad = torch.randn(4, 4)
    with pytest.raises(AssertionError, match="train"):
        opt.step()


def test_checkpoint_taken_in_eval_mode_resumes_on_the_same_trajectory(
    tmp_path,
) -> None:
    """The trainer saves after eval(); restoring must put the run back on `y`."""
    torch.manual_seed(0)
    init = torch.randn(6, 4)
    grads = [torch.randn(6, 4) for _ in range(6)]

    p = nn.Parameter(init.clone())
    opt = NorMuonScheduleFree(_muon_group(p, lr=0.05))
    _drive(opt, p, grads[:3])
    opt.eval()

    p2 = nn.Parameter(p.detach().clone())
    opt2 = _reload(opt, p2, tmp_path)
    assert opt2.param_groups[0]["k"] == 3
    assert opt2.param_groups[0]["train_mode"] is False

    opt.train()
    opt2.train()
    _drive(opt, p, grads[3:])
    _drive(opt2, p2, grads[3:])
    opt.eval()
    opt2.eval()
    assert torch.allclose(p.detach(), p2.detach(), atol=1e-6)


class _Model(nn.Module):
    """One of each parameter kind the split has to route."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(4, 8)
        self.registers = nn.Parameter(torch.zeros(2, 8))
        self.gain = nn.Parameter(torch.ones(8))
        self.hidden = nn.Linear(8, 24)
        self.head = nn.Linear(8, 6)


def test_build_routes_linear_weights_to_muon_and_the_rest_to_adam() -> None:
    model = _Model()
    opt = OptimizerConfig(name=Optimizers.SF_NORMUON).build(model, head=model.head)
    assert isinstance(opt, NorMuonScheduleFree)
    muon = opt.param_groups[0]
    assert [id(p) for p in muon["params"]] == [id(model.hidden.weight)]
    grouped = [p for g in opt.param_groups for p in g["params"]]
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    assert len(grouped) == len(list(model.parameters()))


def test_build_adamw() -> None:
    opt = OptimizerConfig(name=Optimizers.ADAMW, lr=1e-3).build(_Model())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["initial_lr"] == 1e-3

"""Schedule-free NorMuon (arXiv:2605.23061)."""

import math

import torch
from torch.optim import Optimizer

# Update RMS relative to the LR (torch.optim.Muon's "match_rms_adamw" constant).
_RMS_SCALE = 0.2


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """Quintic Newton-Schulz approximation to the polar factor of `G`, in bf16."""
    assert G.ndim == 2, f"expected a matrix, got shape {tuple(G.shape)}"
    a, b, c = coefficients
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / X.norm().clamp(min=eps)
    for _ in range(steps):
        gram = X @ X.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        X = torch.addmm(X, gram_update, X, beta=a)
    return X.T if transposed else X


class NorMuonScheduleFree(Optimizer):
    """Schedule-free NorMuon on `use_muon` groups, schedule-free AdamW on the rest.

    The live weights hold `y = (1 - beta) z + beta x` where gradients are taken;
    `eval()`/`train()` swap them to/from the averaged iterate `x`, so sampling,
    validation and checkpointing must run in eval mode. `betas[0]` is the
    interpolation coefficient, `betas[1]` the second-moment decay.
    """

    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            assert "use_muon" in group, "every param group needs a `use_muon` flag"
            group.setdefault("betas", (0.9, 0.95))
            group.setdefault("eps", 1e-8)
            group.setdefault("weight_decay", 0.0)
            if group["use_muon"]:
                group.setdefault("momentum", 0.8)
            # Kept on the group so state_dict carries them and --resume is exact.
            group.setdefault("k", 0)
            group.setdefault("weight_sum", 0.0)
            group.setdefault("train_mode", True)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def eval(self) -> None:
        """Move the live weights from `y` to the averaged iterate `x`."""
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta = group["betas"][0]
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    p.lerp_(end=state["z"], weight=1.0 - 1.0 / beta)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Move the live weights back from `x` to the gradient iterate `y`."""
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta = group["betas"][0]
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    p.lerp_(end=state["z"], weight=1.0 - beta)
            group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            assert group["train_mode"], (
                "step() needs the live weights on the `y` iterate; call train() first"
            )
            beta, beta2 = group["betas"]
            eps, decay = group["eps"], group["weight_decay"]
            k, lr = group["k"], group["lr"]
            # lr^2-weighted averaging, as in the schedule-free reference.
            weight_sum = group["weight_sum"] = group["weight_sum"] + lr * lr
            ckp1 = (lr * lr) / weight_sum if weight_sum > 0 else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad, state = p.grad, self.state[p]
                if "z" not in state:
                    state["z"] = p.clone()
                    if group["use_muon"]:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_moment"] = torch.zeros(
                            p.shape[0], device=p.device, dtype=torch.float32
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                z = state["z"]

                if group["use_muon"]:
                    # Plain momentum: the gradient at `y` already carries the lookahead.
                    momentum = state["momentum_buffer"]
                    momentum.lerp_(grad, 1.0 - group["momentum"])
                    update = zeropower_via_newtonschulz5(momentum).to(grad.dtype)
                    # Per-neuron RMS normalization, then rescale to Adam-comparable RMS.
                    second_moment = state["second_moment"]
                    row_ms = update.square().mean(dim=1).float()
                    second_moment.mul_(beta2).add_(row_ms, alpha=1.0 - beta2)
                    update = (
                        update / (second_moment.sqrt() + eps).to(grad.dtype)[:, None]
                    )
                    m, n = p.shape
                    update *= (
                        _RMS_SCALE
                        * lr
                        * math.sqrt(m * n)
                        / update.norm().clamp_min(1e-12)
                    )
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = (exp_avg_sq / (1.0 - beta2 ** (k + 1))).sqrt_().add_(eps)
                    update = grad.div(denom).mul_(lr)

                if decay != 0.0:
                    update.add_(p, alpha=lr * decay)  # decoupled decay at `y`
                # y <- (1 - beta) z' + beta x' with x' = lerp(x, z', ckp1), in place.
                p.lerp_(end=z, weight=ckp1)
                p.add_(update, alpha=beta * (1.0 - ckp1) - 1.0)
                z.sub_(update)

            group["k"] = k + 1

        return loss

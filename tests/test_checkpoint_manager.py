import json
from pathlib import Path

import pytest
import torch
from torch import nn

from jwt.training.checkpoint_manager import CheckpointManager
from jwt.training.ema import EMA


def const_model(value: float) -> nn.Module:
    """A 2x2 linear layer with every parameter filled with `value`."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)
    return model


def test_ema_state_round_trips_through_checkpoint(tmp_path: Path) -> None:
    """EMA shadow weights and decay survive a save / load cycle."""
    model = const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for tensor in ema.shadow_weights.values():
            tensor.fill_(7.0)

    manager = CheckpointManager(exp_path=tmp_path)
    manager.save(
        step=10,
        model=model,
        optimizer=optimizer,
        scaler=None,
        best_loss=1.23,
        additional_state={"ema": ema.state_dict()},
    )

    fresh_model = const_model(1.0)
    fresh_ema = EMA(fresh_model, decay=0.9999)
    manager.load_latest(fresh_model, ema=fresh_ema)

    assert fresh_ema.decay == 0.5
    for name, tensor in ema.shadow_weights.items():
        assert torch.equal(fresh_ema.shadow_weights[name], tensor)


def test_load_without_ema_in_checkpoint_warns(tmp_path: Path) -> None:
    """Loading a pre-EMA checkpoint into an EMA leaves it initialized + warns."""
    model = const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())

    manager = CheckpointManager(exp_path=tmp_path)
    manager.save(step=5, model=model, optimizer=optimizer, scaler=None, best_loss=2.0)

    fresh_model = const_model(4.0)
    fresh_ema = EMA(fresh_model, decay=0.9999)
    with pytest.warns(UserWarning, match="no EMA state"):
        manager.load_latest(fresh_model, ema=fresh_ema)

    # EMA keeps its from-init shadow (the 4.0-filled fresh model).
    for tensor in fresh_ema.shadow_weights.values():
        assert torch.allclose(tensor, torch.full_like(tensor, 4.0))


def test_cleanup_keeps_best_latest_and_recent(tmp_path: Path) -> None:
    """Cleanup keeps the best target plus the 2 most recent checkpoints."""
    model = const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(exp_path=tmp_path)

    # Step 10 has the lowest loss; the worse later saves leave `best` pinned
    # to it while `latest` advances.
    manager.save(
        step=10,
        model=model,
        optimizer=optimizer,
        scaler=None,
        best_loss=0.5,
        val_loss=0.5,
    )
    for step in (20, 30, 40, 50):
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=None,
            best_loss=0.5,
            val_loss=0.9,
        )

    manager.cleanup_old_checkpoints(keep_recent=2)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint.[0-9]*.pt"))
    assert remaining == [
        "checkpoint.10.pt",  # best target
        "checkpoint.40.pt",  # 2nd most recent
        "checkpoint.50.pt",  # most recent / latest target
    ]


def test_cleanup_keeps_symlinks_resolvable(tmp_path: Path) -> None:
    """The best and latest symlinks still resolve to real files after cleanup."""
    model = const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(exp_path=tmp_path)

    manager.save(
        step=10,
        model=model,
        optimizer=optimizer,
        scaler=None,
        best_loss=0.5,
        val_loss=0.5,
    )
    for step in (20, 30, 40):
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=None,
            best_loss=0.5,
            val_loss=0.9,
        )

    manager.cleanup_old_checkpoints(keep_recent=2)

    assert manager.best_checkpoint_path.resolve().exists()
    assert manager.latest_checkpoint_path.resolve().exists()


def test_cleanup_keep_recent_zero_keeps_only_symlink_targets(tmp_path: Path) -> None:
    """keep_recent=0 keeps only the best and latest targets, without crashing."""
    model = const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(exp_path=tmp_path)

    manager.save(
        step=10,
        model=model,
        optimizer=optimizer,
        scaler=None,
        best_loss=0.5,
        val_loss=0.5,
    )
    for step in (20, 30):
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=None,
            best_loss=0.5,
            val_loss=0.9,
        )

    manager.cleanup_old_checkpoints(keep_recent=0)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint.[0-9]*.pt"))
    assert remaining == [
        "checkpoint.10.pt",  # best target
        "checkpoint.30.pt",  # latest target
    ]


def save_losses(manager: CheckpointManager, losses: dict[int, float]) -> None:
    """Save one checkpoint per `step: val_loss`, trainer best-so-far alongside."""
    model = const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    best = float("inf")
    for step, val_loss in losses.items():
        best = min(best, val_loss)
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=None,
            best_loss=best,
            val_loss=val_loss,
        )


def best_target(manager: CheckpointManager) -> str:
    return manager.best_checkpoint_path.resolve().name


def test_best_points_at_the_lowest_val_loss(tmp_path: Path) -> None:
    """`best` follows the lowest checkpoint loss, not every save."""
    manager = CheckpointManager(exp_path=tmp_path)
    save_losses(manager, {10: 0.5, 20: 0.3, 30: 0.4})
    # 0.3 at step 20 is the minimum; step 30 is only the latest.
    assert best_target(manager) == "checkpoint.20.pt"
    assert manager.latest_checkpoint_path.resolve().name == "checkpoint.30.pt"


def test_best_survives_cleanup_after_many_worse_saves(tmp_path: Path) -> None:
    """The true best is kept even once it falls out of the recent window."""
    manager = CheckpointManager(exp_path=tmp_path)
    save_losses(manager, {10: 0.5, 20: 0.2, 30: 0.4, 40: 0.6, 50: 0.7, 60: 0.8})
    manager.cleanup_old_checkpoints(keep_recent=2)
    remaining = sorted(p.name for p in tmp_path.glob("checkpoint.[0-9]*.pt"))
    assert remaining == ["checkpoint.20.pt", "checkpoint.50.pt", "checkpoint.60.pt"]


def test_no_best_before_any_validation(tmp_path: Path) -> None:
    """An unknown (inf) loss never creates the `best` symlink."""
    manager = CheckpointManager(exp_path=tmp_path)
    save_losses(manager, {10: float("inf")})
    assert not manager.best_checkpoint_path.is_symlink()


def test_unknown_loss_does_not_move_best(tmp_path: Path) -> None:
    """A save with no validation loss after a real best leaves `best` alone."""
    manager = CheckpointManager(exp_path=tmp_path)
    save_losses(manager, {10: 0.5, 20: float("inf")})
    assert best_target(manager) == "checkpoint.10.pt"


def test_best_val_loss_survives_a_restart(tmp_path: Path) -> None:
    """A fresh manager on the same run knows the best loss from the sidecar."""
    save_losses(CheckpointManager(exp_path=tmp_path), {10: 0.5, 20: 0.3})
    resumed = CheckpointManager(exp_path=tmp_path)
    assert resumed.best_val_loss == pytest.approx(0.3)

    # 0.35 does not beat 0.3; 0.2 does.
    save_losses(resumed, {30: 0.35})
    assert best_target(resumed) == "checkpoint.20.pt"
    save_losses(resumed, {40: 0.2})
    assert best_target(resumed) == "checkpoint.40.pt"


def test_best_without_sidecar_is_replaced(tmp_path: Path) -> None:
    """A pre-fix run's `best` symlink (no sidecar) is taken over by the next loss."""
    manager = CheckpointManager(exp_path=tmp_path)
    save_losses(manager, {10: 0.1})
    manager.best_meta_path.unlink()

    resumed = CheckpointManager(exp_path=tmp_path)
    assert resumed.best_val_loss == float("inf")
    # 0.9 is worse than the old 0.1, but that loss is no longer known.
    save_losses(resumed, {20: 0.9})
    assert best_target(resumed) == "checkpoint.20.pt"


def test_stale_sidecar_is_ignored(tmp_path: Path) -> None:
    """A sidecar naming a checkpoint other than the `best` target is not trusted."""
    manager = CheckpointManager(exp_path=tmp_path)
    save_losses(manager, {10: 0.5})
    manager.best_meta_path.write_text(
        json.dumps({"checkpoint": "checkpoint.99.pt", "val_loss": 0.01})
    )
    assert CheckpointManager(exp_path=tmp_path).best_val_loss == float("inf")


def test_val_loss_round_trips_and_defaults_for_old_checkpoints(tmp_path: Path) -> None:
    """`load` returns the stored loss, and inf for a checkpoint without the key."""
    manager = CheckpointManager(exp_path=tmp_path)
    save_losses(manager, {10: 0.25})
    assert manager.load_latest(const_model(0.0))["val_loss"] == pytest.approx(0.25)

    path = tmp_path / "checkpoint.10.pt"
    data = torch.load(path, weights_only=False)
    del data["val_loss"]
    torch.save(data, path)
    assert manager.load(path, const_model(0.0))["val_loss"] == float("inf")

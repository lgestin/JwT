import pytest
import torch
from torch import nn

from tts.training.checkpoint_manager import CheckpointManager
from tts.training.ema import EMA


def _const_model(value: float) -> nn.Module:
    """A 2x2 linear layer with every parameter filled with ``value``."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)
    return model


def test_ema_state_round_trips_through_checkpoint(tmp_path) -> None:
    """EMA shadow weights and decay survive a save / load cycle."""
    model = _const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for tensor in ema._shadow.values():
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

    fresh_model = _const_model(1.0)
    fresh_ema = EMA(fresh_model, decay=0.9999)
    manager.load_latest(fresh_model, ema=fresh_ema)

    assert fresh_ema.decay == 0.5
    for name, tensor in ema._shadow.items():
        assert torch.equal(fresh_ema._shadow[name], tensor)


def test_load_without_ema_in_checkpoint_warns(tmp_path) -> None:
    """Loading a pre-EMA checkpoint into an EMA leaves it initialized + warns."""
    model = _const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())

    manager = CheckpointManager(exp_path=tmp_path)
    manager.save(
        step=5, model=model, optimizer=optimizer, scaler=None, best_loss=2.0
    )

    fresh_model = _const_model(4.0)
    fresh_ema = EMA(fresh_model, decay=0.9999)
    with pytest.warns(UserWarning, match="no EMA state"):
        manager.load_latest(fresh_model, ema=fresh_ema)

    # EMA keeps its from-init shadow (the 4.0-filled fresh model).
    for tensor in fresh_ema._shadow.values():
        assert torch.allclose(tensor, torch.full_like(tensor, 4.0))


def test_cleanup_keeps_best_latest_and_recent(tmp_path) -> None:
    """Cleanup keeps the best target plus the 2 most recent checkpoints."""
    model = _const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(exp_path=tmp_path)

    # Step 10 is the best; later saves pass best_loss=inf so save() leaves
    # the "best" symlink pinned to step 10 while "latest" advances.
    manager.save(
        step=10, model=model, optimizer=optimizer, scaler=None, best_loss=0.5
    )
    for step in (20, 30, 40, 50):
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=None,
            best_loss=float("inf"),
        )

    manager.cleanup_old_checkpoints(keep_recent=2)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint.[0-9]*.pt"))
    assert remaining == [
        "checkpoint.10.pt",  # best target
        "checkpoint.40.pt",  # 2nd most recent
        "checkpoint.50.pt",  # most recent / latest target
    ]


def test_cleanup_keeps_symlinks_resolvable(tmp_path) -> None:
    """The best and latest symlinks still resolve to real files after cleanup."""
    model = _const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(exp_path=tmp_path)

    manager.save(
        step=10, model=model, optimizer=optimizer, scaler=None, best_loss=0.5
    )
    for step in (20, 30, 40):
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=None,
            best_loss=float("inf"),
        )

    manager.cleanup_old_checkpoints(keep_recent=2)

    assert manager.best_checkpoint_path.resolve().exists()
    assert manager.latest_checkpoint_path.resolve().exists()


def test_cleanup_keep_recent_zero_keeps_only_symlink_targets(tmp_path) -> None:
    """keep_recent=0 keeps only the best and latest targets, without crashing."""
    model = _const_model(1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(exp_path=tmp_path)

    manager.save(
        step=10, model=model, optimizer=optimizer, scaler=None, best_loss=0.5
    )
    for step in (20, 30):
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=None,
            best_loss=float("inf"),
        )

    manager.cleanup_old_checkpoints(keep_recent=0)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint.[0-9]*.pt"))
    assert remaining == [
        "checkpoint.10.pt",  # best target
        "checkpoint.30.pt",  # latest target
    ]

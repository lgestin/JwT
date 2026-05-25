import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from jwt.training.ema import EMA


@dataclass
class CheckpointManager:
    """Manages model checkpoints during training.

    Handles saving checkpoints at regular intervals and tracking
    the best model based on validation loss.
    """

    exp_path: Path
    save_best: bool = True

    def __post_init__(self):
        """Coerce str inputs and create the directory."""
        self.exp_path = Path(self.exp_path)
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.exp_path / "checkpoint.best.pt"
        self.latest_checkpoint_path = self.exp_path / "checkpoint.latest.pt"

    @staticmethod
    def _point_symlink(link_path: Path, target_name: str) -> None:
        """(Re)point a symlink at a sibling file, replacing any existing one."""
        if link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_name)

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        best_loss: float,
        additional_state: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer state to save
            scaler: GradScaler state to save (if using AMP)
            best_loss: Best validation loss so far
            additional_state: Additional state dict to save

        Returns:
            Path to the saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint_data: dict[str, Any] = {
            "step": step,
            "best_loss": best_loss,
            "opt": optimizer.state_dict(),
        }

        # Handle ConditionalFlowMatcher or direct model
        if hasattr(model, "denoiser"):
            # It's a ConditionalFlowMatcher, save the denoiser
            checkpoint_data["model"] = model.denoiser.state_dict()  # ty: ignore[unresolved-attribute]
            if hasattr(model.denoiser, "dims"):
                checkpoint_data["dims"] = model.denoiser.dims
            if hasattr(model.denoiser, "cfg"):
                checkpoint_data["config"] = model.denoiser.cfg
        else:
            # It's a direct model
            checkpoint_data["model"] = model.state_dict()
            if hasattr(model, "dims"):
                checkpoint_data["dims"] = model.dims
            if hasattr(model, "cfg"):
                checkpoint_data["config"] = model.cfg

        if scaler is not None:
            checkpoint_data["scaler"] = scaler.state_dict()

        # Add any additional state
        if additional_state:
            checkpoint_data.update(additional_state)

        # Save checkpoint
        checkpoint_path = self.exp_path / f"checkpoint.{step}.pt"
        torch.save(checkpoint_data, checkpoint_path)

        # Point the "latest" symlink at the checkpoint just written.
        self._point_symlink(self.latest_checkpoint_path, checkpoint_path.name)

        # Point the "best" symlink here if this is the best loss so far.
        if self.save_best and best_loss < float("inf"):
            self._point_symlink(self.best_checkpoint_path, checkpoint_path.name)

        return checkpoint_path

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        ema: EMA | None = None,
        map_location: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Load the most recent checkpoint.

        Follows the "checkpoint.latest.pt" symlink that `save` maintains. Runs
        created before that symlink existed fall back to the highest-numbered
        "checkpoint.<step>.pt" on disk.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scaler: GradScaler to load state into (optional)
            map_location: Device mapping for loading

        Returns:
            Dictionary with checkpoint metadata
        """
        latest = self.latest_checkpoint_path
        if not latest.exists():
            # Fallback for runs predating the latest symlink. Glob only
            # numeric-stepped checkpoints — "checkpoint.*.pt" would also match
            # the "checkpoint.best.pt" symlink, and int("best") raises.
            checkpoints = sorted(
                self.exp_path.glob("checkpoint.[0-9]*.pt"),
                key=lambda p: int(p.stem.split(".")[-1]),
            )
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.exp_path}")
            latest = checkpoints[-1]

        return self.load(
            latest,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            map_location=map_location,
        )

    def load_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        map_location: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Load the best checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scaler: GradScaler to load state into (optional)
            map_location: Device mapping for loading

        Returns:
            Dictionary with checkpoint metadata
        """
        if not self.best_checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {self.best_checkpoint_path}")

        return self.load(
            self.best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=map_location,
        )

    def load(
        self,
        checkpoint_path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        ema: EMA | None = None,
        map_location: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Load a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scaler: GradScaler to load state into (optional)
            map_location: Device mapping for loading

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        # Load checkpoint data. weights_only=False is required because we
        # persist the model config (a dataclass) alongside the state dict.
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Load model state - handle ConditionalFlowMatcher or direct model
        if hasattr(model, "denoiser"):
            # It's a ConditionalFlowMatcher, load into the denoiser
            model.denoiser.load_state_dict(checkpoint_data["model"])  # ty: ignore[unresolved-attribute]
        else:
            # It's a direct model
            model.load_state_dict(checkpoint_data["model"])

        # Load optimizer state if provided
        if optimizer is not None and "opt" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["opt"])

        # Load scaler state if provided
        if scaler is not None and "scaler" in checkpoint_data:
            scaler.load_state_dict(checkpoint_data["scaler"])

        # Load EMA state if provided. A checkpoint predating EMA has no "ema"
        # key — leave the EMA at its from-loaded-weights initialization.
        if ema is not None:
            if "ema" in checkpoint_data:
                ema.load_state_dict(checkpoint_data["ema"])
            else:
                warnings.warn(
                    "checkpoint has no EMA state; EMA initialized from the loaded model weights",
                    stacklevel=2,
                )

        # Return metadata
        metadata = {
            "step": checkpoint_data.get("step", 0),
            "best_loss": checkpoint_data.get("best_loss", float("inf")),
        }

        # Add dims if available
        if "dims" in checkpoint_data:
            metadata["dims"] = checkpoint_data["dims"]

        # Add model config dataclass if available
        if "config" in checkpoint_data:
            metadata["config"] = checkpoint_data["config"]

        return metadata

    def cleanup_old_checkpoints(self, keep_recent: int = 2) -> None:
        """Remove checkpoints not worth keeping during a training run.

        Keeps the checkpoints targeted by the ``best`` and ``latest`` symlinks
        plus the ``keep_recent`` highest-step checkpoints; deletes the rest.

        Args:
            keep_recent: Number of most-recent checkpoints to keep, in addition
                to the ``best`` and ``latest`` symlink targets.
        """
        # Glob only numeric-stepped checkpoints — "checkpoint.*.pt" would also
        # match the "checkpoint.best.pt" symlink, and int("best") raises.
        checkpoints = sorted(
            self.exp_path.glob("checkpoint.[0-9]*.pt"),
            key=lambda p: int(p.stem.split(".")[-1]),
        )

        # Protected set: resolved real paths that must survive cleanup.
        protected: set[Path] = set()
        for link in (self.best_checkpoint_path, self.latest_checkpoint_path):
            if link.is_symlink():
                protected.add(link.resolve())
        # The "if" guard avoids checkpoints[-0:] returning the whole list.
        if keep_recent > 0:
            for checkpoint in checkpoints[-keep_recent:]:
                protected.add(checkpoint.resolve())

        for checkpoint in checkpoints:
            if checkpoint.resolve() not in protected:
                checkpoint.unlink()

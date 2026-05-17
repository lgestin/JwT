from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CheckpointManager:
    """Manages model checkpoints during training.

    Handles saving checkpoints at regular intervals and tracking
    the best model based on validation loss.
    """

    exp_path: Path | str
    save_best: bool = True

    def __post_init__(self):
        """Convert exp_path to Path and create directory."""
        self.exp_path: Path = Path(self.exp_path)
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.exp_path / "checkpoint.best.pt"

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None,
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
        checkpoint_data = {
            "step": step,
            "best_loss": best_loss,
            "opt": optimizer.state_dict(),
        }

        # Handle ConditionalFlowMatcher or direct model
        if hasattr(model, "denoiser"):
            # It's a ConditionalFlowMatcher, save the denoiser
            checkpoint_data["model"] = model.denoiser.state_dict()
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

        # Update best checkpoint symlink if this is the best
        if self.save_best and best_loss < float("inf"):
            # Remove old symlink if it exists
            if self.best_checkpoint_path.is_symlink():
                self.best_checkpoint_path.unlink()
            # Create new symlink to current checkpoint
            self.best_checkpoint_path.symlink_to(checkpoint_path.name)

        return checkpoint_path

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        map_location: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Load the latest checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scaler: GradScaler to load state into (optional)
            map_location: Device mapping for loading

        Returns:
            Dictionary with checkpoint metadata
        """
        # Find latest checkpoint
        checkpoints = sorted(
            self.exp_path.glob("checkpoint.*.pt"),
            key=lambda p: int(p.stem.split(".")[-1]),
        )

        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.exp_path}")

        latest_checkpoint = checkpoints[-1]
        return self.load(
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=map_location,
        )

    def load_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
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
            raise FileNotFoundError(
                f"Best checkpoint not found at {self.best_checkpoint_path}"
            )

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
        scaler: torch.cuda.amp.GradScaler | None = None,
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
        checkpoint_data = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

        # Load model state - handle ConditionalFlowMatcher or direct model
        if hasattr(model, "denoiser"):
            # It's a ConditionalFlowMatcher, load into the denoiser
            model.denoiser.load_state_dict(checkpoint_data["model"])
        else:
            # It's a direct model
            model.load_state_dict(checkpoint_data["model"])

        # Load optimizer state if provided
        if optimizer is not None and "opt" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["opt"])

        # Load scaler state if provided
        if scaler is not None and "scaler" in checkpoint_data:
            scaler.load_state_dict(checkpoint_data["scaler"])

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

    def cleanup_old_checkpoints(self, keep_n: int = 5):
        """Remove old checkpoints, keeping only the most recent ones.

        Args:
            keep_n: Number of recent checkpoints to keep
        """
        checkpoints = sorted(
            self.exp_path.glob("checkpoint.*.pt"),
            key=lambda p: int(p.stem.split(".")[-1]),
        )

        # Don't delete the checkpoint that best.pt points to
        best_target = None
        if self.best_checkpoint_path.is_symlink():
            best_target = self.best_checkpoint_path.resolve()

        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_n]:
            if best_target is None or checkpoint.resolve() != best_target:
                checkpoint.unlink()

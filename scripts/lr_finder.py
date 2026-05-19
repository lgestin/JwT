"""LR range test (Smith) for RollingFlowSpeaker.

Ramps the learning rate exponentially from ``min_lr`` to ``max_lr`` over
``num_steps`` optimizer steps, logging loss and the pre-step gradient
norm. No gradient clipping, no validation, no sampling, no checkpoints.

After the run, writes ``lr_find.png`` (smoothed loss + grad norm vs LR)
to ``output_dir`` and prints the LR at minimum smoothed loss and the LR
at which the run diverged (if it did).

Heuristic for picking a max LR: ~10x below the divergence LR, or just
past where the smoothed-loss slope is steepest.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from simple_parsing import ArgumentParser
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tts.data.collate import collate
from tts.data.dataset import AudioDataset
from tts.data.source import ArrowTTSSource
from tts.data.text import Tokenizer, Vocabulary
from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from tts.model.transformer import TransformerConfig
from tts.training.tensorboard_logger import TensorBoardLogger


@dataclass
class Args:
    # Data (defaults mirror scripts/train.py)
    vocab_path: str = "/data/ljspeech/vocabulary.json"
    arrow_path: str = "data/ljspeech_24khz.arrow"
    sample_rate: int = 24000
    batch_size: int = 64
    num_workers: int = 6
    # Model (defaults mirror scripts/train.py)
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    mel_dim: int = 100
    n_denoising_steps: int = 32
    # LR finder
    output_dir: str = "outputs/lr_finder"
    device: str = "cuda"
    min_lr: float = 1e-7
    max_lr: float = 1e-1
    num_steps: int = 1000
    smooth_beta: float = 0.98
    divergence_factor: float = 4.0


def _training_step(
    model: RollingFlowSpeaker,
    batch,
    device: torch.device,
    amp_dtype: torch.dtype,
    noamp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = batch.to(device)
    mels_values = batch.mels
    if mels_values.ndim == 4 and mels_values.shape[1] == 1:
        mels_values = mels_values.squeeze(1)
    text = MaskedTensor(values=batch.tokens.unsqueeze(1), mask=batch.tokens_mask)
    mels = MaskedTensor(values=mels_values, mask=batch.mels_mask)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=not noamp):
        v_pred, v_target, v_mask, _t = model.training_step(text, mels)
        v_per_pos = (v_pred - v_target).pow(2).mean(-1)
        v_loss = (v_per_pos * v_mask).sum() / v_mask.sum().clamp(min=1)
        loss = v_loss

    return loss, v_loss.detach()


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    if args.num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if args.min_lr <= 0 or args.max_lr <= args.min_lr:
        raise ValueError("require 0 < min_lr < max_lr")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary.from_json(args.vocab_path)
    tokenizer = Tokenizer(vocab)
    source = ArrowTTSSource(args.arrow_path, tokenizer=tokenizer)
    print(f"Source size: {len(source)}")
    dataset = AudioDataset(tts_source=source, sample_rate=args.sample_rate)

    pin = device.type == "cuda"
    train_dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
        drop_last=True,
    )

    model = RollingFlowSpeaker(
        RollingFlowConfig(
            transformer_config=TransformerConfig(
                dim=args.dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
            ),
            vocabulary_size=len(vocab),
            mel_dim=args.mel_dim,
            n_denoising_steps=args.n_denoising_steps,
        )
    ).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.min_lr)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    noamp = device.type != "cuda"

    tb = TensorBoardLogger(log_dir=output_dir / "tb")
    lr_mult = (args.max_lr / args.min_lr) ** (1.0 / (args.num_steps - 1))

    lrs: list[float] = []
    losses: list[float] = []
    smoothed_losses: list[float] = []
    grad_norms: list[float] = []
    ema = 0.0
    best_smoothed = float("inf")
    diverged_at: float | None = None

    train_iter = iter(train_dl)
    try:
        for step in range(args.num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)

            lr = args.min_lr * (lr_mult**step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss, v_loss = _training_step(
                model, batch, device, amp_dtype, noamp
            )
            loss_val = float(loss.detach())

            if not math.isfinite(loss_val):
                diverged_at = lr
                print(f"non-finite loss at step {step}, lr={lr:.3e}; stopping")
                break

            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                )
            )
            optimizer.step()

            ema = args.smooth_beta * ema + (1.0 - args.smooth_beta) * loss_val
            smoothed = ema / (1.0 - args.smooth_beta ** (step + 1))
            best_smoothed = min(best_smoothed, smoothed)

            lrs.append(lr)
            losses.append(loss_val)
            smoothed_losses.append(smoothed)
            grad_norms.append(grad_norm)

            tb.log_metrics(
                {
                    "lr": lr,
                    "loss": loss_val,
                    "smoothed_loss": smoothed,
                    "grad_norm": grad_norm,
                    "v_loss": float(v_loss),
                },
                step,
                prefix="lr_find",
            )

            if step % 20 == 0:
                print(
                    f"step {step:4d}  lr={lr:.3e}  loss={loss_val:.4f}  "
                    f"smoothed={smoothed:.4f}  grad_norm={grad_norm:.3f}"
                )

            if smoothed > args.divergence_factor * best_smoothed:
                diverged_at = lr
                print(
                    f"diverged at step {step}, lr={lr:.3e}  "
                    f"(smoothed {smoothed:.4f} > {args.divergence_factor}x "
                    f"min {best_smoothed:.4f})"
                )
                break
    except KeyboardInterrupt:
        print("interrupted; plotting collected data")
    finally:
        tb.close()

    if not lrs:
        print("no steps completed; nothing to plot")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(lrs, smoothed_losses, color="C0", label="smoothed loss")
    ax1.set_xscale("log")
    ax1.set_xlabel("learning rate (log)")
    ax1.set_ylabel("smoothed loss", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lrs, grad_norms, color="C3", alpha=0.6, label="grad norm")
    ax2.set_yscale("log")
    ax2.set_ylabel("grad norm (log)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    min_idx = min(range(len(smoothed_losses)), key=smoothed_losses.__getitem__)
    min_lr_value = lrs[min_idx]
    ax1.axvline(
        min_lr_value,
        color="green",
        linestyle=":",
        label=f"min loss @ {min_lr_value:.2e}",
    )
    if diverged_at is not None:
        ax1.axvline(
            diverged_at,
            color="black",
            linestyle=":",
            label=f"diverged @ {diverged_at:.2e}",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    out_png = output_dir / "lr_find.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    print()
    print(f"LR @ min smoothed loss: {min_lr_value:.3e}")
    if diverged_at is not None:
        print(f"LR @ divergence:        {diverged_at:.3e}")
        print(f"Suggested max LR:       ~{diverged_at / 10:.3e}")
    else:
        print("(no divergence detected; consider raising max_lr)")
    print(f"Plot saved: {out_png}")


if __name__ == "__main__":
    main()

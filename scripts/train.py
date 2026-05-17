"""End-to-end training entrypoint for RollingFlowSpeaker."""

from dataclasses import dataclass
from pathlib import Path

import torch
from simple_parsing import ArgumentParser
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from tts.data.collate import collate
from tts.data.dataset import AudioDataset
from tts.data.source import ArrowTTSSource
from tts.data.text import Tokenizer, Vocabulary
from tts.model.neural_speaker import RollingFlowConfig, RollingFlowSpeaker
from tts.model.transformer import TransformerConfig
from tts.training.checkpoint_manager import CheckpointManager
from tts.training.console_logger import ConsoleLogger
from tts.training.loggers import Logger, MultiLogger
from tts.training.trainer import (
    AMPDtype,
    TrainerConfig,
    TrainerState,
    TTSRollingFlowMatchingTrainer,
)


@dataclass
class Args:
    # Data
    vocab_path: str = "/data/ljspeech/vocabulary.json"
    arrow_path: str = "data/ljspeech_24khz.arrow"
    sample_rate: int = 24000
    n_valid: int = 64
    n_smp: int = 4
    # Run
    output_dir: str = "outputs/run0"
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 6
    max_steps: int = 30_000
    valid_steps: int = 1_000
    smp_steps: int = 2_500
    checkpoint_steps: int = 5_000
    clip_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    lr: float = 1e-3
    # Model
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    mel_dim: int = 100
    n_denoising_steps: int = 32
    # Codec
    use_codec: bool = True
    # Logging
    use_tensorboard: bool = True
    # Perf
    compile: bool = True


def _make_loader(
    dataset, batch_size: int, num_workers: int, shuffle: bool, pin_memory: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary.from_json(args.vocab_path)
    tokenizer = Tokenizer(vocab)
    source = ArrowTTSSource(args.arrow_path, tokenizer=tokenizer)
    print(f"Source size: {len(source)}")

    full = AudioDataset(tts_source=source, sample_rate=args.sample_rate)
    N = len(full)
    assert args.n_valid + args.n_smp < N, "dataset too small for the requested splits"
    smp_ds = Subset(full, list(range(args.n_smp)))
    valid_ds = Subset(full, list(range(N - args.n_valid, N)))
    train_ds = Subset(full, list(range(args.n_smp, N - args.n_valid)))

    pin = device.type == "cuda"
    train_dl = _make_loader(train_ds, args.batch_size, args.num_workers, True, pin)
    valid_dl = _make_loader(valid_ds, args.batch_size, args.num_workers, False, pin)
    smp_dl = _make_loader(smp_ds, args.n_smp, 0, False, pin)

    codec = None
    mel_dim = args.mel_dim
    if args.use_codec:
        from tts.data.audio.codecs import BigVGAN

        codec = BigVGAN().to(device)
        assert codec.n_mels == mel_dim, (
            f"codec mel_dim {codec.n_mels} != configured mel_dim {mel_dim}"
        )

    model = RollingFlowSpeaker(
        RollingFlowConfig(
            transformer_config=TransformerConfig(
                dim=args.dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
            ),
            vocabulary_size=len(vocab),
            mel_dim=mel_dim,
            n_denoising_steps=args.n_denoising_steps,
        )
    ).to(device)
    if args.compile:
        # Disable inductor's split_reductions pass — its mix_order_reduction
        # codegen can't factor expressions like s13*(s23 + s79) and crashes
        # with CantSplit on AdaLN's backward at dynamic shapes.
        import torch._inductor.config as inductor_config

        inductor_config.split_reductions = False
        model.forward = torch.compile(model.forward, dynamic=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    sub_loggers: list[Logger] = [
        ConsoleLogger(total=args.max_steps, audio_dir=output_dir / "audio")
    ]
    if args.use_tensorboard:
        from tts.training.tensorboard_logger import TensorBoardLogger

        sub_loggers.append(TensorBoardLogger(log_dir=output_dir / "tb"))
    logger: Logger = (
        sub_loggers[0] if len(sub_loggers) == 1 else MultiLogger(*sub_loggers)
    )

    checkpoint_manager = CheckpointManager(exp_path=output_dir / "checkpoints")

    trainer = TTSRollingFlowMatchingTrainer(
        config=TrainerConfig(
            clip_grad_norm=args.clip_grad_norm,
            device=device,
            amp_dtype=AMPDtype.BF16 if device.type == "cuda" else AMPDtype.FP32,
            smp_steps=args.smp_steps,
            valid_steps=args.valid_steps,
            checkpoint_steps=args.checkpoint_steps,
            max_steps=args.max_steps,
            noamp=device.type != "cuda",
            n_smp=args.n_smp,
            grad_accum_steps=args.grad_accum_steps,
        ),
        codec=codec,
        model=model,
        optimizer=optimizer,
        scaler=None,
        logger=logger,
        train_dloader=train_dl,
        valid_dloader=valid_dl,
        smp_dloader=smp_dl,
        state=TrainerState(step=0),
        checkpoint_manager=checkpoint_manager,
    )

    trainer.train()


if __name__ == "__main__":
    main()

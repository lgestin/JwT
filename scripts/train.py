"""End-to-end training entrypoint for RollingFlowSpeaker."""

from dataclasses import dataclass, field
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
from tts.training.checkpoint_manager import CheckpointManager
from tts.training.console_logger import ConsoleLogger
from tts.training.loggers import Logger, MultiLogger
from tts.training.trainer import (
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
    # Run
    output_dir: str = "outputs/run0"
    batch_size: int = 64
    num_workers: int = 6
    lr: float = 1e-3
    # Codec
    use_codec: bool = True
    # Logging
    use_tensorboard: bool = True
    # Perf
    compile: bool = True
    # Model
    model: RollingFlowConfig = field(default_factory=RollingFlowConfig)
    # Trainer
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    device = torch.device(args.trainer.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary.from_json(args.vocab_path)
    tokenizer = Tokenizer(vocab)
    source = ArrowTTSSource(args.arrow_path, tokenizer=tokenizer)
    print(f"Source size: {len(source)}")

    full = AudioDataset(tts_source=source, sample_rate=args.sample_rate)
    N = len(full)
    n_smp = args.trainer.n_smp
    if args.n_valid + n_smp >= N:
        raise ValueError("dataset too small for the requested splits")
    smp_ds = Subset(full, list(range(n_smp)))
    valid_ds = Subset(full, list(range(N - args.n_valid, N)))
    train_ds = Subset(full, list(range(n_smp, N - args.n_valid)))

    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
    )
    smp_dl = DataLoader(
        smp_ds,
        batch_size=n_smp,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=pin,
    )

    codec = None
    if args.use_codec:
        from tts.data.audio.codecs import BigVGAN

        codec = BigVGAN().to(device)
        if codec.n_mels != args.model.mel_dim:
            raise ValueError(
                f"codec mel_dim {codec.n_mels} != configured mel_dim {args.model.mel_dim}"
            )

    args.model.vocabulary_size = len(vocab)
    model = RollingFlowSpeaker(args.model).to(device)
    if args.compile:
        # Disable inductor's split_reductions pass — its mix_order_reduction
        # codegen can't factor expressions like s13*(s23 + s79) and crashes
        # with CantSplit on AdaLN's backward at dynamic shapes.
        import torch._inductor.config as inductor_config

        inductor_config.split_reductions = False
        model.forward = torch.compile(model.forward, dynamic=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    sub_loggers: list[Logger] = [
        ConsoleLogger(total=args.trainer.max_steps, audio_dir=output_dir / "audio")
    ]
    if args.use_tensorboard:
        from tts.training.tensorboard_logger import TensorBoardLogger

        sub_loggers.append(TensorBoardLogger(log_dir=output_dir / "tb"))
    logger: Logger = MultiLogger(*sub_loggers)

    checkpoint_manager = CheckpointManager(exp_path=output_dir / "checkpoints")

    trainer = TTSRollingFlowMatchingTrainer(
        config=args.trainer,
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

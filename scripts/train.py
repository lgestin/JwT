"""End-to-end training entrypoint for RollingFlowSpeaker."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader, Subset

from tts.data.audio.codecs import Codecs
from tts.data.collate import collate
from tts.data.dataset import AudioDataset
from tts.data.source import ArrowTTSSource
from tts.data.text import Tokenizer, Vocabulary
from tts.model.flow import FlowParametrizations
from tts.model.neural_speaker import RollingFlowConfig, RollingFlowSpeaker
from tts.training.checkpoint_manager import CheckpointManager
from tts.training.console_logger import ConsoleLogger
from tts.training.ema import EMA, EMAConfig
from tts.training.loggers import Logger, MultiLogger
from tts.training.optimizer import OptimizerConfig, build_optimizer
from tts.training.trainer import (
    TrainerConfig,
    TrainerState,
    TTSRollingFlowMatchingTrainer,
)


@dataclass
class Args:
    # Data
    vocab_path: str = "/data/ljspeech/vocabulary.json"
    arrow_path: str = "data/ljspeech_24khz_bigvgan.arrow"
    n_valid: int = 64
    # Run
    output_dir: str = "outputs/run0"
    resume: bool = False  # load the latest checkpoint from output_dir/checkpoints
    batch_size: int = 64
    num_workers: int = 6
    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # EMA (exponential moving average of weights)
    ema: EMAConfig = field(default_factory=EMAConfig)
    # Codec
    codec: Codecs = Codecs.BIGVGAN
    # Flow-matching parametrization (locked to the model checkpoint).
    parametrization: FlowParametrizations = FlowParametrizations.RECTIFIED_FLOW
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
    codec_name = str(args.codec).lower()
    source = ArrowTTSSource(args.arrow_path, tokenizer=tokenizer, codec_name=codec_name)
    print(f"Source size: {len(source)}")

    codec = args.codec.codec.to(device)

    full = AudioDataset(tts_source=source, sample_rate=codec.sample_rate)
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

    args.model.vocabulary_size = len(vocab)
    args.model.codec = args.codec
    args.model.parametrization = args.parametrization
    args.model.acoustic_dim = codec.acoustic_dim
    model = RollingFlowSpeaker(args.model).to(device)
    if args.compile:
        # Disable inductor's split_reductions pass — its mix_order_reduction
        # codegen can't factor expressions like s13*(s23 + s79) and crashes
        # with CantSplit on AdaLN's backward at dynamic shapes.
        import torch._inductor.config as inductor_config

        inductor_config.split_reductions = False
        model.forward = torch.compile(model.forward, dynamic=True)

    optimizer = build_optimizer(model, args.optimizer)

    ema = EMA(model, decay=args.ema.decay) if args.ema.enabled else None

    sub_loggers: list[Logger] = [
        ConsoleLogger(total=args.trainer.max_steps, audio_dir=output_dir / "audio")
    ]
    if args.use_tensorboard:
        from tts.training.tensorboard_logger import TensorBoardLogger

        sub_loggers.append(TensorBoardLogger(log_dir=output_dir / "tb"))
    logger: Logger = MultiLogger(*sub_loggers)

    checkpoint_manager = CheckpointManager(exp_path=output_dir / "checkpoints")

    state = TrainerState(step=0)
    if args.resume:
        meta = checkpoint_manager.load_latest(
            model, optimizer, ema=ema, map_location=device
        )
        state = TrainerState(step=meta["step"], best_loss=meta["best_loss"])
        print(f"Resumed from step {state.step} (best_loss={meta['best_loss']})")

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
        state=state,
        checkpoint_manager=checkpoint_manager,
        ema=ema,
    )

    trainer.train()


if __name__ == "__main__":
    main()

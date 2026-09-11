"""End-to-end training entrypoint for RollingFlowSpeaker."""

import warnings
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from jwt.data.audio.codecs import RawAudioPatcher
from jwt.data.collate import collate
from jwt.data.dataset import AudioDataset
from jwt.data.source import ArrowTTSSource, check_same_sample_rate
from jwt.data.splits import split_indices
from jwt.data.text import Tokenizer, Vocabulary
from jwt.model.neural_speaker import RollingFlowSpeaker
from jwt.training.checkpoint_manager import CheckpointManager
from jwt.training.config import (
    Args,
    LoggerBackend,
    check_model_config_consistency,
    dump_config,
    parse_args,
)
from jwt.training.console_logger import ConsoleLogger
from jwt.training.ema import EMA
from jwt.training.loggers import Logger, MultiLogger
from jwt.training.trainer import TrainerState, TTSRollingFlowMatchingTrainer


def main() -> None:
    args: Args = parse_args()

    device = torch.device(args.trainer.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(args, output_dir / "config.yaml")

    vocab = Vocabulary.from_json(args.vocab_path)
    tokenizer = Tokenizer(vocab)
    if not args.datasets:
        raise ValueError("config needs at least one entry under `datasets:`")
    codec = args.codec.codec.to(device)
    if not isinstance(codec, RawAudioPatcher):
        raise ValueError(f"prepared data is raw audio; got codec {args.codec}")
    paths = [d.path for d in args.datasets]
    sources = [ArrowTTSSource(p, tokenizer, codec.patch_size) for p in paths]
    check_same_sample_rate(sources, paths)
    sample_rate = sources[0].sample_rate
    print(f"Sample rate: {sample_rate} Hz")

    train_parts, valid_parts, unseen_parts, weights = [], [], [], []
    for d, source in zip(args.datasets, sources, strict=True):
        train_idx, valid_idx, unseen_idx = split_indices(
            source.speakers, source.utt_ids, d.n_valid, d.n_valid_speakers
        )
        if args.n_train is not None:
            train_idx = train_idx[: args.n_train]
        prompt = args.audio_prompt
        train_parts.append(AudioDataset(source, sample_rate, train_idx, prompt))
        valid_parts.append(AudioDataset(source, sample_rate, valid_idx, prompt, 0))
        unseen_parts.append(AudioDataset(source, sample_rate, unseen_idx, prompt, 0))
        weights += [d.weight / len(train_idx)] * len(train_idx)
        print(
            f"{d.path}: train={len(train_idx)} valid={len(valid_idx)} "
            f"valid_unseen={len(unseen_idx)}"
        )
    train_ds = ConcatDataset(train_parts)
    valid_ds = ConcatDataset(valid_parts)
    unseen_ds = ConcatDataset(unseen_parts)
    n_smp = args.trainer.n_smp
    if len(valid_ds) < n_smp:
        raise ValueError(f"need n_smp={n_smp} valid items, got {len(valid_ds)}")
    smp_ds = Subset(valid_ds, list(range(n_smp)))

    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, len(train_ds), replacement=True),
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
    )
    valid_unseen_dl = None
    if len(unseen_ds):
        valid_unseen_dl = DataLoader(
            unseen_ds,
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
    args.model.acoustic_dim = codec.acoustic_dim
    model = RollingFlowSpeaker(args.model).to(device)
    print(f"Attention: {args.trainer.attention_implementation.name}")
    if args.compile:
        # Disable inductor's split_reductions pass — its mix_order_reduction
        # codegen can't factor expressions like s13*(s23 + s79) and crashes
        # with CantSplit on AdaLN's backward at dynamic shapes.
        import torch._inductor.config as inductor_config

        inductor_config.split_reductions = False
        # Let dynamo trace through `.item()` calls (e.g. `max_seqlen` for
        # FlexAttention) symbolically rather than graph-breaking.
        torch._dynamo.config.capture_scalar_outputs = True
        model.forward = torch.compile(model.forward, dynamic=True)

    optimizer = AdamW(
        model.parameters(),
        lr=args.optimizer.lr,
        betas=args.optimizer.betas,
        weight_decay=args.optimizer.weight_decay,
    )

    ema = EMA(model, decay=args.ema.decay) if args.ema.enabled else None

    sub_loggers: list[Logger] = [
        ConsoleLogger(total=args.trainer.max_steps, audio_dir=output_dir / "audio")
    ]
    if args.logger is LoggerBackend.WANDB:
        from jwt.training.wandb_logger import WandbLogger

        sub_loggers.append(
            WandbLogger(
                log_dir=output_dir,
                run_name=output_dir.name,
                project=args.wandb.project,
                entity=args.wandb.entity,
                group=args.wandb.group,
                tags=args.wandb.tags or None,
            )
        )
    elif args.logger is LoggerBackend.TENSORBOARD:
        from jwt.training.tensorboard_logger import TensorBoardLogger

        sub_loggers.append(TensorBoardLogger(log_dir=output_dir / "tb"))
    logger: Logger = MultiLogger(*sub_loggers)
    logger.log_config(args)

    checkpoint_manager = CheckpointManager(exp_path=output_dir / "checkpoints")

    state = TrainerState(step=0)
    if args.resume:
        meta = checkpoint_manager.load_latest(
            model, optimizer, ema=ema, map_location=device
        )
        if "config" in meta:
            check_model_config_consistency(args.model, meta["config"])
        else:
            warnings.warn(
                "checkpoint has no stored model config; skipping the consistency check",
                stacklevel=2,
            )
        state = TrainerState(step=meta["step"], best_loss=meta["best_loss"])
        print(f"Resumed from step {state.step} (best_loss={meta['best_loss']})")

    trainer = TTSRollingFlowMatchingTrainer(
        config=args.trainer,
        codec=codec,
        sample_rate=sample_rate,
        model=model,
        optimizer=optimizer,
        scaler=None,
        logger=logger,
        train_dloader=train_dl,
        valid_dloader=valid_dl,
        smp_dloader=smp_dl,
        state=state,
        checkpoint_manager=checkpoint_manager,
        valid_unseen_dloader=valid_unseen_dl,
        ema=ema,
    )

    try:
        trainer.train()
    finally:
        logger.close()


if __name__ == "__main__":
    main()

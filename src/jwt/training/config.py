"""Training run config: the `Args` dataclass, file-backed parsing, and the
checkpoint-consistency guard.

Training is launched from a YAML config file given by a required `--config_path`
argument, with individual CLI flags overriding any value. The resolved config a
run used is dumped to `output_dir/config.yaml`; to resume a run, point
`--config_path` at that saved file.
"""

import argparse
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path

from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import load, save

from jwt.data.audio.codecs import Codecs
from jwt.data.audio_prompt import AudioPromptConfig
from jwt.model.neural_speaker import RollingFlowConfig
from jwt.training.ema import EMAConfig
from jwt.training.optimizer import OptimizerConfig
from jwt.training.trainer import TrainerConfig


class LoggerBackend(Enum):
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"


@dataclass
class WandbConfig:
    project: str = "JwT"
    entity: str | None = None
    # Defaults to the config file's stem (e.g. "proxy") in `parse_args`.
    group: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    path: str  # prepared dataset directory (scripts/data/create_arrow.py)
    weight: float = 1.0  # relative sampling weight; weight ∝ hours = concatenation
    n_valid: int = 64  # held-out utterances of training speakers
    n_valid_speakers: int = 0  # whole speakers held out (valid_unseen)


@dataclass
class Args:
    # Data — `datasets` is file-only: simple_parsing can't put a list of
    # dataclasses on the CLI, so parse_args copies it from the config file.
    vocab_path: str = "data/vocabulary.json"
    datasets: list[DatasetConfig] = field(
        default_factory=lambda: [
            DatasetConfig(path="data/prepared/ljspeech_22.050khz")
        ],
        metadata={"cmd": False},
    )
    audio_prompt: AudioPromptConfig | None = None
    n_train: int | None = None  # caps each dataset's training split
    # Run
    output_dir: str = "outputs/run0"
    resume: bool = False  # load the latest checkpoint from output_dir/checkpoints
    batch_size: int = 64
    num_workers: int = 6
    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # EMA (exponential moving average of weights)
    ema: EMAConfig = field(default_factory=EMAConfig)
    # Codec — prepared data is raw waveform, so this must be a RAWAUDIO_* codec
    # (its patch size reshapes the waveform); also copied into `model.codec`.
    codec: Codecs = Codecs.RAWAUDIO_512
    # Logging — configs saved before the LoggerBackend switch need their
    # `use_tensorboard: true` line replaced with `logger: TENSORBOARD`.
    logger: LoggerBackend = LoggerBackend.WANDB
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # Perf
    compile: bool = True
    # Model
    model: RollingFlowConfig = field(default_factory=RollingFlowConfig)
    # Trainer
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self) -> None:
        if self.codec == Codecs.BIGVGAN and self.trainer.aux_mel_weight > 0:
            raise ValueError(
                "aux_mel_weight must be 0 for the BigVGAN codec: its "
                "flow-matching loss is already computed in mel space, so the "
                "auxiliary mel loss is redundant and not supported "
                f"(got aux_mel_weight={self.trainer.aux_mel_weight})"
            )


def dump_config(args: Args, path: Path | str) -> None:
    """Serialize a resolved `Args` to a YAML file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(args, path)


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse training `Args` from the YAML file given by a required
    `--config_path`, with CLI flags overriding individual values.

    Precedence, lowest to highest: `Args` defaults < config file < CLI flags.
    There is no default config file. To resume a run, point `--config_path`
    at that run's saved `<output_dir>/config.yaml`.
    """
    # Pre-parse just `--config_path` so the file can be loaded before its
    # values seed the real parser's defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_path")
    known, _ = pre.parse_known_args(argv)

    parser = ArgumentParser(add_config_path_arg=True)
    if known.config_path is None:
        # No config given. Build a bare parser so `-h/--help` still prints,
        # then fail with a clear message if help was not what was asked.
        parser.add_arguments(Args, dest="args")
        parser.parse_args(argv)  # exits here if -h/--help was passed
        parser.error("--config_path is required")

    defaults = load(Args, known.config_path)
    parser.add_arguments(Args, dest="args", default=defaults)
    args = parser.parse_args(argv).args
    args.datasets = defaults.datasets  # file-only field, see Args.datasets
    if args.wandb.group is None:
        args.wandb.group = Path(known.config_path).stem
    return args


def check_model_config_consistency(
    runtime: RollingFlowConfig, checkpoint: RollingFlowConfig
) -> None:
    """Raise if the run's model config differs from the checkpoint's.

    `RollingFlowConfig` carries `codec` and `parametrization`, so this single
    comparison guards every checkpoint-locked architecture field. A mismatch
    would corrupt a resumed run (or crash `load_state_dict` with a far less
    actionable error), so it is a hard failure.
    """
    diffs = [
        f"  {f.name}: checkpoint={getattr(checkpoint, f.name)!r} "
        f"run={getattr(runtime, f.name)!r}"
        for f in fields(RollingFlowConfig)
        if getattr(checkpoint, f.name) != getattr(runtime, f.name)
    ]
    if diffs:
        raise ValueError(
            "resumed run's model config does not match the checkpoint:\n"
            + "\n".join(diffs)
        )

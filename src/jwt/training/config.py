"""Training run config: the `Args` dataclass, file-backed parsing, and the
checkpoint-consistency guard.

Training is launched from a YAML config file given by a required `--config_path`
argument, with individual CLI flags overriding any value. The resolved config a
run used is dumped to `output_dir/config.yaml`; to resume a run, point
`--config_path` at that saved file.
"""

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path

from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import load, save

from jwt.data.audio.codecs import Codecs
from jwt.model.neural_speaker import RollingFlowConfig
from jwt.training.ema import EMAConfig
from jwt.training.optimizer import OptimizerConfig
from jwt.training.trainer import TrainerConfig


@dataclass
class Args:
    # Data
    vocab_path: str = "data/vocabulary.json"
    arrow_path: str = "data/ljspeech_24khz_bigvgan.arrow"
    n_valid: int = 64
    n_train: int | None = None
    # Run
    output_dir: str = "outputs/run0"
    resume: bool = False  # load the latest checkpoint from output_dir/checkpoints
    batch_size: int = 64
    num_workers: int = 6
    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # EMA (exponential moving average of weights)
    ema: EMAConfig = field(default_factory=EMAConfig)
    # Codec — selects the arrow acoustic column and the codec object; also
    # copied into `model.codec`.
    codec: Codecs = Codecs.BIGVGAN
    # Logging
    use_tensorboard: bool = True
    # Perf
    compile: bool = True
    # Model
    model: RollingFlowConfig = field(default_factory=RollingFlowConfig)
    # Trainer
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self) -> None:
        if self.codec == Codecs.BIGVGAN and self.trainer.aux_stft_weight > 0:
            raise ValueError(
                "aux_stft_weight must be 0 for the BigVGAN codec: its "
                "flow-matching loss is already perceptual (mel-space), so the "
                "auxiliary STFT loss adds little and is not supported "
                f"(got aux_stft_weight={self.trainer.aux_stft_weight})"
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

    parser.add_arguments(Args, dest="args", default=load(Args, known.config_path))
    return parser.parse_args(argv).args


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
        f"  {f.name}: checkpoint={getattr(checkpoint, f.name)!r} run={getattr(runtime, f.name)!r}"
        for f in fields(RollingFlowConfig)
        if getattr(checkpoint, f.name) != getattr(runtime, f.name)
    ]
    if diffs:
        raise ValueError(
            "resumed run's model config does not match the checkpoint:\n" + "\n".join(diffs)
        )

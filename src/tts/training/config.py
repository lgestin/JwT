"""Training run config: the `Args` dataclass, file-backed parsing, and the
checkpoint-consistency guard.

Training is launched from a YAML config file (default `configs/default.yaml`),
with individual CLI flags overriding any value. The resolved config a run used
is dumped to `output_dir/config.yaml` so the run can be resumed faithfully.
"""

import argparse
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path

from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import load, save

from tts.data.audio.codecs import Codecs
from tts.model.flow import FlowParametrizations
from tts.model.neural_speaker import RollingFlowConfig
from tts.training.ema import EMAConfig
from tts.training.optimizer import OptimizerConfig
from tts.training.trainer import TrainerConfig

DEFAULT_CONFIG_PATH = "configs/default.yaml"


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


def dump_config(args: Args, path: Path | str) -> None:
    """Serialize a resolved `Args` to a YAML file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(args, path)


def _select_config_path(argv: list[str] | None) -> str:
    """Pick which config file to load from a lightweight pre-parse of the CLI.

    Defaults to `configs/default.yaml`, or `--config_path` if given. When
    `--resume` is set and `output_dir/config.yaml` exists, that saved config
    is used instead so a resumed run reuses its original config.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_path", default=DEFAULT_CONFIG_PATH)
    pre.add_argument("--resume", action="store_true")
    pre.add_argument("--output_dir", default=None)
    known, _ = pre.parse_known_args(argv)

    config_path = known.config_path
    if known.resume:
        output_dir = known.output_dir
        if output_dir is None:
            output_dir = load(Args, config_path).output_dir
        saved = Path(output_dir) / "config.yaml"
        if saved.exists():
            config_path = str(saved)
        else:
            warnings.warn(
                f"--resume set but no saved config at {saved}; "
                f"falling back to {config_path}",
                stacklevel=2,
            )
    return config_path


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse training `Args` from a config file with CLI overrides.

    Precedence, lowest to highest: `Args` defaults < config file < CLI flags.
    On `--resume`, the config saved next to the checkpoints is preferred over
    `configs/default.yaml`.
    """
    config_path = _select_config_path(argv)
    defaults = load(Args, config_path)

    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(Args, dest="args", default=defaults)
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

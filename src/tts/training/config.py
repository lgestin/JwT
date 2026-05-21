"""Training run config: the `Args` dataclass, file-backed parsing, and the
checkpoint-consistency guard.

Training is launched from a YAML config file (default `configs/default.yaml`),
with individual CLI flags overriding any value. The resolved config a run used
is dumped to `output_dir/config.yaml` so the run can be resumed faithfully.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

from simple_parsing.helpers.serialization import save

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

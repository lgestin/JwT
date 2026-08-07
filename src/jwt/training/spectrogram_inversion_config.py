"""Run config for spectrogram-inversion training: the `SpectrogramInversionArgs`
dataclass, file-backed parsing, and the checkpoint-consistency guard.

Mirrors `jwt.training.config` (the TTS run config) with the model swapped to
`SpectrogramInverterConfig` and the text/vocab machinery removed. Reuse
`jwt.training.config.dump_config` for serialization — it is model-agnostic.
"""

import argparse
from dataclasses import dataclass, field, fields

from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization import load

from jwt.model.spectrogram_inverter import SpectrogramInverterConfig
from jwt.training.ema import EMAConfig
from jwt.training.optimizer import OptimizerConfig
from jwt.training.trainer import TrainerConfig


@dataclass
class SpectrogramInversionArgs:
    # Data — any arrow with a waveform_i16 column works; the per-codec acoustic
    # columns are ignored (patching happens at load time).
    arrow_path: str = "data/ljspeech_24khz_bigvgan.arrow"
    # Window length in patches: window = n_frames * patch_size samples. Clips
    # shorter than the window are excluded from every split.
    n_frames: int = 256
    n_valid: int = 64
    n_train: int | None = None
    # Run
    output_dir: str = "outputs/melinv0"
    resume: bool = False  # load the latest checkpoint from output_dir/checkpoints
    batch_size: int = 32
    num_workers: int = 6
    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # EMA (exponential moving average of weights)
    ema: EMAConfig = field(default_factory=EMAConfig)
    # Logging
    use_tensorboard: bool = True
    # Perf
    compile: bool = True
    # Model — carries the codec (must be a RawAudio* variant) and mel geometry.
    model: SpectrogramInverterConfig = field(default_factory=SpectrogramInverterConfig)
    # Trainer
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def parse_args(argv: list[str] | None = None) -> SpectrogramInversionArgs:
    """Parse `SpectrogramInversionArgs` from the YAML file given by a required
    `--config_path`, with CLI flags overriding individual values.

    Precedence, lowest to highest: defaults < config file < CLI flags.
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
        parser.add_arguments(SpectrogramInversionArgs, dest="args")
        parser.parse_args(argv)  # exits here if -h/--help was passed
        parser.error("--config_path is required")

    parser.add_arguments(
        SpectrogramInversionArgs,
        dest="args",
        default=load(SpectrogramInversionArgs, known.config_path),
    )
    return parser.parse_args(argv).args


def check_model_config_consistency(
    runtime: SpectrogramInverterConfig, checkpoint: SpectrogramInverterConfig
) -> None:
    """Raise if the run's model config differs from the checkpoint's.

    `SpectrogramInverterConfig` carries the codec, parametrization and mel
    geometry, so this single comparison guards every checkpoint-locked
    architecture field. A mismatch would corrupt a resumed run, so it is a
    hard failure.
    """
    diffs = [
        f"  {f.name}: checkpoint={getattr(checkpoint, f.name)!r} "
        f"run={getattr(runtime, f.name)!r}"
        for f in fields(SpectrogramInverterConfig)
        if getattr(checkpoint, f.name) != getattr(runtime, f.name)
    ]
    if diffs:
        raise ValueError(
            "resumed run's model config does not match the checkpoint:\n"
            + "\n".join(diffs)
        )

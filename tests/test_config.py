from pathlib import Path

import pytest
from simple_parsing.helpers.serialization import load

from tts.data.audio.codecs import Codecs
from tts.model.flow import FlowParametrizations
from tts.model.neural_speaker import RollingFlowConfig
from tts.training.config import Args, check_model_config_consistency, dump_config


def test_args_defaults_survive_the_move() -> None:
    """Args keeps its defaults, including nested configs, after moving modules."""
    args = Args()
    assert args.batch_size == 64
    assert args.codec == Codecs.BIGVGAN
    assert args.parametrization == FlowParametrizations.RECTIFIED_FLOW
    assert args.trainer.max_steps == 200_001
    assert args.optimizer.lr == 1e-3
    assert args.ema.enabled is True


def test_dump_config_round_trips(tmp_path: Path) -> None:
    """A dumped Args reloads identically, including nested and enum fields."""
    args = Args(batch_size=13)
    args.trainer.max_steps = 999
    args.optimizer.lr = 5e-4
    path = tmp_path / "config.yaml"

    dump_config(args, path)
    reloaded = load(Args, path)

    assert reloaded == args


def test_dump_config_creates_parent_dirs(tmp_path: Path) -> None:
    """dump_config creates missing parent directories."""
    path = tmp_path / "nested" / "dir" / "config.yaml"
    dump_config(Args(), path)
    assert path.exists()


def test_consistency_passes_for_equal_configs() -> None:
    """Identical model configs pass the check without raising."""
    check_model_config_consistency(RollingFlowConfig(), RollingFlowConfig())


def test_consistency_raises_on_mismatch() -> None:
    """A differing checkpoint-locked field raises ValueError naming the field."""
    runtime = RollingFlowConfig(acoustic_dim=100)
    checkpoint = RollingFlowConfig(acoustic_dim=80)
    with pytest.raises(ValueError, match="acoustic_dim"):
        check_model_config_consistency(runtime, checkpoint)

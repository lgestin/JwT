from pathlib import Path

import pytest
from simple_parsing.helpers.serialization import load

from tts.data.audio.codecs import Codecs
from tts.model.flow import FlowParametrizations
from tts.model.neural_speaker import RollingFlowConfig
from tts.training.config import (
    Args,
    check_model_config_consistency,
    dump_config,
    parse_args,
)


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


def test_parse_args_loads_values_from_config_file(tmp_path: Path) -> None:
    """parse_args reads field values from the --config_path file."""
    cfg = tmp_path / "default.yaml"
    args = Args(batch_size=7)
    args.trainer.max_steps = 321
    dump_config(args, cfg)

    parsed = parse_args(["--config_path", str(cfg)])

    assert parsed.batch_size == 7
    assert parsed.trainer.max_steps == 321


def test_cli_flag_overrides_config_file(tmp_path: Path) -> None:
    """An explicit CLI flag wins over the config file value."""
    cfg = tmp_path / "default.yaml"
    dump_config(Args(batch_size=7), cfg)

    parsed = parse_args(["--config_path", str(cfg), "--batch_size", "99"])

    assert parsed.batch_size == 99


def test_resume_prefers_saved_config(tmp_path: Path) -> None:
    """With --resume, the saved output_dir/config.yaml wins over --config_path."""
    default_cfg = tmp_path / "default.yaml"
    dump_config(Args(batch_size=7), default_cfg)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    dump_config(Args(batch_size=555), output_dir / "config.yaml")

    parsed = parse_args(
        [
            "--config_path", str(default_cfg),
            "--resume",
            "--output_dir", str(output_dir),
        ]
    )

    assert parsed.batch_size == 555
    assert parsed.resume is True


def test_resume_without_saved_config_warns(tmp_path: Path) -> None:
    """--resume with no saved config.yaml falls back to --config_path and warns."""
    default_cfg = tmp_path / "default.yaml"
    dump_config(Args(batch_size=7), default_cfg)

    output_dir = tmp_path / "run"
    output_dir.mkdir()  # no config.yaml inside

    with pytest.warns(UserWarning, match="no saved config"):
        parsed = parse_args(
            [
                "--config_path", str(default_cfg),
                "--resume",
                "--output_dir", str(output_dir),
            ]
        )

    assert parsed.batch_size == 7


def test_default_config_round_trips() -> None:
    """configs/default.yaml deserializes to the Args() defaults."""
    repo_root = Path(__file__).parent.parent
    default_yaml = repo_root / "configs" / "default.yaml"
    assert default_yaml.exists()
    assert load(Args, default_yaml) == Args()

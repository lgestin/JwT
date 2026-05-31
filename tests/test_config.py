from pathlib import Path

import pytest
from simple_parsing.helpers.serialization import load

from jwt.data.audio.codecs import Codecs
from jwt.model.flow import FlowParametrizations
from jwt.model.neural_speaker import RollingFlowConfig
from jwt.training.config import (
    Args,
    check_model_config_consistency,
    dump_config,
    parse_args,
)
from jwt.training.trainer import TrainerConfig


def test_args_defaults_survive_the_move() -> None:
    """Args keeps its defaults, including nested configs, after moving modules."""
    args = Args()
    assert args.batch_size == 64
    assert args.codec == Codecs.BIGVGAN
    assert args.model.parametrization == FlowParametrizations.JWT
    assert args.trainer.max_steps == 200_001
    assert args.optimizer.lr == 1e-3
    assert args.ema.enabled is True


def test_n_train_defaults_to_none() -> None:
    """n_train defaults to None — runs use the full training split uncapped."""
    assert Args().n_train is None


def test_n_train_round_trips(tmp_path: Path) -> None:
    """A capped n_train survives a dump/load round trip."""
    cfg = tmp_path / "run.yaml"
    dump_config(Args(n_train=1024), cfg)
    assert load(Args, cfg).n_train == 1024


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
    cfg = tmp_path / "run.yaml"
    args = Args(batch_size=7)
    args.trainer.max_steps = 321
    dump_config(args, cfg)

    parsed = parse_args(["--config_path", str(cfg)])

    assert parsed.batch_size == 7
    assert parsed.trainer.max_steps == 321


def test_cli_flag_overrides_config_file(tmp_path: Path) -> None:
    """An explicit CLI flag wins over the config file value."""
    cfg = tmp_path / "run.yaml"
    dump_config(Args(batch_size=7), cfg)

    parsed = parse_args(["--config_path", str(cfg), "--batch_size", "99"])

    assert parsed.batch_size == 99


def test_parse_args_requires_config_path() -> None:
    """parse_args exits when --config_path is not given."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_resume_flag_is_parsed(tmp_path: Path) -> None:
    """--resume is a plain flag; resuming means pointing --config_path at the
    saved config."""
    saved = tmp_path / "run" / "config.yaml"
    dump_config(Args(batch_size=42), saved)

    parsed = parse_args(["--config_path", str(saved), "--resume"])

    assert parsed.resume is True
    assert parsed.batch_size == 42


def test_example_config_round_trips() -> None:
    """configs/example.yaml deserializes to the Args() defaults."""
    repo_root = Path(__file__).parent.parent
    example_yaml = repo_root / "configs" / "example.yaml"
    assert example_yaml.exists()
    assert load(Args, example_yaml) == Args()


def test_bigvgan_rejects_nonzero_aux_stft_weight() -> None:
    """BigVGAN's flow-matching loss is already in mel space, so the auxiliary
    STFT loss is redundant and not allowed for it."""
    with pytest.raises(ValueError, match="aux_stft_weight"):
        Args(codec=Codecs.BIGVGAN, trainer=TrainerConfig(aux_stft_weight=0.1))


def test_bigvgan_allows_zero_aux_stft_weight() -> None:
    Args(codec=Codecs.BIGVGAN, trainer=TrainerConfig(aux_stft_weight=0.0))


def test_rawaudio_allows_nonzero_aux_stft_weight() -> None:
    Args(codec=Codecs.RAWAUDIO_256, trainer=TrainerConfig(aux_stft_weight=0.1))


def test_bigvgan_rejects_nonzero_aux_mel_weight() -> None:
    """BigVGAN's flow-matching loss is already in mel space, so the auxiliary
    mel loss is redundant and not allowed for it."""
    with pytest.raises(ValueError, match="aux_mel_weight"):
        Args(codec=Codecs.BIGVGAN, trainer=TrainerConfig(aux_mel_weight=0.1))


def test_bigvgan_allows_zero_aux_mel_weight() -> None:
    Args(codec=Codecs.BIGVGAN, trainer=TrainerConfig(aux_mel_weight=0.0))


def test_rawaudio_allows_nonzero_aux_mel_weight() -> None:
    Args(codec=Codecs.RAWAUDIO_256, trainer=TrainerConfig(aux_mel_weight=0.1))


def test_rawaudio_allows_both_aux_weights_simultaneously() -> None:
    """Both auxiliary losses are independent; either or both can be active."""
    Args(
        codec=Codecs.RAWAUDIO_256,
        trainer=TrainerConfig(aux_stft_weight=0.75, aux_mel_weight=0.1),
    )

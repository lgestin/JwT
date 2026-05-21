from tts.data.audio.codecs import Codecs
from tts.model.flow import FlowParametrizations
from tts.training.config import Args


def test_args_defaults_survive_the_move() -> None:
    """Args keeps its defaults, including nested configs, after moving modules."""
    args = Args()
    assert args.batch_size == 64
    assert args.codec == Codecs.BIGVGAN
    assert args.parametrization == FlowParametrizations.RECTIFIED_FLOW
    assert args.trainer.max_steps == 200_001
    assert args.optimizer.lr == 1e-3
    assert args.ema.enabled is True

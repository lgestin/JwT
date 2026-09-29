from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from jwt.data.audio import AudioFile
from jwt.data.prepared import PreparedMeta, ShardWriter, build_schema, write_meta

TEST_SEED = 42
ASSETS_FOLDER = Path(__file__).parent / "assets"
AUDIO_TEST_FILES = [fpath.as_posix() for fpath in sorted(ASSETS_FOLDER.glob("*.wav"))]


@pytest.fixture(params=AUDIO_TEST_FILES)
def audio_from_file(request: pytest.FixtureRequest) -> AudioFile:
    return AudioFile(request.param)


@pytest.fixture(
    params=[
        (1, 16000, 16000),
        (2, 24000, 24000),
        (1, 48000, 96000),
        (2, 22050, 44100),
        (1, 8000, 4000),
    ]
)
def random_audio(request: pytest.FixtureRequest) -> AudioFile:
    channels, sample_rate, num_samples = request.param
    generator = torch.Generator().manual_seed(TEST_SEED)
    waveform = torch.randn(channels, num_samples, generator=generator)
    return AudioFile(waveform=waveform, sample_rate=sample_rate)


@pytest.fixture
def make_prepared(tmp_path: Path) -> Callable[..., Path]:
    """Tiny prepared datasets: 100 Hz audio (patch 10), a word every 0.5 s."""

    def make(
        name: str,
        utts: list[tuple[str, str, str | None, int | None, int]],
        rows_per_shard: int = 2,
    ) -> Path:
        folder = tmp_path / name
        schema = build_schema()
        with ShardWriter(folder, schema, rows_per_shard, batch_size=1) as writer:
            for utt_id, speaker, session, session_idx, n_words in utts:
                n_frames = 5 * n_words
                wav = torch.arange(n_frames * 10, dtype=torch.int16)
                writer.write(
                    {
                        "utt_id": utt_id,
                        "dataset": name,
                        "speaker": speaker,
                        "session": session,
                        "session_idx": session_idx,
                        "duration": 0.5 * n_words,
                        "text": " ".join(["word"] * n_words),
                        "phonemes": " ".join(["ab"] * n_words),
                        "word_start": [0.5 * k for k in range(n_words)],
                        "word_end": [0.5 * k + 0.3 for k in range(n_words)],
                        "word_phoneme_start": [3 * k for k in range(n_words)],
                        "word_text_start": [5 * k for k in range(n_words)],
                        "waveform_i16": wav.numpy().tobytes(),
                        "num_samples": n_frames * 10,
                        "sample_rate": 100,
                        "loudness": -24.0,
                    }
                )
        write_meta(
            folder,
            PreparedMeta(sample_rate=100, target_loudness=-24.0, aligner="MMS_FA"),
        )
        return folder

    return make

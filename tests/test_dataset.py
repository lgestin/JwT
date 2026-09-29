import pytest
import torch

from jwt.data.audio_prompt import AudioPromptConfig
from jwt.data.collate import collate
from jwt.data.dataset import AudioDataset
from jwt.data.source import ArrowTTSSource, LJTTSSource
from jwt.data.text import Tokenizer, Vocabulary

TOKENIZER = Tokenizer(Vocabulary({"a": 0, "b": 1, " ": 2}))
UTTS = [
    ("u0", "d/a", "s", 0, 10),
    ("u1", "d/a", "s", 1, 10),
    ("u2", "d/a", "s", 2, 6),
    ("u3", "d/a", "s", 3, 8),
]


def dataset_for(make_prepared, cfg, seed=0) -> AudioDataset:
    source = ArrowTTSSource(str(make_prepared("d", UTTS)), TOKENIZER, patch_size=10)
    return AudioDataset(source, sample_rate=100, audio_prompt=cfg, seed=seed)


def test_prompts_off_is_unchanged(make_prepared) -> None:
    """Without audio_prompt, samples and batches carry no prompt."""
    sample = dataset_for(make_prepared, None)[0]
    assert sample.audio_prompt is None
    assert sample.audio.acoustic.shape[-1] == 50
    assert collate([sample]).audio_prompt is None


def test_cut_splits_audio_and_tokens(make_prepared) -> None:
    """A cut splits frames, waveform, text and phonemes at the same word."""
    cfg = AudioPromptConfig(p_drop=0.0, p_other=0.0, max_trim_s=0.0)
    sample = dataset_for(make_prepared, cfg)[0]
    f = sample.audio_prompt.shape[-1]  # prompt frames == cut frame (no trim)
    k = f // 5  # 5 frames per word
    assert f % 5 == 0 and 1.0 <= 0.5 * k <= 4.0
    assert sample.audio.acoustic.shape[-1] == 50 - f
    assert sample.audio.waveform.shape[-1] == (50 - f) * 10
    assert sample.audio.waveform[0, 0].item() == f * 10 / 32768.0
    assert sample.text.phonemes == " ".join(["ab"] * (10 - k))
    assert sample.text.text == " ".join(["word"] * (10 - k))


def test_cut_trim_shortens_prompt_only(make_prepared) -> None:
    """The trim removes up to 0.5 s from the prompt, never from the target."""
    cfg = AudioPromptConfig(p_drop=0.0, p_other=0.0, max_trim_s=0.5)
    sample = dataset_for(make_prepared, cfg)[0]
    target = sample.audio.acoustic.shape[-1]
    assert target % 5 == 0
    assert 50 - target - 5 <= sample.audio_prompt.shape[-1] <= 50 - target


def test_other_prompt_is_a_window_of_the_reference(make_prepared) -> None:
    """An other-utterance prompt is a max_prompt_s window; the target is whole."""
    cfg = AudioPromptConfig(p_drop=0.0, p_other=1.0, max_prompt_s=2.0)
    sample = dataset_for(make_prepared, cfg)[0]
    assert sample.audio_prompt.shape == (1, 10, 20)  # 2 s at 10 frames/s
    assert sample.audio.acoustic.shape[-1] == 50  # target untouched


def test_seeded_prompts_are_deterministic(make_prepared) -> None:
    """A seeded dataset gives the same prompt for an item every time."""
    ds = dataset_for(make_prepared, AudioPromptConfig(), seed=3)
    assert all(torch.equal(ds[i].audio_prompt, ds[i].audio_prompt) for i in range(4))


def test_collate_pads_prompts_including_all_empty(make_prepared) -> None:
    """Prompts pad to the longest, down to zero frames when all dropped."""
    ds = dataset_for(make_prepared, AudioPromptConfig(p_drop=1.0))
    batch = collate([ds[0], ds[1]])
    assert batch.audio_prompt.shape == (2, 1, 10, 0)
    assert batch.audio_prompt_mask.shape == (2, 0)
    cfg = AudioPromptConfig(p_drop=0.0, p_other=1.0, max_prompt_s=2.0)
    batch = collate([dataset_for(make_prepared, cfg)[i] for i in (0, 2)])
    assert batch.audio_prompt.shape == (2, 1, 10, 20)
    assert batch.audio_prompt_mask.all()


def test_dataset_is_a_split_of_the_source(make_prepared) -> None:
    """A dataset indexes the source through its split indices."""
    source = ArrowTTSSource(str(make_prepared("d", UTTS)), TOKENIZER, patch_size=10)
    ds = AudioDataset(source, sample_rate=100, indices=[0, 2])
    assert len(ds) == 2
    assert ds[1].idx == 2
    assert ds[1].audio.acoustic.shape[-1] == 30  # u2: 6 words


def test_split_prompts_never_use_other_splits(make_prepared) -> None:
    """u0's only far session mates are u2 (30 frames) and u3 (40 frames)."""
    source = ArrowTTSSource(str(make_prepared("d", UTTS)), TOKENIZER, patch_size=10)
    cfg = AudioPromptConfig(p_drop=0.0, p_other=1.0, max_prompt_s=5.0)
    ds = AudioDataset(source, 100, indices=[0, 1, 2], audio_prompt=cfg)
    assert {ds[0].audio_prompt.shape[-1] for _ in range(20)} == {30}


def test_audio_prompts_need_prepared_data(tmp_path) -> None:
    """Prompt selection reads speaker and word columns only prepared data has."""
    (tmp_path / "metadata.csv").write_text("LJ001-0001|r|Hi.\n", encoding="utf-8")
    with pytest.raises(TypeError, match="ArrowTTSSource"):
        AudioDataset(LJTTSSource(str(tmp_path)), 100, audio_prompt=AudioPromptConfig())

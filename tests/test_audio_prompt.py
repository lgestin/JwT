import random

from jwt.data.audio_prompt import (
    AudioPromptConfig,
    AudioPromptSampler,
    Cut,
    OtherUtterance,
)
from jwt.data.source import ArrowTTSSource

SESSION = [
    ("u0", "d/a", "s", 0, 10),
    ("u1", "d/a", "s", 1, 10),
    ("u2", "d/a", "s", 2, 10),
    ("u3", "d/a", "s", 3, 10),
]


def sampler_for(make_prepared, utts, **cfg) -> AudioPromptSampler:
    source = ArrowTTSSource(
        str(make_prepared("d", utts)), tokenizer=None, patch_size=10
    )
    return AudioPromptSampler(source, AudioPromptConfig(**cfg))


def draws(sampler: AudioPromptSampler, idx: int = 0, n: int = 100) -> set:
    return {sampler(idx, random.Random(seed)) for seed in range(n)}


def test_drop_always(make_prepared) -> None:
    """p_drop=1 never gives a prompt."""
    assert draws(sampler_for(make_prepared, SESSION, p_drop=1.0)) == {None}


def test_other_skips_session_neighbours(make_prepared) -> None:
    """Other-utterance prompts skip the adjacent clips of the session."""
    sampler = sampler_for(make_prepared, SESSION, p_drop=0.0, p_other=1.0)
    assert draws(sampler) == {OtherUtterance(2), OtherUtterance(3)}


def test_other_falls_back_to_same_speaker(make_prepared) -> None:
    """Without a far session mate, any other clip of the speaker is used."""
    utts = [
        ("u0", "d/a", "s", 0, 10),
        ("u1", "d/a", "s", 1, 10),
        ("u2", "d/a", "t", 0, 10),
    ]
    sampler = sampler_for(make_prepared, utts, p_drop=0.0, p_other=1.0)
    assert draws(sampler) == {OtherUtterance(1), OtherUtterance(2)}


def test_single_utterance_speaker_falls_back_to_cut(make_prepared) -> None:
    """A speaker with one clip gets a same-utterance cut instead."""
    utts = [("u0", "d/a", None, None, 10)]
    sampler = sampler_for(make_prepared, utts, p_drop=0.0, p_other=1.0)
    assert isinstance(sampler(0, random.Random(0)), Cut)


def test_cut_respects_bounds(make_prepared) -> None:
    """Cuts start between min and max prompt length and leave min target."""
    # 10 words of 0.5 s: starts 0.0..4.5 s, duration 5.0 s.
    sampler = sampler_for(
        make_prepared,
        SESSION,
        p_drop=0.0,
        p_other=0.0,
        min_prompt_s=1.0,
        max_prompt_s=3.0,
        min_target_s=1.5,
    )
    assert draws(sampler) == {Cut(k) for k in range(2, 7)}


def test_cut_requires_a_pause_before_the_word(make_prepared) -> None:
    """Fixture words are 0.3 s of speech + 0.2 s of pause, so 0.25 s rejects all."""
    utts = [("u0", "d/a", None, None, 10)]
    sampler = sampler_for(
        make_prepared, utts, p_drop=0.0, p_other=0.0, min_pause_s=0.25
    )
    assert sampler(0, random.Random(0)) is None


def test_cut_without_candidate_is_none(make_prepared) -> None:
    """An utterance too short for any cut gets no prompt."""
    utts = [("u0", "d/a", None, None, 2)]
    sampler = sampler_for(make_prepared, utts, p_drop=0.0, p_other=0.0)
    assert sampler(0, random.Random(0)) is None


def test_other_references_stay_in_the_split(make_prepared) -> None:
    """A split's prompts come from the split only (no validation audio in train)."""
    source = ArrowTTSSource(str(make_prepared("d", SESSION)), None, patch_size=10)
    cfg = AudioPromptConfig(p_drop=0.0, p_other=1.0)
    sampler = AudioPromptSampler(source, cfg, indices=[0, 1, 2])  # u3 held out
    assert draws(sampler) == {OtherUtterance(2)}

from itertools import pairwise
from pathlib import Path

import soundfile as sf
import torch
from misaki.token import MToken

from jwt.data.align import (
    Aligner,
    Aligners,
    mms_word,
    word_phoneme_starts,
    word_text_starts,
)
from jwt.data.text import Text

ASSETS_FOLDER = Path(__file__).parent / "assets"


def token(text: str, phonemes: str, whitespace: str) -> MToken:
    return MToken(text=text, tag="X", whitespace=whitespace, phonemes=phonemes)


def test_word_phoneme_starts_indexes_each_token() -> None:
    """Each token's phonemes plus whitespace advance the offset."""
    toks = [token("Hi", "hI", ""), token(",", ",", " "), token("you", "ju", "")]
    assert word_phoneme_starts(toks, "hI, ju") == [0, 2, 4]


def test_word_phoneme_starts_rejects_mismatch() -> None:
    """Tokens that don't rebuild the phoneme string give None."""
    assert word_phoneme_starts([token("Hi", "hI", "")], "hI!") is None


def test_word_phoneme_starts_on_real_g2p() -> None:
    """misaki's own tokens always rebuild its phoneme string."""
    text = Text("Mr. Smith said: \"It's 1905, isn't it?\"")
    starts = word_phoneme_starts(text.word_tokens, text.phonemes)
    assert starts is not None and len(starts) == len(text.word_tokens)


def test_word_text_starts_indexes_each_token() -> None:
    """Offsets index the text; a text the tokens don't rebuild gives None."""
    toks = [token("Hi", "hI", ""), token(",", ",", " "), token("you", "ju", "")]
    assert word_text_starts(toks, "Hi, you") == [0, 2, 4]
    assert word_text_starts(toks, "Hi,  you") is None


def test_mms_word_maps_letters_digits_punctuation() -> None:
    """Words keep letters and apostrophes, digits become the star token."""
    assert mms_word("Isn't") == "isn't"
    assert mms_word("1905") == "*"
    assert mms_word(",") is None


def test_aligners_build_from_their_name() -> None:
    """create_arrow.py selects the aligner by name (`--aligner MMS_FA`)."""
    assert Aligners("MMS_FA") is Aligners.MMS_FA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clips() -> tuple[list[torch.Tensor], int, list[Text]]:
    """Two clips of different lengths, so the batch needs padding."""
    wav, sr = sf.read(ASSETS_FOLDER / "physicsworks.wav", dtype="float32")
    wav = torch.from_numpy(wav)
    waveforms = [wav[None, : sr * 4], wav[None, sr * 4 : sr * 6]]
    texts = [Text("Physics works in 1905, isn't it?"), Text("It really works.")]
    return waveforms, sr, texts


def test_mms_fa_spans_are_monotonic_and_in_bounds() -> None:
    """Each utterance gets one ordered span per token, inside its audio."""
    waveforms, sr, texts = clips()
    aligner = Aligners.MMS_FA.aligner(DEVICE)
    assert isinstance(aligner, Aligner)
    batch = aligner.align(waveforms, sr, [t.word_tokens for t in texts])
    for spans, wav, text in zip(batch, waveforms, texts, strict=True):
        assert spans is not None and len(spans) == len(text.word_tokens)
        assert all(a[0] <= b[0] for a, b in pairwise(spans))
        assert all(s <= e <= wav.shape[-1] / sr + 1e-3 for s, e in spans)


def test_mms_fa_batch_matches_one_at_a_time() -> None:
    """Padding and per-row normalization keep spans within one 20 ms frame."""
    waveforms, sr, texts = clips()
    aligner = Aligners.MMS_FA.aligner(DEVICE)
    batch = aligner.align(waveforms, sr, [t.word_tokens for t in texts])
    for wav, text, batched in zip(waveforms, texts, batch, strict=True):
        [single] = aligner.align([wav], sr, [text.word_tokens])
        assert single is not None and batched is not None
        for (s1, e1), (s2, e2) in zip(single, batched, strict=True):
            assert abs(s1 - s2) <= 0.021 and abs(e1 - e2) <= 0.021


def test_mms_fa_unalignable_utterance_is_none() -> None:
    """One bad utterance doesn't sink the batch."""
    waveforms, sr, texts = clips()
    aligner = Aligners.MMS_FA.aligner(DEVICE)
    tokens = [texts[0].word_tokens, Text("...").word_tokens]  # no spoken word
    good, bad = aligner.align(waveforms, sr, tokens)
    assert good is not None and bad is None


def test_mms_fa_audio_too_short_for_its_text_is_none() -> None:
    """CTC needs a frame per token: 50 ms can't hold a sentence."""
    waveforms, sr, texts = clips()
    aligner = Aligners.MMS_FA.aligner(DEVICE)
    [spans] = aligner.align([waveforms[0][:, : sr // 20]], sr, [texts[0].word_tokens])
    assert spans is None

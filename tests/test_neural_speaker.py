import pytest
import torch

from tts.data.audio.stft import MelSpectrogram
from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from tts.model.transformer import AdaLN, TransformerConfig

B = 2
T_TEXT = 4
N_MELS = 16
T_MEL_MAX = 16
MAX_MEL_LEN = T_MEL_MAX


@pytest.fixture
def model() -> RollingFlowSpeaker:
    torch.manual_seed(0)
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        mel_dim=N_MELS,
        n_denoising_steps=4,
        max_mel_len=MAX_MEL_LEN,
        eos_n_frames=2,
        eos_mel_value=-15.0,
        eos_detect_threshold=-11.0,
    )
    return RollingFlowSpeaker(cfg).eval()


@pytest.fixture
def text(model: RollingFlowSpeaker) -> MaskedTensor:
    return MaskedTensor(
        values=torch.randint(0, model.cfg.vocabulary_size, (B, 1, T_TEXT)),
        mask=torch.ones(B, T_TEXT, dtype=torch.bool),
    )


@pytest.fixture
def mels(model: RollingFlowSpeaker, audio_from_file) -> MaskedTensor:
    """Mels from the most-energetic T_MEL_MAX-frame window of a real audio asset."""
    waveform = audio_from_file.waveform  # (channels, T_audio)
    mono = waveform.mean(dim=0, keepdim=True)  # (1, T_audio)
    batch_wave = mono.repeat(B, 1)  # (B, T_audio)
    mel_spec = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        n_mels=model.cfg.mel_dim,
        sample_rate=audio_from_file.sample_rate,
    )
    full = mel_spec(batch_wave)  # (B, mel_dim, T_mel_full)
    # Slice the highest-energy window so the test data isn't a leading-silence flat patch.
    energy = full[0].mean(dim=0)  # (T_mel_full,)
    start = int(energy.unfold(0, T_MEL_MAX, 1).mean(dim=-1).argmax().item())
    values = full[..., start : start + T_MEL_MAX]
    return MaskedTensor(values=values, mask=torch.ones(B, T_MEL_MAX, dtype=torch.bool))


def test_speak_shape_and_mask(model: RollingFlowSpeaker, text: MaskedTensor) -> None:
    out = model.speak(text)
    assert out.values.ndim == 3
    assert out.values.shape[0] == B
    assert out.values.shape[1] == N_MELS
    assert out.values.shape[2] <= model.cfg.max_mel_len
    assert out.mask.shape == (B, out.values.shape[2])
    assert (out.mask.sum(-1) >= 0).all()
    assert (out.mask.sum(-1) <= model.cfg.max_mel_len).all()


def test_speak_outputs_are_finite(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    out = model.speak(text)
    assert torch.isfinite(out.values).all()


def test_speak_is_deterministic_with_pinned_x0(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    x_0 = torch.randn(B, N_MELS, model.cfg.max_mel_len)
    a = model.speak(text, x_0=x_0)
    b = model.speak(text, x_0=x_0)
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.mask, b.mask)


def test_speak_actually_updates_positions(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    """Output should differ from the initial noise — speak must integrate."""
    x_0 = torch.randn(B, N_MELS, model.cfg.max_mel_len)
    out = model.speak(text, x_0=x_0)
    mean = model.mel_mean
    std = model.mel_std
    for i in range(B):
        L = int(out.mask[i].sum().item())
        if L == 0:
            continue
        x_0_denorm = x_0[i, :, :L] * std + mean
        assert not torch.allclose(out.values[i, :, :L], x_0_denorm)


def test_speak_stops_on_sentinel(model: RollingFlowSpeaker, text: MaskedTensor) -> None:
    """speak() must stop before max_mel_len when the sentinel fires."""
    # Drive every generated frame to eos_mel_value (normalized) so the sentinel
    # fires as early as possible.
    eos_norm = (model.cfg.eos_mel_value - model.mel_mean) / model.mel_std

    original_forward = model.forward

    def _always_eos(text, mels, t):
        v = original_forward(text, mels, t)
        # Bias velocity so x_1 target is eos_norm everywhere.
        x_t = mels.values.transpose(1, 2)  # (B, T, mel_dim), normalized
        # v = x_1 - x_0; x_t = (1-t)*x_0 + t*x_1 → x_1 = (x_t - (1-t)*x_0) / t
        # Just return a large constant velocity toward eos_norm.
        return torch.full_like(v, float(eos_norm) * 10)

    model.forward = _always_eos
    try:
        out = model.speak(text)
    finally:
        model.forward = original_forward

    assert out.values.shape[2] < model.cfg.max_mel_len


def test_speak_respects_max_mel_len_cap(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    """Output must never exceed max_mel_len even if the sentinel never fires."""
    # Drive frames toward a high value — sentinel won't trigger.
    original_forward = model.forward

    def _never_eos(text, mels, t):
        return torch.full_like(original_forward(text, mels, t), 10.0)

    model.forward = _never_eos
    try:
        out = model.speak(text)
    finally:
        model.forward = original_forward

    assert out.values.shape[2] <= model.cfg.max_mel_len


def test_forward_invariant_to_text_padding(model: RollingFlowSpeaker) -> None:
    """Right-padding the text mask must not change v_pred for the real tokens.

    Regression: the attention mask was built in original (un-packed) coords
    and applied to the packed sequence, so any batch with mixed text lengths
    silently misaligned the mask — dropping real mel positions and admitting
    duplicates of the last mel in their place.
    """
    # Zero-init AdaLN makes every block an identity, which would mask the bug.
    # Perturb so the attention path actually contributes to the output.
    torch.manual_seed(0)
    for m in model.modules():
        if isinstance(m, AdaLN):
            torch.nn.init.normal_(m.linear.weight, std=0.02)
            torch.nn.init.normal_(m.linear.bias, std=0.02)

    T_mel = 5
    text_ids = torch.randint(0, model.cfg.vocabulary_size, (B, 2))
    mel_vals = torch.randn(B, model.cfg.mel_dim, T_mel)
    t = torch.full((B, T_mel), 0.5)
    mels = MaskedTensor(values=mel_vals, mask=torch.ones(B, T_mel, dtype=torch.bool))

    text_unpadded = MaskedTensor(
        values=text_ids.unsqueeze(1),
        mask=torch.ones(B, 2, dtype=torch.bool),
    )
    text_padded = MaskedTensor(
        values=torch.cat([text_ids, torch.zeros(B, 4, dtype=torch.long)], dim=1).unsqueeze(1),
        mask=torch.tensor([[True, True, False, False, False, False]] * B),
    )

    with torch.no_grad():
        v_unpadded = model.forward(text_unpadded, mels, t)
        v_padded = model.forward(text_padded, mels, t)

    assert torch.allclose(v_unpadded, v_padded, atol=1e-5), (
        f"text padding changed v_pred (max diff "
        f"{(v_unpadded - v_padded).abs().max().item():.4e}) — attn mask misaligned"
    )


def test_training_step_covers_warmup_with_negative_mel_front(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    """Negative mel_front must produce the partial-ramp distribution that
    inference sees during warm-up (steps k=0..n-2 in speak)."""
    n = model.cfg.n_denoising_steps
    mel_front = torch.tensor([-(n - 1), -1], dtype=torch.long)
    v_pred, _, v_mask, _ = model.training_step(text, mels, mel_front=mel_front)
    assert torch.isfinite(v_pred).all()
    # Sample 0 (mel_front=-(n-1)): only position 0 is in the supervision window.
    assert v_mask[0].sum().item() == 1
    assert bool(v_mask[0, 0])
    # Sample 1 (mel_front=-1): positions 0..n-2 are supervised.
    assert v_mask[1].sum().item() == n - 1
    assert bool(v_mask[1, : n - 1].all())


def test_training_step_default_mel_front_samples_warmup_region(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    """Default mel_front sampler must cover [-(n-1), mel_lens+eos_n) so the
    training distribution includes the inference warm-up shapes and sentinel."""
    n = model.cfg.n_denoising_steps
    eos_n = model.cfg.eos_n_frames
    mel_lens = mels.mask.sum(-1)
    torch.manual_seed(0)
    fronts = []
    for _ in range(500):
        _, _, _, t = model.training_step(text, mels)
        for b in range(B):
            tb = t[b]
            ones = (tb == 1.0).nonzero(as_tuple=True)[0]
            if len(ones) > 0:
                fronts.append(int(ones[0].item()))
            else:
                fronts.append(int(round((tb[0].item() - 1.0) * (n - 1))))
    fronts_t = torch.tensor(fronts)
    assert fronts_t.min().item() >= -(n - 1), fronts_t.min().item()
    assert fronts_t.max().item() < int((mel_lens + eos_n).max().item())
    assert (fronts_t < 0).any(), "default sampler never produced negative mel_front"


def test_training_step_shapes(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    T_mel = mels.values.shape[-1]
    eos_n = model.cfg.eos_n_frames
    T_ext = T_mel + eos_n
    mel_front = torch.tensor([2, 3], dtype=torch.long)
    v_pred, v_target, v_mask, t = model.training_step(text, mels, mel_front=mel_front)
    assert v_pred.shape == (B, T_ext, N_MELS)
    assert v_target.shape == (B, T_ext, N_MELS)
    assert v_mask.shape == (B, T_ext)
    assert t.shape == (B, T_ext)
    n = model.cfg.n_denoising_steps
    assert (v_mask.sum(-1) == n - 1).all()


def test_training_step_is_deterministic(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    T_mel = mels.values.shape[-1]
    eos_n = model.cfg.eos_n_frames
    mel_front = torch.tensor([2, 3], dtype=torch.long)
    x_0 = torch.randn(B, T_mel + eos_n, N_MELS)
    a = model.training_step(text, mels, mel_front=mel_front, x_0=x_0)
    b = model.training_step(text, mels, mel_front=mel_front, x_0=x_0)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)


def test_training_step_appends_sentinel(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    """training_step must append eos_n_frames sentinel frames to each sample.

    The supervision target (v_target = x_1 - x_0) at sentinel positions must
    correspond to x_1 = normalized(eos_mel_value), independent of the noise x_0.
    We pin mel_front to a position before the real frames so the rolling window
    lands entirely in the real region — sentinel positions will have t=0 (pure
    noise) and fall outside the supervision window, but x_1 at those positions
    is still the normalized sentinel value.
    """
    eos_n = model.cfg.eos_n_frames
    eos_val_norm = (model.cfg.eos_mel_value - float(model.mel_mean)) / float(model.mel_std)
    mel_lens = mels.mask.sum(-1)
    T_mel = mels.values.shape[-1]

    # Pin x_0 to zeros so v_target = x_1 - 0 = x_1 at sentinel positions.
    x_0 = torch.zeros(B, T_mel + eos_n, N_MELS)
    mel_front = torch.zeros(B, dtype=torch.long)  # front at position 0
    _, v_target, _, _ = model.training_step(text, mels, mel_front=mel_front, x_0=x_0)

    for b in range(B):
        L = int(mel_lens[b].item())
        sentinel_positions = slice(L, L + eos_n)
        sentinel_target = v_target[b, sentinel_positions, :]
        assert sentinel_target.shape == (eos_n, N_MELS)
        assert torch.allclose(
            sentinel_target,
            torch.full_like(sentinel_target, eos_val_norm),
            atol=1e-5,
        ), (
            f"sample {b}: sentinel target mean {sentinel_target.mean():.4f} "
            f"!= expected {eos_val_norm:.4f}"
        )

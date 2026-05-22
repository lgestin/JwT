import pytest
import torch

from tts.data.audio.codecs import Codec, Codecs
from tts.data.audio.stft import MelSpectrogram
from tts.model.flow import FlowParametrizations
from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from tts.model.transformer import AdaLN, TransformerConfig
from tts.training.timestep_schedules import TimestepSchedules

B = 2
T_TEXT = 4
N_MELS = 16
T_MEL_MAX = 16
MAX_AC_LEN = T_MEL_MAX


class StubCodec:
    """Lightweight Codec for testing — BigVGAN-shaped normalize/EOS semantics
    without the HF download. Implements the Codec protocol via duck typing.
    """

    required_sample_rate: int | None = None

    def __init__(self, acoustic_dim: int = N_MELS):
        self.acoustic_dim = acoustic_dim
        self.hop_length = 256
        self.mean = -5.0
        self.std = 2.0
        self.eos_value = -15.0
        self.eos_threshold = -11.0

    def encode(self, w: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def eos_frames(
        self, n: int, *, device=None, dtype=torch.float32
    ) -> torch.Tensor:
        return torch.full((self.acoustic_dim, n), self.eos_value, device=device, dtype=dtype)

    def is_eos(self, frame: torch.Tensor) -> torch.BoolTensor:
        return frame.mean(dim=-1) < self.eos_threshold

    def reconstruct(self, w: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


# Make StubCodec satisfy the runtime_checkable Codec protocol.
assert isinstance(StubCodec(), Codec)


@pytest.fixture
def codec() -> StubCodec:
    return StubCodec()


@pytest.fixture
def model(codec: StubCodec) -> RollingFlowSpeaker:
    torch.manual_seed(0)
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        codec=Codecs.BIGVGAN,
        acoustic_dim=N_MELS,
        n_denoising_steps=4,
        max_acoustic_len=MAX_AC_LEN,
        eos_n_frames=2,
    )
    return RollingFlowSpeaker(cfg).eval()


@pytest.fixture
def text(model: RollingFlowSpeaker) -> MaskedTensor:
    return MaskedTensor(
        values=torch.randint(0, model.cfg.vocabulary_size, (B, 1, T_TEXT)),
        mask=torch.ones(B, T_TEXT, dtype=torch.bool),
    )


@pytest.fixture
def acoustic(
    model: RollingFlowSpeaker, codec: StubCodec, audio_from_file
) -> MaskedTensor:
    """Pre-EOS-extended, normalized acoustic features ready for model.training_step.

    Produces T_real real frames from a real audio asset, then appends eos_n
    sentinel frames, then normalizes — mirroring what the trainer's
    prepare_acoustic_batch helper does.
    """
    waveform = audio_from_file.waveform
    mono = waveform.mean(dim=0, keepdim=True)
    batch_wave = mono.repeat(B, 1)
    mel_spec = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        n_mels=model.cfg.acoustic_dim,
        sample_rate=audio_from_file.sample_rate,
    )
    full = mel_spec(batch_wave)  # (B, acoustic_dim, T_full)
    energy = full[0].mean(dim=0)
    start = int(energy.unfold(0, T_MEL_MAX, 1).mean(dim=-1).argmax().item())
    real = full[..., start : start + T_MEL_MAX]  # (B, acoustic_dim, T_real)

    eos_n = model.cfg.eos_n_frames
    eos = codec.eos_frames(eos_n).unsqueeze(0).expand(B, model.cfg.acoustic_dim, eos_n)
    values_ext = torch.cat([real, eos], dim=-1)  # (B, acoustic_dim, T_real+eos_n)
    values_norm = codec.normalize(values_ext)
    mask = torch.ones(B, T_MEL_MAX + eos_n, dtype=torch.bool)
    return MaskedTensor(values=values_norm, mask=mask)


def test_speak_shape_and_mask(
    model: RollingFlowSpeaker, text: MaskedTensor, codec: StubCodec
) -> None:
    out = model.speak(text, codec=codec)
    assert out.values.ndim == 3
    assert out.values.shape[0] == B
    assert out.values.shape[1] == N_MELS
    assert out.values.shape[2] <= model.cfg.max_acoustic_len
    assert out.mask.shape == (B, out.values.shape[2])
    assert (out.mask.sum(-1) >= 0).all()
    assert (out.mask.sum(-1) <= model.cfg.max_acoustic_len).all()


def test_speak_outputs_are_finite(
    model: RollingFlowSpeaker, text: MaskedTensor, codec: StubCodec
) -> None:
    out = model.speak(text, codec=codec)
    assert torch.isfinite(out.values).all()


def test_speak_is_deterministic_with_pinned_x0(
    model: RollingFlowSpeaker, text: MaskedTensor, codec: StubCodec
) -> None:
    x_0 = torch.randn(B, N_MELS, model.cfg.max_acoustic_len)
    a = model.speak(text, codec=codec, x_0=x_0)
    b = model.speak(text, codec=codec, x_0=x_0)
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.mask, b.mask)


def test_speak_actually_updates_positions(
    model: RollingFlowSpeaker, text: MaskedTensor, codec: StubCodec
) -> None:
    """Output should differ from the initial noise — speak must integrate.

    speak() returns features in normalized space, so we compare directly to x_0.
    """
    x_0 = torch.randn(B, N_MELS, model.cfg.max_acoustic_len)
    out = model.speak(text, codec=codec, x_0=x_0)
    for i in range(B):
        L = int(out.mask[i].sum().item())
        if L == 0:
            continue
        assert not torch.allclose(out.values[i, :, :L], x_0[i, :, :L])


def test_speak_stops_on_sentinel(
    model: RollingFlowSpeaker, text: MaskedTensor, codec: StubCodec
) -> None:
    """speak() must stop before max_acoustic_len when the sentinel fires."""
    eos_norm = (codec.eos_value - codec.mean) / codec.std

    original_forward = model.forward

    def _always_eos(text, acoustic, t):
        v = original_forward(text, acoustic, t)
        return torch.full_like(v, float(eos_norm) * 10)

    model.forward = _always_eos
    try:
        out = model.speak(text, codec=codec)
    finally:
        model.forward = original_forward

    assert out.values.shape[2] < model.cfg.max_acoustic_len


def test_speak_respects_max_acoustic_len_cap(
    model: RollingFlowSpeaker, text: MaskedTensor, codec: StubCodec
) -> None:
    """Output must never exceed max_acoustic_len even if the sentinel never fires."""
    original_forward = model.forward

    def _never_eos(text, acoustic, t):
        return torch.full_like(original_forward(text, acoustic, t), 10.0)

    model.forward = _never_eos
    try:
        out = model.speak(text, codec=codec)
    finally:
        model.forward = original_forward

    assert out.values.shape[2] <= model.cfg.max_acoustic_len


def test_forward_invariant_to_text_padding(model: RollingFlowSpeaker) -> None:
    """Right-padding the text mask must not change v_pred for the real tokens.

    Regression: the attention mask was built in original (un-packed) coords
    and applied to the packed sequence, so any batch with mixed text lengths
    silently misaligned the mask — dropping real acoustic positions and
    admitting duplicates of the last frame in their place.
    """
    torch.manual_seed(0)
    for m in model.modules():
        if isinstance(m, AdaLN):
            torch.nn.init.normal_(m.linear.weight, std=0.02)
            torch.nn.init.normal_(m.linear.bias, std=0.02)

    T_ac = 5
    text_ids = torch.randint(0, model.cfg.vocabulary_size, (B, 2))
    ac_vals = torch.randn(B, model.cfg.acoustic_dim, T_ac)
    t = torch.full((B, T_ac), 0.5)
    acoustic = MaskedTensor(values=ac_vals, mask=torch.ones(B, T_ac, dtype=torch.bool))

    text_unpadded = MaskedTensor(
        values=text_ids.unsqueeze(1),
        mask=torch.ones(B, 2, dtype=torch.bool),
    )
    text_padded = MaskedTensor(
        values=torch.cat([text_ids, torch.zeros(B, 4, dtype=torch.long)], dim=1).unsqueeze(1),
        mask=torch.tensor([[True, True, False, False, False, False]] * B),
    )

    with torch.no_grad():
        v_unpadded = model.forward(text_unpadded, acoustic, t)
        v_padded = model.forward(text_padded, acoustic, t)

    assert torch.allclose(v_unpadded, v_padded, atol=1e-5), (
        f"text padding changed v_pred (max diff "
        f"{(v_unpadded - v_padded).abs().max().item():.4e}) — attn mask misaligned"
    )


def test_training_step_covers_warmup_with_negative_acoustic_front(
    model: RollingFlowSpeaker, text: MaskedTensor, acoustic: MaskedTensor
) -> None:
    """Negative acoustic_front must produce the partial-ramp distribution that
    inference sees during warm-up (steps k=0..n-2 in speak)."""
    n = model.cfg.n_denoising_steps
    acoustic_front = torch.tensor([-(n - 1), -1], dtype=torch.long)
    out = model.training_step(text, acoustic, acoustic_front=acoustic_front)
    assert torch.isfinite(out.loss).all()
    assert torch.isfinite(out.x_pred).all()
    # Sample 0 (acoustic_front=-(n-1)): only position 0 is in the supervision window.
    assert out.v_mask[0].sum().item() == 1
    assert bool(out.v_mask[0, 0])
    # Sample 1 (acoustic_front=-1): positions 0..n-2 are supervised.
    assert out.v_mask[1].sum().item() == n - 1
    assert bool(out.v_mask[1, : n - 1].all())


def test_training_step_default_acoustic_front_samples_warmup_region(
    model: RollingFlowSpeaker, text: MaskedTensor, acoustic: MaskedTensor
) -> None:
    """Default acoustic_front sampler must cover [-(n-1), acoustic_lens_ext) so
    the training distribution includes the inference warm-up shapes and sentinel.
    """
    n = model.cfg.n_denoising_steps
    lens_ext = acoustic.mask.sum(-1)
    torch.manual_seed(0)
    fronts = []
    for _ in range(500):
        t = model.training_step(text, acoustic).t
        for b in range(B):
            tb = t[b]
            ones = (tb == 1.0).nonzero(as_tuple=True)[0]
            if len(ones) > 0:
                fronts.append(int(ones[0].item()))
            else:
                fronts.append(int(round((tb[0].item() - 1.0) * (n - 1))))
    fronts_t = torch.tensor(fronts)
    assert fronts_t.min().item() >= -(n - 1), fronts_t.min().item()
    assert fronts_t.max().item() < int(lens_ext.max().item())
    assert (fronts_t < 0).any(), "default sampler never produced negative acoustic_front"


def test_training_step_shapes(
    model: RollingFlowSpeaker, text: MaskedTensor, acoustic: MaskedTensor
) -> None:
    T_ext = acoustic.values.shape[-1]
    acoustic_front = torch.tensor([2, 3], dtype=torch.long)
    out = model.training_step(text, acoustic, acoustic_front=acoustic_front)
    assert out.loss.dim() == 0
    assert out.x_pred.shape == (B, T_ext, N_MELS)
    assert out.v_mask.shape == (B, T_ext)
    assert out.t.shape == (B, T_ext)
    assert out.per_pos_loss.shape == (B, T_ext)
    n = model.cfg.n_denoising_steps
    assert (out.v_mask.sum(-1) == n - 1).all()


def test_training_step_is_deterministic(
    model: RollingFlowSpeaker, text: MaskedTensor, acoustic: MaskedTensor
) -> None:
    T_ext = acoustic.values.shape[-1]
    acoustic_front = torch.tensor([2, 3], dtype=torch.long)
    x_0 = torch.randn(B, T_ext, N_MELS)
    a = model.training_step(text, acoustic, acoustic_front=acoustic_front, x_0=x_0)
    b = model.training_step(text, acoustic, acoustic_front=acoustic_front, x_0=x_0)
    for name in ("loss", "x_pred", "v_mask", "t", "per_pos_loss"):
        assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize(
    "parametrization", [FlowParametrizations.RECTIFIED_FLOW, FlowParametrizations.JWT]
)
def test_training_step_finite_for_both_parametrizations(
    parametrization: FlowParametrizations, audio_from_file
) -> None:
    """RF and JWT should both produce finite loss + x_pred end-to-end."""
    torch.manual_seed(0)
    codec_ = StubCodec()
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        codec=Codecs.BIGVGAN,
        parametrization=parametrization,
        acoustic_dim=N_MELS,
        n_denoising_steps=4,
        max_acoustic_len=MAX_AC_LEN,
        eos_n_frames=2,
    )
    m = RollingFlowSpeaker(cfg).eval()
    txt = MaskedTensor(
        values=torch.randint(0, m.cfg.vocabulary_size, (B, 1, T_TEXT)),
        mask=torch.ones(B, T_TEXT, dtype=torch.bool),
    )
    # Quick stand-in for the `acoustic` fixture so this test stays codec-agnostic.
    eos_n = m.cfg.eos_n_frames
    real = torch.randn(B, m.cfg.acoustic_dim, T_MEL_MAX) * 2 - 5  # roughly log-mel range
    eos_frames = codec_.eos_frames(eos_n).unsqueeze(0).expand(B, m.cfg.acoustic_dim, eos_n)
    ext = codec_.normalize(torch.cat([real, eos_frames], dim=-1))
    ac = MaskedTensor(values=ext, mask=torch.ones(B, T_MEL_MAX + eos_n, dtype=torch.bool))

    out = m.training_step(txt, ac)
    assert torch.isfinite(out.loss), f"{parametrization}: loss not finite"
    assert torch.isfinite(out.x_pred).all(), f"{parametrization}: x_pred not finite"
    assert (out.v_mask.sum() > 0).item()


@pytest.mark.parametrize(
    "parametrization", [FlowParametrizations.RECTIFIED_FLOW, FlowParametrizations.JWT]
)
def test_speak_finite_for_both_parametrizations(
    parametrization: FlowParametrizations,
) -> None:
    """JWT.step divides by (1-t); the t=1 column must not leak NaN into outputs."""
    torch.manual_seed(0)
    codec_ = StubCodec()
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        codec=Codecs.BIGVGAN,
        parametrization=parametrization,
        acoustic_dim=N_MELS,
        n_denoising_steps=4,
        max_acoustic_len=MAX_AC_LEN,
        eos_n_frames=2,
    )
    m = RollingFlowSpeaker(cfg).eval()
    txt = MaskedTensor(
        values=torch.randint(0, m.cfg.vocabulary_size, (B, 1, T_TEXT)),
        mask=torch.ones(B, T_TEXT, dtype=torch.bool),
    )
    out = m.speak(txt, codec=codec_)
    assert torch.isfinite(out.values).all(), f"{parametrization}: speak() produced NaN/Inf"


def test_noise_scale_field_defaults_to_one() -> None:
    """noise_scale defaults to 1.0 — existing configs keep unit-Gaussian noise."""
    assert RollingFlowConfig().noise_scale == 1.0


def test_lognorm_schedule_runs_end_to_end() -> None:
    """A non-linear (logit-normal) schedule must drive both training_step and
    speak to finite outputs — exercises the warped t-ramp and per-position dt."""
    torch.manual_seed(0)
    codec_ = StubCodec()
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        codec=Codecs.BIGVGAN,
        parametrization=FlowParametrizations.JWT,  # JWT.step divides by (1-t)
        timestep_schedule=TimestepSchedules.LOG_NORM,
        acoustic_dim=N_MELS,
        n_denoising_steps=4,
        max_acoustic_len=MAX_AC_LEN,
        eos_n_frames=2,
    )
    m = RollingFlowSpeaker(cfg).eval()
    txt = MaskedTensor(
        values=torch.randint(0, m.cfg.vocabulary_size, (B, 1, T_TEXT)),
        mask=torch.ones(B, T_TEXT, dtype=torch.bool),
    )
    eos_n = m.cfg.eos_n_frames
    real = torch.randn(B, m.cfg.acoustic_dim, T_MEL_MAX) * 2 - 5
    eos_frames = codec_.eos_frames(eos_n).unsqueeze(0).expand(B, m.cfg.acoustic_dim, eos_n)
    ext = codec_.normalize(torch.cat([real, eos_frames], dim=-1))
    ac = MaskedTensor(values=ext, mask=torch.ones(B, T_MEL_MAX + eos_n, dtype=torch.bool))

    out = m.training_step(txt, ac)
    assert torch.isfinite(out.loss), "lognorm: training loss not finite"
    assert torch.isfinite(out.x_pred).all(), "lognorm: x_pred not finite"

    spoken = m.speak(txt, codec=codec_)
    assert torch.isfinite(spoken.values).all(), "lognorm: speak produced NaN/Inf"


def test_sample_noise_applies_cfg_noise_scale() -> None:
    """_sample_noise draws a zero-mean Gaussian with std == cfg.noise_scale, so
    training and inference share the same scaled prior."""
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        codec=Codecs.BIGVGAN,
        acoustic_dim=N_MELS,
        n_denoising_steps=4,
        max_acoustic_len=MAX_AC_LEN,
        eos_n_frames=2,
        noise_scale=0.3,
    )
    speaker = RollingFlowSpeaker(cfg).eval()
    torch.manual_seed(0)
    noise = speaker._sample_noise((40000,), device=torch.device("cpu"))

    assert abs(noise.std().item() - 0.3) < 0.01
    assert abs(noise.mean().item()) < 0.01

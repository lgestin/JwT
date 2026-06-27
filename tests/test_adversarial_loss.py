import torch

from jwt.model.discriminator import MultiResolutionSTFTDiscriminator
from jwt.training.adversarial_loss import AdversarialLoss

HOP = 64
N_FFTS = (256, 128)


def _wav(batch: int, n_frames: int, *, seed: int) -> torch.Tensor:
    """A (batch, n_frames * HOP) waveform — decodes to exactly n_frames frames."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_frames * HOP, generator=generator)


def _adv_loss() -> AdversarialLoss:
    disc = MultiResolutionSTFTDiscriminator(n_ffts=N_FFTS, channels=8)
    return AdversarialLoss(disc, hop_length=HOP)


def test_discriminator_returns_logits_and_features_per_resolution() -> None:
    disc = MultiResolutionSTFTDiscriminator(n_ffts=N_FFTS, channels=8)
    outputs = disc(_wav(2, 32, seed=0))
    assert len(outputs) == len(N_FFTS)
    for logits, feats in outputs:
        assert logits.shape[0] == 2  # batch preserved
        assert len(feats) >= 1
        assert feats[-1] is logits  # final feature map is the logit map


def test_feature_matching_is_zero_for_identical_waveforms() -> None:
    adv = _adv_loss()
    wav = _wav(2, 32, seed=1)
    v_mask = torch.ones(2, 32)
    _, feat = adv.generator_loss(wav, wav.clone(), v_mask)
    assert feat.item() == 0.0


def test_generator_loss_gradient_flows_to_prediction() -> None:
    adv = _adv_loss()
    pred = _wav(2, 32, seed=1).requires_grad_(True)
    target = _wav(2, 32, seed=2)
    adv_g, feat = adv.generator_loss(pred, target, torch.ones(2, 32))
    (adv_g + feat).backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_requires_grad_toggle_separates_generator_and_discriminator() -> None:
    """The trainer freezes D for the generator term and unfreezes it for the
    discriminator term. Frozen, the generator backward must leave D params with
    no grad; unfrozen, the discriminator backward must populate them."""
    adv = _adv_loss()
    pred = _wav(2, 32, seed=1).requires_grad_(True)
    target = _wav(2, 32, seed=2)
    v_mask = torch.ones(2, 32)

    # Generator phase: D frozen.
    for p in adv.discriminator.parameters():
        p.requires_grad_(False)
    adv_g, feat = adv.generator_loss(pred, target, v_mask)
    (adv_g + feat).backward()
    assert all(p.grad is None for p in adv.discriminator.parameters())

    # Discriminator phase: D unfrozen.
    for p in adv.discriminator.parameters():
        p.requires_grad_(True)
    disc_loss = adv.discriminator_loss(pred, target, v_mask)
    disc_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adv.discriminator.parameters())


def test_zero_mask_gives_no_generator_gradient_to_prediction() -> None:
    """With an all-zero v_mask the prediction is replaced by the target, so the
    generator loss does not depend on (and yields no gradient to) the prediction."""
    adv = _adv_loss()
    pred = _wav(2, 32, seed=1).requires_grad_(True)
    target = _wav(2, 32, seed=2)
    adv_g, feat = adv.generator_loss(pred, target, torch.zeros(2, 32))
    (adv_g + feat).backward()
    # The prediction is fully masked out, so its gradient is exactly zero.
    assert pred.grad is None or pred.grad.abs().sum() == 0


def test_discriminator_loss_is_finite() -> None:
    adv = _adv_loss()
    pred, target = _wav(2, 32, seed=1), _wav(2, 32, seed=2)
    loss = adv.discriminator_loss(pred, target, torch.ones(2, 32))
    assert torch.isfinite(loss)
    assert loss.item() > 0  # hinge loss on an untrained D is positive


def test_leading_window_crop_localizes_gradient() -> None:
    """With window_frames=K, the discriminator only sees the K leading frames of
    the rolling window, so the generator gradient lands only on those samples."""
    K = 6
    adv = AdversarialLoss(
        MultiResolutionSTFTDiscriminator(n_ffts=N_FFTS, channels=8),
        hop_length=HOP,
        window_frames=K,
    )
    T = 32
    pred = _wav(2, T, seed=1).requires_grad_(True)
    target = _wav(2, T, seed=2)
    v_mask = torch.zeros(2, T)
    v_mask[:, 10:26] = 1.0  # rolling window = frames 10..25
    t = torch.zeros(2, T)
    t[:, 10:26] = torch.linspace(1.0, 0.0, 16)  # ramps down across the window
    adv_g, feat = adv.generator_loss(pred, target, v_mask, t)
    (adv_g + feat).backward()

    assert pred.grad is not None
    g = pred.grad.abs().sum(dim=0)  # (T*HOP,)
    lead = slice(10 * HOP, (10 + K) * HOP)  # leading-K samples
    assert g[lead].sum() > 0
    outside = torch.ones_like(g, dtype=torch.bool)
    outside[lead] = False
    assert g[outside].sum() == 0  # nothing outside the leading-K crop


def test_t_weighting_changes_adversarial_loss() -> None:
    """t-weighting (w=t**gamma) actually reweights the generator hinge: with the
    same discriminator and inputs, gamma>0 differs from the unweighted mean."""
    disc = MultiResolutionSTFTDiscriminator(n_ffts=N_FFTS, channels=8)
    K = 8
    unweighted = AdversarialLoss(disc, hop_length=HOP, window_frames=K, t_weight_pow=0.0)
    weighted = AdversarialLoss(disc, hop_length=HOP, window_frames=K, t_weight_pow=3.0)
    pred, target = _wav(2, 32, seed=1), _wav(2, 32, seed=2)
    v_mask = torch.zeros(2, 32)
    v_mask[:, 4:28] = 1.0
    t = torch.zeros(2, 32)
    t[:, 4:28] = torch.linspace(1.0, 0.0, 24)

    a0, _ = unweighted.generator_loss(pred, target, v_mask, t)
    a1, _ = weighted.generator_loss(pred, target, v_mask, t)
    assert not torch.allclose(a0, a1)


def test_grad_accumulation_matches_single_step() -> None:
    """Grad accumulation is correct for the adversarial setting: accumulating
    `ga` identical micro-batches (each scaled by 1/ga) yields the same generator
    AND discriminator gradients as a single un-accumulated step. This mirrors the
    trainer's per-micro-step sequence — freeze D for the generator backward,
    unfreeze for the discriminator backward — and so also proves the
    requires_grad toggle never contaminates or clears the accumulating D grads.
    """
    torch.manual_seed(0)
    adv = AdversarialLoss(MultiResolutionSTFTDiscriminator(n_ffts=N_FFTS, channels=8), hop_length=HOP)
    gen = torch.nn.Linear(8, 32 * HOP)  # fixed latent -> waveform "generator"
    latent = torch.randn(2, 8)
    target = _wav(2, 32, seed=7)
    v_mask = torch.ones(2, 32)

    def run(ga: int, n_micro: int):
        gen.zero_grad(set_to_none=True)
        adv.discriminator.zero_grad(set_to_none=True)
        for _ in range(n_micro):
            pred = gen(latent)
            # Generator phase: D frozen — backward flows to `gen`, not to D.
            for p in adv.discriminator.parameters():
                p.requires_grad_(False)
            adv_g, feat = adv.generator_loss(pred, target, v_mask)
            ((adv_g + feat) / ga).backward()
            # Discriminator phase: D unfrozen — grads accumulate on D only.
            for p in adv.discriminator.parameters():
                p.requires_grad_(True)
            (adv.discriminator_loss(pred, target, v_mask) / ga).backward()
        assert gen.weight.grad is not None
        gen_grad = gen.weight.grad.clone()
        disc_grad = torch.cat([p.grad.flatten() for p in adv.discriminator.parameters()])
        return gen_grad, disc_grad

    gen_1, disc_1 = run(ga=1, n_micro=1)
    gen_2, disc_2 = run(ga=2, n_micro=2)

    assert torch.allclose(gen_1, gen_2, atol=1e-5, rtol=1e-4)
    assert torch.allclose(disc_1, disc_2, atol=1e-5, rtol=1e-4)

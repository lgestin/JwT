"""Adversarial + feature-matching loss over decoded waveforms.

`AdversarialLoss` wraps a `MultiResolutionSTFTDiscriminator` and exposes the
hinge GAN losses and the L1 feature-matching loss. Like `MelAuxLoss` it is
codec-agnostic: the trainer decodes the predicted and target acoustic features
to waveforms and passes them here.

Rolling-window masking: outside the supervised `v_mask` region the model's
predicted `x_1` is unsupervised, so it must not reach the discriminator. `v_mask`
(acoustic-frame time) is expanded to sample resolution by the codec `hop_length`
and the prediction is replaced by the target outside the window, so both the
adversarial and feature signals — and the gradient back to the generator — are
confined to the supervised region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jwt.model.discriminator import MultiResolutionSTFTDiscriminator


class AdversarialLoss(nn.Module):
    """Hinge adversarial + L1 feature-matching loss over `(B, S)` waveforms."""

    def __init__(self, discriminator: MultiResolutionSTFTDiscriminator, hop_length: int):
        super().__init__()
        self.discriminator = discriminator
        self.hop_length = hop_length

    def _expand_mask(self, v_mask: torch.Tensor, n_samples: int, dtype: torch.dtype) -> torch.Tensor:
        """(B, T) acoustic-frame mask -> (B, S) sample mask via `hop_length`."""
        m = v_mask.repeat_interleave(self.hop_length, dim=-1).to(dtype)
        assert m.shape[-1] == n_samples, (
            f"expanded v_mask ({m.shape[-1]}) must match waveform length "
            f"({n_samples}); v_mask time axis must be the codec frame axis"
        )
        return m

    def generator_loss(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generator-side losses. Discriminator params should be frozen by the
        caller (``requires_grad_(False)``) so this backprops *through* the
        discriminator to the generator without writing grads on the D params.

        Returns ``(adv, feat)``: the mean hinge generator loss and the mean L1
        feature-matching loss, both averaged over resolutions/layers.
        """
        m = self._expand_mask(v_mask, pred_wav.shape[-1], pred_wav.dtype)
        pred_in = pred_wav * m + target_wav.detach() * (1 - m)

        pred_outputs = self.discriminator(pred_in)
        with torch.no_grad():
            target_outputs = self.discriminator(target_wav)

        adv = pred_wav.new_zeros(())
        feat = pred_wav.new_zeros(())
        n_layers = 0
        for (p_logits, p_feats), (_t_logits, t_feats) in zip(pred_outputs, target_outputs):
            adv = adv + F.relu(1.0 - p_logits).mean()
            for pf, tf in zip(p_feats, t_feats):
                feat = feat + F.l1_loss(pf, tf)
                n_layers += 1
        adv = adv / len(pred_outputs)
        feat = feat / max(n_layers, 1)
        return adv, feat

    def discriminator_loss(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminator hinge loss on detached predictions vs. the target.

        The prediction is detached and masked identically to ``generator_loss``
        so the discriminator sees the same in-window fake distribution.
        """
        m = self._expand_mask(v_mask, pred_wav.shape[-1], pred_wav.dtype)
        fake = pred_wav.detach() * m + target_wav.detach() * (1 - m)

        real_outputs = self.discriminator(target_wav)
        fake_outputs = self.discriminator(fake)

        loss = target_wav.new_zeros(())
        for (r_logits, _), (f_logits, _) in zip(real_outputs, fake_outputs):
            loss = loss + F.relu(1.0 - r_logits).mean() + F.relu(1.0 + f_logits).mean()
        return loss / len(real_outputs)

"""Adversarial + feature-matching loss over decoded waveforms.

`AdversarialLoss` wraps a `MultiResolutionSTFTDiscriminator` and exposes the
hinge GAN losses and the L1 feature-matching loss. Like `MelAuxLoss` it is
codec-agnostic: the trainer decodes the predicted and target acoustic features
to waveforms and passes them here.

Two windowing controls, both off by default (so the default is "discriminate the
whole supervised region, unweighted"):

- `window_frames` (K > 0): crop the discriminator input to the K *leading* frames
  of each sample's rolling window — the most-resolved, about-to-be-committed
  frames (t -> 1). Because the rolling window is a narrow fixed-width band at a
  random per-sample offset, cropping to it makes the discriminator's input a small
  fixed length (`K * hop_length` samples), which slashes its memory/compute and
  removes the padding/boundary contamination of running over the whole clip.

- `t_weight_pow` (gamma > 0): weight the generator's adversarial hinge per frame
  by `t ** gamma`. `t` ramps from ~1 at the window's leading edge down toward 0,
  so this down-weights the noisier, harder-to-resolve (and trivially
  discriminable) low-t frames, concentrating adversarial pressure on the frames
  that actually become output audio.

When both are 0 the discriminator sees the whole sequence and the prediction is
replaced by the target outside `v_mask` (so the generator gradient still only
lands inside the supervised window).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jwt.model.discriminator import MultiResolutionSTFTDiscriminator


class AdversarialLoss(nn.Module):
    """Hinge adversarial + L1 feature-matching loss over `(B, S)` waveforms."""

    def __init__(
        self,
        discriminator: MultiResolutionSTFTDiscriminator,
        hop_length: int,
        window_frames: int = 0,
        t_weight_pow: float = 0.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.hop_length = hop_length
        self.window_frames = window_frames
        self.t_weight_pow = t_weight_pow
        if window_frames > 0:
            max_n_fft = max(d.n_fft for d in discriminator.discriminators)
            assert window_frames * hop_length >= max_n_fft, (
                f"disc_window_frames ({window_frames}) * hop_length ({hop_length}) = "
                f"{window_frames * hop_length} samples is shorter than the largest "
                f"discriminator n_fft ({max_n_fft}); the cropped window can't be STFT'd"
            )

    def _expand_mask(self, v_mask: torch.Tensor, n_samples: int, dtype: torch.dtype) -> torch.Tensor:
        """(B, T) acoustic-frame mask -> (B, S) sample mask via `hop_length`."""
        m = v_mask.repeat_interleave(self.hop_length, dim=-1).to(dtype)
        assert m.shape[-1] == n_samples, (
            f"expanded v_mask ({m.shape[-1]}) must match waveform length "
            f"({n_samples}); v_mask time axis must be the codec frame axis"
        )
        return m

    def _crop_leading(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        v_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Crop to the K leading (highest-t) frames of each sample's window.

        The first ``True`` in ``v_mask`` is the window's leading edge
        (acoustic_front + 1); ``t`` decreases from there, so the K frames starting
        at the leading edge are the most-resolved. Returns the cropped
        ``(pred_wav, target_wav, v_mask, t)`` with waveforms of ``K * hop_length``
        samples and masks/timesteps of ``K`` frames.
        """
        T_ext = v_mask.shape[1]
        K, P = self.window_frames, self.hop_length
        # First supervised frame per sample, clamped so the crop stays in bounds.
        start = v_mask.float().argmax(dim=1).clamp(min=0, max=max(T_ext - K, 0))  # (B,)
        frame_off = torch.arange(K, device=v_mask.device)
        sample_off = torch.arange(K * P, device=pred_wav.device)
        fidx = start[:, None] + frame_off[None, :]  # (B, K)
        sidx = start[:, None] * P + sample_off[None, :]  # (B, K*P)
        return (
            torch.gather(pred_wav, 1, sidx),
            torch.gather(target_wav, 1, sidx),
            torch.gather(v_mask, 1, fidx),
            torch.gather(t, 1, fidx),
        )

    def _prepare(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        v_mask: torch.Tensor,
        t: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.window_frames > 0:
            assert t is not None, "t is required when window_frames > 0"
            pred_wav, target_wav, v_mask, t = self._crop_leading(pred_wav, target_wav, v_mask, t)
        m = self._expand_mask(v_mask, pred_wav.shape[-1], pred_wav.dtype)
        return pred_wav, target_wav, v_mask, t, m

    def _hinge_gen(self, logits: torch.Tensor, w_frames: torch.Tensor | None) -> torch.Tensor:
        """Mean generator hinge `relu(1 - D)`, optionally weighted per time frame.

        `w_frames` is a per-acoustic-frame weight `(B, T_frames)`; it is linearly
        resampled to the logits' time axis and broadcast over frequency, so each
        STFT column is scaled by the `t**gamma` of the frame it covers.
        """
        h = F.relu(1.0 - logits)  # (B, 1, F', T')
        if w_frames is None:
            return h.mean()
        Tp = logits.shape[-1]
        w = F.interpolate(w_frames[:, None, :], size=Tp, mode="linear", align_corners=False)
        w = w[:, :, None, :]  # (B, 1, 1, T')
        return (h * w).sum() / w.expand_as(h).sum().clamp(min=1e-8)

    def _frame_weights(self, v_mask: torch.Tensor, t: torch.Tensor | None) -> torch.Tensor | None:
        """`t**gamma` inside the window, 0 outside — or None when weighting is off."""
        if self.t_weight_pow == 0.0 or t is None:
            return None
        return t.clamp(min=0.0).pow(self.t_weight_pow) * v_mask.to(t.dtype)

    def generator_loss(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        v_mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generator-side losses. Discriminator params should be frozen by the
        caller (``requires_grad_(False)``) so this backprops *through* the
        discriminator to the generator without writing grads on the D params.

        Returns ``(adv, feat)``: the (optionally t-weighted) mean hinge generator
        loss and the mean L1 feature-matching loss.
        """
        pred_wav, target_wav, v_mask, t, m = self._prepare(pred_wav, target_wav, v_mask, t)
        pred_in = pred_wav * m + target_wav.detach() * (1 - m)

        pred_outputs = self.discriminator(pred_in)
        with torch.no_grad():
            target_outputs = self.discriminator(target_wav)

        w_frames = self._frame_weights(v_mask, t)
        adv = pred_wav.new_zeros(())
        feat = pred_wav.new_zeros(())
        n_layers = 0
        for (p_logits, p_feats), (_t_logits, t_feats) in zip(pred_outputs, target_outputs):
            adv = adv + self._hinge_gen(p_logits, w_frames)
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
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discriminator hinge loss on detached predictions vs. the target.

        Cropped to the same leading window as ``generator_loss`` so the
        discriminator trains on exactly the slice it judges. The discriminator's
        own classification is left unweighted (standard).
        """
        pred_wav, target_wav, v_mask, _t, m = self._prepare(pred_wav, target_wav, v_mask, t)
        fake = pred_wav.detach() * m + target_wav.detach() * (1 - m)

        real_outputs = self.discriminator(target_wav)
        fake_outputs = self.discriminator(fake)

        loss = target_wav.new_zeros(())
        for (r_logits, _), (f_logits, _) in zip(real_outputs, fake_outputs):
            loss = loss + F.relu(1.0 - r_logits).mean() + F.relu(1.0 + f_logits).mean()
        return loss / len(real_outputs)

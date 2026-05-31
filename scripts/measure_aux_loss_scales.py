"""Measure the on-device magnitudes of the old MelAuxLoss and the new
MultiResComplexSTFTAuxLoss on a trained checkpoint, using one-step rolling-
window predictions on real audio. Prints both losses and the lambda value that
matches the new loss's gradient contribution to the old mel weight.

Usage:
    uv run python scripts/measure_aux_loss_scales.py \
        --checkpoint outputs/run_22khz_raw512/checkpoints/checkpoint.555000.pt \
        --audio-glob 'outputs/run_22khz_raw512/audio/*_clean_step0000000.wav'
"""

import argparse
import glob

import torch
import torch.nn as nn

from jwt.data.audio.stft import MelSpectrogram
from jwt.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from jwt.training.loss import MultiResComplexSTFTAuxLoss


class _LegacyMelAuxLoss(nn.Module):
    """Verbatim copy of the pre-rename MelAuxLoss for side-by-side comparison."""

    def __init__(self, sample_rate: int, hop_length: int, n_mels: int = 80):
        super().__init__()
        self.mel = MelSpectrogram(
            n_fft=4 * hop_length,
            hop_length=hop_length,
            n_mels=n_mels,
            sample_rate=sample_rate,
            window="hann",
            center=False,
            mel_scale="slaney",
        )

    def forward(
        self, pred_wav: torch.Tensor, target_wav: torch.Tensor, v_mask: torch.Tensor
    ) -> torch.Tensor:
        log_p = self.mel(pred_wav).clamp(min=1e-5).log()
        log_t = self.mel(target_wav).clamp(min=1e-5).log()
        diff = (log_p - log_t).abs().mean(dim=1)
        return (diff * v_mask).sum() / v_mask.sum().clamp(min=1)


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    assert isinstance(cfg, RollingFlowConfig)
    model = RollingFlowSpeaker(cfg).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return cfg, model, ckpt.get("step", "?")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--audio-glob",
        default="outputs/run_22khz_raw512/audio/*_clean_step0000000.wav",
    )
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--n-fronts", type=int, default=4,
                   help="rolling-window positions to sweep per utterance")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import soundfile as sf

    cfg, model, step = _load_model(args.checkpoint, args.device)
    codec = cfg.codec.codec.to(args.device).eval()
    schedule = cfg.timestep_schedule.schedule
    n = cfg.n_denoising_steps

    mel_loss = _LegacyMelAuxLoss(
        sample_rate=args.sample_rate, hop_length=codec.hop_length
    ).to(args.device)
    stft_loss = MultiResComplexSTFTAuxLoss(hop_length=codec.hop_length).to(args.device)

    paths = sorted(glob.glob(args.audio_glob))
    assert paths, f"no audio matched {args.audio_glob!r}"
    print(f"checkpoint step={step}, codec={cfg.codec.value}, "
          f"hop={codec.hop_length}, n_steps={n}")
    print(f"{len(paths)} wavs, {args.n_fronts} fronts/wav "
          f"→ {len(paths) * args.n_fronts} measurements")
    print()

    mel_vals, stft_vals = [], []
    for path in paths:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        assert sr == args.sample_rate, f"{path}: sr={sr} != {args.sample_rate}"
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        w_gt = torch.from_numpy(wav).to(args.device).unsqueeze(0)
        x_1 = codec.normalize(codec.encode(w_gt))
        T_ac = x_1.shape[-1]

        # Sweep `acoustic_front` across the sequence so the rolling window lands
        # in different content (start/middle/end).
        usable = max(1, T_ac - n)
        fronts = [int(i * usable / max(1, args.n_fronts - 1)) for i in range(args.n_fronts)]

        # Token sentinel — text isn't material for this scale measurement at the
        # supervised window's high-t positions; left as a single pad token.
        tokens = torch.zeros(1, 1, dtype=torch.long, device=args.device)
        text_mt = MaskedTensor(
            values=tokens.unsqueeze(1),
            mask=torch.ones(1, 1, dtype=torch.bool, device=args.device),
        )

        for front in fronts:
            ac_idx = torch.arange(T_ac, device=args.device).unsqueeze(0)
            progress = torch.clamp(1.0 - (ac_idx - front).float() / (n - 1), 0.0, 1.0)
            t_per_pos = schedule.timestep(progress)
            t_b = t_per_pos.unsqueeze(1)

            x_0 = cfg.noise_scale * torch.randn_like(x_1)
            x_t = (1.0 - t_b) * x_0 + t_b * x_1

            noisy_mt = MaskedTensor(
                values=x_t,
                mask=torch.ones(1, T_ac, dtype=torch.bool, device=args.device),
            )
            pred = model.forward(text_mt, noisy_mt, t_per_pos)
            # JWT: pred IS x_1_pred in (B, T, D).
            x_1_pred = pred.transpose(1, 2)

            # Match training: v_mask is the supervision window.
            v_mask = (
                (ac_idx > front)
                & (ac_idx < front + n)
            ).to(x_1.dtype)

            with torch.autocast(device_type=args.device, enabled=False):
                pred_wav = codec.decode(codec.unnormalize(x_1_pred.float()))
                target_wav = codec.decode(codec.unnormalize(x_1.float()))
                mel_l1 = mel_loss(pred_wav, target_wav, v_mask)
                stft_l1, _ = stft_loss(pred_wav, target_wav, v_mask)
            mel_vals.append(mel_l1.item())
            stft_vals.append(stft_l1.item())

    mel_t = torch.tensor(mel_vals)
    stft_t = torch.tensor(stft_vals)
    print(f"{'':<32}{'mean':>10}{'median':>10}{'std':>10}{'min':>10}{'max':>10}")
    print(f"{'log-mel L1 (old, magnitude)':<32}"
          f"{mel_t.mean():>10.4f}{mel_t.median():>10.4f}"
          f"{mel_t.std():>10.4f}{mel_t.min():>10.4f}{mel_t.max():>10.4f}")
    print(f"{'complex STFT L1 (new)':<32}"
          f"{stft_t.mean():>10.4f}{stft_t.median():>10.4f}"
          f"{stft_t.std():>10.4f}{stft_t.min():>10.4f}{stft_t.max():>10.4f}")

    ratio = mel_t.mean() / stft_t.mean()
    old_lambda = 0.1  # the value in configs/runpod.yaml the checkpoint trained with
    new_lambda = old_lambda * ratio
    print()
    print(f"mean-ratio mel/stft = {ratio:.4f}")
    print(
        f"To match the gradient magnitude of aux_mel_weight={old_lambda} on the "
        f"old loss, set aux_stft_weight ≈ {new_lambda:.4f}"
    )


if __name__ == "__main__":
    main()

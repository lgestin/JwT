"""Gradio demo server for a trained RollingFlowSpeaker checkpoint.

Usage:
    uv run --extra demo --extra bigvgan python scripts/demo.py \
        --checkpoint outputs/run2/checkpoints/checkpoint.best.pt \
        --vocab-path data/vocabulary.json
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch
from simple_parsing import ArgumentParser

from jwt.data.audio.codecs import BigVGAN
from jwt.data.text import Phonemizer, Tokenizer, Vocabulary
from jwt.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Args:
    checkpoint: str
    vocab_path: str = "data/vocabulary.json"
    device: str = "cuda"
    # Server
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


def _load_model(checkpoint_path: str, device: torch.device) -> tuple[RollingFlowSpeaker, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no 'config' entry. "
            "Retrain to embed the model config in the checkpoint."
        )
    cfg = ckpt["config"]
    assert isinstance(cfg, RollingFlowConfig), (
        f"expected RollingFlowConfig in checkpoint, got {type(cfg).__name__}"
    )
    model = RollingFlowSpeaker(cfg).to(device)

    state = ckpt["model"]
    # Strip torch.compile's "_orig_mod." prefix if the checkpoint came from a compiled model.
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint {checkpoint_path} (step={ckpt.get('step', '?')})")
    return model, ckpt


def _plot_mel(
    mel: torch.Tensor,
    hop_length: int,
    sample_rate: int,
) -> plt.Figure:
    """Render a log-mel spectrogram."""
    m = mel.detach().cpu().float().numpy()
    T = m.shape[-1]
    duration_s = T * hop_length / sample_rate

    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=110)
    im = ax.imshow(
        m,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        extent=(0.0, duration_s, 0, m.shape[0]),
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title(f"log-mel spectrogram — {T} frames ({duration_s:.2f}s)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def _build_synth_fn(
    model: RollingFlowSpeaker,
    codec: BigVGAN,
    phonemizer: Phonemizer,
    tokenizer: Tokenizer,
    device: torch.device,
):
    sr = codec.required_sample_rate
    assert sr is not None  # BigVGAN's vocoder is locked to 24 kHz
    hop = codec.hop_length

    @torch.inference_mode()
    def synthesize(text: str, seed: int):
        if not text or not text.strip():
            raise ValueError("Please enter some text.")

        phonemes, _ = phonemizer(text)
        token_ids = tokenizer.encode(phonemes)
        if len(token_ids) == 0:
            raise ValueError(f"No tokens produced for text: {text!r}")

        tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        text_mt = MaskedTensor(
            values=tokens.unsqueeze(1),
            mask=torch.ones_like(tokens, dtype=torch.bool),
        )

        # Pin noise to the seed for reproducibility. speak() consumes one
        # frame of x_0 per step as the buffer grows, so size it to max_acoustic_len.
        gen = torch.Generator(device=device).manual_seed(int(seed))
        x_0 = torch.randn(
            1,
            model.cfg.acoustic_dim,
            model.cfg.max_acoustic_len,
            device=device,
            generator=gen,
        )

        acoustic_pred = model.speak(text_mt, codec=codec, x_0=x_0)
        ac_len = int(acoustic_pred.mask[0].sum().item())
        ac = acoustic_pred.values[0, :, :ac_len]  # (acoustic_dim, ac_len), normalized
        ac_unnorm = codec.unnormalize(ac.unsqueeze(0))
        wav = codec.decode(ac_unnorm)[0].squeeze(0)  # (T_audio,)

        n_tokens = len(token_ids)
        duration_s = ac_len * hop / sr
        rate = ac_len / n_tokens if n_tokens else 0.0
        length_info = (
            f"Predicted length: {ac_len} frames ({duration_s:.2f}s)\n"
            f"Tokens: {n_tokens}\n"
            f"Rate: {rate:.2f} frames/token"
        )

        wav_np = wav.detach().cpu().float().numpy()
        fig = _plot_mel(ac_unnorm[0], hop_length=hop, sample_rate=sr)
        return (sr, wav_np), fig, phonemes, length_info

    return synthesize


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    # Import gradio lazily so `--help` works without the demo extra installed.
    import gradio as gr

    device = torch.device(args.device)
    assert Path(args.checkpoint).exists(), f"checkpoint not found: {args.checkpoint}"

    vocab = Vocabulary.from_json(args.vocab_path)
    tokenizer = Tokenizer(vocab)
    phonemizer = Phonemizer()
    codec = BigVGAN().to(device)
    model, ckpt = _load_model(args.checkpoint, device)
    assert model.cfg.vocabulary_size == len(vocab), (
        f"vocab size mismatch: checkpoint has {model.cfg.vocabulary_size}, "
        f"vocab file has {len(vocab)}"
    )

    synthesize = _build_synth_fn(model, codec, phonemizer, tokenizer, device)

    with gr.Blocks(title="RollingFlowSpeaker demo") as demo:
        gr.Markdown("# RollingFlowSpeaker demo")
        gr.Markdown(
            f"Checkpoint: `{args.checkpoint}` (step {ckpt.get('step', '?')}) &middot; "
            f"sample rate: {codec.required_sample_rate} Hz &middot; "
            f"n_denoising_steps: {model.cfg.n_denoising_steps}"
        )
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    label="Text",
                    lines=3,
                    value="The quick brown fox jumps over the lazy dog.",
                )
                seed = gr.Number(label="Seed", value=0, precision=0)
                go = gr.Button("Synthesize", variant="primary")
                phonemes_out = gr.Textbox(label="Phonemes", interactive=False)
                length_out = gr.Textbox(label="Length predictor", interactive=False, lines=3)
            with gr.Column(scale=3):
                audio_out = gr.Audio(label="Synthesized audio", type="numpy")
                mel_out = gr.Plot(label="log-mel spectrogram")

        go.click(
            synthesize,
            inputs=[text_in, seed],
            outputs=[audio_out, mel_out, phonemes_out, length_out],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

"""Gradio demo server for a trained RollingFlowSpeaker checkpoint.

Usage:
    uv run --extra demo --extra bigvgan python scripts/demo.py \
        --checkpoint outputs/run2/checkpoints/checkpoint.best.pt \
        --vocab-path /data/ljspeech/vocabulary.json
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch
from simple_parsing import ArgumentParser

from tts.data.audio.codecs import BigVGAN
from tts.data.text import Phonemizer, Tokenizer, Vocabulary
from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Args:
    checkpoint: str
    vocab_path: str = "/data/ljspeech/vocabulary.json"
    device: str = "cuda"
    # Server
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


def _load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[RollingFlowSpeaker, dict]:
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


def _plot_mel(mel: torch.Tensor, hop_length: int, sample_rate: int) -> plt.Figure:
    """Render a log-mel spectrogram as a labeled matplotlib figure."""
    m = mel.detach().cpu().float().numpy()
    T = m.shape[-1]
    duration_s = T * hop_length / sample_rate

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
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
    ax.set_title("log-mel spectrogram")
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
    sr = codec.sample_rate
    hop = codec.hop_length

    @torch.inference_mode()
    def synthesize(text: str, frames_per_token: int, seed: int):
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
        mel_lens = torch.tensor(
            [max(1, len(token_ids) * int(frames_per_token))],
            dtype=torch.long,
            device=device,
        )

        # Pin noise to the seed for reproducibility.
        gen = torch.Generator(device=device).manual_seed(int(seed))
        x_0 = torch.randn(
            1,
            model.cfg.mel_dim,
            int(mel_lens.item()),
            device=device,
            generator=gen,
        )

        mels_pred = model.speak(text_mt, mel_lens, x_0=x_0)
        mel = mels_pred.values[0]  # (mel_dim, T_mel)
        wav = codec.decode(mel.unsqueeze(0))[0].squeeze(0)  # (T_audio,)

        wav_np = wav.detach().cpu().float().numpy()
        fig = _plot_mel(mel, hop_length=hop, sample_rate=sr)
        return (sr, wav_np), fig, phonemes

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
            f"sample rate: {codec.sample_rate} Hz &middot; "
            f"n_denoising_steps: {model.cfg.n_denoising_steps}"
        )
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    label="Text",
                    lines=3,
                    value="The quick brown fox jumps over the lazy dog.",
                )
                frames_per_token = gr.Slider(
                    label="Mel frames per phoneme (controls duration)",
                    minimum=4,
                    maximum=24,
                    step=1,
                    value=12,
                )
                seed = gr.Number(label="Seed", value=0, precision=0)
                go = gr.Button("Synthesize", variant="primary")
                phonemes_out = gr.Textbox(label="Phonemes", interactive=False)
            with gr.Column(scale=3):
                audio_out = gr.Audio(label="Synthesized audio", type="numpy")
                mel_out = gr.Plot(label="log-mel spectrogram")

        go.click(
            synthesize,
            inputs=[text_in, frames_per_token, seed],
            outputs=[audio_out, mel_out, phonemes_out],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

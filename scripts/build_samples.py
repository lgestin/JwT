"""Generate the demo audio set and update docs/index.html.

For each prompt in PROMPTS_SHORT / PROMPTS_LONG:
  - phonemize + tokenize
  - sample audio via model.speak()
  - write docs/samples/<slug>.wav

Then rewrite the "Samples" and "Longer passages" tables in docs/index.html.
The "Known failure modes" section is left alone — fill those in manually.

Usage:
    uv run python scripts/build_samples.py \\
        --checkpoint outputs/run_small_11khz_raw128/checkpoints/checkpoint.best.pt

Add/remove prompts by editing PROMPTS_SHORT / PROMPTS_LONG below.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from simple_parsing import ArgumentParser

from jwt.data.text import Phonemizer, Tokenizer, Vocabulary
from jwt.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)

# Sections shown on docs/index.html, in display order. Each section's <h2>
# heading must match what's in index.html — the script finds the heading,
# then rewrites the next <table>'s body with the rendered rows.
#
# (slug, prompt). Slug becomes <slug>.wav. Edit freely — renaming a slug
# here renames the file and the audio src in the rendered <tr>.
SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Phonetic diagnostics",
        [
            ("quickbrownfox", "The quick brown fox jumps over the lazy dog."),
            ("seashells", "She sells seashells by the seashore."),
            ("peter_piper", "Peter Piper picked a peck of pickled peppers."),
            ("birch_canoe", "The birch canoe slid on the smooth planks."),
        ],
    ),
    (
        "Famous lines",
        [
            ("best_of_times", "It was the best of times, it was the worst of times."),
            ("to_be", "To be, or not to be, that is the question."),
            (
                "happy_families",
                "All happy families are alike; each unhappy family is unhappy in its own way.",
            ),
        ],
    ),
    (
        "Prosody",
        [
            (
                "woodchuck",
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            ),
            ("came_saw_conquered", "We came, we saw, we conquered."),
            (
                "matter_unresolved",
                "Yes, but, as I was saying, the matter remains unresolved.",
            ),
        ],
    ),
    (
        "Long passages",
        [
            (
                "hobbit",
                "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",  # noqa: E501
            ),
            (
                "atlantic",
                "The Atlantic Ocean is the second largest of the world's five oceans, covering roughly one-fifth of the Earth's surface.",  # noqa: E501
            ),
            (
                "forest_dawn",
                "He had been told by many that the journey would be difficult, but nothing had prepared him for the silence of the forest at dawn.",  # noqa: E501
            ),
        ],
    ),
    (
        "Self-referential",
        [
            (
                "no_codec",
                "This audio was generated frame by frame, with no codec and no vocoder.",
            ),
            (
                "flow_matching",
                "Flow matching learns to transport noise toward data along a continuous trajectory.",  # noqa: E501
            ),
            (
                "one_patch",
                "The model speaking these words predicts raw waveform samples, one patch at a time.",  # noqa: E501
            ),
        ],
    ),
]


@dataclass
class Args:
    checkpoint: str
    vocab_path: str = "data/vocabulary.json"
    output_dir: str = "docs/samples"
    index_path: str = "docs/index.html"
    sample_rate: int = 11025  # used when codec.required_sample_rate is None (RawAudio)
    device: str = "cuda"
    seed: int = 0


def load_model(checkpoint_path: str, device: torch.device) -> tuple[RollingFlowSpeaker, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no 'config' entry.")
    cfg = ckpt["config"]
    assert isinstance(cfg, RollingFlowConfig), (
        f"expected RollingFlowConfig in checkpoint, got {type(cfg).__name__}"
    )
    model = RollingFlowSpeaker(cfg).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {checkpoint_path} (step={ckpt.get('step', '?')}, codec={cfg.codec})")
    return model, ckpt


def normalize_text(text: str) -> str:
    """Map punctuation the tokenizer's vocabulary doesn't cover onto something
    it does. Em/en dashes pass through phonemization unchanged and would crash
    Tokenizer.encode (KeyError on '—'). Treat them as comma-strength breaks."""
    return text.replace("—", ",").replace("–", ",")  # noqa: RUF001


@torch.inference_mode()
def synthesize(
    model: RollingFlowSpeaker,
    codec,
    phonemizer: Phonemizer,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, str]:
    text = normalize_text(text)
    phonemes, _ = phonemizer(text)
    token_ids = tokenizer.encode(phonemes)
    if not token_ids:
        raise ValueError(f"No tokens produced for text: {text!r}")

    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    text_mt = MaskedTensor(
        values=tokens.unsqueeze(1),
        mask=torch.ones_like(tokens, dtype=torch.bool),
    )

    gen = torch.Generator(device=device).manual_seed(int(seed))
    x_0 = torch.randn(
        1,
        model.cfg.acoustic_dim,
        model.cfg.max_acoustic_len,
        device=device,
        generator=gen,
    )

    acoustic = model.speak(text_mt, codec=codec, x_0=x_0)
    ac_len = int(acoustic.mask[0].sum().item())
    ac = acoustic.values[0, :, :ac_len]
    ac_unnorm = codec.unnormalize(ac.unsqueeze(0))
    wav = codec.decode(ac_unnorm)[0].squeeze(0).detach().cpu().float()
    return wav, phonemes


def render_row(label: str, prompt: str, phonemes: str, slug: str) -> str:
    return (
        "            <tr>\n"
        f'                <td class="label">{html.escape(label)}</td>\n'
        f'                <td class="prompt">{html.escape(prompt)}</td>\n'
        f'                <td class="phonemes">{html.escape(phonemes)}</td>\n'
        '                <td class="player">\n'
        f'                    <audio controls preload="none" src="samples/{html.escape(slug)}.wav"></audio>\n'  # noqa: E501
        "                </td>\n"
        "            </tr>"
    )


def replace_table_after_h2(text: str, h2_label: str, new_rows: str) -> str:
    """Replace the body of the first <table> that follows <h2>{h2_label}</h2>.

    Preserves the surrounding <table> tags and any text between the heading and
    the table (lede paragraphs etc.).
    """
    pattern = re.compile(
        r"(<h2>" + re.escape(h2_label) + r"</h2>.*?<table>)(.*?)(</table>)",
        re.DOTALL,
    )

    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}\n{new_rows}\n        {m.group(3)}"

    new_text, n = pattern.subn(repl, text, count=1)
    if n != 1:
        raise RuntimeError(
            f"Couldn't find <h2>{h2_label}</h2> followed by <table>...</table> in index"
        )
    return new_text


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    device = torch.device(args.device)
    assert Path(args.checkpoint).exists(), f"checkpoint not found: {args.checkpoint}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary.from_json(args.vocab_path)
    tokenizer = Tokenizer(vocab)
    phonemizer = Phonemizer()

    model, _ = load_model(args.checkpoint, device)
    codec = model.cfg.codec.codec.to(device)
    assert model.cfg.vocabulary_size == len(vocab), (
        f"vocab size mismatch: model={model.cfg.vocabulary_size}, file={len(vocab)}"
    )

    # RawAudio codecs accept any rate; BigVGAN locks to 24 kHz. Trust the codec
    # if it has a hard requirement, otherwise use the CLI sample-rate.
    sr = codec.required_sample_rate or args.sample_rate
    print(f"Writing wavs at {sr} Hz to {args.output_dir}/")

    # Generate every section's audio first, accumulate rendered rows per section.
    # Labels are continuous 01..N across all sections.
    rendered_by_section: dict[str, str] = {}
    counter = 0
    for section_name, prompts in SECTIONS:
        print(f"\n[{section_name}]")
        rows = []
        for slug, prompt in prompts:
            counter += 1
            label = f"{counter:02d}"
            wav, phonemes = synthesize(
                model, codec, phonemizer, tokenizer, prompt, device, args.seed
            )
            out = Path(args.output_dir) / f"{slug}.wav"
            torchaudio.save(str(out), wav.unsqueeze(0), sample_rate=sr)
            print(f"  [{label}] {slug}.wav  ({wav.numel() / sr:.2f}s)  {prompt[:60]}")
            rows.append((label, prompt, phonemes, slug))
        rendered_by_section[section_name] = "\n".join(render_row(*r) for r in rows)

    # Rewrite each section's <table> body in-place; failure-modes and the rest
    # of the page are left alone.
    index_path = Path(args.index_path)
    text = index_path.read_text()
    for section_name, _ in SECTIONS:
        text = replace_table_after_h2(text, section_name, rendered_by_section[section_name])
    index_path.write_text(text)

    print(f"\nDone. {counter} wavs written; {index_path} updated.")


if __name__ == "__main__":
    main()

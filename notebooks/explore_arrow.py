"""Marimo notebook: walk through a prepared dataset (arrow shards + meta.json).

Run with:
    uv run marimo edit notebooks/explore_arrow.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Explore a prepared dataset

    Walk a prepared row through every layer of `jwt.data`:

    1. read `meta.json` and the arrow shards with `pyarrow`
    2. let `ArrowTTSSource` materialise a row into `(Audio, Text)`
    3. look at its word alignment
    4. wrap the source in `AudioDataset` and pull a `Sample`
    5. cut an audio prompt at a word boundary and collate a `Batch` with its padding masks
    """)
    return


@app.cell
def _(mo):
    prepared_dir = mo.ui.text(
        value="data/prepared/ljspeech_22.050khz",
        label="prepared dataset directory",
        full_width=True,
    )
    vocab_path = mo.ui.text(
        value="data/vocabulary.json",
        label="vocabulary json",
        full_width=True,
    )
    patch_size = mo.ui.dropdown(
        options={str(p): p for p in (32, 64, 128, 256, 512)},
        value="512",
        label="raw-audio patch size",
    )
    mo.vstack([prepared_dir, vocab_path, patch_size])
    return patch_size, prepared_dir, vocab_path


@app.cell
def _(prepared_dir):
    import pyarrow as pa

    from jwt.data.prepared import read_meta, shard_paths

    meta = read_meta(prepared_dir.value)
    shards = shard_paths(prepared_dir.value)
    table = pa.concat_tables(
        pa.ipc.open_file(pa.memory_map(str(p), "r")).read_all() for p in shards
    )
    return meta, shards, table


@app.cell
def _(meta, mo, shards, table):
    mo.md(f"""
    ## Prepared directory

    - **sample_rate**: {meta.sample_rate} Hz
    - **target_loudness**: {meta.target_loudness} dB
    - **aligner**: `{meta.aligner}`
    - **dropped**: {meta.dropped or "none"}
    - **shards**: {len(shards)}, **rows**: {table.num_rows}
    - **columns**: {", ".join(f"`{c}`" for c in table.column_names)}
    """)
    return


@app.cell
def _(mo, table):
    row_idx = mo.ui.slider(
        start=0,
        stop=table.num_rows - 1,
        value=0,
        label="row",
        show_value=True,
    )
    row_idx
    return (row_idx,)


@app.cell
def _(row_idx, table):
    scalar_cols = [
        "utt_id",
        "dataset",
        "speaker",
        "session",
        "session_idx",
        "duration",
        "text",
        "phonemes",
        "num_samples",
        "sample_rate",
        "loudness",
    ]
    raw_row = {c: table.column(c)[row_idx.value].as_py() for c in scalar_cols}
    return (raw_row,)


@app.cell
def _(mo, raw_row):
    mo.md(
        f"""
        ### Raw row `{raw_row["utt_id"]}`

        - **dataset**: {raw_row["dataset"]}, **speaker**: {raw_row["speaker"]}
        - **session**: {raw_row["session"]} (#{raw_row["session_idx"]})
        - **text**: {raw_row["text"]}
        - **phonemes**: `{raw_row["phonemes"]}`
        - **samples**: {raw_row["num_samples"]:,} @ {raw_row["sample_rate"]} Hz
        - **duration**: {raw_row["duration"]:.2f}s
        - **loudness**: {raw_row["loudness"]:.2f} dB
        """
    )
    return


@app.cell
def _(patch_size, prepared_dir, vocab_path):
    from jwt.data.source import ArrowTTSSource
    from jwt.data.text import Tokenizer, Vocabulary

    tokenizer = Tokenizer(Vocabulary.from_json(vocab_path.value))
    arrow_source = ArrowTTSSource(prepared_dir.value, tokenizer, patch_size.value)
    return arrow_source, tokenizer


@app.cell
def _(arrow_source, mo):
    n_sessions = len(
        {
            (s, x)
            for s, x in zip(arrow_source.speakers, arrow_source.sessions, strict=True)
        }
    )
    mo.md(f"""
    ## Through `ArrowTTSSource`

    - `len(source)` = {len(arrow_source)}
    - hours: {sum(arrow_source.durations) / 3600:.2f}
    - speakers: {len(set(arrow_source.speakers))}, sessions: {n_sessions}
    """)
    return


@app.cell
def _(arrow_source, row_idx):
    audio, text = arrow_source[row_idx.value]
    return audio, text


@app.cell
def _(audio, mo, text, tokenizer):
    phoneme_tokens = tokenizer.encode(text.phonemes)
    mo.md(
        f"""
        ### `Audio`
        - waveform: shape={tuple(audio.waveform.shape)}, dtype=`{audio.waveform.dtype}`
        - sample_rate: {audio.sample_rate}
        - loudness: {audio.loudness:.2f} dB
        - acoustic [1, patch_size, n_frames]: shape={tuple(audio.acoustic.shape)}

        ### `Text`
        - text: {text.text!r}
        - phonemes: `{text.phonemes}`
        - `tokenizer.encode(phonemes)`: len={len(phoneme_tokens)}
        - first 30 tokens: {phoneme_tokens[:30]}
        """
    )
    return


@app.cell
def _(audio, mo):
    import io

    import soundfile as sf

    def wav(waveform, sample_rate):
        """WAV bytes of a [1, S] waveform, for `mo.audio`."""
        buf = io.BytesIO()
        sf.write(buf, waveform.squeeze(0).cpu().numpy(), sample_rate, format="WAV")
        buf.seek(0)
        return buf

    mo.vstack([mo.md("### Listen"), mo.audio(wav(audio.waveform, audio.sample_rate))])
    return (wav,)


@app.cell
def _(arrow_source, mo, row_idx, text):
    words = arrow_source.words(row_idx.value)
    # Each word runs until the next one starts (or to the end of the utterance).
    ends = [(w.text_start, w.phoneme_start) for w in words[1:]]
    ends.append((len(text.text), len(text.phonemes)))
    word_rows = []
    for word, (text_end, phoneme_end) in zip(words, ends, strict=True):
        word_rows.append(
            {
                "word": text.text[word.text_start : text_end],
                "start_s": round(word.start, 3),
                "end_s": round(word.end, 3),
                "phonemes": text.phonemes[word.phoneme_start : phoneme_end],
            }
        )
    mo.vstack(
        [
            mo.md("### Word alignment (`source.words(idx)`)"),
            mo.ui.table(
                word_rows,
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(arrow_source, mo):
    from jwt.data.dataset import AudioDataset

    dataset = AudioDataset(arrow_source, arrow_source.sample_rate)
    sample = dataset[0]
    mo.md(
        f"""
        ## Through `AudioDataset`

        - `len(dataset)` = {len(dataset)}
        - `dataset[0]` returns a `Sample(idx={sample.idx}, audio=..., text=...)`
        - no audio prompt configured: `sample.audio_prompt` = {sample.audio_prompt}
        - `sample.audio.acoustic.shape` = {tuple(sample.audio.acoustic.shape)}
        """
    )
    return (AudioDataset,)


@app.cell
def _(AudioDataset, arrow_source):
    from jwt.data.audio_prompt import AudioPromptConfig

    cut_dataset = AudioDataset(
        arrow_source,
        arrow_source.sample_rate,
        audio_prompt=AudioPromptConfig(p_drop=0, p_other=0),
        seed=0,
    )
    return (cut_dataset,)


@app.cell
def _(arrow_source, cut_dataset, mo, patch_size, row_idx, wav):
    from jwt.data.audio.codecs import RawAudioPatcher

    cut = cut_dataset[row_idx.value]
    prompt_s = (
        cut.audio_prompt.shape[-1] * arrow_source.hop_length / cut.audio.sample_rate
    )
    prompt_wave = RawAudioPatcher(patch_size.value).decode(cut.audio_prompt)
    mo.vstack(
        [
            mo.md(
                f"""
                ### Audio prompt: same-utterance word cut

                `AudioPromptConfig(p_drop=0, p_other=0)`; an empty prompt means no word
                boundary fits the prompt / target length bounds.

                - prompt: shape={tuple(cut.audio_prompt.shape)} ({prompt_s:.2f}s)
                - target: {cut.audio.waveform.shape[-1] / cut.audio.sample_rate:.2f}s
                - target text: {cut.text.text!r}
                - target phonemes: `{cut.text.phonemes}`
                """
            ),
            mo.audio(wav(prompt_wave, cut.audio.sample_rate)),
            mo.audio(wav(cut.audio.waveform, cut.audio.sample_rate)),
        ]
    )
    return


@app.cell
def _(cut_dataset, mo, row_idx):
    import matplotlib.pyplot as plt
    import torch

    from jwt.data.collate import collate

    # The selected utterance first, then its neighbours, so padding shows.
    batch_rows = [(row_idx.value + i) % len(cut_dataset) for i in range(4)]
    batch = collate([cut_dataset[i] for i in batch_rows])
    # Side by side in the order the model reads them: prompt, text, target.
    masks = {
        "audio_prompt_mask": batch.audio_prompt_mask,
        "tokens_mask": batch.tokens_mask,
        "acoustic_mask": batch.acoustic_mask,
    }
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.imshow(
        torch.cat(list(masks.values()), dim=1).numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="gray",
    )
    offset = 0
    for name, mask in masks.items():
        width = mask.shape[1]
        if offset:
            ax.axvline(offset - 0.5, color="red")
        ax.text(offset + width / 2, -0.7, f"{name} ({width})", ha="center")
        offset += width
    ax.set_yticks(range(len(batch_rows)), [f"row {r}" for r in batch_rows])
    ax.set_xticks([])
    fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"""
        ### `collate` into a `Batch`: padding masks

        Row {batch_rows[0]} (top) is the selected utterance; white is valid,
        black is padding.

        - acoustic: {tuple(batch.acoustic.shape)},
          {int(batch.acoustic_mask[0].sum())} valid frames
        - audio_prompt: {tuple(batch.audio_prompt.shape)},
          {int(batch.audio_prompt_mask[0].sum())} valid frames
        - tokens: {tuple(batch.tokens.shape)},
          {int(batch.tokens_mask[0].sum())} valid tokens
        """
            ),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()

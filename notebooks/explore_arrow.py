"""Marimo notebook: walk through the LJSpeech Arrow data pipeline.

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
    # Explore the LJSpeech Arrow source

    Walk an arrow row through every layer of `tts.data`:

    1. open the file with `pyarrow` and look at the row payload
    2. let `ArrowTTSSource` materialise it into `(Audio, Text)`
    3. wrap that in `AudioDataset` and pull a `Sample`
    4. confirm `BigVGAN.encode(waveform)` matches the stored mel
    5. decode the stored mel back into a waveform for A/B listening
    """)
    return


@app.cell
def _(mo):
    import torch

    arrow_path = mo.ui.text(
        value="data/ljspeech_24khz.arrow",
        label="arrow file",
        full_width=True,
    )
    vocab_path = mo.ui.text(
        value="/data/ljspeech/vocabulary.json",
        label="vocabulary json",
        full_width=True,
    )
    device = mo.ui.dropdown(
        options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
        value="cuda" if torch.cuda.is_available() else "cpu",
        label="device",
    )
    mo.vstack([arrow_path, vocab_path, device])
    return arrow_path, device, torch, vocab_path


@app.cell
def _(arrow_path):
    import pyarrow as pa

    arrow_file = pa.memory_map(arrow_path.value, "r")
    table = pa.ipc.open_file(arrow_file).read_all()
    return (table,)


@app.cell
def _(mo, table):
    mo.md(f"""
    ## Arrow file

    - **rows**: {table.num_rows}
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
        "audio_id",
        "text",
        "phonemes",
        "num_samples",
        "sample_rate",
        "loudness",
        "n_mels",
        "n_frames",
    ]
    raw_row = {c: table.column(c)[row_idx.value].as_py() for c in scalar_cols}
    return (raw_row,)


@app.cell
def _(mo, raw_row):
    raw_duration_s = raw_row["num_samples"] / raw_row["sample_rate"]
    mo.md(
        f"""
        ### Raw row `{raw_row["audio_id"]}`

        - **text**: {raw_row["text"]}
        - **phonemes**: `{raw_row["phonemes"]}`
        - **samples**: {raw_row["num_samples"]:,} @ {raw_row["sample_rate"]} Hz ({raw_duration_s:.2f}s)
        - **loudness**: {raw_row["loudness"]:.2f} dB
        - **mel**: {raw_row["n_mels"]} mels x {raw_row["n_frames"]} frames
        """
    )
    return


@app.cell
def _(arrow_path, vocab_path):
    from tts.data.source import ArrowTTSSource
    from tts.data.text import Tokenizer, Vocabulary

    tokenizer = Tokenizer(Vocabulary.from_json(vocab_path.value))
    arrow_source = ArrowTTSSource(arrow_path.value, tokenizer=tokenizer)
    return arrow_source, tokenizer


@app.cell
def _(arrow_source, row_idx):
    audio, text = arrow_source[row_idx.value]
    return audio, text


@app.cell
def _(audio, mo, text, tokenizer):
    phoneme_tokens = tokenizer.encode(text.phonemes)
    mo.md(
        f"""
        ## Through `ArrowTTSSource`

        ### `Audio`
        - waveform: shape={tuple(audio.waveform.shape)}, dtype=`{audio.waveform.dtype}`
        - sample_rate: {audio.sample_rate}
        - duration_s: {audio.duration_s:.3f}
        - loudness: {audio.loudness:.2f} dB
        - mels: shape={tuple(audio.mels.shape)}, dtype=`{audio.mels.dtype}`

        ### `Text`
        - text: {text.text!r}
        - phonemes: `{text.phonemes}`
        - tokens via `tokenizer.encode(phonemes)` (len={len(phoneme_tokens)}, first 30): {phoneme_tokens[:30]}
        """
    )
    return


@app.cell
def _(audio, mo):
    import io

    import soundfile as sf

    original_buf = io.BytesIO()
    sf.write(
        original_buf,
        audio.waveform.squeeze(0).cpu().numpy(),
        audio.sample_rate,
        format="WAV",
    )
    original_buf.seek(0)
    mo.vstack([mo.md("### Listen (original)"), mo.audio(original_buf)])
    return io, sf


@app.cell
def _(audio, mo):
    stored_mel = audio.mels
    mo.md(
        f"""
        ### Stored mel (log-scaled at encode time)

        - shape: {tuple(stored_mel.shape)} (batch x n_mels x n_frames)
        - dtype: `{stored_mel.dtype}`
        - min: {stored_mel.min().item():.3f}, max: {stored_mel.max().item():.3f}, mean: {stored_mel.mean().item():.3f}

        First 4x6 corner:
        ```
        {stored_mel[0, :4, :6].cpu().tolist()}
        ```
        """
    )
    return (stored_mel,)


@app.cell
def _(io, mo, stored_mel):
    import numpy as np
    from PIL import Image

    log_mel = stored_mel[0].cpu().numpy()[::-1, :]
    m_min, m_max = float(log_mel.min()), float(log_mel.max())
    norm = (log_mel - m_min) / (m_max - m_min + 1e-9)

    stops = np.array(
        [[68, 1, 84], [33, 145, 140], [253, 231, 37]],
        dtype=np.float32,
    )
    idx = norm * (len(stops) - 1)
    lo = np.floor(idx).astype(np.int32).clip(0, len(stops) - 2)
    frac = (idx - lo)[..., None]
    rgb = (stops[lo] * (1 - frac) + stops[lo + 1] * frac).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB").resize(
        (min(log_mel.shape[1] * 2, 1600), 400),
        Image.BILINEAR,
    )
    mel_buf = io.BytesIO()
    img.save(mel_buf, format="PNG")
    mel_buf.seek(0)

    mo.vstack(
        [
            mo.md(
                f"### Log-mel spectrogram\n\n"
                f"range `[{m_min:.2f}, {m_max:.2f}]` "
                f"({log_mel.shape[0]} mels x {log_mel.shape[1]} frames; "
                f"low freq at bottom; viridis colourmap)"
            ),
            mo.image(mel_buf),
        ]
    )
    return (np,)


@app.cell
def _(arrow_source, mo):
    from tts.data.dataset import AudioDataset

    dataset = AudioDataset(tts_source=arrow_source, sample_rate=24000)
    sample = dataset[0]
    mo.md(
        f"""
        ## Through `AudioDataset`

        - `len(dataset)` = {len(dataset)}
        - `dataset[0]` returns a `Sample(idx={sample.idx}, audio=..., text=...)`
        - `sample.audio.waveform.shape` = {tuple(sample.audio.waveform.shape)}
        - `sample.audio.sample_rate` = {sample.audio.sample_rate}
        - `sample.audio.loudness` = {sample.audio.loudness:.2f} dB (normalised in `__getitem__`)

        > `collate.py` / `Batch` have pending fixes for variable-length stacking
        > and field names, so a `DataLoader` step is omitted here.
        """
    )
    return


@app.cell
def _(device, mo):
    try:
        from tts.data.audio.codecs import BigVGAN

        bigvgan = BigVGAN().to(device.value).eval()
        bigvgan_error: str | None = None
    except ImportError as exc:
        bigvgan = None
        bigvgan_error = str(exc)

    if bigvgan_error is not None:
        bigvgan_notice = mo.md(
            f"""
            ### BigVGAN unavailable

            ```
            {bigvgan_error}
            ```

            Install with `uv sync --extra bigvgan` and re-run.
            """
        )
    else:
        bigvgan_notice = mo.md(
            f"""
            ### BigVGAN loaded

            - sample_rate: {bigvgan.sample_rate}
            - n_mels: {bigvgan.n_mels}
            - hop_length: {bigvgan.hop_length}
            - device: `{device.value}`
            """
        )
    bigvgan_notice
    return (bigvgan,)


@app.cell
def _(audio, bigvgan, device, mo, stored_mel, torch):
    if bigvgan is None:
        roundtrip_view = mo.md("_skipped: BigVGAN not loaded_")
    else:
        with torch.inference_mode():
            recomputed_mel = bigvgan.encode(audio.waveform.to(device.value)).cpu()
        diff = (recomputed_mel - stored_mel).abs()
        roundtrip_view = mo.md(
            f"""
            ### `BigVGAN.encode(waveform)` vs. stored mel

            - recomputed shape: {tuple(recomputed_mel.shape)}
            - max |diff|: {diff.max().item():.3e}
            - mean |diff|: {diff.mean().item():.3e}
            """
        )
    roundtrip_view
    return


@app.cell
def _(audio, bigvgan, device, io, mo, sf, stored_mel, torch):
    if bigvgan is None:
        decode_view = mo.md("_skipped: BigVGAN not loaded_")
    else:
        with torch.inference_mode():
            decoded = bigvgan.decode(stored_mel.to(device.value)).cpu()
        decoded_buf = io.BytesIO()
        sf.write(
            decoded_buf,
            decoded.squeeze().numpy(),
            audio.sample_rate,
            format="WAV",
        )
        decoded_buf.seek(0)
        decode_view = mo.vstack(
            [
                mo.md(
                    f"""
                    ### BigVGAN decode (stored mel -> waveform)

                    - decoded shape: {tuple(decoded.shape)}
                    - max |amp|: {decoded.abs().max().item():.3f}
                    """
                ),
                mo.audio(decoded_buf),
            ]
        )
    decode_view
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

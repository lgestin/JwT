"""Attention-map probe for validation diagnostics.

`capture_attention` runs the model behind forward hooks that collect each
self-attention layer's attention matrix. It is a validation-only tool:

- The fused SDPA kernel never materializes the attention matrix, so the model
  must be driven with `attention_implementation=TorchAttention`, whose
  `SelfAttention.forward` returns `(out, attn_weights)`.
- The hooks observe `attn_weights` from that tuple; they do not modify the
  output, so the rest of the forward is unaffected.
- The probe forces an *eager* forward by dropping the compiled `forward`
  attribute, so adding/removing hooks never invalidates the compiled graph.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from jwt.model.registers import Registers
from jwt.model.transformer import SelfAttention
from jwt.training.loggers import colorize


class AttentionCollector:
    """Accumulates per-layer head-averaged attention maps.

    With registers on, every layer sees `(B, n + T, n + T)` — the n register
    tokens head the sequence for the whole block stack. Rows are queries,
    columns keys, so the layer-averaged map splits into:

    - `maps` — `(B, T, T)`, real queries over real keys, in packed
      `[text | audio | pad]` coordinates. Rows sum to <= 1, the remainder
      being mass parked on the registers.
    - `seq_to_registers_maps` — `(B, T, n)`, real queries into register keys:
      that parked mass, per register.
    - `registers_to_seq_maps` — `(B, n, T)`, register queries over real keys:
      what each register reads.

    `images` and `scalars` render these per sample given the text/audio
    lengths.
    """

    def __init__(self, n_registers: int = 0) -> None:
        self._maps: torch.Tensor | None = None
        self._seq_mask: torch.Tensor | None = None
        self.n_registers = n_registers

    def record(
        self, attn_weights: torch.Tensor, seq_mask: torch.Tensor | None = None
    ) -> None:
        """`attn_weights`: (B, H, n + T, n + T) — stored whole, head-averaged.
        `seq_mask`: optional (B, n + T) bool, the widened mask the layer saw;
        identical across layers, so the last one recorded is kept.

        The n register rows/cols are kept here rather than dropped on the spot:
        `maps` crops them, but the two register properties report them, and
        they are gone for good once discarded.
        """
        attn_weights = attn_weights.mean(dim=1)
        if torch.is_tensor(self._maps):
            self._maps = torch.cat((self._maps, attn_weights.unsqueeze(0)))
        else:
            self._maps = attn_weights.unsqueeze(0)
        if seq_mask is not None:
            self._seq_mask = seq_mask[:, self.n_registers :]

    @property
    def seq_mask(self) -> torch.Tensor | None:
        """(B, T) bool in packed coordinates: the queries the model treats as
        real, i.e. `Transformer`'s `seq_mask` with the register prefix cropped.
        `None` when the forward ran unmasked or `record` was not given one."""
        return self._seq_mask

    @property
    def maps(self) -> torch.Tensor:
        if self._maps is None:
            raise RuntimeError(
                "attention maps empty: nothing was passed to `record` — run the "
                "model with `TorchAttention`, the fused backends expose no weights"
            )
        n = self.n_registers
        return self._maps[..., n:, n:].mean(dim=0)

    @property
    def seq_to_registers_maps(self) -> torch.Tensor:
        assert self._maps is not None
        n = self.n_registers
        assert n > 0
        return self._maps[..., n:, :n].mean(dim=0)

    @property
    def registers_to_seq_maps(self) -> torch.Tensor:
        assert self._maps is not None
        n = self.n_registers
        assert n > 0
        return self._maps[..., :n, n:].mean(dim=0)

    def images(
        self, text_lens: torch.Tensor, acoustic_lens: torch.Tensor
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Per-sample heatmaps by sample index: the text->audio map, plus the
        register read/write maps when registers are on."""
        images = {
            i: {"attention": img}
            for i, img in attention_images(self.maps, text_lens, acoustic_lens).items()
        }
        if self.n_registers:
            for i, imgs in registers_attention_images(
                self.registers_to_seq_maps,
                self.seq_to_registers_maps,
                text_lens,
                acoustic_lens,
            ).items():
                images.setdefault(i, {}).update(imgs)
        return images

    def scalars(
        self, text_lens: torch.Tensor, acoustic_lens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Register-mass scalars (see `register_mass`); empty without registers."""
        if not self.n_registers:
            return {}
        return registers_mass(
            self.seq_to_registers_maps, text_lens, acoustic_lens, self.seq_mask
        )


@contextmanager
def capture_attention(model: torch.nn.Module) -> Iterator[AttentionCollector]:
    """Hook every self-attention layer and yield an `AttentionCollector`.

    Inside the block, run the model with `attention_implementation=TorchAttention`
    so the hooks have weights to observe. The compiled `forward` is swapped out
    for the eager one for the duration and restored on exit, along with the
    hooks — the model is left exactly as it was found.
    """
    registers = [m for m in model.modules() if isinstance(m, Registers)]
    collector = AttentionCollector(n_registers=sum(r.n for r in registers))

    def hook(_module: torch.nn.Module, args: tuple, output: object) -> None:
        # SelfAttention.forward returns (out, attn_weights); TorchAttention
        # populates attn_weights, SDPA leaves it None.
        attn_weights = output[1] if isinstance(output, tuple) else None
        if attn_weights is None:
            return
        mask = args[2] if len(args) > 2 else None
        seq_mask = None if mask is None else mask.reshape(mask.shape[0], -1)
        collector.record(attn_weights.detach(), seq_mask)  # ty: ignore[unresolved-attribute]

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, SelfAttention)
    ]
    # Drop the compiled `forward` instance attribute so calls fall through to the
    # eager class method; restore it afterwards. A no-op when compile is off.
    compiled_forward = model.__dict__.pop("forward", None)
    try:
        yield collector
    finally:
        if compiled_forward is not None:
            model.forward = compiled_forward
        for h in handles:
            h.remove()


def attention_images(
    attn_map: torch.Tensor,
    text_lens: torch.Tensor,
    acoustic_lens: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Each sample's text->audio attention block as a heatmap image, by index.

    `attn_map` is `(N, T, T)` in packed `[text | audio | pad]` coordinates. For
    sample `i` the block is `attn_map[i, text:text+audio, :text]` transposed to
    text tokens (rows) attended to by audio frames (columns) — min-max
    normalized to a `(3, text, audio)` viridis-colored image. Samples with no
    text or no audio are skipped.
    """
    tl_all = text_lens.tolist()
    al_all = acoustic_lens.tolist()
    images: dict[int, torch.Tensor] = {}
    for i in range(attn_map.shape[0]):
        tl, al = int(tl_all[i]), int(al_all[i])
        if tl == 0 or al == 0:
            continue
        # Transpose audio-query x text-key -> text (rows) x audio (columns).
        block = attn_map[i, tl : tl + al, :tl].T.detach().cpu().float()
        mn, mx = block.min(), block.max()
        block = (block - mn) / (mx - mn).clamp(min=1e-9)
        # Nearest-neighbor upscale to >=256px on the short side: viewers
        # smooth when scaling, so ship the crisp cells at display size.
        k = max(1, -(-256 // min(block.shape)))
        if k > 1:
            block = block.repeat_interleave(k, 0).repeat_interleave(k, 1)
        images[i] = colorize(block, cmap="viridis")
    return images


def registers_mass(
    seq_to_reg: torch.Tensor,
    text_lens: torch.Tensor,
    acoustic_lens: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Mean attention mass real queries park on the registers.

    `seq_to_reg` is `(N, T, n)` in packed `[text | audio | pad]` coordinates.
    Summing over registers gives each query's parked mass (the complement of
    its row sum in `AttentionCollector.maps`); it is averaged over the text
    queries, the audio queries, and both together.

    "Real" queries are `seq_mask` when given — the model's own `(N, T)` mask,
    which also drops the pure-noise frames past the rolling frontier — else
    the `[0, text + audio)` range from the lengths. Text is always the
    leading `text_lens` positions; audio is the rest of the real ones.
    """
    T = seq_to_reg.shape[1]
    pos = torch.arange(T, device=seq_to_reg.device).unsqueeze(0)
    tl = text_lens.unsqueeze(1)
    in_text = pos < tl
    real = seq_mask if seq_mask is not None else pos < tl + acoustic_lens.unsqueeze(1)
    in_audio = real & ~in_text
    mass = seq_to_reg.float().sum(-1)  # (N, T)
    return {
        "register_mass": mass[in_text | in_audio].mean(),
        "register_mass_text": mass[in_text].mean(),
        "register_mass_audio": mass[in_audio].mean(),
    }


def _registers_image(block: torch.Tensor) -> torch.Tensor:
    """Min-max normalize, upscale, colorize — as `attention_images`, for the
    wide-and-short `(n_registers, seq)` register blocks.

    Only the register axis is upscaled: it is n_registers (~16) rows tall, so
    the >=256px rule gives k ~ 16, and applying that to the hundreds-of-frames
    sequence axis too would make every image ~10k px wide.
    """
    block = block.detach().cpu().float()
    mn, mx = block.min(), block.max()
    block = (block - mn) / (mx - mn).clamp(min=1e-9)
    k = max(1, -(-256 // block.shape[0]))
    if k > 1:
        block = block.repeat_interleave(k, 0)
    return colorize(block, cmap="viridis")


def registers_attention_images(
    reg_to_seq: torch.Tensor,
    seq_to_reg: torch.Tensor,
    text_lens: torch.Tensor,
    acoustic_lens: torch.Tensor,
) -> dict[int, dict[str, torch.Tensor]]:
    """Each sample's register attention as two heatmap images, by index.

    `reg_to_seq` is `(N, n, T)` (registers as queries over real keys) and
    `seq_to_reg` `(N, T, n)` (real queries into register keys), both in packed
    `[text | audio | pad]` sequence coordinates. Per sample the pad tail is
    dropped and each block becomes a registers (rows) x sequence (columns)
    image:

    - `registers_to_seq`   — what register r reads from each position
    - `registers_from_seq` — how much each position writes its attention into
      register r
    """
    tl_all = text_lens.tolist()
    al_all = acoustic_lens.tolist()
    images: dict[int, dict[str, torch.Tensor]] = {}
    for i in range(reg_to_seq.shape[0]):
        real = int(tl_all[i]) + int(al_all[i])
        if real == 0:
            continue
        images[i] = {
            "registers_to_seq": _registers_image(reg_to_seq[i, :, :real]),
            "registers_from_seq": _registers_image(seq_to_reg[i, :real, :].T),
        }
    return images

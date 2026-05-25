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

from jwt.model.transformer import SelfAttention
from jwt.training.loggers import Logger


class AttentionCollector:
    """Accumulates the head-averaged attention map, summed over layers.

    Each hooked layer contributes one `(B, T, T)` head-average; `map` returns
    the mean once the forward pass has fired every layer.
    """

    def __init__(self) -> None:
        self._sum: torch.Tensor | None = None
        self._layers = 0

    def record(self, attn_weights: torch.Tensor) -> None:
        """`attn_weights`: (B, H, T, T) — averaged over heads, summed over layers."""
        head_avg = attn_weights.mean(dim=1)  # (B, T, T)
        self._sum = head_avg if self._sum is None else self._sum + head_avg
        self._layers += 1

    @property
    def map(self) -> torch.Tensor:
        """(B, T, T) attention map averaged over heads and layers."""
        if self._sum is None:
            raise RuntimeError("no attention captured — was the forward run?")
        return self._sum / self._layers


@contextmanager
def capture_attention(model: torch.nn.Module) -> Iterator[AttentionCollector]:
    """Hook every self-attention layer and yield an `AttentionCollector`.

    Inside the block, run the model with `attention_implementation=TorchAttention`
    so the hooks have weights to observe. The compiled `forward` is swapped out
    for the eager one for the duration and restored on exit, along with the
    hooks — the model is left exactly as it was found.
    """
    collector = AttentionCollector()

    def hook(_module: torch.nn.Module, _args: tuple, output: object) -> None:
        # SelfAttention.forward returns (out, attn_weights); TorchAttention
        # populates attn_weights, SDPA leaves it None.
        attn_weights = output[1] if isinstance(output, tuple) else None
        if attn_weights is not None:
            collector.record(attn_weights.detach())  # ty: ignore[unresolved-attribute]

    handles = [
        m.register_forward_hook(hook) for m in model.modules() if isinstance(m, SelfAttention)
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


def log_attention_maps(
    logger: Logger,
    attn_map: torch.Tensor,
    text_lens: torch.Tensor,
    acoustic_lens: torch.Tensor,
    step: int,
) -> None:
    """Log each sample's text->audio attention block as a heatmap.

    `attn_map` is `(N, T, T)` in packed `[text | audio | pad]` coordinates. For
    sample `i` the logged block is `attn_map[i, text:text+audio, :text]`
    transposed to text tokens (rows) attended to by audio frames (columns) —
    min-max normalized to a grayscale image under `valid/attention/{i}`.
    """
    tl_all = text_lens.tolist()
    al_all = acoustic_lens.tolist()
    for i in range(attn_map.shape[0]):
        tl, al = int(tl_all[i]), int(al_all[i])
        if tl == 0 or al == 0:
            continue
        # Transpose audio-query x text-key -> text (rows) x audio (columns).
        block = attn_map[i, tl : tl + al, :tl].T.detach().cpu().float()
        mn, mx = block.min(), block.max()
        block = (block - mn) / (mx - mn).clamp(min=1e-9)
        logger.log_image(f"valid/attention/{i}", block.unsqueeze(0), step)

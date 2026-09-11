import torch


class LayerKVCache:
    """One layer's keys/values for the rolling sampler.

    `stage` appends the new tokens to the `length` committed ones and returns
    `[committed | new]` for attention; `commit_(n)` then keeps the first `n`
    of the new tokens, the rest are dropped by the next stage.
    """

    def __init__(self):
        self.length = 0  # committed tokens
        self.k: torch.Tensor | None = None  # (B, H, L, D), L >= length
        self.v: torch.Tensor | None = None

    def stage(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is not None and self.v is not None:
            k = torch.cat([self.k[:, :, : self.length], k], dim=2)
            v = torch.cat([self.v[:, :, : self.length], v], dim=2)
        self.k, self.v = k, v
        return k, v

    def commit_(self, n: int) -> None:
        assert self.k is not None and self.length + n <= self.k.shape[2]
        self.length += n


class KVCache:
    """A `LayerKVCache` per layer plus the tokens' key mask and commit index,
    which every layer shares. `cache[i]` is layer `i`, created on first use."""

    def __init__(self):
        self.layers: dict[int, LayerKVCache] = {}
        self.length = 0  # committed tokens
        self.valid: torch.Tensor | None = None  # (B, L) bool, True = visible key
        self.commit: torch.Tensor | None = None  # (B, L), see `commit_rule`

    def __getitem__(self, layer: int) -> LayerKVCache:
        return self.layers.setdefault(layer, LayerKVCache())

    def stage(
        self, valid: torch.Tensor, commit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new tokens' (B, n) key mask and commit index; return
        `[committed | new]`."""
        if self.valid is not None and self.commit is not None:
            valid = torch.cat([self.valid[:, : self.length], valid], dim=1)
            commit = torch.cat([self.commit[:, : self.length], commit], dim=1)
        self.valid, self.commit = valid, commit
        return valid, commit

    def commit_(self, n: int) -> None:
        """Keep the first `n` staged tokens, in every layer."""
        assert self.valid is not None and self.length + n <= self.valid.shape[1]
        self.length += n
        for layer in self.layers.values():
            layer.commit_(n)

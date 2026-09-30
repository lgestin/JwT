import torch


class LayerKVCache:
    """One layer's keys/values for the rolling sampler.

    `stage` appends the new tokens to the committed ones and returns
    `[committed | new]` for attention; `commit(length)` then keeps the first
    `length` of them and drops the rest.
    """

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None  # (B, H, L, D)
        self.v: torch.Tensor | None = None

    def stage(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is not None and self.v is not None:
            k = torch.cat([self.k, k], dim=2)
            v = torch.cat([self.v, v], dim=2)
        self.k, self.v = k, v
        return k, v

    def commit(self, length: int) -> None:
        assert self.k is not None and self.v is not None
        assert length <= self.k.shape[2]
        self.k, self.v = self.k[:, :, :length], self.v[:, :, :length]


class KVCache:
    """A `LayerKVCache` per layer plus the tokens' `seq_mask` and `commit_index`,
    which every layer shares. `cache[i]` is layer `i`, created on first use."""

    def __init__(self) -> None:
        self.layers: dict[int, LayerKVCache] = {}
        self.length = 0  # committed tokens
        self.seq_mask: torch.Tensor | None = None  # (B, L) bool, True = visible key
        # (B, L) float, inf = never committed, see `commit_rule`
        self.commit_index: torch.Tensor | None = None

    def __getitem__(self, layer: int) -> LayerKVCache:
        return self.layers.setdefault(layer, LayerKVCache())

    def stage(
        self, seq_mask: torch.Tensor, commit_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new tokens' (B, n) `seq_mask` and `commit_index`; return
        `[committed | new]`."""
        if self.seq_mask is not None and self.commit_index is not None:
            seq_mask = torch.cat([self.seq_mask, seq_mask], dim=1)
            commit_index = torch.cat([self.commit_index, commit_index], dim=1)
        self.seq_mask, self.commit_index = seq_mask, commit_index
        return seq_mask, commit_index

    def commit(self) -> None:
        """Keep the staged tokens with a finite `commit_index`, in every layer.

        The batch shares one length, so only the leading block frozen in every
        sample is kept; `speak` feeds every sample the same layout, so it is
        exact there.
        """
        assert self.seq_mask is not None and self.commit_index is not None
        staged = self.commit_index[:, self.length :]
        leading_frozen = torch.isfinite(staged).long().cumprod(dim=1)
        self.length += int(leading_frozen.sum(dim=1).min())
        self.seq_mask = self.seq_mask[:, : self.length]
        self.commit_index = self.commit_index[:, : self.length]
        for layer in self.layers.values():
            layer.commit(self.length)

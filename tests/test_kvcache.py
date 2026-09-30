import math

import pytest
import torch
from torch import nn

from jwt.model.kvcache import KVCache, LayerKVCache
from jwt.model.transformer import Transformer, TransformerConfig

INF = math.inf


def make_model(n_registers: int = 2, num_layers: int = 3) -> Transformer:
    """Small Transformer with open adaLN gates so attention is observable."""
    torch.manual_seed(0)
    model = Transformer(
        TransformerConfig(
            dim=32, num_heads=4, num_layers=num_layers, n_registers=n_registers
        )
    )
    for block in model.blocks:
        nn.init.normal_(block.adaLN.linear.weight, std=0.02)
        nn.init.normal_(block.adaLN.linear.bias, std=0.02)
    return model.eval()


def make_layout(
    B: int, P: int, C: int, W: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed [prefix | clean | window] sequence: x, t, positions, commit_index."""
    T = P + C + W
    x = torch.randn(B, T, 32)
    t = torch.cat([torch.ones(B, P + C), torch.rand(B, W).clamp(max=0.99)], dim=1)
    positions = torch.arange(T).float().expand(B, T)
    commit_index = torch.cat(
        [
            torch.zeros(B, P),
            torch.arange(1, C + 1).float().expand(B, C),
            torch.full((B, W), INF),
        ],
        dim=1,
    )
    return x, t, positions, commit_index


def test_commit_makes_frozen_tokens_blind_to_later_tokens() -> None:
    """Prefix and earlier clean frames ignore later frames; later ones do not."""
    model = make_model()
    B, P, C, W = 2, 3, 3, 4
    x, t, positions, commit_index = make_layout(B, P, C, W)

    out = model(x, t, positions, commit_index=commit_index)
    x2 = x.clone()
    x2[:, P + C :] = torch.randn(B, W, 32)  # new window contents
    x2[:, P + C - 1] = torch.randn(B, 32)  # and a different last clean frame
    out2 = model(x2, t, positions, commit_index=commit_index)

    torch.testing.assert_close(out[:, : P + C - 1], out2[:, : P + C - 1])
    assert not torch.allclose(out[:, P + C - 1 :], out2[:, P + C - 1 :])


def test_without_commit_index_prefix_sees_window() -> None:
    """Without `commit_index` every token attends to the whole sequence."""
    model = make_model()
    B, P, C, W = 2, 3, 3, 4
    x, t, positions, _ = make_layout(B, P, C, W)
    out = model(x, t, positions)
    x2 = x.clone()
    x2[:, P + C :] = torch.randn(B, W, 32)
    assert not torch.allclose(out[:, :P], model(x2, t, positions)[:, :P])


def test_forward_commits_the_frozen_prefix() -> None:
    """Registers, prefix and clean frames are kept; window frames are not."""
    model = make_model(n_registers=2)
    B, P, C, W = 2, 4, 3, 4
    x, t, positions, commit_index = make_layout(B, P, C, W)
    cache = KVCache()
    model(x, t, positions, commit_index=commit_index, cache=cache)
    assert cache.length == 2 + P + C
    for layer in cache.layers.values():
        assert layer.k is not None and layer.k.shape[2] == 2 + P + C


def test_forward_commits_only_the_prefix_frozen_in_every_sample() -> None:
    """The batch shares one cache length, so the shortest frozen prefix wins."""
    model = make_model(n_registers=0)
    commit_index = torch.tensor([[0.0, 0.0, 1.0, INF], [0.0, 1.0, INF, INF]])
    x, t = torch.randn(2, 4, 32), torch.ones(2, 4)
    cache = KVCache()
    model(
        x,
        t,
        torch.arange(4).float().expand(2, 4),
        commit_index=commit_index,
        cache=cache,
    )
    assert cache.length == 2  # sample 1 has only 2 frozen tokens


def test_cached_windowed_forward_matches_full_forward() -> None:
    """Prefill the prefix, then feed only new tokens with the cache: every
    output must match the full-sequence forward under the commit rule."""
    model = make_model()
    B, P, C, W = 2, 4, 3, 4
    x, t, positions, commit_index = make_layout(B, P, C, W)
    seq_mask = torch.ones(B, P + C + W, dtype=torch.bool)
    seq_mask[1, P - 1] = False  # sample 1 has a shorter prefix (padded)
    full = model(x, t, positions, seq_mask=seq_mask, commit_index=commit_index)

    cache = KVCache()
    sl = slice(0, P)
    model(
        x[:, sl],
        t[:, sl],
        positions[:, sl],
        seq_mask=seq_mask[:, sl],
        commit_index=commit_index[:, sl],
        cache=cache,
    )
    assert cache.length == P + 2

    sl = slice(P, P + C + W)
    out = model(
        x[:, sl],
        t[:, sl],
        positions[:, sl],
        commit_index=commit_index[:, sl],
        cache=cache,
    )
    torch.testing.assert_close(out, full[:, sl], atol=1e-5, rtol=1e-4)
    assert cache.length == P + C + 2

    # A fresh window with nothing committed: same as the full forward on
    # [prefix | clean | new window].
    x_w = torch.randn(B, W, 32)
    t_w = torch.rand(B, W).clamp(max=0.99)
    x_full = torch.cat([x[:, : P + C], x_w], dim=1)
    t_full = torch.cat([t[:, : P + C], t_w], dim=1)
    full2 = model(
        x_full, t_full, positions, seq_mask=seq_mask, commit_index=commit_index
    )
    sl = slice(P + C, P + C + W)
    out2 = model(
        x_w, t_w, positions[:, sl], commit_index=commit_index[:, sl], cache=cache
    )
    torch.testing.assert_close(out2, full2[:, sl], atol=1e-5, rtol=1e-4)
    assert cache.length == P + C + 2

    # Roll one frame: the first window token becomes clean and is committed.
    x_r, t_r = x_w.clone(), t_w.clone()
    t_r[:, 0] = 1.0
    commit_index_r = torch.cat(
        [torch.full((B, 1), float(C + 1)), commit_index[:, -(W - 1) :]], 1
    )
    x_full = torch.cat([x[:, : P + C], x_r], dim=1)
    t_full = torch.cat([t[:, : P + C], t_r], dim=1)
    commit_index_full = torch.cat([commit_index[:, : P + C], commit_index_r], dim=1)
    full3 = model(
        x_full, t_full, positions, seq_mask=seq_mask, commit_index=commit_index_full
    )
    out3 = model(x_r, t_r, positions[:, sl], commit_index=commit_index_r, cache=cache)
    torch.testing.assert_close(out3, full3[:, sl], atol=1e-5, rtol=1e-4)
    assert cache.length == P + C + 1 + 2


def test_layer_cache_stages_then_commits() -> None:
    """Staged K/V are kept only up to the committed count."""
    layer = LayerKVCache()
    k1, v1 = torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2)
    k_all, v_all = layer.stage(k1, v1)
    assert torch.equal(k_all, k1) and torch.equal(v_all, v1)

    layer.commit(1)
    k2, v2 = torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2)
    k_all, _ = layer.stage(k2, v2)
    # [committed first token of k1 | k2]; k1's second token was dropped.
    assert torch.equal(k_all, torch.cat([k1[:, :, :1], k2], dim=2))

    with pytest.raises(AssertionError):
        layer.commit(4)  # only 1 committed + 2 staged tokens


def test_forward_without_cache_commits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training passes no cache, so `commit` (and its GPU sync) never runs."""

    def fail(self: KVCache) -> None:
        raise AssertionError("commit called without a cache")

    monkeypatch.setattr(KVCache, "commit", fail)
    model = make_model()
    x, t, positions, commit_index = make_layout(2, 3, 3, 4)
    model(x, t, positions, commit_index=commit_index)

import torch

from tts.model.transformer import (
    Transformer,
    TransformerBlock,
    precompute_freqs_cis,
)


def test_transformer_shape() -> None:
    model = Transformer(dim=64, num_heads=4, num_layers=2)
    x = torch.randn(2, 16, 64)
    y = model(x)
    assert y.shape == (2, 16, 64)


def test_transformer_backward() -> None:
    model = Transformer(dim=32, num_heads=4, num_layers=2)
    x = torch.randn(1, 8, 32, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None


def test_transformer_block_shape() -> None:
    block = TransformerBlock(dim=32, num_heads=4)
    freqs_cis = precompute_freqs_cis(seq_len=8, head_dim=8)
    x = torch.randn(2, 8, 32)
    y = block(x, freqs_cis)
    assert y.shape == x.shape


def test_causal_mask_changes_output() -> None:
    torch.manual_seed(0)
    model = Transformer(dim=32, num_heads=4, num_layers=2)
    x = torch.randn(1, 8, 32)
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1).logical_not()
    y_masked = model(x, mask=mask)
    y_unmasked = model(x)
    assert y_masked.shape == y_unmasked.shape
    assert not torch.allclose(y_masked, y_unmasked)

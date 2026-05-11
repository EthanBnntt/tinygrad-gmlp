"""Unit tests for gmlp_tinygrad."""

from __future__ import annotations

import numpy as np
import pytest
from tinygrad import Tensor

from gmlp_tinygrad import Identity, TinyAttention, SpatialGatingUnit, gMLPLayer, gMLP


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_identity_returns_input(rng: np.random.Generator) -> None:
    x = Tensor(rng.standard_normal((2, 5, 8), dtype=np.float32))
    ident = Identity()
    assert (ident(x).numpy() == x.numpy()).all()


def test_tiny_attention_output_shape(rng: np.random.Generator) -> None:
    b, n, dim, out_dim, attn_dim = 2, 7, 16, 24, 32
    attn = TinyAttention(dim, out_dim, attn_dim=attn_dim, dropout_p=0.0)
    x = Tensor(rng.standard_normal((b, n, dim), dtype=np.float32))
    y = attn(x, mask=None)
    assert y.numpy().shape == (b, n, out_dim)


def test_tiny_attention_with_mask_has_finite_output(rng: np.random.Generator) -> None:
    b, n, dim = 2, 5, 12
    attn = TinyAttention(dim, 18, attn_dim=24, dropout_p=0.0)
    x = Tensor(rng.standard_normal((b, n, dim), dtype=np.float32))
    mask = Tensor(np.ones((b, n, n), dtype=np.bool_))
    y = attn(x, mask)
    out = y.numpy()
    assert np.isfinite(out).all()


def test_spatial_gating_output_shape(rng: np.random.Generator) -> None:
    b, n, hidden = 3, 6, 32
    sgu = SpatialGatingUnit(hidden, n)
    h = Tensor(rng.standard_normal((b, n, hidden), dtype=np.float32))
    out = sgu(h, residual=0.0, mask=None)
    assert out.numpy().shape == (b, n, hidden // 2)


def test_spatial_gating_wrong_sequence_length_raises(rng: np.random.Generator) -> None:
    n, wrong_n, hidden = 8, 10, 16
    sgu = SpatialGatingUnit(hidden, n)
    h = Tensor(rng.standard_normal((2, wrong_n, hidden), dtype=np.float32))
    with pytest.raises(AssertionError):
        sgu(h)


def test_spatial_gating_with_square_mask(rng: np.random.Generator) -> None:
    b, n, hidden = 2, 4, 24
    sgu = SpatialGatingUnit(hidden, n)
    h = Tensor(rng.standard_normal((b, n, hidden), dtype=np.float32))
    mask = Tensor(np.ones((n, n), dtype=np.bool_))
    out = sgu(h, residual=0.0, mask=mask)
    assert out.numpy().shape == (b, n, hidden // 2)
    assert np.isfinite(out.numpy()).all()


def test_gmlp_layer_invalid_hidden_dim_raises() -> None:
    with pytest.raises(AssertionError, match="hidden_dim must be even"):
        gMLPLayer(dim=8, hidden_dim=15, max_seq_len=4)


def test_gmlp_layer_output_shape_with_attention(rng: np.random.Generator) -> None:
    b, n, dim, hidden = 2, 5, 12, 32
    layer = gMLPLayer(dim, hidden, n, attn_dim=16, use_attn=True, dropout_p=0.0)
    x = Tensor(rng.standard_normal((b, n, dim), dtype=np.float32))
    y = layer(x, mask=None)
    assert y.numpy().shape == (b, n, dim)


def test_gmlp_layer_output_shape_without_attention(rng: np.random.Generator) -> None:
    b, n, dim, hidden = 2, 5, 12, 32
    layer = gMLPLayer(dim, hidden, n, use_attn=False, dropout_p=0.0)
    x = Tensor(rng.standard_normal((b, n, dim), dtype=np.float32))
    y = layer(x, mask=None)
    assert y.numpy().shape == (b, n, dim)


def test_gmlp_stack_output_shape_eval_mode(rng: np.random.Generator) -> None:
    b, n, dim, hidden, layers = 2, 6, 14, 28, 3
    model = gMLP(dim, layers, hidden, n, layer_skip_p=0.0)
    x = Tensor(rng.standard_normal((b, n, dim), dtype=np.float32))
    assert x.training is False
    y = model(x, mask=None)
    assert y.numpy().shape == (b, n, dim)


def test_gmlp_stack_forward_with_training_flag(rng: np.random.Generator) -> None:
    b, n, dim, hidden = 2, 4, 10, 20
    model = gMLP(dim, 2, hidden, n, layer_skip_p=0.3)
    x = Tensor(rng.standard_normal((b, n, dim), dtype=np.float32))
    x.training = True
    y = model(x, mask=None)
    assert y.numpy().shape == (b, n, dim)
    assert np.isfinite(y.numpy()).all()

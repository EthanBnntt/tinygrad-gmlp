"""
gMLP Implementation in TinyGrad, with optional attention mechanism.
"""

from einops import einsum
import numpy as np
from tinygrad import Tensor, nn

class TinyAttention:
    """
    A simple, single headed, attention mechanism.
    """
    def __init__(
        self,
        dim: int,
        out_dim: int, 
        attn_dim: int = 64,
        bias: bool = True,
        dropout_p: float = 0.0):

        self.attn_dim = attn_dim
        self.dropout_p = dropout_p

        self.qkv = nn.Linear(dim, 3 * attn_dim, bias=bias)
        self.out_proj = nn.Linear(attn_dim, out_dim, bias=bias)

    def __call__(self, x: Tensor, mask: Tensor = None) -> Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        a = (einsum(q, k, "b n d, b m d -> b n m") / np.sqrt(self.attn_dim))
        a = a.masked_fill(mask.logical_not(), float("-inf")) if mask is not None else a
        a = a.softmax().dropout(self.dropout_p)
        x = einsum(a, v, "b n m, b m d -> b n d")
        return self.out_proj(x)

class SpatialGatingUnit:
    """
    A spatial gating mechanism for neural networks.
    """
    def __init__(self, dim: int, max_seq_len: int):
        self.max_seq_len = max_seq_len

        self.norm = nn.RMSNorm(dim // 2)
        self.gating_matrix = Tensor(np.random.randn(max_seq_len, max_seq_len)).float()

    def __call__(self, x: Tensor, residual: Tensor = 0.0, mask: Tensor = None) -> Tensor:
        assert x.shape[1] == self.max_seq_len, \
            f"sequence length must match gating matrix dimensions: \
                {x.shape[1]} =/= {self.max_seq_len}"

        gating_matrix = self.gating_matrix.masked_fill(mask.logical_not(), 0) \
            if mask is not None else self.gating_matrix

        u, v = x.chunk(2, dim=-1)
        v = einsum(self.norm(v), gating_matrix, "b n d, m n -> b m d") + residual
        return u * v

class Identity:
    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x

class gMLPLayer:
    """
    A single layer of the gMLP model.
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 max_seq_len: int,
                 attn_dim: int = 64,
                 use_attn: bool = True,
                 dropout_p: float = 0.0):
        assert hidden_dim % 2 == 0, "hidden_dim must be even"

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_attn = use_attn
        self.dropout_p = dropout_p

        self.input_norm = nn.RMSNorm(dim)
        self.channel_proj = nn.Linear(dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim // 2, dim)

        self.spatial_gating_unit = SpatialGatingUnit(hidden_dim, max_seq_len)
        self.tiny_attention = TinyAttention(dim, hidden_dim // 2, attn_dim) \
            if use_attn else Identity()

    def __call__(self, x: Tensor, mask: Tensor = None) -> Tensor:
        shortcut = x
        x = self.input_norm(x)
        h = self.channel_proj(x).gelu().dropout(self.dropout_p)
        attn_out = self.tiny_attention(x, mask) if self.use_attn else 0.0
        h = self.spatial_gating_unit(h, attn_out, mask)
        return self.output_proj(h) + shortcut

class gMLP:
    """
    The gMLP model: a stack of gMLP layers.
    """
    def __init__(self,
                 dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 max_seq_len: int, 
                 layer_skip_p: float = 0.0):
        self.layer_skip_p = layer_skip_p
        self.layers = [gMLPLayer(dim, hidden_dim, max_seq_len) for _ in range(num_layers)]

    def __call__(self, x: Tensor, mask: Tensor = None) -> Tensor:
        for layer_idx, layer in enumerate(self.layers):
            skip_layer = Tensor(np.random.rand(x.shape[0], 1, 1)).float() > self.layer_skip_p
            skip_layer = skip_layer if x.training and layer_idx > 0 else 1.0
            x =  skip_layer * layer(x, mask)
        return x

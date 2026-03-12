"""Causal multi-head attention block for hybrid SSM+Attention architectures.

Used with --block hybrid to replace selected SSM layers with attention.
The Mamba-2 paper shows 10% attention layers is optimal.
"""

import mlx.core as mx
import mlx.nn as nn


class AttentionBlock(nn.Module):
    """Causal multi-head attention with optional sliding window.

    Matches SSDBlock interface: (B, L, D) -> (B, L, D) with pre-norm + residual.
    n_heads auto-derived as d_model // 64 (matches SSD d_head=64).
    """

    def __init__(self, d_model, window=None):
        super().__init__()
        self.d_head = 64
        self.n_heads = d_model // 64
        self.window = window
        self.norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x):
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # Single QKV projection, split into 3 tensors of (B, L, D)
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape to (B, H, L, d_head)
        q = q.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Additive causal mask (mask=None is bidirectional in MLX)
        idx = mx.arange(L)
        diff = idx[:, None] - idx[None, :]
        if self.window is not None:
            causal = (diff >= 0) & (diff < self.window)
        else:
            causal = diff >= 0
        mask = mx.where(causal, mx.zeros((L, L)), mx.array(float("-inf")))

        scale = 1.0 / (self.d_head**0.5)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        # Reshape back: (B, H, L, d_head) -> (B, L, D)
        y = y.transpose(0, 2, 1, 3).reshape(B, L, D)
        return residual + self.out_proj(y)

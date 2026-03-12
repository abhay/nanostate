"""Mamba-2 SSD (State Space Duality) layer for nanostate.

Replaces the S4D FFT convolution with chunked matrix multiplications.
This enables selectivity (input-dependent A, B, C) without custom Metal
kernels — all ops are standard MLX (matmul, cumsum, exp, reshape).

The algorithm:
  1. Split sequence into chunks of Q tokens
  2. Intra-chunk: matmul (parallel across all chunks)
  3. Inter-chunk: sequential scan on T/Q states (tiny)
  4. State-to-output: matmul (parallel across all chunks)

Reference: "Transformers are SSMs" (Dao & Gu, 2024)
  https://arxiv.org/abs/2405.21060
"""

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# SSD core algorithm
# ---------------------------------------------------------------------------


def segsum(x):
    """Stable segment sum for 1-semiseparable matrix construction.

    Given a sequence of log-decay values, computes the pairwise cumulative
    sums needed for the causal mask L[i,j] = exp(sum_{k=j+1}^{i} a_k).

    x: (..., T) -> (..., T, T) lower-triangular log-decay matrix
    """
    T = x.shape[-1]
    x_cumsum = mx.cumsum(x, axis=-1)
    # Outer difference: cumsum[i] - cumsum[j] gives sum from j+1 to i
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    # Mask to lower-triangular (causal)
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    x_segsum = mx.where(mask, x_segsum, mx.array(-1e9))
    return x_segsum


def ssd_forward(X, A, B, C, block_len=64):
    """SSD forward pass: chunked matmul replaces parallel scan.

    Args:
        X: (batch, length, n_heads, d_head)  -- input values
        A: (batch, length, n_heads)           -- scalar log-decay per head
        B: (batch, length, n_heads, d_state)  -- input projection
        C: (batch, length, n_heads, d_state)  -- output projection
        block_len: chunk size Q (default 64)

    Returns:
        Y: (batch, length, n_heads, d_head)   -- output
        final_state: (batch, n_heads, d_head, d_state)
    """
    Bsz, L, H, P = X.shape
    N = B.shape[-1]
    Q = block_len
    n_chunks = L // Q

    # Reshape into chunks: (B, L, ...) -> (B, C, Q, ...)
    X = X.reshape(Bsz, n_chunks, Q, H, P)
    A = A.reshape(Bsz, n_chunks, Q, H)
    B = B.reshape(Bsz, n_chunks, Q, H, N)
    C = C.reshape(Bsz, n_chunks, Q, H, N)

    # A needs to be (B, H, C, Q) for cumsum over the Q dimension
    A = A.transpose(0, 3, 1, 2)  # (B, H, C, Q)
    A_cumsum = mx.cumsum(A, axis=-1)  # (B, H, C, Q)

    # === Step 1: Intra-chunk outputs (diagonal blocks) ===
    # L_mask[i,j] = exp(sum of A from j+1 to i) within each chunk
    L_mask = mx.exp(segsum(A))  # (B, H, C, Q, Q)
    # CB^T gram matrix masked by L, then applied to X
    # Decompose the 4-operand einsum into two steps for MLX compatibility
    CB = mx.einsum("bclhn,bcshn->bhcls", C, B)  # (B, H, C, Q, Q)
    Y_diag = mx.einsum("bhcls,bcshp->bclhp", L_mask * CB, X)  # (B, C, Q, H, P)

    # === Step 2: Chunk states ===
    # Decay factors from each position to end of its chunk
    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (B, H, C, Q)
    states = mx.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)  # (B, C, H, P, N)

    # === Step 3: Inter-chunk recurrence ===
    # Prepend zero initial state
    initial_states = mx.zeros_like(states[:, :1])  # (B, 1, H, P, N)
    states = mx.concatenate([initial_states, states], axis=1)  # (B, C+1, H, P, N)
    # Decay between chunk boundaries
    A_chunk_ends = A_cumsum[:, :, :, -1]  # (B, H, C) — total decay per chunk
    A_chunk_padded = mx.pad(A_chunk_ends, [(0, 0), (0, 0), (1, 0)])  # (B, H, C+1)
    decay_chunk = mx.exp(segsum(A_chunk_padded))  # (B, H, C+1, C+1)
    new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)  # (B, C+1, H, P, N)
    states = new_states[:, :-1]  # (B, C, H, P, N) — initial state for each chunk
    final_state = new_states[:, -1]  # (B, H, P, N)

    # === Step 4: State-to-output (off-diagonal blocks) ===
    state_decay_out = mx.exp(A_cumsum)  # (B, H, C, Q)
    Y_off = mx.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Combine and reshape back
    Y = (Y_diag + Y_off).reshape(Bsz, L, H, P)
    return Y, final_state


# ---------------------------------------------------------------------------
# SSD Block (replaces SSMBlock)
# ---------------------------------------------------------------------------


class SSDLayer(nn.Module):
    """SSD selective state space layer.

    Unlike S4D which uses fixed A/B/C and FFT convolution, SSD makes
    A/B/C input-dependent (selective) and uses chunked matmul for training.
    """

    def __init__(self, d_inner, n_heads, d_state=64, chunk_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_inner // n_heads
        self.d_state = d_state
        self.chunk_size = chunk_size

        # Input-dependent projections for A, B, C
        self.a_proj = nn.Linear(d_inner, n_heads, bias=False)
        self.b_proj = nn.Linear(d_inner, d_state, bias=False)
        self.c_proj = nn.Linear(d_inner, d_state, bias=False)

        # Skip connection (D parameter, like S4D)
        self.D = mx.ones((d_inner,))

    def __call__(self, x):
        """x: (batch, seq_len, d_inner) -> (batch, seq_len, d_inner)"""
        B, L, _ = x.shape

        # Input-dependent A, B, C (this is selectivity)
        A = -nn.softplus(self.a_proj(x))  # (B, L, H) — negative log-decay
        B_proj = self.b_proj(x)  # (B, L, N)
        C_proj = self.c_proj(x)  # (B, L, N)

        # Expand B, C to all heads (shared, MVA pattern)
        B_proj = mx.broadcast_to(B_proj[:, :, None, :], (B, L, self.n_heads, self.d_state))
        C_proj = mx.broadcast_to(C_proj[:, :, None, :], (B, L, self.n_heads, self.d_state))

        # Reshape X for multi-head: (B, L, d_inner) -> (B, L, H, P)
        X = x.reshape(B, L, self.n_heads, self.d_head)

        # Pad sequence to multiple of chunk_size
        Q = self.chunk_size
        pad_len = (Q - L % Q) % Q
        if pad_len > 0:
            X = mx.pad(X, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
            A = mx.pad(A, [(0, 0), (0, pad_len), (0, 0)])
            B_proj = mx.pad(B_proj, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
            C_proj = mx.pad(C_proj, [(0, 0), (0, pad_len), (0, 0), (0, 0)])

        # SSD forward
        Y, _ = ssd_forward(X, A, B_proj, C_proj, block_len=Q)

        # Trim padding and reshape back
        if pad_len > 0:
            Y = Y[:, :L]
        y = Y.reshape(B, L, -1)  # (B, L, d_inner)

        # Skip connection
        return y + x * self.D


class SSDBlock(nn.Module):
    """Mamba-2 style gated block with SSD layer.

    LN -> parallel projections (X, Z, A, B, C) -> conv1d -> SSD -> SiLU gate -> out
    """

    def __init__(self, d_model, d_state=64, d_head=64, expand=2, chunk_size=64):
        super().__init__()
        d_inner = d_model * expand
        n_heads = d_inner // d_head

        self.norm = nn.LayerNorm(d_model)

        # Project to expanded dim: X path + Z gate path
        self.in_proj = nn.Linear(d_model, d_inner * 2)

        # Depthwise conv on X path (causal, kernel=4)
        # MLX Conv1d is channels-last: input (B, L, C)
        # We pad manually on the left for causal convolution
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=0, groups=d_inner)
        self.conv_pad = 3  # left-pad for causal

        # SSD layer
        self.ssd = SSDLayer(d_inner, n_heads, d_state=d_state, chunk_size=chunk_size)

        # Output
        self.out_proj = nn.Linear(d_inner, d_model)

    def __call__(self, x):
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # Parallel projections
        xz = self.in_proj(x)
        x_raw, z = mx.split(xz, 2, axis=-1)

        # Causal conv1d on X path: left-pad then conv
        x_padded = mx.pad(x_raw, [(0, 0), (self.conv_pad, 0), (0, 0)])  # (B, L+3, d_inner)
        x_conv = self.conv1d(x_padded)  # (B, L, d_inner)
        x_conv = nn.silu(x_conv)

        # SSD (selective state space)
        y = self.ssd(x_conv)

        # Gate and project
        y = y * nn.silu(z)
        return residual + self.out_proj(y)

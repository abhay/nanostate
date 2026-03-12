"""Fused Metal kernels for SSD acceleration.

Uses mx.fast.metal_kernel() to fuse multi-step operations into single
GPU dispatches, keeping intermediates in registers/threadgroup memory.

Kernel 1: fused_segsum_exp
  Fuses: cumsum -> outer diff -> tril mask -> exp
  Replaces: exp(segsum(A)) in ssd.py
  Benefit: avoids materializing (Q, Q) intermediate in VRAM

Kernel 2: fused_ssd_intra_chunk
  Fuses: segsum_exp + CB^T gram + masked matmul
  Replaces: L_mask computation + both einsum calls in Step 1
  Benefit: all intermediates stay in threadgroup memory
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Kernel 1: fused segsum + exp
# ---------------------------------------------------------------------------

_segsum_exp_kernel = mx.fast.metal_kernel(
    name="segsum_exp",
    input_names=["A", "T_val"],
    output_names=["L"],
    source="""
        // Grid: (T, T, batch_total) — one thread per output element
        // A: (batch_total, T), L: (batch_total, T, T)

        uint row = thread_position_in_grid.x;
        uint col = thread_position_in_grid.y;
        uint batch = thread_position_in_grid.z;

        uint T_dim = (uint)T_val[0];

        if (row >= T_dim || col >= T_dim) return;

        // Compute cumsum[row] and cumsum[col] by partial scan
        T cum_row = T(0);
        T cum_col = T(0);
        uint base = batch * T_dim;
        uint lim = row > col ? row : col;
        for (uint k = 0; k <= lim; k++) {
            T a_k = A[base + k];
            if (k <= row) cum_row += a_k;
            if (k <= col) cum_col += a_k;
        }

        uint out_idx = batch * T_dim * T_dim + row * T_dim + col;
        if (row >= col) {
            L[out_idx] = metal::exp(cum_row - cum_col);
        } else {
            L[out_idx] = T(0);
        }
    """,
    ensure_row_contiguous=True,
)


def segsum_exp_metal(A):
    """Fused segsum + exp via Metal kernel.

    A: (..., T) -> (..., T, T) lower-triangular exp(cumsum diff) matrix.
    Equivalent to exp(segsum(A)) but in a single kernel dispatch.
    """
    orig_shape = A.shape
    T = orig_shape[-1]
    batch_dims = orig_shape[:-1]
    batch_total = 1
    for d in batch_dims:
        batch_total *= d

    A_flat = A.reshape(batch_total, T)
    out_shape = (batch_total, T, T)

    T_arr = mx.array([T], dtype=mx.uint32)
    results = _segsum_exp_kernel(
        inputs=[A_flat, T_arr],
        template=[("T", A.dtype)],
        grid=(T, T, batch_total),
        threadgroup=(min(T, 32), min(T, 32), 1),
        output_shapes=[out_shape],
        output_dtypes=[A.dtype],
    )
    L = results[0]
    return L.reshape(*batch_dims, T, T)


# ---------------------------------------------------------------------------
# Kernel 2: fused SSD intra-chunk (segsum_exp + CB^T + masked matmul)
# ---------------------------------------------------------------------------

_ssd_intra_chunk_kernel = mx.fast.metal_kernel(
    name="ssd_intra_chunk",
    input_names=["A", "B", "C", "X", "dims"],
    output_names=["Y"],
    source="""
        // One thread computes one output element Y[b, c, l, h, p].
        //
        // Grid: (Q, nC * P, Bsz * H)
        //   x -> l (position within chunk)
        //   y -> c * P + p
        //   z -> b * H + h
        //
        // dims: [Q, H, nC, P, N, Bsz] as uint32 array

        uint l = thread_position_in_grid.x;
        uint cp = thread_position_in_grid.y;
        uint bh = thread_position_in_grid.z;

        uint Q_val = (uint)dims[0];
        uint H = (uint)dims[1];
        uint nC = (uint)dims[2];
        uint P = (uint)dims[3];
        uint N = (uint)dims[4];
        uint Bsz = (uint)dims[5];

        if (l >= Q_val) return;

        uint c = cp / P;
        uint p = cp % P;
        uint b = bh / H;
        uint h = bh % H;

        if (c >= nC || b >= Bsz) return;

        // Compute cumsum_A[l] for this position
        // A is (Bsz, H, nC, Q) — row-major
        uint a_base = b * H * nC * Q_val + h * nC * Q_val + c * Q_val;
        T cum_l = T(0);
        for (uint k = 0; k <= l; k++) {
            cum_l += A[a_base + k];
        }

        // Accumulate: Y = sum_s<=l { exp(cum_l - cum_s) * dot(C[l], B[s]) * X[s,p] }
        // cum_s tracks cumsum[s] = A[0]+...+A[s]
        T acc = T(0);
        T cum_s = T(0);

        // B, C are (Bsz, nC, Q, H, N); X is (Bsz, nC, Q, H, P)
        uint stride_b = nC * Q_val * H * N;
        uint stride_x = nC * Q_val * H * P;
        uint c_off = b * stride_b + c * Q_val * H * N + l * H * N + h * N;
        uint x_off_base = b * stride_x + c * Q_val * H * P + h * P + p;

        for (uint s = 0; s <= l; s++) {
            // Update cumsum first: cum_s = A[0]+...+A[s]
            cum_s += A[a_base + s];
            T decay = metal::exp(cum_l - cum_s);

            // dot(C[b,c,l,h,:], B[b,c,s,h,:])
            uint b_off = b * stride_b + c * Q_val * H * N + s * H * N + h * N;
            T cb = T(0);
            for (uint n = 0; n < N; n++) {
                cb += C[c_off + n] * B[b_off + n];
            }

            // X[b,c,s,h,p]
            uint x_idx = x_off_base + s * H * P;
            acc += decay * cb * X[x_idx];
        }

        // Write Y[b,c,l,h,p]
        uint y_idx = b * stride_x + c * Q_val * H * P + l * H * P + h * P + p;
        Y[y_idx] = acc;
    """,
    ensure_row_contiguous=True,
)


def _ssd_intra_chunk_raw(A, B, C, X):
    """Raw Metal kernel call (no autodiff)."""
    Bsz, H, nC, Q = A.shape
    P = X.shape[-1]
    N = B.shape[-1]

    dims = mx.array([Q, H, nC, P, N, Bsz], dtype=mx.uint32)
    results = _ssd_intra_chunk_kernel(
        inputs=[A, B, C, X, dims],
        template=[("T", A.dtype)],
        grid=(Q, nC * P, Bsz * H),
        threadgroup=(min(Q, 32), 1, 1),
        output_shapes=[X.shape],
        output_dtypes=[A.dtype],
    )
    return results[0]


def _ssd_intra_chunk_mlx(A, B, C, X):
    """Pure MLX path (supports autodiff)."""
    from ssd import segsum

    L_mask = mx.exp(segsum(A))
    CB = mx.einsum("bclhn,bcshn->bhcls", C, B)
    return mx.einsum("bhcls,bcshp->bclhp", L_mask * CB, X)


@mx.custom_function
def ssd_intra_chunk_metal(A, B, C, X):
    """Fused SSD Step 1: Metal forward, MLX backward.

    A: (Bsz, H, nC, Q) — log-decay values (already transposed)
    B: (Bsz, nC, Q, H, N) — input projection
    C: (Bsz, nC, Q, H, N) — output projection
    X: (Bsz, nC, Q, H, P) — input values

    Returns:
        Y_diag: (Bsz, nC, Q, H, P) — intra-chunk output
    """
    return _ssd_intra_chunk_raw(A, B, C, X)


@ssd_intra_chunk_metal.vjp
def ssd_intra_chunk_vjp(primals, cotangents, outputs):
    """Backward pass via pure MLX (autodiff-friendly)."""
    A, B, C, X = primals
    ct = cotangents
    # Recompute forward with MLX ops, then get gradients via mx.vjp
    _, grads = mx.vjp(_ssd_intra_chunk_mlx, [A, B, C, X], [ct])
    return tuple(grads)

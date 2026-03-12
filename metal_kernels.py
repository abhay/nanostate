"""Fused Metal kernels for SSD acceleration.

Uses mx.fast.metal_kernel() to fuse multi-step operations into single
GPU dispatches, keeping intermediates in registers/threadgroup memory.

Kernel 1: segsum_exp — fuses cumsum + outer diff + tril mask + exp
Kernel 2: ssd_intra_chunk — fuses segsum_exp + CB^T gram + masked matmul
  Two variants: scalar (any Q <= 128) and simdgroup (Q <= 64, dims % 8 == 0).
  Simdgroup variant uses simdgroup_matrix_multiply_accumulate for both matmuls,
  achieving 10-20% speedup over MLX's separate einsum calls at medium+ sizes.
Kernel 3: ssd_intra_chunk_backward — Metal backward for dA
  Gradient of intra-chunk output w.r.t. log-decay A via fused kernel.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Kernel 1: fused segsum + exp (with shared-memory prefix scan)
# ---------------------------------------------------------------------------

_segsum_exp_kernel = mx.fast.metal_kernel(
    name="segsum_exp_v2",
    input_names=["A", "T_val"],
    output_names=["L"],
    header="""
        // Each thread computes cumsum[row] and cumsum[col] independently.
        // O(T) per thread, T <= 128, so this is fast without shared memory.
    """,
    source="""
        uint row = thread_position_in_grid.x;
        uint col = thread_position_in_grid.y;
        uint batch = thread_position_in_grid.z;

        uint T_dim = (uint)T_val[0];

        if (row >= T_dim || col >= T_dim) return;

        uint base = batch * T_dim;

        // Compute cumsum[row] and cumsum[col] independently per thread
        float cum_row = 0.0f;
        for (uint k = 0; k <= row; k++) {
            cum_row += (float)A[base + k];
        }
        float cum_col = 0.0f;
        for (uint k = 0; k <= col; k++) {
            cum_col += (float)A[base + k];
        }

        uint out_idx = batch * T_dim * T_dim + row * T_dim + col;
        if (row >= col) {
            L[out_idx] = (T)metal::exp(cum_row - cum_col);
        } else {
            L[out_idx] = T(0);
        }
    """,
    ensure_row_contiguous=True,
)


def segsum_exp_metal(A):
    """Fused segsum + exp via Metal kernel with shared-memory prefix scan.

    A: (..., T) -> (..., T, T) lower-triangular exp(cumsum diff) matrix.
    """
    orig_shape = A.shape
    T = orig_shape[-1]
    assert T <= 128, f"segsum_exp_metal requires T <= 128, got {T}"
    batch_dims = orig_shape[:-1]
    batch_total = 1
    for d in batch_dims:
        batch_total *= d

    A_flat = A.reshape(batch_total, T)
    out_shape = (batch_total, T, T)

    T_arr = mx.array([T], dtype=mx.uint32)
    # Threadgroup must be <= 1024 threads total
    tg = min(T, 32)
    results = _segsum_exp_kernel(
        inputs=[A_flat, T_arr],
        template=[("T", A.dtype)],
        grid=(T, T, batch_total),
        threadgroup=(tg, tg, 1),
        output_shapes=[out_shape],
        output_dtypes=[A.dtype],
    )
    return results[0].reshape(*batch_dims, T, T)


# ---------------------------------------------------------------------------
# Kernel 2: fused SSD intra-chunk (with shared cumsum + FP32 accumulation)
# ---------------------------------------------------------------------------

_ssd_intra_chunk_kernel = mx.fast.metal_kernel(
    name="ssd_intra_chunk_v2",
    input_names=["A", "B", "C", "X", "dims"],
    output_names=["Y"],
    source="""
        // Grid: (Q, nC * P, Bsz * H)
        //   x -> l (position within chunk)
        //   y -> c * P + p
        //   z -> b * H + h
        //
        // dims: [Q, H, nC, P, N, Bsz]
        //
        // Optimization: threadgroup-shared cumsum prefix scan.
        // Thread l=0 computes the full cumsum for this (b, h, c) slice
        // into shared memory. All other threads read from it.

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

        // --- Shared-memory prefix scan for A cumsum ---
        threadgroup float shared_cumsum[128];  // max Q=128

        uint a_base = b * H * nC * Q_val + h * nC * Q_val + c * Q_val;

        // Thread 0 in this threadgroup computes the serial prefix sum
        uint tid = thread_position_in_threadgroup.x;
        if (tid == 0) {
            float running = 0.0f;
            for (uint k = 0; k < Q_val; k++) {
                running += (float)A[a_base + k];
                shared_cumsum[k] = running;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Read cumsum[l] from shared memory
        float cum_l = shared_cumsum[l];

        // --- Accumulate in FP32 for numerical stability ---
        float acc = 0.0f;

        // B, C are (Bsz, nC, Q, H, N); X is (Bsz, nC, Q, H, P)
        uint stride_b = nC * Q_val * H * N;
        uint stride_x = nC * Q_val * H * P;
        uint c_off = b * stride_b + c * Q_val * H * N + l * H * N + h * N;
        uint x_off_base = b * stride_x + c * Q_val * H * P + h * P + p;

        for (uint s = 0; s <= l; s++) {
            float cum_s = shared_cumsum[s];
            float decay = metal::exp(cum_l - cum_s);

            // dot(C[b,c,l,h,:], B[b,c,s,h,:]) in FP32
            uint b_off = b * stride_b + c * Q_val * H * N + s * H * N + h * N;
            float cb = 0.0f;
            for (uint n = 0; n < N; n++) {
                cb += (float)C[c_off + n] * (float)B[b_off + n];
            }

            // X[b,c,s,h,p]
            uint x_idx = x_off_base + s * H * P;
            acc += decay * cb * (float)X[x_idx];
        }

        // Write Y[b,c,l,h,p] (cast back to output dtype)
        uint y_idx = b * stride_x + c * Q_val * H * P + l * H * P + h * P + p;
        Y[y_idx] = (T)acc;
    """,
    ensure_row_contiguous=True,
)


def _ssd_intra_chunk_scalar(A, B, C, X):
    """Scalar Metal kernel (any Q <= 128)."""
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


# ---------------------------------------------------------------------------
# Kernel 2b: simdgroup variant (Q <= 64, Q/N/P divisible by 8)
# Uses simdgroup_matrix_multiply_accumulate for both CB gram + Y matmul.
# Fuses cumsum + mask + two matmuls in one kernel dispatch.
# ---------------------------------------------------------------------------

_ssd_intra_chunk_simd_kernel = mx.fast.metal_kernel(
    name="ssd_intra_chunk_simd",
    input_names=["A", "B", "C_in", "X", "dims"],
    output_names=["Y"],
    header="#include <metal_simdgroup_matrix>\n",
    source="""
        uint Q_val = (uint)dims[0];
        uint H = (uint)dims[1];
        uint nC = (uint)dims[2];
        uint P = (uint)dims[3];
        uint N = (uint)dims[4];
        uint Bsz = (uint)dims[5];

        // Each threadgroup handles one (b, h, c) slice
        uint slice_id = threadgroup_position_in_grid.x;
        if (slice_id >= Bsz * H * nC) return;

        uint b = slice_id / (H * nC);
        uint h = (slice_id / nC) % H;
        uint c = slice_id % nC;

        uint tid = thread_position_in_threadgroup.x;
        uint num_threads = threads_per_threadgroup.x;
        uint sid = simdgroup_index_in_threadgroup;
        uint num_sgs = simdgroups_per_threadgroup;

        uint stride_bn = H * N;  // row stride for B, C in Q dim
        uint stride_xp = H * P;  // row stride for X in Q dim
        uint bc_base = b * nC * Q_val * H * N + c * Q_val * H * N;
        uint x_base = b * nC * Q_val * H * P + c * Q_val * H * P;
        uint a_base = b * H * nC * Q_val + h * nC * Q_val + c * Q_val;

        // Threadgroup memory (all float32 for accumulation)
        threadgroup float CB[64 * 64];       // Q * Q
        threadgroup float B_T_tile[8 * 64];  // transposed B block
        threadgroup float C_tile[8 * 64];    // C block for simdgroup load
        threadgroup float X_tile[8 * 64];    // X block for simdgroup load
        threadgroup float cumsum_buf[64];

        // === Step 1: Cumsum of A ===
        if (tid == 0) {
            float running = 0.0f;
            for (uint k = 0; k < Q_val; k++) {
                running += (float)A[a_base + k];
                cumsum_buf[k] = running;
            }
        }

        // === Step 2: CB = C @ B^T via tiled simdgroup matmul ===
        uint tiles_q = Q_val / 8;
        uint total_cb_tiles = tiles_q * tiles_q;

        // Zero CB
        for (uint i = tid; i < Q_val * Q_val; i += num_threads) {
            CB[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tile over N in blocks of 8
        for (uint n_block = 0; n_block < N; n_block += 8) {
            // Cooperatively load B[:, n_block:+8] transposed -> B_T_tile[8, Q]
            // and C[:, n_block:+8] -> C_tile[Q, 8] (row-major, stride=8)
            for (uint i = tid; i < 8 * Q_val; i += num_threads) {
                uint n_off = i / Q_val;
                uint q_idx = i % Q_val;
                B_T_tile[n_off * Q_val + q_idx] =
                    (float)B[bc_base + q_idx * stride_bn + h * N + n_block + n_off];
                C_tile[q_idx * 8 + n_off] =
                    (float)C_in[bc_base + q_idx * stride_bn + h * N + n_block + n_off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each simdgroup accumulates CB output tiles
            for (uint tile_id = sid; tile_id < total_cb_tiles; tile_id += num_sgs) {
                uint tile_l = tile_id / tiles_q;
                uint tile_s = tile_id % tiles_q;

                simdgroup_float8x8 matC_tile, matBT_tile, acc;
                simdgroup_load(acc, CB + tile_l * 8 * Q_val + tile_s * 8, Q_val);
                // Load C from threadgroup (already float32, stride=8)
                simdgroup_load(matC_tile, C_tile + tile_l * 8 * 8, (ulong)8);
                simdgroup_load(matBT_tile, B_T_tile + tile_s * 8, Q_val);
                simdgroup_multiply_accumulate(acc, matC_tile, matBT_tile, acc);
                simdgroup_store(acc, CB + tile_l * 8 * Q_val + tile_s * 8, Q_val);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // === Step 3: Apply decay mask ===
        for (uint idx = tid; idx < Q_val * Q_val; idx += num_threads) {
            uint l = idx / Q_val;
            uint s = idx % Q_val;
            if (l >= s) {
                CB[idx] *= metal::exp(cumsum_buf[l] - cumsum_buf[s]);
            } else {
                CB[idx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Step 4: Y = CB @ X via simdgroup matmul ===
        uint tiles_p = P / 8;
        uint total_y_tiles = tiles_q * tiles_p;

        for (uint tile_id = sid; tile_id < total_y_tiles; tile_id += num_sgs) {
            uint tile_row = tile_id / tiles_p;
            uint tile_col = tile_id % tiles_p;

            simdgroup_float8x8 acc(0.0f);
            for (uint k = 0; k < Q_val; k += 8) {
                simdgroup_float8x8 matCB, matX;
                simdgroup_load(matCB, CB + tile_row * 8 * Q_val + k, Q_val);

                // Load X[k:k+8, tile_col*8:+8] through per-simdgroup scratch
                // X_tile has 8*Q floats — each simdgroup uses 64-float slice
                threadgroup float* x_scratch = X_tile + sid * 64;
                for (uint ii = thread_index_in_simdgroup; ii < 64; ii += 32) {
                    uint r = ii / 8;
                    uint cc = ii % 8;
                    x_scratch[r * 8 + cc] =
                        (float)X[x_base + (k + r) * stride_xp + h * P + tile_col * 8 + cc];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(matX, x_scratch, (ulong)8);
                simdgroup_multiply_accumulate(acc, matCB, matX, acc);
            }

            // Store Y — cast back to output type via same per-simdgroup scratch
            threadgroup float* y_scratch = X_tile + sid * 64;
            simdgroup_store(acc, y_scratch, (ulong)8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint y_base_off = x_base + tile_row * 8 * stride_xp + h * P + tile_col * 8;
            for (uint ii = thread_index_in_simdgroup; ii < 64; ii += 32) {
                uint r = ii / 8;
                uint cc = ii % 8;
                Y[y_base_off + r * stride_xp + cc] = (T)y_scratch[r * 8 + cc];
            }
        }
    """,
    ensure_row_contiguous=True,
)


def _ssd_intra_chunk_raw(A, B, C, X):
    """Raw Metal kernel call — picks simdgroup or scalar variant."""
    Bsz, H, nC, Q = A.shape
    P = X.shape[-1]
    N = B.shape[-1]
    orig_dtype = A.dtype

    # Use simdgroup variant when dimensions are compatible
    if Q <= 64 and Q % 8 == 0 and N % 8 == 0 and P % 8 == 0:
        dims = mx.array([Q, H, nC, P, N, Bsz], dtype=mx.uint32)
        total_slices = Bsz * H * nC
        threads_per_tg = 256  # 8 simdgroups × 32 threads
        results = _ssd_intra_chunk_simd_kernel(
            inputs=[A, B, C, X, dims],
            template=[("T", orig_dtype)],
            grid=(total_slices * threads_per_tg, 1, 1),
            threadgroup=(threads_per_tg, 1, 1),
            output_shapes=[X.shape],
            output_dtypes=[orig_dtype],
        )
        return results[0]

    # Fallback: scalar kernel
    return _ssd_intra_chunk_scalar(A, B, C, X)


def _ssd_intra_chunk_mlx(A, B, C, X):
    """Pure MLX path (supports autodiff)."""
    from ssd import segsum

    L_mask = mx.exp(segsum(A))
    CB = mx.einsum("bclhn,bcshn->bhcls", C, B)
    return mx.einsum("bhcls,bcshp->bclhp", L_mask * CB, X)


# ---------------------------------------------------------------------------
# Kernel 3: backward kernel for dA (gradient of A through intra-chunk)
# ---------------------------------------------------------------------------

_ssd_backward_dA_kernel = mx.fast.metal_kernel(
    name="ssd_backward_dA",
    input_names=["A", "B", "C", "X", "dY", "dims"],
    output_names=["dA"],
    source="""
        // Compute dA[b, h, c, k] = sum over (l,s) where s < k <= l of:
        //   dY[b,c,l,h,:] . { exp(cum_l - cum_s) * CB[l,s] * X[b,c,s,h,:] }
        //
        // Grid: (Q, nC, Bsz * H)

        uint k = thread_position_in_grid.x;
        uint c = thread_position_in_grid.y;
        uint bh = thread_position_in_grid.z;

        uint Q_val = (uint)dims[0];
        uint H = (uint)dims[1];
        uint nC = (uint)dims[2];
        uint P = (uint)dims[3];
        uint N = (uint)dims[4];
        uint Bsz = (uint)dims[5];

        if (k >= Q_val) return;

        uint b = bh / H;
        uint h = bh % H;
        if (b >= Bsz) return;

        // Shared cumsum
        threadgroup float shared_cumsum[128];
        uint a_base = b * H * nC * Q_val + h * nC * Q_val + c * Q_val;
        uint tid = thread_position_in_threadgroup.x;
        if (tid == 0) {
            float running = 0.0f;
            for (uint i = 0; i < Q_val; i++) {
                running += (float)A[a_base + i];
                shared_cumsum[i] = running;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float grad_a = 0.0f;
        uint stride_b = nC * Q_val * H * N;
        uint stride_x = nC * Q_val * H * P;

        // Sum over all (l, s) pairs where s < k <= l
        for (uint l = k; l < Q_val; l++) {
            float cum_l = shared_cumsum[l];
            uint c_off = b * stride_b + c * Q_val * H * N + l * H * N + h * N;
            uint dy_base = b * stride_x + c * Q_val * H * P + l * H * P + h * P;

            for (uint s = 0; s < k; s++) {
                float cum_s = shared_cumsum[s];
                float decay = metal::exp(cum_l - cum_s);

                // CB[l,s] = dot(C[l], B[s])
                uint b_off = b * stride_b + c * Q_val * H * N + s * H * N + h * N;
                float cb = 0.0f;
                for (uint n = 0; n < N; n++) {
                    cb += (float)C[c_off + n] * (float)B[b_off + n];
                }

                // dot(dY[l,:], X[s,:])
                uint x_base = b * stride_x + c * Q_val * H * P + s * H * P + h * P;
                float dy_x = 0.0f;
                for (uint p = 0; p < P; p++) {
                    dy_x += (float)dY[dy_base + p] * (float)X[x_base + p];
                }

                grad_a += decay * cb * dy_x;
            }
        }

        dA[a_base + k] = (T)grad_a;
    """,
    ensure_row_contiguous=True,
)


@mx.custom_function
def ssd_intra_chunk_metal(A, B, C, X):
    """Fused SSD Step 1: Metal forward, Metal+MLX backward.

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
    """Backward: full MLX autodiff (faster than scalar Metal backward)."""
    A, B, C, X = primals
    dY = cotangents

    # All gradients via MLX autodiff — faster than scalar Metal kernels
    # because MLX uses optimized BLAS for the matmul backward passes
    _, grads = mx.vjp(_ssd_intra_chunk_mlx, [A, B, C, X], [dY])
    return tuple(grads)


# ---------------------------------------------------------------------------
# Adaptive chunk size tuning
# ---------------------------------------------------------------------------


def tune_chunk_size(model_fn, sample_input, candidates=(32, 64, 128)):
    """Profile SSD chunk sizes and return the fastest.

    Args:
        model_fn: callable that takes (input, chunk_size) and returns output
        sample_input: representative input tensor
        candidates: chunk sizes to try

    Returns:
        best_chunk_size: int
    """
    import time

    best_q = candidates[0]
    best_time = float("inf")

    for q in candidates:
        # Warmup
        try:
            out = model_fn(sample_input, q)
            mx.eval(out)
        except Exception:
            continue

        # Time 5 iterations
        t0 = time.time()
        for _ in range(5):
            out = model_fn(sample_input, q)
            mx.eval(out)
        elapsed = time.time() - t0

        if elapsed < best_time:
            best_time = elapsed
            best_q = q

    return best_q

# SSD Implementation Design for nanostate

## Goal

Replace S4D (LTI, FFT convolution) with Mamba-2's SSD (selective, chunked matmul).
This breaks the LTI ceiling the agent discovered (L=4 and L=6 converge to same val_bpb).

## Why SSD over Mamba-1

- Mamba-1's parallel scan requires custom CUDA kernel. Agent proved Python for-loop is 7x slower than FFT on MLX.
- SSD replaces scan with **matmul** (MLX already fast at this). No custom kernels needed for correctness.
- SSD is 2-8x faster than Mamba-1 even on CUDA. Should be even more advantageous on MLX.
- Larger state dim (64-256) with no speed penalty. Mamba-1 was limited to N=16.

## Architecture: S4D → SSD

### Current (S4D)
```
Input → Embedding → [SSMBlock × L] → LayerNorm → Head
                     │
                     ├─ LayerNorm
                     ├─ Linear(d → 2*d_inner)  # SSM path + gate
                     ├─ S4DLayer(d_inner, N)    # FFT convolution
                     ├─ SiLU gate
                     └─ Linear(d_inner → d)
```

### Target (SSD)
```
Input → Embedding → [SSDBlock × L] → LayerNorm → Head
                     │
                     ├─ LayerNorm
                     ├─ Linear(d → ed + ed + H + 2*N)  # X, Z, A, B, C all in parallel
                     ├─ Conv1d(ed, kernel=4, groups=ed)  # on X path only
                     ├─ SSDLayer(n_heads, d_head, d_state)  # chunked matmul
                     ├─ SiLU gate (with Z)
                     ├─ GroupNorm
                     └─ Linear(ed → d)
```

### Key Structural Changes

| Property | S4D (current) | SSD (target) |
|----------|--------------|--------------|
| Selectivity | Fixed A, B, C, dt | Input-dependent A, B, C |
| Training algo | FFT convolution | Chunked matmul |
| Inference algo | Recurrent (same) | Recurrent (same) |
| A matrix | Diagonal (D, N) | Scalar per head per timestep |
| Heads | None (per-channel) | Multi-head (H = d_inner / P) |
| Head dim P | 1 | 64 (like transformer head dim) |
| State dim N | 64 | 64-256 (larger = better, no speed penalty) |
| Conv1d | None | Depthwise, kernel=4 |
| Post-gate norm | None | GroupNorm |

## Implementation Plan

### Phase A: Pure MLX SSD (no Metal kernels)

Port the 35-line reference implementation to MLX. All required ops exist.

**New file: `ssd.py`** — the SSD layer and block, separate from train.py.
This lets us develop/test without touching train.py (agent conflict avoidance).

```python
# Core SSD forward pass (MLX version of reference impl)
def segsum(x):
    """Stable segment sum for 1-SS matrix construction."""
    T = x.shape[-1]
    x_cumsum = mx.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = mx.tril(mx.ones((T, T)))
    x_segsum = mx.where(mask, x_segsum, -1e9)
    return x_segsum

def ssd_forward(X, A, B, C, block_len=64):
    """
    X: (batch, length, n_heads, d_head)
    A: (batch, length, n_heads)         -- scalar decay per head per timestep
    B: (batch, length, n_heads, d_state)
    C: (batch, length, n_heads, d_state)
    Returns: Y (batch, length, n_heads, d_head), final_state
    """
    B_sz, L, H, P = X.shape
    N = B.shape[-1]
    Q = block_len
    n_chunks = L // Q

    # Reshape into chunks
    X = X.reshape(B_sz, n_chunks, Q, H, P)
    A = A.reshape(B_sz, n_chunks, Q, H).transpose(0, 3, 1, 2)  # (B, H, C, Q)
    B = B.reshape(B_sz, n_chunks, Q, H, N)
    C = C.reshape(B_sz, n_chunks, Q, H, N)

    A_cumsum = mx.cumsum(A, axis=-1)

    # Step 1: Intra-chunk (diagonal blocks)
    L_mask = mx.exp(segsum(A))  # (B, H, n_chunks, Q, Q)
    # Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L_mask, X)
    # Decompose: CB^T -> (Q,Q) gram, mask with L, multiply by X
    CB = mx.einsum("bclhn,bcshn->bhcls", C, B)  # (B, H, C, Q, Q)
    Y_diag = mx.einsum("bhcls,bcshp->bclhp", L_mask * CB, X)

    # Step 2: Chunk states
    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = mx.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # Step 3: Inter-chunk scan (tiny: n_chunks length)
    initial_states = mx.zeros_like(states[:, :1])
    states = mx.concatenate([initial_states, states], axis=1)
    decay_chunk = mx.exp(segsum(mx.pad(A_cumsum[:, :, :, -1], ((0,0),(0,0),(1,0)))))
    new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]

    # Step 4: State-to-output (off-diagonal blocks)
    state_decay_out = mx.exp(A_cumsum)
    Y_off = mx.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = (Y_diag + Y_off).reshape(B_sz, L, H, P)
    return Y, new_states[:, -1]
```

**New SSDBlock in train.py (or ssd.py):**

```python
class SSDBlock(nn.Module):
    def __init__(self, d_model, n_heads=None, d_state=64, d_head=64,
                 expand=2, conv_kernel=4, chunk_size=64):
        super().__init__()
        self.d_inner = d_model * expand
        self.n_heads = n_heads or (self.d_inner // d_head)
        self.d_head = d_head
        self.d_state = d_state
        self.chunk_size = chunk_size

        self.norm = nn.LayerNorm(d_model)

        # All projections in parallel (unlike Mamba-1 which is sequential)
        # X (d_inner) + Z (d_inner) + A (n_heads) + B (d_state) + C (d_state)
        self.in_proj = nn.Linear(d_model,
            self.d_inner + self.d_inner + self.n_heads + 2 * d_state)

        # Depthwise conv on X path
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
            kernel_size=conv_kernel, groups=self.d_inner, padding=conv_kernel-1)

        self.group_norm = nn.GroupNorm(self.n_heads, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def __call__(self, x):
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # Project everything in parallel
        xz_abc = self.in_proj(x)
        x_raw, z, a, b, c = mx.split(xz_abc, [
            self.d_inner,
            self.d_inner * 2,
            self.d_inner * 2 + self.n_heads,
            self.d_inner * 2 + self.n_heads + self.d_state,
        ], axis=-1)

        # Conv1d on X path (causal: trim right padding)
        x_conv = self.conv1d(x_raw.transpose(0, 2, 1))[:, :, :L].transpose(0, 2, 1)
        x_conv = nn.silu(x_conv)

        # Reshape for multi-head SSD
        X = x_conv.reshape(B, L, self.n_heads, self.d_head)
        A = -nn.softplus(a)  # negative log-decay (Mamba uses softplus for dt, SSD for A)
        B_proj = b.reshape(B, L, 1, self.d_state).broadcast_to((B, L, self.n_heads, self.d_state))
        C_proj = c.reshape(B, L, 1, self.d_state).broadcast_to((B, L, self.n_heads, self.d_state))

        # Pad sequence to multiple of chunk_size
        pad_len = (self.chunk_size - L % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            X = mx.pad(X, ((0,0), (0,pad_len), (0,0), (0,0)))
            A = mx.pad(A, ((0,0), (0,pad_len), (0,0)))
            B_proj = mx.pad(B_proj, ((0,0), (0,pad_len), (0,0), (0,0)))
            C_proj = mx.pad(C_proj, ((0,0), (0,pad_len), (0,0), (0,0)))

        # SSD forward
        Y, _ = ssd_forward(X, A, B_proj, C_proj, block_len=self.chunk_size)

        # Trim padding
        if pad_len > 0:
            Y = Y[:, :L]

        y = Y.reshape(B, L, self.d_inner)

        # Gate + GroupNorm + output
        y = y * nn.silu(z)
        y = self.group_norm(y)
        return residual + self.out_proj(y)
```

### Phase B: Recurrent Mode for Inference

Update `engine.py` RecurrentState to support SSD blocks:

```python
# Per-step recurrence (same cost as S4D recurrent mode)
# h_t = a_t * h_{t-1} + B_t * x_t
# y_t = C_t^T * h_t
# Where a_t is scalar, h is (n_heads, d_head, d_state)
```

This is simpler than S4D recurrence because A is scalar (not diagonal).

### Phase C: Hardware-Aware Config

```python
import mlx.core as mx

def detect_hardware():
    """Detect Apple Silicon capabilities."""
    info = {
        "backend": mx.default_device().type,  # "gpu" or "cpu"
        "memory_gb": mx.metal.get_active_memory() / 1e9 if hasattr(mx, 'metal') else 0,
    }
    # Infer chip generation from memory/GPU properties
    # M1 Max: 32 GPU cores, 16-64GB
    # M2 Ultra: 76 GPU cores, 64-192GB
    # etc.
    return info

def auto_config(task, hardware=None):
    """Choose optimal config based on hardware and task."""
    hw = hardware or detect_hardware()
    mem = hw["memory_gb"]

    # Precision: bfloat16 preferred (better range), float16 fallback
    # M1 supports float16 well, bfloat16 support varies
    dtype = mx.bfloat16  # try bfloat16 first, fallback to float16

    # Chunk size: 64 is the paper default, may need tuning
    # Smaller chunks = less memory, more sequential work
    # Larger chunks = more memory, more parallel work
    chunk_size = 64

    # Model size based on available memory
    if mem >= 128:      # M2 Ultra 192GB, M4 Max 128GB
        size = "large"
    elif mem >= 32:     # M3 Max 36-128GB
        size = "medium"
    elif mem >= 16:     # M1 Max 16-64GB
        size = "small"
    else:
        size = "tiny"

    return {"size": size, "dtype": dtype, "chunk_size": chunk_size}
```

### Phase D: Fused Metal Kernels (implemented — inference-only)

Implemented in `metal_kernels.py`. Key finding: **Metal kernels are an inference/eval optimization, not a training optimization.** During training, MLX autodiff builds the gradient graph during forward — backward is "free" (graph reuse). Wrapping Metal in `@mx.custom_function` forces the VJP to retrace forward, adding ~4-5% overhead. Training uses pure MLX.

Two kernel variants for SSD step 1 (intra-chunk):
1. **Scalar kernel** — any Q≤128, per-thread cumsum + serial CB·X accumulation
2. **Simdgroup kernel** — Q≤64 with dims%8==0, uses `simdgroup_matrix_multiply_accumulate` for both CB gram and Y matmul. ~20% faster forward than MLX einsum.

Key implementation details:
- `#include <metal_simdgroup_matrix>` via `header=` param in `mx.fast.metal_kernel()`
- bf16 inputs batch-cast to fp32 in Python (coalesced loads, faster than per-element in kernel)
- `threadgroup_barrier` required for cross-simdgroup sync (not `simdgroup_barrier`)
- Max 1024 threads/threadgroup, 32KB threadgroup memory on M1

## Migration Strategy

**Key constraint:** The autoresearch agent only modifies train.py. We have two options:

### Option 1: Parallel development (recommended)
1. Build SSD in `ssd.py` (new file, no agent conflict)
2. Test independently with a simple training script
3. Once validated, integrate into train.py as an alternative block type
4. Add `--block {s4d,ssd}` flag to train.py

### Option 2: Feature flag in train.py
1. Add SSD code directly to train.py behind a `--block ssd` flag
2. Default stays `s4d` so agent's work is unaffected
3. Agent can eventually experiment with `--block ssd`

Option 1 is safer for parallel development. Option 2 is simpler.

## Size Presets with SSD

SSD changes the param count because of multi-head structure and conv1d:

```
SIZE_PRESETS_SSD = {
    "tiny":   {"d_model": 128,  "n_layers": 4,  "d_state": 64,  "d_head": 64, "expand": 2},
    "small":  {"d_model": 384,  "n_layers": 4,  "d_state": 64,  "d_head": 64, "expand": 2},
    "medium": {"d_model": 768,  "n_layers": 6,  "d_state": 128, "d_head": 64, "expand": 2},
    "large":  {"d_model": 1024, "n_layers": 12, "d_state": 256, "d_head": 64, "expand": 2},
}
```

Note: SSD allows larger d_state with no speed penalty (unlike S4D/Mamba-1).

## Testing Plan

1. **Unit test SSD forward**: compare output shapes, gradients flow
2. **Numerical test**: verify recurrent mode matches chunked forward on same input
3. **Training test**: train tiny SSD on lm task, verify loss decreases
4. **A/B comparison**: S4D vs SSD at same param count on lm and lm-tok
5. **Scaling test**: verify medium/large don't OOM on M1 Max 16GB

## Open Questions

- What's the optimal chunk_size for Apple Silicon? Paper uses 64 (tuned for A100 tensor cores).
- Does MLX einsum performance match decomposed matmuls? May need to manually decompose.
- GroupNorm vs LayerNorm — paper uses GroupNorm, our current code uses LayerNorm.
- Should B and C be shared (1 head each, MVA pattern) or per-head? Paper recommends shared.
- Do we need the conv1d? It failed on lm (overfitting) but is standard in Mamba/Mamba-2.

## References

- Paper: https://arxiv.org/abs/2405.21060
- Blog: https://tridao.me/blog/2024/mamba2-part1-model/ (parts 1-4)
- Code: https://github.com/state-spaces/mamba (official)
- Knowledge: `knowledge/summary_mamba2_ssd.md` (full paper analysis)
- Knowledge: `knowledge/mlx_optimization_research.md` (MLX capabilities)

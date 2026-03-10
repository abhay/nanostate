# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Paper:** https://arxiv.org/abs/2312.00752
**Authors:** Albert Gu, Tri Dao (December 2023, revised May 2024)

## Core Thesis

Prior SSMs (including S4D, which our `train.py` implements) are **linear time-invariant (LTI)**: the parameters A, B, C, dt are fixed across the entire sequence. This means the model applies the same dynamics to every token regardless of content. Mamba makes B, C, and dt **input-dependent** (functions of the current token), turning the SSM into a **selective** model that can decide what to remember and what to forget based on content.

The key claim: selectivity is what closes the gap between SSMs and Transformers. Mamba-3B matches Transformer-6B on language modeling benchmarks.

## What Our Code Does (for reference)

In `train.py`, `S4DLayer` is a textbook fixed S4D:

```python
# All parameters are static -- same for every token in the sequence:
self.log_A = mx.random.uniform(...)   # (D, N) -- fixed
self.B = mx.random.normal(...)        # (D, N) -- fixed
self.C = mx.random.normal(...)        # (D, N) -- fixed
self.log_dt = mx.zeros(...)           # (D,)   -- fixed
```

Training uses **FFT convolution**: the kernel `K[h,l] = sum_n C*B*exp(dtA*l)*dt` is precomputed for the full sequence length, then convolved with the input via FFT. This is possible *because* the parameters are time-invariant -- the kernel only depends on the lag `l`, not on position or content.

The `SSMBlock` wraps this in a post-norm residual pattern: `SSM -> LN -> MLP -> LN`, with GELU activation in the MLP.

## The Selection Mechanism (S6)

### What changes from S4 to S6

| Parameter | S4D (our code) | Mamba (S6) |
|-----------|---------------|------------|
| A | `(D, N)` fixed param | `(D, N)` fixed param (unchanged) |
| B | `(D, N)` fixed param | `(B, L, N)` = `Linear_N(x)` -- **input-dependent** |
| C | `(D, N)` fixed param | `(B, L, N)` = `Linear_N(x)` -- **input-dependent** |
| dt | `(D,)` fixed param | `(B, L, D)` = `softplus(param + Linear_1(x))` broadcast to D -- **input-dependent** |

A stays fixed. The selection comes from B, C, and especially dt.

### Why dt matters most

The paper's ablation (Figure 13) shows:
- No selection: 10.93 perplexity
- Selective dt only: 9.81
- Selective B,C only: 9.98
- All selective: 8.71

dt controls the step size in discretization. When dt is large, the model focuses on the current input (the discretized B-bar grows, pumping more of the current token into the state). When dt is small, the model ignores the current input and persists the existing state. This is equivalent to a learned gate:

> When N=1, A=-1, B=1: `g_t = sigmoid(Linear(x_t))`, `h_t = (1-g_t)*h_{t-1} + g_t*x_t`

This recovers classical RNN gating. Mamba's insight is that **SSM discretization is the principled foundation of gating**.

### Consequence: no more FFT convolution

Input-dependent parameters mean the convolution kernel changes at every timestep. You can no longer precompute a single kernel and FFT-convolve. The model must use the **recurrent** form:

```
h_t = A_bar * h_{t-1} + B_bar * x_t
y_t = C_t * h_t
```

where A_bar, B_bar, C_t all vary with t. This is computed via a **parallel scan** instead of FFT.

## Hardware-Aware Parallel Scan

The naive recurrence is sequential: O(L) steps. Mamba solves this with three techniques:

1. **Kernel fusion**: Load (dt, A, B, C) from HBM to SRAM, compute the full discretization + recurrence in SRAM, write only the output back. This avoids materializing the full `(B, L, D, N)` state tensor in HBM.

2. **Parallel associative scan**: The recurrence `h_t = a_t * h_{t-1} + b_t` is an associative operation. You can compute it in O(log L) parallel steps using a prefix scan (same algorithm as parallel prefix sum). This is the key to making the recurrent mode fast despite being "sequential" in principle.

3. **Recomputation in backward pass**: Instead of saving all intermediate states for backprop (which would cost O(BLDN) memory), recompute them during the backward pass. This trades compute for memory, similar to gradient checkpointing.

Result: same memory as FlashAttention, linear FLOPs, 3x faster than convolution-based S4 on A100, and 20-40x faster than FlashAttention-2 beyond 2K sequence length.

**For nanostate on MLX**: We don't have a fused CUDA kernel. But the parallel scan algorithm itself is implementable in pure MLX using `mx.cumsum`-style operations or a manual log-depth scan. The key insight is that we need to move from FFT convolution to a scan-based approach if we want selectivity.

## The Mamba Block Architecture

### Current nanostate block (SSMBlock)

```
x -> S4D -> + -> LayerNorm -> MLP(Linear-GELU-Linear) -> + -> LayerNorm -> out
     ^      |                                             |
     |      x (residual)                                  x (residual)
```

Two sub-blocks (SSM + MLP), each with post-norm residual. This is standard S4 block design.

### Mamba block (simplified, no separate MLP)

```
x -> Linear(D -> E*D) -> [Conv1d(k=4)] -> SiLU -> SSM(S6) -> * -> Linear(E*D -> D) -> out
                                                               ^
x -> Linear(D -> E*D) ------------------------------------> SiLU
```

Key differences:
- **Expand-and-gate**: Input is projected to E*D dimensions (E=2), split into two paths. One goes through the SSM, the other is a gating path. They are combined by elementwise multiplication (like GLU/SwiGLU).
- **No separate MLP block**: The gating structure subsumes the MLP. This cuts the block count in half -- one Mamba block replaces SSM+MLP.
- **Short convolution (k=4)**: A depthwise Conv1d before the SSM provides local context. This helps because the SSM itself is causal-recurrent and benefits from a small local receptive field.
- **SiLU activation**: Used on both the SSM path (after conv) and the gate path. SiLU(x) = x * sigmoid(x).
- **No LayerNorm inside the block** (optional, motivated by RetNet). Normalization is applied at the block boundaries.

### Parameter budget comparison

For our current SSMBlock with D=128, N=64, MLP_RATIO=2:
- SSM params: log_A(128*64) + B(128*64) + C(128*64) + D(128) + log_dt(128) = 24,832
- MLP: Linear(128->256) + Linear(256->128) = 65,792
- LayerNorm x2: 512
- Total per block: ~91K

For a Mamba block with D=128, E=2, N=16 (Mamba uses smaller N):
- Input projections: 2 * Linear(128->256) = 65,792
- Conv1d(k=4, groups=256): 1,024
- SSM projections: Linear_N(x) for B, Linear_N(x) for C, Linear_1(x) for dt = ~8K-ish
- Output projection: Linear(256->128) = 32,896
- SSM A: 256*16 = 4,096
- Total per block: ~112K

More parameters in projections, fewer in SSM internals. The SSM does less "work" per parameter but is more expressive due to selectivity.

## Key Experimental Results

### Ablation: selection mechanism dominance
- H3 architecture with S4: 10.30 ppl -> with S6: 8.95 (1.35 improvement)
- Mamba architecture with S4: 10.56 ppl -> with S6: 8.69 (1.87 improvement)
- Selection matters more than architecture choice.

### Initialization
- S4D-Real (our parameterization): 8.71 ppl
- S4D-Lin (complex): 9.16 ppl
- Random real: 8.71 ppl
- **When using selection, real diagonal init (what we already do) matches or beats complex.** HiPPO init may matter less with selectivity. This is a notable finding for nanostate.

### State dimension
- S4 LTI caps out around N=4-8 (no benefit from larger state)
- S6 selective continues improving up to N=16 and beyond
- Selective models can actually use a larger state effectively.

### Scale
- Mamba-130M: 10.56 ppl on Pile
- Mamba-1.4B: 6.80 ppl (matches Transformer++)
- Mamba-2.8B: 63.3 avg on zero-shot benchmarks (approaches GPT-J-6B at 63.0)

### DNA (directly relevant to our --task dna)
- Mamba matches 3-4x larger models on HG38
- On species classification with 99% DNA similarity: near-perfect accuracy up to 1M sequence length
- HyenaDNA (LTI) plateaus -- validates that selection helps discrete modalities

## What to Try in nanostate (Ordered by Expected Impact)

### 1. Selective dt (highest priority, minimal code change)

Make `log_dt` input-dependent. This is the single most impactful change from the ablation.

```python
# Current: self.log_dt = mx.zeros((d_model,))
# Mamba: dt = softplus(dt_param + Linear(x))
self.dt_proj = nn.Linear(d_model, d_model)  # s_delta(x)
self.dt_bias = mx.random.uniform(low=-4.0, high=0.0, shape=(d_model,))
# In forward: dt = softplus(self.dt_proj(u) + self.dt_bias)  # (B, L, D)
```

This breaks FFT convolution. You need to switch to a recurrent scan. Simplest approach: a sequential Python loop over L (slow but correct), then optimize later.

### 2. Selective B and C

Project B and C from the input:

```python
# Current: self.B, self.C are (D, N) fixed parameters
# Mamba: B = Linear_N(x), C = Linear_N(x)
self.B_proj = nn.Linear(d_model, state_dim)  # (B,L,D) -> (B,L,N)
self.C_proj = nn.Linear(d_model, state_dim)
# In forward: B = self.B_proj(u)  # (B, L, N)
#             C = self.C_proj(u)  # (B, L, N)
```

Note the shape change: B and C go from `(D, N)` shared across all channels to `(B, L, N)` -- one set of SSM coefficients per position per batch element, shared across the D channels (or expand to (B,L,D,N) if per-channel is desired).

### 3. Parallel scan implementation for MLX

Replace FFT convolution with a scan. The recurrence per channel h:

```
h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
y_t = C_t @ h_t
```

A parallel associative scan computes this in O(L log L) work and O(log L) depth. In MLX:

```python
def selective_scan(A_bar, B_bar_x, C):
    """
    A_bar: (B, L, D, N) -- discretized diagonal A
    B_bar_x: (B, L, D, N) -- discretized B * input
    C: (B, L, N) -- output projection
    """
    # Sequential fallback (correct, slow):
    h = mx.zeros((batch, d_model, state_dim))
    ys = []
    for t in range(L):
        h = A_bar[:, t] * h + B_bar_x[:, t]
        y = mx.sum(C[:, t, None, :] * h, axis=-1)  # (B, D)
        ys.append(y)
    return mx.stack(ys, axis=1)  # (B, L, D)
```

For a proper parallel scan, use the associative property: `(a1, b1) . (a2, b2) = (a1*a2, a2*b1 + b2)`. This can be implemented with a log-depth tree reduction using `mx.compile` for performance.

### 4. Mamba block (expand-and-gate with SiLU)

Replace `SSMBlock` with a Mamba-style block:

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, state_dim, expand=2, conv_width=4):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)  # split into ssm_path + gate
        self.conv1d = nn.Conv1d(d_inner, d_inner, conv_width, groups=d_inner, padding=conv_width-1)
        self.ssm = SelectiveS6(d_inner, state_dim)
        self.out_proj = nn.Linear(d_inner, d_model)

    def __call__(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_ssm, z = mx.split(xz, 2, axis=-1)
        x_ssm = nn.silu(self.conv1d(x_ssm)[..., :x.shape[1]])  # causal conv
        x_ssm = self.ssm(x_ssm)
        y = x_ssm * nn.silu(z)  # gated output
        return residual + self.out_proj(y)
```

This eliminates the separate MLP block entirely. One Mamba block = one residual unit.

### 5. Incremental approach (recommended order)

Since switching from FFT to scan is a big change, consider this progression:

1. **Keep FFT, add gating**: Add expand-and-gate (SiLU) around the existing S4D layer without changing the SSM itself. This tests whether the Mamba block structure helps even with fixed parameters. Low risk.

2. **Switch to sequential scan**: Replace `kernel()` + FFT with an explicit recurrent loop. This will be slower but unlocks selectivity. Verify correctness against FFT output.

3. **Add selective dt**: Make dt input-dependent via a linear projection + softplus. This is the biggest expected win per the ablation study.

4. **Add selective B, C**: Project B and C from input. Combined with selective dt, this is the full S6 mechanism.

5. **Optimize scan**: If the sequential scan is too slow, implement a parallel scan or use `mx.compile` to fuse the loop.

### 6. State dimension adjustment

With selectivity, larger state dimensions pay off more than in fixed S4. Current STATE_DIM=64 might actually be oversized for LTI mode (the paper shows S4 caps out at N=4-8) but undersized for selective mode. Consider:
- Reducing to N=16 initially (faster scan, same performance with selectivity)
- Testing N=16 vs N=64 with and without selectivity to see the interaction

### 7. Initialization interaction

The paper finds that with selectivity enabled, **real random init matches HiPPO**. This means:
- HiPPO init is more important for our current fixed S4D (where it's the main lever for quality)
- Once selectivity is added, the init matters less -- the model learns to select regardless
- Still worth trying HiPPO first on the fixed model (known big win), then adding selectivity on top

## Key Differences: Fixed S4D vs Mamba

| Aspect | Our S4DLayer | Mamba |
|--------|-------------|-------|
| Parameters | All fixed (A, B, C, dt) | A fixed; B, C, dt input-dependent |
| Compute mode | FFT convolution (parallel, O(L log L)) | Parallel scan (O(L log L) work, O(log L) depth) |
| State utilization | Caps out at small N | Scales with N |
| Block design | SSM + MLP (two sub-blocks) | Expand-gate-SSM (one block, no separate MLP) |
| Activation | GELU in MLP | SiLU in gating path |
| Normalization | Post-norm (LN after residual add) | Pre-norm (LN before block) |
| Local context | None | Conv1d (k=4) before SSM |
| Skip connection | D parameter (y = conv + D*u) | Gating serves similar purpose |
| Expressivity | Linear time-invariant | Nonlinear, content-aware |

## Limitations and Caveats

- The hardware-aware implementation (fused CUDA kernels, SRAM management) is specific to NVIDIA GPUs. On MLX/Apple Silicon, we need a different optimization strategy. The algorithmic ideas (parallel scan, recomputation) still apply.
- Mamba was validated up to ~3B parameters. The authors note uncertainty about scaling beyond 7B.
- For continuous signals (audio waveforms), the LTI property of S4 can actually be advantageous. Selectivity helps most on discrete modalities (text, DNA) -- which is exactly what our `--task lm` and `--task dna` benchmarks test.
- The paper does not explore fine-tuning, in-context learning, or RLHF -- affordances that Transformers have well-established recipes for.

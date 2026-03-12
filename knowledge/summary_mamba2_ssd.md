# Mamba-2 / State Space Duality (SSD)

**Paper:** "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
**Authors:** Tri Dao and Albert Gu
**URL:** https://arxiv.org/abs/2405.21060
**Venue:** ICML 2024
**Code:** https://github.com/state-spaces/mamba

---

## 1. What is State Space Duality?

SSD shows that structured SSMs and variants of attention are **duals of each other**, connected through semiseparable matrices. The key insight:

- An SSM `y = SSM(A,B,C)(x)` can be written as a matrix multiplication `y = Mx` where M is a **semiseparable matrix** (every submatrix on/below the diagonal has rank at most N, the state dimension).
- When A is **scalar times identity** (not full diagonal as in Mamba-1), the matrix M factors as:
  ```
  M = L . (C B^T)    where L = 1SS(a) is a 1-semiseparable matrix
  ```
  with L_ij = a_i * a_{i-1} * ... * a_{j+1} for i >= j (cumulative products of scalar decay factors).

- This is **exactly** masked kernel attention `Y = (L . QK^T) V` where:
  - C -> Q (queries)
  - B -> K (keys)
  - X -> V (values)
  - L -> structured mask (replacing softmax)
  - a_t -> input-dependent decay factors (replacing causal mask of all 1s)

- When all a_t = 1, SSD reduces to **causal linear attention** (cumsum replaces scan).

**The duality:** Two algorithms to compute the same function:
1. **Linear (recurrent) mode:** `h_t = a_t * h_{t-1} + B_t x_t; y_t = C_t^T h_t` -- O(TN) time, sequential
2. **Quadratic (attention) mode:** Materialize M, multiply `y = Mx` -- O(T^2N) time, parallel matmuls

SSD's algorithm combines both for the best of each.

## 2. Key Differences: SSD vs Mamba-1's Selective SSM

| Property | Mamba-1 (S6) | Mamba-2 (SSD) |
|----------|-------------|---------------|
| A structure | Diagonal (N,) per timestep | **Scalar times identity** -- one scalar per timestep per head |
| A shape | (T, N) | (T,) per head |
| Head dim P | P = 1 (each channel independent) | P = 64 or 128 (shared dynamics within head) |
| State dim N | 16 | 64 to 256 |
| B, C sharing | Per-channel | Shared across heads (1 B, 1 C head -- MVA pattern) |
| Computation | Parallel associative scan | **Chunked matmul + small scan** |
| Hardware | Custom CUDA scan kernel | **Matrix multiplications** (tensor cores) |
| Projections | Sequential (A,B,C depend on conv(X)) | **Parallel** (A,B,C,X projected simultaneously) |

## 3. The SSD Algorithm: Chunked Computation

The core algorithmic contribution. Instead of a full parallel scan over T timesteps, SSD:

1. **Splits the sequence into chunks** of size Q (default **Q = 64**, called `block_len`)
2. Decomposes the semiseparable matrix M into a block grid of (T/Q) x (T/Q) submatrices
3. Diagonal blocks = intra-chunk (self-similar SSM subproblems)
4. Off-diagonal blocks = low-rank (factored through the state h)

### The 4 Steps

**Step 1: Intra-chunk outputs (diagonal blocks)**
- For each chunk, compute output assuming initial state = 0
- Uses the **quadratic (attention) form**: `Y_diag = (L . C B^T) X` within each chunk
- This is a matmul on (Q, Q) matrices -- small enough to be efficient
- All chunks computed **in parallel**

**Step 2: Chunk states (right factors of low-rank blocks)**
- For each chunk, compute the final state assuming initial state = 0
- `states = einsum(B, decay_states, X)` -- a (N, P) matrix per chunk
- This is the B-block factor: what state each chunk produces from its own inputs

**Step 3: Inter-chunk recurrence (center factors)**
- The T/Q chunk states form a **new, much shorter** 1-SS matrix multiplication
- Sequence length reduced from T to T/Q (e.g., 2048 -> 32 with Q=64)
- Can use a small associative scan OR naive dense matmul (both cheap at this scale)
- Produces the **true initial state** for each chunk, accounting for all prior chunks

**Step 4: State-to-output conversion (left factors)**
- For each chunk, given its true initial state (from step 3), compute the output contribution
- `Y_off = einsum(C, states, state_decay_out)` -- another matmul
- All chunks computed **in parallel**

**Final output:** `Y = Y_diag + Y_off` (intra-chunk + inter-chunk contributions)

### Why This is Fast

- Steps 1, 2, 4 are **batched matrix multiplications** -- fully parallel, use tensor cores
- Step 3 operates on length T/Q instead of T (e.g., 32 instead of 2048) -- negligible cost
- A100: 312 TFLOPS BF16 matmul vs 19 TFLOPS FP32 scalar -- **16x hardware advantage for matmul**
- The chunk size Q=64 balances intra-chunk quadratic cost vs inter-chunk sequential cost

### Complexity

|               | Attention | SSM (scan) | SSD |
|---------------|-----------|-----------|-----|
| State size    | T         | N         | N   |
| Training FLOPs| T^2 N     | T N^2     | T N^2 |
| Inference FLOPs| T N      | N^2       | N^2 |
| Memory        | T^2       | T N^2     | **T N** |
| Uses matmul   | Yes       | No        | **Yes** |

## 4. Reference Implementation (~35 lines of PyTorch)

From Listing 1 in the paper (page 21):

```python
def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
       which is equivalent to a scalar SSM."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks / chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len)
                   for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states
    # at chunk boundaries (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```

### Key Tensor Shapes

- `X`: (batch, length, n_heads, d_head) -- P = d_head
- `A`: (batch, length, n_heads) -- **scalar per head per timestep**
- `B`: (batch, length, n_heads, d_state) -- N = d_state
- `C`: (batch, length, n_heads, d_state)
- `states`: (batch, n_chunks, d_head, d_state) per head -- the SSM hidden state
- `L`: (n_heads, n_chunks, block_len, block_len) -- the 1-SS mask within each chunk

## 5. Performance: Mamba-2 vs Mamba-1 vs Transformers

### Speed (Figure 10 in paper, A100 80GB)

- **SSD is 2-8x faster than Mamba's fused associative scan** (for N=64 state dim)
- SSD is **faster than FlashAttention-2** for sequence length >= 2K, and 6x faster at 16K
- Increasing state dimension: Mamba-1 scan time scales linearly with N; SSD barely changes
  - At N=256, SSD is dramatically faster than Mamba scan
- SSD speed is roughly constant across state dimensions because it's dominated by matmul

### Quality (Table 1, Pile pretraining, 300B tokens)

| Model | Size | Pile PPL | Avg downstream acc |
|-------|------|----------|--------------------|
| Pythia | 1.4B | 7.51 | 51.7 |
| Mamba | 1.4B | 6.80 | 56.4 |
| **Mamba-2** | **1.3B** | **6.66** | **56.4** |
| Pythia | 2.8B | 6.73 | 55.7 |
| Mamba | 2.8B | 6.22 | 59.9 |
| **Mamba-2** | **2.7B** | **6.09** | **60.2** |

- Mamba-2 matches or slightly exceeds Mamba-1 quality at every scale
- Mamba-2 2.7B outperforms Pythia-6.9B (trained on same data)
- Scaling laws (Figure 9): Mamba-2 **Pareto dominates** both Mamba-1 and Transformer++ in perplexity vs FLOPs and vs wall-clock time

### MQAR (Multi-Query Associative Recall)

- Mamba-1 struggles on hard MQAR across all settings
- Mamba-2 with N=64 substantially outperforms Mamba-1 even at N=16
- Mamba-2 with N=256 matches or exceeds softmax attention

### Hybrid Models (Table 3, 2.7B scale)

- Pure Mamba-2: 60.2 avg accuracy
- Pure Transformer++: 60.2 avg accuracy
- Mamba-2 + 6 attention layers (out of 64): **61.0** -- best of both worlds
- ~10% attention layers is the sweet spot

## 6. Mamba-2 Architecture Details

### Block Design (Figure 6 -- "Parallel Mamba Block")

```
Input u (L, d)
  |
  +--> W^(x) --> x (L, ed)    [expansion factor e, typically 2]
  |
  +--> W^(z) --> z (L, ed)    [gate branch]
  |
  +--> projection --> A, B, C  [produced IN PARALLEL with x, not from x]
  |
  x --> conv1d(x) --> x_c     [depthwise, independent along d]
  |
  x_c --> SSM_{A,B,C}(x_c) --> y  [SSD layer, independent along d]
  |
  y * SiLU(z) --> y_g         [gating]
  |
  GroupNorm(y_g) --> y_n      [extra normalization, NormFormer-style]
  |
  y_n --> W^(o) --> out (L, d)
```

### Key Architecture Hyperparameters

- **Head dimension P:** 64 or 128 (like Transformer head dim)
- **State dimension N:** 64 default (up to 256; was 16 in Mamba-1)
- **Number of heads H:** D / P (e.g., 768/64 = 12 heads for d_model=768)
- **Head pattern:** Multi-input SSM (MIS) / Multi-value attention (MVA)
  - B and C: 1 head each (shared across all X heads)
  - X: H heads (independent)
  - A: H heads (each head has its own decay)
  - This is analogous to multi-value attention, NOT multi-head attention
- **Expansion factor e:** 2 (like Mamba-1)
- **Conv kernel:** 1D depthwise, typically width 4
- **Kernel feature map psi:** Swish/SiLU applied to B and C branches (default, following Mamba-1)
- **Normalization:** GroupNorm after gating, before output projection

### Parameter Counts (Table 4 ablation, ~130M scale)

- Sequential projections, no norm: 129.3M, ppl 11.76
- Parallel projections + norm (Mamba-2): 126.5M, ppl 11.49
- Parallel projections save ~2% params and improve quality

## 7. Recurrent Inference Mode

For autoregressive generation, SSD reverts to the standard SSM recurrence:

```
h_t = a_t * h_{t-1} + B_t * x_t    # state update (scalar a_t times state + rank-1 update)
y_t = C_t^T * h_t                   # output projection
```

- State size: (n_heads, d_head, d_state) = H * P * N per layer
- Inference FLOPs per step: O(N^2) per head = O(H * P * N + H * N * N) roughly
- No KV cache growing with sequence length (unlike Transformers)

## 8. Porting to MLX on Apple Silicon: Implementation Notes

### Required Operations (all available in MLX)

1. **Matrix multiplication (matmul/einsum):** The dominant operation. Steps 1, 2, 4 are all batched matmuls. MLX has `mx.matmul` and `mx.einsum`.

2. **Cumulative sum (`cumsum`):** Used to compute A_cumsum. MLX has `mx.cumsum`.

3. **Exponential (`exp`):** Applied to cumsum'd A values to get decay factors and the L mask. `mx.exp`.

4. **Segment sum (`segsum`):** Custom function -- outer difference of cumsum + tril mask + masked_fill(-inf). Requires:
   - `cumsum`
   - Broadcasting subtraction (outer diff)
   - `tril` mask generation
   - `where` or masked fill

5. **Reshape/rearrange:** Chunking the sequence into (batch, n_chunks, chunk_len, ...). Standard reshape/transpose ops.

6. **1D depthwise convolution:** For the conv in the block. `mx.conv1d` with groups=channels.

7. **Elementwise ops:** SiLU/Swish activation, multiplication (gating), addition.

8. **GroupNorm:** `mx.fast.layer_norm` or manual implementation with group reshaping.

9. **Linear projections:** Standard matmul for W^(x), W^(z), W^(o), and the A/B/C projections.

### What's CUDA-Specific (from the optimized Triton kernel)

The paper's optimized implementation uses:
- **Triton kernel** for fused SSD computation -- NOT needed for a correct implementation
- **Tensor core** exploitation via matmul -- MLX's Metal backend uses Apple's AMX/GPU matrix units automatically
- The reference PyTorch implementation (Listing 1 above) works on **any backend** -- no custom kernels needed

### What's Portable

- The entire reference implementation uses only standard tensor operations
- **No custom CUDA kernels required** for correctness
- The chunked algorithm naturally maps to matmuls, which MLX handles well
- The inter-chunk scan (step 3) on T/Q length is tiny and can be done with naive matmul

### MLX-Specific Considerations

1. **einsum support:** MLX supports einsum but may be slower than explicit reshape+matmul for complex contractions. Consider decomposing the 4-operand einsum in step 1 into sequential matmuls.

2. **Memory:** SSD's memory footprint is O(TN), much better than attention's O(T^2). The L matrix within chunks is only (Q, Q) = (64, 64) -- small.

3. **Chunk size tuning:** Q=64 is optimized for NVIDIA tensor cores. On Apple Silicon, the optimal chunk size might differ. Experiment with Q=32, 64, 128. Smaller Q = more sequential scan work but smaller intra-chunk matmuls.

4. **dtype:** The reference uses the same dtype for all of X, A, B, C. Use float32 or bfloat16 (M1 Max supports float32 well; bfloat16 on M3+).

5. **The segsum function** creates a (Q, Q) dense matrix per chunk per head -- this is fine for Q=64 (4KB per matrix in float32). At Q=256 it would be 256KB.

6. **Recurrent mode for generation:** Trivial to implement -- just the scalar recurrence. This is where SSMs shine on Apple Silicon: O(1) memory per step, no growing KV cache.

### Minimal MLX Implementation Sketch

The SSD forward pass in MLX would:
1. Project input to get X, Z, A, B, C (standard linear layers)
2. Conv1d on X
3. Reshape all into chunks: (B, T, ...) -> (B, T//Q, Q, ...)
4. Compute A_cumsum = mx.cumsum(A, axis=-1) within chunks
5. Build L = exp(segsum(A)) -- (Q, Q) lower triangular per chunk
6. Step 1: Y_diag via two matmuls (C @ B^T -> QQ gram, mask with L, then @ X)
7. Step 2: states via matmul (B^T @ (decay * X))
8. Step 3: scan chunk states (tiny -- T/Q length, use matmul with decay_chunk matrix)
9. Step 4: Y_off via matmul (C @ states * decay)
10. Y = Y_diag + Y_off, reshape back to (B, T, ...)
11. Gate with SiLU(Z), GroupNorm, output projection

### Differences from Mamba-1 Implementation

- **No parallel associative scan needed** -- the hardest part of Mamba-1 to implement efficiently
- **No hardware-specific scan kernel** -- everything is matmul
- **Larger state dimension** (64-256 vs 16) with no speed penalty
- **Simpler block** -- parallel projections, no sequential dependency for A,B,C on conv(X)
- **A is scalar** not diagonal -- fewer parameters, simpler recurrence

## 9. Summary: Why SSD Matters for nanostate

1. **Simpler to implement correctly** than Mamba-1's parallel scan
2. **Faster training** via matmul instead of scan (2-8x on GPU; should also help on Apple Silicon)
3. **Larger state sizes** (64-256) with minimal speed cost -- better recall, better language modeling
4. **Equal or better quality** than Mamba-1 at all scales tested
5. **Natural matmul decomposition** maps well to any hardware with matrix multiply units
6. **Clean recurrent mode** for generation -- same O(1) per-step cost as Mamba-1
7. **The reference implementation is ~35 lines** -- feasible to port directly to MLX

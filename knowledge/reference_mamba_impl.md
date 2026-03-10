# Reference Mamba Implementation Analysis

Source: [state-spaces/mamba](https://github.com/state-spaces/mamba)

This document compares the official Mamba implementation (Mamba-1 and Mamba-2) with
our nanostate `train.py` S4D implementation, and provides concrete adaptation suggestions.

---

## 1. Architecture Overview: Mamba vs Our S4D

### Official Mamba-1 block (mamba_ssm/modules/mamba_simple.py)

```
Input (B, L, D)
  |
  in_proj  -->  Linear(D, 2*d_inner)  -->  split into x, z   [expand=2, so d_inner=2*D]
  |                                         |
  x branch:                                z branch (gate):
    conv1d (depthwise, kernel=4)              kept aside
    SiLU activation
    x_proj  -->  Linear(d_inner, dt_rank + 2*d_state)
    split into dt_raw, B, C    (B and C are input-dependent!)
    dt = dt_proj(dt_raw)       (low-rank: dt_rank -> d_inner)
    selective_scan(x, dt, A, B, C, D)
    |
  y = scan_output * SiLU(z)     <-- gating with z branch
  |
  out_proj  -->  Linear(d_inner, D)
  |
Output (B, L, D)
```

### Our S4D block (train.py)

```
Input (B, L, D)
  |
  S4DLayer: FFT convolution with fixed B, C
  + skip (D parameter)
  |
  LayerNorm
  |
  MLP: Linear(D, 2D) -> GELU -> Linear(2D, D)
  |
  LayerNorm
  |
Output (B, L, D)
```

**Key structural differences:**
1. Mamba has NO separate MLP -- the expand-and-gate pattern IS the "MLP equivalent"
2. Mamba uses input-dependent B, C (selective) vs our fixed/learned B, C
3. Mamba uses depthwise conv1d before the SSM (local context mixing)
4. Mamba uses SiLU gating instead of a separate MLP block
5. Mamba uses pre-norm (LN -> Mixer -> Add) vs our post-norm (Add -> LN)

---

## 2. State Space Parameterization

### A matrix

**Official Mamba-1:**
```python
# S4D real initialization: A = -exp(log(arange(1, d_state+1)))
# Shape: (d_inner, d_state), same values repeated across d_inner
A = repeat(
    torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
    "n -> d n",
    d=self.d_inner,
).contiguous()
A_log = torch.log(A)  # stored as log for numerical stability
self.A_log = nn.Parameter(A_log)
self.A_log._no_weight_decay = True

# Used as: A = -torch.exp(self.A_log.float())  # always negative
```

**Official Mamba-2:**
```python
# Simpler: one scalar per head, uniform random in [1, 16]
A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
A_log = torch.log(A).to(dtype=dtype)
self.A_log = nn.Parameter(A_log)
```

**Ours:**
```python
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))
# Used as: A = -mx.exp(self.log_A)
```

**Differences:**
- Mamba-1 uses the S4D-Lin initialization: `A_n = -n` for n=1..N. This gives a structured
  set of timescales (1, 1/2, 1/3, ..., 1/N). Our random init lacks this structure.
- Mamba excludes A_log from weight decay. We should do the same.
- Mamba-2 simplifies to one A per head (not per state dimension).

### B and C matrices (the biggest difference)

**Official Mamba-1:** B and C are **input-dependent** (selective):
```python
# x_proj projects the post-conv activation to dt, B, C
self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

# In forward:
x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
```

B and C are freshly computed at every timestep from the input. This is the core
"selective" mechanism -- the SSM parameters adapt to the input.

**Ours:** B and C are fixed learned parameters:
```python
self.B = mx.random.normal((d_model, state_dim)) * 0.01
self.C = mx.random.normal((d_model, state_dim)) * 0.01
```

This is the standard S4D approach. The lack of input dependence means our model
cannot selectively filter information based on content.

### D (skip connection)

Both implementations use `D = ones(d_inner)`. Identical in spirit.

### dt (step size / delta)

**Official Mamba-1:**
```python
# dt_rank is typically d_model/16 (low-rank bottleneck)
self.dt_rank = math.ceil(self.d_model / 16)

# dt is input-dependent, projected through a low-rank bottleneck:
# x -> x_proj -> dt_raw (dt_rank dims) -> dt_proj -> dt (d_inner dims)
self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

# dt_proj bias initialized via inverse-softplus of uniform[dt_min, dt_max]:
dt = torch.exp(
    torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
    + math.log(dt_min)
).clamp(min=dt_init_floor)
inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
self.dt_proj.bias.copy_(inv_dt)

# Applied with softplus to ensure positivity:
# delta_softplus=True in selective_scan_fn
```

**Ours:**
```python
self.log_dt = mx.zeros((d_model,))
# Used as: dt = mx.exp(self.log_dt)
```

**Differences:**
- Mamba's dt is input-dependent (computed from each token)
- Mamba uses softplus (not exp) for positivity
- Mamba uses inverse-softplus initialization to get dt in [0.001, 0.1]
- Mamba uses a low-rank projection (dt_rank) for efficiency

---

## 3. Discretization Method

### Official Mamba (selective_scan_ref):
```python
# ZOH discretization:
deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))  # exp(dt * A)
deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)   # dt * B * u

# Recurrence:
x = deltaA[:,:,i] * x + deltaB_u[:,:,i]                      # x_new = exp(dt*A)*x + dt*B*u
y = torch.einsum('bdn,bn->bd', x, C[:,:,i])                  # y = C*x
```

The discretization is:
- `A_bar = exp(dt * A)`  (ZOH for the diagonal A)
- `B_bar = dt * B`       (Euler approximation for B, since A is diagonal and B is input-dependent)

### Ours:
```python
dtA = A * dt[:, None]                                    # dt * A
V = mx.exp(dtA[:, :, None] * arange[None, None, :])     # exp(dt*A*l) Vandermonde
K = mx.sum(CB[:, :, None] * V, axis=1) * dt[:, None]    # convolution kernel
```

We compute the full convolution kernel analytically using the closed-form S4D formula.
This is equivalent to ZOH discretization but evaluated as a convolution rather than
a recurrence. The key formula: `K[l] = C * exp(A*dt*l) * B * dt`.

**Both are correct ZOH**. The difference is that Mamba computes element-wise recurrences
(for selectivity) while we use FFT convolution (for parallelism with fixed parameters).

---

## 4. Selective Scan Implementation

The core selective scan is a parallel scan over the recurrence. The reference
implementation (selective_scan_ref) shows the logic clearly:

```python
for i in range(L):
    x = deltaA[:, :, i] * x + deltaB_u[:, :, i]   # state update
    y = einsum('bdn,bn->bd', x, C[:, :, i])        # output
```

The CUDA kernel (`selective_scan_cuda`) parallelizes this. The key insight: because
B and C change at every timestep, you cannot use FFT convolution. You must do the
sequential scan (or a parallel prefix scan).

**For our MLX setup**: if we add input-dependent B/C, we need to implement a
sequential scan. MLX does not have a built-in parallel scan, but we can write
the recurrence directly.

---

## 5. Gating and Activation Functions

### Mamba-1 gating pattern:
```python
# in_proj produces x and z (gate) by projecting to 2*d_inner
xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
x, z = xz.chunk(2, dim=1)

# x goes through conv1d + SiLU + SSM
# z is the gate
# Output: y = ssm_output * SiLU(z)
out = y * F.silu(z)
out = self.out_proj(out)
```

### Mamba-2 gating pattern:
```python
# in_proj produces [z, x, B, C, dt] all in one projection
zxbcdt = self.in_proj(u)
z, xBC, dt = split(...)

# x, B, C go through conv1d + SiLU
# SSM scan produces y
# Then: y = RMSNorm(y) * SiLU(z)   (norm before gating)
y = self.norm(y, z)
out = self.out_proj(y)
```

### GatedMLP (used optionally alongside Mamba blocks):
```python
class GatedMLP(nn.Module):
    def forward(self, x):
        y = self.fc1(x)            # project to 2*hidden
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)  # SiLU gating
        y = self.fc2(y)
        return y
```

### Ours:
```python
# No gating. Standard MLP with GELU:
self.mlp = nn.Sequential(
    nn.Linear(d_model, d_model * mlp_ratio),
    nn.GELU(),
    nn.Linear(d_model * mlp_ratio, d_model),
)
```

**The Mamba block IS the gated architecture** -- there is no separate MLP needed.
The expand factor (2x) combined with SiLU gating serves the same purpose as a
gated MLP. The original Mamba paper notes this: "we remove the MLP block entirely."

---

## 6. Block Structure and Residual Connections

### Official Mamba (block.py):
```python
# Pre-norm residual: Add -> LN -> Mixer
# (not the standard LN -> Mixer -> Add, reordered for fused add+norm)
class Block(nn.Module):
    def forward(self, hidden_states, residual=None):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)  # Mamba layer
        # optionally: same pattern for MLP
        if self.mlp is not None:
            residual = hidden_states + residual
            hidden_states = self.norm2(residual)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

### Full model (mixer_seq_simple.py):
```python
class MixerModel(nn.Module):
    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        # Final norm
        residual = (hidden_states + residual)
        hidden_states = self.norm_f(residual)
        return hidden_states
```

### Ours:
```python
class SSMBlock(nn.Module):
    def __call__(self, x):
        x = self.norm1(x + self.ssm(x))   # post-norm
        x = self.norm2(x + self.mlp(x))   # post-norm
        return x
```

**Differences:**
- Mamba uses **pre-norm** (norm before the mixer). We use **post-norm** (norm after add).
- Mamba carries a separate `residual` tensor through the stack for fused add+norm.
- Mamba uses RMSNorm (not LayerNorm) in practice. RMSNorm is simpler and slightly faster.
- With `d_intermediate=0`, Mamba blocks have NO MLP -- just the SSM mixer.

---

## 7. Initialization

### Official Mamba:
```python
# A: S4D-Lin init -- A_n = -n for n in 1..d_state
A = repeat(torch.arange(1, d_state + 1, ...), "n -> d n", d=d_inner)
A_log = torch.log(A)

# dt bias: inverse-softplus of uniform in [dt_min, dt_max]
dt = torch.exp(torch.rand(d_inner) * (log(dt_max) - log(dt_min)) + log(dt_min))
inv_dt = dt + torch.log(-torch.expm1(-dt))

# out_proj and fc2: scaled by 1/sqrt(2*n_layer) for residual accumulation
for name, p in module.named_parameters():
    if name in ["out_proj.weight", "fc2.weight"]:
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        with torch.no_grad():
            p /= math.sqrt(n_residuals_per_layer * n_layer)

# Embedding: normal with std=0.02
nn.init.normal_(module.weight, std=0.02)
```

### Ours:
```python
# A: random uniform in [-4, -1] (log space)
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))

# B, C: small random normal
self.B = mx.random.normal((d_model, state_dim)) * 0.01
self.C = mx.random.normal((d_model, state_dim)) * 0.01

# dt: zeros (so exp(0)=1)
self.log_dt = mx.zeros((d_model,))
```

**Key differences:**
- Our A init is random; Mamba's is structured (1, 2, 3, ..., N) in log space
- We don't scale output projections by 1/sqrt(n_layers)
- Our dt starts at 1.0 (exp(0)); Mamba's starts in [0.001, 0.1]

---

## 8. Concrete Adaptation Suggestions for nanostate

### Priority 1: S4D-Lin A initialization (easy, high impact)

Replace random A init with the structured S4D initialization:

```python
# In S4DLayer.__init__:
# BEFORE:
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))

# AFTER (S4D-Lin):
A = mx.repeat(mx.arange(1, state_dim + 1).astype(mx.float32)[None, :], d_model, axis=0)
self.log_A = mx.log(A)
# Usage stays the same: A = -mx.exp(self.log_A)
```

This gives timescales at 1, 1/2, 1/3, ..., 1/N which captures multi-scale dynamics.

### Priority 2: Better dt initialization (easy, medium impact)

Initialize dt in the [0.001, 0.1] range with softplus instead of exp:

```python
# In S4DLayer.__init__:
# BEFORE:
self.log_dt = mx.zeros((d_model,))

# AFTER:
dt_min, dt_max = 0.001, 0.1
dt = mx.exp(
    mx.random.uniform(shape=(d_model,)) * (math.log(dt_max) - math.log(dt_min))
    + math.log(dt_min)
)
# inverse softplus: log(exp(dt) - 1)
self.log_dt = dt + mx.log(-mx.expm1(-dt))
# In kernel(): dt = mx.softplus(self.log_dt) instead of mx.exp(self.log_dt)
```

### Priority 3: Pre-norm block structure (easy, medium impact)

Switch from post-norm to pre-norm residual:

```python
# BEFORE (post-norm):
def __call__(self, x):
    x = self.norm1(x + self.ssm(x))
    x = self.norm2(x + self.mlp(x))
    return x

# AFTER (pre-norm):
def __call__(self, x):
    x = x + self.ssm(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
```

Pre-norm is more stable for deeper models and is what all modern SSM/Transformer
architectures use.

### Priority 4: Depthwise conv1d before SSM (moderate, medium impact)

Add a short (kernel=4) depthwise conv before the SSM for local context:

```python
# In S4DLayer or SSMBlock:
# Add before SSM processing:
# conv1d with groups=d_model (depthwise), kernel_size=4, causal padding
```

In MLX, this can be done with `nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model)`
plus causal masking (pad left by 3, no right padding).

### Priority 5: SiLU gating (moderate, medium impact)

Replace the separate MLP with a gated architecture. The simplest version:

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, state_dim, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)   # x and z
        self.ssm = S4DLayer(d_inner, state_dim)           # SSM on expanded dim
        self.out_proj = nn.Linear(d_inner, d_model)

    def __call__(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = mx.split(xz, 2, axis=-1)  # split along last dim
        x = self.ssm(x)
        x = x * nn.silu(z)               # SiLU gating
        x = self.out_proj(x)
        return x + residual
```

This eliminates the separate MLP entirely and matches the Mamba architecture more closely.
The expand factor (2x) gives the same parameter budget as our current SSM+MLP.

### Priority 6: Input-dependent B and C (harder, highest impact)

This is the "selective" mechanism. It requires replacing FFT convolution with a
sequential scan:

```python
class SelectiveScan(nn.Module):
    def __init__(self, d_inner, state_dim, dt_rank):
        super().__init__()
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # A, D as before

    def __call__(self, x):
        # x: (B, L, d_inner)
        B_sz, L, D = x.shape
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*state_dim)
        dt_raw, B_input, C_input = mx.split(
            x_dbl, [self.dt_rank, self.state_dim], axis=-1
        )
        dt = mx.softplus(self.dt_proj(dt_raw))  # (B, L, D)

        A = -mx.exp(self.A_log)  # (D, N)

        # Sequential scan (cannot use FFT with input-dependent B, C)
        h = mx.zeros((B_sz, D, self.state_dim))
        outputs = []
        for t in range(L):
            dA = mx.exp(dt[:, t, :, None] * A[None, :, :])     # (B, D, N)
            dB = dt[:, t, :, None] * B_input[:, t, None, :]     # (B, D, N)
            h = dA * h + dB * x[:, t, :, None]                  # (B, D, N)
            y = mx.sum(h * C_input[:, t, None, :], axis=-1)     # (B, D)
            outputs.append(y)
        y = mx.stack(outputs, axis=1)  # (B, L, D)
        return y + x * self.D[None, None, :]
```

**Warning**: The sequential scan is O(L) and cannot be parallelized with FFT.
For training on long sequences this will be slow. However, it enables the selective
mechanism which is the key innovation of Mamba.

A practical middle ground: keep FFT convolution for training but make B and C
input-dependent by projecting from a global summary (e.g., mean of x) rather
than per-timestep. This preserves convolution-mode training while adding some
input dependence.

### Priority 7: Output projection scaling (easy, low-to-medium impact)

Scale output projection weights by 1/sqrt(n_layers) for better gradient flow:

```python
# After model init:
for block in model.blocks:
    block.out_proj.weight *= 1.0 / math.sqrt(n_layers)
```

---

## 9. Summary of Mamba-2 (SSD) Differences

Mamba-2 restructures the selective scan as a structured state space duality (SSD):

1. **Multi-head structure**: nheads = d_inner / headdim (like attention heads)
2. **Scalar A per head**: A is (nheads,) not (d_inner, d_state)
3. **Chunked computation**: sequence split into chunks for semi-parallel processing
4. **RMSNorm before gating**: `y = RMSNorm(y) * SiLU(z)` instead of just `y * SiLU(z)`
5. **All projections in one**: `in_proj` produces `[z, x, B, C, dt]` at once
6. **Larger d_state**: default 128 (vs 16 for Mamba-1)

For nanostate, Mamba-1 adaptations are more practical since Mamba-2's chunked scan
requires custom CUDA/Triton kernels.

---

## 10. What NOT to Adapt

- **CUDA selective scan kernel**: We use MLX, not CUDA. Use sequential recurrence instead.
- **Fused add+norm**: Performance optimization, not algorithmic improvement.
- **Inference cache / step function**: Only needed for autoregressive generation.
- **Tensor parallelism / distributed**: Not relevant for single-file setup.
- **causal_conv1d CUDA kernel**: Use MLX conv1d with manual causal padding.

---

## 11. Recommended Implementation Order

1. **S4D-Lin A init** -- 5 minutes, likely improves convergence
2. **Pre-norm residual** -- 5 minutes, more stable training
3. **dt init with softplus** -- 10 minutes, better default timescales
4. **SiLU gating (replace MLP)** -- 30 minutes, matches Mamba architecture
5. **Depthwise conv1d** -- 30 minutes, adds local context
6. **Input-dependent B, C with sequential scan** -- 1-2 hours, enables selectivity
7. **Output projection scaling** -- 5 minutes, helps deep models

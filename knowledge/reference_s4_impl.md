# Reference S4 Implementation Analysis

Source: https://github.com/state-spaces/s4
Compared against: `/Users/abhay/Dev/nanostate/train.py`

---

## 1. HiPPO Initialization (THE Critical Difference)

The single most important insight in S4 is the HiPPO (High-order Polynomial Projection Operators) initialization of the state matrix A. Our nanostate implementation uses **random initialization**, which is the #1 thing to fix.

### What HiPPO Does

HiPPO defines specific matrices A, B that optimally compress continuous-time input history into a finite-dimensional state. The most common variant is **HiPPO-LegS** (Legendre Scaled), which projects the input history onto a basis of scaled Legendre polynomials.

### The HiPPO-LegS Matrix (from `src/models/hippo/hippo.py`)

```python
# Legendre (scaled) - the default and most important initialization
elif measure == 'legs':
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    B = np.diag(T)[:, None]
    B = B.copy()
```

This produces an NxN matrix A and Nx1 matrix B. The key property: the eigenvalues of this matrix have real part = -1/2 and imaginary parts that grow roughly linearly (approximately `pi * n` for the nth eigenvalue).

### S4D Diagonal Approximation

For the diagonal S4D variant (which is what nanostate implements), the full HiPPO matrix is **diagonalized** using the NPLR (Normal Plus Low-Rank) decomposition. The result is a diagonal A matrix with complex eigenvalues.

**From the simplified `models/s4/s4d.py`:**

```python
class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model

        # dt: log-uniform in [dt_min, dt_max]
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # C: random complex
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        # A diagonal: real part = -0.5, imaginary part = pi * n
        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)
```

**The S4D-Lin initialization (from `src/models/sequence/kernels/dplr.py`):**

```python
elif init in ['linear', 'lin']:
    imag_part = pi * imag_part  # imag_part = arange(N//2)
# So A = -0.5 - j * pi * [0, 1, 2, ..., N/2-1]
```

**The S4D-Inv initialization:**

```python
elif init in ['inverse', 'inv']:
    imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
# Approximates the actual eigenvalues of the HiPPO matrix
```

### What Our Code Does (WRONG)

```python
# Our nanostate: random init!
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))
self.B = mx.random.normal((d_model, state_dim)) * 0.01
self.C = mx.random.normal((d_model, state_dim)) * 0.01
```

Problems:
1. **A is real-only** -- S4D uses complex diagonal A (real + imaginary parts)
2. **A has random magnitudes** -- should have real part exactly -0.5 (or learned from that starting point)
3. **A has no imaginary part** -- the imaginary part `pi * n` creates oscillatory modes at different frequencies, which is the key to long-range modeling
4. **B is random** -- should be initialized from HiPPO (or a constant like `ones`)
5. **No conjugate symmetry** -- S4D uses N/2 complex eigenvalues and takes `2 * real_part` to get real output

### Recommended Fix for nanostate

```python
def __init__(self, d_model: int, state_dim: int):
    super().__init__()
    N = state_dim  # Use N/2 complex pairs internally

    # A diagonal: S4D-Lin initialization
    # Real part: -0.5 (log-parameterized as log(0.5))
    # Imaginary part: pi * [0, 1, 2, ..., N/2 - 1]
    self.log_A_real = mx.full((d_model, N // 2), math.log(0.5))
    self.A_imag = np.pi * mx.broadcast_to(
        mx.arange(N // 2).astype(mx.float32)[None, :],
        (d_model, N // 2)
    )

    # B: ones (constant) or HiPPO-derived
    # Simple version: constant ones (works well for S4D)
    self.B = mx.ones((d_model, N // 2)) * (2.0 ** 0.5)

    # C: random complex (stored as real pairs)
    self.C_re = mx.random.normal((d_model, N // 2)) * 0.5
    self.C_im = mx.random.normal((d_model, N // 2)) * 0.5

    # D: skip connection
    self.D = mx.ones((d_model,))

    # dt: log-uniform in [0.001, 0.1]
    self.log_dt = mx.random.uniform(
        low=math.log(0.001), high=math.log(0.1),
        shape=(d_model,)
    )
```

---

## 2. Discretization

### S4D ZOH Discretization (from `SSMKernelDiag.forward`)

The reference implementation uses Zero-Order Hold (ZOH) discretization:

```python
if self.disc == 'zoh':
    # C_bar = C * (exp(dt*A) - 1) / A
    C = C * (torch.exp(dtA)-1.) / A
    K = log_vandermonde(C, dtA, L)  # sum_n C_n * exp(dtA_n * l)
```

The simplified S4D version (`models/s4/s4d.py`) also uses ZOH:

```python
def forward(self, L):
    dt = torch.exp(self.log_dt)
    C = torch.view_as_complex(self.C)
    A = -torch.exp(self.log_A_real) + 1j * self.A_imag

    dtA = A * dt.unsqueeze(-1)
    K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
    C = C * (torch.exp(dtA)-1.) / A   # <-- ZOH correction factor
    K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
    return K
```

### What Our Code Does

```python
def kernel(self, L: int):
    dt = mx.exp(self.log_dt)
    A = -mx.exp(self.log_A)
    dtA = A * dt[:, None]

    # Simple Vandermonde: exp(dtA * l) for l = 0..L-1
    V = mx.exp(dtA[:, :, None] * arange[None, None, :])
    CB = self.C * self.B
    K = mx.sum(CB[:, :, None] * V, axis=1)
    return K * dt[:, None]
```

**Differences:**
- Our code multiplies by `dt` at the end instead of the ZOH factor `(exp(dtA)-1)/A`
- Our code computes `C * B * exp(dtA*l) * dt` which is a first-order (Euler) approximation
- The reference computes `C * (exp(dtA)-1)/A * exp(dtA*l)` which is the exact ZOH discretization
- The ZOH factor is especially important for stability with large dt*A values

### Recommended Fix

```python
def kernel(self, L: int):
    dt = mx.exp(self.log_dt)  # (H,)
    A_real = -mx.exp(self.log_A_real)  # (H, N/2), negative
    A_imag = -self.A_imag  # (H, N/2), negative
    # For real-only variant, can skip imaginary part

    dtA = A_real * dt[:, None]  # (H, N/2) -- for real-only case

    # ZOH discretization factor
    C_bar = self.C * self.B * (mx.exp(dtA) - 1.0) / (A_real)  # (H, N/2)

    # Vandermonde
    arange = mx.arange(L).astype(mx.float32)
    V = mx.exp(dtA[:, :, None] * arange[None, None, :])  # (H, N/2, L)
    K = mx.sum(C_bar[:, :, None] * V, axis=1)  # (H, L)
    return K
```

---

## 3. dt (Step Size) Initialization

### Reference

```python
# Log-uniform initialization in [dt_min, dt_max], typically [0.001, 0.1]
log_dt = torch.rand(H) * (
    math.log(dt_max) - math.log(dt_min)
) + math.log(dt_min)
```

This is **log-uniform** in [0.001, 0.1], meaning dt values span 2 orders of magnitude.

### Our Code

```python
self.log_dt = mx.zeros((d_model,))  # dt = 1.0 for all channels
```

**Problem:** All channels start with dt=1.0, which is 10x larger than the reference maximum. This is likely causing instability. The reference uses dt in [0.001, 0.1] to ensure `dt*A` doesn't blow up.

### Reference also uses separate learning rate for dt

```yaml
lr:
  dt: 0.001
  A: 0.001
  B: 0.001
```

The SSM parameters (A, B, dt) use a **smaller learning rate** (0.001) compared to the rest of the model. This is critical for training stability.

---

## 4. S4D Layer Internal Structure

### Reference S4D (from `models/s4/s4d.py`)

The S4D layer includes more than just the SSM kernel:

```python
class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Post-SSM processing:
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout)

        # Output mixing: Conv1d -> GLU
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):
        # SSM convolution
        k = self.kernel(L=L)
        y = fft_conv(u, k)
        y = y + u * self.D.unsqueeze(-1)

        # Post-processing (INSIDE the S4D layer, not outside)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)  # <-- GLU mixing layer
        return y
```

### Reference S4 Block (full version, from `src/models/sequence/modules/s4block.py`)

```python
class S4Block(SequenceModule):
    # Default: activation='gelu', final_act='glu'
    def forward(self, x):
        y = self.layer(x)        # FFTConv (SSM kernel + convolution)
        y = self.activation(y)   # GELU
        y = self.drop(y)         # Dropout
        y = self.output_linear(y)  # Linear -> GLU (2H -> H)
        return y
```

### Our Code

```python
class SSMBlock(nn.Module):
    def __init__(self, d_model, state_dim, mlp_ratio=2):
        self.ssm = S4DLayer(d_model, state_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x):
        x = self.norm1(x + self.ssm(x))   # post-norm residual around SSM
        x = self.norm2(x + self.mlp(x))   # post-norm residual around MLP
        return x
```

**Key differences:**
1. Reference has **GELU + GLU** inside the S4D layer itself (after FFT conv, before the residual)
2. Reference uses a separate FFN/MLP block outside with its own residual
3. Reference uses **prenorm** (norm before the layer) by default, not postnorm
4. Reference S4D has an output mixing layer (`Conv1d -> GLU`) that the nanostate MLP partially substitutes for

---

## 5. Residual Block & Normalization

### Reference Architecture (from `backbones/block.py` and `backbones/model.py`)

Default config:
```yaml
prenorm: true
norm: layer
residual: R  # Standard residual connection
```

The block structure is:
```
x -> LayerNorm -> S4Layer(GELU + GLU) -> Dropout -> + x
                                                    |
                                                    v
                                               LayerNorm -> ... (next block)
```

With prenorm, the final norm is applied after all layers:
```python
# In SequenceModel.forward():
for layer in self.layers:
    outputs = layer(outputs)
if self.norm is not None:
    outputs = self.norm(outputs)
```

### Our Code

Uses postnorm:
```python
x = self.norm1(x + self.ssm(x))    # postnorm
x = self.norm2(x + self.mlp(x))    # postnorm
```

**Recommendation:** Switch to prenorm, which is the S4 default and is generally more stable for deep models:
```python
def __call__(self, x):
    x = x + self.ssm(self.norm1(x))   # prenorm residual
    x = x + self.mlp(self.norm2(x))   # prenorm residual
    return x
```

---

## 6. Complex vs Real

### Reference S4D

S4D uses **complex** diagonal A and complex C. The key trick: only N/2 eigenvalues are stored (one from each conjugate pair), and the output is `2 * real_part`:

```python
# From models/s4/s4d.py
C = torch.randn(H, N // 2, dtype=torch.cfloat)  # Complex C
A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # Complex A

K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real  # 2 * Re(...)
```

### Our Code

Uses purely real A and real B, C:
```python
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))
self.B = mx.random.normal((d_model, state_dim)) * 0.01
self.C = mx.random.normal((d_model, state_dim)) * 0.01
```

**Impact:** Without imaginary components, the model cannot represent oscillatory dynamics at different frequencies. This is equivalent to an exponential moving average (EMA) with different decay rates, but loses the "memory" property that makes S4 special for long-range tasks.

### Adaptation for MLX

MLX does not have native complex number support. Two approaches:

**Approach A: Simulate complex with real pairs (recommended for nanostate)**
```python
# Store real and imaginary parts separately
# A = a_re + j * a_im, C = c_re + j * c_im
# K = 2 * Re(sum_n (c_re + j*c_im) * (b_re + j*b_im) * exp((a_re + j*a_im)*dt*l))
# exp(x + jy) = exp(x) * (cos(y) + j*sin(y))

def kernel(self, L):
    dt = mx.exp(self.log_dt)
    A_real = -mx.exp(self.log_A_real)  # (H, N/2)
    A_imag = self.A_imag               # (H, N/2), pi * arange(N/2)

    dtA_real = A_real * dt[:, None]  # (H, N/2)
    dtA_imag = A_imag * dt[:, None]  # (H, N/2)

    # ZOH: C_bar = C * (exp(dtA) - 1) / A
    # For complex: need complex division
    # Simpler: absorb into Vandermonde

    arange = mx.arange(L).astype(mx.float32)
    # exp(dtA * l) = exp(dtA_real*l) * (cos(dtA_imag*l) + j*sin(dtA_imag*l))
    decay = mx.exp(dtA_real[:, :, None] * arange[None, None, :])  # (H, N/2, L)
    angles = dtA_imag[:, :, None] * arange[None, None, :]         # (H, N/2, L)
    V_real = decay * mx.cos(angles)  # (H, N/2, L)
    V_imag = decay * mx.sin(angles)  # (H, N/2, L)

    # K = 2 * Re(C * B * V) where C, B can be real for simplicity
    # If C is real and B is real: K = 2 * C * B * V_real
    CB = self.C * self.B  # (H, N/2)
    K = 2.0 * mx.sum(CB[:, :, None] * V_real, axis=1)  # (H, L)

    # (With ZOH correction applied to CB if desired)
    return K
```

**Approach B: S4D-Real (from the paper)**
Use purely real eigenvalues A = diag(-1, -2, ..., -N). This is simpler but less expressive:
```python
# S4D-Real: A = -diag(1, 2, ..., N)
# (from dplr.py init='real')
real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
# A = -real_part (no imaginary component)
```

---

## 7. Learning Rate Schedule for SSM Parameters

### Reference

The reference uses **different learning rates** for SSM parameters vs other parameters:

```yaml
# SSM params (A, B, dt): lr=0.001, weight_decay=0
lr:
  dt: 0.001
  A: 0.001
  B: 0.001
wd: 0.0
```

```python
# From models/s4/s4d.py - register method
def register(self, name, tensor, lr=None):
    if lr == 0.0:
        self.register_buffer(name, tensor)
    else:
        self.register_parameter(name, nn.Parameter(tensor))
        optim = {"weight_decay": 0.0}
        if lr is not None: optim["lr"] = lr
        setattr(getattr(self, name), "_optim", optim)
```

```python
# From example.py - optimizer setup
def setup_optimizer(model, lr, weight_decay, epochs):
    # General parameters (higher lr, with weight decay)
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # SSM parameters (lower lr=0.001, no weight decay)
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})
```

### Our Code

Uses a single Adam optimizer with uniform lr=0.001 for all parameters:
```python
optimizer = optim.Adam(learning_rate=lr)
```

**This is actually close to the reference for SSM params**, but the rest of the model could benefit from a higher learning rate and/or weight decay. The key insight: **SSM parameters should have NO weight decay**.

---

## 8. Priority-Ordered Implementation Recommendations

### Must-Have (highest impact on model quality)

1. **HiPPO-based A initialization** -- Replace random A with S4D-Lin:
   - Real part: `-0.5` (constant across all state dimensions)
   - Imaginary part: `-pi * [0, 1, 2, ..., N/2-1]` (different frequency for each state dim)
   - Use N/2 complex pairs instead of N real states

2. **Complex diagonal with cos/sin** -- Implement the imaginary component of A using separate sin/cos computation since MLX lacks complex numbers

3. **dt initialization** -- Change from `log_dt = zeros` (dt=1.0) to log-uniform in [log(0.001), log(0.1)]

4. **ZOH discretization** -- Replace `K * dt` with the proper `C * (exp(dtA)-1)/A * exp(dtA*l)` factor

### Should-Have (moderate impact)

5. **Prenorm** -- Switch from postnorm to prenorm residual blocks

6. **GLU output projection** -- Add a GLU layer after GELU activation inside the S4D layer (or replace the MLP with GLU-based mixing)

7. **B initialization** -- Use constant `ones` (or sqrt(2)) instead of small random values

8. **C initialization** -- Use random normal with reasonable scale (not * 0.01)

### Nice-to-Have (refinements)

9. **Separate learning rates** -- Lower lr (0.001) for SSM params, potentially higher for other params; no weight decay on SSM params

10. **D initialization** -- Reference uses `randn` not `ones` for the skip connection D parameter

11. **Dropout** -- Add dropout after activation (before output projection) inside the S4D layer

---

## 9. Minimal Concrete Patch

The single highest-impact change to `/Users/abhay/Dev/nanostate/train.py` is replacing the S4DLayer initialization and kernel computation. Here is a drop-in replacement that stays purely real (S4D-Real style) while being much closer to the reference:

```python
class S4DLayer(nn.Module):
    """Diagonal State Space layer with HiPPO-inspired initialization."""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        N = state_dim

        # A: S4D-Lin initialization (complex diagonal stored as real + imag)
        # Real part: all -0.5
        self.log_A_real = mx.full((d_model, N // 2), math.log(0.5))
        # Imaginary part: pi * [0, 1, 2, ..., N/2 - 1]
        self.A_imag = mx.broadcast_to(
            (math.pi * mx.arange(N // 2).astype(mx.float32))[None, :],
            (d_model, N // 2),
        )

        # B: constant initialization
        self.B_re = mx.ones((d_model, N // 2))
        self.B_im = mx.zeros((d_model, N // 2))

        # C: random (complex, stored as two reals)
        self.C_re = mx.random.normal((d_model, N // 2))
        self.C_im = mx.random.normal((d_model, N // 2))

        # D: skip connection
        self.D = mx.ones((d_model,))

        # dt: log-uniform in [0.001, 0.1]
        self.log_dt = mx.random.uniform(
            low=math.log(0.001), high=math.log(0.1), shape=(d_model,)
        )

    def kernel(self, L: int):
        dt = mx.exp(self.log_dt)                   # (H,)
        A_real = -mx.exp(self.log_A_real)           # (H, N/2), negative
        A_imag = self.A_imag                        # (H, N/2)

        dtA_real = A_real * dt[:, None]             # (H, N/2)
        dtA_imag = A_imag * dt[:, None]             # (H, N/2)

        arange = mx.arange(L).astype(mx.float32)   # (L,)

        # Vandermonde: exp((a_re + j*a_im)*l) = exp(a_re*l) * [cos(a_im*l) + j*sin(a_im*l)]
        decay = mx.exp(dtA_real[:, :, None] * arange[None, None, :])     # (H, N/2, L)
        angles = dtA_imag[:, :, None] * arange[None, None, :]           # (H, N/2, L)
        V_re = decay * mx.cos(angles)
        V_im = decay * mx.sin(angles)

        # ZOH factor: (exp(dtA) - 1) / A, computed for complex A
        # Let dtA = r + jw, A = a + jb
        # (exp(r+jw) - 1) / (a+jb) -- compute real and imag parts
        exp_re = mx.exp(dtA_real) * mx.cos(dtA_imag) - 1.0
        exp_im = mx.exp(dtA_real) * mx.sin(dtA_imag)
        denom = A_real**2 + A_imag**2 + 1e-8
        zoh_re = (exp_re * A_real + exp_im * A_imag) / denom
        zoh_im = (exp_im * A_real - exp_re * A_imag) / denom

        # C_bar = C * B * zoh (complex multiply)
        CB_re = self.C_re * self.B_re - self.C_im * self.B_im
        CB_im = self.C_re * self.B_im + self.C_im * self.B_re
        Cbar_re = CB_re * zoh_re - CB_im * zoh_im
        Cbar_im = CB_re * zoh_im + CB_im * zoh_re

        # K = 2 * Re(Cbar * V)
        K = 2.0 * mx.sum(Cbar_re[:, :, None] * V_re - Cbar_im[:, :, None] * V_im, axis=1)

        return K  # (H, L)

    def __call__(self, u):
        _B, L, _H = u.shape
        K = self.kernel(L)

        u_t = mx.transpose(u, axes=(0, 2, 1))
        fft_size = 2 * L
        K_f = mx.fft.rfft(K, n=fft_size)
        u_f = mx.fft.rfft(u_t, n=fft_size)
        y = mx.fft.irfft(K_f * u_f, n=fft_size)
        y = y[..., :L]

        y = mx.transpose(y, axes=(0, 2, 1))
        return y + u * self.D
```

---

## 10. Key Numerical Details

- **A real part is always negative** (enforced via `-exp(log_A_real)`)
- **A imaginary part is non-positive** by convention (negative frequency)
- **B clipping**: Reference clips B.imag to [-2, 2] for stability (`B_clip=2.0`)
- **dt transform**: Reference default is `softplus` (not `exp`) for numerical stability, though simplified version uses `exp`
- **State dim**: Default is N=64, with N/2=32 complex pairs
- **Channels**: Default is 1 (single-input single-output per SSM copy)

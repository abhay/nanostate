# HiPPO: Recurrent Memory with Optimal Polynomial Projections

Paper: https://arxiv.org/abs/2008.07669
Authors: Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Re (Stanford, 2020)

## Why this paper matters for nanostate

Our `S4DLayer` in `train.py` currently initializes the A matrix with random values:

```python
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))
```

This gives us negative real diagonal entries (via `-exp(log_A)`) in the range roughly
`[-exp(-1), -exp(-4)]` = `[-0.37, -0.018]`. These values have no mathematical relationship
to memory or sequence modeling. The HiPPO paper provides the *principled* initialization
that makes S4 actually work for long-range dependencies. Without HiPPO, S4 is just a
random linear system with no reason to remember anything useful.

The README says it directly: "Random initialization (no HiPPO)" is the first thing to fix.

---

## Core idea

HiPPO frames *memory* as an online function approximation problem. Given a continuous
input signal f(t), at every time t we want to maintain an optimal polynomial approximation
of the entire history f(x) for x <= t, with respect to some measure that weights different
parts of the past.

The key insight: differentiating the optimal projection coefficients c(t) yields an ODE

```
d/dt c(t) = A(t) c(t) + B(t) f(t)
```

where A and B have closed-form expressions determined entirely by the choice of measure.
Different measures = different memory behaviors = different A matrices.

When discretized, this becomes the state space recurrence: `c_{k+1} = A_d c_k + B_d f_k`,
which is exactly what our S4DLayer computes (in convolutional form via FFT).

---

## The three measures and their A matrices

### 1. HiPPO-LegT (Translated Legendre) -- sliding window

Measure: uniform weight on `[t - theta, t]` (fixed-length sliding window of size theta).

**Continuous-time matrices** (ODE: `d/dt c(t) = -(1/theta) A c(t) + (1/theta) B f(t)`):

```
          { (2n+1)^{1/2} (2k+1)^{1/2}    if k <= n    (lower triangular incl. diagonal)
A_nk =    {
          { (-1)^{n-k} (2n+1)^{1/2} (2k+1)^{1/2}   if k > n    (but the paper uses a different normalization)
```

More precisely, with the orthonormal basis (lambda_n = 1):

```
A_nk = (2n+1)^{1/2} (2k+1)^{1/2} * { 1           if k <= n
                                      { (-1)^{n-k}  if k > n

B_n  = (2n+1)^{1/2}
```

With the LMU normalization (lambda_n = (2n+1)^{1/2} (-1)^n):

```
A_nk = (2n+1) * { (-1)^{n-k}  if k <= n
                 { 1           if k > n

B_n  = (2n+1)(-1)^n
```

**Limitation**: theta is a hyperparameter (window length). If you set it wrong, performance
degrades badly. The paper shows LegT can drop from 98% to 70% on pMNIST with wrong theta.

### 2. HiPPO-LagT (Translated Laguerre) -- exponential decay

Measure: `mu(x) = e^{x-t}` for x <= t (exponentially decaying weight on the past).

**Continuous-time matrices** (for the basic case alpha=0, beta=1):

```
A_nk = { 1    if n >= k
        { 0    if n < k

B_n  = 1
```

So A is simply the lower-triangular matrix of all ones (plus (1+beta)/2 = 1 on the diagonal).
The full dynamics are `d/dt c(t) = -Ac(t) + Bf(t)` where A is the lower-triangular ones matrix
with 1 on the diagonal. This is LTI (time-invariant).

### 3. HiPPO-LegS (Scaled Legendre) -- the important one for S4

Measure: `mu_t = (1/t) * 1_{[0,t]}` (uniform weight on the *entire* history [0, t], window
grows with time).

**This is the one used in S4 and the one we should implement.**

**Continuous-time ODE** (from Theorem 2, equation 29 in the paper):

```
d/dt c(t) = -(1/t) A c(t) + (1/t) B f(t)
```

where the A matrix entries are:

```
         { (2n+1)^{1/2} (2k+1)^{1/2}    if n > k
A_nk =   { n + 1                         if n = k
         { 0                              if n < k
```

```
B_n = (2n+1)^{1/2}
```

**In matrix form** (equation 30), A = D * M * D^{-1} where:
- D = diag((2n+1)^{1/2}) for n = 0, ..., N-1
- M is a lower-triangular matrix with M_nk = 2k+1 if k < n, M_nn = n+1, M_nk = 0 if k > n

**Concrete 4x4 example of A for HiPPO-LegS:**

```
A = [ 1     0       0       0    ]
    [ 3^½   2       0       0    ]
    [ 5^½   3·5^½   3       0    ]  (not exact -- see formula)
    [ 7^½   3·7^½   5·7^½   4    ]
```

More precisely, the lower-triangular part:
- Diagonal: A_nn = n + 1
- Below diagonal (n > k): A_nk = sqrt((2n+1)(2k+1))

**Why LegS is special:**
1. **Timescale equivariance** (Prop. 3): if you dilate the input f(alpha*t), the coefficients
   are the same as running HiPPO on f at time alpha*t. No timescale hyperparameter needed.
2. **Bounded gradients** (Prop. 5/7): gradient norm ||dc(t1)/df(t0)|| = Theta(1/t1), meaning
   gradients decay polynomially (not exponentially) -- avoids vanishing gradients.
3. **No hyperparameters**: unlike LegT (needs theta) and LagT (needs theta/beta), LegS
   adapts automatically.

The 1/t factor makes this technically time-varying, but the Euler discretization
`c_{k+1} = (I - A/k) c_k + (1/k) B f_k` removes dependence on step size entirely.

---

## Discretization (continuous -> discrete)

The paper discusses four methods. For our S4D implementation, bilinear (Tustin's method)
is recommended (it's what the S4 paper uses).

Given continuous `d/dt c = Ac + Bf`, the bilinear discretization with step size dt:

```
A_d = (I - dt/2 * A)^{-1} (I + dt/2 * A)
B_d = (I - dt/2 * A)^{-1} * dt * B
```

For **LTI systems** (LegT, LagT), A is constant and this is straightforward.

For **LegS**, the ODE is `d/dt c = -(1/t)Ac + (1/t)Bf`. The discrete recurrence (eq. 4) is:

```
c_{k+1} = (I - A/k) c_k + (1/k) B f_k     (Euler)
```

Or with bilinear:

```
c_{k+1} = (I - A/(k+1) * alpha)^{-1} (I + (1-alpha)/k * A) c_k
         + dt/k * (I - A/(k+1) * alpha)^{-1} B f_k
```

The Zero-Order Hold (ZOH) discretization for LTI systems:

```
A_d = exp(dt * A)
B_d = A^{-1} (exp(dt * A) - I) B
```

---

## What this means for our S4DLayer

### The S4D connection

S4D (Diagonal State Spaces) diagonalizes the HiPPO matrix and works with its eigenvalues.
The full HiPPO-LegS A matrix is N x N lower-triangular (not diagonal), so S4D:

1. Computes the eigenvalues of the HiPPO-LegS matrix
2. Uses these (complex) eigenvalues as the diagonal A matrix
3. Adjusts B and C accordingly

**The key result from the S4D paper**: the eigenvalues of HiPPO-LegS are approximately:

```
lambda_n = -(1/2) + i * pi * (n + 1/2)    for n = 0, 1, ..., N-1
```

But since our code uses **real-valued** diagonal A (no complex numbers), we need the
**NPLR (Normal Plus Low-Rank)** diagonal approximation, or we use the simpler S4D-Lin
initialization:

```
A_n = -(1/2) + i * pi * n    (S4D-Lin, complex)
```

or for a **real-only diagonal** approximation:

```
A_n = -(n + 1/2)    for n = 0, 1, ..., N-1
```

This places eigenvalues at -0.5, -1.5, -2.5, ..., which captures the real part of the HiPPO
spectrum. The imaginary parts (oscillatory modes) are lost, which limits expressivity but
keeps everything real-valued and simple.

### What to change in train.py

**Current code** (line 66):
```python
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))
```

**Option 1: Real-only HiPPO diagonal (simplest, preserves current architecture)**

```python
# HiPPO-inspired: place eigenvalues at -(n + 1/2) for n = 0..N-1
# Then log_A = log(n + 1/2) so that A = -exp(log_A) = -(n + 1/2)
hippo_real = mx.log(mx.arange(1, state_dim + 1).astype(mx.float32) - 0.5)
# Broadcast across d_model (same init for every channel)
self.log_A = mx.broadcast_to(hippo_real[None, :], (d_model, state_dim))
```

This gives A = -0.5, -1.5, -2.5, ..., -(N-0.5). The small eigenvalues (-0.5) create slow
modes that remember long history; the large ones create fast modes for recent detail.

**Option 2: S4D-Lin with complex eigenvalues (more faithful, requires code changes)**

Would require changing the kernel computation to handle complex exponentials:
```python
# A_n = -1/2 + i*pi*n
# This requires complex arithmetic in the Vandermonde computation
```

This is a bigger change but captures the oscillatory modes that let the model represent
richer functions of the input history.

**Option 3: Full HiPPO-LegS matrix (most faithful, but not diagonal)**

Construct the full N x N matrix:
```python
def make_hippo_legs(N):
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = (2*n + 1)**0.5 * (2*k + 1)**0.5
            elif n == k:
                A[n, k] = n + 1
    B = np.array([(2*n + 1)**0.5 for n in range(N)])
    return A, B
```

This would require switching from diagonal to dense state transitions (much more expensive).

### Recommended approach for nanostate

**Start with Option 1** -- it's a one-line change that preserves the diagonal structure:

```python
# Replace random init with HiPPO-inspired spacing
hippo_spacing = mx.arange(1, state_dim + 1).astype(mx.float32) - 0.5
self.log_A = mx.broadcast_to(mx.log(hippo_spacing)[None, :], (d_model, state_dim))
```

This should improve long-range dependency handling immediately because:
- The smallest eigenvalue (-0.5) has a very slow decay rate, remembering long history
- The eigenvalue spacing follows the HiPPO spectrum rather than being random
- It matches what S4D-Real effectively does

Then try **Option 2** (complex S4D-Lin) as a follow-up, which requires modifying the
`kernel()` method to handle complex exponentials but should give the biggest gains.

### B vector initialization

The paper also prescribes B. Currently we have:
```python
self.B = mx.random.normal((d_model, state_dim)) * 0.01
```

For HiPPO-LegS, B_n = (2n+1)^{1/2}. In S4D (after diagonalization), B gets transformed
too, but a reasonable initialization would be:

```python
self.B = mx.broadcast_to(
    mx.sqrt(2 * mx.arange(state_dim).astype(mx.float32) + 1)[None, :],
    (d_model, state_dim)
)
```

### dt (step size) initialization

The paper notes that the discretization step size matters. For LegS, the system is
timescale-invariant, but for the LTI approximation used in S4D, dt controls the effective
timescale. The S4 paper initializes log_dt from a uniform distribution in [log(dt_min),
log(dt_max)] where dt_min=0.001 and dt_max=0.1. Our current `log_dt = zeros` gives
dt=1.0, which may be too large. Consider:

```python
self.log_dt = mx.random.uniform(low=-6.9, high=-2.3, shape=(d_model,))  # [0.001, 0.1]
```

---

## Summary of key formulas

| Measure | A_nk (n > k) | A_nn | A_nk (n < k) | B_n | Time-varying? |
|---------|-------------|------|-------------|-----|--------------|
| LegT | sqrt((2n+1)(2k+1)) | sqrt((2n+1)(2k+1)) | (-1)^{n-k} sqrt((2n+1)(2k+1)) | sqrt(2n+1) | No (LTI, scaled by 1/theta) |
| LagT | 1 | 1 | 0 | 1 | No (LTI) |
| LegS | sqrt((2n+1)(2k+1)) | n+1 | 0 | sqrt(2n+1) | Yes (scaled by 1/t) |

---

## Experimental evidence from the paper

- **pMNIST**: HiPPO-LegS achieves 98.3% (SOTA for recurrent models, beats Transformers at 97.9%)
- **Random init baseline** ("Rand"): 69.93% on pMNIST -- showing that the precise A matrix matters enormously
- **LagT**: 98.15% -- competitive but has a hyperparameter
- **Trajectory classification with timescale shift**: LegS gets 88-95% while LSTM/GRU collapse to 25-35%
- **Function approximation over 10^6 steps**: LegS achieves 0.02 MSE vs LSTM's 0.25

The 28-point accuracy gap between HiPPO-LegS (98.3%) and random init (69.9%) on pMNIST
is the single strongest argument for replacing our random initialization.

---

## Ideas to try in nanostate (ordered by effort)

1. **[Easy] HiPPO real diagonal init**: Replace random log_A with log(n + 0.5) spacing
2. **[Easy] HiPPO B init**: Replace random B with sqrt(2n+1) values
3. **[Easy] dt range init**: Initialize log_dt uniformly in [-6.9, -2.3] instead of zeros
4. **[Medium] S4D-Lin complex init**: Add complex number support to kernel computation,
   use A_n = -1/2 + i*pi*n
5. **[Medium] Separate learning rates**: Use a smaller LR for A (the S4 paper uses 0.001
   for most params but 0.0001 for A), since HiPPO gives a good starting point we don't
   want to move too far from
6. **[Hard] Full NPLR**: Implement the Normal Plus Low-Rank decomposition from the
   original S4 paper, which keeps the full HiPPO structure while enabling efficient computation

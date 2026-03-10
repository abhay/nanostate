# S4: Efficiently Modeling Long Sequences with Structured State Spaces

**Paper**: Gu, Goel, Re (Stanford, 2021). ICLR 2022 Outstanding Paper Honorable Mention.
**arXiv**: https://arxiv.org/abs/2111.00396
**Code**: https://github.com/HazyResearch/state-spaces

## Core idea

State space models (SSMs) map input u(t) to output y(t) through a latent state x(t):

    x'(t) = A x(t) + B u(t)       (continuous)
    y(t)  = C x(t) + D u(t)

The choice of A is critical. Random A gives exponential-decay dynamics and poor long-range performance. The HiPPO matrix gives A structure that provably memorizes input history via polynomial projections, dramatically improving long-range dependency modeling.

The problem: naively computing the SSM with a HiPPO A matrix costs O(N^2 L) operations and O(NL) memory (N = state dimension, L = sequence length). S4 solves this by reparameterizing A as **Normal Plus Low-Rank (NPLR)**, reducing computation to O(N + L) via Cauchy kernels.

## The three views (and why they matter for nanostate)

**1. Continuous-time**: x'(t) = Ax(t) + Bu(t). The "true" model. Parameters live here.

**2. Recurrent (discrete)**: Discretize with step size Delta to get:

    x_k = A_bar x_{k-1} + B_bar u_k
    y_k = C x_k

where A_bar, B_bar depend on the discretization method. This is O(N) per step -- perfect for autoregressive generation. **Our nanostate code does NOT implement this view**, which means we cannot do efficient token-by-token generation. Adding a `step()` method to S4DLayer for recurrent inference is a clear improvement path.

**3. Convolutional**: Unroll the recurrence into a convolution kernel:

    K = (C B, C A_bar B, C A_bar^2 B, ..., C A_bar^{L-1} B)
    y = K * u  (computed via FFT)

This is what our nanostate code does for training. The kernel is computed once, then FFT convolution gives O(L log L) parallel computation.

## HiPPO initialization -- the single biggest missing piece in nanostate

The HiPPO-LegS matrix (equation 2 in the paper):

    A_nk = -( (2n+1)^{1/2} (2k+1)^{1/2} )   if n > k
    A_nk = -(n+1)                              if n = k
    A_nk = 0                                   if n < k

This is a lower-triangular matrix with specific structure derived from Legendre polynomial projections. It allows the state x(t) to maintain an optimal compressed representation of the input history u(s) for s <= t.

**The ablation results (Section 4.4) are striking**: on sequential CIFAR-10, HiPPO initialization reaches ~65% validation accuracy vs ~50% for random diagonal initialization. All initializations reach perfect *training* accuracy, but there is a >15% generalization gap. HiPPO is not just faster convergence -- it gives fundamentally better generalization.

### What this means for nanostate

Our current code (train.py line 66):

    self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))

This is a random real diagonal initialization. The paper shows this is the worst-performing option. Switching to HiPPO-derived diagonal initialization is the highest-impact single change we can make.

## Discretization methods

The paper uses the **bilinear (Tustin) method**:

    A_bar = (I - Delta/2 * A)^{-1} (I + Delta/2 * A)
    B_bar = (I - Delta/2 * A)^{-1} Delta B

For a diagonal A matrix (our case), this simplifies to element-wise operations.

**Our nanostate code uses a different discretization.** Looking at the kernel() method (train.py lines 72-86):

    dt = exp(log_dt)
    A = -exp(log_A)           # negative real
    dtA = A * dt
    V = exp(dtA * l)          # for l = 0..L-1

This is computing exp(A * dt * l), which corresponds to the **exact matrix exponential** (zero-order hold / ZOH discretization). Specifically:

    A_bar = exp(A * dt)    (element-wise for diagonal A)
    B_bar = A^{-1} (A_bar - I) B   (ZOH formula, though our code absorbs this into K)

The kernel formula K[l] = sum_n C[n] * B[n] * exp(A[n] * dt * l) * dt is a Vandermonde-style product that is correct for diagonal SSMs. However, the paper's S4D follow-up (Gu et al., 2022, arXiv:2206.11893) showed that ZOH discretization can behave differently from bilinear, particularly for initialization. The S4D paper recommends ZOH for S4D-Lin (which uses eigenvalues at -1/2 + i*pi*n) and bilinear for S4D-Inv.

### Discretization options to try in nanostate

1. **Bilinear (Tustin)**: For diagonal A, A_bar = (1 + dt*A/2) / (1 - dt*A/2). This preserves stability (maps left half-plane to unit disk). Could improve training dynamics.
2. **ZOH** (what we currently use): A_bar = exp(dt * A). Also stability-preserving for negative real A.
3. **Euler**: A_bar = I + dt * A. Simplest but least stable. Not recommended.

Since our A is real and negative, both ZOH and bilinear are stability-preserving. The choice may matter more when we switch to complex-valued (HiPPO-derived) A.

## The DPLR parameterization and the path to S4D

The full S4 uses A = Lambda - P Q* (Diagonal Plus Low-Rank), which requires Cauchy kernel computation -- complex and hard to implement efficiently. The key insight of the follow-up S4D paper is that **the diagonal restriction alone is surprisingly effective**.

S4D (arXiv:2206.11893) showed that the diagonal approximation of S4's HiPPO matrix recovers the same kernel in the limit of infinite state dimension. The S4D kernel computation is just 2 lines of code (essentially what our nanostate kernel() method does), yet it matches full S4 on almost all benchmarks, averaging 85% on Long Range Arena.

### S4D initialization options for nanostate

**S4D-Lin**: A_n = -1/2 + i * pi * n, for n = 0, 1, ..., N-1.
This places eigenvalues on a vertical line in the left half-plane. The real part -1/2 controls decay rate; the imaginary parts i*pi*n create oscillations at different frequencies, analogous to a Fourier basis. This requires **complex-valued** A, B, C.

**S4D-Inv**: A_n = -1/2 + i * N * (2n+1)^{-1} terms (inverse spacing).

**S4D-Real (what's closest to our code)**: Use only the real parts of the HiPPO eigenvalues: A_n = -(n + 1/2). This is purely real and drops right into our existing code structure. The diagonal of the HiPPO-LegS matrix is -(n+1), so a simple approximation is A_n = -(n+1).

### Concrete change for nanostate

Replace random init with HiPPO-derived diagonal init:

```python
# Instead of:
self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))

# Option A: S4D-Real (simplest, stays real-valued)
# A_n = -(n + 1/2) for n = 0..N-1
A_real = -(mx.arange(state_dim).astype(mx.float32) + 0.5)
# Broadcast across d_model: each feature channel gets the same structured init
self.log_A = mx.broadcast_to(mx.log(-A_real)[None, :], (d_model, state_dim))

# Option B: S4D-Lin (complex-valued, bigger change but more faithful to S4)
# A_n = -1/2 + i*pi*n
# Requires complex B, C and taking real part of output
```

## Architecture details from the paper

- S4 defines a map R^L -> R^L (1-D sequence to 1-D sequence).
- Multiple features handled by H independent copies (one SSM per feature dimension), like depthwise-separable convolution.
- Features mixed by position-wise linear layers between SSM layers.
- The deep S4 model is "a depthwise-separable CNN with global convolution kernels."
- Total parameters per layer: O(H^2) + O(HN) where H = d_model, N = state_dim.

**Our nanostate architecture matches this**: S4DLayer operates independently per d_model channel, SSMBlock adds LayerNorm and MLP for feature mixing. This is correct.

Key differences from the paper's architecture:
- Paper uses **pre-norm** in some experiments, **BatchNorm** in others (Table 11). We use post-norm. Pre-norm is generally more stable.
- Paper uses **GLU activation** after the S4 linear layer for WikiText-103 (to match parameter count). We use GELU in the MLP. GLU could help.
- Paper often uses **dropout** (0.0 to 0.25 depending on task). We have no dropout.
- Paper uses **AdamW** with weight decay (0.01 typical). We use plain Adam.

## Key experimental results

| Benchmark | S4 Result | Notes |
|-----------|-----------|-------|
| LRA Average | 86.09% | First model to solve Path-X (96.35%). All other models fail. |
| sCIFAR-10 | 91.13% | Matches 2-D ResNet with no 2-D inductive bias |
| WikiText-103 | 20.95 ppl | Within 0.8 ppl of Transformer, 60x faster generation |
| Speech Commands (Raw) | 98.32% | 16K-length raw audio, no preprocessing |
| ETTh1 forecasting | 0.116 MSE | Beats Informer on 40/50 settings |

## Practical insights for nanostate experiments

### Highest-impact changes (ordered by expected return)

1. **HiPPO initialization for A** (Section 4.4 ablation shows >15% generalization gap on sCIFAR). Start with S4D-Real: A_n = -(n+0.5). This is a one-line change.

2. **Learnable dt with proper initialization** (log_dt currently initialized to 0, meaning dt=1). The paper uses a learnable step size Delta. The S4D paper suggests initializing log_dt uniformly in [log(dt_min), log(dt_max)] where dt_min=0.001 and dt_max=0.1. Our dt=1 is way too large.

3. **Pre-norm instead of post-norm**. Current code: `x = self.norm1(x + self.ssm(x))`. Change to: `x = x + self.ssm(self.norm1(x))`. This is more stable and is standard practice.

4. **Separate learning rates for SSM parameters vs other parameters**. The paper (Appendix D.2) reduces the learning rate for HiPPO parameters (A, P, Q, B, C, Delta) to a maximum of 0.001 "which improves stability since the HiPPO equation is crucial to performance." Our code uses a single lr for everything.

5. **Weight decay** (AdamW instead of Adam). The paper uses 0.01 weight decay in most experiments.

6. **Dropout**. The paper uses up to 0.25 dropout. Our model has none.

7. **Complex-valued S4D** (S4D-Lin initialization). Bigger change but closer to the theory. Requires complex A, B, C and taking the real part of the output kernel.

### Longer-term directions

- **Recurrent inference mode**: Add a `step()` method that uses the recurrent view for autoregressive generation. This would enable fast inference benchmarking.
- **Resolution adaptation**: The continuous-time parameterization means we can change Delta at test time to handle different sampling rates (shown for speech at 0.5x frequency, Section 4.3).
- **Selective scan (Mamba)**: Make B, C, dt input-dependent. This is the natural next step after getting S4D working well.

## Connection to our kernel() implementation

Our kernel computation (train.py lines 72-86) is mathematically:

    K[h, l] = dt[h] * sum_n C[h,n] * B[h,n] * exp(A[h,n] * dt[h] * l)

This is correct for a diagonal SSM with ZOH discretization. The paper's equation (5) is:

    K = (C B_bar, C A_bar B_bar, ..., C A_bar^{L-1} B_bar)

For diagonal A_bar = exp(A * dt) and B_bar involving A^{-1}(exp(A*dt) - I) * B, these are equivalent up to the handling of the B_bar term. Our formulation absorbs the dt scaling into the kernel directly. This is a valid simplification for the diagonal case.

The Vandermonde structure V[h,n,l] = exp(dtA[h,n] * l) is the key computational primitive. For the full DPLR S4, computing this requires Cauchy kernels. For diagonal S4D (our case), it is trivially a Vandermonde product, which is exactly what our code computes.

## Summary for the autonomous research loop

The S4 paper establishes that:
1. SSMs are a principled sequence model framework with three interchangeable views.
2. The A matrix initialization (HiPPO) is the single most important ingredient.
3. The NPLR/DPLR parameterization makes computation efficient, but the diagonal approximation (S4D) is nearly as good and far simpler.
4. Our nanostate code has the right overall structure but is missing: HiPPO init, proper dt init, pre-norm, weight decay, dropout, and separate SSM learning rates.

The first experiment should be HiPPO-derived A initialization. Everything else is secondary.

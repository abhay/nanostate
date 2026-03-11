# Analysis: autoresearch/mar10

Run tag: mar10 | 21 experiments | Task: lm (TinyShakespeare byte-level)

## Metric Trajectory

| Exp | val_bpb | Change | What |
|-----|---------|--------|------|
| 1 | 2.3249 | baseline | d=128, L=4, random A, post-norm, Adam lr=1e-3 |
| 4 | 2.3031 | -0.9% | Pre-norm residual blocks |
| 9 | 2.239 | -2.8% | Scale to d=384 (width > depth at 1000 steps) |
| 14 | 2.2524 | -0.6% | SiLU gated block (Mamba-style, replaces SSM+MLP) |
| 16 | 2.2225 | -1.3% | Lower LR to 5e-4 |
| 20 | 2.2093 | -0.6% | Cosine LR decay with 100-step warmup |

**Total improvement: 2.3249 -> 2.2093 (-5.0%)**

## What Always Works

- **Pre-norm** over post-norm (small but consistent win)
- **Width scaling** up to d=384 (d=512 hits diminishing returns at 1000 steps)
- **SiLU gating** (Mamba expand-and-gate pattern, more param-efficient than width scaling)
- **Cosine LR decay** with warmup (finer convergence in later training)
- **LR ~5e-4** for the gated d=384 model (1e-3 too high, 3e-4 too low at 1000 steps)

## What Never Works (Don't Retry)

- **HiPPO A init** (3 attempts: alone, with dt fix, combined with complex S4D-Lin — all worse). Root cause: dt and A are tightly coupled. HiPPO eigenvalues span [-0.5, -63.5]; with dt=1.0 most states die (exp(-63.5)=0), with dt in [0.001,0.1] some channels get no dynamics. The random A init in [-0.37, -0.018] with dt=1.0 happens to give a useful range of decay rates for 256-length sequences.
- **dt init alone** [0.001, 0.1] — too small for the random A range, removes all dynamics
- **More steps** beyond 1000 — TinyShakespeare overfits (train drops, val rises) even with cosine decay
- **Weight decay at 1000 steps** — regularization hurts when not yet overfitting
- **More depth** (L=6, L=8) — underfits at 1000 steps with these widths
- **seq_len=512** — overfits more, 2x slower, no benefit
- **state_dim=16** — LTI S4D still benefits from N=64 (unlike selective Mamba which caps at N=16)
- **Output proj scaling + grad clip** — too aggressive with only 4 layers

## What's Promising But Unfinished

1. **HiPPO init with matched dt**: The init itself is mathematically correct but needs careful dt calibration. A narrow dt range (e.g., [0.01, 0.05]) with HiPPO A might work. Alternatively, the Mamba-1 init (log_A = log(arange(1, N+1))) with appropriate dt could be better than random.
2. **Complex S4D-Lin**: Tried once bundled with too many other changes. The oscillatory modes from imaginary eigenvalues should help — but needs to be isolated and tested with the current gated block architecture.
3. **Overfitting as a scaling signal**: The model overfits TinyShakespeare at 2000 steps. Weight decay + longer training could unlock more capacity. Need to try weight decay with MORE steps (e.g., wd=0.1 + 2000 steps) rather than at 1000 steps where it just hurts.

## Recommended Next Experiments (Ranked)

1. **Conv1d before SSM** (Mamba uses depthwise conv1d, kernel=4): Adds local context before the SSM processes. The SSM sees smoothed local features. Should be a clean win.
2. **SiLU activation on SSM path**: Mamba applies SiLU after conv1d, before the SSM. Our block goes straight from in_proj to SSM with no nonlinearity on the x path.
3. **B/C initialization at larger scale**: Current B,C init is 0.01 * randn — very small. Try 0.1 or 1.0 scale, or constant ones for B.
4. **Dropout** (0.1): Light dropout inside the block to regularize.
5. **Batch size sweep**: Try batch=64 or batch=16 to see if batch size matters.
6. **Weight decay + 2000 steps**: wd=0.01 + 2000 steps with cosine decay could unlock longer training.
7. **Selective dt** (input-dependent): Make log_dt a function of input via a linear projection + softplus. Breaks FFT (needs recurrent scan) but is Mamba's highest-impact change.
8. **Larger model with wd**: d=512, L=4, wd=0.05, 1500 steps — bigger model with regularization.
9. **Embedding scaling**: Scale embeddings by sqrt(d_model) as in Transformers.
10. **Complex S4D-Lin (isolated)**: Retry with the current gated block, keeping everything else fixed.

## Key Insight: The Overfitting Regime

At d=384 with 1000 steps, we're in a sweet spot where:
- Train loss (1.22) is close to val loss (1.53) — mild overfitting
- More capacity (width, gating) helps the val metric
- But more training steps pushes into severe overfitting

The path forward is either:
a) Add regularization to enable longer training (dropout, weight decay at higher step counts)
b) Improve the model's inductive bias so it generalizes better at 1000 steps (conv1d, better init, selectivity)

Option (b) is likely higher-leverage for SSM research purposes.

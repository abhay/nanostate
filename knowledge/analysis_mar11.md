# Analysis: autoresearch/mar11

Run tag: mar11 | 16 new experiments (37 total) | Tasks: lm + lm-tok

## Metric Trajectory

### lm (TinyShakespeare byte-level)

| Exp | val_bpb | Change | What |
|-----|---------|--------|------|
| mar10 best | 2.2093 | - | cosine LR + gated block + d=384 |
| 22 | 2.1838 | -1.2% | HiPPO-LegS A+B init (PR#4, dt calibrated) |
| 26 | 2.1745 | -0.4% | LR bump to 7e-4 |

**Total mar11 lm improvement: 2.2093 → 2.1745 (-1.6%)**
**Total cumulative lm improvement: 2.3249 → 2.1745 (-6.5%)**

### lm-tok (FineWebEdu BPE, 50K vocab) — NEW

| Exp | val_bpb | Change | What |
|-----|---------|--------|------|
| 29 | 8.0317 | baseline | d=384, L=4, 1000 steps |
| 34 | 7.4744 | -6.9% | 3000 steps (2.5 passes over data) |

**Total lm-tok improvement: 8.0317 → 7.4744 (-6.9%)**

## What Always Works

*Inherited from mar10:*
- **Pre-norm** over post-norm
- **Width scaling** to d=384 (d=512 diminishing returns)
- **SiLU gating** (Mamba expand-and-gate pattern)
- **Cosine LR decay** with warmup

*New in mar11:*
- **HiPPO-LegS init** with calibrated dt (resolved the mar10 HiPPO failures). Key: dt in [0.008, 0.03] matches HiPPO eigenvalue range [0.5, 63.5], giving decay rates exp(-0.004) to exp(-1.9) — a useful spread for L=256.
- **LR ~7e-4** for HiPPO-init model (was 5e-4 with random A init). HiPPO's structured eigenvalues tolerate slightly higher LR.
- **More training steps on lm-tok**: 3000 steps = 24.6M tokens seen (2.5 passes) gives 7.47 bpb vs 8.03 at 1000 steps. Dataset is 10M tokens, far too large for 1000-step underfitting.

## What Never Works (Don't Retry)

*All of mar10's "never works" remain valid, EXCEPT:*
- ~~HiPPO A init~~ — NOW WORKS with properly calibrated dt. The mar10 conclusion was premature; the failure was due to dt=1.0 (too large) or dt=[0.001,0.1] (too small). The sweet spot is dt=[0.008,0.03].

*New in mar11:*
- **Depthwise conv1d (k=4) + SiLU before SSM on lm**: Increases overfitting without improving generalization. Train loss drops but val loss unchanged. The model already has enough capacity for TinyShakespeare.
- **Embedding scaling sqrt(d_model)**: sqrt(384)≈19.6 is too aggressive. Inflates embedding magnitudes far beyond the residual stream range.
- **Dropout on lm at 1000 steps**: Regularization still hurts. The model isn't overfitting enough for dropout to help.
- **Weight tying on lm-tok**: Hurts at 1000 steps (8.18 vs 8.03) and at 3000 steps (7.53 vs 7.47). The separate head can learn a different projection optimized for output, which helps.
- **Depth scaling (L=6) on lm-tok**: Converges to the exact same val_bpb as L=4 (7.475 vs 7.474). Extra depth learns faster (reaches L=4's 3000-step result by step 2000) but hits the same ceiling. The bottleneck is NOT backbone depth.
- **LR=1e-3 on lm-tok**: Slightly worse than 7e-4 (7.51 vs 7.47).
- **Smaller d_model on lm-tok** (d=128 or d=256): Worse metrics despite being faster. The 50K vocab needs sufficient d_model for the embedding to represent token relationships.
- **Batch=16 on lm-tok**: Sees fewer tokens per step, hurts throughput-adjusted learning.

## Key Structural Insight: The LTI Ceiling

On lm-tok, both L=4 and L=6 converge to val_bpb ≈ 7.47. This is NOT a capacity limit (L=6 has 5% more backbone params) — it's an **inductive bias limit**. The S4D layer is Linear Time-Invariant (LTI): B, C, and dt are fixed parameters, so the model learns a single convolution kernel applied identically to all inputs.

An LTI model cannot:
- Selectively attend to or ignore specific tokens based on content
- Dynamically adjust its "memory horizon" based on context
- Implement content-dependent gating within the SSM itself

This is exactly what Mamba's selective scan addresses: making B, C, dt **input-dependent** so the SSM can selectively remember or forget based on what it sees.

## The lm-tok Bottleneck Map

```
50K vocab embedding: 19.3M params (45%)  ← dominates param count
50K head linear:     19.3M params (45%)  ← dominates param count
SSM backbone:         4.3M params (10%)  ← does the actual modeling
```

90% of parameters are in the embedding/head. The SSM backbone at 4.3M is far too small relative to the vocab size. Solutions:
1. Weight tying (saves 19.3M, but hurts quality as tested)
2. Bigger backbone (L=6 doesn't help — LTI ceiling)
3. Better SSM (selective scan — breaks LTI ceiling)
4. Smaller vocab / subword approach (out of scope, data.py is read-only)

## What's Promising But Unfinished

1. **Selective scan (input-dependent B, C, dt)**: The biggest remaining architectural improvement. Requires replacing FFT convolution with a recurrent scan (for loop over L). Will be slower per step but should dramatically improve quality by breaking the LTI ceiling. MLX lacks `mx.scan`, so needs a Python for loop.

2. **Conv1d + SiLU on lm-tok**: Failed on lm (overfitting), but lm-tok has 10,000x more data. The conv1d's local context mixing might help where overfitting isn't the bottleneck.

3. **More training on lm-tok** (5000+ steps): 3000 steps sees 24.6M tokens (2.5 passes). The model is still improving at step 3000 — more steps could push val_bpb lower. Cosine decay may need a longer schedule.

4. **Softplus for dt** (Mamba uses softplus, we use exp): Different gradient profile — softplus clips dt gradients to [0,1] range, potentially more stable.

5. **RMSNorm instead of LayerNorm**: Mamba-2 uses RMSNorm. Simpler (no mean subtraction) and slightly faster.

6. **C init scale**: Currently 0.01 × randn — very small. Larger init (0.1 or 1.0) might help the SSM output have more influence early in training.

## Recommended Next Experiments (Ranked)

1. **Selective scan on lm-tok** (highest impact): Implement input-dependent B, C, dt with recurrent scan. Even with Python for loop overhead, the quality improvement could be large enough to justify. Try with reduced state_dim=16 (Mamba uses N=16) and seq_len=128 to manage speed.

2. **Conv1d + SiLU before SSM on lm-tok**: Re-test the conv1d approach from mar10 on lm-tok where overfitting isn't an issue. Quick code change, ~77 min test.

3. **5000 steps on lm-tok**: Simple scaling — more training without code changes. ~2 hours but easy.

4. **C init scale on lm-tok**: Try C init 0.1 or 1.0 instead of 0.01. Quick code change.

5. **Softplus dt + Mamba-1 A init** (log_A = log(arange(1, N+1))): Our HiPPO-LegS init uses -(n+0.5) but Mamba uses -n. Try the Mamba variant with softplus dt.

6. **d=384 L=4 + weight tying + 5000 steps**: At 3000 steps weight tying was close (7.53 vs 7.47). With 5000 steps the tied model might catch up while using 45% fewer params.

7. **Gradient accumulation** (effective batch=64 or 128): Smoother gradients for the undertrained regime. Implement by accumulating grads over 2-4 mini-batches.

8. **Complex S4D-Lin on lm-tok** (isolated): Add imaginary eigenvalues for oscillatory modes. Should be tested on lm-tok where overfitting isn't masking the effect.

## Speed Notes

lm-tok runs are ~77 minutes (1450ms/step × 3000 steps). The FFT convolution at d_inner=768 is the bottleneck. Options to speed up:
- Reduce state_dim from 64 to 32 (halves Vandermonde cost, untested on lm-tok)
- Use selective scan with recurrent mode (avoids FFT entirely, but Python loop overhead)
- Use seq_len=128 instead of 256 (halves FFT cost, fewer tokens per step but can compensate with more steps)

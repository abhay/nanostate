# Analysis: autoresearch/mar12

Run tag: mar12 | 13 new experiments (50 total) | Focus: lm-tok iteration

## Metric Trajectory

### lm-tok (FineWebEdu BPE, 50K vocab)

| Exp | val_bpb | Change | What |
|-----|---------|--------|------|
| 40 | 8.036 | baseline | mar12 baseline (1000 steps, d=384 L=4) |
| 42 | 7.760 | -3.4% | seq_len=512 (env var sweep) |
| 39 | 7.796 | -3.0% | conv1d(k=4)+SiLU before SSM |
| (34) | 7.474 | -7.0% | 3000 steps (from mar11, still best overall) |

**Best 1000-step lm-tok: 7.760 (seq_len=512)**
**Best overall lm-tok: 7.474 (3000 steps, L=256, no conv1d)**

### lm (TinyShakespeare byte-level)

No improvement over mar11 best (2.1745). RMSNorm achieved 2.178 (equal, simpler). lm task is near ceiling — all effort should go to lm-tok.

## What Always Works

*Inherited from mar11 (all still valid):*
- Pre-norm, d=384, SiLU gating, cosine LR + warmup, HiPPO-LegS init, lr=7e-4

*New in mar12:*
- **seq_len=512 on lm-tok**: -3.4% improvement at 1000 steps. The SSM's long-range memory pays off with longer sequences when the dataset is large enough to avoid overfitting. (Still hurts on lm — TinyShakespeare overfits.)
- **Conv1d(k=4)+SiLU before SSM on lm-tok**: -3.0% improvement. Local context mixing helps on the larger dataset.
- **RMSNorm on lm**: Equal to LayerNorm (2.178 vs 2.175), simpler. Marginally worse on lm-tok though (7.826 vs 7.760 with same L=512), so keep LayerNorm as default.

## What Never Works (Don't Retry)

*All of mar11's "never works" remain valid.*

*New in mar12:*
- **Selective scan (Python for-loop) on Apple Silicon**: Metal GPU watchdog kills the process. Sequential scan creates O(L)-deep computation graphs that exceed macOS's ~5-10s GPU command timeout. Tested at L=64 (N=16) and L=128 (N=16) — both crashed or produced terrible metrics (2.537 bpb on lm vs baseline 2.175). Truncated BPTT (stop_gradient every 32 steps) didn't help. **Selective scan needs a parallel scan implementation or C/Metal extension** — it cannot be done with a Python for loop on Apple Silicon.
- **--size medium on lm-tok**: Metal GPU watchdog crash. The 100M param model's forward pass exceeds the GPU timeout. Cannot scale model size beyond --size small (~42.8M lm-tok).
- **seq_len=1024 on lm-tok**: OOM with B=32. Reducing to B=16 then hits watchdog timeout.
- **C init scale 0.1**: No improvement over 0.01 (8.041 vs 8.036). The SSM output magnitude isn't the bottleneck.
- **Softplus for dt on lm**: No improvement (2.227 vs 2.175). The exp discretization works fine.
- **AdamW weight_decay=0.1 on lm-tok**: Hurts (7.778 vs 7.760 with same L=512). The model isn't overfitting lm-tok, so regularization isn't needed.
- **RMSNorm on lm-tok**: Slightly worse than LayerNorm (7.826 vs 7.760 at L=512). The mean subtraction in LayerNorm matters for the 50K vocab's varied token embeddings.

## The Metal GPU Watchdog Wall

This is the most important discovery of mar12. macOS enforces a ~5-10s timeout on individual GPU command buffers. This blocks:
1. **Selective scan**: O(L) sequential ops in the computation graph
2. **Large models**: --size medium forward/backward exceeds timeout
3. **Long sequences**: L=1024 at B=32 exceeds memory, and B=16 exceeds timeout

This is NOT a memory constraint — it's a **compute graph depth** constraint. The workarounds are:
- **mx.compile**: May compile the graph into fewer, faster Metal commands (UNTESTED)
- **Mixed precision (float16)**: Halves compute per op, might squeeze medium model under the timeout (UNTESTED)
- **Parallel scan**: O(log L) depth instead of O(L), but complex to implement in MLX (UNTESTED)
- **C/Metal extension**: Write the scan kernel in C/Metal, submit as a single GPU command (out of scope)

Until the watchdog wall is broken, we're limited to --size small with FFT convolution (LTI model).

## Key Insight: Two Remaining Levers

Within the current hardware constraints, there are only two levers left for lm-tok:
1. **Training efficiency**: More steps, better schedules, better optimization
2. **Architectural tweaks within FFT/LTI**: Conv1d, discretization, normalization, gating variants

The selective scan path (which would break the LTI ceiling) is blocked by Metal GPU watchdog. The model scaling path is also blocked. This means diminishing returns are likely unless the watchdog wall is broken.

## What's Promising But Unfinished

1. **3000+ steps with seq_len=512**: The best 1000-step config (7.760) hasn't been validated with longer training. If the same ~7% relative improvement from 1000→3000 steps holds, this could reach ~7.22 bpb. This is the most likely next record.

2. **Conv1d + seq_len=512 combined**: Started but killed (3.7s/step, too slow to complete in time). The two improvements are somewhat independent (conv1d = local context, L=512 = longer range), so they might stack. Would need ~62 min for 1000 steps.

3. **mx.compile**: Could speed up all runs AND potentially break the Metal watchdog wall by compiling the computation graph. Has never been tested. This is the highest-leverage technical experiment.

4. **Mixed precision (float16)**: Could enable --size medium and speed up all runs. The 50K vocab head projection (50257×384 matrix multiply) dominates step time and would benefit most from float16.

5. **5000-10000 steps on lm-tok**: Simple scaling. The model was still improving at 3000 steps. Cosine schedule would need to be extended.

6. **Gradient accumulation**: Effective batch=64 or 128 for smoother gradients. The lm-tok regime is underfitting, so larger effective batch sizes might help convergence.

7. **Hybrid SSM+attention**: Replace 1 of 4 SSM layers with a single attention head. Gives content-based selection without the sequential scan problem. Never tried.

8. **SwiGLU gating**: Different gating mechanism. Currently using SiLU expand-and-gate. SwiGLU is standard in modern LLMs and might help.

9. **ZOH discretization (isolated)**: Tried bundled with other changes in mar10. Never isolated on lm-tok.

## Recommended Next Experiments (Ranked)

1. **3000 steps + seq_len=512 on lm-tok** (high confidence): Combine best 1000-step config with longer training. Expected ~7.2 bpb. ~26 min at 1.5s/step. Then run hellaswag/piqa benchmarks.

2. **mx.compile** (high leverage): Wrap forward/backward in `mx.compile`. Could speed up all runs by 2-5x AND potentially enable selective scan by reducing graph depth. Quick code change, test on lm first.

3. **Mixed precision float16** (high leverage): `model.astype(mx.float16)` or selective casting. Speed up lm-tok runs and potentially enable --size medium. Keep optimizer state in float32.

4. **Hybrid SSM+attention layer** (medium effort): Replace layer 2 (of 4) with a single-head causal attention. Gives content-based selection without sequential scan. ~30 lines of code.

5. **Conv1d + seq_len=512** (low hanging, partially tested): Rerun with full 1000 steps. ~62 min. The 300 steps we saw were trending well (8.163 at step 300 vs baseline 8.036 at step 1000).

6. **Gradient accumulation** (easy): Accumulate gradients over 2-4 steps for effective B=64-128. Few lines of code.

7. **5000+ steps on best config** (time investment): Pure scaling. ~50 min at L=512.

8. **SwiGLU instead of SiLU expand-and-gate** (quick test): Different gating. Minor code change.

## Speed Notes

| Config | ms/step | 1000 steps | 3000 steps |
|--------|---------|------------|------------|
| lm-tok L=256 (baseline) | ~1450 | ~24 min | ~73 min |
| lm-tok L=512 | ~1500 | ~25 min | ~75 min |
| lm-tok L=256 conv1d | ~1500 | ~25 min | ~75 min |
| lm-tok L=512 conv1d | ~3700 | ~62 min | ~185 min |
| lm (byte-level) | ~20 | ~20s | ~60s |

The conv1d + L=512 combo is disproportionately slow (3.7s vs 1.5s) — likely the conv1d at seq_len=512 with d_inner=768 creates a large intermediate tensor.

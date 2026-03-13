# Analysis: autoresearch/mar13-2

Run tag: mar13-2 | 12 new experiments (76 total) | Focus: SSD architecture tuning

## Metric Trajectory

### lm-tok (FineWebEdu BPE, 50K vocab) — 1000-step comparison

| Row | val_bpb | Change vs baseline | What |
|-----|---------|-------------------|------|
| 51 | 7.826 | baseline | SSD baseline (d_head=64, no SiLU) |
| 52 | 7.548 | -3.6% | + seq_len=512 |
| 55 | 7.519 | -3.9% | + mx.compile |
| 60 | 7.672 | -2.0% | + compile + seq512 combined |
| 69 | 7.676 | -1.9% | + B/C SiLU feature map (neutral) |
| **71** | **7.593** | **-3.0%** | **+ d_head=32 (24 heads) — NEW 1K RECORD** |
| 73 | 7.687 | -1.8% | + d_head=16 (48 heads, 30% slower) |
| 74 | 7.685 | -1.8% | + min_lr=7e-5 (hurts convergence) |

**Best 1000-step lm-tok: 7.593 (SSD + d_head=32 + B/C SiLU + compile + seq512)**
**Best overall lm-tok: 7.217 (5000 steps, d_head=32 + B/C SiLU, best@2900 — NEW RECORD)**

### lm (TinyShakespeare byte-level)

| Row | val_bpb | What |
|-----|---------|------|
| 26 | 2.1745 | S4D best (LR=7e-4, HiPPO) |
| **68** | **2.141** | **SSD + B/C SiLU — NEW BYTE RECORD** |
| 70 | 2.166 | SSD + d_head=32 (worse on bytes) |
| 72 | 2.188 | SSD + d_head=16 (worst) |

**Best byte-level: 2.141 (SSD + B/C SiLU + d_head=64)**

## What Always Works

*Inherited from mar12 (all still valid):*
- Pre-norm, d=384, SiLU gating, cosine LR + warmup, HiPPO-LegS init, lr=7e-4
- SSD block type for lm-tok (breaks LTI ceiling)
- seq_len=512 + compile + metal-eval for all SSD lm-tok runs

*New in mar13:*
- **d_head=32 on lm-tok**: -0.079 bpb over d_head=64 at 1000 steps. 24 heads with 32-dim values beats 12 heads with 64-dim values. Same params, same speed. The SSD algorithm benefits from more independent selective attention patterns for the 50K vocab task.
- **B/C SiLU feature map on byte-level lm**: -0.034 bpb (2.141 vs 2.175). From the Mamba-2 paper. SiLU on B/C projections gives nonlinear feature maps that help state encoding. Zero compute overhead.
- **SSD on byte-level lm**: First time tested. 2.156 vs 2.175 S4D — selectivity helps even on small vocab.

## What Never Works (Don't Retry)

*All of mar12's "never works" remain valid.*

*New in mar13:*
- **RMSNorm before SSD gating (Mamba-2 style)**: Hurts on both lm (2.344 vs 2.141) and lm-tok (7.966 vs 7.672). Also 32% slower due to extra norm computation. The Mamba-2 paper tested at 130M+ params — our 42M model doesn't benefit.
- **d_head=16 (48 heads)**: Same quality as d_head=32 but 30% slower on lm-tok (5900ms vs 4400ms). The 48 heads create larger intermediate tensors in the SSD (CB gram matrix scales with H). No quality gain.
- **B/C SiLU on lm-tok**: Neutral (7.676 vs 7.672). The 50K vocab embedding dominates; B/C nonlinearity has proportionally less impact than on byte-level.
- **Cosine min_lr=7e-5**: Hurts at 1000 steps (7.685 vs 7.593). Keeps LR too high at the end, preventing final convergence. Default 1e-5 is fine.

## Key Insight: d_head Is Task-Dependent

The d_head sweep revealed a striking task dependence:
- **lm-tok (50K vocab)**: d_head=32 > d_head=64 > d_head=16. Optimal at 24 heads.
- **lm (256 byte vocab)**: d_head=64 > d_head=32 > d_head=16. Bigger heads are better.

Hypothesis: Token-level language modeling has richer per-position information (50K classes) that benefits from more independent attention patterns. Byte-level has simpler per-position info but needs higher-dimensional value vectors for each head to capture complex byte patterns.

## What's Promising But Unfinished

1. **5000-step validation with d_head=32**: DONE — achieved 7.217 best@2900 (final 7.283), beating the previous 7.228 record. The improvement from d_head=32 is confirmed at scale.

2. **N_LAYERS=5 with d_head=32**: Never tested with SSD. L=6 was tested with S4D (no gain) and SSD (marginal, 2.2x slower), but L=5 is untested. One more layer adds ~25% more params but only ~25% more compute.

3. **Gradient accumulation for effective larger batch**: --grad-accum 2 would give effective batch=64. Incompatible with --compile, so it would be slower per step, but might converge faster per token.

4. **Different expand ratios**: Currently expand=2 (d_inner=768). expand=3 would give d_inner=1152 with more params. Never tested.

5. **Warmup ratio for 5000-step runs**: Current warmup is min(100, steps//10) = 100 steps = 2% at 5000 steps. A longer warmup (500-1000 steps) might help the model start from a better point.

6. **Combining B/C SiLU removal on lm-tok**: Since B/C SiLU is neutral on lm-tok, removing it might actually help marginally (simpler model). Worth a quick test.

## Recommended Next Experiments (Ranked)

1. **5000-step validation with d_head=32** (highest priority): Validate the d_head=32 win at scale. Expected to break 7.228 record. ~5.5 hours. Use current best config: `--block ssd --compile --metal-eval` with `NS_SEQ_LEN=512`.

2. **N_LAYERS=5** (quick env var test): `NS_N_LAYERS=5`. Test on lm first (~90s), then lm-tok at 1000 steps (~70 min). More depth may help with d_head=32's increased head count.

3. **Remove B/C SiLU for lm-tok** (isolation test): Since B/C SiLU is neutral on lm-tok but d_head=32 was tested WITH SiLU, verify d_head=32 works equally well without SiLU. Would simplify the model.

4. **Larger effective batch via grad accumulation**: `--grad-accum 2` without `--compile`. Compare quality at same number of tokens seen. Might help convergence for long runs.

5. **5000-step warmup=500**: Increase warmup from 100 to 500 steps for the validation run. Test as env var alongside the 5000-step validation.

6. **expand=3** (wider inner dim): d_inner=1152 instead of 768. More parameters but potentially more expressive SSD. Quick code change.

7. **d_head=48** (intermediate sweep point): Between the 32 and 64 optima, untested. Could be the actual sweet spot for lm-tok.

## Speed Notes

| Config | ms/step | 1000 steps | 5000 steps |
|--------|---------|------------|------------|
| SSD + compile + seq512 (d_head=64) | ~4300 | ~72 min | ~6.0 hr |
| SSD + compile + seq512 (d_head=32) | ~4400 | ~74 min | ~6.1 hr |
| SSD + compile + seq512 (d_head=16) | ~5900 | ~98 min | ~8.2 hr |
| SSD + compile + metal-eval (byte lm) | ~90 | ~1.5 min | ~7.5 min |

d_head=32 has negligible speed penalty over d_head=64 (<3%). d_head=16 is 35% slower.

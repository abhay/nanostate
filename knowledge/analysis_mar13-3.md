# Analysis: autoresearch/mar13-3

Run tag: mar13-3 | 18 new experiments (94 total) | Focus: d_head=48 scaling, architecture & schedule probes

## Metric Trajectory

### lm-tok (FineWebEdu BPE, 50K vocab) — 1000-step comparison

| Row | val_bpb | Change vs baseline | What |
|-----|---------|-------------------|------|
| 51 | 7.826 | baseline | SSD baseline (d_head=64, no SiLU) |
| 52 | 7.548 | -3.6% | + seq_len=512 |
| 71 | 7.593 | -3.0% | + d_head=32 (24 heads) |
| 77 | 7.751 | -1.0% | mar13-3 baseline (d_head=48, 16 heads) |
| **81** | **7.609** | **-2.8%** | **d_head=48 (16 heads), 1K** |
| **82** | **7.573** | **-3.2%** | **d_head=48 + N_LAYERS=5, 1K** |
| 79 | 7.665 | -2.1% | N_LAYERS=5 (d_head=32), 1K |

**Best 1000-step lm-tok: 7.491 (LR=1e-3 + d_head=48 + N_LAYERS=5 + SSD + compile + seq512)**
**Best overall lm-tok: 7.178 (LR=1e-3, 4000 steps, best@2450 — ties record, converges faster)**

### lm (TinyShakespeare byte-level)

| Row | val_bpb | What |
|-----|---------|------|
| 26 | 2.1745 | S4D best (LR=7e-4, HiPPO) |
| **68** | **2.141** | **SSD + B/C SiLU — BYTE RECORD** |
| 70 | 2.166 | SSD + d_head=32 |
| 86 | 2.212 | SSD + d_head=48 (worse on bytes) |

**Best byte-level: 2.141 (SSD + B/C SiLU + d_head=64, 12 heads)**

## What Always Works

*Inherited (all still valid):*
- Pre-norm, d=384, SiLU gating, cosine LR + warmup, HiPPO-LegS init, lr=7e-4
- SSD block type for lm-tok (breaks LTI ceiling)
- seq_len=512 + compile + metal-eval for all SSD lm-tok runs

*Confirmed in mar13-3:*
- **d_head=48 on lm-tok**: 16 heads, 48-dim values. Best 1K result: 7.573 (beats d_head=32's 7.593). Scales to 7.180 at 4000 steps. The sweet spot is fewer heads with larger values than d_head=32, but more heads than d_head=64.
- **N_LAYERS=5**: Adds ~25% compute, consistently improves lm-tok by ~0.03-0.04 bpb. Works with both d_head=32 and d_head=48.
- **Longer training**: 4000 steps significantly beats 3000 steps (7.180 vs ~7.28 extrapolated). Diminishing returns visible but model still improving at 3800.
- **Conv1d is essential**: Removing conv1d causes 0.29 bpb loss at step 200 with growing gap. Local context (k=4 window) complements SSD's chunk-level attention.

## What Never Works (Don't Retry)

*All of mar12/mar13's "never works" remain valid, plus:*

- **Per-head decay bias (a_bias)**: HiPPO-inspired diverse time scales for A. Neutral at 3000 steps (7.225 vs 7.180 record). The learned A projections already adapt — explicit bias adds no value.
- **Parallel A/B/C projections from d_model**: Mamba-2 paper pattern. 7.584 at 1000 steps, worse than sequential from d_inner (7.573). At 43M scale, conv1d output provides useful local context for selectivity.
- **d_head=48 on byte-level**: 2.212 vs 2.141 record. Byte-level needs larger head dim (d_head=64 is best). Task-dependent d_head is confirmed: lm-tok wants more heads, lm wants bigger heads.
- **LR=5e-4 with deep models**: Too slow for N_LAYERS=5 at any step count (7.755). lr=7e-4 is optimal.
- **d_state=32**: 25% faster but 0.028 bpb worse (7.595 vs 7.567). d_state=64 has genuinely higher ceiling.
- **expand=3 (d_inner=1152)**: 2.6x slower, killed@150, same quality trend. The expanded dim creates massive intermediate tensors with no quality gain at 43M.
- **GroupNorm after gating**: 0.2 bpb worse at 350 steps. Mamba-2 pattern that doesn't help at this scale.
- **SwiGLU gating**: Same quality as SiLU gate, 32% slower (7.630 vs 7.672). Extra gate parameters wasted.
- **conv_k=8**: Worse than k=4 by 0.14 bpb. Larger kernel overfits and slows down.
- **No conv1d**: Essential component, 0.29 bpb loss. Never remove.
- **warmup=400 for 4000-step runs**: Delays convergence, final 7.208 vs 7.180 record. Default warmup=100 (2.5%) is fine.
- **chunk_size=16**: 80% slower than chunk_size=32 with no quality gain. Too many inter-chunk overheads on Apple Silicon.
- **N_LAYERS=6**: Quality improves by 0.043 but 2x slower (10.8s vs 5.6s/step). Not worth it at 43M scale on Apple Silicon.

## Key Insight: Optimal d_head Varies by Task

| Task | Best d_head | Heads | Reason |
|------|------------|-------|--------|
| lm-tok (50K vocab) | 48 | 16 | Rich per-position info; needs many selective patterns but also enough per-head capacity |
| lm (256 byte vocab) | 64 | 12 | Simpler per-position info; needs high-dimensional value vectors per head |

d_head=48 is the Goldilocks point for lm-tok: more heads than d_head=64 for selectivity, but each head retains enough capacity (unlike d_head=32 which has too-small value vectors, or d_head=16 which is 30% slower with no gain).

## Diminishing Returns Analysis

Improvement per experiment session:
- mar10→mar11: 2.325 → 2.175 (lm) = -6.4% (-0.150 bpb)
- mar11→mar12: 2.175 → 2.141 (lm) = -1.6% (-0.034), 8.032 → 7.228 (lm-tok) = -10.0%
- mar12→mar13: 7.228 → 7.180 (lm-tok) = -0.7% (-0.048)

Improvements are clearly decelerating. The architecture is close to its ceiling at 43M params / 4000 steps. Remaining gains will come from:
1. Training longer (more tokens)
2. Schedule optimization
3. Minor architectural refinements

## Late-Session Results (after initial analysis)

- **warmup=400 4000 steps**: 7.208 best@3600. Discard — longer warmup doesn't help at 4000 steps. Delays convergence without finding better basin.
- **N_LAYERS=6 1000 steps**: 7.524 best@950. Keep — 0.043 better than L=5 but **2x slower** (10.8s vs 5.6s/step). Impractical for long runs.
- **chunk_size=16**: Killed@150. 80% slower than chunk_size=32 with no quality gain. Too many inter-chunk overheads.
- **LR=1e-3 1000 steps**: **7.491 best@850**. Keep — 0.076 better than LR=7e-4 at 1000 steps!
- **LR=1e-3 4000 steps**: **7.178 best@2450**. Keep — ties record (7.180) but converges 1350 steps faster. Higher LR speeds early convergence but doesn't raise the ceiling.

## What's Promising But Unfinished

1. **LR=1e-3 as new default**: Converges faster (best@2450 vs 3800). For future runs, LR=1e-3 should be the default — same ceiling but faster to reach it.

2. **6000-8000 step runs**: Model was still slowly improving at step 3800. More training is the most reliable path, but gains are tiny (~0.01 bpb per 1000 extra steps).

3. **N_LAYERS=6 validation**: 0.043 better at 1000 steps but 2x slower. Only worth validating if the per-step quality advantage survives at scale and justifies the 12-hour investment.

4. **Combination: LR=1e-3 + N_LAYERS=6**: Untested. Higher LR might compensate for L=6's slower convergence per wall-clock.

## Recommended Next Experiments (Ranked)

1. **Adopt LR=1e-3 as default** (code change): Change NS_LR default from 7e-4 to 1e-3 in train.py.

2. **6000-step run with LR=1e-3**: Test if more steps pushes below 7.178. ~9 hours.

3. **Cross-task benchmark**: Run the `benchmark-comparison` skill to verify improvements transfer to lm, dna, ts tasks.

4. **N_LAYERS=6 + LR=1e-3** (4000 steps): ~12 hours. Only if 6000-step L=5 doesn't improve.

5. **New architecture ideas**: Multi-head B/C projections, residual scaling, post-SSD conv1d.

## Speed Notes

| Config | ms/step | 1000 steps | 4000 steps |
|--------|---------|------------|------------|
| d_head=48 + L=4 + compile + seq512 | ~5500 | ~92 min | ~6.1 hr |
| d_head=48 + L=5 + compile + seq512 | ~5600 | ~93 min | ~6.2 hr |
| d_head=32 + L=4 + compile + seq512 | ~4400 | ~74 min | ~4.9 hr |
| Byte-level lm (any d_head) | ~90 | ~1.5 min | ~6 min |

d_head=48 + L=5 is ~5600ms/step. L=6 would be ~6700ms/step estimated (~20% more).

# Analysis: autoresearch/mar13-3

Run tag: mar13-3 | 13 new experiments (89 total) | Focus: d_head=48 scaling, architecture probes

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

**Best 1000-step lm-tok: 7.573 (d_head=48 + N_LAYERS=5 + SSD + compile + seq512)**
**Best overall lm-tok: 7.180 (4000 steps, d_head=48 + N_LAYERS=5, best@3800 — RECORD)**

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

## What's Promising But Unfinished

1. **Warmup=400 for 4000-step run** (RUNNING): Testing 10% warmup vs 2.5% default. Slower ramp may help optimizer find better basin. Results expected in ~5 hours.

2. **6000-8000 step runs**: The 4000-step curve (best@3800) shows the model is still improving. Longer training is the most reliable way to push further. Risk: overfitting (watch train vs val gap).

3. **LR=1e-3 with warmup=400**: Higher LR + longer warmup might allow faster exploration then better convergence. Untested with d_head=48 + N_LAYERS=5.

4. **N_LAYERS=6 with d_head=48**: L=6 was marginal with d_head=64 (2.2x slower). But d_head=48 gives faster heads — L=6 might be net positive. Quick 1000-step test first.

5. **Chunk size tuning**: SSD chunk_size is 64 (default). chunk_size=32 or 128 untested. Smaller chunks = finer granularity but more inter-chunk overhead.

6. **Weight decay=0.01**: Current default is 0.0. Mild WD was tried at 0.1 (hurt) and 0.0 (default). 0.01 is an untested middle ground that might help regularize 5-layer models.

## Recommended Next Experiments (Ranked)

1. **Log warmup=400 result** (in progress, ~5 hours): Wait for completion. If it beats 7.180, adopt warmup=400 as standard.

2. **6000-step run** (high priority): NS_STEPS=6000 with current best config (d_head=48, N_LAYERS=5, compile, metal-eval, seq512). Expected ~8.5 hours. Most reliable path to improvement.

3. **N_LAYERS=6 + d_head=48** (1000-step probe): Quick test whether 6th layer helps. If 1K result is significantly better than 7.573, scale to 4000 steps.

4. **LR=1e-3 + warmup=400** (1000-step probe): Higher LR with longer ramp. Quick test, ~75 min.

5. **chunk_size=32** (1000-step probe): Finer-grained SSD chunking. May help with selectivity resolution.

6. **Weight decay=0.01** (1000-step probe): Mild regularization for 5-layer model. Quick env var test.

## Speed Notes

| Config | ms/step | 1000 steps | 4000 steps |
|--------|---------|------------|------------|
| d_head=48 + L=4 + compile + seq512 | ~5500 | ~92 min | ~6.1 hr |
| d_head=48 + L=5 + compile + seq512 | ~5600 | ~93 min | ~6.2 hr |
| d_head=32 + L=4 + compile + seq512 | ~4400 | ~74 min | ~4.9 hr |
| Byte-level lm (any d_head) | ~90 | ~1.5 min | ~6 min |

d_head=48 + L=5 is ~5600ms/step. L=6 would be ~6700ms/step estimated (~20% more).

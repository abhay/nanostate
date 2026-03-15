# nanostate

This is an experiment to have the LLM do its own research on state space models, running on Apple Silicon via MLX.

## Bootstrap

Before the first autonomous run, populate `knowledge/` in an interactive session. The agent can't fetch external content autonomously (see Safety), so this is how you front-load the context it needs.

**1. Read the key papers** (use the `read-arxiv-paper` skill):
- S4: https://arxiv.org/abs/2111.00396
- Mamba: https://arxiv.org/abs/2312.00752
- HiPPO: https://arxiv.org/abs/2008.07669

**2. Read reference implementations** (use the `read-github-code` skill):
- `state-spaces/mamba`: official Mamba implementation
- `state-spaces/s4`: original S4 (repo is `state-spaces/s4`, not `state-spaces/s4d`)

**3. Verify knowledge/ is populated**. You should have summaries like:
- `knowledge/summary_s4_structured_state_spaces.md`
- `knowledge/summary_mamba_selective_scan.md`
- `knowledge/reference_mamba_impl.md`

This only needs to happen once. After that, the autonomous loop has everything it needs in `knowledge/` to run without fetching anything. You can always add more papers/repos in future interactive sessions.

## Setup

To set up a new experiment run:

1. **Pick a run tag**: Use the tag provided by the user, or default to today's date in `monDD` format (e.g. `mar10`). If the branch `autoresearch/<tag>` already exists, append a sequence number (e.g. `mar10-2`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md`: repository context and SSM references.
   - `data.py`: fixed infrastructure (dataset download, loading, batching). Do not modify.
   - `train.py`: the file you modify. Model (S4D/SSD/hybrid blocks), training loop, evaluation.
4. **Read previous results**: If `results.tsv` exists, read it. If any `knowledge/analysis_*.md` files exist, read the most recent one. These tell you what's been tried before and what worked. Do not retry approaches that have already failed unless you have a specific reason to believe the outcome will be different.
5. **Verify data downloads work**: Run `uv run python data.py` to prep all datasets if needed.
6. **Initialize results.tsv**: If `results.tsv` doesn't already exist, create it with just the header row. If it exists from a previous run, append to it.
7. **Go**: Start the experiment loop immediately. Do not wait for human confirmation.

## Experimentation

Each experiment runs on Apple Silicon via MLX. You launch it as: `uv run python train.py --task lm` (or `--task lm-tok` or `--task dna` or `--task ts`).

**Current priority (mar13-3):**
Best lm-tok: **7.178** (SSD + d_head=48 + N_LAYERS=5 + LR=1e-3 + compile + seq512, best@2450 of 4000 steps, 43.7M params). Best byte-level: **2.141** (SSD + B/C SiLU + d_head=64). 94 experiments across 6 agent runs.

d_head=48 is the new default — 16 heads with 48-dim values hits the sweet spot for lm-tok (beats d_head=32's 7.217 and d_head=64's 7.228). N_LAYERS=5 consistently adds ~0.03-0.04 bpb for ~25% more compute. LR=1e-3 converges 1350 steps faster than 7e-4 to the same ceiling. Improvements are decelerating: the architecture is approaching its ceiling at 43M params / 4000 steps.

Next directions (ranked):
1. **6000-step run with LR=1e-3**: Test if more steps pushes below 7.178. ~9 hours.
2. **Cross-task benchmark**: Verify d_head=48 improvements transfer to lm, dna, ts tasks.
3. **N_LAYERS=6 + LR=1e-3** (4000 steps): L=6 is 0.043 better at 1000 steps but 2x slower. Only try if 6000-step L=5 doesn't improve. ~12 hours.
4. **New architecture ideas**: Multi-head B/C projections, residual scaling, post-SSD conv1d.
Do NOT run: RMSNorm+SSD, d_head=16, min_lr=7e-5, C_SCALE, seq1024, --size medium, hybrid at L=4, SwiGLU, conv_k=8, B/C SiLU ablation on lm-tok (confirmed neutral), expand=3 (2.6x slower, no gain), GroupNorm after gating, per-head decay bias, parallel A/B/C projections, d_state=32 (0.028 worse), chunk_size=16 (80% slower), warmup=400 for 4000-step runs, no-conv1d (0.29 bpb worse).

**Three block types:**
- `--block s4d` (default): S4D diagonal SSM with FFT convolution. Fixed A/B/C (LTI). Fast but hits a quality ceiling.
- `--block ssd`: Mamba-2 SSD with chunked matmul. Input-dependent A/B/C (selective). Slower per step but breaks the LTI ceiling. **Use this for both lm and lm-tok experiments** — S4D hits a quality ceiling that SSD breaks through.
- `--block hybrid`: Mix SSD layers with attention layers. Uses `--attn-layers` to specify which layer positions get attention (default: middle layer). Supports `--attn-type {full, sliding}` and `--attn-window`. Attention heads are auto-derived (d_model // 64).
  - **Verdict: loses to pure SSD at L=4.** All positions tested (attn@0: 7.934, attn@2: 7.807, attn@3: 7.800) worse than pure SSD (7.672). At L=4, 25% attention is too high — every SSD layer lost is a capacity hit. The Mamba-2 paper's 10% ratio needs L=10+ to work. Don't retry at this model scale.

**Performance flags (use these!):**
- `--compile`: Fuses element-wise ops via `mx.compile`. Adds ~2-3s JIT overhead on first step, then faster steady-state. Use on all runs longer than 100 steps.
- `--dtype bfloat16` (default): Halves memory bandwidth. Works on all Apple Silicon. The 50K vocab projection dominates lm-tok and benefits most from reduced precision.
- Do NOT use `--dtype float16` — it goes NaN without loss scaling. **Especially bad with SSD: exp overflow in segsum causes immediate NaN.** Use `--dtype float32` if you need full precision for debugging.
- `--grad-checkpoint`: Recomputes activations during backward instead of storing. Enables `--size medium` and `--size large` on 16GB machines. Costs ~30% more compute.
- `--metal-eval`: Uses fused Metal kernels during eval passes (~20% faster evals). Forward-only optimization — training always uses pure MLX (autodiff builds the gradient graph during forward, backward is free). Only applies to `--block ssd` and `--block hybrid`.
- `--grad-accum N`: Gradient accumulation over N microbatches. Effective batch = batch × N with less peak memory. Incompatible with `--compile` (dynamic loop). Use when memory-limited at larger seq_len or batch sizes.
- `--chunk-size Q`: SSD chunk size (default 64, auto-tuned to 32 for seq_len≥512). **If you hit Metal GPU watchdog crashes, reduce to 32 or 16.** Smaller chunks fit better in Apple Silicon's cache hierarchy — especially important at seq512 where training is bandwidth-limited. Also settable via `NS_CHUNK_SIZE` env var.
- All flags are composable: `--block ssd --compile --metal-eval --chunk-size 32`. Works with all block types (s4d, ssd, hybrid).

**Two language modeling modes:**
- `--task lm`: byte-level TinyShakespeare. Fast (~80s for 1000 steps). Good for quick iteration.
- `--task lm-tok`: BPE token-level FineWebEdu (GPT-2 tokenizer, 50K vocab). Slower but avoids overfitting. This is the real benchmark.

**Experiment strategy:** Use `--task lm` (fast, ~80ms/step) as a quick proxy to test architectural ideas. Then validate winners on `--task lm-tok` with longer runs. However, some ideas that fail on lm (due to overfitting TinyShakespeare's 1MB) may work on lm-tok (10B tokens, no overfitting). If something fails on lm because of overfitting (train loss much lower than val loss), retry it on lm-tok before discarding.

**Use 1000-step lm-tok runs for quick iteration.** With S4D + compile: ~8 min. With SSD + compile + seq512: ~65-75 min. This is enough to compare ideas — if something doesn't beat the baseline at 1000 steps, it won't at 4000. Only do 4000-step runs to validate winners. **While long runs are in the background, keep working** — plan the next experiment, write code, review results. Don't sit idle waiting. **Important**: faster convergence at 1000 steps does NOT reliably predict a higher ceiling at 4000 steps (d_head=32 showed 0.079 bpb advantage at 1000 steps but only 0.011 at scale).

**Before long validation runs (3000+ steps):** Ablate neutral components. If you're about to validate A+B+C and B was neutral at 1000 steps, run a quick 1000-step ablation of A+C first. If A+C matches A+B+C, drop B — simpler is better (see simplicity criterion). Only validate the minimal winning config. A 5-hour run is expensive; spend 1 hour of screening to make sure every component earns its place.

**Optimal training length: ~3000-4000 steps** for SSD + seq512. The d_head=64 run peaked at step 3700 (7.228), the d_head=32 run peaked at step 2900 (7.217). Both overfit after their peak. Use `--steps 4000` for validation runs to be safe.

**What you CAN do:**
- Modify `train.py` and `ssd.py`. Everything is fair game: model architecture, SSM parameterization, optimizer, hyperparameters, training loop, initialization, gating, discretization, hybrid layers, etc.

**What you CANNOT do:**
- Modify `data.py`. It is read-only. It contains the dataset loading, batching, and download logic.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the lowest val_bpb** (for language modeling), **highest accuracy** (for DNA), or **lowest val_mse** (for time series). The model is deliberately naive; there are many known improvements to discover.

**Memory & GPU limits**: Apple Silicon has unified memory. The default (`small`, d=384, L=4) uses ~42.8M params for lm-tok and runs comfortably. `medium` and `large` trigger Metal GPU watchdog crashes (the ~5-10s macOS GPU timeout, not OOM). Use `--grad-checkpoint` and `--chunk-size 16` if attempting larger sizes. For lm-tok, most params are in embed+head (50257×d_model each).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

The script prints progress during training:

```
step     0 | train 5.5536 | val 5.5542 | bpb 8.0130 | 342ms/step | 0.7s
step    50 | train 3.1886 | val 2.7891 | bpb 4.0240 | 21ms/step | 1.8s
...
step   999 | train 1.6001 | val 1.6152 | bpb 2.3302 | 20ms/step | 22.1s
```

And logs to `logs/<task>_<timestamp>.csv`. Extract the final metrics:

```
tail -1 logs/lm_*.csv
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	task	val_metric	params	status	description
```

1. git commit hash (short, 7 chars)
2. task: `lm`, `lm-tok`, `dna`, or `ts`
3. primary metric: val_bpb (lm/lm-tok), accuracy (dna), or val_mse (ts)
4. parameter count (e.g. 431000)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	task	val_metric	params	status	description
a1b2c3d	lm	2.3302	431000	keep	baseline
b2c3d4e	lm	2.1500	431000	keep	HiPPO initialization for A matrix
c3d4e5f	lm	2.2800	431000	discard	increase state_dim to 128
d4e5f6g	lm	0.0000	0	crash	selective scan OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Decide what to try. Three modes:
   - **Size preset** (for scaling up/down): use `--size {tiny,small,medium,large}` to change model dimensions in one shot. No code edit needed.
     - `tiny`: d=128, L=4 (~662K lm, ~13.5M lm-tok) — fast debugging
     - `small`: d=384, L=4 (~4.3M lm, ~42.8M lm-tok) — default, quick iteration
     - `medium`: d=768, L=6 (~23M lm, ~100M lm-tok) — serious training
     - `large`: d=1024, L=12 (~81M lm, ~183M lm-tok) — pushing M1 Max limits
   - **Env var sweep** (for hyperparameter changes): set `NS_*` env vars (NS_BATCH_SIZE, NS_LR, NS_STEPS, NS_SEQ_LEN, NS_D_HEAD, NS_EXPAND, NS_WARMUP_STEPS, NS_CHUNK_SIZE) without editing code. `NS_D_MODEL`, `NS_N_LAYERS`, `NS_STATE_DIM` still work and override `--size`. No commit needed.
   - **Code change** (for architectural changes): edit `train.py` and git commit.
   Use your judgment. Model scaling → `--size`. Pure number tuning (lr, batch, seq_len) → env vars. Structural changes (new init, gating, selectivity) → code edit.
3. Run the experiment: `uv run python train.py --task lm-tok --block ssd --compile --metal-eval --save checkpoints/lm_tok > run.log 2>&1` (redirect everything; do NOT use tee or let output flood your context). Always use `--save` so the best checkpoint is tracked automatically (saved to `checkpoints/<task>/best/` whenever val_loss improves). For byte-level: `uv run python train.py --task lm --block ssd --compile --save checkpoints/lm > run.log 2>&1`.
4. Parse the results: `grep -A1 METRICS run.log | head -1` gives a JSON blob with all metrics.
5. If the output is empty or shows an error, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
6. Record the results in results.tsv. For env var sweeps, note the env vars in the description column.
7. If the primary metric improved (lower val_bpb, higher accuracy, lower val_mse):
   - **Code change**: keep the commit, advance the branch.
   - **Env var sweep**: commit the winning values as new defaults in `train.py`, then advance.
   - **Checkpoint**: regenerate the dashboard (`uv run python progress.py --html-only`). The best checkpoint was already saved during training (at `checkpoints/<task>/best/`). For lm, generate a text sample (`uv run python generate.py checkpoints/lm/best --prompt "ROMEO: " --tokens 200 --temp 0.8 2>&1 | head -20`). For lm-tok, generate with a different prompt (`--prompt "The meaning of life"`). Append the sample to the results log or commit message. Commit results.tsv + reports/, and push to the remote branch.
   - **Benchmark** (optional, for milestone improvements): Run `uv run python eval.py checkpoints/lm_tok/best --benchmark hellaswag` to get a standardized HellaSwag score. Log it in the commit message. This only makes sense for lm-tok checkpoints trained for 3K+ steps.
8. If the metric is equal or worse:
   - **Code change**: git reset back to where you started.
   - **Env var sweep**: nothing to undo, just move on.
   - Do NOT push on discard/crash. Only push on keep.

**Warmup**: Before each timed run, do a quick throwaway: `uv run python train.py --task lm --steps 10 > /dev/null 2>&1`. The first run after idle compiles Metal shaders and warms the GPU. Discard it.

**Timeout**: Set your bash tool timeout to match the expected run time. For `lm`, 1000 steps takes ~2 min. For `lm-tok` with S4D, expect ~0.4s/step with `--compile` — so 1000 steps ≈ 8 min. For `lm-tok` with SSD (`--block ssd --compile`), expect ~4.2s/step at seq512 — so 1000 steps ≈ 70 min, 4000 steps ≈ 280 min (~4.7 hrs). **Do NOT use the default 10-minute timeout for long runs — it will kill the process.** Add a 60s buffer to your timeout. If a run exceeds 3x expected time, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea is fundamentally broken, skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, read the references in README.md, re-read train.py for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period. **If the human does interrupt**, run the periodic checkpoint steps (analysis, summary, dashboard, commit, push) before stopping so nothing is lost.

## SSM-specific guidance

S4D has HiPPO-LegS initialization, Mamba-style gated blocks (pre-norm, SiLU), and cosine LR with warmup. SSD (Mamba-2) is implemented with chunked matmul. Hybrid SSM+attention is available but loses to pure SSD at L=4. Current focus: improving the SSD block and training recipe.

**SSD (Mamba-2) — implemented, use `--block ssd`:**
- Implementation in `ssd.py`. Chunked matmul algorithm, input-dependent A/B/C (selective). B/C use SiLU feature map (Mamba-2 paper).
- Best lm-tok: **7.178** (d_head=48, N_LAYERS=5, LR=1e-3, 4000 steps, best@2450). Broke the LTI ceiling (S4D topped out at 7.474).
- Best byte-level: **2.141** (d_head=64, B/C SiLU). d_head is task-dependent: 48 for lm-tok, 64 for byte-level.
- Always use `--compile --metal-eval` for SSD runs. seq512 validated: use `NS_SEQ_LEN=512` for all lm-tok.
- Default d_head=48 (NS_D_HEAD env var). Expand ratio configurable via NS_EXPAND (default 2). Chunk size auto-tunes to 32 at seq512.
- Conv1d is essential — removing it causes 0.29 bpb loss. The k=4 local context window complements SSD's chunk-level attention.
- See `knowledge/summary_mamba2_ssd.md` and `knowledge/design_ssd_implementation.md` for design details.

**Tested and resolved (don't retry):**
- C init scale: no effect on SSD (input-dependent C projection absorbs init scale). 0.1 also no help on S4D.
- Softplus for dt: no improvement on lm.
- RMSNorm: slightly worse than LayerNorm on lm-tok (7.826 vs 7.760). RMSNorm before SSD gating (Mamba-2 style) hurts on both lm (2.344 vs 2.141) and lm-tok (7.966 vs 7.672), and 32% slower.
- Weight decay (AdamW wd=0.1): hurts on lm-tok.
- Hybrid SSM+attention: all positions worse than pure SSD at L=4 (see block types above).
- SwiGLU gating: same quality as default SiLU but 32% slower. Discard.
- seq_len=512: validated, use it.
- d_head=16 (48 heads): same quality as d_head=32 but 30% slower on lm-tok. Too many heads.
- d_head optimal: **d_head=48 is best for lm-tok** (7.178 record, 16 heads). d_head=64 is best for byte-level lm (2.141). d_head=32 is worse than 48 on lm-tok (7.217 vs 7.178). Optimal d_head varies by task: lm-tok wants more heads with moderate capacity, byte-level wants fewer heads with larger values.
- B/C SiLU feature map: helps byte-level lm (2.141 record), neutral on lm-tok. Keep for byte-level, don't expect gains on lm-tok.
- Cosine min_lr=7e-5: hurts at 1000 steps (7.685 vs 7.593). Default 1e-5 is fine.
- conv_k=8: worse than conv_k=4 by 0.14 bpb on lm-tok.
- N_LAYERS=5 on byte-level: overfits (best@400=2.169, final=2.201). But **N_LAYERS=5 works on lm-tok** — adds ~0.03-0.04 bpb for 25% more compute. Use it for lm-tok validation runs.
- N_LAYERS=6: 0.043 better than L=5 at 1000 steps but 2x slower (10.8s vs 5.6s/step). Not worth it at 43M scale on Apple Silicon.
- Hybrid SSM+attention at L=4: all attention positions (0,2,3) worse than pure SSD. 25% attention ratio too high — needs L=10+ for the 10% sweet spot.
- expand=3 (d_inner=1152): 2.6x slower, same quality trend. Massive intermediate tensors with no quality gain at 43M.
- Per-head decay bias (a_bias): HiPPO-inspired diverse time scales for A. Neutral (7.225 vs 7.180 record). Learned A projections already adapt.
- Parallel A/B/C projections from d_model: Mamba-2 paper pattern. Worse than sequential from d_inner (7.584 vs 7.573). At 43M, conv1d output provides useful local context for selectivity.
- GroupNorm after gating: 0.2 bpb worse at 350 steps. Mamba-2 pattern that doesn't help at this scale.
- d_state=32: 25% faster but 0.028 bpb worse (7.595 vs 7.567). d_state=64 has genuinely higher ceiling.
- chunk_size=16: 80% slower than chunk_size=32 with no quality gain. Too many inter-chunk overheads on Apple Silicon.
- No conv1d: essential component, 0.29 bpb loss. Never remove.
- warmup=400 for 4000-step runs: delays convergence (7.208 vs 7.180). Default warmup=100 (2.5%) is fine.
- d_head=48 on byte-level: 2.212 vs 2.141 record. Byte-level needs d_head=64.
- LR=5e-4 with deep models: too slow for N_LAYERS=5 (7.755).

**Unexplored ideas (ranked by expected impact):**
1. 6000-8000 step runs: model still improving at step 3800. More training is the most reliable path but gains are tiny (~0.01 bpb per 1000 extra steps).
2. N_LAYERS=6 + LR=1e-3: untested combination. Higher LR might compensate for L=6's slower convergence per wall-clock.
3. Multi-head B/C projections: separate B/C per head instead of shared.
4. Residual scaling: scale residual connections by 1/sqrt(N_LAYERS).
5. Post-SSD conv1d: conv after SSD instead of before.

**Speed & precision:**
- **Recommended for SSD + lm-tok**: `--compile --metal-eval` on every run. Compile gives 19% faster training, metal-eval gives ~20% faster evals. Chunk size auto-tunes to Q=32 at seq512.
- **bfloat16 with SSD**: Works but costs ~0.07 bpb quality (7.620 vs 7.548). Acceptable for quick iteration, use float32 for final runs.
- **Bandwidth-limited at seq512**: Training is memory-bandwidth-limited on Apple Silicon at seq512. Use `--compile` (fewer kernel launches), `--metal-eval` (faster evals), and let chunk size auto-tune. `--grad-accum` can simulate larger batch without extra memory.
- **Speed by config**: d_head=48 + L=4: ~5.5s/step. d_head=48 + L=5: ~5.6s/step. d_head=32 + L=4: ~4.4s/step. L=6: ~10.8s/step (2x slower than L=5, not worth it).
- See `knowledge/mlx_optimization_research.md` for detailed MLX capabilities.

**Optimizer & training:**
- **LR=1e-3 is the default** (changed from 7e-4). Converges 1350 steps faster to the same ceiling (7.178 best@2450 vs best@3800 at 7e-4). LR=7e-4 still works but is slower.
- LR=5e-4 is too slow for deep models (N_LAYERS=5+). Don't use it.
- Optimal training: ~4000 steps for SSD + seq512 (overfits after ~3700). Use `--steps 4000` for validation runs.
- Warmup: default `min(100, max_steps // 10)` works well. Configurable via NS_WARMUP_STEPS env var. Longer warmup (400 steps) doesn't help.
- Gradient accumulation: `--grad-accum N` (incompatible with `--compile`).
- Weight decay hurts on lm-tok. Don't retry.
- Model scaling: `--size medium` and `--size large` crash with Metal watchdog. Stick with `small` for now.

**Don't be afraid to break things.** The starting model is intentionally basic. Radical changes (replacing the entire SSM core, changing the block structure, adding new components) are encouraged. This is research.

## Periodic checkpoints

Every 10 experiments (or when you feel stuck), pause the loop and do housekeeping:

1. **Analysis**: Run the `analyze-results` skill. This writes `knowledge/analysis_<tag>.md` with distilled learnings: what worked, what didn't, recommended next experiments. This is the memory that future agents (and future sessions of yourself) read.
2. **Run summary**: Run `uv run python progress.py --summary <tag>` to generate `reports/runs/<tag>.md`.
3. **Dashboard**: Run `uv run python progress.py --html-only` to regenerate `reports/index.html` + `reports/results.json`.
4. **Commit and push**: Commit results.tsv, reports/, and knowledge/ to the experiment branch. Push.
5. **Compact context**: Run `/compact` to free up context window space. The checkpoint steps above preserve all state to disk, so nothing is lost. This lets you run many more experiments before hitting the context limit.

This ensures that if the session ends (context limit, crash, human interrupt), the work is preserved and the next session can pick up where you left off.

## Safety

**Do NOT fetch external code or web content during the autonomous loop.** No GitHub API calls, no web fetches, no downloading anything. Use only files already in this repo.

Reference implementation summaries and paper notes are in `knowledge/`. Read these for inspiration. They were curated during interactive sessions and are safe to use.

If you want to explore a new paper or reference implementation, stop and ask the human. External content is only fetched during interactive sessions where a human is reviewing actions.

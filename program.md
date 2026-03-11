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
   - `train.py`: the file you modify. S4D model, training loop, evaluation.
4. **Read previous results**: If `results.tsv` exists, read it. If any `knowledge/analysis_*.md` files exist, read the most recent one. These tell you what's been tried before and what worked. Do not retry approaches that have already failed unless you have a specific reason to believe the outcome will be different.
5. **Verify data downloads work**: Run `uv run python data.py` to prep all datasets if needed.
6. **Initialize results.tsv**: If `results.tsv` doesn't already exist, create it with just the header row. If it exists from a previous run, append to it.
7. **Go**: Start the experiment loop immediately. Do not wait for human confirmation.

## Experimentation

Each experiment runs on Apple Silicon via MLX. You launch it as: `uv run python train.py --task lm` (or `--task lm-tok` or `--task dna` or `--task ts`).

**Two language modeling modes:**
- `--task lm`: byte-level TinyShakespeare. Fast (~80s for 1000 steps). Good for quick iteration.
- `--task lm-tok`: BPE token-level FineWebEdu (GPT-2 tokenizer, 50K vocab). Slower but avoids overfitting. This is the real benchmark.

**Experiment strategy:** Use `--task lm` (fast, ~80ms/step) as a quick proxy to test architectural ideas. Then validate winners on `--task lm-tok` with longer runs. However, some ideas that fail on lm (due to overfitting TinyShakespeare's 1MB) may work on lm-tok (10B tokens, no overfitting). If something fails on lm because of overfitting (train loss much lower than val loss), retry it on lm-tok before discarding.

**Use 1000-step lm-tok runs for quick iteration** (~8 min). This is enough to compare ideas — if something doesn't beat the baseline at 1000 steps, it won't at 3000. Only do 3000+ step runs to validate winners. **While long runs are in the background, keep working** — plan the next experiment, write code, review results. Don't sit idle waiting.

**What you CAN do:**
- Modify `train.py`: the only file you edit. Everything is fair game: model architecture, SSM parameterization, optimizer, hyperparameters, training loop, initialization, gating, discretization, hybrid layers, etc.

**What you CANNOT do:**
- Modify `data.py`. It is read-only. It contains the dataset loading, batching, and download logic.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the lowest val_bpb** (for language modeling), **highest accuracy** (for DNA), or **lowest val_mse** (for time series). The model is deliberately naive; there are many known improvements to discover.

**Memory** is a soft constraint. Apple Silicon has unified memory (16GB), and an OOM crash kills the whole system. The current model (d=384, L=4, N=64) uses ~42.8M params for lm-tok (mostly the 50K vocab embed+head). You still have headroom — d=512, L=8, N=128 should fit. For lm-tok, note that scaling d_model increases the embed/head tables linearly (50257×d_model each).

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
2. Decide what to try. Two modes:
   - **Env var sweep** (for hyperparameter changes): set `NS_*` env vars (NS_D_MODEL, NS_N_LAYERS, NS_STATE_DIM, NS_BATCH_SIZE, NS_LR, NS_STEPS, NS_SEQ_LEN) without editing code. No commit needed.
   - **Code change** (for architectural changes): edit `train.py` and git commit.
   Use your judgment. Pure number tuning (lr, layer count, dimensions) → env vars. Structural changes (new init, gating, selectivity) → code edit.
3. Run the experiment: `uv run python train.py --task lm > run.log 2>&1` (redirect everything; do NOT use tee or let output flood your context). Prepend env vars if sweeping, e.g. `NS_LR=3e-4 NS_D_MODEL=256 uv run python train.py --task lm > run.log 2>&1`.
4. Parse the results: `grep -A1 METRICS run.log | head -1` gives a JSON blob with all metrics.
5. If the output is empty or shows an error, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
6. Record the results in results.tsv. For env var sweeps, note the env vars in the description column.
7. If the primary metric improved (lower val_bpb, higher accuracy, lower val_mse):
   - **Code change**: keep the commit, advance the branch.
   - **Env var sweep**: commit the winning values as new defaults in `train.py`, then advance.
   - **Checkpoint**: regenerate the dashboard (`uv run python progress.py --html-only`). Save a checkpoint with `--save checkpoints/<task>_best` (e.g. `checkpoints/lm_best` or `checkpoints/lm_tok_best`). For lm, generate a text sample (`uv run python generate.py checkpoints/lm_best --prompt "ROMEO: " --tokens 200 --temp 0.8 2>&1 | head -20`). For lm-tok, generate with a different prompt (`--prompt "The meaning of life"`). Append the sample to the results log or commit message. Commit results.tsv + reports/, and push to the remote branch.
   - **Benchmark** (optional, for milestone improvements): Run `uv run python eval.py checkpoints/<task>_best --benchmark hellaswag` to get a standardized HellaSwag score. Log it in the commit message. This only makes sense for lm-tok checkpoints trained for 3K+ steps.
8. If the metric is equal or worse:
   - **Code change**: git reset back to where you started.
   - **Env var sweep**: nothing to undo, just move on.
   - Do NOT push on discard/crash. Only push on keep.

**Warmup**: Before each timed run, do a quick throwaway: `uv run python train.py --task lm --steps 10 > /dev/null 2>&1`. The first run after idle compiles Metal shaders and warms the GPU. Discard it.

**Timeout**: Set your bash tool timeout to match the expected run time. For `lm`, 1000 steps takes ~2 min. For `lm-tok`, expect ~500ms/step — so 1000 steps ≈ 10 min, 3000 steps ≈ 30 min. **Do NOT use the default 10-minute timeout for long runs — it will kill the process.** Add a 60s buffer to your timeout. If a run exceeds 3x expected time, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea is fundamentally broken, skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, read the references in README.md, re-read train.py for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period. **If the human does interrupt**, run the periodic checkpoint steps (analysis, summary, dashboard, commit, push) before stopping so nothing is lost.

## SSM-specific guidance

The model has HiPPO-LegS initialization, Mamba-style gated blocks (pre-norm, SiLU), and cosine LR with warmup. These are all done. Here's what's left to explore, roughly in order of expected impact:

**Selectivity (the Mamba insight) — #1 priority, never tried:**
- Make B, C, dt input-dependent (functions of the input via linear projections, not fixed parameters). This is what makes Mamba work: the model can selectively remember or forget based on content.
- This changes the model from a fixed linear system to a data-dependent one.
- WARNING: selective SSMs cannot use FFT convolution. You must replace the convolutional path with either a sequential scan or a parallel scan. A simple sequential scan (`for t in range(L): ...`) works but is slow. A parallel scan is faster but more complex to implement. Start with the sequential scan to prove the idea works, optimize later.
- **Evidence this matters:** On lm-tok, L=4 and L=6 converge to the exact same val_bpb (~7.47). This is NOT a capacity limit — it's an LTI (Linear Time-Invariant) inductive bias limit. The fixed B, C, dt mean the model applies one convolution kernel to all inputs. Selectivity breaks this ceiling.
- Start with: reduced state_dim=16 (Mamba uses N=16), seq_len=128, to manage speed with Python for loop. Prove quality improves first, optimize later.

**Quick wins to try on lm-tok:**
- **C init scale**: Currently 0.01 × randn — very small. Try 0.1 or 1.0 so the SSM output has more influence early in training.
- **Softplus for dt** instead of exp: Mamba uses softplus(linear(x)). Different gradient profile — clips dt gradients to [0,1], potentially more stable.
- **RMSNorm** instead of LayerNorm: Mamba-2 uses RMSNorm. Simpler (no mean subtraction), slightly faster.
- **Conv1d before SSM on lm-tok**: Failed on lm (overfitting), but lm-tok has 10,000x more data. Worth retrying.
- **seq_len=512 or 1024 on lm-tok**: Hurt on lm (overfitting), but lm-tok doesn't overfit. Longer sequences let the SSM use its long-range memory advantage.

**Architecture:**
- Hybrid: mix SSM layers with 1-2 attention layers. Active research area (Jamba, Zamba).
- Different MLP designs (SwiGLU).

**Discretization:**
- ZOH (zero-order hold) vs bilinear vs Euler discretization.
- The current implementation uses a simple exponential discretization.

**Speed & precision:**
- Mixed precision (float16) training. MLX supports `mx.float16` natively. The big win is on lm-tok where the 50K vocab head projection dominates step time. Use `model.astype(mx.float16)` or cast specific layers. Keep optimizer state in float32 to avoid instability.
- mx.compile for speed — wraps the forward/backward in a compiled graph.
- Profile where time is spent: for lm-tok, ~90% of params are in embed+head (50257×d_model). Optimizing the SSM won't help if the vocab projection is the bottleneck.

**Optimizer & training:**
- Learning rate warmup ratio: current warmup is min(100, steps/10). For longer runs (5K+), the warmup ratio matters — experiment with 1-5% warmup.
- Cosine decay min LR: currently decays to 1e-5. For longer runs, the final LR matters more.
- Gradient accumulation: simulate larger effective batch size without more memory. Accumulate gradients over N steps before updating.
- Weight decay may help on lm-tok (it hurt on lm where the model was overfitting TinyShakespeare, but lm-tok is in an underfitting regime).
- Larger/smaller models, different layer counts.

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

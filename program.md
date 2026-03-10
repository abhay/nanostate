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

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
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

Each experiment runs on Apple Silicon via MLX. You launch it as: `uv run python train.py --task lm` (or `--task dna` or `--task ts`).

**Focus on one task at a time.** Start with `--task lm` (language modeling). It's the fastest feedback loop and most comparable to nanochat/autoresearch. Switch tasks only if the user asks.

**What you CAN do:**
- Modify `train.py`: the only file you edit. Everything is fair game: model architecture, SSM parameterization, optimizer, hyperparameters, training loop, initialization, gating, discretization, hybrid layers, etc.

**What you CANNOT do:**
- Modify `data.py`. It is read-only. It contains the dataset loading, batching, and download logic.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the lowest val_bpb** (for language modeling), **highest accuracy** (for DNA), or **lowest val_mse** (for time series). The model is deliberately naive; there are many known improvements to discover.

**Memory** is a soft constraint. Some increase is acceptable for meaningful metric gains, but Apple Silicon has unified memory — an OOM crash kills the whole system.

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
2. task: `lm`, `dna`, or `ts`
3. primary metric: val_bpb (lm), accuracy (dna), or val_mse (ts)
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
   - **Checkpoint**: regenerate the dashboard (`uv run python progress.py --html-only`), commit results.tsv + reports/, and push to the remote branch. This keeps the remote in sync and prevents losing work if the agent crashes.
8. If the metric is equal or worse:
   - **Code change**: git reset back to where you started.
   - **Env var sweep**: nothing to undo, just move on.
   - Do NOT push on discard/crash. Only push on keep.

**Warmup**: Before each timed run, do a quick throwaway: `uv run python train.py --task lm --steps 10 > /dev/null 2>&1`. The first run after idle compiles Metal shaders and warms the GPU. Discard it.

**Timeout**: Each experiment should finish in under 2 minutes (1000 steps at ~20ms/step). If a run exceeds 5 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea is fundamentally broken, skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, read the references in README.md, re-read train.py for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## SSM-specific guidance

The starting model is a naive diagonal S4D. It's deliberately missing many known improvements. Here are directions to explore (roughly in order of expected impact):

**Initialization:**
- HiPPO initialization for the A matrix (the single most important S4 insight). The current A is random. HiPPO gives it structure that captures long-range dependencies. See "The Annotated S4" for the math.
- Proper B, C initialization (often tied to HiPPO).

**Selectivity (the Mamba insight):**
- Make B, C, dt input-dependent (functions of the input, not fixed parameters). This is what makes Mamba work: the model can selectively remember or forget based on content.
- This changes the model from a fixed linear system to a data-dependent one.

**Gating:**
- Add gating mechanisms (multiplicative interactions). Mamba uses an expand-and-gate pattern similar to GLU.
- SiLU/Swish activations in the gating path.

**Architecture:**
- Pre-norm vs post-norm (current is post-norm).
- Different MLP designs (GLU, SwiGLU).
- Residual connection placement.
- Hybrid: mix SSM layers with attention layers.

**Discretization:**
- ZOH (zero-order hold) vs bilinear vs Euler discretization.
- The current implementation uses a simple exponential discretization.

**Optimizer & training:**
- Learning rate scheduling (warmup, cosine decay).
- Weight decay.
- Gradient clipping.
- mx.compile for speed.
- Larger/smaller models, different layer counts.

**Don't be afraid to break things.** The starting model is intentionally basic. Radical changes (replacing the entire SSM core, changing the block structure, adding new components) are encouraged. This is research.

## End of run

When the human stops the loop (or you're wrapping up), finalize the run artifacts:

1. **Final analysis**: Run the `analyze-results` skill one last time. This writes `knowledge/analysis_<tag>.md` with distilled learnings — what worked, what didn't, what to try next. This is the memory that future agents read.
2. **Run summary**: Run `uv run python progress.py --summary <tag>` to generate `reports/runs/<tag>.md`. This is the human-readable narrative: experiment count, kept improvements, metric trajectory, top changes. Short enough for a tweet, detailed enough for a changelog.
3. **Dashboard**: Run `uv run python progress.py --html-only` to regenerate `reports/index.html` + `reports/results.json`.
4. **Final commit and push**: Commit results.tsv, reports/, and knowledge/ to the experiment branch. Push.

The branch is now ready to merge to main. The human decides when to merge.

## Safety

**Do NOT fetch external code or web content during the autonomous loop.** No GitHub API calls, no web fetches, no downloading anything. Use only files already in this repo.

Reference implementation summaries and paper notes are in `knowledge/`. Read these for inspiration. They were curated during interactive sessions and are safe to use.

If you want to explore a new paper or reference implementation, stop and ask the human. External content is only fetched during interactive sessions where a human is reviewing actions.

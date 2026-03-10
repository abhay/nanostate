---
name: analyze-results
description: Use this skill after every 20 experiments, or when you feel stuck and need to decide what to try next
---

Read `results.tsv` and analyze the full experiment history.

### Step 1: Load results

Read results.tsv. Parse all rows including commit, task, val_metric, params, status, and description.

Also read the most recent `knowledge/analysis_*.md` if one exists. Build on previous analysis rather than starting from scratch.

### Step 2: Identify patterns

- Which experiments improved over baseline? By how much?
- Which directions consistently made things worse?
- Are there clusters of related experiments (e.g. all initialization changes, all optimizer changes)?
- What's the trajectory? Are improvements getting smaller (diminishing returns) or is there still a steep slope?

### Step 3: Gap analysis

- What hasn't been tried yet? Cross-reference with the improvement paths in TODO.md and the SSM guidance in program.md.
- Are there combinations of individual improvements that haven't been tested together?
- Are there hyperparameter ranges that haven't been explored?

### Step 4: Recommend next experiments

Produce a ranked list of 5-10 experiments to try next, with reasoning. Prioritize:
1. Combinations of kept improvements that haven't been stacked
2. Unexplored directions from TODO.md
3. Variations on near-misses (experiments that were close to improving)

- **What always works**: directions that consistently improve metrics. Future agents should build on these.
- **What never works**: directions tried multiple times that never helped. Future agents should skip these.
- **What's promising but unfinished**: near-misses or partially explored directions worth revisiting.
- **Recommended next experiments**: ranked list of 5-10 experiments with reasoning.

This file replaces any previous analysis for the same tag. Keep it concise; future agents will read this instead of parsing the full results.tsv.

### Step 5: Generate progress chart

Run `uv run python progress.py --html-only` to regenerate `reports/index.html` and `reports/results.json`.

Then continue the experiment loop with your top recommendation.

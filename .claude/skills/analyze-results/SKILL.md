---
name: analyze-results
description: Use this skill after every 20 experiments, or when you feel stuck and need to decide what to try next
---

Read `results.tsv` and analyze the full experiment history.

### Step 1: Load results

Read results.tsv. Parse all rows including commit, task, val_metric, params, status, and description.

### Step 2: Identify patterns

- Which experiments improved over baseline? By how much?
- Which directions consistently made things worse?
- Are there clusters of related experiments (e.g. all initialization changes, all optimizer changes)?
- What's the trajectory — are improvements getting smaller (diminishing returns) or is there still a steep slope?

### Step 3: Gap analysis

- What hasn't been tried yet? Cross-reference with the improvement paths in TODO.md and the SSM guidance in program.md.
- Are there combinations of individual improvements that haven't been tested together?
- Are there hyperparameter ranges that haven't been explored?

### Step 4: Recommend next experiments

Produce a ranked list of 5-10 experiments to try next, with reasoning. Prioritize:
1. Combinations of kept improvements that haven't been stacked
2. Unexplored directions from TODO.md
3. Variations on near-misses (experiments that were close to improving)

Write the analysis to `knowledge/analysis_{timestamp}.md` where timestamp is YYYYMMDD_HHMM. Then continue the experiment loop with your top recommendation.

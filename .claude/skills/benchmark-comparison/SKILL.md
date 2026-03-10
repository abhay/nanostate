---
name: benchmark-comparison
description: Use this skill after a significant improvement to test if it transfers across all three tasks (lm, dna, ts)
---

Run all three tasks and produce a comparison table to check if improvements transfer.

### Step 1: Run all tasks

Run each task and capture results:

```bash
uv run python train.py --task lm > run_lm.log 2>&1
uv run python train.py --task dna > run_dna.log 2>&1
uv run python train.py --task ts > run_ts.log 2>&1
```

### Step 2: Parse results

Extract the final metrics from each run using the `---METRICS---` JSON block at the end of each log (or `tail -1` on the CSV logs if no summary block).

### Step 3: Compare to baseline

Produce a markdown table:

| Task | Metric | Baseline | Current | Delta | Transfer? |
|------|--------|----------|---------|-------|-----------|
| LM   | val_bpb | 2.3302  | ???     | ???   | ???       |
| DNA  | accuracy | 0.836  | ???     | ???   | ???       |
| TS   | val_mse | 1.05    | ???     | ???   | ???       |

Mark "Transfer?" as YES if the current model beats baseline on that task, NO if it's worse, NEUTRAL if within noise.

### Step 4: Report

Write the comparison to `knowledge/benchmark_{timestamp}.md`. If the improvement transfers to all three tasks, this is a strong signal that the change is fundamental (not
task-specific overfitting). If it only helps one task, note that — it may still be worth keeping but is less general.

### Step 5: Update baseline

If all three tasks improved, update the baseline numbers in this skill file and in README.md.

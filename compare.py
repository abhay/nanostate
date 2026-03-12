"""Compare block types (s4d, ssd, hybrid) at the same parameter count.

Trains each architecture with identical settings and prints a comparison table.
Useful for measuring the quality impact of selectivity and attention.

Usage:
  python compare.py                        # default: tiny, 200 steps, lm
  python compare.py --size small --steps 1000
  python compare.py --task lm-tok --steps 1000 --size small
  python compare.py --blocks ssd hybrid    # compare subset
"""

import argparse
import json
import re
import subprocess
import sys


def run_training(task, block, size, steps, extra_args=None):
    """Train a model and return the metrics JSON."""
    cmd = [
        sys.executable,
        "train.py",
        "--task",
        task,
        "--block",
        block,
        "--size",
        size,
        "--steps",
        str(steps),
    ]
    # hybrid defaults to middle layer attention, no extra args needed
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    output = result.stdout + result.stderr

    # Extract metrics JSON from ---METRICS--- block
    match = re.search(r"---METRICS---\n(.+)\n---END METRICS---", output)
    if match:
        return json.loads(match.group(1))

    # Training failed
    print(f"  [{block}] FAILED", file=sys.stderr)
    if result.stderr:
        # Print last 5 lines of stderr for diagnosis
        lines = result.stderr.strip().split("\n")
        for line in lines[-5:]:
            print(f"    {line}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare block types at same param count")
    parser.add_argument("--task", default="lm", choices=["lm", "lm-tok", "dna", "ts"])
    parser.add_argument("--size", default="tiny", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--blocks", nargs="+", default=["s4d", "ssd", "hybrid"], choices=["s4d", "ssd", "hybrid"])
    parser.add_argument("--compile", action="store_true", help="Pass --compile to training")
    args = parser.parse_args()

    extra = []
    if args.compile:
        extra.append("--compile")

    print(f"Comparing {', '.join(args.blocks)} | task={args.task} size={args.size} steps={args.steps}")
    print()

    results = {}
    for block in args.blocks:
        print(f"  Training {block}...", end="", flush=True)
        metrics = run_training(args.task, block, args.size, args.steps, extra)
        if metrics:
            results[block] = metrics
            print(f" val_loss={metrics['val_loss']:.4f} ({metrics['step_ms']:.0f}ms/step)")
        else:
            print(" FAILED")

    if len(results) < 2:
        print("\nNeed at least 2 successful runs to compare.")
        sys.exit(1)

    # Print comparison table
    print()
    metric_key = "val_bpb" if args.task in ("lm", "lm-tok") else "val_loss"
    metric_label = "val_bpb" if args.task in ("lm", "lm-tok") else "val_loss"

    header = f"{'block':<10} {'params':>10} {metric_label:>10} {'val_loss':>10} {'ms/step':>8} {'total_s':>8}"
    print(header)
    print("-" * len(header))

    # Sort by metric (lower is better for all current metrics)
    sorted_blocks = sorted(results.items(), key=lambda kv: kv[1].get(metric_key, kv[1]["val_loss"]))
    best_val = sorted_blocks[0][1].get(metric_key, sorted_blocks[0][1]["val_loss"])

    for block, m in sorted_blocks:
        val = m.get(metric_key, m["val_loss"])
        delta = val - best_val
        delta_str = f" (+{delta:.4f})" if delta > 0 else " (best)"
        print(f"{block:<10} {m['params']:>10,} {val:>10.4f} {m['val_loss']:>10.4f} {m['step_ms']:>8.0f} {m['total_seconds']:>8.1f}{delta_str}")


if __name__ == "__main__":
    main()

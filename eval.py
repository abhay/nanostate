"""Evaluate a trained nanostate model.

Supports two modes:
  1. Validation loss on the model's training dataset (default)
  2. Standardized benchmarks (HellaSwag, PIQA) via --benchmark

Prints results as JSON for easy parsing.

Usage:
  python eval.py checkpoints/lm
  python eval.py checkpoints/lm_tok --steps 50 --batch 64
  python eval.py checkpoints/lm_tok --benchmark hellaswag
  python eval.py checkpoints/lm_tok --benchmark piqa
"""

import argparse
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from engine import load_model

# Defaults matching train.py
DNA_TASK = "promoter_no_tata"
ETT_VARIANT = "ETTh1"
PRED_LEN = 96

BENCHMARK_CACHE = os.path.join(os.path.dirname(__file__), "data", "benchmarks")


# ---------------------------------------------------------------------------
# Benchmark data loading
# ---------------------------------------------------------------------------


def _download_file(url, path):
    """Download a file if it doesn't exist."""
    if os.path.exists(path):
        return
    import urllib.request

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url}...", file=sys.stderr)
    urllib.request.urlretrieve(url, path)


def _load_hellaswag():
    """Load HellaSwag validation set. Returns list of (ctx, endings, label)."""
    path = os.path.join(BENCHMARK_CACHE, "hellaswag_val.jsonl")
    _download_file(
        "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
        path,
    )
    examples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            examples.append((obj["ctx"], obj["endings"], obj["label"]))
    return examples


def _load_piqa():
    """Load PIQA validation set. Returns list of (goal, [sol1, sol2], label)."""
    goals_path = os.path.join(BENCHMARK_CACHE, "piqa_valid.lst")
    labels_path = os.path.join(BENCHMARK_CACHE, "piqa_valid_labels.lst")
    _download_file(
        "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl",
        goals_path,
    )
    _download_file(
        "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst",
        labels_path,
    )
    examples = []
    with open(goals_path) as fg, open(labels_path) as fl:
        for goal_line, label_line in zip(fg, fl):
            obj = json.loads(goal_line)
            label = int(label_line.strip())
            examples.append((obj["goal"], [obj["sol1"], obj["sol2"]], label))
    return examples


# ---------------------------------------------------------------------------
# Log-likelihood benchmark evaluation
# ---------------------------------------------------------------------------


def evaluate_benchmark(model, examples, enc):
    """Evaluate multiple-choice examples via log-likelihood ranking.

    Each example is (context, endings, label). For each ending, we compute
    the cross-entropy loss on the completion tokens (masked away from context).
    The ending with lowest average loss is the prediction (acc_norm).

    Returns dict with acc, acc_norm, and num_examples.
    """
    num_correct = 0
    num_correct_norm = 0
    num_total = 0

    for i, (ctx, endings, label) in enumerate(examples):
        ctx_tokens = enc.encode_ordinary(ctx)
        n_choices = len(endings)

        # Tokenize each choice: context + " " + ending
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = enc.encode_ordinary(" " + end)
            row = ctx_tokens + end_tokens
            mask = [0] * len(ctx_tokens) + [1] * len(end_tokens)
            tok_rows.append(row)
            mask_rows.append(mask)

        # Pad to same length
        max_len = max(len(r) for r in tok_rows)
        tokens = np.zeros((n_choices, max_len), dtype=np.int32)
        mask = np.zeros((n_choices, max_len), dtype=np.float32)
        for j, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[j, : len(tok_row)] = tok_row
            mask[j, : len(mask_row)] = mask_row

        # Forward pass: (n_choices, max_len) -> (n_choices, max_len, vocab)
        logits = model(mx.array(tokens))

        # Shift for autoregressive loss: predict token i+1 from position i
        shift_logits = logits[:, :-1, :]
        shift_tokens = mx.array(tokens[:, 1:])
        shift_mask = mx.array(mask[:, 1:])

        # Per-position cross-entropy (unreduced)
        losses = nn.losses.cross_entropy(shift_logits, shift_tokens, reduction="none")
        masked_losses = losses * shift_mask

        # Sum and normalize
        sum_loss = mx.sum(masked_losses, axis=1)
        avg_loss = sum_loss / mx.sum(shift_mask, axis=1)
        mx.eval(sum_loss, avg_loss)

        pred = mx.argmin(sum_loss).item()
        pred_norm = mx.argmin(avg_loss).item()

        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        num_total += 1

        if (i + 1) % 100 == 0:
            acc_n = num_correct_norm / num_total
            print(f"  {i + 1}/{len(examples)} acc_norm={acc_n:.4f}", file=sys.stderr)

    acc = num_correct / num_total
    acc_norm = num_correct_norm / num_total
    return {
        "acc": round(acc, 4),
        "acc_norm": round(acc_norm, 4),
        "num_examples": num_total,
    }


# ---------------------------------------------------------------------------
# Validation loss evaluation
# ---------------------------------------------------------------------------


def evaluate_val_loss(model, config, args):
    """Run validation loss evaluation (original eval.py behavior)."""
    from data import (
        get_batch_lm,
        get_batch_ts,
        load_dna,
        load_ett,
        load_fineweb,
        load_shakespeare,
    )

    task = config["task"]

    if task in ("lm", "lm-tok"):
        val_data = load_fineweb("val") if task == "lm-tok" else load_shakespeare("val")
        losses = []
        for i in range(args.steps):
            xnp, ynp = get_batch_lm(val_data, args.batch, args.seq_len)
            logits = model(mx.array(xnp))
            loss = nn.losses.cross_entropy(logits, mx.array(ynp)).mean()
            mx.eval(loss)
            losses.append(loss.item())
            if (i + 1) % 10 == 0:
                print(f"  eval {i + 1}/{args.steps}...", file=sys.stderr)
        val_loss = float(np.mean(losses))
        val_bpb = val_loss / np.log(2)
        return {"val_loss": round(val_loss, 4), "val_bpb": round(val_bpb, 4)}

    elif task == "dna":
        val_seqs, val_labels, _, _, _ = load_dna("test", DNA_TASK)
        correct, total = 0, 0
        losses = []
        for start in range(0, len(val_seqs), args.batch):
            end = min(start + args.batch, len(val_seqs))
            x = mx.array(val_seqs[start:end])
            y = mx.array(val_labels[start:end])
            logits = model(x)
            loss = nn.losses.cross_entropy(logits, y).mean()
            preds = mx.argmax(logits, axis=-1)
            mx.eval(loss, preds)
            correct += int(mx.sum(preds == y).item())
            total += end - start
            losses.append(loss.item())
        val_loss = float(np.mean(losses))
        return {"val_loss": round(val_loss, 4), "accuracy": round(correct / total, 4)}

    elif task == "ts":
        val_data = load_ett("val", ETT_VARIANT)
        losses, maes = [], []
        for i in range(args.steps):
            xnp, ynp = get_batch_ts(val_data, args.batch, args.seq_len, PRED_LEN)
            x, y = mx.array(xnp), mx.array(ynp)
            pred = model(x)
            mse = mx.mean((pred - y) ** 2)
            mae = mx.mean(mx.abs(pred - y))
            mx.eval(mse, mae)
            losses.append(mse.item())
            maes.append(mae.item())
            if (i + 1) % 10 == 0:
                print(f"  eval {i + 1}/{args.steps}...", file=sys.stderr)
        return {
            "val_loss": round(float(np.mean(losses)), 4),
            "val_mse": round(float(np.mean(losses)), 4),
            "val_mae": round(float(np.mean(maes)), 4),
        }

    else:
        print(f"Unknown task: {task}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained nanostate model")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument(
        "--benchmark",
        choices=["hellaswag", "piqa"],
        help="Run a standardized benchmark instead of val loss",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of eval batches (lm/ts)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    task = config["task"]

    d, n_layers, n = config["d_model"], config["n_layers"], config["state_dim"]
    n_params = sum(x.size for _, x in nn.utils.tree_flatten(model.parameters()))
    print(f"Model: d={d}, L={n_layers}, N={n} | {n_params:,} params | task={task}", file=sys.stderr)

    if args.benchmark:
        if task not in ("lm", "lm-tok"):
            print(f"Benchmarks require an LM model, got '{task}'", file=sys.stderr)
            sys.exit(1)

        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

        if args.benchmark == "hellaswag":
            print("Running HellaSwag (10,042 examples)...", file=sys.stderr)
            examples = _load_hellaswag()
        elif args.benchmark == "piqa":
            print("Running PIQA (1,838 examples)...", file=sys.stderr)
            examples = _load_piqa()

        results = evaluate_benchmark(model, examples, enc)
        results["benchmark"] = args.benchmark
    else:
        results = evaluate_val_loss(model, config, args)

    results["task"] = task
    results["params"] = n_params
    results["checkpoint"] = args.checkpoint
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

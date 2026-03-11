"""Evaluate a trained nanostate model on its validation set.

Loads a checkpoint and runs evaluation batches to compute val_loss and
task-specific metrics. Prints results as JSON for easy parsing.

Usage:
  python eval.py checkpoints/lm
  python eval.py checkpoints/lm_tok --steps 50 --batch 64
  python eval.py checkpoints/lm --seq-len 512
"""

import argparse
import json
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from data import (
    get_batch_lm,
    get_batch_ts,
    load_dna,
    load_ett,
    load_fineweb,
    load_shakespeare,
)
from engine import load_model

# Defaults matching train.py
DNA_TASK = "promoter_no_tata"
ETT_VARIANT = "ETTh1"
PRED_LEN = 96


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained nanostate model")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--steps", type=int, default=50, help="Number of eval batches (lm/ts)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    task = config["task"]

    d, n_layers, n = config["d_model"], config["n_layers"], config["state_dim"]
    n_params = sum(x.size for _, x in nn.utils.tree_flatten(model.parameters()))
    print(f"Model: d={d}, L={n_layers}, N={n} | {n_params:,} params | task={task}", file=sys.stderr)

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
        results = {"val_loss": round(val_loss, 4), "val_bpb": round(val_bpb, 4)}

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
        results = {"val_loss": round(val_loss, 4), "accuracy": round(correct / total, 4)}

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
        results = {
            "val_loss": round(float(np.mean(losses)), 4),
            "val_mse": round(float(np.mean(losses)), 4),
            "val_mae": round(float(np.mean(maes)), 4),
        }

    else:
        print(f"Unknown task: {task}", file=sys.stderr)
        sys.exit(1)

    results["task"] = task
    results["params"] = n_params
    results["checkpoint"] = args.checkpoint
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

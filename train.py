"""nanostate: Naive S4D language/sequence model.

Real diagonal state space, random init, FFT convolution, basic Adam.
This is the file you iterate on.

Usage:
  python train.py --task lm       # byte-level language modeling (TinyShakespeare)
  python train.py --task dna      # DNA sequence classification
  python train.py --task ts       # time series forecasting (ETT)
"""

import argparse
import csv
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from data import (
    get_batch_dna,
    get_batch_lm,
    get_batch_ts,
    load_dna,
    load_ett,
    load_shakespeare,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# model (override via NS_* env vars for sweeps)
D_MODEL = int(os.environ.get("NS_D_MODEL", 384))
N_LAYERS = int(os.environ.get("NS_N_LAYERS", 4))
STATE_DIM = int(os.environ.get("NS_STATE_DIM", 64))
MLP_RATIO = 2

# training
BATCH_SIZE = int(os.environ.get("NS_BATCH_SIZE", 32))
LEARNING_RATE = float(os.environ.get("NS_LR", 5e-4))
MAX_STEPS = int(os.environ.get("NS_STEPS", 1000))
EVAL_INTERVAL = 50
EVAL_STEPS = 10

# task-specific
SEQ_LEN = int(os.environ.get("NS_SEQ_LEN", 256))  # for lm and ts
PRED_LEN = 96  # forecast horizon for ts
DNA_TASK = "promoter_no_tata"
ETT_VARIANT = "ETTh1"

# ---------------------------------------------------------------------------
# S4D Layer (real diagonal, convolutional mode)
# ---------------------------------------------------------------------------


class S4DLayer(nn.Module):
    """Diagonal State Space layer. Real-valued, FFT convolution for training."""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        # A: real diagonal, negative for stability. parameterized as -exp(log_A)
        self.log_A = mx.random.uniform(low=-4.0, high=-1.0, shape=(d_model, state_dim))
        self.B = mx.random.normal((d_model, state_dim)) * 0.01
        self.C = mx.random.normal((d_model, state_dim)) * 0.01
        self.D = mx.ones((d_model,))
        self.log_dt = mx.zeros((d_model,))  # learnable step size

    def kernel(self, L: int):
        """Compute SSM convolution kernel of length L."""
        dt = mx.exp(self.log_dt)  # (H,)
        A = -mx.exp(self.log_A)  # (H, N), negative
        dtA = A * dt[:, None]  # (H, N)

        # Vandermonde: exp(dtA * l) for l = 0..L-1
        arange = mx.arange(L).astype(mx.float32)  # (L,)
        # (H, N, 1) * (1, 1, L) -> (H, N, L)
        V = mx.exp(dtA[:, :, None] * arange[None, None, :])

        # K[h, l] = sum_n C[h,n] * B[h,n] * V[h,n,l] * dt[h]
        CB = self.C * self.B  # (H, N)
        K = mx.sum(CB[:, :, None] * V, axis=1)  # (H, L)
        return K * dt[:, None]

    def __call__(self, u):
        """u: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        _B, L, _H = u.shape
        K = self.kernel(L)  # (H, L)

        # FFT convolution (causal: pad to 2L, take first L)
        u_t = mx.transpose(u, axes=(0, 2, 1))  # (B, H, L)
        fft_size = 2 * L
        K_f = mx.fft.rfft(K, n=fft_size)  # (H, fft_size//2+1)
        u_f = mx.fft.rfft(u_t, n=fft_size)  # (B, H, fft_size//2+1)
        y = mx.fft.irfft(K_f * u_f, n=fft_size)  # (B, H, 2L)
        y = y[..., :L]  # (B, H, L) causal

        # transpose back + skip connection
        y = mx.transpose(y, axes=(0, 2, 1))  # (B, L, H)
        return y + u * self.D


# ---------------------------------------------------------------------------
# Model blocks
# ---------------------------------------------------------------------------


class SSMBlock(nn.Module):
    """Mamba-style gated block: LN -> expand -> SSM + SiLU gate -> project down."""

    def __init__(self, d_model: int, state_dim: int, mlp_ratio: int = 2):
        super().__init__()
        d_inner = d_model * mlp_ratio
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)  # SSM path + gate path
        self.ssm = S4DLayer(d_inner, state_dim)
        self.out_proj = nn.Linear(d_inner, d_model)

    def __call__(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_ssm, z = mx.split(xz, 2, axis=-1)
        x_ssm = self.ssm(x_ssm)
        y = x_ssm * nn.silu(z)  # SiLU gating
        return residual + self.out_proj(y)


# ---------------------------------------------------------------------------
# NanoSSM: task-specific head on shared backbone
# ---------------------------------------------------------------------------


class NanoSSM(nn.Module):
    def __init__(self, task: str, n_classes: int = 2, n_features: int = 7, pred_len: int = 96):
        super().__init__()
        self.task = task

        # embedding
        if task == "lm":
            self.embed = nn.Embedding(256, D_MODEL)
        elif task == "dna":
            self.embed = nn.Embedding(5, D_MODEL)  # A C G T N
        elif task == "ts":
            self.embed = nn.Linear(n_features, D_MODEL)

        # shared backbone
        self.blocks = [SSMBlock(D_MODEL, STATE_DIM, MLP_RATIO) for _ in range(N_LAYERS)]
        self.norm = nn.LayerNorm(D_MODEL)

        # task head
        if task == "lm":
            self.head = nn.Linear(D_MODEL, 256)
        elif task == "dna":
            self.head = nn.Linear(D_MODEL, n_classes)
        elif task == "ts":
            self.head = nn.Linear(D_MODEL, n_features)
            self.pred_len = pred_len

    def __call__(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.task == "lm":
            return self.head(x)  # (B, L, vocab)
        elif self.task == "dna":
            x = mx.mean(x, axis=1)  # (B, H) mean pool
            return self.head(x)  # (B, n_classes)
        elif self.task == "ts":
            x = x[:, -self.pred_len :, :]  # last pred_len positions
            return self.head(x)  # (B, pred_len, n_features)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def loss_lm(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y).mean()


def loss_dna(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y).mean()


def loss_ts(model, x, y):
    pred = model(x)
    return mx.mean((pred - y) ** 2)


LOSS_FN = {"lm": loss_lm, "dna": loss_dna, "ts": loss_ts}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_lm(model, data, batch_size, steps):
    losses = []
    for _ in range(steps):
        xnp, ynp = get_batch_lm(data, batch_size, SEQ_LEN)
        loss = loss_lm(model, mx.array(xnp), mx.array(ynp))
        losses.append(loss.item())
    val_loss = float(np.mean(losses))
    return {"val_loss": val_loss, "val_bpb": val_loss / np.log(2)}


def evaluate_dna(model, seqs, labels, batch_size):
    correct, total = 0, 0
    losses = []
    for start in range(0, len(seqs), batch_size):
        end = min(start + batch_size, len(seqs))
        x = mx.array(seqs[start:end])
        y = mx.array(labels[start:end])
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y).mean()
        preds = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(preds == y).item())
        total += end - start
        losses.append(loss.item())
    val_loss = float(np.mean(losses))
    return {"val_loss": val_loss, "accuracy": correct / total}


def evaluate_ts(model, data, batch_size, steps):
    losses, maes = [], []
    for _ in range(steps):
        xnp, ynp = get_batch_ts(data, batch_size, SEQ_LEN, PRED_LEN)
        x, y = mx.array(xnp), mx.array(ynp)
        pred = model(x)
        mse = mx.mean((pred - y) ** 2).item()
        mae = mx.mean(mx.abs(pred - y)).item()
        losses.append(mse)
        maes.append(mae)
    return {"val_loss": float(np.mean(losses)), "val_mse": float(np.mean(losses)), "val_mae": float(np.mean(maes))}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def count_params(model):
    return sum(x.size for _, x in nn.utils.tree_flatten(model.parameters()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["lm", "dna", "ts"], default="lm")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--save", metavar="DIR", help="Save model checkpoint to DIR after training")
    args = parser.parse_args()

    batch_size = args.batch
    max_steps = args.steps
    lr = args.lr
    task = args.task

    # --- data ---
    if task == "lm":
        train_data = load_shakespeare("train")
        val_data = load_shakespeare("val")
        model = NanoSSM("lm")
    elif task == "dna":
        train_seqs, train_labels, _, n_classes, max_len = load_dna("train", DNA_TASK)
        val_seqs, val_labels, _, _, _ = load_dna("test", DNA_TASK)
        model = NanoSSM("dna", n_classes=n_classes)
    elif task == "ts":
        train_data = load_ett("train", ETT_VARIANT)
        val_data = load_ett("val", ETT_VARIANT)
        n_features = train_data.shape[1]
        model = NanoSSM("ts", n_features=n_features, pred_len=PRED_LEN)

    # materialize parameters
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"NanoSSM ({task}): {N_LAYERS} layers, d={D_MODEL}, state={STATE_DIM}, {n_params:,} params")

    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = LOSS_FN[task]
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # --- logging ---
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"{task}_{timestamp}.csv")

    if task == "lm":
        extra_cols = ["val_bpb"]
    elif task == "dna":
        extra_cols = ["accuracy"]
    elif task == "ts":
        extra_cols = ["val_mse", "val_mae"]

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"] + extra_cols + ["step_ms", "total_s"])

    # --- train ---
    t0 = time.time()
    for step in range(max_steps):
        ts = time.time()

        if task == "lm":
            xnp, ynp = get_batch_lm(train_data, batch_size, SEQ_LEN)
            x, y = mx.array(xnp), mx.array(ynp)
        elif task == "dna":
            xnp, ynp = get_batch_dna(train_seqs, train_labels, batch_size)
            x, y = mx.array(xnp), mx.array(ynp)
        elif task == "ts":
            xnp, ynp = get_batch_ts(train_data, batch_size, SEQ_LEN, PRED_LEN)
            x, y = mx.array(xnp), mx.array(ynp)

        train_loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, train_loss)

        step_ms = (time.time() - ts) * 1000

        if step % EVAL_INTERVAL == 0 or step == max_steps - 1:
            if task == "lm":
                metrics = evaluate_lm(model, val_data, batch_size, EVAL_STEPS)
                extra = [f"{metrics['val_bpb']:.4f}"]
                status = f"bpb {metrics['val_bpb']:.4f}"
            elif task == "dna":
                metrics = evaluate_dna(model, val_seqs, val_labels, batch_size)
                extra = [f"{metrics['accuracy']:.4f}"]
                status = f"acc {metrics['accuracy']:.4f}"
            elif task == "ts":
                metrics = evaluate_ts(model, val_data, batch_size, EVAL_STEPS)
                extra = [f"{metrics['val_mse']:.4f}", f"{metrics['val_mae']:.4f}"]
                status = f"mse {metrics['val_mse']:.4f} mae {metrics['val_mae']:.4f}"

            total_s = time.time() - t0
            tl = train_loss.item()
            vl = metrics["val_loss"]
            print(f"step {step:5d} | train {tl:.4f} | val {vl:.4f} | {status} | {step_ms:.0f}ms/step | {total_s:.1f}s")

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                row = [step, f"{tl:.4f}", f"{vl:.4f}"] + extra + [f"{step_ms:.0f}", f"{total_s:.1f}"]
                writer.writerow(row)

    # --- metrics summary (machine-readable) ---
    total_s = time.time() - t0
    summary = {"task": task, "params": n_params, "steps": max_steps, "d_model": D_MODEL, "n_layers": N_LAYERS, "state_dim": STATE_DIM}
    summary["train_loss"] = round(train_loss.item(), 4)
    summary["val_loss"] = round(metrics["val_loss"], 4)
    summary["step_ms"] = round(step_ms, 1)
    summary["total_seconds"] = round(total_s, 1)
    if task == "lm":
        summary["val_bpb"] = round(metrics["val_bpb"], 4)
    elif task == "dna":
        summary["accuracy"] = round(metrics["accuracy"], 4)
    elif task == "ts":
        summary["val_mse"] = round(metrics["val_mse"], 4)
        summary["val_mae"] = round(metrics["val_mae"], 4)

    # --- checkpoint ---
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        config = {
            "task": task,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "state_dim": STATE_DIM,
            "mlp_ratio": MLP_RATIO,
        }
        model.save_weights(os.path.join(args.save, "model.npz"))
        with open(os.path.join(args.save, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved checkpoint to {args.save}")

    print(f"\nDone. Log: {log_file}")
    print("---METRICS---")
    print(json.dumps(summary))
    print("---END METRICS---")


if __name__ == "__main__":
    main()

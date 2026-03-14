"""nanostate: Naive S4D language/sequence model.

Real diagonal state space, random init, FFT convolution, basic Adam.
This is the file you iterate on.

Usage:
  python train.py --task lm       # byte-level language modeling (TinyShakespeare)
  python train.py --task lm-tok   # BPE token-level language modeling (FineWebEdu)
  python train.py --task dna      # DNA sequence classification
  python train.py --task ts       # time series forecasting (ETT)

Model sizes (byte-level LM params):
  python train.py --size tiny     # d=128, L=4   (~662K params)
  python train.py --size small    # d=384, L=4   (~4.3M params, default)
  python train.py --size medium   # d=768, L=6   (~23M params)
  python train.py --size large    # d=1024, L=12 (~81M params)

Performance flags:
  python train.py --compile                # fuse ops via mx.compile
  python train.py --dtype float32           # full precision (bfloat16 is default)
  python train.py --grad-checkpoint        # trade compute for memory
  python train.py --metal-eval             # fused Metal kernels for eval (~20% faster)
  python train.py --grad-accum 4           # gradient accumulation (eff batch = batch * 4)
  python train.py --chunk-size 32          # SSD chunk size (auto-tuned for seq512)
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
    get_fineweb_vocab_size,
    load_dna,
    load_ett,
    load_fineweb,
    load_shakespeare,
)

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_hardware():
    """Detect Apple Silicon chip and capabilities."""
    info = mx.device_info()
    name = info.get("device_name", "Unknown")
    memory_bytes = info.get("memory_size", 0)
    memory_gb = memory_bytes / (1024**3)

    # Chip generation from device name (M1=1, M2=2, ..., M5=5)
    gen = 1
    for g in range(9, 0, -1):
        if f"M{g}" in name:
            gen = g
            break

    return {
        "name": name,
        "memory_gb": memory_gb,
        "generation": gen,
    }


def auto_config(hw):
    """Suggest training defaults based on detected hardware.

    Returns a dict of suggested values. CLI args and NS_* env vars
    always override these — auto_config just picks smarter defaults
    than the hardcoded ones.
    """
    mem = hw["memory_gb"]

    # Size: fit the biggest model that leaves headroom
    if mem >= 128:
        size = "large"
    elif mem >= 48:
        size = "medium"
    elif mem >= 16:
        size = "small"
    else:
        size = "tiny"

    # Batch: scale with memory, cap at 64
    if mem >= 64:
        batch = 64
    elif mem >= 32:
        batch = 48
    elif mem >= 16:
        batch = 32
    else:
        batch = 16

    # Chunk size: larger memory can handle bigger chunks
    if mem >= 32:
        chunk_size = 128
    elif mem >= 16:
        chunk_size = 64
    else:
        chunk_size = 32

    # Grad checkpoint: suggest for tight memory
    grad_checkpoint = mem < 16

    return {
        "size": size,
        "batch": batch,
        "chunk_size": chunk_size,
        "grad_checkpoint": grad_checkpoint,
    }


def estimate_training_memory_gb(n_params, dtype_bytes=4):
    """Rough estimate of peak training memory.

    params + grads + optimizer state (Adam m,v in fp32) + activations.
    """
    param_bytes = n_params * dtype_bytes
    grad_bytes = param_bytes
    optim_bytes = n_params * 4 * 2  # Adam m, v always float32
    act_bytes = param_bytes * 2  # rough: ~2x params for activations
    return (param_bytes + grad_bytes + optim_bytes + act_bytes) / (1024**3)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Size presets: one knob to scale the model.
# Use --size {tiny,small,medium,large}. NS_* env vars override any preset.
SIZE_PRESETS = {
    "tiny": {"d_model": 128, "n_layers": 4, "state_dim": 64},
    "small": {"d_model": 384, "n_layers": 4, "state_dim": 64},
    "medium": {"d_model": 768, "n_layers": 6, "state_dim": 64},
    "large": {"d_model": 1024, "n_layers": 12, "state_dim": 64},
}

# model defaults (resolved in main() with --size and NS_* env vars)
D_MODEL = 384
N_LAYERS = 4
STATE_DIM = 64
MLP_RATIO = 2
D_HEAD = int(os.environ.get("NS_D_HEAD", 48))
EXPAND = int(os.environ.get("NS_EXPAND", 2))
CHUNK_SIZE = int(os.environ.get("NS_CHUNK_SIZE", 64))

# training
BATCH_SIZE = int(os.environ.get("NS_BATCH_SIZE", 32))
LEARNING_RATE = float(os.environ.get("NS_LR", 7e-4))
MAX_STEPS_DEFAULT = {"lm": 1000, "lm-tok": 3000, "dna": 1000, "ts": 1000}
EVAL_INTERVAL = 50
EVAL_STEPS = 10

# task-specific
SEQ_LEN = int(os.environ.get("NS_SEQ_LEN", 256))  # for lm and ts
PRED_LEN = 96  # forecast horizon for ts
DNA_TASK = "promoter_no_tata"
ETT_VARIANT = "ETTh1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_attn_layers(s, n_layers):
    """Parse comma-separated layer indices for attention placement."""
    indices = set()
    for part in s.split(","):
        idx = int(part.strip())
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"--attn-layers index {idx} out of range [0, {n_layers})")
        indices.add(idx)
    return indices


# ---------------------------------------------------------------------------
# S4D Layer (real diagonal, convolutional mode)
# ---------------------------------------------------------------------------


class S4DLayer(nn.Module):
    """Diagonal State Space layer. Real-valued, FFT convolution for training."""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        # A: HiPPO-LegS diagonal. Eigenvalues -(n + 0.5) for n=0..N-1
        # give a logarithmic spread of decay rates from slow to fast.
        hippo_spacing = mx.arange(1, state_dim + 1, dtype=mx.float32) - 0.5
        self.log_A = mx.broadcast_to(mx.log(hippo_spacing)[None, :], (d_model, state_dim))
        # B: HiPPO-LegS projection weights
        self.B = mx.broadcast_to(
            mx.sqrt(2 * mx.arange(state_dim, dtype=mx.float32) + 1)[None, :],
            (d_model, state_dim),
        )
        self.C = mx.random.normal((d_model, state_dim)) * 0.01
        self.D = mx.ones((d_model,))
        # dt: step size matched to HiPPO eigenvalue range [0.008, 0.03]
        self.log_dt = mx.random.uniform(low=-4.83, high=-3.51, shape=(d_model,))

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
    def __init__(
        self,
        task: str,
        vocab_size: int = 256,
        n_classes: int = 2,
        n_features: int = 7,
        pred_len: int = 96,
        block_type: str = "s4d",
        chunk_size: int = 64,
        attn_layers: list = None,
        attn_type: str = "full",
        attn_window: int = None,
        use_metal: bool = False,
    ):
        super().__init__()
        self.task = task
        self.block_type = block_type
        self._grad_checkpoint = False

        # embedding
        if task in ("lm", "lm-tok"):
            self.embed = nn.Embedding(vocab_size, D_MODEL)
        elif task == "dna":
            self.embed = nn.Embedding(5, D_MODEL)  # A C G T N
        elif task == "ts":
            self.embed = nn.Linear(n_features, D_MODEL)

        # shared backbone
        if block_type == "hybrid":
            from attention import AttentionBlock
            from ssd import SSDBlock

            attn_set = set(attn_layers) if attn_layers else {N_LAYERS // 2}
            window = attn_window if attn_type == "sliding" else None
            self.blocks = []
            for i in range(N_LAYERS):
                if i in attn_set:
                    self.blocks.append(AttentionBlock(D_MODEL, window=window))
                else:
                    self.blocks.append(SSDBlock(D_MODEL, d_state=STATE_DIM, d_head=D_HEAD, expand=EXPAND, chunk_size=chunk_size, use_metal=use_metal))
        elif block_type == "ssd":
            from ssd import SSDBlock

            self.blocks = [SSDBlock(D_MODEL, d_state=STATE_DIM, d_head=D_HEAD, expand=EXPAND, chunk_size=chunk_size, use_metal=use_metal) for _ in range(N_LAYERS)]
        else:
            self.blocks = [SSMBlock(D_MODEL, STATE_DIM, MLP_RATIO) for _ in range(N_LAYERS)]
        self.norm = nn.LayerNorm(D_MODEL)

        # task head
        if task in ("lm", "lm-tok"):
            self.head = nn.Linear(D_MODEL, vocab_size)
        elif task == "dna":
            self.head = nn.Linear(D_MODEL, n_classes)
        elif task == "ts":
            self.head = nn.Linear(D_MODEL, n_features)
            self.pred_len = pred_len

    def __call__(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = mx.checkpoint(block)(x) if self._grad_checkpoint else block(x)
        x = self.norm(x)

        if self.task in ("lm", "lm-tok"):
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


LOSS_FN = {"lm": loss_lm, "lm-tok": loss_lm, "dna": loss_dna, "ts": loss_ts}

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


def set_metal(model, use_metal):
    """Toggle Metal kernels on SSD layers (for eval-only acceleration)."""
    for block in model.blocks:
        if hasattr(block, "ssd"):
            block.ssd.use_metal = use_metal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["lm", "lm-tok", "dna", "ts"], default="lm")
    parser.add_argument("--auto", action="store_true", help="Auto-configure size, batch, chunk-size based on hardware")
    parser.add_argument("--size", choices=list(SIZE_PRESETS), default=None, help="Model size preset (overridden by NS_* env vars)")
    parser.add_argument(
        "--block",
        choices=["s4d", "ssd", "hybrid"],
        default="s4d",
        help="Block type: s4d (FFT conv), ssd (Mamba-2), or hybrid (SSD+attention)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--save", metavar="DIR", help="Save model checkpoint to DIR after training")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="SSD chunk size Q (default 64, smaller = less GPU pressure)")
    parser.add_argument("--compile", action="store_true", help="Fuse ops with mx.compile (faster steps)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16", help="Model precision (default bfloat16)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Gradient checkpointing (less memory, slower)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch * accum)")
    parser.add_argument("--metal-eval", action="store_true", help="Use fused Metal kernels during eval (forward-only, ~20%% faster)")
    parser.add_argument(
        "--attn-layers",
        default=None,
        help="Comma-separated 0-indexed attention layer positions (default: middle)",
    )
    parser.add_argument(
        "--attn-type",
        choices=["full", "sliding"],
        default="full",
        help="Attention type for hybrid (default: full causal)",
    )
    parser.add_argument(
        "--attn-window",
        type=int,
        default=None,
        help="Sliding window size (requires --attn-type sliding)",
    )
    args = parser.parse_args()

    if args.attn_window is not None and args.attn_type != "sliding":
        parser.error("--attn-window requires --attn-type sliding")

    # Auto-configure: detect hardware, suggest defaults (explicit flags override)
    hw = detect_hardware()
    if args.auto:
        ac = auto_config(hw)
        if args.size is None and "NS_D_MODEL" not in os.environ:
            args.size = ac["size"]
        if args.batch == BATCH_SIZE and "NS_BATCH_SIZE" not in os.environ:
            args.batch = ac["batch"]
        if args.chunk_size == CHUNK_SIZE and "NS_CHUNK_SIZE" not in os.environ:
            args.chunk_size = ac["chunk_size"]
        if not args.grad_checkpoint:
            args.grad_checkpoint = ac["grad_checkpoint"]

    # Resolve model dimensions: defaults < --size < NS_* env vars
    global D_MODEL, N_LAYERS, STATE_DIM
    if args.size:
        p = SIZE_PRESETS[args.size]
        D_MODEL, N_LAYERS, STATE_DIM = p["d_model"], p["n_layers"], p["state_dim"]
    if "NS_D_MODEL" in os.environ:
        D_MODEL = int(os.environ["NS_D_MODEL"])
    if "NS_N_LAYERS" in os.environ:
        N_LAYERS = int(os.environ["NS_N_LAYERS"])
    if "NS_STATE_DIM" in os.environ:
        STATE_DIM = int(os.environ["NS_STATE_DIM"])

    batch_size = args.batch
    task = args.task
    block_type = args.block
    max_steps = args.steps or int(os.environ.get("NS_STEPS", MAX_STEPS_DEFAULT.get(task, 1000)))
    lr = args.lr
    use_metal_eval = args.metal_eval and block_type in ("ssd", "hybrid")

    # Chunk size tuning: smaller chunks fit better in Apple Silicon cache at longer sequences.
    # Default Q=64 works well for seq_len<=256; Q=32 is better for seq_len>=512.
    chunk_size = args.chunk_size
    if chunk_size == CHUNK_SIZE and "NS_CHUNK_SIZE" not in os.environ and SEQ_LEN >= 512 and block_type in ("ssd", "hybrid"):
        chunk_size = 32
        print(f"  Auto chunk_size={chunk_size} for seq_len={SEQ_LEN} (override with --chunk-size)")

    attn_layers = None
    if args.attn_layers:
        attn_layers = list(parse_attn_layers(args.attn_layers, N_LAYERS))
    elif block_type == "hybrid":
        attn_layers = [N_LAYERS // 2]

    # --- data ---
    if task == "lm":
        train_data = load_shakespeare("train")
        val_data = load_shakespeare("val")
        model = NanoSSM(
            "lm",
            block_type=block_type,
            chunk_size=chunk_size,
            attn_layers=attn_layers,
            attn_type=args.attn_type,
            attn_window=args.attn_window,
            use_metal=False,
        )
    elif task == "lm-tok":
        train_data = load_fineweb("train")
        val_data = load_fineweb("val")
        vocab_size = get_fineweb_vocab_size()
        model = NanoSSM(
            "lm-tok",
            vocab_size=vocab_size,
            block_type=block_type,
            chunk_size=chunk_size,
            attn_layers=attn_layers,
            attn_type=args.attn_type,
            attn_window=args.attn_window,
            use_metal=False,
        )
    elif task == "dna":
        train_seqs, train_labels, _, n_classes, max_len = load_dna("train", DNA_TASK)
        val_seqs, val_labels, _, _, _ = load_dna("test", DNA_TASK)
        model = NanoSSM(
            "dna",
            n_classes=n_classes,
            block_type=block_type,
            chunk_size=chunk_size,
            attn_layers=attn_layers,
            attn_type=args.attn_type,
            attn_window=args.attn_window,
            use_metal=False,
        )
    elif task == "ts":
        train_data = load_ett("train", ETT_VARIANT)
        val_data = load_ett("val", ETT_VARIANT)
        n_features = train_data.shape[1]
        model = NanoSSM(
            "ts",
            n_features=n_features,
            pred_len=PRED_LEN,
            block_type=block_type,
            chunk_size=chunk_size,
            attn_layers=attn_layers,
            attn_type=args.attn_type,
            attn_window=args.attn_window,
            use_metal=False,
        )

    # materialize parameters
    mx.eval(model.parameters())

    # mixed precision
    if args.dtype != "float32":
        dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
        model.set_dtype(dtype)
        mx.eval(model.parameters())

    # gradient checkpointing
    if args.grad_checkpoint:
        model._grad_checkpoint = True

    n_params = count_params(model)

    # gradient accumulation
    grad_accum = args.grad_accum

    # hardware info + memory warning
    hw = detect_hardware()
    dtype_bytes = 2 if args.dtype != "float32" else 4
    est_gb = estimate_training_memory_gb(n_params, dtype_bytes)
    if est_gb > hw["memory_gb"] * 0.8:
        print(f"WARNING: estimated {est_gb:.1f}GB training memory may exceed {hw['memory_gb']:.0f}GB ({hw['name']})")
        print("  Try: --size smaller, --grad-checkpoint, or reduce --batch")

    flags = []
    if args.dtype != "float32":
        flags.append(args.dtype)
    if args.compile:
        flags.append("compiled")
    if args.grad_checkpoint:
        flags.append("grad-ckpt")
    if use_metal_eval:
        flags.append("metal-eval")
    if grad_accum > 1:
        flags.append(f"grad-accum={grad_accum}")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    print(f"NanoSSM ({task}): {N_LAYERS} layers, d={D_MODEL}, state={STATE_DIM}, {n_params:,} params{flag_str}")
    batch_info = f"batch={batch_size}"
    if grad_accum > 1:
        batch_info += f" (eff={batch_size * grad_accum} via {grad_accum}x accum)"
    print(f"  {hw['name']}, {hw['memory_gb']:.0f}GB | est. {est_gb:.1f}GB training memory | {batch_info}")

    # Cosine decay with linear warmup
    warmup_steps = min(100, max_steps // 10)
    lr_schedule = optim.schedulers.join_schedules(
        [
            optim.schedulers.linear_schedule(1e-7, lr, warmup_steps),
            optim.schedulers.cosine_decay(lr, max_steps - warmup_steps, float(os.environ.get("NS_MIN_LR", 1e-5))),
        ],
        [warmup_steps],
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)
    loss_fn = LOSS_FN[task]
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # training step (optionally compiled)
    state = [model.state, optimizer.state]

    def train_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    if grad_accum <= 1:
        if args.compile:
            train_step = mx.compile(train_step, inputs=state, outputs=state)
    else:
        # Gradient accumulation: average gradients over microbatches
        def microbatch_grad(x, y):
            loss, grads = loss_and_grad(model, x, y)
            return loss, grads

        def accum_train_step(xs, ys):
            total_loss = mx.zeros(())
            acc_grads = None
            for i in range(grad_accum):
                loss, grads = microbatch_grad(xs[i], ys[i])
                total_loss = total_loss + loss
                if acc_grads is None:
                    acc_grads = grads
                else:
                    acc_grads = mx.tree_map(lambda a, b: a + b, acc_grads, grads)
            avg_grads = mx.tree_map(lambda g: g / grad_accum, acc_grads)
            optimizer.update(model, avg_grads)
            return total_loss / grad_accum

        train_step = accum_train_step

    # --- logging ---
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"{task}_{timestamp}.csv")

    if task in ("lm", "lm-tok"):
        extra_cols = ["val_bpb"]
    elif task == "dna":
        extra_cols = ["accuracy"]
    elif task == "ts":
        extra_cols = ["val_mse", "val_mae"]

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"] + extra_cols + ["step_ms", "total_s"])

    # --- checkpoint helper ---
    def save_checkpoint(path):
        os.makedirs(path, exist_ok=True)
        config = {
            "task": task,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "state_dim": STATE_DIM,
            "mlp_ratio": MLP_RATIO,
            "block_type": block_type,
            "chunk_size": chunk_size,
        }
        if task == "lm-tok":
            config["vocab_size"] = vocab_size
        if attn_layers is not None:
            config["attn_layers"] = attn_layers
            config["attn_type"] = args.attn_type
            if args.attn_window is not None:
                config["attn_window"] = args.attn_window
        model.save_weights(os.path.join(path, "model.npz"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # --- train ---
    best_val_loss = float("inf")
    t0 = time.time()
    for step in range(max_steps):
        ts = time.time()

        if grad_accum <= 1:
            if task in ("lm", "lm-tok"):
                xnp, ynp = get_batch_lm(train_data, batch_size, SEQ_LEN)
                x, y = mx.array(xnp), mx.array(ynp)
            elif task == "dna":
                xnp, ynp = get_batch_dna(train_seqs, train_labels, batch_size)
                x, y = mx.array(xnp), mx.array(ynp)
            elif task == "ts":
                xnp, ynp = get_batch_ts(train_data, batch_size, SEQ_LEN, PRED_LEN)
                x, y = mx.array(xnp), mx.array(ynp)
            train_loss = train_step(x, y)
        else:
            micro_bs = batch_size // grad_accum
            xs, ys = [], []
            for _ in range(grad_accum):
                if task in ("lm", "lm-tok"):
                    xnp, ynp = get_batch_lm(train_data, micro_bs, SEQ_LEN)
                elif task == "dna":
                    xnp, ynp = get_batch_dna(train_seqs, train_labels, micro_bs)
                elif task == "ts":
                    xnp, ynp = get_batch_ts(train_data, micro_bs, SEQ_LEN, PRED_LEN)
                xs.append(mx.array(xnp))
                ys.append(mx.array(ynp))
            train_loss = train_step(xs, ys)
        mx.eval(state, train_loss)

        step_ms = (time.time() - ts) * 1000

        if step % EVAL_INTERVAL == 0 or step == max_steps - 1:
            if use_metal_eval:
                set_metal(model, True)
            if task in ("lm", "lm-tok"):
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
            if use_metal_eval:
                set_metal(model, False)

            total_s = time.time() - t0
            tl = train_loss.item()
            vl = metrics["val_loss"]
            print(f"step {step:5d} | train {tl:.4f} | val {vl:.4f} | {status} | {step_ms:.0f}ms/step | {total_s:.1f}s")

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                row = [step, f"{tl:.4f}", f"{vl:.4f}"] + extra + [f"{step_ms:.0f}", f"{total_s:.1f}"]
                writer.writerow(row)

            # Save best checkpoint
            if args.save and vl < best_val_loss:
                best_val_loss = vl
                best_path = os.path.join(args.save, "best")
                save_checkpoint(best_path)
                print(f"  → new best val_loss={vl:.4f}, saved to {best_path}")

    # --- metrics summary (machine-readable) ---
    total_s = time.time() - t0
    summary = {"task": task, "params": n_params, "steps": max_steps, "d_model": D_MODEL, "n_layers": N_LAYERS, "state_dim": STATE_DIM}
    summary["train_loss"] = round(train_loss.item(), 4)
    summary["val_loss"] = round(metrics["val_loss"], 4)
    summary["step_ms"] = round(step_ms, 1)
    summary["total_seconds"] = round(total_s, 1)
    if task in ("lm", "lm-tok"):
        summary["val_bpb"] = round(metrics["val_bpb"], 4)
    elif task == "dna":
        summary["accuracy"] = round(metrics["accuracy"], 4)
    elif task == "ts":
        summary["val_mse"] = round(metrics["val_mse"], 4)
        summary["val_mae"] = round(metrics["val_mae"], 4)

    # --- checkpoint (final) ---
    if args.save:
        save_checkpoint(args.save)
        print(f"Saved checkpoint to {args.save}")
        if best_val_loss < float("inf"):
            print(f"Best checkpoint (val_loss={best_val_loss:.4f}) at {args.save}/best")

    print(f"\nDone. Log: {log_file}")
    print("---METRICS---")
    print(json.dumps(summary))
    print("---END METRICS---")


if __name__ == "__main__":
    main()

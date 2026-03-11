#!/usr/bin/env bash
# Speedrun: reproduce the current best nanostate result.
#
# Byte-level TinyShakespeare, M1 Max, ~80 seconds.
# Target: 2.17 val_bpb (HiPPO-LegS init, gated blocks, cosine LR, lr=7e-4)
#
# Usage:
#   bash runs/speedrun.sh              # train + eval
#   bash runs/speedrun.sh --eval-only  # eval existing checkpoint
set -euo pipefail

CHECKPOINT="checkpoints/speedrun"

if [[ "${1:-}" != "--eval-only" ]]; then
    echo "=== Speedrun: byte-level TinyShakespeare ==="
    echo "Target: ~2.17 val_bpb | d=384, L=4, N=64 | ~4.3M params"
    echo ""

    # Warmup (compiles kernels, doesn't count toward time)
    uv run python train.py --task lm --steps 10 > /dev/null 2>&1

    # Train
    uv run python train.py \
        --task lm \
        --steps 1000 \
        --lr 7e-4 \
        --batch 32 \
        --save "$CHECKPOINT"

    echo ""
fi

# Eval
echo "=== Evaluation ==="
uv run python eval.py "$CHECKPOINT" --steps 100

# Generate sample
echo ""
echo "=== Sample generation ==="
uv run python generate.py "$CHECKPOINT" --prompt "ROMEO: " --tokens 200
echo ""

"""Benchmark recurrent inference speed.

Measures tokens/sec at different context depths to demonstrate the SSM
advantage: constant cost per token regardless of sequence length.

A transformer's KV cache grows linearly, so token 10,000 costs more than
token 100. An SSM's fixed state vector means both cost the same.

Usage:
  python benchmark.py checkpoints/lm
  python benchmark.py checkpoints/lm_tok --tokens 500
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from engine import RecurrentState, load_model


def benchmark_at_position(state, start_token, warmup=10, measure=200):
    """Generate tokens and measure speed.

    Returns tokens/sec averaged over `measure` tokens.
    """
    token = start_token
    # Warmup
    for _ in range(warmup):
        logits = state.step(token)
        token = mx.argmax(logits).item()

    # Measure
    t0 = time.time()
    for _ in range(measure):
        logits = state.step(token)
        token = mx.argmax(logits).item()
    elapsed = time.time() - t0
    return measure / elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark recurrent inference")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens to measure per position")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    task = config["task"]
    d, n_layers, n = config["d_model"], config["n_layers"], config["state_dim"]
    n_params = sum(x.size for _, x in nn.utils.tree_flatten(model.parameters()))
    mlp_ratio = config.get("mlp_ratio", 2)
    d_inner = d * mlp_ratio
    state_bytes = n_layers * d_inner * n * 4

    print(f"Model: d={d}, L={n_layers}, N={n} | {n_params:,} params | task={task}")
    print(f"State: {state_bytes:,} bytes ({state_bytes / 1024:.0f} KB) — fixed, never grows")
    print(f"Measuring {args.tokens} tokens at each context depth (greedy decoding)")
    print()

    state = RecurrentState(model)

    # Determine start token based on task
    if task == "lm-tok":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        start_token = enc.encode_ordinary("\n")[0]
    else:
        start_token = ord("\n")

    # Benchmark at increasing context depths
    # Feed context tokens first, then measure generation speed
    context_depths = [0, 100, 1000, 5000, 10000]
    results = []

    for depth in context_depths:
        state.reset()

        # Build up context
        token = start_token
        for _ in range(depth):
            logits = state.step(token)
            token = mx.argmax(logits).item()

        # Measure generation speed at this depth
        tok_s = benchmark_at_position(state, token, warmup=10, measure=args.tokens)
        results.append((depth, tok_s))
        print(f"  After {depth:>6,} tokens of context: {tok_s:,.0f} tok/s")

    print()

    # Show the constant-cost property
    speeds = [r[1] for r in results]
    min_s, max_s = min(speeds), max(speeds)
    variation = (max_s - min_s) / ((max_s + min_s) / 2) * 100
    print(f"Speed range: {min_s:,.0f} – {max_s:,.0f} tok/s ({variation:.1f}% variation)")
    print(f"Transformer comparison: KV cache at 10K tokens would use ~{10000 * d * n_layers * 2 * 4 / 1024 / 1024:.0f} MB")
    print(f"SSM state is always {state_bytes / 1024:.0f} KB regardless of context length")


if __name__ == "__main__":
    main()

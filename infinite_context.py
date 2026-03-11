"""Infinite context demo: stream a book through the model.

Demonstrates the SSM's constant-memory property by processing an
arbitrarily long document. Memory never grows — the model compresses
everything into its fixed state vector.

After processing the full document, generates a continuation to show
the model "remembers" what it read.

Usage:
  python infinite_context.py checkpoints/lm --file data/shakespeare.txt
  python infinite_context.py checkpoints/lm_tok --file data/shakespeare.txt --generate 200
  python infinite_context.py checkpoints/lm --file data/shakespeare.txt --interval 1000
"""

import argparse
import sys
import time

import mlx.core as mx
import mlx.nn as nn

from engine import RecurrentState, load_model


def get_process_memory_mb():
    """Get current process RSS in MB (macOS/Linux)."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Infinite context demo")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--file", required=True, help="Text file to stream through the model")
    parser.add_argument("--generate", type=int, default=200, help="Tokens to generate after reading")
    parser.add_argument("--interval", type=int, default=5000, help="Print status every N tokens")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature for generation")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    task = config["task"]
    d, n_layers, n = config["d_model"], config["n_layers"], config["state_dim"]
    n_params = sum(x.size for _, x in nn.utils.tree_flatten(model.parameters()))
    mlp_ratio = config.get("mlp_ratio", 2)
    d_inner = d * mlp_ratio
    state_bytes = n_layers * d_inner * n * 4

    # Set up tokenizer for BPE mode
    enc = None
    if task == "lm-tok":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

    # Read the input file
    with open(args.file) as f:
        text = f.read()

    # Tokenize
    if enc:
        tokens = enc.encode_ordinary(text)
    else:
        tokens = list(text.encode("utf-8"))

    print(f"Model: d={d}, L={n_layers}, N={n} | {n_params:,} params", file=sys.stderr)
    print(f"State: {state_bytes:,} bytes ({state_bytes / 1024:.0f} KB) — fixed", file=sys.stderr)
    print(f"File: {args.file} ({len(text):,} chars, {len(tokens):,} tokens)", file=sys.stderr)
    print(f"Mode: {'BPE' if enc else 'byte'}", file=sys.stderr)
    print(file=sys.stderr)

    state = RecurrentState(model)
    mem_start = get_process_memory_mb()

    # Stream the entire document through the model
    print(f"{'Tokens':>10}  {'tok/s':>8}  {'State (KB)':>10}  {'RSS (MB)':>10}", file=sys.stderr)
    print(f"{'------':>10}  {'-----':>8}  {'----------':>10}  {'--------':>10}", file=sys.stderr)

    t0 = time.time()
    for i, token in enumerate(tokens):
        logits = state.step(token)

        if (i + 1) % args.interval == 0:
            elapsed = time.time() - t0
            tok_s = (i + 1) / elapsed
            mem_now = get_process_memory_mb()
            print(
                f"{i + 1:>10,}  {tok_s:>8,.0f}  {state_bytes / 1024:>10,.0f}  {mem_now:>10,.1f}",
                file=sys.stderr,
            )

    elapsed = time.time() - t0
    tok_s = len(tokens) / elapsed
    mem_end = get_process_memory_mb()

    print(file=sys.stderr)
    print(f"Processed {len(tokens):,} tokens in {elapsed:.1f}s ({tok_s:,.0f} tok/s)", file=sys.stderr)
    print(f"State size: {state_bytes / 1024:.0f} KB (constant throughout)", file=sys.stderr)
    print(f"RSS: {mem_start:.1f} MB → {mem_end:.1f} MB", file=sys.stderr)

    # Generate continuation
    if args.generate > 0:
        print(file=sys.stderr)
        print(f"--- Generating {args.generate} tokens from the final state ---", file=sys.stderr)
        print(file=sys.stderr)

        next_token = mx.argmax(logits).item() if args.temp == 0 else _sample(logits, args.temp)

        for _ in range(args.generate):
            if enc:
                sys.stdout.write(enc.decode([next_token]))
            else:
                sys.stdout.buffer.write(bytes([next_token]))
            sys.stdout.flush()

            logits = state.step(next_token)
            next_token = mx.argmax(logits).item() if args.temp == 0 else _sample(logits, args.temp)

        print(file=sys.stderr)


def _sample(logits, temperature=0.8, top_k=40):
    """Sample from logits with temperature and top-k."""
    logits = logits / temperature
    if 0 < top_k < logits.shape[-1]:
        threshold = mx.sort(logits)[-top_k]
        logits = mx.where(logits < threshold, -1e9, logits)
    return mx.random.categorical(logits).item()


if __name__ == "__main__":
    main()

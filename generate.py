"""Generate text from a trained nanostate model.

Runs in recurrent mode: constant memory, constant cost per token.
No KV cache, no sequence buffer. Just a fixed state vector.

Usage:
  python generate.py checkpoints/lm
  python generate.py checkpoints/lm --prompt "ROMEO: "
  python generate.py checkpoints/lm --tokens 500 --temp 0.7
  python generate.py checkpoints/lm --temp 0 --tokens 100  # greedy
  python generate.py checkpoints/lm_tok --prompt "The meaning of life"
"""

import argparse
import sys
import time

import mlx.core as mx

from engine import RecurrentState, load_model


def sample(logits, temperature=0.8, top_k=40):
    """Sample a token from logits with temperature and top-k."""
    if temperature == 0:
        return mx.argmax(logits).item()

    logits = logits / temperature

    if 0 < top_k < logits.shape[-1]:
        threshold = mx.sort(logits)[-top_k]
        logits = mx.where(logits < threshold, -1e9, logits)

    # categorical samples from unnormalized log-probs
    return mx.random.categorical(logits).item()


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained nanostate model")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--prompt", default="", help="Text prompt to condition on")
    parser.add_argument("--tokens", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 = disabled)")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    task = config["task"]
    if task not in ("lm", "lm-tok"):
        print(f"Generation only supported for LM tasks, got '{task}'", file=sys.stderr)
        sys.exit(1)

    # Set up tokenizer for BPE mode
    enc = None
    if task == "lm-tok":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

    state = RecurrentState(model)

    d, n_layers, n = config["d_model"], config["n_layers"], config["state_dim"]
    state_bytes = n_layers * d * n * 4
    mode = "BPE" if enc else "byte"
    print(f"Model: d={d}, L={n_layers}, N={n} | state: {state_bytes:,} bytes | {mode}", file=sys.stderr)
    print(f"Generating {args.tokens} tokens (temp={args.temp}, top_k={args.top_k})", file=sys.stderr)
    print("---", file=sys.stderr)

    # Feed prompt through model to build up state
    if args.prompt:
        sys.stdout.write(args.prompt)
        sys.stdout.flush()
        if enc:
            prompt_tokens = enc.encode_ordinary(args.prompt)
            for tok in prompt_tokens:
                logits = state.step(tok)
        else:
            for b in args.prompt.encode("utf-8"):
                logits = state.step(b)
        next_token = sample(logits, args.temp, args.top_k)
    else:
        # Start with newline
        start_token = enc.encode_ordinary("\n")[0] if enc else ord("\n")
        logits = state.step(start_token)
        next_token = sample(logits, args.temp, args.top_k)

    # Generate
    t0 = time.time()
    for _ in range(args.tokens):
        if enc:
            sys.stdout.write(enc.decode([next_token]))
        else:
            sys.stdout.buffer.write(bytes([next_token]))
        sys.stdout.flush()

        logits = state.step(next_token)
        next_token = sample(logits, args.temp, args.top_k)

    elapsed = time.time() - t0
    tok_per_sec = args.tokens / elapsed if elapsed > 0 else 0
    print(file=sys.stderr)
    print(f"--- {args.tokens} tokens in {elapsed:.1f}s ({tok_per_sec:.0f} tok/s)", file=sys.stderr)


if __name__ == "__main__":
    main()

# nanostate

The [nanochat](https://github.com/karpathy/nanochat) of state space models.

Karpathy stripped GPT training down to something you can read in one sitting. We're doing the same thing for SSMs.

## Why SSMs?

Transformers scale quadratically with sequence length. Every token attends to every other token, so doubling the context 4x's the compute. The KV cache grows with every token you generate, eating memory at inference.

SSMs don't have this problem. They compress the entire history into a fixed-size state vector, so inference cost stays constant regardless of how long the sequence gets. Training can be parallelized as a convolution. You get three views of the same model (continuous, recurrent, convolutional) and can pick whichever one fits: convolutional for fast parallel training, recurrent for cheap autoregressive generation.

The tradeoff is real: a fixed-size state can't do arbitrary lookback the way attention can. But the research keeps closing that gap (Mamba, Mamba-2, hybrid architectures), and for many tasks the linear scaling wins outright.

## The idea

Start with the dumbest possible state space model. Diagonal S4D, random init, vanilla Adam, no gating. ~50 lines for the core. You can hold the whole thing in your head.

Then make it better.

**The naive starting point:**
- Diagonal State Space layers (S4D)
- Random initialization (no HiPPO)
- SSM → LayerNorm → MLP → LayerNorm blocks
- Basic Adam

Right now this runs on an M1 Mac. That's the whole compute budget. Part of the appeal of SSMs: you can actually train something meaningful without a cluster. We'll be looking at other hardware soon.

## Baseline numbers

100 steps on M1, no tuning, random init. These are intentionally bad. That's the point.

| Task | Dataset | Params | Step time | Metric |
|------|---------|--------|-----------|--------|
| Language modeling | TinyShakespeare (byte-level) | 431K | ~22ms | 2.98 BPB |
| DNA classification | Nucleotide Transformer (promoter detection) | 366K | ~28ms | 83.8% accuracy |
| Time series | ETT-h1 (electricity forecasting) | 367K | ~20ms | 1.00 MSE |

## Getting started

```bash
git clone https://github.com/abhay/nanostate.git
cd nanostate
uv sync
uv run python train.py                  # language modeling (default)
uv run python train.py --task dna       # DNA classification
uv run python train.py --task ts        # time series forecasting
```

## References

If you're new to SSMs, these are worth your time (roughly in order of accessibility):

- [A Visual Guide to Mamba and State Space Models](https://maartengrootendorst.com/blog/mamba/) by Maarten Grootendorst. 50+ figures walking you from "what's wrong with Transformers" through SSM fundamentals to Mamba's selective scan. Probably the easiest on-ramp.
- [The Annotated S4](https://srush.github.io/annotated-s4/) by Sasha Rush and Sidd Karamcheti. Line-by-line reimplementation in JAX, in the spirit of "The Annotated Transformer." This is where the math clicks.
- [Hazy Research blog series on S4](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1) by Gu, Goel, Saab, Re. The official companion posts from Stanford. Covers the three representations (continuous, recurrent, convolutional) with good intuition.
- [Introduction to State Space Models](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train) by Loick Bourdois on Hugging Face. Self-contained tutorial covering discretization, HiPPO, and the three views.

The papers:

- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) (S4, Gu et al. 2021). The one that started it all.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu, Dao 2023). Input-dependent parameters and hardware-aware parallel scan. Mamba-3B matches Transformers at 2x the size.
- [Albert Gu's PhD thesis](https://purl.stanford.edu/mb976vf9362) (Stanford 2023). The most complete single-document treatment of the whole SSM line of work, from HiPPO through S4 and beyond.

## Acknowledgements

This project wouldn't exist without [Andrej Karpathy](https://github.com/karpathy)'s work on [nanochat](https://github.com/karpathy/nanochat) and [autoresearch](https://github.com/karpathy/autoresearch). The whole philosophy here (minimal code, obvious architecture, train it yourself) is borrowed directly from that lineage.

We also ported autoresearch's training setup to run natively on Apple Silicon via MLX: [PR](https://github.com/abhay/autoresearch/pull/2) and [branch](https://github.com/abhay/autoresearch/tree/feature/mlx). Got val_bpb down to 1.311 on an M1 Max in 5 minutes. That work is what got us thinking about SSMs on consumer hardware in the first place.

## License

MIT

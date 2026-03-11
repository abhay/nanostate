# Run: mar10

**Task**: lm  
**Experiments**: 20 (7 kept, 13 discarded, 0 crashes)  
**Baseline val_bpb**: 2.3249  
**Best val_bpb**: 2.2093 (+5.0% from baseline)  
**Keep rate**: 35.0%  

## Top improvements

- **d=384 n_layers=4 (wider, same depth)** → 2.2390
- **lr=5e-4 (env var sweep)** → 2.2225
- **pre-norm residual blocks** → 2.3031
- **cosine LR decay with 100-step warmup** → 2.2093
- **d=256 n_layers=6 (env var sweep)** → 2.2945
- **Mamba-style SiLU gated block (replaces SSM+MLP)** → 2.2524

*Generated 2026-03-10 22:55*

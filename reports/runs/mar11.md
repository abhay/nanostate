# Run: mar11

**Task**: lm  
**Experiments**: 36 (11 kept, 25 discarded, 0 crashes)  
**Baseline val_bpb**: 2.3249  
**Best val_bpb**: 2.1745 (+6.5% from baseline)  
**Keep rate**: 30.6%  

## Top improvements

- **3000 steps (env var sweep, 24.6M tokens seen)** → 7.4744
- **d=384 n_layers=4 (wider, same depth)** → 2.2390
- **lr=5e-4 (env var sweep)** → 2.2225
- **baseline mar11 (HiPPO-LegS A+B init from PR#4)** → 2.1838
- **pre-norm residual blocks** → 2.3031
- **cosine LR decay with 100-step warmup** → 2.2093
- **lr=7e-4 (env var sweep, best of 5e-4/7e-4/8e-4/1e-3)** → 2.1745
- **d=256 n_layers=6 (env var sweep)** → 2.2945
- **Mamba-style SiLU gated block (replaces SSM+MLP)** → 2.2524
- **baseline lm-tok (d=384 L=4 N=64 lr=7e-4)** → 8.0317

*Generated 2026-03-11 08:18*

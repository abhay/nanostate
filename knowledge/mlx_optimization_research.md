# MLX Optimization Research

## Key Findings (Mar 2026)

### Parallel Scan Status
- MLX has NO built-in associative_scan primitive (GitHub issue #134, closed without implementation)
- Has `cumsum`, `cumprod`, `cummax` — fixed-op scans only
- alxndrTL/mamba.py tried pure-MLX Blelloch scan: **slower than sequential** due to MLX lacking array views
- Only viable path: custom Metal kernel via `mx.fast.metal_kernel()`

### Mamba-2 SSD Algorithm (Potential Game-Changer)
- Replaces parallel scan with **chunked matmul** (chunk size Q=64)
- Intra-chunk: matmul (parallel, MLX already fast at this)
- Inter-chunk: sequential scan on only T/Q elements (64x shorter sequence)
- 25-line reference implementation, highly portable
- Blog: https://tridao.me/blog/2024/mamba2-part1-model/ (parts 1-4)
- May be faster than a Metal parallel scan kernel since MLX matmul is already well-optimized

### Custom Metal Kernels
- API: `mx.fast.metal_kernel(name, source, ...)` — JIT-compiled Metal
- Wire into autodiff with `@mx.custom_function`
- Reported speedups: 8x forward, 40x backward for fused kernels
- Build kernel object ONCE, reuse (construction = JIT compile = expensive)
- Apple GPU SIMD width: 32 threads (fixed across all Apple Silicon)
- ~208KB register file per core, ~60KB threadgroup shared memory

### Metal-Specific Scan References
- Kieber-Emmons: "Efficient Parallel Prefix Sum in Metal for Apple M1" (Medium)
  - SIMD-group cooperative scan, keeps work in registers
  - Only needs threads_per_threadgroup / 32 shared memory entries
- ShoYamanishi/AppleNumericalComputing (GitHub)
  - 16 algorithms optimized for M1/M2 including prefix-sum
  - Finding: no severe penalty for uncoalesced memory on Apple GPUs (unlike CUDA)
- accelerated-scan (proger, GitHub): fastest GPU associative scan
  - Chunked processing + warp shuffles → maps to Metal SIMD shuffles
  - 11.3ms for seqlen 65K vs 125ms reference
- Metal-Puzzles (abeleinin): port of GPU-Puzzles to Metal via mx.fast.metal_kernel

### Mamba on non-CUDA (AMD/ROCm lessons)
- LightOn: naive CUDA→HIP translation was "an order of magnitude slower"
- Platform-specific optimization of scan and atomic operations was essential
- We cannot simply translate CUDA scan to Metal — need Metal-native optimization

### mx.compile
- Fuses element-wise ops, JIT-compiles to Metal
- Pure functions only (no side effects)
- State must be explicitly declared (model, optimizer, random)
- Won't fuse ops that have dedicated mx.fast implementations
- Easy win: wrap training step

### Mixed Precision
- float16 and bfloat16 both supported on GPU
- `model.set_dtype(mx.float16)` or selective with predicate
- No built-in loss scaler (bfloat16 usually doesn't need one)
- FP16 has better latency at low occupancy on Apple Silicon
- Use Python scalars (not mx.array) for scalar ops to avoid upcasting

### Gradient Checkpointing
- `@mx.checkpoint` decorator
- Recomputes activations during backward instead of storing
- Useful for larger models on 16GB

### Flash Attention
- `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask="causal")`
- Multi-head, GQA, MQA supported
- Softmax always in float32
- Ready for hybrid SSM+attention experiments

### Apple Silicon Hardware Facts
- Memory bandwidth: M1 Max ~400 GB/s, M2 Ultra ~800 GB/s
- NVIDIA A100: 2 TB/s — Apple will always be memory-bandwidth limited
- Compute-bound operations with good data reuse can close the gap
- Unified memory means no CPU↔GPU copies (huge advantage for SSM state)

### Papers to Read
- Mamba-2/SSD: https://arxiv.org/abs/2405.21060
- FlashLinearAttention: https://arxiv.org/abs/2312.06635
- Tiled Flash Linear Attention: https://arxiv.org/abs/2503.14376
- Blelloch scan (GPU Gems 3, Ch. 39): https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- Apple GPU architecture (reverse-engineered): https://dougallj.github.io/applegpu/docs.html

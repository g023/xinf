# g023's TurboXInf — 2x Faster Inference Engine for Qwen3-1.77B

Author: **g023**  - 
License: **MIT** - Created: **April 15, 2026**

(https://huggingface.co/g023) - 
(https://github.com/g023)

A custom inference engine that achieves **2x throughput** over vanilla HuggingFace Transformers for the [g023/Qwen3-1.77B-g023](https://huggingface.co/g023/Qwen3-1.77B-g023) model on an NVIDIA RTX 3060 12GB.

## Results

| Metric | Baseline | TurboXInf | Improvement |
|---|---|---|---|
| **Throughput** | 56.4 tok/s | **113 tok/s** | **2.01x** |
| **Perplexity** | 9.46 | 9.35 | -0.11 (better) |
| **VRAM (model)** | 3.54 GB | 2.40 GB | -32% |
| **VRAM (peak)** | 3.66 GB | 4.80 GB | +31% (compile buffers) |
| TTFT | 22 ms | 33 ms | +50% (compilation overhead) |

## Core Innovation: Custom Triton INT8 GEMV Kernels

The key insight driving TurboXInf: for autoregressive decode at batch=1, every linear layer is a **matrix-vector multiply (GEMV)** that is purely **memory-bandwidth-bound**.

By quantizing weights to INT8 (1 byte/param vs 2 bytes for BF16), we halve the dominant memory traffic. But existing INT8 solutions (torchao, bitsandbytes) failed to deliver speedups on this hardware because their dequantization kernels are not fused with the matmul.

TurboXInf solves this with a **custom Triton GEMV kernel** that:
1. Reads INT8 weights directly from global memory (half the bytes)
2. Dequantizes on-the-fly in registers (zero extra memory traffic)
3. Accumulates in FP32 for numerical stability
4. Applies per-row scaling and outputs BF16

Combined with `torch.compile(fullgraph=True)` to eliminate Python dispatch overhead across 204 linear layers, this achieves near-theoretical bandwidth utilization.

### tldr: Custom Triton INT8 GEMV kernels + torch.compile = 2x throughput for Qwen3-1.77B on RTX 3060, with no quality loss and reduced model VRAM usage. Existing INT8 solutions failed due to unfused dequantization overhead. TurboXInf's fused kernel reads half the data and eliminates extra memory traffic, achieving 55% of the theoretical INT8 bandwidth limit.

### Caveman explanation:
- Each token generation requires reading all model weights (3.3 GB for BF16)
- RTX 3060 has 360 GB/s bandwidth → max ~109 tok/s for BF16
- INT8 halves the weight size → max ~205 tok/s
- Existing INT8 solutions have unfused dequantization → extra memory traffic → slower than BF16
- TurboXInf's custom Triton kernel fuses dequantization → no extra traffic → achieves 113 tok/s = 55% of INT8 theoretical max.

### Why Existing Solutions Failed

| Approach | Result | Problem |
|---|---|---|
| BNB INT8 | 15.3 tok/s (0.27x) | Mixed-precision decomposition overhead |
| BNB INT4 NF4 | 47.3 tok/s (0.84x) | Complex dequant, quality loss (PPL 14.0) |
| torchao INT8 | 20.1 tok/s (0.36x) | Unfused dequant, compiler recompilation limit |
| `torch._weight_int8pack_mm` | 131μs vs 37μs BF16 | Wrong kernel for batch=1 GEMV |
| Speculative decoding | 32 tok/s (0.57x) | Draft model overhead dominates for small models |
| CUDA graphs | Crash | DynamicCache in-place mutations |

### Kernel-Level Benchmarks

```
Layer              Shape          BF16 μs    INT8 μs    Speedup
─────────────────────────────────────────────────────────────────
q_proj/o_proj      [2048, 2048]    35.5       19.8       1.80x
k_proj/v_proj      [1024, 2048]    22.4       13.1       1.71x
gate_proj/up_proj  [6144, 2048]    86.1       44.6       1.93x
down_proj          [2048, 6144]    97.7       49.0       1.99x
lm_head            [151936,2048]  2036.5     1019.1      2.00x
─────────────────────────────────────────────────────────────────
Total per token                  13221       6933        1.91x
```

## Architecture

```
turboxinf/
├── __init__.py          # Package entry
├── config.py            # TurboXInfConfig dataclass
├── model.py             # Model loading + quantization pipeline
├── engine.py            # Core engine (generate, generate_stream)
├── server.py            # FastAPI OpenAI-compatible API server
├── benchmark.py         # Benchmark utilities
├── kernels/
│   ├── __init__.py
│   └── int8_gemv.py     # Custom Triton INT8 GEMV kernel
├── quantize/
│   └── __init__.py
└── plugins/
    └── __init__.py      # Plugin system (PluginBase, PluginManager)
```

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate triton fastapi uvicorn
```

### Generate Text

```bash
python main.py generate "Explain quantum computing" --max-tokens 256
```

### Start API Server

```bash
python main.py serve --port 8000
```

Then use the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "g023/Qwen3-1.77B-g023",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Run Benchmark

```bash
python main.py benchmark
```

### Python API

```python
from turboxinf import TurboXInfConfig, TurboXInfEngine

engine = TurboXInfEngine(TurboXInfConfig())
engine.load()
engine.warmup(runs=10)

# Non-streaming
result = engine.generate("Explain the water cycle")
print(result["content"])
print(f"{result['tokens_per_sec']} tok/s")

# Streaming
for chunk in engine.generate_stream("Write a poem about AI"):
    if not chunk["done"]:
        print(chunk["token"], end="", flush=True)
```

## Configuration

Key config options in `TurboXInfConfig`:

| Option | Default | Description |
|---|---|---|
| `quantize_weights` | `"int8_triton"` | `"none"`, `"int8_triton"`, `"int8_bnb"`, `"int4_bnb"` |
| `use_torch_compile` | `True` | Enable torch.compile |
| `compile_mode` | `"default"` | Compile mode |
| `compile_fullgraph` | `True` | Full-graph compilation |
| `enable_thinking` | `True` | Enable Qwen3 thinking mode |

## Hardware

Benchmarked on:
- **GPU**: NVIDIA RTX 3060 12GB (GA106, 28 SMs, 360 GB/s bandwidth)
- **CPU**: Intel i5-12600K (16 threads)
- **RAM**: 64 GB DDR4
- **CUDA**: 13.0
- **PyTorch**: 2.11.0+cu126
- **Transformers**: 5.5.4
- **Triton**: 3.6.0

## Technical Deep Dive

### Memory Bandwidth Analysis

At BF16, the Qwen3-1.77B model has ~3.3 GB of weights. At 360 GB/s theoretical bandwidth, the theoretical maximum for single-token decode is:

$$\text{max tok/s} = \frac{360 \text{ GB/s}}{3.3 \text{ GB}} \approx 109 \text{ tok/s}$$

My baseline achieves 56.4 tok/s = 52% bandwidth utilization. With INT8 quantization (1.76 GB weights):

$$\text{max tok/s} = \frac{360 \text{ GB/s}}{1.76 \text{ GB}} \approx 205 \text{ tok/s}$$

TurboXInf achieves 113 tok/s = 55% of INT8 theoretical, demonstrating that the Triton kernel maintains equivalent bandwidth utilization while reading half the data.

### Optimization Pipeline

1. **Load BF16 model** from HuggingFace Hub
2. **Per-row INT8 quantization** of all 204 linear layers (including lm_head)
3. **torch.compile(forward, mode="default", fullgraph=True)** to fuse non-linear operations
4. **Warmup** to trigger JIT compilation (first ~10 inferences)
5. **Steady-state inference** at 113 tok/s

### Why fullgraph=True Matters

Without `fullgraph=True`: 109.9 tok/s (1.95x). With `fullgraph=True`: 113.3 tok/s (2.01x).

The +3% comes from eliminating graph breaks that cause the compiler to generate multiple smaller kernels instead of a single optimized execution plan.

## Experiment Log

25+ experiments were conducted across 7 phases:

| Phase | Experiments | Key Finding |
|---|---|---|
| 1. Baseline | SDPA, torch.compile variants | compile(default) = 64 tok/s (1.14x) |
| 2. BNB Quantization | INT8, INT4 NF4 | Both SLOWER — decomposition overhead |
| 3. torchao | INT8, INT4 | INT8 unfused = 21 tok/s, INT4 missing deps |
| 4. Speculative | Qwen3-0.6B draft | 32 tok/s — overhead dominates for small models |
| 5. INT8+Compile | torchao v1 + compile | 20 tok/s — recompilation limit hit |
| 6. **Triton INT8** | **Custom GEMV kernel** | **109 tok/s (1.95x) — breakthrough** |
| 7. **Optimization** | **fullgraph, reduce-overhead** | **113 tok/s (2.01x)** |

## License

MIT

# g023's TurboXInf — High-Performance Inference Engine

Author: **g023**  - 
License: **MIT** - Created: **April 15, 2026**

(https://huggingface.co/g023) - 
(https://github.com/g023)

A custom inference engine achieving **2–2.5x throughput** over vanilla HuggingFace Transformers for [g023/Qwen3-1.77B-g023](https://huggingface.co/g023/Qwen3-1.77B-g023) and [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) on an NVIDIA RTX 3060 12GB.

## Results

Benchmarked on RTX 3060 12GB. All quantized configs use `torch.compile(mode='default')`.

### Qwen3-1.77B (g023/Qwen3-1.77B-g023)

| Configuration | Speedup vs Baseline | VRAM | Notes |
|---|---|---|---|
| BF16 (baseline) | 1.0x | 3.55 GB | Vanilla HF `model.generate()` |
| BF16 + torch.compile | 1.14x | 3.55 GB | Kernel fusion only |
| INT8 + compile | **2.0x** | 2.40 GB | Custom Triton INT8 GEMV |
| **INT4 gs256 + compile** | **2.5x** | **1.53 GB** | Custom Triton INT4 GEMV (fastest) |

Baseline = BF16 without torch.compile (vanilla HuggingFace out-of-the-box).

### Qwen3.5-2B (Qwen/Qwen3.5-2B)

| Configuration | Speedup vs Baseline | VRAM | Notes |
|---|---|---|---|
| BF16 + compile (baseline) | 1.0x | 4.44 GB | fullgraph=False (auto) |
| INT8 + compile | **1.56x** | **3.57 GB** | Best quality/speed |
| INT4 gs256 + compile | **1.88x** | **2.65 GB** | Best throughput |

> Qwen3.5 requires `fullgraph=False` due to data-dependent branching in linear attention, which limits optimization gains compared to Qwen3.

## Core Innovation: Custom Triton GEMV Kernels

### INT8 GEMV Kernel

For autoregressive decode at batch=1, every linear layer is a **matrix-vector multiply (GEMV)** that is purely **memory-bandwidth-bound**. By quantizing to INT8 (1 byte vs 2 bytes BF16), we halve memory traffic. The custom Triton kernel:

1. Reads INT8 weights directly from global memory (half the bytes)
2. Dequantizes on-the-fly in registers (zero extra memory traffic)
3. Accumulates in FP32, applies per-row scaling, outputs BF16

### INT4 GEMV Kernel (NEW)

The INT4 kernel pushes further — 4 bits per weight (0.5 bytes) with group quantization:

1. Packs 2 INT4 values per byte (symmetric quantization: range [-8, 7])
2. Uses group-wise scales (one FP16 scale per `group_size` elements)
3. Iterates one group per step with 1D scale loads for efficiency
4. **Optimal group_size=256** found via sweep (32/64/128/256/512)

**INT4 gs256 achieves 2.5x over BF16** — faster than INT8 — while using only **43% of BF16 VRAM**.

### Why Existing Solutions Failed

| Approach | Result | Problem |
|---|---|---|
| BNB INT8 | 0.27x | Mixed-precision decomposition overhead |
| BNB INT4 NF4 | 0.84x | Complex dequant, quality loss |
| torchao INT8 | 0.36x | Unfused dequant, recompilation limit |
| Speculative decoding | 0.57x | Draft model overhead dominates for small models |

### Profiler Analysis (INT8 on Qwen3-1.77B)

- **83.7%** of CUDA time in INT8 GEMV kernel
- **81%** bandwidth utilization (81% of theoretical 360 GB/s)
- **4.2%** in SDPA attention (already optimized by cutlass)
- Remaining: fused normalization, SiLU, RoPE (torch.compile handles these)

## Architecture

```
turboxinf/
├── __init__.py          # Package entry
├── config.py            # TurboXInfConfig dataclass (all options)
├── model.py             # Multi-model loading, quantization pipeline
├── engine.py            # Core engine (generate, generate_stream)
├── turbo_decode.py      # Custom decode loop with StaticCache + CUDA graphs
├── server.py            # FastAPI OpenAI-compatible API server
├── benchmark.py         # Benchmark utilities
├── kernels/
│   ├── __init__.py
│   ├── int8_gemv.py     # Triton INT8 GEMV kernel + LinearINT8
│   └── int4_gemv.py     # Triton INT4 GEMV kernel + LinearINT4 + mixed quant
├── quantize/
│   └── __init__.py
└── plugins/
    └── __init__.py      # Plugin system (PluginBase, PluginManager)
```

### Multi-Model Support

- **Qwen3** (`Qwen3ForCausalLM`): Standard transformer, 29 layers, fullgraph=True
- **Qwen3.5** (`Qwen3_5ForConditionalGeneration`): Hybrid linear+full attention, 24 layers, vision encoder, fullgraph=False (auto-detected)
- Architecture auto-detected from model config
- Vision encoder automatically excluded from quantization

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate triton fastapi uvicorn

# Install torchao (must use the PyTorch CUDA index to get the correct ABI wheel)
pip install torchao --index-url https://download.pytorch.org/whl/cu126

# Optional: for Qwen3.5-2B fast linear attention
pip install flash-linear-attention causal-conv1d

# Optional: set HF_TOKEN for authenticated downloads (avoids rate limits)
export HF_TOKEN="hf_..."
```

> **Note:** `torchao` must be installed from the PyTorch CUDA index (`--index-url https://download.pytorch.org/whl/cu126`). The default PyPI wheel ships `.so` files built for the wrong Python ABI. `causal-conv1d` requires `nvcc`; if unavailable, Qwen3.5 falls back to a torch implementation automatically with no speed penalty when using `torch.compile`.

### Generate Text

```bash
# Qwen3-1.77B (default, INT4 gs256 = fastest)
python main.py generate "Explain quantum computing" --quantize int4_triton --max-tokens 256

# Qwen3.5-2B (INT8 = fastest for this model)
python main.py generate "What is AI?" --model Qwen/Qwen3.5-2B --quantize int8_triton

# With INT8 (good balance of speed and quality)
python main.py generate "Write a haiku" --quantize int8_triton
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
python main.py benchmark --quantize int4_triton --name int4_gs256
python main.py benchmark --quantize int8_triton --name int8
```

### Python API

```python
from turboxinf import TurboXInfConfig, TurboXInfEngine

config = TurboXInfConfig()
config.quantize_weights = "int4_triton"  # Fastest for Qwen3
config.int4_group_size = 256

engine = TurboXInfEngine(config)
engine.load()
engine.warmup(runs=5)

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
| `model_path` | `"g023/Qwen3-1.77B-g023"` | HuggingFace model ID |
| `quantize_weights` | `"int8_triton"` | `"none"`, `"int8_triton"`, `"int4_triton"`, `"mixed_int4_int8"` |
| `int4_group_size` | `256` | Group size for INT4 quantization (32/64/128/256/512) |
| `use_torch_compile` | `True` | Enable torch.compile |
| `compile_mode` | `"default"` | `"default"`, `"max-autotune"` |
| `compile_fullgraph` | `True` | Full-graph (auto False for Qwen3.5) |
| `enable_thinking` | `True` | Enable thinking/reasoning mode |
| `mixed_int8_patterns` | `["down_proj"]` | Layers to keep as INT8 in mixed mode |

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

At BF16, the Qwen3-1.77B model has ~3.3 GB of weights. At 360 GB/s theoretical bandwidth:

$$\text{BF16 max} = \frac{360}{3.3} \approx 109 \text{ tok/s}$$

$$\text{INT8 max} = \frac{360}{1.76} \approx 205 \text{ tok/s}$$

$$\text{INT4 max} = \frac{360}{0.92} \approx 391 \text{ tok/s}$$

The achieved 2.5x speedup for INT4 over BF16 demonstrates that while INT4 has higher unpacking overhead than INT8, the 4x reduction in weight data dominates at larger group sizes where scale factor overhead is minimal.

### INT4 Group Size Trade-off

| Group Size | Speed | Extra Data | Quality |
|---|---|---|---|
| 32 | ~1.0x | +3.1% scales | Best |
| 64 | ~1.2x | +1.6% scales | Very good |
| 128 | ~1.8x | +0.8% scales | Good |
| **256** | **~2.5x** | **+0.4% scales** | Good |
| 512 | ~2.1x | +0.2% scales | Acceptable |

Group size 256 is optimal: just 0.4% extra scale data while enabling efficient 1-group-per-iteration kernel execution.

### Optimization Pipeline

1. **Load BF16 model** from HuggingFace Hub (HF/transformers warnings auto-suppressed)
2. **Architecture detection** (Qwen3 vs Qwen3.5, auto fullgraph settings)
3. **Quantization** (INT8 or INT4 with group scales, vision encoder skipped)
4. **pad_token_id** set from tokenizer to prevent per-call warnings
5. **torch.compile** with appropriate fullgraph setting (recompile limit auto-tuned for Qwen3.5)
6. **Warmup** to trigger JIT compilation (~5–10 inferences, dynamo/inductor warnings suppressed)
7. **Steady-state inference** at 2–2.5x baseline with clean output

### Qwen3.5-2B Specifics

Qwen3.5-2B is a hybrid linear-attention + full-attention multimodal model:
- 18 linear attention layers + 6 full attention layers
- 248K vocabulary (vs 151K for Qwen3)
- Vision encoder (excluded from quantization)
- Requires `fullgraph=False` due to data-dependent branching in linear attention mask
- `flash-linear-attention` + `causal-conv1d` recommended for optimal speed

## Experiment Log

25+ experiments were conducted across multiple phases:

| Phase | Experiments | Key Finding |
|---|---|---|
| 1. Baseline | SDPA, torch.compile variants | compile(default) = 1.15x |
| 2. BNB Quantization | INT8, INT4 NF4 | Both SLOWER — decomposition overhead |
| 3. torchao | INT8, INT4 | Unfused dequant = 0.36x |
| 4. Speculative | Qwen3-0.6B draft | 0.57x — overhead dominates |
| 5. **Triton INT8** | **Custom GEMV kernel** | **2.0x — breakthrough** |
| 6. **Triton INT4** | **Custom INT4 GEMV + group quant** | **2.5x with gs256** |
| 7. **Multi-model** | **Qwen3.5-2B support** | **2.67x (INT8)** |
| 8. Group sweep | INT4 gs 32/64/128/256/512 | gs256 optimal |
| 9. Mixed quant | INT4 + INT8 for down_proj | 2.3x, good VRAM balance |

## License

MIT

## OUTPUT - Qwen 3.5-2B

```plaintext
python main.py generate "What is AI?" --model Qwen/Qwen3.5-2B --quantize int4_triton
[TurboXInf] Detected architecture: qwen3_5
Loading weights: 100%|████████████| 617/617 [00:00<00:00, 1058.02it/s]
  Replaced 187 linear layers with INT4 (group_size=256, skipped 98)
[TurboXInf] Model loaded in 10.905s
  [WARN] Qwen3.5 linear attention uses data-dependent branching; setting fullgraph=False automatically
[TurboXInf] Warming up...
[TurboXInf] Warmup done in 51.82s
Okay, the user asked "What is AI?" Let me figure out how to respond. First, I need to define what Artificial Intelligence (AI) means clearly but simply. The term comes from Latin roots: artificial for humans, intelligence referring to cognitive functions. So combining them gives a system that mimics human intelligence.

I should consider examples of common uses like image recognition, speech processing, and decision-making. Maybe mention that it's not fully autonomous yet, since some people think machines are already super smart. Also, note that different forms exist: rule-based, machine learning, deep learning, etc. But keep it simple for now.

Wait, maybe also address that AI has ethical considerations. Some users might be concerned about safety or privacy. But the question is just asking what AI is, so focus on definition first, then maybe add context about limitations or future possibilities.

Check if the answer covers key aspects: definition, examples, applications, potential benefits, and maybe caveats about current capabilities vs. future goals. That would make the answer comprehensive without overcomplicating. Make sure to avoid jargon where possible, use clear language.

Also, check for any sensitive topics. Ethical issues are important but the main query is about the concept. Keep

[129.56 tok/s, 256 tokens]
```

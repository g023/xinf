"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — Global Configuration"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurboXInfConfig:
    """Master configuration for the TurboXInf inference engine."""

    # ── Model ────────────────────────────────────────────────────────────────
    model_path: str = "g023/Qwen3-1.77B-g023"
    dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    device: str = "cuda"

    # ── Quantization ─────────────────────────────────────────────────────────
    quantize_weights: str = "int8_triton"  # "none", "int8_triton", "int8_bnb", "int4_bnb"
    quantize_kv_cache: bool = False  # FP8 KV cache
    sensitive_layers_fp16: bool = True  # Keep first/last layers at higher precision

    # ── Generation ───────────────────────────────────────────────────────────
    max_new_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    enable_thinking: bool = True

    # ── Optimization ─────────────────────────────────────────────────────────
    use_torch_compile: bool = True
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    compile_fullgraph: bool = True  # Use fullgraph=True for torch.compile
    use_cuda_graphs: bool = False  # CUDA graphs conflict with dynamic cache
    use_flash_attention: bool = True
    use_static_cache: bool = False

    # ── Speculative Decoding ─────────────────────────────────────────────────
    use_speculation: bool = False  # Slower for small models on consumer GPUs
    speculation_num_draft_tokens: int = 5
    speculation_exit_layer: int = 14  # Use first N layers as draft model

    # ── Adaptive Head Pruning ────────────────────────────────────────────────
    use_head_pruning: bool = False  # Experimental
    head_pruning_threshold: float = 0.1  # Prune heads below this importance

    # ── KV Cache ─────────────────────────────────────────────────────────────
    max_cache_length: int = 4096  # Static cache size for CUDA graphs

    # ── Server ───────────────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_max_concurrent: int = 4

    # ── Plugin System ────────────────────────────────────────────────────────
    plugins_dir: str = "plugins"
    enabled_plugins: list = field(default_factory=list)

    # ── Paths ────────────────────────────────────────────────────────────────
    cache_dir: Optional[str] = None
    benchmark_dir: str = "benchmarks"

    def get_torch_dtype(self):
        import torch
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]

"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — Global Configuration"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


# ── Supported Models ─────────────────────────────────────────────────────────
SUPPORTED_MODELS = {
    "g023/Qwen3-1.77B-g023": "qwen3",
    "Qwen/Qwen3.5-2B": "qwen3_5",
}

# ── Quantization Modes ───────────────────────────────────────────────────────
QUANT_MODES = ["none", "int8_triton", "int4_triton", "mixed_int4_int8", "int8_bnb", "int4_bnb"]


@dataclass
class TurboXInfConfig:
    """Master configuration for the TurboXInf inference engine."""

    # ── Model ────────────────────────────────────────────────────────────────
    model_path: str = "g023/Qwen3-1.77B-g023"
    dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    device: str = "cuda"
    model_arch: str = "auto"  # "auto", "qwen3", "qwen3_5" — auto-detect from config

    # ── Quantization ─────────────────────────────────────────────────────────
    quantize_weights: str = "int8_triton"  # "none", "int8_triton", "int4_triton", "mixed_int4_int8", "int8_bnb", "int4_bnb"
    quantize_kv_cache: bool = False  # FP8 KV cache
    sensitive_layers_fp16: bool = True  # Keep first/last layers at higher precision
    int4_group_size: int = 256  # Group size for INT4 quantization (32, 64, 128, 256)
    mixed_int8_patterns: list = field(default_factory=lambda: ["down_proj"])  # Layers to keep as INT8 in mixed mode

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

    # ── Vision (Qwen3.5 specific) ────────────────────────────────────────────
    skip_vision_quantize: bool = True  # Don't quantize vision encoder

    def get_torch_dtype(self):
        import torch
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]

    def detect_model_arch(self) -> str:
        """Detect model architecture from model_path."""
        if self.model_arch != "auto":
            return self.model_arch
        for model_id, arch in SUPPORTED_MODELS.items():
            if model_id in self.model_path:
                return arch
        # Fallback: try loading config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_path, cache_dir=self.cache_dir)
            if hasattr(config, 'model_type'):
                if 'qwen3_5' in config.model_type:
                    return 'qwen3_5'
                elif 'qwen3' in config.model_type:
                    return 'qwen3'
        except Exception:
            pass
        return "qwen3"  # default fallback

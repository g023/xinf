"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — Optimized Model Loading & Management

Supports:
  - Qwen3ForCausalLM (g023/Qwen3-1.77B-g023)
  - Qwen3_5ForConditionalGeneration (Qwen/Qwen3.5-2B)
  - INT8 Triton GEMV quantization
  - INT4 Triton GEMV quantization (group quantization)
  - Mixed INT4/INT8 quantization
  - BitsAndBytes INT8/INT4 (legacy)
"""

import gc
import time
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import TurboXInfConfig


def _get_skip_patterns(config: TurboXInfConfig, model_arch: str) -> list:
    """Get patterns of module names to skip during quantization."""
    skip = []
    if model_arch == "qwen3_5" and config.skip_vision_quantize:
        skip.extend(["visual", "vision", "merger"])
    return skip


def _get_model_class(model_arch: str):
    """Return the appropriate model class for the architecture."""
    if model_arch == "qwen3_5":
        try:
            from transformers import Qwen3_5ForConditionalGeneration
            return Qwen3_5ForConditionalGeneration
        except ImportError:
            pass
    # Default: AutoModelForCausalLM (handles Qwen3 and most models)
    return AutoModelForCausalLM


def load_model_and_tokenizer(
    config: TurboXInfConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """
    Load model with progressive optimizations.

    Supports Qwen3 and Qwen3.5 architectures with automatic detection.
    Returns: (model, tokenizer, load_info)
    """
    load_info = {"steps": [], "warnings": []}
    t_total = time.time()

    torch.set_float32_matmul_precision("high")

    # ── Detect architecture ──────────────────────────────────────────────
    model_arch = config.detect_model_arch()
    load_info["model_arch"] = model_arch
    print(f"[TurboXInf] Detected architecture: {model_arch}")

    # ── Load tokenizer ───────────────────────────────────────────────────
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        cache_dir=config.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_info["steps"].append(("tokenizer", round(time.time() - t0, 3)))

    # ── Load model ───────────────────────────────────────────────────────
    t0 = time.time()
    model_kwargs = {
        "device_map": config.device,
        "dtype": config.get_torch_dtype(),
        "cache_dir": config.cache_dir,
    }

    # BNB INT8 quantization (legacy, slower)
    if config.quantize_weights == "int8_bnb":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model_kwargs.pop("dtype", None)
        load_info["steps"].append(("quantize_config", "int8_bitsandbytes"))

    elif config.quantize_weights == "int4_bnb":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.get_torch_dtype(),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.pop("dtype", None)
        load_info["steps"].append(("quantize_config", "int4_nf4"))

    # Load with appropriate class
    ModelClass = _get_model_class(model_arch)
    model = ModelClass.from_pretrained(config.model_path, **model_kwargs)
    model.eval()
    load_info["steps"].append(("model_load", round(time.time() - t0, 3)))

    # ── Triton Quantization ──────────────────────────────────────────────
    skip_patterns = _get_skip_patterns(config, model_arch)

    if config.quantize_weights == "int8_triton":
        t0 = time.time()
        from .kernels.int8_gemv import replace_linear_with_int8
        replace_linear_with_int8(model, skip_lm_head=False, skip_patterns=skip_patterns)
        load_info["steps"].append(("int8_triton_quantize", round(time.time() - t0, 3)))

    elif config.quantize_weights == "int4_triton":
        t0 = time.time()
        from .kernels.int4_gemv import replace_linear_with_int4
        replace_linear_with_int4(
            model,
            group_size=config.int4_group_size,
            skip_lm_head=False,
            skip_patterns=skip_patterns,
        )
        load_info["steps"].append(("int4_triton_quantize", {
            "group_size": config.int4_group_size,
            "time": round(time.time() - t0, 3),
        }))

    elif config.quantize_weights == "mixed_int4_int8":
        t0 = time.time()
        from .kernels.int4_gemv import replace_linear_mixed_int4_int8
        replace_linear_mixed_int4_int8(
            model,
            group_size=config.int4_group_size,
            skip_lm_head=False,
            skip_patterns=skip_patterns,
            int8_patterns=config.mixed_int8_patterns,
        )
        load_info["steps"].append(("mixed_int4_int8_quantize", {
            "group_size": config.int4_group_size,
            "int8_patterns": config.mixed_int8_patterns,
            "time": round(time.time() - t0, 3),
        }))

    if torch.cuda.is_available():
        load_info["vram_after_quant"] = round(torch.cuda.memory_allocated() / 1e9, 3)

    # ── torch.compile ────────────────────────────────────────────────────
    if config.use_torch_compile:
        t0 = time.time()
        # Qwen3.5 has data-dependent branching in linear attention — fullgraph breaks
        fullgraph = config.compile_fullgraph
        if model_arch == "qwen3_5" and fullgraph:
            fullgraph = False
            load_info["warnings"].append(
                "Qwen3.5 linear attention uses data-dependent branching; "
                "setting fullgraph=False automatically"
            )
        try:
            model.forward = torch.compile(
                model.forward,
                mode=config.compile_mode,
                fullgraph=fullgraph,
            )
            load_info["steps"].append(("torch_compile", {
                "mode": config.compile_mode,
                "fullgraph": fullgraph,
                "time": round(time.time() - t0, 3),
            }))
        except Exception as e:
            load_info["warnings"].append(f"torch.compile failed: {e}")

    # ── VRAM info ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        load_info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        load_info["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)

    load_info["total_load_time"] = round(time.time() - t_total, 3)
    return model, tokenizer, load_info

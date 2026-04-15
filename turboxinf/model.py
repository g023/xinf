"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — Optimized Model Loading & Management"""

import gc
import time
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import TurboXInfConfig


def load_model_and_tokenizer(
    config: TurboXInfConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """
    Load model with progressive optimizations.

    Returns: (model, tokenizer, load_info)
    """
    load_info = {"steps": [], "warnings": []}
    t_total = time.time()

    torch.set_float32_matmul_precision("high")

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
        del model_kwargs["dtype"]
        load_info["steps"].append(("quantize_config", "int8_bitsandbytes"))

    elif config.quantize_weights == "int4_bnb":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.get_torch_dtype(),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        del model_kwargs["dtype"]
        load_info["steps"].append(("quantize_config", "int4_nf4"))

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    model.eval()
    load_info["steps"].append(("model_load", round(time.time() - t0, 3)))

    # ── Triton INT8 Quantization (custom GEMV kernel) ────────────────────
    if config.quantize_weights == "int8_triton":
        t0 = time.time()
        from .kernels.int8_gemv import replace_linear_with_int8
        replace_linear_with_int8(model, skip_lm_head=False)
        load_info["steps"].append(("int8_triton_quantize", round(time.time() - t0, 3)))
        if torch.cuda.is_available():
            load_info["vram_after_int8"] = round(torch.cuda.memory_allocated() / 1e9, 3)

    # ── torch.compile ────────────────────────────────────────────────────
    if config.use_torch_compile:
        t0 = time.time()
        try:
            model.forward = torch.compile(
                model.forward,
                mode=config.compile_mode,
                fullgraph=config.compile_fullgraph,
            )
            load_info["steps"].append(("torch_compile", {
                "mode": config.compile_mode,
                "fullgraph": config.compile_fullgraph,
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

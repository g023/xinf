"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — Benchmark Utilities"""

import json
import os
import time
from typing import Dict, List, Optional

import torch

from .config import TurboXInfConfig
from .engine import TurboXInfEngine


# ── Standard benchmark prompts ───────────────────────────────────────────────
BENCHMARK_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to find the longest common subsequence of two strings.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "You are completing the next step in a task to create an arcade game in javascript.",
]

PERPLEXITY_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning, there was nothing. Then, there was light. "
    "The universe expanded rapidly in what scientists call the Big Bang. "
    "Over billions of years, matter coalesced into stars, planets, and galaxies. "
    "On one small planet orbiting an ordinary star, life emerged from simple chemistry."
)


def run_benchmark(
    engine: TurboXInfEngine,
    name: str = "unnamed",
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 256,
    warmup_runs: int = 3,
    benchmark_runs: int = 3,
    do_sample: bool = False,
    save_dir: str = "benchmarks",
) -> Dict:
    """Run a complete benchmark suite on the engine."""
    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    os.makedirs(save_dir, exist_ok=True)
    results = {
        "name": name,
        "config": {
            "model_path": engine.config.model_path,
            "dtype": engine.config.dtype,
            "quantize_weights": engine.config.quantize_weights,
            "use_torch_compile": engine.config.use_torch_compile,
            "compile_mode": engine.config.compile_mode,
            "use_cuda_graphs": engine.config.use_cuda_graphs,
            "use_speculation": engine.config.use_speculation,
        },
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
    }

    # ── VRAM baseline ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        results["vram_pre_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)

    # ── Perplexity test ──────────────────────────────────────────────────
    ppl = measure_perplexity(engine)
    results["perplexity"] = ppl
    print(f"  [{name}] Perplexity: {ppl}")

    # ── Coherency test ───────────────────────────────────────────────────
    coherency = measure_coherency(engine)
    results["coherency"] = coherency
    print(f"  [{name}] Coherent: {coherency['coherent']}")

    # ── Warmup ───────────────────────────────────────────────────────────
    print(f"  [{name}] Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        engine.generate("Hello", max_new_tokens=32, do_sample=False)

    # ── Benchmark runs ───────────────────────────────────────────────────
    print(f"  [{name}] Benchmarking ({benchmark_runs} runs × {len(prompts)} prompts)...")
    run_results = []
    for run_idx in range(benchmark_runs):
        for prompt in prompts:
            r = engine.generate(
                prompt, max_new_tokens=max_new_tokens, do_sample=do_sample
            )
            run_results.append({
                "prompt": prompt[:80],
                "output_tokens": r["usage"]["output_tokens"],
                "tokens_per_sec": r["tokens_per_sec"],
                "time_taken": r["time_taken"],
                "ms_per_token": round(1000 / r["tokens_per_sec"], 2) if r["tokens_per_sec"] > 0 else 0,
            })
            print(
                f"    Run {run_idx+1}: {r['usage']['output_tokens']} tok "
                f"@ {r['tokens_per_sec']} tok/s"
            )

    # ── Aggregate stats ──────────────────────────────────────────────────
    all_tps = [r["tokens_per_sec"] for r in run_results]
    all_mspt = [r["ms_per_token"] for r in run_results]
    results["decode"] = {
        "avg_tokens_per_sec": round(sum(all_tps) / len(all_tps), 2),
        "min_tokens_per_sec": round(min(all_tps), 2),
        "max_tokens_per_sec": round(max(all_tps), 2),
        "avg_ms_per_token": round(sum(all_mspt) / len(all_mspt), 2),
        "runs": run_results,
    }

    if torch.cuda.is_available():
        results["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # ── TTFT (Time to First Token) ───────────────────────────────────────
    ttft = measure_ttft(engine, max_tokens=1)
    results["ttft_ms"] = ttft
    print(f"  [{name}] TTFT: {ttft:.1f}ms")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = os.path.join(save_dir, f"{name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n  ┌─ {name} Results ─────────────────")
    print(f"  │ Decode:     {results['decode']['avg_tokens_per_sec']} tok/s avg")
    print(f"  │ Range:      {results['decode']['min_tokens_per_sec']}-{results['decode']['max_tokens_per_sec']} tok/s")
    print(f"  │ TTFT:       {ttft:.1f} ms")
    print(f"  │ Perplexity: {ppl}")
    print(f"  │ Coherent:   {coherency['coherent']}")
    print(f"  │ VRAM Peak:  {results.get('vram_peak_gb', 'N/A')} GB")
    print(f"  └───────────────────────────────────")

    return results


@torch.inference_mode()
def measure_perplexity(engine: TurboXInfEngine) -> float:
    """Compute perplexity on standard test text."""
    inputs = engine.tokenizer(PERPLEXITY_TEXT, return_tensors="pt")
    device = next(engine.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Need to call the underlying model, not generate
    model = engine.model
    # Handle compiled model
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    outputs = model(**inputs, labels=inputs["input_ids"])
    return round(torch.exp(outputs.loss).item(), 4)


def measure_coherency(engine: TurboXInfEngine) -> Dict:
    """Generate response and check basic coherency."""
    result = engine.generate(
        "Explain what water is in exactly three sentences.",
        max_new_tokens=256,
        do_sample=False,
    )
    response = result["content"]
    checks = {
        "non_empty": len(response.strip()) > 0,
        "min_length": len(response) > 20,
        "no_excessive_repetition": (
            len(set(response.split())) > len(response.split()) * 0.3
            if response.split() else False
        ),
        "contains_sentences": "." in response,
    }
    return {
        "response": response[:500],
        "checks": checks,
        "coherent": all(checks.values()),
    }


def measure_ttft(engine: TurboXInfEngine, max_tokens: int = 1) -> float:
    """Measure time to first token in milliseconds."""
    prompt = "Hello"
    inputs = engine._build_input(prompt)
    gen_kwargs = engine._get_generation_kwargs(max_new_tokens=max_tokens, do_sample=False)

    torch.cuda.synchronize()
    t0 = time.time()
    _ = engine.model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    ttft = (time.time() - t0) * 1000
    return round(ttft, 2)

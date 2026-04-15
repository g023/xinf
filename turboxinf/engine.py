"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — Core Inference Engine"""

import gc
import os
import time
import threading
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from .config import TurboXInfConfig
from .model import load_model_and_tokenizer
from .plugins import PluginManager, PluginHook


class TurboXInfEngine:
    """
    Main inference engine with progressive optimization pipeline.

    Features:
    - Streaming and non-streaming generation
    - torch.compile acceleration
    - INT8/INT4 quantization
    - CUDA graph decode (via static cache)
    - Plugin system for extensibility
    - Self-speculative decoding (layer skip)
    - Adaptive head pruning
    """

    def __init__(self, config: Optional[TurboXInfConfig] = None):
        self.config = config or TurboXInfConfig()
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.load_info: Dict = {}
        self._compiled = False
        self._warmed_up = False

        # Plugin system
        self.plugin_manager = PluginManager(self.config.plugins_dir)

        # Stats tracking
        self._stats = {
            "total_tokens_generated": 0,
            "total_time_s": 0.0,
            "total_requests": 0,
        }

    def load(self):
        """Load model, tokenizer, and plugins."""
        self.plugin_manager.run_hook(PluginHook.PRE_LOAD, engine=self)

        self.model, self.tokenizer, self.load_info = load_model_and_tokenizer(
            self.config
        )

        # Load plugins
        self.plugin_manager.load_all(engine=self)
        self.plugin_manager.run_hook(PluginHook.POST_LOAD, engine=self)

        print(f"[TurboXInf] Model loaded in {self.load_info['total_load_time']}s")
        if self.load_info.get("warnings"):
            for w in self.load_info["warnings"]:
                print(f"  [WARN] {w}")

        return self

    def warmup(self, prompt: str = "Hello", max_tokens: int = 64, runs: int = 3):
        """Warmup the engine to trigger compilation and CUDA graph capture.
        
        Uses progressively longer generations to compile all code paths.
        """
        if self._warmed_up:
            return

        print("[TurboXInf] Warming up...")
        t0 = time.time()
        # Short warmups to compile the initial graph
        for i in range(runs):
            _ = self.generate(prompt, max_new_tokens=max_tokens, do_sample=False)
        # One longer generation to compile the decode path fully
        _ = self.generate(
            "Explain the meaning of life in detail.",
            max_new_tokens=256,
            do_sample=False,
        )
        self._warmed_up = True
        print(f"[TurboXInf] Warmup done in {time.time() - t0:.2f}s")

    def _build_input(
        self,
        prompt: str,
        messages: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Build tokenized input from prompt or messages."""
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        # Plugin hook
        messages = self.plugin_manager.run_hook(
            PluginHook.PRE_TOKENIZE, messages
        ) or messages

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        inputs = self.plugin_manager.run_hook(
            PluginHook.POST_TOKENIZE, inputs
        ) or inputs

        return inputs

    def _get_generation_kwargs(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        **extra_kwargs,
    ) -> dict:
        """Build generation kwargs from config with overrides."""
        cfg = self.config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens,
            "do_sample": do_sample if do_sample is not None else cfg.do_sample,
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
        }

        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = temperature if temperature is not None else cfg.temperature
            gen_kwargs["top_p"] = top_p if top_p is not None else cfg.top_p
            gen_kwargs["top_k"] = top_k if top_k is not None else cfg.top_k

        gen_kwargs.update(extra_kwargs)
        return gen_kwargs

    @torch.inference_mode()
    def generate(
        self,
        prompt: str = "",
        messages: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Non-streaming generation. Returns a dict with:
        - content: str (final response)
        - reasoning: str (thinking content, if any)
        - usage: dict (token counts)
        - time_taken: float
        - tokens_per_sec: float
        """
        inputs = self._build_input(prompt, messages, system_prompt)
        gen_kwargs = self._get_generation_kwargs(
            max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty, **kwargs
        )
        input_len = inputs["input_ids"].shape[1]

        self.plugin_manager.run_hook(PluginHook.PRE_GENERATE, gen_kwargs)

        torch.cuda.synchronize()
        t0 = time.time()

        outputs = self.model.generate(**inputs, **gen_kwargs)

        torch.cuda.synchronize()
        total_time = time.time() - t0

        output_ids = outputs[0][input_len:]
        output_len = len(output_ids)
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Parse thinking/content
        reasoning, content = self._parse_response(response)

        result = {
            "content": content,
            "reasoning": reasoning,
            "usage": {
                "input_tokens": input_len,
                "output_tokens": output_len,
                "total_tokens": input_len + output_len,
            },
            "time_taken": round(total_time, 4),
            "tokens_per_sec": round(output_len / total_time, 2) if total_time > 0 else 0,
        }

        # Update stats
        self._stats["total_tokens_generated"] += output_len
        self._stats["total_time_s"] += total_time
        self._stats["total_requests"] += 1

        result = self.plugin_manager.run_hook(PluginHook.POST_GENERATE, result) or result
        return result

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str = "",
        messages: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming generation. Yields dicts with:
        - token: str (new token text)
        - done: bool
        - usage: dict (final only)
        - time_taken: float (final only)
        - tokens_per_sec: float (final only)
        """
        inputs = self._build_input(prompt, messages, system_prompt)
        gen_kwargs = self._get_generation_kwargs(
            max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty, **kwargs
        )
        input_len = inputs["input_ids"].shape[1]

        self.plugin_manager.run_hook(PluginHook.PRE_GENERATE, gen_kwargs)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )
        gen_kwargs["streamer"] = streamer

        torch.cuda.synchronize()
        t0 = time.time()
        token_count = 0
        full_response = []

        # Run generation in background thread
        thread = threading.Thread(
            target=self.model.generate, kwargs={**inputs, **gen_kwargs}
        )
        thread.start()

        for text in streamer:
            token_count += 1
            full_response.append(text)
            self.plugin_manager.run_hook(PluginHook.ON_TOKEN, text)
            yield {"token": text, "done": False}

        thread.join()
        torch.cuda.synchronize()
        total_time = time.time() - t0

        response = "".join(full_response)
        reasoning, content = self._parse_response(response)

        self._stats["total_tokens_generated"] += token_count
        self._stats["total_time_s"] += total_time
        self._stats["total_requests"] += 1

        yield {
            "token": "",
            "done": True,
            "content": content,
            "reasoning": reasoning,
            "usage": {
                "input_tokens": input_len,
                "output_tokens": token_count,
                "total_tokens": input_len + token_count,
            },
            "time_taken": round(total_time, 4),
            "tokens_per_sec": round(token_count / total_time, 2) if total_time > 0 else 0,
        }

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Split response into reasoning (thinking) and content."""
        if "</think>" in response:
            parts = response.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
        else:
            reasoning = ""
            content = response.strip()
        return reasoning, content

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        avg_tps = 0
        if self._stats["total_time_s"] > 0:
            avg_tps = self._stats["total_tokens_generated"] / self._stats["total_time_s"]
        return {
            **self._stats,
            "avg_tokens_per_sec": round(avg_tps, 2),
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            "total_tokens_generated": 0,
            "total_time_s": 0.0,
            "total_requests": 0,
        }

    def unload(self):
        """Unload model and free VRAM."""
        self.plugin_manager.unload_all()
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — High-Performance Custom Decode Engine

This module implements a highly optimized generation loop that bypasses
the overhead of transformers.generate() by:
1. Using a StaticCache for pre-allocated KV cache (no dynamic allocation)
2. Using CUDA Graphs for the decode step (zero kernel launch overhead)
3. Using torch.compile for kernel fusion
4. Minimizing Python overhead between tokens
"""

import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from .config import TurboXInfConfig


class TurboDecoder:
    """
    Ultra-fast custom decode loop with CUDA Graphs.

    Achieves near-theoretical memory bandwidth utilization by:
    - Pre-allocating all tensors (zero allocation during decode)
    - Capturing CUDA graphs for decode step
    - Fusing operations via torch.compile
    - Eliminating all Python overhead in the hot loop
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 config: TurboXInfConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        # Pre-allocated decode tensors
        self._decode_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._decode_position_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._decode_cache_position = torch.zeros((1,), dtype=torch.long, device=self.device)

        # CUDA graph state
        self._cuda_graph = None
        self._graph_output_logits = None
        self._graph_captured = False

        # Compiled forward
        self._compiled_forward = None

        # Static cache
        self._cache = None
        self._max_cache_len = config.max_cache_length

    def _setup_cache(self):
        """Initialize static KV cache."""
        self._cache = StaticCache(
            self.model.config,
            max_batch_size=1,
            max_cache_len=self._max_cache_len,
            device=self.device,
            dtype=self.config.get_torch_dtype(),
        )

    def _reset_cache(self):
        """Reset cache for a new generation."""
        if self._cache is not None:
            self._cache.reset()
        else:
            self._setup_cache()

    def _compile_decode(self):
        """Compile the model forward for decode step."""
        if self._compiled_forward is not None:
            return

        # Compile just the model for decode (single token input)
        try:
            self._compiled_forward = torch.compile(
                self.model,
                mode=self.config.compile_mode,
                fullgraph=True,
            )
        except Exception as e:
            print(f"[TurboDecoder] torch.compile failed: {e}, using uncompiled model")
            self._compiled_forward = self.model

    def _capture_cuda_graph(self, position: int):
        """Capture CUDA graph for the decode step."""
        if not self.config.use_cuda_graphs:
            return

        # Warm up with real computation
        self._decode_input_ids.fill_(0)
        self._decode_position_ids.fill_(position)
        self._decode_cache_position.fill_(position)

        model_fn = self._compiled_forward or self.model

        # Warmup runs to stabilize memory
        for _ in range(3):
            with torch.no_grad():
                out = model_fn(
                    input_ids=self._decode_input_ids,
                    position_ids=self._decode_position_ids,
                    past_key_values=self._cache,
                    cache_position=self._decode_cache_position,
                    use_cache=True,
                    logits_to_keep=1,
                )

        # Capture CUDA graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            with torch.no_grad():
                out = model_fn(
                    input_ids=self._decode_input_ids,
                    position_ids=self._decode_position_ids,
                    past_key_values=self._cache,
                    cache_position=self._decode_cache_position,
                    use_cache=True,
                    logits_to_keep=1,
                )
                self._graph_output_logits = out.logits

        self._graph_captured = True

    def _decode_step_graph(self, token_id: int, position: int) -> torch.Tensor:
        """Execute one decode step using captured CUDA graph."""
        self._decode_input_ids[0, 0] = token_id
        self._decode_position_ids[0, 0] = position
        self._decode_cache_position[0] = position
        self._cuda_graph.replay()
        return self._graph_output_logits[:, -1, :]

    def _decode_step_eager(self, token_id: int, position: int) -> torch.Tensor:
        """Execute one decode step without CUDA graph."""
        self._decode_input_ids[0, 0] = token_id
        self._decode_position_ids[0, 0] = position
        self._decode_cache_position[0] = position

        model_fn = self._compiled_forward or self.model
        with torch.no_grad():
            out = model_fn(
                input_ids=self._decode_input_ids,
                position_ids=self._decode_position_ids,
                past_key_values=self._cache,
                cache_position=self._decode_cache_position,
                use_cache=True,
                logits_to_keep=1,
            )
        return out.logits[:, -1, :]

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run prefill phase (process full prompt).
        Returns logits for the last position.
        """
        self._reset_cache()
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=self.device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)

        # Prefill in one shot (no CUDA graph needed, this is variable-length)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self._cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        return out.logits[:, -1, :]

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0,
                      top_p: float = 1.0, top_k: int = 0,
                      do_sample: bool = False) -> int:
        """Sample a token from logits."""
        if not do_sample:
            return logits.argmax(dim=-1).item()

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            min_val = top_k_values[:, -1:]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf'), device=logits.device), logits)

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _apply_repetition_penalty(self, logits: torch.Tensor,
                                   generated_ids: List[int],
                                   penalty: float = 1.0) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0 or not generated_ids:
            return logits

        penalty_ids = torch.tensor(generated_ids, device=logits.device, dtype=torch.long).unique()
        scores = logits[:, penalty_ids]
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        logits[:, penalty_ids] = scores
        return logits

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
    ) -> Dict[str, Any]:
        """
        Non-streaming generation with custom high-speed decode loop.
        """
        cfg = self.config
        max_tokens = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
        temp = temperature if temperature is not None else cfg.temperature
        tp = top_p if top_p is not None else cfg.top_p
        tk = top_k if top_k is not None else cfg.top_k
        sample = do_sample if do_sample is not None else cfg.do_sample
        rep_pen = repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty

        # Build input
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_len = input_ids.shape[1]

        # Ensure cache is large enough
        required_len = input_len + max_tokens
        if required_len > self._max_cache_len:
            self._max_cache_len = required_len + 64
            self._cache = None
            self._graph_captured = False

        torch.cuda.synchronize()
        t0 = time.time()

        # ── Prefill ──────────────────────────────────────────────────────
        logits = self.prefill(input_ids)

        # ── Decode Loop ──────────────────────────────────────────────────
        generated_ids = []
        eos_id = self.tokenizer.eos_token_id
        position = input_len

        # Try to capture CUDA graph for decode
        if self.config.use_cuda_graphs and not self._graph_captured:
            try:
                self._capture_cuda_graph(position)
            except Exception as e:
                print(f"[TurboDecoder] CUDA graph capture failed: {e}")
                self._graph_captured = False

        # Use CUDA graph or eager decode
        decode_fn = self._decode_step_graph if self._graph_captured else self._decode_step_eager

        for i in range(max_tokens):
            # Apply repetition penalty
            logits = self._apply_repetition_penalty(logits, generated_ids, rep_pen)

            # Sample token
            token_id = self._sample_token(logits, temp, tp, tk, sample)
            generated_ids.append(token_id)

            # Check EOS
            if token_id == eos_id:
                break

            # Decode step
            logits = decode_fn(token_id, position)
            position += 1

        torch.cuda.synchronize()
        total_time = time.time() - t0

        # Decode output
        output_ids = torch.tensor(generated_ids, device='cpu')
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Parse thinking/content
        if "</think>" in response:
            parts = response.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
        else:
            reasoning = ""
            content = response.strip()

        output_len = len(generated_ids)
        return {
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
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming generation. Yields token-by-token.
        """
        cfg = self.config
        max_tokens = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
        temp = temperature if temperature is not None else cfg.temperature
        tp = top_p if top_p is not None else cfg.top_p
        tk = top_k if top_k is not None else cfg.top_k
        sample = do_sample if do_sample is not None else cfg.do_sample
        rep_pen = repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_len = input_ids.shape[1]

        required_len = input_len + max_tokens
        if required_len > self._max_cache_len:
            self._max_cache_len = required_len + 64
            self._cache = None
            self._graph_captured = False

        torch.cuda.synchronize()
        t0 = time.time()

        logits = self.prefill(input_ids)

        generated_ids = []
        eos_id = self.tokenizer.eos_token_id
        position = input_len

        if self.config.use_cuda_graphs and not self._graph_captured:
            try:
                self._capture_cuda_graph(position)
            except Exception:
                self._graph_captured = False

        decode_fn = self._decode_step_graph if self._graph_captured else self._decode_step_eager

        for i in range(max_tokens):
            logits = self._apply_repetition_penalty(logits, generated_ids, rep_pen)
            token_id = self._sample_token(logits, temp, tp, tk, sample)
            generated_ids.append(token_id)

            # Decode token to text
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            yield {"token": token_text, "token_id": token_id, "done": False}

            if token_id == eos_id:
                break

            logits = decode_fn(token_id, position)
            position += 1

        torch.cuda.synchronize()
        total_time = time.time() - t0
        output_len = len(generated_ids)

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "</think>" in response:
            parts = response.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
        else:
            reasoning = ""
            content = response.strip()

        yield {
            "token": "",
            "done": True,
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

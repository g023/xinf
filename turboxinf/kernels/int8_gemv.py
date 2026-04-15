"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

Triton INT8 GEMV kernel and LinearINT8 replacement module.

Provides close to 2x speedup for autoregressive decode (batch=1)
by reading INT8 weights (1 byte) instead of BF16 (2 bytes).
Falls back to BF16 cuBLAS for prefill (batch > 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def int8_gemv_kernel(
    W_ptr, x_ptr, y_ptr, scale_ptr,
    M, K,
    stride_wm, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT8 weight × BF16 activation fused GEMV."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        )
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)

    s = tl.load(scale_ptr + offs_m, mask=mask_m)
    tl.store(y_ptr + offs_m, (acc * s).to(tl.bfloat16), mask=mask_m)


@triton.jit
def int8_gemv_bias_kernel(
    W_ptr, x_ptr, y_ptr, scale_ptr, bias_ptr,
    M, K,
    stride_wm, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT8 weight × BF16 activation fused GEMV with bias add."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        )
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)

    s = tl.load(scale_ptr + offs_m, mask=mask_m)
    b = tl.load(bias_ptr + offs_m, mask=mask_m)
    tl.store(y_ptr + offs_m, (acc * s + b.to(tl.float32)).to(tl.bfloat16), mask=mask_m)


def _select_block_sizes(M, K):
    """Heuristic block size selection based on matrix dimensions."""
    if M >= 4096:
        return 32, 256
    elif M >= 1024:
        return 32, 256
    else:
        return 32, 128


class LinearINT8(nn.Module):
    """Drop-in replacement for nn.Linear using INT8 weights + Triton GEMV.
    
    For single-token decode (batch=1), uses a fused INT8 GEMV kernel.
    For prefill (batch > 1), dequantizes to BF16 and uses cuBLAS.
    """

    def __init__(self, weight_bf16: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        M, K = weight_bf16.shape
        amax = weight_bf16.abs().amax(dim=1).clamp(min=1e-10)
        scale = amax / 127.0
        w_int8 = (weight_bf16 / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.M = M
        self.K = K
        self.BLOCK_M, self.BLOCK_K = _select_block_sizes(M, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.K)
        B = x_2d.shape[0]

        if B == 1:
            # Fused INT8 GEMV
            x_flat = x_2d.squeeze(0).contiguous()
            y = torch.empty(self.M, dtype=torch.bfloat16, device=x.device)
            grid = ((self.M + self.BLOCK_M - 1) // self.BLOCK_M,)

            if self.bias is not None:
                int8_gemv_bias_kernel[grid](
                    self.weight_int8, x_flat, y, self.scale, self.bias,
                    self.M, self.K,
                    self.weight_int8.stride(0), self.weight_int8.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_K=self.BLOCK_K,
                )
            else:
                int8_gemv_kernel[grid](
                    self.weight_int8, x_flat, y, self.scale,
                    self.M, self.K,
                    self.weight_int8.stride(0), self.weight_int8.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_K=self.BLOCK_K,
                )
            y = y.unsqueeze(0)
        else:
            # Fallback: dequantize to BF16 and use cuBLAS
            w_bf16 = self.weight_int8.to(torch.bfloat16) * self.scale.unsqueeze(1)
            y = F.linear(x_2d, w_bf16, self.bias)

        return y.view(*orig_shape[:-1], self.M)


def replace_linear_with_int8(model: nn.Module, skip_lm_head: bool = False) -> nn.Module:
    """Replace all nn.Linear layers in a model with LinearINT8.
    
    Args:
        model: The model to quantize
        skip_lm_head: If True, don't quantize the lm_head (for tied embeddings)
    
    Returns:
        The model with INT8 linear layers
    """
    replaced = 0
    skipped = 0

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name

                if skip_lm_head and "lm_head" in full_name:
                    skipped += 1
                    continue

                int8_linear = LinearINT8(
                    child.weight.data,
                    child.bias.data if child.bias is not None else None,
                )
                setattr(module, child_name, int8_linear)
                replaced += 1

    print(f"  Replaced {replaced} linear layers with INT8 (skipped {skipped})")
    return model

"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

Triton INT4 GEMV kernel with group quantization.

For autoregressive decode (batch=1), reads packed INT4 weights (2 values per byte),
dequantizes with per-group scales in registers, and accumulates in FP32.
Expected ~2x memory bandwidth reduction vs INT8, ~4x vs BF16.

Packing format: two signed INT4 values per uint8 byte
  - Low nibble  (byte & 0xF): even K index (stored as unsigned + 8)
  - High nibble (byte >> 4):   odd K index  (stored as unsigned + 8)
  - Signed range: [-8, 7], unsigned storage: [0, 15]

Group quantization: scale per group of GROUP_SIZE consecutive elements
  - scale[m, g] = absmax(W[m, g*GS:(g+1)*GS]) / 7.0
  - w_int4[m, k] = round(W[m, k] / scale[m, k // GS]).clamp(-8, 7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ── Globals ──────────────────────────────────────────────────────────────────
DEFAULT_GROUP_SIZE = 128
DEFAULT_BLOCK_M = 32


@triton.jit
def int4_gemv_kernel(
    W_ptr,        # uint8 packed weights [M, K//2]
    x_ptr,        # bf16 input activations [K]
    y_ptr,        # bf16 output [M]
    scale_ptr,    # bf16 per-group scales [M, num_groups]
    M, K,
    stride_wm,    # stride of W packed in dim 0 (uint8 units)
    num_groups,   # K // GROUP_SIZE
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    """Optimized INT4 GEMV: processes multiple groups per outer iteration
    with 1D scale loads per group (unrolled inner loop)."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    K_PACKED = K // 2
    HALF_GS: tl.constexpr = GROUP_SIZE // 2  # packed bytes per group

    for k_byte_start in range(0, K_PACKED, HALF_GS):
        offs_k = k_byte_start + tl.arange(0, HALF_GS)
        mask_k = offs_k < K_PACKED

        # Load packed INT4 weights [BLOCK_M, HALF_GS]
        packed = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int32)

        # Unpack two INT4 values per byte
        low = (packed & 0xF) - 8
        high = ((packed >> 4) & 0xF) - 8

        # Load activation pairs
        k_even = offs_k * 2
        k_odd = k_even + 1

        x_even = tl.load(x_ptr + k_even, mask=k_even < K, other=0.0).to(tl.float32)
        x_odd = tl.load(x_ptr + k_odd, mask=k_odd < K, other=0.0).to(tl.float32)

        # 1D scale load — one group per iteration (HALF_GS covers exactly 1 group)
        group_idx = k_byte_start // HALF_GS
        scale = tl.load(
            scale_ptr + offs_m * num_groups + group_idx,
            mask=mask_m, other=1.0,
        ).to(tl.float32)

        # Unscaled dot product [BLOCK_M], then scale
        dot = tl.sum(low.to(tl.float32) * x_even[None, :] +
                     high.to(tl.float32) * x_odd[None, :], axis=1)
        acc += scale * dot

    tl.store(y_ptr + offs_m, acc.to(tl.bfloat16), mask=mask_m)


@triton.jit
def int4_gemv_bias_kernel(
    W_ptr, x_ptr, y_ptr, scale_ptr, bias_ptr,
    M, K,
    stride_wm, num_groups,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    """Optimized INT4 GEMV with fused bias addition."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    K_PACKED = K // 2
    HALF_GS: tl.constexpr = GROUP_SIZE // 2

    for k_byte_start in range(0, K_PACKED, HALF_GS):
        offs_k = k_byte_start + tl.arange(0, HALF_GS)
        mask_k = offs_k < K_PACKED

        packed = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int32)

        low = (packed & 0xF) - 8
        high = ((packed >> 4) & 0xF) - 8

        k_even = offs_k * 2
        k_odd = k_even + 1

        x_even = tl.load(x_ptr + k_even, mask=k_even < K, other=0.0).to(tl.float32)
        x_odd = tl.load(x_ptr + k_odd, mask=k_odd < K, other=0.0).to(tl.float32)

        group_idx = k_byte_start // HALF_GS
        scale = tl.load(
            scale_ptr + offs_m * num_groups + group_idx,
            mask=mask_m, other=1.0,
        ).to(tl.float32)

        dot = tl.sum(low.to(tl.float32) * x_even[None, :] +
                     high.to(tl.float32) * x_odd[None, :], axis=1)
        acc += scale * dot

    b = tl.load(bias_ptr + offs_m, mask=mask_m).to(tl.float32)
    tl.store(y_ptr + offs_m, (acc + b).to(tl.bfloat16), mask=mask_m)


def _select_int4_block_sizes(M, K, group_size=DEFAULT_GROUP_SIZE):
    """Select BLOCK_M. BLOCK_K_PACKED is always GROUP_SIZE//2 (one group per iteration)."""
    if M >= 4096:
        block_m = 64
    elif M >= 1024:
        block_m = 32
    else:
        block_m = 32

    # BLOCK_K_PACKED = GROUP_SIZE // 2 (one group per inner loop iteration)
    block_k_packed = group_size // 2

    # Ensure power of 2
    p = 1
    while p * 2 <= block_k_packed:
        p *= 2
    block_k_packed = p

    return block_m, block_k_packed


class LinearINT4(nn.Module):
    """Drop-in replacement for nn.Linear using INT4 weights + Triton GEMV.

    For single-token decode (batch=1), uses a fused INT4 GEMV kernel.
    For prefill (batch > 1), dequantizes to BF16 and uses cuBLAS.

    Group quantization: weights are quantized per group of `group_size` elements
    along the K dimension. Each group has its own scale factor.
    """

    def __init__(self, weight_bf16: torch.Tensor, bias: torch.Tensor = None,
                 group_size: int = DEFAULT_GROUP_SIZE):
        super().__init__()
        M, K = weight_bf16.shape
        assert K % 2 == 0, f"K ({K}) must be even for INT4 packing"
        assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"

        num_groups = K // group_size

        # ── Per-group symmetric quantization (CPU to avoid OOM) ─────────
        w_cpu = weight_bf16.float().cpu()
        weight_groups = w_cpu.reshape(M, num_groups, group_size)
        amax = weight_groups.abs().amax(dim=2).clamp(min=1e-10)  # [M, num_groups]
        scale = amax / 7.0  # range [-8, 7], max absolute value is 7

        # Quantize to signed INT4 [-8, 7]
        w_int4 = (weight_groups / scale.unsqueeze(2)).round().clamp(-8, 7).to(torch.int8)
        w_int4 = w_int4.reshape(M, K)

        # ── Pack two INT4 values into one uint8 byte ─────────────────────
        w_unsigned = (w_int4 + 8).to(torch.uint8)  # shift to [0, 15]
        w_even = w_unsigned[:, 0::2]  # even K indices → low nibble
        w_odd = w_unsigned[:, 1::2]   # odd K indices → high nibble
        w_packed = (w_even | (w_odd << 4)).contiguous()

        device = weight_bf16.device
        self.register_buffer("weight_packed", w_packed.to(device))  # [M, K//2]
        self.register_buffer("scale", scale.to(torch.bfloat16).contiguous().to(device))  # [M, num_groups]
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16).contiguous().to(device))
        else:
            self.bias = None

        self.M = M
        self.K = K
        self.num_groups = num_groups
        self.group_size = group_size
        self.BLOCK_M, self.BLOCK_K_PACKED = _select_int4_block_sizes(M, K, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.K)
        B = x_2d.shape[0]

        if B == 1:
            # ── Fused INT4 GEMV path (decode) ────────────────────────────
            x_flat = x_2d.squeeze(0).contiguous()
            y = torch.empty(self.M, dtype=torch.bfloat16, device=x.device)
            grid = ((self.M + self.BLOCK_M - 1) // self.BLOCK_M,)

            if self.bias is not None:
                int4_gemv_bias_kernel[grid](
                    self.weight_packed, x_flat, y, self.scale, self.bias,
                    self.M, self.K,
                    self.weight_packed.stride(0),
                    self.num_groups,
                    GROUP_SIZE=self.group_size,
                    BLOCK_M=self.BLOCK_M,
                    BLOCK_K_PACKED=self.BLOCK_K_PACKED,
                )
            else:
                int4_gemv_kernel[grid](
                    self.weight_packed, x_flat, y, self.scale,
                    self.M, self.K,
                    self.weight_packed.stride(0),
                    self.num_groups,
                    GROUP_SIZE=self.group_size,
                    BLOCK_M=self.BLOCK_M,
                    BLOCK_K_PACKED=self.BLOCK_K_PACKED,
                )
            y = y.unsqueeze(0)
        else:
            # ── Fallback: dequantize to BF16 for cuBLAS (prefill) ────────
            w_bf16 = self._dequantize()
            y = F.linear(x_2d, w_bf16, self.bias)

        return y.view(*orig_shape[:-1], self.M)

    def _dequantize(self) -> torch.Tensor:
        """Dequantize packed INT4 weights to BF16 for prefill fallback."""
        packed = self.weight_packed  # [M, K//2] uint8
        # Unpack
        low = (packed.to(torch.int16) & 0xF).to(torch.int8) - 8   # even indices
        high = ((packed.to(torch.int16) >> 4) & 0xF).to(torch.int8) - 8  # odd indices

        # Interleave back to [M, K]
        M, K_half = packed.shape
        w_int4 = torch.stack([low, high], dim=2).reshape(M, K_half * 2)

        # Dequantize with per-group scale
        w_groups = w_int4.reshape(M, self.num_groups, self.group_size).to(torch.bfloat16)
        scale_expanded = self.scale.unsqueeze(2)  # [M, num_groups, 1]
        w_bf16 = (w_groups * scale_expanded).reshape(M, self.K)
        return w_bf16


def replace_linear_with_int4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    skip_lm_head: bool = False,
    skip_patterns: list = None,
) -> nn.Module:
    """Replace nn.Linear layers with LinearINT4.

    Args:
        model: The model to quantize
        group_size: Number of elements per quantization group
        skip_lm_head: If True, don't quantize lm_head
        skip_patterns: List of name patterns to skip (e.g., ["vision", "visual"])

    Returns:
        The model with INT4 linear layers
    """
    skip_patterns = skip_patterns or []
    replaced = 0
    skipped = 0

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name

                # Check skip conditions
                if skip_lm_head and "lm_head" in full_name:
                    skipped += 1
                    continue

                if any(pat in full_name for pat in skip_patterns):
                    skipped += 1
                    continue

                # K must be divisible by group_size and even
                K = child.weight.shape[1]
                if K % 2 != 0 or K % group_size != 0:
                    skipped += 1
                    continue

                int4_linear = LinearINT4(
                    child.weight.data,
                    child.bias.data if child.bias is not None else None,
                    group_size=group_size,
                )
                setattr(module, child_name, int4_linear)
                replaced += 1

    print(f"  Replaced {replaced} linear layers with INT4 (group_size={group_size}, skipped {skipped})")
    return model


def replace_linear_mixed_int4_int8(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    skip_lm_head: bool = False,
    skip_patterns: list = None,
    int8_patterns: list = None,
) -> nn.Module:
    """Mixed quantization: INT4 for FFN layers, INT8 for attention projections.

    This provides near-INT4 speed with better quality preservation for
    attention-sensitive weights.

    Args:
        model: The model to quantize
        group_size: Group size for INT4 quantization
        skip_lm_head: If True, don't quantize lm_head
        skip_patterns: Name patterns to skip entirely
        int8_patterns: Name patterns to quantize as INT8 instead of INT4

    Returns:
        The model with mixed INT4/INT8 linear layers
    """
    from .int8_gemv import LinearINT8

    skip_patterns = skip_patterns or []
    int8_patterns = int8_patterns or ["q_proj", "k_proj", "v_proj", "o_proj"]
    replaced_int4 = 0
    replaced_int8 = 0
    skipped = 0

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name

                if skip_lm_head and "lm_head" in full_name:
                    skipped += 1
                    continue

                if any(pat in full_name for pat in skip_patterns):
                    skipped += 1
                    continue

                # Determine quantization type
                use_int8 = any(pat in full_name for pat in int8_patterns)

                if use_int8:
                    int8_linear = LinearINT8(
                        child.weight.data,
                        child.bias.data if child.bias is not None else None,
                    )
                    setattr(module, child_name, int8_linear)
                    replaced_int8 += 1
                else:
                    K = child.weight.shape[1]
                    if K % 2 != 0 or K % group_size != 0:
                        # Fall back to INT8 if INT4 constraints not met
                        int8_linear = LinearINT8(
                            child.weight.data,
                            child.bias.data if child.bias is not None else None,
                        )
                        setattr(module, child_name, int8_linear)
                        replaced_int8 += 1
                    else:
                        int4_linear = LinearINT4(
                            child.weight.data,
                            child.bias.data if child.bias is not None else None,
                            group_size=group_size,
                        )
                        setattr(module, child_name, int4_linear)
                        replaced_int4 += 1

    print(f"  Mixed quant: {replaced_int4} INT4 + {replaced_int8} INT8 (skipped {skipped})")
    return model

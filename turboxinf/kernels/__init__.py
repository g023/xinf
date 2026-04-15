"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf Kernels — Custom Triton GEMV kernels for quantized inference.

Available kernels:
  - INT8 GEMV: Per-row symmetric quantization, ~2x bandwidth reduction
  - INT4 GEMV: Per-group symmetric quantization, ~4x bandwidth reduction
  - Mixed INT4/INT8: INT4 for FFN, INT8 for attention projections
"""

from .int8_gemv import LinearINT8, replace_linear_with_int8
from .int4_gemv import LinearINT4, replace_linear_with_int4, replace_linear_mixed_int4_int8

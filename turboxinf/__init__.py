"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — High-Performance Inference Engine

2-2.5x faster inference for Qwen3/Qwen3.5 via custom Triton INT8/INT4 GEMV kernels.
"""
__version__ = "2.0.0"

from .config import TurboXInfConfig
from .engine import TurboXInfEngine

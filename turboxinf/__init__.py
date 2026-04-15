"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — Inference Engine

2x faster inference for Qwen3-1.77B on RTX 3060 via custom Triton INT8 GEMV kernels.
"""
__version__ = "1.0.0"

from .config import TurboXInfConfig
from .engine import TurboXInfEngine

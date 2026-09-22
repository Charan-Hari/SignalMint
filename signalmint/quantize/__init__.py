"""INT8 quantization and export to the portable C runtime.

Phase 3: post-training quantization to integer parameters, an integer inference
reference that matches the C runtime bit-for-bit, and a C-header exporter. The
parity between :class:`IntegerRuntime` and the compiled runtime is the headline
deliverable.
"""

from __future__ import annotations

from .export import export_c_header
from .fixedpoint import FRAC_BITS, quantize_multiplier, requantize
from .int_infer import IntegerRuntime
from .quantizer import QuantConv, QuantModel, quantize_model

__all__ = [
    "quantize_model",
    "QuantModel",
    "QuantConv",
    "IntegerRuntime",
    "export_c_header",
    "quantize_multiplier",
    "requantize",
    "FRAC_BITS",
]

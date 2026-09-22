"""Data loading, framing and quantization for 1-D signals.

Phase 1: CWRU bearing-vibration adapter (with offline synthetic fallback),
frame extraction, and mu-law / linear amplitude quantization to an INT8-friendly
categorical alphabet.
"""

from __future__ import annotations

from .cwru import (
    CWRU_FILES,
    SignalRecord,
    get_cwru_records,
    get_synthetic_records,
    synthetic_bearing,
)
from .dataset import FrameDataset, FrameSet, records_to_frames
from .framing import frame_signal, num_frames
from .loader import load_records
from .quantize import dequantize, mu_law_decode, mu_law_encode, quantize

__all__ = [
    "SignalRecord",
    "CWRU_FILES",
    "get_cwru_records",
    "get_synthetic_records",
    "synthetic_bearing",
    "load_records",
    "frame_signal",
    "num_frames",
    "quantize",
    "dequantize",
    "mu_law_encode",
    "mu_law_decode",
    "records_to_frames",
    "FrameSet",
    "FrameDataset",
]

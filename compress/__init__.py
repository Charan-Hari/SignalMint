"""Neural-compression head.

Phase 2: an arithmetic coder driven by the model's per-sample probabilities,
measured in bits-per-sample against gzip / order-0 entropy baselines.
"""

from __future__ import annotations

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder, BitReader, BitWriter
from .baselines import (
    gzip_bits_per_sample,
    order0_entropy_bits_per_sample,
    raw_bits_per_sample,
)
from .coder import CompressionResult, decode_frame, encode_frames, frame_distributions
from .freqs import cumulative, find_symbol, probs_to_freqs

__all__ = [
    "ArithmeticEncoder",
    "ArithmeticDecoder",
    "BitWriter",
    "BitReader",
    "probs_to_freqs",
    "cumulative",
    "find_symbol",
    "encode_frames",
    "decode_frame",
    "frame_distributions",
    "CompressionResult",
    "gzip_bits_per_sample",
    "order0_entropy_bits_per_sample",
    "raw_bits_per_sample",
]

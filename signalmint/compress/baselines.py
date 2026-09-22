"""Baseline compressors for bits-per-sample comparison.

These are the yardsticks the neural codec must beat: a general-purpose
byte compressor (gzip) and the order-0 Shannon entropy (the best any
memoryless coder can do on the symbol stream).
"""

from __future__ import annotations

import gzip
import math

import numpy as np

__all__ = ["gzip_bits_per_sample", "order0_entropy_bits_per_sample", "raw_bits_per_sample"]


def raw_bits_per_sample(num_bins: int) -> float:
    """Uniform fixed-width code cost: ``log2(num_bins)`` bits/sample."""
    return math.log2(num_bins)


def gzip_bits_per_sample(symbols: np.ndarray, num_bins: int, level: int = 9) -> float:
    """Bits/sample when the symbol stream is packed to bytes and gzip'd.

    Symbols are stored in the smallest whole number of bytes that holds
    ``num_bins`` values (1 byte for <=256), then compressed with gzip. This is a
    fair, ubiquitous general-purpose baseline.
    """
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if symbols.size == 0:
        return 0.0
    if num_bins <= 256:
        payload = symbols.astype(np.uint8).tobytes()
    elif num_bins <= 65536:
        payload = symbols.astype("<u2").tobytes()
    else:
        payload = symbols.astype("<u4").tobytes()
    compressed = gzip.compress(payload, compresslevel=level)
    return len(compressed) * 8 / symbols.size


def order0_entropy_bits_per_sample(symbols: np.ndarray, num_bins: int) -> float:
    """Order-0 empirical Shannon entropy in bits/sample (memoryless lower bound)."""
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if symbols.size == 0:
        return 0.0
    counts = np.bincount(symbols, minlength=num_bins).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    return float(-(nz * np.log2(nz)).sum())

"""Convert model probabilities into exact integer frequency tables.

The arithmetic coder needs integer cumulative frequencies that sum to a fixed
``total``. This module maps a float probability vector to non-zero integer
frequencies summing exactly to ``2**total_bits`` -- deterministically, so the
encoder and decoder always agree.
"""

from __future__ import annotations

import numpy as np

__all__ = ["probs_to_freqs", "cumulative", "find_symbol"]


def probs_to_freqs(probs: np.ndarray, total_bits: int = 16) -> np.ndarray:
    """Map a probability vector to positive integer freqs summing to ``2**total_bits``.

    Every symbol receives at least frequency 1 (so it stays decodable), and the
    remaining mass is distributed proportionally with a deterministic tie-break,
    guaranteeing encoder/decoder agreement.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = probs.size
    total = 1 << total_bits
    if n > total:
        raise ValueError("alphabet larger than total frequency budget")

    freqs = np.ones(n, dtype=np.int64)
    remaining = total - n
    freqs += np.floor(probs * remaining).astype(np.int64)

    diff = total - int(freqs.sum())
    order = np.argsort(-probs, kind="stable")
    i = 0
    while diff > 0:
        freqs[order[i % n]] += 1
        diff -= 1
        i += 1
    while diff < 0:
        j = order[i % n]
        if freqs[j] > 1:
            freqs[j] -= 1
            diff += 1
        i += 1
    return freqs


def cumulative(freqs: np.ndarray) -> np.ndarray:
    """Cumulative frequency boundaries: ``cum[s]..cum[s+1]`` is symbol ``s``."""
    cum = np.zeros(freqs.size + 1, dtype=np.int64)
    np.cumsum(freqs, out=cum[1:])
    return cum


def find_symbol(cum: np.ndarray, target: int) -> int:
    """Return symbol ``s`` such that ``cum[s] <= target < cum[s+1]``."""
    return int(np.searchsorted(cum, target, side="right") - 1)

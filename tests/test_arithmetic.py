"""Tests for the arithmetic coder and frequency tables (no torch needed)."""

from __future__ import annotations

import numpy as np

from signalmint.compress.arithmetic import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitReader,
    BitWriter,
)
from signalmint.compress.freqs import cumulative, find_symbol, probs_to_freqs


def test_bit_writer_reader_roundtrip() -> None:
    bits = [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    w = BitWriter()
    for b in bits:
        w.write_bit(b)
    r = BitReader(w.getvalue())
    assert [r.read_bit() for _ in range(len(bits))] == bits


def _static_freqs(num_bins: int, counts: np.ndarray, total_bits: int = 16) -> np.ndarray:
    probs = counts / counts.sum()
    return probs_to_freqs(probs, total_bits)


def test_arithmetic_roundtrip_static_model() -> None:
    """Encode/decode a symbol sequence under a fixed distribution."""
    rng = np.random.default_rng(0)
    num_bins = 16
    counts = rng.integers(1, 50, size=num_bins).astype(np.float64)
    freqs = _static_freqs(num_bins, counts)
    cum = cumulative(freqs)
    total = int(cum[-1])

    symbols = rng.integers(0, num_bins, size=500).astype(np.int64)

    enc = ArithmeticEncoder()
    for s in symbols:
        enc.encode(int(cum[s]), int(cum[s + 1]), total)
    data = enc.finish()

    dec = ArithmeticDecoder(data)
    decoded = []
    for _ in range(len(symbols)):
        target = dec.decode_target(total)
        s = find_symbol(cum, target)
        dec.update(int(cum[s]), int(cum[s + 1]), total)
        decoded.append(s)
    assert np.array_equal(np.asarray(decoded), symbols)


def test_arithmetic_beats_uniform_when_skewed() -> None:
    """A skewed source should compress below log2(num_bins) bits/sample."""
    num_bins = 16
    counts = np.ones(num_bins)
    counts[0] = 1000  # very skewed
    freqs = _static_freqs(num_bins, counts)
    cum = cumulative(freqs)
    total = int(cum[-1])
    symbols = np.zeros(2000, dtype=np.int64)  # all the common symbol
    enc = ArithmeticEncoder()
    for s in symbols:
        enc.encode(int(cum[s]), int(cum[s + 1]), total)
    data = enc.finish()
    bits_per_sample = len(data) * 8 / len(symbols)
    assert bits_per_sample < np.log2(num_bins)


def test_probs_to_freqs_sums_to_total_and_positive() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.random(37)
        p /= p.sum()
        freqs = probs_to_freqs(p, total_bits=16)
        assert freqs.sum() == (1 << 16)
        assert freqs.min() >= 1


def test_probs_to_freqs_handles_zeros() -> None:
    p = np.zeros(8)
    p[3] = 1.0
    freqs = probs_to_freqs(p, total_bits=12)
    assert freqs.sum() == (1 << 12)
    assert freqs.min() >= 1
    assert np.argmax(freqs) == 3

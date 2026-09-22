"""Fixed-point primitives shared by the Python integer reference and the C runtime.

Bit-exact parity between Python and C requires that every arithmetic operation be
integer and identically rounded on both sides. This module defines:

* ``quantize_multiplier`` -- represent a positive float scale as ``M / 2**shift``
  with ``M`` a 31-bit integer (the TFLite-style requant multiplier);
* ``requantize`` -- apply that multiplier to a 64-bit accumulator with
  round-half-up, exactly as the C runtime does;
* ``make_tanh_lut`` / ``make_sigmoid_lut`` and ``lut_lookup`` -- integer lookup
  tables for the gated activation, generated once in Python and emitted to C so
  both use identical entries.

Activations are Q(FRAC_BITS) fixed-point int32 values: the real number ``r`` is
stored as ``round(r * 2**FRAC_BITS)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FRAC_BITS",
    "LUT_SIZE",
    "LUT_RANGE",
    "quantize_multiplier",
    "requantize",
    "requantize_vec",
    "round_shift_vec",
    "make_tanh_lut",
    "make_sigmoid_lut",
    "lut_index",
    "lut_lookup",
    "to_fixed",
    "from_fixed",
]

FRAC_BITS = 12
ONE_Q = 1 << FRAC_BITS

# Gated-activation LUT: inputs clamped to [-LUT_RANGE, LUT_RANGE] real units.
LUT_SIZE = 4096
LUT_RANGE = 8


def to_fixed(x: np.ndarray | float) -> np.ndarray:
    """Convert real values to Q(FRAC_BITS) int32."""
    return np.round(np.asarray(x, dtype=np.float64) * ONE_Q).astype(np.int64)


def from_fixed(x: np.ndarray | int) -> np.ndarray:
    """Convert Q(FRAC_BITS) fixed-point back to real (float)."""
    return np.asarray(x, dtype=np.float64) / ONE_Q


def quantize_multiplier(scale: float) -> tuple[int, int]:
    """Represent a positive ``scale`` as ``(M, total_shift)`` with ``scale == M / 2**total_shift``.

    ``M`` lands in ``[2**30, 2**31)``. Mirrors TFLite's QuantizeMultiplier.
    """
    if scale <= 0.0:
        return 0, 0
    shift = 0
    s = scale
    while s < 0.5:
        s *= 2.0
        shift += 1
    while s >= 1.0:
        s /= 2.0
        shift -= 1
    m = int(round(s * (1 << 31)))
    if m == (1 << 31):
        m //= 2
        shift -= 1
    total_shift = 31 + shift
    return m, total_shift


def requantize(acc: int, multiplier: int, total_shift: int) -> int:
    """Compute ``round(acc * multiplier / 2**total_shift)`` with round-half-up.

    Pure integer; identical to the C implementation. ``acc`` and the product are
    treated as arbitrary-precision here but stay within int64 for the tiny model.
    """
    prod = acc * multiplier
    if total_shift <= 0:
        return prod << (-total_shift)
    half = 1 << (total_shift - 1)
    if prod >= 0:
        return (prod + half) >> total_shift
    # Symmetric round-half-up for negatives (round half away from zero).
    return -((-prod + half) >> total_shift)


def round_shift_vec(prod: np.ndarray, total_shift: int) -> np.ndarray:
    """Vectorized ``round(prod / 2**total_shift)`` with round-half-away-from-zero."""
    prod = np.asarray(prod, dtype=np.int64)
    if total_shift <= 0:
        return prod << (-total_shift)
    half = np.int64(1) << (total_shift - 1)
    pos = (prod + half) >> total_shift
    neg = -((-prod + half) >> total_shift)
    return np.where(prod >= 0, pos, neg)


def requantize_vec(acc: np.ndarray, multiplier: int, total_shift: int) -> np.ndarray:
    """Vectorized :func:`requantize` over an int64 array."""
    prod = np.asarray(acc, dtype=np.int64) * np.int64(multiplier)
    return round_shift_vec(prod, total_shift)


def make_tanh_lut() -> np.ndarray:
    """int32 Q(FRAC_BITS) tanh table over ``[-LUT_RANGE, LUT_RANGE]``."""
    xs = np.linspace(-LUT_RANGE, LUT_RANGE, LUT_SIZE)
    return to_fixed(np.tanh(xs)).astype(np.int64)


def make_sigmoid_lut() -> np.ndarray:
    """int32 Q(FRAC_BITS) sigmoid table over ``[-LUT_RANGE, LUT_RANGE]``."""
    xs = np.linspace(-LUT_RANGE, LUT_RANGE, LUT_SIZE)
    return to_fixed(1.0 / (1.0 + np.exp(-xs))).astype(np.int64)


def lut_index(x_qf: np.ndarray | int) -> np.ndarray:
    """Map Q(FRAC_BITS) inputs to LUT indices with the exact integer formula used in C."""
    range_qf = LUT_RANGE * ONE_Q
    x = np.clip(np.asarray(x_qf, dtype=np.int64), -range_qf, range_qf)
    # index = (x + range) * (LUT_SIZE - 1) / (2 * range)
    return ((x + range_qf) * (LUT_SIZE - 1)) // (2 * range_qf)


def lut_lookup(lut: np.ndarray, x_qf: np.ndarray | int) -> np.ndarray:
    idx = lut_index(x_qf)
    return lut[idx]

"""Tests for fixed-point requantization and LUTs (no torch needed)."""

from __future__ import annotations

import numpy as np

from signalmint.quantize.fixedpoint import (
    FRAC_BITS,
    from_fixed,
    lut_index,
    make_sigmoid_lut,
    make_tanh_lut,
    quantize_multiplier,
    requantize,
    requantize_vec,
    to_fixed,
)


def test_quantize_multiplier_reconstructs_scale() -> None:
    for scale in [0.001, 0.03, 0.5, 0.9999, 1.5, 7.3, 123.0]:
        m, shift = quantize_multiplier(scale)
        approx = m / (2.0**shift)
        assert abs(approx - scale) / scale < 1e-6


def test_requantize_matches_float_rounding() -> None:
    rng = np.random.default_rng(0)
    scale = 0.037
    m, shift = quantize_multiplier(scale)
    for _ in range(200):
        acc = int(rng.integers(-5_000_000, 5_000_000))
        got = requantize(acc, m, shift)
        expected = round(acc * scale)
        # Fixed-point uses round-half-away; allow a 1-LSB tolerance vs Python round
        assert abs(got - expected) <= 1


def test_requantize_vec_matches_scalar() -> None:
    m, shift = quantize_multiplier(0.25)
    accs = np.array([-100, -1, 0, 1, 100, 12345, -6789], dtype=np.int64)
    vec = requantize_vec(accs, m, shift)
    scal = [requantize(int(a), m, shift) for a in accs]
    assert list(vec) == scal


def test_fixed_roundtrip() -> None:
    x = np.array([-1.0, -0.25, 0.0, 0.5, 0.999])
    assert np.allclose(from_fixed(to_fixed(x)), x, atol=1.0 / (1 << FRAC_BITS))


def test_lut_monotonic_and_bounded() -> None:
    tanh_lut = make_tanh_lut()
    sig_lut = make_sigmoid_lut()
    one = 1 << FRAC_BITS
    assert tanh_lut.min() >= -one and tanh_lut.max() <= one
    assert sig_lut.min() >= 0 and sig_lut.max() <= one
    # tanh is odd-ish and increasing.
    assert np.all(np.diff(tanh_lut) >= 0)
    assert np.all(np.diff(sig_lut) >= 0)


def test_lut_index_in_bounds() -> None:
    xs = np.array([-10 << FRAC_BITS, 0, 10 << FRAC_BITS], dtype=np.int64)
    idx = lut_index(xs)
    assert idx.min() >= 0
    assert idx.max() <= make_tanh_lut().size - 1

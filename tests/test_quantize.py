"""Tests for amplitude quantization and framing."""

from __future__ import annotations

import numpy as np
import pytest

from signalmint.data.framing import frame_signal, num_frames
from signalmint.data.quantize import dequantize, mu_law_decode, mu_law_encode, quantize


def test_mu_law_roundtrip_is_near_identity() -> None:
    x = np.linspace(-1, 1, 501)
    back = mu_law_decode(mu_law_encode(x, 255.0), 255.0)
    assert np.max(np.abs(back - x)) < 1e-6


def test_quantize_range_and_dtype() -> None:
    x = np.random.default_rng(0).standard_normal(2000).astype(np.float32)
    sym, scale = quantize(x, num_bins=256, quantizer="mu_law")
    assert sym.dtype == np.int64
    assert sym.min() >= 0 and sym.max() <= 255
    assert scale > 0


def test_quantize_dequantize_bounded_error() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000).astype(np.float32)
    sym, scale = quantize(x, num_bins=256, quantizer="mu_law", normalize=True)
    recon = dequantize(sym, 256, "mu_law", scale=scale)
    # 256-level mu-law reconstruction should track the signal closely.
    rel = np.linalg.norm(recon - x) / np.linalg.norm(x)
    assert rel < 0.15


def test_quantize_rejects_bad_bins() -> None:
    with pytest.raises(ValueError):
        quantize(np.zeros(10), num_bins=1)


def test_num_frames_matches_frame_signal() -> None:
    x = np.arange(1000)
    assert num_frames(1000, 100, 50) == frame_signal(x, 100, 50).shape[0]


def test_frame_signal_content() -> None:
    x = np.arange(10)
    frames = frame_signal(x, 4, 3)
    assert frames.shape == (3, 4)
    assert list(frames[0]) == [0, 1, 2, 3]
    assert list(frames[1]) == [3, 4, 5, 6]
    assert list(frames[2]) == [6, 7, 8, 9]


def test_frame_signal_too_short() -> None:
    assert frame_signal(np.arange(3), 4).shape == (0, 4)

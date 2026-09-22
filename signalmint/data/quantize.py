"""Amplitude quantization for 1-D signals.

Turns a real-valued signal into an INT-friendly categorical alphabet of
``num_bins`` symbols and back. Two schemes are supported:

* ``mu_law``  -- companding that concentrates resolution near zero (good for
  signals whose interesting structure is low-amplitude, e.g. vibration).
* ``linear``  -- uniform bins.

All functions are pure NumPy so they run without torch and are trivially
portable to the C runtime in Phase 3.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mu_law_encode",
    "mu_law_decode",
    "quantize",
    "dequantize",
]


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mu_law_encode(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Companding transform. Expects ``x`` in [-1, 1], returns values in [-1, 1]."""
    x = _as_float(x)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def mu_law_decode(y: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Inverse of :func:`mu_law_encode`."""
    y = _as_float(y)
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu


def quantize(
    x: np.ndarray,
    num_bins: int,
    quantizer: str = "mu_law",
    mu: float = 255.0,
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """Quantize a real signal into integer symbols in ``[0, num_bins)``.

    Args:
        x: 1-D real signal.
        num_bins: Number of quantization levels (the categorical alphabet size).
        quantizer: ``"mu_law"`` or ``"linear"``.
        mu: Companding parameter for mu-law.
        normalize: If True, scale by peak absolute value into [-1, 1] first.

    Returns:
        ``(symbols, scale)`` where ``symbols`` is an ``int64`` array in
        ``[0, num_bins)`` and ``scale`` is the peak-abs used for normalization
        (1.0 when ``normalize`` is False), needed to reconstruct amplitude.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    x = _as_float(x)

    scale = 1.0
    if normalize:
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        scale = peak if peak > 0.0 else 1.0
        x = x / scale
    x = np.clip(x, -1.0, 1.0)

    if quantizer == "mu_law":
        x = mu_law_encode(x, mu)
    elif quantizer != "linear":
        raise ValueError(f"unknown quantizer: {quantizer!r}")

    # Map [-1, 1] -> [0, num_bins - 1] with uniform bins.
    idx = np.floor((x + 1.0) * 0.5 * num_bins).astype(np.int64)
    idx = np.clip(idx, 0, num_bins - 1)
    return idx, scale


def dequantize(
    symbols: np.ndarray,
    num_bins: int,
    quantizer: str = "mu_law",
    mu: float = 255.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Approximately reconstruct amplitudes from integer symbols.

    Reconstruction is lossy: each symbol maps to its bin centre.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    symbols = np.asarray(symbols, dtype=np.int64)
    # Bin centre in [-1, 1].
    y = (symbols.astype(np.float64) + 0.5) / num_bins * 2.0 - 1.0
    if quantizer == "mu_law":
        y = mu_law_decode(y, mu)
    elif quantizer != "linear":
        raise ValueError(f"unknown quantizer: {quantizer!r}")
    return y * scale

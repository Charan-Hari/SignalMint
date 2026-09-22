"""Frame extraction: slice a long 1-D signal into fixed-length training frames."""

from __future__ import annotations

import numpy as np

__all__ = ["frame_signal", "num_frames"]


def num_frames(length: int, frame_length: int, hop_length: int) -> int:
    """Number of complete frames obtainable from a signal of ``length`` samples."""
    if length < frame_length:
        return 0
    return 1 + (length - frame_length) // hop_length


def frame_signal(
    x: np.ndarray,
    frame_length: int,
    hop_length: int | None = None,
) -> np.ndarray:
    """Slice ``x`` into overlapping frames.

    Args:
        x: 1-D signal.
        frame_length: Samples per frame.
        hop_length: Step between frames; defaults to ``frame_length``
            (non-overlapping).

    Returns:
        A ``(n_frames, frame_length)`` array. Uses a strided view where possible,
        returned as a contiguous copy for safety.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("frame_signal expects a 1-D array")
    if hop_length is None:
        hop_length = frame_length
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")

    n = num_frames(x.shape[0], frame_length, hop_length)
    if n == 0:
        return np.empty((0, frame_length), dtype=x.dtype)

    strides = (x.strides[0] * hop_length, x.strides[0])
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, frame_length), strides=strides, writeable=False
    )
    return np.ascontiguousarray(view)

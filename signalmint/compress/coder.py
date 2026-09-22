"""Model-driven neural codec: encode/decode symbol frames with the AR model.

Because the model is strictly causal, the distribution used to code the symbol
at position ``t`` depends only on symbols ``< t``. The encoder (which has the
whole frame) and the decoder (which reconstructs it left-to-right) therefore
compute *identical* distributions, so the round-trip is exact.

Position 0 has no predictor and is coded under a uniform prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.wavenet import WaveNetLite
from .arithmetic import ArithmeticDecoder, ArithmeticEncoder
from .freqs import cumulative, find_symbol, probs_to_freqs

__all__ = ["CompressionResult", "encode_frames", "decode_frame", "frame_distributions"]


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class CompressionResult:
    """Outcome of compressing a set of frames.

    Attributes:
        data: The compressed byte stream (all frames, one arithmetic stream).
        num_symbols: Total symbols coded.
        num_bits: Length of ``data`` in bits.
        bits_per_sample: ``num_bits / num_symbols``.
    """

    data: bytes
    num_symbols: int
    num_bits: int
    bits_per_sample: float


def frame_distributions(model: WaveNetLite, frame: np.ndarray) -> np.ndarray:
    """Per-position coding distributions for a single frame ``(T,)``.

    Returns ``(T, num_bins)``: row 0 is uniform; row ``t`` (t>=1) is the model's
    prediction of symbol ``t`` given symbols ``< t``.
    """
    import torch

    num_bins = model.num_bins
    T = frame.shape[0]
    dist = np.empty((T, num_bins), dtype=np.float64)
    dist[0] = 1.0 / num_bins
    if T > 1:
        with torch.no_grad():
            x = torch.from_numpy(frame.astype(np.int64)).unsqueeze(0)
            logits = model.forward(x)[0].cpu().numpy().astype(np.float64)
        # logits[t] predicts position t+1 -> distributions for positions 1..T-1.
        dist[1:] = _softmax_np(logits[:-1])
    return dist


def encode_frames(
    model: WaveNetLite, frames: np.ndarray, total_bits: int = 16
) -> CompressionResult:
    """Compress ``(N, T)`` integer frames into a single arithmetic stream."""
    import torch

    if frames.ndim != 2:
        raise ValueError("frames must be 2-D (N, T)")
    n, T = frames.shape
    num_bins = model.num_bins

    # Batched teacher-forced forward for all frames at once.
    with torch.no_grad():
        x = torch.from_numpy(frames.astype(np.int64))
        logits = model.forward(x).cpu().numpy().astype(np.float64)  # (N, T, V)

    enc = ArithmeticEncoder()
    num_symbols = 0
    for i in range(n):
        frame = frames[i]
        # position 0: uniform
        uniform = np.full(num_bins, 1.0 / num_bins)
        for t in range(T):
            if t == 0:
                probs = uniform
            else:
                probs = _softmax_np(logits[i, t - 1])
            freqs = probs_to_freqs(probs, total_bits)
            cum = cumulative(freqs)
            s = int(frame[t])
            enc.encode(int(cum[s]), int(cum[s + 1]), int(cum[-1]))
            num_symbols += 1

    data = enc.finish()
    num_bits = len(data) * 8
    return CompressionResult(
        data=data,
        num_symbols=num_symbols,
        num_bits=num_bits,
        bits_per_sample=num_bits / max(num_symbols, 1),
    )


def decode_frame(
    model: WaveNetLite, data: bytes, length: int, total_bits: int = 16
) -> np.ndarray:
    """Decode a *single* frame of ``length`` symbols from ``data``.

    Used to verify round-trip correctness. To stay bit-exact with the encoder,
    the model is always evaluated on a **fixed length-``length`` buffer** (decoded
    prefix followed by zeros). Because the model is causal, the zero-filled future
    positions cannot influence the current prediction, yet keeping the tensor
    shape identical to the encoder's forward pass guarantees identical
    floating-point logits -- without which the arithmetic coder would desync.
    """
    import torch

    num_bins = model.num_bins
    dec = ArithmeticDecoder(data)
    buffer = np.zeros(length, dtype=np.int64)
    uniform = np.full(num_bins, 1.0 / num_bins)
    for t in range(length):
        if t == 0:
            probs = uniform
        else:
            with torch.no_grad():
                x = torch.from_numpy(buffer).unsqueeze(0)  # (1, length)
                logits = model.forward(x)[0].cpu().numpy().astype(np.float64)
            probs = _softmax_np(logits[t - 1])
        freqs = probs_to_freqs(probs, total_bits)
        cum = cumulative(freqs)
        target = dec.decode_target(int(cum[-1]))
        s = find_symbol(cum, target)
        dec.update(int(cum[s]), int(cum[s + 1]), int(cum[-1]))
        buffer[t] = s
    return buffer

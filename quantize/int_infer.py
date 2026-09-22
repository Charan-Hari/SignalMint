"""Integer inference reference for a :class:`QuantModel`.

Pure-integer NumPy that computes the same int32 logits the C runtime produces.
This is the *reference*: parity means the C output equals this exactly. It also
lets us measure quantization quality (INT8 vs float argmax / NLL agreement)
without leaving Python.

All activations are Q(FRAC_BITS) fixed-point int64; convolution accumulators are
int64; requantization and the gated-activation LUTs use the shared primitives in
:mod:`signalmint.quantize.fixedpoint`.
"""

from __future__ import annotations

import numpy as np

from .fixedpoint import (
    FRAC_BITS,
    lut_lookup,
    make_sigmoid_lut,
    make_tanh_lut,
    requantize_vec,
    round_shift_vec,
)
from .quantizer import QuantConv, QuantModel

__all__ = ["IntegerRuntime"]

_HALF_F = 1 << (FRAC_BITS - 1)


def _causal_conv(x_qf: np.ndarray, conv: QuantConv, kernel: int) -> np.ndarray:
    """Dilated causal conv on Q.F input ``(Cin, T)`` -> Q.F output ``(Cout, T)``."""
    cin, T = x_qf.shape
    w = conv.weight.astype(np.int64)  # (Cout, Cin, K)
    pad = (kernel - 1) * conv.dilation
    xp = np.zeros((cin, T + pad), dtype=np.int64)
    xp[:, pad:] = x_qf
    acc = np.zeros((w.shape[0], T), dtype=np.int64)
    for k in range(kernel):
        seg = xp[:, k * conv.dilation : k * conv.dilation + T]  # (Cin, T)
        acc += w[:, :, k] @ seg  # (Cout, T)
    out = requantize_vec(acc, conv.multiplier, conv.shift)
    return out + conv.bias_qf[:, None]


class IntegerRuntime:
    """Executes a :class:`QuantModel` in pure integer arithmetic."""

    def __init__(self, qm: QuantModel):
        self.qm = qm
        self.tanh_lut = make_tanh_lut()
        self.sigmoid_lut = make_sigmoid_lut()

    def logits(self, symbols: np.ndarray) -> np.ndarray:
        """Return int32 Q.F logits ``(T, num_bins)`` for a 1-D symbol frame."""
        qm = self.qm
        symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
        C = qm.residual_channels

        # Embedding lookup -> (C, T) Q.F
        x = qm.embed_qf[symbols].T.astype(np.int64)  # (C, T)

        skip_total = np.zeros((qm.skip_channels, x.shape[1]), dtype=np.int64)
        for blk in qm.blocks:
            h = _causal_conv(x, blk.conv, qm.kernel_size)  # (2C, T)
            f = h[:C]
            g = h[C:]
            tanh_f = lut_lookup(self.tanh_lut, f)
            sig_g = lut_lookup(self.sigmoid_lut, g)
            prod = tanh_f * sig_g  # Q.2F
            z = round_shift_vec(prod, FRAC_BITS)  # Q.F, (C, T)

            residual = _causal_conv(z, blk.residual, 1) + x
            skip = _causal_conv(z, blk.skip, 1)
            skip_total = skip_total + skip
            x = residual

        h = np.maximum(skip_total, 0)  # ReLU
        h = _causal_conv(h, qm.head1, 1)
        h = np.maximum(h, 0)
        logits = _causal_conv(h, qm.head2, 1)  # (num_bins, T)
        return logits.T.astype(np.int64)  # (T, num_bins)

    def predict_argmax(self, symbols: np.ndarray) -> np.ndarray:
        """Argmax next-symbol prediction at each position ``(T,)``."""
        return np.argmax(self.logits(symbols), axis=1).astype(np.int64)

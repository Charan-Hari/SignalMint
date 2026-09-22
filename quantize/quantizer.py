"""Post-training quantization of a trained WaveNetLite to integer parameters.

Produces a :class:`QuantModel` -- INT8 convolution weights with per-tensor
requant multipliers, Q(FRAC_BITS) biases, and a Q(FRAC_BITS) embedding table --
that both the Python integer reference (:mod:`signalmint.quantize.int_infer`) and
the exported C runtime consume. The float model is untouched; this is a separate
integer view for edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model.wavenet import WaveNetLite
from .fixedpoint import quantize_multiplier, to_fixed

__all__ = ["QuantConv", "QuantModel", "quantize_model"]


@dataclass
class QuantConv:
    """A quantized convolution: INT8 weights + requant multiplier + Q.F bias.

    ``weight`` has shape ``(out_ch, in_ch, kernel)`` (int8). Output is computed as
    ``requantize(sum(weight * x_qF), multiplier, shift) + bias_qf``.
    """

    weight: np.ndarray  # int8 (out, in, k)
    bias_qf: np.ndarray  # int32 Q.F (out,)
    multiplier: int
    shift: int
    dilation: int = 1


@dataclass
class QuantBlock:
    conv: QuantConv          # (2C, C, k) gated conv
    residual: QuantConv      # (C, C, 1)
    skip: QuantConv          # (skipC, C, 1)


@dataclass
class QuantModel:
    num_bins: int
    residual_channels: int
    skip_channels: int
    kernel_size: int
    dilation_cycle: int
    embed_qf: np.ndarray                      # int32 Q.F (num_bins, C)
    blocks: list[QuantBlock] = field(default_factory=list)
    head1: QuantConv = None                   # (skipC, skipC, 1)
    head2: QuantConv = None                   # (num_bins, skipC, 1)


def _quantize_conv(weight: np.ndarray, bias: np.ndarray, dilation: int = 1) -> QuantConv:
    """Quantize a conv weight to per-tensor symmetric INT8; bias to Q.F."""
    w = np.asarray(weight, dtype=np.float64)  # (out, in, k)
    max_abs = float(np.max(np.abs(w))) if w.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    w_int8 = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    mult, shift = quantize_multiplier(scale)
    bias_qf = (
        to_fixed(bias).astype(np.int64)
        if bias is not None
        else np.zeros(w.shape[0], dtype=np.int64)
    )
    return QuantConv(weight=w_int8, bias_qf=bias_qf, multiplier=mult, shift=shift, dilation=dilation)


def quantize_model(model: WaveNetLite) -> QuantModel:
    """Quantize a trained :class:`WaveNetLite` into a :class:`QuantModel`."""
    model = model.eval()
    cfg = model.config
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    embed = sd["embed.weight"]  # (num_bins, C)
    embed_qf = to_fixed(embed).astype(np.int64)

    qm = QuantModel(
        num_bins=cfg.num_bins,
        residual_channels=cfg.residual_channels,
        skip_channels=cfg.skip_channels,
        kernel_size=cfg.kernel_size,
        dilation_cycle=cfg.dilation_cycle,
        embed_qf=embed_qf,
    )

    for i in range(cfg.num_layers):
        dil = 2 ** (i % cfg.dilation_cycle)
        conv = _quantize_conv(
            sd[f"blocks.{i}.conv.conv.weight"], sd[f"blocks.{i}.conv.conv.bias"], dilation=dil
        )
        res = _quantize_conv(
            sd[f"blocks.{i}.residual_proj.weight"], sd[f"blocks.{i}.residual_proj.bias"]
        )
        skip = _quantize_conv(
            sd[f"blocks.{i}.skip_proj.weight"], sd[f"blocks.{i}.skip_proj.bias"]
        )
        qm.blocks.append(QuantBlock(conv=conv, residual=res, skip=skip))

    # head = [ReLU, Conv1d(idx 1), ReLU, Conv1d(idx 3)]
    qm.head1 = _quantize_conv(sd["head.1.weight"], sd["head.1.bias"])
    qm.head2 = _quantize_conv(sd["head.3.weight"], sd["head.3.bias"])
    return qm

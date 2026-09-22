"""Static resource-footprint analysis of a :class:`QuantModel`.

Reports the numbers that decide whether the model fits the 1-8 MB / sub-100 mW
target tier:

* **ROM/flash**: INT8 weights + Q.F biases + embedding table + activation LUTs.
* **Streaming RAM**: peak activation memory for a *streaming, memory-bounded*
  implementation -- bounded by the dilated-conv ring buffers and independent of
  sequence length (the whole point of the causal design).
* **MACs/sample**: multiply-accumulates per generated sample, used for latency
  and energy modeling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fixedpoint import LUT_SIZE
from .quantizer import QuantConv, QuantModel

__all__ = ["Footprint", "analyze_footprint"]


def _conv_rom_bytes(c: QuantConv) -> int:
    # INT8 weights + INT32 biases.
    return int(c.weight.size) * 1 + int(c.bias_qf.size) * 4


def _conv_macs(c: QuantConv) -> int:
    out_ch, in_ch, kernel = c.weight.shape
    return out_ch * in_ch * kernel


@dataclass
class Footprint:
    rom_bytes: int
    streaming_ram_bytes: int
    macs_per_sample: int
    num_parameters: int
    receptive_field: int

    @property
    def rom_kb(self) -> float:
        return self.rom_bytes / 1024

    @property
    def streaming_ram_kb(self) -> float:
        return self.streaming_ram_bytes / 1024


def analyze_footprint(qm: QuantModel, activation_bytes: int = 4) -> Footprint:
    """Compute ROM, streaming RAM and MAC counts for ``qm``.

    Args:
        qm: The quantized model.
        activation_bytes: Bytes per streaming activation element (4 = INT32).
    """
    C = qm.residual_channels
    S = qm.skip_channels
    K = qm.kernel_size

    # --- ROM ---------------------------------------------------------------
    rom = 0
    rom += int(qm.embed_qf.size) * 4                 # embedding Q.F table
    rom += 2 * LUT_SIZE * 4                           # tanh + sigmoid LUTs
    macs = 0
    receptive = 1
    streaming_ram = 0
    for i, blk in enumerate(qm.blocks):
        rom += _conv_rom_bytes(blk.conv)
        rom += _conv_rom_bytes(blk.residual)
        rom += _conv_rom_bytes(blk.skip)
        macs += _conv_macs(blk.conv) + _conv_macs(blk.residual) + _conv_macs(blk.skip)
        dilation = blk.conv.dilation
        receptive += (K - 1) * dilation
        # Streaming ring buffer for this dilated layer: (K-1)*dilation past
        # residual-channel activations must be retained.
        streaming_ram += C * (K - 1) * dilation * activation_bytes
    rom += _conv_rom_bytes(qm.head1) + _conv_rom_bytes(qm.head2)
    macs += _conv_macs(qm.head1) + _conv_macs(qm.head2)

    # Working vectors held per step: x (C), gate (2C), z (C), skip accumulator (S),
    # head buffer (S), logits (num_bins).
    streaming_ram += (C + 2 * C + C + S + S + qm.num_bins) * activation_bytes

    num_params = 0
    for blk in qm.blocks:
        num_params += blk.conv.weight.size + blk.residual.weight.size + blk.skip.weight.size
    num_params += qm.head1.weight.size + qm.head2.weight.size + qm.embed_qf.size

    return Footprint(
        rom_bytes=int(rom),
        streaming_ram_bytes=int(streaming_ram),
        macs_per_sample=int(macs),
        num_parameters=int(num_params),
        receptive_field=int(receptive),
    )

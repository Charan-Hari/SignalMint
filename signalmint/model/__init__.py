"""Streaming autoregressive model (WaveNet-lite dilated causal convolutions).

Phase 1: the categorical next-symbol model whose per-sample probabilities drive
both the anomaly and compression heads.
"""

from __future__ import annotations

from .wavenet import CausalConv1d, ResidualBlock, WaveNetLite

__all__ = ["WaveNetLite", "CausalConv1d", "ResidualBlock"]

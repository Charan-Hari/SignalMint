"""SignalMint: tiny explicit-likelihood generative models for edge signals.

One autoregressive model provides two products from the same per-sample
probabilities:

* label-free anomaly detection (negative log-likelihood scoring), and
* neural compression (probabilities drive an entropy coder).

The public surface is intentionally small at Phase 0; later phases populate the
``data``, ``model``, ``anomaly``, ``compress`` and ``quantize`` subpackages.
"""

from __future__ import annotations

from .config import (
    AnomalyConfig,
    CompressConfig,
    DataConfig,
    ModelConfig,
    QuantConfig,
    SignalMintConfig,
    TrainConfig,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SignalMintConfig",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "AnomalyConfig",
    "CompressConfig",
    "QuantConfig",
]

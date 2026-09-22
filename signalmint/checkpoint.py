"""Checkpoint persistence for trained models plus their full config.

A checkpoint bundles the model weights and the :class:`SignalMintConfig` so a
model can be reloaded, evaluated, compressed, or exported to the C runtime
without guessing hyper-parameters.
"""

from __future__ import annotations

from pathlib import Path

from .config import ModelConfig, SignalMintConfig
from .model.wavenet import WaveNetLite

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(model: WaveNetLite, config: SignalMintConfig, path: str | Path) -> Path:
    """Save ``model`` weights and ``config`` to ``path`` (torch archive)."""
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": config.to_dict()},
        p,
    )
    return p


def load_checkpoint(path: str | Path) -> tuple[WaveNetLite, SignalMintConfig]:
    """Reload a ``(model, config)`` pair saved by :func:`save_checkpoint`."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    config = SignalMintConfig.from_dict(payload["config"])
    model = WaveNetLite(ModelConfig(**payload["config"]["model"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config

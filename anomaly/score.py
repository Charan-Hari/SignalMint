"""Likelihood-based anomaly scoring using the trained AR model."""

from __future__ import annotations

import numpy as np

from ..data.dataset import FrameSet
from ..model.wavenet import WaveNetLite

__all__ = ["frame_nll_scores", "calibrate_threshold"]


def frame_nll_scores(
    model: WaveNetLite, frames: FrameSet, batch_size: int = 64, device=None
) -> np.ndarray:
    """Mean per-sample NLL (nats) for each frame -- the anomaly score.

    Higher NLL means the frame is less probable under the healthy-data model,
    i.e. more anomalous.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    symbols = torch.from_numpy(frames.symbols.astype(np.int64))
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, symbols.shape[0], batch_size):
            batch = symbols[start : start + batch_size].to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            per = model.nll(inputs, targets, reduction="none")  # (B, T-1)
            scores.append(per.mean(dim=1).cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(scores).astype(np.float64)


def calibrate_threshold(normal_scores: np.ndarray, percentile: float = 99.0) -> float:
    """Set a detection threshold at a percentile of the normal-score distribution."""
    normal_scores = np.asarray(normal_scores, dtype=np.float64)
    if normal_scores.size == 0:
        return float("nan")
    return float(np.percentile(normal_scores, percentile))

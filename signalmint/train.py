"""Reference training loop for the WaveNet-lite next-symbol model.

Trains by teacher forcing with categorical cross-entropy (which *is* the
negative log-likelihood) and reports per-sample NLL in both nats and
bits-per-sample -- the latter being the theoretical lower bound on the
compression head's output, tying training directly to the product metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import ModelConfig, TrainConfig
from .data.dataset import FrameSet
from .model.wavenet import WaveNetLite

__all__ = ["TrainResult", "resolve_device", "train_model", "evaluate_nll"]

_LN2 = math.log(2.0)


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        model: The trained :class:`WaveNetLite`.
        train_nll: Per-epoch training NLL (nats/sample).
        val_nll: Per-epoch validation NLL (nats/sample); empty if no validation.
        best_val_nll: Best validation NLL observed (or final train NLL).
        bits_per_sample: ``best_val_nll / ln(2)`` -- the compression lower bound.
    """

    model: WaveNetLite
    train_nll: list[float] = field(default_factory=list)
    val_nll: list[float] = field(default_factory=list)
    best_val_nll: float = float("inf")
    bits_per_sample: float = float("inf")


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return perm[n_val:], perm[:n_val]


def evaluate_nll(model: WaveNetLite, frames: FrameSet, batch_size: int, device) -> float:
    """Mean per-sample NLL (nats) over ``frames`` under ``model``."""
    import torch

    if len(frames) == 0:
        return float("nan")
    model.eval()
    symbols = torch.from_numpy(frames.symbols.astype(np.int64))
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, symbols.shape[0], batch_size):
            batch = symbols[start : start + batch_size].to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            loss = model.nll(inputs, targets, reduction="sum")
            total += float(loss.item())
            count += targets.numel()
    return total / max(count, 1)


def train_model(
    frames: FrameSet,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> TrainResult:
    """Train a :class:`WaveNetLite` on healthy frames and report NLL.

    The model is trained on *normal* data only (label 0) so that, at inference,
    faulty signals surface as high-NLL anomalies. Callers should pass a FrameSet
    already filtered to the normal class for the anomaly use case; for pure
    density estimation any frames are acceptable.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(train_config.seed)
    device = resolve_device(train_config.device)

    model = WaveNetLite(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    symbols = torch.from_numpy(frames.symbols.astype(np.int64))
    train_idx, val_idx = _split_indices(
        symbols.shape[0], train_config.val_fraction, train_config.seed
    )
    train_frames = FrameSet(
        frames.symbols[train_idx], frames.labels[train_idx],
        [frames.names[i] for i in train_idx],
    )
    val_frames = (
        FrameSet(
            frames.symbols[val_idx], frames.labels[val_idx],
            [frames.names[i] for i in val_idx],
        )
        if len(val_idx) > 0
        else FrameSet(np.empty((0, frames.symbols.shape[1]), np.int64), np.empty((0,), np.int64), [])
    )

    train_symbols = torch.from_numpy(train_frames.symbols.astype(np.int64))
    loader = DataLoader(
        TensorDataset(train_symbols),
        batch_size=train_config.batch_size,
        shuffle=True,
    )

    result = TrainResult(model=model)
    for _epoch in range(train_config.epochs):
        model.train()
        epoch_total, epoch_count = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad()
            loss = model.nll(inputs, targets, reduction="mean")
            loss.backward()
            if train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            epoch_total += float(loss.item()) * targets.numel()
            epoch_count += targets.numel()

        train_nll = epoch_total / max(epoch_count, 1)
        result.train_nll.append(train_nll)
        if len(val_frames) > 0:
            val_nll = evaluate_nll(model, val_frames, train_config.batch_size, device)
            result.val_nll.append(val_nll)

    reference = result.val_nll[-1] if result.val_nll else result.train_nll[-1]
    result.best_val_nll = min(result.val_nll) if result.val_nll else reference
    result.bits_per_sample = result.best_val_nll / _LN2
    return result

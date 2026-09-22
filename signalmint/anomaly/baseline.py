"""A small 1-D convolutional autoencoder baseline for anomaly detection.

This is the standard "learn to reconstruct normal data; flag high reconstruction
error" approach that SignalMint's likelihood model is measured against. It shares
the same training discipline (normal data only) for a fair comparison.
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import FrameSet

__all__ = ["ConvAutoencoder", "train_autoencoder", "autoencoder_scores"]


def _build(num_bins: int):
    import torch.nn as nn

    class ConvAutoencoder(nn.Module):
        """Strided conv encoder / transposed-conv decoder over [0, 1] frames."""

        def __init__(self, latent_channels: int = 16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(8, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(16, latent_channels, 5, stride=2, padding=2),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(latent_channels, 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(8, 1, 4, stride=2, padding=1),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return ConvAutoencoder


# Exposed for typing/imports; the real class is built lazily to avoid importing
# torch at module import time.
ConvAutoencoder = None  # type: ignore[assignment]


def _frames_to_float(frames: FrameSet, num_bins: int):
    import torch

    x = frames.symbols.astype(np.float32) / max(num_bins - 1, 1)
    return torch.from_numpy(x).unsqueeze(1)  # (N, 1, T)


def train_autoencoder(
    normal_frames: FrameSet,
    num_bins: int,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
):
    """Train the autoencoder on healthy frames; return the fitted module."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    cls = _build(num_bins)
    model = cls()
    x = _frames_to_float(normal_frames, num_bins)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
    model.eval()
    return model


def autoencoder_scores(model, frames: FrameSet, num_bins: int, batch_size: int = 64):
    """Per-frame reconstruction MSE -- the baseline anomaly score."""
    import torch

    x = _frames_to_float(frames, num_bins)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size]
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            scores.append(mse.cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(scores).astype(np.float64)

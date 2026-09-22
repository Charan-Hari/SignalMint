"""Tests for the WaveNet-lite model and the training loop.

Marked to skip cleanly if torch is unavailable so the rest of the suite still
runs on a bare (core-only) install.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signalmint.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from signalmint.data.cwru import get_synthetic_records  # noqa: E402
from signalmint.data.dataset import records_to_frames  # noqa: E402
from signalmint.model.wavenet import WaveNetLite  # noqa: E402
from signalmint.train import train_model  # noqa: E402


def _tiny_model() -> WaveNetLite:
    cfg = ModelConfig(
        num_bins=16, residual_channels=8, skip_channels=8, num_layers=4, dilation_cycle=4
    )
    return WaveNetLite(cfg)


def test_forward_shape() -> None:
    model = _tiny_model()
    x = torch.randint(0, 16, (2, 32))
    logits = model.forward(x)
    assert logits.shape == (2, 32, 16)


def test_model_is_causal() -> None:
    """Changing a future input must not change earlier outputs."""
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randint(0, 16, (1, 24))
    with torch.no_grad():
        base = model.forward(x)
        x2 = x.clone()
        x2[0, 20] = (x2[0, 20] + 1) % 16  # perturb a late timestep
        perturbed = model.forward(x2)
    # Outputs at t < 20 must be identical (no future leakage).
    assert torch.allclose(base[0, :20], perturbed[0, :20], atol=1e-5)


def test_nll_reductions_consistent() -> None:
    model = _tiny_model()
    x = torch.randint(0, 16, (3, 20))
    inp, tgt = x[:, :-1], x[:, 1:]
    per = model.nll(inp, tgt, reduction="none")
    assert per.shape == tgt.shape
    assert torch.allclose(per.mean(), model.nll(inp, tgt, reduction="mean"), atol=1e-6)


def test_training_reduces_nll_below_uniform() -> None:
    """A short run on synthetic data should beat the uniform-code baseline."""
    records = get_synthetic_records(n_normal=3, n_fault=0, n_samples=6000, seed=0)
    data_cfg = DataConfig(frame_length=256, num_bins=16, hop_length=128)
    frames = records_to_frames(records, data_cfg)
    model_cfg = ModelConfig(
        num_bins=16, residual_channels=8, skip_channels=8, num_layers=4, dilation_cycle=4
    )
    train_cfg = TrainConfig(batch_size=16, epochs=3, learning_rate=3e-3, val_fraction=0.2, seed=0)
    result = train_model(frames, model_cfg, train_cfg)
    uniform_bits = np.log2(16)
    # Learned model must compress better than a 4-bit uniform code.
    assert result.bits_per_sample < uniform_bits
    assert result.train_nll[-1] < result.train_nll[0]

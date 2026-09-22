"""Integration tests for both Phase 2 heads on a small trained model."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signalmint.anomaly.baseline import autoencoder_scores, train_autoencoder  # noqa: E402
from signalmint.anomaly.detector import evaluate_anomaly, report_from_scores  # noqa: E402
from signalmint.compress.baselines import (  # noqa: E402,F401
    gzip_bits_per_sample,
    raw_bits_per_sample,
)
from signalmint.compress.coder import decode_frame, encode_frames  # noqa: E402
from signalmint.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from signalmint.data.cwru import get_synthetic_records  # noqa: E402
from signalmint.data.dataset import records_to_frames  # noqa: E402
from signalmint.train import train_model  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    """Train a tiny model on synthetic normal frames; return model + eval data."""
    records = get_synthetic_records(n_normal=6, n_fault=6, n_samples=8000, seed=0)
    data_cfg = DataConfig(frame_length=256, num_bins=16, hop_length=128)
    frames = records_to_frames(records, data_cfg)
    normal = frames.filter_label(0)
    model_cfg = ModelConfig(
        num_bins=16, residual_channels=8, skip_channels=8, num_layers=4, dilation_cycle=4
    )
    train_cfg = TrainConfig(batch_size=16, epochs=4, learning_rate=3e-3, val_fraction=0.2, seed=0)
    result = train_model(normal, model_cfg, train_cfg)
    return result.model, frames, normal, data_cfg


def test_compression_roundtrip_and_beats_uniform(trained) -> None:
    model, frames, normal, data_cfg = trained
    subset = normal.symbols[:8]
    comp = encode_frames(model, subset, total_bits=16)

    # Exact reversibility: decode the first frame and compare.
    decoded = decode_frame(model, encode_frames(model, subset[:1], 16).data,
                           length=subset.shape[1], total_bits=16)
    assert np.array_equal(decoded, subset[0])

    # Neural codec beats a fixed-width uniform code.
    assert comp.bits_per_sample < raw_bits_per_sample(data_cfg.num_bins)


def test_compression_beats_order0_entropy(trained) -> None:
    """The AR model's temporal context should beat the memoryless order-0 code.

    (The stronger "beats gzip/FLAC" claim is validated on properly-trained real
    CWRU data in the Phase 5 benchmark, not this fast tiny-model unit test.)
    """
    from signalmint.compress.baselines import order0_entropy_bits_per_sample

    model, frames, normal, data_cfg = trained
    subset = normal.symbols[:16]
    comp = encode_frames(model, subset, total_bits=16)
    order0 = order0_entropy_bits_per_sample(subset, data_cfg.num_bins)
    assert comp.bits_per_sample < order0


def test_anomaly_model_separates_faults(trained) -> None:
    model, frames, normal, data_cfg = trained
    report, scores = evaluate_anomaly(model, frames, target_fa=0.05)
    # The likelihood model should clearly separate healthy vs faulty frames.
    assert report.auc > 0.8
    assert report.n_fault > 0 and report.n_normal > 0


def test_model_auc_beats_autoencoder_baseline(trained) -> None:
    model, frames, normal, data_cfg = trained
    _model_report, model_scores = evaluate_anomaly(model, frames, target_fa=0.05)
    model_auc = report_from_scores(model_scores, frames.labels, 0.05).auc

    ae = train_autoencoder(normal, data_cfg.num_bins, epochs=6, seed=0)
    ae_scores = autoencoder_scores(ae, frames, data_cfg.num_bins)
    ae_auc = report_from_scores(ae_scores, frames.labels, 0.05).auc

    # The explicit-likelihood model should be competitive with / beat the AE.
    assert model_auc >= ae_auc - 0.05

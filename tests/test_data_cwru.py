"""Tests for the CWRU adapter's synthetic fallback and frame assembly."""

from __future__ import annotations

import numpy as np

from signalmint.config import DataConfig
from signalmint.data.cwru import get_synthetic_records, synthetic_bearing
from signalmint.data.dataset import records_to_frames


def test_synthetic_bearing_shapes_and_determinism() -> None:
    a = synthetic_bearing(5000, fault=False, seed=7)
    b = synthetic_bearing(5000, fault=False, seed=7)
    assert a.shape == (5000,)
    assert np.array_equal(a, b)


def test_fault_signal_has_higher_kurtosis() -> None:
    # Bearing-fault impulses make the amplitude distribution heavier-tailed.
    normal = synthetic_bearing(20000, fault=False, seed=1)
    fault = synthetic_bearing(20000, fault=True, seed=1)

    def kurtosis(x: np.ndarray) -> float:
        x = (x - x.mean()) / (x.std() + 1e-9)
        return float(np.mean(x**4))

    assert kurtosis(fault) > kurtosis(normal)


def test_records_to_frames_labels_align() -> None:
    records = get_synthetic_records(n_normal=2, n_fault=2, n_samples=8000, seed=0)
    cfg = DataConfig(frame_length=256, num_bins=64)
    fs = records_to_frames(records, cfg)
    assert fs.symbols.shape[1] == 256
    assert set(np.unique(fs.labels)).issubset({0, 1})
    assert len(fs.names) == len(fs)
    # Filtering to normal frames drops all fault frames.
    normal = fs.filter_label(0)
    assert set(np.unique(normal.labels)) == {0}
    assert len(normal) < len(fs)


def test_symbols_within_alphabet() -> None:
    records = get_synthetic_records(n_normal=1, n_fault=1, n_samples=4000, seed=3)
    cfg = DataConfig(frame_length=128, num_bins=32)
    fs = records_to_frames(records, cfg)
    assert fs.symbols.min() >= 0
    assert fs.symbols.max() < 32

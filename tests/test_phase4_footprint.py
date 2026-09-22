"""Tests for the edge footprint analyzer (Phase 4)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from signalmint.config import ModelConfig  # noqa: E402
from signalmint.model.wavenet import WaveNetLite  # noqa: E402
from signalmint.quantize.footprint import analyze_footprint  # noqa: E402
from signalmint.quantize.quantizer import quantize_model  # noqa: E402


def test_footprint_positive_and_bounded() -> None:
    cfg = ModelConfig(
        num_bins=16, residual_channels=8, skip_channels=8, num_layers=4, dilation_cycle=4
    )
    qm = quantize_model(WaveNetLite(cfg).eval())
    fp = analyze_footprint(qm)

    assert fp.rom_bytes > 0
    assert fp.streaming_ram_bytes > 0
    assert fp.macs_per_sample > 0
    # Tiny model must fit the target tier with room to spare.
    assert fp.rom_bytes < 8 * 1024 * 1024
    assert fp.streaming_ram_bytes < 1 * 1024 * 1024
    # Receptive field matches the float model's analytic value.
    assert fp.receptive_field == WaveNetLite(cfg).receptive_field


def test_macs_scale_with_channels() -> None:
    small = analyze_footprint(quantize_model(WaveNetLite(
        ModelConfig(num_bins=16, residual_channels=8, skip_channels=8, num_layers=4)
    ).eval()))
    big = analyze_footprint(quantize_model(WaveNetLite(
        ModelConfig(num_bins=16, residual_channels=16, skip_channels=16, num_layers=4)
    ).eval()))
    assert big.macs_per_sample > small.macs_per_sample
    assert big.rom_bytes > small.rom_bytes

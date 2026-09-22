"""Tests for the configuration dataclasses."""

from __future__ import annotations

import pytest

from signalmint.config import (
    DataConfig,
    ModelConfig,
    SignalMintConfig,
    default_config,
)


def test_default_config_roundtrips_json(tmp_path) -> None:
    cfg = default_config()
    path = cfg.to_json(tmp_path / "cfg.json")
    loaded = SignalMintConfig.from_json(path)
    assert loaded.to_dict() == cfg.to_dict()


def test_data_hop_defaults_to_frame_length() -> None:
    cfg = DataConfig(frame_length=512, hop_length=None)
    assert cfg.hop_length == 512


def test_bins_must_match_between_data_and_model() -> None:
    with pytest.raises(ValueError, match="num_bins"):
        SignalMintConfig(
            data=DataConfig(num_bins=64),
            model=ModelConfig(num_bins=256),
        )


def test_receptive_field_grows_with_dilation() -> None:
    # kernel_size=2, 8 layers, dilation resets every 8 -> 1 + sum(2^0..2^7)
    cfg = ModelConfig(kernel_size=2, num_layers=8, dilation_cycle=8)
    assert cfg.receptive_field == 1 + (2**8 - 1)


def test_invalid_quantizer_rejected() -> None:
    with pytest.raises(ValueError, match="quantizer"):
        DataConfig(quantizer="nope")


@pytest.mark.parametrize("percentile", [0.0, -1.0, 101.0])
def test_anomaly_threshold_percentile_bounds(percentile) -> None:
    from signalmint.config import AnomalyConfig

    with pytest.raises(ValueError):
        AnomalyConfig(threshold_percentile=percentile)

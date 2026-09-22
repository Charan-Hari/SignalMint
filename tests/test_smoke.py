"""Smoke tests for the Phase 0 scaffold.

These assert the package imports, the version is well-formed, and the public
config surface behaves. They deliberately avoid heavy dependencies (torch) so
the suite is green on a bare install.
"""

from __future__ import annotations

import signalmint


def test_version_is_semver_like() -> None:
    parts = signalmint.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_public_config_symbols_exported() -> None:
    for name in (
        "SignalMintConfig",
        "DataConfig",
        "ModelConfig",
        "TrainConfig",
        "AnomalyConfig",
        "CompressConfig",
        "QuantConfig",
    ):
        assert hasattr(signalmint, name), f"{name} missing from public API"

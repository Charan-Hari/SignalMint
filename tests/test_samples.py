"""Tests for the bundled sample signals (no torch needed)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_manifest_and_files_present() -> None:
    manifest = SAMPLES / "manifest.csv"
    assert manifest.exists(), "run scripts/make_samples.py"
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 5, "expect at least 5 sample signals"
    labels = {int(r["label"]) for r in rows}
    assert labels == {0, 1}, "samples must include healthy and fault"
    for r in rows:
        assert (SAMPLES / r["file"]).exists()


def test_sample_signals_load_and_are_1d() -> None:
    manifest = SAMPLES / "manifest.csv"
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        sig = np.loadtxt(SAMPLES / r["file"], delimiter=",", skiprows=1)
        assert sig.ndim == 1
        assert sig.size > 100
        assert np.isfinite(sig).all()

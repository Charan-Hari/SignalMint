"""Generate a small set of sample signals users can test SignalMint on.

Writes 8 single-column CSV vibration signals (plus a manifest) under ``samples/``
using the project's own synthetic bearing generator -- so they carry no
third-party data license. Healthy and several fault types are included so users
can immediately see label-free anomaly detection and compression at work via
``scripts/try_samples.py``.

Usage:
    python scripts/make_samples.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from signalmint.data.cwru import synthetic_bearing

SAMPLES_DIR = Path("samples")
N = 3200
SR = 12_000


def _blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Blend two signals (w of b into a) to emulate fault severity."""
    return ((1.0 - w) * a + w * b).astype(np.float32)


def _write_csv(path: Path, signal: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["amplitude"])
        for v in signal:
            w.writerow([f"{float(v):.5f}"])


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    normal_a = synthetic_bearing(N, SR, fault=False, rpm=1797, seed=10)
    normal_b = synthetic_bearing(N, SR, fault=False, rpm=1730, seed=11)
    fault_i = synthetic_bearing(N, SR, fault=True, rpm=1797, seed=20)
    fault_ball = synthetic_bearing(N, SR, fault=True, rpm=1772, seed=21)
    fault_outer = synthetic_bearing(N, SR, fault=True, rpm=1750, seed=22)

    # label: 0 healthy, 1 fault. type is descriptive.
    specs = [
        ("01_normal_1797rpm.csv", normal_a, 0, "normal", "Healthy bearing, 1797 rpm"),
        ("02_normal_1730rpm.csv", normal_b, 0, "normal", "Healthy bearing, 1730 rpm"),
        ("03_normal_lightload.csv",
         synthetic_bearing(N, SR, fault=False, rpm=1760, seed=12), 0, "normal",
         "Healthy bearing, light load"),
        ("04_inner_race_fault.csv", fault_i, 1, "inner_race", "Inner-race defect"),
        ("05_ball_fault.csv", fault_ball, 1, "ball", "Rolling-element (ball) defect"),
        ("06_outer_race_fault.csv", fault_outer, 1, "outer_race", "Outer-race defect"),
        ("07_early_stage_fault.csv", _blend(normal_a, fault_i, 0.35), 1, "incipient",
         "Early-stage / subtle inner-race defect"),
        ("08_severe_fault.csv", _blend(normal_b, fault_outer, 1.4), 1, "severe",
         "Advanced / severe outer-race defect"),
    ]

    manifest = SAMPLES_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "type", "sample_rate", "description"])
        for name, sig, label, typ, desc in specs:
            _write_csv(SAMPLES_DIR / name, np.asarray(sig, dtype=np.float32))
            w.writerow([name, label, typ, SR, desc])

    print(f"wrote {len(specs)} sample signals + manifest to {SAMPLES_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

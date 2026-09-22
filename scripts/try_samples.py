"""Try SignalMint on the bundled sample signals -- no download, no checkpoint.

Loads the CSV signals in ``samples/``, trains a small model on the *healthy* ones
only, then for every sample reports the anomaly score (vs a calibrated threshold)
and the neural compression ratio. One command to see both products at once.

Usage:
    python scripts/try_samples.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from signalmint.anomaly.score import calibrate_threshold, frame_nll_scores
from signalmint.compress.baselines import gzip_bits_per_sample, raw_bits_per_sample
from signalmint.compress.coder import encode_frames
from signalmint.config import DataConfig, ModelConfig, SignalMintConfig, TrainConfig
from signalmint.data.cwru import SignalRecord
from signalmint.data.dataset import records_to_frames
from signalmint.train import train_model


def _load_samples(samples_dir: Path) -> list[SignalRecord]:
    manifest = samples_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(
            f"no manifest at {manifest}. Run: python scripts/make_samples.py"
        )
    records: list[SignalRecord] = []
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sig = np.loadtxt(samples_dir / row["file"], delimiter=",", skiprows=1)
            records.append(
                SignalRecord(
                    name=row["file"].replace(".csv", ""),
                    signal=sig.astype(np.float32),
                    label=int(row["label"]),
                    fault_type=row["type"],
                    source="sample",
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Test SignalMint on sample signals.")
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--num-bins", type=int, default=32)
    parser.add_argument("--frame-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    records = _load_samples(Path(args.samples_dir))
    n_norm = sum(r.label == 0 for r in records)
    print(f"Loaded {len(records)} sample signals ({n_norm} healthy, {len(records) - n_norm} fault).")

    cfg = SignalMintConfig(
        data=DataConfig(num_bins=args.num_bins, frame_length=args.frame_length,
                        hop_length=args.frame_length // 2),
        model=ModelConfig(num_bins=args.num_bins, residual_channels=12, skip_channels=16,
                          num_layers=5, dilation_cycle=5),
        train=TrainConfig(epochs=args.epochs, batch_size=16, learning_rate=3e-3, seed=0),
    )

    frames = records_to_frames(records, cfg.data)
    normal_frames = frames.filter_label(0)

    print(f"Training a tiny model on the healthy signals ({args.epochs} epochs) ...")
    result = train_model(normal_frames, cfg.model, cfg.train)
    model = result.model

    threshold = calibrate_threshold(frame_nll_scores(model, normal_frames), percentile=99.0)

    print("\n" + "=" * 74)
    print(f"{'sample':<26}{'type':<12}{'score':>8}{'verdict':>12}{'bits/sample':>14}")
    print("-" * 74)
    for rec in records:
        fs = records_to_frames([rec], cfg.data)
        score = float(np.mean(frame_nll_scores(model, fs)))
        verdict = "ANOMALY" if score > threshold else "normal"
        comp = encode_frames(model, fs.symbols, total_bits=16)
        print(f"{rec.name:<26}{rec.fault_type:<12}{score:>8.3f}{verdict:>12}"
              f"{comp.bits_per_sample:>10.2f} b/s")
    print("-" * 74)
    print(f"threshold (99th pct of healthy) = {threshold:.3f} nats   "
          f"| uniform baseline = {raw_bits_per_sample(args.num_bins):.2f} b/s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

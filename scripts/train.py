"""Train the WaveNet-lite model and report per-sample NLL.

Example:
    python scripts/train.py --source auto --epochs 15 --out artifacts/model.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from signalmint.checkpoint import save_checkpoint
from signalmint.config import SignalMintConfig
from signalmint.data.dataset import records_to_frames
from signalmint.data.loader import load_records
from signalmint.train import train_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SignalMint WaveNet-lite.")
    parser.add_argument("--source", default="auto", choices=["auto", "cwru", "synthetic"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--residual-channels", type=int, default=32)
    parser.add_argument("--skip-channels", type=int, default=64)
    parser.add_argument("--normal-only", action="store_true", default=True,
                        help="Train on healthy frames only (anomaly-detection setup).")
    parser.add_argument("--all-classes", dest="normal_only", action="store_false")
    parser.add_argument("--out", default="artifacts/model.pt")
    parser.add_argument("--metrics", default="artifacts/train_metrics.json")
    args = parser.parse_args()

    config = SignalMintConfig()
    config.data.dataset = "cwru" if args.source != "synthetic" else "synthetic"
    config.data.data_dir = args.data_dir
    config.data.num_bins = args.num_bins
    config.data.frame_length = args.frame_length
    config.data.hop_length = args.hop_length
    config.model.num_bins = args.num_bins
    config.model.num_layers = args.layers
    config.model.residual_channels = args.residual_channels
    config.model.skip_channels = args.skip_channels
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr

    print(f"Loading records (source={args.source}) ...")
    records = load_records(args.data_dir, source=args.source)
    src = {r.source for r in records}
    n_normal = sum(1 for r in records if r.label == 0)
    n_fault = sum(1 for r in records if r.label == 1)
    print(f"  {len(records)} records from {src}: {n_normal} normal, {n_fault} fault")

    frames = records_to_frames(records, config.data)
    if args.normal_only:
        frames = frames.filter_label(0)
    print(f"  {len(frames)} frames of length {config.data.frame_length}")

    print("Training ...")
    t0 = time.time()
    result = train_model(frames, config.model, config.train)
    elapsed = time.time() - t0

    ckpt = save_checkpoint(result.model, config, args.out)
    metrics = {
        "params": result.model.num_parameters(),
        "receptive_field": result.model.receptive_field,
        "train_nll_nats": result.train_nll,
        "val_nll_nats": result.val_nll,
        "best_val_nll_nats": result.best_val_nll,
        "bits_per_sample": result.bits_per_sample,
        "num_frames": len(frames),
        "elapsed_sec": round(elapsed, 2),
        "sources": sorted(src),
    }
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nDone in {elapsed:.1f}s -> {ckpt}")
    print(f"  parameters:       {metrics['params']:,}")
    print(f"  receptive field:  {metrics['receptive_field']} samples")
    print(f"  best val NLL:     {result.best_val_nll:.4f} nats/sample")
    print(f"  bits/sample:      {result.bits_per_sample:.4f}  "
          f"(uniform baseline = {config.data.num_bins.bit_length() - 1} bits)")
    print(f"  metrics written:  {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate a trained checkpoint: anomaly detection + compression benchmarks.

Produces the Phase 2 deliverable -- real numbers for both product heads from the
*same* model:

* anomaly: ROC AUC and detection@fixed-false-alarm, model vs autoencoder baseline
* compression: bits/sample, neural codec vs gzip / order-0 entropy / uniform

Example:
    python scripts/evaluate.py --checkpoint artifacts/model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signalmint.anomaly.baseline import autoencoder_scores, train_autoencoder
from signalmint.anomaly.detector import evaluate_anomaly, report_from_scores
from signalmint.checkpoint import load_checkpoint
from signalmint.compress.baselines import (
    gzip_bits_per_sample,
    order0_entropy_bits_per_sample,
    raw_bits_per_sample,
)
from signalmint.compress.coder import decode_frame, encode_frames
from signalmint.data.dataset import records_to_frames
from signalmint.data.loader import load_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SignalMint heads.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--source", default="auto", choices=["auto", "cwru", "synthetic"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target-fa", type=float, default=0.01)
    parser.add_argument("--ae-epochs", type=int, default=15)
    parser.add_argument("--max-compress-frames", type=int, default=400)
    parser.add_argument("--out", default="artifacts/eval_metrics.json")
    args = parser.parse_args()

    print(f"Loading checkpoint {args.checkpoint} ...")
    model, config = load_checkpoint(args.checkpoint)

    print(f"Loading records (source={args.source}) ...")
    records = load_records(args.data_dir, source=args.source)
    frames = records_to_frames(records, config.data)
    normal = frames.filter_label(0)
    fault = frames.filter_label(1)
    print(f"  frames: {len(frames)} total, {len(normal)} normal, {len(fault)} fault")

    # ---- Anomaly detection ---------------------------------------------------
    print("Scoring anomalies (model NLL) ...")
    model_report, model_scores = evaluate_anomaly(model, frames, target_fa=args.target_fa)

    print(f"Training autoencoder baseline ({args.ae_epochs} epochs) ...")
    ae = train_autoencoder(normal, config.data.num_bins, epochs=args.ae_epochs, seed=0)
    ae_scores = autoencoder_scores(ae, frames, config.data.num_bins)
    ae_report = report_from_scores(ae_scores, frames.labels, args.target_fa)

    # ---- Compression ---------------------------------------------------------
    # Benchmark on CONTIGUOUS, NON-OVERLAPPING frames built straight from the
    # normal signals. (Using the overlapping training frames would unfairly help
    # gzip, which would just exploit the duplicated overlap.)
    print("Compressing (neural codec) ...")
    from signalmint.data.framing import frame_signal
    from signalmint.data.quantize import quantize

    L = config.data.frame_length
    comp_chunks = []
    for r in records:
        if r.label != 0:
            continue
        sym, _ = quantize(
            r.signal, config.data.num_bins, config.data.quantizer,
            config.data.mu_law_mu, config.data.normalize,
        )
        chunk = frame_signal(sym, L, L)  # non-overlapping
        if chunk.shape[0]:
            comp_chunks.append(chunk)
    subset = np.concatenate(comp_chunks, axis=0)[: args.max_compress_frames]

    comp = encode_frames(model, subset, total_bits=16)
    gzip_bps = gzip_bits_per_sample(subset.reshape(-1), config.data.num_bins)
    order0_bps = order0_entropy_bits_per_sample(subset.reshape(-1), config.data.num_bins)
    uniform_bps = raw_bits_per_sample(config.data.num_bins)

    # Verify exact reversibility on one frame.
    one = encode_frames(model, subset[:1], total_bits=16).data
    decoded = decode_frame(model, one, length=subset.shape[1], total_bits=16)
    roundtrip_ok = bool(np.array_equal(decoded, subset[0]))

    metrics = {
        "anomaly": {
            "model": vars(model_report),
            "autoencoder": vars(ae_report),
        },
        "compression": {
            "neural_bits_per_sample": comp.bits_per_sample,
            "gzip_bits_per_sample": gzip_bps,
            "order0_entropy_bits_per_sample": order0_bps,
            "uniform_bits_per_sample": uniform_bps,
            "compression_ratio_vs_uniform": uniform_bps / comp.bits_per_sample,
            "roundtrip_exact": roundtrip_ok,
            "num_symbols": comp.num_symbols,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- Report --------------------------------------------------------------
    print("\n==================== ANOMALY DETECTION ====================")
    print(f"  {'scorer':<14}{'AUC':>8}{'det@FA':>10}{'thr':>10}{'FA':>8}")
    for name, rep in (("model NLL", model_report), ("autoencoder", ae_report)):
        print(f"  {name:<14}{rep.auc:>8.3f}{rep.detection_rate:>10.3f}"
              f"{rep.threshold:>10.3f}{rep.achieved_fa:>8.3f}")
    print(f"  (target false-alarm = {args.target_fa})")

    print("\n==================== COMPRESSION ==========================")
    print(f"  {'neural codec':<26}{comp.bits_per_sample:>8.3f} bits/sample")
    print(f"  {'gzip':<26}{gzip_bps:>8.3f} bits/sample")
    print(f"  {'order-0 entropy':<26}{order0_bps:>8.3f} bits/sample")
    print(f"  {'uniform (fixed-width)':<26}{uniform_bps:>8.3f} bits/sample")
    print(f"  compression vs uniform:   {uniform_bps / comp.bits_per_sample:.2f}x")
    print(f"  exact round-trip:         {roundtrip_ok}")
    print(f"\n  metrics written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

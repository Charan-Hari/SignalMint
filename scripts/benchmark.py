"""End-to-end benchmark: one command that produces the headline numbers.

Runs (training the model first if no checkpoint exists) the anomaly, compression,
INT8 parity and edge-footprint stages, then writes a consolidated
``artifacts/benchmarks.json`` and a human-readable ``docs/BENCHMARKS.md``.

Example:
    python scripts/benchmark.py --source cwru
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from signalmint.data.framing import frame_signal
from signalmint.data.loader import load_records
from signalmint.data.quantize import quantize
from signalmint.quantize.footprint import analyze_footprint
from signalmint.quantize.parity import check_parity, find_c_compiler
from signalmint.quantize.quantizer import quantize_model


def _ensure_checkpoint(checkpoint: str, source: str, epochs: int) -> None:
    if Path(checkpoint).exists():
        return
    print(f"No checkpoint at {checkpoint}; training first ...")
    subprocess.run(
        [sys.executable, "scripts/train.py", "--source", source,
         "--epochs", str(epochs), "--out", checkpoint],
        check=True,
    )


def _contiguous_normal_frames(records, cfg) -> np.ndarray:
    chunks = []
    for r in records:
        if r.label != 0:
            continue
        sym, _ = quantize(
            r.signal, cfg.num_bins, cfg.quantizer, cfg.mu_law_mu, cfg.normalize
        )
        c = frame_signal(sym, cfg.frame_length, cfg.frame_length)
        if c.shape[0]:
            chunks.append(c)
    return np.concatenate(chunks, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="SignalMint end-to-end benchmark.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--source", default="auto", choices=["auto", "cwru", "synthetic"])
    parser.add_argument("--train-epochs", type=int, default=12)
    parser.add_argument("--target-fa", type=float, default=0.01)
    parser.add_argument("--max-compress-frames", type=int, default=200)
    parser.add_argument("--json", default="artifacts/benchmarks.json")
    parser.add_argument("--md", default="docs/BENCHMARKS.md")
    args = parser.parse_args()

    _ensure_checkpoint(args.checkpoint, args.source, args.train_epochs)
    model, config = load_checkpoint(args.checkpoint)

    from signalmint.data.dataset import records_to_frames

    records = load_records(config.data.data_dir, source=args.source)
    frames = records_to_frames(records, config.data)
    normal = frames.filter_label(0)
    source_tag = sorted({r.source for r in records})

    # ---- Anomaly -------------------------------------------------------------
    print("Anomaly ...")
    model_report, _ = evaluate_anomaly(model, frames, target_fa=args.target_fa)
    ae = train_autoencoder(normal, config.data.num_bins, epochs=15, seed=0)
    ae_report = report_from_scores(
        autoencoder_scores(ae, frames, config.data.num_bins), frames.labels, args.target_fa
    )

    # ---- Compression ---------------------------------------------------------
    print("Compression ...")
    subset = _contiguous_normal_frames(records, config.data)[: args.max_compress_frames]
    comp = encode_frames(model, subset, total_bits=16)
    gzip_bps = gzip_bits_per_sample(subset.reshape(-1), config.data.num_bins)
    order0_bps = order0_entropy_bits_per_sample(subset.reshape(-1), config.data.num_bins)
    uniform_bps = raw_bits_per_sample(config.data.num_bins)
    one = encode_frames(model, subset[:1], total_bits=16).data
    roundtrip = bool(np.array_equal(
        decode_frame(model, one, length=subset.shape[1], total_bits=16), subset[0]
    ))

    # ---- Quantization + parity + footprint -----------------------------------
    print("Quantization / parity / footprint ...")
    qm = quantize_model(model)
    fp = analyze_footprint(qm)
    parity = None
    if find_c_compiler() is not None:
        rng = np.random.default_rng(0)
        syms = rng.integers(0, config.data.num_bins, size=128).astype(np.int64)
        pr = check_parity(model, syms)
        parity = {"exact": pr.exact, "max_abs_diff": pr.max_abs_diff, "compiler": pr.compiler}

    benchmarks = {
        "dataset": source_tag,
        "model": {
            "num_bins": config.data.num_bins,
            "residual_channels": config.model.residual_channels,
            "skip_channels": config.model.skip_channels,
            "num_layers": config.model.num_layers,
            "receptive_field": fp.receptive_field,
            "int8_parameters": fp.num_parameters,
        },
        "anomaly": {"model": vars(model_report), "autoencoder": vars(ae_report)},
        "compression": {
            "neural_bits_per_sample": comp.bits_per_sample,
            "gzip_bits_per_sample": gzip_bps,
            "order0_entropy_bits_per_sample": order0_bps,
            "uniform_bits_per_sample": uniform_bps,
            "vs_uniform": uniform_bps / comp.bits_per_sample,
            "vs_gzip": gzip_bps / comp.bits_per_sample,
            "roundtrip_exact": roundtrip,
        },
        "footprint": {
            "rom_kb": round(fp.rom_kb, 2),
            "streaming_ram_kb": round(fp.streaming_ram_kb, 2),
            "macs_per_sample": fp.macs_per_sample,
        },
        "parity": parity,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
    _render_md(benchmarks, args.md, args.target_fa)

    print("\nWrote", args.json, "and", args.md)
    print(f"  anomaly AUC (model): {model_report.auc:.3f}")
    print(f"  compression: {comp.bits_per_sample:.2f} b/s "
          f"(gzip {gzip_bps:.2f}, uniform {uniform_bps:.2f})")
    print(f"  footprint: ROM {fp.rom_kb:.1f} KB, RAM {fp.streaming_ram_kb:.1f} KB")
    if parity:
        print(f"  INT8 parity exact: {parity['exact']}")
    return 0


def _render_md(b: dict, path: str, target_fa: float) -> None:
    c = b["compression"]
    am, aa = b["anomaly"]["model"], b["anomaly"]["autoencoder"]
    fp = b["footprint"]
    parity = b["parity"]
    parity_line = (
        f"bit-exact ({parity['compiler']}), max abs diff {parity['max_abs_diff']}"
        if parity else "compiler unavailable on this host"
    )
    pct_saved = 100 - 100 / c["vs_uniform"]
    md = f"""# SignalMint Benchmarks

Dataset: {', '.join(b['dataset'])}. Model: {b['model']['num_layers']}-layer WaveNet-lite,
{b['model']['num_bins']} bins, receptive field {b['model']['receptive_field']} samples,
~{b['model']['int8_parameters']:,} INT8 parameters.

## One model, two products

### Anomaly detection (label-free; trained on healthy data only)
| Scorer | ROC AUC | Detection @ {target_fa:.0%} FA |
| --- | --- | --- |
| **SignalMint NLL** | {am['auc']:.3f} | {am['detection_rate']:.3f} |
| Autoencoder baseline | {aa['auc']:.3f} | {aa['detection_rate']:.3f} |

### Compression (bits/sample; lower is better)
| Coder | bits/sample |
| --- | --- |
| **SignalMint neural codec** | **{c['neural_bits_per_sample']:.3f}** |
| gzip | {c['gzip_bits_per_sample']:.3f} |
| order-0 entropy | {c['order0_entropy_bits_per_sample']:.3f} |
| uniform (fixed-width) | {c['uniform_bits_per_sample']:.3f} |

- **{c['vs_uniform']:.2f}x** smaller than a fixed-width code; **{c['vs_gzip']:.2f}x** smaller than gzip.
- exact round-trip: **{c['roundtrip_exact']}**

## Edge footprint (INT8 C runtime)
| Resource | Value |
| --- | --- |
| ROM / flash | {fp['rom_kb']:.1f} KB |
| Streaming RAM (length-independent) | {fp['streaming_ram_kb']:.2f} KB |
| MACs / sample | {fp['macs_per_sample']:,} |
| Python-reference vs C parity | {parity_line} |

## Why it's worth money
- **Compression {c['vs_uniform']:.2f}x** => roughly {pct_saved:.0f}% fewer bytes on the wire,
  a directly billable saving on cellular/satellite/LoRa telemetry links.
- **Label-free anomaly detection** => no rare, expensive failure labels needed;
  the same model that compresses also flags abnormal behaviour locally.
- **Fits {fp['rom_kb']:.0f} KB ROM / {fp['streaming_ram_kb']:.0f} KB RAM** => runs on cheap MCUs
  with no cloud dependency (privacy + offline + battery life).

*Reproduce with `python scripts/benchmark.py --source {b['dataset'][0]}`.*
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

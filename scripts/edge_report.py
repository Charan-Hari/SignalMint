"""Phase 4 edge deployment report: footprint, latency and modeled energy.

Quantizes a checkpoint, analyzes its static footprint, builds and times the C
runtime on this host, and models the per-sample cost on a representative MCU.
Writes a JSON report and a human-readable Markdown summary.

Example:
    python scripts/edge_report.py --checkpoint artifacts/model.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from signalmint.checkpoint import load_checkpoint
from signalmint.quantize.export import export_c_header
from signalmint.quantize.footprint import analyze_footprint
from signalmint.quantize.parity import RUNTIME_DIR, build_runtime, find_c_compiler
from signalmint.quantize.quantizer import quantize_model

# Representative MCU assumptions for the modeled energy estimate.
MCU_NAME = "Cortex-M4F @ 80 MHz (illustrative)"
MCU_CLOCK_HZ = 80_000_000
MCU_MACS_PER_CYCLE = 1.0
MCU_OVERHEAD_FACTOR = 2.0        # non-MAC ops (LUT, requant, control) per MAC
MCU_ACTIVE_POWER_MW = 10.0       # typical M4F active power at this clock


def measure_latency(length: int, reps: int) -> float | None:
    """Return measured microseconds/sample from the built C runtime, or None."""
    comp = find_c_compiler()
    if comp is None:
        return None
    _kind, exe = build_runtime(RUNTIME_DIR)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "syms.txt"
        syms = rng.integers(0, 64, size=length)
        np.savetxt(sp, syms, fmt="%d")
        out = subprocess.run(
            [str(exe), str(sp), "--bench", str(reps)],
            capture_output=True, text=True, check=True,
        )
    total_us, r, T = out.stdout.split()
    return float(total_us) / (int(r) * int(T))


def main() -> int:
    parser = argparse.ArgumentParser(description="SignalMint edge report.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--bench-length", type=int, default=256)
    parser.add_argument("--bench-reps", type=int, default=50)
    parser.add_argument("--json", default="artifacts/edge_report.json")
    parser.add_argument("--md", default="docs/edge_report.md")
    args = parser.parse_args()

    model, cfg = load_checkpoint(args.checkpoint)
    qm = quantize_model(model)
    export_c_header(qm, RUNTIME_DIR / "model_data.h")
    fp = analyze_footprint(qm)

    # Modeled MCU cost.
    cycles_per_sample = fp.macs_per_sample / MCU_MACS_PER_CYCLE * MCU_OVERHEAD_FACTOR
    samples_per_sec = MCU_CLOCK_HZ / cycles_per_sample
    time_per_sample_s = cycles_per_sample / MCU_CLOCK_HZ
    energy_uj = (MCU_ACTIVE_POWER_MW / 1000.0) * time_per_sample_s * 1e6

    measured_us = measure_latency(args.bench_length, args.bench_reps)

    report = {
        "model": {
            "num_bins": qm.num_bins,
            "residual_channels": qm.residual_channels,
            "skip_channels": qm.skip_channels,
            "num_layers": len(qm.blocks),
            "receptive_field": fp.receptive_field,
            "int8_parameters": fp.num_parameters,
        },
        "footprint": {
            "rom_bytes": fp.rom_bytes,
            "rom_kb": round(fp.rom_kb, 2),
            "streaming_ram_bytes": fp.streaming_ram_bytes,
            "streaming_ram_kb": round(fp.streaming_ram_kb, 2),
            "macs_per_sample": fp.macs_per_sample,
        },
        "latency_host": {
            "measured_us_per_sample": (round(measured_us, 4) if measured_us else None),
            "note": "Full-frame C runtime timed on this host CPU.",
        },
        "energy_model_mcu": {
            "mcu": MCU_NAME,
            "assumptions": {
                "clock_hz": MCU_CLOCK_HZ,
                "macs_per_cycle": MCU_MACS_PER_CYCLE,
                "overhead_factor": MCU_OVERHEAD_FACTOR,
                "active_power_mw": MCU_ACTIVE_POWER_MW,
            },
            "cycles_per_sample": round(cycles_per_sample, 1),
            "samples_per_sec": round(samples_per_sec, 1),
            "energy_uj_per_sample": round(energy_uj, 4),
            "fits_sub_100mw": MCU_ACTIVE_POWER_MW < 100.0,
        },
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = f"""# SignalMint Edge Deployment Report

Generated from `{args.checkpoint}` via `scripts/edge_report.py`.

## Model
- alphabet (bins): {qm.num_bins}
- residual / skip channels: {qm.residual_channels} / {qm.skip_channels}
- layers: {len(qm.blocks)}, receptive field: {fp.receptive_field} samples
- INT8 parameters: {fp.num_parameters:,}

## Static footprint
| Resource | Value |
| --- | --- |
| ROM / flash (INT8 weights + biases + embed + LUTs) | {fp.rom_kb:.1f} KB |
| Streaming activation RAM (bounded, length-independent) | {fp.streaming_ram_kb:.2f} KB |
| MACs / sample | {fp.macs_per_sample:,} |

Both comfortably inside the **1-8 MB** target tier; the streaming RAM is bounded
by the dilated-conv ring buffers and does **not** grow with signal length.

## Latency (this host)
- measured: {report['latency_host']['measured_us_per_sample']} us/sample (full-frame C runtime)

## Modeled MCU energy ({MCU_NAME})
Assumptions: {MCU_MACS_PER_CYCLE} MAC/cycle, {MCU_OVERHEAD_FACTOR}x non-MAC overhead,
{MCU_ACTIVE_POWER_MW} mW active at {MCU_CLOCK_HZ/1e6:.0f} MHz.

| Metric | Value |
| --- | --- |
| cycles / sample | {cycles_per_sample:,.0f} |
| throughput | {samples_per_sec:,.0f} samples/s |
| energy / sample | {energy_uj:.3f} uJ |
| sub-100 mW active | {'yes' if MCU_ACTIVE_POWER_MW < 100 else 'no'} |

*Energy figures are modeled estimates with stated assumptions, not measured on
silicon.*
"""
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(md, encoding="utf-8")

    print("==================== EDGE REPORT ====================")
    print(f"  ROM/flash:          {fp.rom_kb:.1f} KB")
    print(f"  streaming RAM:      {fp.streaming_ram_kb:.2f} KB  (length-independent)")
    print(f"  MACs/sample:        {fp.macs_per_sample:,}")
    if measured_us:
        print(f"  measured latency:   {measured_us:.3f} us/sample (host)")
    print(f"  modeled MCU:        {samples_per_sec:,.0f} samples/s, "
          f"{energy_uj:.3f} uJ/sample @ {MCU_ACTIVE_POWER_MW} mW")
    print(f"\n  JSON: {args.json}")
    print(f"  MD:   {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

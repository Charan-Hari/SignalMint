# SignalMint Edge Deployment Report

Generated from `artifacts\model.pt` via `scripts/edge_report.py`.

## Model
- alphabet (bins): 64
- residual / skip channels: 32 / 64
- layers: 8, receptive field: 256 samples
- INT8 parameters: 67,584

## Static footprint
| Resource | Value |
| --- | --- |
| ROM / flash (INT8 weights + biases + embed + LUTs) | 109.5 KB |
| Streaming activation RAM (bounded, length-independent) | 33.12 KB |
| MACs / sample | 65,536 |

Both comfortably inside the **1-8 MB** target tier; the streaming RAM is bounded
by the dilated-conv ring buffers and does **not** grow with signal length.

## Latency (this host)
- measured: 84.6875 us/sample (full-frame C runtime)

## Modeled MCU energy (Cortex-M4F @ 80 MHz (illustrative))
Assumptions: 1.0 MAC/cycle, 2.0x non-MAC overhead,
10.0 mW active at 80 MHz.

| Metric | Value |
| --- | --- |
| cycles / sample | 131,072 |
| throughput | 610 samples/s |
| energy / sample | 16.384 uJ |
| sub-100 mW active | yes |

*Energy figures are modeled estimates with stated assumptions, not measured on
silicon.*

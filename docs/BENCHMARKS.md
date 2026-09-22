# SignalMint Benchmarks

Dataset: cwru. Model: 8-layer WaveNet-lite,
64 bins, receptive field 256 samples,
~67,584 INT8 parameters.

## One model, two products

### Anomaly detection (label-free; trained on healthy data only)
| Scorer | ROC AUC | Detection @ 1% FA |
| --- | --- | --- |
| **SignalMint NLL** | 1.000 | 1.000 |
| Autoencoder baseline | 1.000 | 1.000 |

### Compression (bits/sample; lower is better)
| Coder | bits/sample |
| --- | --- |
| **SignalMint neural codec** | **2.892** |
| gzip | 4.734 |
| order-0 entropy | 5.340 |
| uniform (fixed-width) | 6.000 |

- **2.07x** smaller than a fixed-width code; **1.64x** smaller than gzip.
- exact round-trip: **True**

## Edge footprint (INT8 C runtime)
| Resource | Value |
| --- | --- |
| ROM / flash | 109.5 KB |
| Streaming RAM (length-independent) | 33.12 KB |
| MACs / sample | 65,536 |
| Python-reference vs C parity | bit-exact (msvc), max abs diff 0 |

## Why it's worth money
- **Compression 2.07x** => roughly 52% fewer bytes on the wire,
  a directly billable saving on cellular/satellite/LoRa telemetry links.
- **Label-free anomaly detection** => no rare, expensive failure labels needed;
  the same model that compresses also flags abnormal behaviour locally.
- **Fits 110 KB ROM / 33 KB RAM** => runs on cheap MCUs
  with no cloud dependency (privacy + offline + battery life).

*Reproduce with `python scripts/benchmark.py --source cwru`.*

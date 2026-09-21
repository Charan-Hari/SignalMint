# SignalMint

> Learn normal. Compress better. Detect earlier. Run anywhere.

**Tiny, explicit-likelihood generative models for 1-D edge signals** (vibration,
audio, current, telemetry). One trained model delivers two products at once:

1. **Label-free anomaly detection** — the model outputs a probability for every
   sample it sees. Anything it finds improbable is an anomaly. No labeled failure
   data required (that is the problem the predictive-maintenance market is stuck on).
2. **Neural compression** — the same per-sample probabilities feed an entropy
   coder. "Predict the next sample well" is mathematically identical to "send
   fewer bits", so the model doubles as a codec for bandwidth-constrained links.

Both capabilities come from the **same** autoregressive model. That is the whole
thesis: an explicit-likelihood model is simultaneously a detector and a compressor.

## Why this and not "tiny image generation"

Image generation on tiny hardware is a demo category — nobody buys a 32x32 art
generator. But *generative modeling of signals* has non-art uses where being small
is a genuine requirement. See [docs/thesis.md](docs/thesis.md) for the full
positioning, target market, and success metrics.

## Target tier

| Parameter        | Choice                                                     |
| ---------------- | ---------------------------------------------------------- |
| Memory / RAM     | 1-8 MB (not 264 KB, not 1 GB)                              |
| Power            | sub-100 mW active                                          |
| Precision        | INT8 weights/activations, quantization-aware training      |
| Modality         | 1-D signals first (vibration/audio/telemetry)             |
| Deployment       | software-only on dev boards first; silicon co-design later |
| Primary dataset  | CWRU bearing vibration (labeled normal/fault)             |

## Architecture: one model, three heads

```
   Raw 1-D signal
        |
  [ framing + quantization to N bins ]      (mu-law / learned bins, INT8-friendly)
        |
  [ streaming causal model ]                (dilated causal convs, WaveNet-lite, INT8)
        |  P(next sample | past)
        +----------------+-----------------+
   Likelihood        Entropy coder      Sampler
   (NLL score)       (arithmetic/rANS)  (optional)
        |                |                  |
   ANOMALY score    COMPRESSED bits    synthetic / infill
```

## Project phases (all complete)

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 0 | Scaffold | done — `pip install -e .`, tests green |
| 1 | Data + reference model on CWRU | done — trains, per-sample NLL reported |
| 2 | Both product heads benchmarked | done — anomaly + compression beat baselines |
| 3 | INT8 + portable C runtime | done — bit-exact parity C vs Python |
| 4 | Edge deployment proof | done — ROM/RAM/latency/energy report |
| 5 | Packaging + story | done — benchmark, docs, demo notebook |

## Results (CWRU bearing vibration)

One 8-layer WaveNet-lite (64 bins, ~68 K INT8 params, 256-sample receptive field),
trained on healthy data only. Full table in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Capability | SignalMint | Baseline |
| --- | --- | --- |
| **Anomaly** ROC AUC / detection @1% FA | **1.000 / 1.000** | autoencoder 1.000 / 1.000 |
| **Compression** bits/sample | **2.89** | gzip 4.73, order-0 5.34, uniform 6.00 |
| **Edge footprint** | **110 KB ROM, 33 KB RAM**, 65 K MACs/sample | — |
| **INT8 C runtime parity** | **bit-exact** (max abs diff 0) | — |

- Compression is **2.07× smaller than fixed-width** and **1.64× smaller than gzip**, with an **exact round-trip** — a directly billable ~52% cut in bytes on the wire.
- The **same model** does anomaly detection with **no labeled failures**.
- Fits cheap MCUs with **length-independent** streaming RAM and no cloud dependency.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[train,dev]"
```

Core install (`pip install -e .`) pulls only NumPy/SciPy so the package imports on
any machine. PyTorch is an optional `train` extra so the runtime/eval paths stay light.

## Reproduce

```powershell
python scripts/train.py       --source cwru --epochs 12      # -> artifacts/model.pt
python scripts/evaluate.py    --checkpoint artifacts/model.pt # anomaly + compression
python scripts/export_c.py    --checkpoint artifacts/model.pt # -> runtime/model_data.h
python scripts/parity.py      --checkpoint artifacts/model.pt # bit-exact C parity
python scripts/edge_report.py --checkpoint artifacts/model.pt # ROM/RAM/latency/energy
python scripts/benchmark.py   --source cwru                   # everything -> docs/BENCHMARKS.md
```

Or run [notebooks/demo.ipynb](notebooks/demo.ipynb) for a fast synthetic-data walkthrough (no download).

## License

MIT — see [LICENSE](LICENSE).

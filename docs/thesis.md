# SignalMint Thesis & Positioning

This document is the "why" behind the code. It is the strategic frame that the
repository deliberately keeps public as a credibility piece, while any
customer-specific deployment work stays private.

## The core technical bet

An **autoregressive model outputs an explicit probability for every sample it
observes.** That single property yields two products from one trained model:

* **Anomaly detection** — samples the model finds improbable (high negative
  log-likelihood) are anomalies. This needs *no labeled failure data*, which is
  precisely why the predictive-maintenance market is stuck: failures are rare and
  expensive to collect.
* **Compression** — feeding those same probabilities into an entropy coder is the
  textbook construction of a neural codec. "Predict the next sample well" is
  mathematically identical to "spend fewer bits", giving a directly dollar-linked
  metric (bandwidth cost per device per month).

Everything else in the repo is engineering in service of running this one idea in
1-8 MB of RAM at INT8 precision.

## Why not tiny image generation

Image generation on a microcontroller is an impressive *demo* with no buyer. The
transferable assets from that class of work are: streaming inference that bounds
peak memory, model-aware quantization, and matching a model's arithmetic pattern
to constrained hardware. Point those at **signals** (vibration, audio, current,
telemetry) and there are buyers who pay to work around power, latency, bandwidth,
and privacy limits.

## Target tier (committed)

| Parameter   | Choice                                   | Rationale                                  |
| ----------- | ---------------------------------------- | ------------------------------------------ |
| RAM         | 1-8 MB                                   | 264 KB is too tight for utility; 1 GB is already served |
| Power       | sub-100 mW active                        | physics does not get cheaper over time     |
| Precision   | INT8, quantization-aware                  | fits the tier, matches cheap MAC hardware  |
| Modality    | 1-D signals first                        | the paying problems and 100x cheaper compute |
| Deployment  | dev boards / software first              | validate before spending on silicon        |

## First dataset: CWRU bearing vibration

The Case Western Reserve University bearing dataset is numeric time-series with
labeled normal and fault conditions. It is the canonical predictive-maintenance
benchmark, small enough to iterate quickly, and lets us report both:

* anomaly AUC / detection-rate-at-fixed-false-alarm (train on normal only), and
* bits-per-sample compression against gzip/FLAC,

from the **same** checkpoint.

## Who pays (summary)

The end buyer is an OEM or enterprise deploying many low-power edge devices where
sending raw data to the cloud is too costly, slow, private, or unreliable. The
strongest first beachhead is **industrial predictive maintenance / remote
monitoring** (vibration and telemetry), where downtime is expensive and labeled
failure data is scarce.

## Success metrics (what "working" means)

1. **Compression:** beat gzip/FLAC bits-per-sample on real vibration data.
2. **Anomaly:** match/beat an autoencoder's detection rate at a fixed false-alarm
   rate, with no labeled failures.
3. **Footprint:** single-digit MB RAM, INT8, bounded peak memory, with bit-exact
   parity between the Python reference and the C runtime.

## What we are NOT doing

* Not generating images or competing on generative quality with cloud models.
* Not building custom silicon before paying pilots exist.
* Not using GPU-scale architectures that assume abundant memory.

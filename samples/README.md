# Sample signals

Eight ready-to-run 1-D vibration signals so you can try SignalMint without
downloading any dataset. They are produced by the project's **own** synthetic
bearing generator (`scripts/make_samples.py`), so they carry no third-party data
license. Each file is a single-column CSV (`amplitude`, one sample per line) at a
12 kHz sample rate.

| File | Label | Type | Description |
| --- | --- | --- | --- |
| `01_normal_1797rpm.csv` | healthy | normal | Healthy bearing, 1797 rpm |
| `02_normal_1730rpm.csv` | healthy | normal | Healthy bearing, 1730 rpm |
| `03_normal_lightload.csv` | healthy | normal | Healthy bearing, light load |
| `04_inner_race_fault.csv` | fault | inner_race | Inner-race defect |
| `05_ball_fault.csv` | fault | ball | Rolling-element (ball) defect |
| `06_outer_race_fault.csv` | fault | outer_race | Outer-race defect |
| `07_early_stage_fault.csv` | fault | incipient | Early-stage / subtle defect (hard case) |
| `08_severe_fault.csv` | fault | severe | Advanced outer-race defect |

## Try them

```bash
python scripts/try_samples.py
```

This trains a tiny model on the **healthy** signals only, then scores every
sample for anomaly (vs a threshold calibrated on healthy data) and reports the
neural compression rate. Typical output flags the clear faults as `ANOMALY`
while healthy signals stay `normal`. The `07_early_stage_fault` case is
deliberately subtle and often sits near the threshold — a realistic reminder
that incipient faults are the genuinely hard ones, not a solved problem.

## Regenerate

```bash
python scripts/make_samples.py
```

## Use your own signal

Any single-column CSV of numeric samples works. Point the loader at it, or drop
it into `samples/` and add a row to `manifest.csv`.

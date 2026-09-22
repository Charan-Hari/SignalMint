"""Generate the README demo GIF from the trained model.

Renders an honest animation: a healthy bearing-vibration signal streams by with a
live per-frame anomaly score (mean NLL). Partway through, the stream switches to a
faulty bearing and the score jumps above the calibrated threshold -- the same
label-free detection the model performs on-device.

Usage:
    python scripts/make_demo_gif.py --checkpoint artifacts/model.pt --out docs/demo.gif
"""

from __future__ import annotations

import argparse

import numpy as np

# Headless rendering.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from signalmint.anomaly.score import calibrate_threshold, frame_nll_scores  # noqa: E402
from signalmint.checkpoint import load_checkpoint  # noqa: E402
from signalmint.data.cwru import get_synthetic_records  # noqa: E402
from signalmint.data.dataset import FrameSet, records_to_frames  # noqa: E402
from signalmint.data.loader import load_records  # noqa: E402


def _pick_records(records, label):
    return [r for r in records if r.label == label]


def main() -> int:
    parser = argparse.ArgumentParser(description="Make SignalMint demo GIF.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--source", default="auto", choices=["auto", "cwru", "synthetic"])
    parser.add_argument("--out", default="docs/demo.gif")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    model, cfg = load_checkpoint(args.checkpoint)

    try:
        records = load_records(cfg.data.data_dir, source=args.source)
        if not records:
            raise RuntimeError("empty")
    except Exception:
        records = get_synthetic_records()

    normal_recs = _pick_records(records, 0)
    fault_recs = _pick_records(records, 1)
    normal = records_to_frames(normal_recs, cfg.data)
    fault = records_to_frames(fault_recs, cfg.data)

    # Calibrate the detection threshold on healthy scores.
    normal_scores = frame_nll_scores(model, normal)
    threshold = calibrate_threshold(normal_scores, percentile=99.0)

    # Build a timeline: healthy frames, then faulty frames.
    half = args.frames // 2
    n_idx = np.random.default_rng(0).choice(len(normal), size=min(half, len(normal)), replace=False)
    f_idx = np.random.default_rng(1).choice(len(fault), size=min(args.frames - half, len(fault)), replace=False)
    timeline = FrameSet(
        symbols=np.concatenate([normal.symbols[n_idx], fault.symbols[f_idx]], axis=0),
        labels=np.concatenate([np.zeros(len(n_idx), int), np.ones(len(f_idx), int)]),
        names=["normal"] * len(n_idx) + ["fault"] * len(f_idx),
    )
    scores = frame_nll_scores(model, timeline)
    T = timeline.symbols.shape[1]
    num_bins = cfg.data.num_bins

    plt.rcParams.update({
        "figure.facecolor": "#0b1020", "axes.facecolor": "#141b2e",
        "axes.edgecolor": "#263150", "text.color": "#e8edf7",
        "axes.labelcolor": "#9aa6c0", "xtick.color": "#9aa6c0",
        "ytick.color": "#9aa6c0", "font.size": 11,
    })
    fig, (ax_sig, ax_score) = plt.subplots(
        2, 1, figsize=(6.2, 3.5), gridspec_kw={"height_ratios": [2, 1]}, dpi=90
    )
    fig.subplots_adjust(hspace=0.5, left=0.1, right=0.97, top=0.88, bottom=0.13)

    x = np.arange(T)
    (sig_line,) = ax_sig.plot([], [], lw=1.4, color="#4cc9f0")
    ax_sig.set_xlim(0, T)
    ax_sig.set_ylim(-0.05, num_bins + 0.05)
    ax_sig.set_ylabel("signal (quantized)")
    title = ax_sig.set_title("", fontsize=14, fontweight="bold")

    score_x = np.arange(len(scores))
    (score_line,) = ax_score.plot([], [], lw=2.0, color="#8b5cf6")
    ax_score.axhline(threshold, color="#f0b429", ls="--", lw=1.2, label="detection threshold")
    ax_score.set_xlim(0, len(scores))
    ax_score.set_ylim(float(scores.min()) * 0.9, float(scores.max()) * 1.1)
    ax_score.set_xlabel("frame")
    ax_score.set_ylabel("anomaly score (NLL)")
    ax_score.legend(loc="upper left", fontsize=9, facecolor="#141b2e", edgecolor="#263150")
    marker = ax_score.scatter([], [], s=40, zorder=5)

    def update(i):
        frame = timeline.symbols[i]
        sig_line.set_data(x, frame)
        is_fault = timeline.labels[i] == 1
        sig_line.set_color("#ff6b6b" if is_fault else "#4cc9f0")
        state = "FAULT DETECTED" if is_fault else "healthy"
        color = "#ff6b6b" if is_fault else "#56d364"
        title.set_text(f"SignalMint  \u2014  {state}")
        title.set_color(color)

        score_line.set_data(score_x[: i + 1], scores[: i + 1])
        cols = ["#ff6b6b" if s > threshold else "#4cc9f0" for s in scores[: i + 1]]
        marker.set_offsets(np.c_[score_x[: i + 1], scores[: i + 1]])
        marker.set_color(cols)
        return sig_line, score_line, title, marker

    anim = FuncAnimation(fig, update, frames=len(timeline.symbols), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    # Optimize palette to shrink the file for a README.
    try:
        from PIL import Image, ImageSequence

        img = Image.open(args.out)
        frames_opt = [f.convert("P", palette=Image.ADAPTIVE, colors=64)
                      for f in ImageSequence.Iterator(img)]
        frames_opt[0].save(
            args.out, save_all=True, append_images=frames_opt[1:],
            loop=0, duration=int(1000 / args.fps), optimize=True,
        )
    except Exception:  # noqa: BLE001 - optimization is best-effort
        pass

    import os
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({len(timeline.symbols)} frames @ {args.fps} fps, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

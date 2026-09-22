"""Export a trained checkpoint to the C runtime's ``model_data.h``.

Example:
    python scripts/export_c.py --checkpoint artifacts/model.pt \
        --out runtime/model_data.h
"""

from __future__ import annotations

import argparse
from pathlib import Path

from signalmint.checkpoint import load_checkpoint
from signalmint.quantize.export import export_c_header
from signalmint.quantize.quantizer import quantize_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SignalMint model to C.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--out", default="runtime/model_data.h")
    args = parser.parse_args()

    model, _config = load_checkpoint(args.checkpoint)
    qm = quantize_model(model)
    path = export_c_header(qm, args.out)

    size_kb = Path(path).stat().st_size / 1024
    print(f"exported {path} ({size_kb:.1f} KB header)")
    print(f"  layers={len(qm.blocks)}  res_ch={qm.residual_channels} "
          f"skip_ch={qm.skip_channels}  num_bins={qm.num_bins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

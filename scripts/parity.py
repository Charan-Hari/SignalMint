"""Verify bit-exact parity between the Python integer reference and the C runtime.

Exports a trained checkpoint to C, builds the runtime, and compares int32 logits.

Example:
    python scripts/parity.py --checkpoint artifacts/model.pt --length 128
"""

from __future__ import annotations

import argparse

import numpy as np

from signalmint.checkpoint import load_checkpoint
from signalmint.quantize.parity import check_parity, find_c_compiler


def main() -> int:
    parser = argparse.ArgumentParser(description="SignalMint C parity check.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    comp = find_c_compiler()
    if comp is None:
        print("ERROR: no C compiler found (need gcc/clang or MSVC build tools).")
        return 1
    print(f"Using compiler: {comp[0]}")

    model, cfg = load_checkpoint(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    symbols = rng.integers(0, cfg.data.num_bins, size=args.length).astype(np.int64)

    print(f"Checking parity on {args.length} symbols ({cfg.data.num_bins} bins) ...")
    result = check_parity(model, symbols)

    print(f"  values compared:  {result.num_values}")
    print(f"  mismatches:       {result.num_mismatch}")
    print(f"  max abs diff:     {result.max_abs_diff}")
    print(f"  BIT-EXACT PARITY: {result.exact}")
    return 0 if result.exact else 2


if __name__ == "__main__":
    raise SystemExit(main())

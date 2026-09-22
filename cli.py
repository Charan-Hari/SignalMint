"""Command-line interface for SignalMint.

Phase 0 exposes ``version`` and ``config`` subcommands so the installed
``signalmint`` entry point is real and testable. Later phases add ``train``,
``evaluate``, ``compress`` and ``export`` subcommands.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .config import SignalMintConfig, default_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signalmint",
        description=(
            "Tiny explicit-likelihood generative models for edge signals: "
            "anomaly detection and neural compression from one model."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"signalmint {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    p_config = sub.add_parser(
        "config",
        help="Print or write the default pipeline configuration.",
    )
    p_config.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the config JSON to this path instead of stdout.",
    )

    sub.add_parser("version", help="Print the SignalMint version.")
    return parser


def _cmd_config(output: str | None) -> int:
    config: SignalMintConfig = default_config()
    if output:
        path = config.to_json(output)
        print(f"wrote config to {path}")
    else:
        import json

        print(json.dumps(config.to_dict(), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "version"):
        print(f"signalmint {__version__}")
        return 0
    if args.command == "config":
        return _cmd_config(args.output)

    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

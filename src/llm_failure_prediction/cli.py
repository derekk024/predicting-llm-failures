from __future__ import annotations

import argparse
from pathlib import Path

from llm_failure_prediction.config import load_config
from llm_failure_prediction.pilot import run_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-failures")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser(
        "pilot",
        help="Run paired ARC prompts and cache clean activations",
    )
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--limit", type=int, help="Override the configured question limit")
    pilot.add_argument("--run-id", help="Use a stable output directory name")
    pilot.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.limit is not None:
        config.dataset.limit = args.limit
    if args.device is not None:
        config.model.device = args.device
    if args.command == "pilot":
        run_dir = run_pilot(config, run_id=args.run_id)
        print(f"Pilot artifacts written to {run_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from llm_failure_prediction.config import load_config, load_gate_config
from llm_failure_prediction.gate import run_model_gate
from llm_failure_prediction.heldout import load_heldout_config, run_heldout_evaluation
from llm_failure_prediction.mlp import load_mlp_config, run_mlp_benchmark
from llm_failure_prediction.patching import load_patching_config, run_patching
from llm_failure_prediction.pilot import run_pilot
from llm_failure_prediction.probes import load_probe_config, run_probe_benchmark


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
    gate = subparsers.add_parser(
        "model-gate",
        help="Compare frozen model and answer-scoring prompt configurations",
    )
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--gate-id", help="Use a stable output directory name")
    probe = subparsers.add_parser(
        "probe",
        help="Fit confidence baselines and per-layer activation probes",
    )
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--run-id", help="Use a stable output directory name")
    mlp = subparsers.add_parser(
        "mlp",
        help="Train a regularized MLP over validation-selected activation layers",
    )
    mlp.add_argument("--config", type=Path, required=True)
    mlp.add_argument("--run-id", help="Use a stable output directory name")
    heldout = subparsers.add_parser(
        "heldout",
        help="Evaluate frozen confidence, layer, and MLP models without refitting",
    )
    heldout.add_argument("--config", type=Path, required=True)
    heldout.add_argument("--run-id", help="Use a stable output directory name")
    patch = subparsers.add_parser(
        "patch",
        help="Patch clean final-position activations into flipped perturbed prompts",
    )
    patch.add_argument("--config", type=Path, required=True)
    patch.add_argument("--run-id", help="Use a stable output directory name")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "pilot":
        config = load_config(args.config)
        if args.limit is not None:
            config.dataset.limit = args.limit
        if args.device is not None:
            config.model.device = args.device
        run_dir = run_pilot(config, run_id=args.run_id)
        print(f"Pilot artifacts written to {run_dir}")
    elif args.command == "model-gate":
        config = load_gate_config(args.config)
        gate_dir = run_model_gate(config, gate_id=args.gate_id)
        print(f"Model-gate artifacts written to {gate_dir}")
    elif args.command == "probe":
        config = load_probe_config(args.config)
        output_dir = run_probe_benchmark(config, run_id=args.run_id)
        print(f"Probe artifacts written to {output_dir}")
    elif args.command == "mlp":
        config = load_mlp_config(args.config)
        output_dir = run_mlp_benchmark(config, run_id=args.run_id)
        print(f"MLP artifacts written to {output_dir}")
    elif args.command == "heldout":
        config = load_heldout_config(args.config)
        output_dir = run_heldout_evaluation(config, run_id=args.run_id)
        print(f"Held-out artifacts written to {output_dir}")
    elif args.command == "patch":
        config = load_patching_config(args.config)
        output_dir = run_patching(config, run_id=args.run_id)
        print(f"Patching artifacts written to {output_dir}")


if __name__ == "__main__":
    main()

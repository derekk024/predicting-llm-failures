from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TARGETS = ["incorrect_hint", "reorder_choices", "irrelevant_sentence"]
TARGET_LABELS = ["Incorrect hint", "Reordered choices", "Irrelevant context"]
COLORS = {
    "entropy": "#7A8CA5",
    "logit_features": "#2D6A9F",
    "single_layer": "#D98E3D",
    "mlp": "#A63D40",
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )


def plot_heldout_auroc(results: dict, output_path: Path) -> None:
    models = ["entropy", "logit_features", "single_layer", "mlp"]
    labels = ["Entropy", "Logit features", "Single layer", "Late-layer MLP"]
    x = np.arange(len(TARGETS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for model_index, (model, label) in enumerate(zip(models, labels, strict=True)):
        values = []
        lower = []
        upper = []
        for target in TARGETS:
            model_result = results["targets"][target]["models"][model]
            value = model_result["metrics"]["auroc"]
            interval = model_result["bootstrap_intervals"]["auroc"]
            values.append(value)
            lower.append(value - interval[0])
            upper.append(interval[1] - value)
        offset = (model_index - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[model],
            yerr=np.asarray([lower, upper]),
            capsize=2.5,
            error_kw={"linewidth": 1},
        )
    ax.axhline(0.5, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x, TARGET_LABELS)
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Failure prediction on ARC-Challenge test (95% bootstrap intervals)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_layerwise_validation(probe_results: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#4C78A8", "#F58518", "#B455A0"]
    for target, label, color in zip(TARGETS, TARGET_LABELS, colors, strict=True):
        result = probe_results["targets"][target]
        layers = [item["layer_index"] for item in result["layer_probes"]]
        aurocs = [item["metrics"]["auroc"] for item in result["layer_probes"]]
        best_layer = result["best_layer"]["layer_index"]
        ax.plot(layers, aurocs, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
        ax.scatter(
            [best_layer],
            [aurocs[layers.index(best_layer)]],
            s=55,
            facecolors="white",
            edgecolors=color,
            linewidth=2,
            zorder=4,
        )
    ax.axhline(0.5, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Cached hidden-state index")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Single-state activation probes peak in late representations")
    ax.set_xticks(range(0, 28, 3))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True, sharey=True)
    for ax, target, title in zip(axes, TARGETS, TARGET_LABELS, strict=True):
        ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
        for model, label in [("logit_features", "Logit features"), ("mlp", "MLP")]:
            bins = results["targets"][target]["models"][model]["calibration_bins"]
            predicted = [item["mean_predicted_probability"] for item in bins]
            observed = [item["observed_failure_rate"] for item in bins]
            counts = np.asarray([item["count"] for item in bins])
            ax.plot(
                predicted,
                observed,
                linewidth=1.6,
                color=COLORS[model],
                label=label,
            )
            ax.scatter(
                predicted,
                observed,
                s=(3 + 5 * np.sqrt(counts / counts.max())) ** 2,
                color=COLORS[model],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Observed failure rate")
    axes[1].set_xlabel("Predicted failure probability")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Held-out calibration", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ood_transfer(arc_results: dict, mmlu_results: dict, output_path: Path) -> None:
    series = [
        (arc_results, "logit_features", "ARC logit", "#2D6A9F"),
        (arc_results, "mlp", "ARC MLP", "#A63D40"),
        (mmlu_results, "logit_features", "MMLU logit", "#79A9D1"),
        (mmlu_results, "mlp", "MMLU MLP", "#D67A7D"),
    ]
    x = np.arange(len(TARGETS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for series_index, (results, model, label, color) in enumerate(series):
        values = [
            results["targets"][target]["models"][model]["metrics"]["auroc"] for target in TARGETS
        ]
        offset = (series_index - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.axhline(0.5, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x, TARGET_LABELS)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("AUROC")
    ax.set_title("Frozen ARC-trained predictors transfer to MMLU, but activations do not lead")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-results", type=Path, required=True)
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--ood-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    heldout = _load(args.heldout_results)
    probes = _load(args.probe_results)
    plot_heldout_auroc(heldout, args.output_dir / "heldout-auroc.png")
    plot_layerwise_validation(probes, args.output_dir / "validation-layerwise-auroc.png")
    plot_calibration(heldout, args.output_dir / "heldout-calibration.png")
    if args.ood_results is not None:
        ood = _load(args.ood_results)
        plot_ood_transfer(heldout, ood, args.output_dir / "ood-transfer-auroc.png")


if __name__ == "__main__":
    main()

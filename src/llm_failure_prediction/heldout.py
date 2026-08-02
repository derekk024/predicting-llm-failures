from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch
import yaml
from pydantic import Field

from llm_failure_prediction.config import OutputConfig, StrictModel
from llm_failure_prediction.io import prepare_run_directory, write_json, write_resolved_config
from llm_failure_prediction.mlp import ActivationMLP
from llm_failure_prediction.probes import (
    BootstrapConfig,
    FailureTarget,
    _calibration_bins,
    _standardized_mean_difference,
    bootstrap_intervals,
    load_run_data,
    margin_matched_indices,
    paired_bootstrap_difference_intervals,
    paired_metric_differences,
    probability_metrics,
)


class HeldoutConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    test_run: Path
    probe_results: Path
    probe_models: Path
    mlp_results: Path
    mlp_models: Path
    targets: list[FailureTarget] = Field(min_length=1)
    bootstrap: BootstrapConfig
    calibration_bins: int = Field(default=10, gt=1)
    output: OutputConfig


def load_heldout_config(path: str | Path) -> HeldoutConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return HeldoutConfig.model_validate(raw)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _evaluate_model(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    config: HeldoutConfig,
    seed: int,
) -> dict[str, Any]:
    return {
        "metrics": probability_metrics(
            y_true,
            probabilities,
            calibration_bins=config.calibration_bins,
        ),
        "bootstrap_intervals": bootstrap_intervals(
            y_true,
            probabilities,
            samples=config.bootstrap.samples,
            confidence_level=config.bootstrap.confidence_level,
            calibration_bins=config.calibration_bins,
            seed=seed,
        ),
        "calibration_bins": _calibration_bins(
            y_true,
            probabilities,
            bins=config.calibration_bins,
        ),
    }


def _load_mlp_probabilities(
    checkpoint_path: Path,
    activations: np.ndarray,
    layer_to_position: dict[int, int],
) -> tuple[np.ndarray, list[int]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selected_layers = [int(layer) for layer in checkpoint["selected_layers"]]
    positions = [layer_to_position[layer] for layer in selected_layers]
    features = activations[:, positions, :].reshape(len(activations), -1).astype(np.float32)
    mean = checkpoint["feature_mean"].numpy()
    scale = checkpoint["feature_scale"].numpy()
    standardized = (features - mean) / scale

    model = ActivationMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(standardized))).numpy()
    return probabilities, selected_layers


def _paired_comparison(
    y_true: np.ndarray,
    candidate_probabilities: np.ndarray,
    logit_probabilities: np.ndarray,
    *,
    config: HeldoutConfig,
    seed: int,
) -> dict[str, Any]:
    return {
        "difference_definition": (
            "candidate metric minus logit-feature metric; positive favors the candidate for "
            "AUROC/AUPRC, negative favors the candidate for Brier/ECE"
        ),
        "point_differences": paired_metric_differences(
            y_true,
            candidate_probabilities,
            logit_probabilities,
            calibration_bins=config.calibration_bins,
        ),
        "bootstrap_difference_intervals": paired_bootstrap_difference_intervals(
            y_true,
            candidate_probabilities,
            logit_probabilities,
            samples=config.bootstrap.samples,
            confidence_level=config.bootstrap.confidence_level,
            calibration_bins=config.calibration_bins,
            seed=seed,
        ),
    }


def run_heldout_evaluation(
    config: HeldoutConfig,
    *,
    run_id: str | None = None,
) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = prepare_run_directory(
        config.output.root,
        run_id,
        overwrite=config.output.overwrite,
    )
    write_resolved_config(output_dir / "config.resolved.yaml", config)

    test = load_run_data(config.test_run, config.targets)
    probe_results = _load_json(config.probe_results)
    mlp_results = _load_json(config.mlp_results)
    layer_to_position = {
        int(layer_index): position for position, layer_index in enumerate(test.layer_indices)
    }
    results: dict[str, Any] = {
        "schema_version": config.schema_version,
        "run_id": run_id,
        "evaluation_type": "frozen held-out evaluation; no fitting or selection is performed",
        "test_run": str(config.test_run),
        "test_questions": len(test.question_ids),
        "probe_results": str(config.probe_results),
        "mlp_results": str(config.mlp_results),
        "targets": {},
    }

    for target_index, target in enumerate(config.targets):
        y_true = test.targets[target]
        probe_target = probe_results["targets"][target]
        mlp_target = mlp_results["targets"][target]
        seed_offset = config.seed + target_index * 10_000

        probabilities: dict[str, np.ndarray] = {
            "majority": np.full(len(y_true), probe_target["train_prevalence"]),
        }
        for name, features in {
            "entropy": test.entropy[:, None],
            "margin": test.margin[:, None],
            "logit_features": test.logit_features,
        }.items():
            model = joblib.load(config.probe_models / f"{target}__{name}.joblib")
            probabilities[name] = model.predict_proba(features)[:, 1]

        selected_layer = int(probe_target["best_layer"]["layer_index"])
        layer_model = joblib.load(config.probe_models / f"{target}__layer_{selected_layer}.joblib")
        layer_features = test.activations[
            :,
            layer_to_position[selected_layer],
            :,
        ].astype(np.float32)
        probabilities["single_layer"] = layer_model.predict_proba(layer_features)[:, 1]
        probabilities["mlp"], selected_mlp_layers = _load_mlp_probabilities(
            config.mlp_models / f"{target}__mlp.pt",
            test.activations,
            layer_to_position,
        )
        if selected_mlp_layers != mlp_target["selected_layers"]:
            raise ValueError(f"MLP layer metadata mismatch for {target}")

        target_result: dict[str, Any] = {
            "positive_count": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "selected_single_layer": selected_layer,
            "selected_mlp_layers": selected_mlp_layers,
            "models": {},
        }
        for model_index, (name, model_probabilities) in enumerate(probabilities.items()):
            target_result["models"][name] = _evaluate_model(
                y_true,
                model_probabilities,
                config=config,
                seed=seed_offset + model_index,
            )

        logit_probabilities = probabilities["logit_features"]
        target_result["single_layer_vs_logit_features"] = _paired_comparison(
            y_true,
            probabilities["single_layer"],
            logit_probabilities,
            config=config,
            seed=seed_offset + 100,
        )
        target_result["mlp_vs_logit_features"] = _paired_comparison(
            y_true,
            probabilities["mlp"],
            logit_probabilities,
            config=config,
            seed=seed_offset + 101,
        )

        correct_mask = test.originally_correct
        correct_result: dict[str, Any] = {
            "question_count": int(correct_mask.sum()),
            "positive_count": int(y_true[correct_mask].sum()),
            "models": {},
        }
        for name, model_probabilities in probabilities.items():
            correct_result["models"][name] = probability_metrics(
                y_true[correct_mask],
                model_probabilities[correct_mask],
                calibration_bins=config.calibration_bins,
            )
        target_result["originally_correct_subset"] = correct_result

        matched_indices = margin_matched_indices(y_true, test.margin)
        matched_y = y_true[matched_indices]
        matched_margin = test.margin[matched_indices]
        matched_result: dict[str, Any] = {
            "pair_count": int(len(matched_indices) // 2),
            "margin_standardized_mean_difference": _standardized_mean_difference(
                matched_y,
                matched_margin,
            ),
            "models": {},
        }
        for name, model_probabilities in probabilities.items():
            matched_metrics = probability_metrics(
                matched_y,
                model_probabilities[matched_indices],
                calibration_bins=config.calibration_bins,
            )
            matched_result["models"][name] = {
                metric: matched_metrics[metric] for metric in ("auroc", "auprc")
            }
        matched_result["single_layer_vs_logit_features"] = _paired_comparison(
            matched_y,
            probabilities["single_layer"][matched_indices],
            logit_probabilities[matched_indices],
            config=config,
            seed=seed_offset + 200,
        )
        matched_result["mlp_vs_logit_features"] = _paired_comparison(
            matched_y,
            probabilities["mlp"][matched_indices],
            logit_probabilities[matched_indices],
            config=config,
            seed=seed_offset + 201,
        )
        target_result["margin_matched_subset"] = matched_result
        results["targets"][target] = target_result
        print(
            f"Evaluated {target}: n={len(y_true)}, positives={int(y_true.sum())}, "
            f"MLP AUROC={target_result['models']['mlp']['metrics']['auroc']:.3f}",
            flush=True,
        )

    write_json(output_dir / "heldout-results.json", results)
    return output_dir

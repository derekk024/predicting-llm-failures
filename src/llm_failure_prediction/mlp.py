from __future__ import annotations

import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml
from pydantic import Field

from llm_failure_prediction.config import OutputConfig, StrictModel
from llm_failure_prediction.io import prepare_run_directory, write_json, write_resolved_config
from llm_failure_prediction.modeling import resolve_device
from llm_failure_prediction.probes import (
    BootstrapConfig,
    FailureTarget,
    _standardized_mean_difference,
    bootstrap_intervals,
    load_run_data,
    margin_matched_indices,
    paired_bootstrap_difference_intervals,
    paired_metric_differences,
    probability_metrics,
)


class MLPTrainingConfig(StrictModel):
    hidden_dim: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=64, gt=0)
    max_epochs: int = Field(default=200, gt=0)
    patience: int = Field(default=20, gt=0)
    min_delta: float = Field(default=1e-5, ge=0.0)


class MLPConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    train_run: Path
    evaluation_run: Path
    probe_results: Path
    targets: list[FailureTarget] = Field(min_length=1)
    top_k_layers: int = Field(default=3, gt=0)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    training: MLPTrainingConfig
    bootstrap: BootstrapConfig
    calibration_bins: int = Field(default=10, gt=1)
    output: OutputConfig


class ActivationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def load_mlp_config(path: str | Path) -> MLPConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return MLPConfig.model_validate(raw)


def select_top_layers(target_results: dict[str, Any], top_k: int) -> list[int]:
    layer_results = target_results["layer_probes"]
    if top_k > len(layer_results):
        raise ValueError(f"Requested {top_k} layers but only {len(layer_results)} are available")
    ranked = sorted(
        layer_results,
        key=lambda result: result["metrics"]["auroc"],
        reverse=True,
    )
    return [int(result["layer_index"]) for result in ranked[:top_k]]


def _standardize_features(
    train_features: np.ndarray,
    evaluation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_standardized = ((train_features - mean) / scale).astype(np.float32)
    evaluation_standardized = ((evaluation_features - mean) / scale).astype(np.float32)
    return train_standardized, evaluation_standardized, mean, scale


def _train_mlp(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    *,
    config: MLPConfig,
    target_seed: int,
) -> tuple[ActivationMLP, np.ndarray, dict[str, Any]]:
    random.seed(target_seed)
    np.random.seed(target_seed)
    torch.manual_seed(target_seed)
    device = resolve_device(config.device)

    train_x = torch.from_numpy(train_features).to(device)
    train_y = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    evaluation_x = torch.from_numpy(evaluation_features).to(device)
    evaluation_y = torch.from_numpy(evaluation_labels.astype(np.float32)).to(device)

    model = ActivationMLP(
        input_dim=train_features.shape[1],
        hidden_dim=config.training.hidden_dim,
        dropout=config.training.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(config.training.max_epochs):
        model.train()
        permutation = torch.randperm(len(train_x), device=device)
        batch_losses = []
        for start in range(0, len(train_x), config.training.batch_size):
            indices = permutation[start : start + config.training.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            evaluation_loss = float(loss_function(model(evaluation_x), evaluation_y).detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "evaluation_loss": evaluation_loss,
            }
        )
        if evaluation_loss < best_loss - config.training.min_delta:
            best_loss = evaluation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(evaluation_x)).float().cpu().numpy()
    model.to("cpu")
    training_summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_evaluation_loss": best_loss,
        "epochs_completed": len(history),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    del train_x, train_y, evaluation_x, evaluation_y
    if device.type == "mps":
        torch.mps.empty_cache()
    return model, probabilities, training_summary


def run_mlp_benchmark(config: MLPConfig, *, run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = prepare_run_directory(
        config.output.root,
        run_id,
        overwrite=config.output.overwrite,
    )
    write_resolved_config(output_dir / "config.resolved.yaml", config)

    train = load_run_data(config.train_run, config.targets)
    evaluation = load_run_data(config.evaluation_run, config.targets)
    if set(train.question_ids) & set(evaluation.question_ids):
        raise ValueError("Train and evaluation question IDs overlap")
    with config.probe_results.open(encoding="utf-8") as handle:
        probe_results = json.load(handle)

    results: dict[str, Any] = {
        "schema_version": config.schema_version,
        "run_id": run_id,
        "train_run": str(config.train_run),
        "evaluation_run": str(config.evaluation_run),
        "probe_results": str(config.probe_results),
        "targets": {},
    }
    layer_to_position = {
        int(layer_index): position for position, layer_index in enumerate(train.layer_indices)
    }
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for target_index, target in enumerate(config.targets):
        selected_layers = select_top_layers(
            probe_results["targets"][target],
            config.top_k_layers,
        )
        positions = [layer_to_position[layer] for layer in selected_layers]
        train_features = train.activations[:, positions, :].reshape(len(train.question_ids), -1)
        evaluation_features = evaluation.activations[:, positions, :].reshape(
            len(evaluation.question_ids), -1
        )
        train_features, evaluation_features, feature_mean, feature_scale = _standardize_features(
            train_features.astype(np.float32),
            evaluation_features.astype(np.float32),
        )
        model, probabilities, training_summary = _train_mlp(
            train_features,
            train.targets[target],
            evaluation_features,
            evaluation.targets[target],
            config=config,
            target_seed=config.seed + target_index,
        )

        checkpoint_path = models_dir / f"{target}__mlp.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "selected_layers": selected_layers,
                "feature_mean": torch.from_numpy(feature_mean),
                "feature_scale": torch.from_numpy(feature_scale),
                "input_dim": train_features.shape[1],
                "hidden_dim": config.training.hidden_dim,
                "dropout": config.training.dropout,
            },
            checkpoint_path,
        )

        evaluation_y = evaluation.targets[target]
        probe_models_dir = config.probe_results.parent / "models"
        logit_model = joblib.load(probe_models_dir / f"{target}__logit_features.joblib")
        logit_probabilities = logit_model.predict_proba(evaluation.logit_features)[:, 1]
        seed_offset = config.seed + target_index * 10_000
        target_result: dict[str, Any] = {
            "selected_layers": selected_layers,
            "training": training_summary,
            "metrics": probability_metrics(
                evaluation_y,
                probabilities,
                calibration_bins=config.calibration_bins,
            ),
            "bootstrap_intervals": bootstrap_intervals(
                evaluation_y,
                probabilities,
                samples=config.bootstrap.samples,
                confidence_level=config.bootstrap.confidence_level,
                calibration_bins=config.calibration_bins,
                seed=seed_offset,
            ),
            "mlp_vs_logit_features": {
                "difference_definition": (
                    "MLP metric minus logit-feature metric; positive favors the MLP for "
                    "AUROC/AUPRC, negative favors the MLP for Brier/ECE"
                ),
                "point_differences": paired_metric_differences(
                    evaluation_y,
                    probabilities,
                    logit_probabilities,
                    calibration_bins=config.calibration_bins,
                ),
                "bootstrap_difference_intervals": paired_bootstrap_difference_intervals(
                    evaluation_y,
                    probabilities,
                    logit_probabilities,
                    samples=config.bootstrap.samples,
                    confidence_level=config.bootstrap.confidence_level,
                    calibration_bins=config.calibration_bins,
                    seed=seed_offset + 1,
                ),
            },
        }

        correct_mask = evaluation.originally_correct
        target_result["originally_correct_subset"] = {
            "question_count": int(correct_mask.sum()),
            "positive_count": int(evaluation_y[correct_mask].sum()),
            "mlp_metrics": probability_metrics(
                evaluation_y[correct_mask],
                probabilities[correct_mask],
                calibration_bins=config.calibration_bins,
            ),
            "logit_feature_metrics": probability_metrics(
                evaluation_y[correct_mask],
                logit_probabilities[correct_mask],
                calibration_bins=config.calibration_bins,
            ),
        }

        matched_indices = margin_matched_indices(evaluation_y, evaluation.margin)
        matched_y = evaluation_y[matched_indices]
        matched_margin = evaluation.margin[matched_indices]
        mlp_matched = probability_metrics(
            matched_y,
            probabilities[matched_indices],
            calibration_bins=config.calibration_bins,
        )
        logit_matched = probability_metrics(
            matched_y,
            logit_probabilities[matched_indices],
            calibration_bins=config.calibration_bins,
        )
        target_result["margin_matched_subset"] = {
            "pair_count": int(len(matched_indices) // 2),
            "margin_standardized_mean_difference": _standardized_mean_difference(
                matched_y,
                matched_margin,
            ),
            "mlp": {name: mlp_matched[name] for name in ("auroc", "auprc")},
            "logit_features": {name: logit_matched[name] for name in ("auroc", "auprc")},
        }
        results["targets"][target] = target_result
        print(
            f"Finished {target}: layers={selected_layers}, "
            f"best_epoch={training_summary['best_epoch']}, "
            f"AUROC={target_result['metrics']['auroc']:.3f}",
            flush=True,
        )

    write_json(output_dir / "mlp-results.json", results)
    return output_dir

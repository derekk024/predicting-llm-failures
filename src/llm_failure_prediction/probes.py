from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import yaml
from pydantic import Field, model_validator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llm_failure_prediction.config import OutputConfig, StrictModel
from llm_failure_prediction.io import prepare_run_directory, write_json, write_resolved_config
from llm_failure_prediction.schema import ANSWER_LABELS, QuestionRecord

FailureTarget = Literal["incorrect_hint", "reorder_choices", "irrelevant_sentence"]


class LogisticConfig(StrictModel):
    c: float = Field(default=1.0, gt=0.0)
    max_iter: int = Field(default=2000, gt=0)


class BootstrapConfig(StrictModel):
    samples: int = Field(default=1000, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class ProbeConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    train_run: Path
    evaluation_run: Path
    targets: list[FailureTarget] = Field(min_length=1)
    logistic: LogisticConfig
    bootstrap: BootstrapConfig
    calibration_bins: int = Field(default=10, gt=1)
    output: OutputConfig

    @model_validator(mode="after")
    def validate_unique_targets(self) -> ProbeConfig:
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("Probe targets must be unique")
        return self


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    question_ids: np.ndarray
    activations: np.ndarray
    logit_features: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    originally_correct: np.ndarray
    targets: dict[FailureTarget, np.ndarray]
    layer_indices: np.ndarray


def load_probe_config(path: str | Path) -> ProbeConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProbeConfig.model_validate(raw)


def _read_records(path: Path) -> list[QuestionRecord]:
    with path.open(encoding="utf-8") as handle:
        return [QuestionRecord.model_validate_json(line) for line in handle if line.strip()]


def _clean_logit_features(
    records: list[QuestionRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_logits = np.asarray(
        [[record.runs[0].answer_logits[label] for label in ANSWER_LABELS] for record in records],
        dtype=np.float32,
    )
    centered_logits = raw_logits - raw_logits.mean(axis=1, keepdims=True)
    entropy = np.asarray([record.runs[0].entropy for record in records], dtype=np.float32)
    margin = np.asarray(
        [record.runs[0].top_two_logit_margin for record in records],
        dtype=np.float32,
    )
    features = np.column_stack([centered_logits, entropy, margin]).astype(np.float32)
    return features, entropy, margin


def load_run_data(run_dir: str | Path, targets: list[FailureTarget]) -> RunData:
    run_path = Path(run_dir)
    records = _read_records(run_path / "records.jsonl")
    with np.load(run_path / "activations.npz") as cache:
        activations = cache["activations"]
        cached_ids = cache["question_ids"].astype(str)
        layer_indices = cache["layer_indices"].astype(np.int64)

    record_ids = np.asarray([record.question.question_id for record in records])
    if not np.array_equal(cached_ids, record_ids):
        raise ValueError(f"Activation rows are not aligned with records in {run_path}")
    if activations.ndim != 3 or activations.shape[0] != len(records):
        raise ValueError(f"Unexpected activation shape {activations.shape} in {run_path}")
    if not np.isfinite(activations).all():
        raise ValueError(f"Activation cache contains non-finite values in {run_path}")

    logit_features, entropy, margin = _clean_logit_features(records)
    target_arrays: dict[FailureTarget, np.ndarray] = {}
    for target in targets:
        values = []
        for record in records:
            run = next((run for run in record.runs if run.perturbation == target), None)
            if run is None or run.changed_from_clean is None:
                raise ValueError(
                    f"Missing target {target} for question {record.question.question_id}"
                )
            values.append(run.changed_from_clean)
        target_arrays[target] = np.asarray(values, dtype=np.int8)

    return RunData(
        run_dir=run_path,
        question_ids=record_ids,
        activations=activations,
        logit_features=logit_features,
        entropy=entropy,
        margin=margin,
        originally_correct=np.asarray([record.runs[0].is_correct for record in records]),
        targets=target_arrays,
        layer_indices=layer_indices,
    )


def _ece(y_true: np.ndarray, probabilities: np.ndarray, *, bins: int) -> float:
    bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def _calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> list[dict[str, float | int]]:
    bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    result = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            result.append(
                {
                    "bin": bin_id,
                    "count": int(mask.sum()),
                    "mean_predicted_probability": float(probabilities[mask].mean()),
                    "observed_failure_rate": float(y_true[mask].mean()),
                }
            )
    return result


def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    calibration_bins: int,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64).clip(0.0, 1.0)
    has_both_classes = np.unique(y_true).size == 2
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)) if has_both_classes else None,
        "auprc": (
            float(average_precision_score(y_true, probabilities)) if has_both_classes else None
        ),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": _ece(y_true, probabilities, bins=calibration_bins),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    calibration_bins: int,
    seed: int,
) -> dict[str, list[float] | None]:
    if samples == 0:
        return {name: None for name in ("auroc", "auprc", "brier", "ece")}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {name: [] for name in ("auroc", "auprc", "brier", "ece")}
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        metrics = probability_metrics(
            y_true[indices],
            probabilities[indices],
            calibration_bins=calibration_bins,
        )
        for name, value in metrics.items():
            if value is not None:
                collected[name].append(value)

    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, list[float] | None] = {}
    for name, values in collected.items():
        intervals[name] = (
            [float(x) for x in np.quantile(values, [alpha, 1.0 - alpha])] if values else None
        )
    return intervals


def _fit_logistic(config: LogisticConfig, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=config.c,
                    max_iter=config.max_iter,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def _evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    config: ProbeConfig,
    bootstrap_seed: int,
    with_intervals: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metrics": probability_metrics(
            y_true,
            probabilities,
            calibration_bins=config.calibration_bins,
        )
    }
    if with_intervals:
        result["bootstrap_intervals"] = bootstrap_intervals(
            y_true,
            probabilities,
            samples=config.bootstrap.samples,
            confidence_level=config.bootstrap.confidence_level,
            calibration_bins=config.calibration_bins,
            seed=bootstrap_seed,
        )
        result["calibration_bins"] = _calibration_bins(
            y_true,
            probabilities,
            bins=config.calibration_bins,
        )
    return result


def _fit_and_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    *,
    config: ProbeConfig,
) -> tuple[Pipeline, np.ndarray]:
    pipeline = _fit_logistic(config.logistic, config.seed)
    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(evaluation_x)[:, 1]
    return pipeline, probabilities


def _validate_splits(train: RunData, evaluation: RunData) -> None:
    overlap = set(train.question_ids) & set(evaluation.question_ids)
    if overlap:
        raise ValueError(f"Train and evaluation runs share {len(overlap)} question IDs")
    if train.activations.shape[1:] != evaluation.activations.shape[1:]:
        raise ValueError(
            "Train and evaluation activation shapes differ: "
            f"{train.activations.shape[1:]} vs {evaluation.activations.shape[1:]}"
        )
    if not np.array_equal(train.layer_indices, evaluation.layer_indices):
        raise ValueError("Train and evaluation layer indices differ")


def run_probe_benchmark(
    config: ProbeConfig,
    *,
    run_id: str | None = None,
) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = prepare_run_directory(
        config.output.root,
        run_id,
        overwrite=config.output.overwrite,
    )
    write_resolved_config(output_dir / "config.resolved.yaml", config)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    train = load_run_data(config.train_run, config.targets)
    evaluation = load_run_data(config.evaluation_run, config.targets)
    _validate_splits(train, evaluation)

    results: dict[str, Any] = {
        "schema_version": config.schema_version,
        "run_id": run_id,
        "train_run": str(config.train_run),
        "evaluation_run": str(config.evaluation_run),
        "train_questions": len(train.question_ids),
        "evaluation_questions": len(evaluation.question_ids),
        "activation_shape_train": list(train.activations.shape),
        "logit_feature_names": [
            "centered_logit_A",
            "centered_logit_B",
            "centered_logit_C",
            "centered_logit_D",
            "entropy",
            "top_two_logit_margin",
        ],
        "targets": {},
    }

    for target_index, target in enumerate(config.targets):
        train_y = train.targets[target]
        evaluation_y = evaluation.targets[target]
        if np.unique(train_y).size != 2:
            raise ValueError(f"Training target {target} does not contain both classes")

        target_result: dict[str, Any] = {
            "train_positive_count": int(train_y.sum()),
            "train_prevalence": float(train_y.mean()),
            "evaluation_positive_count": int(evaluation_y.sum()),
            "evaluation_prevalence": float(evaluation_y.mean()),
            "baselines": {},
            "layer_probes": [],
        }
        seed_offset = config.seed + target_index * 10_000
        majority_probabilities = np.full(len(evaluation_y), train_y.mean())
        target_result["baselines"]["majority"] = _evaluate_probabilities(
            evaluation_y,
            majority_probabilities,
            config=config,
            bootstrap_seed=seed_offset,
            with_intervals=True,
        )

        baseline_specs = {
            "entropy": (train.entropy[:, None], evaluation.entropy[:, None]),
            "margin": (train.margin[:, None], evaluation.margin[:, None]),
            "logit_features": (train.logit_features, evaluation.logit_features),
        }
        baseline_probabilities: dict[str, np.ndarray] = {}
        for baseline_index, (name, (train_x, evaluation_x)) in enumerate(
            baseline_specs.items(),
            start=1,
        ):
            fitted, probabilities = _fit_and_score(
                train_x,
                train_y,
                evaluation_x,
                config=config,
            )
            baseline_probabilities[name] = probabilities
            target_result["baselines"][name] = _evaluate_probabilities(
                evaluation_y,
                probabilities,
                config=config,
                bootstrap_seed=seed_offset + baseline_index,
                with_intervals=True,
            )
            joblib.dump(fitted, models_dir / f"{target}__{name}.joblib")

        layer_models: list[Pipeline] = []
        layer_probabilities: list[np.ndarray] = []
        for layer_position, layer_index in enumerate(train.layer_indices):
            fitted, probabilities = _fit_and_score(
                train.activations[:, layer_position, :].astype(np.float32),
                train_y,
                evaluation.activations[:, layer_position, :].astype(np.float32),
                config=config,
            )
            metrics = probability_metrics(
                evaluation_y,
                probabilities,
                calibration_bins=config.calibration_bins,
            )
            target_result["layer_probes"].append(
                {
                    "layer_index": int(layer_index),
                    "metrics": metrics,
                }
            )
            layer_models.append(fitted)
            layer_probabilities.append(probabilities)

        best_position = max(
            range(len(layer_models)),
            key=lambda index: target_result["layer_probes"][index]["metrics"]["auroc"],
        )
        best_layer = int(train.layer_indices[best_position])
        best_probabilities = layer_probabilities[best_position]
        target_result["best_layer"] = {
            "layer_index": best_layer,
            **_evaluate_probabilities(
                evaluation_y,
                best_probabilities,
                config=config,
                bootstrap_seed=seed_offset + 100,
                with_intervals=True,
            ),
        }
        joblib.dump(
            layer_models[best_position],
            models_dir / f"{target}__layer_{best_layer}.joblib",
        )

        correct_mask = evaluation.originally_correct
        correct_result: dict[str, Any] = {
            "question_count": int(correct_mask.sum()),
            "positive_count": int(evaluation_y[correct_mask].sum()),
            "baselines": {},
        }
        for name, probabilities in {
            "majority": majority_probabilities,
            **baseline_probabilities,
        }.items():
            correct_result["baselines"][name] = _evaluate_probabilities(
                evaluation_y[correct_mask],
                probabilities[correct_mask],
                config=config,
                bootstrap_seed=seed_offset + 200,
                with_intervals=False,
            )
        correct_result["best_layer"] = {
            "layer_index": best_layer,
            **_evaluate_probabilities(
                evaluation_y[correct_mask],
                best_probabilities[correct_mask],
                config=config,
                bootstrap_seed=seed_offset + 201,
                with_intervals=False,
            ),
        }
        target_result["originally_correct_subset"] = correct_result
        results["targets"][target] = target_result
        print(
            f"Finished {target}: best layer {best_layer}, "
            f"AUROC={target_result['best_layer']['metrics']['auroc']:.3f}",
            flush=True,
        )

    write_json(output_dir / "probe-results.json", results)
    return output_dir

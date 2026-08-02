from pathlib import Path

import numpy as np

from llm_failure_prediction.probes import (
    bootstrap_intervals,
    load_probe_config,
    probability_metrics,
)


def test_probe_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "probes_arc_validation.yaml"
    config = load_probe_config(path)
    assert config.targets == ["incorrect_hint", "reorder_choices", "irrelevant_sentence"]
    assert config.bootstrap.samples == 1000
    assert config.train_run.name == "qwen3-1.7b_seed42"


def test_probability_metrics_are_calibrated_probability_metrics() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = probability_metrics(y_true, probabilities, calibration_bins=5)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert np.isclose(metrics["brier"], 0.025)
    assert np.isclose(metrics["ece"], 0.15)


def test_bootstrap_intervals_are_deterministic() -> None:
    y_true = np.asarray([0, 0, 0, 1, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    kwargs = {
        "samples": 50,
        "confidence_level": 0.95,
        "calibration_bins": 5,
        "seed": 42,
    }
    first = bootstrap_intervals(y_true, probabilities, **kwargs)
    second = bootstrap_intervals(y_true, probabilities, **kwargs)
    assert first == second
    assert first["auroc"] == [1.0, 1.0]

from pathlib import Path

import numpy as np

from llm_failure_prediction.patching import (
    load_patching_config,
    paired_mean_effect,
    semantic_logit_margin,
)
from llm_failure_prediction.schema import ScoredRun


def test_patching_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "patching_arc_test.yaml"
    config = load_patching_config(path)
    assert config.layer_by_target["reorder_choices"] == 27
    assert config.max_examples_per_target == 67
    assert config.max_sanity_logit_difference == 0.01
    assert config.bootstrap.samples == 1000


def test_semantic_margin_respects_reordered_prompt_labels() -> None:
    run = ScoredRun.model_validate(
        {
            "perturbation": "reorder_choices",
            "prompt": "prompt",
            "metadata": {
                "prompt_to_semantic": {"A": "D", "B": "A", "C": "B", "D": "C"},
            },
            "answer_logits": {"A": 0.0, "B": 3.0, "C": 1.0, "D": 2.0},
            "answer_probabilities": {"A": 0.1, "B": 0.6, "C": 0.1, "D": 0.2},
            "predicted_prompt_label": "B",
            "predicted_semantic_label": "A",
            "entropy": 1.0,
            "top_two_logit_margin": 1.0,
            "is_correct": True,
            "changed_from_clean": False,
            "sequence_length": 10,
        }
    )
    logits = np.asarray([0.0, 3.0, 1.0, 2.0])
    assert semantic_logit_margin(logits, run, "A") == 1.0


def test_paired_mean_effect_is_deterministic() -> None:
    candidate = np.asarray([1.0, 1.0, 0.0, 1.0])
    control = np.asarray([0.0, 1.0, 0.0, 0.0])
    kwargs = {"samples": 100, "confidence_level": 0.95, "seed": 42}
    first = paired_mean_effect(candidate, control, **kwargs)
    second = paired_mean_effect(candidate, control, **kwargs)
    assert first == second
    assert first["point_difference"] == 0.5
    assert first["bootstrap_interval"][0] >= 0.0

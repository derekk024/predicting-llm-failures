import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from llm_failure_prediction.patching import (
    _select_examples,
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

    sensitivity_path = (
        Path(__file__).parents[1] / "configs" / "patching_arc_test_reorder_preterminal.yaml"
    )
    sensitivity = load_patching_config(sensitivity_path)
    assert sensitivity.targets == ["reorder_choices"]
    assert sensitivity.layer_by_target["reorder_choices"] == 26
    assert sensitivity.selection_run is not None


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


def test_selection_run_reuses_question_ids_in_order(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"question_id": "q2", "target": "reorder_choices"},
                    {"question_id": "q1", "target": "reorder_choices"},
                ]
            }
        ),
        encoding="utf-8",
    )
    base = load_patching_config(Path(__file__).parents[1] / "configs" / "patching_arc_test.yaml")
    base.targets = ["reorder_choices"]
    base.layer_by_target = {"reorder_choices": 26}
    base.max_examples_per_target = 2
    base.selection_run = selection_path
    records = []
    for question_id in ("q1", "q2"):
        run = SimpleNamespace(perturbation="reorder_choices", changed_from_clean=True)
        records.append(
            SimpleNamespace(
                question=SimpleNamespace(question_id=question_id),
                runs=[run],
            )
        )
    selected = _select_examples(records, base)
    assert [records[index].question.question_id for index, _, _ in selected] == ["q2", "q1"]

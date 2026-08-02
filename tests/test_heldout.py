from pathlib import Path

import numpy as np
import torch

from llm_failure_prediction.heldout import _load_mlp_probabilities, load_heldout_config
from llm_failure_prediction.mlp import ActivationMLP


def test_heldout_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "heldout_arc_test.yaml"
    config = load_heldout_config(path)
    assert config.bootstrap.samples == 1000
    assert config.test_run.name == "qwen3-1.7b_seed42"
    assert config.targets == ["incorrect_hint", "reorder_choices", "irrelevant_sentence"]


def test_frozen_mlp_loading_does_not_refit(tmp_path: Path) -> None:
    model = ActivationMLP(input_dim=4, hidden_dim=2, dropout=0.0)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = {
        "state_dict": model.state_dict(),
        "selected_layers": [0, 1],
        "feature_mean": torch.zeros(4),
        "feature_scale": torch.ones(4),
        "input_dim": 4,
        "hidden_dim": 2,
        "dropout": 0.0,
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    activations = np.ones((3, 2, 2), dtype=np.float32)

    probabilities, layers = _load_mlp_probabilities(
        checkpoint_path,
        activations,
        {0: 0, 1: 1},
    )
    assert layers == [0, 1]
    assert np.allclose(probabilities, 0.5)

from pathlib import Path

import torch

from llm_failure_prediction.mlp import ActivationMLP, load_mlp_config, select_top_layers


def test_mlp_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "mlp_arc_validation.yaml"
    config = load_mlp_config(path)
    assert config.top_k_layers == 3
    assert config.training.hidden_dim == 64
    assert config.device == "auto"


def test_select_top_layers_orders_by_validation_auroc() -> None:
    result = {
        "layer_probes": [
            {"layer_index": 0, "metrics": {"auroc": 0.6}},
            {"layer_index": 1, "metrics": {"auroc": 0.8}},
            {"layer_index": 2, "metrics": {"auroc": 0.7}},
        ]
    }
    assert select_top_layers(result, 2) == [1, 2]


def test_activation_mlp_returns_one_logit_per_example() -> None:
    model = ActivationMLP(input_dim=12, hidden_dim=4, dropout=0.1)
    logits = model(torch.zeros(5, 12))
    assert logits.shape == (5,)

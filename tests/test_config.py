from pathlib import Path

from llm_failure_prediction.config import load_config


def test_pilot_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "pilot_arc.yaml"
    config = load_config(path)
    assert config.dataset.limit == 200
    assert config.model.enable_thinking is False
    assert config.model.name == "Qwen/Qwen3-0.6B"
    assert config.model.dtype == "bfloat16"

from pathlib import Path

from llm_failure_prediction.config import load_config, load_gate_config


def test_pilot_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "pilot_arc.yaml"
    config = load_config(path)
    assert config.dataset.limit == 200
    assert config.model.enable_thinking is False
    assert config.model.name == "Qwen/Qwen3-0.6B"
    assert config.model.dtype == "bfloat16"
    assert config.prompt.answer_candidates == {"A": "A", "B": "B", "C": "C", "D": "D"}


def test_model_gate_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "prompt_model_gate.yaml"
    config = load_gate_config(path)
    assert config.dataset.split == "validation"
    assert config.dataset.limit == 100
    assert [model.id for model in config.models] == ["qwen3-0.6b", "qwen3-1.7b"]
    assert config.prompt_formats[1].answer_candidates["A"] == " A"


def test_frozen_extraction_configs_cover_official_splits() -> None:
    config_dir = Path(__file__).parents[1] / "configs" / "extraction"
    expected = {"train": 1117, "validation": 295, "test": 1165}
    observed = {}
    for path in config_dir.glob("*.yaml"):
        config = load_config(path)
        observed[config.dataset.split] = config.dataset.limit
        assert config.model.name == "Qwen/Qwen3-1.7B"
        assert config.prompt.assistant_prefill == ""
    assert observed == expected

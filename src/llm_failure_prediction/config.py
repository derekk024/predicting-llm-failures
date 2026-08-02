from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_failure_prediction.schema import ANSWER_LABELS, AnswerLabel


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    format: Literal["arc", "mmlu"] = "arc"
    path: str
    name: str
    split: str
    revision: str | None = None
    limit: int = Field(gt=0)
    shuffle: bool = True


class ModelConfig(StrictModel):
    name: str
    revision: str | None = None
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    enable_thinking: bool = False


class PromptConfig(StrictModel):
    system: str
    answer_instruction: str
    assistant_prefill: str = ""
    answer_candidates: dict[AnswerLabel, str] = Field(
        default_factory=lambda: dict(zip(ANSWER_LABELS, ANSWER_LABELS, strict=True))
    )

    @model_validator(mode="after")
    def validate_answer_candidates(self) -> PromptConfig:
        if set(self.answer_candidates) != set(ANSWER_LABELS):
            raise ValueError(f"answer_candidates must have exactly the keys {ANSWER_LABELS}")
        if any(not candidate for candidate in self.answer_candidates.values()):
            raise ValueError("answer candidate strings must be nonempty")
        return self


class HintConfig(StrictModel):
    enabled: bool = True
    template: str


class ReorderConfig(StrictModel):
    enabled: bool = True


class IrrelevantConfig(StrictModel):
    enabled: bool = True
    template: str


class PerturbationConfig(StrictModel):
    incorrect_hint: HintConfig
    reorder_choices: ReorderConfig
    irrelevant_sentence: IrrelevantConfig


class RuntimeConfig(StrictModel):
    activation_dtype: Literal["float16", "float32"] = "float16"
    deterministic_algorithms: bool = False


class OutputConfig(StrictModel):
    root: Path
    overwrite: bool = False


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    prompt: PromptConfig
    perturbations: PerturbationConfig
    runtime: RuntimeConfig
    output: OutputConfig

    @model_validator(mode="after")
    def require_a_perturbation(self) -> ExperimentConfig:
        enabled = (
            self.perturbations.incorrect_hint.enabled,
            self.perturbations.reorder_choices.enabled,
            self.perturbations.irrelevant_sentence.enabled,
        )
        if not any(enabled):
            raise ValueError("At least one perturbation must be enabled")
        return self


class NamedModelConfig(ModelConfig):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")


class NamedPromptConfig(PromptConfig):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")


class ModelGateConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    dataset: DatasetConfig
    models: list[NamedModelConfig] = Field(min_length=1)
    prompt_formats: list[NamedPromptConfig] = Field(min_length=1)
    perturbations: PerturbationConfig
    runtime: RuntimeConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ModelGateConfig:
        model_ids = [model.id for model in self.models]
        prompt_ids = [prompt.id for prompt in self.prompt_formats]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model IDs must be unique")
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("prompt format IDs must be unique")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)


def load_gate_config(path: str | Path) -> ModelGateConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ModelGateConfig.model_validate(raw)

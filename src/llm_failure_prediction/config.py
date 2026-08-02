from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
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


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)

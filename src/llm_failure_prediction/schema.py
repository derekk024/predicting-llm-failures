from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANSWER_LABELS = ("A", "B", "C", "D")
AnswerLabel = Literal["A", "B", "C", "D"]
PerturbationName = Literal["clean", "incorrect_hint", "reorder_choices", "irrelevant_sentence"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Choice(StrictModel):
    semantic_label: AnswerLabel
    text: str


class Question(StrictModel):
    dataset_path: str
    dataset_name: str
    dataset_split: str
    question_id: str
    question: str
    choices: list[Choice]
    correct_semantic_label: AnswerLabel

    @model_validator(mode="after")
    def validate_choices(self) -> Question:
        labels = tuple(choice.semantic_label for choice in self.choices)
        if labels != ANSWER_LABELS:
            raise ValueError(f"Choices must be normalized and ordered as {ANSWER_LABELS}")
        return self


class PerturbationMetadata(StrictModel):
    hint_semantic_label: AnswerLabel | None = None
    prompt_to_semantic: dict[AnswerLabel, AnswerLabel]
    donor_question_id: str | None = None
    donor_question: str | None = None


class ScoredRun(StrictModel):
    perturbation: PerturbationName
    prompt: str
    metadata: PerturbationMetadata
    answer_logits: dict[AnswerLabel, float]
    answer_probabilities: dict[AnswerLabel, float]
    predicted_prompt_label: AnswerLabel
    predicted_semantic_label: AnswerLabel
    entropy: float = Field(ge=0.0)
    top_two_logit_margin: float = Field(ge=0.0)
    is_correct: bool
    changed_from_clean: bool | None
    sequence_length: int = Field(gt=0)


class QuestionRecord(StrictModel):
    schema_version: Literal[1]
    run_id: str
    activation_row: int = Field(ge=0)
    question: Question
    runs: list[ScoredRun]

    @model_validator(mode="after")
    def validate_runs(self) -> QuestionRecord:
        if not self.runs or self.runs[0].perturbation != "clean":
            raise ValueError("The first run must be the clean run")
        names = [run.perturbation for run in self.runs]
        if len(names) != len(set(names)):
            raise ValueError("Each perturbation may appear at most once")
        if self.runs[0].changed_from_clean is not None:
            raise ValueError("changed_from_clean must be null for the clean run")
        return self

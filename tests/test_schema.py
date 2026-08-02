import pytest
from pydantic import ValidationError

from llm_failure_prediction.schema import Choice, Question


def test_question_requires_normalized_choice_order() -> None:
    with pytest.raises(ValidationError):
        Question(
            dataset_path="dataset",
            dataset_name="config",
            dataset_split="train",
            question_id="q1",
            question="Question?",
            choices=[
                Choice(semantic_label="B", text="b"),
                Choice(semantic_label="A", text="a"),
                Choice(semantic_label="C", text="c"),
                Choice(semantic_label="D", text="d"),
            ],
            correct_semantic_label="A",
        )

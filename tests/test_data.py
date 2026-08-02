from llm_failure_prediction.data import normalize_arc_example, normalize_mmlu_example


def test_normalizes_non_letter_source_labels() -> None:
    example = {
        "id": "q1",
        "question": "Which choice?",
        "choices": {
            "label": ["1", "2", "3", "4"],
            "text": ["one", "two", "three", "four"],
        },
        "answerKey": "3",
    }
    question = normalize_arc_example(
        example,
        dataset_path="allenai/ai2_arc",
        dataset_name="ARC-Challenge",
        dataset_split="train",
    )
    assert question is not None
    assert question.correct_semantic_label == "C"
    assert [choice.semantic_label for choice in question.choices] == ["A", "B", "C", "D"]


def test_skips_questions_without_four_choices() -> None:
    example = {
        "id": "q2",
        "question": "Which choice?",
        "choices": {"label": ["A", "B", "C"], "text": ["a", "b", "c"]},
        "answerKey": "A",
    }
    assert (
        normalize_arc_example(
            example,
            dataset_path="allenai/ai2_arc",
            dataset_name="ARC-Challenge",
            dataset_split="train",
        )
        is None
    )


def test_normalizes_mmlu_example_with_stable_source_id() -> None:
    example = {
        "question": "Which answer?",
        "subject": "test_subject",
        "choices": ["one", "two", "three", "four"],
        "answer": 2,
    }
    question = normalize_mmlu_example(
        example,
        source_index=17,
        dataset_path="cais/mmlu",
        dataset_name="all",
        dataset_split="test",
    )
    assert question is not None
    assert question.question_id == "mmlu:test_subject:17"
    assert question.correct_semantic_label == "C"

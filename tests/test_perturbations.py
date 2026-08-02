from llm_failure_prediction.config import PerturbationConfig
from llm_failure_prediction.perturbations import build_variants
from llm_failure_prediction.schema import Choice, Question


def make_question(question_id: str, correct: str = "B") -> Question:
    return Question(
        dataset_path="allenai/ai2_arc",
        dataset_name="ARC-Challenge",
        dataset_split="train",
        question_id=question_id,
        question="What is correct?",
        choices=[
            Choice(semantic_label="A", text="alpha"),
            Choice(semantic_label="B", text="beta"),
            Choice(semantic_label="C", text="gamma"),
            Choice(semantic_label="D", text="delta"),
        ],
        correct_semantic_label=correct,
    )


def make_config() -> PerturbationConfig:
    return PerturbationConfig.model_validate(
        {
            "incorrect_hint": {
                "enabled": True,
                "template": "Someone says {hint_label}.",
            },
            "reorder_choices": {"enabled": True},
            "irrelevant_sentence": {
                "enabled": True,
                "template": "Unrelated: {donor_question}",
            },
        }
    )


def test_variants_are_deterministic_and_semantically_mapped() -> None:
    question = make_question("q1")
    donor = make_question("q2")
    first = build_variants(question, donor=donor, config=make_config(), seed=42)
    second = build_variants(question, donor=donor, config=make_config(), seed=42)
    assert first == second
    assert [variant.name for variant in first] == [
        "clean",
        "incorrect_hint",
        "reorder_choices",
        "irrelevant_sentence",
    ]
    assert first[1].metadata.hint_semantic_label != question.correct_semantic_label
    assert first[2].metadata.prompt_to_semantic != {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
    }
    for prompt_label, semantic_label in first[2].metadata.prompt_to_semantic.items():
        index = ("A", "B", "C", "D").index(prompt_label)
        expected_text = next(
            choice.text for choice in question.choices if choice.semantic_label == semantic_label
        )
        assert first[2].displayed_choices[index] == expected_text
    assert first[3].metadata.donor_question_id == "q2"

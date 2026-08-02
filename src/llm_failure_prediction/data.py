from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset

from llm_failure_prediction.config import DatasetConfig
from llm_failure_prediction.schema import ANSWER_LABELS, Choice, Question


def normalize_arc_example(
    example: Mapping[str, Any],
    *,
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
) -> Question | None:
    labels = [str(label) for label in example["choices"]["label"]]
    texts = [str(text).strip() for text in example["choices"]["text"]]
    if len(labels) != 4 or len(texts) != 4 or len(set(labels)) != 4:
        return None

    source_answer = str(example["answerKey"])
    if source_answer not in labels:
        return None

    source_to_semantic = dict(zip(labels, ANSWER_LABELS, strict=True))
    choices = [
        Choice(semantic_label=semantic_label, text=text)
        for semantic_label, text in zip(ANSWER_LABELS, texts, strict=True)
    ]
    return Question(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        question_id=str(example["id"]),
        question=str(example["question"]).strip(),
        choices=choices,
        correct_semantic_label=source_to_semantic[source_answer],
    )


def normalize_arc_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
) -> list[Question]:
    normalized = (
        normalize_arc_example(
            example,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
        for example in examples
    )
    return [question for question in normalized if question is not None]


def load_arc_questions(config: DatasetConfig, *, seed: int) -> list[Question]:
    dataset = load_dataset(
        config.path,
        config.name,
        split=config.split,
        revision=config.revision,
    )
    if config.shuffle:
        dataset = dataset.shuffle(seed=seed)

    questions: list[Question] = []
    for example in dataset:
        normalized = normalize_arc_example(
            example,
            dataset_path=config.path,
            dataset_name=config.name,
            dataset_split=config.split,
        )
        if normalized is not None:
            questions.append(normalized)
        if len(questions) == config.limit:
            break

    if len(questions) < config.limit:
        raise ValueError(
            f"Requested {config.limit} four-choice questions but found only {len(questions)}"
        )
    return questions


def normalize_mmlu_example(
    example: Mapping[str, Any],
    *,
    source_index: int,
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
) -> Question | None:
    texts = [str(text).strip() for text in example["choices"]]
    answer_index = int(example["answer"])
    if len(texts) != 4 or answer_index not in range(4):
        return None
    subject = str(example["subject"])
    return Question(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        question_id=f"mmlu:{subject}:{source_index}",
        question=str(example["question"]).strip(),
        choices=[
            Choice(semantic_label=label, text=text)
            for label, text in zip(ANSWER_LABELS, texts, strict=True)
        ],
        correct_semantic_label=ANSWER_LABELS[answer_index],
    )


def load_mmlu_questions(config: DatasetConfig, *, seed: int) -> list[Question]:
    dataset = load_dataset(
        config.path,
        config.name,
        split=config.split,
        revision=config.revision,
    )
    dataset = dataset.add_column("_source_index", list(range(len(dataset))))
    if config.shuffle:
        dataset = dataset.shuffle(seed=seed)

    questions: list[Question] = []
    for example in dataset:
        normalized = normalize_mmlu_example(
            example,
            source_index=int(example["_source_index"]),
            dataset_path=config.path,
            dataset_name=config.name,
            dataset_split=config.split,
        )
        if normalized is not None:
            questions.append(normalized)
        if len(questions) == config.limit:
            break
    if len(questions) < config.limit:
        raise ValueError(
            f"Requested {config.limit} four-choice questions but found only {len(questions)}"
        )
    return questions


def load_questions(config: DatasetConfig, *, seed: int) -> list[Question]:
    if config.format == "arc":
        return load_arc_questions(config, seed=seed)
    if config.format == "mmlu":
        return load_mmlu_questions(config, seed=seed)
    raise ValueError(f"Unsupported dataset format: {config.format}")

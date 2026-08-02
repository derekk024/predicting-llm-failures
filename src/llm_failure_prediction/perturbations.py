from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from llm_failure_prediction.config import PerturbationConfig
from llm_failure_prediction.schema import (
    ANSWER_LABELS,
    AnswerLabel,
    PerturbationMetadata,
    PerturbationName,
    Question,
)


@dataclass(frozen=True)
class PromptVariant:
    name: PerturbationName
    question_text: str
    displayed_choices: tuple[str, str, str, str]
    metadata: PerturbationMetadata


def _identity_mapping() -> dict[AnswerLabel, AnswerLabel]:
    return dict(zip(ANSWER_LABELS, ANSWER_LABELS, strict=True))


def _question_seed(question_id: str, seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _incorrect_label(question: Question, seed: int) -> AnswerLabel:
    candidates = [label for label in ANSWER_LABELS if label != question.correct_semantic_label]
    rng = random.Random(_question_seed(question.question_id, seed, "incorrect_hint"))
    return rng.choice(candidates)


def _reordered_mapping(question: Question, seed: int) -> dict[AnswerLabel, AnswerLabel]:
    semantic_order = list(ANSWER_LABELS)
    rng = random.Random(_question_seed(question.question_id, seed, "reorder_choices"))
    while semantic_order == list(ANSWER_LABELS):
        rng.shuffle(semantic_order)
    return dict(zip(ANSWER_LABELS, semantic_order, strict=True))


def build_variants(
    question: Question,
    *,
    donor: Question,
    config: PerturbationConfig,
    seed: int,
) -> list[PromptVariant]:
    choice_by_label = {choice.semantic_label: choice.text for choice in question.choices}
    clean_choices = tuple(choice.text for choice in question.choices)
    identity = _identity_mapping()
    variants = [
        PromptVariant(
            name="clean",
            question_text=question.question,
            displayed_choices=clean_choices,
            metadata=PerturbationMetadata(prompt_to_semantic=identity),
        )
    ]

    if config.incorrect_hint.enabled:
        hint_label = _incorrect_label(question, seed)
        hint = config.incorrect_hint.template.format(hint_label=hint_label)
        variants.append(
            PromptVariant(
                name="incorrect_hint",
                question_text=f"{question.question}\n\n{hint}",
                displayed_choices=clean_choices,
                metadata=PerturbationMetadata(
                    hint_semantic_label=hint_label,
                    prompt_to_semantic=identity,
                ),
            )
        )

    if config.reorder_choices.enabled:
        mapping = _reordered_mapping(question, seed)
        displayed = tuple(choice_by_label[mapping[label]] for label in ANSWER_LABELS)
        variants.append(
            PromptVariant(
                name="reorder_choices",
                question_text=question.question,
                displayed_choices=displayed,
                metadata=PerturbationMetadata(prompt_to_semantic=mapping),
            )
        )

    if config.irrelevant_sentence.enabled:
        irrelevant = config.irrelevant_sentence.template.format(donor_question=donor.question)
        variants.append(
            PromptVariant(
                name="irrelevant_sentence",
                question_text=f"{question.question}\n\n{irrelevant}",
                displayed_choices=clean_choices,
                metadata=PerturbationMetadata(
                    prompt_to_semantic=identity,
                    donor_question_id=donor.question_id,
                    donor_question=donor.question,
                ),
            )
        )

    return variants

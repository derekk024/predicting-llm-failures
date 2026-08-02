from __future__ import annotations

from llm_failure_prediction.config import PromptConfig
from llm_failure_prediction.perturbations import PromptVariant
from llm_failure_prediction.schema import ANSWER_LABELS


def render_user_prompt(variant: PromptVariant, config: PromptConfig) -> str:
    choices = "\n".join(
        f"{label}. {text}"
        for label, text in zip(ANSWER_LABELS, variant.displayed_choices, strict=True)
    )
    return f"{variant.question_text}\n\n{choices}\n\n{config.answer_instruction}"


def render_chat_prompt(tokenizer, user_prompt: str, config: PromptConfig, *, enable_thinking: bool):
    messages = [
        {"role": "system", "content": config.system},
        {"role": "user", "content": user_prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return f"{rendered}{config.assistant_prefill}"

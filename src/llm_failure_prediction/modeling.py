from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_failure_prediction.config import ModelConfig
from llm_failure_prediction.schema import ANSWER_LABELS, AnswerLabel

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class BatchScores:
    logits: np.ndarray
    probabilities: np.ndarray
    predicted_indices: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    sequence_lengths: np.ndarray
    clean_activations: np.ndarray


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MultipleChoiceModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = resolve_device(config.device)
        dtype = DTYPES[config.dtype]
        if self.device.type == "cpu" and dtype == torch.float16:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(config.name, revision=config.revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # We score at each row's last real token rather than using generation APIs. Right padding
        # avoids non-finite Qwen attention outputs observed for left-padded MPS batches.
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.name,
            revision=config.revision,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.answer_token_ids = self._answer_token_ids()

    def _answer_token_ids(self) -> dict[AnswerLabel, int]:
        ids: dict[AnswerLabel, int] = {}
        for label in ANSWER_LABELS:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Answer label {label!r} maps to {len(token_ids)} tokens; "
                    "single-step answer-logit scoring is invalid for this tokenizer"
                )
            ids[label] = token_ids[0]
        if len(set(ids.values())) != 4:
            raise ValueError("Answer labels do not map to four distinct token IDs")
        return ids

    @torch.inference_mode()
    def score(self, rendered_prompts: list[str], *, clean_index: int = 0) -> BatchScores:
        encoded = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        sequence_lengths = encoded["attention_mask"].sum(dim=-1)
        last_positions = sequence_lengths - 1
        batch_indices = torch.arange(len(rendered_prompts), device=self.device)

        candidate_ids = torch.tensor(
            [self.answer_token_ids[label] for label in ANSWER_LABELS],
            device=self.device,
        )
        final_logits = outputs.logits[batch_indices, last_positions, :]
        answer_logits = final_logits.index_select(-1, candidate_ids).float()
        if not torch.isfinite(answer_logits).all():
            bad_rows = (~torch.isfinite(answer_logits).all(dim=-1)).nonzero().flatten().tolist()
            raise FloatingPointError(
                f"Non-finite answer logits for batch rows {bad_rows} on {self.device} "
                f"with dtype={next(self.model.parameters()).dtype}. Try bfloat16 on Apple "
                "Silicon or float32/CPU as a diagnostic."
            )
        probabilities = torch.softmax(answer_logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        top_two = answer_logits.topk(k=2, dim=-1).values
        margin = top_two[:, 0] - top_two[:, 1]

        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        clean_activations = torch.stack(
            [
                hidden[clean_index, last_positions[clean_index], :]
                for hidden in outputs.hidden_states[1:]
            ],
            dim=0,
        )

        return BatchScores(
            logits=answer_logits.cpu().numpy(),
            probabilities=probabilities.cpu().numpy(),
            predicted_indices=answer_logits.argmax(dim=-1).cpu().numpy(),
            entropy=entropy.cpu().numpy(),
            margin=margin.cpu().numpy(),
            sequence_lengths=sequence_lengths.cpu().numpy(),
            clean_activations=clean_activations.float().cpu().numpy(),
        )

    @property
    def model_commit(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)

    @property
    def tokenizer_commit(self) -> str | None:
        return getattr(self.tokenizer, "_commit_hash", None)

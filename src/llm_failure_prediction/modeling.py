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
    def __init__(self, config: ModelConfig, answer_candidates: dict[AnswerLabel, str]):
        self.config = config
        self.answer_candidates = answer_candidates
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
            candidate = self.answer_candidates[label]
            token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Answer candidate {candidate!r} for label {label} maps to "
                    f"{len(token_ids)} tokens; "
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

    @torch.inference_mode()
    def score_with_patch(
        self,
        rendered_prompts: list[str],
        *,
        patch_index: int,
        layer_index: int,
        patch_vector: np.ndarray,
    ) -> np.ndarray:
        backbone = getattr(self.model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise TypeError("Model does not expose transformer blocks at model.layers")
        if layer_index not in range(len(layers)):
            raise IndexError(f"Layer {layer_index} is outside [0, {len(layers)})")
        if patch_index not in range(len(rendered_prompts)):
            raise IndexError(f"Patch row {patch_index} is outside [0, {len(rendered_prompts)})")

        encoded = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        last_position = int(encoded["attention_mask"][patch_index].sum().item() - 1)
        patch = torch.as_tensor(
            patch_vector,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )

        def replace_final_position(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if patch.shape != hidden.shape[-1:]:
                raise ValueError(
                    f"Patch shape {tuple(patch.shape)} does not match hidden size "
                    f"{hidden.shape[-1]}"
                )
            # Avoid indexed assignment here: it can produce an invalid MPS graph for Qwen's
            # following matrix multiplication even though the eager tensor shape is correct.
            rows = torch.arange(hidden.shape[0], device=self.device)[:, None]
            positions = torch.arange(hidden.shape[1], device=self.device)[None, :]
            mask = (rows == patch_index) & (positions == last_position)
            patched_hidden = torch.where(mask[..., None], patch.reshape(1, 1, -1), hidden)
            if isinstance(output, tuple):
                return (patched_hidden, *output[1:])
            return patched_hidden

        # Hugging Face returns the embedding state, outputs of blocks 0..N-2, and the terminal
        # normalized state. After excluding the embedding, cached feature N-1 therefore maps to
        # the final norm rather than the pre-norm output of block N-1.
        target_module = backbone.norm if layer_index == len(layers) - 1 else layers[layer_index]
        handle = target_module.register_forward_hook(replace_final_position)
        try:
            outputs = self.model(**encoded, output_hidden_states=False, use_cache=False)
        finally:
            handle.remove()
        candidate_ids = torch.tensor(
            [self.answer_token_ids[label] for label in ANSWER_LABELS],
            device=self.device,
        )
        answer_logits = (
            outputs.logits[patch_index, last_position, :].index_select(-1, candidate_ids).float()
        )
        if not torch.isfinite(answer_logits).all():
            raise FloatingPointError("Patched forward pass produced non-finite answer logits")
        return answer_logits.cpu().numpy()

    @property
    def model_commit(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)

    @property
    def tokenizer_commit(self) -> str | None:
        return getattr(self.tokenizer, "_commit_hash", None)

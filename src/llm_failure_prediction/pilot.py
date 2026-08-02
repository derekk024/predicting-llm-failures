from __future__ import annotations

import gc
import platform
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import __version__ as datasets_version

from llm_failure_prediction.config import ExperimentConfig
from llm_failure_prediction.data import load_questions
from llm_failure_prediction.io import (
    prepare_run_directory,
    write_activation_cache,
    write_json,
    write_records,
    write_resolved_config,
)
from llm_failure_prediction.modeling import MultipleChoiceModel
from llm_failure_prediction.perturbations import build_variants
from llm_failure_prediction.prompts import render_chat_prompt, render_user_prompt
from llm_failure_prediction.schema import ANSWER_LABELS, QuestionRecord, ScoredRun


def set_reproducibility(seed: int, *, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _build_run_id(config: ExperimentConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = config.model.name.rsplit("/", 1)[-1].lower()
    return f"{timestamp}_{model_slug}_n{config.dataset.limit}_seed{config.seed}"


def _summarize(records: list[QuestionRecord]) -> dict[str, Any]:
    clean_runs = [record.runs[0] for record in records]
    clean_correct_count = sum(run.is_correct for run in clean_runs)
    clean_predictions = Counter(run.predicted_semantic_label for run in clean_runs)
    summary: dict[str, Any] = {
        "n_questions": len(records),
        "clean_correct_count": clean_correct_count,
        "clean_accuracy": float(clean_correct_count / len(clean_runs)),
        "clean_mean_entropy": float(np.mean([run.entropy for run in clean_runs])),
        "clean_mean_top_two_logit_margin": float(
            np.mean([run.top_two_logit_margin for run in clean_runs])
        ),
        "clean_prediction_counts": {
            label: clean_predictions.get(label, 0) for label in ANSWER_LABELS
        },
        "perturbations": {},
    }
    perturbation_names = [run.perturbation for run in records[0].runs[1:]]
    for name in perturbation_names:
        runs = [next(run for run in record.runs if run.perturbation == name) for record in records]
        flip_count = sum(bool(run.changed_from_clean) for run in runs)
        originally_correct_runs = [
            run for clean, run in zip(clean_runs, runs, strict=True) if clean.is_correct
        ]
        originally_correct_flip_count = sum(
            bool(run.changed_from_clean) for run in originally_correct_runs
        )
        predictions = Counter(run.predicted_semantic_label for run in runs)
        summary["perturbations"][name] = {
            "semantic_flip_count": flip_count,
            "semantic_flip_rate": float(flip_count / len(runs)),
            "accuracy": float(np.mean([run.is_correct for run in runs])),
            "prediction_counts": {label: predictions.get(label, 0) for label in ANSWER_LABELS},
            "originally_correct_count": len(originally_correct_runs),
            "originally_correct_flip_count": originally_correct_flip_count,
            "originally_correct_flip_rate": (
                float(originally_correct_flip_count / len(originally_correct_runs))
                if originally_correct_runs
                else None
            ),
        }
    return summary


def run_pilot(config: ExperimentConfig, *, run_id: str | None = None) -> Path:
    set_reproducibility(
        config.seed,
        deterministic_algorithms=config.runtime.deterministic_algorithms,
    )
    run_id = run_id or _build_run_id(config)
    run_dir = prepare_run_directory(
        config.output.root,
        run_id,
        overwrite=config.output.overwrite,
    )
    write_resolved_config(run_dir / "config.resolved.yaml", config)

    started = time.perf_counter()
    questions = load_questions(config.dataset, seed=config.seed)
    model = MultipleChoiceModel(config.model, config.prompt.answer_candidates)
    records: list[QuestionRecord] = []
    activation_rows: list[np.ndarray] = []

    for index, question in enumerate(questions):
        donor = questions[(index + 1) % len(questions)]
        variants = build_variants(
            question,
            donor=donor,
            config=config.perturbations,
            seed=config.seed,
        )
        user_prompts = [render_user_prompt(variant, config.prompt) for variant in variants]
        rendered_prompts = [
            render_chat_prompt(
                model.tokenizer,
                prompt,
                config.prompt,
                enable_thinking=config.model.enable_thinking,
            )
            for prompt in user_prompts
        ]
        batch = model.score(rendered_prompts)
        clean_semantic_label = None
        scored_runs: list[ScoredRun] = []

        variant_pairs = zip(variants, user_prompts, strict=True)
        for variant_index, (variant, user_prompt) in enumerate(variant_pairs):
            predicted_prompt_label = ANSWER_LABELS[int(batch.predicted_indices[variant_index])]
            predicted_semantic_label = variant.metadata.prompt_to_semantic[predicted_prompt_label]
            if variant.name == "clean":
                clean_semantic_label = predicted_semantic_label
            assert clean_semantic_label is not None
            scored_runs.append(
                ScoredRun(
                    perturbation=variant.name,
                    prompt=user_prompt,
                    metadata=variant.metadata,
                    answer_logits={
                        label: float(batch.logits[variant_index, label_index])
                        for label_index, label in enumerate(ANSWER_LABELS)
                    },
                    answer_probabilities={
                        label: float(batch.probabilities[variant_index, label_index])
                        for label_index, label in enumerate(ANSWER_LABELS)
                    },
                    predicted_prompt_label=predicted_prompt_label,
                    predicted_semantic_label=predicted_semantic_label,
                    entropy=float(batch.entropy[variant_index]),
                    top_two_logit_margin=float(batch.margin[variant_index]),
                    is_correct=predicted_semantic_label == question.correct_semantic_label,
                    changed_from_clean=(
                        None
                        if variant.name == "clean"
                        else predicted_semantic_label != clean_semantic_label
                    ),
                    sequence_length=int(batch.sequence_lengths[variant_index]),
                )
            )

        activation_row = len(activation_rows)
        activation_rows.append(batch.clean_activations)
        records.append(
            QuestionRecord(
                schema_version=config.schema_version,
                run_id=run_id,
                activation_row=activation_row,
                question=question,
                runs=scored_runs,
            )
        )

        if (index + 1) % 10 == 0 or index + 1 == len(questions):
            elapsed = time.perf_counter() - started
            print(f"Scored {index + 1}/{len(questions)} questions in {elapsed:.1f}s", flush=True)

    activation_dtype = np.dtype(config.runtime.activation_dtype)
    activation_array = np.stack(activation_rows).astype(activation_dtype, copy=False)
    write_records(run_dir / "records.jsonl", records)
    write_activation_cache(
        run_dir / "activations.npz",
        activation_array,
        question_ids=[record.question.question_id for record in records],
        layer_indices=np.arange(activation_array.shape[1], dtype=np.int16),
    )
    write_json(run_dir / "summary.json", _summarize(records))
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": config.schema_version,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets_version,
            "device": str(model.device),
            "model_name": config.model.name,
            "requested_model_revision": config.model.revision,
            "resolved_model_commit": model.model_commit,
            "resolved_tokenizer_commit": model.tokenizer_commit,
            "dataset_path": config.dataset.path,
            "dataset_name": config.dataset.name,
            "dataset_split": config.dataset.split,
            "requested_dataset_revision": config.dataset.revision,
            "answer_token_ids": model.answer_token_ids,
            "answer_candidates": config.prompt.answer_candidates,
            "activation_shape": list(activation_array.shape),
            "activation_dtype": str(activation_array.dtype),
            "activation_semantics": (
                "clean run, final prompt position; cache indices 0..N-2 are outputs of "
                "blocks 0..N-2 and index N-1 is the terminal normalized hidden state"
            ),
        },
    )
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return run_dir

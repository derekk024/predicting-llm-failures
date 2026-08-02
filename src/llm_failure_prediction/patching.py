from __future__ import annotations

import gc
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import Field, model_validator

from llm_failure_prediction.config import OutputConfig, StrictModel, load_config
from llm_failure_prediction.io import prepare_run_directory, write_json, write_resolved_config
from llm_failure_prediction.modeling import MultipleChoiceModel
from llm_failure_prediction.probes import BootstrapConfig, FailureTarget, _read_records
from llm_failure_prediction.prompts import render_chat_prompt
from llm_failure_prediction.schema import ANSWER_LABELS, AnswerLabel, QuestionRecord, ScoredRun


class PatchingConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    extraction_run: Path
    selection_run: Path | None = None
    targets: list[FailureTarget] = Field(min_length=1)
    layer_by_target: dict[FailureTarget, int]
    max_examples_per_target: int = Field(default=67, gt=0)
    max_sanity_logit_difference: float = Field(default=0.01, gt=0.0)
    bootstrap: BootstrapConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_targets_and_layers(self) -> PatchingConfig:
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("Patching targets must be unique")
        if set(self.layer_by_target) != set(self.targets):
            raise ValueError("layer_by_target must contain exactly the configured targets")
        if any(layer < 0 for layer in self.layer_by_target.values()):
            raise ValueError("Patching layer indices must be nonnegative")
        return self


def load_patching_config(path: str | Path) -> PatchingConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return PatchingConfig.model_validate(raw)


def semantic_logit_margin(
    logits: np.ndarray,
    run: ScoredRun,
    semantic_label: AnswerLabel,
) -> float:
    semantic_to_prompt = {
        semantic: prompt for prompt, semantic in run.metadata.prompt_to_semantic.items()
    }
    prompt_label = semantic_to_prompt[semantic_label]
    index = ANSWER_LABELS.index(prompt_label)
    other_logits = np.delete(logits, index)
    return float(logits[index] - other_logits.max())


def _predicted_semantic_label(logits: np.ndarray, run: ScoredRun) -> AnswerLabel:
    prompt_label = ANSWER_LABELS[int(np.argmax(logits))]
    return run.metadata.prompt_to_semantic[prompt_label]


def _select_examples(
    records: list[QuestionRecord],
    config: PatchingConfig,
) -> list[tuple[int, FailureTarget, ScoredRun]]:
    if config.selection_run is not None:
        with config.selection_run.open(encoding="utf-8") as handle:
            selection_rows = json.load(handle)["rows"]
        record_by_id = {
            record.question.question_id: (record_index, record)
            for record_index, record in enumerate(records)
        }
        selected = []
        for target in config.targets:
            question_ids = [row["question_id"] for row in selection_rows if row["target"] == target]
            if len(question_ids) < config.max_examples_per_target:
                raise ValueError(
                    f"Selection run has only {len(question_ids)} examples for {target}"
                )
            if len(question_ids) != len(set(question_ids)):
                raise ValueError(f"Selection run contains duplicate {target} question IDs")
            for question_id in question_ids[: config.max_examples_per_target]:
                if question_id not in record_by_id:
                    raise ValueError(f"Selection question {question_id} is absent from extraction")
                record_index, record = record_by_id[question_id]
                run = next(run for run in record.runs if run.perturbation == target)
                if not run.changed_from_clean:
                    raise ValueError(f"Selection question {question_id} is not a {target} flip")
                selected.append((record_index, target, run))
        return selected

    rng = random.Random(config.seed)
    selected = []
    for target in config.targets:
        candidates = []
        for record_index, record in enumerate(records):
            run = next(run for run in record.runs if run.perturbation == target)
            if run.changed_from_clean:
                candidates.append((record_index, target, run))
        rng.shuffle(candidates)
        selected.extend(candidates[: config.max_examples_per_target])
    return selected


def paired_mean_effect(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if candidate.shape != control.shape or candidate.ndim != 1 or not len(candidate):
        raise ValueError("Paired effects require two nonempty one-dimensional arrays")
    differences = candidate - control
    interval = None
    if samples:
        rng = np.random.default_rng(seed)
        bootstrapped = np.empty(samples, dtype=np.float64)
        for sample_index in range(samples):
            indices = rng.integers(0, len(differences), size=len(differences))
            bootstrapped[sample_index] = differences[indices].mean()
        alpha = (1.0 - confidence_level) / 2.0
        interval = [float(value) for value in np.quantile(bootstrapped, [alpha, 1.0 - alpha])]
    return {
        "point_difference": float(differences.mean()),
        "bootstrap_interval": interval,
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    config: PatchingConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_examples": len(rows), "targets": {}}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target"]].append(row)
    for target_index, (target, target_rows) in enumerate(grouped.items()):
        interventions = {}
        for intervention in (
            "perturbed_identity_patch",
            "clean_patch",
            "unrelated_patch",
            "norm_matched_noise",
        ):
            restoration = np.asarray(
                [row[intervention]["restored_clean_answer"] for row in target_rows],
                dtype=np.float64,
            )
            margin_change = np.asarray(
                [row[intervention]["clean_answer_margin_change"] for row in target_rows],
                dtype=np.float64,
            )
            interventions[intervention] = {
                "restoration_count": int(restoration.sum()),
                "restoration_rate": float(restoration.mean()),
                "mean_clean_answer_margin_change": float(margin_change.mean()),
                "max_abs_logit_change": float(
                    max(row[intervention]["max_abs_logit_change"] for row in target_rows)
                ),
            }
        clean_restoration = np.asarray(
            [row["clean_patch"]["restored_clean_answer"] for row in target_rows],
            dtype=np.float64,
        )
        clean_margin_change = np.asarray(
            [row["clean_patch"]["clean_answer_margin_change"] for row in target_rows],
            dtype=np.float64,
        )
        comparisons = {}
        for control_index, control in enumerate(("unrelated_patch", "norm_matched_noise")):
            control_restoration = np.asarray(
                [row[control]["restored_clean_answer"] for row in target_rows],
                dtype=np.float64,
            )
            control_margin_change = np.asarray(
                [row[control]["clean_answer_margin_change"] for row in target_rows],
                dtype=np.float64,
            )
            seed = config.seed + target_index * 100 + control_index * 10
            comparisons[f"clean_patch_minus_{control}"] = {
                "difference_definition": "clean patch effect minus paired control effect",
                "restoration_rate": paired_mean_effect(
                    clean_restoration,
                    control_restoration,
                    samples=config.bootstrap.samples,
                    confidence_level=config.bootstrap.confidence_level,
                    seed=seed,
                ),
                "clean_answer_margin_change": paired_mean_effect(
                    clean_margin_change,
                    control_margin_change,
                    samples=config.bootstrap.samples,
                    confidence_level=config.bootstrap.confidence_level,
                    seed=seed + 1,
                ),
            }
        summary["targets"][target] = {
            "n_examples": len(target_rows),
            "layer_index": target_rows[0]["layer_index"],
            "max_replay_abs_logit_difference": float(
                max(row["replay_max_abs_logit_difference"] for row in target_rows)
            ),
            "interventions": interventions,
            "paired_comparisons": comparisons,
        }
    return summary


def run_patching(config: PatchingConfig, *, run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = prepare_run_directory(
        config.output.root,
        run_id,
        overwrite=config.output.overwrite,
    )
    write_resolved_config(output_dir / "config.resolved.yaml", config)

    extraction_config = load_config(config.extraction_run / "config.resolved.yaml")
    records = _read_records(config.extraction_run / "records.jsonl")
    with np.load(config.extraction_run / "activations.npz") as cache:
        clean_activations = cache["activations"]
        cached_ids = cache["question_ids"].astype(str)
    if cached_ids.tolist() != [record.question.question_id for record in records]:
        raise ValueError("Activation cache and records are misaligned")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = MultipleChoiceModel(
        extraction_config.model,
        extraction_config.prompt.answer_candidates,
    )
    examples = _select_examples(records, config)
    rows = []
    for example_index, (record_index, target, perturbed_run) in enumerate(examples):
        record = records[record_index]
        clean_label = record.runs[0].predicted_semantic_label
        rendered_prompts = [
            render_chat_prompt(
                model.tokenizer,
                run.prompt,
                extraction_config.prompt,
                enable_thinking=extraction_config.model.enable_thinking,
            )
            for run in record.runs
        ]
        perturbed_index = record.runs.index(perturbed_run)
        # Reuse the original four-variant batch shape. In addition to matching extraction, this
        # avoids a Qwen3 bfloat16/MPS graph failure observed for unpadded single-prompt batches.
        unpatched = model.score(rendered_prompts, clean_index=perturbed_index)
        layer_index = config.layer_by_target[target]
        perturbed_vector = unpatched.clean_activations[layer_index]
        clean_vector = clean_activations[record_index, layer_index].astype(np.float32)
        unrelated_index = (record_index + 1 + example_index) % len(records)
        if unrelated_index == record_index:
            unrelated_index = (unrelated_index + 1) % len(records)
        unrelated_vector = clean_activations[unrelated_index, layer_index].astype(np.float32)
        displacement_norm = float(np.linalg.norm(clean_vector - perturbed_vector))
        noise_direction = rng.normal(size=clean_vector.shape).astype(np.float32)
        noise_direction /= max(float(np.linalg.norm(noise_direction)), 1e-12)
        noise_vector = perturbed_vector + noise_direction * displacement_norm

        base_logits = unpatched.logits[perturbed_index]
        recorded_logits = np.asarray(
            [perturbed_run.answer_logits[label] for label in ANSWER_LABELS],
            dtype=np.float32,
        )
        base_margin = semantic_logit_margin(base_logits, perturbed_run, clean_label)
        intervention_vectors = {
            "perturbed_identity_patch": perturbed_vector,
            "clean_patch": clean_vector,
            "unrelated_patch": unrelated_vector,
            "norm_matched_noise": noise_vector,
        }
        row: dict[str, Any] = {
            "question_id": record.question.question_id,
            "target": target,
            "layer_index": layer_index,
            "clean_semantic_label": clean_label,
            "unpatched_semantic_label": _predicted_semantic_label(base_logits, perturbed_run),
            "unpatched_clean_answer_margin": base_margin,
            "replay_max_abs_logit_difference": float(np.max(np.abs(base_logits - recorded_logits))),
            "patch_displacement_norm": displacement_norm,
            "unrelated_question_id": records[unrelated_index].question.question_id,
        }
        for name, vector in intervention_vectors.items():
            patched_logits = model.score_with_patch(
                rendered_prompts,
                patch_index=perturbed_index,
                layer_index=layer_index,
                patch_vector=vector,
            )
            patched_margin = semantic_logit_margin(patched_logits, perturbed_run, clean_label)
            row[name] = {
                "predicted_semantic_label": _predicted_semantic_label(
                    patched_logits,
                    perturbed_run,
                ),
                "restored_clean_answer": (
                    _predicted_semantic_label(patched_logits, perturbed_run) == clean_label
                ),
                "clean_answer_margin": patched_margin,
                "clean_answer_margin_change": patched_margin - base_margin,
                "max_abs_logit_change": float(np.max(np.abs(patched_logits - base_logits))),
            }
        if row["unpatched_semantic_label"] == clean_label:
            raise RuntimeError(
                f"Replayed {target} prompt no longer flips question {record.question.question_id}"
            )
        if row["replay_max_abs_logit_difference"] > config.max_sanity_logit_difference:
            raise RuntimeError(
                f"Replayed logits differ from extraction for {record.question.question_id}"
            )
        identity_difference = row["perturbed_identity_patch"]["max_abs_logit_change"]
        if identity_difference > config.max_sanity_logit_difference:
            raise RuntimeError(f"Identity patch changed logits for {record.question.question_id}")
        rows.append(row)
        if (example_index + 1) % 20 == 0 or example_index + 1 == len(examples):
            print(f"Patched {example_index + 1}/{len(examples)} examples", flush=True)

    write_json(
        output_dir / "patching-results.json",
        {"rows": rows, "summary": _summarize(rows, config=config)},
    )
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return output_dir

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_failure_prediction.config import (
    ExperimentConfig,
    ModelConfig,
    ModelGateConfig,
    PromptConfig,
)
from llm_failure_prediction.io import prepare_run_directory, write_json, write_resolved_config
from llm_failure_prediction.pilot import run_pilot


def _default_gate_id(config: ModelGateConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_n{config.dataset.limit}_seed{config.seed}"


def run_model_gate(config: ModelGateConfig, *, gate_id: str | None = None) -> Path:
    gate_id = gate_id or _default_gate_id(config)
    gate_dir = prepare_run_directory(
        config.output.root,
        gate_id,
        overwrite=config.output.overwrite,
    )
    write_resolved_config(gate_dir / "config.resolved.yaml", config)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for named_model in config.models:
        for named_prompt in config.prompt_formats:
            run_id = f"{named_model.id}__{named_prompt.id}"
            experiment = ExperimentConfig(
                schema_version=config.schema_version,
                seed=config.seed,
                dataset=config.dataset.model_copy(deep=True),
                model=ModelConfig.model_validate(named_model.model_dump(exclude={"id"})),
                prompt=PromptConfig.model_validate(named_prompt.model_dump(exclude={"id"})),
                perturbations=config.perturbations.model_copy(deep=True),
                runtime=config.runtime.model_copy(deep=True),
                output={
                    "root": gate_dir / "runs",
                    "overwrite": config.output.overwrite,
                },
            )
            run_dir = run_pilot(experiment, run_id=run_id)
            with (run_dir / "summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            max_label_share = (
                max(summary["clean_prediction_counts"].values()) / summary["n_questions"]
            )
            results.append(
                {
                    "model_id": named_model.id,
                    "model_name": named_model.name,
                    "model_revision": named_model.revision,
                    "prompt_id": named_prompt.id,
                    "assistant_prefill": named_prompt.assistant_prefill,
                    "answer_candidates": named_prompt.answer_candidates,
                    "run_directory": str(run_dir),
                    "clean_accuracy": summary["clean_accuracy"],
                    "clean_prediction_counts": summary["clean_prediction_counts"],
                    "clean_max_label_share": max_label_share,
                    "perturbations": summary["perturbations"],
                }
            )

    write_json(
        gate_dir / "gate-summary.json",
        {
            "schema_version": config.schema_version,
            "gate_id": gate_id,
            "n_questions": config.dataset.limit,
            "dataset_split": config.dataset.split,
            "elapsed_seconds": time.perf_counter() - started,
            "selection_rule": (
                "Prefer higher clean accuracy subject to acceptable answer-position balance; "
                "inspect perturbation flip counts before freezing the configuration."
            ),
            "results": results,
        },
    )
    return gate_dir

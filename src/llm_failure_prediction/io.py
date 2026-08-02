from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

from llm_failure_prediction.schema import QuestionRecord


def prepare_run_directory(root: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = root / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory {run_dir} is not empty. Choose another run ID or enable overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_records(path: Path, records: list[QuestionRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def write_resolved_config(path: Path, config: BaseModel) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)


def write_activation_cache(
    path: Path,
    activations: np.ndarray,
    *,
    question_ids: list[str],
    layer_indices: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        activations=activations,
        question_ids=np.asarray(question_ids),
        layer_indices=layer_indices,
    )

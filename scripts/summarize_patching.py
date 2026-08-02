from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_failure_prediction.probes import _read_records


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _coordinate_diagnostic(results: dict, records_path: Path) -> dict:
    records = {record.question.question_id: record for record in _read_records(records_path)}
    rows = [row for row in results["rows"] if row["target"] == "reorder_choices"]
    same_letter = []
    for row in rows:
        record = records[row["question_id"]]
        reordered = next(run for run in record.runs if run.perturbation == "reorder_choices")
        semantic_to_prompt = {
            semantic: prompt for prompt, semantic in reordered.metadata.prompt_to_semantic.items()
        }
        same_letter.append(
            semantic_to_prompt[row["clean_semantic_label"]] == row["clean_semantic_label"]
        )
    return {
        "same_displayed_letter_count": sum(same_letter),
        "moved_to_different_letter_count": len(same_letter) - sum(same_letter),
        "clean_patch_restoration_same_letter": sum(
            row["clean_patch"]["restored_clean_answer"]
            for row, same in zip(rows, same_letter, strict=True)
            if same
        ),
        "clean_patch_restoration_moved_letter": sum(
            row["clean_patch"]["restored_clean_answer"]
            for row, same in zip(rows, same_letter, strict=True)
            if not same
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--sensitivity-results", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary = _load(args.primary_results)
    sensitivity = _load(args.sensitivity_results)
    primary_reorder_ids = [
        row["question_id"] for row in primary["rows"] if row["target"] == "reorder_choices"
    ]
    sensitivity_ids = [row["question_id"] for row in sensitivity["rows"]]
    if primary_reorder_ids != sensitivity_ids:
        raise ValueError("Preterminal sensitivity does not use the primary reorder sample")

    output = {
        "schema_version": 1,
        "primary_artifact": str(args.primary_results),
        "primary_summary": primary["summary"],
        "reorder_terminal_coordinate_diagnostic": _coordinate_diagnostic(
            primary,
            args.records,
        ),
        "preterminal_sensitivity_artifact": str(args.sensitivity_results),
        "preterminal_sensitivity_summary": sensitivity["summary"],
        "interpretation_scope": (
            "Controlled final-position interventions on flipped ARC test pairs; this summary does "
            "not establish a general mechanism or incremental predictive value."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

# Week 1 pilot: evaluator and activation cache

## Setup

- Model: `Qwen/Qwen3-0.6B`, thinking disabled, bfloat16 inference on Apple M3 Pro MPS.
- Data: 200 deterministically shuffled four-choice questions from ARC-Challenge train.
- Scoring: next-token logits restricted to the single-token labels `A/B/C/D`.
- Activations: clean prompt only, final non-padding position, 28 transformer-block outputs.
- Runtime: 60.2 seconds after the one-time model download.

The model and dataset revisions are pinned in `configs/pilot_arc.yaml`. The validated activation
cache has shape `[200, 28, 1024]`, uses float16 storage, occupies about 9 MB compressed, and is
aligned to records by both `activation_row` and question ID.

## Sanity-check results

| Condition | Accuracy | Semantic flips | Flip rate |
|---|---:|---:|---:|
| Clean | 43.0% | — | — |
| Incorrect hint | 34.5% | 30 / 200 | 15.0% |
| Reordered choices | 39.0% | 132 / 200 | 66.0% |
| Irrelevant question | 35.5% | 32 / 200 | 16.0% |

Among the 86 questions answered correctly in the clean condition, the corresponding flip counts
were 19 for incorrect hints, 44 for reordered choices, and 18 for irrelevant questions.

## Decision from the pilot

The label balance is adequate for testing the evaluation and probe pipeline. However, the 0.6B
model predicted clean prompt label `A` on 151 of 200 questions. This strong position bias makes the
reordering result especially hard to interpret: remapping is implemented correctly, but many
semantic flips arise because the model preserves a prompt position instead of a choice meaning.

Before scaling extraction, compare prompt formats and `Qwen3-1.7B` on a small development subset.
Choose the format using development data only, then freeze it. The final study should report prompt
label distributions alongside accuracy and flip rates, and should not present this pilot's flip
rates as substantive evidence about activation predictiveness.


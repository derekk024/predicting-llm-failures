# Pilot data schema

Each pilot run writes one directory under `artifacts/pilot_arc/`:

- `config.resolved.yaml`: the complete effective configuration.
- `manifest.json`: package, hardware, model revision, token-ID, timing, and tensor metadata.
- `records.jsonl`: one `QuestionRecord` per original question.
- `activations.npz`: clean-run activations aligned by `activation_row`.
- `summary.json`: pilot accuracy, confidence, and semantic flip-rate checks.

## `QuestionRecord`

The question identity is the grouping key for every later train/validation/test split. Source
choice labels are normalized to semantic labels `A/B/C/D` in source choice order. Each record
contains the clean run followed by the enabled perturbations.

Every scored run includes:

- its exact user prompt and perturbation metadata;
- logits and probabilities restricted to the four answer tokens;
- the predicted prompt label and its remapped semantic label;
- four-way entropy and the top-two raw-logit margin;
- correctness and semantic change relative to the clean answer; and
- tokenized prompt length.

For reordered choices, `metadata.prompt_to_semantic` maps displayed answer labels back to the
original choice identity. `changed_from_clean` is computed only after this mapping.

## Activation cache

`activations.npz` contains:

- `activations`: `[question, transformer_block, hidden_dimension]`, stored as float16 by default;
- `question_ids`: the row alignment key; and
- `layer_indices`: zero-based transformer-block indices.

The embedding output (`hidden_states[0]`) is excluded. Each cached vector is the output of one
transformer block at the final position of the fully rendered clean prompt. Perturbed activations
are neither cached nor used as probe inputs.


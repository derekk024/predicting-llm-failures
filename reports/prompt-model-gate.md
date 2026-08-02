# Prompt and model gate

## Purpose

The Week 1 pilot exposed a strong answer-position bias in Qwen3-0.6B. Before scaling activation
extraction, this gate compared two scoring prompts and both planned model sizes on a fixed sample of
100 ARC-Challenge validation questions. The official ARC test split remained untouched.

The answer-cue format appended `The correct answer is` to the assistant prefix and scored the
single leading-space tokens ` A/ B/ C/ D`. The bare-letter format scored `A/B/C/D` immediately
after Qwen's non-thinking chat template. All model, tokenizer, and dataset revisions were pinned.

## Results

| Model | Prompt | Clean accuracy | Largest label share | Hint flips | Reorder flips | Irrelevant flips |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | Bare letter | 28% | 85% | 6% | 60% | 12% |
| Qwen3-0.6B | Answer cue | 40% | 42% | 45% | 46% | 39% |
| Qwen3-1.7B | Bare letter | **72%** | **27%** | 20% | 24% | 31% |
| Qwen3-1.7B | Answer cue | 69% | 31% | 17% | 24% | 24% |

The 1.7B bare-letter clean predictions were distributed `A=23, B=24, C=27, D=26`. It also
produced 20, 24, and 31 positive failure labels for incorrect hints, reordered choices, and
irrelevant context, respectively. This is enough balance to proceed without strengthening any
perturbation yet.

## Frozen decision

Use Qwen3-1.7B with thinking disabled, the bare-letter prompt, and direct single-token
`A/B/C/D` scoring for the main ARC extraction. Use the official ARC train split for fitting,
validation for model and layer selection, and test only for final held-out evaluation.

This gate selected the experimental configuration; it did not test activation predictiveness. Its
100 validation questions must not be reported as held-out performance. The final test split remains
the only source for headline probe metrics and confidence intervals.


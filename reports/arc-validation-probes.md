# ARC validation: confidence baselines and layer probes

## Setup

Qwen3-1.7B was run on all usable four-choice ARC-Challenge examples: 1,117 train questions and
295 validation questions. Train and validation question IDs were disjoint. Clean final-position
activations had shape `[question, 28, 2048]`; only clean activations were probe inputs.

For each perturbation, the benchmark fit:

- calibrated majority, entropy, and top-two-margin baselines;
- logistic regression on four centered answer logits plus entropy and margin; and
- an independently standardized logistic-regression probe for each transformer layer.

The best layer was selected by validation AUROC. Consequently, these metrics are for model
selection rather than held-out claims. Paired 95% intervals used 1,000 question-level bootstrap
samples and compare the selected activation probe with the combined logit-feature model.

## Validation results

| Failure target | Positives | Entropy AUROC | Logit AUROC | Best layer | Activation AUROC | AUROC difference (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Incorrect hint | 56 / 295 | 0.846 | 0.842 | 26 | 0.722 | −0.120 [−0.186, −0.059] |
| Reordered choices | 54 / 295 | 0.842 | 0.828 | 27 | 0.698 | −0.130 [−0.205, −0.057] |
| Irrelevant context | 86 / 295 | 0.755 | 0.796 | 22 | 0.724 | −0.072 [−0.145, −0.002] |

| Failure target | Logit AUPRC | Activation AUPRC | AUPRC difference (95% CI) | Logit Brier | Activation Brier |
|---|---:|---:|---:|---:|---:|
| Incorrect hint | 0.590 | 0.353 | −0.237 [−0.365, −0.106] | 0.109 | 0.174 |
| Reordered choices | 0.470 | 0.295 | −0.175 [−0.296, −0.069] | 0.115 | 0.198 |
| Irrelevant context | 0.573 | 0.501 | −0.072 [−0.185, 0.043] | 0.166 | 0.221 |

The activation probes also underperformed the logit-feature model on the originally-correct subset.
Their calibration was substantially worse, despite using unweighted logistic regression and the
same train/evaluation split.

## Margin-matched diagnostic

Each failure example was greedily matched without replacement to a non-failure example with the
nearest clean top-two logit margin. The resulting standardized margin differences were between
−0.125 and −0.069.

| Failure target | Pairs | Logit AUROC | Activation AUROC | Difference |
|---|---:|---:|---:|---:|
| Incorrect hint | 56 | 0.628 | 0.563 | −0.065 |
| Reordered choices | 54 | 0.512 | 0.469 | −0.043 |
| Irrelevant context | 86 | 0.595 | 0.669 | +0.074 |

The irrelevant-context matched result is an exploratory exception, not evidence of incremental
signal: matching used validation outcomes, the layer was selected on the same validation set, and
no matched-sample interval has yet been frozen for held-out testing.

## Current conclusion

The single-layer probes predict failures above chance, but they do not add predictive value beyond
the model's answer logits on this validation split. This is already informative: the extraction,
confidence controls, calibration analysis, and paired statistical comparison all work, and they do
not force a positive activation result.

The next planned test is a small MLP over a frozen set of late-layer activations, with early stopping
and regularization selected only on validation. The ARC test split remains untouched until the
architecture and selected layers are frozen.


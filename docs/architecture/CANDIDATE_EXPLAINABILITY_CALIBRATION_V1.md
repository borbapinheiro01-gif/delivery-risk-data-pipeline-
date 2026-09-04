# Candidate Explainability + Calibration V1

## Candidate

`LOGISTIC / M0`

This audit stays on the same 91,254 paired orders and the same 14 temporal folds
used for model selection.

## Reproduction parity

The candidate is re-fit independently on each governed fold and its predictions
are checked against the frozen Block-11 OOF scores.

- Prediction mismatches: 0
- Maximum absolute prediction difference:
  9.899e-17

## Explainability

The selected model is linear in standardized feature space. Therefore each logit
has an exact additive decomposition:

`logit(p) = intercept + Σ beta_j * z_j`

The implementation verifies this identity numerically for every fold.

The audit reports:

- standardized coefficients;
- odds ratio per one-standard-deviation change;
- coefficient sign stability;
- held-out permutation AP sensitivity;
- mean absolute logit contribution;
- most frequent top local driver.

These quantities describe model behavior, not causal effects.

ORDER_CORE_V1 has known algebraic dependencies, so coefficient and one-feature
permutation rankings should not be read as unique structural importance.

A SHAP dependency is not necessary to decompose this linear candidate exactly.

## Calibration

Frozen OOF diagnostic:

- rows: 82271
- prevalence: 0.072832468306
- pooled AP: 0.070932752149
- pooled ROC-AUC: 0.508363078125
- Brier: 0.068589080713
- Log Loss: 0.270791444905
- calibration intercept (ideal 0): -2.164144837600
- calibration slope (ideal 1): 0.126481567740
- quantile-bin ECE: 0.038481244441
- quantile-bin MCE: 0.119574425575

Diagnostic classification:

`CALIBRATION_REVIEW_REQUIRED_BEFORE_PRODUCTION`

No calibration transform is fitted or selected here.

## Release state

No threshold, Top-K budget, calibration transform, or production model is
released.

The next gate must backtest the selected M0 candidate on the entire frozen
supervised cohort, because candidate selection occurred on the 91,254-row
paired M1-eligible population while M0 itself is available on all 96,470 frozen
Gold orders.

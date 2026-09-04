# Governed Algorithm Benchmark V1

## Purpose

The logistic paired gate found no support for incremental M1 utility. This
benchmark tests whether nonlinear classifiers can extract useful signal from the
same Shipping Intelligence features.

## Frozen population and folds

- Paired M1-eligible population: 91,254 orders
- Operational M0 fallback outside estimand: 5,216 orders
- Temporal folds: 14
- Purchase-time leakage: 0
- Label-availability leakage: 0

The exact Block-10 fold counts and test order IDs are reproduced before any
model is fit.

## Algorithms

Benchmarked algorithms:

`HIST_GRADIENT_BOOSTING, LOGISTIC, RANDOM_FOREST, XGBOOST`

Optional runtime failures:

`NONE`

Each algorithm is evaluated with M0 and M1 on identical rows.

No hyperparameter optimization is performed in this gate.

## Selection

Primary ranking criterion:

`mean Average Precision across temporal folds`

Tie-breakers:

1. median temporal-fold Average Precision;
2. pooled OOF Average Precision.

Benchmark candidate:

- algorithm: `LOGISTIC`
- feature arm: `M0`
- mean AP: `0.087683030016`
- median AP: `0.067059430475`
- pooled OOF AP: `0.070932752149`

M1 evidence classification:

`NONLINEAR_OR_LINEAR_M1_INCREMENTAL_UTILITY_SUPPORTED`

## Governance

A benchmark winner is not yet a production release.

The next gate must review candidate robustness and operational Recall@Top-K
before production selection, thresholding, calibration, explainability or
deployment.

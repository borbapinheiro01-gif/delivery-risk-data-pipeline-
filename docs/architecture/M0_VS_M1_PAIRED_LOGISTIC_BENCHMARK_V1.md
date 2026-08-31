# M0 vs M1 Paired Logistic Benchmark V1

## Purpose

This is the first production late-risk feature-utility experiment.

M0 and M1 are evaluated on the same 91,254 M1-eligible orders.

The 5,216 remaining frozen supervised orders are not
discarded from the operational population; they remain governed by the M0
fallback.

## Models

Both arms use the same pipeline:

- StandardScaler
- LogisticRegression with L2 regularization
- C = 1.0
- solver = lbfgs
- max_iter = 2000
- random_state = 42
- sklearn API is version-aware: `penalty="l2"` before 1.8; `l1_ratio=0.0` from 1.8 onward

M0 uses ORDER_CORE_V1.

M1 uses ORDER_CORE_V1 plus:

- expected_freight_oot
- freight_burden
- freight_expected_ratio
- freight_log_ratio

## Temporal protocol

Each test fold is one purchase month.

Training rows must satisfy both:

1. purchase time strictly before the test-month cutoff;
2. label availability time strictly before the same cutoff.

The label-availability timestamp is the observed customer delivery timestamp.
This is a conservative rule: an order is not allowed into training merely
because it was purchased earlier.

Minimum prior M1-eligible months before the first test fold:
3.

Valid folds: 14.

## Primary result

- M0 mean AP: 0.087683030016
- M1 mean AP: 0.087015508293
- Mean paired delta AP: -0.000667521723
- Median paired delta AP: -0.001320838555
- Bootstrap 95% interval: [-0.001852390586, 0.000672168611]
- M1 fold wins: 5
- M0 fold wins: 9
- Ties: 0

Classification:

`LOGISTIC_M1_INCREMENTAL_UTILITY_NOT_SUPPORTED`

## Governance

Average Precision is the primary metric.

No probability threshold is selected in this gate.

M1 is not production-released by this result alone.

The subsequent algorithm benchmark is authorized and must preserve temporal
governance and paired feature-set comparisons.

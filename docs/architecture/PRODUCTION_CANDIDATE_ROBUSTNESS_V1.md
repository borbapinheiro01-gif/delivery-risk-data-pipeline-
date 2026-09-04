# Production Candidate Robustness V1

## Raw benchmark winner

- Algorithm: `LOGISTIC`
- Feature arm: `M0`
- Same-algorithm M1 evidence: `M1_NOT_SUPPORTED`

## Governed candidate

- Algorithm: `LOGISTIC`
- Feature arm: `M0`
- Raw ranking position of governed arm: 1
- Governance reason: `RAW_WINNER_ALREADY_M0`

M1 is retained only when the same algorithm demonstrates supported incremental
utility for M1 over M0. A raw rank advantage alone does not override that rule.

## Candidate vs Logistic M0

- Mean paired ΔAP: 0.000000000000
- Median paired ΔAP: 0.000000000000
- Bootstrap 95% CI: [0.000000000000, 0.000000000000]
- Candidate fold wins: 0
- Logistic M0 fold wins: 0
- Ties: 14
- Leave-one-month-out ΔAP range: [0.000000000000, 0.000000000000]
- Wilcoxon p-value: 1.0
- Exact sign-test p-value: 1.0

Classification:

`LOGISTIC_M0_REMAINS_GOVERNED_CANDIDATE`

## Recall at operational review fractions

Top fractions reviewed:

- 5%
- 10%
- 20%

No operational review budget is selected here. Recall necessarily increases as
the reviewed fraction grows; a budget decision requires capacity or intervention
cost constraints.

## Release state

This block does not select a probability threshold and does not release a
production model.

The governed candidate is authorized for explainability and calibration review.

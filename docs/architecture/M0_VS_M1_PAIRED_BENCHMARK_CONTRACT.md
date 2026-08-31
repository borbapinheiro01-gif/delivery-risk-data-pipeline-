# M0 vs M1 — Paired Benchmark Contract

## Question

Does the PIT-safe Shipping Intelligence layer improve late-delivery prediction
beyond `ORDER_CORE_V1`?

## Paired scientific comparison

M0 and M1 must use:

- identical M1-eligible rows;
- identical training rows;
- identical test rows;
- identical temporal cutoffs;
- the frozen label-availability rule.

Models:

- M0: `ORDER_CORE_V1`
- M1: `ORDER_CORE_V1` + Expected Freight + B + Q + Z

Primary metric:

- Average Precision / PR-AUC

Secondary:

- ROC-AUC
- Brier score
- Log Loss
- Recall@Top-K

Threshold selection is not part of the first paired feature-utility gate.

## Operational population

The paired experiment does not erase unavailable rows.

At serving time:

- complete M1 vector -> candidate M1, subject to benchmark approval;
- unavailable M1 vector -> M0 fallback.

Until the benchmark passes, M1 is not released.

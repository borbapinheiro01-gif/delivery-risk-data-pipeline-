# Recruiter Guide — 60-second tour

## What is this?

A data-engineering and logistics-intelligence project built from the public Olist
Brazilian e-commerce dataset.

## What was built?

- immutable RAW layer;
- Bronze / Silver / Gold architecture;
- data-quality gates;
- frozen supervised Gold cohort;
- purchase-time feature contract;
- point-in-time leakage controls;
- Expected Freight out-of-time feature;
- Shipping Friction diagnostics;
- reproducible reports and lineage.

## Scale

- **96,470** orders in the frozen supervised Gold cohort;
- **6,534** late deliveries;
- **6.77%** late-delivery prevalence;
- **13** audited purchase-time features;
- **91,262** orders with reproduced Expected Freight.

## What makes it different from a notebook-only case?

The emphasis is on whether the data can actually be trusted:

- timestamps are governed;
- feature availability is explicit;
- missingness has semantics;
- derived features are independently reproduced;
- ex-post diagnostics are kept separate from purchase-time predictors;
- contracts and artifacts are versioned.

## Suggested review order

1. `README.md`
2. `docs/portfolio/DATA_PIPELINE_OVERVIEW.md`
3. `docs/architecture/`
4. `reports/`

## One-line summary

**A leakage-safe, reproducible data foundation for delivery-risk and freight
intelligence, designed before model deployment claims are made.**

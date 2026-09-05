# Delivery Risk Intelligence

**Data engineering, quality validation and leakage-safe logistics intelligence for delivery-risk analysis.**

This repository presents an end-to-end data foundation built on the public Olist
Brazilian e-commerce dataset. The focus of this release is the **data layer**:
data quality, reproducible transformations, point-in-time governance and
logistics intelligence that can safely support downstream analytics and ML.

## 60-second overview

| Item | Audited result |
|---|---:|
| Frozen supervised Gold cohort | **96,470 orders** |
| Late deliveries | **6,534** |
| On-time deliveries | **89,936** |
| Late-delivery prevalence | **6.77%** |
| Core purchase-time features | **13** |
| Expected Freight artifact | **96,211 rows** |
| Expected Freight available | **91,262 (94.86%)** |
| Shipping Friction core | **96,211 rows** |
| Prediction timestamp | `order_purchase_timestamp` |

> The central design rule is simple: every predictor intended for a purchase-time
> decision must be reproducible using information available at that moment.

## Why this project exists

A delivery-risk model is only as trustworthy as the data pipeline behind it.
This project therefore treats **data correctness, temporal availability and
provenance as first-class engineering problems** before any production-model
claim is made.

The work covers:

- immutable source preservation;
- structural, semantic and statistical quality gates;
- Bronze / Silver / Gold modeling;
- explicit distinction between relation missingness and attribute missingness;
- construct validation for the late-delivery label;
- point-in-time leakage controls;
- independent feature recomputation;
- reproducible Expected Freight estimation;
- Shipping Friction diagnostics;
- documented contracts, lineage and scientific artifacts.

## Architecture

~~~mermaid
flowchart LR
    A[Olist source files] --> B[RAW immutable]
    B --> C[Bronze]
    C --> D[Silver]
    D --> E[Gold]
    E --> F[Purchase-time feature core]
    E --> G[Logistics context]
    G --> H[Expected Freight OOT]
    H --> I[Shipping Friction diagnostics]
    F --> J[Downstream analytics / ML]
    I --> J
~~~

## Data quality and governance

The project does not silently clean away ambiguous records. Instead, each issue
is classified and documented.

Examples of governed checks include:

- schema and key integrity;
- duplicate detection;
- Brazilian state-code validity;
- missingness by relation and by attribute;
- delivered-order cohort construction;
- late-delivery construct validation;
- purchase-time feature availability;
- historical-label availability;
- independent parity checks for engineered features.

The supervised Gold cohort is frozen at **96,470 delivered orders**, with
**6,534 calendar-day late deliveries**.

## Core feature layer

The purchase-time core contains 13 audited order-level variables spanning:

- item / product / seller counts;
- merchandise price statistics;
- freight statistics;
- merchandise-plus-freight;
- freight-to-price ratio.

The layer is intentionally small, inspectable and reproducible.

## Logistics intelligence

### Expected Freight OOT

Expected Freight is generated under an out-of-time protocol and independently
reproduced for parity. The artifact contains **96,211 orders**, with
**91,262 non-null expected-freight values**.

### Shipping Friction

Shipping Friction compares observed freight with expected freight and provides
an **ex-post logistics diagnostic**. It is not silently treated as a same-order
purchase-time predictor.

This distinction matters: high freight alone is not evidence of a logistics
problem, and a diagnostic constructed after fulfillment must not leak into a
purchase-time model.

## Repository tour

| Path | Purpose |
|---|---|
| `data/` | Medallion runtime structure and data contracts |
| `contracts/` | Dataset / schema contracts |
| `configs/` | Reproducible experiment and governance contracts |
| `src/` | Scientific / data-processing source code |
| `scripts/` | Reproduction and productization runners |
| `artifacts/` | Audited derived artifacts |
| `reports/` | Quality, benchmark and decision reports |
| `docs/architecture/` | Technical architecture notes |
| `docs/portfolio/` | Recruiter-facing project tour |

## What a reviewer should look at first

1. [`docs/portfolio/RECRUITER_GUIDE.md`](docs/portfolio/RECRUITER_GUIDE.md)
2. [`docs/portfolio/DATA_PIPELINE_OVERVIEW.md`](docs/portfolio/DATA_PIPELINE_OVERVIEW.md)
3. [`docs/architecture/`](docs/architecture/)
4. [`reports/`](reports/)

## Important scope note

This release is intentionally presented as a **data foundation and logistics
intelligence case**.

The repository also contains downstream modeling experiments. Those experiments
are retained for technical depth, but they are **not required to understand the
data-engineering contribution**, and no production-model release is claimed here.

## Tech stack

`Python` · `SQL` · `DuckDB` · `pandas` · `scikit-learn` · `Git` · `GitHub`

## Key engineering principles

`RAW_IMMUTABLE`  
`RELATION_MISSING != ATTRIBUTE_MISSING`  
`SOURCE_QUALITY != TASK_QUALITY`  
`NOT_APPLICABLE != MISSING`  
`NOT_YET_OBSERVED != NEGATIVE_LABEL`

---

Built as a portfolio case in data engineering, data quality, temporal governance
and logistics intelligence.

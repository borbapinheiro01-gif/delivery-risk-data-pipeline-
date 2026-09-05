# Data Pipeline Overview

## 1. Source layer

The project starts from the public Olist e-commerce dataset.

The raw source is treated as immutable. Quality problems are documented rather
than silently overwritten.

## 2. Quality gates

The pipeline separates:

- structural validity;
- semantic validity;
- statistical completeness;
- task-specific completeness;
- label / construct validity;
- point-in-time feature validity.

## 3. Medallion architecture

~~~text
RAW
  ↓
BRONZE
  ↓
SILVER
  ↓
GOLD
~~~

The Gold supervised cohort contains **96,470 delivered orders**, of which
**6,534 (6.77%)** are calendar-day late.

## 4. Prediction-time contract

The prediction timestamp is:

`order_purchase_timestamp`

A feature intended for same-order purchase-time use must satisfy:

`available_time(feature) <= order_purchase_timestamp`

Historical outcome-dependent information additionally requires the historical
outcome to have been observed before the current order timestamp.

## 5. Purchase-time core

The audited order-level core contains 13 features covering:

- item counts;
- product / seller counts;
- price aggregates;
- freight aggregates;
- total basket-plus-freight;
- freight-price ratio.

## 6. Expected Freight

Expected Freight is estimated out-of-time and independently reproduced.

Artifact rows: **96,211**

Available estimates: **91,262 (94.86%)**

## 7. Shipping Friction

Shipping Friction compares observed freight with expected freight.

It is treated as an ex-post diagnostic and not silently promoted to a
same-order purchase-time predictor.

## 8. Result of the data phase

The data phase closes with:

- a frozen, auditable supervised cohort;
- reproducible transformations;
- temporal leakage controls;
- governed feature definitions;
- logistics context and freight diagnostics;
- technical documentation suitable for downstream analytics and ML.

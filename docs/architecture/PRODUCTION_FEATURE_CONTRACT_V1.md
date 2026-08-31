# Production Feature Contract V1

## Prediction time

`order_purchase_timestamp`

## Grain

One row per `order_id`.

## M0 — released core

The M0 contract contains the 13 audited `ORDER_CORE_V1` features:

1. `item_count`
2. `unique_product_count`
3. `unique_seller_count`
4. `total_price`
5. `mean_price`
6. `max_price`
7. `min_price`
8. `price_range`
9. `total_freight`
10. `mean_freight`
11. `max_freight`
12. `merchandise_plus_freight`
13. `freight_price_ratio`

The two structural identities remain explicit:

`price_range = max_price - min_price`

`merchandise_plus_freight = total_price + total_freight`

## Outcome separation

`gold.delivery_service_outcomes` is ex-post.

Its columns are never joined into same-order purchase-time predictors.

## M1 — Shipping Intelligence

M1 remains **PENDING_PIT_REPRODUCTION_AUDIT**.

No Expected Freight, burden, ratio or anomaly variable is released into the
production benchmark until it is independently reproduced under the same
Point-in-Time contract.

## Critical Shipping Friction

Critical Shipping Friction is an ex-post diagnostic because it contains the
observed late-delivery outcome. It is forbidden as a same-order predictor.

## Supervised cohort parity

The production supervised cohort must satisfy all three conditions:

- `has_order_items = true`;
- delivery target is observed;
- `order_status = delivered`.

This rule was not assumed from row counts. It was accepted only after the
candidate order-id set exactly reproduced the frozen 96,470-order scientific
cohort. The frozen cohort contains 6,534 calendar-day late deliveries.

The six rows that had observed delivery timestamps but were not part of the
scientific cohort are retained in the Gold outcome table for provenance, but
are excluded from `gold.ml_delivery_supervised_core_v1`.

CREATE SCHEMA IF NOT EXISTS audit;

CREATE OR REPLACE TABLE audit.gold_core_quality AS

SELECT
    'GOLD_FEATURE_GRAIN_UNIQUE_ORDER_ID' AS check_name,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT order_id)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (COUNT(*) - COUNT(DISTINCT order_id))::VARCHAR AS observed,
    '0' AS expected
FROM gold.ml_delivery_features_core

UNION ALL

SELECT
    'GOLD_FEATURE_ROWS_EQUAL_ORDERS',
    CASE
        WHEN (SELECT COUNT(*) FROM gold.ml_delivery_features_core)
           = (SELECT COUNT(*) FROM silver.orders)
        THEN 'PASS' ELSE 'FAIL'
    END,
    (SELECT COUNT(*) FROM gold.ml_delivery_features_core)::VARCHAR,
    (SELECT COUNT(*) FROM silver.orders)::VARCHAR

UNION ALL

SELECT
    'GOLD_OUTCOME_GRAIN_UNIQUE_ORDER_ID',
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT order_id)
        THEN 'PASS' ELSE 'FAIL'
    END,
    (COUNT(*) - COUNT(DISTINCT order_id))::VARCHAR,
    '0'
FROM gold.delivery_service_outcomes

UNION ALL

SELECT
    'IDENTITY_PRICE_RANGE',
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL'
    END,
    COUNT(*)::VARCHAR,
    '0'
FROM gold.ml_delivery_features_core
WHERE has_order_items
  AND ABS(price_range - (max_price - min_price)) > 1e-10

UNION ALL

SELECT
    'IDENTITY_MERCHANDISE_PLUS_FREIGHT',
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL'
    END,
    COUNT(*)::VARCHAR,
    '0'
FROM gold.ml_delivery_features_core
WHERE has_order_items
  AND ABS(
        merchandise_plus_freight
        - (total_price + total_freight)
      ) > 1e-10

UNION ALL

SELECT
    'ITEM_RELATION_FLAG_CONSISTENCY',
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL'
    END,
    COUNT(*)::VARCHAR,
    '0'
FROM gold.ml_delivery_features_core
WHERE
    (has_order_items AND item_count IS NULL)
    OR
    (
        NOT has_order_items
        AND (
            item_count IS NOT NULL
            OR total_price IS NOT NULL
            OR total_freight IS NOT NULL
        )
    )

UNION ALL

SELECT
    'SUPERVISED_TARGET_NOT_NULL',
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL'
    END,
    COUNT(*)::VARCHAR,
    '0'
FROM gold.ml_delivery_supervised_core_v1
WHERE late_delivery_calendar_day IS NULL

UNION ALL

SELECT
    'SUPERVISED_ORDER_ID_UNIQUE',
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT order_id)
        THEN 'PASS' ELSE 'FAIL'
    END,
    (COUNT(*) - COUNT(DISTINCT order_id))::VARCHAR,
    '0'
FROM gold.ml_delivery_supervised_core_v1

UNION ALL

SELECT
    'SUPERVISED_STATUS_DELIVERED',
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL'
    END,
    COUNT(*)::VARCHAR,
    '0'
FROM gold.ml_delivery_supervised_core_v1 s
JOIN gold.delivery_service_outcomes y
  USING(order_id)
WHERE LOWER(TRIM(y.order_status)) <> 'delivered'
   OR y.order_status IS NULL;

CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE TABLE gold.ml_delivery_features_core AS
WITH item_agg AS (
    SELECT
        order_id,
        COUNT(*)::BIGINT AS item_count,
        COUNT(DISTINCT product_id)::BIGINT AS unique_product_count,
        COUNT(DISTINCT seller_id)::BIGINT AS unique_seller_count,
        SUM(price)::DOUBLE AS total_price,
        AVG(price)::DOUBLE AS mean_price,
        MAX(price)::DOUBLE AS max_price,
        MIN(price)::DOUBLE AS min_price,
        (MAX(price) - MIN(price))::DOUBLE AS price_range,
        SUM(freight_value)::DOUBLE AS total_freight,
        AVG(freight_value)::DOUBLE AS mean_freight,
        MAX(freight_value)::DOUBLE AS max_freight,
        (SUM(price) + SUM(freight_value))::DOUBLE AS merchandise_plus_freight,
        (SUM(freight_value) / NULLIF(SUM(price), 0))::DOUBLE AS freight_price_ratio
    FROM silver.order_items
    GROUP BY order_id
)
SELECT
    o.order_id,
    o.order_purchase_timestamp AS prediction_time,
    (a.order_id IS NOT NULL) AS has_order_items,
    a.item_count,
    a.unique_product_count,
    a.unique_seller_count,
    a.total_price,
    a.mean_price,
    a.max_price,
    a.min_price,
    a.price_range,
    a.total_freight,
    a.mean_freight,
    a.max_freight,
    a.merchandise_plus_freight,
    a.freight_price_ratio
FROM silver.orders o
LEFT JOIN item_agg a
    ON o.order_id = a.order_id;

CREATE OR REPLACE TABLE gold.delivery_service_outcomes AS
SELECT
    order_id,
    order_status,
    order_delivered_customer_date,
    order_estimated_delivery_date,
    (
        order_delivered_customer_date IS NOT NULL
        AND order_estimated_delivery_date IS NOT NULL
    ) AS target_observed,
    CASE
        WHEN order_delivered_customer_date IS NULL
          OR order_estimated_delivery_date IS NULL
        THEN NULL
        ELSE
            CAST(order_delivered_customer_date AS DATE)
            >
            CAST(order_estimated_delivery_date AS DATE)
    END AS late_delivery_calendar_day
FROM silver.orders;

CREATE OR REPLACE VIEW gold.ml_delivery_supervised_core_v1 AS
SELECT
    f.*,
    y.late_delivery_calendar_day
FROM gold.ml_delivery_features_core f
JOIN gold.delivery_service_outcomes y
    USING (order_id)
WHERE
    f.has_order_items
    AND y.target_observed
    AND LOWER(TRIM(y.order_status)) = 'delivered';

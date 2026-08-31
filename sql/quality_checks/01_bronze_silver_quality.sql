CREATE SCHEMA IF NOT EXISTS audit;

CREATE OR REPLACE TABLE audit.quality_checks AS
SELECT
    'BRONZE_ROW_COUNT__orders' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.orders)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'orders')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.orders)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'orders')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__order_items' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.order_items)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'order_items')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.order_items)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'order_items')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__customers' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.customers)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'customers')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.customers)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'customers')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__sellers' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.sellers)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'sellers')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.sellers)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'sellers')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__products' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.products)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'products')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.products)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'products')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__payments' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.payments)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'payments')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.payments)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'payments')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__reviews' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.reviews)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'reviews')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.reviews)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'reviews')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__geolocation' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.geolocation)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'geolocation')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.geolocation)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'geolocation')::VARCHAR AS expected
UNION ALL
SELECT
    'BRONZE_ROW_COUNT__product_category_translation' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM bronze.product_category_translation)
           = (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'product_category_translation')
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM bronze.product_category_translation)::VARCHAR AS observed,
    (SELECT expected_rows FROM audit.raw_inventory WHERE table_name = 'product_category_translation')::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__orders' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.orders)
           = (SELECT COUNT(*) FROM bronze.orders)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.orders)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.orders)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__order_items' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.order_items)
           = (SELECT COUNT(*) FROM bronze.order_items)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.order_items)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.order_items)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__customers' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.customers)
           = (SELECT COUNT(*) FROM bronze.customers)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.customers)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.customers)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__sellers' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.sellers)
           = (SELECT COUNT(*) FROM bronze.sellers)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.sellers)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.sellers)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__products' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.products)
           = (SELECT COUNT(*) FROM bronze.products)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.products)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.products)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__payments' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.payments)
           = (SELECT COUNT(*) FROM bronze.payments)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.payments)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.payments)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__reviews' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.reviews)
           = (SELECT COUNT(*) FROM bronze.reviews)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.reviews)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.reviews)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__geolocation' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.geolocation)
           = (SELECT COUNT(*) FROM bronze.geolocation)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.geolocation)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.geolocation)::VARCHAR AS expected
UNION ALL
SELECT
    'SILVER_ROW_COUNT__product_category_translation' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.product_category_translation)
           = (SELECT COUNT(*) FROM bronze.product_category_translation)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (SELECT COUNT(*) FROM silver.product_category_translation)::VARCHAR AS observed,
    (SELECT COUNT(*) FROM bronze.product_category_translation)::VARCHAR AS expected
UNION ALL
SELECT
    'KEY_UNIQUENESS__orders__order_id' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.orders)
           = (SELECT COUNT(DISTINCT order_id) FROM silver.orders)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    ((SELECT COUNT(*) FROM silver.orders)
      - (SELECT COUNT(DISTINCT order_id) FROM silver.orders))::VARCHAR AS observed,
    '0' AS expected
UNION ALL
SELECT
    'KEY_UNIQUENESS__customers__customer_id' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.customers)
           = (SELECT COUNT(DISTINCT customer_id) FROM silver.customers)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    ((SELECT COUNT(*) FROM silver.customers)
      - (SELECT COUNT(DISTINCT customer_id) FROM silver.customers))::VARCHAR AS observed,
    '0' AS expected
UNION ALL
SELECT
    'KEY_UNIQUENESS__sellers__seller_id' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.sellers)
           = (SELECT COUNT(DISTINCT seller_id) FROM silver.sellers)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    ((SELECT COUNT(*) FROM silver.sellers)
      - (SELECT COUNT(DISTINCT seller_id) FROM silver.sellers))::VARCHAR AS observed,
    '0' AS expected
UNION ALL
SELECT
    'KEY_UNIQUENESS__products__product_id' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.products)
           = (SELECT COUNT(DISTINCT product_id) FROM silver.products)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    ((SELECT COUNT(*) FROM silver.products)
      - (SELECT COUNT(DISTINCT product_id) FROM silver.products))::VARCHAR AS observed,
    '0' AS expected
UNION ALL
SELECT
    'KEY_UNIQUENESS__product_category_translation__product_category_name' AS check_name,
    CASE
        WHEN (SELECT COUNT(*) FROM silver.product_category_translation)
           = (SELECT COUNT(DISTINCT product_category_name) FROM silver.product_category_translation)
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    ((SELECT COUNT(*) FROM silver.product_category_translation)
      - (SELECT COUNT(DISTINCT product_category_name) FROM silver.product_category_translation))::VARCHAR AS observed,
    '0' AS expected
UNION ALL
SELECT
    'KEY_UNIQUENESS__order_items__order_id_item_id' AS check_name,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT (order_id, order_item_id))
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (COUNT(*) - COUNT(DISTINCT (order_id, order_item_id)))::VARCHAR AS observed,
    '0' AS expected
FROM silver.order_items
UNION ALL
SELECT
    'KEY_UNIQUENESS__payments__order_id_payment_sequential' AS check_name,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT (order_id, payment_sequential))
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    (COUNT(*) - COUNT(DISTINCT (order_id, payment_sequential)))::VARCHAR AS observed,
    '0' AS expected
FROM silver.payments
UNION ALL
SELECT
    'ITEM_TO_ORDER_ORPHANS' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM silver.order_items i LEFT JOIN silver.orders o ON i.order_id = o.order_id
WHERE o.order_id IS NULL
UNION ALL
SELECT
    'ITEM_TO_PRODUCT_ORPHANS' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM silver.order_items i LEFT JOIN silver.products p ON i.product_id = p.product_id
WHERE p.product_id IS NULL
UNION ALL
SELECT
    'ITEM_TO_SELLER_ORPHANS' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM silver.order_items i LEFT JOIN silver.sellers s ON i.seller_id = s.seller_id
WHERE s.seller_id IS NULL
UNION ALL
SELECT
    'ORDER_TO_CUSTOMER_ORPHANS' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM silver.orders o LEFT JOIN silver.customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL
UNION ALL
SELECT
    'PAYMENT_TO_ORDER_ORPHANS' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM silver.payments p LEFT JOIN silver.orders o ON p.order_id = o.order_id
WHERE o.order_id IS NULL
UNION ALL
SELECT
    'REVIEW_TO_ORDER_ORPHANS' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM silver.reviews r LEFT JOIN silver.orders o ON r.order_id = o.order_id
WHERE o.order_id IS NULL
UNION ALL
SELECT
    'CAST_ORDERS_PURCHASE_TIMESTAMP' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM bronze.orders b JOIN silver.orders s USING(order_id)
WHERE NULLIF(b.order_purchase_timestamp, '') IS NOT NULL AND s.order_purchase_timestamp IS NULL
UNION ALL
SELECT
    'CAST_ORDERS_ESTIMATED_TIMESTAMP' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM bronze.orders b JOIN silver.orders s USING(order_id)
WHERE NULLIF(b.order_estimated_delivery_date, '') IS NOT NULL AND s.order_estimated_delivery_date IS NULL
UNION ALL
SELECT
    'CAST_ITEMS_SHIPPING_LIMIT' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM bronze.order_items b
           JOIN silver.order_items s
             ON b.order_id = s.order_id
            AND TRY_CAST(b.order_item_id AS BIGINT) = s.order_item_id
WHERE NULLIF(b.shipping_limit_date, '') IS NOT NULL AND s.shipping_limit_date IS NULL
UNION ALL
SELECT
    'CAST_ITEMS_PRICE' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM bronze.order_items b
           JOIN silver.order_items s
             ON b.order_id = s.order_id
            AND TRY_CAST(b.order_item_id AS BIGINT) = s.order_item_id
WHERE NULLIF(b.price, '') IS NOT NULL AND s.price IS NULL
UNION ALL
SELECT
    'CAST_PAYMENTS_VALUE' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*)::VARCHAR AS observed,
    '0' AS expected
FROM bronze.payments b
           JOIN silver.payments s
             ON b.order_id = s.order_id
            AND TRY_CAST(b.payment_sequential AS BIGINT) = s.payment_sequential
WHERE NULLIF(b.payment_value, '') IS NOT NULL AND s.payment_value IS NULL;

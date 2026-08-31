CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE TABLE silver.orders AS
SELECT
    order_id::VARCHAR AS order_id,
    customer_id::VARCHAR AS customer_id,
    order_status::VARCHAR AS order_status,
    TRY_CAST(NULLIF(order_purchase_timestamp, '') AS TIMESTAMP) AS order_purchase_timestamp,
    TRY_CAST(NULLIF(order_approved_at, '') AS TIMESTAMP) AS order_approved_at,
    TRY_CAST(NULLIF(order_delivered_carrier_date, '') AS TIMESTAMP) AS order_delivered_carrier_date,
    TRY_CAST(NULLIF(order_delivered_customer_date, '') AS TIMESTAMP) AS order_delivered_customer_date,
    TRY_CAST(NULLIF(order_estimated_delivery_date, '') AS TIMESTAMP) AS order_estimated_delivery_date,
    _source_file
FROM bronze.orders;

CREATE OR REPLACE TABLE silver.order_items AS
SELECT
    order_id::VARCHAR AS order_id,
    TRY_CAST(NULLIF(order_item_id, '') AS BIGINT) AS order_item_id,
    product_id::VARCHAR AS product_id,
    seller_id::VARCHAR AS seller_id,
    TRY_CAST(NULLIF(shipping_limit_date, '') AS TIMESTAMP) AS shipping_limit_date,
    TRY_CAST(NULLIF(price, '') AS DOUBLE) AS price,
    TRY_CAST(NULLIF(freight_value, '') AS DOUBLE) AS freight_value,
    _source_file
FROM bronze.order_items;

CREATE OR REPLACE TABLE silver.customers AS
SELECT
    customer_id::VARCHAR AS customer_id,
    customer_unique_id::VARCHAR AS customer_unique_id,
    customer_zip_code_prefix::VARCHAR AS customer_zip_code_prefix,
    customer_city::VARCHAR AS customer_city,
    customer_state::VARCHAR AS customer_state,
    _source_file
FROM bronze.customers;

CREATE OR REPLACE TABLE silver.sellers AS
SELECT
    seller_id::VARCHAR AS seller_id,
    seller_zip_code_prefix::VARCHAR AS seller_zip_code_prefix,
    seller_city::VARCHAR AS seller_city,
    seller_state::VARCHAR AS seller_state,
    _source_file
FROM bronze.sellers;

CREATE OR REPLACE TABLE silver.products AS
SELECT
    product_id::VARCHAR AS product_id,
    product_category_name::VARCHAR AS product_category_name,
    TRY_CAST(NULLIF(product_name_lenght, '') AS BIGINT) AS product_name_lenght,
    TRY_CAST(NULLIF(product_description_lenght, '') AS BIGINT) AS product_description_lenght,
    TRY_CAST(NULLIF(product_photos_qty, '') AS BIGINT) AS product_photos_qty,
    TRY_CAST(NULLIF(product_weight_g, '') AS DOUBLE) AS product_weight_g,
    TRY_CAST(NULLIF(product_length_cm, '') AS DOUBLE) AS product_length_cm,
    TRY_CAST(NULLIF(product_height_cm, '') AS DOUBLE) AS product_height_cm,
    TRY_CAST(NULLIF(product_width_cm, '') AS DOUBLE) AS product_width_cm,
    _source_file
FROM bronze.products;

CREATE OR REPLACE TABLE silver.payments AS
SELECT
    order_id::VARCHAR AS order_id,
    TRY_CAST(NULLIF(payment_sequential, '') AS BIGINT) AS payment_sequential,
    payment_type::VARCHAR AS payment_type,
    TRY_CAST(NULLIF(payment_installments, '') AS BIGINT) AS payment_installments,
    TRY_CAST(NULLIF(payment_value, '') AS DOUBLE) AS payment_value,
    _source_file
FROM bronze.payments;

CREATE OR REPLACE TABLE silver.reviews AS
SELECT
    review_id::VARCHAR AS review_id,
    order_id::VARCHAR AS order_id,
    TRY_CAST(NULLIF(review_score, '') AS BIGINT) AS review_score,
    review_comment_title::VARCHAR AS review_comment_title,
    review_comment_message::VARCHAR AS review_comment_message,
    TRY_CAST(NULLIF(review_creation_date, '') AS TIMESTAMP) AS review_creation_date,
    TRY_CAST(NULLIF(review_answer_timestamp, '') AS TIMESTAMP) AS review_answer_timestamp,
    _source_file
FROM bronze.reviews;

CREATE OR REPLACE TABLE silver.geolocation AS
SELECT
    geolocation_zip_code_prefix::VARCHAR AS geolocation_zip_code_prefix,
    TRY_CAST(NULLIF(geolocation_lat, '') AS DOUBLE) AS geolocation_lat,
    TRY_CAST(NULLIF(geolocation_lng, '') AS DOUBLE) AS geolocation_lng,
    geolocation_city::VARCHAR AS geolocation_city,
    geolocation_state::VARCHAR AS geolocation_state,
    _source_file
FROM bronze.geolocation;

CREATE OR REPLACE TABLE silver.product_category_translation AS
SELECT
    product_category_name::VARCHAR AS product_category_name,
    product_category_name_english::VARCHAR AS product_category_name_english,
    _source_file
FROM bronze.product_category_translation;

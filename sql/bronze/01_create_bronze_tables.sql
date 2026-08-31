CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE TABLE bronze.orders AS
SELECT *, 'olist_orders_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_orders_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.order_items AS
SELECT *, 'olist_order_items_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_order_items_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.customers AS
SELECT *, 'olist_customers_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_customers_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.sellers AS
SELECT *, 'olist_sellers_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_sellers_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.products AS
SELECT *, 'olist_products_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_products_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.payments AS
SELECT *, 'olist_order_payments_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_order_payments_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.reviews AS
SELECT *, 'olist_order_reviews_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_order_reviews_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.geolocation AS
SELECT *, 'olist_geolocation_dataset.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/olist_geolocation_dataset.csv', header = true, all_varchar = true);

CREATE OR REPLACE TABLE bronze.product_category_translation AS
SELECT *, 'product_category_name_translation.csv'::VARCHAR AS _source_file
FROM read_csv_auto('{{RAW_DIR}}/product_category_name_translation.csv', header = true, all_varchar = true);

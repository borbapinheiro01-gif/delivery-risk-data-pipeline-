#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
RAW = ROOT / "data" / "raw" / "olist"
ART = ROOT / "artifacts" / "spatiotemporal_logistics"
REP = ROOT / "reports" / "spatiotemporal_logistics"

ART.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

YEARS = {2017, 2018}


def norm_text(value):
    if pd.isna(value):
        return ""

    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )

    s = s.upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_one(pattern):
    files = sorted(RAW.glob(pattern))

    if len(files) != 1:
        raise RuntimeError(
            f"Esperado 1 arquivo para {pattern}; encontrados: {files}"
        )

    return files[0]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088

    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        R
        * 2.0
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0.0, 1.0)
            )
        )
    )


print("=" * 110)
print("STAGE 01 — ROUTE CHECKPOINT")
print("=" * 110)

# --------------------------------------------------------------------------------------------------
# 1. ORDERS
# --------------------------------------------------------------------------------------------------

orders = pd.read_csv(
    find_one("*orders_dataset.csv"),
    usecols=[
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    low_memory=False,
)

for c in [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]:
    orders[c] = pd.to_datetime(
        orders[c],
        errors="coerce",
    )

orders["year"] = orders["order_purchase_timestamp"].dt.year
orders["month"] = orders["order_purchase_timestamp"].dt.month
orders["year_month"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m")

cohort_all = orders[
    orders["year"].isin(YEARS)
    &
    orders["order_purchase_timestamp"].notna()
].copy()

status_counts = (
    cohort_all["order_status"]
    .value_counts(dropna=False)
    .rename_axis("order_status")
    .reset_index(name="n")
)

status_counts.to_csv(
    REP / "01_cohort_status_counts.csv",
    index=False,
)

cohort = cohort_all[
    cohort_all["order_status"].eq("delivered")
].copy()

if cohort["order_id"].duplicated().any():
    raise RuntimeError("G01: order_id duplicado na coorte.")

print(f"[PASS] pedidos 2017-2018 totais  : {len(cohort_all):,}")
print(f"[PASS] pedidos delivered          : {len(cohort):,}")


# --------------------------------------------------------------------------------------------------
# 2. GEOLOCATION → ZIP LOOKUP
# --------------------------------------------------------------------------------------------------

geo = pd.read_csv(
    find_one("*geolocation_dataset.csv"),
    usecols=[
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    ],
    dtype={
        "geolocation_zip_code_prefix": "string",
    },
    low_memory=False,
)

geo["zip_prefix"] = (
    geo["geolocation_zip_code_prefix"]
    .astype("string")
    .str.zfill(5)
)

geo["lat"] = pd.to_numeric(
    geo["geolocation_lat"],
    errors="coerce",
)

geo["lon"] = pd.to_numeric(
    geo["geolocation_lng"],
    errors="coerce",
)

geo = geo[
    geo["lat"].between(-35.5, 6.5)
    &
    geo["lon"].between(-75.5, -32.0)
].copy()

geo_zip = (
    geo.groupby(
        "zip_prefix",
        as_index=False,
    )
    .agg(
        geo_lat_median=("lat", "median"),
        geo_lon_median=("lon", "median"),
        geo_n_points=("lat", "count"),
        geo_lat_min=("lat", "min"),
        geo_lat_max=("lat", "max"),
        geo_lon_min=("lon", "min"),
        geo_lon_max=("lon", "max"),
    )
)

geo_zip["geo_lat_spread"] = (
    geo_zip["geo_lat_max"]
    -
    geo_zip["geo_lat_min"]
)

geo_zip["geo_lon_spread"] = (
    geo_zip["geo_lon_max"]
    -
    geo_zip["geo_lon_min"]
)

geo_zip = geo_zip.drop(
    columns=[
        "geo_lat_min",
        "geo_lat_max",
        "geo_lon_min",
        "geo_lon_max",
    ]
)

del geo

print(f"[PASS] ZIP lookup: {len(geo_zip):,} prefixos")


# --------------------------------------------------------------------------------------------------
# 3. CUSTOMER / SELLER LOOKUPS
# --------------------------------------------------------------------------------------------------

customers = pd.read_csv(
    find_one("*customers_dataset.csv"),
    usecols=[
        "customer_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
    dtype={
        "customer_zip_code_prefix": "string",
    },
    low_memory=False,
)

customers["customer_zip_code_prefix"] = (
    customers["customer_zip_code_prefix"]
    .astype("string")
    .str.zfill(5)
)

customers["customer_city_norm"] = (
    customers["customer_city"].map(norm_text)
)

customers["customer_state"] = (
    customers["customer_state"]
    .astype("string")
    .str.upper()
    .str.strip()
)


sellers = pd.read_csv(
    find_one("*sellers_dataset.csv"),
    usecols=[
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ],
    dtype={
        "seller_zip_code_prefix": "string",
    },
    low_memory=False,
)

sellers["seller_zip_code_prefix"] = (
    sellers["seller_zip_code_prefix"]
    .astype("string")
    .str.zfill(5)
)

sellers["seller_city_norm"] = (
    sellers["seller_city"].map(norm_text)
)

sellers["seller_state"] = (
    sellers["seller_state"]
    .astype("string")
    .str.upper()
    .str.strip()
)


# --------------------------------------------------------------------------------------------------
# 4. ITEMS + PRODUCTS
# --------------------------------------------------------------------------------------------------

items = pd.read_csv(
    find_one("*order_items_dataset.csv"),
    usecols=[
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
    ],
    low_memory=False,
)

cohort_ids = set(cohort["order_id"])
items = items[
    items["order_id"].isin(cohort_ids)
].copy()

products = pd.read_csv(
    find_one("*products_dataset.csv"),
    usecols=[
        "product_id",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
    low_memory=False,
)

items = items.merge(
    products,
    on="product_id",
    how="left",
    validate="m:1",
)

for c in [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]:
    items[c] = pd.to_numeric(
        items[c],
        errors="coerce",
    )

items["product_volume_cm3"] = (
    items["product_length_cm"]
    *
    items["product_height_cm"]
    *
    items["product_width_cm"]
)

keys = ["order_id", "seller_id"]
grp = items.groupby(keys, sort=False)

rso_basic = grp.agg(
    seller_item_count=("order_item_id", "count"),
    seller_unique_products=("product_id", "nunique"),
    seller_price=("price", "sum"),
    seller_freight=("freight_value", "sum"),
).reset_index()

physical_sums = (
    grp[
        [
            "product_weight_g",
            "product_volume_cm3",
        ]
    ]
    .sum(min_count=1)
    .reset_index()
    .rename(
        columns={
            "product_weight_g":
                "seller_weight_g",
            "product_volume_cm3":
                "seller_volume_proxy_cm3",
        }
    )
)

physical_counts = (
    grp[
        [
            "product_weight_g",
            "product_volume_cm3",
        ]
    ]
    .count()
    .reset_index()
    .rename(
        columns={
            "product_weight_g":
                "seller_items_with_weight",
            "product_volume_cm3":
                "seller_items_with_volume",
        }
    )
)

rso = (
    rso_basic
    .merge(
        physical_sums,
        on=keys,
        how="left",
        validate="1:1",
    )
    .merge(
        physical_counts,
        on=keys,
        how="left",
        validate="1:1",
    )
)

rso["seller_weight_complete"] = (
    rso["seller_items_with_weight"]
    .eq(rso["seller_item_count"])
    .astype("int8")
)

rso["seller_volume_complete"] = (
    rso["seller_items_with_volume"]
    .eq(rso["seller_item_count"])
    .astype("int8")
)

del items, products, rso_basic, physical_sums, physical_counts


# --------------------------------------------------------------------------------------------------
# 5. METADATA + COORDINATES
# --------------------------------------------------------------------------------------------------

rso = rso.merge(
    cohort[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "year",
            "month",
            "year_month",
        ]
    ],
    on="order_id",
    how="inner",
    validate="m:1",
)

rso = rso.merge(
    customers[
        [
            "customer_id",
            "customer_zip_code_prefix",
            "customer_city_norm",
            "customer_state",
        ]
    ],
    on="customer_id",
    how="left",
    validate="m:1",
)

rso = rso.merge(
    sellers[
        [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city_norm",
            "seller_state",
        ]
    ],
    on="seller_id",
    how="left",
    validate="m:1",
)

seller_geo = geo_zip.rename(
    columns={
        "zip_prefix": "seller_zip_code_prefix",
        "geo_lat_median": "seller_lat",
        "geo_lon_median": "seller_lon",
        "geo_n_points": "seller_geo_n_points",
        "geo_lat_spread": "seller_geo_lat_spread",
        "geo_lon_spread": "seller_geo_lon_spread",
    }
)

customer_geo = geo_zip.rename(
    columns={
        "zip_prefix": "customer_zip_code_prefix",
        "geo_lat_median": "customer_lat",
        "geo_lon_median": "customer_lon",
        "geo_n_points": "customer_geo_n_points",
        "geo_lat_spread": "customer_geo_lat_spread",
        "geo_lon_spread": "customer_geo_lon_spread",
    }
)

rso = rso.merge(
    seller_geo,
    on="seller_zip_code_prefix",
    how="left",
    validate="m:1",
)

rso = rso.merge(
    customer_geo,
    on="customer_zip_code_prefix",
    how="left",
    validate="m:1",
)

rso["great_circle_distance_km"] = haversine_km(
    rso["seller_lat"],
    rso["seller_lon"],
    rso["customer_lat"],
    rso["customer_lon"],
)

if rso.duplicated(["order_id", "seller_id"]).any():
    raise RuntimeError("G03: order×seller duplicado.")

if (rso["great_circle_distance_km"].dropna() < 0).any():
    raise RuntimeError("G04: distância negativa.")


# --------------------------------------------------------------------------------------------------
# 6. ROUTE FLAGS
# --------------------------------------------------------------------------------------------------

state_known = (
    rso["seller_state"].notna()
    &
    rso["customer_state"].notna()
)

rso["state_comparison_available"] = state_known.astype("int8")

rso["is_interstate"] = np.where(
    state_known,
    rso["seller_state"].ne(rso["customer_state"]).astype(int),
    np.nan,
)

rso["weighted_dist"] = (
    rso["great_circle_distance_km"]
    *
    rso["seller_freight"]
)

rso["interstate_freight"] = np.where(
    rso["is_interstate"].eq(1),
    rso["seller_freight"],
    0.0,
)

rso.to_csv(
    ART / "01_ROUTE_SELLER_ORDER.csv",
    index=False,
)


# --------------------------------------------------------------------------------------------------
# 7. ORDER LEVEL — VETORIZADO
# --------------------------------------------------------------------------------------------------

g = rso.groupby("order_id", sort=False)

route_summary = g.agg(
    route_sellers_total=("seller_id", "nunique"),
    route_sellers_with_distance=("great_circle_distance_km", "count"),
    distance_min_km=("great_circle_distance_km", "min"),
    distance_mean_km=("great_circle_distance_km", "mean"),
    distance_max_km=("great_circle_distance_km", "max"),
    total_price=("seller_price", "sum"),
    total_freight=("seller_freight", "sum"),
    weight_complete=("seller_weight_complete", "min"),
    volume_complete=("seller_volume_complete", "min"),
    state_comparisons_available=("state_comparison_available", "sum"),
    any_interstate_route=("is_interstate", "max"),
).reset_index()

sums = (
    g[
        [
            "weighted_dist",
            "seller_weight_g",
            "seller_volume_proxy_cm3",
            "interstate_freight",
        ]
    ]
    .sum(min_count=1)
    .reset_index()
    .rename(
        columns={
            "seller_weight_g":
                "total_weight_g",
            "seller_volume_proxy_cm3":
                "product_volume_sum_proxy_cm3",
        }
    )
)

route_summary = route_summary.merge(
    sums,
    on="order_id",
    how="left",
    validate="1:1",
)

route_summary["distance_freight_weighted_km"] = np.where(
    route_summary["total_freight"] > 0,
    (
        route_summary["weighted_dist"]
        /
        route_summary["total_freight"]
    ),
    route_summary["distance_mean_km"],
)

route_summary["route_distance_coverage"] = (
    route_summary["route_sellers_with_distance"]
    /
    route_summary["route_sellers_total"]
)

route_summary["single_seller_order"] = (
    route_summary["route_sellers_total"]
    .eq(1)
    .astype("int8")
)

route_summary["state_route_complete"] = (
    route_summary["state_comparisons_available"]
    .eq(route_summary["route_sellers_total"])
    .astype("int8")
)

route_summary["same_state_all"] = np.where(
    route_summary["state_route_complete"].eq(1),
    route_summary["any_interstate_route"].fillna(0).eq(0).astype(int),
    np.nan,
)

route_summary["interstate_freight_share"] = np.where(
    route_summary["total_freight"] > 0,
    (
        route_summary["interstate_freight"]
        /
        route_summary["total_freight"]
    ),
    np.nan,
)

route_summary = route_summary.drop(
    columns=[
        "weighted_dist",
        "interstate_freight",
    ]
)

route_summary.to_csv(
    ART / "02_ROUTE_ORDER_LEVEL.csv",
    index=False,
)

print(f"[PASS 01] ROUTE_SELLER_ORDER = {len(rso):,}")
print(f"[PASS 01] ROUTE_ORDER_LEVEL   = {len(route_summary):,}")
print("[PASS 01] RAW_MODIFIED = false")

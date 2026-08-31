#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
RAW = ROOT / "data" / "raw" / "olist"
ART = ROOT / "artifacts" / "spatiotemporal_logistics"
REP = ROOT / "reports" / "spatiotemporal_logistics"


def find_one(pattern):
    files = sorted(RAW.glob(pattern))

    if len(files) != 1:
        raise RuntimeError(
            f"Esperado 1 arquivo: {pattern}"
        )

    return files[0]


print("=" * 110)
print("STAGE 03 — MASTER ORDER TABLE")
print("=" * 110)

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

orders["year"] = (
    orders["order_purchase_timestamp"].dt.year
)

orders["month"] = (
    orders["order_purchase_timestamp"].dt.month
)

orders["year_month"] = (
    orders["order_purchase_timestamp"]
    .dt.strftime("%Y-%m")
)

orders["purchase_dow"] = (
    orders["order_purchase_timestamp"]
    .dt.dayofweek
)

cohort = orders[
    orders["year"].isin([2017, 2018])
    &
    orders["order_status"].eq("delivered")
].copy()

master = cohort[
    [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "year",
        "month",
        "year_month",
        "purchase_dow",
    ]
].copy()

master["actual_delivery_days"] = (
    (
        master["order_delivered_customer_date"]
        -
        master["order_purchase_timestamp"]
    )
    .dt.total_seconds()
    /
    86400.0
)

master["promised_delivery_days"] = (
    (
        master["order_estimated_delivery_date"]
        -
        master["order_purchase_timestamp"]
    )
    .dt.total_seconds()
    /
    86400.0
)

master["lateness_days"] = (
    (
        master["order_delivered_customer_date"]
        -
        master["order_estimated_delivery_date"]
    )
    .dt.total_seconds()
    /
    86400.0
)

master["late_delivery_calendar_day"] = (
    master["order_delivered_customer_date"]
    .dt.floor("D")
    >
    master["order_estimated_delivery_date"]
    .dt.floor("D")
).astype("int8")

master["on_time"] = (
    1
    -
    master["late_delivery_calendar_day"]
).astype("int8")

master["truck_strike_2018"] = (
    master["order_purchase_timestamp"]
    .between(
        "2018-05-21",
        "2018-05-31 23:59:59",
    )
    .astype("int8")
)

master["anp_strike_measurement_disruption_2018"] = (
    master["order_purchase_timestamp"]
    .between(
        "2018-05-27",
        "2018-06-02 23:59:59",
    )
    .astype("int8")
)

route_order = pd.read_csv(
    ART / "02_ROUTE_ORDER_LEVEL.csv"
)

master = master.merge(
    route_order,
    on="order_id",
    how="left",
    validate="1:1",
)

route_seller = pd.read_csv(
    ART / "01_ROUTE_SELLER_ORDER.csv"
)

primary = (
    route_seller.sort_values(
        [
            "order_id",
            "seller_freight",
            "seller_price",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )
    .drop_duplicates(
        "order_id",
        keep="first",
    )
)

master = master.merge(
    primary[
        [
            "order_id",
            "seller_id",
            "customer_city_norm",
            "customer_state",
            "seller_city_norm",
            "seller_state",
        ]
    ],
    on="order_id",
    how="left",
    validate="1:1",
)


# --------------------------------------------------------------------------------------------------
# MUNICIPAL CONTEXT — destination
# --------------------------------------------------------------------------------------------------

muni = pd.read_csv(
    ART / "03_MUNICIPAL_CONTEXT.csv",
    dtype={
        "id_municipio": "string",
    },
)

dest = muni.rename(
    columns={
        "uf":
            "customer_state",
        "city_norm":
            "customer_city_norm",
        "id_municipio":
            "customer_id_municipio",
        "population":
            "customer_population",
        "log_population":
            "customer_log_population",
        "gdp_current":
            "customer_gdp_current",
        "gdp_per_capita":
            "customer_gdp_per_capita",
        "log_gdp_per_capita":
            "customer_log_gdp_per_capita",
    }
)

master = master.merge(
    dest[
        [
            "year",
            "customer_state",
            "customer_city_norm",
            "customer_id_municipio",
            "customer_population",
            "customer_log_population",
            "customer_gdp_current",
            "customer_gdp_per_capita",
            "customer_log_gdp_per_capita",
        ]
    ],
    on=[
        "year",
        "customer_state",
        "customer_city_norm",
    ],
    how="left",
    validate="m:1",
)


# --------------------------------------------------------------------------------------------------
# ANP MUNICIPAL
# --------------------------------------------------------------------------------------------------

anp = pd.read_csv(
    ART / "04_ANP_CONTEXT.csv"
)

anp_dest = anp.rename(
    columns={
        "uf":
            "customer_state",
        "city_norm":
            "customer_city_norm",
        "diesel_common_mean":
            "customer_diesel_common_municipal",
        "diesel_s10_mean":
            "customer_diesel_s10_municipal",
        "diesel_common_n_posts":
            "customer_diesel_common_municipal_n_posts",
        "diesel_s10_n_posts":
            "customer_diesel_s10_municipal_n_posts",
    }
)

master = master.merge(
    anp_dest[
        [
            "year",
            "month",
            "customer_state",
            "customer_city_norm",
            "customer_diesel_common_municipal",
            "customer_diesel_s10_municipal",
            "customer_diesel_common_municipal_n_posts",
            "customer_diesel_s10_municipal_n_posts",
        ]
    ],
    on=[
        "year",
        "month",
        "customer_state",
        "customer_city_norm",
    ],
    how="left",
    validate="m:1",
)


# --------------------------------------------------------------------------------------------------
# ANP STATE — separate, never overwriting municipality
# --------------------------------------------------------------------------------------------------

anp_state = pd.read_csv(
    ART / "04b_ANP_STATE_CONTEXT.csv"
).rename(
    columns={
        "uf":
            "customer_state",
        "diesel_common_state_mean":
            "customer_diesel_common_state",
        "diesel_s10_state_mean":
            "customer_diesel_s10_state",
        "diesel_common_state_n_posts":
            "customer_diesel_common_state_n_posts",
        "diesel_s10_state_n_posts":
            "customer_diesel_s10_state_n_posts",
    }
)

master = master.merge(
    anp_state,
    on=[
        "year",
        "month",
        "customer_state",
    ],
    how="left",
    validate="m:1",
)

municipal_available = (
    master[
        [
            "customer_diesel_common_municipal",
            "customer_diesel_s10_municipal",
        ]
    ]
    .notna()
    .any(axis=1)
)

state_available = (
    master[
        [
            "customer_diesel_common_state",
            "customer_diesel_s10_state",
        ]
    ]
    .notna()
    .any(axis=1)
)

master["customer_diesel_geo_level"] = np.select(
    [
        municipal_available,
        (~municipal_available)
        &
        state_available,
    ],
    [
        "MUNICIPAL",
        "STATE_ONLY",
    ],
    default="MISSING",
)

if master["order_id"].duplicated().any():
    raise RuntimeError(
        "Master criou order_id duplicado."
    )

master.to_csv(
    ART / "05_SPATIOTEMPORAL_CONTEXT_CORE.csv",
    index=False,
)


# --------------------------------------------------------------------------------------------------
# COVERAGE
# --------------------------------------------------------------------------------------------------

coverage_fields = [
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "customer_population",
    "customer_gdp_per_capita",
    "customer_diesel_common_municipal",
    "customer_diesel_common_state",
    "actual_delivery_days",
]

coverage = []

for c in coverage_fields:

    n = int(
        master[c]
        .notna()
        .sum()
    )

    coverage.append(
        {
            "field": c,
            "n_non_null": n,
            "n_orders": len(master),
            "coverage_pct":
                100.0 * n / len(master),
        }
    )

pd.DataFrame(
    coverage
).to_csv(
    REP / "02_join_coverage.csv",
    index=False,
)

print(f"[PASS 03] MASTER = {len(master):,} pedidos")
print("[PASS 03] duplicate_order_id = 0")

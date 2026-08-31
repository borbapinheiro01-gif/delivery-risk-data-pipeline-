#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GATE 05.1 — POINT-IN-TIME FEATURE REGISTRY
Delivery Risk Intelligence Platform
================================================================================

OBJETIVO
--------
Classificar TODAS as colunas RAW segundo sua elegibilidade temporal para
predição no instante:

    t0 = order_purchase_timestamp

REGRA
-----
Uma feature X_ij somente pode ser usada quando:

    t_available(X_ij) <= t0_i

IMPORTANTE
----------
Este passo NÃO:

- treina modelo;
- cria Silver;
- imputa dados;
- elimina registros;
- altera RAW;
- resolve payment provenance;
- resolve shipping_limit_date;
- constrói seller history;
- cria Registry 1.3.

É somente o inventário/contrato temporal das colunas RAW.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

RAW = (
    PROJECT
    / "data"
    / "raw"
    / "olist"
)

OUT = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_05_point_in_time"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# RAW FILES
# =============================================================================

FILES = {
    "customers":
        "olist_customers_dataset.csv",

    "geolocation":
        "olist_geolocation_dataset.csv",

    "order_items":
        "olist_order_items_dataset.csv",

    "payments":
        "olist_order_payments_dataset.csv",

    "reviews":
        "olist_order_reviews_dataset.csv",

    "orders":
        "olist_orders_dataset.csv",

    "products":
        "olist_products_dataset.csv",

    "sellers":
        "olist_sellers_dataset.csv",

    "translation":
        "product_category_name_translation.csv",
}


# =============================================================================
# CLASSIFICATION HELPER
# =============================================================================

def result(
    status,
    module,
    direct_feature,
    aggregation,
    rationale,
    unresolved_issue=""
):
    return {
        "point_in_time_status":
            status,

        "feature_module":
            module,

        "direct_model_feature":
            direct_feature,

        "aggregation_policy":
            aggregation,

        "rationale":
            rationale,

        "unresolved_issue":
            unresolved_issue
    }


def classify(table, column):

    key = f"{table}.{column}"


    # =========================================================================
    # ORDERS
    # =========================================================================

    if key == "orders.order_id":
        return result(
            "IDENTIFIER_ONLY",
            "NONE",
            False,
            "NONE",
            "Primary order identifier. Preserve for lineage; never use as predictor."
        )

    if key == "orders.customer_id":
        return result(
            "JOIN_KEY_ONLY",
            "GEOGRAPHY",
            False,
            "JOIN_ONLY",
            "Order-specific customer foreign key."
        )

    if key == "orders.order_purchase_timestamp":
        return result(
            "DERIVED_SAFE_AT_T0",
            "TEMPORAL",
            False,
            "DERIVE_CALENDAR_FEATURES",
            (
                "Defines prediction time t0. "
                "Use derived hour/weekday/month features instead of raw absolute timestamp."
            )
        )

    if key == "orders.order_estimated_delivery_date":
        return result(
            "DERIVED_SAFE_AT_T0",
            "GEO_PROMISE",
            False,
            "DERIVE_PROMISED_LEAD_TIME",
            (
                "Delivery promise associated with the purchase. "
                "Primary derived feature: promised_lead_days."
            )
        )

    if key == "orders.order_status":
        return result(
            "FORBIDDEN_FUTURE",
            "FORBIDDEN",
            False,
            "NONE",
            (
                "Raw dataset contains final/later order state. "
                "Using it at purchase time would leak post-t0 information."
            )
        )

    if key == "orders.order_approved_at":
        return result(
            "FORBIDDEN_FUTURE",
            "FORBIDDEN",
            False,
            "NONE",
            "Approval is a post-purchase process event."
        )

    if key == "orders.order_delivered_carrier_date":
        return result(
            "FORBIDDEN_FUTURE",
            "FORBIDDEN",
            False,
            "NONE",
            "Carrier handoff occurs after purchase."
        )

    if key == "orders.order_delivered_customer_date":
        return result(
            "TARGET_ONLY",
            "TARGET",
            False,
            "LABEL_ONLY",
            "Observed customer delivery is part of target construction."
        )


    # =========================================================================
    # ORDER ITEMS
    # =========================================================================

    if key == "order_items.order_id":
        return result(
            "JOIN_KEY_ONLY",
            "ORDER",
            False,
            "JOIN_ONLY",
            "Child-table key used to aggregate order_items to order grain."
        )

    if key == "order_items.order_item_id":
        return result(
            "AGGREGATE_SAFE_AT_T0",
            "ORDER",
            False,
            "COUNT",
            "Use to derive item_count at order level."
        )

    if key == "order_items.product_id":
        return result(
            "JOIN_KEY_ONLY",
            "ORDER",
            False,
            "COUNT_DISTINCT_AND_JOIN",
            (
                "Raw product ID is not a direct predictor. "
                "Use for product joins and unique product counts."
            )
        )

    if key == "order_items.seller_id":
        return result(
            "JOIN_KEY_ONLY",
            "SELLER_HISTORY",
            False,
            "COUNT_DISTINCT_AND_JOIN",
            (
                "Raw seller ID is not used directly. "
                "Use for seller aggregation and future point-in-time seller history."
            )
        )

    if key == "order_items.shipping_limit_date":
        return result(
            "HOLD_PROVENANCE",
            "GEO_PROMISE",
            False,
            "BLOCKED",
            (
                "Presence in final CSV does not prove operational availability "
                "at exact purchase time."
            ),
            "ISSUE-SHIPPING-LIMIT-001"
        )

    if key == "order_items.price":
        return result(
            "AGGREGATE_SAFE_AT_T0",
            "ORDER",
            False,
            "SUM_MEAN_MAX_MIN",
            "Current-order commercial value; aggregate to one row per order."
        )

    if key == "order_items.freight_value":
        return result(
            "AGGREGATE_SAFE_AT_T0",
            "ORDER",
            False,
            "SUM_MEAN_MAX_RATIO",
            "Current-order freight value; aggregate to one row per order."
        )


    # =========================================================================
    # PAYMENTS
    # =========================================================================

    if key == "payments.order_id":
        return result(
            "JOIN_KEY_ONLY",
            "PAYMENT",
            False,
            "JOIN_ONLY",
            "Payment child-table key."
        )

    if table == "payments":
        return result(
            "HOLD_PROVENANCE",
            "PAYMENT",
            False,
            "BLOCKED",
            (
                "Exact availability at order_purchase_timestamp has not yet "
                "been independently established."
            ),
            "ISSUE-PAYMENT-AVAILABILITY-001"
        )


    # =========================================================================
    # REVIEWS
    # =========================================================================

    if table == "reviews":

        if column in {
            "review_id",
            "order_id"
        }:
            policy = "IDENTIFIER_OR_JOIN_ONLY"
        else:
            policy = "NONE"

        return result(
            "FORBIDDEN_FUTURE",
            "POST_DELIVERY_NLP",
            False,
            policy,
            (
                "Review data belongs to post-purchase/post-delivery context "
                "and is forbidden in the preventive t0 model."
            )
        )


    # =========================================================================
    # CUSTOMERS
    # =========================================================================

    if key == "customers.customer_id":
        return result(
            "JOIN_KEY_ONLY",
            "GEOGRAPHY",
            False,
            "JOIN_ONLY",
            "Order-specific customer lookup key."
        )

    if key == "customers.customer_unique_id":
        return result(
            "IDENTIFIER_ONLY",
            "NONE",
            False,
            "NONE",
            (
                "Stable customer identity is preserved for lineage but "
                "not admitted as raw high-cardinality model feature."
            )
        )

    if column in {
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state"
    } and table == "customers":
        return result(
            "SAFE_CONTEXT_AT_T0",
            "GEOGRAPHY",
            True,
            "ORDER_REFERENCE_JOIN",
            "Customer destination context associated with the current order."
        )


    # =========================================================================
    # PRODUCTS
    # =========================================================================

    if key == "products.product_id":
        return result(
            "JOIN_KEY_ONLY",
            "ORDER",
            False,
            "JOIN_ONLY",
            "Product catalog lookup key."
        )

    if table == "products":
        return result(
            "STATIC_REFERENCE_ASSUMPTION",
            "ORDER",
            False,
            "AGGREGATE_TO_ORDER",
            (
                "Product catalog attribute can support order features. "
                "Historical catalog snapshots are unavailable, so static-reference "
                "assumption must remain documented."
            )
        )


    # =========================================================================
    # SELLERS
    # =========================================================================

    if key == "sellers.seller_id":
        return result(
            "JOIN_KEY_ONLY",
            "SELLER_HISTORY",
            False,
            "JOIN_ONLY",
            "Seller lookup/entity key."
        )

    if table == "sellers":
        return result(
            "STATIC_REFERENCE_ASSUMPTION",
            "GEOGRAPHY",
            False,
            "AGGREGATE_TO_ORDER",
            (
                "Seller location reference is useful for geography. "
                "Historical seller-address snapshots are unavailable."
            )
        )


    # =========================================================================
    # GEOLOCATION
    # =========================================================================

    if table == "geolocation":
        return result(
            "HOLD_DATA_QUALITY",
            "GEOGRAPHY",
            False,
            "ZIP_LEVEL_CONSOLIDATION_REQUIRED",
            (
                "Geographic reference is static, but duplicated/dispersed ZIP "
                "observations require a documented consolidation rule first."
            ),
            "ISSUE-GEO-001"
        )


    # =========================================================================
    # CATEGORY TRANSLATION
    # =========================================================================

    if (
        table == "translation"
        and
        column == "product_category_name"
    ):
        return result(
            "JOIN_KEY_ONLY",
            "ORDER",
            False,
            "JOIN_ONLY",
            "Category translation lookup key."
        )

    if (
        table == "translation"
        and
        column == "product_category_name_english"
    ):
        return result(
            "STATIC_REFERENCE_ASSUMPTION",
            "ORDER",
            False,
            "DISPLAY_OR_CATEGORY_MAPPING",
            "Human-readable category translation."
        )


    # =========================================================================
    # FALLBACK
    # =========================================================================

    return result(
        "REVIEW_REQUIRED",
        "UNASSIGNED",
        False,
        "NONE",
        "No explicit point-in-time rule has been assigned."
    )


# =============================================================================
# READ RAW HEADERS
# =============================================================================

rows = []


for table, filename in FILES.items():

    path = RAW / filename

    if not path.exists():
        print(
            f"[ERRO] RAW ausente: {path}"
        )
        sys.exit(2)

    header = pd.read_csv(
        path,
        nrows=0
    )

    for column in header.columns:

        classification = classify(
            table,
            column
        )

        rows.append(
            {
                "table":
                    table,

                "column":
                    column,

                "column_key":
                    f"{table}.{column}",

                **classification
            }
        )


registry = pd.DataFrame(
    rows
)


# =============================================================================
# INTERNAL VALIDATION
# =============================================================================

checks = []


def add_check(
    check,
    status,
    observed,
    expected
):

    checks.append(
        {
            "check":
                check,

            "status":
                status,

            "observed":
                observed,

            "expected":
                expected
        }
    )


add_check(
    "raw_column_count",
    "PASS" if len(registry) == 52 else "FAIL",
    len(registry),
    52
)


duplicate_keys = int(
    registry[
        "column_key"
    ]
    .duplicated()
    .sum()
)


add_check(
    "column_key_unique",
    "PASS" if duplicate_keys == 0 else "FAIL",
    duplicate_keys,
    0
)


review_required = int(
    (
        registry[
            "point_in_time_status"
        ]
        ==
        "REVIEW_REQUIRED"
    )
    .sum()
)


add_check(
    "all_columns_classified",
    "PASS" if review_required == 0 else "FAIL",
    review_required,
    0
)


future_direct = int(
    (
        registry[
            "point_in_time_status"
        ]
        .isin(
            [
                "FORBIDDEN_FUTURE",
                "TARGET_ONLY"
            ]
        )
        &
        registry[
            "direct_model_feature"
        ]
        .eq(True)
    )
    .sum()
)


add_check(
    "future_target_not_direct_feature",
    "PASS" if future_direct == 0 else "FAIL",
    future_direct,
    0
)


shipping_status = (
    registry.loc[
        registry[
            "column_key"
        ]
        ==
        "order_items.shipping_limit_date",
        "point_in_time_status"
    ]
    .iloc[0]
)


add_check(
    "shipping_limit_remains_hold",
    (
        "PASS"
        if shipping_status == "HOLD_PROVENANCE"
        else "FAIL"
    ),
    shipping_status,
    "HOLD_PROVENANCE"
)


payment_rows = registry[
    (
        registry[
            "table"
        ]
        ==
        "payments"
    )
    &
    (
        registry[
            "column"
        ]
        !=
        "order_id"
    )
]


payment_hold_count = int(
    (
        payment_rows[
            "point_in_time_status"
        ]
        ==
        "HOLD_PROVENANCE"
    )
    .sum()
)


add_check(
    "payment_features_remain_hold",
    (
        "PASS"
        if payment_hold_count == len(payment_rows)
        else "FAIL"
    ),
    payment_hold_count,
    len(payment_rows)
)


review_df = pd.DataFrame(
    checks
)


step_failures = int(
    (
        review_df[
            "status"
        ]
        ==
        "FAIL"
    )
    .sum()
)


step_status = (
    "PASS"
    if step_failures == 0
    else "FAIL"
)


# =============================================================================
# OUTPUT TABLES
# =============================================================================

registry.to_csv(
    OUT
    / "01_raw_feature_point_in_time_registry.csv",
    index=False
)


status_summary = (
    registry[
        "point_in_time_status"
    ]
    .value_counts()
    .rename_axis(
        "point_in_time_status"
    )
    .reset_index(
        name="columns"
    )
)


status_summary.to_csv(
    OUT
    / "01b_point_in_time_status_summary.csv",
    index=False
)


forbidden = registry[
    registry[
        "point_in_time_status"
    ]
    .isin(
        [
            "FORBIDDEN_FUTURE",
            "TARGET_ONLY"
        ]
    )
].copy()


forbidden.to_csv(
    OUT
    / "01c_forbidden_target_columns.csv",
    index=False
)


hold = registry[
    registry[
        "point_in_time_status"
    ]
    .isin(
        [
            "HOLD_PROVENANCE",
            "HOLD_DATA_QUALITY"
        ]
    )
].copy()


hold.to_csv(
    OUT
    / "01d_hold_columns.csv",
    index=False
)


review_df.to_csv(
    OUT
    / "01e_step_validation.csv",
    index=False
)


# =============================================================================
# SUMMARY
# =============================================================================

summary = {
    "step":
        "DQ_GATE_05_1_POINT_IN_TIME_REGISTRY",

    "step_status":
        step_status,

    "gate_05_status":
        "IN_PROGRESS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "prediction_time":
        "order_purchase_timestamp",

    "raw_tables":
        len(FILES),

    "raw_columns_audited":
        len(registry),

    "classified_columns":
        int(
            (
                registry[
                    "point_in_time_status"
                ]
                !=
                "REVIEW_REQUIRED"
            )
            .sum()
        ),

    "review_required":
        review_required,

    "forbidden_or_target":
        len(forbidden),

    "hold_columns":
        len(hold),

    "step_validation_failures":
        step_failures,

    "raw_modified":
        False,

    "silver_created":
        False,

    "model_trained":
        False
}


with (
    OUT
    / "dq_gate_05_1_summary.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# =============================================================================
# REPORT
# =============================================================================

report = f"""
========================================================================================================
DQ GATE 05.1 — POINT-IN-TIME FEATURE REGISTRY
========================================================================================================

STEP STATUS                     : {step_status}
GATE 05 STATUS                  : IN_PROGRESS

PREDICTION TIME                 : order_purchase_timestamp

RAW TABLES                      : {len(FILES)}
RAW COLUMNS AUDITED             : {len(registry)}
REVIEW REQUIRED                 : {review_required}
FORBIDDEN / TARGET              : {len(forbidden)}
HOLD                            : {len(hold)}

STEP VALIDATION FAILURES        : {step_failures}

RULE
--------------------------------------------------------------------------------------------------------
A feature X_ij is eligible only if:

    t_available(X_ij) <= t0_i

with:

    t0_i = order_purchase_timestamp_i

IMPORTANT
--------------------------------------------------------------------------------------------------------
This step classifies RAW fields only.

It does NOT:
- train any model;
- create Silver;
- resolve payment provenance;
- resolve shipping_limit_date provenance;
- consolidate geolocation;
- construct historical seller features;
- create Registry 1.3.

RAW modified                    : NO
Silver created                  : NO
Model trained                   : NO

========================================================================================================
"""


(
    OUT
    / "DQ_GATE_05_1_POINT_IN_TIME_REPORT.txt"
).write_text(
    report.strip() + "\n",
    encoding="utf-8"
)


# =============================================================================
# TERMINAL
# =============================================================================

print()
print("=" * 108)
print("DQ GATE 05.1 — POINT-IN-TIME FEATURE REGISTRY")
print("=" * 108)
print()

print(
    review_df.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("STATUS SUMMARY")
print("=" * 108)

print(
    status_summary.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("FORBIDDEN / TARGET")
print("=" * 108)

print(
    forbidden[
        [
            "column_key",
            "point_in_time_status",
            "feature_module",
            "rationale"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 108)
print("HOLD")
print("=" * 108)

print(
    hold[
        [
            "column_key",
            "point_in_time_status",
            "feature_module",
            "unresolved_issue",
            "rationale"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 108)
print("RESULTADO")
print("=" * 108)

print(
    f"STEP STATUS       : {step_status}"
)

print(
    "GATE 05 STATUS    : IN_PROGRESS"
)

print(
    f"RAW COLUMNS       : {len(registry)}"
)

print(
    f"UNCLASSIFIED      : {review_required}"
)

print(
    f"FAILURES          : {step_failures}"
)

print(
    "RAW MODIFIED      : NÃO"
)

print(
    "SILVER CREATED    : NÃO"
)

print(
    "MODEL TRAINED     : NÃO"
)


if step_status != "PASS":
    sys.exit(2)


print()
print(
    "[PASS] ETAPA 05.1 VALIDADA."
)

print(
    "Próxima etapa permanece bloqueada até revisão destes resultados."
)


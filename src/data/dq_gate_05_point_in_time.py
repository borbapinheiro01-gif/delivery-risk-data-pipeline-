#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
DQ GATE 05 — POINT-IN-TIME AVAILABILITY & LEAKAGE
Delivery Risk Intelligence Platform
================================================================================

OBJETIVO
--------
Construir o contrato de disponibilidade temporal das features antes de
qualquer treinamento.

REGRA FUNDAMENTAL
-----------------
Para uma feature X_ij ser utilizada para prever o pedido i:

    t_available(X_ij) <= t0_i

onde:

    t0_i = order_purchase_timestamp_i

O Gate separa:

SAFE_AT_T0
DERIVED_SAFE
STATIC_REFERENCE
POINT_IN_TIME_REQUIRED
HOLD_PROVENANCE
HOLD_DATA_QUALITY
FORBIDDEN_FUTURE
TARGET_ONLY
IDENTIFIER_ONLY
JOIN_KEY_ONLY

NÃO FAZ
-------
- não treina modelo;
- não imputa;
- não cria Silver;
- não altera RAW;
- não resolve automaticamente questões de provenance.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
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

REG12 = (
    PROJECT
    / "metadata"
    / "registry_v1_2"
)

REG13 = (
    PROJECT
    / "metadata"
    / "registry_v1_3"
)

CONTRACT = (
    PROJECT
    / "contracts"
    / "DELIVERY_RISK_LABEL_CONTRACT_V1.json"
)

GATE04B = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04b_label_contract"
    / "dq_gate_04b_summary.json"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

REG13.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path):

    if not path.exists():

        raise SystemExit(
            f"[ERRO] Arquivo obrigatório ausente:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(
    path,
    obj
):

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )


# =============================================================================
# 1. PRÉ-REQUISITOS
# =============================================================================

print("=" * 108)
print("DQ GATE 05 — POINT-IN-TIME AVAILABILITY & LEAKAGE")
print("=" * 108)
print()


gate04b = load_json(
    GATE04B
)

label_contract = load_json(
    CONTRACT
)

reg12_manifest = load_json(
    REG12
    / "registry_v1_2_manifest.json"
)


if gate04b.get(
    "status"
) != "PASS":

    raise SystemExit(
        "[BLOQUEADO] Gate 04B não está PASS."
    )


if label_contract.get(
    "status"
) != "FROZEN_FOR_MODELING":

    raise SystemExit(
        "[BLOQUEADO] Label Contract não está congelado."
    )


if reg12_manifest.get(
    "internal_registry_status"
) != "PASS":

    raise SystemExit(
        "[BLOQUEADO] Registry 1.2 não está PASS."
    )


print("[PASS] DQ Gate 04B")
print("[PASS] Label Contract V1")
print("[PASS] Registry 1.2")
print()


# =============================================================================
# 2. RAW FILES
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


raw_columns = []


for table, filename in FILES.items():

    path = RAW / filename

    if not path.exists():

        raise SystemExit(
            f"[ERRO] RAW ausente: {path}"
        )

    cols = pd.read_csv(
        path,
        nrows=0
    ).columns.tolist()

    for col in cols:

        raw_columns.append(
            {
                "table":
                    table,

                "column":
                    col,

                "column_key":
                    f"{table}.{col}"
            }
        )


raw_df = pd.DataFrame(
    raw_columns
)


print(
    f"Colunas RAW auditadas: {len(raw_df)}"
)


# =============================================================================
# 3. CLASSIFICAÇÃO POINT-IN-TIME
# =============================================================================

def classify(
    table,
    col
):

    key = (
        f"{table}.{col}"
    )


    # -------------------------------------------------------------------------
    # ORDERS
    # -------------------------------------------------------------------------

    if key == "orders.order_id":

        return (
            "IDENTIFIER_ONLY",
            False,
            "NONE",
            "Order identifier; never predictive feature."
        )


    if key == "orders.customer_id":

        return (
            "JOIN_KEY_ONLY",
            False,
            "NONE",
            "Order-specific customer join key."
        )


    if key == "orders.order_purchase_timestamp":

        return (
            "DERIVED_SAFE",
            False,
            "TEMPORAL",
            (
                "Defines t0. Raw timestamp itself is not required "
                "as a numeric model feature; calendar derivatives allowed."
            )
        )


    if key == "orders.order_estimated_delivery_date":

        return (
            "DERIVED_SAFE",
            False,
            "GEO_PROMISE",
            (
                "Promise exists for the purchase context. "
                "Use derived promised_lead_days rather than raw absolute date."
            )
        )


    if key == "orders.order_status":

        return (
            "FORBIDDEN_FUTURE",
            False,
            "FORBIDDEN",
            "Final order state contains post-t0 information."
        )


    if key == "orders.order_approved_at":

        return (
            "FORBIDDEN_FUTURE",
            False,
            "FORBIDDEN",
            "Occurs after purchase t0."
        )


    if key == "orders.order_delivered_carrier_date":

        return (
            "FORBIDDEN_FUTURE",
            False,
            "FORBIDDEN",
            "Future logistics event."
        )


    if key == "orders.order_delivered_customer_date":

        return (
            "TARGET_ONLY",
            False,
            "TARGET",
            "Outcome timestamp used to construct Y."
        )


    # -------------------------------------------------------------------------
    # ORDER ITEMS
    # -------------------------------------------------------------------------

    if table == "order_items":

        if col == "order_id":

            return (
                "JOIN_KEY_ONLY",
                False,
                "NONE",
                "Child-to-order join key."
            )


        if col == "order_item_id":

            return (
                "DERIVED_SAFE",
                False,
                "ORDER",
                "May be aggregated to item_count at order level."
            )


        if col == "product_id":

            return (
                "JOIN_KEY_ONLY",
                False,
                "ORDER",
                (
                    "Raw high-cardinality identifier not used directly; "
                    "may support counts and product joins."
                )
            )


        if col == "seller_id":

            return (
                "JOIN_KEY_ONLY",
                False,
                "SELLER_HISTORY",
                (
                    "Raw identifier not used directly; supports seller_count "
                    "and point-in-time seller history."
                )
            )


        if col == "shipping_limit_date":

            return (
                "HOLD_PROVENANCE",
                False,
                "GEO_PROMISE",
                (
                    "Availability at purchase has not yet been "
                    "independently demonstrated."
                )
            )


        if col in {
            "price",
            "freight_value"
        }:

            return (
                "AGGREGATE_SAFE_AT_T0",
                False,
                "ORDER",
                (
                    "Transactional order-item value; use only after "
                    "aggregation to order_id."
                )
            )


    # -------------------------------------------------------------------------
    # PAYMENTS
    # -------------------------------------------------------------------------

    if table == "payments":

        if col == "order_id":

            return (
                "JOIN_KEY_ONLY",
                False,
                "PAYMENT",
                "Payment-to-order join key."
            )

        return (
            "HOLD_PROVENANCE",
            False,
            "PAYMENT",
            (
                "Payment point-in-time availability at purchase "
                "remains unresolved in Registry."
            )
        )


    # -------------------------------------------------------------------------
    # REVIEWS
    # -------------------------------------------------------------------------

    if table == "reviews":

        return (
            "FORBIDDEN_FUTURE",
            False,
            "POST_DELIVERY_NLP",
            "Review information is generated after purchase/delivery."
        )


    # -------------------------------------------------------------------------
    # CUSTOMERS
    # -------------------------------------------------------------------------

    if table == "customers":

        if col == "customer_id":

            return (
                "JOIN_KEY_ONLY",
                False,
                "GEOGRAPHY",
                "Order-specific customer join key."
            )


        if col == "customer_unique_id":

            return (
                "IDENTIFIER_ONLY",
                False,
                "NONE",
                (
                    "Stable customer identity must not enter baseline "
                    "as a raw categorical ID."
                )
            )


        if col in {
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state"
        }:

            return (
                "STATIC_REFERENCE",
                False,
                "GEOGRAPHY",
                "Customer destination context associated with the order."
            )


    # -------------------------------------------------------------------------
    # PRODUCTS
    # -------------------------------------------------------------------------

    if table == "products":

        if col == "product_id":

            return (
                "JOIN_KEY_ONLY",
                False,
                "ORDER",
                "Product join key."
            )

        return (
            "STATIC_REFERENCE",
            False,
            "ORDER",
            (
                "Product catalog attribute. Can only enter after "
                "order-level aggregation; historical snapshots unavailable."
            )
        )


    # -------------------------------------------------------------------------
    # SELLERS
    # -------------------------------------------------------------------------

    if table == "sellers":

        if col == "seller_id":

            return (
                "JOIN_KEY_ONLY",
                False,
                "SELLER_HISTORY",
                "Seller join key."
            )

        return (
            "STATIC_REFERENCE",
            False,
            "GEOGRAPHY",
            (
                "Seller location reference. Static assumption is documented; "
                "no historical seller-location snapshots exist."
            )
        )


    # -------------------------------------------------------------------------
    # GEOLOCATION
    # -------------------------------------------------------------------------

    if table == "geolocation":

        return (
            "HOLD_DATA_QUALITY",
            False,
            "GEOGRAPHY",
            (
                "Static geographic reference, but consolidation policy "
                "for duplicated ZIP-prefix observations remains open."
            )
        )


    # -------------------------------------------------------------------------
    # TRANSLATION
    # -------------------------------------------------------------------------

    if table == "translation":

        if col == "product_category_name":

            return (
                "JOIN_KEY_ONLY",
                False,
                "ORDER",
                "Category translation join key."
            )

        return (
            "STATIC_REFERENCE",
            False,
            "ORDER",
            "Human-readable category reference."
        )


    return (
        "REVIEW_REQUIRED",
        False,
        "UNASSIGNED",
        "No explicit point-in-time rule yet."
    )


rows = []


for _, r in raw_df.iterrows():

    (
        status,
        direct,
        module,
        rationale
    ) = classify(
        r[
            "table"
        ],
        r[
            "column"
        ]
    )

    rows.append(
        {
            **r.to_dict(),

            "point_in_time_status":
                status,

            "direct_raw_feature_allowed":
                direct,

            "feature_module":
                module,

            "rationale":
                rationale
        }
    )


pit = pd.DataFrame(
    rows
)


# =============================================================================
# 4. MERGE COM REGISTRY TEMPORAL ANTERIOR, SE EXISTIR
# =============================================================================

prior_path = (
    PROJECT
    / "metadata"
    / "temporal_availability_registry.csv"
)


if prior_path.exists():

    prior = pd.read_csv(
        prior_path,
        low_memory=False
    )


    possible_cols = [
        c
        for c in [
            "table",
            "column",
            "available_at_t0",
            "allowed_as_baseline_feature",
            "leakage_risk"
        ]
        if c in prior.columns
    ]


    if (
        "table" in possible_cols
        and
        "column" in possible_cols
    ):

        prior = prior[
            possible_cols
        ].copy()


        rename = {
            c:
                f"registry_1_0_{c}"
            for c in possible_cols
            if c not in {
                "table",
                "column"
            }
        }


        prior = prior.rename(
            columns=rename
        )


        pit = pit.merge(
            prior,
            on=[
                "table",
                "column"
            ],
            how="left",
            validate="one_to_one"
        )


# =============================================================================
# 5. DERIVED FEATURE CONTRACT
# =============================================================================

derived = pd.DataFrame(
    [

        {
            "feature":
                "item_count",

            "module":
                "ORDER",

            "source":
                "order_items",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "COUNT(order_item_id) GROUP BY order_id",

            "point_in_time_rule":
                "Current order items only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "unique_product_count",

            "module":
                "ORDER",

            "source":
                "order_items",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "COUNT_DISTINCT(product_id)",

            "point_in_time_rule":
                "Current order items only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "unique_seller_count",

            "module":
                "ORDER",

            "source":
                "order_items",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "COUNT_DISTINCT(seller_id)",

            "point_in_time_rule":
                "Current order items only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "total_price",

            "module":
                "ORDER",

            "source":
                "order_items.price",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "SUM(price)",

            "point_in_time_rule":
                "Current order only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "mean_price",

            "module":
                "ORDER",

            "source":
                "order_items.price",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "MEAN(price)",

            "point_in_time_rule":
                "Current order only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "max_price",

            "module":
                "ORDER",

            "source":
                "order_items.price",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "MAX(price)",

            "point_in_time_rule":
                "Current order only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "total_freight",

            "module":
                "ORDER",

            "source":
                "order_items.freight_value",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "SUM(freight_value)",

            "point_in_time_rule":
                "Current order only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "freight_price_ratio",

            "module":
                "ORDER",

            "source":
                "price + freight_value",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "SUM(freight_value) / SUM(price)",

            "point_in_time_rule":
                "Current order only",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "purchase_weekday",

            "module":
                "TEMPORAL",

            "source":
                "orders.order_purchase_timestamp",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "weekday(t0)",

            "point_in_time_rule":
                "Derived from t0 itself",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "purchase_hour",

            "module":
                "TEMPORAL",

            "source":
                "orders.order_purchase_timestamp",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "hour(t0)",

            "point_in_time_rule":
                "Derived from t0 itself",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "purchase_month",

            "module":
                "TEMPORAL",

            "source":
                "orders.order_purchase_timestamp",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "month(t0)",

            "point_in_time_rule":
                "Derived from t0 itself",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "promised_lead_days",

            "module":
                "GEO_PROMISE",

            "source":
                "orders.order_estimated_delivery_date",

            "availability_class":
                "SAFE_AT_T0",

            "formula":
                "date(estimated_delivery) - date(purchase)",

            "point_in_time_rule":
                "Promise associated with current purchase",

            "status":
                "CANDIDATE_SAFE"
        },

        {
            "feature":
                "seller_late_rate_prior",

            "module":
                "SELLER_HISTORY",

            "source":
                "historical orders",

            "availability_class":
                "POINT_IN_TIME_REQUIRED",

            "formula":
                "prior_late_orders / prior_observed_orders",

            "point_in_time_rule":
                (
                    "Historical order j may contribute only when "
                    "delivery_j < purchase_i"
                ),

            "status":
                "REQUIRES_TEMPORAL_FEATURE_BUILDER"
        },

        {
            "feature":
                "seller_late_rate_30d",

            "module":
                "SELLER_HISTORY",

            "source":
                "historical orders",

            "availability_class":
                "POINT_IN_TIME_REQUIRED",

            "formula":
                "smoothed late rate over observable historical outcomes",

            "point_in_time_rule":
                (
                    "delivery_j < purchase_i AND "
                    "delivery_j >= purchase_i - 30 days"
                ),

            "status":
                "REQUIRES_TEMPORAL_FEATURE_BUILDER"
        },

        {
            "feature":
                "seller_volume_30d",

            "module":
                "SELLER_HISTORY",

            "source":
                "historical orders",

            "availability_class":
                "POINT_IN_TIME_REQUIRED",

            "formula":
                "COUNT(prior seller orders)",

            "point_in_time_rule":
                (
                    "purchase_j < purchase_i AND "
                    "purchase_j >= purchase_i - 30 days"
                ),

            "status":
                "REQUIRES_TEMPORAL_FEATURE_BUILDER"
        },

        {
            "feature":
                "seller_customer_distance_km",

            "module":
                "GEOGRAPHY",

            "source":
                "geolocation",

            "availability_class":
                "STATIC_REFERENCE",

            "formula":
                "Haversine(origin_centroid, destination_centroid)",

            "point_in_time_rule":
                "Static reference",

            "status":
                "BLOCKED_BY_GEO_CONSOLIDATION"
        },

        {
            "feature":
                "payment_type",

            "module":
                "PAYMENT",

            "source":
                "payments.payment_type",

            "availability_class":
                "HOLD_PROVENANCE",

            "formula":
                "aggregated payment representation",

            "point_in_time_rule":
                "Must prove availability at purchase",

            "status":
                "BLOCKED"
        },

        {
            "feature":
                "shipping_limit_days",

            "module":
                "GEO_PROMISE",

            "source":
                "order_items.shipping_limit_date",

            "availability_class":
                "HOLD_PROVENANCE",

            "formula":
                "shipping_limit_date - purchase_timestamp",

            "point_in_time_rule":
                "Must prove shipping_limit_date existed at t0",

            "status":
                "BLOCKED"
        }
    ]
)


# =============================================================================
# 6. MODEL MODULE PLAN
# =============================================================================

module_plan = pd.DataFrame(
    [

        {
            "sequence":
                1,

            "model_id":
                "MODEL_01_ORDER_LOGISTIC",

            "module":
                "ORDER",

            "algorithm":
                "LogisticRegression",

            "role":
                "first interpretable baseline",

            "status":
                "NEXT_AFTER_GATE_05"
        },

        {
            "sequence":
                2,

            "model_id":
                "MODEL_02_ORDER_CATBOOST",

            "module":
                "ORDER",

            "algorithm":
                "CatBoostClassifier",

            "role":
                "nonlinear challenger to Model 01",

            "status":
                "LOCKED_UNTIL_MODEL_01_VALIDATED"
        },

        {
            "sequence":
                3,

            "model_id":
                "MODEL_03_SELLER_EXPERT",

            "module":
                "SELLER_HISTORY",

            "algorithm":
                "TO_BE_SELECTED",

            "role":
                "historical seller reliability",

            "status":
                "LOCKED_UNTIL_PREVIOUS_REVIEW"
        },

        {
            "sequence":
                4,

            "model_id":
                "MODEL_04_GEO_PROMISE_EXPERT",

            "module":
                "GEO_PROMISE",

            "algorithm":
                "TO_BE_SELECTED",

            "role":
                "geography and promise adequacy",

            "status":
                "LOCKED_UNTIL_PREVIOUS_REVIEW"
        },

        {
            "sequence":
                5,

            "model_id":
                "MODEL_05_TEMPORAL_EXPERT",

            "module":
                "TEMPORAL",

            "algorithm":
                "TO_BE_SELECTED",

            "role":
                "temporal/operational pressure",

            "status":
                "LOCKED_UNTIL_PREVIOUS_REVIEW"
        },

        {
            "sequence":
                6,

            "model_id":
                "MODEL_06_FUSION",

            "module":
                "FUSION",

            "algorithm":
                "LogisticRegression initially",

            "role":
                "OOF specialist probability fusion",

            "status":
                "LOCKED_UNTIL_SPECIALISTS_VALIDATED"
        }
    ]
)


# =============================================================================
# 7. CRITICAL LEAKAGE AUDIT
# =============================================================================

forbidden = pit[
    pit[
        "point_in_time_status"
    ]
    .isin(
        [
            "FORBIDDEN_FUTURE",
            "TARGET_ONLY"
        ]
    )
].copy()


holds = pit[
    pit[
        "point_in_time_status"
    ]
    .str.startswith(
        "HOLD",
        na=False
    )
].copy()


review = pit[
    pit[
        "point_in_time_status"
    ]
    ==
    "REVIEW_REQUIRED"
].copy()


critical_violations = int(
    (
        (
            pit[
                "point_in_time_status"
            ]
            .isin(
                [
                    "FORBIDDEN_FUTURE",
                    "TARGET_ONLY"
                ]
            )
        )
        &
        (
            pit[
                "direct_raw_feature_allowed"
            ]
            ==
            True
        )
    )
    .sum()
)


# =============================================================================
# 8. SAVE
# =============================================================================

pit.to_csv(
    REG13
    / "feature_point_in_time_registry.csv",
    index=False
)


pit.to_csv(
    OUT
    / "01_raw_feature_point_in_time_registry.csv",
    index=False
)


derived.to_csv(
    OUT
    / "02_derived_feature_contract.csv",
    index=False
)


module_plan.to_csv(
    OUT
    / "03_model_module_plan.csv",
    index=False
)


forbidden.to_csv(
    OUT
    / "04_forbidden_target_features.csv",
    index=False
)


holds.to_csv(
    OUT
    / "05_hold_features.csv",
    index=False
)


review.to_csv(
    OUT
    / "06_review_required_features.csv",
    index=False
)


# =============================================================================
# 9. STATUS
# =============================================================================

gate_status = (
    "PASS"
    if critical_violations == 0
    else
    "FAIL"
)


status_counts = (
    pit[
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


status_counts.to_csv(
    OUT
    / "07_point_in_time_status_summary.csv",
    index=False
)


summary = {

    "gate":
        "DQ_GATE_05_POINT_IN_TIME_AVAILABILITY",

    "status":
        gate_status,

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "prediction_time":
        "order_purchase_timestamp",

    "raw_columns_audited":
        int(
            len(
                pit
            )
        ),

    "forbidden_or_target_columns":
        int(
            len(
                forbidden
            )
        ),

    "hold_columns":
        int(
            len(
                holds
            )
        ),

    "review_required_columns":
        int(
            len(
                review
            )
        ),

    "critical_leakage_violations":
        critical_violations,

    "next_model":
        "MODEL_01_ORDER_LOGISTIC"
        if gate_status == "PASS"
        else None,

    "model_02_locked":
        True,

    "raw_modified":
        False,

    "silver_created":
        False,

    "model_trained":
        False
}


save_json(
    OUT
    / "dq_gate_05_summary.json",
    summary
)


# =============================================================================
# 10. REGISTRY 1.3 MANIFEST
# =============================================================================

for src in REG12.iterdir():

    if (
        src.is_file()
        and
        src.name
        !=
        "registry_v1_2_manifest.json"
    ):

        dst = (
            REG13
            / src.name
        )

        if not dst.exists():

            shutil.copy2(
                src,
                dst
            )


manifest13 = dict(
    reg12_manifest
)


manifest13.update(
    {

        "registry_version":
            "1.3.0",

        "parent_registry_version":
            "1.2.0",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "internal_registry_status":
            gate_status,

        "point_in_time_contract":
            (
                "metadata/registry_v1_3/"
                "feature_point_in_time_registry.csv"
            ),

        "prediction_time":
            "order_purchase_timestamp",

        "updates": [
            "Point-in-time availability contract added",
            "Feature modules assigned",
            "Future and target fields explicitly forbidden",
            "Payment and shipping_limit_date remain HOLD",
            "Seller history explicitly requires historical outcome availability",
            "Sequential model validation protocol established"
        ],

        "raw_modified":
            False
    }
)


save_json(
    REG13
    / "registry_v1_3_manifest.json",
    manifest13
)


# =============================================================================
# 11. REPORT
# =============================================================================

report = []

report.append(
    "=" * 108
)

report.append(
    "DQ GATE 05 — POINT-IN-TIME AVAILABILITY & LEAKAGE"
)

report.append(
    "=" * 108
)

report.append("")

report.append(
    f"STATUS                         : {gate_status}"
)

report.append(
    f"PREDICTION TIME                : order_purchase_timestamp"
)

report.append(
    f"RAW COLUMNS AUDITED             : {len(pit)}"
)

report.append(
    f"FORBIDDEN / TARGET COLUMNS      : {len(forbidden)}"
)

report.append(
    f"HOLD COLUMNS                    : {len(holds)}"
)

report.append(
    f"REVIEW REQUIRED                 : {len(review)}"
)

report.append(
    f"CRITICAL LEAKAGE VIOLATIONS     : {critical_violations}"
)

report.append("")

report.append(
    "MATHEMATICAL RULE"
)

report.append(
    "-" * 108
)

report.append(
    "A feature X_ij is valid only when:"
)

report.append("")

report.append(
    "t_available(X_ij) <= t0_i"
)

report.append("")

report.append(
    "with t0_i = order_purchase_timestamp_i"
)

report.append("")

report.append(
    "SELLER HISTORY RULE"
)

report.append(
    "-" * 108
)

report.append(
    "A previous seller outcome j may contribute to order i only if:"
)

report.append("")

report.append(
    "order_delivered_customer_date_j < order_purchase_timestamp_i"
)

report.append("")

report.append(
    "NEXT MODEL"
)

report.append(
    "-" * 108
)

if gate_status == "PASS":

    report.append(
        "MODEL_01_ORDER_LOGISTIC"
    )

    report.append(
        "Model 02 remains LOCKED until Model 01 is evaluated."
    )

else:

    report.append(
        "BLOCKED"
    )


report.append("")

report.append(
    "RAW MODIFIED                    : NO"
)

report.append(
    "SILVER CREATED                  : NO"
)

report.append(
    "MODEL TRAINED                   : NO"
)

report.append(
    "=" * 108
)


report_path = (
    OUT
    / "DQ_GATE_05_POINT_IN_TIME_REPORT.txt"
)


report_path.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# =============================================================================
# TERMINAL
# =============================================================================

print()
print("=" * 108)
print("1. POINT-IN-TIME STATUS")
print("=" * 108)

print(
    status_counts.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("2. FORBIDDEN / TARGET")
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
print("3. HOLD")
print("=" * 108)

if holds.empty:

    print(
        "[OK] Nenhuma feature HOLD."
    )

else:

    print(
        holds[
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
print("4. DERIVED FEATURE CONTRACT")
print("=" * 108)

print(
    derived.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("5. MODEL SEQUENCE")
print("=" * 108)

print(
    module_plan.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("DQ GATE 05 — RESULTADO")
print("=" * 108)

print(
    f"STATUS                     : {gate_status}"
)

print(
    f"CRITICAL LEAKAGE VIOLATIONS: {critical_violations}"
)

print(
    f"HOLD                       : {len(holds)}"
)

print(
    f"REVIEW REQUIRED            : {len(review)}"
)

print(
    "RAW MODIFIED               : NÃO"
)

print(
    "MODEL TRAINED              : NÃO"
)


if gate_status == "PASS":

    print()
    print(
        "[PASS] POINT-IN-TIME CONTRACT CONSTRUÍDO."
    )

    print(
        "NEXT: MODEL_01_ORDER_LOGISTIC."
    )

    print(
        "MODEL_02 permanece bloqueado até validação do MODEL_01."
    )

    sys.exit(0)

else:

    print()
    print(
        "[FAIL] LEAKAGE CONTRACT REPROVADO."
    )

    sys.exit(2)


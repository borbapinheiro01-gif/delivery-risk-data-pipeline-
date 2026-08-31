#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DATASET KNOWLEDGE & TRUTH REGISTRY — VERSION 1.1
Delivery Risk Intelligence Platform
===============================================================================

OBJETIVO
--------
Evoluir o Registry 1.0 para uma representação mais rigorosa e reutilizável.

ADICIONA
--------
1. Canonicalização das questões abertas
2. Separação:
       semantic cardinality
       observed cardinality
3. Política explícita de grão:
       source grain
       target ML grain
4. Política explícita de agregação
5. Política de feature direta
6. Grain Compatibility Registry
7. Issue Evidence Map
8. Search Knowledge 1.1
9. Croissant 1.1 JSON-LD
10. Auditoria interna do próprio Registry

IMPORTANTE
----------
- NÃO altera data/raw.
- NÃO resolve automaticamente questões abertas.
- NÃO altera o target.
- NÃO cria Silver.
- NÃO cria features.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = PROJECT / "data" / "raw" / "olist"

META = PROJECT / "metadata"

OUT = META / "registry_v1_1"

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# INPUTS
# =============================================================================

COLUMN_CATALOG = META / "column_catalog.csv"

TABLE_CATALOG = META / "table_catalog.csv"

RELATIONSHIP_REGISTRY = META / "relationship_registry.csv"

TRUTH_REGISTRY = META / "truth_provenance_registry.csv"

TEMPORAL_REGISTRY = META / "temporal_availability_registry.csv"

QUALITY_REGISTRY = META / "quality_observation_registry.csv"

ISSUES = META / "unresolved_questions.csv"

SEARCH_V1 = META / "search_knowledge.jsonl"


required = [
    COLUMN_CATALOG,
    TABLE_CATALOG,
    RELATIONSHIP_REGISTRY,
    TRUTH_REGISTRY,
    TEMPORAL_REGISTRY,
    QUALITY_REGISTRY,
    ISSUES,
]


missing = [
    str(p)
    for p in required
    if not p.exists()
]


if missing:

    raise SystemExit(
        "Arquivos do Registry 1.0 ausentes:\n"
        +
        "\n".join(missing)
    )


# =============================================================================
# RAW FILE MAP
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
# LOAD REGISTRY 1.0
# =============================================================================

print("=" * 100)

print(
    "DATASET KNOWLEDGE & TRUTH REGISTRY — UPGRADE 1.1"
)

print("=" * 100)

print()

print(
    f"Projeto : {PROJECT}"
)

print(
    f"RAW     : {RAW}"
)

print(
    f"Registry: {META}"
)

print(
    f"Saída   : {OUT}"
)


columns = pd.read_csv(
    COLUMN_CATALOG,
    low_memory=False
)


tables = pd.read_csv(
    TABLE_CATALOG,
    low_memory=False
)


relationships = pd.read_csv(
    RELATIONSHIP_REGISTRY,
    low_memory=False
)


truth = pd.read_csv(
    TRUTH_REGISTRY,
    low_memory=False
)


temporal = pd.read_csv(
    TEMPORAL_REGISTRY,
    low_memory=False
)


quality = pd.read_csv(
    QUALITY_REGISTRY,
    low_memory=False
)


issues = pd.read_csv(
    ISSUES,
    low_memory=False
)


print()

print(
    f"[OK] Tables        : {len(tables)}"
)

print(
    f"[OK] Columns       : {len(columns)}"
)

print(
    f"[OK] Relationships : {len(relationships)}"
)

print(
    f"[OK] Issues raw    : {len(issues)}"
)


# =============================================================================
# HELPERS
# =============================================================================

def clean(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def pct(a, b):

    if b == 0:
        return np.nan

    return 100.0 * a / b


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


# =============================================================================
# 1. CANONICAL ISSUE REGISTRY
# =============================================================================

print()

print(
    "[1/9] Canonicalizando questões abertas..."
)


# -----------------------------------------------------------------------------
# mapa conceitual
# -----------------------------------------------------------------------------

CANONICAL_TOPIC_MAP = {

    "target_semantics":
        "ISSUE-TARGET-001",

    "target_definition":
        "ISSUE-TARGET-001",


    "shipping_limit_date_provenance":
        "ISSUE-SHIPPING-LIMIT-001",

    "shipping_limit_date":
        "ISSUE-SHIPPING-LIMIT-001",


    "conditional_missingness":
        "ISSUE-MISSINGNESS-001",


    "external_ground_truth":
        "ISSUE-GROUND-TRUTH-001",


    "seller_history":
        "ISSUE-SELLER-HISTORY-001",


    "multi_seller":
        "ISSUE-MULTI-SELLER-001",

    "multi_seller_orders":
        "ISSUE-MULTI-SELLER-001",


    "geolocation":
        "ISSUE-GEO-001",

    "geolocation_spread":
        "ISSUE-GEO-001",


    "financial_reconciliation":
        "ISSUE-FINANCIAL-001",

    "payment_inconsistencies":
        "ISSUE-FINANCIAL-001",


    "observation_window":
        "ISSUE-OBSERVATION-WINDOW-001",


    "payment_availability":
        "ISSUE-PAYMENT-AVAILABILITY-001",


    "outliers":
        "ISSUE-OUTLIERS-001",


    "numeric_transformations":
        "ISSUE-NUMERIC-TRANSFORMATION-001",


    "numeric_imputation":
        "ISSUE-IMPUTATION-001",
}


issues_v11 = issues.copy()


issues_v11[
    "canonical_issue_id"
] = (
    issues_v11[
        "topic"
    ]
    .map(
        CANONICAL_TOPIC_MAP
    )
)


# qualquer tópico ainda não mapeado recebe ID estável
unknown_topics = sorted(
    issues_v11.loc[
        issues_v11[
            "canonical_issue_id"
        ].isna(),
        "topic"
    ]
    .dropna()
    .unique()
)


for i, topic in enumerate(
    unknown_topics,
    start=1
):

    mask = (
        issues_v11[
            "topic"
        ]
        ==
        topic
    )

    safe_topic = (
        str(topic)
        .upper()
        .replace(
            " ",
            "-"
        )
        .replace(
            "_",
            "-"
        )
    )

    issues_v11.loc[
        mask,
        "canonical_issue_id"
    ] = (
        f"ISSUE-{safe_topic}-{i:03d}"
    )


# -----------------------------------------------------------------------------
# canonical topic
# -----------------------------------------------------------------------------

CANONICAL_NAMES = {

    "ISSUE-TARGET-001":
        "label_and_target_semantics",

    "ISSUE-SHIPPING-LIMIT-001":
        "shipping_limit_date_provenance",

    "ISSUE-MISSINGNESS-001":
        "conditional_and_task_missingness",

    "ISSUE-GROUND-TRUTH-001":
        "external_ground_truth",

    "ISSUE-SELLER-HISTORY-001":
        "seller_point_in_time_history",

    "ISSUE-MULTI-SELLER-001":
        "multi_seller_representation",

    "ISSUE-GEO-001":
        "geolocation_consolidation",

    "ISSUE-FINANCIAL-001":
        "financial_reconciliation",

    "ISSUE-OBSERVATION-WINDOW-001":
        "label_observation_window",

    "ISSUE-PAYMENT-AVAILABILITY-001":
        "payment_point_in_time_availability",

    "ISSUE-OUTLIERS-001":
        "extreme_value_policy",

    "ISSUE-NUMERIC-TRANSFORMATION-001":
        "numeric_transformation_policy",

    "ISSUE-IMPUTATION-001":
        "numeric_imputation_policy",
}


issues_v11[
    "canonical_topic"
] = (
    issues_v11[
        "canonical_issue_id"
    ]
    .map(
        CANONICAL_NAMES
    )
    .fillna(
        issues_v11[
            "topic"
        ]
    )
)


# -----------------------------------------------------------------------------
# duplicate_of
# -----------------------------------------------------------------------------

issues_v11[
    "duplicate_of"
] = ""


for canonical_id, grp in issues_v11.groupby(
    "canonical_issue_id",
    dropna=False
):

    if len(grp) <= 1:
        continue

    # Prefer OPEN-* as primary record
    preferred = grp[
        grp[
            "question_id"
        ]
        .astype(str)
        .str.startswith(
            "OPEN-"
        )
    ]

    if len(preferred):

        primary = preferred.iloc[
            0
        ][
            "question_id"
        ]

    else:

        primary = grp.iloc[
            0
        ][
            "question_id"
        ]


    for idx, row in grp.iterrows():

        if row[
            "question_id"
        ] != primary:

            issues_v11.loc[
                idx,
                "duplicate_of"
            ] = primary


issues_v11[
    "source_issue_id"
] = issues_v11[
    "question_id"
]


issues_v11.to_csv(
    OUT
    / "issue_evidence_map.csv",
    index=False
)


# =============================================================================
# CANONICAL ISSUE SUMMARY
# =============================================================================

canonical_rows = []


for canonical_id, grp in issues_v11.groupby(
    "canonical_issue_id"
):

    priorities = grp[
        "priority"
    ].astype(str)


    ranking = {
        "CRITICAL":
            4,

        "HIGH":
            3,

        "MEDIUM":
            2,

        "LOW":
            1
    }


    best_priority = max(
        priorities,
        key=lambda x:
            ranking.get(
                x,
                0
            )
    )


    statuses = sorted(
        set(
            grp[
                "status"
            ].astype(str)
        )
    )


    canonical_rows.append(
        {
            "canonical_issue_id":
                canonical_id,

            "canonical_topic":
                grp[
                    "canonical_topic"
                ].iloc[0],

            "priority":
                best_priority,

            "source_issue_count":
                len(grp),

            "source_issue_ids":
                ";".join(
                    grp[
                        "question_id"
                    ]
                    .astype(str)
                ),

            "statuses":
                ";".join(
                    statuses
                ),

            "question_summary":
                " | ".join(
                    dict.fromkeys(
                        grp[
                            "question"
                        ]
                        .dropna()
                        .astype(str)
                    )
                ),

            "evidence_summary":
                " | ".join(
                    dict.fromkeys(
                        grp[
                            "current_evidence"
                        ]
                        .dropna()
                        .astype(str)
                    )
                ),

            "required_actions":
                " | ".join(
                    dict.fromkeys(
                        grp[
                            "required_action"
                        ]
                        .dropna()
                        .astype(str)
                    )
                ),

            "blocks":
                ";".join(
                    sorted(
                        set(
                            grp[
                                "blocks"
                            ]
                            .dropna()
                            .astype(str)
                        )
                    )
                )
        }
    )


canonical_issues = pd.DataFrame(
    canonical_rows
)


canonical_issues.to_csv(
    OUT
    / "canonical_issue_registry.csv",
    index=False
)


print(
    f"  Raw issues       : {len(issues_v11)}"
)

print(
    f"  Canonical issues : {len(canonical_issues)}"
)


# =============================================================================
# 2. ML GRAIN POLICY
# =============================================================================

print()

print(
    "[2/9] Construindo ML Grain Policy..."
)


TARGET_ML_GRAIN = (
    "1 row per order_id"
)


SOURCE_GRAIN_MAP = dict(
    zip(
        tables[
            "table"
        ],
        tables[
            "grain"
        ]
    )
)


def feature_policy(
    table,
    column,
    semantic_type
):

    # -------------------------------------------------------------------------
    # DEFAULT
    # -------------------------------------------------------------------------

    out = {
        "target_ml_grain":
            TARGET_ML_GRAIN,

        "aggregation_required":
            "NO",

        "aggregation_policy":
            "DIRECT_OR_DERIVED",

        "direct_feature_allowed":
            "REVIEW",

        "feature_form":
            "REVIEW",

        "grain_compatibility":
            "REVIEW"
    }


    # -------------------------------------------------------------------------
    # ORDERS = target grain
    # -------------------------------------------------------------------------

    if table == "orders":

        out[
            "grain_compatibility"
        ] = "DIRECT_GRAIN_MATCH"


        if column in {
            "order_id",
            "customer_id"
        }:

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "KEY_ONLY"

            out[
                "aggregation_policy"
            ] = "NONE"


        elif column == "order_purchase_timestamp":

            out[
                "direct_feature_allowed"
            ] = "NO_RAW_TIMESTAMP"

            out[
                "feature_form"
            ] = "DERIVED_CALENDAR"

            out[
                "aggregation_policy"
            ] = (
                "derive weekday/hour/month/"
                "seasonality"
            )


        elif column == "order_estimated_delivery_date":

            out[
                "direct_feature_allowed"
            ] = "DERIVE_ONLY"

            out[
                "feature_form"
            ] = "PROMISED_LEAD_TIME"

            out[
                "aggregation_policy"
            ] = (
                "estimated_delivery_date - "
                "purchase_timestamp"
            )


        elif column in {
            "order_status",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date"
        }:

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "FORBIDDEN_T0"

            out[
                "aggregation_policy"
            ] = "NONE"


    # -------------------------------------------------------------------------
    # ORDER ITEMS = child grain
    # -------------------------------------------------------------------------

    elif table == "order_items":

        out[
            "aggregation_required"
        ] = "YES"

        out[
            "grain_compatibility"
        ] = "CHILD_GRAIN_REQUIRES_AGGREGATION"


        if column == "order_id":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "JOIN_KEY"

            out[
                "aggregation_policy"
            ] = "GROUP_BY_ORDER_ID"


        elif column == "order_item_id":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "ORDER_ITEM_COUNT"

            out[
                "aggregation_policy"
            ] = "COUNT"


        elif column == "product_id":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "PRODUCT_AGGREGATES"

            out[
                "aggregation_policy"
            ] = (
                "COUNT_DISTINCT + JOIN_PRODUCT_ATTRIBUTES"
            )


        elif column == "seller_id":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = (
                "SELLER_COUNT_AND_HISTORICAL_FEATURES"
            )

            out[
                "aggregation_policy"
            ] = (
                "COUNT_DISTINCT + POINT_IN_TIME_HISTORY"
            )


        elif column == "price":

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "ORDER_PRICE_AGGREGATES"

            out[
                "aggregation_policy"
            ] = (
                "SUM, MEAN, MAX, MIN"
            )


        elif column == "freight_value":

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "ORDER_FREIGHT_AGGREGATES"

            out[
                "aggregation_policy"
            ] = (
                "SUM, MEAN, MAX, freight/item_ratio"
            )


        elif column == "shipping_limit_date":

            out[
                "direct_feature_allowed"
            ] = "HOLD"

            out[
                "feature_form"
            ] = "NONE_UNTIL_PROVENANCE_CONFIRMED"

            out[
                "aggregation_policy"
            ] = (
                "HOLD — availability at t0 not proven"
            )


    # -------------------------------------------------------------------------
    # PAYMENTS = child grain
    # -------------------------------------------------------------------------

    elif table == "payments":

        out[
            "aggregation_required"
        ] = "YES"

        out[
            "grain_compatibility"
        ] = "CHILD_GRAIN_REQUIRES_AGGREGATION"


        if column == "order_id":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "JOIN_KEY"

            out[
                "aggregation_policy"
            ] = "GROUP_BY_ORDER_ID"


        elif column == "payment_sequential":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "PAYMENT_RECORD_COUNT"

            out[
                "aggregation_policy"
            ] = "COUNT"


        elif column == "payment_type":

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "MULTI_PAYMENT_TYPE"

            out[
                "aggregation_policy"
            ] = (
                "SET/FLAGS + COUNT_DISTINCT + PRIMARY_TYPE"
            )


        elif column == "payment_installments":

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "INSTALLMENT_AGGREGATES"

            out[
                "aggregation_policy"
            ] = "MAX, MEAN"


        elif column == "payment_value":

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "TOTAL_PAYMENT"

            out[
                "aggregation_policy"
            ] = "SUM"


    # -------------------------------------------------------------------------
    # PRODUCTS
    # -------------------------------------------------------------------------

    elif table == "products":

        out[
            "aggregation_required"
        ] = "YES_VIA_ORDER_ITEMS"

        out[
            "grain_compatibility"
        ] = (
            "REFERENCE_ENTITY_REQUIRES_ITEM_TO_ORDER_AGGREGATION"
        )


        if column == "product_id":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "JOIN_KEY"

            out[
                "aggregation_policy"
            ] = "NONE_DIRECT"


        elif column == "product_category_name":

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "CATEGORY_SET_OR_DOMINANT_CATEGORY"

            out[
                "aggregation_policy"
            ] = (
                "MODE/SET/N_DISTINCT"
            )


        else:

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "PRODUCT_ATTRIBUTE_AGGREGATES"

            out[
                "aggregation_policy"
            ] = (
                "ORDER_LEVEL MIN/MEAN/MAX/SUM "
                "as semantically appropriate"
            )


    # -------------------------------------------------------------------------
    # SELLERS
    # -------------------------------------------------------------------------

    elif table == "sellers":

        out[
            "aggregation_required"
        ] = "YES_VIA_ORDER_ITEMS"

        out[
            "grain_compatibility"
        ] = (
            "REFERENCE_ENTITY_REQUIRES_ITEM_TO_ORDER_AGGREGATION"
        )


        if column == "seller_id":

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "JOIN_KEY"

            out[
                "aggregation_policy"
            ] = (
                "COUNT_DISTINCT + POINT_IN_TIME_HISTORY"
            )

        else:

            out[
                "direct_feature_allowed"
            ] = "AGGREGATE_ONLY"

            out[
                "feature_form"
            ] = "SELLER_ATTRIBUTE_AGGREGATE"

            out[
                "aggregation_policy"
            ] = (
                "SET / COUNT_DISTINCT / MULTI_SELLER_POLICY"
            )


    # -------------------------------------------------------------------------
    # CUSTOMERS
    # -------------------------------------------------------------------------

    elif table == "customers":

        out[
            "aggregation_required"
        ] = "NO"

        out[
            "grain_compatibility"
        ] = (
            "ORDER_SPECIFIC_REFERENCE_JOIN"
        )


        if column in {
            "customer_id",
            "customer_unique_id"
        }:

            out[
                "direct_feature_allowed"
            ] = "NO"

            out[
                "feature_form"
            ] = "KEY_OR_HISTORY_ONLY"

            out[
                "aggregation_policy"
            ] = "NONE_DIRECT"

        else:

            out[
                "direct_feature_allowed"
            ] = "YES_CANDIDATE"

            out[
                "feature_form"
            ] = "CUSTOMER_LOCATION"

            out[
                "aggregation_policy"
            ] = "DIRECT_REFERENCE_JOIN"


    # -------------------------------------------------------------------------
    # GEOLOCATION
    # -------------------------------------------------------------------------

    elif table == "geolocation":

        out[
            "aggregation_required"
        ] = "YES"

        out[
            "grain_compatibility"
        ] = (
            "REFERENCE_OBSERVATION_REQUIRES_CONSOLIDATION"
        )

        out[
            "direct_feature_allowed"
        ] = "NO_RAW"

        out[
            "feature_form"
        ] = "ZIP_LEVEL_REFERENCE"

        out[
            "aggregation_policy"
        ] = (
            "CONSOLIDATE_BY_ZIP_PREFIX "
            "before customer/seller join"
        )


    # -------------------------------------------------------------------------
    # REVIEWS
    # -------------------------------------------------------------------------

    elif table == "reviews":

        out[
            "aggregation_required"
        ] = "NOT_APPLICABLE_FOR_BASELINE"

        out[
            "grain_compatibility"
        ] = "POST_OUTCOME_DATA"

        out[
            "direct_feature_allowed"
        ] = "NO"

        out[
            "feature_form"
        ] = "POST_DELIVERY_NLP_ONLY"

        out[
            "aggregation_policy"
        ] = (
            "KEEP_OUTSIDE_DELIVERY_RISK_BASELINE"
        )


    # -------------------------------------------------------------------------
    # TRANSLATION
    # -------------------------------------------------------------------------

    elif table == "translation":

        out[
            "aggregation_required"
        ] = "NO_MODEL_FEATURE"

        out[
            "grain_compatibility"
        ] = "PRESENTATION_REFERENCE"

        out[
            "direct_feature_allowed"
        ] = "NO"

        out[
            "feature_form"
        ] = "DISPLAY_ONLY"

        out[
            "aggregation_policy"
        ] = "LEFT_JOIN_FOR_PRESENTATION"


    return out


grain_rows = []


for _, row in columns.iterrows():

    table = row[
        "table"
    ]

    column = row[
        "column"
    ]

    semantic = clean(
        row.get(
            "semantic_type",
            ""
        )
    )


    policy = feature_policy(
        table,
        column,
        semantic
    )


    grain_rows.append(
        {
            "table":
                table,

            "column":
                column,

            "source_grain":
                SOURCE_GRAIN_MAP.get(
                    table,
                    ""
                ),

            **policy
        }
    )


grain_policy = pd.DataFrame(
    grain_rows
)


grain_policy.to_csv(
    OUT
    / "ml_grain_policy_registry.csv",
    index=False
)


# =============================================================================
# 3. AUGMENT COLUMN CATALOG
# =============================================================================

print()

print(
    "[3/9] Atualizando Column Catalog..."
)


columns_v11 = columns.merge(
    grain_policy,
    on=[
        "table",
        "column"
    ],
    how="left",
    validate="one_to_one"
)


columns_v11[
    "registry_version"
] = "1.1"


columns_v11.to_csv(
    OUT
    / "column_catalog_v1_1.csv",
    index=False
)


# =============================================================================
# 4. OBSERVED RELATIONSHIP CARDINALITY
# =============================================================================

print()

print(
    "[4/9] Recalculando cardinalidades observadas..."
)


observed_relationship_rows = []


for _, rel in relationships.iterrows():

    parent_table = rel[
        "parent_table"
    ]

    child_table = rel[
        "child_table"
    ]

    parent_column = rel[
        "parent_column"
    ]

    child_column = rel[
        "child_column"
    ]


    p = pd.read_csv(
        RAW
        / FILES[
            parent_table
        ],
        usecols=[
            parent_column
        ],
        low_memory=False
    )


    c = pd.read_csv(
        RAW
        / FILES[
            child_table
        ],
        usecols=[
            child_column
        ],
        low_memory=False
    )


    parent_series = (
        p[
            parent_column
        ]
        .dropna()
        .astype(str)
    )


    child_series = (
        c[
            child_column
        ]
        .dropna()
        .astype(str)
    )


    parent_unique = (
        parent_series.nunique()
        ==
        len(
            parent_series
        )
    )


    child_counts = (
        child_series
        .value_counts()
    )


    if len(
        child_counts
    ):

        max_child = int(
            child_counts.max()
        )

        median_child = float(
            child_counts.median()
        )

        mean_child = float(
            child_counts.mean()
        )

    else:

        max_child = 0
        median_child = np.nan
        mean_child = np.nan


    if parent_unique:

        if max_child <= 1:

            observed_cardinality = (
                "1:0..1 OBSERVED"
            )

        else:

            observed_cardinality = (
                "1:N OBSERVED"
            )

    else:

        observed_cardinality = (
            "M:N_OR_NONUNIQUE_PARENT OBSERVED"
        )


    parent_values = set(
        parent_series
    )


    orphan_count = int(
        (
            ~child_series.isin(
                parent_values
            )
        ).sum()
    )


    semantic_cardinality = clean(
        rel.get(
            "expected_relationship",
            ""
        )
    )


    observed_relationship_rows.append(
        {
            **rel.to_dict(),

            "semantic_cardinality":
                semantic_cardinality,

            "observed_cardinality":
                observed_cardinality,

            "parent_key_unique_observed":
                bool(
                    parent_unique
                ),

            "child_rows_per_parent_mean":
                mean_child,

            "child_rows_per_parent_median":
                median_child,

            "child_rows_per_parent_max":
                max_child,

            "child_orphan_rows_recomputed":
                orphan_count,

            "cardinality_note":
                (
                    "Observed cardinality describes this sample; "
                    "semantic cardinality expresses the intended "
                    "data-model meaning."
                )
        }
    )


relationships_v11 = pd.DataFrame(
    observed_relationship_rows
)


relationships_v11[
    "registry_version"
] = "1.1"


relationships_v11.to_csv(
    OUT
    / "relationship_registry_v1_1.csv",
    index=False
)


# =============================================================================
# 5. GRAIN COMPATIBILITY SUMMARY
# =============================================================================

print()

print(
    "[5/9] Construindo Grain Compatibility Summary..."
)


grain_summary = (
    grain_policy
    .groupby(
        [
            "source_grain",
            "grain_compatibility",
            "aggregation_required",
            "aggregation_policy"
        ],
        dropna=False
    )
    .agg(
        columns=(
            "column",
            "count"
        )
    )
    .reset_index()
)


grain_summary.to_csv(
    OUT
    / "grain_compatibility_summary.csv",
    index=False
)


# =============================================================================
# 6. SEARCH KNOWLEDGE 1.1
# =============================================================================

print()

print(
    "[6/9] Construindo Search Knowledge 1.1..."
)


search_records = []


# -----------------------------------------------------------------------------
# original knowledge
# -----------------------------------------------------------------------------

if SEARCH_V1.exists():

    with SEARCH_V1.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                search_records.append(
                    json.loads(
                        line
                    )
                )

            except Exception:
                pass


# -----------------------------------------------------------------------------
# grain policies
# -----------------------------------------------------------------------------

for _, row in grain_policy.iterrows():

    search_records.append(
        {
            "id":
                (
                    "grain:"
                    f"{row['table']}."
                    f"{row['column']}"
                ),

            "type":
                "ml_grain_policy",

            "title":
                (
                    f"{row['table']}."
                    f"{row['column']}"
                ),

            "search_text":
                " ".join(
                    [
                        str(
                            row[
                                "table"
                            ]
                        ),

                        str(
                            row[
                                "column"
                            ]
                        ),

                        str(
                            row[
                                "source_grain"
                            ]
                        ),

                        str(
                            row[
                                "target_ml_grain"
                            ]
                        ),

                        str(
                            row[
                                "aggregation_policy"
                            ]
                        ),

                        str(
                            row[
                                "feature_form"
                            ]
                        ),

                        str(
                            row[
                                "grain_compatibility"
                            ]
                        )
                    ]
                ),

            "payload":
                row.to_dict()
        }
    )


# -----------------------------------------------------------------------------
# canonical issues
# -----------------------------------------------------------------------------

for _, row in canonical_issues.iterrows():

    search_records.append(
        {
            "id":
                (
                    "canonical_issue:"
                    f"{row['canonical_issue_id']}"
                ),

            "type":
                "canonical_issue",

            "title":
                row[
                    "canonical_topic"
                ],

            "search_text":
                " ".join(
                    [
                        str(
                            row[
                                "canonical_topic"
                            ]
                        ),

                        str(
                            row[
                                "question_summary"
                            ]
                        ),

                        str(
                            row[
                                "evidence_summary"
                            ]
                        ),

                        str(
                            row[
                                "required_actions"
                            ]
                        )
                    ]
                ),

            "payload":
                row.to_dict()
        }
    )


# -----------------------------------------------------------------------------
# observed relationships
# -----------------------------------------------------------------------------

for _, row in relationships_v11.iterrows():

    search_records.append(
        {
            "id":
                (
                    "cardinality:"
                    f"{row['parent_table']}."
                    f"{row['parent_column']}"
                    "->"
                    f"{row['child_table']}."
                    f"{row['child_column']}"
                ),

            "type":
                "relationship_cardinality",

            "title":
                (
                    f"{row['parent_table']} -> "
                    f"{row['child_table']}"
                ),

            "search_text":
                " ".join(
                    [
                        str(
                            row[
                                "semantic_cardinality"
                            ]
                        ),

                        str(
                            row[
                                "observed_cardinality"
                            ]
                        ),

                        str(
                            row[
                                "join_type_recommended"
                            ]
                        ),

                        str(
                            row[
                                "purpose"
                            ]
                        )
                    ]
                ),

            "payload":
                row.to_dict()
        }
    )


search_v11_path = (
    OUT
    / "search_knowledge_v1_1.jsonl"
)


with search_v11_path.open(
    "w",
    encoding="utf-8"
) as f:

    for record in search_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            )
            +
            "\n"
        )


# =============================================================================
# 7. CROISSANT 1.1
# =============================================================================

print()

print(
    "[7/9] Gerando Croissant 1.1..."
)


# -----------------------------------------------------------------------------
# JSON-LD context
# -----------------------------------------------------------------------------

CROISSANT_CONTEXT = {

    "@language":
        "en",

    "@vocab":
        "https://schema.org/",

    "sc":
        "https://schema.org/",

    "cr":
        "http://mlcommons.org/croissant/",

    "dct":
        "http://purl.org/dc/terms/",

    "citeAs":
        "cr:citeAs",

    "column":
        "cr:column",

    "conformsTo":
        "dct:conformsTo",

    "data": {
        "@id":
            "cr:data",

        "@type":
            "@json"
    },

    "dataType": {
        "@id":
            "cr:dataType",

        "@type":
            "@vocab"
    },

    "examples": {
        "@id":
            "cr:examples",

        "@type":
            "@json"
    },

    "extract":
        "cr:extract",

    "field":
        "cr:field",

    "fileObject":
        "cr:fileObject",

    "fileProperty":
        "cr:fileProperty",

    "fileSet":
        "cr:fileSet",

    "format":
        "cr:format",

    "includes":
        "cr:includes",

    "isLiveDataset":
        "cr:isLiveDataset",

    "jsonPath":
        "cr:jsonPath",

    "key":
        "cr:key",

    "parentField":
        "cr:parentField",

    "path":
        "cr:path",

    "recordSet":
        "cr:recordSet",

    "references":
        "cr:references",

    "regex":
        "cr:regex",

    "repeated":
        "cr:repeated",

    "replace":
        "cr:replace",

    "separator":
        "cr:separator",

    "source":
        "cr:source",

    "subField":
        "cr:subField",

    "transform":
        "cr:transform"
}


# -----------------------------------------------------------------------------
# dtype mapping
# -----------------------------------------------------------------------------

def croissant_dtype(
    raw_dtype
):

    d = str(
        raw_dtype
    ).lower()

    if "int" in d:

        return "sc:Integer"

    if (
        "float" in d
        or
        "double" in d
    ):

        return "sc:Float"

    if "bool" in d:

        return "sc:Boolean"

    return "sc:Text"


# -----------------------------------------------------------------------------
# distributions
# -----------------------------------------------------------------------------

distributions = []


for _, row in tables.iterrows():

    table = row[
        "table"
    ]

    filename = row[
        "file"
    ]

    path = RAW / filename


    distributions.append(
        {
            "@type":
                "cr:FileObject",

            "@id":
                filename,

            "name":
                filename,

            "description":
                str(
                    row.get(
                        "description_pt",
                        table
                    )
                ),

            "contentUrl":
                (
                    "../../data/raw/olist/"
                    f"{filename}"
                ),

            "encodingFormat":
                "text/csv",

            "sha256":
                sha256(
                    path
                )
        }
    )


# -----------------------------------------------------------------------------
# record sets
# -----------------------------------------------------------------------------

record_sets = []


# PK map from Registry 1.0
pk_map = {}


for _, row in tables.iterrows():

    table = row[
        "table"
    ]

    raw_pk = clean(
        row.get(
            "primary_key",
            ""
        )
    )

    if raw_pk:

        pk_map[
            table
        ] = raw_pk.split(
            "+"
        )

    else:

        pk_map[
            table
        ] = []


# FK map from Registry
reference_map = {}


for _, rel in relationships_v11.iterrows():

    reference_map[
        (
            rel[
                "child_table"
            ],
            rel[
                "child_column"
            ]
        )
    ] = (
        rel[
            "parent_table"
        ],
        rel[
            "parent_column"
        ]
    )


for table in tables[
    "table"
].tolist():

    table_columns = (
        columns_v11[
            columns_v11[
                "table"
            ]
            ==
            table
        ]
        .sort_values(
            "position"
        )
    )


    record_set = {
        "@type":
            "cr:RecordSet",

        "@id":
            table,

        "name":
            table,

        "description":
            str(
                tables.loc[
                    tables[
                        "table"
                    ]
                    ==
                    table,
                    "description_pt"
                ]
                .iloc[0]
            ),

        "field":
            []
    }


    pk = pk_map.get(
        table,
        []
    )


    if pk:

        record_set[
            "key"
        ] = [
            {
                "@id":
                    f"{table}/{col}"
            }
            for col in pk
        ]


    filename = FILES[
        table
    ]


    for _, col in table_columns.iterrows():

        column = col[
            "column"
        ]


        field = {
            "@type":
                "cr:Field",

            "@id":
                f"{table}/{column}",

            "name":
                column,

            "description":
                str(
                    col.get(
                        "description_pt",
                        ""
                    )
                ),

            "dataType":
                croissant_dtype(
                    col[
                        "raw_dtype"
                    ]
                ),

            "source": {
                "fileObject": {
                    "@id":
                        filename
                },

                "extract": {
                    "column":
                        column
                }
            }
        }


        ref = reference_map.get(
            (
                table,
                column
            )
        )


        if ref:

            target_table, target_column = ref

            field[
                "references"
            ] = {
                "@id":
                    (
                        f"{target_table}/"
                        f"{target_column}"
                    )
            }


        record_set[
            "field"
        ].append(
            field
        )


    record_sets.append(
        record_set
    )


croissant = {

    "@context":
        CROISSANT_CONTEXT,

    "@type":
        "sc:Dataset",

    "name":
        "olist_brazilian_ecommerce_delivery_risk",

    "description":
        (
            "Brazilian E-Commerce Public Dataset by Olist, "
            "described for the Delivery Risk Intelligence "
            "Platform with relational schema metadata."
        ),

    "url":
        (
            "https://www.kaggle.com/datasets/"
            "olistbr/brazilian-ecommerce"
        ),

    "license":
        (
            "https://creativecommons.org/licenses/"
            "by-nc-sa/4.0/"
        ),

    "version":
        "2",

    "conformsTo":
        "http://mlcommons.org/croissant/1.1",

    "distribution":
        distributions,

    "recordSet":
        record_sets
}


croissant_path = (
    OUT
    / "croissant_1_1.json"
)


with croissant_path.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        croissant,
        f,
        ensure_ascii=False,
        indent=2
    )


# =============================================================================
# 8. INTERNAL REGISTRY VALIDATION
# =============================================================================

print()

print(
    "[8/9] Auditando o próprio Registry..."
)


validation = []


def add_check(
    check,
    status,
    observed,
    expected,
    details=""
):

    validation.append(
        {
            "check":
                check,

            "status":
                status,

            "observed":
                observed,

            "expected":
                expected,

            "details":
                details
        }
    )


# -----------------------------------------------------------------------------
# 52 columns
# -----------------------------------------------------------------------------

add_check(
    "all_columns_preserved",
    (
        "PASS"
        if len(
            columns_v11
        )
        ==
        len(
            columns
        )
        else
        "FAIL"
    ),
    len(
        columns_v11
    ),
    len(
        columns
    )
)


# -----------------------------------------------------------------------------
# no duplicate table+column
# -----------------------------------------------------------------------------

dup_cols = int(
    columns_v11
    .duplicated(
        [
            "table",
            "column"
        ]
    )
    .sum()
)


add_check(
    "column_key_unique",
    (
        "PASS"
        if dup_cols == 0
        else
        "FAIL"
    ),
    dup_cols,
    0
)


# -----------------------------------------------------------------------------
# all grain policies
# -----------------------------------------------------------------------------

missing_grain_policy = int(
    columns_v11[
        "grain_compatibility"
    ]
    .isna()
    .sum()
)


add_check(
    "all_columns_have_grain_policy",
    (
        "PASS"
        if missing_grain_policy == 0
        else
        "FAIL"
    ),
    missing_grain_policy,
    0
)


# -----------------------------------------------------------------------------
# all canonical issues
# -----------------------------------------------------------------------------

missing_canonical = int(
    issues_v11[
        "canonical_issue_id"
    ]
    .isna()
    .sum()
)


add_check(
    "all_issues_have_canonical_id",
    (
        "PASS"
        if missing_canonical == 0
        else
        "FAIL"
    ),
    missing_canonical,
    0
)


# -----------------------------------------------------------------------------
# future fields not allowed direct
# -----------------------------------------------------------------------------

future_columns = columns_v11[
    columns_v11[
        "leakage_risk"
    ]
    .astype(str)
    .str.contains(
        "CRITICAL|HIGH",
        case=False,
        regex=True
    )
]


future_direct = future_columns[
    future_columns[
        "direct_feature_allowed"
    ]
    .isin(
        [
            "YES",
            "YES_CANDIDATE",
            "AGGREGATE_ONLY"
        ]
    )
]


add_check(
    "high_leakage_not_direct_feature",
    (
        "PASS"
        if len(
            future_direct
        )
        ==
        0
        else
        "FAIL"
    ),
    len(
        future_direct
    ),
    0
)


# -----------------------------------------------------------------------------
# child grain must aggregate
# -----------------------------------------------------------------------------

child_tables = {
    "order_items",
    "payments"
}


child_bad = columns_v11[
    columns_v11[
        "table"
    ]
    .isin(
        child_tables
    )
    &
    (
        columns_v11[
            "aggregation_required"
        ]
        !=
        "YES"
    )
]


add_check(
    "child_grain_requires_aggregation",
    (
        "PASS"
        if len(
            child_bad
        )
        ==
        0
        else
        "FAIL"
    ),
    len(
        child_bad
    ),
    0
)


# -----------------------------------------------------------------------------
# no raw mutation
# -----------------------------------------------------------------------------

add_check(
    "raw_directory_exists",
    (
        "PASS"
        if RAW.exists()
        else
        "FAIL"
    ),
    RAW.exists(),
    True
)


validation_df = pd.DataFrame(
    validation
)


validation_df.to_csv(
    OUT
    / "registry_v1_1_validation.csv",
    index=False
)


critical_fail = int(
    (
        validation_df[
            "status"
        ]
        ==
        "FAIL"
    )
    .sum()
)


registry_status = (
    "PASS"
    if critical_fail == 0
    else
    "FAIL"
)


# =============================================================================
# 9. BUILD MANIFEST / SUMMARY
# =============================================================================

print()

print(
    "[9/9] Gerando manifest e resumo..."
)


manifest = {

    "registry_version":
        "1.1.0",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "project":
        "Delivery Risk Intelligence Platform",

    "dataset":
        "olistbr/brazilian-ecommerce",

    "raw_modified":
        False,

    "tables":
        int(
            len(
                tables
            )
        ),

    "columns":
        int(
            len(
                columns_v11
            )
        ),

    "relationships":
        int(
            len(
                relationships_v11
            )
        ),

    "raw_open_issues":
        int(
            len(
                issues_v11
            )
        ),

    "canonical_open_issues":
        int(
            len(
                canonical_issues
            )
        ),

    "search_records":
        int(
            len(
                search_records
            )
        ),

    "croissant_version":
        "1.1",

    "internal_registry_status":
        registry_status
}


with (
    OUT
    / "registry_v1_1_manifest.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2
    )


summary = []


summary.append(
    "=" * 100
)

summary.append(
    "DATASET KNOWLEDGE & TRUTH REGISTRY — VERSION 1.1"
)

summary.append(
    "=" * 100
)

summary.append("")

summary.append(
    f"Status interno          : {registry_status}"
)

summary.append(
    f"Tabelas                 : {len(tables)}"
)

summary.append(
    f"Colunas                 : {len(columns_v11)}"
)

summary.append(
    f"Relacionamentos         : {len(relationships_v11)}"
)

summary.append(
    f"Questões RAW            : {len(issues_v11)}"
)

summary.append(
    f"Questões canônicas      : {len(canonical_issues)}"
)

summary.append(
    f"Search records          : {len(search_records)}"
)

summary.append("")

summary.append(
    "NOVOS CONCEITOS"
)

summary.append(
    "-" * 100
)

summary.append(
    "1. semantic_cardinality != observed_cardinality"
)

summary.append(
    "2. source_grain != target_ml_grain"
)

summary.append(
    "3. direct_feature_allowed != field_exists"
)

summary.append(
    "4. child table feature -> aggregation required"
)

summary.append(
    "5. canonical issue != individual audit observation"
)

summary.append(
    "6. future information remains forbidden at t0"
)

summary.append(
    "7. geolocation must be consolidated before use"
)

summary.append(
    "8. reviews remain outside the preventive model"
)

summary.append("")

summary.append(
    "ARTEFATOS"
)

summary.append(
    "-" * 100
)


for p in sorted(
    OUT.iterdir()
):

    if p.is_file():

        summary.append(
            f"- {p.name}"
        )


summary.append("")

summary.append(
    "PRÓXIMA ETAPA:"
)

summary.append(
    "DQ Gate 03B — Conditional & Task Completeness"
)

summary.append(
    "=" * 100
)


summary_path = (
    OUT
    / "REGISTRY_V1_1_SUMMARY.txt"
)


summary_path.write_text(
    "\n".join(
        summary
    ),
    encoding="utf-8"
)


# =============================================================================
# TERMINAL
# =============================================================================

print()

print(
    "=" * 100
)

print(
    "REGISTRY 1.1 — RESULTADO"
)

print(
    "=" * 100
)

print(
    f"STATUS             : {registry_status}"
)

print(
    f"Tables             : {len(tables)}"
)

print(
    f"Columns            : {len(columns_v11)}"
)

print(
    f"Relationships      : {len(relationships_v11)}"
)

print(
    f"Raw issues         : {len(issues_v11)}"
)

print(
    f"Canonical issues   : {len(canonical_issues)}"
)

print(
    f"Search records     : {len(search_records)}"
)

print()

print(
    "Validation:"
)

print(
    validation_df.to_string(
        index=False
    )
)

print()

print(
    "Arquivos:"
)

for p in sorted(
    OUT.iterdir()
):

    if p.is_file():

        print(
            f"  - metadata/registry_v1_1/{p.name}"
        )

print()

print(
    "[OK] Nenhum arquivo RAW foi alterado."
)

print(
    f"Resumo: {summary_path}"
)


if registry_status != "PASS":

    sys.exit(2)


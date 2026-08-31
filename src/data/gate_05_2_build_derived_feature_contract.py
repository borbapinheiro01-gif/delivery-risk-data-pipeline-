#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GATE 05.2 — DERIVED FEATURE CONTRACT
Delivery Risk Intelligence Platform
================================================================================

OBJETIVO
--------
Definir formalmente as features derivadas que poderão ser construídas
posteriormente.

NÃO MATERIALIZA FEATURES.

Cada feature recebe:

- nome;
- módulo;
- fontes RAW;
- fórmula/agregação;
- granularidade;
- classe temporal;
- dependências;
- status;
- política para cada modelo.

PRINCÍPIO
---------
Nenhuma feature derivada pode "lavar" leakage.

Se uma feature depende de informação futura, TARGET ou HOLD,
ela não pode ser convertida em SAFE simplesmente por uma transformação.

Para Model 01 criamos o feature set:

    ORDER_CORE_V1

usando apenas informações do próprio pedido que já passaram no Gate 05.1.

NÃO FAZ
-------
- não treina modelo;
- não lê outcome para construir predictor;
- não cria Silver;
- não imputa;
- não normaliza;
- não consolida geolocation;
- não libera payments;
- não libera shipping_limit_date;
- não constrói seller history;
- não cria Registry 1.3.
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

GATE05 = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_05_point_in_time"
)

PIT_PATH = (
    GATE05
    / "01_raw_feature_point_in_time_registry.csv"
)

STEP01_SUMMARY = (
    GATE05
    / "dq_gate_05_1_summary.json"
)

LABEL_CONTRACT = (
    PROJECT
    / "contracts"
    / "DELIVERY_RISK_LABEL_CONTRACT_V1.json"
)

PROTOCOL = (
    PROJECT
    / "configs"
    / "model_validation_protocol_v1.json"
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


# =============================================================================
# 1. PRÉ-REQUISITOS
# =============================================================================

step01 = load_json(
    STEP01_SUMMARY
)

label_contract = load_json(
    LABEL_CONTRACT
)

protocol = load_json(
    PROTOCOL
)


if (
    step01.get("step_status")
    !=
    "PASS"
):

    raise SystemExit(
        "[BLOQUEADO] Gate 05.1 não está PASS."
    )


if (
    label_contract.get("status")
    !=
    "FROZEN_FOR_MODELING"
):

    raise SystemExit(
        "[BLOQUEADO] Label Contract V1 não está congelado."
    )


if (
    protocol.get("principle")
    !=
    "ONE_MODEL_AT_A_TIME"
):

    raise SystemExit(
        "[BLOQUEADO] Protocolo sequencial inválido."
    )


if not PIT_PATH.exists():

    raise SystemExit(
        f"[ERRO] PIT Registry ausente:\n{PIT_PATH}"
    )


pit = pd.read_csv(
    PIT_PATH,
    low_memory=False
)


print("=" * 108)
print("DQ GATE 05.2 — DERIVED FEATURE CONTRACT")
print("=" * 108)
print()

print("[PASS] Gate 05.1")
print("[PASS] Label Contract V1")
print("[PASS] ONE_MODEL_AT_A_TIME")
print()


# =============================================================================
# 2. INDEX DO RAW POINT-IN-TIME REGISTRY
# =============================================================================

pit_by_key = (
    pit
    .set_index(
        "column_key"
    )
    .to_dict(
        orient="index"
    )
)


def require_source(source):

    if source not in pit_by_key:

        raise ValueError(
            f"Fonte não encontrada no PIT Registry: {source}"
        )


def source_status(source):

    require_source(
        source
    )

    return pit_by_key[
        source
    ][
        "point_in_time_status"
    ]


# =============================================================================
# 3. FEATURE BUILDER
# =============================================================================

features = []


def add_feature(
    feature_name,
    module,
    sources,
    formula,
    aggregation,
    availability_class,
    feature_set,
    model_01_policy,
    model_02_policy,
    status,
    notes=""
):

    for source in sources:
        require_source(
            source
        )

    features.append(
        {
            "feature_name":
                feature_name,

            "module":
                module,

            "sources":
                ";".join(
                    sources
                ),

            "source_statuses":
                ";".join(
                    source_status(s)
                    for s in sources
                ),

            "target_grain":
                "ORDER",

            "formula":
                formula,

            "aggregation":
                aggregation,

            "availability_class":
                availability_class,

            "feature_set":
                feature_set,

            "model_01_policy":
                model_01_policy,

            "model_02_policy":
                model_02_policy,

            "status":
                status,

            "notes":
                notes
        }
    )


# =============================================================================
# 4. ORDER_CORE_V1
#
# Primeiro conjunto usado pelo Model 01.
#
# Deliberadamente:
# - numérico;
# - sem payments;
# - sem geolocation;
# - sem shipping_limit_date;
# - sem seller history;
# - sem produto catalog snapshot;
# - sem target/futuro.
# =============================================================================

add_feature(
    "item_count",
    "ORDER",
    [
        "order_items.order_item_id"
    ],
    "COUNT(order_item_id)",
    "COUNT_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD",
    "Number of line items in the current order."
)


add_feature(
    "unique_product_count",
    "ORDER",
    [
        "order_items.product_id"
    ],
    "COUNT_DISTINCT(product_id)",
    "COUNT_DISTINCT_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD",
    "Raw product IDs are not direct predictors."
)


add_feature(
    "unique_seller_count",
    "ORDER",
    [
        "order_items.seller_id"
    ],
    "COUNT_DISTINCT(seller_id)",
    "COUNT_DISTINCT_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD",
    "Seller IDs are used only to derive current-order seller count."
)


add_feature(
    "total_price",
    "ORDER",
    [
        "order_items.price"
    ],
    "SUM(price)",
    "SUM_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "mean_price",
    "ORDER",
    [
        "order_items.price"
    ],
    "MEAN(price)",
    "MEAN_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "max_price",
    "ORDER",
    [
        "order_items.price"
    ],
    "MAX(price)",
    "MAX_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "min_price",
    "ORDER",
    [
        "order_items.price"
    ],
    "MIN(price)",
    "MIN_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "total_freight",
    "ORDER",
    [
        "order_items.freight_value"
    ],
    "SUM(freight_value)",
    "SUM_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "mean_freight",
    "ORDER",
    [
        "order_items.freight_value"
    ],
    "MEAN(freight_value)",
    "MEAN_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "max_freight",
    "ORDER",
    [
        "order_items.freight_value"
    ],
    "MAX(freight_value)",
    "MAX_BY_ORDER",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "order_merchandise_plus_freight",
    "ORDER",
    [
        "order_items.price",
        "order_items.freight_value"
    ],
    "SUM(price) + SUM(freight_value)",
    "DERIVED_AFTER_ORDER_AGGREGATION",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


add_feature(
    "freight_price_ratio",
    "ORDER",
    [
        "order_items.price",
        "order_items.freight_value"
    ],
    "SUM(freight_value) / SUM(price)",
    "DERIVED_AFTER_ORDER_AGGREGATION",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD",
    "Division must explicitly handle zero denominator, although observed item prices are positive."
)


add_feature(
    "price_range",
    "ORDER",
    [
        "order_items.price"
    ],
    "MAX(price) - MIN(price)",
    "DERIVED_AFTER_ORDER_AGGREGATION",
    "SAFE_AT_T0",
    "ORDER_CORE_V1",
    "USE",
    "USE",
    "READY_FOR_BUILD"
)


# =============================================================================
# 5. TEMPORAL CANDIDATES
# =============================================================================

for name, formula in [

    (
        "purchase_hour",
        "hour(order_purchase_timestamp)"
    ),

    (
        "purchase_weekday",
        "weekday(order_purchase_timestamp)"
    ),

    (
        "purchase_day_of_month",
        "day(order_purchase_timestamp)"
    ),

    (
        "purchase_month",
        "month(order_purchase_timestamp)"
    ),

    (
        "purchase_quarter",
        "quarter(order_purchase_timestamp)"
    ),

    (
        "is_weekend",
        "1[weekday in Saturday, Sunday]"
    ),

]:

    add_feature(
        name,
        "TEMPORAL",
        [
            "orders.order_purchase_timestamp"
        ],
        formula,
        "DERIVE_FROM_T0",
        "SAFE_AT_T0",
        "TEMPORAL_V1",
        "DO_NOT_USE_YET",
        "DO_NOT_USE_YET",
        "READY_FOR_FUTURE_MODULE",
        "Reserved for Temporal Expert."
    )


# =============================================================================
# 6. PROMISE CANDIDATE
# =============================================================================

add_feature(
    "promised_lead_days",
    "GEO_PROMISE",
    [
        "orders.order_purchase_timestamp",
        "orders.order_estimated_delivery_date"
    ],
    (
        "date(order_estimated_delivery_date) "
        "- date(order_purchase_timestamp)"
    ),
    "ORDER_LEVEL_DERIVATION",
    "SAFE_AT_T0",
    "GEO_PROMISE_V1",
    "DO_NOT_USE_YET",
    "DO_NOT_USE_YET",
    "READY_FOR_FUTURE_MODULE",
    "Primary promise-length signal."
)


# =============================================================================
# 7. CUSTOMER GEOGRAPHY — DIRECT CONTEXT
# =============================================================================

for name, source in [

    (
        "customer_state",
        "customers.customer_state"
    ),

    (
        "customer_city",
        "customers.customer_city"
    ),

    (
        "customer_zip_prefix",
        "customers.customer_zip_code_prefix"
    ),

]:

    add_feature(
        name,
        "GEOGRAPHY",
        [
            source
        ],
        source,
        "REFERENCE_JOIN_TO_ORDER",
        "SAFE_CONTEXT_AT_T0",
        "GEOGRAPHY_CONTEXT_V1",
        "DO_NOT_USE_YET",
        "DO_NOT_USE_YET",
        "READY_FOR_FUTURE_MODULE"
    )


# =============================================================================
# 8. PRODUCT CATALOG ENRICHMENT
#
# Não vai para MODEL_01.
# Mantemos a hipótese STATIC_REFERENCE_ASSUMPTION explicitamente.
# =============================================================================

for name, source, aggregation in [

    (
        "category_count_distinct",
        "products.product_category_name",
        "COUNT_DISTINCT_BY_ORDER"
    ),

    (
        "total_product_weight_g",
        "products.product_weight_g",
        "SUM_BY_ORDER"
    ),

    (
        "max_product_weight_g",
        "products.product_weight_g",
        "MAX_BY_ORDER"
    ),

    (
        "mean_product_weight_g",
        "products.product_weight_g",
        "MEAN_BY_ORDER"
    ),

    (
        "max_product_length_cm",
        "products.product_length_cm",
        "MAX_BY_ORDER"
    ),

    (
        "max_product_height_cm",
        "products.product_height_cm",
        "MAX_BY_ORDER"
    ),

    (
        "max_product_width_cm",
        "products.product_width_cm",
        "MAX_BY_ORDER"
    ),

]:

    add_feature(
        name,
        "ORDER",
        [
            "order_items.product_id",
            source
        ],
        f"{aggregation}({source})",
        aggregation,
        "STATIC_REFERENCE_ASSUMPTION",
        "ORDER_CATALOG_V1",
        "DO_NOT_USE",
        "CANDIDATE_AFTER_REVIEW",
        "STATIC_ASSUMPTION",
        (
            "No historical product-catalog snapshots exist. "
            "Do not mix with strict Model 01 baseline."
        )
    )


add_feature(
    "total_product_volume_cm3",
    "ORDER",
    [
        "order_items.product_id",
        "products.product_length_cm",
        "products.product_height_cm",
        "products.product_width_cm"
    ],
    (
        "SUM(product_length_cm * "
        "product_height_cm * product_width_cm)"
    ),
    "PRODUCT_THEN_ORDER_AGGREGATION",
    "STATIC_REFERENCE_ASSUMPTION",
    "ORDER_CATALOG_V1",
    "DO_NOT_USE",
    "CANDIDATE_AFTER_REVIEW",
    "STATIC_ASSUMPTION"
)


# =============================================================================
# 9. SELLER STATIC GEO CONTEXT
# =============================================================================

for name, source in [

    (
        "seller_state_context",
        "sellers.seller_state"
    ),

    (
        "seller_city_context",
        "sellers.seller_city"
    ),

    (
        "seller_zip_context",
        "sellers.seller_zip_code_prefix"
    ),

]:

    add_feature(
        name,
        "GEOGRAPHY",
        [
            "order_items.seller_id",
            source
        ],
        f"aggregate({source}) to order",
        "MULTI_SELLER_ORDER_AGGREGATION_REQUIRED",
        "STATIC_REFERENCE_ASSUMPTION",
        "SELLER_GEO_CONTEXT_V1",
        "DO_NOT_USE",
        "DO_NOT_USE_YET",
        "STATIC_ASSUMPTION",
        (
            "Historical seller-address snapshots are unavailable. "
            "Multi-seller representation remains open."
        )
    )


# =============================================================================
# 10. GEOLOCATION-BASED FEATURES — BLOCKED
# =============================================================================

add_feature(
    "seller_customer_distance_km",
    "GEO_PROMISE",
    [
        "customers.customer_zip_code_prefix",
        "sellers.seller_zip_code_prefix",
        "geolocation.geolocation_zip_code_prefix",
        "geolocation.geolocation_lat",
        "geolocation.geolocation_lng"
    ],
    "Haversine(seller_zip_centroid, customer_zip_centroid)",
    "ZIP_CONSOLIDATION_THEN_ORDER_AGGREGATION",
    "HOLD_DATA_QUALITY",
    "GEO_PROMISE_V1",
    "DO_NOT_USE",
    "DO_NOT_USE",
    "BLOCKED",
    "Blocked by ISSUE-GEO-001."
)


# =============================================================================
# 11. PAYMENT FEATURES — BLOCKED
# =============================================================================

for name, source, formula in [

    (
        "payment_record_count",
        "payments.payment_sequential",
        "COUNT(payment_sequential)"
    ),

    (
        "payment_type_count",
        "payments.payment_type",
        "COUNT_DISTINCT(payment_type)"
    ),

    (
        "max_payment_installments",
        "payments.payment_installments",
        "MAX(payment_installments)"
    ),

    (
        "total_payment_value",
        "payments.payment_value",
        "SUM(payment_value)"
    ),

]:

    add_feature(
        name,
        "PAYMENT",
        [
            source
        ],
        formula,
        "AGGREGATE_BY_ORDER",
        "HOLD_PROVENANCE",
        "PAYMENT_V1",
        "DO_NOT_USE",
        "DO_NOT_USE",
        "BLOCKED",
        "Blocked by ISSUE-PAYMENT-AVAILABILITY-001."
    )


# =============================================================================
# 12. SHIPPING LIMIT — BLOCKED
# =============================================================================

add_feature(
    "shipping_limit_days_from_purchase",
    "GEO_PROMISE",
    [
        "order_items.shipping_limit_date",
        "orders.order_purchase_timestamp"
    ],
    (
        "shipping_limit_date "
        "- order_purchase_timestamp"
    ),
    "ITEM_TO_ORDER_AGGREGATION_REQUIRED",
    "HOLD_PROVENANCE",
    "GEO_PROMISE_V1",
    "DO_NOT_USE",
    "DO_NOT_USE",
    "BLOCKED",
    "Blocked by ISSUE-SHIPPING-LIMIT-001."
)


# =============================================================================
# 13. SELLER HISTORY — POINT-IN-TIME REQUIRED
# =============================================================================

seller_history_features = [

    (
        "seller_prior_order_count",
        (
            "COUNT(prior seller orders with "
            "purchase_j < purchase_i)"
        ),
        "PURCHASE_EVENT_HISTORY"
    ),

    (
        "seller_prior_observed_delivery_count",
        (
            "COUNT(prior seller orders with "
            "delivery_j < purchase_i)"
        ),
        "OUTCOME_AVAILABLE_HISTORY"
    ),

    (
        "seller_prior_late_count",
        (
            "COUNT(prior late orders with "
            "delivery_j < purchase_i)"
        ),
        "OUTCOME_AVAILABLE_HISTORY"
    ),

    (
        "seller_late_rate_prior",
        (
            "prior_late_count / "
            "prior_observed_delivery_count"
        ),
        "OUTCOME_AVAILABLE_HISTORY"
    ),

    (
        "seller_late_rate_30d",
        (
            "smoothed late rate among outcomes "
            "available in prior 30 days"
        ),
        "OUTCOME_AVAILABLE_HISTORY"
    ),

    (
        "seller_late_rate_90d",
        (
            "smoothed late rate among outcomes "
            "available in prior 90 days"
        ),
        "OUTCOME_AVAILABLE_HISTORY"
    ),

    (
        "seller_order_volume_7d",
        (
            "COUNT seller purchases in "
            "[purchase_i-7d, purchase_i)"
        ),
        "PURCHASE_EVENT_HISTORY"
    ),

    (
        "seller_order_volume_30d",
        (
            "COUNT seller purchases in "
            "[purchase_i-30d, purchase_i)"
        ),
        "PURCHASE_EVENT_HISTORY"
    ),
]


for name, formula, history_type in seller_history_features:

    add_feature(
        name,
        "SELLER_HISTORY",
        [
            "order_items.seller_id",
            "orders.order_purchase_timestamp"
        ],
        formula,
        history_type,
        "POINT_IN_TIME_REQUIRED",
        "SELLER_HISTORY_V1",
        "DO_NOT_USE",
        "DO_NOT_USE",
        "REQUIRES_TEMPORAL_BUILDER",
        (
            "For outcome-dependent history, order j can contribute only "
            "when its delivery outcome was already observable before "
            "purchase_i."
        )
    )


# =============================================================================
# 14. DATAFRAME
# =============================================================================

contract = pd.DataFrame(
    features
)


# =============================================================================
# 15. DEPENDENCY LEAKAGE AUDIT
# =============================================================================

blocked_source_statuses = {
    "FORBIDDEN_FUTURE",
    "TARGET_ONLY"
}


def has_forbidden_dependency(row):

    statuses = set(
        str(
            row[
                "source_statuses"
            ]
        ).split(";")
    )

    return bool(
        statuses
        &
        blocked_source_statuses
    )


contract[
    "has_forbidden_or_target_dependency"
] = contract.apply(
    has_forbidden_dependency,
    axis=1
)


# Seller-history formulas refer conceptually to historical outcomes,
# but the raw outcome column is deliberately NOT listed as a current-order
# predictor source. It will be audited separately by the temporal builder.
#
# This prevents a false claim that outcome is a direct feature.


# =============================================================================
# 16. MODEL 01 FEATURE SET
# =============================================================================

model01 = contract[
    contract[
        "model_01_policy"
    ]
    ==
    "USE"
].copy()


# =============================================================================
# 17. MODEL 01 STRICT SAFETY CHECK
# =============================================================================

safe_dependency_statuses_model01 = {
    "AGGREGATE_SAFE_AT_T0",
    "JOIN_KEY_ONLY"
}


model01_dependency_violations = []


for _, row in model01.iterrows():

    statuses = set(
        str(
            row[
                "source_statuses"
            ]
        ).split(";")
    )

    invalid = (
        statuses
        -
        safe_dependency_statuses_model01
    )

    if invalid:

        model01_dependency_violations.append(
            {
                "feature_name":
                    row[
                        "feature_name"
                    ],

                "invalid_source_statuses":
                    ";".join(
                        sorted(
                            invalid
                        )
                    )
            }
        )


model01_violations_df = pd.DataFrame(
    model01_dependency_violations
)


# =============================================================================
# 18. BLOCKED FEATURES
# =============================================================================

blocked = contract[
    contract[
        "status"
    ]
    ==
    "BLOCKED"
].copy()


# =============================================================================
# 19. MODULE SUMMARY
# =============================================================================

module_summary = (
    contract
    .groupby(
        [
            "module",
            "status"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="feature_count"
    )
    .sort_values(
        [
            "module",
            "status"
        ]
    )
)


# =============================================================================
# 20. VALIDATION
# =============================================================================

checks = []


def add_check(
    check,
    condition,
    observed,
    expected
):

    checks.append(
        {
            "check":
                check,

            "status":
                (
                    "PASS"
                    if condition
                    else
                    "FAIL"
                ),

            "observed":
                observed,

            "expected":
                expected
        }
    )


add_check(
    "derived_feature_names_unique",
    (
        contract[
            "feature_name"
        ]
        .duplicated()
        .sum()
        ==
        0
    ),
    int(
        contract[
            "feature_name"
        ]
        .duplicated()
        .sum()
    ),
    0
)


add_check(
    "no_direct_future_target_dependencies",
    (
        contract[
            "has_forbidden_or_target_dependency"
        ]
        .sum()
        ==
        0
    ),
    int(
        contract[
            "has_forbidden_or_target_dependency"
        ]
        .sum()
    ),
    0
)


add_check(
    "model01_has_features",
    len(
        model01
    ) > 0,
    len(
        model01
    ),
    "> 0"
)


add_check(
    "model01_no_hold_static_future_target",
    len(
        model01_dependency_violations
    ) == 0,
    len(
        model01_dependency_violations
    ),
    0
)


add_check(
    "payment_remains_blocked",
    (
        (
            contract[
                "module"
            ]
            ==
            "PAYMENT"
        )
        &
        (
            contract[
                "status"
            ]
            ==
            "BLOCKED"
        )
    ).all()
    if (
        contract[
            "module"
        ]
        .eq(
            "PAYMENT"
        )
        .any()
    )
    else False,
    int(
        (
            (
                contract[
                    "module"
                ]
                ==
                "PAYMENT"
            )
            &
            (
                contract[
                    "status"
                ]
                ==
                "BLOCKED"
            )
        )
        .sum()
    ),
    int(
        (
            contract[
                "module"
            ]
            ==
            "PAYMENT"
        )
        .sum()
    )
)


shipping = contract[
    contract[
        "feature_name"
    ]
    ==
    "shipping_limit_days_from_purchase"
]


add_check(
    "shipping_limit_derived_remains_blocked",
    (
        len(
            shipping
        )
        ==
        1
        and
        shipping.iloc[
            0
        ][
            "status"
        ]
        ==
        "BLOCKED"
    ),
    (
        shipping.iloc[
            0
        ][
            "status"
        ]
        if len(
            shipping
        )
        else "MISSING"
    ),
    "BLOCKED"
)


seller_hist = contract[
    contract[
        "module"
    ]
    ==
    "SELLER_HISTORY"
]


add_check(
    "seller_history_requires_temporal_builder",
    (
        len(
            seller_hist
        )
        >
        0
        and
        seller_hist[
            "status"
        ]
        .eq(
            "REQUIRES_TEMPORAL_BUILDER"
        )
        .all()
    ),
    int(
        seller_hist[
            "status"
        ]
        .eq(
            "REQUIRES_TEMPORAL_BUILDER"
        )
        .sum()
    ),
    len(
        seller_hist
    )
)


validation = pd.DataFrame(
    checks
)


failures = int(
    (
        validation[
            "status"
        ]
        ==
        "FAIL"
    )
    .sum()
)


step_status = (
    "PASS"
    if failures == 0
    else
    "FAIL"
)


# =============================================================================
# 21. OUTPUTS
# =============================================================================

contract.to_csv(
    GATE05
    / "02_derived_feature_contract.csv",
    index=False
)


model01.to_csv(
    GATE05
    / "02b_model_01_order_core_v1.csv",
    index=False
)


blocked.to_csv(
    GATE05
    / "02c_blocked_derived_features.csv",
    index=False
)


module_summary.to_csv(
    GATE05
    / "02d_module_feature_summary.csv",
    index=False
)


validation.to_csv(
    GATE05
    / "02e_step_validation.csv",
    index=False
)


model01_violations_df.to_csv(
    GATE05
    / "02f_model01_dependency_violations.csv",
    index=False
)


# =============================================================================
# 22. SUMMARY JSON
# =============================================================================

summary = {
    "step":
        "DQ_GATE_05_2_DERIVED_FEATURE_CONTRACT",

    "step_status":
        step_status,

    "gate_05_status":
        "IN_PROGRESS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "derived_features_defined":
        len(
            contract
        ),

    "model_01_feature_set":
        "ORDER_CORE_V1",

    "model_01_features":
        len(
            model01
        ),

    "blocked_features":
        len(
            blocked
        ),

    "seller_history_features":
        len(
            seller_hist
        ),

    "direct_future_target_dependency_violations":
        int(
            contract[
                "has_forbidden_or_target_dependency"
            ]
            .sum()
        ),

    "model01_dependency_violations":
        len(
            model01_dependency_violations
        ),

    "validation_failures":
        failures,

    "model_01_unlocked":
        False,

    "raw_modified":
        False,

    "silver_created":
        False,

    "model_trained":
        False
}


with (
    GATE05
    / "dq_gate_05_2_summary.json"
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
# 23. REPORT
# =============================================================================

report = f"""
========================================================================================================
DQ GATE 05.2 — DERIVED FEATURE CONTRACT
========================================================================================================

STEP STATUS                         : {step_status}
GATE 05 STATUS                      : IN_PROGRESS

DERIVED FEATURES DEFINED            : {len(contract)}
MODEL 01 FEATURE SET                : ORDER_CORE_V1
MODEL 01 FEATURES                   : {len(model01)}

BLOCKED FEATURES                    : {len(blocked)}
SELLER HISTORY DEFINITIONS          : {len(seller_hist)}

DIRECT FUTURE/TARGET VIOLATIONS     : {int(contract['has_forbidden_or_target_dependency'].sum())}
MODEL 01 DEPENDENCY VIOLATIONS      : {len(model01_dependency_violations)}
VALIDATION FAILURES                 : {failures}

MODEL 01
--------------------------------------------------------------------------------------------------------
Still LOCKED.

The feature contract exists, but no feature matrix has been materialized yet.

ORDER_CORE_V1 deliberately excludes:
- payments;
- shipping_limit_date;
- geolocation coordinates;
- seller historical outcomes;
- reviews;
- final order status;
- approval/carrier/delivery timestamps;
- product static-reference enrichment.

IMPORTANT
--------------------------------------------------------------------------------------------------------
This step defines transformations only.

RAW modified                        : NO
Silver created                      : NO
Model trained                       : NO

========================================================================================================
"""


(
    GATE05
    / "DQ_GATE_05_2_DERIVED_FEATURE_CONTRACT_REPORT.txt"
).write_text(
    report.strip()
    +
    "\n",
    encoding="utf-8"
)


# =============================================================================
# TERMINAL
# =============================================================================

print()
print("=" * 108)
print("DQ GATE 05.2 — VALIDATION")
print("=" * 108)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("MODEL 01 — ORDER_CORE_V1")
print("=" * 108)

print(
    model01[
        [
            "feature_name",
            "formula",
            "source_statuses",
            "availability_class",
            "status"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 108)
print("BLOCKED DERIVED FEATURES")
print("=" * 108)

print(
    blocked[
        [
            "feature_name",
            "module",
            "availability_class",
            "status",
            "notes"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 108)
print("MODULE SUMMARY")
print("=" * 108)

print(
    module_summary.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("RESULTADO")
print("=" * 108)

print(
    f"STEP STATUS              : {step_status}"
)

print(
    "GATE 05 STATUS           : IN_PROGRESS"
)

print(
    f"DERIVED FEATURES         : {len(contract)}"
)

print(
    f"MODEL 01 FEATURES        : {len(model01)}"
)

print(
    f"BLOCKED FEATURES         : {len(blocked)}"
)

print(
    f"VALIDATION FAILURES      : {failures}"
)

print(
    "MODEL 01 UNLOCKED        : NÃO"
)

print(
    "RAW MODIFIED             : NÃO"
)

print(
    "SILVER CREATED           : NÃO"
)

print(
    "MODEL TRAINED            : NÃO"
)


if step_status != "PASS":

    sys.exit(2)


print()
print(
    "[PASS] ETAPA 05.2 VALIDADA TECNICAMENTE."
)

print(
    "Gate 05 continua IN_PROGRESS."
)

print(
    "Não executar modelo ainda."
)


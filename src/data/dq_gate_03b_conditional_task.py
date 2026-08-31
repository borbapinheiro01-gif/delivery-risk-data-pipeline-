#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DQ GATE 03B — CONDITIONAL & TASK COMPLETENESS
Delivery Risk Intelligence Platform
===============================================================================

OBJETIVO
--------
Separar corretamente:

    SOURCE COMPLETENESS
        !=
    TASK COMPLETENESS

e:

    RELATION_MISSING
        !=
    ATTRIBUTE_MISSING
        !=
    NOT_APPLICABLE
        !=
    NOT_YET_OBSERVED
        !=
    CENSORED

Este gate NÃO:
    - define definitivamente o label;
    - imputa;
    - remove;
    - winsoriza;
    - cria Silver;
    - cria Gold;
    - treina modelos;
    - altera RAW.

O Gate 04 será responsável por Label / Construct Validity.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import numpy as np
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

CONFIG = (
    PROJECT
    / "configs"
    / "dq_gate_03b_conditional_task.json"
)

OUT = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# PREREQUISITES
# =============================================================================

PREREQUISITES = {

    "DQ_GATE_01":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_01_structural"
        / "dq_gate_01_summary.json",

    "DQ_GATE_02":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_02_semantic"
        / "dq_gate_02_summary.json",

    "DQ_GATE_03":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_03_statistical"
        / "dq_gate_03_summary.json",

    "REGISTRY_1_1":
        PROJECT
        / "metadata"
        / "registry_v1_1"
        / "registry_v1_1_manifest.json",
}


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


for name, path in PREREQUISITES.items():

    if not path.exists():

        print(
            f"[BLOQUEADO] Pré-requisito ausente: {name}"
        )

        print(
            path
        )

        sys.exit(2)


gate1 = load_json(
    PREREQUISITES["DQ_GATE_01"]
)

gate2 = load_json(
    PREREQUISITES["DQ_GATE_02"]
)

gate3 = load_json(
    PREREQUISITES["DQ_GATE_03"]
)

registry = load_json(
    PREREQUISITES["REGISTRY_1_1"]
)


for name, obj in [
    ("DQ_GATE_01", gate1),
    ("DQ_GATE_02", gate2),
    ("DQ_GATE_03", gate3)
]:

    if obj.get(
        "status"
    ) != "PASS":

        print(
            f"[BLOQUEADO] {name} não está PASS."
        )

        sys.exit(2)


if registry.get(
    "internal_registry_status"
) != "PASS":

    print(
        "[BLOQUEADO] Registry 1.1 não está PASS."
    )

    sys.exit(2)


# =============================================================================
# REGISTRY POLICY
# =============================================================================

GRAIN_POLICY = (
    PROJECT
    / "metadata"
    / "registry_v1_1"
    / "ml_grain_policy_registry.csv"
)

COLUMN_CATALOG = (
    PROJECT
    / "metadata"
    / "registry_v1_1"
    / "column_catalog_v1_1.csv"
)


if not GRAIN_POLICY.exists():

    raise SystemExit(
        f"[ERRO] Grain Policy ausente: {GRAIN_POLICY}"
    )


grain_policy = pd.read_csv(
    GRAIN_POLICY,
    low_memory=False
)


# Verificar regra que acabamos de formalizar:
# order_items/payments sempre devem exigir agregação.

for child_table in [
    "order_items",
    "payments"
]:

    bad = grain_policy[
        (
            grain_policy[
                "table"
            ]
            ==
            child_table
        )
        &
        (
            grain_policy[
                "aggregation_required"
            ]
            !=
            "YES"
        )
    ]

    if len(
        bad
    ):

        print(
            f"[BLOQUEADO] Grain Policy inválida em {child_table}."
        )

        print(
            bad.to_string(
                index=False
            )
        )

        sys.exit(2)


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

    "orders":
        "olist_orders_dataset.csv",

    "products":
        "olist_products_dataset.csv",

    "sellers":
        "olist_sellers_dataset.csv",
}


dfs = {}


for table, filename in FILES.items():

    path = RAW / filename

    if not path.exists():

        raise SystemExit(
            f"[ERRO] RAW ausente: {path}"
        )

    dfs[
        table
    ] = pd.read_csv(
        path,
        low_memory=False
    )


orders = dfs["orders"]
items = dfs["order_items"]
payments = dfs["payments"]
customers = dfs["customers"]
products = dfs["products"]
sellers = dfs["sellers"]
geo = dfs["geolocation"]


# =============================================================================
# HELPERS
# =============================================================================

def pct(
    numerator,
    denominator
):

    if denominator == 0:

        return np.nan

    return (
        100.0
        *
        numerator
        /
        denominator
    )


results = []


def add_result(
    check_id,
    dimension,
    scope,
    entity,
    condition,
    severity,
    status,
    numerator,
    denominator,
    expected,
    details=""
):

    results.append(
        {
            "gate":
                "DQ_GATE_03B_CONDITIONAL_TASK_COMPLETENESS",

            "check_id":
                check_id,

            "dimension":
                dimension,

            "scope":
                scope,

            "entity":
                entity,

            "applicability_condition":
                condition,

            "severity":
                severity,

            "status":
                status,

            "affected":
                int(
                    numerator
                ),

            "denominator":
                int(
                    denominator
                ),

            "affected_pct":
                pct(
                    numerator,
                    denominator
                ),

            "expected":
                expected,

            "details":
                details
        }
    )


def zero_tolerance_status(
    value
):

    return (
        "PASS"
        if value == 0
        else
        "FAIL"
    )


def warning_status(
    value
):

    return (
        "PASS"
        if value == 0
        else
        "WARN"
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 104)

print(
    "DQ GATE 03B — CONDITIONAL & TASK COMPLETENESS"
)

print(
    "Delivery Risk Intelligence Platform"
)

print("=" * 104)

print()

print(
    "[PASS] Gate 01"
)

print(
    "[PASS] Gate 02"
)

print(
    "[PASS] Gate 03"
)

print(
    "[PASS] Registry 1.1"
)

print()

print(
    "Princípio central:"
)

print(
    "RELATION_MISSING != ATTRIBUTE_MISSING"
)

print(
    "SOURCE_QUALITY    != TASK_QUALITY"
)


# =============================================================================
# 1. TASK CANDIDATE COHORT
# =============================================================================

print()
print("=" * 104)
print("1. TASK CANDIDATE COHORT")
print("=" * 104)


purchase = pd.to_datetime(
    orders[
        "order_purchase_timestamp"
    ],
    errors="coerce"
)

estimated = pd.to_datetime(
    orders[
        "order_estimated_delivery_date"
    ],
    errors="coerce"
)

delivered = pd.to_datetime(
    orders[
        "order_delivered_customer_date"
    ],
    errors="coerce"
)


task_mask = (
    orders[
        "order_status"
    ]
    .eq(
        "delivered"
    )
    &
    purchase.notna()
    &
    estimated.notna()
    &
    delivered.notna()
)


task = (
    orders.loc[
        task_mask
    ]
    .copy()
)


task[
    "purchase_timestamp_parsed"
] = purchase[
    task_mask
]


task[
    "purchase_month"
] = (
    task[
        "purchase_timestamp_parsed"
    ]
    .dt
    .to_period(
        "M"
    )
    .astype(str)
)


n_source_orders = len(
    orders
)

n_task = len(
    task
)


print(
    f"Source orders                 : {n_source_orders:,}"
)

print(
    f"Task candidate observable     : {n_task:,}"
)

print(
    f"Task candidate / source       : {pct(n_task, n_source_orders):.4f}%"
)


add_result(
    check_id=
        "TASK-001",

    dimension=
        "task_cohort",

    scope=
        "TASK",

    entity=
        "order",

    condition=
        (
            "status=delivered AND purchase observed AND "
            "estimated delivery observed AND actual delivery observed"
        ),

    severity=
        "CRITICAL",

    status=
        (
            "PASS"
            if n_task > 0
            else "FAIL"
        ),

    numerator=
        0
        if n_task > 0
        else 1,

    denominator=
        max(
            n_source_orders,
            1
        ),

    expected=
        "> 0 eligible observations",

    details=
        (
            "This is a label-observable candidate cohort. "
            "Final label semantics/window remain Gate 04 decisions."
        )
)


# =============================================================================
# 2. SOURCE-LEVEL RELATIONSHIP COVERAGE
# =============================================================================

print()
print("=" * 104)
print("2. SOURCE RELATIONSHIP COVERAGE")
print("=" * 104)


source_order_ids = set(
    orders[
        "order_id"
    ]
)

item_order_ids = set(
    items[
        "order_id"
    ]
)

payment_order_ids = set(
    payments[
        "order_id"
    ]
)

customer_ids = set(
    customers[
        "customer_id"
    ]
)

product_ids = set(
    products[
        "product_id"
    ]
)

seller_ids = set(
    sellers[
        "seller_id"
    ]
)


src_orders_without_items = int(
    (
        ~orders[
            "order_id"
        ]
        .isin(
            item_order_ids
        )
    )
    .sum()
)


src_orders_without_payments = int(
    (
        ~orders[
            "order_id"
        ]
        .isin(
            payment_order_ids
        )
    )
    .sum()
)


src_orders_without_customer = int(
    (
        ~orders[
            "customer_id"
        ]
        .isin(
            customer_ids
        )
    )
    .sum()
)


src_item_product_orphans = int(
    (
        ~items[
            "product_id"
        ]
        .isin(
            product_ids
        )
    )
    .sum()
)


src_item_seller_orphans = int(
    (
        ~items[
            "seller_id"
        ]
        .isin(
            seller_ids
        )
    )
    .sum()
)


source_relation_rows = pd.DataFrame(
    [
        {
            "metric":
                "orders_without_items",

            "denominator":
                len(
                    orders
                ),

            "affected":
                src_orders_without_items,

            "affected_pct":
                pct(
                    src_orders_without_items,
                    len(
                        orders
                    )
                )
        },

        {
            "metric":
                "orders_without_payments",

            "denominator":
                len(
                    orders
                ),

            "affected":
                src_orders_without_payments,

            "affected_pct":
                pct(
                    src_orders_without_payments,
                    len(
                        orders
                    )
                )
        },

        {
            "metric":
                "orders_without_customer_resolution",

            "denominator":
                len(
                    orders
                ),

            "affected":
                src_orders_without_customer,

            "affected_pct":
                pct(
                    src_orders_without_customer,
                    len(
                        orders
                    )
                )
        },

        {
            "metric":
                "item_product_orphans",

            "denominator":
                len(
                    items
                ),

            "affected":
                src_item_product_orphans,

            "affected_pct":
                pct(
                    src_item_product_orphans,
                    len(
                        items
                    )
                )
        },

        {
            "metric":
                "item_seller_orphans",

            "denominator":
                len(
                    items
                ),

            "affected":
                src_item_seller_orphans,

            "affected_pct":
                pct(
                    src_item_seller_orphans,
                    len(
                        items
                    )
                )
        }
    ]
)


source_relation_rows.to_csv(
    OUT
    /
    "01_source_relationship_coverage.csv",
    index=False
)


print(
    source_relation_rows.to_string(
        index=False,
        formatters={
            "affected_pct":
                "{:.6f}".format
        }
    )
)


# =============================================================================
# 3. TASK RELATIONSHIP COVERAGE
# =============================================================================

print()
print("=" * 104)
print("3. TASK RELATIONSHIP COVERAGE")
print("=" * 104)


task_ids = set(
    task[
        "order_id"
    ]
)


task_items = (
    items[
        items[
            "order_id"
        ]
        .isin(
            task_ids
        )
    ]
    .copy()
)


task_payments = (
    payments[
        payments[
            "order_id"
        ]
        .isin(
            task_ids
        )
    ]
    .copy()
)


task_item_orders = set(
    task_items[
        "order_id"
    ]
)


task_payment_orders = set(
    task_payments[
        "order_id"
    ]
)


task_orders_without_items = int(
    (
        ~task[
            "order_id"
        ]
        .isin(
            task_item_orders
        )
    )
    .sum()
)


task_orders_without_payment = int(
    (
        ~task[
            "order_id"
        ]
        .isin(
            task_payment_orders
        )
    )
    .sum()
)


task_orders_without_customer = int(
    (
        ~task[
            "customer_id"
        ]
        .isin(
            customer_ids
        )
    )
    .sum()
)


task_product_orphans = int(
    (
        ~task_items[
            "product_id"
        ]
        .isin(
            product_ids
        )
    )
    .sum()
)


task_seller_orphans = int(
    (
        ~task_items[
            "seller_id"
        ]
        .isin(
            seller_ids
        )
    )
    .sum()
)


checks = [

    (
        "TASK-REL-001",
        "task_orders_without_items",
        task_orders_without_items,
        len(task),
        "CRITICAL"
    ),

    (
        "TASK-REL-002",
        "task_orders_without_customer",
        task_orders_without_customer,
        len(task),
        "CRITICAL"
    ),

    (
        "TASK-REL-003",
        "task_item_product_orphans",
        task_product_orphans,
        len(task_items),
        "CRITICAL"
    ),

    (
        "TASK-REL-004",
        "task_item_seller_orphans",
        task_seller_orphans,
        len(task_items),
        "CRITICAL"
    ),

    (
        "TASK-REL-005",
        "task_orders_without_payment",
        task_orders_without_payment,
        len(task),
        "WARNING"
    )
]


for (
    check_id,
    metric,
    value,
    denominator,
    severity
) in checks:

    if severity == "CRITICAL":

        status = zero_tolerance_status(
            value
        )

    else:

        status = warning_status(
            value
        )


    add_result(
        check_id=
            check_id,

        dimension=
            "conditional_relationship_completeness",

        scope=
            "TASK",

        entity=
            metric,

        condition=
            "entity/relation is applicable to candidate ML order",

        severity=
            severity,

        status=
            status,

        numerator=
            value,

        denominator=
            denominator,

        expected=
            "0 relationship-missing observations",

        details=
            (
                "Relationship missingness is evaluated separately "
                "from child attribute missingness."
            )
    )


    print(
        f"[{status:<4}] "
        f"{metric:<40} "
        f"{value:>7,} "
        f"({pct(value, denominator):.6f}%)"
    )


# =============================================================================
# 4. ITEM / PRODUCT CONDITIONAL COMPLETENESS
# =============================================================================

print()
print("=" * 104)
print("4. PRODUCT ATTRIBUTES — CONDITIONED ON PRODUCT RESOLUTION")
print("=" * 104)


product_cols = [
    "product_id",
    "product_category_name",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]


task_item_product = (
    task_items
    .merge(
        products[
            product_cols
        ],
        on=
            "product_id",

        how=
            "left",

        validate=
            "many_to_one",

        indicator=
            "__product_merge"
    )
)


task_item_product[
    "product_resolved"
] = (
    task_item_product[
        "__product_merge"
    ]
    ==
    "both"
)


resolved_product_rows = task_item_product[
    task_item_product[
        "product_resolved"
    ]
].copy()


resolved_product_rows[
    "category_missing"
] = (
    resolved_product_rows[
        "product_category_name"
    ]
    .isna()
)


physical_cols = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]


resolved_product_rows[
    "physical_missing"
] = (
    resolved_product_rows[
        physical_cols
    ]
    .isna()
    .any(
        axis=1
    )
)


category_missing_exposure = int(
    resolved_product_rows[
        "category_missing"
    ]
    .sum()
)


physical_missing_exposure = int(
    resolved_product_rows[
        "physical_missing"
    ]
    .sum()
)


# Unique products used by task cohort
task_product_unique = (
    resolved_product_rows[
        [
            "product_id",
            "product_category_name",
            *physical_cols
        ]
    ]
    .drop_duplicates(
        subset=[
            "product_id"
        ]
    )
)


task_unique_product_category_missing = int(
    task_product_unique[
        "product_category_name"
    ]
    .isna()
    .sum()
)


task_unique_product_physical_missing = int(
    task_product_unique[
        physical_cols
    ]
    .isna()
    .any(
        axis=1
    )
    .sum()
)


conditional_rows = []


def add_conditional_row(
    scope,
    entity,
    attribute,
    condition,
    denominator,
    missing,
    missing_reason,
    ml_relevance,
    severity
):

    status = (
        zero_tolerance_status(
            missing
        )
        if severity == "CRITICAL"
        else
        warning_status(
            missing
        )
    )

    conditional_rows.append(
        {
            "scope":
                scope,

            "entity":
                entity,

            "attribute":
                attribute,

            "applicability_condition":
                condition,

            "denominator":
                denominator,

            "missing":
                missing,

            "missing_pct":
                pct(
                    missing,
                    denominator
                ),

            "completeness_pct":
                (
                    100
                    -
                    pct(
                        missing,
                        denominator
                    )
                    if denominator
                    else np.nan
                ),

            "missing_reason":
                missing_reason,

            "ml_relevance":
                ml_relevance,

            "severity":
                severity,

            "status":
                status
        }
    )


add_conditional_row(
    scope=
        "TASK_ITEM_EXPOSURE",

    entity=
        "product",

    attribute=
        "product_category_name",

    condition=
        "order has item AND product_id resolves",

    denominator=
        len(
            resolved_product_rows
        ),

    missing=
        category_missing_exposure,

    missing_reason=
        "ATTRIBUTE_MISSING",

    ml_relevance=
        "optional candidate feature",

    severity=
        "WARNING"
)


add_conditional_row(
    scope=
        "TASK_ITEM_EXPOSURE",

    entity=
        "product",

    attribute=
        "physical_dimensions_or_weight",

    condition=
        "order has item AND product_id resolves",

    denominator=
        len(
            resolved_product_rows
        ),

    missing=
        physical_missing_exposure,

    missing_reason=
        "ATTRIBUTE_MISSING",

    ml_relevance=
        "optional candidate feature",

    severity=
        "WARNING"
)


add_conditional_row(
    scope=
        "TASK_UNIQUE_PRODUCT",

    entity=
        "product",

    attribute=
        "product_category_name",

    condition=
        "product is referenced by at least one task item",

    denominator=
        len(
            task_product_unique
        ),

    missing=
        task_unique_product_category_missing,

    missing_reason=
        "ATTRIBUTE_MISSING",

    ml_relevance=
        "entity-level diagnostic",

    severity=
        "WARNING"
)


add_conditional_row(
    scope=
        "TASK_UNIQUE_PRODUCT",

    entity=
        "product",

    attribute=
        "physical_dimensions_or_weight",

    condition=
        "product is referenced by at least one task item",

    denominator=
        len(
            task_product_unique
        ),

    missing=
        task_unique_product_physical_missing,

    missing_reason=
        "ATTRIBUTE_MISSING",

    ml_relevance=
        "entity-level diagnostic",

    severity=
        "WARNING"
)


# =============================================================================
# 5. ITEM COMMERCIAL ATTRIBUTES
# =============================================================================

print()
print("=" * 104)
print("5. ITEM COMMERCIAL ATTRIBUTES — CONDITIONED ON ITEM EXISTENCE")
print("=" * 104)


task_item_commercial_missing = (
    task_items[
        [
            "price",
            "freight_value"
        ]
    ]
    .isna()
    .any(
        axis=1
    )
)


commercial_missing = int(
    task_item_commercial_missing.sum()
)


add_conditional_row(
    scope=
        "TASK_ITEM",

    entity=
        "order_item",

    attribute=
        "price_or_freight",

    condition=
        "order item exists",

    denominator=
        len(
            task_items
        ),

    missing=
        commercial_missing,

    missing_reason=
        "ATTRIBUTE_MISSING",

    ml_relevance=
        "core purchase-time candidate features",

    severity=
        "CRITICAL"
)


# =============================================================================
# 6. CUSTOMER GEO CONDITIONAL COMPLETENESS
# =============================================================================

print()
print("=" * 104)
print("6. CUSTOMER GEO — CONDITIONED ON CUSTOMER RESOLUTION")
print("=" * 104)


geo_zip_set = set(
    geo[
        "geolocation_zip_code_prefix"
    ]
    .dropna()
)


task_customer = (
    task[
        [
            "order_id",
            "customer_id"
        ]
    ]
    .merge(
        customers[
            [
                "customer_id",
                "customer_zip_code_prefix",
                "customer_state"
            ]
        ],
        on=
            "customer_id",

        how=
            "left",

        validate=
            "many_to_one",

        indicator=
            "__customer_merge"
    )
)


task_customer[
    "customer_resolved"
] = (
    task_customer[
        "__customer_merge"
    ]
    ==
    "both"
)


# -------------------------------------------------------------------------
# IMPORTANTE:
#
# customer_geo_missing pertence ao DataFrame task_customer porque esse
# mesmo DataFrame será usado posteriormente para construir customer_flags.
#
# Para clientes não resolvidos, geolocation NÃO é simplesmente "missing":
# a relação upstream customer já está ausente. Por isso usamos pd.NA.
#
# Somente quando customer_id resolve para customers é que avaliamos se o
# ZIP possui referência na tabela geolocation.
# -------------------------------------------------------------------------

task_customer[
    "customer_geo_missing"
] = pd.Series(
    pd.NA,
    index=task_customer.index,
    dtype="boolean"
)


_customer_resolved_mask = (
    task_customer[
        "customer_resolved"
    ]
    .fillna(False)
)


task_customer.loc[
    _customer_resolved_mask,
    "customer_geo_missing"
] = (
    ~task_customer.loc[
        _customer_resolved_mask,
        "customer_zip_code_prefix"
    ]
    .isin(
        geo_zip_set
    )
)


customer_resolved_rows = task_customer.loc[
    _customer_resolved_mask
].copy()


customer_geo_missing = int(
    customer_resolved_rows[
        "customer_geo_missing"
    ]
    .sum()
)


add_conditional_row(
    scope=
        "TASK_ORDER",

    entity=
        "customer",

    attribute=
        "geolocation_reference",

    condition=
        "customer_id resolves to customers",

    denominator=
        len(
            customer_resolved_rows
        ),

    missing=
        customer_geo_missing,

    missing_reason=
        "RELATION_MISSING",

    ml_relevance=
        "geographic candidate features",

    severity=
        "WARNING"
)


# =============================================================================
# 7. SELLER GEO CONDITIONAL COMPLETENESS
# =============================================================================

print()
print("=" * 104)
print("7. SELLER GEO — CONDITIONED ON SELLER RESOLUTION")
print("=" * 104)


task_item_seller = (
    task_items
    .merge(
        sellers[
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_state"
            ]
        ],
        on=
            "seller_id",

        how=
            "left",

        validate=
            "many_to_one",

        indicator=
            "__seller_merge"
    )
)


task_item_seller[
    "seller_resolved"
] = (
    task_item_seller[
        "__seller_merge"
    ]
    ==
    "both"
)


seller_resolved_rows = task_item_seller[
    task_item_seller[
        "seller_resolved"
    ]
].copy()


seller_resolved_rows[
    "seller_geo_missing"
] = (
    ~seller_resolved_rows[
        "seller_zip_code_prefix"
    ]
    .isin(
        geo_zip_set
    )
)


seller_geo_missing_exposure = int(
    seller_resolved_rows[
        "seller_geo_missing"
    ]
    .sum()
)


unique_task_sellers = (
    seller_resolved_rows[
        [
            "seller_id",
            "seller_zip_code_prefix"
        ]
    ]
    .drop_duplicates(
        subset=[
            "seller_id"
        ]
    )
)


unique_task_sellers[
    "seller_geo_missing"
] = (
    ~unique_task_sellers[
        "seller_zip_code_prefix"
    ]
    .isin(
        geo_zip_set
    )
)


unique_seller_geo_missing = int(
    unique_task_sellers[
        "seller_geo_missing"
    ]
    .sum()
)


add_conditional_row(
    scope=
        "TASK_ITEM_EXPOSURE",

    entity=
        "seller",

    attribute=
        "geolocation_reference",

    condition=
        "seller_id resolves to sellers",

    denominator=
        len(
            seller_resolved_rows
        ),

    missing=
        seller_geo_missing_exposure,

    missing_reason=
        "RELATION_MISSING",

    ml_relevance=
        "geographic candidate features",

    severity=
        "WARNING"
)


add_conditional_row(
    scope=
        "TASK_UNIQUE_SELLER",

    entity=
        "seller",

    attribute=
        "geolocation_reference",

    condition=
        "seller referenced by at least one task item",

    denominator=
        len(
            unique_task_sellers
        ),

    missing=
        unique_seller_geo_missing,

    missing_reason=
        "RELATION_MISSING",

    ml_relevance=
        "entity-level diagnostic",

    severity=
        "WARNING"
)


# =============================================================================
# 8. PAYMENT CONDITIONAL COMPLETENESS
# =============================================================================

print()
print("=" * 104)
print("8. PAYMENT ATTRIBUTES — CONDITIONED ON PAYMENT EXISTENCE")
print("=" * 104)


payment_fields = [
    "payment_type",
    "payment_installments",
    "payment_value"
]


payment_attribute_missing = int(
    task_payments[
        payment_fields
    ]
    .isna()
    .any(
        axis=1
    )
    .sum()
)


add_conditional_row(
    scope=
        "TASK_PAYMENT_RECORD",

    entity=
        "payment",

    attribute=
        "payment_core_fields",

    condition=
        "payment relation exists",

    denominator=
        len(
            task_payments
        ),

    missing=
        payment_attribute_missing,

    missing_reason=
        "ATTRIBUTE_MISSING",

    ml_relevance=
        (
            "candidate features; point-in-time availability "
            "still unresolved"
        ),

    severity=
        "WARNING"
)


# =============================================================================
# 9. SAVE CONDITIONAL COMPLETENESS
# =============================================================================

conditional_df = pd.DataFrame(
    conditional_rows
)


conditional_df.to_csv(
    OUT
    /
    "02_conditional_attribute_completeness.csv",
    index=False
)


print(
    conditional_df.to_string(
        index=False,
        formatters={
            "missing_pct":
                "{:.6f}".format,

            "completeness_pct":
                "{:.6f}".format
        }
    )
)


# Transfer conditional checks into Gate scorecard.

for i, row in conditional_df.iterrows():

    add_result(
        check_id=
            f"TASK-ATTR-{i+1:03d}",

        dimension=
            "conditional_attribute_completeness",

        scope=
            row[
                "scope"
            ],

        entity=
            (
                f"{row['entity']}."
                f"{row['attribute']}"
            ),

        condition=
            row[
                "applicability_condition"
            ],

        severity=
            row[
                "severity"
            ],

        status=
            row[
                "status"
            ],

        numerator=
            row[
                "missing"
            ],

        denominator=
            row[
                "denominator"
            ],

        expected=
            (
                "0 missing for CRITICAL; "
                "otherwise document observed missingness"
            ),

        details=
            (
                f"reason={row['missing_reason']} | "
                f"{row['ml_relevance']}"
            )
    )


# =============================================================================
# 10. POST-PURCHASE EVENTS — INFORMATIONAL
# =============================================================================

print()
print("=" * 104)
print("9. POST-PURCHASE EVENTS — INFORMATIONAL ONLY")
print("=" * 104)


task_order_raw = orders[
    orders[
        "order_id"
    ]
    .isin(
        task_ids
    )
].copy()


approval_missing_task = int(
    task_order_raw[
        "order_approved_at"
    ]
    .isna()
    .sum()
)


carrier_missing_task = int(
    task_order_raw[
        "order_delivered_carrier_date"
    ]
    .isna()
    .sum()
)


post_event_rows = pd.DataFrame(
    [
        {
            "event":
                "order_approved_at",

            "task_orders":
                n_task,

            "missing":
                approval_missing_task,

            "missing_pct":
                pct(
                    approval_missing_task,
                    n_task
                ),

            "feature_policy":
                "FORBIDDEN_AT_T0",

            "interpretation":
                (
                    "Informational process completeness; "
                    "not baseline feature completeness."
                )
        },

        {
            "event":
                "order_delivered_carrier_date",

            "task_orders":
                n_task,

            "missing":
                carrier_missing_task,

            "missing_pct":
                pct(
                    carrier_missing_task,
                    n_task
                ),

            "feature_policy":
                "FORBIDDEN_AT_T0",

            "interpretation":
                (
                    "Informational process completeness; "
                    "not baseline feature completeness."
                )
        }
    ]
)


post_event_rows.to_csv(
    OUT
    /
    "03_post_purchase_event_completeness.csv",
    index=False
)


print(
    post_event_rows.to_string(
        index=False,
        formatters={
            "missing_pct":
                "{:.6f}".format
        }
    )
)


# =============================================================================
# 11. BUILD ORDER-LEVEL READINESS FLAGS
# =============================================================================

print()
print("=" * 104)
print("10. TASK ORDER FEATURE READINESS")
print("=" * 104)


readiness = task[
    [
        "order_id",
        "customer_id",
        "purchase_month"
    ]
].copy()


# -------------------------------------------------------------------------
# ITEMS
# -------------------------------------------------------------------------

item_order = (
    task_items
    .assign(
        item_core_complete=(
            task_items[
                [
                    "product_id",
                    "seller_id",
                    "price",
                    "freight_value"
                ]
            ]
            .notna()
            .all(
                axis=1
            )
        )
    )
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        item_count=(
            "order_item_id",
            "count"
        ),

        product_count=(
            "product_id",
            "nunique"
        ),

        seller_count=(
            "seller_id",
            "nunique"
        ),

        all_item_core_complete=(
            "item_core_complete",
            "all"
        )
    )
)


readiness = readiness.merge(
    item_order,
    on=
        "order_id",

    how=
        "left",

    validate=
        "one_to_one"
)


readiness[
    "has_items"
] = (
    readiness[
        "item_count"
    ]
    .notna()
)


# -------------------------------------------------------------------------
# PRODUCT ORDER FLAGS
# -------------------------------------------------------------------------

if len(
    resolved_product_rows
):

    product_order_flags = (
        resolved_product_rows
        .groupby(
            "order_id",
            as_index=False
        )
        .agg(
            all_products_resolved=(
                "product_resolved",
                "all"
            ),

            any_product_category_missing=(
                "category_missing",
                "any"
            ),

            any_product_physical_missing=(
                "physical_missing",
                "any"
            )
        )
    )

else:

    product_order_flags = pd.DataFrame(
        columns=[
            "order_id",
            "all_products_resolved",
            "any_product_category_missing",
            "any_product_physical_missing"
        ]
    )


readiness = readiness.merge(
    product_order_flags,
    on=
        "order_id",

    how=
        "left",

    validate=
        "one_to_one"
)


# Important:
# for no-item orders these are NOT attribute missing;
# the upstream relationship itself is absent.

readiness[
    "product_metadata_applicable"
] = readiness[
    "has_items"
]


readiness[
    "product_metadata_complete"
] = (
    readiness[
        "has_items"
    ]
    &
    readiness[
        "all_products_resolved"
    ]
    .fillna(
        False
    )
    &
    ~readiness[
        "any_product_category_missing"
    ]
    .fillna(
        False
    )
    &
    ~readiness[
        "any_product_physical_missing"
    ]
    .fillna(
        False
    )
)


# -------------------------------------------------------------------------
# SELLER ORDER FLAGS
# -------------------------------------------------------------------------

seller_order_flags = (
    seller_resolved_rows
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        all_sellers_resolved=(
            "seller_resolved",
            "all"
        ),

        any_seller_geo_missing=(
            "seller_geo_missing",
            "any"
        )
    )
)


readiness = readiness.merge(
    seller_order_flags,
    on=
        "order_id",

    how=
        "left",

    validate=
        "one_to_one"
)


readiness[
    "seller_reference_complete"
] = (
    readiness[
        "has_items"
    ]
    &
    readiness[
        "all_sellers_resolved"
    ]
    .fillna(
        False
    )
)


readiness[
    "seller_geo_reference_complete"
] = (
    readiness[
        "seller_reference_complete"
    ]
    &
    ~readiness[
        "any_seller_geo_missing"
    ]
    .fillna(
        False
    )
)


# -------------------------------------------------------------------------
# CUSTOMER FLAGS
# -------------------------------------------------------------------------

customer_flags = task_customer[
    [
        "order_id",
        "customer_resolved",
        "customer_geo_missing"
    ]
].copy()


readiness = readiness.merge(
    customer_flags,
    on=
        "order_id",

    how=
        "left",

    validate=
        "one_to_one"
)


readiness[
    "customer_geo_reference_complete"
] = (
    readiness[
        "customer_resolved"
    ]
    .fillna(
        False
    )
    &
    ~readiness[
        "customer_geo_missing"
    ]
    .fillna(
        True
    )
)


# -------------------------------------------------------------------------
# PAYMENT FLAGS
# -------------------------------------------------------------------------

payment_order_flags = (
    task_payments
    .assign(
        payment_fields_complete=(
            task_payments[
                payment_fields
            ]
            .notna()
            .all(
                axis=1
            )
        )
    )
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        payment_record_count=(
            "payment_sequential",
            "count"
        ),

        payment_type_count=(
            "payment_type",
            "nunique"
        ),

        all_payment_fields_complete=(
            "payment_fields_complete",
            "all"
        )
    )
)


readiness = readiness.merge(
    payment_order_flags,
    on=
        "order_id",

    how=
        "left",

    validate=
        "one_to_one"
)


readiness[
    "has_payment"
] = (
    readiness[
        "payment_record_count"
    ]
    .notna()
)


readiness[
    "payment_data_complete"
] = (
    readiness[
        "has_payment"
    ]
    &
    readiness[
        "all_payment_fields_complete"
    ]
    .fillna(
        False
    )
)


# -------------------------------------------------------------------------
# CORE READINESS
# -------------------------------------------------------------------------

readiness[
    "core_purchase_context_complete"
] = (
    readiness[
        "has_items"
    ]
    &
    readiness[
        "all_item_core_complete"
    ]
    .fillna(
        False
    )
    &
    readiness[
        "customer_resolved"
    ]
    .fillna(
        False
    )
)


readiness[
    "geo_reference_complete"
] = (
    readiness[
        "customer_geo_reference_complete"
    ]
    &
    readiness[
        "seller_geo_reference_complete"
    ]
)


readiness.to_csv(
    OUT
    /
    "04_task_order_feature_readiness.csv",
    index=False
)


# =============================================================================
# 12. READINESS SUMMARY
# =============================================================================

flag_cols = [

    "has_items",
    "all_item_core_complete",
    "customer_resolved",
    "core_purchase_context_complete",

    "product_metadata_complete",
    "seller_reference_complete",
    "payment_data_complete",
    "customer_geo_reference_complete",
    "seller_geo_reference_complete",
    "geo_reference_complete"
]


readiness_rows = []


for col in flag_cols:

    true_count = int(
        readiness[
            col
        ]
        .fillna(
            False
        )
        .sum()
    )

    readiness_rows.append(
        {
            "readiness_dimension":
                col,

            "task_orders":
                n_task,

            "ready_orders":
                true_count,

            "not_ready_orders":
                n_task
                -
                true_count,

            "ready_pct":
                pct(
                    true_count,
                    n_task
                )
        }
    )


readiness_summary = pd.DataFrame(
    readiness_rows
)


readiness_summary.to_csv(
    OUT
    /
    "05_task_feature_readiness_summary.csv",
    index=False
)


print(
    readiness_summary.to_string(
        index=False,
        formatters={
            "ready_pct":
                "{:.6f}".format
        }
    )
)


# =============================================================================
# 13. MONTHLY TASK COMPLETENESS
# =============================================================================

print()
print("=" * 104)
print("11. TASK COMPLETENESS BY MONTH")
print("=" * 104)


monthly_rows = []


for month, sub in readiness.groupby(
    "purchase_month"
):

    r = {
        "purchase_month":
            month,

        "task_orders":
            len(
                sub
            )
    }

    for col in flag_cols:

        ready = int(
            sub[
                col
            ]
            .fillna(
                False
            )
            .sum()
        )

        r[
            f"{col}_pct"
        ] = pct(
            ready,
            len(
                sub
            )
        )

    monthly_rows.append(
        r
    )


monthly = pd.DataFrame(
    monthly_rows
).sort_values(
    "purchase_month"
)


monthly.to_csv(
    OUT
    /
    "06_task_completeness_by_month.csv",
    index=False
)


print(
    monthly.to_string(
        index=False,
        formatters={
            c:
                "{:.4f}".format
            for c
            in monthly.columns
            if c.endswith(
                "_pct"
            )
        }
    )
)


# =============================================================================
# 14. MISSING-REASON TAXONOMY
# =============================================================================

print()
print("=" * 104)
print("12. MISSING REASON TAXONOMY")
print("=" * 104)


reason_rows = []


def reason(
    component,
    scope,
    state,
    count,
    denominator,
    condition,
    note=""
):

    reason_rows.append(
        {
            "component":
                component,

            "scope":
                scope,

            "state":
                state,

            "count":
                int(
                    count
                ),

            "denominator":
                int(
                    denominator
                ),

            "pct":
                pct(
                    count,
                    denominator
                ),

            "condition":
                condition,

            "note":
                note
        }
    )


reason(
    "order_items",
    "TASK_ORDER",
    "RELATION_MISSING",
    task_orders_without_items,
    n_task,
    "task order exists",
    "No child order_item rows."
)


reason(
    "payments",
    "TASK_ORDER",
    "RELATION_MISSING",
    task_orders_without_payment,
    n_task,
    "task order exists",
    "No child payment rows."
)


reason(
    "customer",
    "TASK_ORDER",
    "RELATION_MISSING",
    task_orders_without_customer,
    n_task,
    "task order exists",
    "customer_id does not resolve."
)


reason(
    "customer_geolocation",
    "TASK_ORDER",
    "RELATION_MISSING",
    customer_geo_missing,
    len(
        customer_resolved_rows
    ),
    "customer_id resolves",
    "Customer exists, but ZIP has no geolocation reference."
)


reason(
    "product",
    "TASK_ITEM",
    "RELATION_MISSING",
    task_product_orphans,
    len(
        task_items
    ),
    "order item exists",
    "product_id does not resolve."
)


reason(
    "product_category",
    "TASK_ITEM",
    "ATTRIBUTE_MISSING",
    category_missing_exposure,
    len(
        resolved_product_rows
    ),
    "product_id resolves",
    "Entity exists; category attribute is NULL."
)


reason(
    "product_physical",
    "TASK_ITEM",
    "ATTRIBUTE_MISSING",
    physical_missing_exposure,
    len(
        resolved_product_rows
    ),
    "product_id resolves",
    "Entity exists; one or more physical attributes are NULL."
)


reason(
    "seller",
    "TASK_ITEM",
    "RELATION_MISSING",
    task_seller_orphans,
    len(
        task_items
    ),
    "order item exists",
    "seller_id does not resolve."
)


reason(
    "seller_geolocation",
    "TASK_ITEM",
    "RELATION_MISSING",
    seller_geo_missing_exposure,
    len(
        seller_resolved_rows
    ),
    "seller_id resolves",
    "Seller exists, but ZIP has no geolocation reference."
)


reason(
    "order_approved_at",
    "TASK_ORDER",
    "ATTRIBUTE_MISSING",
    approval_missing_task,
    n_task,
    "task order exists",
    "Post-purchase field; forbidden as baseline t0 feature."
)


reason(
    "order_delivered_carrier_date",
    "TASK_ORDER",
    "ATTRIBUTE_MISSING",
    carrier_missing_task,
    n_task,
    "task order exists",
    "Future process field; forbidden as baseline t0 feature."
)


reason_df = pd.DataFrame(
    reason_rows
)


reason_df.to_csv(
    OUT
    /
    "07_missing_reason_registry.csv",
    index=False
)


print(
    reason_df.to_string(
        index=False,
        formatters={
            "pct":
                "{:.6f}".format
        }
    )
)


# =============================================================================
# 15. SOURCE VS TASK
# =============================================================================

print()
print("=" * 104)
print("13. SOURCE QUALITY vs TASK QUALITY")
print("=" * 104)


source_vs_task = pd.DataFrame(
    [
        {
            "metric":
                "orders_without_items",

            "source_denominator":
                len(
                    orders
                ),

            "source_affected":
                src_orders_without_items,

            "source_affected_pct":
                pct(
                    src_orders_without_items,
                    len(
                        orders
                    )
                ),

            "task_denominator":
                n_task,

            "task_affected":
                task_orders_without_items,

            "task_affected_pct":
                pct(
                    task_orders_without_items,
                    n_task
                )
        },

        {
            "metric":
                "orders_without_payments",

            "source_denominator":
                len(
                    orders
                ),

            "source_affected":
                src_orders_without_payments,

            "source_affected_pct":
                pct(
                    src_orders_without_payments,
                    len(
                        orders
                    )
                ),

            "task_denominator":
                n_task,

            "task_affected":
                task_orders_without_payment,

            "task_affected_pct":
                pct(
                    task_orders_without_payment,
                    n_task
                )
        },

        {
            "metric":
                "orders_without_customer_resolution",

            "source_denominator":
                len(
                    orders
                ),

            "source_affected":
                src_orders_without_customer,

            "source_affected_pct":
                pct(
                    src_orders_without_customer,
                    len(
                        orders
                    )
                ),

            "task_denominator":
                n_task,

            "task_affected":
                task_orders_without_customer,

            "task_affected_pct":
                pct(
                    task_orders_without_customer,
                    n_task
                )
        },

        {
            "metric":
                "item_product_orphans",

            "source_denominator":
                len(
                    items
                ),

            "source_affected":
                src_item_product_orphans,

            "source_affected_pct":
                pct(
                    src_item_product_orphans,
                    len(
                        items
                    )
                ),

            "task_denominator":
                len(
                    task_items
                ),

            "task_affected":
                task_product_orphans,

            "task_affected_pct":
                pct(
                    task_product_orphans,
                    len(
                        task_items
                    )
                )
        },

        {
            "metric":
                "item_seller_orphans",

            "source_denominator":
                len(
                    items
                ),

            "source_affected":
                src_item_seller_orphans,

            "source_affected_pct":
                pct(
                    src_item_seller_orphans,
                    len(
                        items
                    )
                ),

            "task_denominator":
                len(
                    task_items
                ),

            "task_affected":
                task_seller_orphans,

            "task_affected_pct":
                pct(
                    task_seller_orphans,
                    len(
                        task_items
                    )
                )
        }
    ]
)


source_vs_task[
    "difference_task_minus_source_pp"
] = (
    source_vs_task[
        "task_affected_pct"
    ]
    -
    source_vs_task[
        "source_affected_pct"
    ]
)


source_vs_task.to_csv(
    OUT
    /
    "08_source_vs_task_completeness.csv",
    index=False
)


print(
    source_vs_task.to_string(
        index=False,
        formatters={
            "source_affected_pct":
                "{:.6f}".format,

            "task_affected_pct":
                "{:.6f}".format,

            "difference_task_minus_source_pp":
                "{:.6f}".format
        }
    )
)


# =============================================================================
# 16. MULTI-ENTITY STRUCTURE
# =============================================================================

print()
print("=" * 104)
print("14. MULTI-ENTITY STRUCTURE")
print("=" * 104)


multi = readiness[
    [
        "order_id",
        "item_count",
        "product_count",
        "seller_count",
        "payment_record_count",
        "payment_type_count"
    ]
].copy()


multi_summary = pd.DataFrame(
    [
        {
            "metric":
                "orders_multi_item",

            "orders":
                int(
                    (
                        multi[
                            "item_count"
                        ]
                        >
                        1
                    )
                    .sum()
                ),

            "pct":
                pct(
                    int(
                        (
                            multi[
                                "item_count"
                            ]
                            >
                            1
                        )
                        .sum()
                    ),
                    n_task
                ),

            "maximum":
                float(
                    multi[
                        "item_count"
                    ]
                    .max()
                )
        },

        {
            "metric":
                "orders_multi_product",

            "orders":
                int(
                    (
                        multi[
                            "product_count"
                        ]
                        >
                        1
                    )
                    .sum()
                ),

            "pct":
                pct(
                    int(
                        (
                            multi[
                                "product_count"
                            ]
                            >
                            1
                        )
                        .sum()
                    ),
                    n_task
                ),

            "maximum":
                float(
                    multi[
                        "product_count"
                    ]
                    .max()
                )
        },

        {
            "metric":
                "orders_multi_seller",

            "orders":
                int(
                    (
                        multi[
                            "seller_count"
                        ]
                        >
                        1
                    )
                    .sum()
                ),

            "pct":
                pct(
                    int(
                        (
                            multi[
                                "seller_count"
                            ]
                            >
                            1
                        )
                        .sum()
                    ),
                    n_task
                ),

            "maximum":
                float(
                    multi[
                        "seller_count"
                    ]
                    .max()
                )
        },

        {
            "metric":
                "orders_multi_payment_record",

            "orders":
                int(
                    (
                        multi[
                            "payment_record_count"
                        ]
                        >
                        1
                    )
                    .sum()
                ),

            "pct":
                pct(
                    int(
                        (
                            multi[
                                "payment_record_count"
                            ]
                            >
                            1
                        )
                        .sum()
                    ),
                    n_task
                ),

            "maximum":
                float(
                    multi[
                        "payment_record_count"
                    ]
                    .max()
                )
        },

        {
            "metric":
                "orders_multi_payment_type",

            "orders":
                int(
                    (
                        multi[
                            "payment_type_count"
                        ]
                        >
                        1
                    )
                    .sum()
                ),

            "pct":
                pct(
                    int(
                        (
                            multi[
                                "payment_type_count"
                            ]
                            >
                            1
                        )
                        .sum()
                    ),
                    n_task
                ),

            "maximum":
                float(
                    multi[
                        "payment_type_count"
                    ]
                    .max()
                )
        }
    ]
)


multi_summary.to_csv(
    OUT
    /
    "09_multi_entity_structure.csv",
    index=False
)


print(
    multi_summary.to_string(
        index=False,
        formatters={
            "pct":
                "{:.6f}".format
        }
    )
)


# =============================================================================
# 17. COHORT FUNNEL
# =============================================================================

funnel = pd.DataFrame(
    [
        {
            "stage":
                "SOURCE_ORDERS",

            "orders":
                len(
                    orders
                )
        },

        {
            "stage":
                "STATUS_DELIVERED",

            "orders":
                int(
                    orders[
                        "order_status"
                    ]
                    .eq(
                        "delivered"
                    )
                    .sum()
                )
        },

        {
            "stage":
                "DELIVERY_OUTCOME_OBSERVED",

            "orders":
                int(
                    (
                        orders[
                            "order_delivered_customer_date"
                        ]
                        .notna()
                    )
                    .sum()
                )
        },

        {
            "stage":
                "TASK_CANDIDATE_OBSERVABLE",

            "orders":
                n_task
        },

        {
            "stage":
                "TASK_WITH_ITEMS",

            "orders":
                n_task
                -
                task_orders_without_items
        },

        {
            "stage":
                "TASK_CORE_PURCHASE_CONTEXT_COMPLETE",

            "orders":
                int(
                    readiness[
                        "core_purchase_context_complete"
                    ]
                    .sum()
                )
        }
    ]
)


funnel[
    "pct_source"
] = (
    100
    *
    funnel[
        "orders"
    ]
    /
    len(
        orders
    )
)


funnel.to_csv(
    OUT
    /
    "10_task_cohort_funnel.csv",
    index=False
)


# =============================================================================
# 18. SCORECARD
# =============================================================================

scorecard = pd.DataFrame(
    results
)


scorecard.to_csv(
    OUT
    /
    "dq_gate_03b_scorecard.csv",
    index=False
)


exceptions = scorecard[
    scorecard[
        "status"
    ]
    .isin(
        [
            "FAIL",
            "WARN"
        ]
    )
].copy()


exceptions.to_csv(
    OUT
    /
    "dq_gate_03b_exceptions.csv",
    index=False
)


critical_failures = int(
    (
        (
            scorecard[
                "severity"
            ]
            ==
            "CRITICAL"
        )
        &
        (
            scorecard[
                "status"
            ]
            ==
            "FAIL"
        )
    )
    .sum()
)


warnings = int(
    (
        scorecard[
            "status"
        ]
        ==
        "WARN"
    )
    .sum()
)


passes = int(
    (
        scorecard[
            "status"
        ]
        ==
        "PASS"
    )
    .sum()
)


gate_status = (
    "PASS"
    if critical_failures == 0
    else
    "FAIL"
)


# =============================================================================
# 19. ISSUE EVIDENCE
# =============================================================================

issue_evidence = pd.DataFrame(
    [
        {
            "canonical_issue_id":
                "ISSUE-MISSINGNESS-001",

            "gate":
                "DQ_GATE_03B",

            "evidence_status":
                (
                    "READY_FOR_REVIEW"
                    if gate_status == "PASS"
                    else
                    "BLOCKED"
                ),

            "evidence":
                (
                    "Relationship missingness and attribute missingness "
                    "have been measured separately at source and task scopes."
                ),

            "recommended_next_action":
                (
                    "Review Gate 03B outputs and decide whether "
                    "ISSUE-MISSINGNESS-001 can be closed."
                )
        }
    ]
)


issue_evidence.to_csv(
    OUT
    /
    "11_issue_evidence_gate_03b.csv",
    index=False
)


# =============================================================================
# 20. SUMMARY JSON
# =============================================================================

summary = {

    "gate":
        "DQ_GATE_03B_CONDITIONAL_TASK_COMPLETENESS",

    "status":
        gate_status,

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_orders":
        int(
            len(
                orders
            )
        ),

    "task_candidate_orders":
        int(
            n_task
        ),

    "task_candidate_pct":
        pct(
            n_task,
            len(
                orders
            )
        ),

    "task_orders_without_items":
        task_orders_without_items,

    "task_orders_without_payment":
        task_orders_without_payment,

    "task_product_orphans":
        task_product_orphans,

    "task_seller_orphans":
        task_seller_orphans,

    "conditional_product_category_missing_item_exposure":
        category_missing_exposure,

    "conditional_product_physical_missing_item_exposure":
        physical_missing_exposure,

    "customer_geo_missing_task_orders":
        customer_geo_missing,

    "seller_geo_missing_task_item_exposure":
        seller_geo_missing_exposure,

    "critical_failures":
        critical_failures,

    "warnings":
        warnings,

    "passes":
        passes,

    "raw_modified":
        False,

    "target_finalized":
        False,

    "silver_created":
        False
}


with (
    OUT
    /
    "dq_gate_03b_summary.json"
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
# 21. REPORT
# =============================================================================

report = []


report.append(
    "=" * 104
)

report.append(
    "DQ GATE 03B — CONDITIONAL & TASK COMPLETENESS"
)

report.append(
    "=" * 104
)

report.append("")

report.append(
    f"STATUS                    : {gate_status}"
)

report.append(
    f"SOURCE ORDERS             : {len(orders):,}"
)

report.append(
    f"TASK CANDIDATE ORDERS     : {n_task:,}"
)

report.append(
    f"TASK / SOURCE             : {pct(n_task, len(orders)):.4f}%"
)

report.append(
    f"PASS                      : {passes}"
)

report.append(
    f"WARNINGS                  : {warnings}"
)

report.append(
    f"CRITICAL FAILURES         : {critical_failures}"
)

report.append("")

report.append(
    "PRINCIPAL DISTINÇÃO METODOLÓGICA"
)

report.append(
    "-" * 104
)

report.append(
    "RELATION_MISSING != ATTRIBUTE_MISSING"
)

report.append(
    "SOURCE_QUALITY    != TASK_QUALITY"
)

report.append(
    "NOT_APPLICABLE    != MISSING"
)

report.append(
    "NOT_YET_OBSERVED  != NEGATIVE LABEL"
)

report.append("")

report.append(
    "CHECKS NÃO-PASS"
)

report.append(
    "-" * 104
)


if exceptions.empty:

    report.append(
        "Nenhum."
    )

else:

    for _, row in exceptions.iterrows():

        report.append(
            f"[{row['status']}] "
            f"{row['check_id']} | "
            f"{row['entity']} | "
            f"scope={row['scope']} | "
            f"condition={row['applicability_condition']} | "
            f"affected={row['affected']}/{row['denominator']} "
            f"({row['affected_pct']:.6f}%)"
        )


report.append("")

report.append(
    "IMPORTANTE"
)

report.append(
    "-" * 104
)

report.append(
    "Nenhum dado foi imputado."
)

report.append(
    "Nenhum registro foi removido."
)

report.append(
    "Nenhuma decisão final sobre target foi tomada."
)

report.append(
    "Nenhuma Silver foi criada."
)

report.append(
    "RAW permaneceu imutável."
)

report.append("")

report.append(
    "Próxima etapa após revisão:"
)

report.append(
    "DQ Gate 04 — Label / Construct Validity."
)

report.append(
    "=" * 104
)


report_path = (
    OUT
    /
    "DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt"
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
print("=" * 104)
print("DQ GATE 03B — RESULTADO")
print("=" * 104)

print(
    f"STATUS                    : {gate_status}"
)

print(
    f"SOURCE ORDERS             : {len(orders):,}"
)

print(
    f"TASK CANDIDATE ORDERS     : {n_task:,}"
)

print(
    f"TASK / SOURCE             : {pct(n_task, len(orders)):.4f}%"
)

print(
    f"PASS                      : {passes}"
)

print(
    f"WARNINGS                  : {warnings}"
)

print(
    f"CRITICAL FAILURES         : {critical_failures}"
)

print()

if not exceptions.empty:

    print(
        "CHECKS NÃO-PASS:"
    )

    print(
        exceptions[
            [
                "status",
                "check_id",
                "scope",
                "entity",
                "affected",
                "denominator",
                "affected_pct"
            ]
        ]
        .to_string(
            index=False
        )
    )


print()
print("Arquivos gerados:")


for p in sorted(
    OUT.iterdir()
):

    if p.is_file():

        print(
            f"  - {p}"
        )


print()

print(
    "[OK] RAW NÃO MODIFICADO."
)

print(
    "[OK] Nenhuma imputação ou exclusão aplicada."
)


if gate_status == "PASS":

    print()

    print(
        "[PASS] DQ GATE 03B APROVADO."
    )

    print(
        "Conditional Completeness e Task Completeness "
        "foram medidos separadamente."
    )

    print(
        "Próxima etapa após revisão: "
        "DQ Gate 04 — Label / Construct Validity."
    )

    sys.exit(0)

else:

    print()

    print(
        "[FAIL] DQ GATE 03B REPROVADO."
    )

    print(
        "Gate 04 permanece bloqueado."
    )

    sys.exit(2)


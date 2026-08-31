#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DQ GATE 02 — SEMANTIC / VALIDITY
Delivery Risk Intelligence Platform

Pré-requisito:
    DQ Gate 01 — Structural = PASS

Valida:
    1. domínios categóricos
    2. UFs brasileiras
    3. ranges numéricos
    4. coordenadas geográficas
    5. review_score
    6. parseabilidade de timestamps
    7. consistência temporal
    8. lead times semanticamente possíveis
    9. categorias sem tradução
   10. CEP-prefixos associados a múltiplos estados
   11. zeros semanticamente suspeitos

NÃO valida neste gate:
    - missingness analítica
    - distribuições / outliers
    - drift
    - representatividade
    - qualidade do label
    - imputação
    - modelagem

Regras:
    CRITICAL FAIL -> bloqueia próximo gate.
    WARNING       -> documentado, não bloqueia.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

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
    / "dq_gate_02_semantic.json"
)

GATE1_SUMMARY = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_01_structural"
    / "dq_gate_01_summary.json"
)

OUT = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_02_semantic"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# PRÉ-REQUISITO — GATE 1
# ======================================================================

if not GATE1_SUMMARY.exists():

    print(
        "[ERRO] Resultado do DQ Gate 01 não encontrado:"
    )

    print(
        GATE1_SUMMARY
    )

    print(
        "Rode e aprove o Gate 01 antes do Gate 02."
    )

    sys.exit(2)


with GATE1_SUMMARY.open(
    "r",
    encoding="utf-8"
) as f:

    gate1 = json.load(f)


if gate1.get("status") != "PASS":

    print(
        "[BLOQUEADO] DQ Gate 01 não está aprovado."
    )

    print(
        f"Status Gate 01: {gate1.get('status')}"
    )

    sys.exit(2)


# ======================================================================
# CONTRATO
# ======================================================================

with CONFIG.open(
    "r",
    encoding="utf-8"
) as f:

    CONTRACT = json.load(f)


UF = set(
    CONTRACT[
        "allowed_brazilian_states"
    ]
)

ORDER_STATUS = set(
    CONTRACT[
        "allowed_order_status"
    ]
)

PAYMENT_TYPE = set(
    CONTRACT[
        "allowed_payment_type"
    ]
)


# ======================================================================
# CARREGAMENTO
# ======================================================================

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


dfs = {}


for name, filename in FILES.items():

    path = RAW / filename

    if not path.exists():

        print(
            f"[ERRO] Arquivo ausente: {path}"
        )

        sys.exit(2)

    dfs[name] = pd.read_csv(
        path,
        low_memory=False
    )


customers = dfs["customers"]
geo = dfs["geolocation"]
items = dfs["order_items"]
payments = dfs["payments"]
reviews = dfs["reviews"]
orders = dfs["orders"]
products = dfs["products"]
sellers = dfs["sellers"]
translation = dfs["translation"]


# ======================================================================
# RESULTADOS
# ======================================================================

results = []

exception_records = []


def add_result(
    check_id,
    table,
    variable,
    dimension,
    severity,
    status,
    observed,
    expected,
    affected_rows=0,
    affected_pct=0.0,
    details=""
):

    results.append(
        {
            "gate":
                "DQ_GATE_02_SEMANTIC",

            "check_id":
                check_id,

            "table":
                table,

            "variable":
                variable,

            "dimension":
                dimension,

            "severity":
                severity,

            "status":
                status,

            "affected_rows":
                int(affected_rows),

            "affected_pct":
                float(affected_pct),

            "observed":
                str(observed),

            "expected":
                str(expected),

            "details":
                str(details),
        }
    )


def percentage(
    n,
    total
):

    if total == 0:
        return 0.0

    return (
        100.0
        *
        n
        /
        total
    )


def result_from_count(
    check_id,
    table,
    variable,
    dimension,
    violations,
    total,
    expected,
    severity="CRITICAL",
    observed=None,
    details=""
):

    violations = int(
        violations
    )

    if observed is None:

        observed = (
            f"{violations} violation(s)"
        )

    if violations == 0:

        status = "PASS"

    elif severity == "CRITICAL":

        status = "FAIL"

    else:

        status = "WARN"

    add_result(
        check_id=check_id,
        table=table,
        variable=variable,
        dimension=dimension,
        severity=severity,
        status=status,
        observed=observed,
        expected=expected,
        affected_rows=violations,
        affected_pct=percentage(
            violations,
            total
        ),
        details=details,
    )


# ======================================================================
# HEADER
# ======================================================================

print("=" * 96)

print(
    "DQ GATE 02 — SEMANTIC / VALIDITY"
)

print(
    "Delivery Risk Intelligence Platform"
)

print("=" * 96)

print()

print(
    "[PASS] Pré-requisito: "
    "DQ Gate 01 — Structural aprovado."
)

print(
    f"RAW      : {RAW}"
)

print(
    f"CONTRACT : {CONFIG}"
)

print(
    f"OUTPUT   : {OUT}"
)


# ======================================================================
# 1. DOMÍNIOS CATEGÓRICOS
# ======================================================================

print()
print("=" * 96)
print("1. DOMÍNIOS CATEGÓRICOS")
print("=" * 96)


def check_allowed_set(
    df,
    table,
    column,
    allowed,
    check_id,
    severity="CRITICAL"
):

    s = (
        df[column]
        .dropna()
        .astype(str)
    )

    invalid_mask = (
        ~s.isin(
            allowed
        )
    )

    invalid = sorted(
        s[
            invalid_mask
        ]
        .unique()
        .tolist()
    )

    count = int(
        invalid_mask.sum()
    )

    result_from_count(
        check_id=check_id,
        table=table,
        variable=column,
        dimension="domain_validity",
        violations=count,
        total=len(s),
        expected=sorted(allowed),
        severity=severity,
        observed=invalid if invalid else "all valid",
        details=(
            "Nulls não são avaliados neste gate."
        ),
    )

    print(
        f"[{'PASS' if count == 0 else 'FAIL'}] "
        f"{table}.{column} | "
        f"invalid={count:,}"
    )

    if count > 0:

        for value in invalid:

            exception_records.append(
                {
                    "check_id":
                        check_id,

                    "table":
                        table,

                    "variable":
                        column,

                    "exception_value":
                        value,

                    "exception_type":
                        "OUTSIDE_ALLOWED_SET",
                }
            )


check_allowed_set(
    orders,
    "orders",
    "order_status",
    ORDER_STATUS,
    "SEM-001"
)


check_allowed_set(
    payments,
    "payments",
    "payment_type",
    PAYMENT_TYPE,
    "SEM-002"
)


check_allowed_set(
    customers,
    "customers",
    "customer_state",
    UF,
    "SEM-003"
)


check_allowed_set(
    sellers,
    "sellers",
    "seller_state",
    UF,
    "SEM-004"
)


check_allowed_set(
    geo,
    "geolocation",
    "geolocation_state",
    UF,
    "SEM-005"
)


# ======================================================================
# 2. RANGES NUMÉRICOS
# ======================================================================

print()
print("=" * 96)
print("2. RANGES NUMÉRICOS")
print("=" * 96)


TABLE_LOOKUP = {
    "order_items":
        items,

    "payments":
        payments,

    "reviews":
        reviews,

    "products":
        products,

    "geolocation":
        geo,
}


numeric_counter = 10


for key, rule in CONTRACT[
    "numeric_rules"
].items():

    numeric_counter += 1

    table_name, column = (
        key.split(
            ".",
            1
        )
    )

    df = TABLE_LOOKUP[
        table_name
    ]

    x = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    observed_non_null = (
        df[column]
        .notna()
    )

    x_non_null = x[
        observed_non_null
    ]

    violation = pd.Series(
        False,
        index=x_non_null.index
    )


    if "min" in rule:

        if rule.get(
            "min_inclusive",
            True
        ):

            violation |= (
                x_non_null
                <
                rule["min"]
            )

        else:

            violation |= (
                x_non_null
                <=
                rule["min"]
            )


    if "max" in rule:

        if rule.get(
            "max_inclusive",
            True
        ):

            violation |= (
                x_non_null
                >
                rule["max"]
            )

        else:

            violation |= (
                x_non_null
                >=
                rule["max"]
            )


    violation_count = int(
        violation.sum()
    )


    bounds = {
        k:
            rule[k]
        for k in [
            "min",
            "max",
            "min_inclusive",
            "max_inclusive"
        ]
        if k in rule
    }


    result_from_count(
        check_id=
            f"SEM-{numeric_counter:03d}",

        table=
            table_name,

        variable=
            column,

        dimension=
            "numeric_validity",

        violations=
            violation_count,

        total=
            len(x_non_null),

        expected=
            bounds,

        severity=
            rule.get(
                "severity",
                "CRITICAL"
            ),

        observed=(
            "all valid"
            if violation_count == 0
            else
            (
                f"min={x_non_null.min()}, "
                f"max={x_non_null.max()}"
            )
        ),

        details=(
            "Missing values são tratados "
            "posteriormente no DQ Gate de completude."
        ),
    )


    status_text = (
        "PASS"
        if violation_count == 0
        else
        "FAIL"
    )


    print(
        f"[{status_text}] "
        f"{table_name}.{column} | "
        f"violations={violation_count:,}"
    )


    if violation_count > 0:

        bad = df.loc[
            violation.index[
                violation
            ]
        ].copy()

        bad[
            "__dq_check"
        ] = (
            f"SEM-{numeric_counter:03d}"
        )

        bad.to_csv(
            OUT
            /
            (
                f"exceptions_"
                f"{table_name}_"
                f"{column}.csv"
            ),
            index=False
        )


# ======================================================================
# 3. VALORES ZERO SUSPEITOS
# ======================================================================

print()
print("=" * 96)
print("3. ZEROS SEMANTICAMENTE SUSPEITOS")
print("=" * 96)


zero_rules = [
    (
        "SEM-030",
        payments,
        "payments",
        "payment_value",
        "payment_value == 0",
    ),

    (
        "SEM-031",
        payments,
        "payments",
        "payment_installments",
        "payment_installments == 0",
    ),

    (
        "SEM-032",
        products,
        "products",
        "product_weight_g",
        "product_weight_g == 0",
    ),
]


for (
    check_id,
    df,
    table,
    column,
    description
) in zero_rules:

    s = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    mask = (
        s.notna()
        &
        (s == 0)
    )

    count = int(
        mask.sum()
    )

    result_from_count(
        check_id=check_id,
        table=table,
        variable=column,
        dimension="semantic_suspicion",
        violations=count,
        total=len(df),
        expected="> 0 preferred",
        severity="WARNING",
        observed=f"{count} zero value(s)",
        details=description,
    )


    print(
        f"[{'PASS' if count == 0 else 'WARN'}] "
        f"{table}.{column} | "
        f"zero={count:,}"
    )


    if count > 0:

        out = (
            df.loc[
                mask
            ]
            .copy()
        )

        out[
            "__dq_check"
        ] = check_id

        out.to_csv(
            OUT
            /
            f"{check_id}_zero_values.csv",
            index=False
        )


# ======================================================================
# 4. FRETE ZERO
# ======================================================================

print()
print("=" * 96)
print("4. FRETE ZERO — PLAUSÍVEL / INFORMATIVO")
print("=" * 96)


freight_zero = (
    pd.to_numeric(
        items[
            "freight_value"
        ],
        errors="coerce"
    )
    ==
    0
)


freight_zero_count = int(
    freight_zero.sum()
)


add_result(
    check_id="SEM-033",
    table="order_items",
    variable="freight_value",
    dimension="business_plausibility",
    severity="INFO",
    status="PASS",
    observed=(
        f"{freight_zero_count} zero freight rows"
    ),
    expected=(
        "zero freight allowed; negative freight prohibited"
    ),
    affected_rows=
        freight_zero_count,

    affected_pct=
        percentage(
            freight_zero_count,
            len(items)
        ),

    details=(
        "Frete zero não é tratado automaticamente "
        "como erro."
    ),
)


print(
    f"[PASS/INFO] freight_value == 0: "
    f"{freight_zero_count:,}"
)


# ======================================================================
# 5. PARSEABILIDADE DE DATAS
# ======================================================================

print()
print("=" * 96)
print("5. PARSEABILIDADE DE TIMESTAMPS")
print("=" * 96)


date_specs = [
    (
        "orders",
        orders,
        "order_purchase_timestamp"
    ),

    (
        "orders",
        orders,
        "order_approved_at"
    ),

    (
        "orders",
        orders,
        "order_delivered_carrier_date"
    ),

    (
        "orders",
        orders,
        "order_delivered_customer_date"
    ),

    (
        "orders",
        orders,
        "order_estimated_delivery_date"
    ),

    (
        "order_items",
        items,
        "shipping_limit_date"
    ),

    (
        "reviews",
        reviews,
        "review_creation_date"
    ),

    (
        "reviews",
        reviews,
        "review_answer_timestamp"
    ),
]


parsed_dates = {}


for i, (
    table,
    df,
    column
) in enumerate(
    date_specs,
    start=40
):

    source_non_null = (
        df[column]
        .notna()
    )

    parsed = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    parsed_dates[
        f"{table}.{column}"
    ] = parsed

    invalid = (
        source_non_null
        &
        parsed.isna()
    )

    count = int(
        invalid.sum()
    )

    result_from_count(
        check_id=
            f"SEM-{i:03d}",

        table=
            table,

        variable=
            column,

        dimension=
            "datetime_parseability",

        violations=
            count,

        total=
            int(
                source_non_null.sum()
            ),

        expected=
            "all non-null timestamps parseable",

        severity=
            "CRITICAL",

        observed=
            f"{count} non-null unparseable values",
    )


    print(
        f"[{'PASS' if count == 0 else 'FAIL'}] "
        f"{table}.{column} | "
        f"unparseable={count:,}"
    )


# ======================================================================
# 6. CONSISTÊNCIA TEMPORAL
# ======================================================================

print()
print("=" * 96)
print("6. CONSISTÊNCIA TEMPORAL")
print("=" * 96)


purchase = parsed_dates[
    "orders.order_purchase_timestamp"
]

approval = parsed_dates[
    "orders.order_approved_at"
]

carrier = parsed_dates[
    "orders.order_delivered_carrier_date"
]

customer_delivery = parsed_dates[
    "orders.order_delivered_customer_date"
]

estimated_delivery = parsed_dates[
    "orders.order_estimated_delivery_date"
]


def temporal_check(
    check_id,
    name,
    left,
    right,
    operator,
    severity,
    expected,
    df_source=orders
):

    valid = (
        left.notna()
        &
        right.notna()
    )

    if operator == ">":

        bad = (
            valid
            &
            (left > right)
        )

    elif operator == ">=":

        bad = (
            valid
            &
            (left >= right)
        )

    elif operator == "<":

        bad = (
            valid
            &
            (left < right)
        )

    elif operator == "<=":

        bad = (
            valid
            &
            (left <= right)
        )

    else:

        raise ValueError(
            operator
        )


    count = int(
        bad.sum()
    )

    result_from_count(
        check_id=check_id,
        table="orders",
        variable=name,
        dimension="cross_field_temporal_consistency",
        violations=count,
        total=int(valid.sum()),
        expected=expected,
        severity=severity,
        observed=f"{count} violation(s)",
    )


    print(
        f"["
        f"{'PASS' if count == 0 else ('FAIL' if severity == 'CRITICAL' else 'WARN')}"
        f"] "
        f"{name} | violations={count:,}"
    )


    if count > 0:

        cols = [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        (
            df_source
            .loc[
                bad,
                cols
            ]
            .to_csv(
                OUT
                /
                f"{check_id}_{name}.csv",
                index=False
            )
        )


# purchase NÃO pode ocorrer depois da aprovação
temporal_check(
    "SEM-050",
    "purchase_after_approval",
    purchase,
    approval,
    ">",
    "CRITICAL",
    "purchase <= approval"
)


# aprovação depois de carrier já foi observada na fonte;
# documentar como warning, sem modificar a origem
temporal_check(
    "SEM-051",
    "approval_after_carrier",
    approval,
    carrier,
    ">",
    "WARNING",
    "approval <= carrier"
)


# carrier após entrega ao cliente é anomalia
temporal_check(
    "SEM-052",
    "carrier_after_customer_delivery",
    carrier,
    customer_delivery,
    ">",
    "WARNING",
    "carrier <= customer_delivery"
)


# entrega real antes da compra seria impossível
temporal_check(
    "SEM-053",
    "purchase_after_customer_delivery",
    purchase,
    customer_delivery,
    ">",
    "CRITICAL",
    "purchase <= customer_delivery"
)


# promessa antes/igual à compra não faz sentido para este case
valid_promise = (
    purchase.notna()
    &
    estimated_delivery.notna()
)

bad_promise = (
    valid_promise
    &
    (
        estimated_delivery
        <=
        purchase
    )
)


promise_count = int(
    bad_promise.sum()
)


result_from_count(
    check_id="SEM-054",
    table="orders",
    variable="promised_lead_time",
    dimension="business_temporal_validity",
    violations=promise_count,
    total=int(
        valid_promise.sum()
    ),
    expected=(
        "estimated_delivery_date > purchase_timestamp"
    ),
    severity="CRITICAL",
    observed=f"{promise_count} invalid promised lead times",
)


print(
    f"[{'PASS' if promise_count == 0 else 'FAIL'}] "
    f"promised_lead_time | "
    f"violations={promise_count:,}"
)


# ======================================================================
# 7. CATEGORIA x TRADUÇÃO
# ======================================================================

print()
print("=" * 96)
print("7. CONSISTÊNCIA DE CATEGORIAS / TRADUÇÃO")
print("=" * 96)


product_categories = set(
    products[
        "product_category_name"
    ]
    .dropna()
    .astype(str)
    .unique()
)


translation_categories = set(
    translation[
        "product_category_name"
    ]
    .dropna()
    .astype(str)
    .unique()
)


without_translation = sorted(
    product_categories
    -
    translation_categories
)


count_without_translation = (
    products[
        "product_category_name"
    ]
    .isin(
        without_translation
    )
    .sum()
)


result_from_count(
    check_id="SEM-060",
    table="products",
    variable="product_category_name",
    dimension="cross_table_semantic_consistency",
    violations=int(
        count_without_translation
    ),
    total=int(
        products[
            "product_category_name"
        ]
        .notna()
        .sum()
    ),
    expected=(
        "category available in translation table"
    ),
    severity="WARNING",
    observed=without_translation,
    details=(
        "Tradução é apenas apoio/apresentação; "
        "produto não será removido."
    ),
)


print(
    f"[{'PASS' if not without_translation else 'WARN'}] "
    f"Categorias sem tradução: "
    f"{len(without_translation)}"
)


for cat in without_translation:

    print(
        f"  - {cat}"
    )


# ======================================================================
# 8. GEOLOCATION — CEP ASSOCIADO A MÚLTIPLOS ESTADOS
# ======================================================================

print()
print("=" * 96)
print("8. CONSISTÊNCIA GEOGRÁFICA POR CEP-PREFIXO")
print("=" * 96)


geo_state_profile = (
    geo
    .groupby(
        "geolocation_zip_code_prefix"
    )[
        "geolocation_state"
    ]
    .nunique(
        dropna=True
    )
)


multi_state_zip = (
    geo_state_profile
    [
        geo_state_profile
        >
        1
    ]
)


multi_state_count = int(
    len(
        multi_state_zip
    )
)


result_from_count(
    check_id="SEM-061",
    table="geolocation",
    variable="geolocation_zip_code_prefix",
    dimension="geographic_consistency",
    violations=multi_state_count,
    total=int(
        geo[
            "geolocation_zip_code_prefix"
        ]
        .nunique()
    ),
    expected="1 state per zip prefix",
    severity="WARNING",
    observed=(
        f"{multi_state_count} prefixes associated "
        f"with more than one state"
    ),
    details=(
        "Será investigado antes da consolidação "
        "definitiva da geolocalização."
    ),
)


if multi_state_count:

    geo_multi_state = (
        geo[
            geo[
                "geolocation_zip_code_prefix"
            ]
            .isin(
                multi_state_zip.index
            )
        ]
        .sort_values(
            [
                "geolocation_zip_code_prefix",
                "geolocation_state"
            ]
        )
    )

    geo_multi_state.to_csv(
        OUT
        /
        "SEM-061_zip_prefix_multiple_states.csv",
        index=False
    )


print(
    f"[{'PASS' if multi_state_count == 0 else 'WARN'}] "
    f"CEP-prefixos com múltiplos estados: "
    f"{multi_state_count:,}"
)


# ======================================================================
# 9. VALORES NÃO FINITOS
# ======================================================================

print()
print("=" * 96)
print("9. VALORES NUMÉRICOS NÃO FINITOS")
print("=" * 96)


finite_specs = [
    (
        "order_items",
        items,
        "price"
    ),

    (
        "order_items",
        items,
        "freight_value"
    ),

    (
        "payments",
        payments,
        "payment_value"
    ),

    (
        "payments",
        payments,
        "payment_installments"
    ),

    (
        "products",
        products,
        "product_weight_g"
    ),

    (
        "products",
        products,
        "product_length_cm"
    ),

    (
        "products",
        products,
        "product_height_cm"
    ),

    (
        "products",
        products,
        "product_width_cm"
    ),

    (
        "geolocation",
        geo,
        "geolocation_lat"
    ),

    (
        "geolocation",
        geo,
        "geolocation_lng"
    ),
]


for i, (
    table,
    df,
    column
) in enumerate(
    finite_specs,
    start=70
):

    x = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    original_non_null = (
        df[column]
        .notna()
    )

    nonfinite = (
        original_non_null
        &
        ~np.isfinite(
            x
        )
    )


    count = int(
        nonfinite.sum()
    )


    result_from_count(
        check_id=
            f"SEM-{i:03d}",

        table=
            table,

        variable=
            column,

        dimension=
            "finite_numeric_validity",

        violations=
            count,

        total=
            int(
                original_non_null.sum()
            ),

        expected=
            "all non-null numeric values finite",

        severity=
            "CRITICAL",

        observed=
            f"{count} non-finite value(s)",
    )


    print(
        f"[{'PASS' if count == 0 else 'FAIL'}] "
        f"{table}.{column} | "
        f"non-finite={count:,}"
    )


# ======================================================================
# SCORECARD
# ======================================================================

scorecard = pd.DataFrame(
    results
)


scorecard.to_csv(
    OUT
    /
    "dq_gate_02_scorecard.csv",
    index=False
)


exceptions = (
    scorecard[
        scorecard[
            "status"
        ]
        .isin(
            [
                "FAIL",
                "WARN"
            ]
        )
    ]
    .copy()
)


exceptions.to_csv(
    OUT
    /
    "dq_gate_02_exceptions.csv",
    index=False
)


if exception_records:

    pd.DataFrame(
        exception_records
    ).to_csv(
        OUT
        /
        "dq_gate_02_domain_exception_values.csv",
        index=False
    )


# ======================================================================
# RESUMO
# ======================================================================

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


total_checks = int(
    len(
        scorecard
    )
)


gate_status = (
    "PASS"
    if critical_failures == 0
    else
    "FAIL"
)


summary = {
    "gate":
        "DQ_GATE_02_SEMANTIC",

    "status":
        gate_status,

    "prerequisite_gate_01":
        gate1.get("status"),

    "total_checks":
        total_checks,

    "passes":
        passes,

    "warnings":
        warnings,

    "critical_failures":
        critical_failures,

    "raw_directory":
        str(RAW),

    "contract":
        str(CONFIG),
}


with (
    OUT
    /
    "dq_gate_02_summary.json"
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


# ======================================================================
# RELATÓRIO TXT
# ======================================================================

report = []

report.append(
    "=" * 96
)

report.append(
    "DQ GATE 02 — SEMANTIC / VALIDITY"
)

report.append(
    "=" * 96
)

report.append("")

report.append(
    "PRÉ-REQUISITO"
)

report.append(
    f"DQ GATE 01       : {gate1.get('status')}"
)

report.append("")

report.append(
    f"STATUS            : {gate_status}"
)

report.append(
    f"TOTAL CHECKS      : {total_checks}"
)

report.append(
    f"PASS              : {passes}"
)

report.append(
    f"WARNINGS          : {warnings}"
)

report.append(
    f"CRITICAL FAILURES : {critical_failures}"
)

report.append("")

report.append(
    "CHECKS NÃO-PASS:"
)

report.append(
    "-" * 96
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
            f"{row['table']}.{row['variable']} | "
            f"{row['dimension']} | "
            f"afetados={row['affected_rows']} "
            f"({row['affected_pct']:.4f}%) | "
            f"observado={row['observed']} | "
            f"esperado={row['expected']}"
        )


report.append("")

report.append(
    "-" * 96
)

report.append(
    "INTERPRETAÇÃO:"
)

report.append(
    "CRITICAL FAIL -> DQ Gate 02 reprovado."
)

report.append(
    "WARNING -> anomalia real documentada, "
    "mas não bloqueante."
)

report.append(
    "PASS com warnings -> pode avançar ao "
    "DQ Gate 03 — Statistical / Completeness."
)

report.append(
    "Nenhum dado RAW é alterado por este gate."
)

report.append(
    "-" * 96
)


report_path = (
    OUT
    /
    "DQ_GATE_02_SEMANTIC_REPORT.txt"
)


report_path.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# ======================================================================
# TERMINAL
# ======================================================================

print()
print("=" * 96)
print("DQ GATE 02 — RESULTADO")
print("=" * 96)

print(
    f"STATUS            : {gate_status}"
)

print(
    f"TOTAL CHECKS      : {total_checks}"
)

print(
    f"PASS              : {passes}"
)

print(
    f"WARNINGS          : {warnings}"
)

print(
    f"CRITICAL FAILURES : {critical_failures}"
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
                "table",
                "variable",
                "affected_rows",
                "affected_pct",
                "observed",
            ]
        ]
        .to_string(
            index=False
        )
    )


print()

print(
    "Arquivos gerados:"
)

for path in sorted(
    OUT.glob("*")
):

    print(
        f"  - {path}"
    )


print()


if gate_status == "PASS":

    print(
        "[PASS] DQ GATE 02 — SEMANTIC / VALIDITY APROVADO."
    )

    print(
        "Os dados passaram pelas regras semânticas críticas."
    )

    print(
        "Warnings permanecem documentados para tratamento posterior."
    )

    sys.exit(0)


else:

    print(
        "[FAIL] DQ GATE 02 — SEMANTIC / VALIDITY REPROVADO."
    )

    print(
        "O próximo gate fica BLOQUEADO "
        "até revisão das falhas críticas."
    )

    sys.exit(2)

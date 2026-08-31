#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DQ GATE 03 — STATISTICAL / COMPLETENESS
Delivery Risk Intelligence Platform

PRÉ-REQUISITO:
    Gate 01 — Structural = PASS
    Gate 02 — Semantic / Validity = PASS

VALIDA:
    1. completude por variável
    2. cobertura entre fontes
    3. estatísticas robustas das variáveis numéricas
    4. quantis e comportamento de cauda
    5. candidatos a extremos por IQR e MAD
    6. concentração / variáveis quase constantes
    7. cardinalidade categórica
    8. missingness por mês
    9. cobertura geográfica

NÃO FAZ:
    - imputação
    - remoção de outliers
    - winsorization
    - clipping
    - scaling
    - feature engineering
    - target definitivo
    - split de modelagem
    - Silver

IMPORTANTE:
    Extremos são sinalizados, nunca removidos automaticamente.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = PROJECT / "data" / "raw" / "olist"

CONFIG = PROJECT / "configs" / "dq_gate_03_statistical.json"

GATE1 = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_01_structural"
    / "dq_gate_01_summary.json"
)

GATE2 = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_02_semantic"
    / "dq_gate_02_summary.json"
)

OUT = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03_statistical"
)

OUT.mkdir(parents=True, exist_ok=True)


# =====================================================================
# PRÉ-REQUISITOS
# =====================================================================

def load_gate(path, name):

    if not path.exists():

        print(f"[BLOQUEADO] {name} não encontrado:")
        print(path)

        sys.exit(2)

    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if result.get("status") != "PASS":

        print(
            f"[BLOQUEADO] {name} = "
            f"{result.get('status')}"
        )

        sys.exit(2)

    return result


gate1 = load_gate(
    GATE1,
    "DQ Gate 01"
)

gate2 = load_gate(
    GATE2,
    "DQ Gate 02"
)


# =====================================================================
# CONFIG
# =====================================================================

with CONFIG.open(
    "r",
    encoding="utf-8"
) as f:

    CONTRACT = json.load(f)


# =====================================================================
# ARQUIVOS
# =====================================================================

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
        "product_category_name_translation.csv"
}


dfs = {}

for table, filename in FILES.items():

    path = RAW / filename

    if not path.exists():

        print(f"[ERRO] Arquivo ausente: {path}")
        sys.exit(2)

    dfs[table] = pd.read_csv(
        path,
        low_memory=False
    )


# =====================================================================
# RESULTADOS
# =====================================================================

results = []


def pct(n, d):

    if d == 0:
        return np.nan

    return 100.0 * n / d


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
                "DQ_GATE_03_STATISTICAL_COMPLETENESS",

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
                float(affected_pct)
                if pd.notna(affected_pct)
                else np.nan,

            "observed":
                str(observed),

            "expected":
                str(expected),

            "details":
                str(details)
        }
    )


# =====================================================================
# HEADER
# =====================================================================

print("=" * 100)

print(
    "DQ GATE 03 — STATISTICAL / COMPLETENESS"
)

print(
    "Delivery Risk Intelligence Platform"
)

print("=" * 100)

print()

print(
    "[PASS] DQ Gate 01 — Structural"
)

print(
    "[PASS] DQ Gate 02 — Semantic / Validity"
)

print()

print(
    f"RAW      : {RAW}"
)

print(
    f"CONTRACT : {CONFIG}"
)

print(
    f"OUTPUT   : {OUT}"
)


# =====================================================================
# 1. COMPLETUDE POR VARIÁVEL
# =====================================================================

print()
print("=" * 100)
print("1. COMPLETUDE POR VARIÁVEL")
print("=" * 100)


completeness_rows = []


for i, (
    key,
    rule
) in enumerate(
    CONTRACT[
        "completeness_rules"
    ].items(),
    start=1
):

    table, column = key.split(
        ".",
        1
    )

    df = dfs[table]

    missing = int(
        df[column]
        .isna()
        .sum()
    )

    total = len(df)

    missing_pct = pct(
        missing,
        total
    )

    completeness_pct = (
        100.0
        -
        missing_pct
    )

    max_missing = float(
        rule[
            "max_missing_pct"
        ]
    )

    severity = rule[
        "severity"
    ]

    if missing_pct <= max_missing:

        status = "PASS"

    elif severity == "CRITICAL":

        status = "FAIL"

    elif severity == "WARNING":

        status = "WARN"

    else:

        status = "INFO"


    completeness_rows.append(
        {
            "table":
                table,

            "column":
                column,

            "role":
                rule.get(
                    "role",
                    ""
                ),

            "rows":
                total,

            "missing":
                missing,

            "missing_pct":
                missing_pct,

            "completeness_pct":
                completeness_pct,

            "threshold_max_missing_pct":
                max_missing,

            "severity":
                severity,

            "status":
                status
        }
    )


    add_result(
        check_id=
            f"STA-COMP-{i:03d}",

        table=
            table,

        variable=
            column,

        dimension=
            "completeness",

        severity=
            severity,

        status=
            status,

        observed=
            f"{missing_pct:.4f}% missing",

        expected=
            f"<= {max_missing:.4f}% missing",

        affected_rows=
            missing,

        affected_pct=
            missing_pct,

        details=
            rule.get(
                "role",
                ""
            )
    )


    print(
        f"[{status:<4}] "
        f"{table}.{column:<38} "
        f"missing={missing:>7,} "
        f"({missing_pct:>8.4f}%) | "
        f"limite={max_missing:.4f}%"
    )


completeness_df = pd.DataFrame(
    completeness_rows
)


completeness_df.to_csv(
    OUT
    / "01_completeness_scorecard.csv",
    index=False
)


# =====================================================================
# 2. COBERTURA ENTRE FONTES
# =====================================================================

print()
print("=" * 100)
print("2. COBERTURA ENTRE FONTES")
print("=" * 100)


orders = dfs["orders"]
items = dfs["order_items"]
payments = dfs["payments"]
customers = dfs["customers"]
sellers = dfs["sellers"]
geo = dfs["geolocation"]
products = dfs["products"]


all_order_ids = set(
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


orders_without_items = (
    all_order_ids
    -
    item_order_ids
)


orders_without_payments = (
    all_order_ids
    -
    payment_order_ids
)


geo_zips = set(
    geo[
        "geolocation_zip_code_prefix"
    ]
    .dropna()
    .astype(int)
)


customers_without_geo = (
    ~customers[
        "customer_zip_code_prefix"
    ]
    .isin(
        geo_zips
    )
)


sellers_without_geo = (
    ~sellers[
        "seller_zip_code_prefix"
    ]
    .isin(
        geo_zips
    )
)


coverage_specs = [

    (
        "STA-COV-001",
        "orders",
        "order_items_coverage",
        len(
            orders_without_items
        ),
        len(
            orders
        ),
        CONTRACT[
            "coverage_warning_thresholds"
        ][
            "orders_without_items_pct"
        ],
        "Orders without order_items"
    ),

    (
        "STA-COV-002",
        "orders",
        "payment_coverage",
        len(
            orders_without_payments
        ),
        len(
            orders
        ),
        CONTRACT[
            "coverage_warning_thresholds"
        ][
            "orders_without_payments_pct"
        ],
        "Orders without payment"
    ),

    (
        "STA-COV-003",
        "customers",
        "geolocation_coverage",
        int(
            customers_without_geo.sum()
        ),
        len(
            customers
        ),
        CONTRACT[
            "coverage_warning_thresholds"
        ][
            "customers_without_geolocation_pct"
        ],
        "Customers whose ZIP is absent from geolocation"
    ),

    (
        "STA-COV-004",
        "sellers",
        "geolocation_coverage",
        int(
            sellers_without_geo.sum()
        ),
        len(
            sellers
        ),
        CONTRACT[
            "coverage_warning_thresholds"
        ][
            "sellers_without_geolocation_pct"
        ],
        "Sellers whose ZIP is absent from geolocation"
    )
]


coverage_rows = []


for (
    check_id,
    table,
    variable,
    affected,
    total,
    threshold,
    description
) in coverage_specs:

    affected_pct = pct(
        affected,
        total
    )

    status = (
        "PASS"
        if affected_pct <= threshold
        else
        "WARN"
    )


    coverage_rows.append(
        {
            "check_id":
                check_id,

            "table":
                table,

            "coverage_measure":
                variable,

            "total":
                total,

            "uncovered":
                affected,

            "uncovered_pct":
                affected_pct,

            "warning_threshold_pct":
                threshold,

            "status":
                status,

            "description":
                description
        }
    )


    add_result(
        check_id=
            check_id,

        table=
            table,

        variable=
            variable,

        dimension=
            "source_coverage",

        severity=
            "WARNING",

        status=
            status,

        observed=
            f"{affected_pct:.4f}% uncovered",

        expected=
            f"<= {threshold:.4f}%",

        affected_rows=
            affected,

        affected_pct=
            affected_pct,

        details=
            description
    )


    print(
        f"[{status}] "
        f"{description:<48} "
        f"{affected:>7,} "
        f"({affected_pct:.4f}%)"
    )


coverage_df = pd.DataFrame(
    coverage_rows
)


coverage_df.to_csv(
    OUT
    / "02_source_coverage.csv",
    index=False
)


# =====================================================================
# 3. PERFIL ESTATÍSTICO NUMÉRICO
# =====================================================================

print()
print("=" * 100)
print("3. PERFIL ESTATÍSTICO NUMÉRICO")
print("=" * 100)


def numeric_profile(
    table,
    column,
    series
):

    x = pd.to_numeric(
        series,
        errors="coerce"
    )

    x = x.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    valid = x.dropna()

    n_total = len(x)
    n_valid = len(valid)

    if n_valid == 0:

        return {
            "table": table,
            "variable": column,
            "n_total": n_total,
            "n_valid": 0,
            "missing": n_total
        }


    q = valid.quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            0.995
        ]
    )


    q1 = float(
        q.loc[0.25]
    )

    q3 = float(
        q.loc[0.75]
    )

    median = float(
        q.loc[0.50]
    )

    iqr = q3 - q1


    mad = float(
        np.median(
            np.abs(
                valid.to_numpy(
                    dtype=float
                )
                -
                median
            )
        )
    )


    lower_iqr = (
        q1
        -
        1.5 * iqr
    )

    upper_iqr = (
        q3
        +
        1.5 * iqr
    )


    iqr_flag = (
        (valid < lower_iqr)
        |
        (valid > upper_iqr)
    )


    if mad > 0:

        modified_z = (
            0.6745
            *
            (
                valid
                -
                median
            )
            /
            mad
        )

        mad_flag = (
            modified_z.abs()
            >
            3.5
        )

        mad_count = int(
            mad_flag.sum()
        )

    else:

        mad_count = 0


    counts = valid.value_counts(
        dropna=False
    )

    dominant_share = (
        100.0
        *
        counts.iloc[0]
        /
        n_valid
    )


    return {

        "table":
            table,

        "variable":
            column,

        "n_total":
            n_total,

        "n_valid":
            n_valid,

        "missing":
            int(
                x.isna().sum()
            ),

        "missing_pct":
            pct(
                int(
                    x.isna().sum()
                ),
                n_total
            ),

        "unique":
            int(
                valid.nunique()
            ),

        "dominant_value_share_pct":
            dominant_share,

        "zero_count":
            int(
                (valid == 0).sum()
            ),

        "zero_pct":
            pct(
                int(
                    (valid == 0).sum()
                ),
                n_valid
            ),

        "min":
            float(
                valid.min()
            ),

        "p01":
            float(
                q.loc[0.01]
            ),

        "p05":
            float(
                q.loc[0.05]
            ),

        "q1":
            q1,

        "median":
            median,

        "mean":
            float(
                valid.mean()
            ),

        "q3":
            q3,

        "p95":
            float(
                q.loc[0.95]
            ),

        "p99":
            float(
                q.loc[0.99]
            ),

        "p995":
            float(
                q.loc[0.995]
            ),

        "max":
            float(
                valid.max()
            ),

        "std":
            float(
                valid.std()
            )
            if n_valid > 1
            else 0.0,

        "iqr":
            float(
                iqr
            ),

        "mad":
            mad,

        "skewness":
            float(
                valid.skew()
            )
            if n_valid > 2
            else np.nan,

        "kurtosis":
            float(
                valid.kurt()
            )
            if n_valid > 3
            else np.nan,

        "iqr_lower":
            float(
                lower_iqr
            ),

        "iqr_upper":
            float(
                upper_iqr
            ),

        "iqr_candidate_count":
            int(
                iqr_flag.sum()
            ),

        "iqr_candidate_pct":
            pct(
                int(
                    iqr_flag.sum()
                ),
                n_valid
            ),

        "mad_candidate_count":
            mad_count,

        "mad_candidate_pct":
            pct(
                mad_count,
                n_valid
            )
    }


profiles = []


for key in CONTRACT[
    "numeric_profile"
]:

    table, column = key.split(
        ".",
        1
    )

    profiles.append(
        numeric_profile(
            table,
            column,
            dfs[table][column]
        )
    )


numeric_df = pd.DataFrame(
    profiles
)


numeric_df.to_csv(
    OUT
    / "03_numeric_statistical_profile.csv",
    index=False
)


print(
    numeric_df[
        [
            "table",
            "variable",
            "missing_pct",
            "min",
            "median",
            "mean",
            "p99",
            "max",
            "skewness",
            "iqr_candidate_pct",
            "mad_candidate_pct"
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "missing_pct":
                "{:.4f}".format,

            "skewness":
                "{:.4f}".format,

            "iqr_candidate_pct":
                "{:.4f}".format,

            "mad_candidate_pct":
                "{:.4f}".format
        }
    )
)


print()

print(
    "[INFO] IQR e MAD somente sinalizam extremos. "
    "Nenhum registro foi removido."
)


# =====================================================================
# 4. CONSTANTES / QUASE CONSTANTES
# =====================================================================

print()
print("=" * 100)
print("4. VARIÁVEIS CONSTANTES / QUASE CONSTANTES")
print("=" * 100)


near_constant_threshold = float(
    CONTRACT[
        "statistical_rules"
    ][
        "near_constant_dominant_share_pct"
    ]
)


for i, row in numeric_df.iterrows():

    unique = int(
        row["unique"]
    )

    dominant = float(
        row[
            "dominant_value_share_pct"
        ]
    )


    if unique <= 1:

        status = "WARN"

        reason = "constant column"

    elif dominant >= near_constant_threshold:

        status = "WARN"

        reason = (
            f"dominant value >= "
            f"{near_constant_threshold}%"
        )

    else:

        status = "PASS"

        reason = "adequate variability"


    add_result(
        check_id=
            f"STA-VAR-{i+1:03d}",

        table=
            row["table"],

        variable=
            row["variable"],

        dimension=
            "statistical_variability",

        severity=
            "WARNING",

        status=
            status,

        observed=
            (
                f"unique={unique}; "
                f"dominant={dominant:.4f}%"
            ),

        expected=
            "non-constant / non-degenerate",

        affected_rows=
            0,

        affected_pct=
            0.0,

        details=
            reason
    )


    print(
        f"[{status}] "
        f"{row['table']}.{row['variable']} | "
        f"unique={unique:,} | "
        f"dominant={dominant:.4f}%"
    )


# =====================================================================
# 5. PERFIL CATEGÓRICO
# =====================================================================

print()
print("=" * 100)
print("5. PERFIL CATEGÓRICO")
print("=" * 100)


categorical_summary = []


for key in CONTRACT[
    "categorical_profile"
]:

    table, column = key.split(
        ".",
        1
    )

    s = dfs[table][column]

    non_null = s.dropna()

    counts = (
        non_null
        .astype(str)
        .value_counts()
    )


    if len(counts):

        dominant_value = str(
            counts.index[0]
        )

        dominant_count = int(
            counts.iloc[0]
        )

        dominant_pct = pct(
            dominant_count,
            len(non_null)
        )

    else:

        dominant_value = ""
        dominant_count = 0
        dominant_pct = np.nan


    singleton_levels = int(
        (counts == 1)
        .sum()
    )


    categorical_summary.append(
        {
            "table":
                table,

            "variable":
                column,

            "rows":
                len(s),

            "missing":
                int(
                    s.isna().sum()
                ),

            "missing_pct":
                pct(
                    int(
                        s.isna().sum()
                    ),
                    len(s)
                ),

            "cardinality":
                int(
                    non_null.nunique()
                ),

            "dominant_value":
                dominant_value,

            "dominant_count":
                dominant_count,

            "dominant_pct":
                dominant_pct,

            "singleton_levels":
                singleton_levels
        }
    )


    frequency = (
        counts
        .rename_axis(
            column
        )
        .reset_index(
            name="count"
        )
    )


    frequency[
        "pct"
    ] = (
        100
        *
        frequency[
            "count"
        ]
        /
        len(non_null)
        if len(non_null)
        else np.nan
    )


    frequency.to_csv(
        OUT
        /
        (
            "categorical_frequency_"
            f"{table}_{column}.csv"
        ),
        index=False
    )


categorical_df = pd.DataFrame(
    categorical_summary
)


categorical_df.to_csv(
    OUT
    / "04_categorical_profile.csv",
    index=False
)


print(
    categorical_df.to_string(
        index=False,
        formatters={
            "missing_pct":
                "{:.4f}".format,

            "dominant_pct":
                lambda x:
                    ""
                    if pd.isna(x)
                    else f"{x:.4f}"
        }
    )
)


# =====================================================================
# 6. MISSINGNESS POR MÊS
# =====================================================================

print()
print("=" * 100)
print("6. MISSINGNESS / COBERTURA POR MÊS")
print("=" * 100)


orders_time = orders.copy()


orders_time[
    "order_purchase_timestamp"
] = pd.to_datetime(
    orders_time[
        "order_purchase_timestamp"
    ],
    errors="coerce"
)


orders_time[
    "purchase_month"
] = (
    orders_time[
        "order_purchase_timestamp"
    ]
    .dt
    .to_period("M")
    .astype(str)
)


# ---------------------------------------------------------------------
# PRESENÇA DE ITEM / PAYMENT POR PEDIDO
# ---------------------------------------------------------------------

item_presence = (
    items[
        [
            "order_id"
        ]
    ]
    .drop_duplicates()
    .assign(
        has_items=1
    )
)


payment_presence = (
    payments[
        [
            "order_id"
        ]
    ]
    .drop_duplicates()
    .assign(
        has_payment=1
    )
)


# ---------------------------------------------------------------------
# METADADOS DE PRODUTO POR PEDIDO
# ---------------------------------------------------------------------

item_product = (
    items[
        [
            "order_id",
            "product_id"
        ]
    ]
    .merge(
        products[
            [
                "product_id",
                "product_category_name",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm"
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one"
    )
)


item_product[
    "product_category_missing"
] = (
    item_product[
        "product_category_name"
    ]
    .isna()
    .astype(int)
)


physical_cols = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]


item_product[
    "physical_missing"
] = (
    item_product[
        physical_cols
    ]
    .isna()
    .any(axis=1)
    .astype(int)
)


product_order_quality = (
    item_product
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        product_category_missing=(
            "product_category_missing",
            "max"
        ),

        product_physical_missing=(
            "physical_missing",
            "max"
        )
    )
)


# ---------------------------------------------------------------------
# CUSTOMER GEO
# ---------------------------------------------------------------------

geo_zips_series = set(
    geo[
        "geolocation_zip_code_prefix"
    ]
    .dropna()
)


customer_quality = (
    customers[
        [
            "customer_id",
            "customer_zip_code_prefix"
        ]
    ]
    .copy()
)


customer_quality[
    "customer_geo_missing"
] = (
    ~customer_quality[
        "customer_zip_code_prefix"
    ]
    .isin(
        geo_zips_series
    )
).astype(int)


# ---------------------------------------------------------------------
# SELLER GEO POR PEDIDO
# ---------------------------------------------------------------------

seller_quality = sellers[
    [
        "seller_id",
        "seller_zip_code_prefix"
    ]
].copy()


seller_quality[
    "seller_geo_missing"
] = (
    ~seller_quality[
        "seller_zip_code_prefix"
    ]
    .isin(
        geo_zips_series
    )
).astype(int)


order_seller_quality = (
    items[
        [
            "order_id",
            "seller_id"
        ]
    ]
    .merge(
        seller_quality,
        on="seller_id",
        how="left",
        validate="many_to_one"
    )
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        seller_geo_missing=(
            "seller_geo_missing",
            "max"
        )
    )
)


# ---------------------------------------------------------------------
# DATASET DIAGNÓSTICO TEMPORÁRIO
# ---------------------------------------------------------------------

month_diag = (
    orders_time
    .merge(
        item_presence,
        on="order_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        payment_presence,
        on="order_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        product_order_quality,
        on="order_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        customer_quality[
            [
                "customer_id",
                "customer_geo_missing"
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )
    .merge(
        order_seller_quality,
        on="order_id",
        how="left",
        validate="one_to_one"
    )
)


month_diag[
    "item_missing"
] = (
    month_diag[
        "has_items"
    ]
    .isna()
    .astype(int)
)


month_diag[
    "payment_missing"
] = (
    month_diag[
        "has_payment"
    ]
    .isna()
    .astype(int)
)


for col in [
    "product_category_missing",
    "product_physical_missing",
    "customer_geo_missing",
    "seller_geo_missing"
]:

    month_diag[col] = (
        month_diag[col]
        .fillna(1)
        .astype(int)
    )


monthly_rows = []


monthly_indicators = [
    "item_missing",
    "payment_missing",
    "product_category_missing",
    "product_physical_missing",
    "customer_geo_missing",
    "seller_geo_missing"
]


for (
    month,
    sub
) in month_diag.groupby(
    "purchase_month"
):

    if month == "NaT":
        continue

    record = {
        "purchase_month":
            month,

        "orders":
            len(sub)
    }


    for col in monthly_indicators:

        count = int(
            sub[col]
            .sum()
        )

        record[
            f"{col}_count"
        ] = count

        record[
            f"{col}_pct"
        ] = pct(
            count,
            len(sub)
        )


    monthly_rows.append(
        record
    )


monthly_df = pd.DataFrame(
    monthly_rows
).sort_values(
    "purchase_month"
)


monthly_df.to_csv(
    OUT
    / "05_monthly_missingness_coverage.csv",
    index=False
)


print(
    monthly_df.to_string(
        index=False,
        formatters={
            c:
                "{:.4f}".format
            for c in monthly_df.columns
            if c.endswith("_pct")
        }
    )
)


# =====================================================================
# 7. ESTABILIDADE DA MISSINGNESS
# =====================================================================

print()
print("=" * 100)
print("7. ESTABILIDADE TEMPORAL DA MISSINGNESS — DIAGNÓSTICO")
print("=" * 100)


stable_months = monthly_df[
    monthly_df[
        "orders"
    ]
    >=
    100
].copy()


stability_rows = []


for indicator in monthly_indicators:

    col = (
        f"{indicator}_pct"
    )

    if stable_months.empty:

        continue


    min_pct = float(
        stable_months[col]
        .min()
    )

    max_pct = float(
        stable_months[col]
        .max()
    )

    range_pp = (
        max_pct
        -
        min_pct
    )


    stability_rows.append(
        {
            "indicator":
                indicator,

            "months_evaluated":
                len(stable_months),

            "min_pct":
                min_pct,

            "max_pct":
                max_pct,

            "range_percentage_points":
                range_pp
        }
    )


    add_result(
        check_id=
            f"STA-MISS-{len(stability_rows):03d}",

        table=
            "__order_level__",

        variable=
            indicator,

        dimension=
            "missingness_temporal_profile",

        severity=
            "INFO",

        status=
            "INFO",

        observed=
            (
                f"min={min_pct:.4f}%; "
                f"max={max_pct:.4f}%; "
                f"range={range_pp:.4f} pp"
            ),

        expected=
            "diagnostic only; formal drift evaluated later",

        details=
            "Months with at least 100 orders."
    )


    print(
        f"{indicator:<34} "
        f"min={min_pct:>8.4f}% | "
        f"max={max_pct:>8.4f}% | "
        f"range={range_pp:>8.4f} pp"
    )


stability_df = pd.DataFrame(
    stability_rows
)


stability_df.to_csv(
    OUT
    / "06_missingness_temporal_stability.csv",
    index=False
)


# =====================================================================
# 8. OUTLIER FLAGS — SOMENTE QUANTIFICAÇÃO
# =====================================================================

print()
print("=" * 100)
print("8. EXTREMOS — IQR / MAD")
print("=" * 100)


tail_df = numeric_df[
    [
        "table",
        "variable",
        "iqr_lower",
        "iqr_upper",
        "iqr_candidate_count",
        "iqr_candidate_pct",
        "mad",
        "mad_candidate_count",
        "mad_candidate_pct"
    ]
].copy()


tail_df.to_csv(
    OUT
    / "07_extreme_value_flags_summary.csv",
    index=False
)


for _, row in tail_df.iterrows():

    add_result(
        check_id=
            (
                "STA-TAIL-"
                f"{len(results)+1:03d}"
            ),

        table=
            row["table"],

        variable=
            row["variable"],

        dimension=
            "tail_behavior",

        severity=
            "INFO",

        status=
            "INFO",

        observed=
            (
                f"IQR flags="
                f"{int(row['iqr_candidate_count']):,} "
                f"({row['iqr_candidate_pct']:.4f}%); "
                f"MAD flags="
                f"{int(row['mad_candidate_count']):,} "
                f"({row['mad_candidate_pct']:.4f}%)"
            ),

        expected=
            "diagnostic only",

        affected_rows=
            int(
                row[
                    "iqr_candidate_count"
                ]
            ),

        affected_pct=
            float(
                row[
                    "iqr_candidate_pct"
                ]
            ),

        details=
            "Extremo não implica erro; nenhuma remoção aplicada."
    )


    print(
        f"{row['table']}.{row['variable']:<30} | "
        f"IQR={row['iqr_candidate_pct']:.4f}% | "
        f"MAD={row['mad_candidate_pct']:.4f}%"
    )


# =====================================================================
# 9. RESUMO GLOBAL DE MISSING
# =====================================================================

print()
print("=" * 100)
print("9. RESUMO GLOBAL DE MISSINGNESS")
print("=" * 100)


global_missing_rows = []


for table, df in dfs.items():

    cells = (
        df.shape[0]
        *
        df.shape[1]
    )

    missing_cells = int(
        df
        .isna()
        .sum()
        .sum()
    )

    global_missing_rows.append(
        {
            "table":
                table,

            "rows":
                df.shape[0],

            "columns":
                df.shape[1],

            "cells":
                cells,

            "missing_cells":
                missing_cells,

            "missing_cells_pct":
                pct(
                    missing_cells,
                    cells
                )
        }
    )


global_missing_df = pd.DataFrame(
    global_missing_rows
)


global_missing_df.to_csv(
    OUT
    / "08_table_missingness_summary.csv",
    index=False
)


print(
    global_missing_df.to_string(
        index=False,
        formatters={
            "missing_cells_pct":
                "{:.4f}".format
        }
    )
)


# =====================================================================
# 10. SCORECARD FINAL
# =====================================================================

scorecard = pd.DataFrame(
    results
)


scorecard.to_csv(
    OUT
    / "dq_gate_03_scorecard.csv",
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
    / "dq_gate_03_exceptions.csv",
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
    ).sum()
)


warnings = int(
    (
        scorecard[
            "status"
        ]
        ==
        "WARN"
    ).sum()
)


passes = int(
    (
        scorecard[
            "status"
        ]
        ==
        "PASS"
    ).sum()
)


infos = int(
    (
        scorecard[
            "status"
        ]
        ==
        "INFO"
    ).sum()
)


total_checks = len(
    scorecard
)


gate_status = (
    "PASS"
    if critical_failures == 0
    else
    "FAIL"
)


summary = {

    "gate":
        "DQ_GATE_03_STATISTICAL_COMPLETENESS",

    "status":
        gate_status,

    "prerequisite_gate_01":
        gate1.get("status"),

    "prerequisite_gate_02":
        gate2.get("status"),

    "total_checks":
        total_checks,

    "passes":
        passes,

    "warnings":
        warnings,

    "informational":
        infos,

    "critical_failures":
        critical_failures,

    "raw_directory":
        str(RAW),

    "contract":
        str(CONFIG),

    "raw_modified":
        False
}


with (
    OUT
    / "dq_gate_03_summary.json"
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


# =====================================================================
# 11. RELATÓRIO TXT
# =====================================================================

report = []


report.append(
    "=" * 100
)

report.append(
    "DQ GATE 03 — STATISTICAL / COMPLETENESS"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    f"DQ GATE 01       : "
    f"{gate1.get('status')}"
)

report.append(
    f"DQ GATE 02       : "
    f"{gate2.get('status')}"
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
    f"INFORMATIONAL     : {infos}"
)

report.append(
    f"CRITICAL FAILURES : {critical_failures}"
)

report.append("")

report.append(
    "CHECKS NÃO-PASS:"
)

report.append(
    "-" * 100
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
    "-" * 100
)

report.append(
    "INTERPRETAÇÃO"
)

report.append(
    "CRITICAL FAIL -> Gate 03 reprovado."
)

report.append(
    "WARN -> problema/tolerância que exige documentação."
)

report.append(
    "INFO -> perfil estatístico, sem decisão automática."
)

report.append(
    "IQR/MAD são sinalizadores de cauda, não regras de exclusão."
)

report.append(
    "Nenhum missing foi imputado."
)

report.append(
    "Nenhum extremo foi removido."
)

report.append(
    "Nenhum arquivo RAW foi modificado."
)

report.append(
    "-" * 100
)


report_path = (
    OUT
    / "DQ_GATE_03_STATISTICAL_REPORT.txt"
)


report_path.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# =====================================================================
# 12. TERMINAL
# =====================================================================

print()
print("=" * 100)
print("DQ GATE 03 — RESULTADO")
print("=" * 100)

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
    f"INFORMATIONAL     : {infos}"
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
                "dimension",
                "affected_rows",
                "affected_pct",
                "observed"
            ]
        ].to_string(
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
        "[PASS] DQ GATE 03 — "
        "STATISTICAL / COMPLETENESS APROVADO."
    )

    print(
        "Nenhuma transformação foi aplicada aos dados RAW."
    )

    print(
        "Próxima etapa: DQ Gate 04 — Label Quality."
    )

    sys.exit(0)

else:

    print(
        "[FAIL] DQ GATE 03 REPROVADO."
    )

    print(
        "O próximo gate está BLOQUEADO "
        "até revisão das falhas críticas."
    )

    sys.exit(2)

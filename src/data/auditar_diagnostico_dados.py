#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA 03
========================================================================
Delivery Risk Intelligence Platform
Diagnóstico Estatístico, Semântico e Plano de Tratamento — Olist
========================================================================

OBJETIVO
--------
Antes de limpar, transformar, criar Silver ou treinar modelos:

1. auditar a definição correta do target;
2. analisar estatisticamente as variáveis numéricas;
3. identificar missing values e seus padrões;
4. sinalizar extremos sem removê-los;
5. verificar consistência financeira;
6. verificar consistência física dos produtos;
7. auditar geolocalização;
8. analisar categóricas e cardinalidades;
9. estudar temporalidade;
10. estudar sellers;
11. auditar IDs;
12. produzir um RASCUNHO formal do plano de tratamento.

IMPORTANTE
----------
Este script NÃO altera nenhum arquivo em data/raw.
Nenhuma imputação, clipping, winsorization ou exclusão é aplicada.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ======================================================================
# CAMINHOS
# ======================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = PROJECT / "data" / "raw" / "olist"

OUT = (
    PROJECT
    / "reports"
    / "audit"
    / "03_data_diagnosis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


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


# ======================================================================
# LOG
# ======================================================================

lines = []


def log(msg=""):

    print(msg)

    lines.append(
        str(msg)
    )


def section(title):

    log()

    log("=" * 96)

    log(title)

    log("=" * 96)


def pct(n, d):

    if d == 0:
        return np.nan

    return 100.0 * n / d


def safe_ratio(a, b):

    return np.where(
        (
            pd.notna(b)
            &
            (b != 0)
        ),
        a / b,
        np.nan
    )


# ======================================================================
# RESUMO NUMÉRICO ROBUSTO
# ======================================================================

def numeric_summary(
    source,
    variable,
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

    n = len(x)

    nv = len(valid)

    base = {
        "source": source,
        "variable": variable,

        "n_total": n,

        "n_valid": nv,

        "n_missing":
            int(
                x.isna().sum()
            ),

        "missing_pct":
            pct(
                int(
                    x.isna().sum()
                ),
                n
            ),

        "n_zero":
            int(
                (valid == 0).sum()
            )
            if nv
            else 0,

        "n_negative":
            int(
                (valid < 0).sum()
            )
            if nv
            else 0,

        "n_positive":
            int(
                (valid > 0).sum()
            )
            if nv
            else 0,

        "n_unique":
            int(
                valid.nunique()
            )
            if nv
            else 0,
    }

    if nv == 0:

        for key in [
            "min",
            "p01",
            "p05",
            "q1",
            "median",
            "mean",
            "q3",
            "p95",
            "p99",
            "p995",
            "max",
            "std",
            "iqr",
            "mad",
            "skew",
            "kurtosis",
            "iqr_lower",
            "iqr_upper",
            "iqr_candidate_count",
            "iqr_candidate_pct",
            "modified_z_candidate_count",
            "modified_z_candidate_pct",
        ]:

            base[key] = np.nan

        return base

    q = valid.quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            0.995,
        ]
    )

    med = float(
        valid.median()
    )

    q1 = float(
        q.loc[0.25]
    )

    q3 = float(
        q.loc[0.75]
    )

    iqr = (
        q3
        -
        q1
    )

    mad = float(
        np.median(
            np.abs(
                valid.to_numpy(
                    dtype=float
                )
                -
                med
            )
        )
    )

    lower = (
        q1
        -
        1.5 * iqr
    )

    upper = (
        q3
        +
        1.5 * iqr
    )

    iqr_flag = (
        (valid < lower)
        |
        (valid > upper)
    )

    if mad > 0:

        modified_z = (
            0.6745
            *
            (
                valid
                -
                med
            )
            /
            mad
        )

        mz_flag = (
            modified_z.abs()
            >
            3.5
        )

        mz_count = int(
            mz_flag.sum()
        )

    else:

        mz_count = 0

    base.update(
        {
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
                med,

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
                    valid.std(
                        ddof=1
                    )
                )
                if nv > 1
                else 0.0,

            "iqr":
                float(
                    iqr
                ),

            "mad":
                mad,

            "skew":
                float(
                    valid.skew()
                )
                if nv > 2
                else np.nan,

            "kurtosis":
                float(
                    valid.kurt()
                )
                if nv > 3
                else np.nan,

            "iqr_lower":
                float(
                    lower
                ),

            "iqr_upper":
                float(
                    upper
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
                    nv
                ),

            "modified_z_candidate_count":
                mz_count,

            "modified_z_candidate_pct":
                pct(
                    mz_count,
                    nv
                ),
        }
    )

    return base


# ======================================================================
# HAVERSINE
# ======================================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(
        lat1.astype(float)
    )

    lon1 = np.radians(
        lon1.astype(float)
    )

    lat2 = np.radians(
        lat2.astype(float)
    )

    lon2 = np.radians(
        lon2.astype(float)
    )

    dlat = (
        lat2
        -
        lat1
    )

    dlon = (
        lon2
        -
        lon1
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(
            dlon / 2
        ) ** 2
    )

    a = np.clip(
        a,
        0,
        1
    )

    return (
        6371.0088
        *
        2
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


# ======================================================================
# INÍCIO
# ======================================================================

section(
    "AUDITORIA 03 — DIAGNÓSTICO ESTATÍSTICO, SEMÂNTICO E PLANO DE TRATAMENTO"
)

log(
    f"Projeto : {PROJECT}"
)

log(
    f"RAW     : {RAW}"
)

log(
    f"Saídas  : {OUT}"
)


# ======================================================================
# CHECAR ARQUIVOS
# ======================================================================

missing_files = [
    filename
    for filename in FILES.values()
    if not (
        RAW / filename
    ).exists()
]

if missing_files:

    raise SystemExit(
        "Arquivos ausentes no RAW: "
        +
        str(
            missing_files
        )
    )


# ======================================================================
# 1. CARREGAMENTO
# ======================================================================

section(
    "1. CARREGAMENTO"
)

dfs = {}

for name, filename in FILES.items():

    path = (
        RAW
        /
        filename
    )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    dfs[name] = df

    log(
        f"[OK] "
        f"{name:<15} "
        f"{len(df):>10,} linhas x "
        f"{df.shape[1]:>2} colunas"
    )


orders = dfs[
    "orders"
].copy()

items = dfs[
    "order_items"
].copy()

payments = dfs[
    "payments"
].copy()

customers = dfs[
    "customers"
].copy()

products = dfs[
    "products"
].copy()

sellers = dfs[
    "sellers"
].copy()

geo = dfs[
    "geolocation"
].copy()

reviews = dfs[
    "reviews"
].copy()


# ======================================================================
# DATAS
# ======================================================================

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in DATE_COLS:

    orders[col] = pd.to_datetime(
        orders[col],
        errors="coerce"
    )


items[
    "shipping_limit_date"
] = pd.to_datetime(
    items[
        "shipping_limit_date"
    ],
    errors="coerce"
)


# ======================================================================
# 2. TARGET
# ======================================================================

section(
    "2. DEFINIÇÃO DO TARGET — AUDITORIA DE SEMÂNTICA TEMPORAL"
)


cohort_mask = (
    (
        orders[
            "order_status"
        ]
        ==
        "delivered"
    )
    &
    orders[
        "order_purchase_timestamp"
    ].notna()
    &
    orders[
        "order_delivered_customer_date"
    ].notna()
    &
    orders[
        "order_estimated_delivery_date"
    ].notna()
)


cohort = orders.loc[
    cohort_mask
].copy()


# ----------------------------------------------------------------------
# TARGET 1 — comparação de TIMESTAMP
# ----------------------------------------------------------------------

cohort[
    "late_timestamp"
] = (
    cohort[
        "order_delivered_customer_date"
    ]
    >
    cohort[
        "order_estimated_delivery_date"
    ]
).astype(
    "int8"
)


# ----------------------------------------------------------------------
# TARGET 2 — comparação de DIA CALENDÁRIO
# ----------------------------------------------------------------------

actual_day = (
    cohort[
        "order_delivered_customer_date"
    ]
    .dt
    .normalize()
)

promised_day = (
    cohort[
        "order_estimated_delivery_date"
    ]
    .dt
    .normalize()
)


cohort[
    "late_calendar"
] = (
    actual_day
    >
    promised_day
).astype(
    "int8"
)


cohort[
    "same_promised_calendar_day"
] = (
    actual_day
    ==
    promised_day
).astype(
    "int8"
)


cohort[
    "target_disagreement"
] = (
    cohort[
        "late_timestamp"
    ]
    !=
    cohort[
        "late_calendar"
    ]
).astype(
    "int8"
)


# ----------------------------------------------------------------------
# DISTÂNCIAS TEMPORAIS
# ----------------------------------------------------------------------

cohort[
    "delay_calendar_days"
] = (
    actual_day
    -
    promised_day
).dt.days


cohort[
    "delay_exact_days"
] = (
    (
        cohort[
            "order_delivered_customer_date"
        ]
        -
        cohort[
            "order_estimated_delivery_date"
        ]
    )
    .dt
    .total_seconds()
    /
    86400.0
)


cohort[
    "promised_lead_days"
] = (
    (
        cohort[
            "order_estimated_delivery_date"
        ]
        -
        cohort[
            "order_purchase_timestamp"
        ]
    )
    .dt
    .total_seconds()
    /
    86400.0
)


cohort[
    "actual_lead_days"
] = (
    (
        cohort[
            "order_delivered_customer_date"
        ]
        -
        cohort[
            "order_purchase_timestamp"
        ]
    )
    .dt
    .total_seconds()
    /
    86400.0
)


# ----------------------------------------------------------------------
# HORÁRIO PRESENTE NO PRAZO ESTIMADO
# ----------------------------------------------------------------------

estimated_hour_counts = (
    cohort[
        "order_estimated_delivery_date"
    ]
    .dt
    .time
    .astype(str)
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "estimated_delivery_time"
    )
    .reset_index(
        name="count"
    )
)


estimated_hour_counts[
    "pct"
] = (
    100
    *
    estimated_hour_counts[
        "count"
    ]
    /
    len(cohort)
)


estimated_hour_counts.to_csv(
    OUT
    /
    "01_estimated_delivery_time_distribution.csv",
    index=False
)


# ----------------------------------------------------------------------
# COMPARAÇÃO DOS DOIS TARGETS
# ----------------------------------------------------------------------

comparison = pd.DataFrame(
    [
        {
            "definition":
                "timestamp_strict",

            "eligible_orders":
                len(cohort),

            "late_orders":
                int(
                    cohort[
                        "late_timestamp"
                    ].sum()
                ),

            "late_rate_pct":
                100
                *
                cohort[
                    "late_timestamp"
                ].mean(),
        },

        {
            "definition":
                "calendar_date",

            "eligible_orders":
                len(cohort),

            "late_orders":
                int(
                    cohort[
                        "late_calendar"
                    ].sum()
                ),

            "late_rate_pct":
                100
                *
                cohort[
                    "late_calendar"
                ].mean(),
        },
    ]
)


comparison.to_csv(
    OUT
    /
    "02_target_definition_comparison.csv",
    index=False
)


same_day = int(
    cohort[
        "same_promised_calendar_day"
    ].sum()
)

disagree = int(
    cohort[
        "target_disagreement"
    ].sum()
)


log(
    comparison.to_string(
        index=False,
        formatters={
            "late_rate_pct":
                "{:.4f}".format
        }
    )
)

log(
    f"Entregues no mesmo DIA da promessa   : "
    f"{same_day:,}"
)

log(
    f"Discordâncias timestamp x calendário : "
    f"{disagree:,}"
)

log(
    "[DECISÃO PENDENTE] "
    "Não congelar o target antes de revisar essa comparação."
)


# ----------------------------------------------------------------------
# DISTRIBUIÇÃO DOS DIAS ADIANTADO/ATRASADO
# ----------------------------------------------------------------------

delay_bins = [
    -np.inf,
    -14,
    -7,
    -3,
    -1,
    0,
    1,
    3,
    7,
    14,
    30,
    np.inf,
]


delay_labels = [
    "<= -14",
    "-13 a -7",
    "-6 a -3",
    "-2 a -1",
    "0",
    "1",
    "2 a 3",
    "4 a 7",
    "8 a 14",
    "15 a 30",
    "> 30",
]


cohort[
    "delay_band_calendar"
] = pd.cut(
    cohort[
        "delay_calendar_days"
    ],
    bins=delay_bins,
    labels=delay_labels,
    right=True
)


delay_dist = (
    cohort[
        "delay_band_calendar"
    ]
    .value_counts(
        sort=False,
        dropna=False
    )
    .rename_axis(
        "delay_band"
    )
    .reset_index(
        name="orders"
    )
)


delay_dist[
    "pct"
] = (
    100
    *
    delay_dist[
        "orders"
    ]
    /
    len(cohort)
)


delay_dist.to_csv(
    OUT
    /
    "03_delay_distribution_calendar_days.csv",
    index=False
)


# ======================================================================
# 3. AGREGAÇÕES SOMENTE PARA DIAGNÓSTICO
# ======================================================================

section(
    "3. AGREGAÇÃO TRANSACIONAL PARA DIAGNÓSTICO — SEM CRIAR SILVER"
)


item_agg = (
    items
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        item_rows=(
            "order_item_id",
            "size"
        ),

        n_products=(
            "product_id",
            "nunique"
        ),

        n_sellers=(
            "seller_id",
            "nunique"
        ),

        total_item_value=(
            "price",
            "sum"
        ),

        mean_item_price=(
            "price",
            "mean"
        ),

        max_item_price=(
            "price",
            "max"
        ),

        total_freight_value=(
            "freight_value",
            "sum"
        ),

        mean_freight_value=(
            "freight_value",
            "mean"
        ),

        min_shipping_limit_date=(
            "shipping_limit_date",
            "min"
        ),

        max_shipping_limit_date=(
            "shipping_limit_date",
            "max"
        ),
    )
)


payment_agg = (
    payments
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        payment_records=(
            "payment_sequential",
            "size"
        ),

        n_payment_types=(
            "payment_type",
            "nunique"
        ),

        total_payment_value=(
            "payment_value",
            "sum"
        ),

        max_installments=(
            "payment_installments",
            "max"
        ),
    )
)


# ----------------------------------------------------------------------
# PRODUTOS ASSOCIADOS AOS ITENS
# ----------------------------------------------------------------------

product_cols = [
    "product_id",
    "product_category_name",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
]


items_products = (
    items[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
        ]
    ]
    .merge(
        products[
            product_cols
        ],
        on="product_id",
        how="left",
        validate="many_to_one"
    )
)


for col in [
    "product_category_name",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]:

    items_products[
        f"missing_{col}"
    ] = (
        items_products[
            col
        ]
        .isna()
        .astype(
            "int8"
        )
    )


product_missing_order = (
    items_products
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        any_missing_category=(
            "missing_product_category_name",
            "max"
        ),

        any_missing_weight=(
            "missing_product_weight_g",
            "max"
        ),

        any_missing_length=(
            "missing_product_length_cm",
            "max"
        ),

        any_missing_height=(
            "missing_product_height_cm",
            "max"
        ),

        any_missing_width=(
            "missing_product_width_cm",
            "max"
        ),
    )
)


# ----------------------------------------------------------------------
# TABELA TEMPORÁRIA PARA AUDITORIA
# ----------------------------------------------------------------------

order_diag = cohort.merge(
    item_agg,
    on="order_id",
    how="left",
    validate="one_to_one"
)


order_diag = order_diag.merge(
    payment_agg,
    on="order_id",
    how="left",
    validate="one_to_one"
)


order_diag = order_diag.merge(
    customers[
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ]
    ],
    on="customer_id",
    how="left",
    validate="many_to_one"
)


order_diag = order_diag.merge(
    product_missing_order,
    on="order_id",
    how="left",
    validate="one_to_one"
)


order_diag[
    "items_plus_freight"
] = (
    order_diag[
        "total_item_value"
    ]
    +
    order_diag[
        "total_freight_value"
    ]
)


order_diag[
    "payment_delta"
] = (
    order_diag[
        "total_payment_value"
    ]
    -
    order_diag[
        "items_plus_freight"
    ]
)


order_diag[
    "abs_payment_delta"
] = (
    order_diag[
        "payment_delta"
    ].abs()
)


order_diag[
    "freight_to_item_value"
] = safe_ratio(
    order_diag[
        "total_freight_value"
    ],
    order_diag[
        "total_item_value"
    ]
)


log(
    f"Coorte supervisionada          : "
    f"{len(order_diag):,}"
)

log(
    f"Sem item agregado              : "
    f"{order_diag['item_rows'].isna().sum():,}"
)

log(
    f"Sem pagamento agregado         : "
    f"{order_diag['total_payment_value'].isna().sum():,}"
)


# ======================================================================
# 4. PERFIL NUMÉRICO
# ======================================================================

section(
    "4. ESTATÍSTICAS NUMÉRICAS E CANDIDATOS A EXTREMOS"
)


numeric_records = []


raw_numeric = [
    (
        "order_items",
        "price",
        items[
            "price"
        ]
    ),

    (
        "order_items",
        "freight_value",
        items[
            "freight_value"
        ]
    ),

    (
        "payments",
        "payment_value",
        payments[
            "payment_value"
        ]
    ),

    (
        "payments",
        "payment_installments",
        payments[
            "payment_installments"
        ]
    ),

    (
        "products",
        "product_weight_g",
        products[
            "product_weight_g"
        ]
    ),

    (
        "products",
        "product_length_cm",
        products[
            "product_length_cm"
        ]
    ),

    (
        "products",
        "product_height_cm",
        products[
            "product_height_cm"
        ]
    ),

    (
        "products",
        "product_width_cm",
        products[
            "product_width_cm"
        ]
    ),

    (
        "cohort",
        "promised_lead_days",
        cohort[
            "promised_lead_days"
        ]
    ),

    (
        "cohort",
        "actual_lead_days",
        cohort[
            "actual_lead_days"
        ]
    ),

    (
        "cohort",
        "delay_calendar_days",
        cohort[
            "delay_calendar_days"
        ]
    ),

    (
        "cohort",
        "delay_exact_days",
        cohort[
            "delay_exact_days"
        ]
    ),

    (
        "order_agg",
        "item_rows",
        order_diag[
            "item_rows"
        ]
    ),

    (
        "order_agg",
        "n_products",
        order_diag[
            "n_products"
        ]
    ),

    (
        "order_agg",
        "n_sellers",
        order_diag[
            "n_sellers"
        ]
    ),

    (
        "order_agg",
        "total_item_value",
        order_diag[
            "total_item_value"
        ]
    ),

    (
        "order_agg",
        "total_freight_value",
        order_diag[
            "total_freight_value"
        ]
    ),

    (
        "order_agg",
        "total_payment_value",
        order_diag[
            "total_payment_value"
        ]
    ),

    (
        "order_agg",
        "payment_delta",
        order_diag[
            "payment_delta"
        ]
    ),

    (
        "order_agg",
        "freight_to_item_value",
        order_diag[
            "freight_to_item_value"
        ]
    ),
]


# ----------------------------------------------------------------------
# DERIVADAS FÍSICAS
# ----------------------------------------------------------------------

products_diag = (
    products.copy()
)


products_diag[
    "volume_cm3"
] = (
    products_diag[
        "product_length_cm"
    ]
    *
    products_diag[
        "product_height_cm"
    ]
    *
    products_diag[
        "product_width_cm"
    ]
)


products_diag[
    "apparent_density_g_cm3"
] = safe_ratio(
    products_diag[
        "product_weight_g"
    ],
    products_diag[
        "volume_cm3"
    ]
)


raw_numeric.extend(
    [
        (
            "products_derived",
            "volume_cm3",
            products_diag[
                "volume_cm3"
            ]
        ),

        (
            "products_derived",
            "apparent_density_g_cm3",
            products_diag[
                "apparent_density_g_cm3"
            ]
        ),
    ]
)


for (
    src,
    var,
    series
) in raw_numeric:

    numeric_records.append(
        numeric_summary(
            src,
            var,
            series
        )
    )


numeric_df = pd.DataFrame(
    numeric_records
)


numeric_df.to_csv(
    OUT
    /
    "04_numeric_statistical_profile.csv",
    index=False
)


log(
    numeric_df[
        [
            "source",
            "variable",
            "n_missing",
            "n_zero",
            "n_negative",
            "min",
            "median",
            "mean",
            "p99",
            "max",
            "iqr_candidate_count",
            "modified_z_candidate_count",
        ]
    ].to_string(
        index=False
    )
)


log(
    "[NOTA] IQR/MAD apenas SINALIZAM candidatos a extremos; "
    "nenhum valor será removido automaticamente."
)


# ======================================================================
# 5. VALIDADE SEMÂNTICA
# ======================================================================

section(
    "5. REGRAS DE VALIDADE / QUALIDADE SEMÂNTICA"
)


quality_checks = []


def add_check(
    table,
    variable,
    rule,
    count,
    total,
    severity="CHECK"
):

    quality_checks.append(
        {
            "table":
                table,

            "variable":
                variable,

            "rule":
                rule,

            "count":
                int(count),

            "pct":
                pct(
                    int(count),
                    int(total)
                ),

            "severity":
                severity,
        }
    )


add_check(
    "order_items",
    "price",
    "price < 0",
    (
        items[
            "price"
        ]
        <
        0
    ).sum(),
    len(items),
    "INVALID_IF_PRESENT"
)


add_check(
    "order_items",
    "price",
    "price == 0",
    (
        items[
            "price"
        ]
        ==
        0
    ).sum(),
    len(items),
    "REVIEW"
)


add_check(
    "order_items",
    "freight_value",
    "freight_value < 0",
    (
        items[
            "freight_value"
        ]
        <
        0
    ).sum(),
    len(items),
    "INVALID_IF_PRESENT"
)


add_check(
    "order_items",
    "freight_value",
    "freight_value == 0",
    (
        items[
            "freight_value"
        ]
        ==
        0
    ).sum(),
    len(items),
    "PLAUSIBLE_REVIEW"
)


add_check(
    "payments",
    "payment_value",
    "payment_value < 0",
    (
        payments[
            "payment_value"
        ]
        <
        0
    ).sum(),
    len(payments),
    "INVALID_IF_PRESENT"
)


add_check(
    "payments",
    "payment_value",
    "payment_value == 0",
    (
        payments[
            "payment_value"
        ]
        ==
        0
    ).sum(),
    len(payments),
    "REVIEW"
)


add_check(
    "payments",
    "payment_installments",
    "payment_installments < 0",
    (
        payments[
            "payment_installments"
        ]
        <
        0
    ).sum(),
    len(payments),
    "INVALID_IF_PRESENT"
)


add_check(
    "payments",
    "payment_installments",
    "payment_installments == 0",
    (
        payments[
            "payment_installments"
        ]
        ==
        0
    ).sum(),
    len(payments),
    "REVIEW"
)


for c in [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]:

    add_check(
        "products",
        c,
        f"{c} < 0",
        (
            products[c]
            <
            0
        ).sum(),
        len(products),
        "INVALID_IF_PRESENT"
    )

    add_check(
        "products",
        c,
        f"{c} == 0",
        (
            products[c]
            ==
            0
        ).sum(),
        len(products),
        "REVIEW"
    )


add_check(
    "cohort",
    "promised_lead_days",
    "promised_lead_days <= 0",
    (
        cohort[
            "promised_lead_days"
        ]
        <=
        0
    ).sum(),
    len(cohort),
    "REVIEW"
)


add_check(
    "cohort",
    "actual_lead_days",
    "actual_lead_days < 0",
    (
        cohort[
            "actual_lead_days"
        ]
        <
        0
    ).sum(),
    len(cohort),
    "INVALID_IF_PRESENT"
)


quality_df = pd.DataFrame(
    quality_checks
)


quality_df.to_csv(
    OUT
    /
    "05_semantic_quality_checks.csv",
    index=False
)


log(
    quality_df.to_string(
        index=False,
        formatters={
            "pct":
                "{:.4f}".format
        }
    )
)


# ======================================================================
# 6. CONSISTÊNCIA FINANCEIRA
# ======================================================================

section(
    "6. CONSISTÊNCIA FINANCEIRA POR PEDIDO"
)


fin = order_diag[
    [
        "order_id",
        "total_item_value",
        "total_freight_value",
        "items_plus_freight",
        "total_payment_value",
        "payment_delta",
        "abs_payment_delta",
        "late_calendar",
    ]
].copy()


fin[
    "within_1_cent"
] = (
    fin[
        "abs_payment_delta"
    ]
    <=
    0.01
)


fin[
    "within_1_real"
] = (
    fin[
        "abs_payment_delta"
    ]
    <=
    1.00
)


fin[
    "over_10_reais"
] = (
    fin[
        "abs_payment_delta"
    ]
    >
    10.00
)


fin.to_csv(
    OUT
    /
    "06_financial_consistency_by_order.csv",
    index=False
)


valid_fin = (
    fin[
        "abs_payment_delta"
    ]
    .notna()
    .sum()
)


fin_summary = pd.DataFrame(
    [
        {
            "orders":
                len(fin),

            "missing_payment":
                int(
                    fin[
                        "total_payment_value"
                    ]
                    .isna()
                    .sum()
                ),

            "within_1_cent":
                int(
                    fin[
                        "within_1_cent"
                    ].sum()
                ),

            "within_1_cent_pct":
                pct(
                    int(
                        fin[
                            "within_1_cent"
                        ].sum()
                    ),
                    valid_fin
                ),

            "within_1_real":
                int(
                    fin[
                        "within_1_real"
                    ].sum()
                ),

            "within_1_real_pct":
                pct(
                    int(
                        fin[
                            "within_1_real"
                        ].sum()
                    ),
                    valid_fin
                ),

            "over_10_reais":
                int(
                    fin[
                        "over_10_reais"
                    ].sum()
                ),

            "payment_delta_median":
                fin[
                    "payment_delta"
                ].median(),

            "payment_delta_p01":
                fin[
                    "payment_delta"
                ].quantile(
                    0.01
                ),

            "payment_delta_p99":
                fin[
                    "payment_delta"
                ].quantile(
                    0.99
                ),

            "payment_delta_min":
                fin[
                    "payment_delta"
                ].min(),

            "payment_delta_max":
                fin[
                    "payment_delta"
                ].max(),
        }
    ]
)


fin_summary.to_csv(
    OUT
    /
    "07_financial_consistency_summary.csv",
    index=False
)


log(
    fin_summary.to_string(
        index=False
    )
)


log(
    "[NOTA] Diferença financeira é diagnóstico; "
    "não será corrigida sem evidência da semântica do registro."
)


# ======================================================================
# 7. MISSING VALUES
# ======================================================================

section(
    "7. MISSINGNESS — GLOBAL, POR TARGET E POR TEMPO"
)


missing_global = []


for table_name, df in dfs.items():

    for col in df.columns:

        m = int(
            df[
                col
            ]
            .isna()
            .sum()
        )

        missing_global.append(
            {
                "table":
                    table_name,

                "column":
                    col,

                "rows":
                    len(df),

                "missing":
                    m,

                "missing_pct":
                    pct(
                        m,
                        len(df)
                    ),
            }
        )


missing_global_df = (
    pd.DataFrame(
        missing_global
    )
    .sort_values(
        [
            "missing_pct",
            "table"
        ],
        ascending=[
            False,
            True
        ]
    )
)


missing_global_df.to_csv(
    OUT
    /
    "08_missingness_global.csv",
    index=False
)


missing_candidate_cols = [
    "item_rows",
    "n_products",
    "n_sellers",
    "total_item_value",
    "total_freight_value",
    "total_payment_value",
    "max_installments",
    "customer_state",
    "customer_zip_code_prefix",
    "any_missing_category",
    "any_missing_weight",
    "any_missing_length",
    "any_missing_height",
    "any_missing_width",
]


missing_by_target = []


for col in missing_candidate_cols:

    if col not in order_diag.columns:
        continue

    for y in [
        0,
        1
    ]:

        sub = order_diag[
            order_diag[
                "late_calendar"
            ]
            ==
            y
        ]

        m = int(
            sub[
                col
            ]
            .isna()
            .sum()
        )

        missing_by_target.append(
            {
                "column":
                    col,

                "late_calendar":
                    y,

                "rows":
                    len(sub),

                "missing":
                    m,

                "missing_pct":
                    pct(
                        m,
                        len(sub)
                    ),
            }
        )


missing_by_target_df = pd.DataFrame(
    missing_by_target
)


missing_by_target_df.to_csv(
    OUT
    /
    "09_missingness_by_target_calendar.csv",
    index=False
)


order_diag[
    "purchase_month"
] = (
    order_diag[
        "order_purchase_timestamp"
    ]
    .dt
    .to_period(
        "M"
    )
    .astype(str)
)


missing_month_cols = [
    "total_payment_value",
    "any_missing_category",
    "any_missing_weight",
]


rows = []


for (
    month,
    sub
) in order_diag.groupby(
    "purchase_month"
):

    for col in missing_month_cols:

        if col.startswith(
            "any_"
        ):

            qtd = int(
                sub[col]
                .fillna(1)
                .astype(float)
                .gt(0)
                .sum()
            )

        else:

            qtd = int(
                sub[col]
                .isna()
                .sum()
            )

        rows.append(
            {
                "purchase_month":
                    month,

                "column":
                    col,

                "orders":
                    len(sub),

                "missing_or_flagged":
                    qtd,
            }
        )


missing_month_df = pd.DataFrame(
    rows
)


missing_month_df[
    "pct"
] = (
    100
    *
    missing_month_df[
        "missing_or_flagged"
    ]
    /
    missing_month_df[
        "orders"
    ]
)


missing_month_df.to_csv(
    OUT
    /
    "10_missingness_by_month.csv",
    index=False
)


log(
    "[OK] Missingness auditada globalmente, "
    "por target candidato e por mês."
)


# ======================================================================
# 8. GEOLOCALIZAÇÃO
# ======================================================================

section(
    "8. GEOLOCALIZAÇÃO — VALIDADE, DUPLICIDADE E COBERTURA"
)


lat_invalid = (
    geo[
        "geolocation_lat"
    ]
    .notna()
    &
    ~geo[
        "geolocation_lat"
    ]
    .between(
        -90,
        90
    )
)


lng_invalid = (
    geo[
        "geolocation_lng"
    ]
    .notna()
    &
    ~geo[
        "geolocation_lng"
    ]
    .between(
        -180,
        180
    )
)


geo_zip = (
    geo
    .groupby(
        "geolocation_zip_code_prefix",
        as_index=False
    )
    .agg(
        geo_points=(
            "geolocation_lat",
            "size"
        ),

        lat_median=(
            "geolocation_lat",
            "median"
        ),

        lng_median=(
            "geolocation_lng",
            "median"
        ),

        lat_min=(
            "geolocation_lat",
            "min"
        ),

        lat_max=(
            "geolocation_lat",
            "max"
        ),

        lng_min=(
            "geolocation_lng",
            "min"
        ),

        lng_max=(
            "geolocation_lng",
            "max"
        ),

        n_geo_cities=(
            "geolocation_city",
            "nunique"
        ),

        n_geo_states=(
            "geolocation_state",
            "nunique"
        ),
    )
)


geo_zip[
    "lat_range"
] = (
    geo_zip[
        "lat_max"
    ]
    -
    geo_zip[
        "lat_min"
    ]
)


geo_zip[
    "lng_range"
] = (
    geo_zip[
        "lng_max"
    ]
    -
    geo_zip[
        "lng_min"
    ]
)


geo_zip.to_csv(
    OUT
    /
    "11_geolocation_zip_profile.csv",
    index=False
)


# ----------------------------------------------------------------------
# COBERTURA DE CLIENTES
# ----------------------------------------------------------------------

customer_geo = (
    customers[
        [
            "customer_id",
            "customer_zip_code_prefix",
        ]
    ]
    .merge(
        geo_zip[
            [
                "geolocation_zip_code_prefix",
                "lat_median",
                "lng_median",
            ]
        ],

        left_on=
            "customer_zip_code_prefix",

        right_on=
            "geolocation_zip_code_prefix",

        how="left",

        validate="many_to_one"
    )
    .rename(
        columns={
            "lat_median":
                "customer_lat",

            "lng_median":
                "customer_lng",
        }
    )
)


# ----------------------------------------------------------------------
# COBERTURA DE SELLERS
# ----------------------------------------------------------------------

seller_geo = (
    sellers[
        [
            "seller_id",
            "seller_zip_code_prefix",
        ]
    ]
    .merge(
        geo_zip[
            [
                "geolocation_zip_code_prefix",
                "lat_median",
                "lng_median",
            ]
        ],

        left_on=
            "seller_zip_code_prefix",

        right_on=
            "geolocation_zip_code_prefix",

        how="left",

        validate="many_to_one"
    )
    .rename(
        columns={
            "lat_median":
                "seller_lat",

            "lng_median":
                "seller_lng",
        }
    )
)


geo_summary = pd.DataFrame(
    [
        {
            "geo_rows":
                len(geo),

            "exact_duplicate_rows":
                int(
                    geo
                    .duplicated()
                    .sum()
                ),

            "unique_zip_prefixes":
                int(
                    geo[
                        "geolocation_zip_code_prefix"
                    ]
                    .nunique()
                ),

            "invalid_lat_rows":
                int(
                    lat_invalid.sum()
                ),

            "invalid_lng_rows":
                int(
                    lng_invalid.sum()
                ),

            "zip_with_multiple_states":
                int(
                    (
                        geo_zip[
                            "n_geo_states"
                        ]
                        >
                        1
                    ).sum()
                ),

            "zip_with_multiple_cities":
                int(
                    (
                        geo_zip[
                            "n_geo_cities"
                        ]
                        >
                        1
                    ).sum()
                ),

            "customers_without_geo":
                int(
                    customer_geo[
                        "customer_lat"
                    ]
                    .isna()
                    .sum()
                ),

            "sellers_without_geo":
                int(
                    seller_geo[
                        "seller_lat"
                    ]
                    .isna()
                    .sum()
                ),
        }
    ]
)


geo_summary.to_csv(
    OUT
    /
    "12_geolocation_quality_summary.csv",
    index=False
)


log(
    geo_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------------------
# DISTÂNCIA SELLER → CLIENTE
# SOMENTE PARA DIAGNÓSTICO
# ----------------------------------------------------------------------

order_customer_geo = (
    cohort[
        [
            "order_id",
            "customer_id",
        ]
    ]
    .merge(
        customer_geo[
            [
                "customer_id",
                "customer_lat",
                "customer_lng",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )
)


order_seller = (
    items[
        [
            "order_id",
            "seller_id",
        ]
    ]
    .drop_duplicates()
)


order_seller_geo = (
    order_seller
    .merge(
        seller_geo[
            [
                "seller_id",
                "seller_lat",
                "seller_lng",
            ]
        ],
        on="seller_id",
        how="left",
        validate="many_to_one"
    )
    .merge(
        order_customer_geo,
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
)


valid_geo_pair = (
    order_seller_geo[
        [
            "seller_lat",
            "seller_lng",
            "customer_lat",
            "customer_lng",
        ]
    ]
    .notna()
    .all(
        axis=1
    )
)


order_seller_geo[
    "distance_km"
] = np.nan


order_seller_geo.loc[
    valid_geo_pair,
    "distance_km"
] = haversine_km(

    order_seller_geo.loc[
        valid_geo_pair,
        "seller_lat"
    ],

    order_seller_geo.loc[
        valid_geo_pair,
        "seller_lng"
    ],

    order_seller_geo.loc[
        valid_geo_pair,
        "customer_lat"
    ],

    order_seller_geo.loc[
        valid_geo_pair,
        "customer_lng"
    ],
)


order_distance = (
    order_seller_geo
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        seller_customer_distance_min_km=(
            "distance_km",
            "min"
        ),

        seller_customer_distance_mean_km=(
            "distance_km",
            "mean"
        ),

        seller_customer_distance_max_km=(
            "distance_km",
            "max"
        ),

        seller_pairs=(
            "seller_id",
            "nunique"
        ),
    )
)


order_diag = order_diag.merge(
    order_distance,
    on="order_id",
    how="left",
    validate="one_to_one"
)


distance_profile = pd.DataFrame(
    [
        numeric_summary(
            "geo_derived",
            "seller_customer_distance_mean_km",
            order_diag[
                "seller_customer_distance_mean_km"
            ]
        ),

        numeric_summary(
            "geo_derived",
            "seller_customer_distance_max_km",
            order_diag[
                "seller_customer_distance_max_km"
            ]
        ),
    ]
)


distance_profile.to_csv(
    OUT
    /
    "13_distance_statistical_profile.csv",
    index=False
)


log(
    f"Pedidos sem distância calculável : "
    f"{order_diag['seller_customer_distance_mean_km'].isna().sum():,}"
)


# ======================================================================
# 9. NUMÉRICAS X TARGET
# ======================================================================

section(
    "9. ANÁLISE BIVARIADA NUMÉRICA — TARGET CALENDÁRIO CANDIDATO"
)


bivariate_numeric_cols = [
    "promised_lead_days",
    "actual_lead_days",
    "total_item_value",
    "total_freight_value",
    "freight_to_item_value",
    "item_rows",
    "n_products",
    "n_sellers",
    "total_payment_value",
    "max_installments",
    "seller_customer_distance_mean_km",
]


biv_rows = []


for col in bivariate_numeric_cols:

    if col not in order_diag.columns:
        continue

    for y in [
        0,
        1
    ]:

        s = pd.to_numeric(
            order_diag.loc[
                order_diag[
                    "late_calendar"
                ]
                ==
                y,
                col
            ],
            errors="coerce"
        ).dropna()

        biv_rows.append(
            {
                "variable":
                    col,

                "late_calendar":
                    y,

                "n":
                    len(s),

                "mean":
                    s.mean()
                    if len(s)
                    else np.nan,

                "median":
                    s.median()
                    if len(s)
                    else np.nan,

                "q1":
                    s.quantile(
                        0.25
                    )
                    if len(s)
                    else np.nan,

                "q3":
                    s.quantile(
                        0.75
                    )
                    if len(s)
                    else np.nan,

                "std":
                    s.std()
                    if len(s) > 1
                    else np.nan,
            }
        )


biv_df = pd.DataFrame(
    biv_rows
)


biv_df.to_csv(
    OUT
    /
    "14_numeric_by_target_calendar.csv",
    index=False
)


log(
    "[OK] Comparações numéricas por classe salvas."
)


# ======================================================================
# 10. CATEGÓRICAS
# ======================================================================

section(
    "10. CATEGÓRICAS — CARDINALIDADE, FREQUÊNCIA E TAXA DE ATRASO"
)


cat_outputs = []


def category_target_table(
    df,
    col,
    target="late_calendar",
    order_col="order_id"
):

    tmp = (
        df[
            [
                order_col,
                col,
                target
            ]
        ]
        .dropna(
            subset=[
                col
            ]
        )
        .drop_duplicates(
            [
                order_col,
                col
            ]
        )
    )

    out = (
        tmp
        .groupby(
            col,
            dropna=False
        )
        .agg(
            orders=(
                order_col,
                "nunique"
            ),

            late_orders=(
                target,
                "sum"
            ),

            late_rate_pct=(
                target,
                lambda x:
                    100
                    *
                    x.mean()
            ),
        )
        .reset_index()
    )

    out[
        "share_of_order_category_pairs_pct"
    ] = (
        100
        *
        out[
            "orders"
        ]
        /
        out[
            "orders"
        ].sum()
    )

    return out.sort_values(
        [
            "orders",
            col
        ],
        ascending=[
            False,
            True
        ]
    )


# ----------------------------------------------------------------------
# ESTADO DO CLIENTE
# ----------------------------------------------------------------------

cust_cat = order_diag[
    [
        "order_id",
        "customer_state",
        "late_calendar",
    ]
].copy()


customer_state_rates = (
    category_target_table(
        cust_cat,
        "customer_state"
    )
)


customer_state_rates.to_csv(
    OUT
    /
    "15_customer_state_late_rate.csv",
    index=False
)


cat_outputs.append(
    (
        "customer_state",

        order_diag[
            "customer_state"
        ].nunique(
            dropna=True
        ),

        len(
            customer_state_rates
        )
    )
)


# ----------------------------------------------------------------------
# PAYMENT TYPE
# ----------------------------------------------------------------------

pay_cat = (
    payments[
        [
            "order_id",
            "payment_type",
        ]
    ]
    .merge(
        cohort[
            [
                "order_id",
                "late_calendar",
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
)


payment_type_rates = (
    category_target_table(
        pay_cat,
        "payment_type"
    )
)


payment_type_rates.to_csv(
    OUT
    /
    "16_payment_type_late_rate.csv",
    index=False
)


cat_outputs.append(
    (
        "payment_type",

        payments[
            "payment_type"
        ].nunique(
            dropna=True
        ),

        len(
            payment_type_rates
        )
    )
)


# ----------------------------------------------------------------------
# CATEGORIA DO PRODUTO
# ----------------------------------------------------------------------

item_cat = (
    items_products[
        [
            "order_id",
            "product_category_name",
        ]
    ]
    .merge(
        cohort[
            [
                "order_id",
                "late_calendar",
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
)


category_rates = (
    category_target_table(
        item_cat,
        "product_category_name"
    )
)


category_rates.to_csv(
    OUT
    /
    "17_product_category_late_rate.csv",
    index=False
)


cat_outputs.append(
    (
        "product_category_name",

        products[
            "product_category_name"
        ].nunique(
            dropna=True
        ),

        len(
            category_rates
        )
    )
)


# ----------------------------------------------------------------------
# ESTADO DO SELLER
# ----------------------------------------------------------------------

seller_state_map = sellers[
    [
        "seller_id",
        "seller_state",
    ]
]


seller_cat = (
    items[
        [
            "order_id",
            "seller_id",
        ]
    ]
    .merge(
        seller_state_map,
        on="seller_id",
        how="left",
        validate="many_to_one"
    )
)


seller_cat = seller_cat.merge(
    cohort[
        [
            "order_id",
            "late_calendar",
        ]
    ],
    on="order_id",
    how="inner",
    validate="many_to_one"
)


seller_state_rates = (
    category_target_table(
        seller_cat,
        "seller_state"
    )
)


seller_state_rates.to_csv(
    OUT
    /
    "18_seller_state_late_rate.csv",
    index=False
)


cat_outputs.append(
    (
        "seller_state",

        sellers[
            "seller_state"
        ].nunique(
            dropna=True
        ),

        len(
            seller_state_rates
        )
    )
)


cat_summary = pd.DataFrame(
    cat_outputs,
    columns=[
        "variable",
        "raw_cardinality",
        "groups_in_rate_table",
    ]
)


cat_summary.to_csv(
    OUT
    /
    "19_categorical_cardinality_summary.csv",
    index=False
)


log(
    cat_summary.to_string(
        index=False
    )
)


log(
    "[NOTA] Payment/category/seller-state podem ter múltiplos "
    "valores no mesmo pedido; os grupos não são necessariamente aditivos."
)


# ======================================================================
# 11. TEMPORALIDADE
# ======================================================================

section(
    "11. TEMPORALIDADE DO TARGET CANDIDATO"
)


order_diag[
    "purchase_month"
] = (
    order_diag[
        "order_purchase_timestamp"
    ]
    .dt
    .to_period(
        "M"
    )
    .astype(str)
)


order_diag[
    "purchase_weekday"
] = (
    order_diag[
        "order_purchase_timestamp"
    ]
    .dt
    .day_name()
)


order_diag[
    "purchase_hour"
] = (
    order_diag[
        "order_purchase_timestamp"
    ]
    .dt
    .hour
)


monthly = (
    order_diag
    .groupby(
        "purchase_month",
        as_index=False
    )
    .agg(
        orders=(
            "order_id",
            "size"
        ),

        late_timestamp_rate_pct=(
            "late_timestamp",
            lambda x:
                100
                *
                x.mean()
        ),

        late_calendar_rate_pct=(
            "late_calendar",
            lambda x:
                100
                *
                x.mean()
        ),

        promised_lead_median_days=(
            "promised_lead_days",
            "median"
        ),

        promised_lead_mean_days=(
            "promised_lead_days",
            "mean"
        ),
    )
)


monthly.to_csv(
    OUT
    /
    "20_target_by_month.csv",
    index=False
)


weekday = (
    order_diag
    .groupby(
        "purchase_weekday",
        as_index=False
    )
    .agg(
        orders=(
            "order_id",
            "size"
        ),

        late_calendar_rate_pct=(
            "late_calendar",
            lambda x:
                100
                *
                x.mean()
        ),
    )
)


weekday.to_csv(
    OUT
    /
    "21_target_by_purchase_weekday.csv",
    index=False
)


hourly = (
    order_diag
    .groupby(
        "purchase_hour",
        as_index=False
    )
    .agg(
        orders=(
            "order_id",
            "size"
        ),

        late_calendar_rate_pct=(
            "late_calendar",
            lambda x:
                100
                *
                x.mean()
        ),
    )
)


hourly.to_csv(
    OUT
    /
    "22_target_by_purchase_hour.csv",
    index=False
)


log(
    monthly.to_string(
        index=False,

        formatters={
            "late_timestamp_rate_pct":
                "{:.2f}".format,

            "late_calendar_rate_pct":
                "{:.2f}".format,

            "promised_lead_median_days":
                "{:.2f}".format,

            "promised_lead_mean_days":
                "{:.2f}".format,
        }
    )
)


# ======================================================================
# 12. SELLERS
# ======================================================================

section(
    "12. SELLERS — ESTABILIDADE AMOSTRAL (SOMENTE EDA)"
)


seller_order = (
    items[
        [
            "order_id",
            "seller_id",
        ]
    ]
    .drop_duplicates()
    .merge(
        cohort[
            [
                "order_id",
                "order_purchase_timestamp",
                "late_calendar",
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
)


seller_stats = (
    seller_order
    .groupby(
        "seller_id",
        as_index=False
    )
    .agg(
        orders=(
            "order_id",
            "nunique"
        ),

        late_orders=(
            "late_calendar",
            "sum"
        ),

        late_rate_pct=(
            "late_calendar",
            lambda x:
                100
                *
                x.mean()
        ),

        first_purchase=(
            "order_purchase_timestamp",
            "min"
        ),

        last_purchase=(
            "order_purchase_timestamp",
            "max"
        ),
    )
)


seller_stats[
    "volume_band"
] = pd.cut(
    seller_stats[
        "orders"
    ],

    bins=[
        0,
        1,
        5,
        20,
        100,
        np.inf
    ],

    labels=[
        "1",
        "2-5",
        "6-20",
        "21-100",
        ">100"
    ]
)


seller_stats.to_csv(
    OUT
    /
    "23_seller_full_period_eda.csv",
    index=False
)


seller_volume = (
    seller_stats
    .groupby(
        "volume_band",
        observed=False,
        as_index=False
    )
    .agg(
        sellers=(
            "seller_id",
            "size"
        ),

        orders_median=(
            "orders",
            "median"
        ),

        late_rate_median_pct=(
            "late_rate_pct",
            "median"
        ),
    )
)


seller_volume.to_csv(
    OUT
    /
    "24_seller_volume_stability.csv",
    index=False
)


log(
    seller_volume.to_string(
        index=False
    )
)


log(
    "[LEAKAGE] O late_rate acima é SOMENTE EDA usando o período completo. "
    "Não poderá virar feature dessa forma."
)


# ======================================================================
# 13. IDs E CARDINALIDADE
# ======================================================================

section(
    "13. IDs E ALTA CARDINALIDADE"
)


id_candidates = [
    (
        "orders",
        "order_id",
        orders[
            "order_id"
        ]
    ),

    (
        "orders",
        "customer_id",
        orders[
            "customer_id"
        ]
    ),

    (
        "customers",
        "customer_unique_id",
        customers[
            "customer_unique_id"
        ]
    ),

    (
        "order_items",
        "product_id",
        items[
            "product_id"
        ]
    ),

    (
        "order_items",
        "seller_id",
        items[
            "seller_id"
        ]
    ),

    (
        "reviews",
        "review_id",
        reviews[
            "review_id"
        ]
    ),
]


id_rows = []


for (
    src,
    col,
    s
) in id_candidates:

    unique = s.nunique(
        dropna=True
    )

    id_rows.append(
        {
            "source":
                src,

            "variable":
                col,

            "rows":
                len(s),

            "unique":
                unique,

            "unique_ratio_pct":
                100
                *
                unique
                /
                len(s)
                if len(s)
                else np.nan,

            "direct_model_feature_recommendation":
                "NO_RAW_ID",
        }
    )


id_df = pd.DataFrame(
    id_rows
)


id_df.to_csv(
    OUT
    /
    "25_id_cardinality.csv",
    index=False
)


log(
    id_df.to_string(
        index=False,

        formatters={
            "unique_ratio_pct":
                "{:.2f}".format
        }
    )
)


# ======================================================================
# 14. PLANO DE TRATAMENTO — RASCUNHO
# ======================================================================

section(
    "14. PLANO DE TRATAMENTO — RASCUNHO PARA REVISÃO, NÃO APLICADO"
)


treatment_plan = [

    [
        "orders",
        "order_id",
        "identifier",
        "KEEP_AS_KEY_ONLY",
        "Chave técnica; nunca como feature bruta",
    ],

    [
        "orders",
        "customer_id",
        "high_cardinality_id",
        "KEY_JOIN_ONLY",
        "Não usar ID bruto no modelo",
    ],

    [
        "customers",
        "customer_unique_id",
        "high_cardinality_id",
        "EDA_OR_HISTORY_ONLY",
        "Somente se houver hipótese clara e construção point-in-time",
    ],

    [
        "order_items",
        "product_id",
        "high_cardinality_id",
        "NO_RAW_MODEL_ID",
        "Preferir atributos do produto",
    ],

    [
        "order_items",
        "seller_id",
        "high_cardinality_id",
        "HISTORICAL_FEATURES_LATER",
        "Histórico deverá usar apenas informação anterior ao pedido",
    ],

    [
        "orders",
        "order_purchase_timestamp",
        "prediction_time",
        "KEEP_DERIVE_CALENDAR",
        "t0 candidato; derivar calendário e sazonalidade",
    ],

    [
        "orders",
        "order_estimated_delivery_date",
        "promise",
        "KEEP_DERIVE_PROMISED_LEAD",
        "Revisar semântica de data antes de congelar target",
    ],

    [
        "orders",
        "order_delivered_customer_date",
        "future_label",
        "TARGET_ONLY",
        "Informação futura; proibida como feature",
    ],

    [
        "orders",
        "order_delivered_carrier_date",
        "future_event",
        "EXCLUDE_FEATURE",
        "Informação posterior ao instante da compra",
    ],

    [
        "orders",
        "order_approved_at",
        "post_purchase_event",
        "EXCLUDE_BASELINE_FEATURE",
        "Posterior ao t0 e apresenta anomalias temporais",
    ],

    [
        "order_items",
        "shipping_limit_date",
        "provenance_uncertain",
        "HOLD_PROVENANCE_CHECK",
        "Somente liberar se confirmarmos disponibilidade no t0",
    ],

    [
        "order_items",
        "price",
        "numeric",
        "KEEP_DIAGNOSE_TAIL",
        "Não remover extremos automaticamente",
    ],

    [
        "order_items",
        "freight_value",
        "numeric",
        "KEEP_DIAGNOSE_TAIL",
        "Frete zero pode ser plausível; negativos seriam inválidos",
    ],

    [
        "payments",
        "payment_value",
        "numeric_multirecord",
        "AGGREGATE_BY_ORDER",
        "Somar por pedido antes do join final",
    ],

    [
        "payments",
        "payment_type",
        "categorical_multirecord",
        "AGGREGATE_MULTI_PAYMENT",
        "Um pedido pode ter vários registros/tipos",
    ],

    [
        "payments",
        "payment_installments",
        "numeric_multirecord",
        "AGGREGATE_BY_ORDER",
        "Revisar zeros antes da regra definitiva",
    ],

    [
        "products",
        "product_category_name",
        "categorical_missing",
        "MISSING_AS_UNKNOWN_CANDIDATE",
        "Decisão final somente após revisar padrão de missing",
    ],

    [
        "products",
        "product_weight_g",
        "numeric_missing",
        "IMPUTE_TRAIN_ONLY_PLUS_FLAG_CANDIDATE",
        "Não aprender imputação usando validation/test",
    ],

    [
        "products",
        "product_length_cm",
        "numeric_missing",
        "IMPUTE_TRAIN_ONLY_PLUS_FLAG_CANDIDATE",
        "Não aprender imputação usando validation/test",
    ],

    [
        "products",
        "product_height_cm",
        "numeric_missing",
        "IMPUTE_TRAIN_ONLY_PLUS_FLAG_CANDIDATE",
        "Não aprender imputação usando validation/test",
    ],

    [
        "products",
        "product_width_cm",
        "numeric_missing",
        "IMPUTE_TRAIN_ONLY_PLUS_FLAG_CANDIDATE",
        "Não aprender imputação usando validation/test",
    ],

    [
        "geolocation",
        "geolocation_zip_code_prefix",
        "many_rows_per_zip",
        "AGGREGATE_MEDIAN_BY_ZIP",
        "Nunca fazer join direto das ~1M linhas",
    ],

    [
        "geolocation",
        "lat/lng",
        "geospatial",
        "VALIDATE_THEN_DISTANCE",
        "Consolidar coordenada antes de calcular Haversine",
    ],

    [
        "customers",
        "customer_state",
        "categorical",
        "KEEP",
        "Informação geográfica disponível no pedido",
    ],

    [
        "sellers",
        "seller_state",
        "categorical",
        "AGGREGATE_MULTI_SELLER",
        "Pedido pode conter mais de um seller",
    ],

    [
        "reviews",
        "review_score/comments",
        "post_delivery",
        "EXCLUDE_MAIN_MODEL",
        "Reservar para módulo NLP pós-entrega",
    ],

    [
        "translation",
        "product_category_name_english",
        "presentation",
        "PRESENTATION_ONLY",
        "Não excluir produtos por falta de tradução",
    ],
]


treatment_df = pd.DataFrame(
    treatment_plan,

    columns=[
        "table",
        "variable",
        "issue_type",
        "proposed_action",
        "rationale",
    ]
)


treatment_df.to_csv(
    OUT
    /
    "26_data_treatment_plan_DRAFT.csv",
    index=False
)


log(
    treatment_df.to_string(
        index=False
    )
)


# ======================================================================
# 15. DECISÕES PENDENTES
# ======================================================================

section(
    "15. CHECKLIST DE DECISÕES QUE AINDA NÃO PODEM SER AUTOMATIZADAS"
)


pending = {

    "target_definition":
        (
            "Comparar timestamp_strict versus calendar_date "
            "e decidir a semântica correta da promessa."
        ),

    "outliers":
        (
            "Investigar extremos plausíveis e suspeitos. "
            "Nenhuma exclusão automática."
        ),

    "numeric_transformations":
        (
            "Log, clipping ou winsorization somente após EDA. "
            "Se adotados, parâmetros serão ajustados somente no treino."
        ),

    "numeric_imputation":
        (
            "Método e estatísticas somente depois do split. "
            "Fit apenas no conjunto de treino."
        ),

    "shipping_limit_date":
        (
            "Confirmar se a informação existia no instante t0 "
            "antes de liberar como feature."
        ),

    "seller_history":
        (
            "Definir janela, smoothing e construção point-in-time "
            "sem utilizar o futuro."
        ),

    "multi_seller_orders":
        (
            "Definir regra definitiva para sellers e distâncias "
            "quando um pedido possui vários sellers."
        ),

    "payment_inconsistencies":
        (
            "Revisar a distribuição de payment_delta "
            "antes de qualquer correção ou exclusão."
        ),

    "geolocation_spread":
        (
            "Inspecionar CEPs com vários estados/cidades "
            "ou grande dispersão antes da consolidação definitiva."
        ),
}


with open(
    OUT
    /
    "27_pending_data_decisions.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        pending,
        f,
        ensure_ascii=False,
        indent=2
    )


for (
    k,
    v
) in pending.items():

    log(
        f"- {k}: {v}"
    )


# ======================================================================
# 16. RESULTADO
# ======================================================================

section(
    "16. RESULTADO"
)


log(
    "[OK] Auditoria somente leitura: "
    "nenhum CSV em data/raw foi modificado."
)


log(
    f"Arquivos gerados em: {OUT}"
)


for path in sorted(
    OUT.glob("*")
):

    log(
        f"  - {path.name}"
    )


report_path = (
    OUT
    /
    "AUDITORIA_03_RELATORIO.txt"
)


report_path.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


log()

log(
    f"Relatório principal: "
    f"{report_path}"
)


log(
    "[PRÓXIMA ETAPA] "
    "Revisar os resultados e aprovar o plano de tratamento "
    "ANTES de criar a camada Silver."
)

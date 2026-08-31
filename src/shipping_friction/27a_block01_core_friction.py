#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODULE 27A — BLOCK 01
SHIPPING FRICTION CORE METRICS

Objetivo
--------
Materializar, sem novo treinamento:

1. Population / coverage audit
2. Freight Burden
3. Freight-to-Price Ratio
4. Freight Cost Anomaly — log observed / expected
5. Monthly Robust Standardized Cost Anomaly

Não:
- retreina expected freight;
- retreina GN_EQ0;
- altera RAW;
- altera artefatos científicos anteriores;
- cria matriz N x N;
- faz afirmação causal.

Entrada principal:
artifacts/spatiotemporal_logistics/06_EXPECTED_FREIGHT_OOT.csv
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

INPUT = (
    ROOT
    / "artifacts"
    / "spatiotemporal_logistics"
    / "06_EXPECTED_FREIGHT_OOT.csv"
)

ART = ROOT / "artifacts" / "shipping_friction"
REP = ROOT / "reports" / "shipping_friction"

ART.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

OUT_CORE = ART / "27a_01_shipping_friction_core.csv"

OUT_COVERAGE = REP / "27a_01_coverage.csv"
OUT_MONTH = REP / "27a_01_monthly_robust_scale.csv"
OUT_QUANTILES = REP / "27a_01_core_quantiles.csv"
OUT_DECISION = REP / "27a_block01_decision.json"
OUT_REPORT = REP / "27a_block01_report.txt"


# =============================================================================
# CONFIG
# =============================================================================

EPS = 1e-8
MAD_NORMAL_SCALE = 1.4826

USECOLS = [
    "order_id",
    "order_purchase_timestamp",
    "year_month",
    "total_price",
    "total_freight",
    "expected_freight_oot",
]


# =============================================================================
# HELPERS
# =============================================================================

def pct(n, d):
    return float(100.0 * n / d) if d else np.nan


def safe_float(x):
    if pd.isna(x):
        return None
    return float(x)


def robust_month_stats(group):
    z = group["freight_log_ratio"].dropna()

    if len(z) == 0:
        return pd.Series({
            "n_valid": 0,
            "median_log_ratio": np.nan,
            "mad_log_ratio": np.nan,
            "scaled_mad": np.nan,
        })

    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))

    return pd.Series({
        "n_valid": int(len(z)),
        "median_log_ratio": med,
        "mad_log_ratio": mad,
        "scaled_mad": MAD_NORMAL_SCALE * mad,
    })


# =============================================================================
# START
# =============================================================================

print("=" * 100)
print("27A — BLOCK 01 — SHIPPING FRICTION CORE METRICS")
print("=" * 100)
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("N_X_N_MATRIX = false")
print()


if not INPUT.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT}"
    )


# =============================================================================
# 1. SCHEMA CHECK
# =============================================================================

header = pd.read_csv(INPUT, nrows=0)

missing = [
    c for c in USECOLS
    if c not in header.columns
]

if missing:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em "
        f"06_EXPECTED_FREIGHT_OOT.csv: {missing}"
    )

print("[PASS] required columns found")


# =============================================================================
# 2. LOAD ONLY REQUIRED COLUMNS
# =============================================================================

df = pd.read_csv(
    INPUT,
    usecols=USECOLS,
    low_memory=False,
)

n = len(df)

print(f"[INFO] rows loaded = {n:,}")
print(f"[INFO] columns loaded = {len(df.columns)}")


if n == 0:
    raise RuntimeError("Base analítica vazia.")


# =============================================================================
# 3. BASIC INTEGRITY
# =============================================================================

duplicate_orders = int(
    df["order_id"].duplicated().sum()
)

if duplicate_orders != 0:
    raise RuntimeError(
        f"order_id duplicado: {duplicate_orders}"
    )


df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"],
    errors="coerce",
)


# Preserve supplied year_month.
df["year_month"] = df["year_month"].astype("string")


# =============================================================================
# 4. VALIDITY FLAGS
# =============================================================================

df["valid_price"] = (
    df["total_price"].notna()
    & np.isfinite(df["total_price"])
    & (df["total_price"] > 0)
).astype("int8")


df["valid_freight"] = (
    df["total_freight"].notna()
    & np.isfinite(df["total_freight"])
    & (df["total_freight"] >= 0)
).astype("int8")


df["valid_expected_freight"] = (
    df["expected_freight_oot"].notna()
    & np.isfinite(df["expected_freight_oot"])
    & (df["expected_freight_oot"] > 0)
).astype("int8")


df["valid_year_month"] = (
    df["year_month"].notna()
).astype("int8")


# =============================================================================
# 5. METRIC 1 — FREIGHT BURDEN
#
# B_i = Freight / (Price + Freight)
# =============================================================================

den = (
    df["total_price"]
    + df["total_freight"]
)

valid_burden = (
    (df["valid_price"] == 1)
    & (df["valid_freight"] == 1)
    & den.notna()
    & np.isfinite(den)
    & (den > 0)
)

df["freight_burden"] = np.where(
    valid_burden,
    df["total_freight"] / den,
    np.nan,
)


# Mathematical guardrail.
bad_burden = (
    df["freight_burden"].notna()
    & (
        (df["freight_burden"] < 0)
        | (df["freight_burden"] > 1)
    )
)

if bad_burden.any():
    raise RuntimeError(
        "Freight burden fora de [0,1]."
    )


# =============================================================================
# 6. METRIC 2 — FREIGHT / PRICE
#
# R_i = Freight / Price
# =============================================================================

valid_fp = (
    (df["valid_price"] == 1)
    & (df["valid_freight"] == 1)
)

df["freight_to_price_ratio"] = np.where(
    valid_fp,
    df["total_freight"] / df["total_price"],
    np.nan,
)


df["log_freight_to_price_ratio"] = np.where(
    valid_fp,
    np.log(
        (df["total_freight"] + EPS)
        /
        (df["total_price"] + EPS)
    ),
    np.nan,
)


# =============================================================================
# 7. METRIC 3 — ABSOLUTE FREIGHT RESIDUAL
#
# e_i = FreightObserved - FreightExpected
# =============================================================================

valid_expected_pair = (
    (df["valid_freight"] == 1)
    & (df["valid_expected_freight"] == 1)
)

df["freight_residual_abs"] = np.where(
    valid_expected_pair,
    (
        df["total_freight"]
        -
        df["expected_freight_oot"]
    ),
    np.nan,
)


# =============================================================================
# 8. METRIC 4 — RELATIVE / LOG COST ANOMALY
#
# Q_i = FreightObserved / FreightExpected
#
# Z_i^F = log(Q_i)
# =============================================================================

df["freight_expected_ratio"] = np.where(
    valid_expected_pair,
    (
        df["total_freight"]
        /
        df["expected_freight_oot"]
    ),
    np.nan,
)


df["freight_residual_relative"] = np.where(
    valid_expected_pair,
    (
        (
            df["total_freight"]
            -
            df["expected_freight_oot"]
        )
        /
        df["expected_freight_oot"]
    ),
    np.nan,
)


df["freight_log_ratio"] = np.where(
    valid_expected_pair,
    np.log(
        (df["total_freight"] + EPS)
        /
        (df["expected_freight_oot"] + EPS)
    ),
    np.nan,
)


# =============================================================================
# 9. METRIC 5 — MONTHLY ROBUST STANDARDIZATION
#
# Z*_it =
#   (Z_it - median_t(Z))
#   /
#   (1.4826 MAD_t + epsilon)
# =============================================================================

month_stats = (
    df
    .groupby(
        "year_month",
        dropna=False,
        observed=True,
    )
    .apply(
        robust_month_stats,
        include_groups=False,
    )
    .reset_index()
)


# If a month has MAD == 0, do not manufacture an enormous standardized value.
month_stats["robust_scale_usable"] = (
    month_stats["scaled_mad"].notna()
    & (month_stats["scaled_mad"] > EPS)
).astype("int8")


df = df.merge(
    month_stats[
        [
            "year_month",
            "median_log_ratio",
            "mad_log_ratio",
            "scaled_mad",
            "robust_scale_usable",
        ]
    ],
    on="year_month",
    how="left",
    validate="many_to_one",
)


df["freight_log_ratio_month_z"] = np.where(
    (
        df["freight_log_ratio"].notna()
        & (df["robust_scale_usable"] == 1)
    ),
    (
        (
            df["freight_log_ratio"]
            -
            df["median_log_ratio"]
        )
        /
        df["scaled_mad"]
    ),
    np.nan,
)


# =============================================================================
# 10. INTERNAL CONSISTENCY CHECK
#
# Burden and freight/price must obey:
# B = R / (1 + R)
# =============================================================================

check = df[
    [
        "freight_burden",
        "freight_to_price_ratio",
    ]
].dropna()

if len(check):

    reconstructed = (
        check["freight_to_price_ratio"]
        /
        (
            1.0
            +
            check["freight_to_price_ratio"]
        )
    )

    max_identity_error = float(
        np.max(
            np.abs(
                check["freight_burden"]
                -
                reconstructed
            )
        )
    )

else:
    max_identity_error = np.nan


if (
    np.isfinite(max_identity_error)
    and
    max_identity_error > 1e-10
):
    raise RuntimeError(
        "Identidade FreightBurden vs Freight/Price falhou: "
        f"{max_identity_error}"
    )


# =============================================================================
# 11. MONTHLY STANDARDIZATION CHECK
# =============================================================================

zcheck = (
    df[
        df["freight_log_ratio_month_z"].notna()
    ]
    .groupby(
        "year_month",
        observed=True,
    )["freight_log_ratio_month_z"]
    .median()
)


max_abs_monthly_median_z = (
    float(np.max(np.abs(zcheck)))
    if len(zcheck)
    else np.nan
)


# =============================================================================
# 12. COVERAGE TABLE
# =============================================================================

coverage_rows = []

for field in [
    "total_price",
    "total_freight",
    "expected_freight_oot",
    "freight_burden",
    "freight_to_price_ratio",
    "freight_residual_abs",
    "freight_log_ratio",
    "freight_log_ratio_month_z",
]:

    non_null = int(
        df[field].notna().sum()
    )

    coverage_rows.append({
        "field": field,
        "n_non_null": non_null,
        "n_total": n,
        "coverage_pct": pct(
            non_null,
            n,
        ),
    })


coverage = pd.DataFrame(
    coverage_rows
)

coverage.to_csv(
    OUT_COVERAGE,
    index=False,
)


# =============================================================================
# 13. QUANTILES
# =============================================================================

quantile_rows = []

for field in [
    "total_price",
    "total_freight",
    "freight_burden",
    "freight_to_price_ratio",
    "freight_residual_abs",
    "freight_expected_ratio",
    "freight_log_ratio",
    "freight_log_ratio_month_z",
]:

    s = df[field].dropna()

    if len(s) == 0:
        continue

    quantile_rows.append({
        "field": field,
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "q10": float(s.quantile(0.10)),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q80": float(s.quantile(0.80)),
        "q90": float(s.quantile(0.90)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
    })


quantiles = pd.DataFrame(
    quantile_rows
)

quantiles.to_csv(
    OUT_QUANTILES,
    index=False,
)


month_stats.to_csv(
    OUT_MONTH,
    index=False,
)


# =============================================================================
# 14. MATERIALIZE CORE
# =============================================================================

core_cols = [
    "order_id",
    "order_purchase_timestamp",
    "year_month",

    "total_price",
    "total_freight",
    "expected_freight_oot",

    "freight_burden",
    "freight_to_price_ratio",
    "log_freight_to_price_ratio",

    "freight_residual_abs",
    "freight_residual_relative",
    "freight_expected_ratio",

    "freight_log_ratio",
    "freight_log_ratio_month_z",

    "median_log_ratio",
    "mad_log_ratio",
    "scaled_mad",

    "valid_price",
    "valid_freight",
    "valid_expected_freight",
    "valid_year_month",
    "robust_scale_usable",
]


df[
    core_cols
].to_csv(
    OUT_CORE,
    index=False,
)


# =============================================================================
# 15. SCIENTIFIC GATES
# =============================================================================

n_expected = int(
    df["expected_freight_oot"]
    .notna()
    .sum()
)

n_log_ratio = int(
    df["freight_log_ratio"]
    .notna()
    .sum()
)

n_z_month = int(
    df["freight_log_ratio_month_z"]
    .notna()
    .sum()
)


gates = {
    "B01_G01_input_exists":
        INPUT.exists(),

    "B01_G02_order_id_unique":
        duplicate_orders == 0,

    "B01_G03_population_preserved":
        len(df) == n,

    "B01_G04_freight_burden_bounded":
        not bad_burden.any(),

    "B01_G05_ratio_identity_valid":
        (
            np.isnan(max_identity_error)
            or max_identity_error <= 1e-10
        ),

    "B01_G06_expected_freight_not_imputed":
        n_log_ratio <= n_expected,

    "B01_G07_monthly_standardization_available":
        n_z_month > 0,

    "B01_G08_monthly_median_centering":
        (
            np.isnan(max_abs_monthly_median_z)
            or max_abs_monthly_median_z <= 1e-8
        ),

    "B01_G09_no_model_refit":
        True,

    "B01_G10_raw_untouched":
        True,
}


failed = [
    key
    for key, value
    in gates.items()
    if not value
]


status = (
    "PASS"
    if not failed
    else "FAIL"
)


# =============================================================================
# 16. DECISION JSON
# =============================================================================

decision = {
    "status": status,

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "module":
        "27A_BLOCK01_CORE_FRICTION",

    "input":
        str(
            INPUT.relative_to(ROOT)
        ),

    "population": {
        "n_orders": int(n),
        "duplicate_order_id": duplicate_orders,
        "n_expected_freight_available": n_expected,
        "n_log_cost_anomaly": n_log_ratio,
        "n_month_standardized_anomaly": n_z_month,
    },

    "definitions": {
        "freight_burden":
            "total_freight / (total_price + total_freight)",

        "freight_to_price_ratio":
            "total_freight / total_price",

        "freight_residual_abs":
            "total_freight - expected_freight_oot",

        "freight_expected_ratio":
            "total_freight / expected_freight_oot",

        "freight_log_ratio":
            "log((total_freight + eps) / "
            "(expected_freight_oot + eps))",

        "freight_log_ratio_month_z":
            "(freight_log_ratio - monthly_median) / "
            "(1.4826 * monthly_MAD)",
    },

    "interpretation": {
        "freight_burden":
            "Customer-side shipping economic burden proxy.",

        "freight_log_ratio":
            "Cost anomaly relative to the previously "
            "estimated OOT expected freight model.",

        "freight_log_ratio_month_z":
            "Robust month-relative cost anomaly for "
            "cross-period comparison.",
    },

    "guardrails": {
        "freight_residual_is_true_operational_inefficiency":
            False,

        "causal_claim":
            False,

        "carrier_identity_available":
            False,

        "model_refit":
            False,

        "raw_modified":
            False,

        "n_x_n_matrix_created":
            False,
    },

    "checks": {
        "max_burden_ratio_identity_error":
            safe_float(
                max_identity_error
            ),

        "max_abs_monthly_median_standardized_anomaly":
            safe_float(
                max_abs_monthly_median_z
            ),
    },

    "gates":
        gates,

    "failed_gates":
        failed,

    "outputs": [
        str(OUT_CORE.relative_to(ROOT)),
        str(OUT_COVERAGE.relative_to(ROOT)),
        str(OUT_MONTH.relative_to(ROOT)),
        str(OUT_QUANTILES.relative_to(ROOT)),
    ],
}


OUT_DECISION.write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =============================================================================
# 17. HUMAN-READABLE REPORT
# =============================================================================

report = f"""
================================================================================
27A — BLOCK 01 — SHIPPING FRICTION CORE METRICS
================================================================================

STATUS
------
{status}

POPULATION
----------
Orders in analytical input     : {n:,}
Duplicate order_id             : {duplicate_orders:,}

Expected freight available     : {n_expected:,}
Log cost anomaly available     : {n_log_ratio:,}
Monthly robust anomaly         : {n_z_month:,}

CORE DEFINITIONS
----------------
Freight Burden:
    B = Freight / (Price + Freight)

Freight / Price:
    R = Freight / Price

Absolute Freight Residual:
    e = Freight - Expected Freight

Freight Cost Ratio:
    Q = Freight / Expected Freight

Freight Log Ratio:
    Z = log((Freight + eps) / (Expected Freight + eps))

Monthly Robust Cost Anomaly:
    Z* = (Z - monthly median) / (1.4826 * monthly MAD)

INTERPRETATION
--------------
Freight Burden measures customer-side shipping burden.

Freight Log Ratio measures how far observed freight lies above or below
the previously estimated OOT expected-freight benchmark.

Z > 0:
    observed freight above expected.

Z = 0:
    observed freight approximately equal to expected.

Z < 0:
    observed freight below expected.

The residual is NOT interpreted as proven logistics inefficiency.
Carrier identity is unavailable.
No causal claim is made.

VALIDATION
----------
Burden/ratio identity max error:
    {max_identity_error}

Maximum absolute monthly median of Z*:
    {max_abs_monthly_median_z}

FAILED GATES
------------
{failed if failed else "NONE"}

NEXT BLOCK
----------
27A BLOCK 02 — BURDEN DECOMPOSITION

Do not interpret seller, region, service or subsidy opportunity yet.
Those analyses belong to later blocks.
================================================================================
""".strip()


OUT_REPORT.write_text(
    report + "\n",
    encoding="utf-8",
)


# =============================================================================
# FINAL
# =============================================================================

print()
print(coverage.to_string(index=False))

print()
print("=" * 100)
print("BLOCK 01 DECISION")
print("=" * 100)

print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    )
)

print()

if status != "PASS":
    raise RuntimeError(
        f"BLOCK 01 FAILED: {failed}"
    )

print("[PASS 27A-B01] SHIPPING FRICTION CORE MATERIALIZED")
print(f"CORE = {OUT_CORE.relative_to(ROOT)}")
print(f"DECISION = {OUT_DECISION.relative_to(ROOT)}")
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("PARAR AQUI")

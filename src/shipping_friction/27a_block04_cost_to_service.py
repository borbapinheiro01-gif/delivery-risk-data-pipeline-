#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODULE 27A — BLOCK 04
COST-TO-SERVICE AUDIT

Cinco componentes congelados:

16. Cost anomaly x Delivery Gap
17. Cost anomaly x Official Late Delivery
18. Freight Burden x Service
19. Cost x Service Quadrants
20. Critical Shipping Friction

Pergunta central
----------------
O frete relativamente mais alto compra serviço melhor?

Objetos principais
------------------
Z* = robust monthly standardized freight log-ratio
B  = freight burden
G  = continuous delivery gap
Y  = official calendar-day late-delivery target

Modelos diagnósticos
--------------------
Continuous:

    G ~ Z* + B + physical + promise + route + month

Binary:

    logit P(Y=1)
        ~ Z* + B + physical + promise + route + month

Interpretação
-------------
Associacional / diagnóstica.

Não:
- retreina GN_EQ0;
- retreina expected freight;
- altera RAW;
- estima causalidade;
- estima abandono/conversão;
- calcula lucro;
- cria matriz N x N.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.multitest import multipletests


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

FRICTION = (
    ROOT
    / "artifacts"
    / "shipping_friction"
    / "27a_01_shipping_friction_core.csv"
)

SERVICE = (
    ROOT
    / "artifacts"
    / "shipping_friction"
    / "27a_03_service_performance_core.csv"
)

MASTER = (
    ROOT
    / "artifacts"
    / "spatiotemporal_logistics"
    / "06_EXPECTED_FREIGHT_OOT.csv"
)

ART = ROOT / "artifacts" / "shipping_friction"
REP = ROOT / "reports" / "shipping_friction"

OUT_CORE = (
    ART
    / "27a_04_cost_service_core.csv"
)

OUT_MODELS = (
    REP
    / "27a_04_cost_service_models.csv"
)

OUT_DECILES = (
    REP
    / "27a_04_cost_anomaly_deciles.csv"
)

OUT_QUADRANTS = (
    REP
    / "27a_04_cost_service_quadrants.csv"
)

OUT_CRITICAL = (
    REP
    / "27a_04_critical_friction_sensitivity.csv"
)

OUT_THRESHOLDS = (
    REP
    / "27a_04_thresholds.csv"
)

OUT_DECISION = (
    REP
    / "27a_block04_decision.json"
)

OUT_REPORT = (
    REP
    / "27a_block04_report.txt"
)


warnings.filterwarnings("ignore")


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

FRICTION_COLS = [
    "order_id",
    "year_month",
    "freight_burden",
    "freight_log_ratio",
    "freight_log_ratio_month_z",
]

SERVICE_COLS = [
    "order_id",
    "delivery_gap_days_continuous",
    "delivery_gap_calendar_days",
    "positive_tardiness_days",
    "late_delivery_calendar_day",
]

MASTER_COLS = [
    "order_id",
    "total_price",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "route_sellers_total",
    "any_interstate_route",
    "promised_delivery_days",
]


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(x):
    if x is None or pd.isna(x):
        return None
    return float(x)


def pct(n, d):
    if not d:
        return np.nan
    return 100.0 * n / d


def read_required(path, cols):

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo ausente: {path}"
        )

    header = pd.read_csv(
        path,
        nrows=0,
    )

    missing = [
        c
        for c in cols
        if c not in header.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path.name}: colunas ausentes: {missing}"
        )

    return pd.read_csv(
        path,
        usecols=cols,
        low_memory=False,
    )


def qcut_safe(series, q=10):

    try:
        return pd.qcut(
            series,
            q=q,
            labels=False,
            duplicates="drop",
        )

    except Exception:

        return pd.Series(
            np.nan,
            index=series.index,
        )


# =============================================================================
# START
# =============================================================================

print("=" * 108)
print("27A — BLOCK 04 — COST-TO-SERVICE AUDIT")
print("=" * 108)

print("FINAL_PREDICTIVE_MODEL_REFIT = false")
print("DIAGNOSTIC_ASSOCIATION_MODELS = true")
print("RAW_MODIFIED = false")
print("CAUSAL_CLAIM = false")
print("N_X_N_MATRIX = false")
print()


# =============================================================================
# 1. LOAD
# =============================================================================

friction = read_required(
    FRICTION,
    FRICTION_COLS,
)

service = read_required(
    SERVICE,
    SERVICE_COLS,
)

master = read_required(
    MASTER,
    MASTER_COLS,
)


for name, frame in [
    ("friction", friction),
    ("service", service),
    ("master", master),
]:

    if frame["order_id"].duplicated().any():
        raise RuntimeError(
            f"{name}: order_id duplicado"
        )


print(
    f"[INFO] friction rows = {len(friction):,}"
)

print(
    f"[INFO] service rows  = {len(service):,}"
)

print(
    f"[INFO] master rows   = {len(master):,}"
)


# =============================================================================
# 2. ONE-TO-ONE MERGE
# =============================================================================

df = (
    friction
    .merge(
        service,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        master,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
)


n_total = len(df)


if n_total != len(friction):
    raise RuntimeError(
        "Merge alterou a população."
    )


# =============================================================================
# 3. NUMERIC TYPES
# =============================================================================

numeric_cols = [
    "freight_burden",
    "freight_log_ratio",
    "freight_log_ratio_month_z",

    "delivery_gap_days_continuous",
    "delivery_gap_calendar_days",
    "positive_tardiness_days",
    "late_delivery_calendar_day",

    "total_price",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "route_sellers_total",
    "any_interstate_route",
    "promised_delivery_days",
]


for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


df["year_month"] = (
    df["year_month"]
    .astype("string")
)


# =============================================================================
# 4. MODEL TRANSFORMATIONS
# =============================================================================

df["log_price"] = np.log1p(
    df["total_price"].where(
        df["total_price"] >= 0
    )
)

df["log_distance"] = np.log1p(
    df["distance_freight_weighted_km"].where(
        df["distance_freight_weighted_km"] >= 0
    )
)

df["log_weight"] = np.log1p(
    df["total_weight_g"].where(
        df["total_weight_g"] >= 0
    )
)

df["log_volume"] = np.log1p(
    df["product_volume_sum_proxy_cm3"].where(
        df["product_volume_sum_proxy_cm3"] >= 0
    )
)


# Coefficient = change associated with +10 percentage points burden.
df["burden_10pp"] = (
    df["freight_burden"]
    /
    0.10
)


# =============================================================================
# 5. ANALYTICAL SAMPLES
# =============================================================================

CONTROL_COLS = [
    "freight_log_ratio_month_z",
    "burden_10pp",
    "log_price",
    "log_distance",
    "log_weight",
    "log_volume",
    "route_sellers_total",
    "any_interstate_route",
    "promised_delivery_days",
    "year_month",
]


continuous_required = (
    CONTROL_COLS
    +
    [
        "delivery_gap_days_continuous"
    ]
)


binary_required = (
    CONTROL_COLS
    +
    [
        "late_delivery_calendar_day"
    ]
)


continuous = (
    df
    .dropna(
        subset=continuous_required
    )
    .copy()
)


binary = (
    df
    .dropna(
        subset=binary_required
    )
    .copy()
)


binary = binary[
    binary[
        "late_delivery_calendar_day"
    ].isin([0, 1])
].copy()


print(
    f"[INFO] continuous complete sample = {len(continuous):,}"
)

print(
    f"[INFO] binary complete sample     = {len(binary):,}"
)


if len(continuous) < 1000:
    raise RuntimeError(
        "Amostra contínua muito pequena."
    )

if len(binary) < 1000:
    raise RuntimeError(
        "Amostra binária muito pequena."
    )


# =============================================================================
# 6. MODELS
#
# C0 / R0:
#     cost anomaly only
#
# C1 / R1:
#     cost anomaly + burden + physical + promise + route + month
#
# Preferred:
#     C1 and R1
# =============================================================================

formula_c0 = (
    "delivery_gap_days_continuous ~ "
    "freight_log_ratio_month_z"
)

formula_c1 = (
    "delivery_gap_days_continuous ~ "
    "freight_log_ratio_month_z + "
    "burden_10pp + "
    "log_price + "
    "log_distance + "
    "log_weight + "
    "log_volume + "
    "route_sellers_total + "
    "any_interstate_route + "
    "promised_delivery_days + "
    "C(year_month)"
)


formula_r0 = (
    "late_delivery_calendar_day ~ "
    "freight_log_ratio_month_z"
)

formula_r1 = (
    "late_delivery_calendar_day ~ "
    "freight_log_ratio_month_z + "
    "burden_10pp + "
    "log_price + "
    "log_distance + "
    "log_weight + "
    "log_volume + "
    "route_sellers_total + "
    "any_interstate_route + "
    "promised_delivery_days + "
    "C(year_month)"
)


print()
print("[INFO] fitting continuous C0 ...")

fit_c0 = smf.ols(
    formula_c0,
    data=continuous,
).fit(
    cov_type="HC1"
)


print("[INFO] fitting continuous C1 ...")

fit_c1 = smf.ols(
    formula_c1,
    data=continuous,
).fit(
    cov_type="HC1"
)


print("[INFO] fitting reliability R0 ...")

fit_r0 = smf.glm(
    formula=formula_r0,
    data=binary,
    family=sm.families.Binomial(),
).fit(
    maxiter=200,
    tol=1e-8,
    cov_type="HC1",
)


print("[INFO] fitting reliability R1 ...")

fit_r1 = smf.glm(
    formula=formula_r1,
    data=binary,
    family=sm.families.Binomial(),
).fit(
    maxiter=200,
    tol=1e-8,
    cov_type="HC1",
)


print("[PASS] diagnostic cost-to-service models fitted")


# =============================================================================
# 7. MODEL COEFFICIENT REGISTRY
# =============================================================================

focal_terms = [
    "freight_log_ratio_month_z",
    "burden_10pp",
]


model_rows = []


for model_name, model_type, fit in [
    (
        "C0_UNADJUSTED_GAP",
        "OLS_HC1",
        fit_c0,
    ),
    (
        "C1_ADJUSTED_GAP",
        "OLS_HC1",
        fit_c1,
    ),
    (
        "R0_UNADJUSTED_LATE",
        "GLM_BINOMIAL_HC1",
        fit_r0,
    ),
    (
        "R1_ADJUSTED_LATE",
        "GLM_BINOMIAL_HC1",
        fit_r1,
    ),
]:

    for term in focal_terms:

        if term not in fit.params.index:
            continue

        coef = float(
            fit.params[term]
        )

        se = float(
            fit.bse[term]
        )

        p = float(
            fit.pvalues[term]
        )

        row = {
            "model":
                model_name,

            "model_type":
                model_type,

            "term":
                term,

            "coef":
                coef,

            "std_err_HC1":
                se,

            "p_value":
                p,

            "nobs":
                int(
                    fit.nobs
                ),

            "effect_scale":
                (
                    "DAYS"
                    if model_type == "OLS_HC1"
                    else
                    "LOG_ODDS"
                ),

            "odds_ratio":
                (
                    np.nan
                    if model_type == "OLS_HC1"
                    else
                    float(
                        np.exp(coef)
                    )
                ),
        }

        model_rows.append(row)


models_df = pd.DataFrame(
    model_rows
)


# FDR across the small frozen focal family.
reject, padj, _, _ = multipletests(
    models_df["p_value"].to_numpy(),
    alpha=0.05,
    method="fdr_bh",
)


models_df["p_value_fdr_bh"] = padj
models_df["fdr_reject_005"] = reject.astype(int)


models_df.to_csv(
    OUT_MODELS,
    index=False,
)


# =============================================================================
# 8. ANOMALY DECILE CURVE
#
# Descriptive only.
# =============================================================================

decile_base = df.dropna(
    subset=[
        "freight_log_ratio_month_z",
        "freight_burden",
        "late_delivery_calendar_day",
    ]
).copy()


decile_base["cost_anomaly_decile"] = qcut_safe(
    decile_base[
        "freight_log_ratio_month_z"
    ],
    10,
)


decile_rows = []


for decile, grp in decile_base.groupby(
    "cost_anomaly_decile",
    dropna=True,
    observed=True,
):

    service_grp = grp[
        grp[
            "delivery_gap_days_continuous"
        ].notna()
    ]

    late = grp[
        "late_delivery_calendar_day"
    ].dropna()

    decile_rows.append({
        "cost_anomaly_decile":
            int(decile),

        "n":
            int(len(grp)),

        "cost_anomaly_median":
            float(
                grp[
                    "freight_log_ratio_month_z"
                ].median()
            ),

        "freight_log_ratio_median":
            float(
                grp[
                    "freight_log_ratio"
                ].median()
            ),

        "freight_burden_median":
            float(
                grp[
                    "freight_burden"
                ].median()
            ),

        "delivery_gap_mean":
            (
                float(
                    service_grp[
                        "delivery_gap_days_continuous"
                    ].mean()
                )
                if len(service_grp)
                else np.nan
            ),

        "delivery_gap_median":
            (
                float(
                    service_grp[
                        "delivery_gap_days_continuous"
                    ].median()
                )
                if len(service_grp)
                else np.nan
            ),

        "late_rate":
            (
                float(
                    late.mean()
                )
                if len(late)
                else np.nan
            ),
    })


deciles_df = pd.DataFrame(
    decile_rows
)

deciles_df.to_csv(
    OUT_DECILES,
    index=False,
)


# =============================================================================
# 9. THRESHOLDS
#
# Frozen sensitivities:
#
# 75 / 80 / 90 percentiles.
# =============================================================================

cost_valid = df[
    "freight_log_ratio_month_z"
].dropna()

burden_valid = df[
    "freight_burden"
].dropna()


threshold_rows = []


for q in [0.25, 0.75, 0.80, 0.90]:

    threshold_rows.append({
        "metric":
            "cost_anomaly",

        "quantile":
            q,

        "value":
            float(
                cost_valid.quantile(q)
            ),
    })


for q in [0.25, 0.75, 0.80, 0.90]:

    threshold_rows.append({
        "metric":
            "freight_burden",

        "quantile":
            q,

        "value":
            float(
                burden_valid.quantile(q)
            ),
    })


thresholds_df = pd.DataFrame(
    threshold_rows
)

thresholds_df.to_csv(
    OUT_THRESHOLDS,
    index=False,
)


def get_threshold(metric, q):

    row = thresholds_df[
        (
            thresholds_df["metric"]
            == metric
        )
        &
        (
            thresholds_df["quantile"]
            == q
        )
    ]

    return float(
        row.iloc[0]["value"]
    )


cost_q25 = get_threshold(
    "cost_anomaly",
    0.25,
)

cost_q75 = get_threshold(
    "cost_anomaly",
    0.75,
)

burden_q75 = get_threshold(
    "freight_burden",
    0.75,
)


# =============================================================================
# 10. COST x SERVICE QUADRANTS
#
# Keep middle-cost orders as NEUTRAL.
#
# Service:
#     GOOD = official on-time
#     POOR = official late
# =============================================================================

quad = df.dropna(
    subset=[
        "freight_log_ratio_month_z",
        "late_delivery_calendar_day",
    ]
).copy()


quad["cost_service_quadrant"] = "NEUTRAL_MIDDLE_COST"


low_cost = (
    quad[
        "freight_log_ratio_month_z"
    ]
    <= cost_q25
)


high_cost = (
    quad[
        "freight_log_ratio_month_z"
    ]
    >= cost_q75
)


good_service = (
    quad[
        "late_delivery_calendar_day"
    ]
    == 0
)


poor_service = (
    quad[
        "late_delivery_calendar_day"
    ]
    == 1
)


quad.loc[
    low_cost & good_service,
    "cost_service_quadrant",
] = "BENCHMARK_LOW_COST_GOOD_SERVICE"


quad.loc[
    low_cost & poor_service,
    "cost_service_quadrant",
] = "LOW_COST_POOR_SERVICE"


quad.loc[
    high_cost & good_service,
    "cost_service_quadrant",
] = "PREMIUM_COST_GOOD_SERVICE"


quad.loc[
    high_cost & poor_service,
    "cost_service_quadrant",
] = "OPERATIONAL_LEAKAGE_CANDIDATE"


quadrant_rows = []


for group, grp in quad.groupby(
    "cost_service_quadrant",
    observed=True,
):

    service_grp = grp[
        grp[
            "delivery_gap_days_continuous"
        ].notna()
    ]

    quadrant_rows.append({
        "cost_service_quadrant":
            group,

        "n":
            int(len(grp)),

        "pct_of_classifiable":
            float(
                100.0
                *
                len(grp)
                /
                len(quad)
            ),

        "cost_anomaly_median":
            float(
                grp[
                    "freight_log_ratio_month_z"
                ].median()
            ),

        "freight_log_ratio_median":
            float(
                grp[
                    "freight_log_ratio"
                ].median()
            ),

        "freight_burden_median":
            float(
                grp[
                    "freight_burden"
                ].median()
            ),

        "delivery_gap_median":
            (
                float(
                    service_grp[
                        "delivery_gap_days_continuous"
                    ].median()
                )
                if len(service_grp)
                else np.nan
            ),

        "late_rate":
            float(
                grp[
                    "late_delivery_calendar_day"
                ].mean()
            ),
    })


quadrants_df = pd.DataFrame(
    quadrant_rows
)

quadrants_df.to_csv(
    OUT_QUADRANTS,
    index=False,
)


# =============================================================================
# 11. CRITICAL SHIPPING FRICTION
#
# CF_q =
#
# high customer burden
# AND
# high month-relative cost anomaly
# AND
# official late delivery
#
# Sensitivity:
# q75, q80, q90
# =============================================================================

critical_rows = []


for q in [0.75, 0.80, 0.90]:

    cq = get_threshold(
        "cost_anomaly",
        q,
    )

    bq = get_threshold(
        "freight_burden",
        q,
    )

    eligible = df[
        [
            "freight_log_ratio_month_z",
            "freight_burden",
            "late_delivery_calendar_day",
        ]
    ].notna().all(axis=1)


    flag = (
        eligible
        &
        (
            df[
                "freight_log_ratio_month_z"
            ]
            >= cq
        )
        &
        (
            df[
                "freight_burden"
            ]
            >= bq
        )
        &
        (
            df[
                "late_delivery_calendar_day"
            ]
            == 1
        )
    )


    df[
        f"critical_friction_q{int(q*100)}"
    ] = flag.astype("int8")


    n_eligible = int(
        eligible.sum()
    )

    n_critical = int(
        flag.sum()
    )


    critical_rows.append({
        "threshold_quantile":
            q,

        "cost_anomaly_threshold":
            cq,

        "freight_burden_threshold":
            bq,

        "n_eligible":
            n_eligible,

        "n_critical":
            n_critical,

        "critical_pct_eligible":
            pct(
                n_critical,
                n_eligible,
            ),
    })


critical_df = pd.DataFrame(
    critical_rows
)

critical_df.to_csv(
    OUT_CRITICAL,
    index=False,
)


# =============================================================================
# 12. MATERIALIZE CORE FOR DOWNSTREAM SEGMENT ANALYSIS
# =============================================================================

quad_map = quad[
    [
        "order_id",
        "cost_service_quadrant",
    ]
].copy()


df = df.merge(
    quad_map,
    on="order_id",
    how="left",
    validate="one_to_one",
)


out_cols = [
    "order_id",
    "year_month",

    "freight_burden",
    "freight_log_ratio",
    "freight_log_ratio_month_z",

    "delivery_gap_days_continuous",
    "delivery_gap_calendar_days",
    "positive_tardiness_days",
    "late_delivery_calendar_day",

    "total_price",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "route_sellers_total",
    "any_interstate_route",
    "promised_delivery_days",

    "cost_service_quadrant",

    "critical_friction_q75",
    "critical_friction_q80",
    "critical_friction_q90",
]


df[
    out_cols
].to_csv(
    OUT_CORE,
    index=False,
)


# =============================================================================
# 13. EXTRACT FOCAL RESULTS
# =============================================================================

def get_model_row(model, term):

    x = models_df[
        (
            models_df["model"]
            == model
        )
        &
        (
            models_df["term"]
            == term
        )
    ]

    if len(x) != 1:
        return None

    return x.iloc[0]


c1_cost = get_model_row(
    "C1_ADJUSTED_GAP",
    "freight_log_ratio_month_z",
)

c1_burden = get_model_row(
    "C1_ADJUSTED_GAP",
    "burden_10pp",
)

r1_cost = get_model_row(
    "R1_ADJUSTED_LATE",
    "freight_log_ratio_month_z",
)

r1_burden = get_model_row(
    "R1_ADJUSTED_LATE",
    "burden_10pp",
)


# =============================================================================
# 14. GATES
# =============================================================================

critical_counts = (
    critical_df
    .set_index(
        "threshold_quantile"
    )["n_critical"]
    .to_dict()
)


quadrant_count_sum = int(
    quadrants_df["n"].sum()
)


gates = {
    "B04_G01_inputs_exist":
        (
            FRICTION.exists()
            and SERVICE.exists()
            and MASTER.exists()
        ),

    "B04_G02_population_preserved":
        len(df) == len(friction),

    "B04_G03_order_id_unique":
        not df["order_id"].duplicated().any(),

    "B04_G04_continuous_sample_nontrivial":
        len(continuous) >= 1000,

    "B04_G05_binary_sample_nontrivial":
        len(binary) >= 1000,

    "B04_G06_reliability_R0_converged":
        bool(
            fit_r0.converged
        ),

    "B04_G07_reliability_R1_converged":
        bool(
            fit_r1.converged
        ),

    "B04_G08_focal_model_results_finite":
        bool(
            np.isfinite(
                models_df["coef"]
            ).all()
        ),

    "B04_G09_quadrant_population_reconciles":
        quadrant_count_sum == len(quad),

    "B04_G10_critical_monotonic_with_threshold":
        bool(
            critical_counts.get(
                0.90,
                0,
            )
            <=
            critical_counts.get(
                0.80,
                0,
            )
            <=
            critical_counts.get(
                0.75,
                0,
            )
        ),

    "B04_G11_final_predictive_model_not_refit":
        True,

    "B04_G12_raw_untouched":
        True,

    "B04_G13_no_causal_claim":
        True,
}


failed = [
    key
    for key, ok
    in gates.items()
    if not ok
]


status = (
    "PASS"
    if not failed
    else "FAIL"
)


# =============================================================================
# 15. INTERPRETATION CLASSIFIER
#
# This does NOT decide statistical causality.
#
# It summarizes the direction of adjusted association.
# =============================================================================

cost_gap_coef = float(
    c1_cost["coef"]
)

cost_gap_p = float(
    c1_cost["p_value_fdr_bh"]
)

cost_late_or = float(
    r1_cost["odds_ratio"]
)

cost_late_p = float(
    r1_cost["p_value_fdr_bh"]
)


if (
    cost_gap_p < 0.05
    and cost_late_p < 0.05
    and cost_gap_coef < 0
    and cost_late_or < 1
):

    cost_service_interpretation = (
        "HIGHER_RELATIVE_COST_ASSOCIATED_WITH_BETTER_SERVICE"
    )

elif (
    cost_gap_p < 0.05
    and cost_late_p < 0.05
    and cost_gap_coef > 0
    and cost_late_or > 1
):

    cost_service_interpretation = (
        "HIGHER_RELATIVE_COST_ASSOCIATED_WITH_WORSE_SERVICE"
    )

elif (
    cost_gap_p >= 0.05
    and cost_late_p >= 0.05
):

    cost_service_interpretation = (
        "NO_CLEAR_ADJUSTED_COST_SERVICE_ASSOCIATION"
    )

else:

    cost_service_interpretation = (
        "MIXED_COST_SERVICE_EVIDENCE"
    )


# =============================================================================
# 16. DECISION JSON
# =============================================================================

decision = {
    "status":
        status,

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "module":
        "27A_BLOCK04_COST_TO_SERVICE",

    "population": {
        "n_orders":
            int(n_total),

        "n_continuous_complete_case":
            int(
                len(continuous)
            ),

        "n_binary_complete_case":
            int(
                len(binary)
            ),

        "n_quadrant_classifiable":
            int(
                len(quad)
            ),
    },

    "adjusted_cost_service": {
        "cost_anomaly_to_delivery_gap": {
            "coefficient_days_per_1_robust_scale":
                safe_float(
                    c1_cost["coef"]
                ),

            "p_value_fdr_bh":
                safe_float(
                    c1_cost["p_value_fdr_bh"]
                ),
        },

        "cost_anomaly_to_late": {
            "odds_ratio_per_1_robust_scale":
                safe_float(
                    r1_cost["odds_ratio"]
                ),

            "p_value_fdr_bh":
                safe_float(
                    r1_cost["p_value_fdr_bh"]
                ),
        },

        "burden_10pp_to_delivery_gap": {
            "coefficient_days":
                safe_float(
                    c1_burden["coef"]
                ),

            "p_value_fdr_bh":
                safe_float(
                    c1_burden["p_value_fdr_bh"]
                ),
        },

        "burden_10pp_to_late": {
            "odds_ratio":
                safe_float(
                    r1_burden["odds_ratio"]
                ),

            "p_value_fdr_bh":
                safe_float(
                    r1_burden["p_value_fdr_bh"]
                ),
        },
    },

    "cost_service_interpretation":
        cost_service_interpretation,

    "critical_friction": {
        (
            f"q{int(row['threshold_quantile']*100)}"
        ): {
            "n_critical":
                int(
                    row[
                        "n_critical"
                    ]
                ),

            "critical_pct_eligible":
                float(
                    row[
                        "critical_pct_eligible"
                    ]
                ),

            "cost_threshold":
                float(
                    row[
                        "cost_anomaly_threshold"
                    ]
                ),

            "burden_threshold":
                float(
                    row[
                        "freight_burden_threshold"
                    ]
                ),
        }

        for _, row
        in critical_df.iterrows()
    },

    "quadrant_definition": {
        "BENCHMARK_LOW_COST_GOOD_SERVICE":
            "cost anomaly <= Q25 and official on-time",

        "LOW_COST_POOR_SERVICE":
            "cost anomaly <= Q25 and official late",

        "PREMIUM_COST_GOOD_SERVICE":
            "cost anomaly >= Q75 and official on-time",

        "OPERATIONAL_LEAKAGE_CANDIDATE":
            "cost anomaly >= Q75 and official late",

        "NEUTRAL_MIDDLE_COST":
            "cost anomaly between Q25 and Q75",
    },

    "critical_friction_definition":
        (
            "high freight burden AND high month-relative "
            "cost anomaly AND official calendar-day late"
        ),

    "interpretation_guardrails": {
        "higher_freight_means_true_operating_cost":
            False,

        "cost_anomaly_equals_proven_inefficiency":
            False,

        "operational_leakage_candidate_is_proven_waste":
            False,

        "critical_friction_is_causal":
            False,

        "freight_burden_is_conversion_probability":
            False,

        "post_outcome_service_valid_for_purchase_time_prediction":
            False,

        "carrier_identity_available":
            False,

        "final_predictive_model_refit":
            False,

        "raw_modified":
            False,
    },

    "gates":
        gates,

    "failed_gates":
        failed,

    "outputs": [
        str(
            OUT_CORE.relative_to(ROOT)
        ),

        str(
            OUT_MODELS.relative_to(ROOT)
        ),

        str(
            OUT_DECILES.relative_to(ROOT)
        ),

        str(
            OUT_QUADRANTS.relative_to(ROOT)
        ),

        str(
            OUT_CRITICAL.relative_to(ROOT)
        ),

        str(
            OUT_THRESHOLDS.relative_to(ROOT)
        ),
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
====================================================================================================
27A — BLOCK 04 — COST-TO-SERVICE AUDIT
====================================================================================================

STATUS
------
{status}

POPULATION
----------
Total orders:
    {n_total:,}

Continuous adjusted sample:
    {len(continuous):,}

Binary reliability adjusted sample:
    {len(binary):,}

Quadrant-classifiable:
    {len(quad):,}

PRIMARY QUESTION
----------------
Does paying relatively more than expected buy better service?

ADJUSTED COST ANOMALY -> DELIVERY GAP
-------------------------------------
Coefficient per +1 monthly robust anomaly scale:
    {float(c1_cost["coef"]):.8f} days

FDR-adjusted p:
    {float(c1_cost["p_value_fdr_bh"]):.8g}

Negative coefficient:
    relatively higher cost associated with earlier/better
    delivery relative to promise.

Positive coefficient:
    relatively higher cost associated with worse/later
    service relative to promise.

ADJUSTED COST ANOMALY -> OFFICIAL LATE
--------------------------------------
Odds ratio per +1 monthly robust anomaly scale:
    {float(r1_cost["odds_ratio"]):.8f}

FDR-adjusted p:
    {float(r1_cost["p_value_fdr_bh"]):.8g}

OR < 1:
    higher relative cost associated with lower late odds.

OR > 1:
    higher relative cost associated with higher late odds.

BURDEN +10 PERCENTAGE POINTS -> DELIVERY GAP
--------------------------------------------
Coefficient:
    {float(c1_burden["coef"]):.8f} days

FDR-adjusted p:
    {float(c1_burden["p_value_fdr_bh"]):.8g}

BURDEN +10 PERCENTAGE POINTS -> LATE
------------------------------------
Odds ratio:
    {float(r1_burden["odds_ratio"]):.8f}

FDR-adjusted p:
    {float(r1_burden["p_value_fdr_bh"]):.8g}

FINAL COST-SERVICE DIRECTION
----------------------------
{cost_service_interpretation}

QUADRANTS
---------
{quadrants_df.to_string(index=False)}

CRITICAL SHIPPING FRICTION SENSITIVITY
--------------------------------------
{critical_df.to_string(index=False)}

DEFINITION
----------
Critical Shipping Friction requires simultaneously:

    high Freight Burden
    AND
    high month-relative Cost Anomaly
    AND
    official calendar-day Late Delivery

Threshold sensitivities:

    Q75
    Q80
    Q90

INTERPRETATION GUARDRAILS
-------------------------
1. Cost anomaly is relative to the OOT expected-freight model.

2. Cost anomaly is NOT proven operational inefficiency.

3. Freight Burden is customer-side economic friction.

4. Freight Burden is NOT observed abandonment/conversion.

5. Operational Leakage Candidate is a diagnostic label,
   not proof of waste or mismanagement.

6. Continuous Delivery Gap is used for service magnitude.

7. LATE_DELIVERY_CALENDAR_DAY remains the official reliability failure.

8. All models in this block are associational diagnostics.

9. No carrier identity is available.

10. No causal claim is made.

FAILED GATES
------------
{failed if failed else "NONE"}

NEXT BLOCK
----------
27A BLOCK 05 — SEGMENT CONCENTRATION

The next block will answer:

    WHERE does Critical Shipping Friction concentrate?

including:

    seller
    region
    distance band
    ticket band
    multi-seller
    interstate

PARAR AQUI
====================================================================================================
""".strip()


OUT_REPORT.write_text(
    report + "\n",
    encoding="utf-8",
)


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 108)
print("FOCAL COST-TO-SERVICE MODELS")
print("=" * 108)

print(
    models_df.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("COST-ANOMALY DECILES")
print("=" * 108)

print(
    deciles_df.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("COST x SERVICE QUADRANTS")
print("=" * 108)

print(
    quadrants_df.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("CRITICAL FRICTION")
print("=" * 108)

print(
    critical_df.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("BLOCK 04 DECISION")
print("=" * 108)

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
        f"BLOCK 04 FAILED: {failed}"
    )


print("[PASS 27A-B04] COST-TO-SERVICE AUDIT COMPLETE")
print(
    f"COST_SERVICE_INTERPRETATION = "
    f"{cost_service_interpretation}"
)
print("FINAL_PREDICTIVE_MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("CAUSAL_CLAIM = false")
print("PARAR AQUI")

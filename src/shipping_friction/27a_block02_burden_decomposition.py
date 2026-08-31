#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODULE 27A — BLOCK 02
FREIGHT BURDEN DECOMPOSITION

Pergunta científica
-------------------
Por que o frete representa uma parcela elevada do desembolso
em alguns pedidos?

Cinco componentes congelados:

06. Burden x preço do produto
07. Burden x distância
08. Burden x peso
09. Burden x volume
10. Burden x estrutura da rota
    - número de sellers/rotas
    - interestadual

Método principal
----------------
Fractional logit / quasi-binomial mean model:

    E(B_i | X_i) = logistic(X_i beta)

onde:

    B_i = Freight / (Price + Freight)

O modelo é DIAGNÓSTICO/ASSOCIACIONAL.

Não:
- retreina GN_EQ0;
- retreina expected freight;
- altera RAW;
- estima abandono de carrinho;
- estima elasticidade causal;
- calcula lucro;
- cria matriz N x N.

Observação crucial:
O efeito de preço sobre Freight Burden é parcialmente mecânico,
porque Price aparece no denominador do próprio índice.
Ele NÃO será interpretado causalmente.
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

CORE = (
    ROOT
    / "artifacts"
    / "shipping_friction"
    / "27a_01_shipping_friction_core.csv"
)

MASTER = (
    ROOT
    / "artifacts"
    / "spatiotemporal_logistics"
    / "06_EXPECTED_FREIGHT_OOT.csv"
)

ART = ROOT / "artifacts" / "shipping_friction"
REP = ROOT / "reports" / "shipping_friction"

OUT_SEGMENTS = ART / "27a_02_burden_segments.csv"

OUT_ASSOC = REP / "27a_02_burden_associations.csv"
OUT_DECILES = REP / "27a_02_burden_decile_curves.csv"
OUT_GROUPS = REP / "27a_02_route_structure_groups.csv"

OUT_COEFS = REP / "27a_02_fractional_logit_coefficients.csv"
OUT_EFFECTS = REP / "27a_02_scenario_effects.csv"
OUT_THRESHOLDS = REP / "27a_02_burden_thresholds.csv"

OUT_DECISION = REP / "27a_block02_decision.json"
OUT_REPORT = REP / "27a_block02_report.txt"


# =============================================================================
# CONFIG
# =============================================================================

CORE_COLS = [
    "order_id",
    "year_month",
    "total_price",
    "total_freight",
    "freight_burden",
    "freight_to_price_ratio",
]

MASTER_REQUIRED = [
    "order_id",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "route_sellers_total",
    "any_interstate_route",
]

FOCAL_TERMS = [
    "log_price",
    "log_distance",
    "log_weight",
    "log_volume",
    "route_sellers_total",
    "any_interstate_route",
]

warnings.filterwarnings("ignore")


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(x):
    if x is None or pd.isna(x):
        return None
    return float(x)


def safe_int(x):
    if x is None or pd.isna(x):
        return None
    return int(x)


def add_logs(frame):
    frame = frame.copy()

    frame["log_price"] = np.log1p(
        frame["total_price"]
    )

    frame["log_distance"] = np.log1p(
        frame["distance_freight_weighted_km"]
    )

    frame["log_weight"] = np.log1p(
        frame["total_weight_g"]
    )

    frame["log_volume"] = np.log1p(
        frame["product_volume_sum_proxy_cm3"]
    )

    return frame


def qcut_safe(series, q):
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


def make_decile_table(frame, variable):

    sub = frame[
        [
            variable,
            "freight_burden",
            "total_price",
            "total_freight",
        ]
    ].dropna().copy()

    if len(sub) == 0:
        return pd.DataFrame()

    sub["decile"] = qcut_safe(
        sub[variable],
        10,
    )

    rows = []

    for decile, grp in sub.groupby(
        "decile",
        dropna=True,
        observed=True,
    ):

        rows.append({
            "variable": variable,
            "decile": int(decile),
            "n": int(len(grp)),
            "x_median": float(
                grp[variable].median()
            ),
            "burden_mean": float(
                grp["freight_burden"].mean()
            ),
            "burden_median": float(
                grp["freight_burden"].median()
            ),
            "price_median": float(
                grp["total_price"].median()
            ),
            "freight_median": float(
                grp["total_freight"].median()
            ),
        })

    return pd.DataFrame(rows)


def scenario_effect(
    fit,
    frame,
    variable,
    relative_change=0.10,
):

    """
    Average predicted burden change after a +10% change
    in one original continuous variable.
    """

    base_pred = np.asarray(
        fit.predict(frame),
        dtype=float,
    )

    alt = frame.copy()

    alt[variable] = (
        alt[variable]
        *
        (1.0 + relative_change)
    )

    alt = add_logs(alt)

    alt_pred = np.asarray(
        fit.predict(alt),
        dtype=float,
    )

    delta = alt_pred - base_pred

    return {
        "scenario":
            f"{variable}_PLUS_{int(relative_change*100)}PCT",

        "variable":
            variable,

        "change":
            relative_change,

        "mean_predicted_burden_delta":
            float(np.mean(delta)),

        "median_predicted_burden_delta":
            float(np.median(delta)),

        "p10_delta":
            float(np.quantile(delta, 0.10)),

        "p90_delta":
            float(np.quantile(delta, 0.90)),
    }


# =============================================================================
# START
# =============================================================================

print("=" * 104)
print("27A — BLOCK 02 — FREIGHT BURDEN DECOMPOSITION")
print("=" * 104)
print("FINAL_PREDICTIVE_MODEL_REFIT = false")
print("DIAGNOSTIC_FRACTIONAL_MODEL_FIT = true")
print("RAW_MODIFIED = false")
print("CAUSAL_CLAIM = false")
print("N_X_N_MATRIX = false")
print()


# =============================================================================
# 1. INPUT VALIDATION
# =============================================================================

for path in [CORE, MASTER]:

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo ausente: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Arquivo vazio: {path}"
        )


core_header = pd.read_csv(
    CORE,
    nrows=0,
)

missing_core = [
    c for c in CORE_COLS
    if c not in core_header.columns
]

if missing_core:
    raise RuntimeError(
        f"Colunas ausentes no CORE: {missing_core}"
    )


master_header = pd.read_csv(
    MASTER,
    nrows=0,
)

missing_master = [
    c for c in MASTER_REQUIRED
    if c not in master_header.columns
]

if missing_master:
    raise RuntimeError(
        f"Colunas ausentes no MASTER: {missing_master}"
    )


print("[PASS] input schemas validated")


# =============================================================================
# 2. LOAD ONLY REQUIRED COLUMNS
# =============================================================================

core = pd.read_csv(
    CORE,
    usecols=CORE_COLS,
    low_memory=False,
)

master = pd.read_csv(
    MASTER,
    usecols=MASTER_REQUIRED,
    low_memory=False,
)


print(f"[INFO] core rows   = {len(core):,}")
print(f"[INFO] master rows = {len(master):,}")


if core["order_id"].duplicated().any():
    raise RuntimeError(
        "CORE possui order_id duplicado."
    )

if master["order_id"].duplicated().any():
    raise RuntimeError(
        "MASTER possui order_id duplicado."
    )


df = core.merge(
    master,
    on="order_id",
    how="left",
    validate="one_to_one",
)


if len(df) != len(core):
    raise RuntimeError(
        "Merge alterou a população."
    )


# =============================================================================
# 3. VALIDITY + TRANSFORMATIONS
# =============================================================================

numeric_cols = [
    "total_price",
    "total_freight",
    "freight_burden",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "route_sellers_total",
    "any_interstate_route",
]

for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


valid = (
    df["freight_burden"].notna()
    &
    df["total_price"].notna()
    &
    (df["total_price"] > 0)
    &
    df["distance_freight_weighted_km"].notna()
    &
    (df["distance_freight_weighted_km"] >= 0)
    &
    df["total_weight_g"].notna()
    &
    (df["total_weight_g"] >= 0)
    &
    df["product_volume_sum_proxy_cm3"].notna()
    &
    (df["product_volume_sum_proxy_cm3"] >= 0)
    &
    df["route_sellers_total"].notna()
    &
    (df["route_sellers_total"] >= 1)
    &
    df["any_interstate_route"].isin([0, 1])
    &
    df["year_month"].notna()
)


analysis = df.loc[valid].copy()

analysis = add_logs(
    analysis
)


n_total = int(len(df))
n_analysis = int(len(analysis))


print(
    f"[INFO] complete burden-decomposition sample = "
    f"{n_analysis:,} / {n_total:,} "
    f"({100*n_analysis/n_total:.2f}%)"
)


if n_analysis < 1000:
    raise RuntimeError(
        "Amostra completa inesperadamente pequena."
    )


# =============================================================================
# 4. BURDEN THRESHOLDS
# =============================================================================

burden_quantiles = {
    "q10": float(
        df["freight_burden"].quantile(0.10)
    ),
    "q25": float(
        df["freight_burden"].quantile(0.25)
    ),
    "q50": float(
        df["freight_burden"].quantile(0.50)
    ),
    "q75": float(
        df["freight_burden"].quantile(0.75)
    ),
    "q80": float(
        df["freight_burden"].quantile(0.80)
    ),
    "q90": float(
        df["freight_burden"].quantile(0.90)
    ),
    "q95": float(
        df["freight_burden"].quantile(0.95)
    ),
}


pd.DataFrame(
    [
        {
            "quantile": key,
            "freight_burden": value,
        }
        for key, value
        in burden_quantiles.items()
    ]
).to_csv(
    OUT_THRESHOLDS,
    index=False,
)


for qname in ["q75", "q80", "q90"]:

    threshold = burden_quantiles[qname]

    df[
        f"high_burden_{qname}"
    ] = (
        df["freight_burden"]
        >= threshold
    ).astype("int8")


# =============================================================================
# 5. DESCRIPTIVE ASSOCIATIONS
#
# Pearson + Spearman.
#
# NOTE:
# Price/Burden has a definitional component because price
# enters the burden denominator.
# =============================================================================

association_variables = [
    "total_price",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "route_sellers_total",
    "any_interstate_route",
]


assoc_rows = []

for variable in association_variables:

    sub = analysis[
        [
            "freight_burden",
            variable,
        ]
    ].dropna()

    pearson = sub[
        "freight_burden"
    ].corr(
        sub[variable],
        method="pearson",
    )

    spearman = sub[
        "freight_burden"
    ].corr(
        sub[variable],
        method="spearman",
    )

    assoc_rows.append({
        "variable":
            variable,

        "n":
            int(len(sub)),

        "pearson_r":
            safe_float(pearson),

        "spearman_rho":
            safe_float(spearman),

        "interpretation_guardrail":
            (
                "PARTLY_DEFINITIONAL"
                if variable == "total_price"
                else
                "ASSOCIATIONAL_ONLY"
            ),
    })


assoc_df = pd.DataFrame(
    assoc_rows
)

assoc_df.to_csv(
    OUT_ASSOC,
    index=False,
)


# =============================================================================
# 6. DECILE CURVES
# =============================================================================

decile_tables = []

for variable in [
    "total_price",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
]:

    tmp = make_decile_table(
        analysis,
        variable,
    )

    if len(tmp):
        decile_tables.append(tmp)


if decile_tables:

    deciles = pd.concat(
        decile_tables,
        ignore_index=True,
    )

else:
    deciles = pd.DataFrame()


deciles.to_csv(
    OUT_DECILES,
    index=False,
)


# =============================================================================
# 7. ROUTE-STRUCTURE DESCRIPTIVES
# =============================================================================

analysis["multi_seller_order"] = (
    analysis["route_sellers_total"] > 1
).astype("int8")


group_rows = []


for field in [
    "multi_seller_order",
    "any_interstate_route",
]:

    for level, grp in analysis.groupby(
        field,
        observed=True,
    ):

        group_rows.append({
            "group_variable":
                field,

            "group_level":
                int(level),

            "n":
                int(len(grp)),

            "burden_mean":
                float(
                    grp["freight_burden"].mean()
                ),

            "burden_median":
                float(
                    grp["freight_burden"].median()
                ),

            "price_median":
                float(
                    grp["total_price"].median()
                ),

            "freight_median":
                float(
                    grp["total_freight"].median()
                ),

            "distance_median":
                float(
                    grp[
                        "distance_freight_weighted_km"
                    ].median()
                ),
        })


group_df = pd.DataFrame(
    group_rows
)

group_df.to_csv(
    OUT_GROUPS,
    index=False,
)


# =============================================================================
# 8. FRACTIONAL LOGIT
#
# Papke-Wooldridge style conditional mean:
#
# E(B | X) = logistic(X beta)
#
# M0:
#   price only
#
# M1:
#   price + physical + route structure
#
# M2:
#   M1 + month fixed effects
#
# M2 is the preferred decomposition model.
#
# Robust HC1 covariance.
# =============================================================================

formula_m0 = (
    "freight_burden ~ "
    "log_price"
)

formula_m1 = (
    "freight_burden ~ "
    "log_price + "
    "log_distance + "
    "log_weight + "
    "log_volume + "
    "route_sellers_total + "
    "any_interstate_route"
)

formula_m2 = (
    "freight_burden ~ "
    "log_price + "
    "log_distance + "
    "log_weight + "
    "log_volume + "
    "route_sellers_total + "
    "any_interstate_route + "
    "C(year_month)"
)


def fit_fractional(formula):

    model = smf.glm(
        formula=formula,
        data=analysis,
        family=sm.families.Binomial(),
    )

    result = model.fit(
        maxiter=200,
        tol=1e-8,
        cov_type="HC1",
    )

    return result


print()
print("[INFO] fitting fractional logit M0 ...")

fit_m0 = fit_fractional(
    formula_m0
)


print("[INFO] fitting fractional logit M1 ...")

fit_m1 = fit_fractional(
    formula_m1
)


print("[INFO] fitting fractional logit M2 ...")

fit_m2 = fit_fractional(
    formula_m2
)


print("[PASS] fractional models fitted")


# =============================================================================
# 9. COEFFICIENT TABLE + FDR FOR FOCAL FAMILY
# =============================================================================

coef_rows = []


for model_name, fit in [
    ("M0_PRICE", fit_m0),
    ("M1_PHYSICAL", fit_m1),
    ("M2_PHYSICAL_TIME", fit_m2),
]:

    for term in fit.params.index:

        coef_rows.append({
            "model":
                model_name,

            "term":
                term,

            "coef":
                safe_float(
                    fit.params[term]
                ),

            "std_err_HC1":
                safe_float(
                    fit.bse[term]
                ),

            "z":
                safe_float(
                    fit.tvalues[term]
                ),

            "p_value":
                safe_float(
                    fit.pvalues[term]
                ),

            "converged":
                bool(
                    fit.converged
                ),
        })


coef_df = pd.DataFrame(
    coef_rows
)


# FDR only across the six focal M2 structural terms.
mask_focal = (
    (coef_df["model"] == "M2_PHYSICAL_TIME")
    &
    coef_df["term"].isin(
        FOCAL_TERMS
    )
)


focal_p = coef_df.loc[
    mask_focal,
    "p_value",
].to_numpy(
    dtype=float
)


if len(focal_p):

    reject, p_adj, _, _ = multipletests(
        focal_p,
        alpha=0.05,
        method="fdr_bh",
    )

    coef_df.loc[
        mask_focal,
        "p_value_fdr_bh",
    ] = p_adj

    coef_df.loc[
        mask_focal,
        "fdr_reject_005",
    ] = reject.astype(int)

else:

    coef_df["p_value_fdr_bh"] = np.nan
    coef_df["fdr_reject_005"] = np.nan


coef_df.to_csv(
    OUT_COEFS,
    index=False,
)


# =============================================================================
# 10. BUSINESS-READABLE +10% SCENARIO EFFECTS
#
# These are ASSOCIATIONAL MODEL CONTRASTS.
# They are NOT causal elasticities.
# =============================================================================

effect_rows = []


for variable in [
    "total_price",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
]:

    effect_rows.append(
        scenario_effect(
            fit_m2,
            analysis,
            variable,
            relative_change=0.10,
        )
    )


# Interstate discrete contrast.
base0 = analysis.copy()
base1 = analysis.copy()

base0["any_interstate_route"] = 0
base1["any_interstate_route"] = 1

pred0 = np.asarray(
    fit_m2.predict(base0),
    dtype=float,
)

pred1 = np.asarray(
    fit_m2.predict(base1),
    dtype=float,
)

delta_interstate = (
    pred1 - pred0
)


effect_rows.append({
    "scenario":
        "INTERSTATE_0_TO_1",

    "variable":
        "any_interstate_route",

    "change":
        1,

    "mean_predicted_burden_delta":
        float(
            np.mean(
                delta_interstate
            )
        ),

    "median_predicted_burden_delta":
        float(
            np.median(
                delta_interstate
            )
        ),

    "p10_delta":
        float(
            np.quantile(
                delta_interstate,
                0.10,
            )
        ),

    "p90_delta":
        float(
            np.quantile(
                delta_interstate,
                0.90,
            )
        ),
})


effects_df = pd.DataFrame(
    effect_rows
)

effects_df.to_csv(
    OUT_EFFECTS,
    index=False,
)


# =============================================================================
# 11. SEGMENT MATERIALIZATION
#
# No interpretation as subsidy candidate yet.
# These flags are only distributional burden markers.
# =============================================================================

ticket_q = qcut_safe(
    df["total_price"],
    4,
)

distance_q = qcut_safe(
    df["distance_freight_weighted_km"],
    4,
)

weight_q = qcut_safe(
    df["total_weight_g"],
    4,
)

volume_q = qcut_safe(
    df["product_volume_sum_proxy_cm3"],
    4,
)


segments = pd.DataFrame({
    "order_id":
        df["order_id"],

    "year_month":
        df["year_month"],

    "freight_burden":
        df["freight_burden"],

    "high_burden_q75":
        df["high_burden_q75"],

    "high_burden_q80":
        df["high_burden_q80"],

    "high_burden_q90":
        df["high_burden_q90"],

    "ticket_quartile":
        ticket_q,

    "distance_quartile":
        distance_q,

    "weight_quartile":
        weight_q,

    "volume_quartile":
        volume_q,

    "route_sellers_total":
        df["route_sellers_total"],

    "multi_seller_order":
        (
            df["route_sellers_total"] > 1
        ).astype("int8"),

    "any_interstate_route":
        df["any_interstate_route"],
})


segments.to_csv(
    OUT_SEGMENTS,
    index=False,
)


# =============================================================================
# 12. SCIENTIFIC CHECKS
# =============================================================================

q75 = burden_quantiles["q75"]
q80 = burden_quantiles["q80"]
q90 = burden_quantiles["q90"]


focal_m2 = coef_df[
    (
        coef_df["model"]
        == "M2_PHYSICAL_TIME"
    )
    &
    coef_df["term"].isin(
        FOCAL_TERMS
    )
].copy()


finite_focal = bool(
    np.isfinite(
        focal_m2["coef"]
    ).all()
)


effects_finite = bool(
    np.isfinite(
        effects_df[
            "mean_predicted_burden_delta"
        ]
    ).all()
)


gates = {
    "B02_G01_block01_core_exists":
        CORE.exists(),

    "B02_G02_population_preserved_after_merge":
        len(df) == len(core),

    "B02_G03_order_id_unique":
        not df["order_id"].duplicated().any(),

    "B02_G04_complete_analysis_sample_nontrivial":
        n_analysis >= 1000,

    "B02_G05_burden_bounded":
        bool(
            df["freight_burden"]
            .dropna()
            .between(0, 1)
            .all()
        ),

    "B02_G06_threshold_order_valid":
        bool(
            q75 <= q80 <= q90
        ),

    "B02_G07_fractional_M0_converged":
        bool(
            fit_m0.converged
        ),

    "B02_G08_fractional_M1_converged":
        bool(
            fit_m1.converged
        ),

    "B02_G09_fractional_M2_converged":
        bool(
            fit_m2.converged
        ),

    "B02_G10_focal_coefficients_finite":
        finite_focal,

    "B02_G11_scenario_effects_finite":
        effects_finite,

    "B02_G12_final_predictive_model_not_refit":
        True,

    "B02_G13_raw_untouched":
        True,
}


failed = [
    name
    for name, ok
    in gates.items()
    if not ok
]


status = (
    "PASS"
    if not failed
    else "FAIL"
)


# =============================================================================
# 13. SCIENTIFIC SUMMARY
# =============================================================================

effect_map = {
    row["variable"]:
        row["mean_predicted_burden_delta"]

    for _, row
    in effects_df.iterrows()
}


m2_map = {
    row["term"]:
        row["coef"]

    for _, row
    in focal_m2.iterrows()
}


decision = {
    "status":
        status,

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "module":
        "27A_BLOCK02_BURDEN_DECOMPOSITION",

    "population": {
        "n_orders":
            n_total,

        "n_complete_case_fractional_model":
            n_analysis,

        "complete_case_pct":
            float(
                100
                *
                n_analysis
                /
                n_total
            ),
    },

    "burden_thresholds": {
        "q75":
            q75,

        "q80":
            q80,

        "q90":
            q90,
    },

    "preferred_model": {
        "name":
            "M2_PHYSICAL_TIME",

        "family":
            "Fractional Logit / Binomial GLM",

        "robust_covariance":
            "HC1",

        "formula":
            formula_m2,

        "converged":
            bool(
                fit_m2.converged
            ),
    },

    "focal_coefficients": {
        term:
            safe_float(
                m2_map.get(term)
            )

        for term in FOCAL_TERMS
    },

    "plus_10pct_average_predicted_burden_change": {
        key:
            safe_float(value)

        for key, value
        in effect_map.items()
    },

    "interpretation_guardrails": {
        "price_effect_partly_definitional":
            True,

        "fractional_model_is_causal":
            False,

        "scenario_effects_are_causal_elasticities":
            False,

        "high_burden_equals_operational_inefficiency":
            False,

        "high_burden_equals_subsidy_candidate":
            False,

        "final_predictive_model_refit":
            False,

        "raw_modified":
            False,

        "carrier_identity_available":
            False,
    },

    "gates":
        gates,

    "failed_gates":
        failed,

    "outputs": [
        str(
            OUT_SEGMENTS.relative_to(ROOT)
        ),
        str(
            OUT_ASSOC.relative_to(ROOT)
        ),
        str(
            OUT_DECILES.relative_to(ROOT)
        ),
        str(
            OUT_GROUPS.relative_to(ROOT)
        ),
        str(
            OUT_COEFS.relative_to(ROOT)
        ),
        str(
            OUT_EFFECTS.relative_to(ROOT)
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
# 14. HUMAN-READABLE REPORT
# =============================================================================

focal_display = focal_m2[
    [
        "term",
        "coef",
        "std_err_HC1",
        "p_value",
        "p_value_fdr_bh",
        "fdr_reject_005",
    ]
].copy()


report = f"""
====================================================================================================
27A — BLOCK 02 — FREIGHT BURDEN DECOMPOSITION
====================================================================================================

STATUS
------
{status}

POPULATION
----------
Total orders:
    {n_total:,}

Complete-case burden decomposition sample:
    {n_analysis:,}

Coverage:
    {100*n_analysis/n_total:.2f}%

BURDEN THRESHOLDS
-----------------
Q75:
    {q75:.8f}

Q80:
    {q80:.8f}

Q90:
    {q90:.8f}

PREFERRED DIAGNOSTIC MODEL
--------------------------
Fractional Logit / Binomial GLM
HC1 robust covariance
Month fixed effects included.

Formula:
    {formula_m2}

Converged:
    {fit_m2.converged}

FOCAL COEFFICIENTS
------------------
{focal_display.to_string(index=False)}

BUSINESS-READABLE MODEL CONTRASTS
---------------------------------
Average predicted change in Freight Burden under +10% covariate contrasts.

IMPORTANT:
These are model-based ASSOCIATIONAL contrasts.
They are NOT causal elasticities.

{effects_df.to_string(index=False)}

INTERPRETATION GUARDRAILS
-------------------------
1. Freight Burden measures consumer-side shipping burden.

2. A high Freight Burden does NOT, by itself, imply logistics inefficiency.

3. Price is present in the denominator of Freight Burden.
   Therefore the Price/Burden association is partly mathematical by construction.

4. Distance, weight, volume and route structure are interpreted associationally.

5. No abandonment or conversion effect is estimated.

6. No subsidy opportunity is declared in Block 02.

7. Seller, regional and service-performance interpretation belongs to later blocks.

FAILED GATES
------------
{failed if failed else "NONE"}

NEXT BLOCK
----------
27A BLOCK 03 — SERVICE PERFORMANCE

The next block will construct and audit:

    Actual Delivery Time
    Promised Delivery Time
    Delivery Gap
    Positive Tardiness
    Late Delivery / reliability

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
print("=" * 104)
print("BURDEN ASSOCIATIONS")
print("=" * 104)
print(
    assoc_df.to_string(
        index=False
    )
)

print()
print("=" * 104)
print("FRACTIONAL LOGIT — FOCAL TERMS")
print("=" * 104)
print(
    focal_display.to_string(
        index=False
    )
)

print()
print("=" * 104)
print("SCENARIO EFFECTS")
print("=" * 104)
print(
    effects_df.to_string(
        index=False
    )
)

print()
print("=" * 104)
print("BLOCK 02 DECISION")
print("=" * 104)
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
        f"BLOCK 02 FAILED: {failed}"
    )

print("[PASS 27A-B02] FREIGHT BURDEN DECOMPOSITION COMPLETE")
print("FINAL_PREDICTIVE_MODEL_REFIT = false")
print("DIAGNOSTIC_FRACTIONAL_MODEL_FIT = true")
print("RAW_MODIFIED = false")
print("CAUSAL_CLAIM = false")
print("PARAR AQUI")

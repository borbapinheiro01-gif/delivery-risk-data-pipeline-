#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

ART = (
    ROOT
    / "artifacts"
    / "spatiotemporal_logistics"
)

BASE_REP = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
)

SCI = BASE_REP / "scientific"

SCI.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 110)
print("SCIENTIFIC AUDIT 10 — VALIDATION OF MODULES 06–09")
print("=" * 110)


# =====================================================================
# 1. MODULE 06
# =====================================================================

print()
print("=" * 110)
print("1. MODULE 06 — COST STRUCTURE")
print("=" * 110)

with open(
    SCI / "06h_COST_STRUCTURE_AUDIT.json",
    "r",
    encoding="utf-8"
) as f:
    m06 = json.load(f)

sens06 = pd.read_csv(
    SCI / "06g_cost_sensitivity.csv"
)

print("M06 status:", m06["status"])
print()
print(sens06.to_string(index=False))


# =====================================================================
# 2. MODULE 07 — REAL RELATIVE AUDIT
# =====================================================================

print()
print("=" * 110)
print("2. MODULE 07 — EXPECTED FREIGHT / RESIDUAL")
print("=" * 110)

comp = pd.read_csv(
    BASE_REP
    / "09_freight_model_comparison.csv"
)

print("MODEL COMPARISON FROM STAGE 04:")
print(comp.to_string(index=False))

# Normalize names for robust discovery.
cols_lower = {
    c.lower(): c
    for c in comp.columns
}

model_col = next(
    (
        c
        for c in comp.columns
        if "model" in c.lower()
    ),
    None
)

if model_col is None:
    raise RuntimeError(
        "Não encontrei coluna de modelo em "
        "09_freight_model_comparison.csv"
    )

mae_col = next(
    (
        c
        for c in comp.columns
        if c.lower() == "mae"
        or "mae" in c.lower()
    ),
    None
)

if mae_col is None:
    raise RuntimeError(
        "Não encontrei MAE na comparação de frete."
    )

comp["_model_norm"] = (
    comp[model_col]
    .astype(str)
    .str.upper()
    .str.strip()
)

def find_model(prefix):
    q = comp[
        comp["_model_norm"]
        .str.contains(prefix, regex=False)
    ]

    if q.empty:
        return None

    return q.iloc[0]

f0 = find_model("F0")
f1 = find_model("F1")
f2 = find_model("F2")

if f0 is None or f2 is None:
    raise RuntimeError(
        "F0 e/ou F2 não encontrados na comparação."
    )

mae_f0 = float(f0[mae_col])
mae_f2 = float(f2[mae_col])

mae_gain_f2_vs_f0 = (
    1.0
    -
    mae_f2 / mae_f0
    if mae_f0 > 0
    else np.nan
)

f2_beats_f0 = bool(
    mae_f2 < mae_f0
)

if f1 is not None:
    mae_f1 = float(f1[mae_col])
    f2_vs_f1 = mae_f2 - mae_f1
else:
    mae_f1 = np.nan
    f2_vs_f1 = np.nan

res_assoc = pd.read_csv(
    SCI
    / "07f_residual_structural_associations.csv"
)

max_abs_pearson = float(
    res_assoc["pearson_r"]
    .abs()
    .max()
)

max_abs_spearman = float(
    res_assoc["spearman_rho"]
    .abs()
    .max()
)

drift = pd.read_csv(
    SCI
    / "07g_residual_temporal_drift.csv"
)

median_drift_range = float(
    drift["residual_median"].max()
    -
    drift["residual_median"].min()
)

# No arbitrary cutoffs here.
# Report evidence dimensions rather than inventing a universal pass line.
relative_evidence = {
    "f2_beats_f0_mae": f2_beats_f0,
    "mae_f0": mae_f0,
    "mae_f1": None if np.isnan(mae_f1) else mae_f1,
    "mae_f2": mae_f2,
    "mae_gain_f2_vs_f0": mae_gain_f2_vs_f0,
    "mae_difference_f2_minus_f1":
        None if np.isnan(f2_vs_f1) else f2_vs_f1,
    "max_abs_residual_pearson":
        max_abs_pearson,
    "max_abs_residual_spearman":
        max_abs_spearman,
    "residual_monthly_median_range":
        median_drift_range,
}

print()
print("RELATIVE EVIDENCE:")
print(
    json.dumps(
        relative_evidence,
        indent=2,
        ensure_ascii=False
    )
)


# =====================================================================
# 3. MODULE 08 — REFIT WITH EXPLICIT CONVERGENCE CHECK
# =====================================================================

print()
print("=" * 110)
print("3. MODULE 08 — MIXEDLM CONVERGENCE AUDIT")
print("=" * 110)

master = pd.read_csv(
    ART
    / "06_EXPECTED_FREIGHT_OOT.csv"
)

speed = master[
    master["actual_delivery_days"].notna()
    &
    master["distance_freight_weighted_km"].notna()
].copy()

speed["log_speed"] = np.log1p(
    speed["actual_delivery_days"]
    .where(
        speed["actual_delivery_days"] >= 0
    )
)

speed["log_distance"] = np.log1p(
    speed["distance_freight_weighted_km"]
    .where(
        speed["distance_freight_weighted_km"] >= 0
    )
)

speed["log_weight"] = np.log1p(
    speed["total_weight_g"]
    .where(
        speed["total_weight_g"] >= 0
    )
)

speed["log_volume"] = np.log1p(
    speed["product_volume_sum_proxy_cm3"]
    .where(
        speed["product_volume_sum_proxy_cm3"] >= 0
    )
)

speed["log_freight"] = np.log1p(
    speed["total_freight"]
    .where(
        speed["total_freight"] >= 0
    )
)

den = (
    speed["total_price"]
    +
    speed["total_freight"]
)

speed["freight_burden"] = np.where(
    den > 0,
    speed["total_freight"] / den,
    np.nan
)

speed["log_pop"] = np.log1p(
    speed["customer_population"]
    .where(
        speed["customer_population"] >= 0
    )
)

speed["log_gdp"] = np.log1p(
    speed["customer_gdp_per_capita"]
    .where(
        speed["customer_gdp_per_capita"] >= 0
    )
)

predictors = [
    "log_distance",
    "log_weight",
    "log_volume",
    "log_freight",
    "freight_burden",
    "freight_residual_OOT",
    "log_pop",
    "log_gdp",
]

speed_std = speed.dropna(
    subset=
        predictors
        +
        [
            "customer_state",
            "log_speed",
        ]
).copy()

for col in predictors:

    sd = float(
        speed_std[col].std()
    )

    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(
            f"Preditor sem variabilidade: {col}"
        )

    speed_std[
        col + "_std"
    ] = (
        speed_std[col]
        -
        speed_std[col].mean()
    ) / sd


formula = (
    "log_speed ~ "
    "log_distance_std + "
    "log_weight_std + "
    "log_volume_std + "
    "log_freight_std + "
    "freight_burden_std + "
    "freight_residual_OOT_std + "
    "log_pop_std + "
    "log_gdp_std + "
    "truck_strike_2018"
)

model = smf.mixedlm(
    formula,
    speed_std,
    groups=speed_std[
        "customer_state"
    ],
)

optimizers = [
    "lbfgs",
    "bfgs",
    "cg",
]

attempts = []

best_fit = None

for method in optimizers:

    caught = []

    try:

        with warnings.catch_warnings(
            record=True
        ) as w:

            warnings.simplefilter(
                "always",
                ConvergenceWarning
            )

            fit = model.fit(
                method=method,
                reml=False,
                maxiter=1000,
                disp=False,
            )

            caught = [
                str(x.message)
                for x in w
                if issubclass(
                    x.category,
                    ConvergenceWarning
                )
            ]

        converged = bool(
            getattr(
                fit,
                "converged",
                False
            )
        )

        attempts.append(
            {
                "optimizer": method,
                "converged": converged,
                "llf": float(fit.llf),
                "warnings":
                    " | ".join(caught),
            }
        )

        print(
            f"{method:<8} "
            f"converged={converged} "
            f"llf={fit.llf:.6f}"
        )

        if converged:

            if (
                best_fit is None
                or
                fit.llf > best_fit.llf
            ):
                best_fit = fit

    except Exception as exc:

        attempts.append(
            {
                "optimizer": method,
                "converged": False,
                "llf": np.nan,
                "warnings":
                    f"{type(exc).__name__}: {exc}",
            }
        )

        print(
            f"{method:<8} FAIL "
            f"{type(exc).__name__}: {exc}"
        )


attempt_df = pd.DataFrame(
    attempts
)

attempt_df.to_csv(
    SCI
    / "10a_mixedlm_optimizer_audit.csv",
    index=False
)


if best_fit is None:

    mixed_status = (
        "NON_CONVERGED"
    )

    residual_result = None

    print()
    print(
        "[FAIL] Nenhum otimizador "
        "convergiu."
    )

else:

    mixed_status = (
        "CONVERGED"
    )

    term = (
        "freight_residual_OOT_std"
    )

    coef = float(
        best_fit.params[term]
    )

    se = float(
        best_fit.bse[term]
    )

    residual_result = {
        "coef": coef,
        "se": se,
        "ci95_lower":
            coef - 1.96 * se,
        "ci95_upper":
            coef + 1.96 * se,
    }

    print()
    print(
        "[PASS] MixedLM convergence "
        "validated."
    )

    print(
        json.dumps(
            residual_result,
            indent=2
        )
    )


# =====================================================================
# 4. MODULE 09 — BETA-BINOMIAL + FREQUENTIST OR
# =====================================================================

print()
print("=" * 110)
print("4. MODULE 09 — RELIABILITY RESULTS")
print("=" * 110)

beta = pd.read_csv(
    SCI
    / "09c_beta_binomial_residual_groups.csv"
)

odds = pd.read_csv(
    SCI
    / "09h_frequentist_odds_ratios.csv"
)

with open(
    SCI
    / "09m_RELIABILITY_DECISION.json",
    "r",
    encoding="utf-8"
) as f:
    rel_decision = json.load(f)

print()
print("BETA-BINOMIAL:")
print(
    beta.to_string(
        index=False
    )
)

print()
print("FREQUENTIST ODDS RATIOS:")
print(
    odds.to_string(
        index=False
    )
)

print()
print("DECISION:")
print(
    json.dumps(
        rel_decision,
        indent=2,
        ensure_ascii=False
    )
)


# =====================================================================
# 5. INTEGRATED SCIENTIFIC CHECKPOINT
# =====================================================================

checkpoint = {
    "module_06_status":
        m06["status"],

    "module_07_relative_evidence":
        relative_evidence,

    "module_08_mixedlm_status":
        mixed_status,

    "module_08_freight_residual":
        residual_result,

    "module_09_beta_binomial": {
        "P_p_low_gt_p_normal":
            rel_decision[
                "P_p_low_gt_p_normal"
            ],

        "P_p_high_gt_p_normal":
            rel_decision[
                "P_p_high_gt_p_normal"
            ],
    },

    "module_09_odds_ratio_residual": {
        "R_A":
            rel_decision[
                "odds_ratio_residual_observed_R_A"
            ],

        "R_B_promise_adjusted":
            rel_decision[
                "odds_ratio_residual_promise_adjusted_R_B"
            ],
    },
}

with open(
    SCI
    / "10b_SCIENTIFIC_CHECKPOINT.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        checkpoint,
        f,
        indent=2,
        ensure_ascii=False
    )


report_lines = [
    "=" * 110,
    "SCIENTIFIC CHECKPOINT — MODULES 06–09",
    "=" * 110,
    "",
    f"M06 COST STRUCTURE            : {m06['status']}",
    "",
    "M07 FREIGHT MODEL",
    f"F2 beats F0 MAE              : {f2_beats_f0}",
    f"MAE gain F2 vs F0            : {mae_gain_f2_vs_f0:.6f}",
    f"Max |Pearson residual|        : {max_abs_pearson:.6f}",
    f"Max |Spearman residual|       : {max_abs_spearman:.6f}",
    "",
    "M08 DELIVERY SPEED",
    f"MixedLM status                : {mixed_status}",
]

if residual_result:

    report_lines += [
        f"Residual standardized coef    : {residual_result['coef']:.6f}",
        f"Residual 95% CI               : "
        f"[{residual_result['ci95_lower']:.6f}, "
        f"{residual_result['ci95_upper']:.6f}]",
    ]

report_lines += [
    "",
    "M09 DELIVERY RELIABILITY",
    f"P(p_LOW > p_NORMAL | data)    : "
    f"{rel_decision['P_p_low_gt_p_normal']:.6f}",

    f"P(p_HIGH > p_NORMAL | data)   : "
    f"{rel_decision['P_p_high_gt_p_normal']:.6f}",

    f"Residual OR R-A               : "
    f"{rel_decision['odds_ratio_residual_observed_R_A']:.6f}",

    f"Residual OR R-B promise adj.  : "
    f"{rel_decision['odds_ratio_residual_promise_adjusted_R_B']:.6f}",

    "",
    "=" * 110,
    "END",
    "=" * 110,
]

report = "\n".join(
    report_lines
)

(
    SCI
    / "10c_SCIENTIFIC_CHECKPOINT_REPORT.txt"
).write_text(
    report,
    encoding="utf-8"
)

print()
print(report)

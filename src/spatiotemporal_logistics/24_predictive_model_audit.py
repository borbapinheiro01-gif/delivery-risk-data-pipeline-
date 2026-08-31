#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

SCI = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
    / "scientific"
)

GN = (
    SCI
    / "gnadmm_delivery"
)

OUT = (
    ROOT
    / "reports"
    / "final_closeout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


def read_csv_required(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo obrigatório ausente: {path}"
        )

    return pd.read_csv(path)


def read_json_required(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo obrigatório ausente: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


print("=" * 105)
print("MODULE 24 — FINAL PREDICTIVE MODEL AUDIT")
print("=" * 105)


# ==============================================================================================
# 1. LOAD EXISTING RESULTS
# ==============================================================================================

nonlinear = read_csv_required(
    SCI
    / "11b_nonlinear_oot_performance.csv"
)

gn_primary = read_csv_required(
    GN
    / "18c_gnadmm_oot_model_summary.csv"
)

ablation = read_csv_required(
    GN
    / "20c_constraint_ablation_summary.csv"
)

calibration = read_csv_required(
    GN
    / "21a_oot_calibration_summary.csv"
)

temporal_decision = read_json_required(
    GN
    / "19d_PAIRED_TEMPORAL_DECISION.json"
)

ablation_decision = read_json_required(
    GN
    / "20d_CONSTRAINT_ABLATION_DECISION.json"
)

calibration_decision = read_json_required(
    GN
    / "21d_CALIBRATION_DECISION.json"
)


# ==============================================================================================
# 2. BUILD PREDICTIVE REGISTRY — NONLINEAR FAMILY
# ==============================================================================================

nonlinear_summary = (
    nonlinear
    .groupby("model")
    .agg(
        months=(
            "test_month",
            "nunique"
        ),
        mean_AP=(
            "ap_oot",
            "mean"
        ),
        median_AP=(
            "ap_oot",
            "median"
        ),
        mean_Brier=(
            "brier_oot",
            "mean"
        ),
        median_Brier=(
            "brier_oot",
            "median"
        ),
    )
    .reset_index()
)

nonlinear_summary[
    "family"
] = "NONLINEAR_RELIABILITY"

nonlinear_summary[
    "mean_LogLoss"
] = np.nan

nonlinear_summary[
    "mean_AUC"
] = np.nan


# ==============================================================================================
# 3. GN PRIMARY FAMILY
# ==============================================================================================

gn_reg = gn_primary.rename(
    columns={
        "mean_ap":
            "mean_AP",

        "median_ap":
            "median_AP",

        "mean_brier":
            "mean_Brier",

        "mean_log_loss":
            "mean_LogLoss",

        "mean_auc":
            "mean_AUC",
    }
).copy()

gn_reg[
    "median_Brier"
] = np.nan

gn_reg[
    "family"
] = "GNADMM_PRIMARY"


# ==============================================================================================
# 4. GN CONSTRAINT ABLATION FAMILY
# ==============================================================================================

abl_reg = ablation[
    [
        "model",
        "months",
        "mean_AP",
        "median_AP",
        "mean_Brier",
        "mean_LogLoss",
        "mean_AUC",
    ]
].copy()

abl_reg[
    "median_Brier"
] = np.nan

abl_reg[
    "family"
] = "GNADMM_CONSTRAINT_ABLATION"


# ==============================================================================================
# 5. UNIFIED REGISTRY
# ==============================================================================================

columns_final = [
    "family",
    "model",
    "months",
    "mean_AP",
    "median_AP",
    "mean_Brier",
    "median_Brier",
    "mean_LogLoss",
    "mean_AUC",
]

registry = pd.concat(
    [
        nonlinear_summary[
            columns_final
        ],

        gn_reg[
            columns_final
        ],

        abl_reg[
            columns_final
        ],
    ],
    ignore_index=True,
)

registry.to_csv(
    OUT
    / "24a_predictive_model_registry.csv",
    index=False,
)


# ==============================================================================================
# 6. CALIBRATION REGISTRY
# ==============================================================================================

calibration.to_csv(
    OUT
    / "24b_calibration_registry.csv",
    index=False,
)


# ==============================================================================================
# 7. METRIC WINNERS — USE CONSTRAINT ABLATION AS FINAL GN COMPARISON
# ==============================================================================================

winner_rows = []

metric_rules = [
    (
        "Average Precision",
        "mean_AP",
        "MAX"
    ),

    (
        "Brier",
        "mean_Brier",
        "MIN"
    ),

    (
        "LogLoss",
        "mean_LogLoss",
        "MIN"
    ),

    (
        "ROC-AUC",
        "mean_AUC",
        "MAX"
    ),
]


for label, column, direction in metric_rules:

    x = ablation.dropna(
        subset=[column]
    )

    if x.empty:
        continue

    if direction == "MAX":
        row = x.loc[
            x[column].idxmax()
        ]

    else:
        row = x.loc[
            x[column].idxmin()
        ]

    winner_rows.append(
        {
            "metric":
                label,

            "direction":
                direction,

            "winner":
                str(
                    row["model"]
                ),

            "value":
                float(
                    row[column]
                ),
        }
    )


winners = pd.DataFrame(
    winner_rows
)

winners.to_csv(
    OUT
    / "24c_metric_winners.csv",
    index=False,
)


# ==============================================================================================
# 8. CALIBRATION WINNERS — KEEP METRICS SEPARATE
# ==============================================================================================

cal_winners = {
    "best_pooled_Brier":
        calibration_decision.get(
            "best_pooled_OOT_Brier"
        ),

    "best_ECE":
        calibration_decision.get(
            "best_pooled_OOT_ECE"
        ),

    "best_intercept_slope_proximity":
        calibration_decision.get(
            "best_calibration_intercept_slope_proximity"
        ),

    "important_guardrail":
        (
            "Brier is an overall proper probabilistic score "
            "and is not interpreted as calibration alone."
        ),
}


# ==============================================================================================
# 9. UNIVERSAL-WINNER TEST
# ==============================================================================================

metric_winner_names = (
    winners["winner"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

cal_winner_names = [
    value
    for key, value
    in cal_winners.items()
    if (
        key.startswith("best_")
        and
        value is not None
    )
]

all_winner_names = (
    metric_winner_names
    +
    cal_winner_names
)

unique_all = sorted(
    set(
        all_winner_names
    )
)

universal_winner = (
    unique_all[0]
    if len(unique_all) == 1
    else None
)


# ==============================================================================================
# 10. NONLINEAR FINAL CHECK
# ==============================================================================================

nl_linear = nonlinear_summary[
    nonlinear_summary["model"].eq(
        "N1_Linear"
    )
]

nl_flex = nonlinear_summary[
    nonlinear_summary["model"].isin(
        [
            "N2_Quadratic",
            "N3_Natural_Spline_df4",
        ]
    )
]

nonlinear_final = {
    "status":
        "INCONCLUSIVE"
}

if (
    not nl_linear.empty
    and
    not nl_flex.empty
):

    linear_ap = float(
        nl_linear.iloc[0][
            "mean_AP"
        ]
    )

    linear_brier = float(
        nl_linear.iloc[0][
            "mean_Brier"
        ]
    )

    best_flex_ap = float(
        nl_flex[
            "mean_AP"
        ].max()
    )

    best_flex_brier = float(
        nl_flex[
            "mean_Brier"
        ].min()
    )

    nonlinear_final = {
        "status":
            (
                "NO_ROBUST_OOT_GAIN"
                if (
                    best_flex_ap <= linear_ap
                    or
                    best_flex_brier >= linear_brier
                )
                else
                "FLEXIBLE_MODEL_GAIN"
            ),

        "linear_mean_AP":
            linear_ap,

        "best_flexible_mean_AP":
            best_flex_ap,

        "linear_mean_Brier":
            linear_brier,

        "best_flexible_mean_Brier":
            best_flex_brier,
    }


# ==============================================================================================
# 11. FINAL DECISION
# ==============================================================================================

decision = {
    "status":
        "PASS",

    "evaluation_scope":
        "TEMPORAL_OUT_OF_TIME",

    "primary_metric":
        "AVERAGE_PRECISION",

    "nonlinear_audit":
        nonlinear_final,

    "gnadmm_temporal_AP_evidence":
        temporal_decision.get(
            "AP_evidence_classification"
        ),

    "mean_GNADMM_AP_improvement_vs_logit":
        temporal_decision.get(
            "mean_AP_improvement"
        ),

    "metric_winners":
        winners.to_dict(
            orient="records"
        ),

    "calibration_winners":
        cal_winners,

    "universal_winner":
        universal_winner,

    "universal_superiority_supported":
        bool(
            universal_winner is not None
        ),

    "scientific_interpretation":
        (
            "Performance is evaluated across ranking/discrimination, "
            "proper probabilistic losses and calibration diagnostics. "
            "Different metric winners are preserved rather than forcing "
            "a single universal model ranking."
        ),

    "raw_modified":
        False,
}


(
    OUT
    / "24d_PREDICTIVE_MODEL_AUDIT.json"
).write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==============================================================================================
# 12. HUMAN-READABLE REPORT
# ==============================================================================================

lines = []

lines.append(
    "=" * 105
)

lines.append(
    "FINAL PREDICTIVE MODEL AUDIT"
)

lines.append(
    "=" * 105
)

lines.append("")

lines.append(
    "METRIC WINNERS"
)

lines.append(
    "-" * 105
)

lines.append(
    winners.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "CALIBRATION"
)

lines.append(
    "-" * 105
)

lines.append(
    json.dumps(
        cal_winners,
        indent=2,
        ensure_ascii=False,
    )
)

lines.append("")

lines.append(
    "NONLINEAR AUDIT"
)

lines.append(
    "-" * 105
)

lines.append(
    json.dumps(
        nonlinear_final,
        indent=2,
        ensure_ascii=False,
    )
)

lines.append("")

lines.append(
    "GNADMM TEMPORAL EVIDENCE"
)

lines.append(
    "-" * 105
)

lines.append(
    (
        f"classification="
        f"{temporal_decision.get('AP_evidence_classification')}"
    )
)

lines.append(
    (
        f"mean_delta_AP="
        f"{temporal_decision.get('mean_AP_improvement')}"
    )
)

lines.append(
    (
        f"block_bootstrap_CI="
        f"{temporal_decision.get('block_bootstrap_AP_CI95')}"
    )
)

lines.append("")

lines.append(
    "UNIVERSAL WINNER"
)

lines.append(
    "-" * 105
)

lines.append(
    str(
        universal_winner
    )
)

lines.append("")

lines.append(
    "A universal winner is not declared when "
    "different metrics or calibration diagnostics favor "
    "different models."
)

lines.append("")

lines.append(
    "=" * 105
)

lines.append(
    "END"
)

lines.append(
    "=" * 105
)


(
    OUT
    / "24e_PREDICTIVE_MODEL_AUDIT_REPORT.txt"
).write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8",
)


print()
print("PREDICTIVE REGISTRY")
print("-" * 105)

print(
    registry.to_string(
        index=False
    )
)

print()
print("METRIC WINNERS")
print("-" * 105)

print(
    winners.to_string(
        index=False
    )
)

print()

print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    )
)

print()

print(
    "[PASS 24] FINAL PREDICTIVE MODEL AUDIT COMPLETE."
)

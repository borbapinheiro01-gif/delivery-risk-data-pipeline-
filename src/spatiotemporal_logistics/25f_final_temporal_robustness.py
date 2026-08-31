#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25F — FINAL TEMPORAL ROBUSTNESS

Compara GN_EQ0 vs LOGIT_MLE nos resultados OOT existentes.

Não:
- retreina;
- altera RAW;
- recalcula features;
- cria matriz N x N.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

INPUT = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
    / "scientific"
    / "gnadmm_delivery"
    / "20a_constraint_ablation_oot_metrics.csv"
)

OUT = ROOT / "reports" / "final_closeout"
OUT.mkdir(parents=True, exist_ok=True)

MONTHLY_OUT = OUT / "25f_temporal_GN_EQ0_vs_LOGIT.csv"
SUMMARY_OUT = OUT / "25g_temporal_robustness_summary.csv"
DECISION_OUT = OUT / "25h_TEMPORAL_ROBUSTNESS_DECISION.json"


print("=" * 100)
print("25F — FINAL TEMPORAL ROBUSTNESS")
print("=" * 100)
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print()


if not INPUT.exists():
    raise FileNotFoundError(INPUT)


df = pd.read_csv(INPUT)


def norm(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


norm_map = {
    norm(c): c
    for c in df.columns
}


def find_col(names, required=True):
    for name in names:
        key = norm(name)

        if key in norm_map:
            return norm_map[key]

    if required:
        raise RuntimeError(
            f"Coluna ausente. Candidatas={names}; "
            f"disponíveis={list(df.columns)}"
        )

    return None


model_col = find_col([
    "model",
    "modelo",
])

month_col = find_col([
    "test_month",
    "year_month",
    "month",
    "mes",
])


metric_candidates = {
    "AP": [
        "AP",
        "average_precision",
        "ap_oot",
    ],

    "Brier": [
        "Brier",
        "brier_score",
        "brier_oot",
    ],

    "LogLoss": [
        "LogLoss",
        "log_loss",
        "logloss",
        "log_loss_oot",
    ],

    "AUC": [
        "AUC",
        "roc_auc",
        "auc_oot",
    ],
}


metrics = {}

for name, candidates in metric_candidates.items():
    col = find_col(
        candidates,
        required=False,
    )

    if col is not None:
        metrics[name] = col


if "AP" not in metrics:
    raise RuntimeError(
        "AP precisa existir no arquivo 20a."
    )


direction = {
    "AP": "HIGHER",
    "Brier": "LOWER",
    "LogLoss": "LOWER",
    "AUC": "HIGHER",
}


df = df.copy()

df[model_col] = (
    df[model_col]
    .astype(str)
    .str.strip()
)


available_models = sorted(
    df[model_col]
    .unique()
    .tolist()
)


print("INPUT =", INPUT.relative_to(ROOT))
print("MODELS =", available_models)
print("METRICS =", list(metrics))
print()


for model in [
    "GN_EQ0",
    "LOGIT_MLE",
]:
    if model not in available_models:
        raise RuntimeError(
            f"Modelo ausente: {model}"
        )


summary_rows = []
monthly_rows = []


for metric, metric_col in metrics.items():

    gn = (
        df[
            df[model_col].eq("GN_EQ0")
        ][[
            month_col,
            metric_col,
        ]]
        .rename(
            columns={
                metric_col: "GN_EQ0"
            }
        )
    )


    logit = (
        df[
            df[model_col].eq("LOGIT_MLE")
        ][[
            month_col,
            metric_col,
        ]]
        .rename(
            columns={
                metric_col: "LOGIT_MLE"
            }
        )
    )


    paired = gn.merge(
        logit,
        on=month_col,
        how="inner",
        validate="one_to_one",
    )


    paired = (
        paired
        .dropna()
        .sort_values(month_col)
        .reset_index(drop=True)
    )


    if direction[metric] == "HIGHER":

        paired["delta_oriented"] = (
            paired["GN_EQ0"]
            -
            paired["LOGIT_MLE"]
        )

    else:

        paired["delta_oriented"] = (
            paired["LOGIT_MLE"]
            -
            paired["GN_EQ0"]
        )


    d = paired[
        "delta_oriented"
    ].to_numpy(dtype=float)


    n = len(d)

    if n == 0:
        continue


    wins = int(
        np.sum(d > 0)
    )

    losses = int(
        np.sum(d < 0)
    )

    ties = int(
        n - wins - losses
    )


    nz = d[
        ~np.isclose(
            d,
            0.0,
            atol=1e-15,
        )
    ]


    if len(nz) >= 2:
        test = wilcoxon(
            nz,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        wilcoxon_p = float(
            test.pvalue
        )

    else:
        wilcoxon_p = np.nan


    n_sign = wins + losses

    if n_sign:
        sign_p = float(
            binomtest(
                wins,
                n=n_sign,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    else:
        sign_p = np.nan


    # Leave-one-month-out:
    # o sinal da média continua positivo
    # mesmo removendo qualquer mês?
    lomo = []

    if n > 1:

        for i in range(n):
            lomo.append(
                float(
                    np.delete(
                        d,
                        i,
                    ).mean()
                )
            )

        lomo_min = float(
            min(lomo)
        )

        lomo_max = float(
            max(lomo)
        )

    else:

        lomo_min = np.nan
        lomo_max = np.nan


    mean_delta = float(
        np.mean(d)
    )

    median_delta = float(
        np.median(d)
    )


    robust_sign = (
        np.isfinite(lomo_min)
        and
        lomo_min > 0
    )


    if (
        mean_delta > 0
        and
        wins > losses
        and
        robust_sign
        and
        np.isfinite(wilcoxon_p)
        and
        wilcoxon_p < 0.05
    ):

        classification = (
            "TEMPORAL_ADVANTAGE_SUPPORTED"
        )

    elif (
        mean_delta > 0
        and
        wins > losses
        and
        robust_sign
    ):

        classification = (
            "FAVORABLE_AND_LOMO_ROBUST_NOT_DECISIVE"
        )

    elif (
        mean_delta > 0
        and
        wins > losses
    ):

        classification = (
            "FAVORABLE_NOT_DECISIVE"
        )

    elif mean_delta > 0:

        classification = (
            "BETTER_MEAN_BUT_TEMPORALLY_MIXED"
        )

    else:

        classification = (
            "NO_GN_EQ0_ADVANTAGE"
        )


    summary_rows.append({
        "metric":
            metric,

        "direction":
            direction[metric],

        "n_months":
            n,

        "mean_oriented_delta":
            mean_delta,

        "median_oriented_delta":
            median_delta,

        "months_GN_EQ0_better":
            wins,

        "months_LOGIT_better":
            losses,

        "ties":
            ties,

        "wilcoxon_p":
            wilcoxon_p,

        "sign_test_p":
            sign_p,

        "LOMO_mean_min":
            lomo_min,

        "LOMO_mean_max":
            lomo_max,

        "LOMO_all_positive":
            robust_sign,

        "classification":
            classification,
    })


    for _, row in paired.iterrows():

        monthly_rows.append({
            "metric":
                metric,

            "test_month":
                row[month_col],

            "GN_EQ0":
                float(row["GN_EQ0"]),

            "LOGIT_MLE":
                float(row["LOGIT_MLE"]),

            "delta_oriented":
                float(
                    row["delta_oriented"]
                ),
        })


summary = pd.DataFrame(
    summary_rows
)

monthly = pd.DataFrame(
    monthly_rows
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)

monthly.to_csv(
    MONTHLY_OUT,
    index=False,
)


ap = (
    summary[
        summary["metric"].eq("AP")
    ]
    .iloc[0]
)


decision = {
    "status":
        "PASS",

    "model_refit":
        False,

    "raw_modified":
        False,

    "comparison":
        "GN_EQ0_vs_LOGIT_MLE",

    "primary_metric":
        "AP",

    "n_months":
        int(ap["n_months"]),

    "mean_AP_delta":
        float(
            ap["mean_oriented_delta"]
        ),

    "median_AP_delta":
        float(
            ap["median_oriented_delta"]
        ),

    "months_GN_EQ0_better":
        int(
            ap["months_GN_EQ0_better"]
        ),

    "months_LOGIT_better":
        int(
            ap["months_LOGIT_better"]
        ),

    "wilcoxon_p":
        (
            None
            if pd.isna(
                ap["wilcoxon_p"]
            )
            else
            float(
                ap["wilcoxon_p"]
            )
        ),

    "sign_test_p":
        (
            None
            if pd.isna(
                ap["sign_test_p"]
            )
            else
            float(
                ap["sign_test_p"]
            )
        ),

    "LOMO_mean_min":
        float(
            ap["LOMO_mean_min"]
        ),

    "LOMO_mean_max":
        float(
            ap["LOMO_mean_max"]
        ),

    "classification":
        str(
            ap["classification"]
        ),
}


DECISION_OUT.write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("=" * 100)
print("RESULTS")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("AP DECISION")
print("=" * 100)

print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("[PASS 25F] TEMPORAL ROBUSTNESS COMPLETE")

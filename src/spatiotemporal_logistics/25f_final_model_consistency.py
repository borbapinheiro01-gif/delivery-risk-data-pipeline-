#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25F — FINAL MODEL CONSISTENCY CHECK

Objetivo:
Comparar GN_EQ0 vs LOGIT_MLE nos meses OOT já calculados.

Não:
- retreina modelo;
- altera RAW;
- recalcula features;
- cria matriz N x N.

Responde apenas:
O GN_EQ0, que teve melhores médias em AP/Brier/LogLoss,
também apresenta comportamento temporal consistente?
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

MONTHLY_OUT = OUT / "25f_GN_EQ0_vs_LOGIT_monthly.csv"
SUMMARY_OUT = OUT / "25g_GN_EQ0_vs_LOGIT_summary.csv"
DECISION_OUT = OUT / "25h_FINAL_MODEL_DECISION.json"


print("=" * 100)
print("25F — FINAL MODEL CONSISTENCY CHECK")
print("=" * 100)
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print()


if not INPUT.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT}"
    )


df = pd.read_csv(INPUT)

print("INPUT:")
print(INPUT.relative_to(ROOT))
print()
print("COLUMNS:")
print(list(df.columns))
print()


# ======================================================================================
# IDENTIFICAR COLUNAS
# ======================================================================================

def normalize(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


column_map = {
    normalize(c): c
    for c in df.columns
}


def find_col(names):
    for name in names:
        key = normalize(name)

        if key in column_map:
            return column_map[key]

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

ap_col = find_col([
    "AP",
    "average_precision",
    "ap_oot",
])

brier_col = find_col([
    "Brier",
    "brier_score",
    "brier_oot",
])

logloss_col = find_col([
    "LogLoss",
    "log_loss",
    "log_loss_oot",
])

auc_col = find_col([
    "AUC",
    "roc_auc",
    "auc_oot",
])


if model_col is None:
    raise RuntimeError("Coluna de modelo não encontrada.")

if month_col is None:
    raise RuntimeError("Coluna temporal não encontrada.")

if ap_col is None:
    raise RuntimeError("Coluna AP não encontrada.")


metrics = {
    "AP": (ap_col, "HIGHER"),
}

if brier_col is not None:
    metrics["Brier"] = (
        brier_col,
        "LOWER",
    )

if logloss_col is not None:
    metrics["LogLoss"] = (
        logloss_col,
        "LOWER",
    )

if auc_col is not None:
    metrics["AUC"] = (
        auc_col,
        "HIGHER",
    )


df = df.copy()

df[model_col] = (
    df[model_col]
    .astype(str)
    .str.strip()
)


models = sorted(
    df[model_col]
    .unique()
)

print("MODELS:")
print(models)
print()


for required in [
    "GN_EQ0",
    "LOGIT_MLE",
]:
    if required not in models:
        raise RuntimeError(
            f"Modelo ausente: {required}"
        )


# ======================================================================================
# COMPARAÇÃO PAREADA
# ======================================================================================

summary_rows = []
monthly_rows = []


for metric, (
    metric_col,
    direction,
) in metrics.items():

    a = (
        df[
            df[model_col]
            ==
            "GN_EQ0"
        ][[
            month_col,
            metric_col,
        ]]
        .rename(
            columns={
                metric_col:
                    "GN_EQ0"
            }
        )
    )

    b = (
        df[
            df[model_col]
            ==
            "LOGIT_MLE"
        ][[
            month_col,
            metric_col,
        ]]
        .rename(
            columns={
                metric_col:
                    "LOGIT_MLE"
            }
        )
    )


    paired = a.merge(
        b,
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


    if direction == "HIGHER":

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


    delta = paired[
        "delta_oriented"
    ].to_numpy(dtype=float)


    wins = int(
        np.sum(delta > 0)
    )

    losses = int(
        np.sum(delta < 0)
    )

    ties = int(
        np.sum(
            np.isclose(
                delta,
                0.0,
                atol=1e-15,
            )
        )
    )


    nonzero = delta[
        ~np.isclose(
            delta,
            0.0,
            atol=1e-15,
        )
    ]


    if len(nonzero) >= 2:

        w = wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        wilcoxon_p = float(
            w.pvalue
        )

    else:

        wilcoxon_p = np.nan


    if (
        wins
        +
        losses
        >
        0
    ):

        sign_p = float(
            binomtest(
                wins,
                n=wins + losses,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    else:

        sign_p = np.nan


    mean_delta = float(
        np.mean(delta)
    )

    median_delta = float(
        np.median(delta)
    )


    # Leave-one-month-out:
    # verifica se um único mês está puxando o resultado.
    lomo = []

    if len(delta) > 1:

        for i in range(
            len(delta)
        ):

            lomo.append(
                float(
                    np.delete(
                        delta,
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


    if (
        mean_delta > 0
        and
        wins > losses
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
    ):

        classification = (
            "FAVORABLE_NOT_DECISIVE"
        )

    elif mean_delta > 0:

        classification = (
            "BETTER_MEAN_BUT_NOT_TEMPORALLY_CONSISTENT"
        )

    else:

        classification = (
            "NO_ADVANTAGE"
        )


    summary_rows.append({
        "metric":
            metric,

        "n_months":
            len(delta),

        "mean_oriented_delta":
            mean_delta,

        "median_oriented_delta":
            median_delta,

        "months_GN_EQ0_better":
            wins,

        "months_LOGIT_MLE_better":
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
                float(
                    row["GN_EQ0"]
                ),

            "LOGIT_MLE":
                float(
                    row["LOGIT_MLE"]
                ),

            "delta_oriented":
                float(
                    row[
                        "delta_oriented"
                    ]
                ),
        })


# ======================================================================================
# SALVAR
# ======================================================================================

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


ap_result = summary[
    summary["metric"]
    ==
    "AP"
].iloc[0]


decision = {
    "status":
        "PASS",

    "purpose":
        "FINAL_MODEL_SELECTION_CHECK",

    "model_refit":
        False,

    "raw_modified":
        False,

    "primary_comparison":
        "GN_EQ0_vs_LOGIT_MLE",

    "primary_metric":
        "AP",

    "n_months":
        int(
            ap_result[
                "n_months"
            ]
        ),

    "mean_AP_delta_GN_EQ0_minus_LOGIT":
        float(
            ap_result[
                "mean_oriented_delta"
            ]
        ),

    "median_AP_delta":
        float(
            ap_result[
                "median_oriented_delta"
            ]
        ),

    "months_GN_EQ0_better":
        int(
            ap_result[
                "months_GN_EQ0_better"
            ]
        ),

    "months_LOGIT_MLE_better":
        int(
            ap_result[
                "months_LOGIT_MLE_better"
            ]
        ),

    "wilcoxon_p":
        (
            None
            if pd.isna(
                ap_result[
                    "wilcoxon_p"
                ]
            )
            else
            float(
                ap_result[
                    "wilcoxon_p"
                ]
            )
        ),

    "LOMO_mean_min":
        float(
            ap_result[
                "LOMO_mean_min"
            ]
        ),

    "LOMO_mean_max":
        float(
            ap_result[
                "LOMO_mean_max"
            ]
        ),

    "classification":
        ap_result[
            "classification"
        ],
}


# Regra prática para fechamento do case:
#
# SUPPORTED:
# GN_EQ0 pode ser apresentado como melhor escolha
# para AP temporal entre os dois.
#
# FAVORABLE_NOT_DECISIVE:
# GN_EQ0 é melhor em média, mas não há evidência
# suficiente para superioridade universal.
#
# BETTER_MEAN...:
# média melhor, mas inconsistência mensal.
#
# NO_ADVANTAGE:
# Logit permanece benchmark preferível.

if (
    decision["classification"]
    ==
    "TEMPORAL_ADVANTAGE_SUPPORTED"
):

    decision[
        "final_modeling_interpretation"
    ] = (
        "GN_EQ0 shows a supported temporal AP "
        "advantage over LOGIT_MLE in the evaluated OOT months."
    )

elif (
    decision["classification"]
    ==
    "FAVORABLE_NOT_DECISIVE"
):

    decision[
        "final_modeling_interpretation"
    ] = (
        "GN_EQ0 has favorable average and monthly AP behavior, "
        "but temporal evidence is not decisive enough to claim "
        "universal superiority over LOGIT_MLE."
    )

elif (
    decision["classification"]
    ==
    "BETTER_MEAN_BUT_NOT_TEMPORALLY_CONSISTENT"
):

    decision[
        "final_modeling_interpretation"
    ] = (
        "GN_EQ0 has a better mean AP, but the advantage is not "
        "temporally consistent across the evaluated months."
    )

else:

    decision[
        "final_modeling_interpretation"
    ] = (
        "GN_EQ0 does not establish an AP advantage over LOGIT_MLE."
    )


DECISION_OUT.write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ======================================================================================
# TERMINAL
# ======================================================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("FINAL MODEL DECISION")
print("=" * 100)

print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("[PASS 25F] FINAL MODEL CONSISTENCY CHECK COMPLETE")
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25I — PREQUENTIAL PROBABILITY BLEND
FINAL LOW-COST MODEL OPTIMIZATION

Objetivo
--------
Testar um único blend probabilístico:

    p_blend = alpha * p_GN_EQ0 + (1-alpha) * p_LOGIT_MLE

alpha in {0, .25, .50, .75, 1}

Regra temporal
--------------
Para avaliar o mês t, alpha é escolhido SOMENTE com observações
dos meses OOT anteriores.

Não:
- retreina GN_EQ0;
- retreina LOGIT_MLE;
- modifica RAW;
- recalcula features;
- cria matriz N x N.

IMPORTANTE
----------
Não há fallback por posição/cumcount.
As previsões precisam poder ser alinhadas de forma segura.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

INPUT = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
    / "scientific"
    / "gnadmm_delivery"
    / "20b_constraint_ablation_oot_predictions.csv"
)

OUT = ROOT / "reports" / "final_closeout"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


PRED_OUT = (
    OUT
    / "25i_prequential_blend_predictions.csv"
)

MONTHLY_OUT = (
    OUT
    / "25j_prequential_blend_monthly.csv"
)

SUMMARY_OUT = (
    OUT
    / "25k_prequential_blend_summary.csv"
)

DECISION_OUT = (
    OUT
    / "25l_FINAL_BLEND_DECISION.json"
)


ALPHAS = np.array(
    [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
    ],
    dtype=float,
)


print("=" * 108)
print("25I — PREQUENTIAL PROBABILITY BLEND")
print("=" * 108)
print("BASE MODELS = GN_EQ0 + LOGIT_MLE")
print("ALPHA GRID =", ALPHAS.tolist())
print("BASE_MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("N_X_N_MATRIX_CREATED = false")
print()


# =====================================================================================
# 1. INPUT
# =====================================================================================

if not INPUT.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT}"
    )


raw = pd.read_csv(
    INPUT
)


print(
    "INPUT =",
    INPUT.relative_to(ROOT),
)

print(
    "rows =",
    len(raw),
)

print(
    "columns =",
    list(raw.columns),
)

print()


# =====================================================================================
# 2. COLUMN DISCOVERY
# =====================================================================================

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
    for c in raw.columns
}


def find_col(
    candidates,
    required=False,
):

    for candidate in candidates:

        key = norm(
            candidate
        )

        if key in norm_map:
            return norm_map[key]

    if required:

        raise RuntimeError(
            "Coluna obrigatória não encontrada. "
            f"Candidatas={candidates}; "
            f"disponíveis={list(raw.columns)}"
        )

    return None


month_col = find_col(
    [
        "test_month",
        "year_month",
        "month",
        "mes",
    ],
    required=True,
)


target_col = find_col(
    [
        "y_true",
        "target",
        "y",
        "late_delivery_calendar_day",
        "LATE_DELIVERY_CALENDAR_DAY",
    ],
)


model_col = find_col(
    [
        "model",
        "modelo",
    ],
)


prob_col = find_col(
    [
        "y_prob",
        "prob",
        "probability",
        "pred_prob",
        "prediction",
        "p_hat",
        "predicted_probability",
    ],
)


id_col = find_col(
    [
        "order_id",
        "observation_id",
        "obs_id",
        "row_id",
    ],
)


# =====================================================================================
# 3. PREPARAR FORMATO WIDE
#
# esperado:
#
# test_month | y_true | GN_EQ0 | LOGIT_MLE
#
# O código aceita:
# A) arquivo LONG com model + probability;
# B) arquivo WIDE com probabilidades já separadas.
# =====================================================================================

if (
    model_col is not None
    and
    prob_col is not None
):

    print(
        "FORMAT = LONG"
    )

    if target_col is None:

        raise RuntimeError(
            "Formato LONG detectado, mas target não foi encontrado."
        )


    if id_col is None:

        raise RuntimeError(
            "Formato LONG detectado, porém não existe um identificador "
            "seguro da observação (ex.: order_id). "
            "O experimento foi interrompido para evitar pareamento incorreto."
        )


    df = raw.copy()


    df[model_col] = (
        df[model_col]
        .astype(str)
        .str.strip()
    )


    df = df[
        df[model_col].isin(
            [
                "GN_EQ0",
                "LOGIT_MLE",
            ]
        )
    ].copy()


    duplicate_check = (
        df.duplicated(
            subset=[
                month_col,
                id_col,
                model_col,
            ]
        )
        .sum()
    )


    if duplicate_check != 0:

        raise RuntimeError(
            f"Existem {duplicate_check} duplicatas "
            "month + id + model."
        )


    targets = (
        df[
            [
                month_col,
                id_col,
                target_col,
            ]
        ]
        .drop_duplicates()
    )


    target_conflicts = (
        targets.groupby(
            [
                month_col,
                id_col,
            ]
        )[target_col]
        .nunique()
        .max()
    )


    if target_conflicts > 1:

        raise RuntimeError(
            "Target inconsistente entre modelos para a mesma observação."
        )


    targets = (
        targets
        .drop_duplicates(
            subset=[
                month_col,
                id_col,
            ]
        )
    )


    probs = (
        df.pivot(
            index=[
                month_col,
                id_col,
            ],
            columns=model_col,
            values=prob_col,
        )
        .reset_index()
    )


    wide = targets.merge(
        probs,
        on=[
            month_col,
            id_col,
        ],
        how="inner",
        validate="one_to_one",
    )


    wide = wide.rename(
        columns={
            target_col:
                "y_true"
        }
    )


else:

    print(
        "FORMAT = WIDE"
    )


    gn_col = find_col(
        [
            "GN_EQ0",
            "pred_GN_EQ0",
            "p_GN_EQ0",
            "prob_GN_EQ0",
            "GN_EQ0_prob",
        ],
    )


    logit_col = find_col(
        [
            "LOGIT_MLE",
            "pred_LOGIT_MLE",
            "p_LOGIT_MLE",
            "prob_LOGIT_MLE",
            "LOGIT_MLE_prob",
        ],
    )


    if (
        gn_col is None
        or
        logit_col is None
        or
        target_col is None
    ):

        raise RuntimeError(
            "Não foi possível identificar com segurança o formato "
            "de 20b_constraint_ablation_oot_predictions.csv."
        )


    wide = raw.copy()


    wide = wide.rename(
        columns={
            target_col:
                "y_true",

            gn_col:
                "GN_EQ0",

            logit_col:
                "LOGIT_MLE",
        }
    )


# =====================================================================================
# 4. VALIDATION
# =====================================================================================

needed = [
    month_col,
    "y_true",
    "GN_EQ0",
    "LOGIT_MLE",
]


for c in needed:

    if c not in wide.columns:

        raise RuntimeError(
            f"Coluna final ausente: {c}"
        )


wide = (
    wide[
        needed
    ]
    .dropna()
    .copy()
)


wide[month_col] = (
    wide[month_col]
    .astype(str)
)


wide["y_true"] = (
    wide["y_true"]
    .astype(int)
)


unique_y = sorted(
    wide["y_true"]
    .unique()
    .tolist()
)


if not set(
    unique_y
).issubset(
    {
        0,
        1,
    }
):

    raise RuntimeError(
        f"Target não-binário: {unique_y}"
    )


for c in [
    "GN_EQ0",
    "LOGIT_MLE",
]:

    wide[c] = (
        wide[c]
        .astype(float)
    )


    bad_prob = (
        (~np.isfinite(wide[c]))
        |
        (wide[c] < 0)
        |
        (wide[c] > 1)
    )


    if bad_prob.any():

        raise RuntimeError(
            f"Probabilidades inválidas em {c}: "
            f"{int(bad_prob.sum())}"
        )


    wide[c] = np.clip(
        wide[c],
        1e-12,
        1 - 1e-12,
    )


months = sorted(
    wide[month_col]
    .unique()
    .tolist()
)


if len(months) < 3:

    raise RuntimeError(
        f"Apenas {len(months)} meses disponíveis."
    )


counts = (
    wide.groupby(
        month_col
    )
    .size()
)


print(
    "aligned observations =",
    len(wide),
)

print(
    "months =",
    len(months),
)

print(
    "first month =",
    months[0],
)

print(
    "last month =",
    months[-1],
)

print(
    "month N min/max =",
    int(counts.min()),
    int(counts.max()),
)

print()


# =====================================================================================
# 5. METRICS
# =====================================================================================

def safe_auc(
    y,
    p,
):

    if np.unique(
        y
    ).size < 2:

        return np.nan

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def evaluate(
    y,
    p,
):

    return {
        "AP":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),

        "LogLoss":
            float(
                log_loss(
                    y,
                    p,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "AUC":
            safe_auc(
                y,
                p,
            ),
    }


def ece_quantile(
    y,
    p,
    n_bins=10,
):

    temp = pd.DataFrame(
        {
            "y":
                y,

            "p":
                p,
        }
    )


    try:

        temp["bin"] = pd.qcut(
            temp["p"],
            q=n_bins,
            duplicates="drop",
        )

    except Exception:

        return np.nan


    total = len(
        temp
    )


    ece = 0.0


    for _, g in temp.groupby(
        "bin",
        observed=True,
    ):

        if len(g) == 0:
            continue


        ece += (
            len(g)
            /
            total
        ) * abs(
            g["y"].mean()
            -
            g["p"].mean()
        )


    return float(
        ece
    )


# =====================================================================================
# 6. PREQUENTIAL SELECTION
#
# Para o mês t:
#
# alpha_t é escolhido pelo menor Brier agregado
# usando somente meses anteriores.
#
# Primeiro mês = warm-up.
# =====================================================================================

prediction_parts = []
monthly_rows = []
alpha_rows = []


for i in range(
    1,
    len(months),
):

    current_month = months[
        i
    ]


    past_months = months[
        :i
    ]


    history = wide[
        wide[month_col].isin(
            past_months
        )
    ]


    test = wide[
        wide[month_col].eq(
            current_month
        )
    ].copy()


    y_hist = (
        history[
            "y_true"
        ]
        .to_numpy()
    )


    gn_hist = (
        history[
            "GN_EQ0"
        ]
        .to_numpy()
    )


    logit_hist = (
        history[
            "LOGIT_MLE"
        ]
        .to_numpy()
    )


    candidates = []


    for alpha in ALPHAS:

        p = (
            alpha
            *
            gn_hist
            +
            (
                1.0
                -
                alpha
            )
            *
            logit_hist
        )


        score = float(
            brier_score_loss(
                y_hist,
                p,
            )
        )


        candidates.append(
            {
                "alpha":
                    float(alpha),

                "past_brier":
                    score,
            }
        )


    candidate_df = pd.DataFrame(
        candidates
    )


    best_brier = (
        candidate_df[
            "past_brier"
        ]
        .min()
    )


    tied = candidate_df[
        np.isclose(
            candidate_df[
                "past_brier"
            ],
            best_brier,
            rtol=0,
            atol=1e-15,
        )
    ].copy()


    # empate:
    # solução mais conservadora,
    # alpha mais próximo de 0.5.
    tied[
        "distance_half"
    ] = abs(
        tied[
            "alpha"
        ]
        -
        0.5
    )


    selected_alpha = float(
        tied.sort_values(
            [
                "distance_half",
                "alpha",
            ]
        )
        .iloc[0][
            "alpha"
        ]
    )


    test["ENSEMBLE"] = (
        selected_alpha
        *
        test["GN_EQ0"]
        +
        (
            1.0
            -
            selected_alpha
        )
        *
        test["LOGIT_MLE"]
    )


    test[
        "selected_alpha"
    ] = selected_alpha


    prediction_parts.append(
        test
    )


    alpha_rows.append(
        {
            "test_month":
                current_month,

            "n_prior_months":
                i,

            "alpha_GN_EQ0":
                selected_alpha,

            "alpha_LOGIT_MLE":
                1.0
                -
                selected_alpha,

            "historical_selection_Brier":
                float(
                    best_brier
                ),
        }
    )


    y_test = (
        test[
            "y_true"
        ]
        .to_numpy()
    )


    for model in [
        "GN_EQ0",
        "LOGIT_MLE",
        "ENSEMBLE",
    ]:

        p_test = (
            test[
                model
            ]
            .to_numpy()
        )


        result = evaluate(
            y_test,
            p_test,
        )


        monthly_rows.append(
            {
                "test_month":
                    current_month,

                "model":
                    model,

                "selected_alpha":
                    (
                        selected_alpha
                        if model
                        ==
                        "ENSEMBLE"
                        else
                        np.nan
                    ),

                **result,
            }
        )


pred = pd.concat(
    prediction_parts,
    ignore_index=True,
)


monthly = pd.DataFrame(
    monthly_rows
)


alpha_history = pd.DataFrame(
    alpha_rows
)


# =====================================================================================
# 7. SAME-PERIOD FINAL EVALUATION
#
# Todos os modelos avaliados exatamente nos mesmos meses:
# meses 2...14.
# =====================================================================================

y_all = (
    pred[
        "y_true"
    ]
    .to_numpy()
)


summary_rows = []


for model in [
    "GN_EQ0",
    "LOGIT_MLE",
    "ENSEMBLE",
]:

    p = (
        pred[
            model
        ]
        .to_numpy()
    )


    pooled = evaluate(
        y_all,
        p,
    )


    ece = ece_quantile(
        y_all,
        p,
        n_bins=10,
    )


    m = monthly[
        monthly[
            "model"
        ].eq(
            model
        )
    ]


    summary_rows.append(
        {
            "model":
                model,

            "n_eval_months":
                len(months)
                -
                1,

            "n_predictions":
                len(pred),

            "pooled_AP":
                pooled["AP"],

            "pooled_Brier":
                pooled["Brier"],

            "pooled_LogLoss":
                pooled["LogLoss"],

            "pooled_AUC":
                pooled["AUC"],

            "pooled_ECE10_quantile":
                ece,

            "mean_monthly_AP":
                float(
                    m["AP"].mean()
                ),

            "mean_monthly_Brier":
                float(
                    m["Brier"].mean()
                ),

            "mean_monthly_LogLoss":
                float(
                    m["LogLoss"].mean()
                ),

            "mean_monthly_AUC":
                float(
                    m["AUC"].mean()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# =====================================================================================
# 8. MONTHLY ROBUSTNESS OF ENSEMBLE
# =====================================================================================

pivot_ap = (
    monthly.pivot(
        index="test_month",
        columns="model",
        values="AP",
    )
)


ens_vs_gn = (
    pivot_ap[
        "ENSEMBLE"
    ]
    -
    pivot_ap[
        "GN_EQ0"
    ]
)


ens_vs_logit = (
    pivot_ap[
        "ENSEMBLE"
    ]
    -
    pivot_ap[
        "LOGIT_MLE"
    ]
)


ens_months_better_gn = int(
    (
        ens_vs_gn
        >
        0
    ).sum()
)


ens_months_better_logit = int(
    (
        ens_vs_logit
        >
        0
    ).sum()
)


# =====================================================================================
# 9. DECISION
#
# AP = primary metric.
#
# O ensemble só vira recomendação se:
#
# 1. pooled AP >= melhor base;
# 2. pooled Brier <= piora máxima minúscula de 0.0001
#    em relação ao melhor base;
# 3. pooled LogLoss <= piora máxima 0.001
#    em relação ao melhor base.
#
# Caso contrário:
# modelo de maior AP entre GN_EQ0 e LOGIT.
#
# Essas tolerâncias NÃO representam significância.
# São apenas guardrails operacionais anti-degradação.
# =====================================================================================

ens = (
    summary[
        summary[
            "model"
        ].eq(
            "ENSEMBLE"
        )
    ]
    .iloc[0]
)


base = summary[
    summary[
        "model"
    ].isin(
        [
            "GN_EQ0",
            "LOGIT_MLE",
        ]
    )
]


best_base_ap = float(
    base[
        "pooled_AP"
    ].max()
)


best_base_brier = float(
    base[
        "pooled_Brier"
    ].min()
)


best_base_logloss = float(
    base[
        "pooled_LogLoss"
    ].min()
)


ensemble_ap_gain = float(
    ens[
        "pooled_AP"
    ]
    -
    best_base_ap
)


ensemble_brier_gap = float(
    ens[
        "pooled_Brier"
    ]
    -
    best_base_brier
)


ensemble_logloss_gap = float(
    ens[
        "pooled_LogLoss"
    ]
    -
    best_base_logloss
)


ensemble_guardrails_pass = bool(
    ensemble_ap_gain
    >=
    0
    and
    ensemble_brier_gap
    <=
    0.0001
    and
    ensemble_logloss_gap
    <=
    0.001
)


if ensemble_guardrails_pass:

    recommended_model = (
        "ENSEMBLE"
    )

    classification = (
        "ENSEMBLE_ACCEPTED"
    )


else:

    base_winner = (
        base.sort_values(
            [
                "pooled_AP",
                "pooled_Brier",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )


    recommended_model = str(
        base_winner[
            "model"
        ]
    )


    classification = (
        "ENSEMBLE_REJECTED_KEEP_BASE_MODEL"
    )


decision = {
    "status":
        "PASS",

    "experiment":
        "PREQUENTIAL_PROBABILITY_BLEND",

    "base_model_refit":
        False,

    "raw_modified":
        False,

    "n_x_n_matrix_created":
        False,

    "alpha_grid":
        ALPHAS.tolist(),

    "alpha_selection":
        (
            "minimum historical Brier using prior OOT months only"
        ),

    "warmup_month":
        months[0],

    "n_evaluation_months":
        len(months)
        -
        1,

    "primary_metric":
        "AP",

    "ensemble_AP_gain_vs_best_base":
        ensemble_ap_gain,

    "ensemble_Brier_gap_vs_best_base":
        ensemble_brier_gap,

    "ensemble_LogLoss_gap_vs_best_base":
        ensemble_logloss_gap,

    "ensemble_months_AP_better_than_GN_EQ0":
        ens_months_better_gn,

    "ensemble_months_AP_better_than_LOGIT":
        ens_months_better_logit,

    "guardrails_pass":
        ensemble_guardrails_pass,

    "classification":
        classification,

    "recommended_model":
        recommended_model,

    "scientific_guardrail":
        (
            "No universal superiority claim. "
            "Blend is retained only if it improves primary OOT AP "
            "without material degradation of probability scores."
        ),
}


# =====================================================================================
# 10. SAVE
# =====================================================================================

pred.to_csv(
    PRED_OUT,
    index=False,
)


monthly.to_csv(
    MONTHLY_OUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


alpha_history.to_csv(
    OUT
    / "25m_prequential_alpha_history.csv",
    index=False,
)


DECISION_OUT.write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =====================================================================================
# 11. REPORT
# =====================================================================================

print()
print("=" * 108)
print("SAME-PERIOD FINAL COMPARISON")
print("=" * 108)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("ALPHA HISTORY")
print("=" * 108)

print(
    alpha_history.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("FINAL BLEND DECISION")
print("=" * 108)

print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    )
)


print()
print("[PASS 25I] PREQUENTIAL BLEND COMPLETE")
print("BASE_MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("N_X_N_MATRIX_CREATED = false")

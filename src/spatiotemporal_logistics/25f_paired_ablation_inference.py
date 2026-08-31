#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODULE 25F — PAIRED TEMPORAL INFERENCE FOR CONSTRAINT ABLATION

Objetivo
--------
Reutilizar os resultados OOT já calculados no módulo 20 e testar:

1) GN_EQ0 vs LOGIT_MLE
2) GN_EQ0 vs GNADMM_GE0

Não retreina nenhum modelo.

Para cada métrica:
- delta mensal orientado de forma que positivo = GN_EQ0 melhor
- média
- mediana
- número de meses ganhos
- Wilcoxon pareado
- teste exato de sign-flip/permutação
- teste binomial de sinais
- leave-one-month-out
- moving-block bootstrap com blocos 2 e 3

Métricas:
- AP: maior é melhor
- Brier: menor é melhor
- LogLoss: menor é melhor
- AUC: maior é melhor
"""

from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

GN = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
    / "scientific"
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

INPUT = (
    GN
    / "20a_constraint_ablation_oot_metrics.csv"
)

SUMMARY_OUT = (
    OUT
    / "25f_paired_ablation_inference.csv"
)

DELTAS_OUT = (
    OUT
    / "25g_paired_ablation_monthly_deltas.csv"
)

DECISION_OUT = (
    OUT
    / "25h_PAIRED_ABLATION_DECISION.json"
)


print("=" * 108)
print("MODULE 25F — PAIRED TEMPORAL ABLATION INFERENCE")
print("=" * 108)
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print()


# ==============================================================================================
# 1. LOAD
# ==============================================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Arquivo ausente: {INPUT}"
    )

df = pd.read_csv(
    INPUT
)

print(
    f"INPUT = {INPUT.relative_to(ROOT)}"
)

print(
    f"rows = {len(df)}"
)

print(
    f"columns = {list(df.columns)}"
)


# ==============================================================================================
# 2. COLUMN DISCOVERY
# ==============================================================================================

def norm(s):
    return (
        str(s)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


norm_map = {
    norm(c): c
    for c in df.columns
}


def find_col(candidates, required=True):
    for name in candidates:
        key = norm(name)

        if key in norm_map:
            return norm_map[key]

    if required:
        raise KeyError(
            "Nenhuma coluna encontrada entre: "
            + ", ".join(candidates)
            + f"\nDisponíveis: {list(df.columns)}"
        )

    return None


model_col = find_col(
    [
        "model",
        "modelo",
    ]
)

month_col = find_col(
    [
        "test_month",
        "year_month",
        "month",
        "mes",
    ]
)


metric_candidates = {
    "AP": [
        "AP",
        "ap",
        "ap_oot",
        "average_precision",
    ],

    "Brier": [
        "Brier",
        "brier",
        "brier_oot",
        "brier_score",
    ],

    "LogLoss": [
        "LogLoss",
        "logloss",
        "log_loss",
        "log_loss_oot",
    ],

    "AUC": [
        "AUC",
        "auc",
        "roc_auc",
        "auc_oot",
    ],
}


metric_cols = {}

for metric, candidates in metric_candidates.items():

    col = find_col(
        candidates,
        required=False
    )

    if col is not None:
        metric_cols[
            metric
        ] = col


if "AP" not in metric_cols:
    raise RuntimeError(
        "AP não foi encontrada no arquivo 20a."
    )


print()
print(
    "MODEL COLUMN =",
    model_col
)

print(
    "MONTH COLUMN =",
    month_col
)

print(
    "METRICS =",
    metric_cols
)


# ==============================================================================================
# 3. NORMALIZE MODEL NAMES
# ==============================================================================================

df = df.copy()

df[
    model_col
] = (
    df[
        model_col
    ]
    .astype(str)
    .str.strip()
)

available_models = sorted(
    df[
        model_col
    ]
    .unique()
    .tolist()
)

print()
print(
    "MODELS =",
    available_models
)


required_models = [
    "GN_EQ0",
    "LOGIT_MLE",
    "GNADMM_GE0",
]


missing_models = [
    model
    for model in required_models
    if model not in available_models
]


if missing_models:
    raise RuntimeError(
        f"Modelos ausentes: {missing_models}"
    )


# ==============================================================================================
# 4. PAIR CONSTRUCTION
# ==============================================================================================

comparisons = [
    (
        "GN_EQ0",
        "LOGIT_MLE",
    ),

    (
        "GN_EQ0",
        "GNADMM_GE0",
    ),
]


metric_direction = {
    "AP": "HIGHER",
    "Brier": "LOWER",
    "LogLoss": "LOWER",
    "AUC": "HIGHER",
}


def paired_values(model_a, model_b, metric):

    metric_col = metric_cols[
        metric
    ]

    a = (
        df[
            df[
                model_col
            ].eq(
                model_a
            )
        ][
            [
                month_col,
                metric_col,
            ]
        ]
        .rename(
            columns={
                metric_col:
                    "A"
            }
        )
    )

    b = (
        df[
            df[
                model_col
            ].eq(
                model_b
            )
        ][
            [
                month_col,
                metric_col,
            ]
        ]
        .rename(
            columns={
                metric_col:
                    "B"
            }
        )
    )

    p = a.merge(
        b,
        on=month_col,
        how="inner",
        validate="one_to_one",
    )

    p = p.dropna(
        subset=[
            "A",
            "B",
        ]
    ).copy()

    p = p.sort_values(
        month_col
    ).reset_index(
        drop=True
    )

    if metric_direction[
        metric
    ] == "HIGHER":

        p[
            "delta_oriented"
        ] = (
            p["A"]
            -
            p["B"]
        )

    else:

        p[
            "delta_oriented"
        ] = (
            p["B"]
            -
            p["A"]
        )

    return p


# ==============================================================================================
# 5. EXACT SIGN-FLIP TEST
# ==============================================================================================

def exact_signflip_pvalue(delta):

    delta = np.asarray(
        delta,
        dtype=float,
    )

    delta = delta[
        np.isfinite(
            delta
        )
    ]

    n = len(
        delta
    )

    if n == 0:
        return np.nan

    observed = abs(
        delta.mean()
    )

    # 14 months -> 2^14 = 16384 combinations.
    # Small enough for an exact deterministic test.
    if n <= 20:

        signs = np.array(
            list(
                product(
                    [-1.0, 1.0],
                    repeat=n,
                )
            ),
            dtype=float,
        )

        means = (
            signs
            *
            delta
        ).mean(
            axis=1
        )

        return float(
            (
                np.abs(
                    means
                )
                >=
                observed
                -
                1e-15
            ).mean()
        )

    rng = np.random.default_rng(
        20260831
    )

    count = 0
    reps = 50000

    for _ in range(
        reps
    ):

        signs = rng.choice(
            [-1.0, 1.0],
            size=n,
        )

        stat = abs(
            np.mean(
                delta
                *
                signs
            )
        )

        if stat >= observed:
            count += 1

    return float(
        (
            count
            +
            1
        )
        /
        (
            reps
            +
            1
        )
    )


# ==============================================================================================
# 6. MOVING-BLOCK BOOTSTRAP
# ==============================================================================================

def moving_block_bootstrap(
    delta,
    block_length,
    reps=5000,
    seed=20260831,
):

    x = np.asarray(
        delta,
        dtype=float,
    )

    n = len(
        x
    )

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    if block_length <= 1:
        block_length = 1

    rng = np.random.default_rng(
        seed
        +
        block_length
    )

    # Circular blocks preserve local temporal adjacency.
    starts = np.arange(
        n
    )

    boot_means = np.empty(
        reps,
        dtype=float,
    )

    n_blocks = int(
        np.ceil(
            n
            /
            block_length
        )
    )

    for r in range(
        reps
    ):

        pieces = []

        chosen_starts = rng.choice(
            starts,
            size=n_blocks,
            replace=True,
        )

        for start in chosen_starts:

            idx = (
                start
                +
                np.arange(
                    block_length
                )
            ) % n

            pieces.append(
                x[
                    idx
                ]
            )

        sample = np.concatenate(
            pieces
        )[:n]

        boot_means[
            r
        ] = sample.mean()

    lo, hi = np.quantile(
        boot_means,
        [
            0.025,
            0.975,
        ]
    )

    prob_positive = float(
        np.mean(
            boot_means
            >
            0
        )
    )

    return (
        float(lo),
        float(hi),
        prob_positive,
    )


# ==============================================================================================
# 7. LEAVE-ONE-MONTH-OUT
# ==============================================================================================

def lomo_summary(delta):

    x = np.asarray(
        delta,
        dtype=float,
    )

    n = len(
        x
    )

    if n <= 1:
        return (
            np.nan,
            np.nan,
        )

    means = []

    for i in range(
        n
    ):

        means.append(
            np.delete(
                x,
                i,
            ).mean()
        )

    return (
        float(
            np.min(
                means
            )
        ),
        float(
            np.max(
                means
            )
        ),
    )


# ==============================================================================================
# 8. RUN ALL COMPARISONS
# ==============================================================================================

summary_rows = []
monthly_rows = []


for model_a, model_b in comparisons:

    for metric in metric_cols:

        p = paired_values(
            model_a,
            model_b,
            metric,
        )

        delta = p[
            "delta_oriented"
        ].to_numpy(
            dtype=float
        )

        n = len(
            delta
        )

        if n == 0:
            continue


        mean_delta = float(
            np.mean(
                delta
            )
        )

        median_delta = float(
            np.median(
                delta
            )
        )

        wins = int(
            np.sum(
                delta
                >
                0
            )
        )

        losses = int(
            np.sum(
                delta
                <
                0
            )
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


        # Wilcoxon
        nonzero = delta[
            ~np.isclose(
                delta,
                0.0,
                atol=1e-15,
            )
        ]

        if len(
            nonzero
        ) >= 2:

            try:

                wx = wilcoxon(
                    nonzero,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )

                wilcoxon_stat = float(
                    wx.statistic
                )

                wilcoxon_p = float(
                    wx.pvalue
                )

            except Exception:

                wilcoxon_stat = np.nan
                wilcoxon_p = np.nan

        else:

            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan


        # Exact sign test
        n_sign = wins + losses

        if n_sign > 0:

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


        signflip_p = exact_signflip_pvalue(
            delta
        )


        block2_lo, block2_hi, block2_prob = (
            moving_block_bootstrap(
                delta,
                block_length=2,
            )
        )

        block3_lo, block3_hi, block3_prob = (
            moving_block_bootstrap(
                delta,
                block_length=3,
            )
        )


        lomo_min, lomo_max = lomo_summary(
            delta
        )


        # Data-driven evidence label.
        # Positive oriented delta always means model A is better.
        ci2_positive = (
            np.isfinite(
                block2_lo
            )
            and
            block2_lo
            >
            0
        )

        ci3_positive = (
            np.isfinite(
                block3_lo
            )
            and
            block3_lo
            >
            0
        )

        classical_support = (
            (
                np.isfinite(
                    wilcoxon_p
                )
                and
                wilcoxon_p
                <
                0.05
            )
            or
            (
                np.isfinite(
                    signflip_p
                )
                and
                signflip_p
                <
                0.05
            )
        )


        if (
            mean_delta
            >
            0
            and
            ci2_positive
            and
            ci3_positive
            and
            classical_support
        ):

            evidence = (
                "TEMPORAL_ADVANTAGE_SUPPORTED"
            )

        elif (
            mean_delta
            >
            0
            and
            wins
            >
            losses
        ):

            evidence = (
                "FAVORABLE_NOT_DECISIVE"
            )

        elif (
            mean_delta
            >
            0
        ):

            evidence = (
                "POSITIVE_MEAN_MIXED_MONTHLY"
            )

        elif (
            mean_delta
            <
            0
        ):

            evidence = (
                "NO_FAVORABLE_SIGNAL"
            )

        else:

            evidence = (
                "NEUTRAL"
            )


        summary_rows.append(
            {
                "model_A":
                    model_a,

                "model_B":
                    model_b,

                "metric":
                    metric,

                "direction":
                    metric_direction[
                        metric
                    ],

                "n_months":
                    n,

                "mean_oriented_delta":
                    mean_delta,

                "median_oriented_delta":
                    median_delta,

                "months_A_better":
                    wins,

                "months_B_better":
                    losses,

                "ties":
                    ties,

                "wilcoxon_stat":
                    wilcoxon_stat,

                "wilcoxon_p":
                    wilcoxon_p,

                "exact_sign_test_p":
                    sign_p,

                "exact_signflip_p":
                    signflip_p,

                "block2_CI95_low":
                    block2_lo,

                "block2_CI95_high":
                    block2_hi,

                "block2_prob_delta_positive":
                    block2_prob,

                "block3_CI95_low":
                    block3_lo,

                "block3_CI95_high":
                    block3_hi,

                "block3_prob_delta_positive":
                    block3_prob,

                "LOMO_mean_min":
                    lomo_min,

                "LOMO_mean_max":
                    lomo_max,

                "evidence_classification":
                    evidence,
            }
        )


        for _, row in p.iterrows():

            monthly_rows.append(
                {
                    "model_A":
                        model_a,

                    "model_B":
                        model_b,

                    "metric":
                        metric,

                    "test_month":
                        row[
                            month_col
                        ],

                    "value_A":
                        float(
                            row[
                                "A"
                            ]
                        ),

                    "value_B":
                        float(
                            row[
                                "B"
                            ]
                        ),

                    "delta_oriented":
                        float(
                            row[
                                "delta_oriented"
                            ]
                        ),
                }
            )


# ==============================================================================================
# 9. SAVE
# ==============================================================================================

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
    DELTAS_OUT,
    index=False,
)


# ==============================================================================================
# 10. MAIN AP DECISIONS
# ==============================================================================================

def get_ap_decision(
    model_b
):

    row = summary[
        summary[
            "model_A"
        ].eq(
            "GN_EQ0"
        )
        &
        summary[
            "model_B"
        ].eq(
            model_b
        )
        &
        summary[
            "metric"
        ].eq(
            "AP"
        )
    ]

    if row.empty:
        return None

    r = row.iloc[
        0
    ]

    return {
        "comparison":
            f"GN_EQ0_vs_{model_b}",

        "mean_AP_oriented_delta":
            float(
                r[
                    "mean_oriented_delta"
                ]
            ),

        "median_AP_oriented_delta":
            float(
                r[
                    "median_oriented_delta"
                ]
            ),

        "months_GN_EQ0_better":
            int(
                r[
                    "months_A_better"
                ]
            ),

        "months_other_better":
            int(
                r[
                    "months_B_better"
                ]
            ),

        "wilcoxon_p":
            (
                None
                if pd.isna(
                    r[
                        "wilcoxon_p"
                    ]
                )
                else
                float(
                    r[
                        "wilcoxon_p"
                    ]
                )
            ),

        "exact_signflip_p":
            (
                None
                if pd.isna(
                    r[
                        "exact_signflip_p"
                    ]
                )
                else
                float(
                    r[
                        "exact_signflip_p"
                    ]
                )
            ),

        "block2_CI95":
            [
                float(
                    r[
                        "block2_CI95_low"
                    ]
                ),
                float(
                    r[
                        "block2_CI95_high"
                    ]
                ),
            ],

        "block3_CI95":
            [
                float(
                    r[
                        "block3_CI95_low"
                    ]
                ),
                float(
                    r[
                        "block3_CI95_high"
                    ]
                ),
            ],

        "LOMO_mean_range":
            [
                float(
                    r[
                        "LOMO_mean_min"
                    ]
                ),
                float(
                    r[
                        "LOMO_mean_max"
                    ]
                ),
            ],

        "classification":
            r[
                "evidence_classification"
            ],
    }


eq0_vs_logit = get_ap_decision(
    "LOGIT_MLE"
)

eq0_vs_ge0 = get_ap_decision(
    "GNADMM_GE0"
)


decision = {
    "status":
        "PASS",

    "model_refit":
        False,

    "raw_modified":
        False,

    "primary_metric":
        "AP",

    "interpretation":
        (
            "Positive oriented delta means GN_EQ0 "
            "performed better than the comparison model."
        ),

    "GN_EQ0_vs_LOGIT_MLE":
        eq0_vs_logit,

    "GN_EQ0_vs_GNADMM_GE0":
        eq0_vs_ge0,
}


DECISION_OUT.write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==============================================================================================
# 11. TERMINAL REPORT
# ==============================================================================================

print()
print("=" * 108)
print("PAIRED TEMPORAL RESULTS")
print("=" * 108)

display_cols = [
    "model_A",
    "model_B",
    "metric",
    "n_months",
    "mean_oriented_delta",
    "median_oriented_delta",
    "months_A_better",
    "months_B_better",
    "wilcoxon_p",
    "exact_signflip_p",
    "block2_CI95_low",
    "block2_CI95_high",
    "block3_CI95_low",
    "block3_CI95_high",
    "LOMO_mean_min",
    "LOMO_mean_max",
    "evidence_classification",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False
    )
)

print()
print("=" * 108)
print("FINAL AP DECISION")
print("=" * 108)

print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print(
    "[PASS 25F] PAIRED ABLATION INFERENCE COMPLETE."
)

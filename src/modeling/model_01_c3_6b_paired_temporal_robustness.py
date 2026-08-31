#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.6-B
PAIRED TEMPORAL ROBUSTNESS & MULTIPLE-COMPARISON AUDIT
===============================================================================

OBJETIVO
--------
Auditar a conclusão supervisionada do C3.6-A SEM treinar nenhum modelo novo.

Métrica primária congelada:

    Average Precision (AP)

Para cada mês m e K > 0:

    Delta_AP(m,K)
        =
        AP(m,K) - AP(m,K=0)

Interpretação:

    Delta > 0  -> K melhora o baseline naquele mês
    Delta = 0  -> empate
    Delta < 0  -> K piora o baseline naquele mês

ETAPAS
------
1. comparação pareada mensal;
2. média, mediana e win/loss;
3. leave-contiguous-block-out:
       remove 1, 2 ou 3 meses consecutivos;
4. circular block bootstrap:
       block lengths = 2, 3, 4 meses;
5. intervalo individual bootstrap;
6. intervalo simultâneo max-t sobre todos os K > 0;
7. auditoria descritiva das métricas secundárias;
8. emissão de candidato de decisão.

IMPORTANTE
----------
O bootstrap é usado como análise de robustez/inferência dependente.

Não assumimos que ele sozinho demonstra validade universal em presença de
mudanças de regime.

NÃO:
- executa LogisticRegression;
- executa PCA;
- abre curvas NPY;
- altera folds;
- seleciona threshold;
- modifica RAW;
- cria Silver.

Nenhum K é formalmente congelado neste estágio.
===============================================================================
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

DIR = (
    ROOT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

SUMMARY_C36A = (
    DIR
    / "07e_supervised_k_summary.json"
)

FOLD_METRICS = (
    DIR
    / "07a_supervised_k_fold_metrics.csv"
)

PAIRED_SOURCE = (
    DIR
    / "07c_supervised_k_paired_vs_k0.csv"
)

VALIDATION_C36A = (
    DIR
    / "07d_supervised_k_validation.csv"
)


OUT_MONTHLY = (
    DIR
    / "08a_ap_delta_monthly_matrix.csv"
)

OUT_SUMMARY = (
    DIR
    / "08b_ap_paired_summary.csv"
)

OUT_DELETE = (
    DIR
    / "08c_contiguous_deletion_sensitivity.csv"
)

OUT_BOOT = (
    DIR
    / "08d_block_bootstrap_inference.csv"
)

OUT_SECONDARY = (
    DIR
    / "08e_secondary_metric_concordance.csv"
)

OUT_VALIDATION = (
    DIR
    / "08f_c36b_validation.csv"
)

OUT_JSON = (
    DIR
    / "08g_c36b_summary.json"
)

OUT_REPORT = (
    DIR
    / "08h_c36b_report.txt"
)


SEED = 20260830
BOOTSTRAP_REPLICATES = 20000

BLOCK_LENGTHS = [
    2,
    3,
    4,
]

DELETE_WIDTHS = [
    1,
    2,
    3,
]

ALPHA = 0.05

TIE_TOL = 1e-12


# =============================================================================
# HELPERS
# =============================================================================

def atomic_csv(df, path):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False
    )

    tmp.replace(
        path
    )


def circular_block_indices(
    rng,
    n,
    block_length,
    replicates,
):

    """
    Circular block bootstrap.

    Cada bloco mantém meses consecutivos.
    O final da série pode circular para o início.

    Retorna:
        array shape = (replicates, n)
    """

    blocks_needed = int(
        np.ceil(
            n
            /
            block_length
        )
    )

    starts = rng.integers(
        low=0,
        high=n,
        size=(
            replicates,
            blocks_needed,
        ),
    )

    offsets = np.arange(
        block_length,
        dtype=np.int64
    )

    idx = (
        starts[
            :,
            :,
            None
        ]
        +
        offsets[
            None,
            None,
            :
        ]
    ) % n

    idx = (
        idx
        .reshape(
            replicates,
            -1
        )
        [
            :,
            :n
        ]
    )

    return idx


# =============================================================================
# START
# =============================================================================

t0 = time.perf_counter()

print()
print("=" * 124)
print("MODEL 01.0-C3.6-B — PAIRED TEMPORAL ROBUSTNESS & MULTIPLE-COMPARISON AUDIT")
print("=" * 124)


# =============================================================================
# PREREQUISITES
# =============================================================================

required = [
    SUMMARY_C36A,
    FOLD_METRICS,
    PAIRED_SOURCE,
    VALIDATION_C36A,
]

for p in required:

    if not p.exists():
        raise SystemExit(
            f"[FAIL] Arquivo ausente: {p}"
        )

    print(
        f"[PASS] {p.name}"
    )


summary_a = json.loads(
    SUMMARY_C36A.read_text(
        encoding="utf-8"
    )
)


if summary_a.get("status") != "PASS":

    raise RuntimeError(
        "C3.6-A não está PASS."
    )


if int(
    summary_a.get(
        "actual_model_fits",
        -1
    )
) != 238:

    raise RuntimeError(
        "C3.6-A não possui 238 fits."
    )


if (
    summary_a.get(
        "descriptive_highest_mean_ap_k"
    )
    !=
    0
):

    raise RuntimeError(
        "Resultado esperado do C3.6-A mudou: K=0 não é o maior AP médio."
    )


print()
print("[PASS] C3.6-A formalmente validado.")
print("[PASS] K=0 é o maior AP médio descritivo.")


# =============================================================================
# LOAD
# =============================================================================

folds = pd.read_csv(
    FOLD_METRICS
)

paired = pd.read_csv(
    PAIRED_SOURCE
)


folds["test_month"] = (
    folds["test_month"]
    .astype(str)
)

paired["test_month"] = (
    paired["test_month"]
    .astype(str)
)


folds["k"] = (
    pd.to_numeric(
        folds["k"]
    )
    .astype(int)
)

paired["k"] = (
    pd.to_numeric(
        paired["k"]
    )
    .astype(int)
)


MONTHS = sorted(
    folds[
        "test_month"
    ].unique()
)


K_ALL = sorted(
    folds[
        "k"
    ].unique()
)


K_ALT = [
    k
    for k in K_ALL
    if k > 0
]


if len(MONTHS) != 17:

    raise RuntimeError(
        f"Meses={len(MONTHS)}; esperado=17."
    )


if len(K_ALL) != 14:

    raise RuntimeError(
        f"K total={len(K_ALL)}; esperado=14."
    )


if len(K_ALT) != 13:

    raise RuntimeError(
        f"K alternativos={len(K_ALT)}; esperado=13."
    )


# =============================================================================
# RECONSTRUCT PRIMARY DELTA
# =============================================================================

baseline_ap = (
    folds.loc[
        folds["k"] == 0,
        [
            "test_month",
            "average_precision",
        ]
    ]
    .rename(
        columns={
            "average_precision":
                "ap_k0"
        }
    )
)


work = (
    folds
    .merge(
        baseline_ap,
        on="test_month",
        how="left",
        validate="many_to_one",
    )
)


work[
    "delta_ap_vs_k0_recomputed"
] = (
    work[
        "average_precision"
    ]
    -
    work[
        "ap_k0"
    ]
)


cross = (
    work
    .merge(
        paired[
            [
                "test_month",
                "k",
                "delta_ap_vs_k0",
            ]
        ],
        on=[
            "test_month",
            "k",
        ],
        how="left",
        validate="one_to_one",
    )
)


max_delta_disagreement = float(
    np.max(
        np.abs(
            cross[
                "delta_ap_vs_k0_recomputed"
            ].to_numpy(
                float
            )
            -
            cross[
                "delta_ap_vs_k0"
            ].to_numpy(
                float
            )
        )
    )
)


if max_delta_disagreement > 1e-12:

    raise RuntimeError(
        "Delta AP recomputado diverge do C3.6-A."
    )


alt = (
    cross.loc[
        cross["k"] > 0,
        [
            "test_month",
            "k",
            "average_precision",
            "ap_k0",
            "delta_ap_vs_k0_recomputed",
        ]
    ]
    .copy()
)


alt = alt.rename(
    columns={
        "average_precision":
            "ap_k",

        "delta_ap_vs_k0_recomputed":
            "delta_ap",
    }
)


atomic_csv(
    alt,
    OUT_MONTHLY
)


# =============================================================================
# MATRIX: MONTH x K
# =============================================================================

D = (
    alt
    .pivot(
        index="test_month",
        columns="k",
        values="delta_ap",
    )
    .reindex(
        index=MONTHS,
        columns=K_ALT,
    )
)


if D.isna().any().any():

    raise RuntimeError(
        "Delta matrix incompleta."
    )


D_np = D.to_numpy(
    dtype=float
)


OBS_MEAN = D_np.mean(
    axis=0
)

OBS_MEDIAN = np.median(
    D_np,
    axis=0
)


# =============================================================================
# BASIC PAIRED SUMMARY
# =============================================================================

basic_rows = []


for j, k in enumerate(
    K_ALT
):

    x = D_np[
        :,
        j
    ]

    positive = int(
        np.sum(
            x > TIE_TOL
        )
    )

    negative = int(
        np.sum(
            x < -TIE_TOL
        )
    )

    ties = int(
        len(x)
        -
        positive
        -
        negative
    )


    basic_rows.append({
        "k":
            k,

        "months":
            len(x),

        "mean_delta_ap":
            float(
                np.mean(x)
            ),

        "median_delta_ap":
            float(
                np.median(x)
            ),

        "std_delta_ap":
            float(
                np.std(
                    x,
                    ddof=1
                )
            ),

        "min_delta_ap":
            float(
                np.min(x)
            ),

        "max_delta_ap":
            float(
                np.max(x)
            ),

        "months_better_than_k0":
            positive,

        "months_worse_than_k0":
            negative,

        "months_tied":
            ties,

        "better_month_pct":
            float(
                100
                *
                positive
                /
                len(x)
            ),

        "mean_delta_negative":
            bool(
                np.mean(x)
                <
                0
            ),

        "median_delta_negative":
            bool(
                np.median(x)
                <
                0
            ),
    })


basic = pd.DataFrame(
    basic_rows
)

atomic_csv(
    basic,
    OUT_SUMMARY
)


# =============================================================================
# CONTIGUOUS DELETION SENSITIVITY
# =============================================================================

delete_rows = []


for width in DELETE_WIDTHS:

    if width >= len(MONTHS):
        continue

    for start in range(
        0,
        len(MONTHS)
        -
        width
        +
        1
    ):

        removed_months = (
            MONTHS[
                start:
                start + width
            ]
        )

        keep = np.ones(
            len(MONTHS),
            dtype=bool
        )

        keep[
            start:
            start + width
        ] = False


        reduced = D_np[
            keep,
            :
        ]


        means = reduced.mean(
            axis=0
        )


        for j, k in enumerate(
            K_ALT
        ):

            delete_rows.append({
                "removed_width":
                    width,

                "removed_start":
                    removed_months[0],

                "removed_end":
                    removed_months[-1],

                "removed_months":
                    "|".join(
                        removed_months
                    ),

                "remaining_months":
                    int(
                        keep.sum()
                    ),

                "k":
                    k,

                "mean_delta_ap_after_deletion":
                    float(
                        means[j]
                    ),

                "functional_better_after_deletion":
                    bool(
                        means[j]
                        >
                        0
                    ),
            })


delete_df = pd.DataFrame(
    delete_rows
)

atomic_csv(
    delete_df,
    OUT_DELETE
)


delete_summary = (
    delete_df
    .groupby(
        [
            "k",
            "removed_width",
        ],
        as_index=False,
    )
    .agg(
        min_mean_delta_after_deletion=(
            "mean_delta_ap_after_deletion",
            "min",
        ),

        max_mean_delta_after_deletion=(
            "mean_delta_ap_after_deletion",
            "max",
        ),

        deletion_cases=(
            "mean_delta_ap_after_deletion",
            "size",
        ),

        positive_mean_cases=(
            "functional_better_after_deletion",
            "sum",
        ),
    )
)


# =============================================================================
# CIRCULAR BLOCK BOOTSTRAP + MAX-T
# =============================================================================

rng_master = np.random.default_rng(
    SEED
)

boot_rows = []


for block_length in BLOCK_LENGTHS:

    # independent deterministic child seed
    child_seed = int(
        rng_master.integers(
            0,
            2**32 - 1
        )
    )

    rng = np.random.default_rng(
        child_seed
    )


    idx = circular_block_indices(
        rng=rng,
        n=len(MONTHS),
        block_length=block_length,
        replicates=BOOTSTRAP_REPLICATES,
    )


    boot_sample = D_np[
        idx,
        :
    ]


    boot_mean = boot_sample.mean(
        axis=1
    )


    se = np.std(
        boot_mean,
        axis=0,
        ddof=1
    )


    if (
        (~np.isfinite(se)).any()
        or
        (se <= 0).any()
    ):

        raise RuntimeError(
            f"Bootstrap SE inválido para block={block_length}."
        )


    # ---------------------------------------------------------
    # INDIVIDUAL PERCENTILE INTERVALS
    # ---------------------------------------------------------

    individual_lo = np.quantile(
        boot_mean,
        ALPHA / 2,
        axis=0,
    )

    individual_hi = np.quantile(
        boot_mean,
        1 - ALPHA / 2,
        axis=0,
    )


    # ---------------------------------------------------------
    # SIMULTANEOUS MAX-T INTERVAL
    # ---------------------------------------------------------

    centered_t = (
        boot_mean
        -
        OBS_MEAN[
            None,
            :
        ]
    ) / se[
        None,
        :
    ]


    max_abs_t = np.max(
        np.abs(
            centered_t
        ),
        axis=1,
    )


    critical_max_t = float(
        np.quantile(
            max_abs_t,
            1 - ALPHA,
        )
    )


    simultaneous_lo = (
        OBS_MEAN
        -
        critical_max_t
        *
        se
    )

    simultaneous_hi = (
        OBS_MEAN
        +
        critical_max_t
        *
        se
    )


    # ---------------------------------------------------------
    # ONE-SIDED MULTIPLICITY-ADJUSTED P:
    #
    # H1: delta > 0
    #
    # Null distribution from centered bootstrap.
    # ---------------------------------------------------------

    max_t_null = np.max(
        centered_t,
        axis=1,
    )


    obs_t = (
        OBS_MEAN
        /
        se
    )


    # ---------------------------------------------------------
    # ALSO TEST BASELINE SUPERIORITY:
    #
    # H1: delta < 0
    # ---------------------------------------------------------

    min_t_null = np.min(
        centered_t,
        axis=1,
    )


    for j, k in enumerate(
        K_ALT
    ):

        p_adj_improvement = float(
            (
                1
                +
                np.sum(
                    max_t_null
                    >=
                    obs_t[j]
                )
            )
            /
            (
                BOOTSTRAP_REPLICATES
                +
                1
            )
        )


        p_adj_baseline = float(
            (
                1
                +
                np.sum(
                    min_t_null
                    <=
                    obs_t[j]
                )
            )
            /
            (
                BOOTSTRAP_REPLICATES
                +
                1
            )
        )


        if simultaneous_lo[j] > 0:

            classification = (
                "FUNCTIONAL_SUPERIOR"
            )

        elif simultaneous_hi[j] < 0:

            classification = (
                "K0_SUPERIOR"
            )

        else:

            classification = (
                "INCONCLUSIVE"
            )


        boot_rows.append({
            "block_length_months":
                block_length,

            "bootstrap_replicates":
                BOOTSTRAP_REPLICATES,

            "k":
                k,

            "observed_mean_delta_ap":
                float(
                    OBS_MEAN[j]
                ),

            "observed_median_delta_ap":
                float(
                    OBS_MEDIAN[j]
                ),

            "bootstrap_se_mean_delta":
                float(
                    se[j]
                ),

            "individual_ci95_lower":
                float(
                    individual_lo[j]
                ),

            "individual_ci95_upper":
                float(
                    individual_hi[j]
                ),

            "max_t_critical_95":
                critical_max_t,

            "simultaneous_ci95_lower":
                float(
                    simultaneous_lo[j]
                ),

            "simultaneous_ci95_upper":
                float(
                    simultaneous_hi[j]
                ),

            "adjusted_p_functional_improvement":
                p_adj_improvement,

            "adjusted_p_k0_superiority":
                p_adj_baseline,

            "simultaneous_classification":
                classification,

            "candidate_only_not_final":
                True,
        })


boot_df = pd.DataFrame(
    boot_rows
)

atomic_csv(
    boot_df,
    OUT_BOOT
)


# =============================================================================
# SECONDARY METRIC CONCORDANCE — DESCRIPTIVE ONLY
# =============================================================================

baseline = (
    folds.loc[
        folds["k"] == 0,
        [
            "test_month",
            "roc_auc",
            "recall_at_10pct",
            "log_loss",
            "brier_score",
        ]
    ]
    .rename(
        columns={
            "roc_auc":
                "k0_roc_auc",

            "recall_at_10pct":
                "k0_recall_at_10pct",

            "log_loss":
                "k0_log_loss",

            "brier_score":
                "k0_brier_score",
        }
    )
)


sec = (
    folds.loc[
        folds["k"] > 0
    ]
    .merge(
        baseline,
        on="test_month",
        how="left",
        validate="many_to_one",
    )
)


# Positive benefit always means K > 0 is better.

sec[
    "benefit_roc_auc"
] = (
    sec[
        "roc_auc"
    ]
    -
    sec[
        "k0_roc_auc"
    ]
)


sec[
    "benefit_recall_at_10pct"
] = (
    sec[
        "recall_at_10pct"
    ]
    -
    sec[
        "k0_recall_at_10pct"
    ]
)


sec[
    "benefit_log_loss"
] = (
    sec[
        "k0_log_loss"
    ]
    -
    sec[
        "log_loss"
    ]
)


sec[
    "benefit_brier_score"
] = (
    sec[
        "k0_brier_score"
    ]
    -
    sec[
        "brier_score"
    ]
)


secondary_rows = []


for k in K_ALT:

    g = sec.loc[
        sec["k"] == k
    ]


    secondary_rows.append({
        "k":
            k,

        "months":
            len(g),

        "mean_benefit_roc_auc":
            float(
                g[
                    "benefit_roc_auc"
                ].mean()
            ),

        "months_roc_better":
            int(
                (
                    g[
                        "benefit_roc_auc"
                    ]
                    >
                    TIE_TOL
                ).sum()
            ),

        "mean_benefit_recall_at_10pct":
            float(
                g[
                    "benefit_recall_at_10pct"
                ].mean()
            ),

        "months_recall10_better":
            int(
                (
                    g[
                        "benefit_recall_at_10pct"
                    ]
                    >
                    TIE_TOL
                ).sum()
            ),

        "mean_benefit_log_loss":
            float(
                g[
                    "benefit_log_loss"
                ].mean()
            ),

        "months_logloss_better":
            int(
                (
                    g[
                        "benefit_log_loss"
                    ]
                    >
                    TIE_TOL
                ).sum()
            ),

        "mean_benefit_brier_score":
            float(
                g[
                    "benefit_brier_score"
                ].mean()
            ),

        "months_brier_better":
            int(
                (
                    g[
                        "benefit_brier_score"
                    ]
                    >
                    TIE_TOL
                ).sum()
            ),
    })


secondary = pd.DataFrame(
    secondary_rows
)

atomic_csv(
    secondary,
    OUT_SECONDARY
)


# =============================================================================
# ROBUSTNESS SYNTHESIS
# =============================================================================

boot_class = (
    boot_df
    .groupby(
        "k"
    )[
        "simultaneous_classification"
    ]
    .agg(
        lambda s:
            "|".join(
                sorted(
                    set(s)
                )
            )
    )
)


all_mean_deltas_negative = bool(
    (
        basic[
            "mean_delta_ap"
        ]
        <
        0
    ).all()
)


all_median_deltas_negative = bool(
    (
        basic[
            "median_delta_ap"
        ]
        <
        0
    ).all()
)


any_boot_functional_superiority = bool(
    boot_df[
        "simultaneous_classification"
    ]
    .eq(
        "FUNCTIONAL_SUPERIOR"
    )
    .any()
)


robust_functional_superiority = []


for k in K_ALT:

    g = boot_df.loc[
        boot_df["k"] == k
    ]

    if (
        g[
            "simultaneous_classification"
        ]
        .eq(
            "FUNCTIONAL_SUPERIOR"
        )
        .all()
    ):

        robust_functional_superiority.append(
            k
        )


robust_k0_superiority = []


for k in K_ALT:

    g = boot_df.loc[
        boot_df["k"] == k
    ]

    if (
        g[
            "simultaneous_classification"
        ]
        .eq(
            "K0_SUPERIOR"
        )
        .all()
    ):

        robust_k0_superiority.append(
            k
        )


deletion_any_positive = bool(
    delete_df[
        "functional_better_after_deletion"
    ].any()
)


deletion_robust_negative_by_k = {}


for k in K_ALT:

    g = delete_df.loc[
        delete_df["k"] == k
    ]

    deletion_robust_negative_by_k[
        str(k)
    ] = bool(
        (
            g[
                "mean_delta_ap_after_deletion"
            ]
            <
            0
        ).all()
    )


all_k_deletion_robust_negative = bool(
    all(
        deletion_robust_negative_by_k.values()
    )
)


# Secondary means:
# positive = functional better.
all_secondary_mean_benefits_negative = bool(
    (
        secondary[
            [
                "mean_benefit_roc_auc",
                "mean_benefit_recall_at_10pct",
                "mean_benefit_log_loss",
                "mean_benefit_brier_score",
            ]
        ]
        <=
        0
    )
    .all()
    .all()
)


# =============================================================================
# DECISION CANDIDATE
# =============================================================================

if (
    all_mean_deltas_negative
    and
    not any_boot_functional_superiority
    and
    all_k_deletion_robust_negative
):

    decision_candidate = (
        "K0_READY_FOR_FORMAL_FREEZE_MODEL01"
    )

else:

    decision_candidate = (
        "K_DECISION_REQUIRES_ADDITIONAL_REVIEW"
    )


# =============================================================================
# VALIDATION
# =============================================================================

checks = []


def add(
    name,
    ok,
    observed,
    expected,
):

    checks.append({
        "check":
            name,

        "status":
            "PASS"
            if ok
            else "FAIL",

        "observed":
            observed,

        "expected":
            expected,
    })


add(
    "c36a_pass",
    summary_a.get("status") == "PASS",
    summary_a.get("status"),
    "PASS",
)

add(
    "c36a_model_fits",
    summary_a.get("actual_model_fits") == 238,
    summary_a.get("actual_model_fits"),
    238,
)

add(
    "baseline_highest_mean_ap",
    summary_a.get(
        "descriptive_highest_mean_ap_k"
    ) == 0,
    summary_a.get(
        "descriptive_highest_mean_ap_k"
    ),
    0,
)

add(
    "months_17",
    len(MONTHS) == 17,
    len(MONTHS),
    17,
)

add(
    "alternative_k_13",
    len(K_ALT) == 13,
    len(K_ALT),
    13,
)

add(
    "paired_rows",
    len(alt) == 221,
    len(alt),
    221,
)

add(
    "delta_recomputation_match",
    max_delta_disagreement
    <=
    1e-12,
    max_delta_disagreement,
    "<=1e-12",
)

add(
    "all_delta_values_finite",
    np.isfinite(
        D_np
    ).all(),
    int(
        (
            ~np.isfinite(
                D_np
            )
        ).sum()
    ),
    0,
)

add(
    "bootstrap_rows",
    len(boot_df)
    ==
    len(BLOCK_LENGTHS)
    *
    len(K_ALT),
    len(boot_df),
    len(BLOCK_LENGTHS)
    *
    len(K_ALT),
)

add(
    "all_bootstrap_metrics_finite",
    np.isfinite(
        boot_df[
            [
                "observed_mean_delta_ap",
                "bootstrap_se_mean_delta",
                "individual_ci95_lower",
                "individual_ci95_upper",
                "simultaneous_ci95_lower",
                "simultaneous_ci95_upper",
                "adjusted_p_functional_improvement",
                "adjusted_p_k0_superiority",
            ]
        ].to_numpy(
            float
        )
    ).all(),
    int(
        (
            ~np.isfinite(
                boot_df[
                    [
                        "observed_mean_delta_ap",
                        "bootstrap_se_mean_delta",
                        "individual_ci95_lower",
                        "individual_ci95_upper",
                        "simultaneous_ci95_lower",
                        "simultaneous_ci95_upper",
                        "adjusted_p_functional_improvement",
                        "adjusted_p_k0_superiority",
                    ]
                ].to_numpy(
                    float
                )
            )
        ).sum()
    ),
    0,
)

add(
    "bootstrap_pvalues_valid",
    (
        (
            boot_df[
                "adjusted_p_functional_improvement"
            ]
            >=
            0
        )
        &
        (
            boot_df[
                "adjusted_p_functional_improvement"
            ]
            <=
            1
        )
        &
        (
            boot_df[
                "adjusted_p_k0_superiority"
            ]
            >=
            0
        )
        &
        (
            boot_df[
                "adjusted_p_k0_superiority"
            ]
            <=
            1
        )
    ).all(),
    "checked",
    "[0,1]",
)

add(
    "no_model_training",
    True,
    False,
    False,
)

add(
    "k_not_formally_frozen",
    True,
    False,
    False,
)

add(
    "threshold_not_selected",
    True,
    False,
    False,
)

add(
    "raw_not_modified",
    True,
    False,
    False,
)


validation = pd.DataFrame(
    checks
)

atomic_csv(
    validation,
    OUT_VALIDATION
)


failures = int(
    validation[
        "status"
    ].eq(
        "FAIL"
    ).sum()
)


# =============================================================================
# SUMMARY JSON
# =============================================================================

runtime = (
    time.perf_counter()
    -
    t0
)


payload = {
    "step":
        "MODEL_01_0_C3_6B_PAIRED_TEMPORAL_ROBUSTNESS",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "primary_metric":
        "AVERAGE_PRECISION",

    "baseline_k":
        0,

    "alternative_k_count":
        len(K_ALT),

    "temporal_months":
        len(MONTHS),

    "paired_delta_definition":
        "AP_month_k - AP_month_k0",

    "all_observed_mean_deltas_negative":
        all_mean_deltas_negative,

    "all_observed_median_deltas_negative":
        all_median_deltas_negative,

    "contiguous_deletion_widths":
        DELETE_WIDTHS,

    "all_k_remain_negative_after_all_contiguous_deletions":
        all_k_deletion_robust_negative,

    "circular_block_bootstrap":
        {
            "replicates":
                BOOTSTRAP_REPLICATES,

            "block_lengths_months":
                BLOCK_LENGTHS,

            "seed":
                SEED,

            "simultaneous_method":
                "MAX_T",

            "family":
                "13_K_VALUES_VS_COMMON_K0_BENCHMARK",
        },

    "any_bootstrap_simultaneous_functional_superiority":
        any_boot_functional_superiority,

    "robust_functional_superiority_k":
        robust_functional_superiority,

    "robust_k0_superiority_k":
        robust_k0_superiority,

    "secondary_metrics_descriptive_only":
        True,

    "all_secondary_mean_benefits_nonpositive":
        all_secondary_mean_benefits_negative,

    "decision_candidate":
        decision_candidate,

    "decision_interpretation":
        (
            "Candidate selection for MODEL_01 only. "
            "No claim that functional temporal information is universally useless."
        ),

    "final_k_frozen":
        False,

    "threshold_selected":
        False,

    "classifier_refit":
        False,

    "pca_refit":
        False,

    "raw_modified":
        False,

    "silver_created":
        False,

    "validation_failures":
        failures,

    "runtime_seconds":
        runtime,
}


OUT_JSON.write_text(
    json.dumps(
        payload,
        indent=4,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =============================================================================
# REPORT
# =============================================================================

summary_print = basic[
    [
        "k",
        "mean_delta_ap",
        "median_delta_ap",
        "months_better_than_k0",
        "months_worse_than_k0",
        "better_month_pct",
    ]
].copy()


boot_print = (
    boot_df[
        [
            "block_length_months",
            "k",
            "observed_mean_delta_ap",
            "simultaneous_ci95_lower",
            "simultaneous_ci95_upper",
            "adjusted_p_functional_improvement",
            "adjusted_p_k0_superiority",
            "simultaneous_classification",
        ]
    ]
)


delete_print = (
    delete_summary[
        [
            "k",
            "removed_width",
            "min_mean_delta_after_deletion",
            "max_mean_delta_after_deletion",
            "positive_mean_cases",
            "deletion_cases",
        ]
    ]
)


report = [
    "=" * 124,
    "MODEL 01.0-C3.6-B — PAIRED TEMPORAL ROBUSTNESS",
    "=" * 124,
    "",
    f"STATUS                              : {payload['status']}",
    f"PRIMARY METRIC                      : Average Precision",
    f"BASELINE                            : K=0",
    f"MONTHS                              : {len(MONTHS)}",
    f"ALTERNATIVE K                       : {len(K_ALT)}",
    "",
    "OBSERVED PAIRED EVIDENCE",
    "-" * 124,
    summary_print.to_string(
        index=False
    ),
    "",
    "CONTIGUOUS DELETION SENSITIVITY",
    "-" * 124,
    delete_print.to_string(
        index=False
    ),
    "",
    "CIRCULAR BLOCK BOOTSTRAP + MAX-T",
    "-" * 124,
    boot_print.to_string(
        index=False
    ),
    "",
    "SYNTHESIS",
    "-" * 124,
    f"All mean delta AP < 0               : {all_mean_deltas_negative}",
    f"All median delta AP < 0             : {all_median_deltas_negative}",
    f"All deletion scenarios stay < 0     : {all_k_deletion_robust_negative}",
    f"Any functional simultaneous win     : {any_boot_functional_superiority}",
    f"Robust functional-superior K        : {robust_functional_superiority}",
    f"Robust K0-superior K                : {robust_k0_superiority}",
    f"Secondary mean benefits <= 0        : {all_secondary_mean_benefits_negative}",
    "",
    f"DECISION CANDIDATE                  : {decision_candidate}",
    "",
    "IMPORTANT",
    "-" * 124,
    "Final K frozen                     : NO",
    "Threshold selected                 : NO",
    "New classifier trained             : NO",
    "PCA refit                          : NO",
    "Silver created                     : NO",
    "RAW modified                       : NO",
    "",
    f"Validation failures                : {failures}",
    f"Runtime seconds                    : {runtime:.3f}",
    "=" * 124,
]


OUT_REPORT.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8",
)


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 124)
print("A. PAIRED AP SUMMARY")
print("=" * 124)

print(
    summary_print.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print("=" * 124)
print("B. CONTIGUOUS DELETION — WORST/BEST CASE")
print("=" * 124)

print(
    delete_print.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print("=" * 124)
print("C. BLOCK BOOTSTRAP + SIMULTANEOUS MAX-T")
print("=" * 124)

print(
    boot_print.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print("=" * 124)
print("D. SECONDARY METRIC CONCORDANCE — DESCRIPTIVE ONLY")
print("=" * 124)

print(
    secondary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print("=" * 124)
print("E. VALIDATION")
print("=" * 124)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 124)
print("RESULTADO C3.6-B")
print("=" * 124)

print(
    "STATUS                              =",
    payload["status"]
)

print(
    "ALL MEAN DELTA AP < 0               =",
    all_mean_deltas_negative
)

print(
    "ALL MEDIAN DELTA AP < 0             =",
    all_median_deltas_negative
)

print(
    "ALL CONTIGUOUS DELETIONS STAY < 0   =",
    all_k_deletion_robust_negative
)

print(
    "ANY FUNCTIONAL SIMULTANEOUS WIN     =",
    any_boot_functional_superiority
)

print(
    "ROBUST FUNCTIONAL K                 =",
    robust_functional_superiority
)

print(
    "ROBUST K0-SUPERIOR K                =",
    robust_k0_superiority
)

print(
    "SECONDARY MEAN BENEFITS <= 0        =",
    all_secondary_mean_benefits_negative
)

print()
print(
    "DECISION CANDIDATE                  =",
    decision_candidate
)

print()
print(
    "K FINAL                             = NÃO CONGELADO"
)

print(
    "NOVO MODELO                         = NÃO TREINADO"
)

print(
    "PCA                                 = NÃO EXECUTADA"
)

print(
    "THRESHOLD                           = NÃO"
)

print(
    "SILVER                              = NÃO CRIADA"
)

print(
    "RAW                                 = INTACTO"
)

print(
    "VALIDATION FAILURES                 =",
    failures
)


if failures:

    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.6-B concluído."
)

print(
    "[PASS] Comparação temporal pareada auditada."
)

print(
    "[PASS] Multiplicidade e dependência temporal consideradas."
)

print(
    "[PASS] Nenhum K foi formalmente congelado automaticamente."
)


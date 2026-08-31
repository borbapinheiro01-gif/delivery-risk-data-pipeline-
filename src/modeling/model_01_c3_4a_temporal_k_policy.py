#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.4-A
RAW-30D — TEMPORAL COMPONENT-COUNT POLICY
===============================================================================

OBJETIVO
--------
Avaliar, SEM TARGET, políticas temporais para escolha do número K
de componentes da representação funcional RAW-30D.

Políticas candidatas de proporção de variância explicada:

    q = 0.80
        0.85
        0.90
        0.95
        0.99

Para cada mês futuro m:

    TRAIN = pedidos anteriores a m
    TEST  = pedidos do mês m

A PCA é aprendida somente no TRAIN.

Para cada q:

    K_q =
        menor K cuja variância acumulada no TRAIN >= q

e avaliamos no TEST:

    RE(q,m) =
        ||X_test - Xhat_test(q)||_F
        --------------------------------
        ||X_test||_F

IMPORTANTE
----------
Este passo NÃO escolhe K final.

Ele produz:

- trade-off K x erro futuro;
- estabilidade temporal de K;
- comparação entre políticas PVE;
- candidato parcimonioso via 1-SE;
- candidato conjunto entre os dois canais.

NÃO:
- usa target;
- treina classificador;
- altera RAW;
- cria Silver;
- congela folds finais.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import sys
import time

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

REPORT_DIR = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

ARTIFACT_DIR = (
    PROJECT
    / "artifacts"
    / "model_01_order_logistic"
    / "functional_feasibility"
)

DECISION = (
    REPORT_DIR
    / "03s_functional_representation_decision.json"
)

ORDER_INDEX = (
    REPORT_DIR
    / "02a_functional_pit_order_index.csv"
)

VOLUME_PATH = (
    ARTIFACT_DIR
    / "02g_purchase_volume_curve_90d.npy"
)

FREIGHT_PATH = (
    ARTIFACT_DIR
    / "02h_purchase_freight_curve_90d.npy"
)

RAW_ORDERS = (
    PROJECT
    / "data"
    / "raw"
    / "olist"
    / "olist_orders_dataset.csv"
)


OUT_FOLDS = (
    REPORT_DIR
    / "04a_raw30_k_policy_folds.csv"
)

OUT_SUMMARY = (
    REPORT_DIR
    / "04b_raw30_k_policy_summary.csv"
)

OUT_ONE_SE = (
    REPORT_DIR
    / "04c_raw30_k_policy_one_se.csv"
)

OUT_VALIDATION = (
    REPORT_DIR
    / "04d_raw30_k_policy_validation.csv"
)

OUT_JSON = (
    REPORT_DIR
    / "04e_raw30_k_policy_summary.json"
)

OUT_REPORT = (
    REPORT_DIR
    / "04f_raw30_k_policy_report.txt"
)


# =============================================================================
# CONFIG
# =============================================================================

WINDOW = 30

PVE_POLICIES = [
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]

MIN_TRAIN = 5000
MIN_TEST = 500

CHANNELS = {
    "purchase_volume":
        VOLUME_PATH,

    "purchase_freight":
        FREIGHT_PATH,
}


# =============================================================================
# HELPERS
# =============================================================================

def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def k_from_pve(
    cumulative,
    q
):
    return int(
        np.searchsorted(
            cumulative,
            q,
            side="left"
        )
        +
        1
    )


def reconstruction_error(
    X,
    V,
    k
):
    Vk = V[:, :k]

    scores = (
        X
        @
        Vk
    )

    reconstruction = (
        scores
        @
        Vk.T
    )

    residual = (
        X
        -
        reconstruction
    )

    denominator = float(
        np.linalg.norm(
            X,
            ord="fro"
        )
    )

    numerator = float(
        np.linalg.norm(
            residual,
            ord="fro"
        )
    )

    if denominator <= 0:
        return np.nan

    return (
        numerator
        /
        denominator
    )


def add_check(
    rows,
    name,
    condition,
    observed,
    expected
):
    rows.append({
        "check":
            name,

        "status":
            "PASS"
            if bool(condition)
            else
            "FAIL",

        "observed":
            str(observed),

        "expected":
            str(expected),
    })


# =============================================================================
# START
# =============================================================================

start = time.perf_counter()

print()
print("=" * 116)
print("MODEL 01.0-C3.4-A — RAW-30D TEMPORAL K POLICY")
print("=" * 116)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required = [
    DECISION,
    ORDER_INDEX,
    VOLUME_PATH,
    FREIGHT_PATH,
    RAW_ORDERS,
]


for path in required:

    if not path.is_file():

        print(
            "[FAIL] Ausente:",
            path
        )

        sys.exit(2)

    print(
        "[PASS]",
        path.name
    )


decision = read_json(
    DECISION
)


primary = decision.get(
    "primary_representation_for_next_stage",
    {}
)


if (
    decision.get("status") != "PASS"
    or
    primary.get("representation")
    !=
    "RAW_30D"
    or
    primary.get("status")
    !=
    "PRIMARY_FOR_NEXT_FUNCTIONAL_STAGE"
):

    print(
        "[FAIL] RAW-30D ainda não está autorizado."
    )

    sys.exit(2)


print()
print("[PASS] D2.1-A autoriza RAW-30D.")


# =============================================================================
# 2. ORDER INDEX
# =============================================================================

index = pd.read_csv(
    ORDER_INDEX,
    low_memory=False
)


time_candidates = [
    "order_purchase_timestamp",
    "prediction_time",
]


time_col = None


for candidate in time_candidates:

    if candidate in index.columns:
        time_col = candidate
        break


if time_col is None:

    raise RuntimeError(
        "Timestamp de previsão não encontrado no order index."
    )


index[time_col] = pd.to_datetime(
    index[time_col],
    errors="coerce"
)


if index[time_col].isna().any():

    raise RuntimeError(
        "Há timestamps inválidos no order index."
    )


if len(index) != 96470:

    raise RuntimeError(
        "Order index não possui 96,470 pedidos."
    )


index["purchase_month"] = (
    index[time_col]
    .dt
    .to_period("M")
)


# =============================================================================
# 3. HISTORICAL SUPPORT — 30D
# =============================================================================

raw_time = pd.read_csv(
    RAW_ORDERS,
    usecols=[
        "order_purchase_timestamp"
    ]
)


raw_time[
    "order_purchase_timestamp"
] = pd.to_datetime(
    raw_time[
        "order_purchase_timestamp"
    ],
    errors="coerce"
)


raw_start = (
    raw_time[
        "order_purchase_timestamp"
    ]
    .min()
)


support_start = (
    raw_start
    +
    pd.Timedelta(
        days=WINDOW
    )
)


index[
    "full_support_30d"
] = (
    index[time_col]
    >=
    support_start
)


full_support_count = int(
    index[
        "full_support_30d"
    ].sum()
)


print()
print("=" * 116)
print("30D SUPPORT")
print("=" * 116)

print(
    "RAW purchase start :",
    raw_start
)

print(
    "30D support start  :",
    support_start
)

print(
    "Full support rows  :",
    f"{full_support_count:,}"
)


# =============================================================================
# 4. TEMPORAL MONTHS
# =============================================================================

months = sorted(
    index[
        "purchase_month"
    ]
    .dropna()
    .unique()
)


eligible_months = []


for month in months:

    train_mask = (
        index[
            "full_support_30d"
        ]
        &
        (
            index[
                "purchase_month"
            ]
            <
            month
        )
    )

    test_mask = (
        index[
            "full_support_30d"
        ]
        &
        (
            index[
                "purchase_month"
            ]
            ==
            month
        )
    )

    n_train = int(
        train_mask.sum()
    )

    n_test = int(
        test_mask.sum()
    )


    if (
        n_train >= MIN_TRAIN
        and
        n_test >= MIN_TEST
    ):
        eligible_months.append(
            month
        )


print()
print(
    "Eligible months     :",
    len(
        eligible_months
    )
)

print(
    "First test month    :",
    eligible_months[0]
)

print(
    "Last test month     :",
    eligible_months[-1]
)


# =============================================================================
# 5. TEMPORAL PCA
# =============================================================================

fold_rows = []


for channel, path in CHANNELS.items():

    print()
    print("=" * 116)
    print(
        "CHANNEL:",
        channel
    )
    print("=" * 116)

    curves = np.load(
        path,
        mmap_mode="r"
    )


    if curves.shape != (
        len(index),
        90
    ):

        raise RuntimeError(
            f"{channel}: shape inesperado {curves.shape}"
        )


    for month in eligible_months:

        train_idx = np.flatnonzero(
            (
                index[
                    "full_support_30d"
                ]
                &
                (
                    index[
                        "purchase_month"
                    ]
                    <
                    month
                )
            ).to_numpy()
        )


        test_idx = np.flatnonzero(
            (
                index[
                    "full_support_30d"
                ]
                &
                (
                    index[
                        "purchase_month"
                    ]
                    ==
                    month
                )
            ).to_numpy()
        )


        H_train = np.asarray(
            curves[
                train_idx,
                :WINDOW
            ],
            dtype=np.float64
        )


        H_test = np.asarray(
            curves[
                test_idx,
                :WINDOW
            ],
            dtype=np.float64
        )


        mu = H_train.mean(
            axis=0
        )


        X_train = (
            H_train
            -
            mu
        )


        X_test = (
            H_test
            -
            mu
        )


        # =============================================================
        # PCA via covariance 30 x 30
        # =============================================================

        gram = (
            X_train.T
            @
            X_train
        )


        eigenvalues, eigenvectors = (
            np.linalg.eigh(
                gram
            )
        )


        order = np.argsort(
            eigenvalues
        )[::-1]


        eigenvalues = np.asarray(
            eigenvalues[
                order
            ],
            dtype=np.float64
        )


        eigenvectors = np.asarray(
            eigenvectors[
                :,
                order
            ],
            dtype=np.float64
        )


        # Somente ruído numérico negativo minúsculo.
        eigenvalues = np.clip(
            eigenvalues,
            0.0,
            None
        )


        total_variance = float(
            eigenvalues.sum()
        )


        if total_variance <= 0:

            raise RuntimeError(
                f"{channel}/{month}: variância nula."
            )


        explained = (
            eigenvalues
            /
            total_variance
        )


        cumulative = np.cumsum(
            explained
        )


        month_print = []


        for q in PVE_POLICIES:

            k = k_from_pve(
                cumulative,
                q
            )


            error = reconstruction_error(
                X_test,
                eigenvectors,
                k
            )


            captured_future = (
                1.0
                -
                error ** 2
            )


            fold_rows.append({
                "channel":
                    channel,

                "current_month":
                    str(
                        month
                    ),

                "train_orders":
                    int(
                        len(
                            train_idx
                        )
                    ),

                "test_orders":
                    int(
                        len(
                            test_idx
                        )
                    ),

                "pve_policy":
                    float(
                        q
                    ),

                "k_train":
                    int(
                        k
                    ),

                "train_cumulative_pve":
                    float(
                        cumulative[
                            k - 1
                        ]
                    ),

                "future_relative_error":
                    float(
                        error
                    ),

                "future_variance_captured":
                    float(
                        captured_future
                    ),

                "train_before_test":
                    True,
            })


            month_print.append(
                (
                    q,
                    k,
                    error
                )
            )


        compact = " | ".join(
            [
                (
                    f"q={q:.2f}:"
                    f"K={k}:"
                    f"RE={err:.4f}"
                )
                for q, k, err
                in month_print
            ]
        )


        print(
            str(month),
            "|",
            compact
        )


        del (
            H_train,
            H_test,
            X_train,
            X_test,
            gram,
            eigenvalues,
            eigenvectors
        )


folds = pd.DataFrame(
    fold_rows
)


folds.to_csv(
    OUT_FOLDS,
    index=False
)


# =============================================================================
# 6. POLICY SUMMARY
# =============================================================================

summary_rows = []


for (
    channel,
    q
), g in folds.groupby(
    [
        "channel",
        "pve_policy"
    ],
    sort=True
):

    errors = (
        g[
            "future_relative_error"
        ]
        .astype(float)
    )


    ks = (
        g[
            "k_train"
        ]
        .astype(float)
    )


    summary_rows.append({
        "channel":
            channel,

        "pve_policy":
            float(
                q
            ),

        "temporal_tests":
            int(
                len(g)
            ),

        "mean_future_error":
            float(
                errors.mean()
            ),

        "median_future_error":
            float(
                errors.median()
            ),

        "p95_future_error":
            float(
                errors.quantile(
                    0.95
                )
            ),

        "sd_future_error":
            float(
                errors.std(
                    ddof=1
                )
            ),

        "se_future_error":
            float(
                errors.std(
                    ddof=1
                )
                /
                math.sqrt(
                    len(
                        errors
                    )
                )
            ),

        "mean_future_variance_captured":
            float(
                g[
                    "future_variance_captured"
                ]
                .mean()
            ),

        "mean_k":
            float(
                ks.mean()
            ),

        "median_k":
            float(
                ks.median()
            ),

        "min_k":
            int(
                ks.min()
            ),

        "max_k":
            int(
                ks.max()
            ),

        "sd_k":
            float(
                ks.std(
                    ddof=1
                )
            ),
    })


summary = pd.DataFrame(
    summary_rows
)


# =============================================================================
# 7. MARGINAL COMPLEXITY / ERROR TRADE-OFF
# =============================================================================

summary[
    "previous_policy"
] = np.nan

summary[
    "added_components_vs_previous"
] = np.nan

summary[
    "error_reduction_vs_previous"
] = np.nan


for channel in CHANNELS:

    idx_rows = (
        summary[
            "channel"
        ]
        ==
        channel
    )


    sub = (
        summary.loc[
            idx_rows
        ]
        .sort_values(
            "pve_policy"
        )
        .copy()
    )


    previous_q = (
        sub[
            "pve_policy"
        ]
        .shift(
            1
        )
    )


    added_k = (
        sub[
            "mean_k"
        ]
        -
        sub[
            "mean_k"
        ]
        .shift(
            1
        )
    )


    error_gain = (
        sub[
            "mean_future_error"
        ]
        .shift(
            1
        )
        -
        sub[
            "mean_future_error"
        ]
    )


    summary.loc[
        sub.index,
        "previous_policy"
    ] = previous_q


    summary.loc[
        sub.index,
        "added_components_vs_previous"
    ] = added_k


    summary.loc[
        sub.index,
        "error_reduction_vs_previous"
    ] = error_gain


summary.to_csv(
    OUT_SUMMARY,
    index=False
)


# =============================================================================
# 8. ONE-STANDARD-ERROR CANDIDATE
# =============================================================================

one_se_rows = []


for channel in CHANNELS:

    sub = (
        summary.loc[
            summary[
                "channel"
            ]
            ==
            channel
        ]
        .sort_values(
            "pve_policy"
        )
        .copy()
    )


    best_idx = (
        sub[
            "mean_future_error"
        ]
        .idxmin()
    )


    best = sub.loc[
        best_idx
    ]


    threshold = (
        float(
            best[
                "mean_future_error"
            ]
        )
        +
        float(
            best[
                "se_future_error"
            ]
        )
    )


    eligible = sub.loc[
        sub[
            "mean_future_error"
        ]
        <=
        threshold
    ].copy()


    candidate = (
        eligible.sort_values(
            [
                "mean_k",
                "pve_policy"
            ],
            ascending=[
                True,
                True
            ]
        )
        .iloc[
            0
        ]
    )


    one_se_rows.append({
        "channel":
            channel,

        "best_error_policy":
            float(
                best[
                    "pve_policy"
                ]
            ),

        "best_mean_future_error":
            float(
                best[
                    "mean_future_error"
                ]
            ),

        "best_error_standard_error":
            float(
                best[
                    "se_future_error"
                ]
            ),

        "one_se_error_threshold":
            float(
                threshold
            ),

        "one_se_candidate_policy":
            float(
                candidate[
                    "pve_policy"
                ]
            ),

        "candidate_mean_k":
            float(
                candidate[
                    "mean_k"
                ]
            ),

        "candidate_median_k":
            float(
                candidate[
                    "median_k"
                ]
            ),

        "candidate_min_k":
            int(
                candidate[
                    "min_k"
                ]
            ),

        "candidate_max_k":
            int(
                candidate[
                    "max_k"
                ]
            ),

        "candidate_mean_future_error":
            float(
                candidate[
                    "mean_future_error"
                ]
            ),

        "candidate_only_not_final":
            True,
    })


one_se = pd.DataFrame(
    one_se_rows
)


joint_candidate_q = float(
    one_se[
        "one_se_candidate_policy"
    ]
    .max()
)


one_se[
    "joint_candidate_policy"
] = (
    joint_candidate_q
)


one_se.to_csv(
    OUT_ONE_SE,
    index=False
)


# =============================================================================
# 9. VALIDATION
# =============================================================================

checks = []


add_check(
    checks,
    "decision_pass",
    decision.get(
        "status"
    )
    ==
    "PASS",
    decision.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "raw30_authorized",
    primary.get(
        "representation"
    )
    ==
    "RAW_30D",
    primary.get(
        "representation"
    ),
    "RAW_30D"
)


add_check(
    checks,
    "task_rows",
    len(
        index
    )
    ==
    96470,
    len(
        index
    ),
    96470
)


add_check(
    checks,
    "full_support_30d",
    full_support_count
    ==
    96421,
    full_support_count,
    96421
)


add_check(
    checks,
    "eligible_months_17",
    len(
        eligible_months
    )
    ==
    17,
    len(
        eligible_months
    ),
    17
)


add_check(
    checks,
    "first_month_2017_04",
    str(
        eligible_months[
            0
        ]
    )
    ==
    "2017-04",
    str(
        eligible_months[
            0
        ]
    ),
    "2017-04"
)


add_check(
    checks,
    "last_month_2018_08",
    str(
        eligible_months[
            -1
        ]
    )
    ==
    "2018-08",
    str(
        eligible_months[
            -1
        ]
    ),
    "2018-08"
)


expected_rows = (
    2
    *
    17
    *
    len(
        PVE_POLICIES
    )
)


add_check(
    checks,
    "fold_policy_rows",
    len(
        folds
    )
    ==
    expected_rows,
    len(
        folds
    ),
    expected_rows
)


nonfinite = int(
    (
        ~np.isfinite(
            folds[
                [
                    "k_train",
                    "train_cumulative_pve",
                    "future_relative_error",
                    "future_variance_captured"
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )
    ).sum()
)


add_check(
    checks,
    "all_metrics_finite",
    nonfinite == 0,
    nonfinite,
    0
)


invalid_k = int(
    (
        (
            folds[
                "k_train"
            ]
            <
            1
        )
        |
        (
            folds[
                "k_train"
            ]
            >
            WINDOW
        )
    ).sum()
)


add_check(
    checks,
    "k_within_1_30",
    invalid_k == 0,
    invalid_k,
    0
)


# K deve ser não decrescente quando q cresce.
k_monotonic_failures = 0

# Erro de reconstrução deve ser não crescente quando K cresce.
error_monotonic_failures = 0


for (
    channel,
    month
), g in folds.groupby(
    [
        "channel",
        "current_month"
    ]
):

    g = g.sort_values(
        "pve_policy"
    )


    kvals = (
        g[
            "k_train"
        ]
        .to_numpy()
    )


    evals = (
        g[
            "future_relative_error"
        ]
        .to_numpy()
    )


    if np.any(
        np.diff(
            kvals
        )
        <
        0
    ):
        k_monotonic_failures += 1


    if np.any(
        np.diff(
            evals
        )
        >
        1e-10
    ):
        error_monotonic_failures += 1


add_check(
    checks,
    "k_monotonic_with_pve",
    k_monotonic_failures
    ==
    0,
    k_monotonic_failures,
    0
)


add_check(
    checks,
    "error_nonincreasing_with_pve",
    error_monotonic_failures
    ==
    0,
    error_monotonic_failures,
    0
)


# =============================================================================
# 10. CROSS-CHECK q=0.90 AGAINST D2
# =============================================================================

expected_k90 = {
    x[
        "channel"
    ]:
        float(
            x[
                "raw_k90_median"
            ]
        )
    for x in decision[
        "paired_evidence"
    ][
        "channels"
    ]
}


observed_k90 = (
    folds.loc[
        np.isclose(
            folds[
                "pve_policy"
            ],
            0.90
        )
    ]
    .groupby(
        "channel"
    )[
        "k_train"
    ]
    .median()
    .to_dict()
)


for channel in CHANNELS:

    add_check(
        checks,
        "d2_k90_reproduced_"
        +
        channel,

        float(
            observed_k90[
                channel
            ]
        )
        ==
        float(
            expected_k90[
                channel
            ]
        ),

        observed_k90[
            channel
        ],

        expected_k90[
            channel
        ]
    )


validation = pd.DataFrame(
    checks
)


validation.to_csv(
    OUT_VALIDATION,
    index=False
)


failures = int(
    validation[
        "status"
    ]
    .eq(
        "FAIL"
    )
    .sum()
)


# =============================================================================
# 11. SUMMARY JSON
# =============================================================================

runtime = (
    time.perf_counter()
    -
    start
)


summary_json = {
    "step":
        "MODEL_01_0_C3_4A_TEMPORAL_K_POLICY",

    "status":
        (
            "PASS"
            if failures == 0
            else
            "FAIL"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "representation":
        "RAW_30D",

    "method":
        (
            "EXPANDING_HISTORY_PCA_"
            "PVE_POLICY_OUT_OF_TIME_RECONSTRUCTION"
        ),

    "window_days":
        WINDOW,

    "pve_policies":
        PVE_POLICIES,

    "minimum_train_rows":
        MIN_TRAIN,

    "minimum_test_rows":
        MIN_TEST,

    "temporal_test_months":
        len(
            eligible_months
        ),

    "first_test_month":
        str(
            eligible_months[
                0
            ]
        ),

    "last_test_month":
        str(
            eligible_months[
                -1
            ]
        ),

    "fold_policy_rows":
        len(
            folds
        ),

    "one_se_candidates":
        one_se.to_dict(
            orient="records"
        ),

    "joint_candidate_policy":
        joint_candidate_q,

    "joint_candidate_status":
        "CANDIDATE_ONLY_NOT_FROZEN",

    "target_used":
        False,

    "final_pve_policy_selected":
        False,

    "final_k_selected":
        False,

    "folds_frozen":
        False,

    "classifier_trained":
        False,

    "silver_created":
        False,

    "raw_modified":
        False,

    "validation_failures":
        failures,

    "runtime_seconds":
        runtime,
}


OUT_JSON.write_text(
    json.dumps(
        summary_json,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 12. REPORT
# =============================================================================

lines = [
    "=" * 116,
    "MODEL 01.0-C3.4-A — RAW-30D TEMPORAL COMPONENT-COUNT POLICY",
    "=" * 116,
    "",
    (
        "STATUS                         : "
        +
        summary_json[
            "status"
        ]
    ),
    (
        "REPRESENTATION                 : RAW-30D"
    ),
    "",
    "DESIGN",
    "-" * 116,
    (
        "PCA fit                        : PAST ONLY"
    ),
    (
        "K(q) fit                       : PAST ONLY"
    ),
    (
        "Test                           : FUTURE MONTH"
    ),
    (
        "Target                         : NOT USED"
    ),
    "",
    "PVE POLICIES",
    "-" * 116,
    str(
        PVE_POLICIES
    ),
    "",
    "ONE-SE CANDIDATES",
    "-" * 116,
    one_se.to_string(
        index=False
    ),
    "",
    (
        "Joint candidate policy         : "
        +
        str(
            joint_candidate_q
        )
    ),
    (
        "Joint candidate status         : CANDIDATE ONLY"
    ),
    "",
    "IMPORTANT",
    "-" * 116,
    "Final PVE policy selected          : NO",
    "Final K selected                   : NO",
    "Folds frozen                       : NO",
    "Classifier trained                 : NO",
    "Silver created                     : NO",
    "RAW modified                       : NO",
    "",
    (
        "Validation failures             : "
        +
        str(
            failures
        )
    ),
    "=" * 116,
]


OUT_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 13. PRINT RESULTS
# =============================================================================

print()
print("=" * 116)
print("POLICY SUMMARY")
print("=" * 116)

show_cols = [
    "channel",
    "pve_policy",
    "temporal_tests",
    "mean_future_error",
    "median_future_error",
    "p95_future_error",
    "se_future_error",
    "mean_k",
    "median_k",
    "min_k",
    "max_k",
    "sd_k",
    "mean_future_variance_captured",
]


print(
    summary[
        show_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print("=" * 116)
print("ONE-STANDARD-ERROR CANDIDATES")
print("=" * 116)

print(
    one_se.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print(
    "JOINT CANDIDATE POLICY =",
    joint_candidate_q
)

print(
    "STATUS                 = CANDIDATE ONLY — NOT FROZEN"
)


print()
print("=" * 116)
print("VALIDATION")
print("=" * 116)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 116)
print("RESULTADO C3.4-A")
print("=" * 116)

print(
    "FAILURES                 =",
    failures
)

print(
    "TEMPORAL MONTHS          =",
    len(
        eligible_months
    )
)

print(
    "POLICIES                 =",
    len(
        PVE_POLICIES
    )
)

print(
    "CHANNELS                 =",
    len(
        CHANNELS
    )
)

print(
    "FOLD/POLICY ROWS         =",
    len(
        folds
    )
)

print(
    "JOINT CANDIDATE          =",
    joint_candidate_q
)

print()
print("TARGET USED              = NÃO")
print("FINAL POLICY SELECTED    = NÃO")
print("FINAL K SELECTED         = NÃO")
print("FOLDS FROZEN             = NÃO")
print("CLASSIFIER TRAINED       = NÃO")
print("SILVER CREATED           = NÃO")
print("RAW MODIFIED             = NÃO")

print()
print(
    "RUNTIME SECONDS          =",
    runtime
)


if failures:
    sys.exit(2)


print()
print("[PASS] C3.4-A temporal K-policy sensitivity concluída.")
print("[PASS] Nenhum K final foi escolhido.")
print("[PASS] Parar aqui para revisar o trade-off.")

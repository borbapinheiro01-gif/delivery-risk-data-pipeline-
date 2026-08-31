#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.6-A
FIRST SUPERVISED TEMPORAL EXPERIMENT
FUNCTIONAL K-UTILITY
===============================================================================

PERGUNTA CENTRAL
----------------
A representação funcional RAW-30D acrescenta informação preditiva
sobre atraso além das 13 features ORDER_CORE_V1?

Para cada mês futuro m e cada K:

    K = 0:
        ORDER_CORE_V1

    K > 0:
        ORDER_CORE_V1
        + K PCs de purchase_volume
        + K PCs de purchase_freight

CONTRATO TEMPORAL
-----------------

TRAIN_m = {
    j :
        purchase_j < cutoff_m
        AND
        delivery_j < cutoff_m
}

TEST_m = pedidos comprados no mês m.

PCA:
    fit somente em TRAIN.

Scaler:
    fit somente em TRAIN.

LogisticRegression:
    fit somente em TRAIN.

Nenhum threshold é escolhido.
Nenhum K final é escolhido.

O código salva checkpoints mensais e pode ser retomado.
===============================================================================
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

from sklearn.exceptions import ConvergenceWarning


# =============================================================================
# PATHS
# =============================================================================

ROOT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

MODEL_DIR = (
    ROOT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

PRED_DIR = (
    ROOT
    / "artifacts"
    / "model_01_order_logistic"
    / "supervised_predictions"
)

CONTRACT = (
    ROOT
    / "configs"
    / "model_01_supervised_temporal_contract_v1.json"
)

FREEZE_SUMMARY = (
    MODEL_DIR
    / "06g_supervised_protocol_freeze_summary.json"
)

FREEZE_VALIDATION = (
    MODEL_DIR
    / "06i_supervised_protocol_reproducibility_validation.json"
)

FOLD_AUDIT = (
    MODEL_DIR
    / "06a_supervised_temporal_fold_audit.csv"
)

MATRIX = (
    ROOT
    / "artifacts"
    / "model_01_order_logistic"
    / "pretraining"
    / "ORDER_CORE_V1_AUDIT_MATRIX.csv"
)

ORDER_INDEX = (
    MODEL_DIR
    / "02a_functional_pit_order_index.csv"
)

VOLUME_NPY = (
    ROOT
    / "artifacts"
    / "model_01_order_logistic"
    / "functional_feasibility"
    / "02g_purchase_volume_curve_90d.npy"
)

FREIGHT_NPY = (
    ROOT
    / "artifacts"
    / "model_01_order_logistic"
    / "functional_feasibility"
    / "02h_purchase_freight_curve_90d.npy"
)

ORDERS_RAW = (
    ROOT
    / "data"
    / "raw"
    / "olist"
    / "olist_orders_dataset.csv"
)

FEATURE_CONTRACT = (
    ROOT
    / "configs"
    / "order_core_v1_feature_contract.json"
)


FOLD_METRICS = (
    MODEL_DIR
    / "07a_supervised_k_fold_metrics.csv"
)

K_SUMMARY = (
    MODEL_DIR
    / "07b_supervised_k_summary.csv"
)

PAIRED = (
    MODEL_DIR
    / "07c_supervised_k_paired_vs_k0.csv"
)

VALIDATION = (
    MODEL_DIR
    / "07d_supervised_k_validation.csv"
)

SUMMARY_JSON = (
    MODEL_DIR
    / "07e_supervised_k_summary.json"
)

REPORT = (
    MODEL_DIR
    / "07f_supervised_k_report.txt"
)


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


def recall_at_fraction(
    y,
    p,
    fraction,
):

    y = np.asarray(
        y,
        dtype=int
    )

    p = np.asarray(
        p,
        dtype=float
    )

    positives = int(
        y.sum()
    )

    if positives == 0:
        return np.nan

    n = max(
        1,
        int(
            math.ceil(
                fraction
                *
                len(y)
            )
        )
    )

    order = np.argsort(
        -p,
        kind="mergesort"
    )

    captured = int(
        y[
            order[:n]
        ].sum()
    )

    return (
        captured
        /
        positives
    )


def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# =============================================================================
# START
# =============================================================================

START_TIME = time.perf_counter()

print()
print("=" * 120)
print("MODEL 01.0-C3.6-A — FIRST SUPERVISED TEMPORAL K-UTILITY")
print("=" * 120)


# =============================================================================
# PREREQUISITES
# =============================================================================

required = [
    CONTRACT,
    FREEZE_SUMMARY,
    FREEZE_VALIDATION,
    FOLD_AUDIT,
    MATRIX,
    ORDER_INDEX,
    VOLUME_NPY,
    FREIGHT_NPY,
    ORDERS_RAW,
    FEATURE_CONTRACT,
]

for p in required:

    if not p.exists():
        raise SystemExit(
            f"[FAIL] Arquivo ausente: {p}"
        )

    print(
        f"[PASS] {p.name}"
    )


# =============================================================================
# CONTRACT
# =============================================================================

contract = json.loads(
    CONTRACT.read_text(
        encoding="utf-8"
    )
)

freeze = json.loads(
    FREEZE_SUMMARY.read_text(
        encoding="utf-8"
    )
)

freeze_validation = json.loads(
    FREEZE_VALIDATION.read_text(
        encoding="utf-8"
    )
)


if (
    contract.get("status")
    !=
    "FROZEN_BEFORE_MODEL_PERFORMANCE"
):
    raise RuntimeError(
        "Contrato supervisionado não está congelado."
    )


if freeze.get("status") != "PASS":
    raise RuntimeError(
        "C3.5-B não está PASS."
    )


if (
    freeze_validation.get("status")
    !=
    "PASS"
):
    raise RuntimeError(
        "C3.5-B1 não está PASS."
    )


current_hash = sha256(
    CONTRACT
)

expected_hash = freeze.get(
    "contract_sha256"
)

if (
    expected_hash
    and
    current_hash != expected_hash
):
    raise RuntimeError(
        "Hash do contrato divergiu após o freeze."
    )


K_GRID = [
    int(k)
    for k in contract["k_grid"]
]

MONTHS = [
    str(m)
    for m
    in contract[
        "temporal_protocol"
    ]["test_months"]
]


if len(K_GRID) != 14:
    raise RuntimeError(
        "Esperados 14 valores de K."
    )

if len(MONTHS) != 17:
    raise RuntimeError(
        "Esperados 17 meses."
    )


print()
print("[PASS] Contrato congelado conferido.")
print(
    "K GRID =",
    K_GRID
)
print(
    "MONTHS =",
    len(MONTHS)
)


# =============================================================================
# FEATURE CONTRACT
# =============================================================================

feature_contract = json.loads(
    FEATURE_CONTRACT.read_text(
        encoding="utf-8"
    )
)

FEATURES = [
    x["feature_name"]
    for x
    in feature_contract["features"]
]


if len(FEATURES) != 13:
    raise RuntimeError(
        "ORDER_CORE_V1 deveria possuir 13 features."
    )


# =============================================================================
# LOAD ORDER CORE
# =============================================================================

matrix = pd.read_csv(
    MATRIX,
    parse_dates=[
        "order_purchase_timestamp",
    ],
)


if len(matrix) != 96470:
    raise RuntimeError(
        f"ORDER_CORE possui {len(matrix)} linhas; esperado 96470."
    )


if matrix["order_id"].duplicated().any():
    raise RuntimeError(
        "order_id duplicado na matriz."
    )


# =============================================================================
# LOAD ORDERS
# =============================================================================

orders = pd.read_csv(
    ORDERS_RAW,
    usecols=[
        "order_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ],
    parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ],
)


if orders["order_id"].duplicated().any():
    raise RuntimeError(
        "order_id duplicado em orders RAW."
    )


# =============================================================================
# CURVE ROW INDEX
# =============================================================================

order_index = pd.read_csv(
    ORDER_INDEX
)


if "order_id" not in order_index.columns:
    raise RuntimeError(
        "02a_functional_pit_order_index.csv não possui order_id."
    )


if len(order_index) != 96470:
    raise RuntimeError(
        "Order index não possui 96470 linhas."
    )


if order_index["order_id"].duplicated().any():
    raise RuntimeError(
        "Order index possui order_id duplicado."
    )


order_index = order_index[
    ["order_id"]
].copy()

order_index[
    "curve_row"
] = np.arange(
    len(order_index),
    dtype=np.int64
)


# =============================================================================
# JOIN SUPERVISED FRAME
# =============================================================================

data = (
    matrix
    .merge(
        orders[
            [
                "order_id",
                "order_delivered_customer_date",
            ]
        ],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        order_index,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
)


if (
    data[
        "order_delivered_customer_date"
    ].isna().any()
):
    raise RuntimeError(
        "Pedido supervisionado sem delivery timestamp."
    )


if data["curve_row"].isna().any():
    raise RuntimeError(
        "Pedido supervisionado sem linha da curva."
    )


data[
    "curve_row"
] = (
    data["curve_row"]
    .astype(np.int64)
)


# =============================================================================
# FULL 30-DAY SUPPORT
# =============================================================================

raw_start = orders[
    "order_purchase_timestamp"
].min()

support_start = (
    raw_start
    +
    pd.Timedelta(
        days=30
    )
)


data[
    "full_support_30d"
] = (
    data[
        "order_purchase_timestamp"
    ]
    >=
    support_start
)


full = data.loc[
    data[
        "full_support_30d"
    ]
].copy()


if len(full) != 96421:
    raise RuntimeError(
        f"Full support = {len(full)}; esperado 96421."
    )


full[
    "purchase_month"
] = (
    full[
        "order_purchase_timestamp"
    ]
    .dt
    .to_period("M")
    .astype(str)
)


# =============================================================================
# CURVES — MEMORY MAP
# =============================================================================

volume = np.load(
    VOLUME_NPY,
    mmap_mode="r"
)

freight = np.load(
    FREIGHT_NPY,
    mmap_mode="r"
)


if volume.shape != (96470, 90):
    raise RuntimeError(
        f"Volume shape inesperado: {volume.shape}"
    )

if freight.shape != (96470, 90):
    raise RuntimeError(
        f"Freight shape inesperado: {freight.shape}"
    )


# =============================================================================
# FOLD AUDIT EXPECTATIONS
# =============================================================================

fold_audit = pd.read_csv(
    FOLD_AUDIT
)

fold_audit[
    "test_month"
] = fold_audit[
    "test_month"
].astype(str)

audit_by_month = (
    fold_audit
    .set_index(
        "test_month"
    )
)


# =============================================================================
# RESUME STATE
# =============================================================================

if FOLD_METRICS.exists():

    old_metrics = pd.read_csv(
        FOLD_METRICS
    )

else:

    old_metrics = pd.DataFrame()


complete_months = set()


if not old_metrics.empty:

    for month, g in old_metrics.groupby(
        "test_month"
    ):

        pred_path = (
            PRED_DIR
            /
            f"predictions_{month}.csv.gz"
        )

        observed_k = set(
            pd.to_numeric(
                g["k"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        if (
            observed_k
            ==
            set(K_GRID)
            and
            pred_path.exists()
        ):
            complete_months.add(
                str(month)
            )


print()
print("=" * 120)
print("RESUME STATUS")
print("=" * 120)

print(
    "Meses já completos:",
    len(complete_months),
    "/",
    len(MONTHS)
)


# =============================================================================
# TEMPORAL EXPERIMENT
# =============================================================================

all_metrics = (
    old_metrics.copy()
    if not old_metrics.empty
    else
    pd.DataFrame()
)


fold_guard_failures = 0
probability_failures = 0


for month_no, month in enumerate(
    MONTHS,
    start=1,
):

    if month in complete_months:

        print(
            f"[SKIP] {month} já possui checkpoint completo."
        )

        continue


    month_start_time = time.perf_counter()


    period = pd.Period(
        month,
        freq="M"
    )

    cutoff = period.start_time


    train = full.loc[
        (
            full[
                "order_purchase_timestamp"
            ]
            <
            cutoff
        )
        &
        (
            full[
                "order_delivered_customer_date"
            ]
            <
            cutoff
        )
    ].copy()


    test = full.loc[
        full[
            "purchase_month"
        ]
        ==
        month
    ].copy()


    # -------------------------------------------------------------------------
    # GUARD AGAINST C3.5-A
    # -------------------------------------------------------------------------

    expected = audit_by_month.loc[
        month
    ]

    expected_train = int(
        expected[
            "label_available_train_rows"
        ]
    )

    expected_test = int(
        expected[
            "test_rows"
        ]
    )


    if len(train) != expected_train:

        fold_guard_failures += 1

        raise RuntimeError(
            f"{month}: TRAIN={len(train)}, esperado={expected_train}"
        )


    if len(test) != expected_test:

        fold_guard_failures += 1

        raise RuntimeError(
            f"{month}: TEST={len(test)}, esperado={expected_test}"
        )


    y_train = (
        train[
            "late_delivery_calendar_day"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    y_test = (
        test[
            "late_delivery_calendar_day"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )


    if len(np.unique(y_train)) != 2:
        raise RuntimeError(
            f"{month}: TRAIN não possui duas classes."
        )

    if len(np.unique(y_test)) != 2:
        raise RuntimeError(
            f"{month}: TEST não possui duas classes."
        )


    train_curve_rows = (
        train[
            "curve_row"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    test_curve_rows = (
        test[
            "curve_row"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    # -------------------------------------------------------------------------
    # CORE FEATURES
    # -------------------------------------------------------------------------

    core_train = (
        train[
            FEATURES
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    core_test = (
        test[
            FEATURES
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    # -------------------------------------------------------------------------
    # RAW-30D CURVES
    # -------------------------------------------------------------------------

    volume_train = np.asarray(
        volume[
            train_curve_rows,
            :30
        ],
        dtype=np.float64
    )

    volume_test = np.asarray(
        volume[
            test_curve_rows,
            :30
        ],
        dtype=np.float64
    )

    freight_train = np.asarray(
        freight[
            train_curve_rows,
            :30
        ],
        dtype=np.float64
    )

    freight_test = np.asarray(
        freight[
            test_curve_rows,
            :30
        ],
        dtype=np.float64
    )


    # -------------------------------------------------------------------------
    # PCA FIT ONLY ON TRAIN
    # -------------------------------------------------------------------------

    pca_volume = PCA(
        n_components=30,
        svd_solver="full",
        whiten=False,
    )

    pca_freight = PCA(
        n_components=30,
        svd_solver="full",
        whiten=False,
    )


    volume_train_scores = (
        pca_volume.fit_transform(
            volume_train
        )
    )

    volume_test_scores = (
        pca_volume.transform(
            volume_test
        )
    )


    freight_train_scores = (
        pca_freight.fit_transform(
            freight_train
        )
    )

    freight_test_scores = (
        pca_freight.transform(
            freight_test
        )
    )


    prevalence = float(
        y_test.mean()
    )


    month_metrics = []


    pred_frame = pd.DataFrame({
        "order_id":
            test[
                "order_id"
            ].to_numpy(),

        "test_month":
            month,

        "y_true":
            y_test,
    })


    print()
    print(
        f"[{month_no:02d}/17] {month} | "
        f"train={len(train):,} | "
        f"test={len(test):,} | "
        f"late={int(y_test.sum()):,} "
        f"({100*prevalence:.3f}%)"
    )


    # -------------------------------------------------------------------------
    # K GRID
    # -------------------------------------------------------------------------

    for k in K_GRID:

        if k == 0:

            X_train = core_train
            X_test = core_test

        else:

            X_train = np.hstack([
                core_train,
                volume_train_scores[
                    :,
                    :k
                ],
                freight_train_scores[
                    :,
                    :k
                ],
            ])

            X_test = np.hstack([
                core_test,
                volume_test_scores[
                    :,
                    :k
                ],
                freight_test_scores[
                    :,
                    :k
                ],
            ])


        # ---------------------------------------------------------------------
        # STANDARDIZE FINAL DESIGN — TRAIN ONLY
        # ---------------------------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )


        # ---------------------------------------------------------------------
        # LOGISTIC REGRESSION — FROZEN CONTRACT
        # ---------------------------------------------------------------------

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            tol=1e-4,
            fit_intercept=True,
            class_weight=None,
            random_state=20260830,
        )


        with warnings.catch_warnings(
            record=True
        ) as caught:

            warnings.simplefilter(
                "always"
            )

            clf.fit(
                X_train_scaled,
                y_train
            )


        convergence_warning = int(
            any(
                issubclass(
                    w.category,
                    ConvergenceWarning
                )
                for w
                in caught
            )
        )


        p = (
            clf.predict_proba(
                X_test_scaled
            )[:, 1]
        )


        if (
            not np.isfinite(p).all()
            or
            (p < 0).any()
            or
            (p > 1).any()
        ):

            probability_failures += 1

            raise RuntimeError(
                f"{month} K={k}: probabilidades inválidas."
            )


        # ---------------------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------------------

        ap = float(
            average_precision_score(
                y_test,
                p
            )
        )

        roc = float(
            roc_auc_score(
                y_test,
                p
            )
        )

        ll = float(
            log_loss(
                y_test,
                p,
                labels=[
                    0,
                    1,
                ],
            )
        )

        brier = float(
            brier_score_loss(
                y_test,
                p
            )
        )

        ap_lift = float(
            ap
            /
            prevalence
        )

        r05 = float(
            recall_at_fraction(
                y_test,
                p,
                0.05
            )
        )

        r10 = float(
            recall_at_fraction(
                y_test,
                p,
                0.10
            )
        )

        r20 = float(
            recall_at_fraction(
                y_test,
                p,
                0.20
            )
        )


        row = {
            "test_month":
                month,

            "fit_cutoff":
                cutoff,

            "k":
                k,

            "functional_dimensions":
                2 * k,

            "total_model_dimensions":
                len(FEATURES)
                +
                (2 * k),

            "train_rows":
                len(train),

            "train_positive":
                int(
                    y_train.sum()
                ),

            "train_positive_pct":
                float(
                    100
                    *
                    y_train.mean()
                ),

            "test_rows":
                len(test),

            "test_positive":
                int(
                    y_test.sum()
                ),

            "test_prevalence":
                prevalence,

            "average_precision":
                ap,

            "ap_lift_over_prevalence":
                ap_lift,

            "roc_auc":
                roc,

            "log_loss":
                ll,

            "brier_score":
                brier,

            "recall_at_05pct":
                r05,

            "recall_at_10pct":
                r10,

            "recall_at_20pct":
                r20,

            "convergence_warning":
                convergence_warning,

            "n_iter":
                int(
                    np.max(
                        clf.n_iter_
                    )
                ),

            "coef_finite":
                bool(
                    np.isfinite(
                        clf.coef_
                    ).all()
                ),
        }


        month_metrics.append(
            row
        )


        pred_frame[
            f"p_k{k}"
        ] = p


        print(
            f"   K={k:02d} | "
            f"AP={ap:.6f} | "
            f"lift={ap_lift:.3f} | "
            f"ROC={roc:.6f} | "
            f"R@10={r10:.4f} | "
            f"iter={row['n_iter']}"
        )


    # -------------------------------------------------------------------------
    # MONTH CHECKPOINT
    # -------------------------------------------------------------------------

    month_metrics_df = pd.DataFrame(
        month_metrics
    )


    if not all_metrics.empty:

        all_metrics = all_metrics.loc[
            all_metrics[
                "test_month"
            ].astype(str)
            !=
            month
        ].copy()


    all_metrics = pd.concat(
        [
            all_metrics,
            month_metrics_df,
        ],
        ignore_index=True,
    )


    all_metrics = (
        all_metrics
        .sort_values(
            [
                "test_month",
                "k",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    atomic_csv(
        all_metrics,
        FOLD_METRICS
    )


    pred_path = (
        PRED_DIR
        /
        f"predictions_{month}.csv.gz"
    )


    pred_frame.to_csv(
        pred_path,
        index=False,
        compression="gzip",
    )


    elapsed_month = (
        time.perf_counter()
        -
        month_start_time
    )


    print(
        f"[CHECKPOINT] {month} salvo | "
        f"{elapsed_month:.2f}s"
    )


    # release large arrays before next month
    del train
    del test

    del core_train
    del core_test

    del volume_train
    del volume_test

    del freight_train
    del freight_test

    del volume_train_scores
    del volume_test_scores

    del freight_train_scores
    del freight_test_scores


# =============================================================================
# RELOAD CANONICAL METRICS
# =============================================================================

metrics = pd.read_csv(
    FOLD_METRICS
)


metrics[
    "test_month"
] = metrics[
    "test_month"
].astype(str)

metrics[
    "k"
] = pd.to_numeric(
    metrics["k"]
).astype(int)


# =============================================================================
# PAIRED COMPARISON AGAINST K=0
# =============================================================================

baseline = (
    metrics.loc[
        metrics["k"] == 0,
        [
            "test_month",
            "average_precision",
        ]
    ]
    .rename(
        columns={
            "average_precision":
                "baseline_k0_average_precision"
        }
    )
)


paired = metrics.merge(
    baseline,
    on="test_month",
    how="left",
    validate="many_to_one",
)


paired[
    "delta_ap_vs_k0"
] = (
    paired[
        "average_precision"
    ]
    -
    paired[
        "baseline_k0_average_precision"
    ]
)


TIE_TOL = 1e-12


paired[
    "comparison_vs_k0"
] = np.select(
    [
        paired[
            "delta_ap_vs_k0"
        ]
        >
        TIE_TOL,

        paired[
            "delta_ap_vs_k0"
        ]
        <
        -TIE_TOL,
    ],
    [
        "BETTER",
        "WORSE",
    ],
    default="TIE",
)


atomic_csv(
    paired,
    PAIRED
)


# =============================================================================
# K SUMMARY
# =============================================================================

summary_rows = []


for k in K_GRID:

    g = metrics.loc[
        metrics["k"] == k
    ].copy()

    p = paired.loc[
        paired["k"] == k
    ].copy()


    summary_rows.append({
        "k":
            k,

        "functional_dimensions":
            2 * k,

        "total_model_dimensions":
            len(FEATURES)
            +
            2 * k,

        "temporal_tests":
            len(g),

        "mean_average_precision":
            float(
                g[
                    "average_precision"
                ].mean()
            ),

        "median_average_precision":
            float(
                g[
                    "average_precision"
                ].median()
            ),

        "std_average_precision":
            float(
                g[
                    "average_precision"
                ].std(
                    ddof=1
                )
            ),

        "mean_ap_lift":
            float(
                g[
                    "ap_lift_over_prevalence"
                ].mean()
            ),

        "mean_roc_auc":
            float(
                g[
                    "roc_auc"
                ].mean()
            ),

        "mean_log_loss":
            float(
                g[
                    "log_loss"
                ].mean()
            ),

        "mean_brier_score":
            float(
                g[
                    "brier_score"
                ].mean()
            ),

        "mean_recall_at_05pct":
            float(
                g[
                    "recall_at_05pct"
                ].mean()
            ),

        "mean_recall_at_10pct":
            float(
                g[
                    "recall_at_10pct"
                ].mean()
            ),

        "mean_recall_at_20pct":
            float(
                g[
                    "recall_at_20pct"
                ].mean()
            ),

        "mean_delta_ap_vs_k0":
            float(
                p[
                    "delta_ap_vs_k0"
                ].mean()
            ),

        "median_delta_ap_vs_k0":
            float(
                p[
                    "delta_ap_vs_k0"
                ].median()
            ),

        "months_better_than_k0":
            int(
                p[
                    "comparison_vs_k0"
                ].eq(
                    "BETTER"
                ).sum()
            ),

        "months_tied_with_k0":
            int(
                p[
                    "comparison_vs_k0"
                ].eq(
                    "TIE"
                ).sum()
            ),

        "months_worse_than_k0":
            int(
                p[
                    "comparison_vs_k0"
                ].eq(
                    "WORSE"
                ).sum()
            ),

        "convergence_warnings":
            int(
                g[
                    "convergence_warning"
                ].sum()
            ),

        "max_iterations_used":
            int(
                g[
                    "n_iter"
                ].max()
            ),
    })


k_summary = pd.DataFrame(
    summary_rows
)


k_summary[
    "descriptive_mean_ap_rank"
] = (
    k_summary[
        "mean_average_precision"
    ]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


k_summary = (
    k_summary
    .sort_values(
        "k"
    )
    .reset_index(
        drop=True
    )
)


atomic_csv(
    k_summary,
    K_SUMMARY
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


numeric_metric_columns = [
    "average_precision",
    "ap_lift_over_prevalence",
    "roc_auc",
    "log_loss",
    "brier_score",
    "recall_at_05pct",
    "recall_at_10pct",
    "recall_at_20pct",
]


metric_array = (
    metrics[
        numeric_metric_columns
    ]
    .to_numpy(
        dtype=float
    )
)


per_month_counts = (
    metrics
    .groupby(
        "test_month"
    )["k"]
    .nunique()
)


per_k_counts = (
    metrics
    .groupby(
        "k"
    )["test_month"]
    .nunique()
)


pred_files = list(
    PRED_DIR.glob(
        "predictions_*.csv.gz"
    )
)


add(
    "frozen_contract_hash",
    current_hash
    ==
    expected_hash,
    current_hash,
    expected_hash,
)

add(
    "fold_metric_rows",
    len(metrics)
    ==
    238,
    len(metrics),
    238,
)

add(
    "seventeen_months",
    metrics[
        "test_month"
    ].nunique()
    ==
    17,
    metrics[
        "test_month"
    ].nunique(),
    17,
)

add(
    "fourteen_k_values",
    metrics[
        "k"
    ].nunique()
    ==
    14,
    metrics[
        "k"
    ].nunique(),
    14,
)

add(
    "each_month_has_14_k",
    (
        per_month_counts
        ==
        14
    ).all(),
    int(
        (
            per_month_counts
            !=
            14
        ).sum()
    ),
    0,
)

add(
    "each_k_has_17_months",
    (
        per_k_counts
        ==
        17
    ).all(),
    int(
        (
            per_k_counts
            !=
            17
        ).sum()
    ),
    0,
)

add(
    "all_metrics_finite",
    np.isfinite(
        metric_array
    ).all(),
    int(
        (
            ~np.isfinite(
                metric_array
            )
        ).sum()
    ),
    0,
)

add(
    "probability_failures",
    probability_failures
    ==
    0,
    probability_failures,
    0,
)

add(
    "fold_guard_failures",
    fold_guard_failures
    ==
    0,
    fold_guard_failures,
    0,
)

add(
    "all_coefficients_finite",
    metrics[
        "coef_finite"
    ].astype(bool).all(),
    int(
        (
            ~metrics[
                "coef_finite"
            ].astype(bool)
        ).sum()
    ),
    0,
)

add(
    "zero_convergence_warnings",
    int(
        metrics[
            "convergence_warning"
        ].sum()
    )
    ==
    0,
    int(
        metrics[
            "convergence_warning"
        ].sum()
    ),
    0,
)

add(
    "prediction_month_files",
    len(pred_files)
    ==
    17,
    len(pred_files),
    17,
)

add(
    "k_final_not_selected",
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


validation = pd.DataFrame(
    checks
)

atomic_csv(
    validation,
    VALIDATION
)


failures = int(
    validation[
        "status"
    ].eq(
        "FAIL"
    ).sum()
)


# =============================================================================
# DESCRIPTIVE WINNER — NOT A FINAL SELECTION
# =============================================================================

descriptive_best_row = (
    k_summary
    .sort_values(
        [
            "mean_average_precision",
            "k",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .iloc[0]
)


descriptive_best_k = int(
    descriptive_best_row[
        "k"
    ]
)


baseline_row = (
    k_summary.loc[
        k_summary["k"] == 0
    ]
    .iloc[0]
)


# =============================================================================
# SUMMARY JSON
# =============================================================================

runtime = (
    time.perf_counter()
    -
    START_TIME
)


summary_payload = {
    "step":
        "MODEL_01_0_C3_6A_SUPERVISED_K_UTILITY",

    "status":
        (
            "PASS"
            if failures == 0
            else "FAIL"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "representation":
        "RAW_30D",

    "classifier":
        "LOGISTIC_REGRESSION_L2_C1",

    "temporal_design":
        "EXPANDING_HISTORY_MONTHLY_OUT_OF_TIME",

    "training_purchase_rule":
        "purchase < fit_cutoff",

    "training_label_rule":
        "delivery < fit_cutoff",

    "pca_fit_scope":
        "TRAIN_ONLY",

    "scaler_fit_scope":
        "TRAIN_ONLY",

    "temporal_months":
        17,

    "k_grid":
        K_GRID,

    "expected_model_fits":
        238,

    "actual_model_fits":
        len(metrics),

    "primary_metric":
        "UNWEIGHTED_MEAN_MONTHLY_AVERAGE_PRECISION",

    "baseline_k":
        0,

    "baseline_mean_average_precision":
        float(
            baseline_row[
                "mean_average_precision"
            ]
        ),

    "descriptive_highest_mean_ap_k":
        descriptive_best_k,

    "descriptive_highest_mean_ap":
        float(
            descriptive_best_row[
                "mean_average_precision"
            ]
        ),

    "descriptive_only_not_final_selection":
        True,

    "final_k_selected":
        False,

    "threshold_selected":
        False,

    "production_claim_allowed":
        False,

    "predictions_saved":
        True,

    "prediction_month_files":
        len(pred_files),

    "contract_sha256":
        current_hash,

    "raw_modified":
        False,

    "silver_created":
        False,

    "validation_failures":
        failures,

    "runtime_seconds":
        runtime,
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary_payload,
        indent=4,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =============================================================================
# REPORT
# =============================================================================

top5 = (
    k_summary
    .sort_values(
        "mean_average_precision",
        ascending=False
    )
    .head(5)
)


report_lines = [
    "=" * 120,
    "MODEL 01.0-C3.6-A — FIRST SUPERVISED K-UTILITY EXPERIMENT",
    "=" * 120,
    "",
    f"STATUS                    : {summary_payload['status']}",
    "",
    "DESIGN",
    "-" * 120,
    "Representation            : RAW-30D",
    "Classifier                : Logistic Regression / L2 / C=1",
    "PCA                       : TRAIN ONLY",
    "Scaler                    : TRAIN ONLY",
    "Temporal months           : 17",
    "K candidates              : 14",
    "Model fits                : 238",
    "",
    "PRIMARY METRIC",
    "-" * 120,
    "Unweighted mean monthly Average Precision",
    "",
    "BASELINE",
    "-" * 120,
    f"K=0 mean AP               : {summary_payload['baseline_mean_average_precision']:.10f}",
    "",
    "TOP 5 BY MEAN AP — DESCRIPTIVE ONLY",
    "-" * 120,
    top5.to_string(index=False),
    "",
    "IMPORTANT",
    "-" * 120,
    "Final K selected           : NO",
    "Threshold selected         : NO",
    "Production claim           : NO",
    "Silver created             : NO",
    "RAW modified               : NO",
    "",
    f"Validation failures        : {failures}",
    f"Runtime seconds            : {runtime:.3f}",
    "=" * 120,
]


REPORT.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 120)
print("K SUMMARY")
print("=" * 120)

print(
    k_summary[
        [
            "k",
            "functional_dimensions",
            "mean_average_precision",
            "median_average_precision",
            "mean_delta_ap_vs_k0",
            "months_better_than_k0",
            "months_tied_with_k0",
            "months_worse_than_k0",
            "mean_roc_auc",
            "mean_recall_at_10pct",
            "mean_log_loss",
            "mean_brier_score",
            "descriptive_mean_ap_rank",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


print()
print("=" * 120)
print("VALIDATION")
print("=" * 120)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("RESULTADO C3.6-A")
print("=" * 120)

print(
    "STATUS                      =",
    summary_payload[
        "status"
    ]
)

print(
    "MODEL FITS                  =",
    len(metrics)
)

print(
    "BASELINE K=0 MEAN AP        =",
    f"{summary_payload['baseline_mean_average_precision']:.8f}"
)

print(
    "HIGHEST MEAN AP K           =",
    descriptive_best_k,
    "(DESCRIPTIVE ONLY)"
)

print(
    "HIGHEST MEAN AP             =",
    f"{summary_payload['descriptive_highest_mean_ap']:.8f}"
)

print()
print(
    "K FINAL                     = NÃO"
)

print(
    "THRESHOLD                   = NÃO"
)

print(
    "PRODUCTION CLAIM            = NÃO"
)

print(
    "SILVER                      = NÃO CRIADA"
)

print(
    "RAW                         = INTACTO"
)

print(
    "VALIDATION FAILURES         =",
    failures
)

print(
    "RUNTIME                     =",
    f"{runtime:.2f}s"
)


if failures:
    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.6-A concluído."
)

print(
    "[PASS] Primeiro experimento supervisionado concluído."
)

print(
    "[PASS] Nenhum K final foi escolhido automaticamente."
)


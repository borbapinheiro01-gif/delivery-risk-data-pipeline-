#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.4-B
TEMPORAL BLOCK BI-CROSS-VALIDATION OF LOW-RANK STRUCTURE
===============================================================================

OBJETIVO
--------
Avaliar o rank K da representação funcional RAW-30D por
Bi-Cross-Validation adaptada à estrutura temporal.

Para cada mês futuro m:

    TRAIN = pedidos de meses anteriores
    TEST  = pedidos do mês m

Logo:

    max(time_train) < min(time_test)

Em cada TEST, também omitimos um bloco de lags.

Particionamos a matriz centrada:

             colunas omitidas   colunas mantidas

TEST                A                 B
TRAIN               C                 D

e usamos a predição BCV:

    A_hat(K) = B D_K^+ C

onde D_K^+ é a pseudoinversa da aproximação de posto K
da matriz D.

ERRO:

    BCV_RE(K) =
        ||A - A_hat(K)||_F
        ------------------
              ||A||_F

IMPORTANTE
----------
Este passo:

- NÃO usa target;
- NÃO escolhe K final;
- NÃO congela folds finais;
- NÃO cria feature para classificador;
- NÃO treina modelo;
- NÃO cria Silver;
- NÃO modifica RAW.

Três sensibilidades de coluna:

    3 lags omitidos  -> 10 blocos -> máximo teórico K=27
    6 lags omitidos  ->  5 blocos -> máximo teórico K=24
   15 lags omitidos  ->  2 blocos -> máximo teórico K=15

Os blocos de lag são contíguos e cobrem os 30 lags exatamente
uma vez em cada esquema.

IMPLEMENTAÇÃO
-------------
Para evitar reprocessamento pesado, o cálculo usa sufficient statistics:

    n
    sum(H)
    H^T H

por mês.

Assim, a BCV trabalha praticamente apenas com matrizes <= 30 x 30.
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

DIR = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

ART = (
    PROJECT
    / "artifacts"
    / "model_01_order_logistic"
    / "functional_feasibility"
)

C34A1 = (
    DIR
    / "04j_rank_identifiability_summary.json"
)

ORDER_INDEX = (
    DIR
    / "02a_functional_pit_order_index.csv"
)

VOLUME = (
    ART
    / "02g_purchase_volume_curve_90d.npy"
)

FREIGHT = (
    ART
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
    DIR
    / "04m_temporal_bcv_fold_rank_errors.csv"
)

OUT_PROFILE = (
    DIR
    / "04n_temporal_bcv_rank_profiles.csv"
)

OUT_CANDIDATES = (
    DIR
    / "04o_temporal_bcv_scheme_candidates.csv"
)

OUT_MONTHLY = (
    DIR
    / "04p_temporal_bcv_monthly_best_rank.csv"
)

OUT_VALIDATION = (
    DIR
    / "04q_temporal_bcv_validation.csv"
)

OUT_JSON = (
    DIR
    / "04r_temporal_bcv_summary.json"
)

OUT_REPORT = (
    DIR
    / "04s_temporal_bcv_report.txt"
)


# =============================================================================
# CONFIG
# =============================================================================

WINDOW = 30

MIN_TRAIN = 5000
MIN_TEST = 500

CHANNELS = {
    "purchase_volume":
        VOLUME,

    "purchase_freight":
        FREIGHT,
}

SCHEMES = {
    "HOLDOUT_03_CONTIGUOUS":
        3,

    "HOLDOUT_06_CONTIGUOUS":
        6,

    "HOLDOUT_15_CONTIGUOUS":
        15,
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


def add_check(
    rows,
    name,
    ok,
    observed,
    expected
):

    rows.append({
        "check":
            name,

        "status":
            "PASS"
            if bool(ok)
            else
            "FAIL",

        "observed":
            str(observed),

        "expected":
            str(expected),
    })


def make_blocks(
    width
):

    if WINDOW % width != 0:

        raise ValueError(
            f"30 não divisível por {width}"
        )

    blocks = []

    for start in range(
        0,
        WINDOW,
        width
    ):

        hold = np.arange(
            start,
            start + width,
            dtype=int
        )

        keep = np.array(
            [
                j
                for j in range(WINDOW)
                if j not in set(
                    hold.tolist()
                )
            ],
            dtype=int
        )

        blocks.append(
            (
                hold,
                keep
            )
        )

    return blocks


def centered_train_gram(
    n,
    total,
    cross
):

    mu = (
        total
        /
        n
    )

    gram = (
        cross
        -
        n
        *
        np.outer(
            mu,
            mu
        )
    )

    gram = (
        gram
        +
        gram.T
    ) / 2.0

    return (
        mu,
        gram
    )


def centered_test_gram(
    n,
    total,
    cross,
    train_mu
):

    gram = (
        cross
        -
        np.outer(
            train_mu,
            total
        )
        -
        np.outer(
            total,
            train_mu
        )
        +
        n
        *
        np.outer(
            train_mu,
            train_mu
        )
    )

    gram = (
        gram
        +
        gram.T
    ) / 2.0

    return gram


def eigen_system(
    gram
):

    values, vectors = (
        np.linalg.eigh(
            gram
        )
    )

    order = np.argsort(
        values
    )[::-1]

    values = np.asarray(
        values[
            order
        ],
        dtype=np.float64
    )

    vectors = np.asarray(
        vectors[
            :,
            order
        ],
        dtype=np.float64
    )

    largest = (
        max(
            float(
                values[0]
            ),
            0.0
        )
        if len(values)
        else
        0.0
    )

    tol = (
        max(
            gram.shape
        )
        *
        np.finfo(
            np.float64
        ).eps
        *
        max(
            largest,
            1.0
        )
    )

    rank = int(
        np.sum(
            values
            >
            tol
        )
    )

    return (
        values,
        vectors,
        tol,
        rank
    )


def bcv_fold_path(
    train_gram,
    test_gram,
    hold,
    keep
):

    # -------------------------------------------------------------
    # Owen-Perry block notation:
    #
    # A = future rows / held-out lags
    # B = future rows / retained lags
    # C = past rows   / held-out lags
    # D = past rows   / retained lags
    #
    # A_hat = B D_k^+ C
    # -------------------------------------------------------------

    G_DD = train_gram[
        np.ix_(
            keep,
            keep
        )
    ]

    G_DC = train_gram[
        np.ix_(
            keep,
            hold
        )
    ]

    G_BB = test_gram[
        np.ix_(
            keep,
            keep
        )
    ]

    G_BA = test_gram[
        np.ix_(
            keep,
            hold
        )
    ]

    G_AA = test_gram[
        np.ix_(
            hold,
            hold
        )
    ]


    energy = float(
        np.trace(
            G_AA
        )
    )


    if (
        not np.isfinite(
            energy
        )
        or
        energy <= 0
    ):

        raise RuntimeError(
            "Energia não positiva no bloco A."
        )


    eig,
    V,
    tol,
    rank = eigen_system(
        G_DD
    )


    if rank <= 0:

        raise RuntimeError(
            "Matriz D possui rank numérico zero."
        )


    # -------------------------------------------------------------
    # D_k^+ C pode ser escrito sem materializar D:
    #
    # D = U S V^T
    #
    # D_k^+ C
    # =
    # V_k S_k^{-1} U_k^T C
    #
    # e
    #
    # U_k = D V_k S_k^{-1}
    #
    # portanto
    #
    # D_k^+ C
    # =
    # V_k diag(1 / s_k^2) V_k^T D^T C
    #
    # =
    # V_k diag(1 / lambda_k) V_k^T G_DC
    # -------------------------------------------------------------

    coef = np.zeros(
        (
            len(keep),
            len(hold)
        ),
        dtype=np.float64
    )


    result = []

    large_negative_sse = 0


    for k in range(
        0,
        rank + 1
    ):

        if k > 0:

            j = (
                k
                -
                1
            )

            v = V[
                :,
                j
            ]

            lam = float(
                eig[
                    j
                ]
            )

            if lam <= tol:

                break


            right = (
                v
                @
                G_DC
            ) / lam


            coef += np.outer(
                v,
                right
            )


        # ---------------------------------------------------------
        # ||A - B M||_F^2
        #
        # =
        # ||A||_F^2
        # - 2 tr(M^T B^T A)
        # + tr(M^T B^T B M)
        # ---------------------------------------------------------

        sse_raw = float(
            energy
            -
            2.0
            *
            np.sum(
                coef
                *
                G_BA
            )
            +
            np.sum(
                coef
                *
                (
                    G_BB
                    @
                    coef
                )
            )
        )


        tolerance_sse = (
            1e-9
            *
            max(
                energy,
                1.0
            )
        )


        if sse_raw < -tolerance_sse:

            large_negative_sse += 1


        sse = max(
            sse_raw,
            0.0
        )


        relative_squared_error = (
            sse
            /
            energy
        )


        relative_error = float(
            math.sqrt(
                relative_squared_error
            )
        )


        result.append({
            "k":
                int(
                    k
                ),

            "rank_d":
                int(
                    rank
                ),

            "svd_tolerance":
                float(
                    tol
                ),

            "heldout_energy":
                energy,

            "sse":
                sse,

            "sse_raw":
                sse_raw,

            "relative_squared_error":
                float(
                    relative_squared_error
                ),

            "relative_error":
                relative_error,
        })


    return (
        result,
        large_negative_sse
    )


# =============================================================================
# INDEPENDENT IMPLEMENTATION CHECKS
# =============================================================================

def synthetic_checks():

    rng = np.random.default_rng(
        20260830
    )


    # -------------------------------------------------------------
    # Teste 1:
    # matriz exatamente rank-3.
    # BCV deve ser autoconsistente em K=3.
    # -------------------------------------------------------------

    L = rng.normal(
        size=(
            24,
            3
        )
    )

    R = rng.normal(
        size=(
            3,
            12
        )
    )

    X = (
        L
        @
        R
    )


    A = X[
        :6,
        :4
    ]

    B = X[
        :6,
        4:
    ]

    C = X[
        6:,
        :4
    ]

    D = X[
        6:,
        4:
    ]


    U, s, VT = np.linalg.svd(
        D,
        full_matrices=False
    )


    k = 3


    Dplus_k = (
        VT[
            :k,
            :
        ].T
        @
        np.diag(
            1.0
            /
            s[
                :k
            ]
        )
        @
        U[
            :,
            :k
        ].T
    )


    Ahat = (
        B
        @
        Dplus_k
        @
        C
    )


    self_consistency = float(
        np.linalg.norm(
            A
            -
            Ahat,
            ord="fro"
        )
        /
        np.linalg.norm(
            A,
            ord="fro"
        )
    )


    # -------------------------------------------------------------
    # Teste 2:
    # conferir álgebra via Gram contra pseudoinversa direta.
    # -------------------------------------------------------------

    X2 = rng.normal(
        size=(
            40,
            10
        )
    )


    A2 = X2[
        :8,
        :3
    ]

    B2 = X2[
        :8,
        3:
    ]

    C2 = X2[
        8:,
        :3
    ]

    D2 = X2[
        8:,
        3:
    ]


    k2 = 4


    U2, s2, VT2 = np.linalg.svd(
        D2,
        full_matrices=False
    )


    direct_coef = (
        VT2[
            :k2,
            :
        ].T
        @
        np.diag(
            1.0
            /
            s2[
                :k2
            ]
        )
        @
        U2[
            :,
            :k2
        ].T
        @
        C2
    )


    gram = (
        D2.T
        @
        D2
    )


    cross = (
        D2.T
        @
        C2
    )


    eig, V, _, _ = eigen_system(
        gram
    )


    gram_coef = np.zeros_like(
        direct_coef
    )


    for j in range(
        k2
    ):

        v = V[
            :,
            j
        ]

        gram_coef += np.outer(
            v,
            (
                v
                @
                cross
            )
            /
            eig[
                j
            ]
        )


    coef_relative_difference = float(
        np.linalg.norm(
            direct_coef
            -
            gram_coef,
            ord="fro"
        )
        /
        np.linalg.norm(
            direct_coef,
            ord="fro"
        )
    )


    direct_pred = (
        B2
        @
        direct_coef
    )

    gram_pred = (
        B2
        @
        gram_coef
    )


    prediction_relative_difference = float(
        np.linalg.norm(
            direct_pred
            -
            gram_pred,
            ord="fro"
        )
        /
        np.linalg.norm(
            direct_pred,
            ord="fro"
        )
    )


    return {
        "rank3_self_consistency_relative_error":
            self_consistency,

        "gram_vs_direct_coef_relative_difference":
            coef_relative_difference,

        "gram_vs_direct_prediction_relative_difference":
            prediction_relative_difference,
    }


# =============================================================================
# START
# =============================================================================

start_runtime = time.perf_counter()


print()
print("=" * 118)
print("MODEL 01.0-C3.4-B — TEMPORAL BLOCK BI-CROSS-VALIDATION")
print("=" * 118)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required = [
    C34A1,
    ORDER_INDEX,
    VOLUME,
    FREIGHT,
    RAW_ORDERS,
]


for p in required:

    if not p.is_file():

        print(
            "[FAIL] Ausente:",
            p
        )

        sys.exit(2)

    print(
        "[PASS]",
        p.name
    )


c34a1 = read_json(
    C34A1
)


if (
    c34a1.get(
        "status"
    )
    !=
    "PASS"
):

    print(
        "[FAIL] C3.4-A.1 não está PASS."
    )

    sys.exit(2)


if (
    c34a1.get(
        "criterion_classification"
    )
    !=
    "NON_IDENTIFYING_FOR_INTRINSIC_RANK"
):

    print(
        "[FAIL] Classificação de C3.4-A.1 inesperada."
    )

    sys.exit(2)


print(
    "[PASS] C3.4-A.1 autoriza mudança de critério."
)


# =============================================================================
# 2. IMPLEMENTATION SELF-CHECK
# =============================================================================

synthetic = synthetic_checks()


print()
print("=" * 118)
print("IMPLEMENTATION SELF-CHECK")
print("=" * 118)


for key, value in synthetic.items():

    print(
        f"{key:50s}: "
        f"{value:.12e}"
    )


# =============================================================================
# 3. ORDER INDEX
# =============================================================================

index = pd.read_csv(
    ORDER_INDEX,
    low_memory=False
)


time_col = None


for candidate in [
    "order_purchase_timestamp",
    "prediction_time",
]:

    if candidate in index.columns:

        time_col = candidate
        break


if time_col is None:

    raise RuntimeError(
        "Timestamp de previsão ausente."
    )


index[
    time_col
] = pd.to_datetime(
    index[
        time_col
    ],
    errors="coerce"
)


if index[
    time_col
].isna().any():

    raise RuntimeError(
        "Timestamp inválido no índice."
    )


if len(
    index
) != 96470:

    raise RuntimeError(
        f"Task rows inesperadas: {len(index)}"
    )


index[
    "purchase_month"
] = (
    index[
        time_col
    ]
    .dt
    .to_period(
        "M"
    )
)


# =============================================================================
# 4. FULL HISTORICAL SUPPORT 30D
# =============================================================================

raw_time = pd.read_csv(
    RAW_ORDERS,
    usecols=[
        "order_purchase_timestamp"
    ],
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
    index[
        time_col
    ]
    >=
    support_start
)


full_support_rows = int(
    index[
        "full_support_30d"
    ].sum()
)


print()
print("=" * 118)
print("SUPPORT / TEMPORAL DESIGN")
print("=" * 118)

print(
    "RAW start           :",
    raw_start
)

print(
    "30D support start   :",
    support_start
)

print(
    "Full support rows   :",
    full_support_rows
)


# =============================================================================
# 5. ELIGIBLE TEMPORAL MONTHS
# =============================================================================

months = sorted(
    index[
        "purchase_month"
    ]
    .dropna()
    .unique()
)


eligible_months = []

temporal_boundary_failures = 0


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


        train_max = (
            index.loc[
                train_mask,
                time_col
            ]
            .max()
        )


        test_min = (
            index.loc[
                test_mask,
                time_col
            ]
            .min()
        )


        if not (
            train_max
            <
            test_min
        ):

            temporal_boundary_failures += 1


print(
    "Eligible months     :",
    len(
        eligible_months
    )
)

print(
    "First future month  :",
    eligible_months[
        0
    ]
)

print(
    "Last future month   :",
    eligible_months[
        -1
    ]
)


# =============================================================================
# 6. COLUMN BLOCK DESIGN
# =============================================================================

scheme_blocks = {}


column_coverage_failures = 0


for scheme, width in SCHEMES.items():

    blocks = make_blocks(
        width
    )

    scheme_blocks[
        scheme
    ] = blocks


    coverage = np.zeros(
        WINDOW,
        dtype=int
    )


    for hold, keep in blocks:

        coverage[
            hold
        ] += 1


        if len(
            np.intersect1d(
                hold,
                keep
            )
        ) != 0:

            column_coverage_failures += 1


        if (
            len(
                hold
            )
            +
            len(
                keep
            )
            !=
            WINDOW
        ):

            column_coverage_failures += 1


    if not np.all(
        coverage
        ==
        1
    ):

        column_coverage_failures += 1


print()
print("=" * 118)
print("COLUMN HOLDOUT SCHEMES")
print("=" * 118)


for scheme, width in SCHEMES.items():

    print(
        f"{scheme:28s}"
        f" width={width:2d}"
        f" blocks={len(scheme_blocks[scheme]):2d}"
        f" retained={WINDOW-width:2d}"
    )


# =============================================================================
# 7. BCV
# =============================================================================

fold_rows = []

large_negative_sse_total = 0

numerical_rank_failures = 0


for channel, path in CHANNELS.items():

    print()
    print("=" * 118)
    print(
        "CHANNEL:",
        channel
    )
    print("=" * 118)


    curves = np.load(
        path,
        mmap_mode="r"
    )


    if curves.shape != (
        len(
            index
        ),
        90
    ):

        raise RuntimeError(
            f"{channel}: shape inesperado {curves.shape}"
        )


    # -----------------------------------------------------------------
    # Monthly sufficient statistics.
    # Cada mês é processado apenas uma vez.
    # -----------------------------------------------------------------

    stats = {}


    for month in months:

        mask = (
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


        idx = np.flatnonzero(
            mask.to_numpy()
        )


        if len(
            idx
        ) == 0:

            continue


        H = np.asarray(
            curves[
                idx,
                :WINDOW
            ],
            dtype=np.float64
        )


        stats[
            month
        ] = {
            "n":
                int(
                    len(
                        idx
                    )
                ),

            "sum":
                H.sum(
                    axis=0,
                    dtype=np.float64
                ),

            "cross":
                (
                    H.T
                    @
                    H
                ),
        }


        del H


    # -----------------------------------------------------------------
    # Expanding-history temporal BCV.
    # -----------------------------------------------------------------

    for month in eligible_months:

        prior_months = [
            m
            for m in stats
            if m < month
        ]


        n_train = int(
            sum(
                stats[
                    m
                ][
                    "n"
                ]
                for m in prior_months
            )
        )


        train_sum = np.sum(
            [
                stats[
                    m
                ][
                    "sum"
                ]
                for m in prior_months
            ],
            axis=0
        )


        train_cross = np.sum(
            [
                stats[
                    m
                ][
                    "cross"
                ]
                for m in prior_months
            ],
            axis=0
        )


        test = stats[
            month
        ]


        n_test = int(
            test[
                "n"
            ]
        )


        train_mu, train_gram = centered_train_gram(
            n_train,
            train_sum,
            train_cross
        )


        test_gram = centered_test_gram(
            n_test,
            test[
                "sum"
            ],
            test[
                "cross"
            ],
            train_mu
        )


        print(
            f"{month}"
            f" | train={n_train:,}"
            f" | test={n_test:,}"
        )


        for scheme, width in SCHEMES.items():

            blocks = scheme_blocks[
                scheme
            ]


            for block_id, (
                hold,
                keep
            ) in enumerate(
                blocks,
                start=1
            ):

                path_rows, neg_sse = bcv_fold_path(
                    train_gram,
                    test_gram,
                    hold,
                    keep
                )


                large_negative_sse_total += (
                    neg_sse
                )


                expected_max_rank = (
                    len(
                        keep
                    )
                )


                actual_rank = int(
                    path_rows[
                        -1
                    ][
                        "rank_d"
                    ]
                )


                if (
                    actual_rank <= 0
                    or
                    actual_rank
                    >
                    expected_max_rank
                ):

                    numerical_rank_failures += 1


                hold_lags = [
                    int(
                        j
                        +
                        1
                    )
                    for j in hold
                ]


                for r in path_rows:

                    fold_rows.append({
                        "channel":
                            channel,

                        "current_month":
                            str(
                                month
                            ),

                        "scheme":
                            scheme,

                        "holdout_width":
                            int(
                                width
                            ),

                        "block_id":
                            int(
                                block_id
                            ),

                        "holdout_lag_first":
                            int(
                                hold_lags[
                                    0
                                ]
                            ),

                        "holdout_lag_last":
                            int(
                                hold_lags[
                                    -1
                                ]
                            ),

                        "holdout_lags":
                            "|".join(
                                str(
                                    x
                                )
                                for x in hold_lags
                            ),

                        "retained_lags":
                            int(
                                len(
                                    keep
                                )
                            ),

                        "train_orders":
                            n_train,

                        "test_orders":
                            n_test,

                        "k":
                            int(
                                r[
                                    "k"
                                ]
                            ),

                        "rank_d":
                            int(
                                r[
                                    "rank_d"
                                ]
                            ),

                        "heldout_energy":
                            float(
                                r[
                                    "heldout_energy"
                                ]
                            ),

                        "sse":
                            float(
                                r[
                                    "sse"
                                ]
                            ),

                        "relative_squared_error":
                            float(
                                r[
                                    "relative_squared_error"
                                ]
                            ),

                        "relative_error":
                            float(
                                r[
                                    "relative_error"
                                ]
                            ),
                    })


folds = pd.DataFrame(
    fold_rows
)


folds.to_csv(
    OUT_FOLDS,
    index=False
)


# =============================================================================
# 8. RANK PROFILES
# =============================================================================

profile_rows = []


for (
    channel,
    scheme,
    k
), g in folds.groupby(
    [
        "channel",
        "scheme",
        "k"
    ],
    sort=True
):

    width = int(
        g[
            "holdout_width"
        ].iloc[
            0
        ]
    )


    blocks_per_month = len(
        scheme_blocks[
            scheme
        ]
    )


    expected_folds = (
        len(
            eligible_months
        )
        *
        blocks_per_month
    )


    pooled_error = float(
        math.sqrt(
            g[
                "sse"
            ].sum()
            /
            g[
                "heldout_energy"
            ].sum()
        )
    )


    fold_error = (
        g[
            "relative_error"
        ]
        .astype(
            float
        )
    )


    profile_rows.append({
        "channel":
            channel,

        "scheme":
            scheme,

        "holdout_width":
            width,

        "k":
            int(
                k
            ),

        "folds_observed":
            int(
                len(
                    g
                )
            ),

        "folds_expected":
            int(
                expected_folds
            ),

        "complete_fold_coverage":
            bool(
                len(
                    g
                )
                ==
                expected_folds
            ),

        "temporal_months":
            int(
                g[
                    "current_month"
                ].nunique()
            ),

        "pooled_relative_error":
            pooled_error,

        "mean_fold_relative_error":
            float(
                fold_error.mean()
            ),

        "median_fold_relative_error":
            float(
                fold_error.median()
            ),

        "p95_fold_relative_error":
            float(
                fold_error.quantile(
                    0.95
                )
            ),

        "sd_fold_relative_error":
            float(
                fold_error.std(
                    ddof=1
                )
            ),

        "se_fold_relative_error":
            float(
                fold_error.std(
                    ddof=1
                )
                /
                math.sqrt(
                    len(
                        fold_error
                    )
                )
            ),
    })


profile = pd.DataFrame(
    profile_rows
)


profile.to_csv(
    OUT_PROFILE,
    index=False
)


# =============================================================================
# 9. CANDIDATES BY SCHEME
# =============================================================================

candidate_rows = []


for (
    channel,
    scheme
), g in profile.groupby(
    [
        "channel",
        "scheme"
    ],
    sort=True
):

    complete = (
        g.loc[
            g[
                "complete_fold_coverage"
            ]
        ]
        .sort_values(
            "k"
        )
        .copy()
    )


    if complete.empty:

        raise RuntimeError(
            f"Sem rank com cobertura completa: {channel}/{scheme}"
        )


    best_idx = (
        complete[
            "pooled_relative_error"
        ]
        .idxmin()
    )


    best = complete.loc[
        best_idx
    ]


    best_k = int(
        best[
            "k"
        ]
    )


    # -------------------------------------------------------------
    # 1-SE apenas como candidato parcimonioso.
    #
    # Não é decisão final.
    # -------------------------------------------------------------

    threshold = (
        float(
            best[
                "mean_fold_relative_error"
            ]
        )
        +
        float(
            best[
                "se_fold_relative_error"
            ]
        )
    )


    eligible = (
        complete.loc[
            complete[
                "mean_fold_relative_error"
            ]
            <=
            threshold
        ]
        .sort_values(
            "k"
        )
    )


    one_se_k = int(
        eligible.iloc[
            0
        ][
            "k"
        ]
    )


    max_tested_k = int(
        complete[
            "k"
        ].max()
    )


    candidate_rows.append({
        "channel":
            channel,

        "scheme":
            scheme,

        "holdout_width":
            int(
                best[
                    "holdout_width"
                ]
            ),

        "max_tested_k":
            max_tested_k,

        "best_bcv_k":
            best_k,

        "best_at_max_tested_rank":
            bool(
                best_k
                ==
                max_tested_k
            ),

        "best_pooled_relative_error":
            float(
                best[
                    "pooled_relative_error"
                ]
            ),

        "best_mean_fold_relative_error":
            float(
                best[
                    "mean_fold_relative_error"
                ]
            ),

        "best_se_fold_relative_error":
            float(
                best[
                    "se_fold_relative_error"
                ]
            ),

        "one_se_threshold":
            threshold,

        "one_se_candidate_k":
            one_se_k,

        "candidate_only_not_final":
            True,
    })


candidates = pd.DataFrame(
    candidate_rows
)


candidates.to_csv(
    OUT_CANDIDATES,
    index=False
)


# =============================================================================
# 10. MONTH-BY-MONTH BEST RANK
# =============================================================================

monthly_rows = []


for (
    channel,
    scheme,
    month,
    k
), g in folds.groupby(
    [
        "channel",
        "scheme",
        "current_month",
        "k"
    ],
    sort=True
):

    expected_blocks = len(
        scheme_blocks[
            scheme
        ]
    )


    if len(
        g
    ) != expected_blocks:

        continue


    pooled = float(
        math.sqrt(
            g[
                "sse"
            ].sum()
            /
            g[
                "heldout_energy"
            ].sum()
        )
    )


    monthly_rows.append({
        "channel":
            channel,

        "scheme":
            scheme,

        "current_month":
            month,

        "k":
            int(
                k
            ),

        "pooled_relative_error":
            pooled,
    })


monthly_profile = pd.DataFrame(
    monthly_rows
)


monthly_best_rows = []


for (
    channel,
    scheme,
    month
), g in monthly_profile.groupby(
    [
        "channel",
        "scheme",
        "current_month"
    ],
    sort=True
):

    g = g.sort_values(
        "k"
    )


    row = g.loc[
        g[
            "pooled_relative_error"
        ].idxmin()
    ]


    monthly_best_rows.append({
        "channel":
            channel,

        "scheme":
            scheme,

        "current_month":
            month,

        "best_k":
            int(
                row[
                    "k"
                ]
            ),

        "best_pooled_relative_error":
            float(
                row[
                    "pooled_relative_error"
                ]
            ),

        "max_comparable_k":
            int(
                g[
                    "k"
                ].max()
            ),
    })


monthly_best = pd.DataFrame(
    monthly_best_rows
)


monthly_best.to_csv(
    OUT_MONTHLY,
    index=False
)


# =============================================================================
# 11. VALIDATION
# =============================================================================

checks = []


add_check(
    checks,
    "c34a1_pass",
    c34a1.get(
        "status"
    )
    ==
    "PASS",
    c34a1.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "criterion_non_identifying",
    c34a1.get(
        "criterion_classification"
    )
    ==
    "NON_IDENTIFYING_FOR_INTRINSIC_RANK",
    c34a1.get(
        "criterion_classification"
    ),
    "NON_IDENTIFYING_FOR_INTRINSIC_RANK"
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
    full_support_rows
    ==
    96421,
    full_support_rows,
    96421
)


add_check(
    checks,
    "eligible_months",
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
    "first_month",
    str(
        eligible_months[
            0
        ]
    )
    ==
    "2017-04",
    eligible_months[
        0
    ],
    "2017-04"
)


add_check(
    checks,
    "last_month",
    str(
        eligible_months[
            -1
        ]
    )
    ==
    "2018-08",
    eligible_months[
        -1
    ],
    "2018-08"
)


add_check(
    checks,
    "strict_temporal_boundaries",
    temporal_boundary_failures
    ==
    0,
    temporal_boundary_failures,
    0
)


add_check(
    checks,
    "column_partition_failures",
    column_coverage_failures
    ==
    0,
    column_coverage_failures,
    0
)


add_check(
    checks,
    "synthetic_rank3_self_consistency",
    synthetic[
        "rank3_self_consistency_relative_error"
    ]
    <
    1e-10,
    synthetic[
        "rank3_self_consistency_relative_error"
    ],
    "<1e-10"
)


add_check(
    checks,
    "gram_formula_matches_direct_coef",
    synthetic[
        "gram_vs_direct_coef_relative_difference"
    ]
    <
    1e-10,
    synthetic[
        "gram_vs_direct_coef_relative_difference"
    ],
    "<1e-10"
)


add_check(
    checks,
    "gram_formula_matches_direct_prediction",
    synthetic[
        "gram_vs_direct_prediction_relative_difference"
    ]
    <
    1e-10,
    synthetic[
        "gram_vs_direct_prediction_relative_difference"
    ],
    "<1e-10"
)


add_check(
    checks,
    "numerical_rank_failures",
    numerical_rank_failures
    ==
    0,
    numerical_rank_failures,
    0
)


add_check(
    checks,
    "large_negative_sse",
    large_negative_sse_total
    ==
    0,
    large_negative_sse_total,
    0
)


numeric_cols = [
    "heldout_energy",
    "sse",
    "relative_squared_error",
    "relative_error",
]


nonfinite = int(
    (
        ~np.isfinite(
            folds[
                numeric_cols
            ].to_numpy(
                dtype=float
            )
        )
    ).sum()
)


add_check(
    checks,
    "all_fold_metrics_finite",
    nonfinite
    ==
    0,
    nonfinite,
    0
)


negative_errors = int(
    (
        folds[
            "relative_error"
        ]
        <
        0
    ).sum()
)


add_check(
    checks,
    "no_negative_relative_error",
    negative_errors
    ==
    0,
    negative_errors,
    0
)


baseline = folds.loc[
    folds[
        "k"
    ]
    ==
    0
]


baseline_deviation = float(
    np.max(
        np.abs(
            baseline[
                "relative_error"
            ].to_numpy(
                dtype=float
            )
            -
            1.0
        )
    )
)


add_check(
    checks,
    "k0_is_mean_only_baseline",
    baseline_deviation
    <
    1e-10,
    baseline_deviation,
    "<1e-10"
)


add_check(
    checks,
    "six_channel_scheme_candidates",
    len(
        candidates
    )
    ==
    6,
    len(
        candidates
    ),
    6
)


monthly_expected = (
    2
    *
    3
    *
    17
)


add_check(
    checks,
    "monthly_best_rows",
    len(
        monthly_best
    )
    ==
    monthly_expected,
    len(
        monthly_best
    ),
    monthly_expected
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
    ].eq(
        "FAIL"
    ).sum()
)


# =============================================================================
# 12. SUMMARY JSON
# =============================================================================

runtime = (
    time.perf_counter()
    -
    start_runtime
)


candidate_records = (
    candidates[
        [
            "channel",
            "scheme",
            "holdout_width",
            "max_tested_k",
            "best_bcv_k",
            "best_at_max_tested_rank",
            "best_pooled_relative_error",
            "one_se_candidate_k",
        ]
    ]
    .to_dict(
        orient="records"
    )
)


one_se_values = (
    candidates[
        "one_se_candidate_k"
    ]
    .astype(
        int
    )
)


summary = {
    "step":
        "MODEL_01_0_C3_4B_TEMPORAL_BLOCK_BCV",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "representation":
        "RAW_30D",

    "method":
        "TEMPORAL_ROWS_PLUS_CONTIGUOUS_LAG_BLOCK_BICROSSVALIDATION",

    "bcv_formula":
        "A_hat_k = B D_k^+ C",

    "window_days":
        WINDOW,

    "temporal_design":
        "EXPANDING_HISTORY_VS_FUTURE_MONTH",

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

    "column_holdout_widths":
        list(
            SCHEMES.values()
        ),

    "column_holdout_schemes":
        SCHEMES,

    "fold_rank_error_rows":
        int(
            len(
                folds
            )
        ),

    "rank_profile_rows":
        int(
            len(
                profile
            )
        ),

    "scheme_candidates":
        candidate_records,

    "one_se_candidate_range_across_channel_schemes": [
        int(
            one_se_values.min()
        ),
        int(
            one_se_values.max()
        ),
    ],

    "candidate_status":
        "DIAGNOSTIC_ONLY_NOT_FROZEN",

    "target_used":
        False,

    "final_k_selected":
        False,

    "final_bcv_scheme_selected":
        False,

    "folds_frozen":
        False,

    "classifier_trained":
        False,

    "functional_feature_created":
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
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 13. REPORT
# =============================================================================

candidate_text = (
    candidates[
        [
            "channel",
            "scheme",
            "holdout_width",
            "max_tested_k",
            "best_bcv_k",
            "best_at_max_tested_rank",
            "best_pooled_relative_error",
            "one_se_candidate_k",
        ]
    ]
    .to_string(
        index=False
    )
)


monthly_stability = (
    monthly_best.groupby(
        [
            "channel",
            "scheme"
        ]
    )
    .agg(
        months=(
            "current_month",
            "count"
        ),

        best_k_median=(
            "best_k",
            "median"
        ),

        best_k_min=(
            "best_k",
            "min"
        ),

        best_k_max=(
            "best_k",
            "max"
        ),
    )
    .reset_index()
)


report = f"""
======================================================================================================================
MODEL 01.0-C3.4-B — TEMPORAL BLOCK BI-CROSS-VALIDATION
======================================================================================================================

STATUS
----------------------------------------------------------------------------------------------------------------------
Validation status                    : {summary["status"]}

REPRESENTATION
----------------------------------------------------------------------------------------------------------------------
RAW-30D

TEMPORAL DESIGN
----------------------------------------------------------------------------------------------------------------------
Train                                : all eligible months before current month
Test                                 : current future month

First test month                     : {summary["first_test_month"]}
Last test month                      : {summary["last_test_month"]}
Temporal test months                 : {summary["temporal_test_months"]}

BCV
----------------------------------------------------------------------------------------------------------------------
A_hat(K) = B D_K^+ C

A = future rows / held-out lags
B = future rows / retained lags
C = past rows   / held-out lags
D = past rows   / retained lags

COLUMN SENSITIVITY
----------------------------------------------------------------------------------------------------------------------
Holdout widths                       : 3, 6, 15 consecutive lags

CANDIDATES — DIAGNOSTIC ONLY
----------------------------------------------------------------------------------------------------------------------
{candidate_text}

TEMPORAL STABILITY
----------------------------------------------------------------------------------------------------------------------
{monthly_stability.to_string(index=False)}

IMPORTANT
----------------------------------------------------------------------------------------------------------------------
No final K is selected here.

No single holdout size is promoted here.

The purpose is to determine whether low-rank structure predicts
simultaneously unseen future rows and held-out lag coordinates.

Target used                          : NO
Final K selected                     : NO
Final BCV scheme selected            : NO
Folds frozen                         : NO
Functional model feature created     : NO
Classifier trained                   : NO
Silver created                       : NO
RAW modified                         : NO

Validation failures                  : {failures}
Runtime seconds                      : {runtime:.3f}
======================================================================================================================
""".strip()


OUT_REPORT.write_text(
    report + "\n",
    encoding="utf-8"
)


# =============================================================================
# 14. PRINT COMPACT RESULTS
# =============================================================================

print()
print("=" * 118)
print("BCV SCHEME CANDIDATES — NÃO CONGELADOS")
print("=" * 118)


print(
    candidates.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print("=" * 118)
print("MONTHLY BEST-RANK STABILITY")
print("=" * 118)


print(
    monthly_stability.to_string(
        index=False
    )
)


print()
print("=" * 118)
print("VALIDATION")
print("=" * 118)


print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 118)
print("RESULTADO C3.4-B")
print("=" * 118)


print(
    "STATUS                  =",
    summary[
        "status"
    ]
)

print(
    "TEMPORAL MONTHS         =",
    len(
        eligible_months
    )
)

print(
    "BCV ERROR ROWS          =",
    len(
        folds
    )
)

print(
    "SCHEME CANDIDATES       =",
    len(
        candidates
    )
)

print(
    "1-SE CANDIDATE RANGE    =",
    int(
        one_se_values.min()
    ),
    "..",
    int(
        one_se_values.max()
    )
)


print()
print(
    "TARGET USED             = NÃO"
)

print(
    "K FINAL                 = NÃO"
)

print(
    "BCV SCHEME FINAL        = NÃO"
)

print(
    "FOLDS FROZEN            = NÃO"
)

print(
    "CLASSIFIER TRAINED      = NÃO"
)

print(
    "SILVER CREATED          = NÃO"
)

print(
    "RAW MODIFIED            = NÃO"
)


print()
print(
    "RUNTIME SECONDS         =",
    runtime
)


if failures:

    sys.exit(
        2
    )


print()
print(
    "[PASS] C3.4-B temporal block BCV concluído."
)

print(
    "[PASS] Nenhum K foi congelado."
)

print(
    "[PASS] Revisar candidatos e estabilidade antes da decisão final."
)

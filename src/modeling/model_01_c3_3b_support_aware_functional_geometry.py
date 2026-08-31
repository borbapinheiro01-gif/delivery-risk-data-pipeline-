#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-B
SUPPORT-AWARE FUNCTIONAL GEOMETRY
===============================================================================

OBJETIVO
--------
Auditar a geometria das curvas PIT construídas no C3.3-A antes de qualquer:

- smoothing;
- spline;
- FPCA;
- feature funcional;
- treinamento.

CORREÇÃO METODOLÓGICA
---------------------
Uma curva histórica próxima ao início do dataset pode possuir bins anteriores
ao começo da observação RAW.

Esses bins NÃO significam necessariamente zero atividade.

Distinção:

    OBSERVED ZERO
        histórico estava dentro da janela observável e nenhum evento ocorreu.

    UNOBSERVED LEFT HISTORY
        janela solicitada ultrapassa o início observável do dataset.

Para uma janela L:

    support_i(L) =
        1[t_i - L >= t_RAW_min]

Somente pedidos com support_i(L)=1 participam da SVD daquela janela.

IMPORTANTE
----------
Isto NÃO remove os demais pedidos do projeto.
É apenas uma máscara de validade para o diagnóstico funcional.

GEOMETRIA
---------
Para cada:

    channel ∈ {purchase_volume, purchase_freight}
    L ∈ {30, 60, 90}

construímos:

    H^(channel,L)

usando apenas pedidos com histórico completamente observável dentro do dataset.

Centralizamos cada lag:

    Hc = H - mean(H, axis=0)

e aplicamos:

    Hc = U Sigma V^T

Sem scaling por lag.

A energia da componente k é:

    e_k = sigma_k^2 / sum_j sigma_j^2

e a energia acumulada:

    E_K = sum_{k=1}^K e_k

Registramos o menor K tal que:

    E_K >= 0.80
    E_K >= 0.90
    E_K >= 0.95
    E_K >= 0.99

NÃO:
- escolhe K final;
- escolhe janela final;
- aplica FPCA;
- cria features de modelo;
- congela threshold;
- congela folds;
- treina modelo;
- cria Silver;
- altera RAW.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = (
    PROJECT
    / "data"
    / "raw"
    / "olist"
)

OUT = (
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

C33A_SUMMARY = (
    OUT
    / "02j_functional_pit_summary.json"
)

ORDER_INDEX = (
    OUT
    / "02a_functional_pit_order_index.csv"
)

VOLUME_NPY = (
    ART
    / "02g_purchase_volume_curve_90d.npy"
)

FREIGHT_NPY = (
    ART
    / "02h_purchase_freight_curve_90d.npy"
)


# =============================================================================
# OUTPUTS C3.3-B
# =============================================================================

SUPPORT_ORDER = (
    OUT
    / "02l_history_support_by_order.csv"
)

SUPPORT_SUMMARY = (
    OUT
    / "02m_history_support_summary.csv"
)

SUPPORT_MONTH = (
    OUT
    / "02n_history_support_by_month.csv"
)

SPECTRUM = (
    OUT
    / "02o_functional_svd_spectrum.csv"
)

GEOMETRY = (
    OUT
    / "02p_functional_window_geometry_summary.csv"
)

LAG_PROFILE = (
    OUT
    / "02q_functional_lag_profile_90d.csv"
)

ADJACENT_CORR = (
    OUT
    / "02r_functional_adjacent_lag_correlation_90d.csv"
)

VALIDATION = (
    OUT
    / "02s_functional_geometry_validation.csv"
)

SUMMARY_JSON = (
    OUT
    / "02t_functional_geometry_summary.json"
)

REPORT = (
    OUT
    / "02u_functional_geometry_report.txt"
)


OUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# CONSTANTS
# =============================================================================

WINDOWS = [
    30,
    60,
    90
]

EXPECTED_ROWS = 96470
EXPECTED_LAGS = 90

DAY = pd.Timedelta(
    days=1
)


# =============================================================================
# HELPERS
# =============================================================================

def add_check(
    rows,
    check,
    condition,
    observed,
    expected
):

    rows.append({
        "check":
            check,

        "status":
            "PASS"
            if bool(condition)
            else "FAIL",

        "observed":
            observed,

        "expected":
            expected,
    })


def first_k_at(
    cumulative,
    threshold
):

    hits = np.where(
        cumulative >= threshold
    )[0]

    if len(hits) == 0:
        return None

    return int(
        hits[0] + 1
    )


def svd_geometry(
    matrix,
    channel,
    window
):

    """
    Matrix:
        rows = orders with full historical support
        cols = retrospective daily lags

    PCA-equivalent diagnostic:
        center only, no scaling.
    """

    A = np.asarray(
        matrix,
        dtype=np.float64
    )

    n, p = A.shape

    mean_vector = A.mean(
        axis=0
    )

    centered = (
        A
        -
        mean_vector
    )

    lag_std = centered.std(
        axis=0,
        ddof=0
    )

    zero_variance_lags = int(
        (
            lag_std == 0
        ).sum()
    )

    # Singular values only:
    # avoids allocating the very large U matrix.
    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False
    )

    singular_values = np.asarray(
        singular_values,
        dtype=np.float64
    )

    eps = np.finfo(
        singular_values.dtype
    ).eps

    tolerance = float(
        singular_values[0]
        *
        max(
            centered.shape
        )
        *
        eps
    )

    rank = int(
        (
            singular_values
            >
            tolerance
        ).sum()
    )

    energy = (
        singular_values
        **
        2
    )

    total_energy = float(
        energy.sum()
    )

    if (
        total_energy <= 0
        or
        not np.isfinite(
            total_energy
        )
    ):

        raise RuntimeError(
            f"Energia inválida: "
            f"{channel} {window}d"
        )

    energy_ratio = (
        energy
        /
        total_energy
    )

    cumulative = np.cumsum(
        energy_ratio
    )

    k80 = first_k_at(
        cumulative,
        0.80
    )

    k90 = first_k_at(
        cumulative,
        0.90
    )

    k95 = first_k_at(
        cumulative,
        0.95
    )

    k99 = first_k_at(
        cumulative,
        0.99
    )

    def residual_fraction(k):

        if k is None:
            return np.nan

        return float(
            np.sqrt(
                max(
                    0.0,
                    1.0
                    -
                    cumulative[
                        k - 1
                    ]
                )
            )
        )

    spectrum_rows = []

    for component, (
        sigma,
        er,
        cer
    ) in enumerate(
        zip(
            singular_values,
            energy_ratio,
            cumulative
        ),
        start=1
    ):

        spectrum_rows.append({
            "channel":
                channel,

            "window_days":
                window,

            "component":
                component,

            "singular_value":
                float(
                    sigma
                ),

            "relative_to_first":
                float(
                    sigma
                    /
                    singular_values[0]
                ),

            "energy_ratio":
                float(
                    er
                ),

            "energy_pct":
                float(
                    100.0
                    *
                    er
                ),

            "cumulative_energy_ratio":
                float(
                    cer
                ),

            "cumulative_energy_pct":
                float(
                    100.0
                    *
                    cer
                ),

            "above_numerical_rank_tolerance":
                bool(
                    sigma
                    >
                    tolerance
                ),
        })

    summary = {
        "channel":
            channel,

        "window_days":
            window,

        "orders_full_support":
            n,

        "lags":
            p,

        "zero_variance_lags":
            zero_variance_lags,

        "numerical_rank":
            rank,

        "nullity":
            int(
                p - rank
            ),

        "svd_tolerance":
            tolerance,

        "largest_singular_value":
            float(
                singular_values[0]
            ),

        "smallest_singular_value":
            float(
                singular_values[-1]
            ),

        "pc1_energy_pct":
            float(
                100.0
                *
                energy_ratio[0]
            ),

        "k80":
            k80,

        "k90":
            k90,

        "k95":
            k95,

        "k99":
            k99,

        "relative_frobenius_error_at_k90":
            residual_fraction(
                k90
            ),

        "relative_frobenius_error_at_k95":
            residual_fraction(
                k95
            ),
    }

    return (
        summary,
        spectrum_rows
    )


# =============================================================================
# START
# =============================================================================

start_time = time.perf_counter()

print()
print("=" * 116)
print("MODEL 01.0-C3.3-B — SUPPORT-AWARE FUNCTIONAL GEOMETRY")
print("=" * 116)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required = [
    (
        C33A_SUMMARY,
        "C3.3-A summary"
    ),
    (
        ORDER_INDEX,
        "C3.3-A order index"
    ),
    (
        VOLUME_NPY,
        "volume curve"
    ),
    (
        FREIGHT_NPY,
        "freight curve"
    ),
    (
        RAW
        / "olist_orders_dataset.csv",
        "orders RAW"
    ),
]

for path, label in required:

    if not path.exists():

        raise SystemExit(
            f"[FAIL] {label} ausente: {path}"
        )

    print(
        f"[PASS] {label}"
    )


with C33A_SUMMARY.open(
    encoding="utf-8"
) as f:

    c33a = json.load(f)


if c33a.get(
    "status"
) != "PASS":

    raise SystemExit(
        "[STOP] C3.3-A não está PASS. "
        "C3.3-B não será executado."
    )


if c33a.get(
    "stage_assessment"
) != "FUNCTIONAL_CURVE_CONSTRUCTION_VIABLE":

    raise SystemExit(
        "[STOP] C3.3-A não declarou "
        "FUNCTIONAL_CURVE_CONSTRUCTION_VIABLE."
    )


if c33a.get(
    "volume_bruteforce_bin_mismatches"
) != 0:

    raise SystemExit(
        "[STOP] C3.3-A possui mismatch de volume."
    )


if c33a.get(
    "freight_bruteforce_bin_mismatches"
) != 0:

    raise SystemExit(
        "[STOP] C3.3-A possui mismatch de freight."
    )


print()
print("[PASS] C3.3-A formalmente validado.")
print("[PASS] Brute-force mismatches = 0.")


# =============================================================================
# 2. LOAD ORDER INDEX
# =============================================================================

orders_index = pd.read_csv(
    ORDER_INDEX,
    parse_dates=[
        "order_purchase_timestamp"
    ]
)


if len(
    orders_index
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Order index possui {len(orders_index)} linhas; "
        f"esperado {EXPECTED_ROWS}."
    )


if not orders_index[
    "order_id"
].is_unique:

    raise RuntimeError(
        "order_id duplicado no order index."
    )


orders_index = (
    orders_index
    .sort_values(
        "curve_row_index"
    )
    .reset_index(
        drop=True
    )
)


expected_index = np.arange(
    EXPECTED_ROWS,
    dtype=np.int64
)


if not np.array_equal(
    orders_index[
        "curve_row_index"
    ].to_numpy(
        dtype=np.int64
    ),
    expected_index
):

    raise RuntimeError(
        "curve_row_index não é 0..96469."
    )


# =============================================================================
# 3. LOAD CURVES WITH MMAP
# =============================================================================

volume = np.load(
    VOLUME_NPY,
    mmap_mode="r",
    allow_pickle=False
)

freight = np.load(
    FREIGHT_NPY,
    mmap_mode="r",
    allow_pickle=False
)


print()
print("Volume shape :", volume.shape)
print("Volume dtype :", volume.dtype)
print("Freight shape:", freight.shape)
print("Freight dtype:", freight.dtype)


if volume.shape != (
    EXPECTED_ROWS,
    EXPECTED_LAGS
):

    raise RuntimeError(
        f"Volume shape inválido: {volume.shape}"
    )


if freight.shape != (
    EXPECTED_ROWS,
    EXPECTED_LAGS
):

    raise RuntimeError(
        f"Freight shape inválido: {freight.shape}"
    )


# =============================================================================
# 4. DATASET OBSERVATION START
# =============================================================================

raw_orders = pd.read_csv(
    RAW
    / "olist_orders_dataset.csv",
    usecols=[
        "order_purchase_timestamp"
    ]
)


raw_purchase = pd.to_datetime(
    raw_orders[
        "order_purchase_timestamp"
    ],
    errors="coerce"
)


if raw_purchase.isna().all():

    raise RuntimeError(
        "Nenhum purchase timestamp válido no RAW."
    )


dataset_start = raw_purchase.min()

dataset_end = raw_purchase.max()


print()
print("=" * 116)
print("OBSERVATION BOUNDARY")
print("=" * 116)

print(
    "RAW purchase start:",
    dataset_start
)

print(
    "RAW purchase end  :",
    dataset_end
)


# =============================================================================
# 5. HISTORICAL SUPPORT
# =============================================================================

support = orders_index[
    [
        "curve_row_index",
        "order_id",
        "order_purchase_timestamp",
        "purchase_month",
    ]
].copy()


support[
    "observable_history_days"
] = (
    (
        support[
            "order_purchase_timestamp"
        ]
        -
        dataset_start
    )
    /
    DAY
)


for window in WINDOWS:

    support[
        f"full_support_{window}d"
    ] = (
        support[
            "order_purchase_timestamp"
        ]
        -
        pd.Timedelta(
            days=window
        )
        >=
        dataset_start
    )


support.to_csv(
    SUPPORT_ORDER,
    index=False
)


# =============================================================================
# 6. SUPPORT SUMMARY
# =============================================================================

support_rows = []


for window in WINDOWS:

    col = (
        f"full_support_{window}d"
    )

    full = int(
        support[
            col
        ].sum()
    )

    incomplete = int(
        len(
            support
        )
        -
        full
    )

    first_valid = (
        support.loc[
            support[
                col
            ],
            "order_purchase_timestamp"
        ]
        .min()
    )

    support_rows.append({
        "window_days":
            window,

        "task_orders":
            len(
                support
            ),

        "full_support_orders":
            full,

        "incomplete_left_history_orders":
            incomplete,

        "full_support_pct":
            float(
                100.0
                *
                full
                /
                len(
                    support
                )
            ),

        "incomplete_left_history_pct":
            float(
                100.0
                *
                incomplete
                /
                len(
                    support
                )
            ),

        "first_full_support_prediction_time":
            first_valid,

        "dataset_purchase_start":
            dataset_start,
    })


support_summary = pd.DataFrame(
    support_rows
)

support_summary.to_csv(
    SUPPORT_SUMMARY,
    index=False
)


# =============================================================================
# 7. SUPPORT BY MONTH
# =============================================================================

monthly_rows = []


for month, group in support.groupby(
    "purchase_month",
    sort=True
):

    row = {
        "purchase_month":
            month,

        "orders":
            len(
                group
            ),
    }

    for window in WINDOWS:

        col = (
            f"full_support_{window}d"
        )

        row[
            f"full_support_{window}d_orders"
        ] = int(
            group[
                col
            ].sum()
        )

        row[
            f"full_support_{window}d_pct"
        ] = float(
            100.0
            *
            group[
                col
            ].mean()
        )

    monthly_rows.append(
        row
    )


support_month = pd.DataFrame(
    monthly_rows
)

support_month.to_csv(
    SUPPORT_MONTH,
    index=False
)


# =============================================================================
# 8. PRINT SUPPORT
# =============================================================================

print()
print("=" * 116)
print("HISTORICAL SUPPORT")
print("=" * 116)

print(
    support_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# 9. SVD GEOMETRY — FULL SUPPORT ONLY
# =============================================================================

geometry_rows = []
spectrum_rows = []


channels = {
    "purchase_volume":
        volume,

    "purchase_freight":
        freight,
}


print()
print("=" * 116)
print("SUPPORT-AWARE SVD")
print("=" * 116)


for channel, curve in channels.items():

    for window in WINDOWS:

        support_mask = (
            support[
                f"full_support_{window}d"
            ]
            .to_numpy(
                dtype=bool
            )
        )

        n_support = int(
            support_mask.sum()
        )

        if n_support < 100:

            raise RuntimeError(
                f"Suporte insuficiente para "
                f"{channel} / {window}d: "
                f"{n_support}"
            )


        # ---------------------------------------------------------
        # IMPORTANTE:
        # apenas esta matriz temporária é convertida para float64.
        # O arquivo .npy original permanece intacto.
        # ---------------------------------------------------------

        matrix = np.asarray(
            curve[
                support_mask,
                :window
            ],
            dtype=np.float64
        )


        if not np.isfinite(
            matrix
        ).all():

            raise RuntimeError(
                f"NaN/Inf em {channel}/{window}d."
            )


        if (
            matrix < 0
        ).any():

            raise RuntimeError(
                f"Valor negativo em "
                f"{channel}/{window}d."
            )


        geom, spec = svd_geometry(
            matrix,
            channel,
            window
        )

        geometry_rows.append(
            geom
        )

        spectrum_rows.extend(
            spec
        )


        print()
        print(
            f"{channel} | {window}d"
        )

        print(
            "  full-support orders :",
            n_support
        )

        print(
            "  rank                :",
            geom[
                "numerical_rank"
            ]
        )

        print(
            "  nullity             :",
            geom[
                "nullity"
            ]
        )

        print(
            "  PC1 energy          :",
            f"{geom['pc1_energy_pct']:.4f}%"
        )

        print(
            "  K80 / K90 / K95/K99 :",
            geom[
                "k80"
            ],
            "/",
            geom[
                "k90"
            ],
            "/",
            geom[
                "k95"
            ],
            "/",
            geom[
                "k99"
            ]
        )

        print(
            "  rel.error@K90       :",
            f"{geom['relative_frobenius_error_at_k90']:.8f}"
        )

        print(
            "  rel.error@K95       :",
            f"{geom['relative_frobenius_error_at_k95']:.8f}"
        )


geometry_df = pd.DataFrame(
    geometry_rows
)

spectrum_df = pd.DataFrame(
    spectrum_rows
)


geometry_df.to_csv(
    GEOMETRY,
    index=False
)

spectrum_df.to_csv(
    SPECTRUM,
    index=False
)


# =============================================================================
# 10. 90-DAY LAG PROFILE — FULL SUPPORT ONLY
# =============================================================================

lag_profile_rows = []


support90 = (
    support[
        "full_support_90d"
    ]
    .to_numpy(
        dtype=bool
    )
)


for channel, curve in channels.items():

    A = np.asarray(
        curve[
            support90,
            :90
        ],
        dtype=np.float64
    )

    for b in range(
        90
    ):

        x = A[
            :,
            b
        ]

        lag_profile_rows.append({
            "channel":
                channel,

            "lag_day":
                b + 1,

            "orders_full_90d_support":
                len(
                    x
                ),

            "mean":
                float(
                    np.mean(
                        x
                    )
                ),

            "std":
                float(
                    np.std(
                        x,
                        ddof=0
                    )
                ),

            "median":
                float(
                    np.median(
                        x
                    )
                ),

            "p05":
                float(
                    np.quantile(
                        x,
                        0.05
                    )
                ),

            "p25":
                float(
                    np.quantile(
                        x,
                        0.25
                    )
                ),

            "p75":
                float(
                    np.quantile(
                        x,
                        0.75
                    )
                ),

            "p95":
                float(
                    np.quantile(
                        x,
                        0.95
                    )
                ),

            "zero_pct":
                float(
                    100.0
                    *
                    (
                        x == 0
                    ).mean()
                ),
        })


lag_profile_df = pd.DataFrame(
    lag_profile_rows
)

lag_profile_df.to_csv(
    LAG_PROFILE,
    index=False
)


# =============================================================================
# 11. ADJACENT-LAG CORRELATION
# =============================================================================

corr_rows = []


for channel, curve in channels.items():

    A = np.asarray(
        curve[
            support90,
            :90
        ],
        dtype=np.float64
    )

    for b in range(
        89
    ):

        x = A[
            :,
            b
        ]

        y = A[
            :,
            b + 1
        ]

        sx = float(
            np.std(
                x
            )
        )

        sy = float(
            np.std(
                y
            )
        )

        if (
            sx == 0
            or
            sy == 0
        ):

            corr = np.nan

        else:

            corr = float(
                np.corrcoef(
                    x,
                    y
                )[0, 1]
            )

        corr_rows.append({
            "channel":
                channel,

            "lag_day_a":
                b + 1,

            "lag_day_b":
                b + 2,

            "pearson_correlation":
                corr,
        })


corr_df = pd.DataFrame(
    corr_rows
)

corr_df.to_csv(
    ADJACENT_CORR,
    index=False
)


# =============================================================================
# 12. VALIDATION
# =============================================================================

checks = []


add_check(
    checks,
    "c33a_pass",
    c33a[
        "status"
    ] == "PASS",
    c33a[
        "status"
    ],
    "PASS"
)

add_check(
    checks,
    "volume_shape",
    volume.shape
    ==
    (
        EXPECTED_ROWS,
        EXPECTED_LAGS
    ),
    str(
        volume.shape
    ),
    "(96470, 90)"
)

add_check(
    checks,
    "freight_shape",
    freight.shape
    ==
    (
        EXPECTED_ROWS,
        EXPECTED_LAGS
    ),
    str(
        freight.shape
    ),
    "(96470, 90)"
)

add_check(
    checks,
    "order_index_unique",
    orders_index[
        "order_id"
    ].is_unique,
    int(
        orders_index[
            "order_id"
        ].duplicated().sum()
    ),
    0
)


s30 = support[
    "full_support_30d"
].to_numpy(
    dtype=bool
)

s60 = support[
    "full_support_60d"
].to_numpy(
    dtype=bool
)

s90 = support[
    "full_support_90d"
].to_numpy(
    dtype=bool
)


add_check(
    checks,
    "support_nesting_90_in_60",
    bool(
        (
            ~s90
            |
            s60
        ).all()
    ),
    int(
        (
            s90
            &
            ~s60
        ).sum()
    ),
    0
)

add_check(
    checks,
    "support_nesting_60_in_30",
    bool(
        (
            ~s60
            |
            s30
        ).all()
    ),
    int(
        (
            s60
            &
            ~s30
        ).sum()
    ),
    0
)


for window in WINDOWS:

    n_support = int(
        support[
            f"full_support_{window}d"
        ].sum()
    )

    add_check(
        checks,
        f"support_{window}d_nonempty",
        n_support > 0,
        n_support,
        ">0"
    )


add_check(
    checks,
    "six_geometry_runs",
    len(
        geometry_df
    ) == 6,
    len(
        geometry_df
    ),
    6
)


add_check(
    checks,
    "all_svd_ranks_positive",
    bool(
        (
            geometry_df[
                "numerical_rank"
            ]
            >
            0
        ).all()
    ),
    int(
        (
            geometry_df[
                "numerical_rank"
            ]
            <=
            0
        ).sum()
    ),
    0
)


add_check(
    checks,
    "all_svd_rank_within_window",
    bool(
        (
            geometry_df[
                "numerical_rank"
            ]
            <=
            geometry_df[
                "window_days"
            ]
        ).all()
    ),
    int(
        (
            geometry_df[
                "numerical_rank"
            ]
            >
            geometry_df[
                "window_days"
            ]
        ).sum()
    ),
    0
)


add_check(
    checks,
    "all_energy_finite",
    bool(
        np.isfinite(
            spectrum_df[
                "energy_ratio"
            ].to_numpy(
                dtype=float
            )
        ).all()
    ),
    int(
        (
            ~np.isfinite(
                spectrum_df[
                    "energy_ratio"
                ].to_numpy(
                    dtype=float
                )
            )
        ).sum()
    ),
    0
)


energy_checks = (
    spectrum_df.groupby(
        [
            "channel",
            "window_days"
        ]
    )[
        "energy_ratio"
    ]
    .sum()
)


energy_error = float(
    np.max(
        np.abs(
            energy_checks.to_numpy()
            -
            1.0
        )
    )
)


add_check(
    checks,
    "energy_sums_to_one",
    energy_error
    <
    1e-10,
    energy_error,
    "<1e-10"
)


k_order_ok = bool(
    (
        (
            geometry_df[
                "k80"
            ]
            <=
            geometry_df[
                "k90"
            ]
        )
        &
        (
            geometry_df[
                "k90"
            ]
            <=
            geometry_df[
                "k95"
            ]
        )
        &
        (
            geometry_df[
                "k95"
            ]
            <=
            geometry_df[
                "k99"
            ]
        )
    ).all()
)


add_check(
    checks,
    "k_energy_order",
    k_order_ok,
    k_order_ok,
    True
)


add_check(
    checks,
    "lag_profile_rows",
    len(
        lag_profile_df
    ) == 180,
    len(
        lag_profile_df
    ),
    180
)


add_check(
    checks,
    "adjacent_correlation_rows",
    len(
        corr_df
    ) == 178,
    len(
        corr_df
    ),
    178
)


validation_df = pd.DataFrame(
    checks
)

validation_df.to_csv(
    VALIDATION,
    index=False
)


failures = int(
    validation_df[
        "status"
    ]
    .eq(
        "FAIL"
    )
    .sum()
)


# =============================================================================
# 13. DESCRIPTIVE CORRELATION SUMMARY
# =============================================================================

corr_summary = (
    corr_df.groupby(
        "channel"
    )[
        "pearson_correlation"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
    .reset_index()
)


# =============================================================================
# 14. SUMMARY
# =============================================================================

runtime = float(
    time.perf_counter()
    -
    start_time
)


summary = {
    "step":
        "MODEL_01_0_C3_3B_SUPPORT_AWARE_FUNCTIONAL_GEOMETRY",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "dataset_purchase_start":
        str(
            dataset_start
        ),

    "dataset_purchase_end":
        str(
            dataset_end
        ),

    "task_orders":
        int(
            len(
                support
            )
        ),

    "windows_days":
        WINDOWS,

    "full_support_orders": {
        str(
            window
        ):
            int(
                support[
                    f"full_support_{window}d"
                ].sum()
            )
        for window
        in WINDOWS
    },

    "incomplete_left_history_orders": {
        str(
            window
        ):
            int(
                (
                    ~support[
                        f"full_support_{window}d"
                    ]
                ).sum()
            )
        for window
        in WINDOWS
    },

    "geometry_runs":
        int(
            len(
                geometry_df
            )
        ),

    "geometry_method":
        "CENTERED_SVD_FULL_SUPPORT_ONLY",

    "pointwise_scaling":
        False,

    "left_history_policy":
        "DO_NOT_INTERPRET_PRE_DATASET_BINS_AS_OBSERVED_ZERO",

    "smoothing_applied":
        False,

    "basis_applied":
        False,

    "fpca_applied":
        False,

    "window_selected":
        False,

    "component_count_selected":
        False,

    "functional_module_unlocked":
        False,

    "feature_added_to_model":
        False,

    "folds_frozen":
        False,

    "model_trained":
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


SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 15. REPORT
# =============================================================================

lines = [
    "=" * 104,
    "MODEL 01.0-C3.3-B — SUPPORT-AWARE FUNCTIONAL GEOMETRY",
    "=" * 104,
    "",
    f"STATUS                      : {summary['status']}",
    f"TASK ORDERS                 : {len(support):,}",
    f"RAW PURCHASE START          : {dataset_start}",
    f"RAW PURCHASE END            : {dataset_end}",
    "",
    "LEFT-HISTORY POLICY",
    "-" * 104,
    "Pre-dataset history is NOT interpreted as observed zero.",
    "SVD uses only orders with full within-dataset support for each window.",
    "",
    "SUPPORT",
    "-" * 104,
]

for row in support_rows:

    lines.append(
        f"{int(row['window_days']):>2}d | "
        f"full={int(row['full_support_orders']):,} | "
        f"incomplete={int(row['incomplete_left_history_orders']):,} | "
        f"coverage={row['full_support_pct']:.4f}%"
    )


lines += [
    "",
    "FUNCTIONAL GEOMETRY",
    "-" * 104,
]


for _, row in geometry_df.iterrows():

    lines.append(
        f"{row['channel']} | "
        f"{int(row['window_days'])}d | "
        f"n={int(row['orders_full_support']):,} | "
        f"rank={int(row['numerical_rank'])} | "
        f"K80={int(row['k80'])} | "
        f"K90={int(row['k90'])} | "
        f"K95={int(row['k95'])} | "
        f"K99={int(row['k99'])} | "
        f"PC1={row['pc1_energy_pct']:.4f}%"
    )


lines += [
    "",
    "IMPORTANT",
    "-" * 104,
    "K is NOT selected.",
    "Window is NOT selected.",
    "Smoothing is NOT applied.",
    "FPCA is NOT applied.",
    "No predictor is added to MODEL_01.",
    "No fold is frozen.",
    "No model is trained.",
    "RAW is not modified.",
    "",
    f"VALIDATION FAILURES          : {failures}",
    f"RUNTIME                     : {runtime:.3f} s",
    "=" * 104,
]


REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 16. PRINT
# =============================================================================

print()
print("=" * 116)
print("FUNCTIONAL WINDOW GEOMETRY")
print("=" * 116)

print(
    geometry_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


print()
print("=" * 116)
print("ADJACENT-LAG CORRELATION SUMMARY — 90D")
print("=" * 116)

print(
    corr_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


print()
print("=" * 116)
print("VALIDATION")
print("=" * 116)

print(
    validation_df.to_string(
        index=False
    )
)


print()
print("=" * 116)
print("RESULTADO C3.3-B")
print("=" * 116)

print(
    "FAILURES                  =",
    failures
)

for window in WINDOWS:

    print(
        f"FULL SUPPORT {window:02d}D          =",
        int(
            support[
                f"full_support_{window}d"
            ].sum()
        )
    )


print(
    "GEOMETRY RUNS             =",
    len(
        geometry_df
    )
)

print()
print("WINDOW SELECTED            = NÃO")
print("K SELECTED                 = NÃO")
print("SMOOTHING APPLIED          = NÃO")
print("BASIS APPLIED              = NÃO")
print("FPCA APPLIED               = NÃO")
print("FUNCTIONAL MODULE UNLOCKED = NÃO")
print("FEATURE ADDED TO MODEL     = NÃO")
print("FOLDS FROZEN               = NÃO")
print("MODEL TRAINED              = NÃO")
print("SILVER CREATED             = NÃO")
print("RAW MODIFIED               = NÃO")


if failures:

    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.3-B support-aware geometry concluída."
)

print(
    "[PASS] Parar antes de smoothing / FPCA."
)

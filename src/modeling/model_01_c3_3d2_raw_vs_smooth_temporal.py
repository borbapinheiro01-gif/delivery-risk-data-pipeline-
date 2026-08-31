#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-D2
RAW 30D vs SMOOTH 30D EDF=0.75
PAIRED TEMPORAL PORTABILITY AUDIT
===============================================================================

PERGUNTA
--------
O smoothing moderado:

    30 dias
    EDF ratio = 0.75

produz uma representação funcional temporalmente mais eficiente que RAW 30d
SEM introduzir perda end-to-end excessiva em relação à curva RAW futura?

===============================================================================
PRINCÍPIO DE COMPARAÇÃO
===============================================================================

Para cada mês futuro m:

TRAIN:
    pedidos anteriores ao mês m
    com suporte histórico completo de 30 dias

TEST:
    pedidos do próprio mês m
    com suporte histórico completo de 30 dias

Os MESMOS pedidos são usados para:

    A) RAW 30d
    B) SMOOTH 30d / EDF=0.75

Nenhum target é utilizado.

===============================================================================
ROTA RAW
===============================================================================

Se:

    H_train

é a matriz RAW histórica:

    mu_R = mean(H_train)

    Hc_train = H_train - mu_R

    Hc_train = U Sigma V^T

K90_RAW é aprendido SOMENTE no passado:

    K90_RAW =
        min {
            K :
            sum_{k<=K} sigma_k^2
            /
            sum_j sigma_j^2
            >= 0.90
        }

No futuro:

    scores_R =
        (H_test - mu_R) V_K

    Hhat_RAW =
        mu_R + scores_R V_K^T

Erro:

    RE_RAW =
        ||H_test - Hhat_RAW||_F
        /
        ||H_test - mu_R||_F

===============================================================================
ROTA SMOOTH
===============================================================================

Operador já auditado:

    S_lambda =
        (I + lambda D2^T D2)^(-1)

com:

    window = 30
    EDF ratio = 0.75

Então:

    H_train^S =
        H_train S_lambda

    H_test^S =
        H_test S_lambda

A PCA/SVD é aprendida SOMENTE em H_train^S.

Reconstrução futura suavizada:

    Hhat_S =
        mu_S
        +
        [(H_test^S-mu_S)V_K]V_K^T

===============================================================================
DUAS MEDIDAS DISTINTAS
===============================================================================

1. ERRO INTERNO DO SMOOTH

    RE_S_INTERNAL =
        ||H_test^S - Hhat_S||_F
        /
        ||H_test^S - mu_S||_F

Mede quão bem o espaço suavizado futuro é representado.

2. ERRO END-TO-END CONTRA RAW

    RE_S_RAW =
        ||H_test - Hhat_S||_F
        /
        ||H_test - mu_R||_F

Este é o confronto justo com RAW.

Ele incorpora:

    smoothing distortion
    +
    truncamento PCA
    +
    portabilidade temporal.

Logo comparamos diretamente:

    RE_RAW
    versus
    RE_S_RAW

===============================================================================
IMPORTANTE
===============================================================================

NÃO:
- usa target;
- faz clipping;
- corrige negativos;
- escolhe smoothing;
- escolhe lambda novo;
- escolhe janela final;
- congela K;
- adiciona feature ao modelo;
- congela folds;
- treina classificador;
- cria Silver;
- modifica RAW.

EDF=0.75 / 30d é apenas CHALLENGER pré-especificado pelos experimentos
anteriores.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd

from scipy.linalg import subspace_angles


# =============================================================================
# PATHS
# =============================================================================

PROJECT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
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


# -----------------------------------------------------------------------------
# PREVIOUS STEPS
# -----------------------------------------------------------------------------

C33C_SUMMARY = (
    OUT
    / "02z_functional_temporal_portability_summary.json"
)

D1_SUMMARY = (
    OUT
    / "03e_smoothing_sensitivity_summary.json"
)

D11_SUMMARY = (
    OUT
    / "03i_negative_smoothing_summary.json"
)

D1_OPERATOR = (
    OUT
    / "03b_smoothing_operator_grid.csv"
)

ORDER_INDEX = (
    OUT
    / "02a_functional_pit_order_index.csv"
)

SUPPORT = (
    OUT
    / "02l_history_support_by_order.csv"
)

VOLUME_PATH = (
    ART
    / "02g_purchase_volume_curve_90d.npy"
)

FREIGHT_PATH = (
    ART
    / "02h_purchase_freight_curve_90d.npy"
)


# =============================================================================
# OUTPUTS
# =============================================================================

P_FOLDS = (
    OUT
    / "03k_raw_vs_smooth_temporal_folds.csv"
)

P_SUMMARY_TABLE = (
    OUT
    / "03l_raw_vs_smooth_temporal_summary.csv"
)

P_PAIRED = (
    OUT
    / "03m_raw_vs_smooth_paired_comparison.csv"
)

P_VALIDATION = (
    OUT
    / "03n_raw_vs_smooth_validation.csv"
)

P_SUMMARY_JSON = (
    OUT
    / "03o_raw_vs_smooth_summary.json"
)

P_REPORT = (
    OUT
    / "03p_raw_vs_smooth_report.txt"
)


# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW = 30

EDF_RATIO = 0.75

EXPECTED_ROWS = 96470

MIN_TRAIN_ROWS = 5000

MIN_TEST_ROWS = 500

CHANNELS = [
    "purchase_volume",
    "purchase_freight",
]

NEGATIVE_TOL = 1e-10


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


def parse_bool(
    series
):
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.to_numpy(
            dtype=bool
        )

    x = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not x.isin(
        [
            "true",
            "false",
        ]
    ).all():

        raise RuntimeError(
            f"Boolean inválido em {series.name}"
        )

    return (
        x.eq("true")
        .to_numpy(
            dtype=bool
        )
    )


def second_difference_matrix(
    length
):
    D = np.zeros(
        (
            length - 2,
            length
        ),
        dtype=np.float64
    )

    i = np.arange(
        length - 2
    )

    D[
        i,
        i
    ] = 1.0

    D[
        i,
        i + 1
    ] = -2.0

    D[
        i,
        i + 2
    ] = 1.0

    return D


def smoother_matrix(
    length,
    lam
):
    D2 = second_difference_matrix(
        length
    )

    penalty = (
        D2.T
        @
        D2
    )

    A = (
        np.eye(
            length,
            dtype=np.float64
        )
        +
        float(lam)
        *
        penalty
    )

    S = np.linalg.solve(
        A,
        np.eye(
            length,
            dtype=np.float64
        )
    )

    # Remover apenas assimetria numérica microscópica.
    return (
        S + S.T
    ) / 2.0


def first_k(
    cumulative,
    level
):
    idx = np.flatnonzero(
        cumulative >= level
    )

    if len(idx) == 0:
        return len(
            cumulative
        )

    return int(
        idx[0] + 1
    )


def fit_pca_basis(
    matrix
):
    """
    PCA por SVD.

    Centering aprendido SOMENTE no conjunto fornecido.
    Não há scaling ponto a ponto.
    """

    A = np.asarray(
        matrix,
        dtype=np.float64
    )

    mean = A.mean(
        axis=0
    )

    centered = (
        A - mean
    )

    _, singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False
    )

    energy = (
        singular_values ** 2
    )

    total = float(
        energy.sum()
    )

    if (
        total <= 0
        or
        not np.isfinite(
            total
        )
    ):
        raise RuntimeError(
            "Energia PCA inválida."
        )

    ratio = (
        energy
        /
        total
    )

    cumulative = np.cumsum(
        ratio
    )

    k80 = first_k(
        cumulative,
        0.80
    )

    k90 = first_k(
        cumulative,
        0.90
    )

    k95 = first_k(
        cumulative,
        0.95
    )

    return {
        "mean":
            mean,

        "singular_values":
            singular_values,

        "vt":
            vt,

        "cumulative":
            cumulative,

        "k80":
            k80,

        "k90":
            k90,

        "k95":
            k95,

        "pc1_energy_pct":
            float(
                100.0
                *
                ratio[0]
            ),
    }


def reconstruct(
    matrix,
    mean,
    vt,
    k
):
    A = np.asarray(
        matrix,
        dtype=np.float64
    )

    centered = (
        A - mean
    )

    basis = (
        vt[
            :k,
            :
        ]
        .T
    )

    scores = (
        centered
        @
        basis
    )

    reconstructed = (
        mean
        +
        scores
        @
        basis.T
    )

    return reconstructed


def relative_error(
    observed,
    reconstructed,
    reference_mean
):
    A = np.asarray(
        observed,
        dtype=np.float64
    )

    R = np.asarray(
        reconstructed,
        dtype=np.float64
    )

    mu = np.asarray(
        reference_mean,
        dtype=np.float64
    )

    numerator = float(
        np.linalg.norm(
            A - R,
            ord="fro"
        )
    )

    denominator = float(
        np.linalg.norm(
            A - mu,
            ord="fro"
        )
    )

    if denominator == 0:
        return 0.0

    return (
        numerator
        /
        denominator
    )


def smoothing_distortion(
    raw,
    smooth,
    raw_train_mean
):
    """
    Distorção do smoothing em relação à mesma escala
    usada pelo benchmark RAW.
    """

    numerator = float(
        np.linalg.norm(
            smooth - raw,
            ord="fro"
        )
    )

    denominator = float(
        np.linalg.norm(
            raw
            -
            raw_train_mean,
            ord="fro"
        )
    )

    if denominator == 0:
        return 0.0

    return (
        numerator
        /
        denominator
    )


def train_internal_error(
    cumulative,
    k
):
    explained = float(
        cumulative[
            k - 1
        ]
    )

    return float(
        np.sqrt(
            max(
                0.0,
                1.0
                -
                explained
            )
        )
    )


def basis_temporal_angles(
    train_basis_info,
    test_matrix
):
    """
    Mede mudança geométrica entre base do passado
    e base que seria obtida APENAS para diagnóstico
    no mês futuro.

    A base TEST nunca participa da reconstrução/predição.
    """

    test_basis_info = fit_pca_basis(
        test_matrix
    )

    k = min(
        int(
            train_basis_info[
                "k90"
            ]
        ),
        int(
            test_basis_info[
                "k90"
            ]
        )
    )

    A = (
        train_basis_info[
            "vt"
        ][
            :k,
            :
        ]
        .T
    )

    B = (
        test_basis_info[
            "vt"
        ][
            :k,
            :
        ]
        .T
    )

    angles = np.degrees(
        subspace_angles(
            A,
            B
        )
    )

    return {
        "dimension":
            int(
                k
            ),

        "mean_angle_deg":
            float(
                np.mean(
                    angles
                )
            ),

        "max_angle_deg":
            float(
                np.max(
                    angles
                )
            ),
    }


# =============================================================================
# START
# =============================================================================

start = time.perf_counter()

print()
print("=" * 120)
print("MODEL 01.0-C3.3-D2 — RAW 30D vs SMOOTH 30D EDF=0.75")
print("=" * 120)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required = [
    (
        C33C_SUMMARY,
        "C3.3-C summary"
    ),
    (
        D1_SUMMARY,
        "D1 summary"
    ),
    (
        D11_SUMMARY,
        "D1.1 summary"
    ),
    (
        D1_OPERATOR,
        "D1 operator grid"
    ),
    (
        ORDER_INDEX,
        "order index"
    ),
    (
        SUPPORT,
        "history support"
    ),
    (
        VOLUME_PATH,
        "volume curves"
    ),
    (
        FREIGHT_PATH,
        "freight curves"
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


def load_json(
    path
):
    with path.open(
        encoding="utf-8"
    ) as f:
        return json.load(
            f
        )


c33c = load_json(
    C33C_SUMMARY
)

d1 = load_json(
    D1_SUMMARY
)

d11 = load_json(
    D11_SUMMARY
)


for name, obj in [
    (
        "C3.3-C",
        c33c
    ),
    (
        "D1",
        d1
    ),
    (
        "D1.1",
        d11
    ),
]:

    if obj.get(
        "status"
    ) != "PASS":

        raise SystemExit(
            f"[STOP] {name} não está PASS."
        )

    if obj.get(
        "validation_failures"
    ) != 0:

        raise SystemExit(
            f"[STOP] {name} possui validation failures."
        )


if c33c.get(
    "target_used"
) is not False:

    raise SystemExit(
        "[STOP] C3.3-C target_used != false."
    )


if d1.get(
    "target_used"
) is not False:

    raise SystemExit(
        "[STOP] D1 target_used != false."
    )


if d11.get(
    "target_used"
) is not False:

    raise SystemExit(
        "[STOP] D1.1 target_used != false."
    )


if d11.get(
    "negative_values_clipped"
) is not False:

    raise SystemExit(
        "[STOP] D1.1 indica clipping."
    )


print()
print("[PASS] Cadeia C3.3-C -> D1 -> D1.1 validada.")


# =============================================================================
# 2. RECOVER THE PRE-SPECIFIED SMOOTHER
# =============================================================================

operator = pd.read_csv(
    D1_OPERATOR
)


candidate = operator.loc[
    (
        operator[
            "window_days"
        ]
        .eq(
            WINDOW
        )
    )
    &
    (
        np.isclose(
            operator[
                "edf_ratio_target"
            ],
            EDF_RATIO
        )
    )
].copy()


if len(
    candidate
) != 1:

    raise RuntimeError(
        "Esperávamos exatamente uma configuração "
        "30d / EDF 0.75."
    )


candidate = candidate.iloc[
    0
]


LAMBDA = float(
    candidate[
        "lambda"
    ]
)

ACTUAL_EDF = float(
    candidate[
        "actual_effective_df"
    ]
)


if not np.isclose(
    ACTUAL_EDF,
    22.5,
    atol=1e-8,
    rtol=0
):

    raise RuntimeError(
        f"EDF inesperado: {ACTUAL_EDF}"
    )


S = smoother_matrix(
    WINDOW,
    LAMBDA
)


print()
print("=" * 120)
print("FIXED CHALLENGER")
print("=" * 120)

print(
    "Window             :",
    WINDOW
)

print(
    "EDF ratio          :",
    EDF_RATIO
)

print(
    "Effective df       :",
    ACTUAL_EDF
)

print(
    "Lambda             :",
    f"{LAMBDA:.12g}"
)

print(
    "Clipping           : NÃO"
)


# =============================================================================
# 3. LOAD METADATA
# =============================================================================

index = pd.read_csv(
    ORDER_INDEX,
    parse_dates=[
        "order_purchase_timestamp"
    ]
)


support = pd.read_csv(
    SUPPORT
)


if len(
    index
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Order index rows={len(index)}"
    )


if len(
    support
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Support rows={len(support)}"
    )


if not np.array_equal(
    index[
        "curve_row_index"
    ].to_numpy(
        dtype=np.int64
    ),
    support[
        "curve_row_index"
    ].to_numpy(
        dtype=np.int64
    )
):

    raise RuntimeError(
        "curve_row_index desalinhado."
    )


full_support = parse_bool(
    support[
        "full_support_30d"
    ]
)


meta = index[
    [
        "curve_row_index",
        "order_id",
        "order_purchase_timestamp",
        "purchase_month",
    ]
].copy()


meta[
    "full_support_30d"
] = full_support


months = sorted(
    meta[
        "purchase_month"
    ]
    .dropna()
    .unique()
    .tolist()
)


print()
print(
    "Months:",
    len(
        months
    ),
    "|",
    months[
        0
    ],
    "->",
    months[
        -1
    ]
)


# =============================================================================
# 4. LOAD CURVES
# =============================================================================

volume = np.load(
    VOLUME_PATH,
    mmap_mode="r",
    allow_pickle=False
)

freight = np.load(
    FREIGHT_PATH,
    mmap_mode="r",
    allow_pickle=False
)


if volume.shape != (
    EXPECTED_ROWS,
    90
):

    raise RuntimeError(
        f"Volume shape={volume.shape}"
    )


if freight.shape != (
    EXPECTED_ROWS,
    90
):

    raise RuntimeError(
        f"Freight shape={freight.shape}"
    )


curves = {
    "purchase_volume":
        volume,

    "purchase_freight":
        freight,
}


# =============================================================================
# 5. PAIRED TEMPORAL AUDIT
# =============================================================================

fold_rows = []


print()
print("=" * 120)
print("PAIRED EXPANDING-HISTORY AUDIT")
print("=" * 120)


for channel in CHANNELS:

    curve = curves[
        channel
    ]


    for current_month in months:

        train_mask = (
            meta[
                "full_support_30d"
            ]
            &
            (
                meta[
                    "purchase_month"
                ]
                <
                current_month
            )
        ).to_numpy(
            dtype=bool
        )


        test_mask = (
            meta[
                "full_support_30d"
            ]
            &
            (
                meta[
                    "purchase_month"
                ]
                ==
                current_month
            )
        ).to_numpy(
            dtype=bool
        )


        n_train = int(
            train_mask.sum()
        )

        n_test = int(
            test_mask.sum()
        )


        if (
            n_train
            <
            MIN_TRAIN_ROWS
            or
            n_test
            <
            MIN_TEST_ROWS
        ):

            continue


        # ---------------------------------------------------------------------
        # SAME RAW DATA FOR BOTH ROUTES
        # ---------------------------------------------------------------------

        raw_train = np.asarray(
            curve[
                train_mask,
                :WINDOW
            ],
            dtype=np.float64
        )


        raw_test = np.asarray(
            curve[
                test_mask,
                :WINDOW
            ],
            dtype=np.float64
        )


        # ---------------------------------------------------------------------
        # RAW ROUTE
        # ---------------------------------------------------------------------

        raw_basis = fit_pca_basis(
            raw_train
        )


        raw_k90 = int(
            raw_basis[
                "k90"
            ]
        )


        raw_reconstruction = reconstruct(
            raw_test,
            raw_basis[
                "mean"
            ],
            raw_basis[
                "vt"
            ],
            raw_k90
        )


        raw_future_error = relative_error(
            raw_test,
            raw_reconstruction,
            raw_basis[
                "mean"
            ]
        )


        raw_train_error = train_internal_error(
            raw_basis[
                "cumulative"
            ],
            raw_k90
        )


        raw_inflation = (
            raw_future_error
            /
            raw_train_error
            if raw_train_error > 0
            else np.nan
        )


        raw_angles = basis_temporal_angles(
            raw_basis,
            raw_test
        )


        # ---------------------------------------------------------------------
        # SMOOTH ROUTE
        # ---------------------------------------------------------------------

        smooth_train = (
            raw_train
            @
            S
        )


        smooth_test = (
            raw_test
            @
            S
        )


        smooth_basis = fit_pca_basis(
            smooth_train
        )


        smooth_k90 = int(
            smooth_basis[
                "k90"
            ]
        )


        smooth_reconstruction = reconstruct(
            smooth_test,
            smooth_basis[
                "mean"
            ],
            smooth_basis[
                "vt"
            ],
            smooth_k90
        )


        # ---------------------------------------------------------------------
        # INTERNAL SMOOTH ERROR
        # ---------------------------------------------------------------------

        smooth_internal_future_error = relative_error(
            smooth_test,
            smooth_reconstruction,
            smooth_basis[
                "mean"
            ]
        )


        smooth_train_error = train_internal_error(
            smooth_basis[
                "cumulative"
            ],
            smooth_k90
        )


        smooth_internal_inflation = (
            smooth_internal_future_error
            /
            smooth_train_error
            if smooth_train_error > 0
            else np.nan
        )


        # ---------------------------------------------------------------------
        # END-TO-END ERROR AGAINST ORIGINAL RAW TEST CURVE
        #
        # Crucial fairness comparison.
        # SAME denominator as RAW benchmark.
        # ---------------------------------------------------------------------

        smooth_end_to_end_raw_error = relative_error(
            raw_test,
            smooth_reconstruction,
            raw_basis[
                "mean"
            ]
        )


        # ---------------------------------------------------------------------
        # SMOOTHING-ONLY DISTORTION
        # ---------------------------------------------------------------------

        smooth_distortion = smoothing_distortion(
            raw_test,
            smooth_test,
            raw_basis[
                "mean"
            ]
        )


        # ---------------------------------------------------------------------
        # TEMPORAL SUBSPACE CHANGE
        # ---------------------------------------------------------------------

        smooth_angles = basis_temporal_angles(
            smooth_basis,
            smooth_test
        )


        # ---------------------------------------------------------------------
        # NEGATIVE VALUES — NO CORRECTION
        # ---------------------------------------------------------------------

        negative_cells = int(
            (
                smooth_test
                <
                -NEGATIVE_TOL
            ).sum()
        )


        negative_cell_pct = float(
            100.0
            *
            negative_cells
            /
            smooth_test.size
        )


        affected_orders = int(
            (
                smooth_test
                <
                -NEGATIVE_TOL
            )
            .any(
                axis=1
            )
            .sum()
        )


        # ---------------------------------------------------------------------
        # PAIRED COMPARISON
        # ---------------------------------------------------------------------

        error_delta = float(
            smooth_end_to_end_raw_error
            -
            raw_future_error
        )


        error_ratio = (
            float(
                smooth_end_to_end_raw_error
                /
                raw_future_error
            )
            if raw_future_error > 0
            else np.nan
        )


        k_delta = int(
            smooth_k90
            -
            raw_k90
        )


        fold_rows.append({
            "channel":
                channel,

            "current_month":
                current_month,

            "train_orders":
                n_train,

            "test_orders":
                n_test,

            "train_first_month":
                str(
                    meta.loc[
                        train_mask,
                        "purchase_month"
                    ].min()
                ),

            "train_last_month":
                str(
                    meta.loc[
                        train_mask,
                        "purchase_month"
                    ].max()
                ),

            "window_days":
                WINDOW,

            "smooth_edf_ratio":
                EDF_RATIO,

            "smooth_lambda":
                LAMBDA,

            # RAW
            "raw_k80_train":
                int(
                    raw_basis[
                        "k80"
                    ]
                ),

            "raw_k90_train":
                raw_k90,

            "raw_k95_train":
                int(
                    raw_basis[
                        "k95"
                    ]
                ),

            "raw_train_relative_error_k90":
                raw_train_error,

            "raw_future_relative_error_k90":
                raw_future_error,

            "raw_future_error_inflation_k90":
                raw_inflation,

            "raw_pc1_energy_pct_train":
                float(
                    raw_basis[
                        "pc1_energy_pct"
                    ]
                ),

            "raw_basis_angle_dimension":
                int(
                    raw_angles[
                        "dimension"
                    ]
                ),

            "raw_basis_mean_angle_deg":
                float(
                    raw_angles[
                        "mean_angle_deg"
                    ]
                ),

            "raw_basis_max_angle_deg":
                float(
                    raw_angles[
                        "max_angle_deg"
                    ]
                ),

            # SMOOTH
            "smooth_k80_train":
                int(
                    smooth_basis[
                        "k80"
                    ]
                ),

            "smooth_k90_train":
                smooth_k90,

            "smooth_k95_train":
                int(
                    smooth_basis[
                        "k95"
                    ]
                ),

            "smooth_train_relative_error_k90":
                smooth_train_error,

            "smooth_internal_future_error_k90":
                smooth_internal_future_error,

            "smooth_internal_error_inflation_k90":
                smooth_internal_inflation,

            "smooth_end_to_end_raw_error_k90":
                smooth_end_to_end_raw_error,

            "smooth_test_distortion_from_raw":
                smooth_distortion,

            "smooth_pc1_energy_pct_train":
                float(
                    smooth_basis[
                        "pc1_energy_pct"
                    ]
                ),

            "smooth_basis_angle_dimension":
                int(
                    smooth_angles[
                        "dimension"
                    ]
                ),

            "smooth_basis_mean_angle_deg":
                float(
                    smooth_angles[
                        "mean_angle_deg"
                    ]
                ),

            "smooth_basis_max_angle_deg":
                float(
                    smooth_angles[
                        "max_angle_deg"
                    ]
                ),

            # NEGATIVE AUDIT
            "smooth_negative_cells_test":
                negative_cells,

            "smooth_negative_cell_pct_test":
                negative_cell_pct,

            "smooth_affected_orders_test":
                affected_orders,

            # PAIRED
            "k90_delta_smooth_minus_raw":
                k_delta,

            "end_to_end_error_delta_smooth_minus_raw":
                error_delta,

            "end_to_end_error_ratio_smooth_over_raw":
                error_ratio,

            "smooth_lower_end_to_end_error":
                bool(
                    smooth_end_to_end_raw_error
                    <
                    raw_future_error
                ),
        })


folds = pd.DataFrame(
    fold_rows
)


if folds.empty:

    raise RuntimeError(
        "Nenhum fold temporal elegível."
    )


folds.to_csv(
    P_FOLDS,
    index=False
)


# =============================================================================
# 6. SUMMARIES
# =============================================================================

summary_rows = []


for channel, g in folds.groupby(
    "channel",
    sort=True
):

    summary_rows.append({
        "channel":
            channel,

        "temporal_tests":
            int(
                len(
                    g
                )
            ),

        "first_test_month":
            str(
                g[
                    "current_month"
                ].min()
            ),

        "last_test_month":
            str(
                g[
                    "current_month"
                ].max()
            ),

        # DIMENSION
        "raw_k90_median":
            float(
                g[
                    "raw_k90_train"
                ].median()
            ),

        "raw_k90_min":
            int(
                g[
                    "raw_k90_train"
                ].min()
            ),

        "raw_k90_max":
            int(
                g[
                    "raw_k90_train"
                ].max()
            ),

        "smooth_k90_median":
            float(
                g[
                    "smooth_k90_train"
                ].median()
            ),

        "smooth_k90_min":
            int(
                g[
                    "smooth_k90_train"
                ].min()
            ),

        "smooth_k90_max":
            int(
                g[
                    "smooth_k90_train"
                ].max()
            ),

        "k90_delta_median":
            float(
                g[
                    "k90_delta_smooth_minus_raw"
                ].median()
            ),

        "k90_reduction_mean":
            float(
                (
                    g[
                        "raw_k90_train"
                    ]
                    -
                    g[
                        "smooth_k90_train"
                    ]
                ).mean()
            ),

        # RAW FUTURE
        "raw_future_error_mean":
            float(
                g[
                    "raw_future_relative_error_k90"
                ].mean()
            ),

        "raw_future_error_median":
            float(
                g[
                    "raw_future_relative_error_k90"
                ].median()
            ),

        "raw_future_error_p95":
            float(
                g[
                    "raw_future_relative_error_k90"
                ].quantile(
                    0.95
                )
            ),

        # SMOOTH INTERNAL
        "smooth_internal_future_error_mean":
            float(
                g[
                    "smooth_internal_future_error_k90"
                ].mean()
            ),

        "smooth_internal_future_error_median":
            float(
                g[
                    "smooth_internal_future_error_k90"
                ].median()
            ),

        # FAIR END-TO-END
        "smooth_end_to_end_raw_error_mean":
            float(
                g[
                    "smooth_end_to_end_raw_error_k90"
                ].mean()
            ),

        "smooth_end_to_end_raw_error_median":
            float(
                g[
                    "smooth_end_to_end_raw_error_k90"
                ].median()
            ),

        "smooth_end_to_end_raw_error_p95":
            float(
                g[
                    "smooth_end_to_end_raw_error_k90"
                ].quantile(
                    0.95
                )
            ),

        "paired_end_to_end_delta_mean":
            float(
                g[
                    "end_to_end_error_delta_smooth_minus_raw"
                ].mean()
            ),

        "paired_end_to_end_delta_median":
            float(
                g[
                    "end_to_end_error_delta_smooth_minus_raw"
                ].median()
            ),

        "paired_end_to_end_ratio_mean":
            float(
                g[
                    "end_to_end_error_ratio_smooth_over_raw"
                ].mean()
            ),

        "smooth_better_end_to_end_folds":
            int(
                g[
                    "smooth_lower_end_to_end_error"
                ].sum()
            ),

        "smooth_better_end_to_end_fold_pct":
            float(
                100.0
                *
                g[
                    "smooth_lower_end_to_end_error"
                ].mean()
            ),

        # SMOOTHING DISTORTION
        "smooth_test_distortion_mean":
            float(
                g[
                    "smooth_test_distortion_from_raw"
                ].mean()
            ),

        "smooth_test_distortion_p95":
            float(
                g[
                    "smooth_test_distortion_from_raw"
                ].quantile(
                    0.95
                )
            ),

        # TEMPORAL BASIS
        "raw_basis_mean_angle_deg_mean":
            float(
                g[
                    "raw_basis_mean_angle_deg"
                ].mean()
            ),

        "smooth_basis_mean_angle_deg_mean":
            float(
                g[
                    "smooth_basis_mean_angle_deg"
                ].mean()
            ),

        "raw_basis_max_angle_deg_max":
            float(
                g[
                    "raw_basis_max_angle_deg"
                ].max()
            ),

        "smooth_basis_max_angle_deg_max":
            float(
                g[
                    "smooth_basis_max_angle_deg"
                ].max()
            ),

        # NEGATIVITY
        "smooth_negative_cell_pct_mean":
            float(
                g[
                    "smooth_negative_cell_pct_test"
                ].mean()
            ),

        "smooth_negative_cell_pct_max":
            float(
                g[
                    "smooth_negative_cell_pct_test"
                ].max()
            ),
    })


summary_table = pd.DataFrame(
    summary_rows
)

summary_table.to_csv(
    P_SUMMARY_TABLE,
    index=False
)


# =============================================================================
# 7. PAIRED MONTH-BY-MONTH VIEW
# =============================================================================

paired_cols = [
    "channel",
    "current_month",
    "train_orders",
    "test_orders",
    "raw_k90_train",
    "smooth_k90_train",
    "k90_delta_smooth_minus_raw",
    "raw_future_relative_error_k90",
    "smooth_internal_future_error_k90",
    "smooth_end_to_end_raw_error_k90",
    "smooth_test_distortion_from_raw",
    "end_to_end_error_delta_smooth_minus_raw",
    "end_to_end_error_ratio_smooth_over_raw",
    "smooth_lower_end_to_end_error",
    "raw_basis_mean_angle_deg",
    "smooth_basis_mean_angle_deg",
    "smooth_negative_cell_pct_test",
]


paired = (
    folds[
        paired_cols
    ]
    .sort_values(
        [
            "channel",
            "current_month"
        ]
    )
    .reset_index(
        drop=True
    )
)


paired.to_csv(
    P_PAIRED,
    index=False
)


# =============================================================================
# 8. VALIDATION
# =============================================================================

checks = []


add_check(
    checks,
    "c33c_pass",
    c33c.get(
        "status"
    ) == "PASS",
    c33c.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "d1_pass",
    d1.get(
        "status"
    ) == "PASS",
    d1.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "d11_pass",
    d11.get(
        "status"
    ) == "PASS",
    d11.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "candidate_window_30",
    WINDOW == 30,
    WINDOW,
    30
)


add_check(
    checks,
    "candidate_edf_075",
    np.isclose(
        EDF_RATIO,
        0.75
    ),
    EDF_RATIO,
    0.75
)


add_check(
    checks,
    "candidate_effective_df",
    np.isclose(
        ACTUAL_EDF,
        22.5,
        atol=1e-8,
        rtol=0
    ),
    ACTUAL_EDF,
    22.5
)


add_check(
    checks,
    "paired_folds_exist",
    len(
        folds
    ) > 0,
    len(
        folds
    ),
    ">0"
)


expected_channels = set(
    CHANNELS
)

actual_channels = set(
    folds[
        "channel"
    ].unique()
)


add_check(
    checks,
    "both_channels_present",
    actual_channels
    ==
    expected_channels,
    str(
        sorted(
            actual_channels
        )
    ),
    str(
        sorted(
            expected_channels
        )
    )
)


temporal_bad = int(
    (
        folds[
            "train_last_month"
        ]
        >=
        folds[
            "current_month"
        ]
    ).sum()
)


add_check(
    checks,
    "all_train_before_test",
    temporal_bad == 0,
    temporal_bad,
    0
)


# Same number of temporal test folds for both channels.
counts = (
    folds.groupby(
        "channel"
    )
    .size()
)


same_fold_count = (
    counts.nunique()
    ==
    1
)


add_check(
    checks,
    "same_fold_count_between_channels",
    same_fold_count,
    str(
        counts.to_dict()
    ),
    "equal"
)


# Expected 17 based on C3.3-C 30d result.
fold_count_each = int(
    counts.iloc[
        0
    ]
)


add_check(
    checks,
    "expected_17_temporal_tests_per_channel",
    fold_count_each == 17,
    fold_count_each,
    17
)


finite_cols = [
    "raw_train_relative_error_k90",
    "raw_future_relative_error_k90",
    "raw_future_error_inflation_k90",
    "smooth_train_relative_error_k90",
    "smooth_internal_future_error_k90",
    "smooth_internal_error_inflation_k90",
    "smooth_end_to_end_raw_error_k90",
    "smooth_test_distortion_from_raw",
    "end_to_end_error_delta_smooth_minus_raw",
    "end_to_end_error_ratio_smooth_over_raw",
    "raw_basis_mean_angle_deg",
    "raw_basis_max_angle_deg",
    "smooth_basis_mean_angle_deg",
    "smooth_basis_max_angle_deg",
]


nonfinite = int(
    (
        ~np.isfinite(
            folds[
                finite_cols
            ].to_numpy(
                dtype=float
            )
        )
    ).sum()
)


add_check(
    checks,
    "all_core_metrics_finite",
    nonfinite == 0,
    nonfinite,
    0
)


negative_error_values = int(
    (
        folds[
            [
                "raw_future_relative_error_k90",
                "smooth_internal_future_error_k90",
                "smooth_end_to_end_raw_error_k90",
                "smooth_test_distortion_from_raw",
            ]
        ]
        <
        0
    )
    .sum()
    .sum()
)


add_check(
    checks,
    "no_negative_error_metrics",
    negative_error_values == 0,
    negative_error_values,
    0
)


k_bad = int(
    (
        ~(
            (
                folds[
                    "raw_k80_train"
                ]
                <=
                folds[
                    "raw_k90_train"
                ]
            )
            &
            (
                folds[
                    "raw_k90_train"
                ]
                <=
                folds[
                    "raw_k95_train"
                ]
            )
            &
            (
                folds[
                    "raw_k95_train"
                ]
                <=
                WINDOW
            )
            &
            (
                folds[
                    "smooth_k80_train"
                ]
                <=
                folds[
                    "smooth_k90_train"
                ]
            )
            &
            (
                folds[
                    "smooth_k90_train"
                ]
                <=
                folds[
                    "smooth_k95_train"
                ]
            )
            &
            (
                folds[
                    "smooth_k95_train"
                ]
                <=
                WINDOW
            )
        )
    ).sum()
)


add_check(
    checks,
    "k_order_valid",
    k_bad == 0,
    k_bad,
    0
)


# No clipping:
# negative values should still be observable somewhere
# because D1.1 already proved their existence.
negative_total = int(
    folds[
        "smooth_negative_cells_test"
    ].sum()
)


add_check(
    checks,
    "negative_values_preserved_not_clipped",
    negative_total > 0,
    negative_total,
    ">0"
)


add_check(
    checks,
    "target_not_used",
    True,
    False,
    False
)


validation = pd.DataFrame(
    checks
)

validation.to_csv(
    P_VALIDATION,
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
# 9. SUMMARY JSON
# =============================================================================

runtime = float(
    time.perf_counter()
    -
    start
)


summary_json = {
    "step":
        "MODEL_01_0_C3_3D2_RAW_VS_SMOOTH_TEMPORAL",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "comparison":
        "RAW_30D_VS_SMOOTH_30D_EDF_0_75",

    "comparison_design":
        "PAIRED_EXPANDING_HISTORY_OUT_OF_TIME",

    "window_days":
        WINDOW,

    "smooth_edf_ratio":
        EDF_RATIO,

    "smooth_effective_df":
        ACTUAL_EDF,

    "smooth_lambda":
        LAMBDA,

    "channels":
        CHANNELS,

    "temporal_fold_rows":
        int(
            len(
                folds
            )
        ),

    "temporal_tests_per_channel":
        int(
            fold_count_each
        ),

    "same_orders_per_pair":
        True,

    "same_temporal_boundaries_per_pair":
        True,

    "pca_fit_on_past_only":
        True,

    "k90_fit_on_past_only":
        True,

    "fair_end_to_end_raw_error_computed":
        True,

    "target_used":
        False,

    "negative_values_clipped":
        False,

    "smoothing_selected":
        False,

    "lambda_selected_as_final":
        False,

    "window_selected_as_final":
        False,

    "component_count_selected":
        False,

    "functional_feature_created":
        False,

    "functional_module_unlocked":
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


P_SUMMARY_JSON.write_text(
    json.dumps(
        summary_json,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 10. REPORT
# =============================================================================

lines = [
    "=" * 112,
    "MODEL 01.0-C3.3-D2 — RAW 30D vs SMOOTH 30D EDF=0.75",
    "=" * 112,
    "",
    f"STATUS                         : {summary_json['status']}",
    f"WINDOW                         : {WINDOW} days",
    f"SMOOTH EDF RATIO               : {EDF_RATIO}",
    f"SMOOTH EFFECTIVE DF            : {ACTUAL_EDF}",
    f"LAMBDA                         : {LAMBDA:.12g}",
    "",
    "DESIGN",
    "-" * 112,
    "Paired expanding-history temporal comparison.",
    "Same orders in RAW and SMOOTH routes.",
    "PCA learned only from past curves.",
    "K90 learned only from past curves.",
    "Target not used.",
    "",
    "KEY FAIRNESS METRIC",
    "-" * 112,
    "RAW route is reconstructed against RAW future curves.",
    "SMOOTH route is ALSO evaluated against RAW future curves.",
    "Therefore smoothing distortion is not hidden by evaluating only in smoothed space.",
    "",
]


for _, row in summary_table.iterrows():

    lines += [
        f"{row['channel']}",
        (
            f"  temporal tests              : "
            f"{int(row['temporal_tests'])}"
        ),
        (
            f"  RAW K90 median              : "
            f"{row['raw_k90_median']:.2f}"
        ),
        (
            f"  SMOOTH K90 median           : "
            f"{row['smooth_k90_median']:.2f}"
        ),
        (
            f"  mean K90 reduction          : "
            f"{row['k90_reduction_mean']:.4f}"
        ),
        (
            f"  RAW future error mean       : "
            f"{row['raw_future_error_mean']:.6f}"
        ),
        (
            f"  SMOOTH internal error mean  : "
            f"{row['smooth_internal_future_error_mean']:.6f}"
        ),
        (
            f"  SMOOTH end-to-end RAW mean  : "
            f"{row['smooth_end_to_end_raw_error_mean']:.6f}"
        ),
        (
            f"  paired delta mean           : "
            f"{row['paired_end_to_end_delta_mean']:.6f}"
        ),
        (
            f"  smooth lower error folds    : "
            f"{int(row['smooth_better_end_to_end_folds'])}"
            f"/{int(row['temporal_tests'])}"
        ),
        "",
    ]


lines += [
    "IMPORTANT",
    "-" * 112,
    "No automatic winner is declared.",
    "No smoothing configuration is promoted.",
    "No clipping is applied.",
    "No target is used.",
    "No final K is selected.",
    "No fold is frozen.",
    "No model is trained.",
    "No Silver is created.",
    "RAW is not modified.",
    "",
    f"VALIDATION FAILURES            : {failures}",
    f"RUNTIME                        : {runtime:.3f}s",
    "=" * 112,
]


P_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 11. PRINT
# =============================================================================

print()
print("=" * 120)
print("PAIRED TEMPORAL SUMMARY")
print("=" * 120)


display_cols = [
    "channel",
    "temporal_tests",
    "raw_k90_median",
    "raw_k90_min",
    "raw_k90_max",
    "smooth_k90_median",
    "smooth_k90_min",
    "smooth_k90_max",
    "k90_reduction_mean",
    "raw_future_error_mean",
    "raw_future_error_p95",
    "smooth_internal_future_error_mean",
    "smooth_end_to_end_raw_error_mean",
    "smooth_end_to_end_raw_error_p95",
    "paired_end_to_end_delta_mean",
    "paired_end_to_end_ratio_mean",
    "smooth_better_end_to_end_folds",
    "smooth_better_end_to_end_fold_pct",
    "smooth_test_distortion_mean",
    "smooth_test_distortion_p95",
    "raw_basis_mean_angle_deg_mean",
    "smooth_basis_mean_angle_deg_mean",
    "smooth_negative_cell_pct_mean",
]


print(
    summary_table[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


print()
print("=" * 120)
print("MONTH-BY-MONTH PAIRED COMPARISON")
print("=" * 120)


paired_display = [
    "channel",
    "current_month",
    "train_orders",
    "test_orders",
    "raw_k90_train",
    "smooth_k90_train",
    "raw_future_relative_error_k90",
    "smooth_internal_future_error_k90",
    "smooth_end_to_end_raw_error_k90",
    "smooth_test_distortion_from_raw",
    "end_to_end_error_delta_smooth_minus_raw",
    "end_to_end_error_ratio_smooth_over_raw",
    "smooth_lower_end_to_end_error",
]


print(
    paired[
        paired_display
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
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
print("RESULTADO C3.3-D2")
print("=" * 120)


print(
    "FAILURES                       =",
    failures
)

print(
    "TEMPORAL FOLD ROWS             =",
    len(
        folds
    )
)

print(
    "TEMPORAL TESTS / CHANNEL       =",
    fold_count_each
)

print(
    "WINDOW                         =",
    WINDOW
)

print(
    "EDF RATIO                      =",
    EDF_RATIO
)

print(
    "LAMBDA                         =",
    LAMBDA
)


print()
print("TARGET USED                    = NÃO")
print("NEGATIVES CLIPPED              = NÃO")
print("SMOOTHING SELECTED             = NÃO")
print("FINAL LAMBDA SELECTED          = NÃO")
print("FINAL WINDOW SELECTED          = NÃO")
print("FINAL K SELECTED               = NÃO")
print("FEATURE CREATED                = NÃO")
print("FUNCTIONAL MODULE UNLOCKED     = NÃO")
print("FOLDS FROZEN                   = NÃO")
print("CLASSIFIER TRAINED             = NÃO")
print("SILVER CREATED                 = NÃO")
print("RAW MODIFIED                   = NÃO")


if failures:

    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.3-D2 paired temporal audit concluída."
)

print(
    "[PASS] Parar aqui antes de escolher RAW ou SMOOTH."
)

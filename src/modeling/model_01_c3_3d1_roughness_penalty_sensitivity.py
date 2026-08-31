#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-D1
ROUGHNESS-PENALTY SMOOTHING SENSITIVITY
===============================================================================

OBJETIVO
--------
Testar se uma regularização funcional de segunda diferença:

    1. reduz rugosidade;
    2. aumenta compressibilidade espectral;
    3. preserva propriedades fundamentais das curvas;
    4. não introduz distorção excessiva;
    5. não gera valores negativos relevantes.

Nenhum target é usado.

Não escolhemos lambda diretamente.

Usamos effective degrees of freedom:

    df(lambda) =
        trace[(I + lambda D2^T D2)^(-1)]

e avaliamos razões-alvo:

    1.00
    0.75
    0.50
    0.25

da dimensão original.

===============================================================================
FORMULAÇÃO
===============================================================================

Para uma curva h in R^L:

    z_lambda =
        argmin_z {
            ||h-z||_2^2
            +
            lambda ||D2 z||_2^2
        }

onde:

    D2 z =
        segunda diferença discreta.

Solução:

    S_lambda =
        (I + lambda D2^T D2)^(-1)

    z_lambda =
        S_lambda h

Para curvas armazenadas por linha:

    H_lambda =
        H S_lambda

===============================================================================
IMPORTANTE
===============================================================================

NÃO:
- escolhe lambda;
- escolhe janela;
- escolhe K;
- usa target;
- cria FPCA scores;
- adiciona feature ao modelo;
- congela folds;
- treina classificador;
- cria Silver;
- modifica RAW.
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

C33C_SUMMARY = (
    OUT
    / "02z_functional_temporal_portability_summary.json"
)

SUPPORT_PATH = (
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

P_OPERATOR = (
    OUT
    / "03b_smoothing_operator_grid.csv"
)

P_RESULTS = (
    OUT
    / "03c_smoothing_geometry_sensitivity.csv"
)

P_VALIDATION = (
    OUT
    / "03d_smoothing_sensitivity_validation.csv"
)

P_SUMMARY = (
    OUT
    / "03e_smoothing_sensitivity_summary.json"
)

P_REPORT = (
    OUT
    / "03f_smoothing_sensitivity_report.txt"
)


# =============================================================================
# CONSTANTS
# =============================================================================

WINDOWS = [
    30,
    60,
    90,
]

CHANNELS = [
    "purchase_volume",
    "purchase_freight",
]

EDF_RATIOS = [
    1.00,
    0.75,
    0.50,
    0.25,
]

EXPECTED_ROWS = 96470

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


def second_difference_matrix(
    length
):
    """
    D2 @ z produz:

        z[t] - 2 z[t+1] + z[t+2]
    """

    D = np.zeros(
        (
            length - 2,
            length
        ),
        dtype=np.float64
    )

    rows = np.arange(
        length - 2
    )

    D[
        rows,
        rows
    ] = 1.0

    D[
        rows,
        rows + 1
    ] = -2.0

    D[
        rows,
        rows + 2
    ] = 1.0

    return D


def effective_df_from_eigenvalues(
    eigenvalues,
    lam
):
    return float(
        np.sum(
            1.0
            /
            (
                1.0
                +
                lam
                *
                eigenvalues
            )
        )
    )


def lambda_for_target_df(
    eigenvalues,
    target_df
):
    """
    Resolve aproximadamente:

        df(lambda) = target_df

    sem scipy.optimize.

    A penalização D2 possui nullspace de dimensão 2,
    portanto df(lambda) -> 2 quando lambda -> infinito.
    """

    length = len(
        eigenvalues
    )

    if target_df >= length:
        return 0.0

    if target_df <= 2:
        raise ValueError(
            "target_df deve ser > 2."
        )

    lo = 0.0
    hi = 1.0

    while (
        effective_df_from_eigenvalues(
            eigenvalues,
            hi
        )
        >
        target_df
    ):

        hi *= 10.0

        if hi > 1e18:
            raise RuntimeError(
                "Não foi possível encontrar limite superior para lambda."
            )

    # Bisseção.
    for _ in range(
        100
    ):

        mid = (
            lo + hi
        ) / 2.0

        value = (
            effective_df_from_eigenvalues(
                eigenvalues,
                mid
            )
        )

        if value > target_df:
            lo = mid
        else:
            hi = mid

    return float(
        (
            lo + hi
        )
        /
        2.0
    )


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
        lam
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

    # Resolver numericamente pode produzir assimetria
    # microscópica. Simetrizamos sem mudar o operador
    # matemático.
    S = (
        S
        +
        S.T
    ) / 2.0

    return (
        S,
        D2,
        penalty
    )


def spectral_geometry(
    matrix
):
    A = np.asarray(
        matrix,
        dtype=np.float64
    )

    centered = (
        A
        -
        A.mean(
            axis=0
        )
    )

    s = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False
    )

    energy = (
        s ** 2
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
            "Energia espectral inválida."
        )

    ratio = (
        energy
        /
        total
    )

    cumulative = np.cumsum(
        ratio
    )

    def first_k(
        level
    ):
        idx = np.flatnonzero(
            cumulative >= level
        )

        return int(
            idx[0] + 1
        )

    return {
        "rank":
            int(
                np.linalg.matrix_rank(
                    centered
                )
            ),

        "pc1_energy_pct":
            float(
                100.0
                *
                ratio[0]
            ),

        "k80":
            first_k(
                0.80
            ),

        "k90":
            first_k(
                0.90
            ),

        "k95":
            first_k(
                0.95
            ),

        "k99":
            first_k(
                0.99
            ),

        "relative_error_k90":
            float(
                np.sqrt(
                    max(
                        0.0,
                        1.0
                        -
                        cumulative[
                            first_k(
                                0.90
                            )
                            -
                            1
                        ]
                    )
                )
            ),
    }


def adjacent_correlation_mean(
    matrix
):
    A = np.asarray(
        matrix,
        dtype=np.float64
    )

    correlations = []

    for j in range(
        A.shape[1] - 1
    ):

        x = A[
            :,
            j
        ]

        y = A[
            :,
            j + 1
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
            continue

        correlations.append(
            float(
                np.corrcoef(
                    x,
                    y
                )[0, 1]
            )
        )

    if not correlations:
        return np.nan

    return float(
        np.mean(
            correlations
        )
    )


# =============================================================================
# START
# =============================================================================

start = time.perf_counter()

print()
print("=" * 118)
print("MODEL 01.0-C3.3-D1 — ROUGHNESS-PENALTY SENSITIVITY")
print("=" * 118)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required = [
    (
        C33C_SUMMARY,
        "C3.3-C summary"
    ),
    (
        SUPPORT_PATH,
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


with C33C_SUMMARY.open(
    encoding="utf-8"
) as f:

    c33c = json.load(f)


if c33c.get(
    "status"
) != "PASS":

    raise SystemExit(
        "[STOP] C3.3-C não está PASS."
    )


if c33c.get(
    "validation_failures"
) != 0:

    raise SystemExit(
        "[STOP] C3.3-C possui falhas."
    )


if c33c.get(
    "target_used"
) is not False:

    raise SystemExit(
        "[STOP] C3.3-C registrou target_used != false."
    )


print()
print("[PASS] C3.3-C validado.")


# =============================================================================
# 2. SUPPORT
# =============================================================================

support = pd.read_csv(
    SUPPORT_PATH
)


if len(
    support
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Support possui {len(support)} linhas; "
        f"esperado {EXPECTED_ROWS}."
    )


def parse_boolean_series(
    series
):
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.to_numpy(
            dtype=bool
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid = normalized.isin(
        [
            "true",
            "false",
        ]
    )

    if not valid.all():
        raise RuntimeError(
            f"Boolean inválido em {series.name}."
        )

    return (
        normalized
        .eq("true")
        .to_numpy(
            dtype=bool
        )
    )


support_masks = {}

for window in WINDOWS:

    col = (
        f"full_support_{window}d"
    )

    if col not in support.columns:
        raise RuntimeError(
            f"{col} ausente."
        )

    support_masks[
        window
    ] = parse_boolean_series(
        support[
            col
        ]
    )


# =============================================================================
# 3. CURVES
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
        f"Volume shape: {volume.shape}"
    )


if freight.shape != (
    EXPECTED_ROWS,
    90
):
    raise RuntimeError(
        f"Freight shape: {freight.shape}"
    )


curves = {
    "purchase_volume":
        volume,

    "purchase_freight":
        freight,
}


# =============================================================================
# 4. BUILD COMPARABLE EDF GRID
# =============================================================================

operator_rows = []
operator_cache = {}


for window in WINDOWS:

    D2 = second_difference_matrix(
        window
    )

    penalty = (
        D2.T
        @
        D2
    )

    eigenvalues = np.linalg.eigvalsh(
        penalty
    )

    # Zera erros numéricos negativos minúsculos.
    eigenvalues[
        np.abs(
            eigenvalues
        )
        <
        1e-12
    ] = 0.0


    for edf_ratio in EDF_RATIOS:

        target_df = float(
            edf_ratio
            *
            window
        )

        # Não permitimos target <= dimensão do nullspace.
        target_df = max(
            target_df,
            2.000001
        )


        lam = lambda_for_target_df(
            eigenvalues,
            target_df
        )


        S, D2_check, penalty_check = (
            smoother_matrix(
                window,
                lam
            )
        )


        actual_df = float(
            np.trace(
                S
            )
        )


        constant = np.ones(
            window,
            dtype=np.float64
        )

        linear = np.arange(
            window,
            dtype=np.float64
        )


        constant_error = float(
            np.max(
                np.abs(
                    S @ constant
                    -
                    constant
                )
            )
        )

        linear_error = float(
            np.max(
                np.abs(
                    S @ linear
                    -
                    linear
                )
            )
        )


        operator_rows.append({
            "window_days":
                window,

            "edf_ratio_target":
                edf_ratio,

            "target_effective_df":
                target_df,

            "lambda":
                lam,

            "actual_effective_df":
                actual_df,

            "actual_edf_ratio":
                float(
                    actual_df
                    /
                    window
                ),

            "penalty_rank":
                int(
                    np.linalg.matrix_rank(
                        penalty
                    )
                ),

            "penalty_nullity":
                int(
                    window
                    -
                    np.linalg.matrix_rank(
                        penalty
                    )
                ),

            "constant_reproduction_max_error":
                constant_error,

            "linear_reproduction_max_error":
                linear_error,
        })


        operator_cache[
            (
                window,
                edf_ratio
            )
        ] = (
            S,
            D2
        )


operator_df = pd.DataFrame(
    operator_rows
)

operator_df.to_csv(
    P_OPERATOR,
    index=False
)


# =============================================================================
# 5. SMOOTHING SENSITIVITY
# =============================================================================

result_rows = []


print()
print("=" * 118)
print("SMOOTHING GRID")
print("=" * 118)


for channel in CHANNELS:

    curve = curves[
        channel
    ]


    for window in WINDOWS:

        mask = support_masks[
            window
        ]

        raw = np.asarray(
            curve[
                mask,
                :window
            ],
            dtype=np.float64
        )


        if not np.isfinite(
            raw
        ).all():
            raise RuntimeError(
                f"NaN/Inf em {channel}/{window}."
            )


        raw_centered = (
            raw
            -
            raw.mean(
                axis=0
            )
        )

        raw_scale = float(
            np.linalg.norm(
                raw_centered,
                ord="fro"
            )
        )


        if raw_scale <= 0:
            raise RuntimeError(
                f"Variabilidade nula em {channel}/{window}."
            )


        raw_sum = raw.sum(
            axis=1
        )

        time_vector = np.arange(
            window,
            dtype=np.float64
        )

        raw_first_moment = (
            raw
            @
            time_vector
        )


        for edf_ratio in EDF_RATIOS:

            S, D2 = operator_cache[
                (
                    window,
                    edf_ratio
                )
            ]


            smoothed = (
                raw
                @
                S
            )


            if not np.isfinite(
                smoothed
            ).all():
                raise RuntimeError(
                    f"Smoothing não-finito: "
                    f"{channel}/{window}/{edf_ratio}"
                )


            distortion = float(
                np.linalg.norm(
                    smoothed
                    -
                    raw,
                    ord="fro"
                )
                /
                raw_scale
            )


            raw_rough = float(
                np.linalg.norm(
                    raw
                    @
                    D2.T,
                    ord="fro"
                )
            )


            smooth_rough = float(
                np.linalg.norm(
                    smoothed
                    @
                    D2.T,
                    ord="fro"
                )
            )


            roughness_ratio = (
                smooth_rough
                /
                raw_rough
                if raw_rough > 0
                else 0.0
            )


            smooth_sum = (
                smoothed.sum(
                    axis=1
                )
            )

            sum_den = np.maximum(
                1.0,
                np.abs(
                    raw_sum
                )
            )

            max_row_sum_relative_error = float(
                np.max(
                    np.abs(
                        smooth_sum
                        -
                        raw_sum
                    )
                    /
                    sum_den
                )
            )


            smooth_first_moment = (
                smoothed
                @
                time_vector
            )

            moment_den = np.maximum(
                1.0,
                np.abs(
                    raw_first_moment
                )
            )

            max_first_moment_relative_error = float(
                np.max(
                    np.abs(
                        smooth_first_moment
                        -
                        raw_first_moment
                    )
                    /
                    moment_den
                )
            )


            negative_mask = (
                smoothed
                <
                -NEGATIVE_TOL
            )

            negative_cells = int(
                negative_mask.sum()
            )

            negative_cell_pct = float(
                100.0
                *
                negative_cells
                /
                smoothed.size
            )


            minimum_value = float(
                smoothed.min()
            )


            geom = spectral_geometry(
                smoothed
            )


            adjacent_corr = (
                adjacent_correlation_mean(
                    smoothed
                )
            )


            op = operator_df.loc[
                (
                    operator_df[
                        "window_days"
                    ]
                    .eq(
                        window
                    )
                    &
                    np.isclose(
                        operator_df[
                            "edf_ratio_target"
                        ],
                        edf_ratio
                    )
                )
            ].iloc[
                0
            ]


            result_rows.append({
                "channel":
                    channel,

                "window_days":
                    window,

                "orders_full_support":
                    int(
                        mask.sum()
                    ),

                "edf_ratio_target":
                    edf_ratio,

                "lambda":
                    float(
                        op[
                            "lambda"
                        ]
                    ),

                "effective_df":
                    float(
                        op[
                            "actual_effective_df"
                        ]
                    ),

                "effective_df_ratio":
                    float(
                        op[
                            "actual_edf_ratio"
                        ]
                    ),

                "distortion_relative_to_centered_raw":
                    distortion,

                "roughness_ratio":
                    roughness_ratio,

                "roughness_reduction_pct":
                    float(
                        100.0
                        *
                        (
                            1.0
                            -
                            roughness_ratio
                        )
                    ),

                "max_row_sum_relative_error":
                    max_row_sum_relative_error,

                "max_first_moment_relative_error":
                    max_first_moment_relative_error,

                "negative_cells":
                    negative_cells,

                "negative_cell_pct":
                    negative_cell_pct,

                "minimum_smoothed_value":
                    minimum_value,

                "adjacent_lag_correlation_mean":
                    adjacent_corr,

                "numerical_rank":
                    geom[
                        "rank"
                    ],

                "pc1_energy_pct":
                    geom[
                        "pc1_energy_pct"
                    ],

                "k80":
                    geom[
                        "k80"
                    ],

                "k90":
                    geom[
                        "k90"
                    ],

                "k95":
                    geom[
                        "k95"
                    ],

                "k99":
                    geom[
                        "k99"
                    ],

                "relative_error_k90":
                    geom[
                        "relative_error_k90"
                    ],
            })


            print(
                f"{channel:17s} "
                f"{window:>2}d "
                f"EDF={edf_ratio:>4.2f} "
                f"lambda={float(op['lambda']):.6g} "
                f"dist={distortion:.4f} "
                f"rough={roughness_ratio:.4f} "
                f"neg={negative_cell_pct:.5f}% "
                f"K90={geom['k90']}"
            )


results = pd.DataFrame(
    result_rows
)

results.to_csv(
    P_RESULTS,
    index=False
)


# =============================================================================
# 6. VALIDATION
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
    "operator_rows",
    len(
        operator_df
    ) == 12,
    len(
        operator_df
    ),
    12
)


add_check(
    checks,
    "sensitivity_rows",
    len(
        results
    ) == 24,
    len(
        results
    ),
    24
)


add_check(
    checks,
    "penalty_nullity_two",
    bool(
        operator_df[
            "penalty_nullity"
        ]
        .eq(
            2
        )
        .all()
    ),
    int(
        (
            operator_df[
                "penalty_nullity"
            ]
            !=
            2
        ).sum()
    ),
    0
)


edf_error = float(
    np.max(
        np.abs(
            operator_df[
                "actual_effective_df"
            ]
            -
            operator_df[
                "target_effective_df"
            ]
        )
    )
)


add_check(
    checks,
    "effective_df_target_match",
    edf_error
    <
    1e-6,
    edf_error,
    "<1e-6"
)


max_constant_error = float(
    operator_df[
        "constant_reproduction_max_error"
    ].max()
)


max_linear_error = float(
    operator_df[
        "linear_reproduction_max_error"
    ].max()
)


add_check(
    checks,
    "constant_nullspace_preserved",
    max_constant_error
    <
    1e-8,
    max_constant_error,
    "<1e-8"
)


add_check(
    checks,
    "linear_nullspace_preserved",
    max_linear_error
    <
    1e-7,
    max_linear_error,
    "<1e-7"
)


finite_cols = [
    "lambda",
    "effective_df",
    "distortion_relative_to_centered_raw",
    "roughness_ratio",
    "roughness_reduction_pct",
    "max_row_sum_relative_error",
    "max_first_moment_relative_error",
    "negative_cell_pct",
    "minimum_smoothed_value",
    "pc1_energy_pct",
    "relative_error_k90",
]


nonfinite = int(
    (
        ~np.isfinite(
            results[
                finite_cols
            ].to_numpy(
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


max_sum_error = float(
    results[
        "max_row_sum_relative_error"
    ].max()
)


add_check(
    checks,
    "row_sum_preserved",
    max_sum_error
    <
    1e-8,
    max_sum_error,
    "<1e-8"
)


max_moment_error = float(
    results[
        "max_first_moment_relative_error"
    ].max()
)


add_check(
    checks,
    "first_moment_preserved",
    max_moment_error
    <
    1e-8,
    max_moment_error,
    "<1e-8"
)


baseline = results.loc[
    np.isclose(
        results[
            "edf_ratio_target"
        ],
        1.0
    )
]


baseline_distortion_max = float(
    baseline[
        "distortion_relative_to_centered_raw"
    ].max()
)


add_check(
    checks,
    "edf_1_is_identity",
    baseline_distortion_max
    <
    1e-12,
    baseline_distortion_max,
    "<1e-12"
)


# EDF deve cair quando a razão-alvo cai.
edf_monotonic_failures = 0

lambda_monotonic_failures = 0


for window, group in operator_df.groupby(
    "window_days"
):

    g = group.sort_values(
        "edf_ratio_target",
        ascending=False
    )

    edf = g[
        "actual_effective_df"
    ].to_numpy(
        dtype=float
    )

    lam = g[
        "lambda"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.diff(
            edf
        )
        <=
        1e-8
    ):
        edf_monotonic_failures += 1

    if not np.all(
        np.diff(
            lam
        )
        >=
        -1e-12
    ):
        lambda_monotonic_failures += 1


add_check(
    checks,
    "edf_monotonic",
    edf_monotonic_failures == 0,
    edf_monotonic_failures,
    0
)


add_check(
    checks,
    "lambda_monotonic",
    lambda_monotonic_failures == 0,
    lambda_monotonic_failures,
    0
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
# 7. SUMMARY
# =============================================================================

runtime = float(
    time.perf_counter()
    -
    start
)


negative_any = int(
    (
        results[
            "negative_cells"
        ]
        >
        0
    ).sum()
)


summary = {
    "step":
        "MODEL_01_0_C3_3D1_ROUGHNESS_PENALTY_SENSITIVITY",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "method":
        "SECOND_DIFFERENCE_PENALIZED_LEAST_SQUARES",

    "formula":
        "H_lambda = H (I + lambda D2^T D2)^(-1)",

    "lambda_parameterization":
        "EFFECTIVE_DEGREES_OF_FREEDOM",

    "edf_ratios":
        EDF_RATIOS,

    "windows":
        WINDOWS,

    "channels":
        CHANNELS,

    "operator_configurations":
        int(
            len(
                operator_df
            )
        ),

    "sensitivity_runs":
        int(
            len(
                results
            )
        ),

    "configurations_with_negative_values":
        negative_any,

    "target_used":
        False,

    "lambda_selected":
        False,

    "window_selected":
        False,

    "component_count_selected":
        False,

    "smoothing_committed":
        False,

    "fpca_applied":
        False,

    "model_feature_created":
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


P_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 8. REPORT
# =============================================================================

lines = [
    "=" * 110,
    "MODEL 01.0-C3.3-D1 — ROUGHNESS-PENALTY SMOOTHING SENSITIVITY",
    "=" * 110,
    "",
    f"STATUS                       : {summary['status']}",
    f"SENSITIVITY RUNS             : {len(results)}",
    f"NEGATIVE-VALUE CONFIGS       : {negative_any}",
    "",
    "FORMULATION",
    "-" * 110,
    "min_z ||h-z||^2 + lambda ||D2 z||^2",
    "H_lambda = H (I + lambda D2^T D2)^(-1)",
    "",
    "LAMBDA CONTROL",
    "-" * 110,
    "Lambda is NOT manually chosen.",
    "Smoothing strength is compared through effective degrees of freedom.",
    "EDF ratios: 1.00, 0.75, 0.50, 0.25.",
    "",
    "IMPORTANT",
    "-" * 110,
    "No lambda selected.",
    "No window selected.",
    "No K selected.",
    "No target used.",
    "No FPCA score produced.",
    "No model trained.",
    "No Silver created.",
    "RAW not modified.",
    "",
    f"VALIDATION FAILURES          : {failures}",
    f"RUNTIME                      : {runtime:.3f}s",
    "=" * 110,
]


P_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 9. PRINT
# =============================================================================

print()
print("=" * 118)
print("OPERATOR GRID")
print("=" * 118)

print(
    operator_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.10g}"
    )
)


print()
print("=" * 118)
print("SENSITIVITY RESULTS")
print("=" * 118)

display_cols = [
    "channel",
    "window_days",
    "edf_ratio_target",
    "lambda",
    "effective_df",
    "distortion_relative_to_centered_raw",
    "roughness_ratio",
    "roughness_reduction_pct",
    "negative_cell_pct",
    "adjacent_lag_correlation_mean",
    "pc1_energy_pct",
    "k90",
    "k95",
]

print(
    results[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
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
print("RESULTADO C3.3-D1")
print("=" * 118)

print(
    "FAILURES                    =",
    failures
)

print(
    "OPERATOR CONFIGURATIONS     =",
    len(
        operator_df
    )
)

print(
    "SENSITIVITY RUNS            =",
    len(
        results
    )
)

print(
    "CONFIGS WITH NEGATIVE CELLS =",
    negative_any
)

print()
print("TARGET USED                 = NÃO")
print("LAMBDA SELECTED             = NÃO")
print("WINDOW SELECTED             = NÃO")
print("K SELECTED                  = NÃO")
print("SMOOTHING COMMITTED         = NÃO")
print("FPCA APPLIED                = NÃO")
print("MODEL FEATURE CREATED       = NÃO")
print("FUNCTIONAL MODULE           = AINDA NÃO LIBERADO")
print("FOLDS FROZEN                = NÃO")
print("CLASSIFIER TRAINED          = NÃO")
print("SILVER CREATED              = NÃO")
print("RAW MODIFIED                = NÃO")


if failures:
    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.3-D1 smoothing sensitivity concluída."
)

print(
    "[PASS] Parar antes de escolher smoothing/lambda."
)

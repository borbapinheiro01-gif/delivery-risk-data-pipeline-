#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-C
TEMPORAL PORTABILITY OF FUNCTIONAL PCA/SVD BASIS
===============================================================================

PERGUNTA
--------
Uma base funcional aprendida SOMENTE no passado consegue representar
adequadamente curvas futuras?

Para cada:
    channel = purchase_volume / purchase_freight
    window  = 30 / 60 / 90 dias
    current_month = mês futuro

fazemos:

    TRAIN = todos os pedidos anteriores ao current_month
            com full historical support

    TEST  = pedidos do current_month
            com full historical support

A base é aprendida APENAS no TRAIN.

Nenhum target é utilizado.

===============================================================================
FORMULAÇÃO
===============================================================================

Se H_train é a matriz de curvas:

    mu = mean(H_train)

    Hc_train = H_train - mu

    Hc_train = U Sigma V^T

K90 é definido SOMENTE no TRAIN:

    K90 = min {
        K :
        sum_{k<=K} sigma_k^2 / sum_j sigma_j^2 >= 0.90
    }

Para uma curva futura h_i:

    score_i = (h_i - mu) V_K

    hhat_i = mu + score_i V_K^T

Erro relativo:

    RE =
        ||H_test - Hhat_test||_F
        /
        ||H_test - mu||_F

Também calculamos K80 e K95.

IMPORTANTE
----------
Isto ainda NÃO é o Temporal Expert.

NÃO:
- usa Y;
- seleciona janela final;
- congela K;
- aplica smoothing;
- treina classificador;
- cria feature definitiva;
- congela folds;
- cria Silver;
- altera RAW.
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

B_SUMMARY = (
    OUT
    / "02t_functional_geometry_summary.json"
)

ORDER_INDEX = (
    OUT
    / "02a_functional_pit_order_index.csv"
)

SUPPORT = (
    OUT
    / "02l_history_support_by_order.csv"
)

VOLUME = (
    ART
    / "02g_purchase_volume_curve_90d.npy"
)

FREIGHT = (
    ART
    / "02h_purchase_freight_curve_90d.npy"
)


# =============================================================================
# OUTPUTS
# =============================================================================

P_FOLDS = (
    OUT
    / "02v_functional_temporal_portability_folds.csv"
)

P_SUMMARY_TABLE = (
    OUT
    / "02w_functional_temporal_portability_summary.csv"
)

P_K_STABILITY = (
    OUT
    / "02x_functional_k_temporal_stability.csv"
)

P_VALIDATION = (
    OUT
    / "02y_functional_temporal_portability_validation.csv"
)

P_SUMMARY_JSON = (
    OUT
    / "02z_functional_temporal_portability_summary.json"
)

P_REPORT = (
    OUT
    / "03a_functional_temporal_portability_report.txt"
)


# =============================================================================
# CONSTANTS
# =============================================================================

WINDOWS = [30, 60, 90]

CHANNELS = [
    "purchase_volume",
    "purchase_freight",
]

MIN_TRAIN_ROWS = 5000
MIN_TEST_ROWS = 500

ENERGY_LEVELS = [
    0.80,
    0.90,
    0.95,
]


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
        "check": check,
        "status": "PASS" if bool(condition) else "FAIL",
        "observed": observed,
        "expected": expected,
    })


def first_k(
    cumulative,
    level
):
    where = np.flatnonzero(
        cumulative >= level
    )

    if len(where) == 0:
        return len(cumulative)

    return int(
        where[0] + 1
    )


def fit_basis(
    train
):
    """
    PCA/SVD centrada.
    Não escala os lags.
    """

    A = np.asarray(
        train,
        dtype=np.float64
    )

    mean = A.mean(
        axis=0
    )

    centered = (
        A - mean
    )

    _, s, vt = np.linalg.svd(
        centered,
        full_matrices=False
    )

    energy = s ** 2

    ratio = (
        energy
        /
        energy.sum()
    )

    cumulative = np.cumsum(
        ratio
    )

    ks = {
        level:
            first_k(
                cumulative,
                level
            )
        for level
        in ENERGY_LEVELS
    }

    return (
        mean,
        s,
        vt,
        cumulative,
        ks
    )


def relative_reconstruction_error(
    test,
    mean,
    vt,
    k
):
    """
    Erro no conjunto futuro usando base aprendida no passado.
    """

    A = np.asarray(
        test,
        dtype=np.float64
    )

    centered = (
        A - mean
    )

    basis = (
        vt[:k, :]
        .T
    )

    scores = (
        centered
        @
        basis
    )

    reconstructed_centered = (
        scores
        @
        basis.T
    )

    residual = (
        centered
        -
        reconstructed_centered
    )

    denominator = float(
        np.linalg.norm(
            centered,
            ord="fro"
        )
    )

    numerator = float(
        np.linalg.norm(
            residual,
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


# =============================================================================
# START
# =============================================================================

start = time.perf_counter()

print()
print("=" * 118)
print("MODEL 01.0-C3.3-C — TEMPORAL FUNCTIONAL BASIS PORTABILITY")
print("=" * 118)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required = [
    (B_SUMMARY, "C3.3-B summary"),
    (ORDER_INDEX, "order index"),
    (SUPPORT, "support matrix"),
    (VOLUME, "volume curves"),
    (FREIGHT, "freight curves"),
]

for path, label in required:

    if not path.exists():
        raise SystemExit(
            f"[FAIL] {label} ausente: {path}"
        )

    print(
        f"[PASS] {label}"
    )


with B_SUMMARY.open(
    encoding="utf-8"
) as f:
    b_summary = json.load(f)


if b_summary.get(
    "status"
) != "PASS":

    raise SystemExit(
        "[STOP] C3.3-B não está PASS."
    )


if b_summary.get(
    "validation_failures"
) != 0:

    raise SystemExit(
        "[STOP] C3.3-B possui falhas."
    )


print()
print("[PASS] C3.3-B validado.")


# =============================================================================
# 2. LOAD METADATA
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


if len(index) != len(support):
    raise RuntimeError(
        "Order index e support possuem tamanhos diferentes."
    )


if not np.array_equal(
    index[
        "curve_row_index"
    ].to_numpy(dtype=np.int64),
    support[
        "curve_row_index"
    ].to_numpy(dtype=np.int64)
):
    raise RuntimeError(
        "curve_row_index desalinhado."
    )


meta = index[
    [
        "curve_row_index",
        "order_id",
        "order_purchase_timestamp",
        "purchase_month",
    ]
].copy()


for window in WINDOWS:

    meta[
        f"full_support_{window}d"
    ] = (
        support[
            f"full_support_{window}d"
        ]
        .astype(bool)
        .to_numpy()
    )


volume = np.load(
    VOLUME,
    mmap_mode="r",
    allow_pickle=False
)

freight = np.load(
    FREIGHT,
    mmap_mode="r",
    allow_pickle=False
)


curves = {
    "purchase_volume":
        volume,

    "purchase_freight":
        freight,
}


# =============================================================================
# 3. MONTHS
# =============================================================================

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
    "Months available:",
    len(months)
)

print(
    "First:",
    months[0],
    "| Last:",
    months[-1]
)


# =============================================================================
# 4. TEMPORAL PORTABILITY
# =============================================================================

fold_rows = []


print()
print("=" * 118)
print("EXPANDING-HISTORY PORTABILITY")
print("=" * 118)


for channel in CHANNELS:

    curve = curves[
        channel
    ]

    for window in WINDOWS:

        support_col = (
            f"full_support_{window}d"
        )

        for current_month in months:

            train_mask = (
                meta[
                    support_col
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
                    support_col
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


            train = np.asarray(
                curve[
                    train_mask,
                    :window
                ],
                dtype=np.float64
            )

            test = np.asarray(
                curve[
                    test_mask,
                    :window
                ],
                dtype=np.float64
            )


            mean_train, s_train, vt_train, cum_train, ks = (
                fit_basis(
                    train
                )
            )


            # ---------------------------------------------------------
            # Também ajustamos a base no mês futuro SOMENTE para medir
            # mudança geométrica. Ela NÃO é usada na reconstrução.
            # ---------------------------------------------------------

            mean_test, s_test, vt_test, cum_test, ks_test = (
                fit_basis(
                    test
                )
            )


            row = {
                "channel":
                    channel,

                "window_days":
                    window,

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

                "test_month":
                    current_month,
            }


            for level in ENERGY_LEVELS:

                pct = int(
                    round(
                        level * 100
                    )
                )

                k = ks[
                    level
                ]

                future_error = (
                    relative_reconstruction_error(
                        test,
                        mean_train,
                        vt_train,
                        k
                    )
                )

                train_residual = float(
                    np.sqrt(
                        max(
                            0.0,
                            1.0
                            -
                            cum_train[
                                k - 1
                            ]
                        )
                    )
                )

                row[
                    f"k{pct}_train"
                ] = int(
                    k
                )

                row[
                    f"train_relative_error_k{pct}"
                ] = (
                    train_residual
                )

                row[
                    f"future_relative_error_k{pct}"
                ] = (
                    future_error
                )

                row[
                    f"future_error_inflation_k{pct}"
                ] = float(
                    future_error
                    /
                    train_residual
                ) if train_residual > 0 else np.nan


            # ---------------------------------------------------------
            # Subspace angle — usamos K90 aprendido no TRAIN.
            #
            # Test basis entra apenas como diagnóstico geométrico.
            # ---------------------------------------------------------

            k_angle = min(
                int(
                    ks[0.90]
                ),
                int(
                    ks_test[0.90]
                )
            )


            train_basis = (
                vt_train[
                    :k_angle,
                    :
                ]
                .T
            )

            test_basis = (
                vt_test[
                    :k_angle,
                    :
                ]
                .T
            )


            angles = np.degrees(
                subspace_angles(
                    train_basis,
                    test_basis
                )
            )


            row[
                "basis_angle_dimension"
            ] = int(
                k_angle
            )

            row[
                "basis_max_angle_deg"
            ] = float(
                np.max(
                    angles
                )
            )

            row[
                "basis_mean_angle_deg"
            ] = float(
                np.mean(
                    angles
                )
            )

            row[
                "train_pc1_energy_pct"
            ] = float(
                100.0
                *
                (
                    s_train[0] ** 2
                )
                /
                np.sum(
                    s_train ** 2
                )
            )

            row[
                "test_pc1_energy_pct"
            ] = float(
                100.0
                *
                (
                    s_test[0] ** 2
                )
                /
                np.sum(
                    s_test ** 2
                )
            )


            fold_rows.append(
                row
            )


folds = pd.DataFrame(
    fold_rows
)


if folds.empty:
    raise RuntimeError(
        "Nenhum fold possui suporte suficiente."
    )


folds.to_csv(
    P_FOLDS,
    index=False
)


# =============================================================================
# 5. SUMMARY PER CHANNEL/WINDOW
# =============================================================================

summary_rows = []


for (
    channel,
    window
), group in folds.groupby(
    [
        "channel",
        "window_days"
    ],
    sort=True
):

    summary_rows.append({
        "channel":
            channel,

        "window_days":
            int(
                window
            ),

        "temporal_tests":
            int(
                len(
                    group
                )
            ),

        "first_test_month":
            group[
                "current_month"
            ].min(),

        "last_test_month":
            group[
                "current_month"
            ].max(),

        "k90_median":
            float(
                group[
                    "k90_train"
                ].median()
            ),

        "k90_min":
            int(
                group[
                    "k90_train"
                ].min()
            ),

        "k90_max":
            int(
                group[
                    "k90_train"
                ].max()
            ),

        "future_error_k90_mean":
            float(
                group[
                    "future_relative_error_k90"
                ].mean()
            ),

        "future_error_k90_median":
            float(
                group[
                    "future_relative_error_k90"
                ].median()
            ),

        "future_error_k90_p95":
            float(
                group[
                    "future_relative_error_k90"
                ].quantile(
                    0.95
                )
            ),

        "future_error_inflation_k90_mean":
            float(
                group[
                    "future_error_inflation_k90"
                ].mean()
            ),

        "future_error_inflation_k90_max":
            float(
                group[
                    "future_error_inflation_k90"
                ].max()
            ),

        "basis_mean_angle_deg_mean":
            float(
                group[
                    "basis_mean_angle_deg"
                ].mean()
            ),

        "basis_max_angle_deg_max":
            float(
                group[
                    "basis_max_angle_deg"
                ].max()
            ),

        "train_pc1_energy_pct_mean":
            float(
                group[
                    "train_pc1_energy_pct"
                ].mean()
            ),

        "test_pc1_energy_pct_mean":
            float(
                group[
                    "test_pc1_energy_pct"
                ].mean()
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
# 6. K TEMPORAL STABILITY
# =============================================================================

k_stability = (
    folds[
        [
            "channel",
            "window_days",
            "current_month",
            "train_orders",
            "k80_train",
            "k90_train",
            "k95_train",
        ]
    ]
    .copy()
)

k_stability.to_csv(
    P_K_STABILITY,
    index=False
)


# =============================================================================
# 7. VALIDATION
# =============================================================================

checks = []


add_check(
    checks,
    "c33b_pass",
    b_summary.get(
        "status"
    ) == "PASS",
    b_summary.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "folds_exist",
    len(
        folds
    ) > 0,
    len(
        folds
    ),
    ">0"
)


add_check(
    checks,
    "all_train_before_test",
    bool(
        (
            folds[
                "train_last_month"
            ]
            <
            folds[
                "current_month"
            ]
        ).all()
    ),
    int(
        (
            folds[
                "train_last_month"
            ]
            >=
            folds[
                "current_month"
            ]
        ).sum()
    ),
    0
)


numerical_cols = [
    "train_relative_error_k80",
    "future_relative_error_k80",
    "train_relative_error_k90",
    "future_relative_error_k90",
    "train_relative_error_k95",
    "future_relative_error_k95",
    "basis_max_angle_deg",
    "basis_mean_angle_deg",
]


finite_bad = int(
    (
        ~np.isfinite(
            folds[
                numerical_cols
            ].to_numpy(
                dtype=float
            )
        )
    ).sum()
)


add_check(
    checks,
    "all_diagnostics_finite",
    finite_bad == 0,
    finite_bad,
    0
)


error_bounds_bad = 0

for c in [
    "train_relative_error_k80",
    "future_relative_error_k80",
    "train_relative_error_k90",
    "future_relative_error_k90",
    "train_relative_error_k95",
    "future_relative_error_k95",
]:

    error_bounds_bad += int(
        (
            (
                folds[c] < 0
            )
            |
            (
                folds[c] > 1
            )
        ).sum()
    )


add_check(
    checks,
    "reconstruction_errors_between_0_1",
    error_bounds_bad == 0,
    error_bounds_bad,
    0
)


k_bad = int(
    (
        ~(
            (
                folds[
                    "k80_train"
                ]
                <=
                folds[
                    "k90_train"
                ]
            )
            &
            (
                folds[
                    "k90_train"
                ]
                <=
                folds[
                    "k95_train"
                ]
            )
            &
            (
                folds[
                    "k95_train"
                ]
                <=
                folds[
                    "window_days"
                ]
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


expected_groups = (
    len(
        CHANNELS
    )
    *
    len(
        WINDOWS
    )
)


add_check(
    checks,
    "six_channel_window_groups",
    len(
        summary_table
    ) == expected_groups,
    len(
        summary_table
    ),
    expected_groups
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
# 8. JSON SUMMARY
# =============================================================================

runtime = float(
    time.perf_counter()
    -
    start
)


summary_json = {
    "step":
        "MODEL_01_0_C3_3C_TEMPORAL_FUNCTIONAL_BASIS_PORTABILITY",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "method":
        "EXPANDING_HISTORY_PCA_BASIS_OUT_OF_TIME_RECONSTRUCTION",

    "target_used":
        False,

    "minimum_train_rows":
        MIN_TRAIN_ROWS,

    "minimum_test_rows":
        MIN_TEST_ROWS,

    "windows_days":
        WINDOWS,

    "channels":
        CHANNELS,

    "temporal_fold_rows":
        int(
            len(
                folds
            )
        ),

    "channel_window_groups":
        int(
            len(
                summary_table
            )
        ),

    "k_learned_from_past_only":
        True,

    "window_selected":
        False,

    "component_count_selected":
        False,

    "smoothing_applied":
        False,

    "fpca_model_feature_created":
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
# 9. REPORT
# =============================================================================

lines = [
    "=" * 108,
    "MODEL 01.0-C3.3-C — TEMPORAL FUNCTIONAL BASIS PORTABILITY",
    "=" * 108,
    "",
    f"STATUS                     : {summary_json['status']}",
    f"TEMPORAL TEST ROWS         : {len(folds)}",
    f"CHANNEL/WINDOW GROUPS      : {len(summary_table)}",
    "",
    "CONTRACT",
    "-" * 108,
    "Basis learned on historical orders only.",
    "K learned from historical curves only.",
    "Current/future month never participates in basis fitting.",
    "Target is not used.",
    "",
    "SUMMARY",
    "-" * 108,
]


for _, row in summary_table.iterrows():

    lines.append(
        f"{row['channel']} | "
        f"{int(row['window_days'])}d | "
        f"tests={int(row['temporal_tests'])} | "
        f"K90 median={row['k90_median']:.1f} | "
        f"K90 range={int(row['k90_min'])}-{int(row['k90_max'])} | "
        f"future RE90 mean={row['future_error_k90_mean']:.4f} | "
        f"future RE90 p95={row['future_error_k90_p95']:.4f} | "
        f"max inflation={row['future_error_inflation_k90_max']:.4f}"
    )


lines += [
    "",
    "IMPORTANT",
    "-" * 108,
    "Window NOT selected.",
    "K NOT selected.",
    "No target used.",
    "No classifier trained.",
    "No model feature created.",
    "No fold frozen.",
    "No Silver created.",
    "RAW not modified.",
    "",
    f"VALIDATION FAILURES        : {failures}",
    f"RUNTIME                    : {runtime:.3f}s",
    "=" * 108,
]


P_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 10. PRINT
# =============================================================================

print()
print("=" * 118)
print("TEMPORAL PORTABILITY SUMMARY")
print("=" * 118)

print(
    summary_table.to_string(
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
print("RESULTADO C3.3-C")
print("=" * 118)

print(
    "FAILURES                 =",
    failures
)

print(
    "TEMPORAL FOLD ROWS       =",
    len(
        folds
    )
)

print(
    "CHANNEL/WINDOW GROUPS    =",
    len(
        summary_table
    )
)

print()
print("TARGET USED               = NÃO")
print("WINDOW SELECTED           = NÃO")
print("K SELECTED                = NÃO")
print("SMOOTHING APPLIED         = NÃO")
print("MODEL FEATURE CREATED     = NÃO")
print("FUNCTIONAL MODULE         = AINDA NÃO LIBERADO")
print("FOLDS FROZEN              = NÃO")
print("CLASSIFIER TRAINED        = NÃO")
print("SILVER CREATED            = NÃO")
print("RAW MODIFIED              = NÃO")


if failures:
    raise SystemExit(2)


print()
print(
    "[PASS] C3.3-C temporal portability concluída."
)

print(
    "[PASS] Parar antes de escolher janela/K ou treinar qualquer modelo."
)

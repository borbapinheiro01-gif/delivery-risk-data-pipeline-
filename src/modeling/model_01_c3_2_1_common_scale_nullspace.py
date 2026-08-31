#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.2.1
COMMON-SCALE TEMPORAL NULLSPACE VALIDATION
===============================================================================

Correção metodológica do C3.2.

PROBLEMA DO C3.2 ORIGINAL
-------------------------
O global usava:

    Z_global = (X - mu_global) / sigma_global

mas cada mês usava:

    Z_month = (X_month - mu_month) / sigma_month

Logo, os vetores que representam as relações lineares eram expressos
em sistemas de coordenadas diferentes.

Neste patch:

    Z_month_common =
        (X_month - mu_global) / sigma_global

para TODOS os meses.

Assim:
- mesma origem de referência;
- mesma escala;
- nullspaces comparáveis;
- subspace_angles passa a medir a diferença geométrica real.

IMPORTANTE
----------
Este script é apenas diagnóstico.

NÃO:
- remove features;
- altera RAW;
- cria Silver;
- congela threshold;
- congela boundary;
- congela folds;
- treina modelo.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles, orth


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

MATRIX = (
    PROJECT
    / "artifacts"
    / "model_01_order_logistic"
    / "pretraining"
    / "ORDER_CORE_V1_AUDIT_MATRIX.csv"
)

CONTRACT = (
    PROJECT
    / "configs"
    / "order_core_v1_feature_contract.json"
)

OLD_SUMMARY = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
    / "01y_sensitivity_rank_summary.json"
)

OUT = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


OUT_TEMPORAL = OUT / "01aa_common_scale_nullspace_temporal.csv"

OUT_RELATIONS = OUT / "01ab_common_scale_relation_validation.csv"

OUT_VALIDATION = OUT / "01ac_common_scale_nullspace_validation.csv"

OUT_SUMMARY = OUT / "01ad_common_scale_nullspace_summary.json"

OUT_REPORT = OUT / "01ae_common_scale_nullspace_report.txt"


# =============================================================================
# HELPERS
# =============================================================================

def svd_rank_and_nullspace(matrix):

    matrix = np.asarray(
        matrix,
        dtype=np.float64
    )

    _, singular_values, vt = np.linalg.svd(
        matrix,
        full_matrices=False
    )

    eps = np.finfo(
        singular_values.dtype
    ).eps

    tolerance = (
        singular_values.max()
        *
        max(matrix.shape)
        *
        eps
    )

    rank = int(
        np.sum(
            singular_values > tolerance
        )
    )

    nullity = int(
        matrix.shape[1] - rank
    )

    if nullity > 0:

        null_basis = (
            vt[
                rank:,
                :
            ]
            .T
        )

    else:

        null_basis = np.empty(
            (
                matrix.shape[1],
                0
            )
        )

    return {
        "rank":
            rank,

        "nullity":
            nullity,

        "singular_values":
            singular_values,

        "tolerance":
            float(
                tolerance
            ),

        "null_basis":
            null_basis
    }


def relation_vector(
    features,
    coefficients
):

    vector = np.zeros(
        len(features),
        dtype=np.float64
    )

    index = {
        feature: i
        for i, feature
        in enumerate(features)
    }

    for feature, coefficient in coefficients.items():

        vector[
            index[
                feature
            ]
        ] = coefficient

    return vector


# =============================================================================
# 1. PRÉ-REQUISITOS
# =============================================================================

print()
print("=" * 112)
print("MODEL 01.0-C3.2.1 — COMMON-SCALE NULLSPACE VALIDATION")
print("=" * 112)

for path, name in [
    (MATRIX, "ORDER_CORE_V1 matrix"),
    (CONTRACT, "Feature contract"),
    (OLD_SUMMARY, "C3.2 summary"),
]:

    if not path.exists():

        raise SystemExit(
            f"[FAIL] {name} ausente: {path}"
        )

    print(
        f"[PASS] {name} encontrado."
    )


# =============================================================================
# 2. VALIDAR C3.2 ANTERIOR
# =============================================================================

with OLD_SUMMARY.open(
    encoding="utf-8"
) as f:

    old_summary = json.load(f)

if old_summary.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "C3.2 anterior não está PASS."
    )

if old_summary.get(
    "global_rank"
) != 11:

    raise RuntimeError(
        "C3.2 anterior não possui rank global 11."
    )

if old_summary.get(
    "global_nullity"
) != 2:

    raise RuntimeError(
        "C3.2 anterior não possui nullity global 2."
    )

print()
print("[PASS] C3.2 anterior confirmado: rank=11, nullity=2.")


# =============================================================================
# 3. FEATURES
# =============================================================================

with CONTRACT.open(
    encoding="utf-8"
) as f:

    contract = json.load(f)

features = [
    row["feature_name"]
    for row
    in contract["features"]
]

if len(features) != 13:

    raise RuntimeError(
        f"Esperadas 13 features; obtidas {len(features)}."
    )


# =============================================================================
# 4. MATRIZ
# =============================================================================

df = pd.read_csv(
    MATRIX,
    parse_dates=[
        "order_purchase_timestamp"
    ]
)

X = (
    df[features]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=np.float64
    )
)

if not np.isfinite(X).all():

    raise RuntimeError(
        "Matriz possui NaN ou Inf."
    )


df["purchase_month"] = (
    df[
        "order_purchase_timestamp"
    ]
    .dt.to_period(
        "M"
    )
    .astype(str)
)


# =============================================================================
# 5. TRANSFORMAÇÃO GLOBAL COMUM
# =============================================================================

global_mean = X.mean(
    axis=0
)

global_scale = X.std(
    axis=0,
    ddof=0
)

if np.any(
    global_scale == 0
):

    zero_features = [
        features[i]
        for i
        in np.where(
            global_scale == 0
        )[0]
    ]

    raise RuntimeError(
        "Zero variance global: "
        +
        str(
            zero_features
        )
    )


Z_global = (
    X - global_mean
) / global_scale


global_diag = svd_rank_and_nullspace(
    Z_global
)

global_nullspace = global_diag[
    "null_basis"
]


print()
print("=" * 112)
print("GLOBAL COMMON COORDINATE SYSTEM")
print("=" * 112)

print(
    "Rows            :",
    len(df)
)

print(
    "Features        :",
    len(features)
)

print(
    "Global rank     :",
    global_diag[
        "rank"
    ]
)

print(
    "Global nullity  :",
    global_diag[
        "nullity"
    ]
)

print(
    "SVD tolerance   :",
    f"{global_diag['tolerance']:.8e}"
)


# =============================================================================
# 6. ESPAÇO NULO TEÓRICO
# =============================================================================

relations = {
    "PRICE_RANGE": {
        "price_range":
            1.0,

        "max_price":
            -1.0,

        "min_price":
            1.0
    },

    "MERCHANDISE_PLUS_FREIGHT": {
        "merchandise_plus_freight":
            1.0,

        "total_price":
            -1.0,

        "total_freight":
            -1.0
    }
}


raw_relation_vectors = []

for relation in relations.values():

    raw_relation_vectors.append(
        relation_vector(
            features,
            relation
        )
    )


raw_relation_matrix = np.column_stack(
    raw_relation_vectors
)


# -------------------------------------------------------------------------
# Se:
#
# Xc @ c = 0
#
# e:
#
# Z = Xc D^{-1}
#
# então:
#
# Z @ (D c) = 0
#
# Portanto o vetor no sistema padronizado é:
#
#     c_standardized = D c
# -------------------------------------------------------------------------

standardized_relation_matrix = (
    global_scale[:, None]
    *
    raw_relation_matrix
)


theoretical_nullspace = orth(
    standardized_relation_matrix
)


if theoretical_nullspace.shape[1] != 2:

    raise RuntimeError(
        "As duas relações teóricas não são independentes."
    )


angles_global_theoretical = np.degrees(
    subspace_angles(
        global_nullspace,
        theoretical_nullspace
    )
)


print()
print("=" * 112)
print("GLOBAL NUMERICAL NULLSPACE vs THEORETICAL NULLSPACE")
print("=" * 112)

print(
    "Angles (deg):",
    angles_global_theoretical
)

print(
    "Maximum angle:",
    float(
        np.max(
            angles_global_theoretical
        )
    )
)


# =============================================================================
# 7. RELAÇÕES — VALIDAÇÃO GLOBAL E TEMPORAL
# =============================================================================

relation_rows = []


for relation_name, coefficients in relations.items():

    c_raw = relation_vector(
        features,
        coefficients
    )

    raw_global_residual = (
        X @ c_raw
    )

    c_common = (
        global_scale
        *
        c_raw
    )

    c_common = (
        c_common
        /
        np.linalg.norm(
            c_common
        )
    )

    common_global_residual = (
        Z_global
        @
        c_common
    )

    relation_rows.append({
        "scope":
            "GLOBAL",

        "purchase_month":
            "ALL",

        "relation":
            relation_name,

        "rows":
            len(df),

        "raw_max_abs_residual":
            float(
                np.max(
                    np.abs(
                        raw_global_residual
                    )
                )
            ),

        "common_scale_max_abs_residual":
            float(
                np.max(
                    np.abs(
                        common_global_residual
                    )
                )
            ),

        "raw_allclose_zero":
            bool(
                np.allclose(
                    raw_global_residual,
                    0.0,
                    rtol=1e-10,
                    atol=1e-10
                )
            ),

        "common_scale_allclose_zero":
            bool(
                np.allclose(
                    common_global_residual,
                    0.0,
                    rtol=1e-10,
                    atol=1e-10
                )
            ),
    })


# =============================================================================
# 8. NULLSPACE TEMPORAL EM ESCALA COMUM
# =============================================================================

temporal_rows = []

p = len(features)


for month, group in df.groupby(
    "purchase_month",
    sort=True
):

    Xm = (
        group[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    n = len(
        Xm
    )


    # ---------------------------------------------------------------------
    # As relações RAW também são verificadas em cada mês.
    # ---------------------------------------------------------------------

    for relation_name, coefficients in relations.items():

        c_raw = relation_vector(
            features,
            coefficients
        )

        raw_residual = (
            Xm @ c_raw
        )

        c_common = (
            global_scale
            *
            c_raw
        )

        c_common = (
            c_common
            /
            np.linalg.norm(
                c_common
            )
        )


        Zm_common_for_relation = (
            Xm
            -
            global_mean
        ) / global_scale


        common_residual = (
            Zm_common_for_relation
            @
            c_common
        )


        relation_rows.append({
            "scope":
                "MONTH",

            "purchase_month":
                month,

            "relation":
                relation_name,

            "rows":
                n,

            "raw_max_abs_residual":
                float(
                    np.max(
                        np.abs(
                            raw_residual
                        )
                    )
                ),

            "common_scale_max_abs_residual":
                float(
                    np.max(
                        np.abs(
                            common_residual
                        )
                    )
                ),

            "raw_allclose_zero":
                bool(
                    np.allclose(
                        raw_residual,
                        0.0,
                        rtol=1e-10,
                        atol=1e-10
                    )
                ),

            "common_scale_allclose_zero":
                bool(
                    np.allclose(
                        common_residual,
                        0.0,
                        rtol=1e-10,
                        atol=1e-10
                    )
                ),
        })


    # ---------------------------------------------------------------------
    # Meses muito pequenos não suportam diagnóstico de rank.
    # ---------------------------------------------------------------------

    if n < p + 1:

        temporal_rows.append({
            "purchase_month":
                month,

            "orders":
                n,

            "computable":
                False,

            "numerical_rank":
                np.nan,

            "nullity":
                np.nan,

            "angle_1_global_vs_month_deg":
                np.nan,

            "angle_2_global_vs_month_deg":
                np.nan,

            "max_angle_global_vs_month_deg":
                np.nan,

            "mean_angle_global_vs_month_deg":
                np.nan,

            "angle_1_theoretical_vs_month_deg":
                np.nan,

            "angle_2_theoretical_vs_month_deg":
                np.nan,

            "max_angle_theoretical_vs_month_deg":
                np.nan,

            "mean_angle_theoretical_vs_month_deg":
                np.nan,

            "smallest_retained_relative_singular_value":
                np.nan,
        })

        continue


    # ---------------------------------------------------------------------
    # CORREÇÃO PRINCIPAL:
    #
    # GLOBAL mean + GLOBAL scale para todos os meses.
    # ---------------------------------------------------------------------

    Zm_common = (
        Xm
        -
        global_mean
    ) / global_scale


    dm = svd_rank_and_nullspace(
        Zm_common
    )


    month_nullspace = dm[
        "null_basis"
    ]


    if (
        dm["nullity"] > 0
        and
        global_nullspace.shape[1] > 0
    ):

        angles_global = np.degrees(
            subspace_angles(
                global_nullspace,
                month_nullspace
            )
        )

    else:

        angles_global = np.array(
            [
                np.nan,
                np.nan
            ]
        )


    if (
        dm["nullity"] > 0
        and
        theoretical_nullspace.shape[1] > 0
    ):

        angles_theoretical = np.degrees(
            subspace_angles(
                theoretical_nullspace,
                month_nullspace
            )
        )

    else:

        angles_theoretical = np.array(
            [
                np.nan,
                np.nan
            ]
        )


    if dm["rank"] > 0:

        smallest_relative = float(
            dm[
                "singular_values"
            ][
                dm["rank"] - 1
            ]
            /
            dm[
                "singular_values"
            ][0]
        )

    else:

        smallest_relative = np.nan


    temporal_rows.append({
        "purchase_month":
            month,

        "orders":
            n,

        "computable":
            True,

        "numerical_rank":
            dm[
                "rank"
            ],

        "nullity":
            dm[
                "nullity"
            ],

        "angle_1_global_vs_month_deg":
            (
                float(
                    angles_global[0]
                )
                if len(
                    angles_global
                ) > 0
                else np.nan
            ),

        "angle_2_global_vs_month_deg":
            (
                float(
                    angles_global[1]
                )
                if len(
                    angles_global
                ) > 1
                else np.nan
            ),

        "max_angle_global_vs_month_deg":
            float(
                np.nanmax(
                    angles_global
                )
            ),

        "mean_angle_global_vs_month_deg":
            float(
                np.nanmean(
                    angles_global
                )
            ),

        "angle_1_theoretical_vs_month_deg":
            (
                float(
                    angles_theoretical[0]
                )
                if len(
                    angles_theoretical
                ) > 0
                else np.nan
            ),

        "angle_2_theoretical_vs_month_deg":
            (
                float(
                    angles_theoretical[1]
                )
                if len(
                    angles_theoretical
                ) > 1
                else np.nan
            ),

        "max_angle_theoretical_vs_month_deg":
            float(
                np.nanmax(
                    angles_theoretical
                )
            ),

        "mean_angle_theoretical_vs_month_deg":
            float(
                np.nanmean(
                    angles_theoretical
                )
            ),

        "smallest_retained_relative_singular_value":
            smallest_relative,
    })


temporal_df = pd.DataFrame(
    temporal_rows
)

relation_df = pd.DataFrame(
    relation_rows
)


temporal_df.to_csv(
    OUT_TEMPORAL,
    index=False
)

relation_df.to_csv(
    OUT_RELATIONS,
    index=False
)


# =============================================================================
# 9. RESULTADOS TEMPORAIS
# =============================================================================

computable = temporal_df.loc[
    temporal_df[
        "computable"
    ].eq(
        True
    )
].copy()


rank_min = int(
    computable[
        "numerical_rank"
    ].min()
)

rank_max = int(
    computable[
        "numerical_rank"
    ].max()
)

nullity_min = int(
    computable[
        "nullity"
    ].min()
)

nullity_max = int(
    computable[
        "nullity"
    ].max()
)


max_global_angle = float(
    computable[
        "max_angle_global_vs_month_deg"
    ].max()
)

mean_global_angle = float(
    computable[
        "max_angle_global_vs_month_deg"
    ].mean()
)

max_theoretical_angle = float(
    computable[
        "max_angle_theoretical_vs_month_deg"
    ].max()
)

mean_theoretical_angle = float(
    computable[
        "max_angle_theoretical_vs_month_deg"
    ].mean()
)


# =============================================================================
# 10. VALIDATION
# =============================================================================

checks = []


def add(
    check,
    condition,
    observed,
    expected
):

    checks.append({
        "check":
            check,

        "status":
            (
                "PASS"
                if bool(
                    condition
                )
                else "FAIL"
            ),

        "observed":
            observed,

        "expected":
            expected,
    })


add(
    "global_rank",
    global_diag["rank"] == 11,
    global_diag["rank"],
    11
)

add(
    "global_nullity",
    global_diag["nullity"] == 2,
    global_diag["nullity"],
    2
)

add(
    "theoretical_nullspace_dimension",
    theoretical_nullspace.shape[1] == 2,
    theoretical_nullspace.shape[1],
    2
)

add(
    "all_raw_relations_validate",
    bool(
        relation_df[
            "raw_allclose_zero"
        ].all()
    ),
    int(
        (
            ~relation_df[
                "raw_allclose_zero"
            ]
        ).sum()
    ),
    0
)

add(
    "all_common_scale_relations_validate",
    bool(
        relation_df[
            "common_scale_allclose_zero"
        ].all()
    ),
    int(
        (
            ~relation_df[
                "common_scale_allclose_zero"
            ]
        ).sum()
    ),
    0
)

add(
    "all_computable_months_rank_11",
    bool(
        (
            computable[
                "numerical_rank"
            ] == 11
        ).all()
    ),
    int(
        (
            computable[
                "numerical_rank"
            ] != 11
        ).sum()
    ),
    0
)

add(
    "all_computable_months_nullity_2",
    bool(
        (
            computable[
                "nullity"
            ] == 2
        ).all()
    ),
    int(
        (
            computable[
                "nullity"
            ] != 2
        ).sum()
    ),
    0
)

add(
    "all_angles_finite",
    bool(
        np.isfinite(
            computable[
                [
                    "max_angle_global_vs_month_deg",
                    "max_angle_theoretical_vs_month_deg",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        ).all()
    ),
    int(
        (
            ~np.isfinite(
                computable[
                    [
                        "max_angle_global_vs_month_deg",
                        "max_angle_theoretical_vs_month_deg",
                    ]
                ]
                .to_numpy(
                    dtype=float
                )
            )
        ).sum()
    ),
    0
)


validation_df = pd.DataFrame(
    checks
)

validation_df.to_csv(
    OUT_VALIDATION,
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
# 11. SUMMARY
# =============================================================================

summary = {
    "step":
        "MODEL_01_0_C3_2_1_COMMON_SCALE_NULLSPACE",

    "status":
        (
            "PASS"
            if failures == 0
            else "FAIL"
        ),

    "methodological_patch":
        "COMMON_GLOBAL_MEAN_AND_SCALE_FOR_ALL_TEMPORAL_SUBSPACE_COMPARISONS",

    "rows":
        int(
            len(
                df
            )
        ),

    "features":
        int(
            len(
                features
            )
        ),

    "global_rank":
        int(
            global_diag[
                "rank"
            ]
        ),

    "global_nullity":
        int(
            global_diag[
                "nullity"
            ]
        ),

    "known_independent_relations":
        int(
            theoretical_nullspace.shape[1]
        ),

    "computable_months":
        int(
            len(
                computable
            )
        ),

    "temporal_rank_min":
        rank_min,

    "temporal_rank_max":
        rank_max,

    "temporal_nullity_min":
        nullity_min,

    "temporal_nullity_max":
        nullity_max,

    "global_vs_theoretical_max_angle_deg":
        float(
            np.max(
                angles_global_theoretical
            )
        ),

    "temporal_global_max_angle_deg":
        max_global_angle,

    "temporal_global_mean_max_angle_deg":
        mean_global_angle,

    "temporal_theoretical_max_angle_deg":
        max_theoretical_angle,

    "temporal_theoretical_mean_max_angle_deg":
        mean_theoretical_angle,

    "structural_nullspace_assessment":
        (
            "INVARIANT_IDENTITIES_CONFIRMED"
            if failures == 0
            else "REVIEW_REQUIRED"
        ),

    "automatic_feature_removal":
        False,

    "functional_module_unlocked":
        False,

    "threshold_frozen":
        False,

    "boundaries_frozen":
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
}


OUT_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 12. REPORT
# =============================================================================

lines = []

lines.append(
    "=" * 100
)

lines.append(
    "MODEL 01.0-C3.2.1 — COMMON-SCALE NULLSPACE VALIDATION"
)

lines.append(
    "=" * 100
)

lines.append(
    ""
)

lines.append(
    f"STATUS                     : {summary['status']}"
)

lines.append(
    f"GLOBAL RANK                : {summary['global_rank']}"
)

lines.append(
    f"GLOBAL NULLITY             : {summary['global_nullity']}"
)

lines.append(
    f"KNOWN RELATIONS            : {summary['known_independent_relations']}"
)

lines.append(
    f"COMPUTABLE MONTHS          : {summary['computable_months']}"
)

lines.append(
    f"TEMPORAL RANK              : {rank_min} .. {rank_max}"
)

lines.append(
    f"TEMPORAL NULLITY           : {nullity_min} .. {nullity_max}"
)

lines.append(
    ""
)

lines.append(
    "COMMON-SCALE ANGLES"
)

lines.append(
    "-" * 100
)

lines.append(
    "Global numerical vs theoretical max angle: "
    f"{summary['global_vs_theoretical_max_angle_deg']:.12e} deg"
)

lines.append(
    "Maximum month/global angle: "
    f"{max_global_angle:.12e} deg"
)

lines.append(
    "Mean maximum month/global angle: "
    f"{mean_global_angle:.12e} deg"
)

lines.append(
    "Maximum month/theoretical angle: "
    f"{max_theoretical_angle:.12e} deg"
)

lines.append(
    "Mean maximum month/theoretical angle: "
    f"{mean_theoretical_angle:.12e} deg"
)

lines.append(
    ""
)

lines.append(
    "ASSESSMENT"
)

lines.append(
    "-" * 100
)

lines.append(
    summary[
        "structural_nullspace_assessment"
    ]
)

lines.append(
    ""
)

lines.append(
    "No feature was removed."
)

lines.append(
    "No temporal boundary was frozen."
)

lines.append(
    "No fold was frozen."
)

lines.append(
    "No model was trained."
)

lines.append(
    "RAW was not modified."
)

lines.append(
    "=" * 100
)


OUT_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 13. PRINT
# =============================================================================

print()
print("=" * 112)
print("VALIDATION")
print("=" * 112)

print(
    validation_df.to_string(
        index=False
    )
)


print()
print("=" * 112)
print("COMMON-SCALE TEMPORAL NULLSPACE")
print("=" * 112)

print(
    temporal_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.12e}"
    )
)


print()
print("=" * 112)
print("KNOWN RELATION VALIDATION — SUMMARY")
print("=" * 112)

relation_summary = (
    relation_df
    .groupby(
        "relation",
        as_index=False
    )
    .agg(
        rows_checked=(
            "rows",
            "sum"
        ),

        max_raw_residual=(
            "raw_max_abs_residual",
            "max"
        ),

        max_common_scale_residual=(
            "common_scale_max_abs_residual",
            "max"
        ),

        raw_failures=(
            "raw_allclose_zero",
            lambda s: int(
                (~s).sum()
            )
        ),

        common_scale_failures=(
            "common_scale_allclose_zero",
            lambda s: int(
                (~s).sum()
            )
        ),
    )
)

print(
    relation_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.12e}"
    )
)


print()
print("=" * 112)
print("RESULTADO C3.2.1")
print("=" * 112)

print(
    "FAILURES                           =",
    failures
)

print(
    "GLOBAL RANK                        =",
    global_diag[
        "rank"
    ]
)

print(
    "GLOBAL NULLITY                     =",
    global_diag[
        "nullity"
    ]
)

print(
    "COMPUTABLE MONTHS                  =",
    len(
        computable
    )
)

print(
    "TEMPORAL RANK                      =",
    f"{rank_min} .. {rank_max}"
)

print(
    "TEMPORAL NULLITY                   =",
    f"{nullity_min} .. {nullity_max}"
)

print(
    "GLOBAL/THEORETICAL MAX ANGLE DEG   =",
    float(
        np.max(
            angles_global_theoretical
        )
    )
)

print(
    "MONTH/GLOBAL MAX ANGLE DEG         =",
    max_global_angle
)

print(
    "MONTH/THEORETICAL MAX ANGLE DEG    =",
    max_theoretical_angle
)

print()
print(
    "STRUCTURAL NULLSPACE ASSESSMENT    =",
    summary[
        "structural_nullspace_assessment"
    ]
)

print()
print("FEATURE REMOVAL                     = NÃO")
print("FUNCTIONAL MODULE UNLOCKED          = NÃO")
print("THRESHOLD FROZEN                    = NÃO")
print("BOUNDARIES FROZEN                   = NÃO")
print("FOLDS FROZEN                        = NÃO")
print("MODEL TRAINED                       = NÃO")
print("SILVER CREATED                      = NÃO")
print("RAW MODIFIED                        = NÃO")


if failures:

    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.2.1 common-scale validation concluída."
)

print(
    "[PASS] Nenhuma decisão de modelagem aplicada."
)

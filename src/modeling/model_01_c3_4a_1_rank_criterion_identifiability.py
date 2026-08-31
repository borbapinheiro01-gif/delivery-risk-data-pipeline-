#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.4-A.1
AUDIT OF RANK-CRITERION IDENTIFIABILITY
===============================================================================

OBJETIVO
--------
Auditar se o critério usado em C3.4-A:

    future full-vector reconstruction error

é capaz, sozinho, de identificar um K ótimo.

RESULTADO TEÓRICO
-----------------
Para PCA aninhada:

    span(V_K) subset span(V_{K+1})

portanto:

    ||X - X P_{K+1}||_F <= ||X - X P_K||_F

onde:

    P_K = V_K V_K^T

No posto completo p=30:

    P_30 = I_30

logo:

    ||X - X P_30||_F = 0

Assim, minimizar somente erro de reconstrução full-vector
favorece estruturalmente o maior K possível.

ESTE SCRIPT:
- valida empiricamente a monotonicidade observada;
- mede dimensionalidade retida;
- mede ganhos marginais;
- classifica o critério;
- NÃO escolhe K;
- NÃO executa PCA/SVD;
- NÃO usa target;
- NÃO altera RAW.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import numpy as np
import pandas as pd


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

FOLDS = (
    DIR
    / "04a_raw30_k_policy_folds.csv"
)

SUMMARY = (
    DIR
    / "04b_raw30_k_policy_summary.csv"
)

ONE_SE = (
    DIR
    / "04c_raw30_k_policy_one_se.csv"
)

SUMMARY_JSON = (
    DIR
    / "04e_raw30_k_policy_summary.json"
)

OUT_AUDIT = (
    DIR
    / "04g_rank_criterion_identifiability_audit.csv"
)

OUT_MARGINAL = (
    DIR
    / "04h_rank_policy_marginal_tradeoff.csv"
)

OUT_VALIDATION = (
    DIR
    / "04i_rank_criterion_identifiability_validation.csv"
)

OUT_JSON = (
    DIR
    / "04j_rank_criterion_identifiability_summary.json"
)

OUT_REPORT = (
    DIR
    / "04k_rank_criterion_identifiability_report.txt"
)


P = 30


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def add_check(
    rows,
    name,
    condition,
    observed,
    expected
):
    rows.append({
        "check": name,
        "status": "PASS" if bool(condition) else "FAIL",
        "observed": str(observed),
        "expected": str(expected)
    })


print()
print("=" * 116)
print("MODEL 01.0-C3.4-A.1 — RANK CRITERION IDENTIFIABILITY")
print("=" * 116)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

for path in [
    FOLDS,
    SUMMARY,
    ONE_SE,
    SUMMARY_JSON
]:
    if not path.is_file():
        print("[FAIL] Ausente:", path)
        sys.exit(2)

    print("[PASS]", path.name)


meta = load_json(
    SUMMARY_JSON
)

if meta.get("status") != "PASS":
    print("[FAIL] C3.4-A não está PASS.")
    sys.exit(2)


folds = pd.read_csv(
    FOLDS
)

summary = pd.read_csv(
    SUMMARY
)

one_se = pd.read_csv(
    ONE_SE
)


# =============================================================================
# 2. EMPIRICAL MONOTONICITY
# =============================================================================

audit_rows = []

monotonic_error_failures = 0
monotonic_k_failures = 0
groups = 0


for (
    channel,
    month
), g in folds.groupby(
    [
        "channel",
        "current_month"
    ],
    sort=True
):

    groups += 1

    g = g.sort_values(
        "pve_policy"
    ).copy()

    q = g[
        "pve_policy"
    ].to_numpy(
        dtype=float
    )

    k = g[
        "k_train"
    ].to_numpy(
        dtype=float
    )

    error = g[
        "future_relative_error"
    ].to_numpy(
        dtype=float
    )

    k_nondecreasing = bool(
        np.all(
            np.diff(k) >= 0
        )
    )

    error_nonincreasing = bool(
        np.all(
            np.diff(error) <= 1e-10
        )
    )

    if not k_nondecreasing:
        monotonic_k_failures += 1

    if not error_nonincreasing:
        monotonic_error_failures += 1

    audit_rows.append({
        "channel":
            channel,

        "current_month":
            month,

        "policies":
            len(g),

        "k_min":
            int(
                k.min()
            ),

        "k_max":
            int(
                k.max()
            ),

        "error_at_lowest_pve":
            float(
                error[0]
            ),

        "error_at_highest_pve":
            float(
                error[-1]
            ),

        "k_nondecreasing":
            k_nondecreasing,

        "error_nonincreasing":
            error_nonincreasing,
    })


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    OUT_AUDIT,
    index=False
)


# =============================================================================
# 3. MARGINAL TRADE-OFF
# =============================================================================

marginal_rows = []


for channel, g in summary.groupby(
    "channel",
    sort=True
):

    g = g.sort_values(
        "pve_policy"
    ).reset_index(
        drop=True
    )


    for i, row in g.iterrows():

        mean_k = float(
            row[
                "mean_k"
            ]
        )

        error = float(
            row[
                "mean_future_error"
            ]
        )

        retained_pct = (
            100.0
            *
            mean_k
            /
            P
        )

        if i == 0:

            previous_q = np.nan
            added_k = np.nan
            error_reduction = np.nan
            gain_per_component = np.nan

        else:

            prev = g.iloc[
                i - 1
            ]

            previous_q = float(
                prev[
                    "pve_policy"
                ]
            )

            added_k = (
                mean_k
                -
                float(
                    prev[
                        "mean_k"
                    ]
                )
            )

            error_reduction = (
                float(
                    prev[
                        "mean_future_error"
                    ]
                )
                -
                error
            )

            if added_k > 0:

                gain_per_component = (
                    error_reduction
                    /
                    added_k
                )

            else:

                gain_per_component = np.nan


        marginal_rows.append({
            "channel":
                channel,

            "pve_policy":
                float(
                    row[
                        "pve_policy"
                    ]
                ),

            "mean_k":
                mean_k,

            "mean_dimension_retained_pct":
                retained_pct,

            "mean_future_error":
                error,

            "previous_policy":
                previous_q,

            "added_mean_components":
                added_k,

            "future_error_reduction":
                error_reduction,

            "error_reduction_per_added_component":
                gain_per_component,
        })


marginal = pd.DataFrame(
    marginal_rows
)

marginal.to_csv(
    OUT_MARGINAL,
    index=False
)


# =============================================================================
# 4. q=.99 DIAGNOSTIC
# =============================================================================

q99 = marginal.loc[
    np.isclose(
        marginal[
            "pve_policy"
        ],
        0.99
    )
].copy()


q99_retained_min = float(
    q99[
        "mean_dimension_retained_pct"
    ].min()
)

q99_retained_max = float(
    q99[
        "mean_dimension_retained_pct"
    ].max()
)


# =============================================================================
# 5. FULL-RANK MATHEMATICAL LIMIT
# =============================================================================

# This is a linear-algebra identity, not an empirical threshold:
#
# V_30 orthogonal complete basis => V_30 V_30^T = I
#
# Therefore:
#
# X - X V_30 V_30^T = 0.
#
# We record it as theoretical structure.

full_rank_dimension = P
full_rank_relative_error = 0.0


# =============================================================================
# 6. IDENTIFIABILITY CLASSIFICATION
# =============================================================================

criterion_status = (
    "NON_IDENTIFYING_FOR_INTRINSIC_RANK_SELECTION"
)


criterion_reason = (
    "Full-vector PCA reconstruction error is nested and "
    "nonincreasing in K; at K=30 the complete orthonormal "
    "basis reconstructs any centered 30-dimensional test "
    "vector exactly. Therefore minimizing this criterion "
    "alone structurally favors maximal rank."
)


# =============================================================================
# 7. VALIDATION
# =============================================================================

checks = []


add_check(
    checks,
    "c34a_pass",
    meta.get(
        "status"
    )
    ==
    "PASS",
    meta.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "channel_month_groups_34",
    groups == 34,
    groups,
    34
)


add_check(
    checks,
    "k_monotonic_failures_zero",
    monotonic_k_failures == 0,
    monotonic_k_failures,
    0
)


add_check(
    checks,
    "error_monotonic_failures_zero",
    monotonic_error_failures == 0,
    monotonic_error_failures,
    0
)


joint = float(
    meta.get(
        "joint_candidate_policy"
    )
)


add_check(
    checks,
    "original_one_se_candidate_099",
    np.isclose(
        joint,
        0.99
    ),
    joint,
    0.99
)


add_check(
    checks,
    "q099_retains_most_dimensions",
    q99_retained_min > 80.0,
    (
        q99_retained_min,
        q99_retained_max
    ),
    ">80% mean dimensions retained"
)


add_check(
    checks,
    "full_rank_dimension_30",
    full_rank_dimension == 30,
    full_rank_dimension,
    30
)


add_check(
    checks,
    "full_rank_theoretical_reconstruction_error_zero",
    full_rank_relative_error == 0.0,
    full_rank_relative_error,
    0.0
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
# 8. SUMMARY
# =============================================================================

result = {
    "step":
        "MODEL_01_0_C3_4A_1_RANK_CRITERION_IDENTIFIABILITY",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_step":
        "MODEL_01_0_C3_4A_TEMPORAL_K_POLICY",

    "representation":
        "RAW_30D",

    "feature_dimension":
        30,

    "channel_month_groups":
        groups,

    "monotonic_k_failures":
        monotonic_k_failures,

    "monotonic_error_failures":
        monotonic_error_failures,

    "original_one_se_candidate":
        joint,

    "q099_mean_dimension_retained_pct_range": [
        q99_retained_min,
        q99_retained_max
    ],

    "full_rank_k":
        30,

    "full_rank_theoretical_error":
        0.0,

    "criterion_status":
        criterion_status,

    "criterion_reason":
        criterion_reason,

    "interpretation": {
        "q099_valid_as_reconstruction_candidate":
            True,

        "q099_valid_as_intrinsic_rank_estimate":
            False,

        "final_k_selected":
            False,

        "next_method":
            "TEMPORAL_BI_CROSS_VALIDATION_RANK_SELECTION"
    },

    "governance": {
        "target_used":
            False,

        "pca_executed":
            False,

        "svd_executed":
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
            False
    },

    "validation_failures":
        failures
}


OUT_JSON.write_text(
    json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 9. REPORT
# =============================================================================

lines = [
    "=" * 116,
    "MODEL 01.0-C3.4-A.1 — RANK CRITERION IDENTIFIABILITY",
    "=" * 116,
    "",
    "STATUS",
    "-" * 116,
    (
        "Validation                         : "
        +
        result[
            "status"
        ]
    ),
    (
        "Criterion                          : "
        +
        criterion_status
    ),
    "",
    "EMPIRICAL RESULT",
    "-" * 116,
    (
        "Channel-month groups               : "
        +
        str(
            groups
        )
    ),
    (
        "K monotonic failures               : "
        +
        str(
            monotonic_k_failures
        )
    ),
    (
        "Error monotonic failures           : "
        +
        str(
            monotonic_error_failures
        )
    ),
    (
        "Original 1-SE candidate             : "
        +
        str(
            joint
        )
    ),
    (
        "q=.99 mean dimension retained       : "
        +
        f"{q99_retained_min:.2f}% .. "
        +
        f"{q99_retained_max:.2f}%"
    ),
    "",
    "MATHEMATICAL LIMIT",
    "-" * 116,
    "For nested PCA subspaces:",
    "",
    "    E_(K+1) <= E_K",
    "",
    "and for the complete 30-dimensional basis:",
    "",
    "    E_30 = 0.",
    "",
    "INTERPRETATION",
    "-" * 116,
    (
        "q=.99 is valid as the best candidate among the "
        "tested policies for full-vector reconstruction."
    ),
    (
        "It is NOT identified as the intrinsic or optimal "
        "dimension of the functional representation."
    ),
    "",
    "NEXT METHOD",
    "-" * 116,
    "Temporal bi-cross-validation of SVD/PCA rank.",
    "Hold out both future rows and lag columns.",
    "",
    "GOVERNANCE",
    "-" * 116,
    "Target used                         : NO",
    "New PCA/SVD executed                : NO",
    "Final K selected                    : NO",
    "Folds frozen                        : NO",
    "Classifier trained                  : NO",
    "Silver created                      : NO",
    "RAW modified                        : NO",
    "=" * 116,
]


OUT_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 10. PRINT
# =============================================================================

print()
print("=" * 116)
print("MARGINAL TRADE-OFF")
print("=" * 116)

print(
    marginal.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
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
print("RESULTADO C3.4-A.1")
print("=" * 116)

print(
    "STATUS                         =",
    result[
        "status"
    ]
)

print(
    "CRITERION                      =",
    criterion_status
)

print(
    "CHANNEL-MONTH GROUPS           =",
    groups
)

print(
    "ERROR MONOTONIC FAILURES       =",
    monotonic_error_failures
)

print(
    "ORIGINAL 1-SE CANDIDATE        =",
    joint
)

print(
    "q=.99 DIMENSIONS RETAINED (%)  =",
    f"{q99_retained_min:.2f}",
    "..",
    f"{q99_retained_max:.2f}"
)

print(
    "FULL RANK K                    =",
    full_rank_dimension
)

print(
    "THEORETICAL ERROR AT K=30      =",
    full_rank_relative_error
)

print()
print("q=.99 reconstruction candidate : SIM")
print("q=.99 intrinsic rank selected   : NÃO")
print("K final                         : NÃO")
print("Target usado                    : NÃO")
print("PCA/SVD executada               : NÃO")
print("Modelo treinado                 : NÃO")
print("RAW alterado                    : NÃO")


if failures:
    sys.exit(2)


print()
print("[PASS] Critério C3.4-A auditado.")
print("[PASS] Não congelar q=.99 como dimensão ótima.")
print("[PASS] Próxima etapa: temporal bi-cross-validation.")

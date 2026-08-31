#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.4-A.1
RECONSTRUCTION-RANK IDENTIFIABILITY AUDIT
===============================================================================

OBJETIVO
--------
Formalizar a limitação identificada no C3.4-A.

Para PCA com subespaços aninhados:

    S_K subset S_{K+1}

logo:

    ||X - P_{K+1}X|| <= ||X - P_K X||

e, no posto completo p=30:

    P_30 = I
    erro_30 = 0

Portanto, erro de reconstrução full-vector não identifica,
sozinho, uma dimensão intrínseca ótima.

NÃO EXECUTA:
- PCA
- SVD
- smoothing
- BCV
- target
- modelo

Apenas lê os artefatos existentes do C3.4-A.
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

FOLDS = DIR / "04a_raw30_k_policy_folds.csv"
SUMMARY = DIR / "04b_raw30_k_policy_summary.csv"
ONESE = DIR / "04c_raw30_k_policy_one_se.csv"
C34A = DIR / "04e_raw30_k_policy_summary.json"

OUT_AUDIT = DIR / "04g_rank_identifiability_audit.csv"
OUT_TRADEOFF = DIR / "04h_rank_marginal_tradeoff.csv"
OUT_VALIDATION = DIR / "04i_rank_identifiability_validation.csv"
OUT_JSON = DIR / "04j_rank_identifiability_summary.json"
OUT_REPORT = DIR / "04k_rank_identifiability_report.txt"


P = 30


def add_check(rows, name, ok, observed, expected):
    rows.append({
        "check": name,
        "status": "PASS" if bool(ok) else "FAIL",
        "observed": str(observed),
        "expected": str(expected),
    })


print()
print("=" * 112)
print("MODEL 01.0-C3.4-A.1 — RANK IDENTIFIABILITY AUDIT")
print("=" * 112)


# =============================================================================
# 1. PRÉ-REQUISITOS
# =============================================================================

for p in [FOLDS, SUMMARY, ONESE, C34A]:

    if not p.is_file():
        print("[FAIL] Ausente:", p)
        sys.exit(2)

    print("[PASS]", p.name)


meta = json.loads(
    C34A.read_text(encoding="utf-8")
)

if meta.get("status") != "PASS":
    raise SystemExit(
        "[FAIL] C3.4-A não está PASS."
    )


folds = pd.read_csv(FOLDS)
summary = pd.read_csv(SUMMARY)
one_se = pd.read_csv(ONESE)


# =============================================================================
# 2. MONOTONICIDADE EMPÍRICA
# =============================================================================

audit_rows = []

k_failures = 0
error_failures = 0


for (channel, month), g in folds.groupby(
    ["channel", "current_month"],
    sort=True
):

    g = g.sort_values(
        "pve_policy"
    ).copy()

    k = g["k_train"].to_numpy(
        dtype=float
    )

    err = g[
        "future_relative_error"
    ].to_numpy(
        dtype=float
    )

    k_ok = bool(
        np.all(
            np.diff(k) >= 0
        )
    )

    error_ok = bool(
        np.all(
            np.diff(err) <= 1e-10
        )
    )

    if not k_ok:
        k_failures += 1

    if not error_ok:
        error_failures += 1

    audit_rows.append({
        "channel": channel,
        "current_month": month,
        "policies_tested": len(g),
        "k_min": int(k.min()),
        "k_max": int(k.max()),
        "lowest_pve_error": float(err[0]),
        "highest_pve_error": float(err[-1]),
        "k_nondecreasing": k_ok,
        "error_nonincreasing": error_ok,
    })


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    OUT_AUDIT,
    index=False
)


# =============================================================================
# 3. TRADE-OFF MARGINAL
# =============================================================================

rows = []


for channel, g in summary.groupby(
    "channel",
    sort=True
):

    g = (
        g.sort_values("pve_policy")
        .reset_index(drop=True)
    )

    for i, r in g.iterrows():

        q = float(
            r["pve_policy"]
        )

        mean_k = float(
            r["mean_k"]
        )

        error = float(
            r["mean_future_error"]
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

            previous = g.iloc[i - 1]

            previous_q = float(
                previous["pve_policy"]
            )

            added_k = (
                mean_k
                -
                float(
                    previous["mean_k"]
                )
            )

            error_reduction = (
                float(
                    previous[
                        "mean_future_error"
                    ]
                )
                -
                error
            )

            gain_per_component = (
                error_reduction / added_k
                if added_k > 0
                else np.nan
            )

        rows.append({
            "channel": channel,
            "pve_policy": q,
            "mean_k": mean_k,
            "dimension_retained_pct": retained_pct,
            "dimension_removed_pct": 100.0 - retained_pct,
            "mean_future_error": error,
            "previous_policy": previous_q,
            "added_mean_components": added_k,
            "future_error_reduction": error_reduction,
            "error_reduction_per_added_component": gain_per_component,
        })


tradeoff = pd.DataFrame(
    rows
)

tradeoff.to_csv(
    OUT_TRADEOFF,
    index=False
)


# =============================================================================
# 4. q=.99
# =============================================================================

q99 = tradeoff.loc[
    np.isclose(
        tradeoff["pve_policy"],
        0.99
    )
].copy()


q99_min_retained = float(
    q99[
        "dimension_retained_pct"
    ].min()
)

q99_max_retained = float(
    q99[
        "dimension_retained_pct"
    ].max()
)


joint_candidate = float(
    meta[
        "joint_candidate_policy"
    ]
)


# =============================================================================
# 5. VALIDAÇÃO
# =============================================================================

checks = []


add_check(
    checks,
    "c34a_pass",
    meta.get("status") == "PASS",
    meta.get("status"),
    "PASS"
)


add_check(
    checks,
    "channel_month_groups",
    len(audit) == 34,
    len(audit),
    34
)


add_check(
    checks,
    "k_monotonic_failures",
    k_failures == 0,
    k_failures,
    0
)


add_check(
    checks,
    "reconstruction_error_monotonic_failures",
    error_failures == 0,
    error_failures,
    0
)


add_check(
    checks,
    "c34a_candidate_is_099",
    np.isclose(
        joint_candidate,
        0.99
    ),
    joint_candidate,
    0.99
)


add_check(
    checks,
    "q099_retains_over_80pct_dimensions",
    q99_min_retained > 80.0,
    (
        f"{q99_min_retained:.4f}% .. "
        f"{q99_max_retained:.4f}%"
    ),
    ">80%"
)


add_check(
    checks,
    "full_rank_dimension",
    P == 30,
    P,
    30
)


# Identidade teórica:
#
# V_p V_p^T = I
# => erro de projeção no posto completo = 0.

theoretical_full_rank_error = 0.0


add_check(
    checks,
    "theoretical_full_rank_error",
    theoretical_full_rank_error == 0.0,
    theoretical_full_rank_error,
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
    ].eq("FAIL").sum()
)


# =============================================================================
# 6. CLASSIFICAÇÃO
# =============================================================================

criterion_class = (
    "NON_IDENTIFYING_FOR_INTRINSIC_RANK"
)


decision = {
    "step":
        "MODEL_01_0_C3_4A_1_RANK_IDENTIFIABILITY",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source":
        "MODEL_01_0_C3_4A_TEMPORAL_K_POLICY",

    "representation":
        "RAW_30D",

    "dimension":
        P,

    "channel_month_groups":
        len(audit),

    "k_monotonic_failures":
        k_failures,

    "reconstruction_error_monotonic_failures":
        error_failures,

    "original_reconstruction_candidate":
        joint_candidate,

    "q099_mean_dimension_retained_pct_range": [
        q99_min_retained,
        q99_max_retained,
    ],

    "full_rank_theoretical_error":
        0.0,

    "criterion_classification":
        criterion_class,

    "interpretation": {
        "q099_best_tested_reconstruction_policy":
            True,

        "q099_intrinsic_rank_estimate":
            False,

        "final_k_selected":
            False,

        "reason":
            (
                "Nested PCA reconstruction error "
                "decreases structurally as K increases "
                "and is zero at complete rank."
            ),

        "next_method":
            (
                "TEMPORAL_BI_CROSS_VALIDATION_"
                "OF_LOW_RANK_STRUCTURE"
            ),
    },

    "governance": {
        "target_used": False,
        "pca_executed": False,
        "svd_executed": False,
        "bcv_executed": False,
        "final_k_selected": False,
        "folds_frozen": False,
        "classifier_trained": False,
        "silver_created": False,
        "raw_modified": False,
    },

    "validation_failures":
        failures,
}


OUT_JSON.write_text(
    json.dumps(
        decision,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 7. RELATÓRIO
# =============================================================================

report = f"""
================================================================================================================
MODEL 01.0-C3.4-A.1 — RANK IDENTIFIABILITY AUDIT
================================================================================================================

STATUS
----------------------------------------------------------------------------------------------------------------
Validation status                       : {decision["status"]}
Criterion classification                : {criterion_class}

EMPIRICAL EVIDENCE
----------------------------------------------------------------------------------------------------------------
Channel-month groups                    : {len(audit)}
K monotonic failures                    : {k_failures}
Reconstruction-error monotonic failures : {error_failures}

C3.4-A reconstruction candidate         : {joint_candidate}

q=.99 dimension retained
----------------------------------------------------------------------------------------------------------------
Minimum channel mean                    : {q99_min_retained:.4f}%
Maximum channel mean                    : {q99_max_retained:.4f}%

MATHEMATICAL STRUCTURE
----------------------------------------------------------------------------------------------------------------
For nested PCA spaces:

    S_K subset S_(K+1)

therefore:

    ||X - P_(K+1)X||_F <= ||X - P_K X||_F

At complete rank p = 30:

    P_30 = I

and therefore:

    ||X - P_30 X||_F = 0

INTERPRETATION
----------------------------------------------------------------------------------------------------------------
q=.99 is the best TESTED reconstruction policy.

It is NOT established as an intrinsic-rank estimate.

Full-vector reconstruction error alone is NON-IDENTIFYING
for intrinsic-rank selection in this experiment.

NEXT
----------------------------------------------------------------------------------------------------------------
Temporal bi-cross-validation of low-rank structure.

No final K is frozen yet.

GOVERNANCE
----------------------------------------------------------------------------------------------------------------
Target used                    : NO
PCA executed in this step      : NO
SVD executed in this step      : NO
BCV executed                   : NO
Final K selected               : NO
Folds frozen                   : NO
Classifier trained             : NO
Silver created                 : NO
RAW modified                   : NO

Validation failures            : {failures}
================================================================================================================
""".strip()


OUT_REPORT.write_text(
    report + "\n",
    encoding="utf-8"
)


# =============================================================================
# 8. OUTPUT
# =============================================================================

print()
print("=" * 112)
print("MARGINAL TRADE-OFF")
print("=" * 112)

print(
    tradeoff.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print()
print("=" * 112)
print("VALIDATION")
print("=" * 112)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 112)
print("RESULTADO C3.4-A.1")
print("=" * 112)

print(
    "STATUS                          :",
    decision["status"]
)

print(
    "CRITERION                       :",
    criterion_class
)

print(
    "CHANNEL-MONTH GROUPS            :",
    len(audit)
)

print(
    "K MONOTONIC FAILURES            :",
    k_failures
)

print(
    "ERROR MONOTONIC FAILURES        :",
    error_failures
)

print(
    "q=.99 RECONSTRUCTION CANDIDATE  :",
    "SIM"
)

print(
    "q=.99 INTRINSIC RANK ESTIMATE   :",
    "NÃO"
)

print(
    "DIMENSION RETAINED q=.99        :",
    f"{q99_min_retained:.2f}% .. "
    f"{q99_max_retained:.2f}%"
)

print()
print("K FINAL                         : NÃO")
print("TARGET                          : NÃO")
print("PCA/SVD NESTA ETAPA             : NÃO")
print("BCV                             : NÃO")
print("MODELO                          : NÃO")
print("SILVER                          : NÃO")
print("RAW                             : INTACTO")


if failures:
    sys.exit(2)


print()
print("[PASS] C3.4-A.1 concluído.")
print("[PASS] Não congelar q=.99 como rank ótimo.")
print("[PASS] Próxima etapa: C3.4-B temporal BCV.")

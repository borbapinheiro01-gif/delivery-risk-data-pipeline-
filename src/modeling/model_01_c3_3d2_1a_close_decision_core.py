#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-D2.1-A
CORE FUNCTIONAL REPRESENTATION DECISION
===============================================================================

OBJETIVO
--------
Fechar metodologicamente a comparação:

    RAW-30D
    versus
    SMOOTH-30D-EDF0.75

usando SOMENTE os artefatos já calculados.

NÃO EXECUTA:
- PCA;
- SVD;
- smoothing;
- curvas PIT;
- classificação;
- target;
- Silver.

NÃO MODIFICA:
- D2 original;
- RAW;
- matrizes funcionais.

CORREÇÃO
--------
O D2 original falhou somente porque exigia encontrar valores negativos
dentro dos folds futuros:

    negative_values_preserved_not_clipped > 0

Essa exigência é incorreta.

D1.1 já demonstrou negativos na população global auditada.

D2 encontrou zero negativos nos seus próprios subconjuntos temporais.

Esses dois fatos são compatíveis.
"""

from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, median
import csv
import hashlib
import json
import sys


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

D2_SUMMARY = DIR / "03o_raw_vs_smooth_summary.json"
D2_VALIDATION = DIR / "03n_raw_vs_smooth_validation.csv"
D2_PAIRED = DIR / "03m_raw_vs_smooth_paired_comparison.csv"
D2_TABLE = DIR / "03l_raw_vs_smooth_temporal_summary.csv"

D11_SUMMARY = DIR / "03i_negative_smoothing_summary.json"

OUT_VALIDATION = DIR / "03r_d2_methodological_validation_v2.csv"
OUT_DECISION = DIR / "03s_functional_representation_decision.json"
OUT_REPORT = DIR / "03t_functional_representation_decision_report.txt"


# =============================================================================
# HELPERS
# =============================================================================

def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_csv(path):
    with path.open(
        encoding="utf-8",
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def to_float(x):
    return float(
        str(x).strip()
    )


def to_int(x):
    return int(
        float(
            str(x).strip()
        )
    )


def to_bool(x):
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()

    if s == "true":
        return True

    if s == "false":
        return False

    raise ValueError(
        f"Boolean inválido: {x!r}"
    )


checks = []


def check(
    name,
    condition,
    observed,
    expected,
    interpretation=""
):
    checks.append({
        "check":
            name,

        "status":
            "PASS"
            if condition
            else "FAIL",

        "observed":
            str(observed),

        "expected":
            str(expected),

        "interpretation":
            interpretation
    })


# =============================================================================
# START
# =============================================================================

print()
print("=" * 108)
print("MODEL 01.0-C3.3-D2.1-A — CORE REPRESENTATION DECISION")
print("=" * 108)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required = [
    D2_SUMMARY,
    D2_VALIDATION,
    D2_PAIRED,
    D2_TABLE,
    D11_SUMMARY
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


# =============================================================================
# 2. HASHES — PRESERVAR D2
# =============================================================================

original_hashes_before = {
    path.name:
        sha256(path)
    for path in [
        D2_SUMMARY,
        D2_VALIDATION,
        D2_PAIRED,
        D2_TABLE
    ]
}


# =============================================================================
# 3. LOAD
# =============================================================================

d2 = read_json(
    D2_SUMMARY
)

d11 = read_json(
    D11_SUMMARY
)

validation_original = read_csv(
    D2_VALIDATION
)

paired = read_csv(
    D2_PAIRED
)


# =============================================================================
# 4. ORIGINAL D2 FAILURE
# =============================================================================

failed_original = [
    row
    for row in validation_original
    if str(
        row.get(
            "status",
            ""
        )
    ).upper() == "FAIL"
]


expected_bad_check = (
    "negative_values_preserved_not_clipped"
)


only_invalid_assertion_failed = (
    len(
        failed_original
    ) == 1
    and
    failed_original[
        0
    ].get(
        "check"
    )
    ==
    expected_bad_check
)


check(
    "original_d2_only_failed_invalid_negative_assertion",
    only_invalid_assertion_failed,
    [
        row.get(
            "check"
        )
        for row in failed_original
    ],
    [
        expected_bad_check
    ],
    (
        "D2 original is preserved. "
        "Its only failure was the invalid requirement "
        "that future subsets must contain negative cells."
    )
)


# =============================================================================
# 5. D1.1 — GLOBAL NEGATIVE EVIDENCE
# =============================================================================

d11_status = d11.get(
    "status"
)

d11_failures = d11.get(
    "validation_failures"
)

negative_configs = int(
    d11.get(
        "configurations_with_negative_cells",
        0
    )
)

audited_configs = int(
    d11.get(
        "configurations_audited",
        0
    )
)


check(
    "d11_status_pass",
    (
        d11_status == "PASS"
        and
        d11_failures == 0
    ),
    (
        d11_status,
        d11_failures
    ),
    (
        "PASS",
        0
    )
)


check(
    "d11_global_negative_artifacts_exist",
    negative_configs > 0,
    (
        f"{negative_configs}/"
        f"{audited_configs}"
    ),
    ">0",
    (
        "D1.1 remains the canonical evidence that "
        "unconstrained smoothing may generate negatives."
    )
)


# =============================================================================
# 6. D2 METADATA
# =============================================================================

check(
    "d2_target_not_used",
    d2.get(
        "target_used"
    ) is False,
    d2.get(
        "target_used"
    ),
    False
)


check(
    "d2_no_clipping",
    d2.get(
        "negative_values_clipped"
    ) is False,
    d2.get(
        "negative_values_clipped"
    ),
    False
)


check(
    "d2_raw_not_modified",
    d2.get(
        "raw_modified"
    ) is False,
    d2.get(
        "raw_modified"
    ),
    False
)


# =============================================================================
# 7. PAIRED DATA
# =============================================================================

required_columns = [
    "channel",
    "current_month",
    "raw_k90_train",
    "smooth_k90_train",
    "raw_future_relative_error_k90",
    "smooth_end_to_end_raw_error_k90",
    "end_to_end_error_delta_smooth_minus_raw",
    "end_to_end_error_ratio_smooth_over_raw",
    "smooth_lower_end_to_end_error",
    "smooth_negative_cell_pct_test"
]


missing_columns = [
    col
    for col in required_columns
    if (
        not paired
        or
        col not in paired[0]
    )
]


check(
    "paired_required_columns",
    len(
        missing_columns
    ) == 0,
    missing_columns,
    []
)


if missing_columns:
    print(
        "[STOP] Colunas ausentes."
    )
    sys.exit(2)


# =============================================================================
# 8. STRUCTURE
# =============================================================================

total_pairs = len(
    paired
)


channels = sorted(
    set(
        row[
            "channel"
        ]
        for row in paired
    )
)


counts = {
    channel:
        sum(
            1
            for row in paired
            if row[
                "channel"
            ]
            ==
            channel
        )
    for channel in channels
}


check(
    "paired_comparisons_34",
    total_pairs == 34,
    total_pairs,
    34
)


check(
    "expected_channels",
    channels
    ==
    [
        "purchase_freight",
        "purchase_volume"
    ],
    channels,
    [
        "purchase_freight",
        "purchase_volume"
    ]
)


check(
    "seventeen_tests_per_channel",
    counts
    ==
    {
        "purchase_freight":
            17,

        "purchase_volume":
            17
    },
    counts,
    {
        "purchase_freight":
            17,

        "purchase_volume":
            17
    }
)


# =============================================================================
# 9. NUMERICAL EVIDENCE
# =============================================================================

deltas = [
    to_float(
        row[
            "end_to_end_error_delta_smooth_minus_raw"
        ]
    )
    for row in paired
]


ratios = [
    to_float(
        row[
            "end_to_end_error_ratio_smooth_over_raw"
        ]
    )
    for row in paired
]


raw_k = [
    to_int(
        row[
            "raw_k90_train"
        ]
    )
    for row in paired
]


smooth_k = [
    to_int(
        row[
            "smooth_k90_train"
        ]
    )
    for row in paired
]


raw_wins = sum(
    x > 0
    for x in deltas
)

smooth_wins = sum(
    x < 0
    for x in deltas
)

ties = sum(
    abs(
        x
    ) <= 1e-12
    for x in deltas
)


compression_wins = sum(
    sk < rk
    for sk, rk in zip(
        smooth_k,
        raw_k
    )
)


check(
    "raw_lower_end_to_end_error_all_pairs",
    raw_wins == total_pairs,
    raw_wins,
    total_pairs,
    (
        "Positive delta = smooth reconstruction has "
        "higher error against future RAW curve."
    )
)


check(
    "smooth_zero_end_to_end_wins",
    smooth_wins == 0,
    smooth_wins,
    0
)


check(
    "all_end_to_end_ratios_above_one",
    all(
        x > 1.0
        for x in ratios
    ),
    min(
        ratios
    ),
    ">1"
)


check(
    "smoothing_compresses_k90_all_pairs",
    compression_wins
    ==
    total_pairs,
    compression_wins,
    total_pairs,
    (
        "Smoothing gives dimensional compression, "
        "but that alone is not evidence of superior fidelity."
    )
)


# =============================================================================
# 10. NEGATIVE CELLS IN D2 TEST FOLDS
# =============================================================================

negative_test_rows = sum(
    to_float(
        row[
            "smooth_negative_cell_pct_test"
        ]
    )
    >
    0
    for row in paired
)


# This is descriptive, not a requirement for >0.
check(
    "d2_test_negative_observation_is_descriptive",
    negative_test_rows >= 0,
    negative_test_rows,
    "DESCRIPTIVE_ONLY",
    (
        "Zero negatives in D2 subsets does not imply clipping. "
        "D1.1 and D2 concern different evaluated populations."
    )
)


# =============================================================================
# 11. CHANNEL SUMMARIES
# =============================================================================

channel_summary = []


for channel in channels:

    rows = [
        row
        for row in paired
        if row[
            "channel"
        ]
        ==
        channel
    ]


    raw_error = [
        to_float(
            row[
                "raw_future_relative_error_k90"
            ]
        )
        for row in rows
    ]


    smooth_error = [
        to_float(
            row[
                "smooth_end_to_end_raw_error_k90"
            ]
        )
        for row in rows
    ]


    delta = [
        to_float(
            row[
                "end_to_end_error_delta_smooth_minus_raw"
            ]
        )
        for row in rows
    ]


    ratio = [
        to_float(
            row[
                "end_to_end_error_ratio_smooth_over_raw"
            ]
        )
        for row in rows
    ]


    rk = [
        to_int(
            row[
                "raw_k90_train"
            ]
        )
        for row in rows
    ]


    sk = [
        to_int(
            row[
                "smooth_k90_train"
            ]
        )
        for row in rows
    ]


    channel_summary.append({
        "channel":
            channel,

        "temporal_tests":
            len(
                rows
            ),

        "raw_wins":
            sum(
                x > 0
                for x in delta
            ),

        "smooth_wins":
            sum(
                x < 0
                for x in delta
            ),

        "raw_future_error_mean":
            mean(
                raw_error
            ),

        "smooth_end_to_end_raw_error_mean":
            mean(
                smooth_error
            ),

        "delta_mean":
            mean(
                delta
            ),

        "delta_median":
            median(
                delta
            ),

        "ratio_mean":
            mean(
                ratio
            ),

        "raw_k90_median":
            median(
                rk
            ),

        "smooth_k90_median":
            median(
                sk
            ),

        "mean_k90_reduction":
            mean(
                rk_i - sk_i
                for rk_i, sk_i in zip(
                    rk,
                    sk
                )
            )
    })


# =============================================================================
# 12. VALIDATION V2
# =============================================================================

failures = sum(
    row[
        "status"
    ]
    ==
    "FAIL"
    for row in checks
)


with OUT_VALIDATION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "check",
            "status",
            "observed",
            "expected",
            "interpretation"
        ]
    )

    writer.writeheader()

    writer.writerows(
        checks
    )


# =============================================================================
# 13. DECISION
# =============================================================================

decision_supported = (
    failures == 0
    and
    raw_wins == 34
    and
    smooth_wins == 0
    and
    compression_wins == 34
)


decision = {
    "decision_id":
        "FUNCTIONAL_REPRESENTATION_D2_V1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        (
            "PASS"
            if decision_supported
            else
            "REVIEW_REQUIRED"
        ),

    "scientific_scope":
        (
            "UNSUPERVISED_TEMPORAL_FUNCTIONAL_REPRESENTATION"
        ),

    "primary_representation_for_next_stage": {
        "representation":
            "RAW_30D",

        "status":
            (
                "PRIMARY_FOR_NEXT_FUNCTIONAL_STAGE"
                if decision_supported
                else
                "HOLD"
            )
    },

    "smooth_challenger": {
        "representation":
            "SMOOTH_30D_EDF_0_75",

        "status":
            (
                "REJECTED_AS_PRIMARY_UNSUPERVISED_PREPROCESSING"
                if decision_supported
                else
                "HOLD"
            ),

        "not_rejected_for_all_predictive_uses":
            True
    },

    "paired_evidence": {
        "comparisons":
            total_pairs,

        "raw_wins":
            raw_wins,

        "smooth_wins":
            smooth_wins,

        "ties":
            ties,

        "smooth_k90_lower_pairs":
            compression_wins,

        "channels":
            channel_summary
    },

    "original_d2": {
        "status":
            d2.get(
                "status"
            ),

        "validation_failures":
            d2.get(
                "validation_failures"
            ),

        "preserved":
            True,

        "only_failed_check":
            (
                failed_original[
                    0
                ].get(
                    "check"
                )
                if len(
                    failed_original
                ) == 1
                else None
            )
    },

    "corrected_validation": {
        "status":
            (
                "PASS"
                if failures == 0
                else
                "FAIL"
            ),

        "validation_failures":
            failures,

        "artifact":
            str(
                OUT_VALIDATION.relative_to(
                    PROJECT
                )
            )
    },

    "scope_limits": {
        "target_used":
            False,

        "predictive_superiority_claimed":
            False,

        "final_k_selected":
            False,

        "production_representation_frozen":
            False
    },

    "next_stage": {
        "representation":
            "RAW_30D",

        "task":
            "TEMPORAL_COMPONENT_COUNT_POLICY"
    },

    "governance": {
        "clipping_applied":
            False,

        "smoothing_committed":
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

    "provenance_sha256":
        original_hashes_before
}


OUT_DECISION.write_text(
    json.dumps(
        decision,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 14. VERIFY ORIGINAL FILES UNCHANGED
# =============================================================================

original_hashes_after = {
    path.name:
        sha256(path)
    for path in [
        D2_SUMMARY,
        D2_VALIDATION,
        D2_PAIRED,
        D2_TABLE
    ]
}


originals_preserved = (
    original_hashes_before
    ==
    original_hashes_after
)


if not originals_preserved:
    print(
        "[FAIL] Arquivo original do D2 mudou."
    )
    sys.exit(4)


# =============================================================================
# 15. TEXT REPORT
# =============================================================================

lines = [
    "=" * 108,
    "MODEL 01.0-C3.3-D2.1-A — FUNCTIONAL REPRESENTATION DECISION",
    "=" * 108,
    "",
    (
        "STATUS                            : "
        + decision[
            "status"
        ]
    ),
    "",
    "DECISION",
    "-" * 108,
    (
        "RAW-30D                           : "
        +
        decision[
            "primary_representation_for_next_stage"
        ][
            "status"
        ]
    ),
    (
        "SMOOTH-30D-EDF0.75                : "
        +
        decision[
            "smooth_challenger"
        ][
            "status"
        ]
    ),
    "",
    "EMPIRICAL EVIDENCE",
    "-" * 108,
    (
        f"Paired temporal comparisons       : "
        f"{total_pairs}"
    ),
    (
        f"RAW wins                          : "
        f"{raw_wins}"
    ),
    (
        f"SMOOTH wins                       : "
        f"{smooth_wins}"
    ),
    (
        f"Ties                              : "
        f"{ties}"
    ),
    (
        f"Pairs with lower SMOOTH K90       : "
        f"{compression_wins}"
    ),
    ""
]


for row in channel_summary:

    lines.extend([
        row[
            "channel"
        ],
        (
            "  temporal tests                 : "
            f"{row['temporal_tests']}"
        ),
        (
            "  RAW wins                       : "
            f"{row['raw_wins']}"
        ),
        (
            "  SMOOTH wins                    : "
            f"{row['smooth_wins']}"
        ),
        (
            "  RAW future error mean          : "
            f"{row['raw_future_error_mean']:.10f}"
        ),
        (
            "  SMOOTH->RAW error mean         : "
            f"{row['smooth_end_to_end_raw_error_mean']:.10f}"
        ),
        (
            "  delta mean                     : "
            f"{row['delta_mean']:.10f}"
        ),
        (
            "  ratio mean                     : "
            f"{row['ratio_mean']:.10f}"
        ),
        (
            "  RAW K90 median                 : "
            f"{row['raw_k90_median']:.2f}"
        ),
        (
            "  SMOOTH K90 median              : "
            f"{row['smooth_k90_median']:.2f}"
        ),
        ""
    ])


lines.extend([
    "D2 VALIDATION CORRECTION",
    "-" * 108,
    (
        f"Original D2 status                : "
        f"{d2.get('status')}"
    ),
    (
        f"Original D2 failures              : "
        f"{d2.get('validation_failures')}"
    ),
    (
        "Original failed check             : "
        +
        expected_bad_check
    ),
    (
        f"D1.1 negative configs             : "
        f"{negative_configs}/{audited_configs}"
    ),
    (
        f"D2 test subsets with negatives    : "
        f"{negative_test_rows}"
    ),
    (
        f"Corrected validation failures      : "
        f"{failures}"
    ),
    "",
    "SCIENTIFIC SCOPE",
    "-" * 108,
    "Target used                        : NO",
    "Predictive superiority claimed     : NO",
    "Final K selected                   : NO",
    "Folds frozen                       : NO",
    "Classifier trained                 : NO",
    "Silver created                     : NO",
    "RAW modified                       : NO",
    "",
    (
        "Original D2 artifacts preserved : "
        +
        (
            "YES"
            if originals_preserved
            else
            "NO"
        )
    ),
    "",
    "NEXT STAGE",
    "-" * 108,
    "RAW-30D",
    "Temporal component-count policy",
    "=" * 108
])


OUT_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 16. PRINT
# =============================================================================

print()
print("=" * 108)
print("VALIDATION V2")
print("=" * 108)

for row in checks:

    print(
        f"[{row['status']}] "
        f"{row['check']} | "
        f"observed={row['observed']} | "
        f"expected={row['expected']}"
    )


print()
print(
    "VALIDATION FAILURES =",
    failures
)


print()
print("=" * 108)
print("CHANNEL EVIDENCE")
print("=" * 108)

for row in channel_summary:

    print()
    print(
        row[
            "channel"
        ]
    )

    print(
        "  tests             =",
        row[
            "temporal_tests"
        ]
    )

    print(
        "  RAW wins          =",
        row[
            "raw_wins"
        ]
    )

    print(
        "  SMOOTH wins       =",
        row[
            "smooth_wins"
        ]
    )

    print(
        "  RAW error mean    =",
        f"{row['raw_future_error_mean']:.10f}"
    )

    print(
        "  SMOOTH error mean =",
        f"{row['smooth_end_to_end_raw_error_mean']:.10f}"
    )

    print(
        "  delta mean        =",
        f"{row['delta_mean']:.10f}"
    )

    print(
        "  ratio mean        =",
        f"{row['ratio_mean']:.10f}"
    )

    print(
        "  RAW K90 median    =",
        row[
            "raw_k90_median"
        ]
    )

    print(
        "  SMOOTH K90 median =",
        row[
            "smooth_k90_median"
        ]
    )


print()
print("=" * 108)
print("FINAL DECISION")
print("=" * 108)

print(
    "STATUS             :",
    decision[
        "status"
    ]
)

print(
    "RAW-30D            :",
    decision[
        "primary_representation_for_next_stage"
    ][
        "status"
    ]
)

print(
    "SMOOTH-30D         :",
    decision[
        "smooth_challenger"
    ][
        "status"
    ]
)

print(
    "RAW wins           :",
    raw_wins,
    "/",
    total_pairs
)

print(
    "SMOOTH wins        :",
    smooth_wins,
    "/",
    total_pairs
)

print(
    "D2 original        :",
    d2.get(
        "status"
    ),
    "(preserved)"
)

print(
    "D2 V2 failures     :",
    failures
)

print(
    "Originals preserved:",
    originals_preserved
)

print()
print("Target usado       : NÃO")
print("K final            : NÃO")
print("Folds congelados   : NÃO")
print("Modelo treinado    : NÃO")
print("Silver criada      : NÃO")
print("RAW alterado       : NÃO")


if failures != 0:
    sys.exit(2)


if not decision_supported:
    sys.exit(3)


print()
print("[PASS] D2.1-A FECHADO.")
print("[PASS] RAW-30D avança para a próxima etapa.")
print("[PASS] Nenhum experimento pesado foi reexecutado.")

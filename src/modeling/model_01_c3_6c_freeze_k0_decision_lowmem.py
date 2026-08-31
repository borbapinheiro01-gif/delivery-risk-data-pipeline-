#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
MODEL 01 — C3.6-C LOW-MEMORY
FORMAL FREEZE OF K*=0
================================================================================

Esta etapa NÃO treina modelo e NÃO executa PCA.

Objetivo:
    formalizar a decisão K*=0 para MODEL_01_ORDER_LOGISTIC
    usando exclusivamente os resultados já produzidos em C3.6-A/B.

Bibliotecas:
    somente Python standard library.

Não faz:
    - pandas
    - numpy
    - sklearn
    - PCA
    - bootstrap
    - treinamento
    - threshold
    - Silver
    - alteração de RAW
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

REPORT = (
    ROOT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

CONTRACT = (
    ROOT
    / "configs"
    / "model_01_supervised_temporal_contract_v1.json"
)

C36A_FOLDS = REPORT / "07a_supervised_k_fold_metrics.csv"
C36A_VALID = REPORT / "07d_supervised_k_validation.csv"
C36A_SUMMARY = REPORT / "07e_supervised_k_summary.json"

C36B_PAIRED = REPORT / "08b_ap_paired_summary.csv"
C36B_DELETION = REPORT / "08c_contiguous_deletion_sensitivity.csv"
C36B_BOOT = REPORT / "08d_block_bootstrap_inference.csv"
C36B_SECONDARY = REPORT / "08e_secondary_metric_concordance.csv"
C36B_VALID = REPORT / "08f_c36b_validation.csv"
C36B_SUMMARY = REPORT / "08g_c36b_summary.json"

OUT_DECISION = REPORT / "09a_model01_k0_formal_decision.json"
OUT_VALIDATION = REPORT / "09b_model01_k0_freeze_validation.csv"
OUT_REPORT = REPORT / "09c_model01_k0_freeze_report.txt"

OUT_DOC = ROOT / "docs" / "MODEL_01_FUNCTIONAL_K_DECISION.md"


# =============================================================================
# CONSTANTS FROZEN BEFORE PERFORMANCE
# =============================================================================

EXPECTED_CONTRACT_SHA256 = (
    "c417cdb43a336033a2ffd6a409e93a55"
    "f48376fa0da644051c44afdcce91706b"
)

EXPECTED_K = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    10,
    15,
    20,
    23,
    25,
    27,
    30,
]

EXPECTED_MONTHS = 17
EXPECTED_FITS = 238
EXPECTED_ALTERNATIVES = 13


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def bool_value(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "sim",
        "pass",
    }


def find_column(fieldnames, candidates):
    lowered = {
        str(x).strip().lower(): x
        for x in fieldnames
        if x is not None
    }

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_write_csv(path: Path, rows):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    fieldnames = [
        "check",
        "status",
        "observed",
        "expected",
    ]

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        tmp,
        path,
    )


checks = []


def check(name, condition, observed, expected):
    checks.append(
        {
            "check": name,
            "status": (
                "PASS"
                if bool(condition)
                else "FAIL"
            ),
            "observed": str(observed),
            "expected": str(expected),
        }
    )


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 112)
print("MODEL 01 — C3.6-C LOW-MEMORY")
print("FORMAL FREEZE OF K*=0")
print("=" * 112)


# =============================================================================
# REQUIRED FILES
# =============================================================================

required = [
    CONTRACT,
    C36A_FOLDS,
    C36A_VALID,
    C36A_SUMMARY,
    C36B_PAIRED,
    C36B_DELETION,
    C36B_BOOT,
    C36B_SECONDARY,
    C36B_VALID,
    C36B_SUMMARY,
]

print()
print("PRÉ-REQUISITOS")
print("-" * 112)

missing = []

for path in required:

    ok = path.exists() and path.stat().st_size > 0

    print(
        f"[{'PASS' if ok else 'FAIL'}] "
        f"{path.name}"
    )

    if not ok:
        missing.append(str(path))


if missing:

    print()
    print("[FAIL] Arquivos ausentes:")

    for x in missing:
        print(" -", x)

    sys.exit(2)


# =============================================================================
# LOAD SMALL FILES ONLY
# =============================================================================

a_summary = load_json(C36A_SUMMARY)
b_summary = load_json(C36B_SUMMARY)

a_valid = load_csv(C36A_VALID)
b_valid = load_csv(C36B_VALID)

folds = load_csv(C36A_FOLDS)
paired = load_csv(C36B_PAIRED)


# =============================================================================
# CONTRACT PROVENANCE
# =============================================================================

contract_hash = sha256_file(
    CONTRACT
)

check(
    "frozen_contract_sha256",
    contract_hash
    == EXPECTED_CONTRACT_SHA256,
    contract_hash,
    EXPECTED_CONTRACT_SHA256,
)


# =============================================================================
# C3.6-A STATUS
# =============================================================================

check(
    "c36a_status_pass",
    str(
        a_summary.get(
            "status",
            "",
        )
    ).upper()
    == "PASS",
    a_summary.get(
        "status"
    ),
    "PASS",
)


def validation_failure_count(rows):

    if not rows:
        return 1

    fields = list(
        rows[0].keys()
    )

    status_col = find_column(
        fields,
        [
            "status",
            "result",
            "validation_status",
        ],
    )

    if status_col is None:
        return 1

    return sum(
        1
        for row in rows
        if str(
            row.get(
                status_col,
                "",
            )
        ).strip().upper()
        not in {
            "PASS",
            "OK",
            "TRUE",
        }
    )


a_validation_failures = (
    validation_failure_count(
        a_valid
    )
)

b_validation_failures = (
    validation_failure_count(
        b_valid
    )
)


check(
    "c36a_validation_zero_failures",
    a_validation_failures == 0,
    a_validation_failures,
    0,
)

check(
    "c36b_validation_zero_failures",
    b_validation_failures == 0,
    b_validation_failures,
    0,
)


# =============================================================================
# INDEPENDENT RECOMPUTATION OF K MEAN AP FROM 07a
# =============================================================================

if not folds:
    raise RuntimeError(
        "07a está vazio."
    )

fold_fields = list(
    folds[0].keys()
)

k_col = find_column(
    fold_fields,
    [
        "k",
        "K",
        "functional_k",
    ],
)

ap_col = find_column(
    fold_fields,
    [
        "average_precision",
        "ap",
        "avg_precision",
    ],
)


if k_col is None:
    raise RuntimeError(
        f"Coluna K não encontrada em 07a. "
        f"Colunas: {fold_fields}"
    )

if ap_col is None:
    raise RuntimeError(
        f"Coluna AP não encontrada em 07a. "
        f"Colunas: {fold_fields}"
    )


ap_by_k = defaultdict(list)

for row in folds:

    k = int(
        float(
            row[k_col]
        )
    )

    ap = float(
        row[ap_col]
    )

    ap_by_k[k].append(
        ap
    )


observed_k = sorted(
    ap_by_k.keys()
)

check(
    "k_grid_exact",
    observed_k == EXPECTED_K,
    observed_k,
    EXPECTED_K,
)


fit_count = sum(
    len(v)
    for v in ap_by_k.values()
)

check(
    "model_fit_count",
    fit_count == EXPECTED_FITS,
    fit_count,
    EXPECTED_FITS,
)


counts_by_k = {
    k: len(v)
    for k, v
    in sorted(
        ap_by_k.items()
    )
}

all_17 = all(
    n == EXPECTED_MONTHS
    for n in counts_by_k.values()
)

check(
    "seventeen_months_per_k",
    all_17,
    counts_by_k,
    f"{EXPECTED_MONTHS} por K",
)


mean_ap = {
    k: statistics.fmean(values)
    for k, values
    in ap_by_k.items()
}


best_k = max(
    mean_ap,
    key=lambda k: (
        mean_ap[k],
        -k,
    ),
)


baseline_ap = mean_ap[0]


check(
    "k0_highest_descriptive_mean_ap",
    best_k == 0,
    {
        "best_k": best_k,
        "mean_ap": mean_ap[best_k],
    },
    "K=0",
)


# =============================================================================
# C3.6-B SUMMARY — PRIMARY SCIENTIFIC DECISION
# =============================================================================

check(
    "c36b_status_pass",
    str(
        b_summary.get(
            "status",
            "",
        )
    ).upper()
    == "PASS",
    b_summary.get(
        "status"
    ),
    "PASS",
)


check(
    "primary_metric_average_precision",
    str(
        b_summary.get(
            "primary_metric",
            "",
        )
    ).upper()
    == "AVERAGE_PRECISION",
    b_summary.get(
        "primary_metric"
    ),
    "AVERAGE_PRECISION",
)


check(
    "baseline_k_zero",
    int(
        b_summary.get(
            "baseline_k",
            -999,
        )
    )
    == 0,
    b_summary.get(
        "baseline_k"
    ),
    0,
)


check(
    "alternative_k_count",
    int(
        b_summary.get(
            "alternative_k_count",
            -1,
        )
    )
    == EXPECTED_ALTERNATIVES,
    b_summary.get(
        "alternative_k_count"
    ),
    EXPECTED_ALTERNATIVES,
)


check(
    "temporal_month_count",
    int(
        b_summary.get(
            "temporal_months",
            -1,
        )
    )
    == EXPECTED_MONTHS,
    b_summary.get(
        "temporal_months"
    ),
    EXPECTED_MONTHS,
)


check(
    "all_observed_mean_deltas_negative",
    bool_value(
        b_summary.get(
            "all_observed_mean_deltas_negative"
        )
    ),
    b_summary.get(
        "all_observed_mean_deltas_negative"
    ),
    True,
)


check(
    "all_observed_median_deltas_negative",
    bool_value(
        b_summary.get(
            "all_observed_median_deltas_negative"
        )
    ),
    b_summary.get(
        "all_observed_median_deltas_negative"
    ),
    True,
)


check(
    "all_contiguous_deletions_negative",
    bool_value(
        b_summary.get(
            "all_k_remain_negative_after_all_contiguous_deletions"
        )
    ),
    b_summary.get(
        "all_k_remain_negative_after_all_contiguous_deletions"
    ),
    True,
)


check(
    "no_simultaneous_functional_superiority",
    not bool_value(
        b_summary.get(
            "any_bootstrap_simultaneous_functional_superiority"
        )
    ),
    b_summary.get(
        "any_bootstrap_simultaneous_functional_superiority"
    ),
    False,
)


robust_functional = (
    b_summary.get(
        "robust_functional_superiority_k",
        [],
    )
    or []
)

check(
    "robust_functional_superiority_empty",
    len(
        robust_functional
    )
    == 0,
    robust_functional,
    [],
)


# IMPORTANT:
# isto impede escrever a conclusão mais forte que os dados permitem.
robust_k0 = (
    b_summary.get(
        "robust_k0_superiority_k",
        [],
    )
    or []
)

check(
    "robust_k0_superiority_empty",
    len(
        robust_k0
    )
    == 0,
    robust_k0,
    [],
)


check(
    "secondary_mean_benefits_nonpositive",
    bool_value(
        b_summary.get(
            "all_secondary_mean_benefits_nonpositive"
        )
    ),
    b_summary.get(
        "all_secondary_mean_benefits_nonpositive"
    ),
    True,
)


decision_candidate = (
    b_summary.get(
        "decision_candidate"
    )
)

check(
    "decision_candidate_ready",
    decision_candidate
    == "K0_READY_FOR_FORMAL_FREEZE_MODEL01",
    decision_candidate,
    "K0_READY_FOR_FORMAL_FREEZE_MODEL01",
)


check(
    "k_not_previously_frozen",
    not bool_value(
        b_summary.get(
            "final_k_frozen",
            False,
        )
    ),
    b_summary.get(
        "final_k_frozen",
        False,
    ),
    False,
)


check(
    "threshold_not_selected",
    not bool_value(
        b_summary.get(
            "threshold_selected",
            False,
        )
    ),
    b_summary.get(
        "threshold_selected",
        False,
    ),
    False,
)


check(
    "raw_unmodified",
    not bool_value(
        b_summary.get(
            "raw_modified",
            False,
        )
    ),
    b_summary.get(
        "raw_modified",
        False,
    ),
    False,
)


check(
    "silver_not_created",
    not bool_value(
        b_summary.get(
            "silver_created",
            False,
        )
    ),
    b_summary.get(
        "silver_created",
        False,
    ),
    False,
)


# =============================================================================
# PAIRED CSV INDEPENDENT CHECK
# =============================================================================

if not paired:
    raise RuntimeError(
        "08b está vazio."
    )

paired_fields = list(
    paired[0].keys()
)

paired_k_col = find_column(
    paired_fields,
    ["k"],
)

mean_delta_col = find_column(
    paired_fields,
    [
        "mean_delta_ap",
        "mean_ap_delta",
    ],
)

median_delta_col = find_column(
    paired_fields,
    [
        "median_delta_ap",
        "median_ap_delta",
    ],
)


if (
    paired_k_col is None
    or mean_delta_col is None
    or median_delta_col is None
):
    raise RuntimeError(
        "Colunas necessárias não encontradas em 08b. "
        f"Colunas: {paired_fields}"
    )


paired_alt = [
    row
    for row in paired
    if int(
        float(
            row[paired_k_col]
        )
    )
    > 0
]


check(
    "paired_alternative_rows",
    len(
        paired_alt
    )
    == EXPECTED_ALTERNATIVES,
    len(
        paired_alt
    ),
    EXPECTED_ALTERNATIVES,
)


paired_all_mean_negative = all(
    float(
        row[mean_delta_col]
    )
    < 0
    for row in paired_alt
)

paired_all_median_negative = all(
    float(
        row[median_delta_col]
    )
    < 0
    for row in paired_alt
)


check(
    "paired_csv_all_mean_deltas_negative",
    paired_all_mean_negative,
    paired_all_mean_negative,
    True,
)


check(
    "paired_csv_all_median_deltas_negative",
    paired_all_median_negative,
    paired_all_median_negative,
    True,
)


# =============================================================================
# FINAL VALIDATION
# =============================================================================

failure_count = sum(
    row["status"] == "FAIL"
    for row in checks
)

status = (
    "FROZEN"
    if failure_count == 0
    else "REVIEW_REQUIRED"
)


# =============================================================================
# WRITE VALIDATION FIRST
# =============================================================================

atomic_write_csv(
    OUT_VALIDATION,
    checks,
)


# =============================================================================
# DECISION
# =============================================================================

decision = {
    "decision_id":
        "MODEL01_C36C_FUNCTIONAL_K_FREEZE_V1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        status,

    "model":
        "MODEL_01_ORDER_LOGISTIC",

    "prediction_time":
        "ORDER_PURCHASE_TIMESTAMP",

    "target":
        "LATE_DELIVERY_CALENDAR_DAY",

    "primary_metric":
        "AVERAGE_PRECISION",

    "protocol":
        "FROZEN_EXPANDING_HISTORY_MONTHLY_OUT_OF_TIME",

    "contract_sha256":
        contract_hash,

    "temporal_months":
        EXPECTED_MONTHS,

    "model_fits_audited":
        fit_count,

    "selected_k":
        (
            0
            if failure_count == 0
            else None
        ),

    "selected_input":
        (
            "ORDER_CORE_V1_ONLY"
            if failure_count == 0
            else None
        ),

    "functional_dimensions_added":
        (
            0
            if failure_count == 0
            else None
        ),

    "baseline_mean_average_precision":
        baseline_ap,

    "descriptive_best_k":
        best_k,

    "decision":
        (
            "REJECT_FUNCTIONAL_COMPONENTS_FOR_PRIMARY_MODEL_01"
            if failure_count == 0
            else
            "NO_FREEZE_REVIEW_REQUIRED"
        ),

    "scientific_language": {
        "supported": (
            "No robust evidence of predictive gain "
            "from RAW-30D functional principal components "
            "was found relative to K=0 under the frozen "
            "temporal MODEL 01 logistic-regression protocol."
        ),

        "not_supported": (
            "K=0 is universally or statistically superior "
            "to every K under all possible conditions."
        ),

        "important_scope": (
            "The decision applies to MODEL_01_ORDER_LOGISTIC "
            "and does not establish that temporal information, "
            "regional interactions or nonlinear representations "
            "are universally non-predictive."
        ),
    },

    "inference_guardrail": {
        "robust_functional_superiority_k":
            robust_functional,

        "robust_k0_superiority_k":
            robust_k0,

        "interpretation":
            (
                "Model-selection evidence supports not adding "
                "functional PCs to MODEL 01, while simultaneous "
                "inference does not justify a universal superiority "
                "claim for K=0."
            ),
    },

    "governance": {
        "k_final_frozen":
            failure_count == 0,

        "threshold_selected":
            False,

        "classifier_refit":
            False,

        "pca_refit":
            False,

        "bootstrap_rerun":
            False,

        "raw_modified":
            False,

        "silver_created":
            False,
    },

    "next_branch": {
        "name":
            "SPATIOTEMPORAL_LOGISTICS_AUDIT",

        "status":
            (
                "READY_TO_START"
                if failure_count == 0
                else
                "BLOCKED_PENDING_REVIEW"
            ),

        "initial_axes": [
            "product_price",
            "freight_absolute",
            "freight_to_price_ratio",
            "delivery_delay",
            "delivery_speed",
            "origin_destination",
            "region",
            "distance",
            "weight_and_volume",
            "calendar_period",
            "external_events",
            "regional_context",
        ],
    },

    "provenance_sha256": {
        path.name:
            sha256_file(path)

        for path in required
    },

    "validation_failures":
        failure_count,
}


atomic_write_text(
    OUT_DECISION,
    json.dumps(
        decision,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
)


# =============================================================================
# REPORT
# =============================================================================

report_lines = [
    "=" * 112,
    "MODEL 01 — C3.6-C LOW-MEMORY — FORMAL K FREEZE",
    "=" * 112,
    "",
    f"STATUS                          : {status}",
    f"VALIDATION FAILURES             : {failure_count}",
    "",
    f"MODEL                           : MODEL_01_ORDER_LOGISTIC",
    f"PRIMARY METRIC                  : Average Precision",
    f"TEMPORAL MONTHS                 : {EXPECTED_MONTHS}",
    f"MODEL FITS AUDITED              : {fit_count}",
    "",
    f"CONTRACT SHA256                 : {contract_hash}",
    f"CONTRACT HASH MATCH             : {contract_hash == EXPECTED_CONTRACT_SHA256}",
    "",
    f"K0 MEAN AP                      : {baseline_ap:.10f}",
    f"DESCRIPTIVE BEST K              : {best_k}",
    f"ALL PAIRED MEAN DELTAS < 0      : {paired_all_mean_negative}",
    f"ALL PAIRED MEDIAN DELTAS < 0    : {paired_all_median_negative}",
    "",
    f"ROBUST FUNCTIONAL SUPERIOR K    : {robust_functional}",
    f"ROBUST K0 SUPERIOR K            : {robust_k0}",
    "",
    "INTERPRETATION:",
    "",
    "No robust predictive improvement was found for K>0 relative to K=0",
    "under the frozen MODEL 01 temporal protocol.",
    "",
    "This DOES NOT mean that temporal information is universally useless.",
    "This DOES NOT establish universal statistical superiority of K=0.",
    "",
    f"FORMAL K MODEL 01               : {0 if failure_count == 0 else 'NOT FROZEN'}",
    f"SELECTED INPUT                  : {'ORDER_CORE_V1_ONLY' if failure_count == 0 else 'REVIEW'}",
    "",
    "NEW TRAINING                   : NO",
    "PCA RERUN                      : NO",
    "BOOTSTRAP RERUN                : NO",
    "THRESHOLD                      : NOT SELECTED",
    "SILVER                         : NOT CREATED",
    "RAW                            : INTACT",
    "",
    "NEXT BRANCH:",
    "SPATIOTEMPORAL_LOGISTICS_AUDIT",
    "",
    "=" * 112,
]

atomic_write_text(
    OUT_REPORT,
    "\n".join(
        report_lines
    )
    + "\n",
)


# =============================================================================
# SCIENTIFIC DOCUMENT
# =============================================================================

if failure_count == 0:

    doc = f"""
# MODEL 01 — Formal Functional-K Decision

## C3.6-C

The frozen temporal supervised experiment compared the baseline

\\[
K=0
\\]

against thirteen alternatives containing RAW-30D functional
principal-component scores.

For future month \\(m\\) and candidate \\(K>0\\),

\\[
\\Delta_{{m,K}}
=
AP_{{m,K}}
-
AP_{{m,0}}.
\\]

The baseline contains only `ORDER_CORE_V1`.

The frozen experiment contained

\\[
14\\times17 = 238
\\]

logistic-regression fits.

The independently recomputed mean Average Precision of the baseline was

\\[
AP_{{K=0}}
=
{baseline_ap:.10f}.
\\]

For every evaluated \\(K>0\\), the observed mean and median paired
Average-Precision differences were negative.

The conclusion remained negative under contiguous temporal deletion
sensitivity.

Circular block-bootstrap simultaneous inference identified no robust
functional challenger.

Importantly, simultaneous inference also does not support the stronger
claim that \\(K=0\\) is universally statistically superior to every
alternative.

Therefore the decision for the first logistic model is

\\[
\\boxed{{K^\\star_{{MODEL\\ 01}}=0}}.
\\]

Thus

\\[
\\boxed{{MODEL\\ 01 = ORDER\\_CORE\\_V1\\ only}}.
\\]

## Scope

This decision applies specifically to `MODEL_01_ORDER_LOGISTIC`.

It does not establish that temporal structure is universally
non-predictive, nor does it preclude nonlinear, geographic,
regime-dependent or external-context interactions.

## Next analytical branch

The next independent explanatory branch is the
`SPATIOTEMPORAL_LOGISTICS_AUDIT`, beginning with the joint study of

\\[
\\text{{price}}
+
\\text{{freight}}
+
\\text{{delivery outcome}}
+
\\text{{origin/destination}}
+
\\text{{region}}
+
\\text{{time}}
+
\\text{{external context}}.
\\]

The branch will be kept analytically separate from the frozen MODEL 01
decision.
""".strip() + "\n"

    atomic_write_text(
        OUT_DOC,
        doc,
    )


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 112)
print("VALIDATION")
print("=" * 112)

for row in checks:

    print(
        f"[{row['status']}] "
        f"{row['check']}"
        f" | observed={row['observed']}"
    )


print()
print("=" * 112)
print("FINAL DECISION")
print("=" * 112)

print(
    "STATUS                     :",
    status,
)

print(
    "VALIDATION FAILURES        :",
    failure_count,
)

print(
    "CONTRACT HASH MATCH        :",
    contract_hash
    == EXPECTED_CONTRACT_SHA256,
)

print(
    "MODEL FITS AUDITED         :",
    fit_count,
)

print(
    "K0 MEAN AP                 :",
    f"{baseline_ap:.10f}",
)

print(
    "DESCRIPTIVE BEST K         :",
    best_k,
)

print(
    "ROBUST FUNCTIONAL K        :",
    robust_functional,
)

print(
    "ROBUST K0 SUPERIOR K       :",
    robust_k0,
)

if failure_count == 0:

    print()
    print(
        "[PASS] C3.6-C concluído."
    )

    print(
        "[PASS] K*=0 formalmente congelado "
        "para MODEL_01_ORDER_LOGISTIC."
    )

    print(
        "[PASS] ORDER_CORE_V1_ONLY é a "
        "representação primária do MODEL 01."
    )

    print(
        "[PASS] SPATIOTEMPORAL_LOGISTICS_AUDIT "
        "está liberada para iniciar."
    )

else:

    print()
    print(
        "[FAIL] Não congelar K."
    )

    print(
        "[FAIL] Revisar 09b_model01_k0_freeze_validation.csv."
    )


sys.exit(
    0
    if failure_count == 0
    else 2
)
